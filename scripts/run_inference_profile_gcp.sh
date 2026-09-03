#!/usr/bin/env bash
# Explicit maintainer workflow for the measured L4 profile.
# The textbook build never invokes this script.

set -euo pipefail

PROJECT="potent-arcade-491015-g7"
ACCOUNT="pierreluc@carbonforge.ai"
VM_NAME="pierreluc-l4-rlbook-profile"
MACHINE_TYPE="g2-standard-8"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
REMOTE_SCRIPT="/var/tmp/profile_inference_gpu.py"
REMOTE_OUTPUT="/var/tmp/inference-serving-profile"
LOCAL_OUTPUT="${1:-data/inference_serving}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STAGING_ROOT="${TMPDIR:-/private/tmp}"
STAGING_OUTPUT="${STAGING_ROOT%/}/rlbook-inference-profile-${VM_NAME}-${RUN_STAMP}"
# Keep the accelerator and host shape fixed while searching more capacity.
# Canadian zones are included for Spot; the paid fallback is restricted to US
# regions covered by the price guard below.
SPOT_ZONES=(
  "northamerica-northeast2-a"
  "northamerica-northeast2-b"
  "northamerica-northeast1-b"
  "northamerica-northeast1-c"
  "us-central1-a"
  "us-central1-b"
  "us-west1-a"
  "us-west1-b"
  "us-west4-a"
  "us-west4-c"
  "us-east1-b"
  "us-east1-c"
  "us-east1-d"
  "us-east4-a"
  "us-east4-c"
)
STANDARD_ZONES=(
  "us-central1-a"
  "us-central1-b"
  "us-central1-c"
)
PROFILE_PROVISIONING_MODE="${RLBOOK_PROFILE_PROVISIONING:-AUTO}"
if [[ "${PROFILE_PROVISIONING_MODE}" != "AUTO" && "${PROFILE_PROVISIONING_MODE}" != "STANDARD" ]]; then
  echo "RLBOOK_PROFILE_PROVISIONING must be AUTO or STANDARD." >&2
  exit 2
fi

# This final retry uses a two-hour provider cap. Earlier standard and Spot
# attempts, including the completed 100-minute guarded attempt, consumed an
# estimated USD 4.35. Including that ledger here keeps the worst-case cumulative
# exposure below the explicitly bounded USD 6.75 ceiling, with conservative
# disk and miscellaneous headroom. DELETE termination means the guard does not
# rely only on the local cleanup trap. Price checked
# 2026-09-02 against:
# https://cloud.google.com/products/compute/pricing/accelerator-optimized
MAX_RUNTIME_HOURS="2"
MAX_RUNTIME="2h"
MAX_EXPOSURE_USD="6.75"
PRIOR_PROFILE_COMPUTE_USD="4.35"
ON_DEMAND_USD_PER_HOUR="0.853624312"
DISK_AND_MISC_HEADROOM_USD="0.50"
SOURCE_GIT_REVISION="$(git rev-parse HEAD 2>/dev/null || true)"
LAUNCHER_SCRIPT_SHA256="$(shasum -a 256 "$0" | awk '{print $1}')"
if SOURCE_STATUS_OUTPUT="$(git status --porcelain --untracked-files=all 2>/dev/null)"; then
  if [[ -n "${SOURCE_STATUS_OUTPUT}" ]]; then
    SOURCE_WORKTREE_STATE="dirty"
  else
    SOURCE_WORKTREE_STATE="clean"
  fi
else
  SOURCE_WORKTREE_STATE="unknown"
fi
READINESS_ATTEMPTS=90
READINESS_INTERVAL_S=10
SSH_READY_ATTEMPTS=30
SSH_READY_INTERVAL_S=5

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required." >&2
  exit 2
fi

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)"
if [[ "${ACTIVE_ACCOUNT}" != "${ACCOUNT}" ]]; then
  echo "Authenticate gcloud as ${ACCOUNT} before starting the paid profile run." >&2
  exit 2
fi

if [[ "$(gcloud config get-value project 2>/dev/null)" != "${PROJECT}" ]]; then
  echo "Set the active gcloud project to ${PROJECT} before running this command." >&2
  exit 2
fi

if [[ -n "$(gcloud compute instances list \
  --project "${PROJECT}" \
  --filter="name=${VM_NAME}" \
  --format='value(name)' 2>/dev/null)" ]]; then
  echo "${VM_NAME} already exists. Resolve or remove it before creating a measured run." >&2
  exit 2
fi

if [[ -n "$(gcloud compute disks list \
  --project "${PROJECT}" \
  --filter="name=${VM_NAME}" \
  --format='value(name)' 2>/dev/null)" ]]; then
  echo "A disk named ${VM_NAME} already exists. Resolve or remove it before creating a measured run." >&2
  exit 2
fi

ESTIMATED_MAX_EXPOSURE_USD="$(python3 - \
  "${PRIOR_PROFILE_COMPUTE_USD}" \
  "${MAX_RUNTIME_HOURS}" \
  "${ON_DEMAND_USD_PER_HOUR}" \
  "${DISK_AND_MISC_HEADROOM_USD}" <<'PY'
from decimal import Decimal
import sys

prior, hours, hourly, headroom = map(Decimal, sys.argv[1:])
print(prior + hours * hourly + headroom)
PY
)"
python3 - "${ESTIMATED_MAX_EXPOSURE_USD}" "${MAX_EXPOSURE_USD}" <<'PY'
from decimal import Decimal
import sys

estimated, ceiling = map(Decimal, sys.argv[1:])
if estimated > ceiling:
    raise SystemExit(
        f"guarded on-demand exposure ${estimated} exceeds the ${ceiling} ceiling"
    )
PY

SELECTED_ZONE=""
SELECTED_PROVISIONING_MODEL=""
CLEANUP_ARMED=0

cleanup() {
  local status=$?
  if [[ "${CLEANUP_ARMED}" -eq 1 && -n "${SELECTED_ZONE}" ]]; then
    gcloud compute instances delete "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${SELECTED_ZONE}" \
      --delete-disks=all \
      --quiet || true
    if gcloud compute disks describe "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${SELECTED_ZONE}" >/dev/null 2>&1; then
      gcloud compute disks delete "${VM_NAME}" \
        --project "${PROJECT}" \
        --zone "${SELECTED_ZONE}" \
        --quiet || true
    fi
    if gcloud compute instances describe "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${SELECTED_ZONE}" >/dev/null 2>&1; then
      echo "The profiling VM still exists; inspect it immediately to avoid further charges." >&2
      status=3
    fi
    if gcloud compute disks describe "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${SELECTED_ZONE}" >/dev/null 2>&1; then
      echo "The profiling disk still exists; inspect it immediately to avoid further charges." >&2
      status=3
    fi
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

create_profile_vm() {
  local zone="$1"
  local provisioning_model="$2"
  local model_label
  local create_output
  model_label="$(printf '%s' "${provisioning_model}" | tr '[:upper:]' '[:lower:]')"

  echo "Trying ${provisioning_model} in ${zone}..."
  if create_output="$(gcloud compute instances create "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${zone}" \
    --machine-type "${MACHINE_TYPE}" \
    --provisioning-model "${provisioning_model}" \
    --instance-termination-action DELETE \
    --max-run-duration "${MAX_RUNTIME}" \
    --maintenance-policy TERMINATE \
    --no-restart-on-failure \
    --reservation-affinity=none \
    --boot-disk-size 200GB \
    --boot-disk-type pd-balanced \
    --boot-disk-auto-delete \
    --no-deletion-protection \
    --image-family "${IMAGE_FAMILY}" \
    --image-project "${IMAGE_PROJECT}" \
    --no-service-account \
    --no-scopes \
    --metadata enable-oslogin=TRUE,install-nvidia-driver=True \
    --labels "purpose=rlbook-profile,owner=pierreluc,provisioning=${model_label}" \
    2>&1)"; then
    printf '%s\n' "${create_output}"
    return 0
  fi
  printf '%s\n' "${create_output}" >&2

  # A network or API error can be ambiguous. Fail closed: delete any resource
  # that appeared despite the nonzero command status, then abort this run.
  if gcloud compute instances describe "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${zone}" >/dev/null 2>&1; then
    echo "The create command failed but the VM exists; deleting the uncertain resource." >&2
    gcloud compute instances delete "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${zone}" \
      --delete-disks=all \
      --quiet || true
    if gcloud compute disks describe "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${zone}" >/dev/null 2>&1; then
      gcloud compute disks delete "${VM_NAME}" \
        --project "${PROJECT}" \
        --zone "${zone}" \
        --quiet || true
    fi
    if gcloud compute instances describe "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${zone}" >/dev/null 2>&1 || \
       gcloud compute disks describe "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${zone}" >/dev/null 2>&1; then
      echo "An uncertain profiling resource remains after cleanup." >&2
      return 21
    fi
    return 20
  fi

  if grep -Eq \
    'ZONE_RESOURCE_POOL_EXHAUSTED|reason: stockout|does not have enough resources' \
    <<<"${create_output}"; then
    return 10
  fi

  echo "Non-capacity creation failure; refusing to broaden or enter the paid fallback." >&2
  return 20
}

if [[ "${PROFILE_PROVISIONING_MODE}" == "AUTO" ]]; then
  for zone in "${SPOT_ZONES[@]}"; do
    if create_profile_vm "${zone}" "SPOT"; then
      SELECTED_ZONE="${zone}"
      SELECTED_PROVISIONING_MODEL="SPOT"
      CLEANUP_ARMED=1
      break
    else
      create_status=$?
      if [[ "${create_status}" -ne 10 ]]; then
        exit "${create_status}"
      fi
    fi
  done
fi

if [[ -z "${SELECTED_ZONE}" ]]; then
  if [[ "${PROFILE_PROVISIONING_MODE}" == "STANDARD" ]]; then
    echo "Using the explicitly selected guarded on-demand L4."
  else
    echo "No approved zone accepted a Spot L4. Trying the guarded on-demand fallback."
  fi
  echo "Estimated maximum exposure: USD ${ESTIMATED_MAX_EXPOSURE_USD} (ceiling: USD ${MAX_EXPOSURE_USD})."
  for STANDARD_ZONE in "${STANDARD_ZONES[@]}"; do
    if create_profile_vm "${STANDARD_ZONE}" "STANDARD"; then
      SELECTED_ZONE="${STANDARD_ZONE}"
      SELECTED_PROVISIONING_MODEL="STANDARD"
      CLEANUP_ARMED=1
      break
    else
      create_status=$?
      if [[ "${create_status}" -ne 10 ]]; then
        exit "${create_status}"
      fi
    fi
  done
fi

if [[ -z "${SELECTED_ZONE}" ]]; then
  echo "No approved L4 zone accepted either Spot or guarded on-demand creation." >&2
  exit 2
fi

echo "Created ${VM_NAME} in ${SELECTED_ZONE} as ${SELECTED_PROVISIONING_MODEL}. Waiting for SSH..."
SSH_READY=0
for ((attempt = 1; attempt <= SSH_READY_ATTEMPTS; attempt++)); do
  if gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "true" \
    --ssh-flag "-o ConnectTimeout=10" >/dev/null 2>&1; then
    SSH_READY=1
    break
  fi
  sleep "${SSH_READY_INTERVAL_S}"
done
if [[ "${SSH_READY}" -ne 1 ]]; then
  echo "SSH/OS Login did not become ready within the bounded wait." >&2
  exit 2
fi

mkdir -p "${LOCAL_OUTPUT}"
mkdir -p "${STAGING_OUTPUT}"

echo "SSH is available. Ensuring Docker and the NVIDIA container runtime are available..."
if ! gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${SELECTED_ZONE}" \
  --command "command -v docker >/dev/null 2>&1" \
  --ssh-flag "-o ConnectTimeout=10" >/dev/null 2>&1; then
  gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "sudo -n apt-get update" \
    --ssh-flag "-o ConnectTimeout=10"
  gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io" \
    --ssh-flag "-o ConnectTimeout=10"
fi
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${SELECTED_ZONE}" \
  --command "command -v nvidia-ctk >/dev/null 2>&1" \
  --ssh-flag "-o ConnectTimeout=10"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${SELECTED_ZONE}" \
  --command "sudo -n nvidia-ctk runtime configure --runtime=docker && sudo -n systemctl restart docker" \
  --ssh-flag "-o ConnectTimeout=10"

echo "SSH is available. Waiting for the NVIDIA driver and Docker daemon..."
READY=0
for ((attempt = 1; attempt <= READINESS_ATTEMPTS; attempt++)); do
  VM_STATUS="$(gcloud compute instances describe "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --format='value(status)' 2>/dev/null || true)"
  if [[ "${VM_STATUS}" != "RUNNING" ]]; then
    break
  fi

  if gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --ssh-flag "-o ConnectTimeout=10" \
    --command "sudo -n bash -lc \"mkdir -p '${REMOTE_OUTPUT}' && date -Is >>'${REMOTE_OUTPUT}/bootstrap.log' && nvidia-smi -L >>'${REMOTE_OUTPUT}/bootstrap.log' 2>&1 && docker version --format '{{.Server.Version}}' >>'${REMOTE_OUTPUT}/bootstrap.log' 2>&1\"" \
    >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep "${READINESS_INTERVAL_S}"
done

if [[ "${READY}" -ne 1 ]]; then
  gcloud compute scp \
    "${VM_NAME}:${REMOTE_OUTPUT}/bootstrap.log" \
    "${STAGING_OUTPUT}/bootstrap.log" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" >/dev/null 2>&1 || true
  gcloud compute instances get-serial-port-output "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    >"${STAGING_OUTPUT}/serial-port.log" 2>&1 || true
  echo "GPU/Docker initialization did not complete within the bounded readiness window. Logs remain in ${STAGING_OUTPUT}." >&2
  exit 2
fi

# The guest driver can become ready after Docker was first configured. Restart
# the daemon once more after host NVML is healthy, then require the NVIDIA
# container runtime itself to enumerate the L4 before pulling or launching
# vLLM. This prevents a host-ready/container-NVML race from reaching the
# expensive model startup.
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${SELECTED_ZONE}" \
  --ssh-flag "-o ConnectTimeout=10" \
  --command "sudo -n nvidia-ctk runtime configure --runtime=docker && sudo -n systemctl restart docker && sleep 5 && nvidia-smi -L && sudo -n bash -lc \"nvidia-container-cli info >>'${REMOTE_OUTPUT}/bootstrap.log' 2>&1\""

gcloud compute scp scripts/profile_inference_gpu.py \
  "${VM_NAME}:${REMOTE_SCRIPT}" \
  --project "${PROJECT}" \
  --zone "${SELECTED_ZONE}"

START_COMMAND="mkdir -p '${REMOTE_OUTPUT}'; rm -f '${REMOTE_OUTPUT}/profile.complete' '${REMOTE_OUTPUT}/profile.failed'; nohup python3 '${REMOTE_SCRIPT}' --output-directory '${REMOTE_OUTPUT}' --source-git-revision '${SOURCE_GIT_REVISION}' --source-worktree-state '${SOURCE_WORKTREE_STATE}' --launcher-script-sha256 '${LAUNCHER_SCRIPT_SHA256}' --cloud-project '${PROJECT}' --cloud-zone '${SELECTED_ZONE}' --provisioning-model '${SELECTED_PROVISIONING_MODEL}' --machine-type '${MACHINE_TYPE}' >'${REMOTE_OUTPUT}/driver.log' 2>&1 </dev/null & profile_pid=\$!; echo \$profile_pid >'${REMOTE_OUTPUT}/profile.pid'"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${SELECTED_ZONE}" \
  --command "sudo -n bash -lc \"${START_COMMAND}\""

SEEN_SWEEPS=""

copy_progress() {
  local availability
  local remote_file
  for remote_file in l4_profile.csv l4_profile_all_requested.csv l4_profile_raw.csv l4_telemetry.csv profile_manifest.json vllm.log driver.log bootstrap.log; do
    if ! availability="$(gcloud compute ssh "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${SELECTED_ZONE}" \
      --command "sudo -n bash -lc \"if test -r '${REMOTE_OUTPUT}/${remote_file}'; then printf present; else printf absent; fi\"" 2>/dev/null)"; then
      echo "Could not query ${remote_file}; leaving the VM running and retrying." >&2
      return 1
    fi
    if [[ "${availability}" == "present" ]]; then
      if ! gcloud compute scp \
        "${VM_NAME}:${REMOTE_OUTPUT}/${remote_file}" \
        "${STAGING_OUTPUT}/${remote_file}" \
        --project "${PROJECT}" \
        --zone "${SELECTED_ZONE}" >/dev/null; then
        echo "Could not copy ${remote_file}; leaving the VM running and retrying." >&2
        return 1
      fi
    fi
  done
}

while true; do
  # A transient API/auth/config-lock failure must not be mistaken for a stopped
  # VM: the EXIT trap would then delete a healthy paid run.  The provider-side
  # max-run-duration remains the fail-safe while local monitoring retries.
  if ! VM_STATUS="$(gcloud compute instances describe "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --format='value(status)' 2>/dev/null)"; then
    echo "Could not read the profiling VM status; retrying without cleanup." >&2
    sleep 30
    continue
  fi
  if [[ "${VM_STATUS}" != "RUNNING" ]]; then
    copy_progress || true
    echo "The profiling VM stopped before the profile completed. Partial artifacts remain in ${STAGING_OUTPUT}." >&2
    exit 2
  fi

  if ! REMOTE_SWEEPS="$(gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "sudo -n find '${REMOTE_OUTPUT}' -maxdepth 1 -name 'sweep-*-mhz.done' -print 2>/dev/null | sort" 2>/dev/null)"; then
    echo "Could not query completed sweeps; retrying without cleanup." >&2
    sleep 30
    continue
  fi
  if [[ "${REMOTE_SWEEPS}" != "${SEEN_SWEEPS}" ]]; then
    if copy_progress; then
      SEEN_SWEEPS="${REMOTE_SWEEPS}"
      if [[ -n "${SEEN_SWEEPS}" ]]; then
        echo "Copied completed clock sweep artifacts."
      fi
    else
      sleep 30
      continue
    fi
  fi

  if gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "sudo -n test -f '${REMOTE_OUTPUT}/profile.complete'" >/dev/null 2>&1; then
    if copy_progress; then
      break
    fi
    sleep 30
    continue
  fi
  if gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "sudo -n test -f '${REMOTE_OUTPUT}/profile.failed'" >/dev/null 2>&1; then
    if ! copy_progress; then
      sleep 30
      continue
    fi
    echo "The profiler failed. Inspect ${STAGING_OUTPUT}/driver.log and profile_manifest.json." >&2
    exit 2
  fi
  sleep 30
done

PYTHONPATH=code uv run --frozen python - \
  "${STAGING_OUTPUT}" \
  "$(pwd)/scripts/profile_inference_gpu.py" \
  "$(pwd)/scripts/run_inference_profile_gcp.sh" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from inference_serving import load_profile

root = Path(sys.argv[1]).resolve()
local_profiler = Path(sys.argv[2]).resolve()
local_launcher = Path(sys.argv[3]).resolve()
path = root / "profile_manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
if manifest.get("status") != "complete":
    raise SystemExit(f"profile manifest is not complete: {manifest.get('status')!r}")
for required in (
    "l4_profile.csv",
    "l4_profile_all_requested.csv",
    "l4_profile_raw.csv",
    "l4_telemetry.csv",
):
    if not (root / required).is_file():
        raise SystemExit(f"missing copied artifact: {required}")

profile = load_profile(root / "l4_profile.csv")
if not profile.measurement_validated:
    raise SystemExit("the staged L4 profile did not pass fail-closed validation")

checksums = manifest.get("sha256")
if not isinstance(checksums, dict) or not checksums:
    raise SystemExit("the completed manifest contains no declared checksums")
for name, expected in checksums.items():
    if Path(name).name != name:
        raise SystemExit(f"unsafe checksum path in manifest: {name!r}")
    artifact = root / name
    if not artifact.is_file():
        raise SystemExit(f"missing checksummed artifact: {name}")
    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"checksum mismatch for {name}")

profiler_digest = hashlib.sha256(local_profiler.read_bytes()).hexdigest()
if manifest.get("profiler_script_sha256") != profiler_digest:
    raise SystemExit("remote profiler script does not match the local pinned source")
launcher_digest = hashlib.sha256(local_launcher.read_bytes()).hexdigest()
if manifest.get("launcher_script_sha256") != launcher_digest:
    raise SystemExit("profile manifest does not match the local launcher source")

print(
    f"Verified {manifest['row_count']} measured request rows, "
    f"{len(checksums)} artifact checksums, and the profiler source hash."
)
PY

# Promote the verified manifest last so readers never mistake a partial sweep
# for a completed measured profile.
for completed_file in l4_profile.csv l4_profile_all_requested.csv l4_profile_raw.csv l4_telemetry.csv vllm.log driver.log bootstrap.log profile_manifest.json; do
  if [[ -f "${STAGING_OUTPUT}/${completed_file}" ]]; then
    mv "${STAGING_OUTPUT}/${completed_file}" "${LOCAL_OUTPUT}/${completed_file}"
  fi
done

echo "The verified measured artifacts are in ${LOCAL_OUTPUT}. The VM will now be deleted."
