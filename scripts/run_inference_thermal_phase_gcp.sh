#!/usr/bin/env bash
# Explicit maintainer workflow for the isolated L4 workload-phase confirmation run.
# It never promotes or replaces the completed full or thermal-identification profiles.

set -euo pipefail

PROJECT="potent-arcade-491015-g7"
ACCOUNT="pierreluc@carbonforge.ai"
VM_NAME="pierreluc-l4-rlbook-thermal-phase"
MACHINE_TYPE="g2-standard-8"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
REMOTE_SCRIPT="/var/tmp/profile_inference_gpu.py"
REMOTE_OUTPUT="/var/tmp/inference-serving-thermal-phase"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_OUTPUT="${1:-data/inference_serving/thermal-phase-identification-${RUN_STAMP}}"
STAGING_ROOT="${TMPDIR:-/private/tmp}"
STAGING_OUTPUT="${STAGING_ROOT%/}/rlbook-inference-thermal-phase-${VM_NAME}-${RUN_STAMP}"
MAX_RUNTIME_HOURS="2"
MAX_RUNTIME="2h"
ON_DEMAND_USD_PER_HOUR="0.853624312"
DISK_AND_MISC_HEADROOM_USD="0.50"
PRIOR_FAILED_TRIAL_USD="0.40"
PRIOR_COMPLETED_RUN_USD="1.60"
MAX_NEW_COMPUTE_USD="1.707248624"
EXPECTED_MAX_CUMULATIVE_USD="4.207248624"
MAX_EXPOSURE_USD="4.80"
SOURCE_GIT_REVISION="$(git rev-parse HEAD 2>/dev/null || true)"
LAUNCHER_SCRIPT_SHA256="$(shasum -a 256 "$0" | awk '{print $1}')"
READINESS_ATTEMPTS=90
READINESS_INTERVAL_S=10
SSH_READY_ATTEMPTS=30
SSH_READY_INTERVAL_S=5
STANDARD_ZONES=(
  "us-central1-b"
  "us-central1-c"
)

if [[ "${RLBOOK_RUN_THERMAL_PHASE_IDENTIFICATION:-}" != "YES" ]]; then
  echo "Set RLBOOK_RUN_THERMAL_PHASE_IDENTIFICATION=YES to authorize this bounded paid run." >&2
  exit 2
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required." >&2
  exit 2
fi

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)"
if [[ "${ACTIVE_ACCOUNT}" != "${ACCOUNT}" ]]; then
  echo "Authenticate gcloud as ${ACCOUNT} before starting the thermal run." >&2
  exit 2
fi
if [[ "$(gcloud config get-value project 2>/dev/null)" != "${PROJECT}" ]]; then
  echo "Set the active gcloud project to ${PROJECT} before starting the thermal run." >&2
  exit 2
fi

if [[ -e "${LOCAL_OUTPUT}" ]]; then
  echo "Local output already exists: ${LOCAL_OUTPUT}. Choose a new directory." >&2
  exit 2
fi

if [[ -n "$(gcloud compute instances list \
  --project "${PROJECT}" \
  --filter="name=${VM_NAME}" \
  --format='value(name)' 2>/dev/null)" ]]; then
  echo "${VM_NAME} already exists. Resolve it before creating a thermal run." >&2
  exit 2
fi
if [[ -n "$(gcloud compute disks list \
  --project "${PROJECT}" \
  --filter="name=${VM_NAME}" \
  --format='value(name)' 2>/dev/null)" ]]; then
  echo "A disk named ${VM_NAME} already exists. Resolve it before creating a thermal run." >&2
  exit 2
fi

ESTIMATED_MAX_EXPOSURE_USD="$(python3 - \
  "${MAX_RUNTIME_HOURS}" \
  "${ON_DEMAND_USD_PER_HOUR}" \
  "${DISK_AND_MISC_HEADROOM_USD}" \
  "${PRIOR_FAILED_TRIAL_USD}" \
  "${PRIOR_COMPLETED_RUN_USD}" \
  "${MAX_NEW_COMPUTE_USD}" <<'PY'
from decimal import Decimal
import sys

hours, hourly, headroom, failed, completed, maximum_compute = map(
    Decimal, sys.argv[1:]
)
computed_maximum = hours * hourly
if computed_maximum != maximum_compute:
    raise SystemExit(
        f"two-hour compute guard ${computed_maximum} does not equal "
        f"the fixed ${maximum_compute}"
    )
print(failed + completed + computed_maximum + headroom)
PY
)"
python3 - \
  "${ESTIMATED_MAX_EXPOSURE_USD}" \
  "${EXPECTED_MAX_CUMULATIVE_USD}" \
  "${MAX_EXPOSURE_USD}" <<'PY'
from decimal import Decimal
import sys

estimated, expected, ceiling = map(Decimal, sys.argv[1:])
if estimated != expected:
    raise SystemExit(
        f"guarded cumulative phase-run exposure ${estimated} does not equal "
        f"the fixed ${expected}"
    )
if estimated > ceiling:
    raise SystemExit(
        f"guarded cumulative thermal-run exposure ${estimated} exceeds "
        f"the ${ceiling} ceiling"
    )
PY

if SOURCE_STATUS_OUTPUT="$(git status --porcelain --untracked-files=all 2>/dev/null)"; then
  if [[ -n "${SOURCE_STATUS_OUTPUT}" ]]; then
    SOURCE_WORKTREE_STATE="dirty"
  else
    SOURCE_WORKTREE_STATE="clean"
  fi
else
  SOURCE_WORKTREE_STATE="unknown"
fi

SELECTED_ZONE=""
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
      echo "The thermal VM still exists; inspect it immediately to avoid charges." >&2
      status=3
    fi
    if gcloud compute disks describe "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${SELECTED_ZONE}" >/dev/null 2>&1; then
      echo "The thermal disk still exists; inspect it immediately to avoid charges." >&2
      status=3
    fi
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

create_thermal_vm() {
  local zone="$1"
  local create_output
  echo "Trying guarded STANDARD L4 in ${zone}..."
  if create_output="$(gcloud compute instances create "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${zone}" \
    --machine-type "${MACHINE_TYPE}" \
    --provisioning-model STANDARD \
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
    --labels purpose=rlbook-thermal-phase,owner=pierreluc,provisioning=standard \
    2>&1)"; then
    printf '%s\n' "${create_output}"
    return 0
  fi
  printf '%s\n' "${create_output}" >&2

  if gcloud compute instances describe "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${zone}" >/dev/null 2>&1; then
    echo "Creation failed ambiguously but a VM exists; deleting it." >&2
    gcloud compute instances delete "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${zone}" \
      --delete-disks=all \
      --quiet || true
    return 20
  fi
  if gcloud compute disks describe "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${zone}" >/dev/null 2>&1; then
    echo "Creation failed ambiguously but a disk exists; deleting it." >&2
    gcloud compute disks delete "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${zone}" \
      --quiet || true
    if gcloud compute disks describe "${VM_NAME}" \
      --project "${PROJECT}" \
      --zone "${zone}" >/dev/null 2>&1; then
      echo "The ambiguous thermal disk still exists; inspect it immediately." >&2
    fi
    return 20
  fi
  if grep -Eq \
    'ZONE_RESOURCE_POOL_EXHAUSTED|reason: stockout|does not have enough resources' \
    <<<"${create_output}"; then
    return 10
  fi
  echo "Non-capacity creation failure; refusing to alter project, account, or machine type." >&2
  return 20
}

echo "Maximum guarded cumulative exposure: USD ${ESTIMATED_MAX_EXPOSURE_USD}."
for zone in "${STANDARD_ZONES[@]}"; do
  if create_thermal_vm "${zone}"; then
    SELECTED_ZONE="${zone}"
    CLEANUP_ARMED=1
    break
  else
    create_status=$?
    if [[ "${create_status}" -ne 10 ]]; then
      exit "${create_status}"
    fi
  fi
done
if [[ -z "${SELECTED_ZONE}" ]]; then
  echo "No approved zone accepted the guarded Standard L4." >&2
  exit 2
fi

echo "Created ${VM_NAME} in ${SELECTED_ZONE}. Waiting for SSH..."
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

mkdir -p "${STAGING_OUTPUT}"

echo "Ensuring Docker and the NVIDIA container runtime are available..."
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

READY=0
for ((attempt = 1; attempt <= READINESS_ATTEMPTS; attempt++)); do
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
  echo "GPU/Docker initialization did not complete within the bounded wait." >&2
  exit 2
fi

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${SELECTED_ZONE}" \
  --ssh-flag "-o ConnectTimeout=10" \
  --command "sudo -n nvidia-ctk runtime configure --runtime=docker && sudo -n systemctl restart docker && sleep 5 && nvidia-smi -L && sudo -n bash -lc \"nvidia-container-cli info >>'${REMOTE_OUTPUT}/bootstrap.log' 2>&1\""

gcloud compute scp scripts/profile_inference_gpu.py \
  "${VM_NAME}:${REMOTE_SCRIPT}" \
  --project "${PROJECT}" \
  --zone "${SELECTED_ZONE}"

START_COMMAND="mkdir -p '${REMOTE_OUTPUT}'; rm -f '${REMOTE_OUTPUT}/thermal_phase.complete' '${REMOTE_OUTPUT}/thermal_phase.failed'; nohup python3 '${REMOTE_SCRIPT}' --mode thermal-phase-identification --output-directory '${REMOTE_OUTPUT}' --source-git-revision '${SOURCE_GIT_REVISION}' --source-worktree-state '${SOURCE_WORKTREE_STATE}' --launcher-script-sha256 '${LAUNCHER_SCRIPT_SHA256}' --cloud-project '${PROJECT}' --cloud-zone '${SELECTED_ZONE}' --provisioning-model STANDARD --machine-type '${MACHINE_TYPE}' >'${REMOTE_OUTPUT}/driver.log' 2>&1 </dev/null & thermal_pid=\$!; echo \$thermal_pid >'${REMOTE_OUTPUT}/thermal_phase.pid'"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${SELECTED_ZONE}" \
  --command "sudo -n bash -lc \"${START_COMMAND}\""

copy_one_if_present() {
  local remote_file="$1"
  local availability
  if ! availability="$(gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "sudo -n bash -lc \"if test -r '${REMOTE_OUTPUT}/${remote_file}'; then printf present; else printf absent; fi\"" \
    2>/dev/null)"; then
    return 1
  fi
  if [[ "${availability}" == "present" ]]; then
    gcloud compute scp \
      "${VM_NAME}:${REMOTE_OUTPUT}/${remote_file}" \
      "${STAGING_OUTPUT}/${remote_file}" \
      --project "${PROJECT}" \
      --zone "${SELECTED_ZONE}" >/dev/null
  fi
}

copy_progress() {
  local remote_file
  for remote_file in l4_thermal_phase_telemetry.csv l4_thermal_phase_requests.csv thermal_phase_manifest.json thermal_phase_vllm.log driver.log bootstrap.log thermal_phase.complete thermal_phase.failed; do
    if ! copy_one_if_present "${remote_file}"; then
      echo "Could not copy ${remote_file}; leaving the bounded VM running and retrying." >&2
      return 1
    fi
  done
  local marker_list
  if ! marker_list="$(gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "sudo -n find '${REMOTE_OUTPUT}' -maxdepth 1 -type f -name 'thermal-phase-block-*' -printf '%f\\n' 2>/dev/null | sort" \
    2>/dev/null)"; then
    return 1
  fi
  while IFS= read -r remote_file; do
    if [[ -n "${remote_file}" ]] && ! copy_one_if_present "${remote_file}"; then
      return 1
    fi
  done <<<"${marker_list}"
}

SEEN_BLOCKS=""
while true; do
  if ! VM_STATUS="$(gcloud compute instances describe "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --format='value(status)' 2>/dev/null)"; then
    echo "Could not read the thermal VM status; retrying without cleanup." >&2
    sleep 30
    continue
  fi
  if [[ "${VM_STATUS}" != "RUNNING" ]]; then
    copy_progress || true
    echo "The thermal VM stopped before completion. Partial data remain in ${STAGING_OUTPUT}." >&2
    exit 2
  fi

  if ! REMOTE_BLOCKS="$(gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "sudo -n find '${REMOTE_OUTPUT}' -maxdepth 1 -name 'thermal-phase-block-*.done' -print 2>/dev/null | sort" \
    2>/dev/null)"; then
    echo "Could not query thermal block checkpoints; retrying without cleanup." >&2
    sleep 30
    continue
  fi
  if [[ "${REMOTE_BLOCKS}" != "${SEEN_BLOCKS}" ]]; then
    if copy_progress; then
      SEEN_BLOCKS="${REMOTE_BLOCKS}"
      echo "Copied completed thermal block checkpoints."
    else
      sleep 30
      continue
    fi
  fi

  if gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "sudo -n test -f '${REMOTE_OUTPUT}/thermal_phase.complete'" >/dev/null 2>&1; then
    if copy_progress; then
      break
    fi
  elif gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT}" \
    --zone "${SELECTED_ZONE}" \
    --command "sudo -n test -f '${REMOTE_OUTPUT}/thermal_phase.failed'" >/dev/null 2>&1; then
    copy_progress || true
    echo "Thermal acquisition failed. Inspect ${STAGING_OUTPUT}/driver.log." >&2
    exit 2
  fi
  sleep 30
done

python3 - \
  "${STAGING_OUTPUT}" \
  "$(pwd)/scripts/profile_inference_gpu.py" \
  "$(pwd)/scripts/run_inference_thermal_phase_gcp.sh" <<'PY'
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
local_profiler = Path(sys.argv[2]).resolve()
local_launcher = Path(sys.argv[3]).resolve()
manifest = json.loads((root / "thermal_phase_manifest.json").read_text(encoding="utf-8"))
if (
    manifest.get("status") != "complete"
    or manifest.get("mode") != "thermal-phase-identification"
):
    raise SystemExit("thermal phase manifest is not a complete phase-identification run")
if (
    manifest.get("schema_version") != 2
    or manifest.get("protocol") != "cold-start-phase-pairs-v1"
):
    raise SystemExit("thermal phase manifest is not the fixed phase-pair protocol")
for required in ("l4_thermal_phase_telemetry.csv", "l4_thermal_phase_requests.csv"):
    if not (root / required).is_file():
        raise SystemExit(f"missing copied artifact: {required}")

expected = [
    ("phase_training_pulse_00", "training", "phase_training", 46, 75.0, "decode", 128, 32, 8),
    ("phase_training_pulse_01", "training", "phase_training", 61, 45.0, "prefill", 4096, 1, 8),
    ("phase_training_pulse_02", "training", "phase_training", 46, 75.0, "prefill", 4096, 1, 8),
    ("phase_training_pulse_03", "training", "phase_training", 61, 45.0, "decode", 128, 32, 8),
    ("phase_validation_pulse_00", "validation", "phase_validation", 55, 60.0, "decode", 128, 32, 8),
    ("phase_validation_pulse_01", "validation", "phase_validation", 55, 60.0, "prefill", 4096, 1, 8),
]
observed = [
    (
        block["block_id"],
        block["split"],
        block["sequence"],
        int(block["requested_power_limit_w"]),
        float(block["duration_s"]),
        block["condition"]["phase"],
        int(block["condition"]["prompt_tokens"]),
        int(block["condition"]["output_tokens"]),
        int(block["condition"]["concurrency"]),
    )
    for sequence in manifest["schedule"]
    for block in sequence["blocks"]
]
if observed != expected:
    raise SystemExit("thermal phase pulse order, split, or condition changed")
scheduled = [item[0] for item in expected]
completed = manifest.get("completed_block_ids", [])
if completed != scheduled:
    raise SystemExit("thermal phase completion order does not match the fixed schedule")
if [sequence["split"] for sequence in manifest["schedule"]] != [
    "training",
    "validation",
]:
    raise SystemExit("thermal phase train/validation split changed")
if not all(
    sequence.get("requires_cooldown_before_every_pulse") is True
    for sequence in manifest["schedule"]
):
    raise SystemExit("thermal schedule does not require a cooldown before every pulse")
cooldowns = manifest.get("cooldown_events")
if not isinstance(cooldowns, list) or [
    event.get("before_block_id") for event in cooldowns
] != scheduled:
    raise SystemExit("thermal cooldown events do not match every scheduled pulse")
if any(event.get("status") != "complete" for event in cooldowns):
    raise SystemExit("a thermal pulse lacks a completed cold-start cooldown")

checkpoints = manifest.get("block_checkpoints")
if not isinstance(checkpoints, list) or len(checkpoints) != len(scheduled):
    raise SystemExit("thermal manifest does not contain one checkpoint per block")
abort_temperature_c = float(manifest["safety_protocol"]["abort_temperature_c"])
for expected_block_id, checkpoint in zip(scheduled, checkpoints, strict=True):
    if checkpoint.get("status") != "complete":
        raise SystemExit(f"incomplete thermal checkpoint: {expected_block_id}")
    if checkpoint.get("block", {}).get("block_id") != expected_block_id:
        raise SystemExit("thermal checkpoint order does not match the schedule")
    if int(checkpoint.get("block_telemetry_rows", 0)) <= 0:
        raise SystemExit(f"thermal block has no telemetry: {expected_block_id}")
    if float(checkpoint.get("maximum_temperature_c", abort_temperature_c)) >= abort_temperature_c:
        raise SystemExit(f"thermal block reached the abort temperature: {expected_block_id}")

with (root / "l4_thermal_phase_telemetry.csv").open(newline="", encoding="utf-8") as stream:
    telemetry_data = list(csv.DictReader(stream))
telemetry_rows = len(telemetry_data)
with (root / "l4_thermal_phase_requests.csv").open(newline="", encoding="utf-8") as stream:
    request_rows = sum(1 for _ in csv.DictReader(stream))
if telemetry_rows != int(manifest.get("telemetry_row_count", -1)):
    raise SystemExit("thermal telemetry row count does not match the manifest")
if request_rows != int(manifest.get("request_row_count", -1)):
    raise SystemExit("thermal request row count does not match the manifest")
observed_block_ids = {row.get("block_id") for row in telemetry_data}
missing_block_ids = [block_id for block_id in scheduled if block_id not in observed_block_ids]
if missing_block_ids:
    raise SystemExit(f"scheduled thermal blocks lack labeled telemetry: {missing_block_ids}")
temperatures = [float(row["temperature_c"]) for row in telemetry_data]
if not temperatures or any(not math.isfinite(value) for value in temperatures):
    raise SystemExit("thermal telemetry temperatures are absent or non-finite")
if max(temperatures) >= abort_temperature_c:
    raise SystemExit("thermal telemetry reached the abort temperature")

checksums = manifest.get("sha256")
if not isinstance(checksums, dict) or not checksums:
    raise SystemExit("thermal manifest has no artifact checksums")
for name, expected in checksums.items():
    if Path(name).name != name:
        raise SystemExit(f"unsafe checksum path: {name!r}")
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"checksum mismatch for {name}")
if manifest.get("profiler_script_sha256") != hashlib.sha256(local_profiler.read_bytes()).hexdigest():
    raise SystemExit("remote thermal profiler does not match local source")
if manifest.get("launcher_script_sha256") != hashlib.sha256(local_launcher.read_bytes()).hexdigest():
    raise SystemExit("thermal manifest does not match the local launcher")
print(f"Verified {len(completed)} thermal blocks and {telemetry_rows} telemetry rows.")
PY

mkdir -p "${LOCAL_OUTPUT}"
for completed_file in l4_thermal_phase_telemetry.csv l4_thermal_phase_requests.csv thermal_phase_vllm.log driver.log bootstrap.log thermal_phase.complete thermal_phase_manifest.json; do
  if [[ -f "${STAGING_OUTPUT}/${completed_file}" ]]; then
    mv "${STAGING_OUTPUT}/${completed_file}" "${LOCAL_OUTPUT}/${completed_file}"
  fi
done
for marker in "${STAGING_OUTPUT}"/thermal-phase-block-*; do
  if [[ -f "${marker}" ]]; then
    mv "${marker}" "${LOCAL_OUTPUT}/$(basename "${marker}")"
  fi
done

echo "Verified thermal-phase-identification data are in ${LOCAL_OUTPUT}. The VM will now be deleted."
