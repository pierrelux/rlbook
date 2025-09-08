from dataclasses import dataclass
import math, random

# ---------------------------
# PROGRAM = "HTTP retrier with backoff"
# ---------------------------

@dataclass
class State:
    t: float            # wall-clock time (s)
    k: int              # attempt index
    done: bool          # success flag
    code: int | None    # last HTTP code or None
    jitter: float       # per-run jitter (simulates clock/socket noise)

# Controls (decision variables): per-step wait times (backoff schedule)
# u[k] can be optimized; in a fixed policy you'd set u[k] = base * gamma**k
# We'll keep them bounded for realism.
def clamp(x, lo, hi): return max(lo, min(hi, x))

# Simulated environment: availability is time-varying (spiky outage)
def server_success_prob(t: float) -> float:
    # Low availability for the first 2 seconds, then rebounds
    base = 0.15 if t < 2.0 else 0.85
    # Some diurnal-like wobble (toy)
    wobble = 0.1 * math.sin(2 * math.pi * (t / 3.0))
    return clamp(base + wobble, 0.01, 0.99)

def http_request():
    # Just returns a code; success = 200, failure = 503
    return 200 if random.random() < 0.5 else 503

# -------- DOCP ingredients --------
# State x_k = (t, k, done, code, jitter)
# Control u_k = wait time before next attempt (our backoff schedule entry)
# Transition Phi_k: one "program step" = (optional wait) + (one request) + (branch)
def Phi(state: State, u_k: float) -> State:
    if state.done:
        # No-ops after success (absorbing state)
        return State(state.t, state.k, True, state.code, state.jitter)

    # 1) Wait according to control (backoff schedule) + jitter
    wait = clamp(u_k + 0.02 * state.jitter, 0.0, 3.0)
    t = state.t + wait

    # 2) Environment: success probability depends on time t
    p = server_success_prob(t)

    # 3) "Perform request": success with prob p; otherwise 503
    code = 200 if random.random() < p else 503
    done = (code == 200)

    # 4) Advance attempt counter and wall clock
    return State(t=t, k=state.k + 1, done=done, code=code, jitter=state.jitter)

# Stage cost: latency penalty each step; heavy penalty if still failing late
def stage_cost(state: State, u_k: float) -> float:
    # Latency/energy per unit wait + small per-step overhead when not done
    return 0.20 * u_k + (0.00 if state.done else 0.002)

# Terminal cost: if failed after horizon, big penalty; if succeeded, pay total time
def terminal_cost(state: State, max_attempts: int) -> float:
    # Pay for elapsed time; fail late incurs extra penalty
    return 0.3 * state.t + (5.0 if (not state.done and state.k >= max_attempts) else 0.0)

def rollout(u, max_attempts=8, seed=0):
    random.seed(seed)
    s = State(t=0.0, k=0, done=False, code=None, jitter=random.uniform(-1,1))
    J = 0.0
    for k in range(max_attempts):
        J += stage_cost(s, u[k])
        s = Phi(s, u[k])
        if s.done:  # early stop like a real program
            break
    J += terminal_cost(s, max_attempts)
    return J, s  # return final state for debugging if needed

# ---------- helpers for SPSA with common random numbers ----------
def eval_policy(u, seeds, max_attempts=8):
    # Average over a fixed set of seeds (CRN helps SPSA a lot)
    Js = []
    for sd in seeds:
        J, _ = rollout(u, max_attempts=max_attempts, seed=sd)
        Js.append(J)
    return sum(Js) / len(Js)

def project_waits(u):
    # Keep waits in [0, 3] for realism
    return [max(0.0, min(3.0, x)) for x in u]

# ---------- schedule parameterizations ----------
def schedule_exp(base, gamma, K):
    # u[k] = base * gamma**k
    return [base * (gamma ** k) for k in range(K)]

# If you prefer per-step but monotone nonnegative waits, use softplus increments:
def schedule_softplus(z, K):
    # z in R^K -> u monotone via cumulative softplus increments
    def softplus(x):
        return math.log1p(math.exp(-abs(x))) + max(x, 0.0)
    inc = [softplus(zi) for zi in z]
    u = []
    s_accum = 0.0
    for i in range(K):
        s_accum += inc[i]
        u.append(s_accum)
    return u

# ---------------------------
# Black-box optimization (SPSA) of the schedule u[0:K]
# ---------------------------
def spsa_optimize(K=8, iters=200, seed=0):
    random.seed(seed)
    # Initialize a conservative schedule (small linear backoff)
    u = [0.05 + 0.1*k for k in range(K)]
    alpha = 0.2      # learning rate
    c0 = 0.1         # perturbation scale
    for t in range(1, iters+1):
        c = c0 / (t ** 0.101)
        # Rademacher perturbation
        delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(K)]
        u_plus  = [clamp(u[i] + c * delta[i], 0.0, 3.0) for i in range(K)]
        u_minus = [clamp(u[i] - c * delta[i], 0.0, 3.0) for i in range(K)]

        Jp, _ = rollout(u_plus, seed=seed + 10*t + 1)
        Jm, _ = rollout(u_minus, seed=seed + 10*t + 2)

        # SPSA gradient estimate
        g = [(Jp - Jm) / (2.0 * c * delta[i]) for i in range(K)]
        # Update (project back to bounds)
        u = [clamp(u[i] - alpha * g[i], 0.0, 3.0) for i in range(K)]
    return u

# ---------- SPSA over 2 parameters (base, gamma) with CRN ----------
def spsa_optimize_exp(K=8, iters=200, seed=0, Nmc=16):
    random.seed(seed)
    # fixed seeds reused every iteration (CRN)
    seeds = [seed + 1000 + i for i in range(Nmc)]

    # init: small base, mild growth
    base, gamma = 0.05, 1.4
    alpha0, c0 = 0.15, 0.2  # learning rate and perturbation scales

    for t in range(1, iters + 1):
        a_t = alpha0 / (t ** 0.602)   # standard SPSA decay
        c_t = c0 / (t ** 0.101)

        # Rademacher perturbations for 2 params
        d_base = 1.0 if random.random() < 0.5 else -1.0
        d_gamma = 1.0 if random.random() < 0.5 else -1.0

        base_plus  = base  + c_t * d_base
        base_minus = base  - c_t * d_base
        gamma_plus  = gamma + c_t * d_gamma
        gamma_minus = gamma - c_t * d_gamma

        u_plus  = project_waits(schedule_exp(base_plus,  gamma_plus,  K))
        u_minus = project_waits(schedule_exp(base_minus, gamma_minus, K))

        Jp = eval_policy(u_plus, seeds, max_attempts=K)
        Jm = eval_policy(u_minus, seeds, max_attempts=K)

        # SPSA gradient estimate
        g_base  = (Jp - Jm) / (2.0 * c_t * d_base)
        g_gamma = (Jp - Jm) / (2.0 * c_t * d_gamma)

        # Update
        base  = max(0.0, base  - a_t * g_base)
        gamma = max(0.5, gamma - a_t * g_gamma)  # keep reasonable

    return base, gamma

if __name__ == "__main__":
    K = 8
    # Baseline linear schedule
    u0 = [0.05 + 0.1*k for k in range(K)]
    J0, s0 = rollout(u0, seed=42)

    # Optimize per-step waits (K-dim SPSA)
    u_opt = spsa_optimize(K=K, iters=200, seed=123)
    J1, s1 = rollout(u_opt, seed=999)

    # Optimize exponential schedule parameters (2-dim SPSA with CRN)
    base_opt, gamma_opt = spsa_optimize_exp(K=K, iters=200, seed=321, Nmc=16)
    u_exp = project_waits(schedule_exp(base_opt, gamma_opt, K))
    J2, s2 = rollout(u_exp, seed=777)

    print("Initial schedule:", [round(x,3) for x in u0], "  Cost ≈", round(J0,3))
    print("Optimized (per-step SPSA):", [round(x,3) for x in u_opt], "  Cost ≈", round(J1,3))
    print("Optimized (exp base, gamma): base=", round(base_opt,3), " gamma=", round(gamma_opt,3),
          "  schedule=", [round(x,3) for x in u_exp], "  Cost ≈", round(J2,3))
    print("Attempts (init → per-step → exp):", s0.k, "→", s1.k, "→", s2.k,
          "  Success codes:", s0.code, s1.code, s2.code)
