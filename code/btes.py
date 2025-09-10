import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import spsolve
import cvxpy as cp
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

class BTESModel:
    """
    Borehole Thermal Energy Storage Model with MPC Control
    Based on van Randenborgh et al. (2025)
    """

    def __init__(self, nx=47, ny=47, num_bhe=9, num_segments=3, dt=2.0, domain_scale=3.0, clip_temps=False):
        # Grid parameters
        # Scale grid and domain consistently to push boundaries away while preserving dx, dy
        self.nx = int(round(nx * domain_scale))
        self.ny = int(round(ny * domain_scale))
        self.num_cells = self.nx * self.ny

        # Domain size (scaled to push boundaries away)
        # Paper nominal is 20m x 20m; increase by domain_scale to reduce boundary influence
        self.Lx = 20.0 * domain_scale  # m
        self.Ly = 20.0 * domain_scale  # m
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny

        # Time parameters
        self.dt = float(dt)  # seconds
        self.clip_temps = clip_temps

        # Ground and fluid properties
        self.cg = 2.30e6  # J/m³K - volumetric heat capacity of ground
        self.cw_ground = 4.20e6  # J/m³K - volumetric heat capacity of groundwater
        self.cp_fluid = 4.18e3   # J/kgK - specific heat capacity of process fluid (water)
        self.lambda_g = 2.3  # W/mK - heat conduction coefficient
        self.phi = 0.8  # porosity
        self.T_ambient = 295.15  # K (22°C)

        # Groundwater flow (5 mm/h from SW to NE)
        self.vx = 1.39e-6  # m/s
        self.vy = 1.39e-6  # m/s

        # BHE parameters
        self.num_bhe = num_bhe
        self.num_segments = num_segments
        self.segment_length = 3.66  # m
        self.q_flow = 197.4e-3  # kg/s - mass flow rate

        # Thermal capacities and resistances (from Bauer et al. methodology)
        self.Cw = 2.45e3  # J/mK - fluid thermal capacity per unit length
        self.Cb = 20.36e3  # J/mK - backfill thermal capacity per unit length
        self.Rfb = 261e-3  # mK/W - fluid-backfill resistance
        self.Rbb = 453.87e-3  # mK/W - backfill-backfill resistance
        self.Rgb = 69.31e-3  # mK/W - ground-backfill resistance

        # BHE positions (3x3 grid with 2m spacing)
        self.bhe_positions = self._setup_bhe_positions()

        # Initialize state vectors
        self.setup_state_space()

    def _setup_bhe_positions(self):
        """Setup BHE positions in a 3x3 grid"""
        positions = []
        center_x, center_y = self.Lx/2, self.Ly/2
        spacing = 2.0  # m

        for i in range(3):
            for j in range(3):
                x = center_x + (i-1) * spacing
                y = center_y + (j-1) * spacing
                # Convert to grid indices
                ix = int(x / self.dx)
                iy = int(y / self.dy)
                positions.append((ix, iy))

        return positions[:self.num_bhe]

    def setup_state_space(self):
        """Setup the complete state-space system"""
        # State vector components:
        # - Ground temperatures: num_cells states
        # - BHE temperatures: num_bhe * num_segments * 4 states (Tf0, Tf1, Tb0, Tb1)
        # - APU temperatures: 2 states (Tin, Tout)

        self.n_ground = self.num_cells
        self.n_bhe = self.num_bhe * self.num_segments * 4
        self.n_apu = 2
        self.n_states = self.n_ground + self.n_bhe + self.n_apu

        # Initialize state vector with ambient temperature
        self.x = np.ones(self.n_states) * self.T_ambient

        # Build system matrices
        self.build_system_matrices()

        # Run a few initialization steps to stabilize
        for _ in range(10):
            self.x = self.A @ self.x + self.f

    def build_system_matrices(self):
        """Build the A, B, and f matrices for the state-space model"""
        # Initialize matrices
        self.A = np.zeros((self.n_states, self.n_states))
        self.B = np.zeros((self.n_states, 1))
        self.f = np.zeros(self.n_states)

        # 1. Ground model matrix (finite volume discretization)
        self.build_ground_matrix()

        # 2. BHE model matrices
        self.build_bhe_matrices()

        # 3. APU model matrix
        self.build_apu_matrix()

        # 4. Coupling between models
        self.add_coupling_terms()

    def build_ground_matrix(self):
        """Build the ground heat transport matrix using finite volume method"""
        # Indices for ground states
        idx_start = 0
        idx_end = self.n_ground

        # Build discretization matrix with improved numerical stability
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny + j

                # Boundary cells - Dirichlet condition
                if i == 0 or i == self.nx-1 or j == 0 or j == self.ny-1:
                    self.A[idx, idx] = 1.0
                    self.f[idx] = self.T_ambient
                else:
                    # Interior cells - heat equation discretization
                    # Diffusion terms (with stability check)
                    diff_x = self.lambda_g * self.dt / (self.cg * self.dx**2)
                    diff_y = self.lambda_g * self.dt / (self.cg * self.dy**2)

                    # Limit diffusion for stability (CFL-like)
                    max_diff = 0.25
                    diff_x = min(diff_x, max_diff)
                    diff_y = min(diff_y, max_diff)

                    # Advection terms (upwind scheme)
                    adv_x_mag = abs(self.cw_ground * self.phi * self.vx * self.dt / (self.cg * self.dx))
                    adv_y_mag = abs(self.cw_ground * self.phi * self.vy * self.dt / (self.cg * self.dy))

                    # Central cell (ensure diagonal dominance, include advection losses)
                    self.A[idx, idx] = 1 - 2*diff_x - 2*diff_y - adv_x_mag - adv_y_mag

                    # Neighboring cells (with upwind for advection)
                    # West/East diffusion contributions
                    if i > 0:
                        self.A[idx, idx-self.ny] = self.A[idx, idx-self.ny] + diff_x
                    if i < self.nx-1:
                        self.A[idx, idx+self.ny] = self.A[idx, idx+self.ny] + diff_x

                    # South/North diffusion contributions
                    if j > 0:
                        self.A[idx, idx-1] = self.A[idx, idx-1] + diff_y
                    if j < self.ny-1:
                        self.A[idx, idx+1] = self.A[idx, idx+1] + diff_y

                    # Upwind advection contributions
                    if self.vx > 0 and i > 0:  # flow west -> east: take from West
                        self.A[idx, idx-self.ny] = self.A[idx, idx-self.ny] + adv_x_mag
                    elif self.vx < 0 and i < self.nx-1:  # flow east -> west: take from East
                        self.A[idx, idx+self.ny] = self.A[idx, idx+self.ny] + adv_x_mag

                    if self.vy > 0 and j > 0:  # flow south -> north: take from South
                        self.A[idx, idx-1] = self.A[idx, idx-1] + adv_y_mag
                    elif self.vy < 0 and j < self.ny-1:  # flow north -> south: take from North
                        self.A[idx, idx+1] = self.A[idx, idx+1] + adv_y_mag

    def build_bhe_matrices(self):
        """Build BHE thermal resistance-capacitance model matrices"""
        for bhe_idx in range(self.num_bhe):
            for seg in range(self.num_segments):
                # State indices for this BHE segment
                # Order: Tf0, Tf1, Tb0, Tb1
                base_idx = self.n_ground + bhe_idx * self.num_segments * 4 + seg * 4

                # Tf0 (descending fluid) equation
                idx_tf0 = base_idx + 0
                idx_tf1 = base_idx + 1
                idx_tb0 = base_idx + 2
                idx_tb1 = base_idx + 3

                # Fluid equations
                alpha_f = self.dt / self.Cw
                beta_f = self.q_flow * self.cp_fluid * self.dt / (self.segment_length * self.Cw)

                # Tf0 equation
                self.A[idx_tf0, idx_tf0] = 1 - alpha_f/self.Rfb - beta_f
                self.A[idx_tf0, idx_tb0] = alpha_f/self.Rfb

                # Flow from previous segment
                if seg > 0:
                    prev_tf0 = base_idx - 4
                    self.A[idx_tf0, prev_tf0] = beta_f

                # Tf1 (ascending fluid) equation
                self.A[idx_tf1, idx_tf1] = 1 - alpha_f/self.Rfb - beta_f
                self.A[idx_tf1, idx_tb1] = alpha_f/self.Rfb

                # Flow from next segment (or bottom U-turn)
                if seg < self.num_segments - 1:
                    next_tf1 = base_idx + 4 + 1
                    self.A[idx_tf1, next_tf1] = beta_f
                else:
                    # U-turn at bottom: ascending gets from descending
                    self.A[idx_tf1, idx_tf0] = beta_f

                # Backfill equations
                alpha_b = self.dt / self.Cb

                # Tb0 equation
                self.A[idx_tb0, idx_tb0] = 1 - alpha_b/self.Rfb - alpha_b/self.Rbb - alpha_b/self.Rgb
                self.A[idx_tb0, idx_tf0] = alpha_b/self.Rfb
                self.A[idx_tb0, idx_tb1] = alpha_b/self.Rbb
                # Ground coupling added later

                # Tb1 equation
                self.A[idx_tb1, idx_tb1] = 1 - alpha_b/self.Rfb - alpha_b/self.Rbb - alpha_b/self.Rgb
                self.A[idx_tb1, idx_tf1] = alpha_b/self.Rfb
                self.A[idx_tb1, idx_tb0] = alpha_b/self.Rbb
                # Ground coupling added later

    def build_apu_matrix(self):
        """Build APU (auxiliary power unit) model matrix"""
        # APU states are at the end
        idx_tin = self.n_states - 2
        idx_tout = self.n_states - 1

        # Initialize APU states properly
        self.x[idx_tin] = self.T_ambient
        self.x[idx_tout] = self.T_ambient

        # Simple model: Tin = Tout + u/(nu*q*cp)
        self.A[idx_tin, idx_tin] = 0.9  # Some thermal inertia
        self.A[idx_tin, idx_tout] = 0.1
        self.B[idx_tin, 0] = 10 * self.dt / (self.num_bhe * self.q_flow * self.cp_fluid)  # Scale factor [K/W] - increased gain

        # Tout comes from BHE outlets (ascending fluid at top segment)
        self.A[idx_tout, idx_tout] = 0.1  # Some retention
        for bhe_idx in range(self.num_bhe):
            # Top segment ascending fluid
            idx_tf1_top = self.n_ground + bhe_idx * self.num_segments * 4 + 1
            if idx_tf1_top < self.n_states:
                self.A[idx_tout, idx_tf1_top] = 0.9 / self.num_bhe

    def add_coupling_terms(self):
        """Add coupling between ground and BHE models"""
        for bhe_idx, (ix, iy) in enumerate(self.bhe_positions):
            # Ground cell index
            ground_idx = ix * self.ny + iy

            # Average temperature from neighboring ground cells
            neighbors = []
            # Cardinal neighbors
            if ix > 0: neighbors.append((ix-1) * self.ny + iy)
            if ix < self.nx-1: neighbors.append((ix+1) * self.ny + iy)
            if iy > 0: neighbors.append(ix * self.ny + (iy-1))
            if iy < self.ny-1: neighbors.append(ix * self.ny + (iy+1))
            # Diagonal neighbors (improve cylindrical approximation)
            if ix > 0 and iy > 0: neighbors.append((ix-1) * self.ny + (iy-1))
            if ix > 0 and iy < self.ny-1: neighbors.append((ix-1) * self.ny + (iy+1))
            if ix < self.nx-1 and iy > 0: neighbors.append((ix+1) * self.ny + (iy-1))
            if ix < self.nx-1 and iy < self.ny-1: neighbors.append((ix+1) * self.ny + (iy+1))

            for seg in range(self.num_segments):
                base_idx = self.n_ground + bhe_idx * self.num_segments * 4 + seg * 4
                idx_tb0 = base_idx + 2
                idx_tb1 = base_idx + 3

                # Backfill-ground coupling
                alpha_b = self.dt / self.Cb
                for n_idx in neighbors:
                    self.A[idx_tb0, n_idx] = alpha_b / (self.Rgb * len(neighbors))
                    self.A[idx_tb1, n_idx] = alpha_b / (self.Rgb * len(neighbors))

                # Ground-BHE heat flux
                heat_flux_coeff = self.dt / (self.cg * self.dx * self.dy * self.Rgb)
                self.A[ground_idx, idx_tb0] += heat_flux_coeff
                self.A[ground_idx, idx_tb1] += heat_flux_coeff
                self.A[ground_idx, ground_idx] -= 2 * heat_flux_coeff

        # Connect APU inlet to BHE inlets
        idx_tin = self.n_states - 2
        for bhe_idx in range(self.num_bhe):
            # Top segment descending fluid gets from APU
            idx_tf0_top = self.n_ground + bhe_idx * self.num_segments * 4
            self.A[idx_tf0_top, idx_tin] = self.q_flow * self.cp_fluid * self.dt / (self.segment_length * self.Cw)

    def simulate_step(self, u):
        """Simulate one time step with control input u"""
        # Apply state update
        self.x = self.A @ self.x + self.B.flatten() * u + self.f

        # Optional clipping for numerical safety (disabled by default)
        if self.clip_temps:
            self.x = np.clip(self.x, 263.15, 353.15)

        # Enforce physical bounds on APU temperatures (relaxed clamps)
        idx_tin = self.n_states - 2
        idx_tout = self.n_states - 1
        # Allow Tin to go as low as -10°C to enable negative power tracking
        self.x[idx_tin] = np.clip(self.x[idx_tin], 263.15, 323.15)  # -10°C to 50°C
        self.x[idx_tout] = np.clip(self.x[idx_tout], 263.15, 323.15)

        # Ensure APU temperatures are properly updated if they're NaN
        idx_tin = self.n_states - 2
        idx_tout = self.n_states - 1

        if np.isnan(self.x[idx_tout]) or self.x[idx_tout] <= 0:
            # Set outlet as average of BHE outlets
            temps = []
            for bhe_idx in range(self.num_bhe):
                idx_tf1_top = self.n_ground + bhe_idx * self.num_segments * 4 + 1
                if idx_tf1_top < self.n_states and not np.isnan(self.x[idx_tf1_top]):
                    temps.append(self.x[idx_tf1_top])
            self.x[idx_tout] = np.mean(temps) if temps else self.T_ambient

        if np.isnan(self.x[idx_tin]) or self.x[idx_tin] <= 0:
            # Inlet temperature based on outlet plus control input
            self.x[idx_tin] = self.x[idx_tout] + u * self.dt / (self.num_bhe * self.q_flow * self.cp_fluid)

        return self.x.copy()

    def get_ground_temperature_field(self):
        """Extract 2D ground temperature field from state vector"""
        ground_temps = self.x[:self.n_ground]
        return ground_temps.reshape(self.nx, self.ny)

    def get_outlet_temperature(self):
        """Get BTES outlet temperature"""
        # Outlet is the last state (APU Tout)
        if not np.isnan(self.x[-1]) and self.x[-1] > 0:
            return self.x[-1]
        else:
            # Fallback: average of BHE outlet temperatures
            temps = []
            for bhe_idx in range(self.num_bhe):
                # Top segment ascending fluid (outlet)
                idx_tf1_top = self.n_ground + bhe_idx * self.num_segments * 4 + 1
                if idx_tf1_top < len(self.x):
                    temps.append(self.x[idx_tf1_top])
            return np.mean(temps) if temps else self.T_ambient

    def get_inlet_temperature(self):
        """Get BTES inlet temperature"""
        # Inlet is the second-to-last state (APU Tin)
        if not np.isnan(self.x[-2]) and self.x[-2] > 0:
            return self.x[-2]
        else:
            # Fallback: average of BHE inlet temperatures
            temps = []
            for bhe_idx in range(self.num_bhe):
                # Top segment descending fluid (inlet)
                idx_tf0_top = self.n_ground + bhe_idx * self.num_segments * 4
                if idx_tf0_top < len(self.x):
                    temps.append(self.x[idx_tf0_top])
            return np.mean(temps) if temps else self.T_ambient

    def get_delivered_power(self):
        """Compute delivered thermal power y = nu * q * cp * (Tout - Tin) [W]."""
        idx_tin = self.n_states - 2
        idx_tout = self.n_states - 1
        tin = self.x[idx_tin]
        tout = self.x[idx_tout]
        # Fallbacks if needed
        if np.isnan(tout) or np.isnan(tin) or tout <= 0 or tin <= 0:
            tout = self.get_outlet_temperature()
            tin = self.get_inlet_temperature()
        return self.num_bhe * self.q_flow * self.cp_fluid * (tout - tin)


class MPCController:
    """Model Predictive Controller for BTES"""

    def __init__(self, model, horizon=80):
        self.model = model
        self.horizon = horizon
        self.R = 0.1  # Tracking weight
        self.Q = 0.01  # Input change weight

        # Control limits - increased range
        self.u_min = -10000.0  # W
        self.u_max = 10000.0   # W

        # Temperature constraints (relaxed for numerical stability)
        self.T_min = 273.15  # 0°C
        self.T_max = 323.15  # 50°C

    def solve(self, x0, y_ref_sequence, u_prev=0):
        """Solve MPC optimization problem"""
        try:
            # Decision variables
            u = cp.Variable(self.horizon)

            # Objective function
            obj = 0
            for k in range(self.horizon):
                # Tracking cost
                obj += self.R * cp.square(u[k] - y_ref_sequence[k])

                # Input change cost
                if k == 0:
                    obj += self.Q * cp.square(u[k] - u_prev)
                else:
                    obj += self.Q * cp.square(u[k] - u[k-1])

            # Constraints - only input constraints for stability
            constraints = []
            constraints += [u >= self.u_min, u <= self.u_max]

            # Solve
            prob = cp.Problem(cp.Minimize(obj), constraints)

            # Try to solve with OSQP first, fallback to ECOS if needed
            try:
                prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
            except:
                prob.solve(solver=cp.ECOS, verbose=False)

            if prob.status in ['optimal', 'optimal_inaccurate']:
                return u.value[0], u.value
            else:
                print(f"MPC solve status: {prob.status}, using fallback")
                # Fallback: simple proportional control
                error = y_ref_sequence[0] - u_prev
                u_fallback = np.clip(u_prev + 0.1 * error, self.u_min, self.u_max)
                return u_fallback, np.ones(self.horizon) * u_fallback

        except Exception as e:
            print(f"MPC solve exception: {e}, using fallback")
            # Fallback control
            return y_ref_sequence[0], np.ones(self.horizon) * y_ref_sequence[0]


def run_simulation(debug_log=True, log_every_hours=1, log_first_minutes=10):
    """Run complete BTES simulation with MPC control"""

    # Initialize model with paper-like discretization and smaller timestep
    print("Initializing BTES model...")
    model = BTESModel(nx=47, ny=47, num_bhe=9, num_segments=3, dt=2.0, domain_scale=1.0, clip_temps=False)

    # Initialize MPC controller
    print("Initializing MPC controller...")
    mpc = MPCController(model, horizon=20)

    # Simulation parameters
    hours = 24
    steps_per_hour = int(round(3600.0 / model.dt))
    total_steps = hours * steps_per_hour
    log_every_steps = max(1, int(log_every_hours * steps_per_hour))
    log_first_steps = int((log_first_minutes * 60) / model.dt)

    # Generate reference demand profile (varying between -500W and -1000W)
    # Negative means heat extraction from BTES
    t_hours = np.linspace(0, hours, total_steps)
    y_ref = -750 - 250 * np.sin(2 * np.pi * t_hours / 12)  # Daily variation

    # Storage for results
    u_history = []
    y_history = []  # delivered power
    T_out_history = []
    T_in_history = []
    ground_temp_snapshots = []
    snapshot_times = [0, 6, 12, 18, 24]  # hours
    snapshot_steps = set(int(t * steps_per_hour) for t in snapshot_times)

    # Initialize temperatures
    T_out_history.append(model.T_ambient)
    T_in_history.append(model.T_ambient)

    # Pre-heat the storage (as in the paper's 26 h heating phase)
    preheat_hours = 0 # Changed from 26 to 0
    preheat_steps = preheat_hours * steps_per_hour
    if preheat_steps > 0:
        print(f"Preheating for {preheat_hours} hours...")
        for k in range(preheat_steps):
            # constant heating power (positive injects heat)
            model.simulate_step(u=800.0)

    # Run simulation
    print(f"Running {hours}-hour simulation...")
    u_prev = 0
    # Simple outer-loop PI on delivered power (with basic anti-windup)
    pi_integral = 0.0
    Kp = 0.05
    Ki = 0.001

    for step in range(total_steps):
        if step % steps_per_hour == 0:
            print(f"  Hour {step//steps_per_hour}/{hours}")

        # Get current state
        x_current = model.x.copy()

        # Create reference sequence for MPC horizon
        ref_indices = np.arange(step, min(step + mpc.horizon, total_steps))
        y_ref_sequence = y_ref[ref_indices]
        if len(y_ref_sequence) < mpc.horizon:
            y_ref_sequence = np.pad(y_ref_sequence, (0, mpc.horizon - len(y_ref_sequence)),
                                   mode='constant', constant_values=y_ref[-1])

        # Outer-loop PI correction based on current delivered power
        y_meas = model.get_delivered_power()
        # Define error so that positive error => y too high
        e_y = y_meas - y_ref_sequence[0]
        # PI control with sign that reduces y when it's too high
        u_pi = -Kp * e_y - Ki * pi_integral

        # Solve MPC
        u_opt, _ = mpc.solve(x_current, y_ref_sequence, u_prev)

        # Combine MPC with PI correction and apply
        u_cmd_pre = u_opt + u_pi
        u_cmd = np.clip(u_cmd_pre, mpc.u_min, mpc.u_max)
        # Anti-windup: only integrate when not saturated
        if np.isclose(u_cmd, u_cmd_pre):
            pi_integral += e_y * model.dt
        model.simulate_step(u_cmd)

        # Record results
        u_history.append(u_cmd)
        y_delivered = model.get_delivered_power()
        y_history.append(y_delivered)

        # Get temperatures with validation
        t_out = model.get_outlet_temperature()
        t_in = model.get_inlet_temperature()

        # Ensure valid temperatures
        if np.isnan(t_out) or t_out <= 0:
            t_out = T_out_history[-1] if T_out_history else model.T_ambient
        if np.isnan(t_in) or t_in <= 0:
            t_in = T_in_history[-1] if T_in_history else model.T_ambient

        T_out_history.append(t_out)
        T_in_history.append(t_in)

        # Save ground temperature snapshots at fixed times
        current_hour = step / steps_per_hour
        if step in snapshot_steps:
            ground_temp_snapshots.append({
                'time': current_hour,
                'field': model.get_ground_temperature_field().copy()
            })

        u_prev = u_opt

        # Debug logging
        if debug_log and (step < log_first_steps or step % log_every_steps == 0):
            idx_tin = model.n_states - 2
            idx_tout = model.n_states - 1
            tin = model.x[idx_tin]
            tout = model.x[idx_tout]
            deltaT = tout - tin
            y_now = y_delivered
            m_dot_total = model.num_bhe * model.q_flow
            y_upper = m_dot_total * model.cp_fluid * (323.15 - tout)
            y_lower = m_dot_total * model.cp_fluid * (263.15 - tout)
            B_tin = model.B[idx_tin, 0]
            clamp_low = np.isclose(tin, 263.15, atol=1e-6)
            clamp_high = np.isclose(tin, 323.15, atol=1e-6)
            print(
                f"    t={step/steps_per_hour:.2f} h | y_ref={y_ref_sequence[0]:7.1f} W, y={y_now:7.1f} W, ΔT={deltaT:5.3f} K, "
                f"Tin={tin-273.15:5.2f}°C, Tout={tout-273.15:5.2f}°C | u_opt={u_opt:7.1f} W, u_PI={u_pi:7.1f} W, u_cmd={u_cmd:7.1f} W | "
                f"B_tin={B_tin:.3e} K/W | clamp_low={clamp_low} clamp_high={clamp_high} | y∈[{y_lower:7.1f},{y_upper:7.1f}]"
            )

    # Remove the initialization values
    T_out_history = T_out_history[1:]
    T_in_history = T_in_history[1:]

    return {
        't_hours': t_hours,
        'u_history': np.array(u_history),
        'y_history': np.array(y_history),
        'y_ref': y_ref,
        'T_out': np.array(T_out_history),
        'T_in': np.array(T_in_history),
        'ground_snapshots': ground_temp_snapshots,
        'model': model
    }


def plot_results(results):
    """Create comprehensive visualization of results"""

    fig = plt.figure(figsize=(16, 12))
    model = results['model']

    # 1. Control performance
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(results['t_hours'], results['y_ref'], 'b--', label='Reference demand (y_ref)', alpha=0.7)
    ax1.plot(results['t_hours'], results['y_history'], 'r-', label='Delivered power (y)', linewidth=1)
    # Overlay instantaneous feasible band based on Tin clamp
    model = results['model']
    m_dot_total = model.num_bhe * model.q_flow
    y_upper = m_dot_total * model.cp_fluid * (323.15 - results['T_out'])
    y_lower = m_dot_total * model.cp_fluid * (263.15 - results['T_out'])  # Updated to -10°C
    ax1.fill_between(results['t_hours'], y_lower, y_upper, color='gray', alpha=0.2, label='Feasible band')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Power (W)')
    ax1.set_title('MPC Tracking Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Temperature evolution
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(results['t_hours'], results['T_in'] - 273.15, 'r-', label='Inlet temp', linewidth=1)
    ax2.plot(results['t_hours'], results['T_out'] - 273.15, 'b-', label='Outlet temp', linewidth=1)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('BTES Inlet/Outlet Temperatures')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Tracking error
    ax3 = plt.subplot(3, 3, 3)
    error = results['y_history'] - results['y_ref']
    ax3.plot(results['t_hours'], error, 'g-', linewidth=1)
    ax3.fill_between(results['t_hours'], 0, error, alpha=0.3, color='green')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Tracking Error (W)')
    ax3.set_title('Control Error')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # 4-8. Ground temperature snapshots
    for idx, snapshot in enumerate(results['ground_snapshots'][:5]):
        ax = plt.subplot(3, 3, 4 + idx)
        im = ax.imshow(snapshot['field'].T - 273.15, origin='lower',
                      cmap='RdBu_r', vmin=20, vmax=30)

        # Mark BHE positions
        for bhe_x, bhe_y in model.bhe_positions:
            circle = Circle((bhe_x, bhe_y), 0.5, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(circle)

        ax.set_title(f't = {snapshot["time"]:.0f} hours')
        ax.set_xlabel('x (cells)')
        ax.set_ylabel('y (cells)')
        plt.colorbar(im, ax=ax, label='°C')

    # 9. Energy balance
    ax9 = plt.subplot(3, 3, 9)
    cumulative_energy = np.cumsum(results['y_history']) * model.dt / 3600  # Wh
    ax9.plot(results['t_hours'], cumulative_energy / 1000, 'purple', linewidth=2)
    ax9.set_xlabel('Time (hours)')
    ax9.set_ylabel('Cumulative Energy (kWh)')
    ax9.set_title('Total Energy Exchange')
    ax9.grid(True, alpha=0.3)
    ax9.fill_between(results['t_hours'], 0, cumulative_energy/1000, alpha=0.3, color='purple')

    plt.suptitle('BTES System with MPC Control - 24 Hour Simulation', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Print performance metrics
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)

    # Calculate metrics
    rmse = np.sqrt(np.mean((results['y_history'] - results['y_ref'])**2))
    mae = np.mean(np.abs(results['y_history'] - results['y_ref']))
    total_energy_kwh = np.sum(results['y_history']) * model.dt / 3600 / 1000  # kWh
    temp_range = np.max(results['T_out']) - np.min(results['T_out'])

    print(f"Tracking RMSE: {rmse:.2f} W")
    print(f"Tracking MAE: {mae:.2f} W")
    print(f"Total energy exchanged: {total_energy_kwh:.3f} kWh")
    print(f"Outlet temperature range: {temp_range:.2f} K")
    print(f"Min outlet temp: {np.min(results['T_out'])-273.15:.2f} °C")
    print(f"Max outlet temp: {np.max(results['T_out'])-273.15:.2f} °C")
    # COP not computed here since we don't model electrical input explicitly

    # Additional diagnostics
    print("\n" + "-"*50)
    print("DIAGNOSTICS")
    print("-"*50)
    m_dot_total = model.num_bhe * model.q_flow
    deltaT = results['T_out'] - results['T_in']
    y_calc = m_dot_total * model.cp_fluid * deltaT
    # Consistency check with stored y_history
    y_consistency = np.max(np.abs(y_calc - results['y_history']))
    y_mean = np.mean(results['y_history'])
    y_min = np.min(results['y_history'])
    y_max = np.max(results['y_history'])
    y_ref_mean = np.mean(results['y_ref'])
    y_ref_min = np.min(results['y_ref'])
    y_ref_max = np.max(results['y_ref'])
    corr = float(np.corrcoef(results['y_history'], results['y_ref'])[0, 1])
    rel_rmse = rmse / (np.std(results['y_ref']) + 1e-9)
    # Temperature clamp activity
    tin = results['T_in']
    clamp_hits = np.mean((np.isclose(tin, 263.15, atol=1e-6)) | (np.isclose(tin, 323.15, atol=1e-6)))
    # Instantaneous feasible power bounds from clamps
    y_upper = m_dot_total * model.cp_fluid * (323.15 - results['T_out'])
    y_lower = m_dot_total * model.cp_fluid * (263.15 - results['T_out'])  # Updated to -10°C
    ref_outside_bounds = np.mean((results['y_ref'] > y_upper) | (results['y_ref'] < y_lower))

    print(f"Total mass flow (nu*q): {m_dot_total:.3f} kg/s")
    print(f"Mean ΔT (Tout−Tin): {np.mean(deltaT):.3f} K  |  Std: {np.std(deltaT):.3f} K")
    print(f"Delivered power y [kW]: mean {y_mean/1000:.2f}, min {y_min/1000:.2f}, max {y_max/1000:.2f}")
    print(f"Reference y_ref [kW]: mean {y_ref_mean/1000:.2f}, min {y_ref_min/1000:.2f}, max {y_ref_max/1000:.2f}")
    print(f"y vs y_ref correlation: {corr:.3f}  |  Relative RMSE (RMSE/std(y_ref)): {rel_rmse:.2f}")
    print(f"Max |y_calc - y_history| consistency check: {y_consistency:.6f} W")
    print(f"Fraction of steps hitting Tin clamp: {clamp_hits*100:.1f}%")
    print(f"Fraction of y_ref outside instantaneous feasible bounds: {ref_outside_bounds*100:.1f}%")

    plt.show()

    return fig


if __name__ == "__main__":
    print("="*50)
    print("BTES MODEL WITH PREDICTIVE CONTROL")
    print("Based on van Randenborgh et al. (2025)")
    print("="*50 + "\n")

    # Run simulation
    results = run_simulation()

    # Plot results
    fig = plot_results(results)

    print("\nSimulation complete!")