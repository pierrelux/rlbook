import numpy as np
import do_mpc
from casadi import *  # noqa: F401 - do-mpc constructs CasADi symbols under the hood
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import matplotlib.patches as mpatches
from contextlib import redirect_stdout, redirect_stderr
import io


def build_model():
    model_type = 'continuous'
    model = do_mpc.model.Model(model_type)

    # States
    X_s = model.set_variable('_x', 'X_s')  # biomass
    S_s = model.set_variable('_x', 'S_s')  # substrate
    P_s = model.set_variable('_x', 'P_s')  # product
    V_s = model.set_variable('_x', 'V_s')  # volume

    # Control input (feed flow)
    inp = model.set_variable('_u', 'inp')

    # Certain parameters
    mu_m = 0.02
    K_m = 0.05
    K_i = 5.0
    v_par = 0.004
    Y_p = 1.2

    # Uncertain parameters
    Y_x = model.set_variable('_p', 'Y_x')
    S_in = model.set_variable('_p', 'S_in')

    # Auxiliary term for specific growth rate
    mu = mu_m * S_s / (K_m + S_s + S_s**2 / K_i)
    model.set_expression('mu', mu)

    # Differential equations
    model.set_rhs('X_s', mu * X_s - inp / V_s * X_s)
    model.set_rhs('S_s', -mu * X_s / Y_x - v_par * X_s / Y_p + inp / V_s * (S_in - S_s))
    model.set_rhs('P_s', v_par * X_s - inp / V_s * P_s)
    model.set_rhs('V_s', inp)

    model.setup()
    return model


def setup_mpc(model):
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 20,
        't_step': 1.0,
        'n_robust': 1,
        'store_full_solution': True,
        # Silence IPOPT/CasADi prints
        'nlpsol_opts': {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
        },
    }
    mpc.set_param(**setup_mpc)

    # Objective
    mterm = model.aux['mu']  # terminal cost on growth
    lterm = model.aux['mu']  # stage cost on growth
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Constraints
    mpc.bounds['lower', '_u', 'inp'] = 0.0
    mpc.bounds['upper', '_u', 'inp'] = 0.2
    
    mpc.bounds['lower', '_x', 'X_s'] = 0.0
    mpc.bounds['lower', '_x', 'S_s'] = 0.0
    mpc.bounds['lower', '_x', 'P_s'] = 0.0
    mpc.bounds['lower', '_x', 'V_s'] = 0.0

    # Uncertain parameters
    Y_x_values = np.array([0.5])  # single scenario to avoid prediction dim issues
    S_in_values = np.array([200.0])
    mpc.set_uncertainty_values(Y_x=Y_x_values, S_in=S_in_values)

    mpc.setup()
    return mpc


def setup_simulator(model):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=1.0)

    # Set uncertain parameters for simulation
    p_template = simulator.get_p_template()
    p_template['Y_x'] = 0.5
    p_template['S_in'] = 200.0
    simulator.set_p_fun(lambda t_now: p_template)

    simulator.setup()
    return simulator


def setup_estimator(model):
    estimator = do_mpc.estimator.StateFeedback(model)
    return estimator


def run_closed_loop_simulation():
    """Run the closed-loop simulation and return results (silencing solver output)."""
    # Build system
    model = build_model()
    mpc = setup_mpc(model)
    simulator = setup_simulator(model)
    estimator = setup_estimator(model)

    # Initial state
    x0 = np.array([1.0, 150.0, 0.0, 120.0]).reshape(-1, 1)
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0
    mpc.set_initial_guess()

    # Storage for results
    results = {
        't': [],
        'x': [],
        'u': [],
    }

    # Silence IPOPT/CasADi output during the loop
    fnull = io.StringIO()
    with redirect_stdout(fnull), redirect_stderr(fnull):
        for k in range(60):
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)
            
            # Store results
            results['t'].append(k)
            results['x'].append(x0.flatten())
            results['u'].append(u0.flatten())

    # Convert to arrays
    results['x'] = np.array(results['x'])
    results['u'] = np.array(results['u'])
    
    return results


def create_animation(results):
    """Create an animated visualization of the batch bioreactor process."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Tank visualization (left side)
    ax_tank = fig.add_subplot(gs[:, 0])
    
    # State plots (middle and right)
    ax_biomass = fig.add_subplot(gs[0, 1])
    ax_substrate = fig.add_subplot(gs[0, 2])
    ax_product = fig.add_subplot(gs[1, 1])
    ax_volume = fig.add_subplot(gs[1, 2])
    ax_control = fig.add_subplot(gs[2, 1:])
    
    # Setup axes
    n_steps = len(results['t'])
    
    # State axes setup
    for ax, label, ylim in [
        (ax_biomass, 'Biomass X_s [g/L]', (0, 6)),
        (ax_substrate, 'Substrate S_s [g/L]', (0, 250)),
        (ax_product, 'Product P_s [g/L]', (0, 50)),
        (ax_volume, 'Volume V_s [L]', (100, 200))
    ]:
        ax.set_xlim(0, n_steps)
        ax.set_ylim(ylim)
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    
    ax_control.set_xlim(0, n_steps)
    ax_control.set_ylim(-0.01, 0.21)
    ax_control.set_xlabel('Time [h]')
    ax_control.set_ylabel('Feed Flow u [L/h]')
    ax_control.grid(True, alpha=0.3)
    
    # Tank setup
    ax_tank.set_xlim(-1.5, 1.5)
    ax_tank.set_ylim(0, 3)
    ax_tank.set_aspect('equal')
    ax_tank.axis('off')
    ax_tank.set_title('Batch Bioreactor', fontsize=14, fontweight='bold')
    
    # Initialize plot elements
    lines = {
        'biomass': ax_biomass.plot([], [], 'g-', lw=2, label='Biomass')[0],
        'substrate': ax_substrate.plot([], [], 'b-', lw=2, label='Substrate')[0],
        'product': ax_product.plot([], [], 'r-', lw=2, label='Product')[0],
        'volume': ax_volume.plot([], [], 'm-', lw=2, label='Volume')[0],
        'control': ax_control.plot([], [], 'k-', lw=2, label='Control')[0],
    }
    
    # Current point markers
    markers = {
        'biomass': ax_biomass.plot([], [], 'go', markersize=8)[0],
        'substrate': ax_substrate.plot([], [], 'bo', markersize=8)[0],
        'product': ax_product.plot([], [], 'ro', markersize=8)[0],
        'volume': ax_volume.plot([], [], 'mo', markersize=8)[0],
        'control': ax_control.plot([], [], 'ko', markersize=8)[0],
    }
    
    # Tank components
    tank_outline = mpatches.FancyBboxPatch(
        (-1, 0.2), 2, 2, boxstyle="round,pad=0.02",
        linewidth=3, edgecolor='black', facecolor='none'
    )
    ax_tank.add_patch(tank_outline)
    
    # Liquid in tank (will be updated)
    liquid = mpatches.Rectangle((-0.95, 0.25), 1.9, 1.0, 
                                facecolor='lightblue', alpha=0.6)
    ax_tank.add_patch(liquid)
    
    # Biomass particles (circles)
    biomass_particles = []
    for _ in range(10):
        particle = plt.Circle((0, 1), 0.05, color='green', alpha=0.7)
        ax_tank.add_patch(particle)
        biomass_particles.append(particle)
    
    # Feed pipe and valve
    feed_pipe = mpatches.Rectangle((0.8, 2.2), 0.1, 0.5, 
                                   facecolor='gray', edgecolor='black')
    ax_tank.add_patch(feed_pipe)
    
    valve = mpatches.FancyBboxPatch(
        (0.75, 2.15), 0.2, 0.1, boxstyle="round,pad=0.01",
        linewidth=2, edgecolor='black', facecolor='red', alpha=0.5
    )
    ax_tank.add_patch(valve)
    
    # Text displays
    time_text = fig.text(0.02, 0.98, '', fontsize=12, fontweight='bold',
                         transform=fig.transFigure)
    
    tank_text = ax_tank.text(0, 2.7, '', ha='center', fontsize=10)
    
    # Add legend to one subplot
    ax_biomass.legend(loc='upper right')
    
    def init():
        """Initialize animation."""
        for line in lines.values():
            line.set_data([], [])
        for marker in markers.values():
            marker.set_data([], [])
        return list(lines.values()) + list(markers.values())
    
    def animate(frame):
        """Animation function."""
        # Update time text
        time_text.set_text(f'Time: {frame:.0f} h')
        
        # Update history lines
        t_data = results['t'][:frame+1]
        x_data = results['x'][:frame+1]
        u_data = results['u'][:frame+1]
        
        if frame > 0:
            lines['biomass'].set_data(t_data, x_data[:, 0])
            lines['substrate'].set_data(t_data, x_data[:, 1])
            lines['product'].set_data(t_data, x_data[:, 2])
            lines['volume'].set_data(t_data, x_data[:, 3])
            lines['control'].set_data(t_data, u_data[:, 0])
            
            # Update current point markers
            markers['biomass'].set_data([frame], [x_data[frame, 0]])
            markers['substrate'].set_data([frame], [x_data[frame, 1]])
            markers['product'].set_data([frame], [x_data[frame, 2]])
            markers['volume'].set_data([frame], [x_data[frame, 3]])
            markers['control'].set_data([frame], [u_data[frame, 0]])
        
        # Update tank visualization
        if frame < len(x_data):
            # Update liquid level based on volume
            volume = x_data[frame, 3]
            liquid_height = 1.5 * (volume / 200.0)  # Normalize to tank height
            liquid.set_height(liquid_height)
            
            # Update biomass particles
            biomass_conc = x_data[frame, 0]
            n_visible = int(10 * min(biomass_conc / 5.0, 1.0))  # Scale particles
            
            for i, particle in enumerate(biomass_particles):
                if i < n_visible:
                    # Random position in liquid
                    x = np.random.uniform(-0.8, 0.8)
                    y = np.random.uniform(0.3, 0.25 + liquid_height * 0.9)
                    particle.set_center((x, y))
                    particle.set_alpha(0.7)
                else:
                    particle.set_alpha(0)
            
            # Update valve color based on control input
            u_val = u_data[frame, 0] if frame < len(u_data) else 0
            valve_color = plt.cm.RdYlGn_r(u_val / 0.2)  # Red=high flow, Green=low
            valve.set_facecolor(valve_color)
            valve.set_alpha(0.8 if u_val > 0.01 else 0.3)
            
            # Update tank text
            tank_text.set_text(
                f'V={volume:.1f}L, X={biomass_conc:.2f}g/L\n'
                f'S={x_data[frame, 1]:.1f}g/L, P={x_data[frame, 2]:.1f}g/L'
            )
        
        return (list(lines.values()) + list(markers.values()) + 
                biomass_particles + [liquid, valve, tank_text, time_text])
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=n_steps, interval=100, blit=False)
    
    plt.suptitle('Batch Bioreactor Control with do-mpc', fontsize=16, fontweight='bold')
    
    return fig, anim


# Run simulation and create animation (no prints)
results = run_closed_loop_simulation()
fig, anim = create_animation(results)

# Render like the pendulum example: JS HTML animation and no extra prints
js_anim = anim.to_jshtml()
plt.close(fig)
display(HTML(js_anim))