import numpy as np
import do_mpc
from casadi import *  # noqa: F401 - do-mpc constructs CasADi symbols under the hood


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

    # Specific growth rate
    mu_S = mu_m * S_s / (K_m + S_s + (S_s**2 / K_i))

    # Dynamics
    model.set_rhs('X_s', mu_S * X_s - inp / V_s * X_s)
    model.set_rhs('S_s', -mu_S * X_s / Y_x - v_par * X_s / Y_p + inp / V_s * (S_in - S_s))
    model.set_rhs('P_s', v_par * X_s - inp / V_s * P_s)
    model.set_rhs('V_s', inp)

    model.setup()
    return model


def build_mpc(model):
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': 30,
        't_step': 1.0,
        'n_robust': 1,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    # Objective: encourage product formation and small inputs (economic-like MPC)
    X_s = model.x['X_s']
    S_s = model.x['S_s']
    P_s = model.x['P_s']
    V_s = model.x['V_s']
    inp = model.u['inp']

    mterm = -P_s  # maximize product at horizon end
    lterm = -P_s + 1e-4 * inp**2  # small input penalty
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Box constraints
    mpc.bounds['lower', '_x', 'X_s'] = 0.0
    mpc.bounds['lower', '_x', 'S_s'] = -0.01
    mpc.bounds['lower', '_x', 'P_s'] = 0.0
    mpc.bounds['lower', '_x', 'V_s'] = 0.0
    mpc.bounds['upper', '_x', 'X_s'] = 3.7
    mpc.bounds['upper', '_x', 'P_s'] = 3.0
    mpc.bounds['lower', '_u', 'inp'] = 0.0
    mpc.bounds['upper', '_u', 'inp'] = 0.2

    # Uncertainty scenarios (shared control across scenarios)
    Y_x_values = np.array([0.5, 0.4, 0.3])
    S_in_values = np.array([200.0, 220.0, 180.0])
    mpc.set_uncertainty_values(Y_x=Y_x_values, S_in=S_in_values)

    mpc.setup()
    return mpc


def build_estimator(model):
    return do_mpc.estimator.StateFeedback(model)


def build_simulator(model):
    simulator = do_mpc.simulator.Simulator(model)
    params_sim = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 1.0,
    }
    simulator.set_param(**params_sim)

    # Realizations of uncertain parameters used by the simulator
    p_num = simulator.get_p_template()
    p_num['Y_x'] = 0.4
    p_num['S_in'] = 200.0

    def p_fun(t_now):
        return p_num

    simulator.set_p_fun(p_fun)
    simulator.setup()
    return simulator


def run_closed_loop():
    model = build_model()
    mpc = build_mpc(model)
    estimator = build_estimator(model)
    simulator = build_simulator(model)

    # Initial state
    x0 = np.array([1.0, 0.5, 0.0, 120.0])
    mpc.x0 = x0
    estimator.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    # Closed-loop simulation
    n_steps = 60
    for _ in range(n_steps):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

    # Visualization
    import matplotlib.pyplot as plt

    mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
    sim_graphics = do_mpc.graphics.Graphics(simulator.data)

    fig, ax = plt.subplots(5, sharex=True, figsize=(12, 8))
    fig.align_ylabels()

    for g in [sim_graphics, mpc_graphics]:
        g.add_line(var_type='_x', var_name='X_s', axis=ax[0], color='#1f77b4')
        g.add_line(var_type='_x', var_name='S_s', axis=ax[1], color='#1f77b4')
        g.add_line(var_type='_x', var_name='P_s', axis=ax[2], color='#1f77b4')
        g.add_line(var_type='_x', var_name='V_s', axis=ax[3], color='#1f77b4')
        g.add_line(var_type='_u', var_name='inp', axis=ax[4], color='#1f77b4')

    ax[0].set_ylabel('X_s [mol/l]')
    ax[1].set_ylabel('S_s [mol/l]')
    ax[2].set_ylabel('P_s [mol/l]')
    ax[3].set_ylabel('V_s [m^3]')
    ax[4].set_ylabel('u_inp [m^3/min]')
    ax[4].set_xlabel('t [min]')

    # Plot full horizon results
    sim_graphics.plot_results()
    mpc_graphics.plot_predictions()
    mpc_graphics.reset_axes()
    plt.tight_layout()
    plt.show()


# Run the closed-loop simulation
run_closed_loop()


