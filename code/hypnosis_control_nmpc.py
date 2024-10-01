import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Patient:
    def __init__(self, age, weight):
        self.age = age
        self.weight = weight
        self.set_pk_params()
        self.set_pd_params()

    def set_pk_params(self):
        self.v1 = 4.27 * (self.weight / 70) ** 0.71 * (self.age / 30) ** (-0.39)
        self.v2 = 18.9 * (self.weight / 70) ** 0.64 * (self.age / 30) ** (-0.62)
        self.v3 = 238 * (self.weight / 70) ** 0.95
        self.cl1 = 1.89 * (self.weight / 70) ** 0.75 * (self.age / 30) ** (-0.25)
        self.cl2 = 1.29 * (self.weight / 70) ** 0.62
        self.cl3 = 0.836 * (self.weight / 70) ** 0.77
        self.k10 = self.cl1 / self.v1
        self.k12 = self.cl2 / self.v1
        self.k13 = self.cl3 / self.v1
        self.k21 = self.cl2 / self.v2
        self.k31 = self.cl3 / self.v3
        self.ke0 = 0.456

    def set_pd_params(self):
        self.E0 = 100
        self.Emax = 100
        self.EC50 = 3.4
        self.gamma = 3

def pk_model(x, u, patient):
    x1, x2, x3, xe = x
    dx1 = -(patient.k10 + patient.k12 + patient.k13) * x1 + patient.k21 * x2 + patient.k31 * x3 + u / patient.v1
    dx2 = patient.k12 * x1 - patient.k21 * x2
    dx3 = patient.k13 * x1 - patient.k31 * x3
    dxe = patient.ke0 * (x1 - xe)
    return np.array([dx1, dx2, dx3, dxe])

def pd_model(ce, patient):
    return patient.E0 - patient.Emax * (ce ** patient.gamma) / (ce ** patient.gamma + patient.EC50 ** patient.gamma)

def simulate_step(x, u, patient, dt):
    x_next = x + dt * pk_model(x, u, patient)
    bis = pd_model(x_next[3], patient)
    return x_next, bis

def objective(u, x0, patient, dt, N, target_bis):
    x = x0.copy()
    total_cost = 0
    for i in range(N):
        x, bis = simulate_step(x, u[i], patient, dt)
        total_cost += (bis - target_bis)**2 + 0.1 * u[i]**2
    return total_cost

def mpc_step(x0, patient, dt, N, target_bis):
    u0 = 10 * np.ones(N)  # Initial guess
    bounds = [(0, 20)] * N  # Infusion rate between 0 and 20 mg/kg/h
    
    result = minimize(objective, u0, args=(x0, patient, dt, N, target_bis),
                      method='SLSQP', bounds=bounds)
    
    return result.x[0]  # Return only the first control input

def run_mpc_simulation(patient, T, dt, N, target_bis):
    steps = int(T / dt)
    x = np.zeros((steps+1, 4))
    bis = np.zeros(steps+1)
    u = np.zeros(steps)
    
    for i in range(steps):
        # Add noise to the current state to simulate real-world uncertainty
        x_noisy = x[i] + np.random.normal(0, 0.01, size=4)
        
        # Use noisy state for MPC planning
        u[i] = mpc_step(x_noisy, patient, dt, N, target_bis)
        
        # Evolve the true state using the deterministic model
        x[i+1], bis[i] = simulate_step(x[i], u[i], patient, dt)
    
    bis[-1] = pd_model(x[-1, 3], patient)
    return x, bis, u

# Set up the problem
patient = Patient(age=40, weight=70)
T = 120  # Total time in minutes
dt = 0.5  # Time step in minutes
N = 20  # Prediction horizon
target_bis = 50  # Target BIS value

# Run MPC simulation
x, bis, u = run_mpc_simulation(patient, T, dt, N, target_bis)

# Plot results
t = np.arange(0, T+dt, dt)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax1.plot(t, bis)
ax1.set_ylabel('BIS')
ax1.set_ylim(0, 100)
ax1.axhline(y=target_bis, color='r', linestyle='--')

ax2.plot(t[:-1], u)
ax2.set_ylabel('Infusion Rate (mg/kg/h)')

ax3.plot(t, x[:, 3])
ax3.set_ylabel('Effect-site Concentration (µg/mL)')
ax3.set_xlabel('Time (min)')

plt.tight_layout()
plt.show()

print(f"Initial BIS: {bis[0]:.2f}")
print(f"Final BIS: {bis[-1]:.2f}")
print(f"Mean infusion rate: {np.mean(u):.2f} mg/kg/h")
print(f"Final effect-site concentration: {x[-1, 3]:.2f} µg/mL")