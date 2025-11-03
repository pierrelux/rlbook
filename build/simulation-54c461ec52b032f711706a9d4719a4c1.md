

## Programs as Models

Up to this point, we have described models using systems of equations—either differential or difference equations—that express how a system evolves over time. These **analytical models** define the transition structure explicitly. For instance, in discrete time, the evolution of the state is governed by a known function:

$$
\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)
$$

Given access to $f$, we can construct trajectories, analyze system behavior, and design control policies. The important feature here is not that the model evolves one step at a time, but that we are given the **local dynamics function** $f$ itself.

In contrast, **simulation-based models** do not expose $f$ directly. Instead, they define a procedure—implemented in code—that takes an initial state and input sequence and returns the resulting trajectory:

$$
\{\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_T\} = \mathcal{S}(\mathbf{x}_0, \{\mathbf{u}_t\}_{t=0}^{T-1})
$$

Here, $\mathcal{S}$ represents the full simulator. Internally, it may apply numerical integration, scheduling logic, branching rules, or other computations. But these details are encapsulated. From the outside, we can only query the simulator by running it.

This distinction is subtle but important. Both types of models can generate trajectories. What matters is the **interface**: analytical models provide direct access to $f$; simulation models do not. They offer a trajectory-generation interface, but hide the internal structure that produces it.

**Case Study: Robotics — *MuJoCo***

MuJoCo illustrates this distinction well. It simulates the dynamics of articulated rigid bodies under contact constraints. The equations it solves include:

$
M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}}) = \boldsymbol{\tau} + J^\top \boldsymbol{\lambda}
$

$
\phi(\mathbf{q}) = 0, \quad \boldsymbol{\lambda} \geq 0, \quad \boldsymbol{\lambda}^\top \phi = 0
$

Here $\mathbf{q}$ are joint positions, $M$ is the mass matrix, and $\boldsymbol{\lambda}$ are contact forces enforcing non-penetration. But these physical equations are part of a larger simulator that also includes:

* collision detection,
* contact force models,
* sensor and actuator emulation,
* and visual rendering.

The full behavior of a robot interacting with its environment emerges only when the simulator is executed. While the underlying physics are well-understood, the complexity of contact dynamics, collision detection, and sensor modeling makes it impractical to expose the local dynamics function $f$ directly.

### Systems with Discrete Events

Many simulation models arise when a system's dynamics are driven not by time-continuous evolution, but by the occurrence of events. These **discrete-event systems** (DES) change state only at specific, often asynchronous points in time. Between events, the state remains fixed.

A discrete-event system can be described by:

* a set of discrete states $\mathcal{X}$,
* a set of events $\mathcal{E}$,
* a transition function $f: \mathcal{X} \times \mathcal{E} \rightarrow \mathcal{X}$,
* and a time-advance function $t_a: \mathcal{X} \rightarrow \mathbb{R}_{\geq 0}$.

At each point, the system checks which events are enabled and advances to the next scheduled one.

**Example: Network Traffic Control System**

Consider a software-defined networking (SDN) controller managing traffic routing in a data center. The system must make real-time decisions about packet forwarding paths based on network conditions and service requirements.

The discrete states $\mathcal{X}$ represent the current network configuration: active routing tables, link utilization levels, and quality-of-service priority queues at each switch.

The events $\mathcal{E}$ include:
- New flow requests arriving (video streaming, database queries, file transfers)
- Link failures or congestion threshold violations
- Flow completion notifications
- Load balancing triggers when servers exceed capacity
- Network policy updates from administrators

The transition function $f$ captures how routing decisions change the network state. When a high-priority video conference flow arrives while a link is congested, the controller might transition to a new state where low-priority background traffic is rerouted through alternative paths.

The time-advance function $t_a$ determines when the next routing decision occurs. Flow arrivals follow traffic patterns (bursty during business hours), while link failures are rare but unpredictable events.

Between events, packets follow the established routing rules—the same forwarding tables remain active across all switches. The control problem here is to adapt routing decisions to discrete network events, balancing throughput, latency, and reliability constraints.

### Hybrid Systems

Some systems evolve continuously most of the time but undergo discrete jumps in response to certain conditions. These **hybrid systems** are common in control applications.

The system consists of:

* a set of discrete modes $q \in \mathcal{Q}$,
* continuous dynamics in each mode: $\dot{\mathbf{x}} = f_q(\mathbf{x})$,
* guards that specify when transitions between modes occur,
* and reset maps that update the state during such transitions.

**Example: Thermostat Control**

An HVAC system can be in one of several modes: `heating`, `cooling`, or `off`. The temperature evolves continuously according to physical laws, but when it crosses certain thresholds, the system switches modes:

```python
if x < setpoint - delta:
    mode = "heating"
elif x > setpoint + delta:
    mode = "cooling"
else:
    mode = "off"
```

Within each mode, a different differential equation applies. This results in a piecewise-smooth trajectory with mode-dependent dynamics.

**Case Study: Building Energy — *EnergyPlus***

EnergyPlus provides a sophisticated example of hybrid systems in building energy simulation. At its core are physical equations describing heat flows:

$$
C_i \frac{dT_i}{dt} = \sum_j h_{ij} A_{ij}(T_j - T_i) + Q_i
$$

It also solves implicit equations representing HVAC component behavior:

$$
0 = f(T, \dot{m}, P)
$$

But the actual simulator includes hundreds of thousands of lines of code handling:

* interpolated weather data,
* occupancy schedules,
* equipment performance curves,
* and control logic implemented as finite-state machines.

The result is a program that emulates how a building behaves over time, given environmental inputs and schedules. The hybrid nature emerges from the interaction between continuous thermal dynamics and discrete control decisions made by thermostats, occupancy sensors, and HVAC equipment.

### Agent-Based Models

Some simulation models do not describe systems via global state transitions, but instead simulate the behavior of many individual components or **agents**, each following local rules. These **agent-based models** (ABMs) are widely used in epidemiology, ecology, and social modeling.

Each agent maintains its own internal state and acts according to probabilistic or rule-based logic. The system's behavior arises from the interactions among agents.

**Example: Residential Energy Consumption under Dynamic Pricing**

Consider a neighborhood where each household is an agent making energy consumption decisions based on real-time electricity pricing and thermal comfort preferences. Each household agent has:

- **Internal state**: current temperature, HVAC settings, comfort preferences, price sensitivity
- **Local decision rules**: MPC algorithms that optimize the trade-off between energy cost and thermal comfort
- **Unique characteristics**: different utility functions, thermal mass, occupancy patterns

The simulator might execute something like:

```python
for household in neighborhood:
    # Each household solves its own MPC optimization
    current_price = utility.get_current_price()
    comfort_weight = household.comfort_preference
    
    # Optimize over prediction horizon
    optimal_setpoint = household.mpc_controller.optimize(
        current_temp=household.temperature,
        price_forecast=utility.price_forecast,
        comfort_weight=comfort_weight
    )
    
    household.set_hvac_setpoint(optimal_setpoint)
    
    # Update shared grid load
    neighborhood.total_demand += household.power_consumption
```

The macro-level demand patterns—peak shifting, load leveling, rebound effects—emerge from individual household optimization decisions. No single equation describes the neighborhood's energy consumption; it arises from the collective behavior of autonomous agents each solving their own control problems.

**Case Study: Traffic Simulation — *SUMO***

SUMO demonstrates agent-based modeling in transportation systems. Each vehicle is an agent with its own route, driving behavior, and decision-making logic. The Krauss car-following rule shows how individual vehicle agents behave:

```python
def update_vehicle(v, v_leader, gap, dt):
    v_safe = v_leader + (gap - min_gap) / tau
    v_desired = min(v_max, v_safe)
    ε = random.uniform(0, 1)
    v_new = max(0, v_desired - ε * a_max * dt)
    x_new = x + v_new * dt
    return x_new, v_new
```

Beyond car-following, each vehicle agent also:

* plans routes through the network based on travel time estimates,
* responds to traffic signals and road conditions,
* makes lane-changing decisions based on utility functions,
* and exhibits individual driving characteristics (aggressiveness, reaction time).

The emergent traffic patterns—congestion formation, traffic waves, bottlenecks—arise from the collective behavior of thousands of individual vehicle agents, each following local rules and making autonomous decisions.

**Case Study: Modeling Curbside Access at Montréal–Trudeau (YUL)**

Afternoon traffic at Montréal–Trudeau airport regularly backs up along the two-lane ramp leading to the departures curb. As passenger volumes rebound, the mix of private drop-offs, taxis, and shuttles converging in a confined space produces frequent delays. When curb dwell times rise—especially around wide-body departures—queues can spill back onto the access road and interfere with other flows on the airport campus.

To manage the situation, the airport operator relies on a dense sensor network. Cameras and license plate readers track vehicle trajectories across virtual gates, generating a real-time stream of entry points, curb interactions, and exit times. According to public statements, AI-based forecasting solutions have been deployed to anticipate congestion and suggest alternative routing options for passengers and drivers. While no technical details have been disclosed, this is a typical instance of a traffic prediction and control problem that lends itself to **agent-based modeling**.

In such a model, each vehicle is treated as an individual agent with internal state:

$$
s_t = \bigl(\text{lane},\;x_t,\;v_t,\;\text{intent}\bigr),
$$

where $x_t$ and $v_t$ denote longitudinal position and speed, and lane and intent capture higher-level behavioural traits. The simulation proceeds in discrete time. At each step, agents update their acceleration based on local traffic density (e.g., via a car-following model like IDM), evaluate potential lane changes (e.g., using a utility or incentive rule), and advance position accordingly.

The layout of the ramp—its geometry, merges, and constraints—is fixed. What changes are the traffic patterns and driver behaviours. These can be estimated from historical trajectories and updated as new data arrives. In a real-time setting, a filtering step adjusts the simulation so that its predicted flows remain consistent with current observations.

While the behaviour of each individual agent is governed by program logic and heuristics—such as car-following rules, desired speeds, or gap acceptance—some parameters are identified offline from historical data, while others are estimated online. This adaptation helps the model track observed conditions. But even with such adjustments, not all effects are easily captured.

Construction activity, weather disturbances, and irregular flight scheduling can introduce sudden shifts in flow that lie outside the scope of the structural model. To account for these, one can overlay a data-driven correction on top of the simulation. Suppose the simulator produces a queue length forecast $q^{\text{sim}}_{t+h}$ over horizon $h$. A statistical model can be trained to predict the residual between this forecast and the observed outcome:

$$
r_{t+h} = q^{\text{obs}}_{t+h} - q^{\text{sim}}_{t+h},
$$

as a function of exogenous features $z_t$, such as weather, incident flags, or scheduled arrivals. The final forecast then becomes:

$$
\widehat{q}_{t+h} = q^{\text{sim}}_{t+h} + \phi(z_t),
$$

where $\phi$ is a learned mapping from external conditions to the expected correction. The result is a hybrid model: the simulation enforces physical structure and agent-level behaviour, while the residual model compensates for aspects of the system that are harder to express analytically.

## Looking Ahead

There's no universally correct way to model a system. Your choice depends on what you know, what you can observe, what you care about, and what tools you have.

This chapter laid out a spectrum—from explicit, mechanistic models to black-box simulators and learned dynamics. In every case, modeling choices define the structure of the problem and the space of possible solutions. In the next chapters, we'll see how they anchor learning, optimization, and decision-making.
