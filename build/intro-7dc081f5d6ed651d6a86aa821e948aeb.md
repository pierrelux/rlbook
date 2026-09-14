# From Models to Learning

Reinforcement learning studies decisions whose consequences unfold over time.
Before an algorithm can learn, someone must decide what the system is, which
quantities form its state, which actions are available, what can be observed,
and how success is measured. In benchmark environments these choices have
already been made. In applications they are often the hardest part of the work.

Sensors are noisy, constraints are non-negotiable, and objectives may conflict.
A controller that performs well in a simulator can fail because the simulator
permits an impossible force or omits a mode transition. A learned policy can
optimize the reward it was given while violating a requirement that never
entered the model. These are failures of formulation rather than failures of a
particular reinforcement-learning algorithm.

The same issue appears across several communities. Control theory, dynamic
programming, operations research, economics, and reinforcement learning use
different vocabularies, but each asks how actions change future outcomes. In
deployed optimization systems, learned predictors often sit inside explicit
objectives and constraints. Ride matching and robot control provide examples at
large scale {cite:p}`Uber2025,ANYbotics2023`, while applied work more generally
faces the difficulty of learning both the objective and the environment
{cite:p}`Iskhakov2020`.

## The Decision Problem

The term *reinforcement learning* can name a problem, a family of algorithms,
or a research community. This book uses it first as a decision problem: an
agent receives information, chooses actions, and experiences consequences. That
description includes model predictive control, dynamic programming, direct
policy search, and many systems that are not usually labeled as RL.

The problem is rarely handed to us in finished form. A useful formulation must
answer five questions:

1. Which variables summarize the information needed for future decisions?
2. Which actions can the decision maker actually apply?
3. How do actions, disturbances, and uncertainty change the system?
4. Which outcomes should the objective reward or penalize?
5. Which physical, operational, or safety constraints must always hold?

The answers determine which computations are available. An explicit
differential equation can expose derivatives and conservation laws. A simulator
may provide only trajectory samples. Logged data cannot answer arbitrary
counterfactual queries without additional assumptions. Interaction provides new
data, but it also consumes time and can damage the system.

## The Book's Progression

The technical development follows the information available about the decision
problem. It begins with models because every later method consumes a model
interface, even when that interface consists only of sampled transitions.

| Part | Question | Result |
|---|---|---|
| Modeling | What evolves, what can be observed, and where does an action enter? | A state, action, disturbance, objective, constraint set, and model interface |
| Trajectory optimization | What action sequence performs well from one initial condition? | An open-loop plan |
| Model predictive control | How can the plan react to new measurements? | Feedback through repeated replanning |
| Dynamic programming | How can decisions be optimized for many possible future states? | A value function and state-contingent policy |
| Approximation and learning | Which required object is unavailable or too expensive to compute exactly? | An approximation learned from models, samples, or interaction |

This order does not assume that a complete mechanistic model is always
available. It makes the information restriction explicit. Known conservation
laws, geometry, and constraints remain in the formulation; uncertain
parameters, residual dynamics, value functions, or policies become learning
targets. Black-box neural networks enter as components when their flexibility
matches the uncertain part of the problem.

The chapters repeatedly test this distinction on concrete systems. A
playground swing exposes internal actuation and an omitted slack-chain mode. A
cart-pole separates global trajectory planning from local stabilization. An
overhead crane and a wave-energy converter show how actuator geometry and hard
constraints shape optimization. The examples are intentionally different: the
method should follow from the structure of the problem, rather than from a
preferred benchmark.
