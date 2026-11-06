# From Models to Learning

Reinforcement learning provides a precise language for decisions whose
consequences unfold over time. Once a state, a set of actions, a reward, and a
transition mechanism have been specified, it asks how experience can improve a
policy. This abstraction is one of the field's strengths: the same mathematical
ideas can describe games, robots, treatment policies, and resource allocation.

An application does not arrive with that interface attached. Someone must decide
where the system ends, which information is available when an action is chosen,
which interventions are physically possible, how success is measured, and which
requirements cannot be traded for reward. Each choice is a claim about the
system. A capable optimizer may expose a missing constraint or an incorrect
boundary more efficiently, but it cannot restore information that the formulation
left out.

## The Agent-Environment Interface

{cite:t}`SuttonBarto2018` are explicit about the scope of their abstraction. The
agent-environment boundary is determined after particular states, actions, and
rewards have been selected. They also take the state signal as the output of a
preprocessing system and set aside the problem of constructing it so that they
can focus on decision making. This is a deliberate and productive division of
labor, not a claim that representation or formulation is unimportant.

This book studies both sides of that division. It develops the standard theory
for choosing actions after a sequential decision problem has been specified. It
also studies how that problem is obtained from a system whose dynamics may be
partly known, whose observations may be incomplete, and whose actions and
outcomes are constrained. Its distinctive emphasis is the reasoning that
constructs, audits, and exploits this interface before and around policy
optimization.

The distinction matters outside benchmark environments. A peer-reviewed study
of real-world reinforcement learning identifies limited interaction, delays,
partial observability, nonstationarity, safety constraints, multi-objective or
poorly specified rewards, real-time computation, offline logs, and operator
interpretability as recurring challenges {cite:p}`DulacArnold2021`. These are not
independent details added after an algorithm has been selected. They determine
the state, action, model, objective, data, and evaluation protocol on which the
algorithm operates.

## Evidence from a Deployed Controller

The magnetic controller for the TCV tokamak developed by
{cite:t}`Degrave2022` illustrates how these pieces meet in a consequential
system. A neural policy controlled all 19 magnetic coils at 10 kHz and produced
several plasma configurations on the physical machine. The policy was trained in
a simulator built from coil-circuit dynamics, a free-boundary plasma model, and
plasma-current dynamics. The training environment also included identified
models of sensors, power supplies, delays, noise, and bias. Uncertain physical
parameters were varied over experimentally informed ranges, and objectives and
termination conditions encoded operating requirements. Conventional control
formed and stabilized the plasma before handing control to the learned policy.

The result is properly an achievement in reinforcement learning, but policy
optimization was one component of the implemented system. Its transfer to the
tokamak depended on decisions about what to model, randomize, observe, constrain,
and retain under conventional control. The example does not imply that every
application needs a detailed physical simulator. It shows that using learning
does not remove the need to identify and exploit available structure.

This combination also has a general mathematical form. In their review of safe
learning in robotics, {cite:t}`Brunke2022` decompose dynamics, objectives, and
constraints into nominal components representing prior knowledge and unknown
components learned from data. A fully specified mechanistic model and a purely
sample-based interface are two ends of a spectrum, and many useful systems lie
between them.

## Relation to Standard Reinforcement Learning

A reinforcement-learning course centered on {cite:t}`SuttonBarto2018` develops
general solution methods once the sequential decision interface has been chosen:
Bellman equations, dynamic programming and planning, Monte Carlo and
temporal-difference methods, value-function approximation, and policy gradients.
These foundations are also central to this book. The distinction is therefore
not between model-free and model-based reinforcement learning; the standard
theory contains both.

The difference is one of starting point and emphasis. This course also studies
the modeling and control steps that construct and surround the standard
reinforcement-learning interface. When dynamics and constraints are known,
trajectory optimization may produce a useful plan directly. Replanning can turn
that plan into feedback, and a local linear model may be sufficient for
stabilization. Dynamic programming becomes relevant when decisions must account
for many possible future states. If a required model component, value function,
or policy is unknown or too expensive to compute, learning supplies an
approximation from data or interaction.

When only transition samples and rewards are available, the standard
reinforcement-learning interface is the appropriate one. When equations,
geometry, conservation laws, or hard constraints are also available, replacing
them all with samples removes information that computation could use. Partial
models lead to a third case: known structure remains explicit while data are
used for uncertain parameters, residual dynamics, value functions, or policies.

The course therefore prepares students to move between formulations. They
should be able to inspect a new sequential decision problem, identify what is
known, choose a computation that uses that information, state what must be
learned, and design evidence that could reveal a modeling failure. This scope
complements a standard RL curriculum. It does not replace the algorithms or
theory developed there.

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

## Course Progression

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

Dynamic programming, function approximation, and optimization provide a shared
mathematical foundation for these methods. The same structures recur in
reinforcement learning, control theory, operations research, and other areas of
sequential decision-making. Their common form makes it possible to transfer an
argument or algorithm from one setting to another while preserving the
assumptions that make it valid.

## Self-checks

:::{exercise} Problem before algorithm
:label: ex-intro-check-1

A production controller must respect a hard storage limit. Is selecting a learning algorithm enough to define the reinforcement-learning problem? Name two pieces of structure that must be specified first.
:::

:::{solution} ex-intro-check-1
:class: dropdown

No. At minimum, the state and available decisions must be defined, together with the objective and the hard storage constraint. The transition or feedback structure must also say how decisions affect future states.
:::

:::{exercise} Shared structure
:label: ex-intro-check-2

What feature makes trajectory optimization, model predictive control, and reinforcement learning members of the same decision-making family?
:::

:::{solution} ex-intro-check-2
:class: dropdown

All three choose actions whose consequences unfold through a dynamical system over time, so present decisions must account for future objectives and constraints.
:::

:::{exercise} Course scope
:label: ex-intro-check-3

Suppose an environment already supplies a Markov state, an action set, a reward,
and a transition sampler. Which formulation work has already been done? What
additional computations become possible if differentiable dynamics and hard
constraints are also available?
:::

:::{solution} ex-intro-check-3
:class: dropdown

The environment designer has already chosen the system boundary, state, actions,
transition interface, and objective. Sample-based reinforcement-learning methods
can operate on that interface. Differentiable dynamics and explicit constraints
also permit direct trajectory optimization, model predictive control, local
feedback design, and model-based dynamic programming when their other
assumptions are satisfied.
:::
