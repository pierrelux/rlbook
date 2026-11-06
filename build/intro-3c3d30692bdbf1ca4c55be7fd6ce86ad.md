# From Models to Learning

Reinforcement learning provides a precise language for decisions whose
consequences unfold over time. Once a state, a set of actions, a reward, and a
transition interface have been specified, it asks how models or experience can
be used to evaluate and improve a policy. This abstraction is one of the field's
strengths: the same mathematical ideas can describe games, robots, treatment
policies, and resource allocation.

:::{admonition} Definition used in this course
:class: course-definition

**Reinforcement learning** is the study of systems that learn from data how to
make good decisions in a dynamical system, whose future evolution can be
affected by present actions.

This names a problem class. Temporal-difference learning, policy-gradient
methods, and value-based methods are important algorithm families within it.
What counts as a good decision is determined by the objective and constraints in
the formulation.
:::

An application does not arrive with that interface attached. Someone must decide
where the system ends, which information is available when an action is chosen,
which interventions are physically possible, how success is measured, and which
requirements cannot be traded for reward. Each choice is a claim about the
system. An optimizer can act only on the variables, objectives, and constraints
supplied by the formulation. An omitted requirement cannot affect the computed
solution.

## The Agent-Environment Interface

{cite:t}`SuttonBarto2018` are explicit about the scope of their abstraction. The
agent-environment boundary is determined after particular states, actions, and
rewards have been selected. They also take the state signal as the output of a
preprocessing system and largely set aside the problem of constructing it so
that they can focus on decision making. This is a deliberate and productive
division of labor, not a claim that representation or formulation is
unimportant.

The familiar agent-environment loop is not unique to reinforcement learning.
Control theory, operations research, and econometrics describe the same causal
exchange with different objects and ask different questions of it.

```{raw} html
<div id="fig-interaction-loop-vocabularies" class="interaction-lenses-figure" role="figure" aria-label="Three vocabularies for a shared causal interaction.">
<div class="interaction-lenses" role="group" aria-label="The same sequential interaction described in the vocabularies of reinforcement learning, control theory, and operations research or econometrics.">
  <section class="interaction-lens">
    <div class="interaction-lens-title">Reinforcement learning</div>
    <div class="interaction-loop" role="img" aria-label="An agent sends an action to an environment and receives the next state and reward.">
      <div class="interaction-node">
        <span>Agent</span>
        <small>policy <i>π</i></small>
      </div>
      <div class="interaction-arrows" aria-hidden="true">
        <span>action <i>A</i><sub>t</sub></span>
        <svg viewBox="0 0 80 48" focusable="false">
          <path d="M4 13 H70" />
          <path d="M70 13 l-8 -5 M70 13 l-8 5" />
          <path d="M76 35 H10" />
          <path d="M10 35 l8 -5 M10 35 l8 5" />
        </svg>
        <span>state <i>S</i><sub>t+1</sub><br />reward <i>R</i><sub>t+1</sub></span>
      </div>
      <div class="interaction-node interaction-node-world">
        <span>Environment</span>
        <small>transition <i>p</i></small>
      </div>
    </div>
    <p>How can models or experience evaluate and improve a policy?</p>
    <div class="interaction-context">objective encoded by reward</div>
  </section>

  <section class="interaction-lens">
    <div class="interaction-lens-title">Control theory</div>
    <div class="interaction-loop" role="img" aria-label="A controller sends an input to a plant and receives a measurement as feedback.">
      <div class="interaction-node">
        <span>Controller</span>
        <small>feedback law <i>κ</i></small>
      </div>
      <div class="interaction-arrows" aria-hidden="true">
        <span>input <i>u</i><sub>t</sub></span>
        <svg viewBox="0 0 80 48" focusable="false">
          <path d="M4 13 H70" />
          <path d="M70 13 l-8 -5 M70 13 l-8 5" />
          <path d="M76 35 H10" />
          <path d="M10 35 l8 -5 M10 35 l8 5" />
        </svg>
        <span>measurement <i>y</i><sub>t</sub></span>
      </div>
      <div class="interaction-node interaction-node-world">
        <span>Plant</span>
        <small>dynamics <i>f</i></small>
      </div>
    </div>
    <p>Which feedback law makes the plant meet its specification?</p>
    <div class="interaction-context">reference · disturbances · cost · constraints</div>
  </section>

  <section class="interaction-lens">
    <div class="interaction-lens-title">OR and econometrics</div>
    <div class="interaction-loop" role="img" aria-label="A decision rule applies an intervention to a system or population and receives the next state and outcome.">
      <div class="interaction-node">
        <span>Decision rule</span>
        <small>objective + constraints</small>
      </div>
      <div class="interaction-arrows" aria-hidden="true">
        <span>decision <i>d</i><sub>t</sub></span>
        <svg viewBox="0 0 80 48" focusable="false">
          <path d="M4 13 H70" />
          <path d="M70 13 l-8 -5 M70 13 l-8 5" />
          <path d="M76 35 H10" />
          <path d="M10 35 l8 -5 M10 35 l8 5" />
        </svg>
        <span>covariates and<br />outcome</span>
      </div>
      <div class="interaction-node interaction-node-world">
        <span>System or population</span>
        <small>data-generating process</small>
      </div>
    </div>
    <p>What would each feasible intervention cause, and which should be selected?</p>
    <div class="interaction-context">shocks · feasibility · causal assumptions · welfare</div>
  </section>
</div>

<div class="interaction-lenses-caption"><strong>Three vocabularies for a shared causal interaction.</strong>
Reinforcement learning describes an agent acting in an environment; control
theory describes a controller acting on a plant; operations research and
econometrics describe decisions or interventions applied to an evolving system.
The correspondence is structural rather than exact: a measurement need not
reveal the state, a reward or cost must be specified, and econometric
counterfactuals require identification assumptions. The reinforcement-learning
panel follows the interaction convention of Sutton and Barto (2018).</div>
</div>
```

This book studies both sides of that division. It develops the standard theory
for choosing actions after a sequential decision problem has been specified. It
also studies how that problem is obtained from a system whose dynamics may be
partly known, whose observations may be incomplete, and whose actions and
outcomes are constrained. Its distinctive emphasis is the reasoning that
constructs and audits the decision problem, selects a solution method, and
evaluates the resulting policy.

The distinction matters outside benchmark environments. A study of real-world
reinforcement learning identifies limited samples, delays, high-dimensional
state and action spaces, safety constraints, partial observability,
stochasticity and nonstationarity, multi-objective or poorly specified rewards,
real-time action selection, offline logs, and explainability to system operators
as recurring challenges {cite:p}`DulacArnold2021`. These are not independent
details added after an algorithm has been selected. They determine the state,
action, model, objective, data, and evaluation protocol on which the algorithm
operates.

## Magnetic Control on the TCV Tokamak

TCV is a fusion-research device that confines plasma using magnetic fields. In
the controller developed by {cite:t}`Degrave2022`, conventional control first
formed and stabilized the plasma. A learned policy then issued voltage commands
to its 19 control coils at 10 kHz and controlled several plasma configurations
on the physical machine.

The simulator combined conductor-circuit dynamics, a free-boundary plasma model,
and lumped plasma-current dynamics. Data-identified sensor and power-supply
models represented filtering, delays, measurement noise, and voltage offsets.
Uncertain physical parameters were varied over experimentally informed ranges,
and objectives and termination conditions encoded operating requirements.

The learned policy became the feedback controller after handover. The surrounding
system fixed the simulator, observations, action interface, objectives,
termination rules, randomized physical parameters, and handover state. In this
experiment, reinforcement learning used physical models and conventional
plasma-formation control rather than replacing them. The example does not imply
that every application needs a detailed physical simulator. It shows that using
learning does not remove the need to identify and exploit available structure.

{cite:t}`Brunke2022` formalize the same division for safe robot control by
decomposing dynamics, cost, and constraints into nominal components that encode
prior knowledge and uncertain components learned from data. Their review treats
model-driven and data-driven approaches as endpoints and studies methods that
combine them.

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
stabilization. Dynamic programming becomes useful when computation must produce
state-contingent decisions across many states or stochastic future branches.
Learning becomes useful when a required model component, value function, or
policy must be inferred from data, or when an exact computation must be replaced
by an approximation.

When only transition samples and rewards are available, sample-based
reinforcement-learning methods can operate directly on that interface. When
equations, geometry, conservation laws, or hard constraints are also available,
replacing them all with samples discards directly usable information. Partial
models lead to a third case: known structure remains explicit while data are used
for uncertain parameters, residual dynamics, value functions, or policies.

The course therefore prepares students to move between formulations. They
should be able to inspect a new sequential decision problem, identify what is
known, choose a computation that uses that information, state what must be
learned, and design evidence that could reveal a modeling failure. This scope
complements a standard RL curriculum. It does not replace the algorithms or
theory developed there.

## Formulating the Decision Problem

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
problem. Every later method depends on some account of how actions lead to future
observations, whether supplied by equations, a simulator, logged transitions, or
live interaction.

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
targets. Flexible function approximators are used when an uncertain or
computationally intractable component must be estimated from data.

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

The environment designer has selected a boundary, state representation, actions,
transition interface, and reward, although those choices may still need to be
audited against the intended application. Sample-based reinforcement-learning
methods can operate on that interface. Differentiable dynamics and explicit
constraints also permit direct trajectory optimization, model predictive
control, local feedback design, and model-based dynamic programming when their
other assumptions are satisfied.
:::
