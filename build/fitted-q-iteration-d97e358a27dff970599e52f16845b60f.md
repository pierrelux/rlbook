---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Fitted Q-Iteration and DQN

The algorithms of the previous chapters chose the states at which targets
were computed and drew as many successors as they liked. In practice the data
are given. A log of transitions $(s, a, r, s')$ was recorded by whoever
operated the system, with one successor per transition, and no new successor
can be drawn; or a buffer fills as the agent acts, so that the data change
while the function is being fitted. This chapter is about fitted Q-iteration
on such data. The batch algorithms come first: the tree-based FQI of Ernst and
neural fitted Q-iteration, run on two logs, one from the inference-serving
scheduler of the infinite-horizon chapter and one of the collision warnings
that ESA received for its satellites. Both show that the error of a fitted
policy has two sources, approximation where two actions are nearly tied and
coverage where the log never tried an action, and the second shows what to
do when a few rare events dominate the regression. The chapter then opens
the fitting step. Writing out the gradient steps of a neural fit and merging
them with the outer loop turns fitted Q-iteration into DQN, with its target
network and its online network, and letting the buffer evolve during
learning gives DQN proper, Double DQN, and, at the limit of one transition
per update, Q-learning. A synthesis section lays out the design space that
these choices span, and the chapter closes with losses that model the noise
of Bellman targets better than squared error.

How do the buffer, the Bellman target, the function approximator and the
optimization budget combine when only one observed successor is available
for each stored transition?

::::{admonition} Learning Goals
:class: note

After completing this chapter, you should be able to:
- Write fitted Q-iteration on a fixed set of transitions, and say where the empirical distribution of the data replaces the model
- Attribute the errors of a fitted policy to approximation, where actions are nearly tied, or to coverage, where the data never tried an action
- Scale the targets of a regression so that a few rare, costly events do not dominate the fit
- Flatten the nested loops of neural fitted Q-iteration into DQN, and explain the roles of its online and target networks
- Place an algorithm in the design space of fitted Q-iteration by its buffer, target function, fit, initialization and loss

**Prerequisites:** fitted value iteration and its error analysis ([approximate Bellman equations](approximate-bellman-equations.md)), action-value functions, the single-sample paradigm and maximization bias ([Monte Carlo Bellman estimation](monte-carlo-bellman-estimation.md)).
::::

(sec-fqi-batch)=
## Learning from a Fixed Set of Transitions

How does fitted Q-iteration run when the transitions are fixed before
learning begins, and what replaces the model?

The Bellman operator on action values contains an expectation under the
transition law, which we write $P$ over transitions $\tau = (s, a, r, s')$:

$$
(\Bellman q)(s,a) = r(s,a) + \gamma \int \max_{a'} q(s',a')\, P(ds' \mid s,a).
$$

A batch algorithm never sees $P$. It receives a data set
$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$ and works with the
**empirical distribution**

$$
\hat{P}_{\mathcal{D}} = \frac{1}{N} \sum_{\tau \in \mathcal{D}} \delta_\tau,
$$

where $\delta_\tau$ is a point mass at the transition $\tau$. Sampling from
$\hat{P}_{\mathcal{D}}$ means picking a recorded transition uniformly at
random, and the expectation of any function of a transition under $P$ is
estimated by its average over the data set, the sample-average estimator of
the [Monte Carlo chapter](monte-carlo-bellman-estimation.md) applied to
transitions. Fitted Q-iteration on a batch is value iteration under this
empirical law. The target of a transition is $y_i = r_i + \gamma \max_{a'}
q(s'_i, a')$, with the one successor that was recorded, and the fit averages
the error of the targets over the transitions in the data set. In the
language of statistics this is a plug-in estimate: where the algorithm needs
the unknown $P$, it substitutes $\hat{P}_{\mathcal{D}}$.

```{prf:remark} The empirical distribution as a probability measure
:class: dropdown

The empirical distribution $\hat{P}_{\mathcal{D}} = \frac{1}{N}\sum_{\tau \in \mathcal{D}} \delta_{\tau}$ is a discrete probability measure over $N$ points, whether the state and action spaces are continuous or discrete. For any measurable set $A \subseteq \mathcal{S} \times \mathcal{A} \times \mathbb{R} \times \mathcal{S}$,

$$
\hat{P}_{\mathcal{D}}(A) = \frac{1}{N}\sum_{\tau \in \mathcal{D}} \mathbb{1}[\tau \in A] = \frac{|\{\tau \in \mathcal{D} : \tau \in A\}|}{N},
$$

so it assigns probability $1/N$ to each recorded tuple and zero elsewhere. A transition that never occurred has no weight, which is the formal statement of the coverage problem that the two experiments below exhibit.
```

```{prf:algorithm} Fitted Q-Iteration on a Batch
:label: fitted-q-iteration-batch

**Input:** Dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$, function approximator $q(s,a; \boldsymbol{\theta})$, discount factor $\gamma$, maximum iterations $n_{\max}$

**Output:** Learned Q-function parameters $\boldsymbol{\theta}$

1. Initialize $\boldsymbol{\theta}_0$
2. $n \leftarrow 0$
3. **repeat**  $\quad$ // **Outer loop: value iteration**
4. $\quad$ **// Construct the regression data set with Bellman targets**
5. $\quad$ $X^{(n)} \leftarrow \{(s_i, a_i) : (s_i, a_i, r_i, s'_i) \in \mathcal{D}\}$
6. $\quad$ **for** each $(s_i, a_i, r_i, s'_i) \in \mathcal{D}$ **do**
7. $\quad\quad$ $y_i^{(n)} \leftarrow r_i + \gamma \max_{a' \in \mathcal{A}} q(s'_i, a'; \boldsymbol{\theta}_n)$
8. $\quad$ **end for**
9. $\quad$ **// Inner loop: fit the Q-function to the targets**
10. $\quad$ $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(X^{(n)}, y^{(n)}, \boldsymbol{\theta}_{\text{init}}, K)$
11. $\quad$ $n \leftarrow n+1$
12. **until** convergence or $n \geq n_{\max}$
13. **return** $\boldsymbol{\theta}_n$
```

The state-action pairs $X^{(n)}$ are the same at every iteration; only the
targets change, because they are computed with the current Q-function. The
`fit` operation in line 10 minimizes $\sum_i \ell(q(s_i, a_i; \boldsymbol{\theta}),
y_i^{(n)})$ for a loss $\ell$, by whatever procedure the function class
calls for, from an initialization $\boldsymbol{\theta}_{\text{init}}$ and
with an optimization budget $K$. Supervised learning would initialize at
random and run to convergence. The two original batch algorithms differ in
exactly these choices.

{cite:t}`ernst2005tree` instantiated the template with extremely randomized
trees, an ensemble whose prediction is the average of the targets in a leaf.
Each fit grows the trees from scratch, a cold start, and runs to completion,
since tree construction has no incremental form. A tree with a fixed
partition is an averager, so the convergence theorem of the
[approximate Bellman chapter](approximate-bellman-equations.md) applies once
the partitions stop changing, and trees take high-dimensional inputs without
feature design. {cite:t}`riedmiller2005neural` replaced the ensemble by a
neural network $q(s,a;\boldsymbol\theta)$, which interpolates smoothly across
the state-action space, and fitted it by gradient descent (RProp, chosen for
its insensitivity to hyperparameters) until the loss stopped decreasing. The
network is warm-started: each fit begins from the parameters of the previous
iteration, which is natural for a gradient method and is the first of the
choices that the second half of this chapter turns into DQN.

```{prf:remark} Goal states in neural fitted Q-iteration
:class: dropdown

For episodic tasks with goal states $S^+$ and forbidden states $S^-$, Riedmiller modified the targets:

$$
y_i^{(n)} = \begin{cases}
c(s_i, a_i, s'_i) & \text{if } s'_i \in S^+ \text{ (goal reached)} \\
C^- & \text{if } s'_i \in S^- \text{ (forbidden state, typically } C^- = 1\text{)} \\
c(s_i, a_i, s'_i) + \gamma \max_{a'} q(s'_i, a'; \boldsymbol{\theta}_n) & \text{otherwise,}
\end{cases}
$$

where $c(s, a, s')$ is the immediate cost. A goal state has no future cost and no bootstrapped term, a forbidden state carries a fixed penalty, and every other state uses the Bellman backup. The hint-to-goal heuristic adds synthetic transitions from goal states with target zero, which pins the Q-function to zero in the goal region and encodes the boundary condition explicitly.
```

### Experiment: Scheduling from Fixed Transition Data

The reduced inference-scheduling MDP in [Infinite-Horizon MDPs](infinite-horizon-mdps.md#exact-scheduling-mdp-for-inference-serving)
has an exact value-iteration solution because its transition matrix is known.
Suppose instead that only a fixed table of transitions is retained. Each row
contains a reduced state $(p,d)$, a scheduling action, the cost $c$, and the
next reduced state. The generic reward-maximizing form above becomes a
cost-minimizing form after a sign change. FQI can reuse those rows, but it
receives no direct evidence for a state-action region that the behavior policy
never visits. Any prediction there comes from the regressor's inductive bias.

Each transition row is sampled from the finite MDP, not logged directly from
vLLM. Its arrival model comes from the normalized Azure
trace, while its phase rates and powers come from one NVIDIA L4 measurement run
of Qwen/Qwen2.5-7B-Instruct served by vLLM 0.28.0. The data sets therefore
inherit a measured hardware calibration and all of the reduced model's
aggregation assumptions.

Two sampling procedures produce matched data sets of 50,000 transitions. The
**broad** data set draws a reduced state uniformly and then draws uniformly
among its feasible actions. The **narrow** data set follows trajectories from
the empty system under a decode-priority heuristic and selects a random
feasible action with probability 0.05. Both use seed 0 and the same MDP. Their
different state-action sampling distributions are the controlled change.

Each fitted-Q sweep constructs targets

$$
y_i^{(n)}=c_i+0.99\min_{u'\in\mathcal A(s_i')}
q_n(s_i',u')
$$

and fits a joint state-action Extra-Trees regressor. The ensemble contains 200
trees with maximum depth 12 and minimum leaf size 2. The random seed is 0;
successive tree fits use the deterministic seeds 0 through 49 across 50
fitted-Q sweeps. The three scheduling actions enter as a one-hot input,
matching the joint state-action representation in the generic batch algorithm
above. Before each fit, rows with the same state and action are merged into one
point whose target is their average and whose weight is their count. The
minimum leaf size of 1 therefore counts distinct state-action pairs, not rows:
a larger value would force every leaf to average the targets of at least two
different pairs, which blurs the small differences between actions in the same
state.

Changing only the sampling distribution tests how coverage affects an
otherwise identical FQI procedure. The exact dynamic-programming policy and
both fitted policies are evaluated on the same 2,000 seeded episodes, each 300
steps long. Return and policy disagreement measure agreement with the reduced
MDP objective and its exact solution. Mean waiting time, dropped arrivals, and
observed state-action coverage locate operational differences. Decode stalls
count periods in which decode jobs were active but the policy did something
else. In this model they are not errors: the exact policy stalls decode on
purpose while it prefills to grow the batch.

```{code-cell} python
:tags: [remove-input]
:label: fig-inference-scheduling-fqi
:caption: Extra-Trees FQI on two fixed 50,000-transition data sets sampled from the same measured-L4-calibrated scheduling MDP. These are model transitions, not vLLM logs. The broad data set samples states and feasible actions uniformly; the narrow data set follows decode priority with 5 percent random actions. Each policy table covers all 49 states and marks whether the displayed action occurs in that policy's training buffer.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="fqi",
)))
```

:::{figure} _static/inference_serving/fqi.svg
:label: fig-inference-scheduling-fqi-fallback
:class: pdf-fallback
:alt: Static comparison of exact value iteration with fitted Q policies learned from broad and narrow transition data.

Policy tables over all 49 states. At each state, a white circle marks that the
buffer contains visits to the action selected by the displayed policy. The
interactive version adds playback, stepping, and scrubbing through a queue
trajectory sampled from the reduced transition model rather than observed
directly from vLLM.
:::

Exact dynamic programming and the two fitted-Q policies use the same 2,000
seeded episodes of 300 steps. Coverage is computed from the feasible
state-action pairs present in each fixed transition buffer.

```{code-cell} python
:tags: [remove-input]

import pandas as pd

fqi_metrics = pd.read_csv(
    "artifacts/inference_serving/metrics_fqi.csv"
).set_index("controller")
fqi_metrics[
    [
        "mean_discounted_return",
        "mean_queue_length",
        "mean_waiting_time_s",
        "mean_decode_stalls",
        "mean_dropped_arrivals",
        "coverage_fraction",
        "policy_disagreement_fraction",
    ]
].T.rename_axis("metric").round(3)
```

{download}`Download every fitted-Q metric (CSV) <artifacts/inference_serving/metrics_fqi.csv>`

The broad buffer covers every feasible state-action pair, and its fitted policy
disagrees with the exact policy in 6 of the 49 states (12.24 percent). All six
lie next to the switching curve, where the exact policy prefills and the fitted
policy decodes. Each of these state-action pairs appears about 340 times in the
buffer, so the errors do not come from missing data. In those states the exact
action values of prefill and decode differ by at most 1.24, while the values
themselves range from 254 to 891. A regressor that is accurate to a fraction of
a percent can still rank two actions this close in the wrong order. The broad
policy's mean discounted return is $-242.25$, compared with $-230.04$ for the
exact policy, and its mean waiting time is 5.76 seconds rather than 5.24.

The narrow buffer covers 41.73 percent of feasible pairs and disagrees with the
exact policy in 19 states (38.78 percent). In 15 of those states the buffer
contains no visit to the action the fitted policy selects. The log follows
decode priority, which never prefills while a request is decoding, so the
regressor has no data on the batching decisions that the exact policy makes.
Its choices there are extrapolations: it idles in several states with six
requests decoding, and it prefills past the point where the exact policy
decodes. Its mean discounted return is $-286.08$, its mean waiting time is
11.20 seconds, and it drops 1.40 arrivals per episode against 0.20 for the
exact policy. Return is the negative of discounted cost, so higher values are
better.

The two data sets have equal size, so the difference between their policies
cannot be attributed to the number of transition rows. The broad buffer's
errors come from approximation where the action gap is small; the narrow
buffer's errors come from coverage, where the behavior policy never took the
action in question. These are the first and the third difficulty named after
the error bound of the [approximate Bellman chapter](approximate-bellman-equations.md#sec-abe-error):
a fit is accurate on average, not at the states where two actions are nearly
tied, and it is controlled only where there are data.

The hardware calibration is measured, while training and evaluation remain
inside the stated 49-state MDP. These results concern offline coverage under
that abstraction. They are not a benchmark of vLLM scheduling or of NVIDIA L4
hardware across deployments.

:::{dropdown} Inspect the fitted-Q experiment
```{literalinclude} code/inference_control.py
:language: python
:start-at: def fit_fqi
:end-before: def _reduced_policy_trajectory
:linenos:
```
{download}`Download the complete inference-control implementation <code/inference_control.py>`
:::

(sec-fqi-conjunctions)=
### Example: Collision Avoidance Learned from Logged Warnings

What does fitted Q-iteration look like when there is no model, only a log?
When a satellite and another object, most often a piece of debris, are
predicted to pass close to each other, the operator receives about three
conjunction data messages per day over the week before the time of closest
approach. Each message gives a collision probability estimated from the latest
tracking. The operator can plan an avoidance maneuver or wait for the next
message. Waiting buys information, because most alarms go away as the
tracking improves. It also makes the maneuver more expensive: closer to the
encounter, a larger change of velocity is needed for the same miss distance,
and there is less time to plan. ESA plans its maneuvers at least two days
before closest approach, and its missions in low orbit react to collision
probabilities between $10^{-5}$ and $10^{-4}$ {cite}`Uriot2022`. ESA released
the messages of 13,154 conjunctions of its satellites for a machine-learning
competition {cite}`Uriot2022`, and we use them as the log.

We pose an optimal stopping problem on these sequences. A decision is taken at
each message received at least two days before closest approach, and once
more at the two-day deadline with the latest message. A maneuver $t$ days
before closest approach costs $c(t) = (1 + 2/t)/2$, in units of a maneuver two
days out. An event without a maneuver costs $C = 10^5$ times the collision
probability of its last message, the best estimate available, so that a
maneuver at the deadline breaks even at a probability of $10^{-5}$. The state
is what the operator knows at a decision: the time $t$ to closest approach,
the latest log-probability $\rho$, and the age of that message. The costs are
teaching choices, and ESA's analysts look at much more than one number.

Two features of the problem let fitted Q-iteration work from the log alone.
A maneuver ends the problem at a known cost, so $q(s, \text{maneuver}) = c(t)$
needs no learning. And waiting does not change which messages arrive: the
released sequences contain no maneuvers, so each one shows what follows
"wait". Fitted Q-iteration only has to learn $W(s) = q(s, \text{wait})$, by
regression on the logged transitions $(s_i, s_i')$ with targets

$$
y_i = \begin{cases} C\, p_{\text{final}} & \text{if } s_i \text{ is the deadline,} \\
\min\{c(t_i'),\, W_k(s_i')\} & \text{otherwise,} \end{cases}
\qquad W_{k+1} = \operatorname{fit}\big(\{(s_i, y_i)\}\big).
$$

The minimum over the two actions is the $\max$ of the Bellman operator for
costs. There is no expectation to compute: each logged successor is one draw
of $s'$, the single-sample paradigm of the
[Monte Carlo chapter](monte-carlo-bellman-estimation.md). We fit extra
trees, the averager of tree-based FQI {cite}`ernst2005tree`, on the logged
decision epochs, and judge every policy out of sample: each event is decided
by a policy fitted without it.

:::{figure} _static/conjunction_wait_or_burn/wait_or_burn.svg
:label: fig-conjunction-wait-or-burn

Collision avoidance from ESA's conjunction messages. (a) The collision
probability of the 742 events whose estimate crossed the break-even
probability, about $10^{-5}$ (dashed), before the deadline. The 59 in color
ended above it. (b) Total cost of each policy over all events, split into the
cost of maneuvers and of expected collisions. Never maneuvering costs 2,668.
(c) The policy learned by fitted Q-iteration on all events, at a message that
has just arrived. Above the blue boundary it maneuvers. The dashed line is the
break-even probability of the message taken at face value.
:::

:::{include} artifacts/conjunction_wait_or_burn/results.md
:::

Maneuvering at the first alarm is safe but wasteful: it makes 742
maneuvers for 62 events that needed one, because 92 of every 100 alarms
resolve ({numref}`fig-conjunction-wait-or-burn`, panel (a)). Waiting until the
deadline and then acting on the latest message halves the cost. This rule is
what an operator would write by hand. Fitted Q-iteration was told neither
that waiting pays nor at what probability to act, and it comes within about 7% of
the hand rule, 313 against 291. Its learned rule (panel (c)) waits through a
new alarm more than three days out unless the probability is twenty to forty-five
times the break-even value, in which case it maneuvers early, while a
maneuver is still cheap. The bar comes down as the deadline nears, but at the
deadline it still asks for about three times the break-even probability, a
bet that later messages will come down, and that bet misses more of the
events that needed a maneuver, 20 against 6 for the hand rule. Two choices
decide whether fitted Q-iteration works here at all, and both follow from
the error analysis of the previous chapters.

1. **The fit must fit the right thing.** The raw target $C\, p_{\text{final}}$
   is heavy-tailed. One event in the log ends at a probability of 2% and
   costs about 2,070 if nothing is done, more than three times the cost of
   all other events together. Extra trees predict the average of the targets
   in a leaf, so this one cost is shared by every state in the leaves where
   its epochs fall, and those states look dangerous. Fitted on all events,
   the raw-target policy makes 342 maneuvers; fitted without that single
   event, 141. Out of sample it makes 374 maneuvers and costs 441, half again
   as much as the hand rule. The second version fits the ratio
   $y_i / (C\, 10^{\rho_i})$, the cost relative to what the current message
   alone predicts, and multiplies the fit back. The regressor then learns a
   correction factor to the operator's own estimate. For the dangerous event
   this factor stays between 0.7 and 7, an ordinary value, and the policy
   makes 166 maneuvers with the event in the data and 175 without it.
2. **The fit is only as good as the data where decisions are made.** Five
   in six of the 123,882 decision epochs have probabilities below $10^{-7}$,
   where waiting is obviously right, and we leave them out of the fit. Only
   62 events needed a maneuver. This is the coverage problem behind the
   concentrability coefficients of the
   [approximate Bellman chapter](approximate-bellman-equations.md#sec-abe-error):
   the regression error that matters is the one at the few states where the
   two actions are close, and no amount of data elsewhere reduces it. The
   hindsight policy, at 147, is not a target: it knows the estimate of the
   messages that arrive after the deadline, which no policy can use.

- {download}`Conjunction model and fitted Q-iteration <code/conjunction_wait_or_burn.py>`
- {download}`Conjunction experiment builder <scripts/build_conjunction_wait_or_burn_artifacts.py>`
- {download}`Conjunction results <artifacts/conjunction_wait_or_burn/metrics.json>`
- {download}`Conjunction messages used (columns of the ESA training set, CC BY 4.0) <data/conjunctions/kelvins_cdm_extract.csv.gz>`

(sec-fqi-flattening)=
## From Nested Loops to Target Networks

Can the outer Bellman loop and the inner regression loop be interleaved
without losing track of which parameters generate and fit each target?

Fitted Q-iteration has a nested structure: the outer loop computes Bellman
targets, and the inner loop fits the function approximator to them. For a
tree ensemble the inner loop is one call that builds the trees, and the
nesting is the only possible presentation. For a neural network the inner
loop is a sequence of gradient steps,

$$
\boldsymbol{\theta}_n^{(k+1)} = \boldsymbol{\theta}_n^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}\big(\boldsymbol{\theta}_n^{(k)}; \mathcal{D}_n^{\text{fit}}\big), \qquad k = 0, \ldots, K-1,
$$

starting from $\boldsymbol{\theta}_n^{(0)} = \boldsymbol{\theta}_n$ and
ending with $\boldsymbol{\theta}_{n+1} = \boldsymbol{\theta}_n^{(K)}$, where
$\mathcal{D}_n^{\text{fit}}$ is the regression data set of the $n$-th outer
iteration. Two loops of counted steps can be written as one, and the
rewriting shows where the target network of DQN comes from.

### Making the Nested Structure Explicit

Starting from {prf:ref}`fitted-q-iteration-batch`, replacing the `fit` call
by $K$ explicit gradient steps gives the following.

```{prf:algorithm} Neural Fitted Q-Iteration with Explicit Inner Loop
:label: nfqi-explicit-inner

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, inner optimization steps $K$, initialization $\boldsymbol{\theta}_0$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$  $\quad$ // **Outer loop counter (value iteration)**
3. **repeat**
    1. **// Compute Bellman targets using current Q-function**
    2. $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
    3. **for** each $(s,a,r,s') \in \mathcal{D}$ **do**
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    4. **// Inner optimization loop: fit to targets via gradient descent**
    5. $\boldsymbol{\theta}_n^{(0)} \leftarrow \boldsymbol{\theta}_n$ $\quad$ // Warm start from previous outer iteration
    6. $k \leftarrow 0$ $\quad$ // **Inner loop counter (regression)**
    7. **repeat**
        1. $\boldsymbol{\theta}_n^{(k+1)} \leftarrow \boldsymbol{\theta}_n^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n^{(k)}; \mathcal{D}_n^{\text{fit}})$
        2. $k \leftarrow k + 1$
    8. **until** $k = K$ $\quad$ // Partial optimization: exactly K gradient steps
    9. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n^{(K)}$
    10. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```

The targets of outer iteration $n$ are computed with $\boldsymbol{\theta}_n$,
the parameters at the end of the previous outer iteration, and they stay
fixed for all $K$ inner steps. The inner iterates
$\boldsymbol{\theta}_n^{(k)}$ for $k > 0$ never enter a target. The fit is
warm-started from $\boldsymbol{\theta}_n$, the very parameters that computed
the targets, and it stops after $K$ steps rather than at convergence: these
are the two departures from supervised learning that neural fitted
Q-iteration introduced, and they are kept by everything that follows.

### Flattening into a Single Loop

Replace the pair of counters $(n, k)$ by a single step counter $t = nK + k$.
The parameters that compute the targets, $\boldsymbol{\theta}_n$, are held
in a separate copy that changes only when $t$ crosses a multiple of $K$, the
boundary between two outer iterations. That copy is the **target network**
$\boldsymbol{\theta}_{\text{target}}$. The parameters that the gradient
steps update, $\boldsymbol{\theta}_t$, correspond to the inner iterates
$\boldsymbol{\theta}_n^{(k)}$, and are called the **online network**, online
in the sense of being continually updated, not in the sense of how the data
were collected. Nothing else changes.

```{prf:algorithm} Flattened Neural Fitted Q-Iteration
:label: nfqi-flattened

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, initialization $\boldsymbol{\theta}_0$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. $t \leftarrow 0$ $\quad$ // **Single flattened loop counter**
4. **while** training **do**
    1. **// Compute targets using fixed target network**
    2. $\mathcal{D}_t^{\text{fit}} \leftarrow \emptyset$
    3. **for** each $(s,a,r,s') \in \mathcal{D}$ **do**
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{\text{target}})$
        2. $\mathcal{D}_t^{\text{fit}} \leftarrow \mathcal{D}_t^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    4. **// Gradient step on online network**
    5. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t; \mathcal{D}_t^{\text{fit}})$
    6. **// Periodic target network update (marks outer iteration boundary)**
    7. **if** $t \bmod K = 0$ **then**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    8. $t \leftarrow t + 1$
5. **return** $\boldsymbol{\theta}_t$
```

At step $t$, the outer iteration is $n = \lfloor t/K \rfloor$ and the inner
step is $k = t \bmod K$. The target network equals $\boldsymbol{\theta}_n$
for $t$ from $nK$ to $(n+1)K - 1$ and becomes $\boldsymbol{\theta}_{n+1}$ at
$t = (n+1)K$, and $\boldsymbol{\theta}_t$ is $\boldsymbol{\theta}_n^{(k)}$.
With $K = 10$, the inner step $k = 7$ of outer iteration $n = 3$ is the
global step $t = 37$. The data set is still the fixed $\mathcal{D}$; only the
indexing has changed. A target network, in this reading, is the boundary of
an outer iteration of fitted Q-iteration, kept in memory. When DQN is
presented with a target network updated every $K$ steps, it is this
flattened fitted Q-iteration with an evolving buffer, which the next section
adds.

### Smooth Target Updates via Exponential Moving Average

The periodic copy can be replaced by a target network that tracks the online
network a little at every step,

$$
\boldsymbol{\theta}_{\text{target}} \leftarrow \tau\,\boldsymbol{\theta}_{t+1} + (1-\tau)\,\boldsymbol{\theta}_{\text{target}},
$$

an **exponential moving average** (also called Polyak averaging) with rate
$\tau \in (0, 1)$. For small $\tau$, typically between $0.001$ and $0.01$,
the target lags the online network by roughly $1/\tau$ steps, and the targets
drift continuously instead of jumping every $K$ steps. The scheme became
standard with DDPG {cite}`lillicrap2015continuous` and is used by TD3
{cite}`fujimoto2018addressing` and SAC {cite}`haarnoja2018soft`.

```{prf:algorithm} Flattened Neural Fitted Q-Iteration with EMA Target Updates
:label: nfqi-flattened-ema

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, EMA rate $\tau \in (0, 1)$, initialization $\boldsymbol{\theta}_0$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. $t \leftarrow 0$ $\quad$ // **Single flattened loop counter**
4. **while** training **do**
    1. **// Compute targets using fixed target network**
    2. $\mathcal{D}_t^{\text{fit}} \leftarrow \emptyset$
    3. **for** each $(s,a,r,s') \in \mathcal{D}$ **do**
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{\text{target}})$
        2. $\mathcal{D}_t^{\text{fit}} \leftarrow \mathcal{D}_t^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    4. **// Gradient step on online network**
    5. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t; \mathcal{D}_t^{\text{fit}})$
    6. **// CHANGED: Smooth EMA update at every step**
    7. $\boldsymbol{\theta}_{\text{target}} \leftarrow \tau\boldsymbol{\theta}_{t+1} + (1-\tau)\boldsymbol{\theta}_{\text{target}}$
    8. $t \leftarrow t + 1$
5. **return** $\boldsymbol{\theta}_t$
```

(sec-fqi-online)=
## Online Data: DQN and Its Variants

What changes when the buffer evolves during learning and a neural Q-function
is updated from replayed transitions?

The flattened algorithm sampled from a fixed data set. Let the data set
change instead: a **replay buffer** $\mathcal{B}_t$ of fixed capacity $B$
receives one new transition at every step of interaction and drops the
oldest when full, so that $\mathcal{B}_t = \{\tau_{t-B+1}, \ldots, \tau_t\}$
and its empirical distribution $\hat{P}_{\mathcal{B}_t}$ slides forward in
time. Each gradient step uses a mini-batch of $b$ transitions drawn
uniformly from the current buffer. The online network acquires a second
role: besides being the parameters under training, it is the policy that
collects the new transitions, through $\varepsilon$-greedy action selection.
"Online" now has both meanings at once, continually updated parameters and
data collected during learning.

Deep Q-Network (DQN) {cite}`mnih2013atari` is this algorithm with a
convolutional network and moderate settings: a buffer of about $10^6$
transitions, mini-batches of about 32, and a target network copied every
$K \approx 10^4$ steps. It is flattened fitted Q-iteration with an evolving
buffer, and each of its components has a place in the nested view: the
target network marks the outer iterations, the mini-batch gradient step is
the inner loop, and the buffer is the data set, now changing slowly under
the policy that the online network defines.

```{prf:algorithm} Deep Q-Network (DQN)
:label: dqn

**Input**: MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, replay buffer capacity $B$, mini-batch size $b$, exploration rate $\varepsilon$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. Initialize replay buffer $\mathcal{B}$ with capacity $B$
4. $t \leftarrow 0$
5. **while** training **do**
    1. Observe current state $s$
    2. **// Use online network to collect data**
    3. Select action: $a \leftarrow \begin{cases} \arg\max_{a'} q(s,a';\boldsymbol{\theta}_t) & \text{with probability } 1-\varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}$
    4. Execute $a$, observe reward $r$ and next state $s'$
    5. Store $(s,a,r,s')$ in $\mathcal{B}$, replacing oldest if full
    6. Sample mini-batch of $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$ from $\mathcal{B}$
    7. For each sampled transition $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma \color{blue}{\max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{\text{target}})}$ (both selection AND evaluation with target network)
    8. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b (q(s_i,a_i;\boldsymbol{\theta}_t) - y_i)^2$
    9. **if** $t \bmod K = 0$ **then**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    10. $t \leftarrow t + 1$
6. **return** $\boldsymbol{\theta}_t$
```

The deadly triad of the [approximate Bellman chapter](approximate-bellman-equations.md)
is all here: a neural network, bootstrapped targets, and a buffer whose
transitions were collected by earlier policies. The design choices are what
keep it in check. Targets are held fixed for $K$ steps by the target network,
each fit is warm-started, and the large buffer averages the data over many
policies, so that the distribution the fit sees changes slowly even though
the policy changes at every step.

(sec-fqi-double-dqn)=
### Double DQN

The target of DQN, $\max_{a'} q(s', a'; \boldsymbol{\theta}_{\text{target}})$,
uses the target network twice, to select the best next action and to value
it, and the [Monte Carlo chapter](monte-carlo-bellman-estimation.md#sec-mc-overestimation)
showed that such a maximum is biased upward by the reuse of the winning
noise. Double DQN {cite}`van2016deep` separates the two roles between the
two networks that DQN already maintains: the online network selects,
$a^*_i = \arg\max_{a'} q(s'_i, a'; \boldsymbol{\theta}_t)$, and the target
network evaluates, $y_i = r_i + \gamma\, q(s'_i, a^*_i; \boldsymbol{\theta}_{\text{target}})$.
The buffer and the fit are unchanged; only the target function differs.

```{prf:algorithm} Double Deep Q-Network (Double DQN)
:label: double-dqn

**Input**: MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, replay buffer capacity $B$, mini-batch size $b$, exploration rate $\varepsilon$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. Initialize replay buffer $\mathcal{B}$ with capacity $B$
4. $t \leftarrow 0$
5. **while** training **do**
    1. Observe current state $s$
    2. **// Use online network to collect data**
    3. Select action: $a \leftarrow \begin{cases} \arg\max_{a'} q(s,a';\boldsymbol{\theta}_t) & \text{with probability } 1-\varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}$
    4. Execute $a$, observe reward $r$ and next state $s'$
    5. Store $(s,a,r,s')$ in $\mathcal{B}$, replacing oldest if full
    6. Sample mini-batch of $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$ from $\mathcal{B}$
    7. For each sampled transition $(s_i,a_i,r_i,s_i')$:
        1. $a^*_i \leftarrow \color{red}{\arg\max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_t)}$ (action selection with online network)
        2. $y_i \leftarrow r_i + \gamma \color{green}{q(s_i',a^*_i; \boldsymbol{\theta}_{\text{target}})}$ (action evaluation with target network)
    8. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b (q(s_i,a_i;\boldsymbol{\theta}_t) - y_i)^2$
    9. **if** $t \bmod K = 0$ **then**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    10. $t \leftarrow t + 1$
6. **return** $\boldsymbol{\theta}_t$
```

The two networks are not independent estimates in the sense of the Monte
Carlo chapter's argument, since the target network is a delayed copy of the
online one. Their errors are nonetheless less correlated than the errors of
one network used twice, because they were fitted to different targets at
different times, and the reduction in overestimation that this buys is
substantial in practice.

### Q-Learning: The Limiting Case

DQN uses a buffer of about $10^6$ transitions, a target network updated
every $10^4$ steps and mini-batches of 32. At the opposite extreme, with a
buffer of one transition, a target network updated at every step and a
mini-batch of one, the algorithm is Q-learning {cite}`watkins1989learning`.
Each transition is used once and discarded, $\hat{P}_{\mathcal{B}_t}$ is a
point mass at the current transition, and the parameters that compute the
target are the ones just updated.

Q-learning is a **stochastic approximation** method. The Robbins–Monro
procedure solves an equation $F(\boldsymbol{\theta}) = 0$ in which
$F(\boldsymbol{\theta}) = \mathbb{E}[f(\boldsymbol{\theta}; Z)]$ is an
expectation that is never computed, by the update
$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t f(\boldsymbol{\theta}_t; Z_t)$
on one noisy draw $Z_t$ at a time; under step sizes with $\alpha_t \to 0$
and $\sum_t \alpha_t = \infty$ the iterates converge to a solution. For
Q-learning the equation is the Bellman equation in residual form,
$\Bellman q - q = 0$, and the draw is a transition. The residual at a
transition $(s,a,r,s')$ is the **TD error**

$$
R(s,a,r,s'; \boldsymbol{\theta}) = r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}) - q(s,a; \boldsymbol{\theta}),
$$

whose expectation over transitions is the Bellman residual at $(s,a)$,
averaged over the distribution of the behavior policy. Each transition is
one noisy sample of it, and the update takes a gradient step on the squared
TD error with the target held fixed, the semi-gradient of the approximate
Bellman chapter.

```{prf:algorithm} Q-Learning
:label: q-learning

**Input**: MDP $(S, A, P, R, \gamma)$, function approximator $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, exploration rate $\varepsilon$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$
2. $t \leftarrow 0$
3. **while** training **do**
    1. Observe current state $s$
    2. Select action: $a \leftarrow \begin{cases} \arg\max_{a' \in A} q(s,a';\boldsymbol{\theta}_t) & \text{with probability } 1-\varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}$
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Compute TD target: $y \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_t)$
    5. Update: $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} (q(s,a;\boldsymbol{\theta}_t) - y)^2$
    6. $t \leftarrow t + 1$
4. **return** $\boldsymbol{\theta}_t$
```

The gradient in line 5 depends on the function class. With one-hot features
$\boldsymbol{\phi}(s,a) = \boldsymbol{e}_{(s,a)}$ the update is the tabular
rule $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max_{a'} Q(s',a') - Q(s,a))$,
which converges under the stochastic-approximation conditions. With linear
features $q(s,a;\boldsymbol\theta) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(s,a)$
the gradient is $\boldsymbol{\phi}(s,a)$, and convergence is not guaranteed:
least squares is not an averager, and the max operator keeps the on-policy
argument from applying. With a neural network the gradient comes from
backpropagation and there are no guarantees at all.

### Convergence Analysis via the ODE Method

Tabular Q-learning converges under diminishing step sizes and sufficient
exploration {cite}`watkins1992qlearning,jaakkola1994convergence,tsitsiklis1994asynchronous`.
The standard proof technique is the **ordinary differential equation (ODE)
method** {cite}`kushner2003stochastic,borkar2000ode`, which studies a
stochastic approximation through an associated deterministic dynamical
system. The argument has two steps. As the step sizes shrink, the stochastic
recursion $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha_t h(\boldsymbol{\theta}_t, Z_t)$
tracks the ODE $\dot{\boldsymbol{\theta}} = \bar{h}(\boldsymbol{\theta})$,
where $\bar{h}(\boldsymbol{\theta}) = \mathbb{E}[h(\boldsymbol{\theta}, Z)]$
is the expected update direction. Then the ODE is shown to converge, with a
Lyapunov function, and the stochastic algorithm converges to the same point.

For tabular Q-learning, the update at the visited pair $(s,a)$ is
$Q_{t+1}(s,a) = Q_t(s,a) + \alpha_t [r + \gamma \max_{a'} Q_t(s',a') - Q_t(s,a)]$,
and its expectation over the transition, given that $(s,a)$ is visited, is
the Bellman residual at $(s,a)$. The ODE is

$$
\frac{dQ(s,a)}{dt} = \sum_{s'} p(s'|s,a) \big[r(s,a,s') + \gamma \max_{a'} Q(s',a') \big] - Q(s,a),
$$

whose equilibria are the solutions of the Bellman optimality equation, that
is, $Q^*$ alone. The distance $\|Q - Q^*\|_\infty$ decreases along its
trajectories because the Bellman operator is a contraction, which is the
Lyapunov function. Exploration enters through the requirement that every
pair be visited infinitely often; under ergodicity the frequency with which
each $(s,a)$ is updated converges to the stationary distribution of the
behavior policy, which the ODE absorbs as a per-coordinate time scale. With
linear features the ODE no longer has the Bellman residual as its drift at
every coordinate, the projection enters, and the [approximate Bellman
chapter](approximate-bellman-equations.md) showed what can go wrong then.

(sec-fqi-design-space)=
## The Design Space of Fitted Q-Iteration

Which choices generate the algorithms of this chapter, and how do they
combine?

Every algorithm above has the same two-level structure, with three
ingredients at any time $t$:

1. A **buffer** $\mathcal{B}_t$ of transitions, with its empirical
   distribution $\hat{P}_{\mathcal{B}_t}$. It is a fixed data set
   $\mathcal{D}$ in the batch algorithms, a circular buffer of capacity $B$
   in DQN, and the single current transition in Q-learning.
2. A **target function** $g(s,a,r,s'; \boldsymbol{\theta})$ that computes a
   regression target from one transition and the parameters of the current
   Q-function: $r + \gamma \max_{a'} q(s',a';\boldsymbol\theta)$ for the hard
   maximum, the log-sum-exp of the [regularized chapter](regularized-dp.md)
   for the smooth one, and the decoupled selection and evaluation of Double
   DQN. Unlike the Bellman operator, which acts on functions, $g$ acts on
   tuples.
3. A **fit**: a loss $\ell$, an initialization $\boldsymbol{\theta}_{\text{init}}$,
   and an optimization budget $K$.

Pushing the buffer through the target function turns the distribution over
transitions into a distribution over regression pairs,
$\hat{P}_t^{\text{fit}} = (\mathrm{id}, g(\cdot;\boldsymbol\theta_t))_\#\hat{P}_{\mathcal{B}_t}$,
and the fit minimizes the empirical risk
$\mathbb{E}_{((s,a),y) \sim \hat{P}_t^{\text{fit}}}[\ell(q(s,a;\boldsymbol\theta), y)]$.
The buffer distribution changes only when data are collected; the
regression distribution changes every time the targets are recomputed. In
the abstract, the whole family is

$$
\begin{aligned}
&\textbf{repeat } n = 0, 1, 2, \ldots \\
&\quad \text{Sample transitions from } \hat{P}_{\mathcal{B}_n} \text{ and construct targets via } g(\cdot; \boldsymbol{\theta}_n) \\
&\quad \boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\hat{P}_n^{\text{fit}}, \boldsymbol{\theta}_{\text{init}}, K) \\
&\textbf{until } \text{convergence.}
\end{aligned}
$$

How often the data are reused is measured by the **replay ratio**, the
number of optimization steps per transition collected:

$$
\text{Replay ratio} = \begin{cases}
K \cdot N_{\text{epochs}} \cdot N / b & \text{batch (batch size } b \text{ from } N \text{ transitions)} \\
b & \text{DQN (one step per transition, batch size } b \text{)} \\
1 & \text{Q-learning.}
\end{cases}
$$

A large replay ratio reuses the same empirical distribution many times,
which is sample-average approximation; a replay ratio of one uses each
transition once, which is stochastic approximation. More reuse lowers the
variance of the estimates under $\hat{P}_{\mathcal{B}_t}$ and raises the
risk of fitting the idiosyncrasies of a buffer that reflects old policies or
narrow coverage.

```{prf:remark} Two notions of bootstrap
:class: dropdown

Two different "bootstraps" meet in fitted Q-iteration. Sampling with replacement from $\hat{P}_{\mathcal{B}_t}$ approximates expectations under the unknown $P$ by expectations under the empirical distribution, which is the bootstrap of statistics; mini-batch training is this resampling, and the replay ratio is the number of bootstrap samples per transition. Computing $y = r + \gamma \max_{a'} q(s',a';\boldsymbol{\theta})$ replaces the unknown continuation value by the current estimate, which is the bootstrap of temporal-difference learning and the source of the maximization bias of the previous chapter. The buffer encodes the first, the target function the second, and the bias corrections (double estimation, learned corrections, Gumbel losses) act on the second alone.
```

The table places the algorithms of this chapter, and of the next, in this
space.

| **Algorithm** | **Approximator** | **Target** | **Fit budget** | **Initialization** | **Data** | **Bias fix** |
|:--------------|:-----------------|:-----------|:---------------|:-------------------|:---------|:-------------|
| FQI {cite}`ernst2005tree` | Extra trees | Hard max | Full | Cold | Fixed batch | None |
| NFQI {cite}`riedmiller2005neural` | Neural net | Hard max | Full | Warm | Fixed batch | None |
| Q-learning {cite}`watkins1989learning` | Any | Hard max | $K=1$ | Warm | One transition | None |
| DQN {cite}`mnih2013atari` | Deep net | Hard max | $K=1$ | Warm | Replay buffer | None |
| Double DQN {cite}`van2016deep` | Deep net | Decoupled max | $K=1$ | Warm | Replay buffer | Double estimation |
| Soft Q {cite}`haarnoja2017reinforcement` | Neural net | Log-sum-exp | $K$ steps | Warm | Replay buffer | None |

The continuous-action methods NFQCA, DDPG and SAC add one more axis, the
actor that replaces the maximum in the target, and belong to the
[next chapter](amortized-action-optimization.md).

## Regression Losses and Noise Models

How should the fitting loss change when Bellman targets have asymmetric or
heavy-tailed noise rather than Gaussian residuals?

At a fixed time $t$, the fit is a statistical estimation problem: given
samples from $\hat{P}_t^{\text{fit}}$, choose $\boldsymbol{\theta}$ to
minimize $\mathbb{E}_{((s,a),y)\sim \hat{P}_t^{\text{fit}}}[\ell(q(s,a;\boldsymbol{\theta}), y)]$.
The loss $\ell$ is a noise model for the targets. Squared error is the
likelihood of Gaussian noise, absolute error of Laplace noise, Gumbel
regression of extreme-value noise, and a classification loss over value bins
makes no parametric assumption. Bellman targets are noisy, biased by the
maximum, and bootstrapped, so the choice is not neutral, and two
alternatives to squared error have done well in practice.

### Gumbel Regression

Extreme value theory says that the maximum of several Gaussian errors has
Gumbel-distributed tails. {cite:t}`garg2023extreme` took the distribution
seriously in Extreme Q-Learning (XQL): instead of modeling the error of a
target as Gaussian,

$$
y_i = q^*(s_i, a_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2),
$$

they model it as Gumbel,

$$
y_i = q^*(s_i, a_i) + \varepsilon_i, \quad \varepsilon_i \sim -\text{Gumbel}(0, \beta),
$$

with the sign chosen because the targets overestimate. The maximum
likelihood loss is Gumbel regression,

$$
\mathcal{L}_{\text{Gumbel}}(\boldsymbol{\theta}) = \sum_i \left[\frac{q(s_i, a_i; \boldsymbol{\theta}) - y_i}{\beta} + \exp\left(\frac{q(s_i, a_i; \boldsymbol{\theta}) - y_i}{\beta}\right)\right],
$$

with a temperature $\beta$ that sets the heaviness of the tail, and its
gradient with respect to $q$ is

$$
\frac{\partial \mathcal{L}_{\text{Gumbel}}}{\partial q} = \frac{1}{\beta}\left[1 + \exp\left(\frac{q - y}{\beta}\right)\right].
$$

When $q < y$ the exponential is small and the gradient is mild; when $q > y$
it grows exponentially with the error. The loss penalizes overestimation
much more than underestimation, which pushes the fit toward conservative
values when the targets are biased upward by the maximum. Squared error is
the right likelihood for averages, by the central limit theorem; Gumbel
regression is the right likelihood for maxima of light-tailed noise, by
extreme value theory.

```{code-cell} python
:tags: [hide-input]
:label: fig-gumbel-loss
:caption: Gumbel regression against squared error, as functions of the error $q - y$ of a prediction. (a) The loss grows exponentially for overestimates ($q > y$) and linearly for underestimates; a smaller temperature $\beta$ makes the asymmetry sharper. (b) Its derivative with respect to $q$, which sets how hard the fit pushes back: steep for overestimates, flat near $1/\beta$ for underestimates, while the derivative of squared error (dashed) is symmetric.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))
from book_figure_theme import INK, MUTED, PALETTE, figure_style, show

error = np.linspace(-3, 3, 500)                       # q - y
betas = {0.5: PALETTE["vermilion"], 1.0: PALETTE["orange"], 2.0: PALETTE["blue"]}

with figure_style():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for beta, color in betas.items():
        ax.plot(error, error / beta + np.exp(error / beta), color=color)
        bx.plot(error, (1 + np.exp(error / beta)) / beta, color=color)
    ax.plot(error, error ** 2, color=MUTED, lw=1, ls="--")
    bx.plot(error, 2 * error, color=MUTED, lw=1, ls="--")
    for axis in (ax, bx):
        axis.axvline(0, color=MUTED, lw=.6)
        axis.set_xlabel("error $q - y$")
        axis.set_xlim(-3, 3)
    ax.set_ylim(0, 10)
    bx.set_ylim(-5, 10)
    ax.text(.55, 9.2, r"$\beta = 0.5$", color=betas[0.5], fontsize=8.5, ha="right")
    ax.text(1.75, 8.0, r"$\beta = 1$", color=betas[1.0], fontsize=8.5, ha="right")
    ax.text(2.95, 3.2, r"$\beta = 2$", color=betas[2.0], fontsize=8.5, ha="right")
    ax.text(-2.9, 6.0, "squared\nerror", color=MUTED, fontsize=8.5)
    bx.text(-2.9, -3.8, "squared\nerror", color=MUTED, fontsize=8.5)
    ax.set_ylabel("loss")
    bx.set_ylabel(r"derivative with respect to $q$")
    ax.set_title("(a) Gumbel loss")
    bx.set_title("(b) Its gradient")
    fig.tight_layout(w_pad=2)
show(fig)
```

{numref}`fig-gumbel-loss` draws the loss and its gradient for three
temperatures. The loss handles the asymmetry of the noise on its own, with no variance to
estimate and no weights to compute, and XQL has improved both value-based
and actor-critic methods, in particular in offline reinforcement learning,
where the bias of the maximum compounds across iterations without new data
to correct it.

```{prf:remark} XQL and soft Q-learning: target function against loss
:class: dropdown

XQL and soft Q-learning both involve the Gumbel distribution, at different places in the design space. Soft Q-learning changes the target function, using the smooth Bellman operator of the [regularized chapter](regularized-dp.md),

$$
g^{\text{soft}}(s,a,r,s'; \boldsymbol{\theta}) = r + \gamma \frac{1}{\beta}\log\sum_{a'} \exp(\beta q(s',a'; \boldsymbol{\theta})),
$$

with the squared loss unchanged; the Gumbel distribution enters through the Gumbel-max construction of the log-sum-exp, which is about making the optimal policy stochastic, not about the noise of the targets. XQL keeps the hard maximum in the target and changes the loss to $\ell_{\text{Gumbel}}(q,y) = (q-y)/\beta + \exp((q-y)/\beta)$, where the Gumbel distribution models the noise of the TD errors. The log-sum-exp reduces maximization bias by averaging over actions, a property of the target; the Gumbel loss penalizes overestimation, a property of the loss. The two can be combined.
```

### Classification-Based Q-Learning

Nothing changes upstream: transitions are sampled from $\hat{P}_{\mathcal{B}_t}$
and pushed through the same target function. What changes is the
representation of the value and the loss. Choose a grid
$z_1 < z_2 < \cdots < z_K$ that spans the plausible returns. The network
outputs logits $\ell_{\boldsymbol{\theta}}(s,a) \in \mathbb{R}^K$, converted
to probabilities and to a scalar value by

$$
p_{\boldsymbol{\theta}}(k \mid s,a) = \frac{\exp(\ell_{\boldsymbol{\theta}}(s,a)_k)}{\sum_{j=1}^K \exp(\ell_{\boldsymbol{\theta}}(s,a)_j)}, \quad q(s,a; \boldsymbol{\theta}) = \sum_{k=1}^K z_k \, p_{\boldsymbol{\theta}}(k \mid s,a).
$$

Each scalar target $y_i$ becomes a distribution over the bins by the
**two-hot encoding**. If $y_i$ falls between $z_j$ and $z_{j+1}$,

$$
q_j(y_i) = \frac{z_{j+1} - y_i}{z_{j+1} - z_j}, \quad q_{j+1}(y_i) = \frac{y_i - z_j}{z_{j+1} - z_j}, \quad q_k(y_i) = 0 \text{ for } k \notin \{j, j+1\}.
$$

These are barycentric coordinates, and $\sum_k z_k q_k(y_i) = y_i$ recovers
the scalar exactly: the encoding is the linear interpolation of the
[finite-horizon chapter](finite-horizon-dp.md) ({prf:ref}`backward-recursion-interp`)
applied to the value axis.

```{code-cell} python
:tags: [hide-input]
:label: fig-two-hot
:caption: Two-hot encoding on 11 bins spanning $[-5, 5]$. (a) The target $y = 0.7$ puts weight $0.7$ on the bin at $1$ and $0.3$ on the bin at $0$, the barycentric coordinates of $0.7$ between them, so that the weighted average of the bins is $0.7$ again. (b) Three targets, each spread over the two bins around it.

def two_hot(y, bins):
    """Weights q_k(y): linear interpolation of y between the two bins around it."""
    q = np.zeros(len(bins))
    y = np.clip(y, bins[0], bins[-1])
    j = min(np.searchsorted(bins, y, side="right") - 1, len(bins) - 2)
    t = (y - bins[j]) / (bins[j + 1] - bins[j])
    q[j], q[j + 1] = 1 - t, t
    return q

bins = np.linspace(-5, 5, 11)
with figure_style():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    q = two_hot(0.7, bins)
    ax.bar(bins, q, width=.55, color=PALETTE["blue"])
    ax.axvline(0.7, ymax=.88, color=PALETTE["vermilion"], lw=1.2, ls="--")
    ax.text(0.7, 1.0, "$y = 0.7$", color=PALETTE["vermilion"], fontsize=8.5, ha="center")
    for k in np.flatnonzero(q):
        side = 1 if bins[k] > 0.7 else -1
        ax.text(bins[k] + .35 * side, q[k] - .03, f"{q[k]:.1f}", ha="left" if side > 0 else "right",
                va="top", fontsize=8.5, color=INK)
    ax.text(-4.8, .55, rf"$\Sigma_k\, z_k\, q_k(y) = {bins @ q:.1f}$", fontsize=8.5, color=INK)
    ax.set_title("(a) One target")
    for y, color, shift in ((-2.3, PALETTE["green"], -.18), (0.7, PALETTE["blue"], 0), (3.8, PALETTE["purple"], .18)):
        bx.bar(bins + shift, two_hot(y, bins), width=.18, color=color)
        bx.text(y, 1.0, f"$y = {y}$", color=color, fontsize=8.5, ha="center")
    bx.set_title("(b) Three targets")
    for axis in (ax, bx):
        axis.set_xticks(bins, [f"${b:.0f}$" for b in bins])
        axis.set_xlabel("bin value $z_k$")
    ax.set_ylabel("weight $q_k(y)$")
    ax.set_ylim(0, 1.12)
    fig.tight_layout(w_pad=2)
show(fig)
```

In {numref}`fig-two-hot`, each target puts its mass on the two bins around
it, with a weight that decreases with the distance to each bin. The loss is
the cross-entropy

$$
\mathcal{L}_{\text{CE}}(\boldsymbol{\theta}) = -\mathbb{E}_{((s,a),y) \sim \hat{P}_t^{\text{fit}}}\left[ \sum_{k=1}^K q_k(y) \log p_{\boldsymbol{\theta}}(k \mid s, a) \right],
$$

which projects the target distribution onto the predicted one in the
geometry of the simplex rather than of the real line. Its gradient with
respect to the logits is $p_{\boldsymbol\theta} - q$, bounded whatever the
size of the target. This is where its robustness comes from. Each sample
contributes a bounded gradient per bin, where squared error contributes a
gradient proportional to the error; the finite grid clips extreme targets
to its end bins, so that an outlier cannot set the scale of the regression;
and the two-hot encoding spreads each target over two bins, a label
smoothing that averages noisy targets at the same $(s,a)$. The encoding is
an averager in the sense of {prf:ref}`gordon-averager`, a convex combination
that preserves order and bounds, paired with a network that is not, so that
classification-based Q-learning is a hybrid of a monotone target structure
and a flexible fit. {cite:t}`farebrother2024stop` found that DQN and CQL
with squared error degrade when the Q-network is scaled to a large ResNet,
while the classification loss (HL-Gauss, which smooths the target with a
Gaussian instead of the two-hot encoding) keeps its performance.

## Summary

Fitted Q-iteration on data is fitted value iteration with the model replaced
by the empirical distribution of a buffer: targets from recorded
transitions, one successor each, and a fit that averages its error over
them. The two experiments located the errors of a fitted policy. In the
scheduling log with full coverage, the policy erred only where two actions
were nearly tied, an approximation error that more data would not remove;
in the log that followed one heuristic, it erred where the heuristic had
never tried the exact policy's action, a coverage error that no regressor
can repair. On ESA's collision warnings, fitted Q-iteration learned from
the log alone a rule within about 7% of the one an operator would write by
hand, once its targets were scaled so that one rare event could not dominate
the fit.

Writing out the gradient steps of a neural fit and merging them with the
outer loop produces the online and target networks of DQN: the target
network is the outer iteration's parameters kept in memory, and the online
network is the inner iterate, which also collects the data once the buffer
is allowed to evolve. Double DQN changes only the target function, to
decouple selection from evaluation, and Q-learning is the limit of one
transition per update, a stochastic approximation whose convergence the ODE
method establishes for tables. All of them occupy one design space, with a
buffer, a target function and a fit as its coordinates, and the loss of the
fit is a noise model for the targets, which Gumbel regression and
classification losses match better than squared error.

Fitted Q-iteration can now learn action values from fixed or evolving
buffers, but every target still requires $\max_a q(s,a)$. What happens when
that action search is itself a nonlinear program? [Amortized action
optimization](amortized-action-optimization.md) learns an actor to replace
the repeated search while retaining the fitted-Q critic.

## Self-checks

:::{exercise} Two loops
:label: ex-fqi-check-1

What happens in the outer loop and the inner loop of fitted Q-iteration?
:::

:::{solution} ex-fqi-check-1
:class: dropdown

The outer loop freezes the current Q-function and constructs Bellman targets. The inner loop fits a new function approximator to those targets.
:::

:::{exercise} Offline support
:label: ex-fqi-check-2

Why is a highly accurate regression fit still dangerous for actions poorly represented in an offline dataset?
:::

:::{solution} ex-fqi-check-2
:class: dropdown

The fit is unconstrained off the data support. Maximization can select extrapolated action values that are numerically large but unsupported by transitions in the dataset.
:::

:::{exercise} Frozen targets
:label: ex-fqi-check-3

What instability does a target network reduce?
:::

:::{solution} ex-fqi-check-3
:class: dropdown

It slows the movement of regression targets. Without it, each parameter update simultaneously changes the predictions and the targets being fitted, creating a rapidly moving objective.
:::

:::{exercise} Matched coverage comparison
:label: ex-fqi-check-4

The two inference-scheduling buffers contain the same number of transitions.
Why can the buffer with fewer distinct state-action pairs still attain lower
training error while producing a worse policy?
:::

:::{solution} ex-fqi-check-4
:class: dropdown

Repeated transitions make the empirical regression problem easier on the
visited region, but they supply no Bellman targets for omitted actions and
states. Greedy action selection can therefore depend on unsupported
extrapolation even when the in-buffer fit is accurate.
:::

:::{exercise} Scaling the targets
:label: ex-fqi-check-5

In the collision-avoidance example, why did one event with a 2% collision probability change the maneuvers of the fitted policy at hundreds of other states, and what did fitting a ratio instead of the raw cost change?
:::

:::{solution} ex-fqi-check-5
:class: dropdown

Extra trees predict the average of the targets in a leaf, so a single cost three times larger than all the others combined raises the prediction of every state that shares a leaf with that event's epochs. Fitting the cost relative to the current message's own estimate turns the target into a correction factor of ordinary size, so the event no longer dominates the leaves it falls in.
:::
