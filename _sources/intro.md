# Why This Book?

Reinforcement learning often captures headlines with impressive breakthroughs: AlphaGo defeating world champions, agents mastering Atari, chatbots engaging millions in conversation. Yet compared to other branches of machine learning—especially supervised learning, which has become routine in industry and research—RL has not yet achieved the same widespread adoption in everyday decision-making and operations.

For perspective, consider supervised learning. Standardized tools like scikit-learn, TensorFlow, or PyTorch have made it straightforward for data scientists to integrate classification and regression models into production workflows. In medicine, hundreds of FDA-approved devices now incorporate supervised machine learning. Randomized controlled trials regularly evaluate diagnostic and predictive models built with supervised learning techniques.

By contrast, reinforcement learning's adoption is still in earlier stages. Among more than a thousand FDA-cleared AI-enabled medical devices as of 2025, none explicitly reference reinforcement learning in their public summaries {cite:p}`FDA2025`. A recent *Lancet Digital Health* review of 86 clinical trials involving AI identified only two studies that tested RL-based decision rules {cite:p}`Kleijnen2024`. Most AI solutions in healthcare remain supervised learning models trained on labeled datasets.

Outside healthcare, promising results exist, but scale remains modest. In building automation, a 2025 survey identified many field demonstrations of reinforcement learning and Model Predictive Control (MPC) for HVAC control. Yet, fewer than a third met basic methodological criteria. Among reliable studies, average cost reductions were around 13–16% {cite:p}`Chen2025`. This is encouraging, yet still short of broad, industry-wide adoption.

Even in high-profile reinforcement learning deployments, adoption at scale is sometimes uncertain or difficult to confirm. Google DeepMind famously reported significant cooling-energy reductions in data centers using RL back in 2018 {cite:p}`Evans2018`. However, more recent public confirmation of widespread autonomous use has been lacking. Meta, in a 2024 engineering blog, provided clear evidence that an RL-based airflow controller was achieving meaningful reductions in energy and water usage, and stated that broader deployment was underway, though specifics on scale were not disclosed {cite:p}`Meta2024`.

Nevertheless, notable successes continue to emerge. Uber has successfully integrated reinforcement learning into its ride-matching system, with measurable improvements across hundreds of cities {cite:p}`Uber2025`. ANYbotics has commercially deployed quadruped robots whose locomotion policies, trained entirely via RL in simulation, now reliably perform complex industrial inspections {cite:p}`ANYbotics2023`.

These examples illustrate genuine progress. Yet it is clear reinforcement learning has not yet reached the "plug-and-play" status enjoyed by supervised learning. This is largely due to fundamental differences in problem structure. Supervised learning tasks typically involve clearly defined inputs, outputs, and objective metrics. Reinforcement learning, by contrast, requires explicit problem formulation, exploration, interactive data collection, and a nuanced understanding of the environment's structure.

As {cite:t}`Iskhakov2020` notes in econometrics, a primary challenge facing adoption of any sequential decision-making tool is:

> *“The difficulty of learning about the objective function and environment facing real-world decision-makers.”*

In reinforcement learning, this difficulty is integral. We cannot sidestep defining the problem, the objective, and the constraints; these are not incidental but central. Supervised learning allows practitioners to abstract away most of these concerns into standardized data formats and evaluation metrics. Reinforcement learning practitioners do not have this luxury.

That is where this book begins.

I did not fully appreciate this difference as a PhD student. Like many others trained in machine learning, I focused on tuning algorithms, chasing benchmarks, and climbing leaderboards. Problem definition was abstracted away, often assumed or left as someone else's responsibility.

Working in industry and consulting changed that. Real problems rarely fit neatly into a predefined framework. Sensors produce noisy data; constraints are non-negotiable; objectives may shift or conflict. I discovered firsthand that most effort goes into formulating the decision problem clearly, long before selecting an algorithm.

John Rust captures this precisely when reflecting on dynamic programming in practice:

> *“The range of known real-world applications of Dynamic Programming seems disappointingly small, given the immense computer power and decades of research that have produced a myriad of alternative solution methods for DP problems. I believe the biggest constraint on progress is not limited computer power, but instead the difficulty of learning the underlying structure of the decision problem.”* {cite:t}`Rust2008`

In other words, solving a real-world decision problem starts with formulating it correctly. What are we optimizing? What can we observe and control? How does information flow? These questions are not secondary details; they define the problem itself.

The chapters that follow address these challenges explicitly. They offer strategies to bridge the gap from theoretical reinforcement learning formulations to practically useful systems. By carefully structuring decision problems, we can help reinforcement learning achieve broader impact, just as supervised learning already has.

## What Problem Are We Solving?

The term *reinforcement learning* gets used in many different ways. In the formal sense defined by {cite:t}`SuttonBarto2018`, RL is a problem: learning to act through interaction with an environment. But in common usage, reinforcement learning can mean a family of algorithms, a research community, or even a long-term scientific agenda. For some, it is part of an effort to “solve intelligence.” For others, it is a toolbox for solving control problems with data.

This book takes the latter view.

We are not in the business of solving intelligence. Our concern is more immediate: helping systems make good decisions using the data they already produce. That means we treat reinforcement learning not as an end in itself, but as a vocabulary for reasoning about decisions under uncertainty. When optimization, feedback, and data intersect, we are in the territory of reinforcement learning, whether we use temporal-difference learning, model-based planning, or simple policy rules. What unifies these approaches is not a specific algorithm, but a shared structure: decision-making through experience.

Being clear about the problem matters. If the goal is to understand cognition, then abstraction and simulation are appropriate tools. But if the goal is to improve real-world systems, whether in energy, logistics, health, or agriculture, then the hard part is not choosing an algorithm. It is defining the task. What are we optimizing? What constraints apply? What information is available, and when? These are modeling questions, and they sit upstream of any learning method.

This book begins with the problem, not the solution. We use reinforcement learning in its broadest sense—as a perspective on how to improve decision-making from data, not just a collection of algorithms.

In doing so, we take inspiration from Sutton’s philosophy, while shifting its emphasis. Sutton famously advises: *“Approximate the solution, not the problem.”* In his view, the reinforcement learning agent should be shaped by experience, not by handcrafted structure or strong priors. That framing has led to powerful ideas and a remarkable degree of generality.

But it also reflects a particular philosophy. For Sutton, *“the problem”* is the world itself—complex, unknown, and handed to us as-is. We do not design it; we simply confront it. Reinforcement learning, in this view, is a path toward understanding intelligence through interaction, not engineering through modeling.

This book takes a more pragmatic stance. In practice, the problem is rarely just “given.” It must be defined: what are the goals, what decisions are available, what feedback is observable, and under what constraints? Before we can solve a problem, we have to formulate it—and that formulation shapes everything that follows.

## What Does It Mean to Model a Decision Problem?

Modeling is not about feeding data into a black box. It is about deciding what matters. It involves structuring a problem: defining objectives, specifying constraints, clarifying what is observable, and determining how decisions unfold over time.

Take an HVAC system. The goal might be “maximize comfort while minimizing energy.” But what does comfort mean? A fixed temperature? Acceptable humidity? Rate of change? Occupant preferences? Is comfort linearly traded against energy, or are there thresholds? And how do you ensure safety and respect equipment limitations?

In irrigation, similar questions arise. Should irrigation be based on soil dryness, weather forecasts, plant health, or electricity prices? Should we water now or wait? How often should we revisit this choice? The answers depend on sensor availability, environmental dynamics, and risk tolerance.

Even time plays a central role. Are we planning for the next few minutes, or the next growing season? Should we model time in discrete steps or continuously? These are not afterthoughts. They shape what is learnable, controllable, and feasible.

Real-world systems come with hard constraints: physical limits, budgets, safety regulations, human expectations. Ignoring them may simplify the math, but it makes any solution irrelevant in practice. Good modeling incorporates these constraints from the beginning.

This kind of modeling is what Operations Research (OR) has long emphasized. By the 1970s, the foundational theory of dynamic programming was already in place. Today’s OR community often focuses on solving concrete decision problems, drawing on tools like mixed-integer linear programming when appropriate.

Through consulting, I came to appreciate OR’s pragmatism. My colleagues built decision-support systems connected to real-time data. In practice, they were doing reinforcement learning, just without Temporal Difference methods. They could not rely on massive data streams. Instead, they had to wrestle directly with business logic, real-world variability, and engineering constraints.

That does not mean OR has all the answers, or that reinforcement learning has been misguided. On the contrary, general-purpose solutions, even on toy problems, have advanced theory in meaningful ways. Simplified settings enable validation without full domain understanding: Did the pole balance? Did the agent reach the goal? Did it beat the Atari score? These abstractions follow good software engineering principles: separation of concerns, clear interfaces, and rapid iteration.

But abstraction is only one part of the equation. It is useful until it is not. A strong framework offers clarity at first, but eventually gets in the way, layering on complexity, edge cases, and configuration knobs. This mirrors the lifecycle of many software tools, where the initial elegance gives way to accumulated mess. I have come to treat modeling the same way: start small, surface the hard constraints early, and only add structure when it is required by the data, physics, or policy context.

We should not try to cram every control task into the discounted Markov Decision Process (MDP) format just because it is the default interface. Instead, we should keep a lean toolbox and reach for what the problem demands. This mindset, start simple, avoid premature generality, do not confuse abstraction with robustness, is well known to engineers. If that means choosing a basic model-predictive controller over a trendy reinforcement learning library, or the reverse, so be it.

Over time, this habit of zooming in when needed and zooming out when possible reshaped how I approached research. Once I started questioning the abstractions, I began to see what they were hiding.

Peeling back the layers revealed dynamics not captured by benchmarks, constraints ignored by rewards, and theory that only emerged in contact with the real world. This convinced me that some of the most meaningful discoveries still lie beneath the surface.

Reinforcement learning is a framework for learning to make decisions through experience. But experience is only useful if we have posed the right problem. Modeling determines not just *how* learning proceeds, but *what* we can learn at all.

At first, this might sound easy, just specify a reward and let the agent learn. That is the promise behind the “reward is enough” hypothesis. But in the real world, rewards are not handed to us. They must be constructed, inferred, or negotiated. And even when we manage that, rewards only express part of what matters. They do not tell us what information is available, what tradeoffs are acceptable, or how to handle ambiguity, delay, or disagreement.

In short, posing the right problem is itself a hard problem, and one for which we have few systematic tools.

Often, the only place we can turn is to people. Domain experts act, react, and judge, even when they cannot explain their reasoning explicitly. Their preferences show up in behavior, in corrections, in choices they make under pressure. If we cannot write down what we want, perhaps we can learn it indirectly from them.

---

## Learning From Humans

When we cannot write down the right behavior, we often try to learn it from examples. This is the idea behind imitation learning: watch the expert, then generalize. But in practice, good demonstrations are hard to collect and rarely cover the full range of relevant situations, especially the rare or risky ones.

That is why many real-world applications require more than direct imitation. If we cannot show what to do in every case, we must express what we want. This is where reward design, cost functions, and preference modeling come in. These tools attempt to capture the underlying objective by observing not just what people do, but what they seem to value.

Preference elicitation offers one route. Rather than specifying the optimal solution directly, we infer it from comparisons, rankings, or feedback. Under mild assumptions, the von Neumann–Morgenstern theorem tells us that such preferences correspond to a utility function. This principle forms the basis of *Reinforcement Learning from Human Feedback* (RLHF), now central to training large models.

But here, too, we face limits. Once we have inferred preferences or objectives from humans, what comes next? In many systems, the default answer is to treat this as a standard supervised learning problem: fit a black-box model to human-labeled data, then optimize the resulting predictions.

This approach can go surprisingly far. Recent work, such as Decision Transformers, has shown that supervised learning can recover policies that perform competitively, sometimes even state-of-the-art. But these successes are often built on vast datasets, carefully curated environments, and tight control over evaluation. In the real world, we rarely have that luxury.

Supervised learning assumes that if we show enough examples, the system will generalize appropriately. But generalization is fragile when data is limited, feedback is partial, or the stakes are high. Without the right structure in place, we risk building policies that extrapolate poorly, violate constraints, or break in unexpected ways.

This is where modeling matters again.

By modeling the decision process, which includes the constraints, objectives, time structure, and information flows, we introduce the *right inductive biases* into the learning system. These biases are not arbitrary. They reflect how the world works and what the agent can and cannot do. They make learning tractable even when data is scarce and help ensure that the resulting decisions behave reasonably under uncertainty.

So while supervised learning plays a role, it is not the full story. The point is not just to imitate or infer, but to **embed** what we have learned into a framework where decision-making remains accountable, robust, and grounded in structure.

That is the aim of this book: to take what we can learn from humans and from data, and combine it with modeling discipline to build systems that do not just act, but act for the right reasons.

---

## The Path Forward

Rust’s critique remains timely. After decades of algorithmic progress, we still struggle to help people make better decisions in the settings that matter most. Reinforcement learning has pushed the boundaries of simulation. But in practice, its reach remains limited, not because the tools are broken, but because we have struggled to formulate problems in ways that connect to the real world.

It is not that decision problems cannot be solved. It is that we often fail to pose them in solvable form.

This is not a limitation of algorithms. It is a limitation in how we frame our goals.

But that may be changing.

Rust once wrote:

> *“Humans are learning to replicate the type of subconscious model building that goes on inside their brains and bring it to the conscious, formal level, but they are doing this modeling themselves, since it is not clear **how to teach computers how to model**.”*

That might have been true then. But today, we are beginning to see what it might look like to *teach* machines to model, or at least assist us in doing so.

One school of thought, inspired by Sutton’s long-term vision, imagines that we will not need to model at all. Instead, we build general-purpose agents trained on vast, unstructured experience. We do not hand them objectives or constraints. We do not define environments. We let them learn everything from scratch. The hope is that once such an agent is sufficiently broad and capable, it can generalize everywhere, even to tasks we have not yet imagined.

That is a bold and fascinating bet. But it is still a bet. And if your goal is to solve meaningful problems now, in healthcare, infrastructure, climate, or logistics, then the challenge of formulation does not disappear. Even the most flexible agent cannot act reliably in a domain where the goals, tradeoffs, and structure remain unclear.

What is more, our current models of generality, especially large language models, are not agents in the classical sense. They produce text, but they do not act in the world. They reason in language, but they do not optimize over time. Despite growing trends to describe these systems as “agents,” they are better understood as powerful modeling tools: systems trained on enormous corpora of human knowledge, capable of reflecting, translating, and helping us express complex ideas.

And that might be their greatest strength.

We may not hand full control to a language model anytime soon, but we *will* use these systems to help us model. They will assist in articulating objectives, surfacing hidden assumptions, identifying constraints, and mapping informal goals into structured forms. They will not replace modeling; they will augment it.

And what takes action, what makes real decisions in the world, will still rely on explicit optimization, grounded in formal structure, and designed to behave predictably. The language model may help us design that system, but it will not be the one executing it.

That is where I believe the near future lies: general-purpose models as modeling assistants, paired with optimization and control systems that retain structure, constraints, and accountability.

This book is about building that bridge: from goals to models, from data to decisions, from abstraction to action.

This is the modeling mindset. And it is what turns reinforcement learning into a practical tool for solving real problems.