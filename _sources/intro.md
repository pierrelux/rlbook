# Why This Book?

Reinforcement learning often captures headlines with dramatic feats: AlphaGo defeating world champions, agents mastering Atari, chatbots engaging millions in conversation. Yet despite these accomplishments, RL’s real-world impact on decision-making remains surprisingly limited.

As {cite:t}`Iskhakov2020` observed in econometrics, the “daunting problem” hindering the broader adoption of decision-making algorithms is:

> *“The difficulty of learning about the objective function and environment facing real-world decision-makers.”*

In RL, this same difficulty often goes unaddressed. There is a tendency to focus on solving benchmark tasks, sometimes without fully confronting the question of whether the problem itself has been well posed. As a result, we risk treating RL less as a tool for tackling real-world challenges and more as an idealized model of intelligence in controlled settings.

That gap is where this book begins.

It's easy to build agents that succeed in artificial environments with well-defined rules and carefully shaped rewards. It's much harder to design systems that support better decisions in real-world contexts: messy, constrained, and often high-stakes. And the difficulty usually has little to do with the algorithm itself.

I didn’t fully understand this as a PhD student. Like many of us trained in machine learning, I focused on the agent side of the RL setup: tuning models, benchmarking algorithms, chasing leaderboard scores. I never touched the environment. I never defined the problem. That part was invisible, assumed, or treated as someone else’s responsibility.

Modeling was seen as something to abstract away. We were taught to value elegance through abstraction, and looking beyond the Gym API felt like breaking the rules—introducing complications where there should be cleanliness.

It took experience outside of academia to realize how much of the hard work happens before choosing an algorithm.

While working as a consultant, I finally experienced the other side of the interface. I was no longer tweaking a policy. I was shaping the problem itself. The sensors were noisy, constraints were real, and objectives were vague or contradictory. The system needed to make decisions quickly, with partial information and conflicting goals.

That’s when John Rust’s words truly resonated:

> *“The range of known real-world applications of Dynamic Programming seems disappointingly small, given the immense computer power and the decades of research that have produced a myriad of alternative solution methods for DP problems.”*

We do have powerful tools. The bottleneck lies earlier in the process.

Rust continues:

> *“I believe the biggest constraint on progress is not limited computer power, but instead the **difficulty of learning the underlying structure** of the decision problem.”*

Before we can solve a problem, we have to understand what the problem is. That means identifying objectives, actions, constraints, and the flow of information. What’s observable? What’s controllable? What matters? These aren’t secondary concerns. They are the problem.

As Rust puts it:

> *“Calculating an optimal solution to the wrong objective, or misspecifying the constraints and opportunities the actor actually confronts, may not result in helpful advice to the actor. **It is like providing the right answer to the wrong question.**”*

This is the heart of the issue. In academic RL, we often leap to reward functions and standard frameworks without spending enough time on the underlying formulation. We end up with elegant algorithms applied to poorly posed problems: solutions in search of a useful application.

The challenge isn’t about expressiveness. It’s about translating vague intentions into precise formulations. What constitutes a state? What actions are feasible, and when? What time horizon is meaningful? How do we encode preferences when even the stakeholders disagree?

That act of translation—from informal goals to formal decision problems—is where the real work lies. And it's the focus of this book.

## What Problem Are We Solving?

The term *reinforcement learning* gets used in many different ways. In the formal sense defined by {cite:t}`SuttonBarto2018`, RL is a problem: learning to act through interaction with an environment. But in common usage, RL can mean a family of algorithms, a research community, or even a long-term scientific agenda. For some, it’s part of an effort to “solve intelligence.” For others, it’s a toolbox for solving control problems with data.

This book takes the latter view.

We are not in the business of solving intelligence. Our concern is more immediate: helping systems make good decisions using the data they already produce. That means we treat RL not as an end in itself, but as a vocabulary for reasoning about decisions under uncertainty. When optimization, feedback, and data intersect, we’re in the territory of RL—whether we use temporal-difference learning, model-based planning, or simple policy rules. What unifies these approaches isn’t a specific algorithm, but a shared structure: decision-making through experience.

Being clear about the problem matters. If the goal is to understand cognition, then abstraction and simulation are appropriate tools. But if the goal is to improve real-world systems—whether in energy, logistics, health, or agriculture—then the hard part isn’t choosing an algorithm. It’s defining the task. What are we optimizing? What constraints apply? What information is available, and when? These are modeling questions, and they sit upstream of any learning method.

This book begins with the problem, not the solution. We take reinforcement learning seriously not as a brand of algorithms, but as a way of thinking about how decisions get better with data.

In doing so, we take inspiration from Sutton’s philosophy, but also shift its emphasis. Sutton famously advises: *“Approximate the solution, not the problem.”* In his view, the RL agent should be shaped by experience, not by handcrafted structure or strong priors. That framing has led to a great deal of insight and elegant generality. But in real-world applications, the situation is often reversed: the problem itself is underspecified, and the main work lies in giving it shape. Before we can even begin to approximate a solution, we must first decide what the problem actually is.

That’s where this book begins—not with the agent, but with the world it must reason about. Not with the function class, but with the feedback structure. In short: not just how to learn, but what it means to have something worth learning.

## What Does It Mean to Model a Decision Problem?

Modeling isn’t about feeding data into a black box. It’s about deciding what matters. It involves structuring a problem: defining objectives, specifying constraints, clarifying what’s observable, and determining how decisions unfold over time.

Take an HVAC system. The goal might be “maximize comfort while minimizing energy.” But what does comfort mean? A fixed temperature? Acceptable humidity? Rate of change? Occupant preferences? Is comfort linearly traded against energy, or are there thresholds? And how do you ensure safety and respect equipment limitations?

In irrigation, similar questions arise. Should irrigation be based on soil dryness, weather forecasts, plant health, or electricity prices? Should we water now or wait? How often should we revisit this choice? The answers depend on sensor availability, environmental dynamics, and risk tolerance.

Even time plays a central role. Are we planning for the next few minutes, or the next growing season? Should we model time in discrete steps or continuously? These aren’t afterthoughts. They shape what’s learnable, controllable, and feasible.

Real-world systems come with hard constraints: physical limits, budgets, safety regulations, human expectations. Ignoring them may simplify the math, but makes any solution irrelevant in practice. Good modeling incorporates these constraints from the beginning.

This kind of modeling is what Operations Research (OR) has long emphasized. By the 1970s, the foundational theory of dynamic programming was already in place. Today’s OR community often focuses on solving concrete decision problems, drawing on tools like mixed-integer linear programming when appropriate.

Through consulting, I came to appreciate OR’s pragmatism. My colleagues built decision-support systems connected to real-time data. In practice, they were doing reinforcement learning, just without Temporal Difference methods. They couldn’t rely on massive data streams. Instead, they had to wrestle directly with business logic, real-world variability, and engineering constraints.

That doesn’t mean OR has all the answers, or that RL has been misguided. On the contrary, general-purpose solutions—even on toy problems—have advanced theory in meaningful ways. Simplified settings enable validation without full domain understanding: Did the pole balance? Did the agent reach the goal? Did it beat the Atari score? These abstractions follow good software engineering principles: separation of concerns, clear interfaces, and rapid iteration.

But abstraction is only one part of the equation. It’s useful until it isn’t. A strong framework offers clarity at first, but eventually gets in the way—layering on complexity, edge cases, and configuration knobs. This mirrors the lifecycle of many software tools, where the initial elegance gives way to accumulated mess. I’ve come to treat modeling the same way: start small, surface the hard constraints early, and only add structure when it’s required by the data, physics, or policy context.

We shouldn’t try to cram every control task into the discounted MDP format just because it's the default interface. Instead, we should keep a lean toolbox and reach for what the problem demands. This mindset—start simple, avoid premature generality, don’t confuse abstraction with robustness—is well known to engineers. If that means choosing a basic model-predictive controller over a trendy RL library, or the reverse, so be it.

Over time, this habit of zooming in when needed and zooming out when possible reshaped how I approached research. Once I started questioning the abstractions, I began to see what they were hiding.

Peeling back the layers revealed dynamics not captured by benchmarks, constraints ignored by rewards, and theory that only emerged in contact with the real world. This convinced me that some of the most meaningful discoveries still lie beneath the surface.

Reinforcement learning is a framework for learning to make decisions through experience. But experience is only useful if we’ve posed the right problem. Modeling determines not just *how* learning proceeds, but *what* we can learn at all.

At first, this might sound easy—just specify a reward and let the agent learn. That’s the promise behind the “reward is enough” hypothesis. But in the real world, rewards aren’t handed to us. They must be constructed, inferred, or negotiated. And even when we manage that, rewards only express part of what matters. They don’t tell us what information is available, what tradeoffs are acceptable, or how to handle ambiguity, delay, or disagreement.

In short, posing the right problem is itself a hard problem—and one for which we have few systematic tools.

Often, the only place we can turn is to people. Domain experts act, react, and judge, even when they can’t explain their reasoning explicitly. Their preferences show up in behavior, in corrections, in choices they make under pressure. If we can't write down what we want, perhaps we can learn it indirectly—from them.

## Learning From Humans

When we can’t write down the right behavior, we often try to learn it from examples. This is the idea behind imitation learning: watch the expert, then generalize. But in practice, good demonstrations are hard to collect and rarely cover the full range of relevant situations—especially the rare or risky ones.

That’s why many real-world applications require more than direct imitation. If we can’t show what to do in every case, we must express what we want. This is where reward design, cost functions, and preference modeling come in. These tools attempt to capture the underlying objective by observing not just what people do, but what they seem to value.

Preference elicitation offers one route. Rather than specifying the optimal solution directly, we infer it from comparisons, rankings, or feedback. Under mild assumptions, the von Neumann–Morgenstern theorem tells us that such preferences correspond to a utility function. This principle forms the basis of *Reinforcement Learning from Human Feedback* (RLHF), now central to training large models.

But here, too, we face limits. Once we’ve inferred preferences or objectives from humans, what comes next? In many systems, the default answer is to treat this as a standard supervised learning problem: fit a black-box model to human-labeled data, then optimize the resulting predictions.

This approach can go surprisingly far. Recent work, such as Decision Transformers, has shown that supervised learning can recover policies that perform competitively—sometimes even state-of-the-art. But these successes are often built on vast datasets, carefully curated environments, and tight control over evaluation. In the real world, we rarely have that luxury.

Supervised learning assumes that if we show enough examples, the system will generalize appropriately. But generalization is fragile when data is limited, feedback is partial, or the stakes are high. Without the right structure in place, we risk building policies that extrapolate poorly, violate constraints, or break in unexpected ways.

This is where modeling matters again.

By modeling the decision process—the constraints, objectives, time structure, and information flows—we introduce the *right inductive biases* into the learning system. These biases aren’t arbitrary. They reflect how the world works and what the agent can and cannot do. They make learning tractable even when data is scarce and help ensure that the resulting decisions behave reasonably under uncertainty.

So while supervised learning plays a role, it’s not the full story. The point is not just to imitate or infer, but to **embed** what we’ve learned into a framework where decision-making remains accountable, robust, and grounded in structure.

That’s the aim of this book: to take what we can learn from humans and from data, and combine it with modeling discipline to build systems that don’t just act—but act for the right reasons.

## The Path Forward

Rust’s critique remains timely. After decades of algorithmic progress, we still struggle to help people make better decisions in the settings that matter most. RL has pushed the boundaries of simulation. But in practice, its reach remains limited—not because the tools are broken, but because we’ve struggled to formulate problems in ways that connect to the real world.

It’s not that decision problems can’t be solved. It’s that we often fail to pose them in solvable form.

This isn’t a limitation of algorithms. It’s a limitation in how we frame our goals.

But that may be changing.

Rust once wrote:

> *“Humans are learning to replicate the type of subconscious model building that goes on inside their brains and bring it to the conscious, formal level—but they are doing this modeling themselves, since it is not clear **how to teach computers how to model**.”*

That might have been true then. But today, we’re beginning to see what it might look like to *teach* machines to model—or at least assist us in doing so.

One school of thought, inspired by Sutton’s long-term vision, imagines that we won’t need to model at all. Instead, we build general-purpose agents trained on vast, unstructured experience. We don’t hand them objectives or constraints. We don’t define environments. We let them learn everything from scratch. The hope is that once such an agent is sufficiently broad and capable, it can generalize everywhere—even to tasks we haven’t yet imagined.

That is a bold and fascinating bet. But it is still a bet. And if your goal is to solve meaningful problems now—in healthcare, infrastructure, climate, logistics—then the challenge of formulation doesn’t disappear. Even the most flexible agent can’t act reliably in a domain where the goals, tradeoffs, and structure remain unclear.

What’s more, our current models of generality—especially large language models—are not agents in the classical sense. They produce text, but they don’t act in the world. They reason in language, but they don’t optimize over time. Despite growing trends to describe these systems as “agents,” they are better understood as powerful modeling tools: systems trained on enormous corpora of human knowledge, capable of reflecting, translating, and helping us express complex ideas.

And that might be their greatest strength.

We may not hand full control to a language model anytime soon, but we *will* use these systems to help us model. They will assist in articulating objectives, surfacing hidden assumptions, identifying constraints, and mapping informal goals into structured forms. They won’t replace modeling—they’ll augment it.

And what takes action—what makes real decisions in the world—will still rely on explicit optimization, grounded in formal structure, and designed to behave predictably. The language model may help us design that system, but it won’t be the one executing it.

That’s where I believe the near future lies: general-purpose models as modeling assistants, paired with optimization and control systems that retain structure, constraints, and accountability.

This book is about building that bridge: from goals to models, from data to decisions, from abstraction to action.

This is the modeling mindset. And it’s what turns reinforcement learning into a practical tool for solving real problems.