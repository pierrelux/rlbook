# Why This Book?

Reinforcement learning offers a powerful framework for decision-making: systems that learn to act through interaction with their environment. From AlphaGo defeating world champions to Uber's ride-matching system optimizing across hundreds of cities {cite:p}`Uber2025`, from chatbots engaging millions to ANYbotics' quadruped robots performing complex industrial inspections with policies trained entirely in simulation {cite:p}`ANYbotics2023`, RL has demonstrated remarkable capabilities.

Yet compared to supervised learning, which has become routine in industry, reinforcement learning has not achieved the same widespread adoption. Supervised learning benefits from standardized tools and well-defined interfaces: inputs, outputs, and objective metrics. Reinforcement learning, by contrast, requires explicit problem formulation: defining objectives, constraints, and how decisions unfold over time. This additional structure is also what makes RL applicable to a broader class of problems.

As {cite:t}`Iskhakov2020` notes, a primary challenge is *"the difficulty of learning about the objective function and environment facing real-world decision-makers."* We cannot sidestep defining the problem, the objective, and the constraints.

Working in industry and consulting taught me what my PhD did not: real problems rarely fit neatly into predefined frameworks. Sensors produce noisy data; constraints are non-negotiable; objectives may shift or conflict. Most effort goes into formulating the decision problem, long before selecting an algorithm.

The chapters that follow address this challenge explicitly. They offer strategies to bridge the gap from theoretical RL formulations to practically useful systems. By carefully structuring decision problems, we can help reinforcement learning achieve broader impact.

## The Decision Problem

The term *reinforcement learning* gets used in many ways. In the formal sense defined by {cite:t}`SuttonBarto2018`, RL is a problem: learning to act through interaction with an environment. But in common usage, it can mean a family of algorithms, a research community, or a long-term scientific agenda.

This book takes a practical view: reinforcement learning as a vocabulary for reasoning about decisions under uncertainty. When optimization, feedback, and data intersect, we are in the territory of reinforcement learning, whether we use temporal-difference learning, model-based planning, or simple policy rules. What unifies these approaches is not a specific algorithm, but a shared structure: decision-making through experience.

In mainstream RL research, the problem is often treated as given. Sutton famously advises: *"Approximate the solution, not the problem."* The agent should be shaped by experience, not by handcrafted structure. For Sutton, *the problem* is the world itself: complex, unknown, and handed to us as-is. We do not design it; we confront it.

This book takes a different stance. We consider a variety of decision problems, each with its own structure: finite or infinite horizon, discrete or continuous time, deterministic or stochastic dynamics. The problem is not handed to us; it must be defined. What are the goals? What decisions are available? What feedback is observable, and under what constraints?

This is what Operations Research has long emphasized. Through consulting, I saw it firsthand: production systems running overnight, MIP solvers optimizing decisions with XGBoost predictors as inputs, all live and working. In practice, these systems are doing reinforcement learning, just without calling it that. They define objectives, encode constraints, and optimize over time. The vocabulary differs, but the structure is the same.

## What This Book Offers

Reinforcement learning did not develop in isolation. Its foundations draw from control theory, dynamic programming, operations research, and economics. Many of the same ideas appear under different names in different communities, often with complementary perspectives.

This book aims to give you a broader view of that landscape. Where do RL algorithms come from? What mathematical structures underlie them? How do they connect to classical methods in optimization and control? Understanding these connections helps you see when a method applies, when it does not, and what alternatives exist.

The goal is not to survey every technique superficially. It is to go deep on the mathematical foundations that are shared across methods: dynamic programming, function approximation, optimization, and the interplay between them. These structures recur throughout sequential decision-making, whether in reinforcement learning, control theory, or operations research. Master them once, and you can recognize them in different guises.
