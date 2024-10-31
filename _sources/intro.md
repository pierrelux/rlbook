# Introduction
In contemporary RL research, there is often a focus on "solving" benchmark problems, sometimes at the expense of properly formulating them in the first place. This has led to a gap between the potential of RL as a powerful tool for addressing real-world challenges and its current role as an idealized theoretical model of human intelligence. The difficulty in formulating RL or control problems has been recognized in various fields. For example, {cite:t}`Iskhakov2020` describe the "daunting problem" hindering the broader adoption of decision-making algorithms in econometrics as "the difficulty of learning about the objective function and environment facing real-world decision-makers." This course suggests that these challenges can be addressed by taking a more flexible approach -- building a diverse set of tools to better formulate and solve these problems. Just as understanding others' objectives requires an open mind, tackling real-world RL problems calls for adaptable methods.

The crux of the problem here is not about the theoretical expressiveness of one framework over another, but rather a deeply human challenge: communication. It’s about knowing how to translate complex problems into mathematical formulations and then into executable, societally beneficial code. 

Outside academia, many challenges arise beyond just choosing an RL algorithm. These include defining time units (discrete, continuous, start points, durations), identifying states, determining actions, and understanding how these evolve over time. Setting the right objectives for a system is especially difficult, as real-world systems that interact with humans must account for preferences, values, and judgments that even domain experts find hard to articulate.  

While most presentations of RL emphasize the trial-and-error approach popularized by Sutton and Barto, this course aims to provide a broader perspective on the diverse tools and approaches that can help tackle RL problems effectively in practice. This perspective enables us to incorporate a variety of techniques, including optimal control, system identification methods, and ideas from simulation, alongside the usual tools of temporal difference learning.

## The Reinforcement Learning Problem

Reinforcement learning (RL) addresses the problem of making good decisions based on experience (data), inspired by animal psychology and early AI research in the 1980s. The broader topic of sequential decision-making gained prominence during and after World War II, driven by advancements in ballistics, aerospace, and resource planning. These developments laid the groundwork for several research disciplines, such as control theory and operations research (OR). Although not explicitly referred to as "learning," both fields have incorporated learning concepts through adaptation (influenced by Wiener's cybernetics) and system identification.

But what is learning, after all? While it’s challenging to define learning perfectly, it fundamentally involves data—measurements of some phenomenon of interest. However, data alone is insufficient; we must use it to improve the quality of our decisions, or in other words, to adapt. Supervised learning, the main form of machine learning in our everyday lives, involves data and the process of adapting the model's parameters. The question then arises: How can we frame the problem of learning to make good decisions from data as a supervised learning problem?

### From Imitation to Optimization

One method is to collect demonstrations from experts showing the appropriate actions in various situations. Using supervised learning, we would hope that our model generalizes to unseen situations. This approach, known as imitation learning (IL), faces two main challenges: the availability of high-quality data and sufficient data coverage. Often, we want to develop an RL system because we either don't know how to solve the problem well enough to provide demonstrations or aim to outperform the available demonstrations (superhuman AI).

When expert demonstrations are unavailable or inadequate, we take a different approach. Instead of requiring input-output pairs for imitation, we ask the system designer to provide a description of what they want to achieve through a mathematical function, usually a sum of costs or rewards, and a description of the "laws of the world" by which this objective can be achieved (the dynamics). The underlying assumption is that deriving good decisions from this description is easier than directly using IL. In this sense, choosing between demonstrations or reward/dynamics pairs is like specifying a "basis" for generating new and better ways of acting.

### Challenges in Problem Specification

Specifying objectives and dynamics can be complex. Consider dynamically adjusting temperature setpoints in a building to minimize energy usage while maintaining thermal comfort and adhering to equipment limitations. Thermal comfort preferences vary greatly among individuals and are influenced by factors like humidity, which sensors might not always capture accurately. Moreover, ensuring responsible deployment requires safeguards, as pushing close to equipment limits could disrupt service, damage trust, and impede user adoption. In practice, knowledge about controlling building HVAC systems is often held tacitly by experienced building managers, passed down through informal interactions and undocumented practices, posing a risk of knowledge loss with staff turnover. This challenge of understanding human decision-making preferences is also prevalent in other industries, such as water treatment plants, where operators must manage water outflow, turbidity, and pH levels under varying demands. These tasks require intricate, often tacit knowledge that encompasses forecasting, risk perception, and overall intuition about the system's state.

The difficulty of specifying what people want has given rise to the field of preference elicitation, which aims to infer optimal actions based on human preferences. In preference modeling, we seek to derive preferences from data, usually obtained through preference elicitation. This is also a learning problem closely related to supervised learning, but instead of directly representing the desired actions, we infer a preference ordering from the data. Preference data is typically collected in a pairwise manner, and under the Von Neumann theorem, these preferences can be preserved by their equivalent representation as a scalar utility function. This inferred utility function is the foundation of Reinforcement Learning from Human Feedback (RLHF), used in state-of-the-art Large Language Models (LLMs). Compared to IL, RLHF simplifies data collection as preferences are easier to obtain at scale than expert demonstrations. Additionally, RLHF allows for data reuse, as preference queries can be collected offline and relabeled later; correcting demonstrations post hoc is more challenging.

Ultimately, whether we choose imitation learning, preference-based modeling, or direct modeling of objectives and dynamics depends on how easily we can express our intentions to a machine. This challenge of conveying human intents to computers can be seen as a human-computer interaction (HCI) problem. Throughout this course, the aim is to equip you with a range of tools to help you find the right approach for your problem by selecting or combining these techniques effectively.

