# Lecture 1 instructor route

This page is an instructor note, not part of the student-facing table of
contents. The examples now live beside the concepts that explain them, so this
route links across the textbook rather than reproducing a separate lecture
page.

| time | book location | classroom move | question left visible |
|---:|---|---|---|
| 0–8 min | [system boundaries and action channels](../dynamics.md#system-boundaries-and-action-channels) | ask for predictions before naming the inputs | What can each controller physically change? |
| 8–22 min | [SwingRL internal actuation](../dynamics.md#internal-actuation-in-swingrl) | let the class locate the driven and parametric terms | Where does the rider enter the mechanics? |
| 22–34 min | [rod and chain evidence](../dynamics.md#internal-actuation-in-swingrl) | reveal the negative suspension force last | Did the controller succeed, or did the model fail? |
| 34–49 min | [overhead-crane collocation](../collocation.md#overhead-crane-point-to-point-motion) | compute the shaping delay, then compare it with the constrained trajectory | Which structure was useful without being globally exact? |
| 49–64 min | [wave-energy economic MPC](../mpc.md#wave-energy-capture) | first ask students to maximize instantaneous power | Why can a trajectory objective disagree with the greedy choice? |
| 64–75 min | [recorded SwingRL PPO experiment](../pg.md#experiment-ppo-on-the-swingrl-plant) | stop the replay at intermediate checkpoints and ask for a prediction | Which failure belongs to optimization, and which belongs to the model? |
| 75–80 min | [assessment and exit ticket](../syllabus.md) | collect a five-line formulation on paper | State, action, disturbance, objective, constraint? |

The final paper response is best treated as a diagnostic exit ticket in the
first meeting. It reveals whether students can formulate a decision problem
without turning the exercise into a test of syntax or generated prose. A later
in-class assessment can reuse the same five-line format with an unfamiliar
system and require students to defend one modeling choice.
