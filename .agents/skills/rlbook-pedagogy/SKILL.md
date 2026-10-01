---
name: rlbook-pedagogy
description: Design, refactor, or review chapters, worked examples, exercises, and learning sequences for this RL and control textbook. Use when adding teaching material, improving conceptual progression, introducing a derivation, or aligning learning goals with practice. Routine code maintenance without a teaching-content change does not need this skill.
---

# RL textbook pedagogy

Use the repository's [PEDAGOGICAL_GUIDELINES.md](../../../PEDAGOGICAL_GUIDELINES.md)
as the source of pedagogical guidance. Read it before designing or substantially
restructuring teaching material. When drafting prose, also read
[WRITING_GUIDELINES.md](../../../WRITING_GUIDELINES.md). Resolve these paths
relative to this skill directory, not the current working directory; retain
the root files as the maintained sources rather than copying them into skills.

Teach graduate students in CS or applied mathematics who know linear algebra,
probability, multivariate calculus, and some machine learning, but may have
little classical control background. Preserve the user's chosen topic,
chapter placement, examples, and scope.

## Shape the learning sequence

- Identify what the reader can already do and what the new material should
  enable. Give chapters a short motivation, 3–7 actionable learning goals,
  and prerequisite links. Apply this structure at chapter scale, not to every
  small edit or subsection.
- Introduce unfamiliar abstractions through a concrete problem. Develop the
  definitions and equations from that problem, then return to it to check
  their meaning. Keep analytic or numerical checks near the calculations
  they verify.
- Explain intermediate derivation steps and why they are needed. Introduce
  notation gradually, state assumptions where they enter, and distinguish
  exact results from computational approximations.
- Connect prose, equations, pseudocode, code, and figures using consistent
  notation. Use recurring case studies when they serve the explanation;
  state enough of the model that specialist domain knowledge is unnecessary.
- Let later examples require more independent reasoning. Use occasional
  prediction or self-explanation prompts and exercises covering calculation,
  conceptual reasoning, computation, and extensions. Match exercises to the
  stated goals and revisit earlier ideas where useful.
- State what each experiment tests, with interpretable comparisons,
  parameters, randomization, and outcomes. A motivating question is optional;
  do not make question-led openings a template. Keep code focused on the concept
  being taught. Conclude with what the result supports and how it connects
  to the next topic.

For revisions, diagnose gaps in the conceptual progression before rearranging
paragraphs. Check that a reader can follow the assumptions, reproduce a worked
calculation, and attempt the exercises using material already introduced.
When changing computational material or MyST structure, verify the affected
examples and rendered output with the repository's relevant checks. Scale
verification to the actual change.
