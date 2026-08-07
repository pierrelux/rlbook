---
# The page outline is suppressed: headings live inside language tabs, so an
# outline would list both languages at once and half its links would point
# into a hidden panel.
site:
  hide_outline: true
---

# Syllabus · Plan de cours

**IFT 6162 — Apprentissage par renforcement, commande optimale
(Reinforcement Learning, Optimal Control)**

:::{note}
This page is bilingual. Use the tabs below to switch between English and French.
Cette page est bilingue. Utilisez les onglets ci-dessous pour basculer entre l'anglais et le français.
:::

`````{tab-set}

````{tab-item} English
:sync: en

## Course information

| | |
|---|---|
| **Course code** | IFT 6162 |
| **Title** | Reinforcement Learning, Optimal Control (official registrar title is in French: *Apprentissage par renforcement, commande optimale*) |
| **Credits** | *TBD* |
| **Term** | Fall 2026 |
| **Schedule** | Mondays 10:30–12:29 and Thursdays 13:30–15:29, Aug 31 – Oct 16 and Oct 26 – Dec 9, 2026 |
| **Room** | Campus Montréal — room *TBD* |
| **Instructor** | Pierre-Luc Bacon |
| **Email** | *TBD* |
| **Office hours** | *TBD* |
| **Teaching assistants** | *TBD* |
| **Language of instruction** | *TBD* (course materials are in English) |
| **Course website** | <https://pierrelucbacon.com/rlbook/> |

## Description

This course builds reinforcement learning up from its foundations in dynamics,
optimization, and control. Rather than starting from the tabular Markov decision
process and working outward, we start from the decision problem itself: how to
write down a model of a system, how to state an objective that unfolds over time,
and how to solve the resulting optimization problem numerically. Dynamic
programming, model predictive control, and modern deep RL algorithms then appear
as different answers to the same question, each carrying its own assumptions and
failure modes.

The organizing theme is that most of the work in applying RL happens *before* an
algorithm is chosen. Sensors produce noisy data, constraints are non-negotiable,
and objectives shift or conflict. A practitioner who can only reach for a policy
gradient method is poorly equipped for that reality. A practitioner who
recognizes the same mathematical structure across trajectory optimization, MPC,
dynamic programming, and deep RL can choose the right tool and explain why.

Reinforcement learning did not develop in isolation: its foundations draw on
control theory, dynamic programming, operations research, and economics. The same
ideas recur under different names in each community. Making those connections
explicit is a central goal of the course.

## Prerequisites

Students are expected to be comfortable with:

- **Mathematics.** Linear algebra, multivariable calculus, and probability at the
  undergraduate level.
- **Optimization.** Unconstrained and constrained optimization; gradient-based
  methods. Familiarity with KKT conditions is helpful but will be reviewed.
- **Programming.** Python, with working knowledge of NumPy. The course uses JAX
  and nonlinear programming solvers; no prior experience with either is assumed.
- **Machine learning.** A prior course covering supervised learning and neural
  network training.

Prior exposure to reinforcement learning is *not* required.

## Learning outcomes

By the end of the course, you should be able to:

1. Formulate a sequential decision problem as a discrete-time optimal control
   problem or an MDP, stating dynamics, objective, and constraints explicitly,
   and justify the modeling choices you made.
2. Solve trajectory optimization problems numerically using single shooting,
   multiple shooting, and direct collocation, and explain the trade-offs among
   them in terms of conditioning, sparsity, and solver behavior.
3. Implement model predictive control, and reason about recursive feasibility,
   stability, constraint softening, and fallback behavior.
4. Derive dynamic programming recursions for finite- and infinite-horizon
   problems, and characterize Bellman operators as contractions on an appropriate
   space.
5. Explain approximate dynamic programming as the projection of a Bellman
   residual onto an approximation space, and connect this view to fitted value
   and fitted Q iteration.
6. Analyze the Monte Carlo estimators used throughout RL, identifying sources of
   bias and variance, including maximization bias.
7. Situate deep RL algorithms — DQN and its extensions, DDPG, TD3, path
   consistency learning, MPPI, and policy gradient methods — within this
   framework, and explain each design choice as a response to a specific
   difficulty.
8. Read and critically assess research papers spanning the RL, control, and
   operations research literatures.

## Textbook

The primary text is the course book, written for this class and freely available
online:

> Pierre-Luc Bacon. *Building Up RL: From Dynamics and Control to Learning.*
> <https://pierrelucbacon.com/rlbook/>

The book is executable: figures and examples are generated from code you can run
and modify. The source is on
[GitHub](https://github.com/pierrelux/rlbook); corrections and issues are welcome.

### Supplementary references

These are recommended for depth on particular topics; none is required.

- D. P. Bertsekas. *Dynamic Programming and Optimal Control*, Vols. I–II.
  Athena Scientific.
- M. L. Puterman. *Markov Decision Processes: Discrete Stochastic Dynamic
  Programming.* Wiley, 1994.
- R. S. Sutton and A. G. Barto. *Reinforcement Learning: An Introduction*,
  2nd ed. MIT Press, 2018.
- J. T. Betts. *Practical Methods for Optimal Control and Estimation Using
  Nonlinear Programming*, 2nd ed. SIAM.
- J. B. Rawlings, D. Q. Mayne, and M. M. Diehl. *Model Predictive Control:
  Theory, Computation, and Design*, 2nd ed. Nob Hill.
- J. Nocedal and S. J. Wright. *Numerical Optimization*, 2nd ed. Springer.
- W. B. Powell. *Reinforcement Learning and Stochastic Optimization.* Wiley, 2022.

## Evaluation

:::{warning}
The evaluation scheme below is **not final**. Weights and dates will be
confirmed at the first lecture and this page will be updated accordingly.
:::

| Component | Weight | Due |
|---|---|---|
| Assignments | *TBD* | *TBD* |
| Paper presentation | *TBD* | *TBD* |
| Midterm | *TBD* | *TBD* |
| Final project — proposal | *TBD* | *TBD* |
| Final project — report and presentation | *TBD* | *TBD* |
| **Total** | **100%** | |

**Grading scale.** *TBD*

**Late policy.** *TBD*

## Tentative schedule

The plan below maps the course onto the book. It is indicative: pacing will be
adjusted as the term goes on. Week numbers are nominal, not tied one-to-one to
calendar dates — the class meets Mondays and Thursdays from August 31 to
December 9, 2026, with no sessions between October 16 and October 26; exact
dates (including any additional holidays) will be confirmed in class.

| Week | Topic | Reading |
|---|---|---|
| 1 | The decision problem; why formulation comes first | [Why This Book?](intro.md) |
| 2 | Dynamics models and state space; deterministic to stochastic; partial observability | [Dynamics](dynamics.md) |
| 3 | Discrete-time optimal control problems; existence and optimality conditions | [Trajectory Optimization](trajectories.md) |
| 4 | Sequential and simultaneous methods; single and multiple shooting; adjoints | [Trajectory Optimization](trajectories.md) |
| 5 | Direct transcription and collocation; polynomial interpolation | [Collocation](collocation.md) |
| 6 | Model predictive control: closing the loop by replanning; theoretical guarantees; variants | [MPC](mpc.md) |
| 7 | MPC in practice: constraint softening, feasibility restoration, backup controllers; parametric optimization | [MPC](mpc.md) |
| 8 | Dynamic programming: backward recursion, continuous spaces, the linear quadratic regulator | [Dynamic Programming](dp.md) |
| 9 | Markov decision processes: Bellman operators, infinite horizon, value and policy iteration | [Dynamic Programming](dp.md) |
| 10 | Smoothing and regularized MDPs; projection and weighted residual methods | [Smoothing](smoothing.md), [Projection](projection.md) |
| 11 | Monte Carlo methods; overestimation bias and its mitigation | [Monte Carlo](montecarlo.md) |
| 12 | Fitted Q iteration, NFQ, DQN and its extensions | [FQI](fqi.md) |
| 13 | Amortized optimization (NFQCA, DDPG, TD3, PCL, MPPI); policy gradient methods | [Amortization](amortization.md), [Policy Gradients](pg.md) |

The appendices on [worked examples](appendix_examples.md),
[initial value problems](appendix_ivps.md), and
[nonlinear programming](appendix_nlp.md) are reference material used throughout
the term rather than assigned to a specific week.

## Course policies

**Attendance.** *TBD*

**Collaboration.** Discussing ideas with classmates is encouraged. Unless an
assignment states otherwise, the work you submit must be written by you alone,
and you must name anyone you worked with.

**Generative AI.** *TBD* — the specific policy will be stated on each assignment.
Where AI assistance is permitted, you remain fully responsible for the
correctness of everything you submit and you must disclose how it was used.

**Academic integrity.** All work is subject to the University's regulations on
plagiarism and fraud. See
<https://integrite.umontreal.ca/> for the full policy.

**Accommodations.** Students registered with the *Soutien aux personnes
étudiantes en situation de handicap* (SESH) service should contact the instructor
early in the term so that arrangements can be made. See
<https://vieetudiante.umontreal.ca/a-propos/service/soutien-personnes-etudiantes-situation-handicap>
or write to <soutienhandicap@sve.umontreal.ca>.

**Changes to this syllabus.** This document may be revised during the term.
Changes will be announced in class and reflected on this page.
````

````{tab-item} Français
:sync: fr

## Renseignements généraux

| | |
|---|---|
| **Sigle** | IFT 6162 |
| **Titre** | Apprentissage par renforcement, commande optimale |
| **Crédits** | *À déterminer* |
| **Trimestre** | Automne 2026 |
| **Horaire** | Lundi 10 h 30 – 12 h 29 et jeudi 13 h 30 – 15 h 29, du 31 août au 16 octobre et du 26 octobre au 9 décembre 2026 |
| **Local** | Campus Montréal — local *à déterminer* |
| **Enseignant** | Pierre-Luc Bacon |
| **Courriel** | *À déterminer* |
| **Disponibilités** | *À déterminer* |
| **Auxiliaires d'enseignement** | *À déterminer* |
| **Langue d'enseignement** | *À déterminer* (le matériel du cours est en anglais) |
| **Site du cours** | <https://pierrelucbacon.com/rlbook/> |

## Description

Ce cours construit l'apprentissage par renforcement à partir de ses fondements en
dynamique, en optimisation et en commande. Plutôt que de partir du processus
décisionnel de Markov tabulaire pour élargir ensuite le cadre, nous partons du
problème de décision lui-même : comment écrire le modèle d'un système, comment
formuler un objectif qui se déploie dans le temps, et comment résoudre
numériquement le problème d'optimisation qui en résulte. La programmation
dynamique, la commande prédictive et les algorithmes modernes d'apprentissage par
renforcement profond apparaissent alors comme autant de réponses à une même
question, chacune portant ses propres hypothèses et ses propres modes de
défaillance.

Le fil conducteur est que l'essentiel du travail d'application de
l'apprentissage par renforcement se fait *avant* le choix d'un algorithme. Les
capteurs produisent des données bruitées, certaines contraintes ne sont pas
négociables, et les objectifs évoluent ou entrent en conflit. Une personne qui ne
dispose que d'une méthode de gradient de politique est mal outillée devant cette
réalité. Celle qui reconnaît la même structure mathématique dans l'optimisation
de trajectoires, la commande prédictive, la programmation dynamique et
l'apprentissage profond peut choisir le bon outil et justifier son choix.

L'apprentissage par renforcement ne s'est pas développé en vase clos : ses
fondements puisent dans la théorie de la commande, la programmation dynamique, la
recherche opérationnelle et l'économie. Les mêmes idées y reviennent sous des
noms différents. Rendre ces liens explicites est un objectif central du cours.

## Préalables

On attend des étudiantes et étudiants une aisance avec :

- **Mathématiques.** Algèbre linéaire, calcul à plusieurs variables et
  probabilités de niveau premier cycle.
- **Optimisation.** Optimisation avec et sans contraintes ; méthodes fondées sur
  le gradient. Une familiarité avec les conditions KKT est utile, mais celles-ci
  seront revues.
- **Programmation.** Python, avec une connaissance pratique de NumPy. Le cours
  utilise JAX et des solveurs de programmation non linéaire ; aucune expérience
  préalable n'est présumée pour l'un ou l'autre.
- **Apprentissage automatique.** Un cours antérieur couvrant l'apprentissage
  supervisé et l'entraînement de réseaux de neurones.

Aucune exposition préalable à l'apprentissage par renforcement n'est requise.

## Objectifs d'apprentissage

Au terme du cours, vous devriez être en mesure de :

1. Formuler un problème de décision séquentielle comme un problème de commande
   optimale en temps discret ou comme un MDP, en énonçant explicitement la
   dynamique, l'objectif et les contraintes, et justifier vos choix de
   modélisation.
2. Résoudre numériquement des problèmes d'optimisation de trajectoires par tir
   simple, tir multiple et collocation directe, et expliquer les compromis entre
   ces approches en matière de conditionnement, de creux et de comportement du
   solveur.
3. Mettre en œuvre la commande prédictive et raisonner sur la faisabilité
   récursive, la stabilité, l'assouplissement des contraintes et les
   comportements de repli.
4. Établir les récurrences de la programmation dynamique en horizon fini et
   infini, et caractériser les opérateurs de Bellman comme des contractions sur
   un espace approprié.
5. Expliquer la programmation dynamique approchée comme la projection d'un résidu
   de Bellman sur un espace d'approximation, et relier ce point de vue à
   l'itération sur la valeur ajustée et à l'itération sur $Q$ ajustée.
6. Analyser les estimateurs de Monte-Carlo employés en apprentissage par
   renforcement en identifiant les sources de biais et de variance, dont le biais
   de maximisation.
7. Situer les algorithmes d'apprentissage profond — DQN et ses extensions, DDPG,
   TD3, l'apprentissage par cohérence de chemin, MPPI et les méthodes de gradient
   de politique — dans ce cadre, et expliquer chaque choix de conception comme
   une réponse à une difficulté précise.
8. Lire et évaluer de façon critique des articles de recherche issus des
   littératures de l'apprentissage par renforcement, de la commande et de la
   recherche opérationnelle.

## Manuel

Le texte principal est le livre du cours, rédigé pour cette classe et accessible
gratuitement en ligne :

> Pierre-Luc Bacon. *Building Up RL: From Dynamics and Control to Learning.*
> <https://pierrelucbacon.com/rlbook/>

Le livre est exécutable : les figures et les exemples sont produits à partir de
code que vous pouvez exécuter et modifier. Le code source est sur
[GitHub](https://github.com/pierrelux/rlbook) ; les corrections et les signalements
sont les bienvenus.

### Références complémentaires

Recommandées pour approfondir certains sujets ; aucune n'est obligatoire.

- D. P. Bertsekas. *Dynamic Programming and Optimal Control*, vol. I–II.
  Athena Scientific.
- M. L. Puterman. *Markov Decision Processes: Discrete Stochastic Dynamic
  Programming.* Wiley, 1994.
- R. S. Sutton et A. G. Barto. *Reinforcement Learning: An Introduction*,
  2<sup>e</sup> éd. MIT Press, 2018.
- J. T. Betts. *Practical Methods for Optimal Control and Estimation Using
  Nonlinear Programming*, 2<sup>e</sup> éd. SIAM.
- J. B. Rawlings, D. Q. Mayne et M. M. Diehl. *Model Predictive Control: Theory,
  Computation, and Design*, 2<sup>e</sup> éd. Nob Hill.
- J. Nocedal et S. J. Wright. *Numerical Optimization*, 2<sup>e</sup> éd. Springer.
- W. B. Powell. *Reinforcement Learning and Stochastic Optimization.* Wiley, 2022.

## Évaluation

:::{warning}
Le barème ci-dessous **n'est pas définitif**. Les pondérations et les dates
seront confirmées au premier cours et cette page sera mise à jour en conséquence.
:::

| Élément | Pondération | Échéance |
|---|---|---|
| Travaux pratiques | *À déterminer* | *À déterminer* |
| Présentation d'un article | *À déterminer* | *À déterminer* |
| Examen de mi-session | *À déterminer* | *À déterminer* |
| Projet final — proposition | *À déterminer* | *À déterminer* |
| Projet final — rapport et présentation | *À déterminer* | *À déterminer* |
| **Total** | **100 %** | |

**Barème de notation.** *À déterminer*

**Politique sur les retards.** *À déterminer*

## Calendrier provisoire

Le plan ci-dessous met le cours en correspondance avec le livre. Il est indicatif :
le rythme sera ajusté au fil du trimestre. La numérotation des semaines est
nominale et ne correspond pas nécessairement aux dates du calendrier : le cours
a lieu les lundis et jeudis du 31 août au 9 décembre 2026, sans séance entre le
16 et le 26 octobre ; les dates exactes (incluant tout congé additionnel) seront
confirmées en classe.

| Semaine | Sujet | Lecture |
|---|---|---|
| 1 | Le problème de décision ; pourquoi la formulation vient en premier | [Why This Book?](intro.md) |
| 2 | Modèles de dynamique et espace d'états ; du déterministe au stochastique ; observabilité partielle | [Dynamics](dynamics.md) |
| 3 | Problèmes de commande optimale en temps discret ; existence et conditions d'optimalité | [Trajectory Optimization](trajectories.md) |
| 4 | Méthodes séquentielles et simultanées ; tir simple et tir multiple ; états adjoints | [Trajectory Optimization](trajectories.md) |
| 5 | Transcription directe et collocation ; interpolation polynomiale | [Collocation](collocation.md) |
| 6 | Commande prédictive : boucler la boucle par replanification ; garanties théoriques ; variantes | [MPC](mpc.md) |
| 7 | La commande prédictive en pratique : assouplissement des contraintes, restauration de la faisabilité, contrôleurs de secours ; optimisation paramétrique | [MPC](mpc.md) |
| 8 | Programmation dynamique : récurrence arrière, espaces continus, régulateur linéaire quadratique | [Dynamic Programming](dp.md) |
| 9 | Processus décisionnels de Markov : opérateurs de Bellman, horizon infini, itération sur la valeur et sur la politique | [Dynamic Programming](dp.md) |
| 10 | Lissage et MDP régularisés ; méthodes de projection et de résidus pondérés | [Smoothing](smoothing.md), [Projection](projection.md) |
| 11 | Méthodes de Monte-Carlo ; biais de surestimation et stratégies d'atténuation | [Monte Carlo](montecarlo.md) |
| 12 | Itération sur $Q$ ajustée, NFQ, DQN et ses extensions | [FQI](fqi.md) |
| 13 | Optimisation amortie (NFQCA, DDPG, TD3, PCL, MPPI) ; méthodes de gradient de politique | [Amortization](amortization.md), [Policy Gradients](pg.md) |

Les annexes sur les [exemples détaillés](appendix_examples.md), les
[problèmes à valeur initiale](appendix_ivps.md) et la
[programmation non linéaire](appendix_nlp.md) servent de matériel de référence
tout au long du trimestre plutôt que d'être rattachées à une semaine précise.

## Règlements du cours

**Présence.** *À déterminer*

**Collaboration.** Les échanges d'idées entre collègues sont encouragés. Sauf
indication contraire dans l'énoncé d'un travail, le texte que vous remettez doit
être rédigé par vous seul, et vous devez nommer les personnes avec qui vous avez
travaillé.

**Intelligence artificielle générative.** *À déterminer* — la politique
applicable sera précisée dans chaque énoncé de travail. Lorsque l'usage d'une IA
est permis, vous demeurez entièrement responsable de l'exactitude de ce que vous
remettez et vous devez déclarer comment elle a été utilisée.

**Intégrité intellectuelle.** Tous les travaux sont soumis au règlement
disciplinaire de l'Université sur le plagiat et la fraude. Voir
<https://integrite.umontreal.ca/> pour le texte complet.

**Accommodements.** Les personnes inscrites au service de Soutien aux personnes
étudiantes en situation de handicap (SESH) sont invitées à communiquer avec
l'enseignant tôt dans le trimestre afin que les mesures nécessaires soient mises
en place. Voir
<https://vieetudiante.umontreal.ca/a-propos/service/soutien-personnes-etudiantes-situation-handicap>
ou écrire à <soutienhandicap@sve.umontreal.ca>.

**Modifications au plan de cours.** Ce document peut être révisé en cours de
trimestre. Toute modification sera annoncée en classe et reflétée sur cette page.
````

`````
