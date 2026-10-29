---
# The page outline is suppressed: headings live inside language tabs, so an
# outline would list both languages at once and half its links would point
# into a hidden panel.
site:
  hide_outline: true
---

# Syllabus · Plan de cours

**IFT 6162: Apprentissage par renforcement, commande optimale
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
| **Schedule** | Mondays 10:30–12:29 and Thursdays 13:30–15:29; Aug 31 – Oct 16 and Oct 26 – Dec 9, 2026 (14 teaching weeks, ~25 sessions) |
| **Room** | Campus Montréal, room *TBD* |
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
7. Situate deep RL algorithms, including DQN and its extensions, DDPG, TD3, path
   consistency learning, MPPI, and policy gradient methods, within this
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

| Component | Weight | Due |
|---|---|---|
| Applied project | 30% | *TBD* |
| In-class midterm 1 | 15% | *TBD* |
| In-class midterm 2 | 15% | *TBD* |
| In-class final examination | 40% | *TBD* |
| **Total** | **100%** | |

The three examinations are completed in person on paper, without electronic
devices or generative-AI tools. Each student may bring one double-sided
reference sheet that they prepared themselves. Questions may ask students to
derive or interpret a result, inspect supplied code, identify a flaw in a model
or algorithm, and justify a diagnosis. Memorizing Python syntax is not an
examination objective.

Exercises, self-checks, and coding practice are formative and ungraded. They
prepare students for the analytical and diagnostic work required on the
examinations and in the project.

The applied project is completed in teams of exactly three. If enrollment makes
that impossible, exceptions will be arranged privately by the instructor.
Each team will formulate and investigate a substantive control-and-learning
problem rather than reproduce a canned benchmark. The deliverables are working
code, a poster, and an oral defense. A short proposal is an ungraded checkpoint.
The team artifact establishes the base project grade; an individual's grade may
be adjusted when the oral defense shows a materially different level of
understanding.

**Grading scale.** *TBD*

**Project checkpoint and late policy.** Dates and late arrangements are *TBD*.

## Tentative schedule

The course meets **twice a week**, Mondays 10:30–12:29 and Thursdays
13:30–15:29, over **14 teaching weeks**, for roughly 25 sessions in total. Each
row below therefore covers two meetings, except where a holiday intervenes.
Pacing is indicative and will be adjusted as the term goes on.

| Week | Topic | Reading |
|---|---|---|
| 1 · Aug 31 – Sep 3 | Course overview; system boundaries; state and action | [Controlled Systems](modeling-controlled-systems.md) |
| 2 · Sep 7 – 10 | Dynamics models and the state-space perspective | [Controlled Systems](modeling-controlled-systems.md) |
| 3 · Sep 14 – 17 | Stochastic dynamics; partial observation; programs and data as model interfaces | [Stochastic Dynamics](stochastic-dynamics-observation.md), [Model Interfaces](model-interfaces.md) |
| 4 · Sep 21 – 24 | Discrete-time optimal control problems; existence and optimality conditions | [Finite-Horizon Optimal Control](discrete-time-optimal-control.md) |
| 5 · Sep 28 – Oct 1 | Adjoints and the Pontryagin principle; single and multiple shooting | [Discrete-Time PMP](discrete-time-pmp.md), [Numerical Trajectory Optimization](numerical-trajectory-optimization.md) |
| 6 · Oct 5 – 8 | Direct transcription; polynomial interpolation | [Continuous-Time Collocation](continuous-time-collocation.md) |
| 7 · Oct 12 – 15 | A compendium of direct transcription methods; worked examples | [Continuous-Time Collocation](continuous-time-collocation.md) |
| 8 · Oct 26 – 29 | Closing the loop by replanning; theoretical guarantees; MPC variants | [Receding-Horizon Control](receding-horizon-control.md), [MPC Variants](mpc-variants-reliability.md) |
| 9 · Nov 2 – 5 | MPC failure handling; parametric optimization and approximate controllers | [Reliable MPC](mpc-variants-reliability.md), [Parametric Controllers](parametric-controllers.md) |
| 10 · Nov 9 – 12 | Backward recursion, continuous spaces, and the linear quadratic regulator | [Finite-Horizon Dynamic Programming](finite-horizon-dp.md) |
| 11 · Nov 16 – 19 | Stochastic and infinite-horizon MDPs; Bellman operators; value and policy iteration | [Stochastic DP](stochastic-dp.md), [Infinite-Horizon MDPs](infinite-horizon-mdps.md) |
| 12 · Nov 23 – 26 | Regularized MDPs; weighted residuals; approximate Bellman equations | [Regularized DP](regularized-dp.md), [Weighted Residuals](weighted-residual-methods.md), [Approximate Bellman Equations](approximate-bellman-equations.md) |
| 13 · Nov 30 – Dec 3 | Monte Carlo methods and overestimation bias; fitted Q iteration, NFQ, DQN | [Monte Carlo](monte-carlo-bellman-estimation.md), [FQI](fitted-q-iteration.md) |
| 14 · Dec 7 | Amortized action optimization; stochastic gradient estimators; regularized and direct policy optimization | [Amortized Action Optimization](amortized-action-optimization.md), [Gradient Estimation](gradient-estimation.md), [Regularized Policy Learning](regularized-policy-learning.md), [Policy Gradients](policy-gradients.md) |

**No class on:**

- Monday **September 7**: Labour Day (*Fête du travail*)
- Monday **October 12**: Thanksgiving (*Action de grâce*)
- **October 19–25**: *période d'activités libres* (no meetings; this is the gap
  between the two blocks in the registrar's listing)
- Thursday **December 10** falls outside the course's end date of December 9,
  so week 14 has a Monday session only.

:::{note}
The registrar lists the course as beginning **Monday, August 31**, while the
University's academic calendar gives the *rentrée* as **Tuesday, September 1**.
The date of the first Monday meeting will be confirmed before the term starts.
:::

The appendices on [worked examples](appendix_examples.md),
[initial value problems](appendix_ivps.md), and
[nonlinear programming](appendix_nlp.md) are reference material used throughout
the term rather than assigned to a specific week.

## Course policies

**Attendance.** *TBD*

**Collaboration.** Discussion and collaboration are encouraged on formative
work. The project is collaborative within the assigned team. Each student must
be able to explain and defend the team's complete submission.

**Generative AI.** Generative AI may be used for formative work and the applied
project. Disclosure is not required. Each student remains responsible for every
claim, equation, baseline, experiment, and line of submitted code and must be
able to defend them during the oral defense. Generative AI and electronic
devices are prohibited during in-class examinations.

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
| **Horaire** | Lundi 10 h 30 – 12 h 29 et jeudi 13 h 30 – 15 h 29; du 31 août au 16 octobre et du 26 octobre au 9 décembre 2026 (14 semaines de cours, ~25 séances) |
| **Local** | Campus Montréal, local *à déterminer* |
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
7. Situer les algorithmes d'apprentissage profond, dont DQN et ses extensions, DDPG,
   TD3, l'apprentissage par cohérence de chemin, MPPI et les méthodes de gradient
   de politique, dans ce cadre, et expliquer chaque choix de conception comme
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

| Élément | Pondération | Échéance |
|---|---|---|
| Projet appliqué | 30 % | *À déterminer* |
| Premier examen de mi-session en classe | 15 % | *À déterminer* |
| Deuxième examen de mi-session en classe | 15 % | *À déterminer* |
| Examen final en classe | 40 % | *À déterminer* |
| **Total** | **100 %** | |

Les trois examens se font en personne, sur papier, sans appareil électronique
ni outil d'intelligence artificielle générative. Chaque personne peut apporter
une feuille de référence recto verso qu'elle a préparée elle-même. Les questions
peuvent demander d'établir ou d'interpréter un résultat, d'examiner du code
fourni, de repérer une erreur dans un modèle ou un algorithme et de justifier le
diagnostic. La mémorisation de la syntaxe Python ne constitue pas un objectif
d'évaluation.

Les exercices, les autoévaluations et les activités de programmation sont
formatifs et ne sont pas notés. Ils préparent au travail d'analyse et de
diagnostic demandé dans les examens et le projet.

Le projet appliqué se réalise en équipes d'exactement trois personnes. Si le
nombre d'inscriptions rend cette règle impossible, l'enseignant réglera les
exceptions en privé. Chaque équipe formulera et étudiera un problème substantiel
de commande et d'apprentissage, plutôt que de reproduire un banc d'essai
préfabriqué. Les livrables sont du code fonctionnel, une affiche et une défense
orale. Une courte proposition sert de jalon non noté. La production d'équipe
établit la note de base du projet; la note individuelle peut être ajustée si la
défense orale indique un niveau de compréhension sensiblement différent.

**Barème de notation.** *À déterminer*

**Jalon du projet et politique sur les retards.** Les dates et les modalités
sont *à déterminer*.

## Calendrier provisoire

Le cours a lieu **deux fois par semaine**, le lundi de 10 h 30 à 12 h 29 et le
jeudi de 13 h 30 à 15 h 29, sur **14 semaines de cours**, soit environ 25
séances au total. Chaque ligne ci-dessous couvre donc deux séances, sauf lorsqu'un
congé s'intercale. Le rythme est indicatif et sera ajusté au fil du trimestre.

| Semaine | Sujet | Lecture |
|---|---|---|
| 1 · 31 août – 3 sept. | Présentation du cours ; frontière du système ; état et action | [Controlled Systems](modeling-controlled-systems.md) |
| 2 · 7 – 10 sept. | Modèles de dynamique et perspective de l'espace d'états | [Controlled Systems](modeling-controlled-systems.md) |
| 3 · 14 – 17 sept. | Dynamique stochastique ; observation partielle ; programmes et données comme interfaces de modèle | [Stochastic Dynamics](stochastic-dynamics-observation.md), [Model Interfaces](model-interfaces.md) |
| 4 · 21 – 24 sept. | Problèmes de commande optimale en temps discret ; existence et conditions d'optimalité | [Finite-Horizon Optimal Control](discrete-time-optimal-control.md) |
| 5 · 28 sept. – 1<sup>er</sup> oct. | États adjoints et principe de Pontryagin ; tir simple et tir multiple | [Discrete-Time PMP](discrete-time-pmp.md), [Numerical Trajectory Optimization](numerical-trajectory-optimization.md) |
| 6 · 5 – 8 oct. | Transcription directe ; interpolation polynomiale | [Continuous-Time Collocation](continuous-time-collocation.md) |
| 7 · 12 – 15 oct. | Panorama des méthodes de transcription directe ; exemples détaillés | [Continuous-Time Collocation](continuous-time-collocation.md) |
| 8 · 26 – 29 oct. | Boucler la boucle par replanification ; garanties théoriques ; variantes de commande prédictive | [Receding-Horizon Control](receding-horizon-control.md), [MPC Variants](mpc-variants-reliability.md) |
| 9 · 2 – 5 nov. | Gestion des défaillances en commande prédictive ; optimisation paramétrique et contrôleurs approchés | [Reliable MPC](mpc-variants-reliability.md), [Parametric Controllers](parametric-controllers.md) |
| 10 · 9 – 12 nov. | Récurrence arrière, espaces continus et régulateur linéaire quadratique | [Finite-Horizon Dynamic Programming](finite-horizon-dp.md) |
| 11 · 16 – 19 nov. | MDP stochastiques et à horizon infini ; opérateurs de Bellman ; itérations sur la valeur et la politique | [Stochastic DP](stochastic-dp.md), [Infinite-Horizon MDPs](infinite-horizon-mdps.md) |
| 12 · 23 – 26 nov. | MDP régularisés ; résidus pondérés ; équations de Bellman approchées | [Regularized DP](regularized-dp.md), [Weighted Residuals](weighted-residual-methods.md), [Approximate Bellman Equations](approximate-bellman-equations.md) |
| 13 · 30 nov. – 3 déc. | Méthodes de Monte-Carlo et biais de surestimation ; itération sur $Q$ ajustée, NFQ, DQN | [Monte Carlo](monte-carlo-bellman-estimation.md), [FQI](fitted-q-iteration.md) |
| 14 · 7 déc. | Optimisation amortie des actions ; estimateurs de gradient stochastique ; optimisation régularisée et directe des politiques | [Amortized Action Optimization](amortized-action-optimization.md), [Gradient Estimation](gradient-estimation.md), [Regularized Policy Learning](regularized-policy-learning.md), [Policy Gradients](policy-gradients.md) |

**Aucune séance :**

- Lundi **7 septembre**: congé universitaire, Fête du travail
- Lundi **12 octobre**: congé universitaire, Action de grâce
- **19 au 25 octobre**: période d'activités libres (c'est l'intervalle entre les
  deux blocs de dates au répertoire des cours)
- Le jeudi **10 décembre** dépasse la date de fin du cours (9 décembre) : la
  semaine 14 ne comporte donc qu'une séance, le lundi.

:::{note}
Le répertoire des cours indique un début le **lundi 31 août**, alors que le
calendrier universitaire fixe la rentrée au **mardi 1<sup>er</sup> septembre**.
La date de la première séance du lundi sera confirmée avant le début du trimestre.
:::

Les annexes sur les [exemples détaillés](appendix_examples.md), les
[problèmes à valeur initiale](appendix_ivps.md) et la
[programmation non linéaire](appendix_nlp.md) servent de matériel de référence
tout au long du trimestre plutôt que d'être rattachées à une semaine précise.

## Règlements du cours

**Présence.** *À déterminer*

**Collaboration.** Les échanges et la collaboration sont encouragés dans les
activités formatives. Le projet se réalise en collaboration au sein de l'équipe
assignée. Chaque personne doit pouvoir expliquer et défendre l'ensemble de la
production de son équipe.

**Intelligence artificielle générative.** L'intelligence artificielle générative
peut être utilisée dans les activités formatives et le projet appliqué. Aucune
déclaration d'utilisation n'est exigée. Chaque personne demeure responsable de
chaque affirmation, équation, méthode de référence, expérience et ligne de code
remise, et doit pouvoir les défendre lors de la défense orale. L'intelligence
artificielle générative et les appareils électroniques sont interdits pendant
les examens en classe.

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
