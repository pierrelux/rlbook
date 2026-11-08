# Glossaire et conventions de la version française

Ce fichier n'est pas une page du livre. Il fixe les choix de traduction pour
que tous les chapitres emploient les mêmes termes. Ajouter ici tout nouveau
terme récurrent avant de l'employer dans un chapitre.

## Règles générales

- Aucun anglicisme : pas de « RL », « MPC », « policy », « reward »,
  « feedback », « benchmark », etc. On écrit « apprentissage par
  renforcement » en toutes lettres plutôt qu'un sigle.
- Les cellules de code, leurs commentaires et les étiquettes des figures
  produites par le code restent identiques à la version anglaise. Le build
  français réutilise ainsi le cache d'exécution anglais.
- Les étiquettes MyST (`(label)=`, `:label:`), les noms de directives et les
  clés de citation ne se traduisent pas.
- Notation du livre : pas de notation `d` pour les différentielles, π partout
  pour la politique, `log` et non `ln`.
- Typographie québécoise : guillemets « » avec espaces insécables, espace
  insécable avant le deux-points seulement (pas avant ; ! ?). Décimales avec
  un point partout, y compris dans le texte, pour rester cohérent avec les
  formules, le code et les figures.
- Les titres de sections suivent la casse française (majuscule au premier mot
  seulement).

## Termes

| Anglais | Français |
|---|---|
| reinforcement learning | apprentissage par renforcement |
| agent / environment | agent / environnement |
| policy | politique |
| reward / return | récompense / rendement |
| value function | fonction de valeur |
| action-value function | fonction de valeur d'action |
| state / observation / action | état / observation / action |
| transition | transition |
| discount factor | facteur d'actualisation |
| Markov decision process | processus de décision markovien |
| Bellman equation / operator | équation / opérateur de Bellman |
| dynamic programming | programmation dynamique |
| approximate dynamic programming | programmation dynamique approchée |
| temporal difference | différence temporelle |
| Monte Carlo | Monte-Carlo |
| policy gradient | gradient de politique |
| actor-critic | acteur-critique |
| fitted Q iteration | itération Q ajustée |
| function approximation | approximation de fonctions |
| amortized | amorti |
| control (theory) | commande, théorie de la commande |
| optimal control | contrôle optimal |
| model predictive control | commande prédictive |
| controller | correcteur |
| feedback / feedback law | rétroaction / loi de rétroaction |
| closed loop / open loop | boucle fermée / boucle ouverte |
| plant | procédé |
| input / measurement | entrée / mesure |
| disturbance / noise | perturbation / bruit |
| trajectory optimization | optimisation de trajectoires |
| shooting (single, multiple) | tir (simple, multiple) |
| collocation | collocation |
| horizon / receding horizon | horizon / horizon glissant |
| constraint / objective / cost | contrainte / objectif / coût |
| operations research | recherche opérationnelle |
| data-generating process | processus générateur des données |
| covariates | covariables |
| logged data | données historiques |
| sample / sampler | échantillon / échantillonneur |
| simulator | simulateur |
| benchmark | banc d'essai |
| state-contingent | conditionnel à l'état |
| self-check | autoévaluation |
| working definition | définition de travail |
| initial-value problem | problème à valeur initiale |
| nonlinear programming | programmation non linéaire |
| system identification | identification de système |
| partial observability | observabilité partielle |
| sim-to-real gap | écart entre simulation et réalité |
| discrete-time optimal control problem (DOCP) | problème de contrôle optimal en temps discret (PCOTD) |
| nonlinear program (NLP) | programme non linéaire (PNL) |
| sequential quadratic programming (SQP) | programmation quadratique successive (PQS) |
| quadratic program (QP) | programme quadratique |
| ODE | équation différentielle ordinaire (EDO) |
| LQR | commande linéaire quadratique |
| costate / adjoint | état adjoint |
| replay / rollout | rejeu / déroulement |
| token / prompt (LLM) | jeton / invite |
| prefill / decode | préremplissage / décodage |
| throughput | débit |
| governor (battery current) | limiteur de courant |
| fitted value iteration / value iteration | itération sur les valeurs ajustée / itération sur les valeurs |
| Bellman backup | mise à jour de Bellman |
| greedy / ε-greedy | glouton / ε-glouton |
| replay buffer / replay ratio | tampon de rejeu / taux de rejeu |
| target network / online network | réseau cible / réseau en ligne |
| warm start / cold start | démarrage à chaud / à froid |
| mini-batch | mini-lot |
| behavior policy | politique de comportement |
| bootstrapping (TD) | autoamorçage |
| TD error | erreur de différence temporelle |
| importance sampling | échantillonnage préférentiel |
| action gap | écart entre actions |
| deadly triad | triade mortelle |
| policy iteration / policy evaluation | itération sur les politiques / évaluation de politique |
| on-policy / off-policy | sur politique / hors politique |
| test function / hat functions | fonction test / fonctions chapeau |
| Chebyshev | Tchebychev |
| semi-gradient | semi-gradient |
| soft actor-critic | acteur-critique souple |
| MDP | processus de décision markovien (en toutes lettres) |
| sup-norm | norme sup |
| stochastic shortest path | plus court chemin stochastique |
| iLQR / DDP | commande linéaire quadratique itérative (iLQR) / programmation dynamique différentielle (DDP) |
| cost-to-go | coût restant |
| backward / forward pass | passe arrière / passe avant |
| line search | recherche linéaire |
| decision epoch | époque de décision |
| terminal ingredients | éléments terminaux |
| recursive feasibility | faisabilité récursive |
| setpoint | consigne |
| tube / economic MPC | commande prédictive à tubes / économique |
| probability distribution | loi (loi de proposition, loi cible) |
| self-normalized | autonormalisé |
| surrogate model | modèle de substitution |
| PDE | équation aux dérivées partielles (EDP) |
| effective sample size | taille d'échantillon effective |
| seed | germe |
| clipping | écrêtage |
| curse of dimensionality | fléau de la dimension |
| slack variables | variables d'écart |
| backup controller | correcteur de secours |
| reference governor | gouverneur de référence |
| replay buffer | tampon de rejeu |
| target network | réseau cible |
| actor / critic | acteur / critique |
| soft max / soft Bellman | maximum adouci / Bellman adouci |
| score-function estimator | estimateur par fonction de score |
| reparameterization trick | astuce de reparamétrisation |
| likelihood ratio | rapport de vraisemblance |
| control variate | variable de contrôle |
| baseline | référence |
| advantage / GAE | avantage / estimation généralisée de l'avantage (GAE) |
| surrogate loss | perte de substitution |
| trust region | région de confiance |
| checkpoint | point de contrôle |
| epoch | époque |
| credit assignment | attribution du mérite |
| algorithm names (DQN, PPO, SAC, TD3, REINFORCE, Q-learning…) | kept as is |
| smooth Bellman equation | équation de Bellman lisse (soft → adouci) |
| value / policy iteration | itération sur les valeurs / sur les politiques |
| greedy | glouton |
| regularizer | régularisateur |
| inverse RL | apprentissage par renforcement inverse |
| path integral | intégrale de chemin |
| certainty equivalence | équivalence de certitude |
| IVP | problème à valeur initiale |
| stiff | raide |
| zero-order hold | bloqueur d'ordre zéro |
| feasible | réalisable |
| saddle point | point de selle |
| QP | programme quadratique |
| direct transcription | transcription directe |
| defect | défaut |
| mesh / node | maillage / nœud |
| stage (collocation, RK) / stage cost | étage / coût d'étage |
| sparse | creux |
| interior-point methods | méthodes de points intérieurs |
| condensing | condensation |
| cart-pole / swing-up | chariot-pendule / redressement |
| input shaper | façonneur d'entrée |
| initial guess | amorce (random seed → germe) |

## Liens vers les titres

Les liens implicites vers un titre (`#decision-rules-and-policies`) utilisent
le titre anglais. Quand on traduit un titre ciblé ainsi, ajouter l'étiquette
anglaise juste au-dessus : `(decision-rules-and-policies)=`.
