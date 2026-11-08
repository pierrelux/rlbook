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

## Liens vers les titres

Les liens implicites vers un titre (`#decision-rules-and-policies`) utilisent
le titre anglais. Quand on traduit un titre ciblé ainsi, ajouter l'étiquette
anglaise juste au-dessus : `(decision-rules-and-policies)=`.
