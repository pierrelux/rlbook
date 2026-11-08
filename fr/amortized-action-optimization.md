---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Optimisation amortie des actions

Le chapitre précédent a montré comment l'itération Q ajustée traite les grands espaces d'états au moyen de l'approximation de fonctions. L'itération Q ajustée (FQI) conserve une structure unifiée dans les cadres par lots et en ligne : un tampon de rejeu $\mathcal{B}_t$ qui induit une distribution empirique $\hat{P}_{\mathcal{B}_t}$, une application cible $T_q$ dérivée de l'opérateur de Bellman, une fonction de perte $\ell$ et un budget d'optimisation. Les algorithmes diffèrent par la façon dont ils instancient ces composantes (évolution du tampon, opérateur de Bellman strict ou adouci, fréquence des mises à jour), mais tous suivent le même gabarit.

Ce cadre cesse toutefois de fonctionner lorsque l'espace d'actions devient grand ou continu. Le calcul des cibles de Bellman exige d'évaluer $\max_{a' \in \mathcal{A}} q(s',a';\boldsymbol{\theta})$ pour chaque état suivant $s'$. Lorsque les actions sont continues ($\mathcal{A} \subset \mathbb{R}^m$), cette maximisation exige de résoudre un programme non linéaire à chaque calcul de cible. Pour un tampon de rejeu contenant des millions de transitions, cela devient d'un coût de calcul prohibitif.

Peut-on remplacer la maximisation répétée par une application apprise des états vers les actions?
L'**amortissement** investit du calcul pendant l'entraînement afin que la sélection
des actions n'exige qu'une passe avant au moment de l'exécution.

Les stratégies que nous examinons sont :

1. **Optimisation explicite** (section 2) : résoudre numériquement la maximisation pour un sous-ensemble d'états, en acceptant le coût de calcul en échange de solutions exactes.

2. **Amortissement par un réseau de politique** : apprendre un réseau de politique
   déterministe $\pi_{\boldsymbol{w}}$ qui approche $\arg\max_a
   q(s,a;\boldsymbol{\theta})$, ce qui permet de sélectionner rapidement une action
   par une seule passe avant. NFQCA, DDPG et TD3 instancient cette voie.

Chaque approche représente un point différent du compromis entre calcul et précision, et toutes s'inscrivent dans le gabarit de l'itération Q ajustée en modifiant la façon dont les cibles sont calculées.

## Optimisation explicite

Quel est le coût de calcul d'une résolution directe de la maximisation sur des
actions continues pour chaque cible de l'itération Q ajustée?

Rappelons que, dans les méthodes Q ajustées, l'idée principale est de calculer l'opérateur de Bellman seulement sur un sous-ensemble de tous les états, en comptant sur l'approximation de fonctions pour généraliser aux autres états. À chaque étape de la boucle d'approximations successives, on construit un jeu de données qui associe des paires état-action d'entrée aux évaluations correspondantes de l'opérateur d'optimalité :

$$
\mathcal{D}_n = \{((s, a), (\Bellman q)(s, a; \boldsymbol{\theta}_n)) \mid (s,a) \in \mathcal{B}\}
$$

Ce jeu de données est ensuite fourni à notre approximateur de fonctions (réseau de neurones, forêt aléatoire, modèle linéaire) pour obtenir le jeu de paramètres suivant :

$$
\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_n)
$$

Bien que cette stratégie permette de traiter des espaces d'états très grands, voire infinis (continus), elle exige encore une maximisation sur les actions ($\max_{a \in \mathcal{A}}$) lors de la création du jeu de données, au moment de calculer l'opérateur $\Bellman$ en chaque point de base. Cette maximisation devient coûteuse en calcul pour de grands espaces d'actions. On peut y remédier en ajoutant un autre niveau d'optimisation : pour chaque échantillon ajouté à notre jeu de données de régression, on emploie des méthodes d'optimisation numérique pour trouver des actions qui maximisent l'opérateur de Bellman pour l'état donné.

```{prf:algorithm} Itération Q ajustée avec optimisation explicite
:label: fitted-q-iteration-explicit

**Entrée :** processus de décision markovien $(S, \mathcal{A}, P, R, \gamma)$, points de base $\mathcal{B}$, approximateur de fonctions $q(s,a; \boldsymbol{\theta})$, nombre maximal d'itérations $N$, tolérance $\varepsilon > 0$

**Sortie :** paramètres $\boldsymbol{\theta}$ de l'approximation de la fonction Q

1. Initialiser $\boldsymbol{\theta}_0$
2. $n \leftarrow 0$
3. **répéter**
    1. $\mathcal{D} \leftarrow \emptyset$ $\quad$ // Jeu de données de régression
    2. **pour** chaque $(s,a,r,s') \in \mathcal{B}$ **faire** $\quad$ // Monte-Carlo avec un seul échantillon
        1. $y_{s,a} \leftarrow r + \gamma \texttt{maximize}(q(s', \cdot; \boldsymbol{\theta}_n))$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D})$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}||\mathcal{A}|}\sum_{(s,a) \in \mathcal{D} \times \mathcal{A}} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **jusqu'à ce que** $\delta < \varepsilon$ ou $n \geq N$
5. **retourner** $\boldsymbol{\theta}_n$
```

Le pseudocode ci-dessus introduit une routine générique $\texttt{maximize}$ qui représente toute méthode d'optimisation numérique cherchant une action qui maximise la fonction donnée. Cette approche est polyvalente et s'adapte à différents types d'espaces d'actions. Pour des espaces d'actions continus, on peut employer des méthodes standard d'optimisation non linéaire comme la descente de gradient ou L-BFGS (par exemple avec scipy.optimize.minimize). Pour de grands espaces d'actions discrets, on peut utiliser des solveurs de programmation en nombres entiers : programmation linéaire en nombres entiers si l'approximateur de la fonction Q est linéaire en les actions, ou solveurs de programmation non linéaire mixte en nombres entiers (MINLP) pour des fonctions Q non linéaires. Le choix du solveur dépend de la structure de notre approximateur de la fonction Q et des contraintes sur notre espace d'actions.

Bien que l'optimisation explicite fournisse des solutions exactes, elle devient coûteuse en calcul lorsqu'il faut calculer des cibles pour des millions de transitions dans un tampon de rejeu. Peut-on éviter de résoudre un problème d'optimisation à chaque décision? La réponse est l'amortissement.

## Approche par optimisation amortie

Les recherches d'actions coûteuses effectuées pendant l'entraînement peuvent-elles
superviser une application qui renvoie des actions à faible coût au déploiement?

Résoudre explicitement la maximisation pour chaque échantillon exige beaucoup de calcul. On peut « amortir » une partie de ce calcul en remplaçant l'optimisation explicite pour chaque échantillon par une application directe qui fournit immédiatement un maximiseur approché.
Pour les fonctions Q, rappelons que l'opérateur est donné par :

$$
(\Bellman q)(s,a) = r(s,a) + \gamma \int p(ds'|s,a)\max_{a' \in \mathcal{A}(s')} q(s', a')
$$

Si $q^*$ est la fonction de valeur d'action optimale, alors $v^*(s) = \max_a q^*(s,a)$, et on peut obtenir directement la politique optimale en calculant la règle de décision :

$$
\pi^\star(s) = \arg\max_{a \in \mathcal{A}(s)} q^\star(s,a)
$$

Puisque $q^*$ est un point fixe de $\Bellman$, on peut écrire :

$$
\begin{align*}
q^\star(s,a) &= (\Bellman q^*)(s,a) \\
&= r(s,a) + \gamma \int p(ds'|s,a) \max_{a' \in \mathcal{A}(s')} q^\star(s', a') \\
&= r(s,a) + \gamma \int p(ds'|s,a) q^\star(s', \pi^\star(s'))
\end{align*}
$$

Remarquons que $\pi^\star$ est réalisée par notre solveur numérique $\texttt{maximize}$ dans la procédure ci-dessus. Une stratégie pratique consisterait à recueillir ces maximiseurs à chaque étape et à s'en servir pour entraîner un approximateur de fonctions qui prédit directement ces solutions. En raison des contraintes de calcul, on peut vouloir calculer ces maximiseurs exacts seulement pour un sous-ensemble d'états, selon un certain budget de calcul, et utiliser la règle de décision ajustée pour généraliser aux autres états. On aboutit à la version amortie suivante :

```{prf:algorithm} Itération Q ajustée avec optimisation amortie
:label: fitted-q-iteration-amortized

**Entrée :** processus de décision markovien $(S, \mathcal{A}, P, R, \gamma)$, points de base $\mathcal{B}$, sous-ensemble pour l'optimisation exacte $\mathcal{B}_{\text{opt}} \subset \mathcal{B}$, approximateur de la fonction Q $q(s,a; \boldsymbol{\theta})$, approximateur de la politique $\pi_{\boldsymbol{w}}$, nombre maximal d'itérations $N$, tolérance $\varepsilon > 0$

**Sortie :** paramètres $\boldsymbol{\theta}$ de la fonction Q, $\boldsymbol{w}$ de la politique

1. Initialiser $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$
2. $n \leftarrow 0$
3. **répéter**
    1. $\mathcal{D}_q \leftarrow \emptyset$ $\quad$ // Jeu de données de régression de la fonction Q
    2. $\mathcal{D}_\pi \leftarrow \emptyset$ $\quad$ // Jeu de données de régression de la politique
    3. **pour** chaque $(s,a,r,s') \in \mathcal{B}$ **faire**
        1. **// Déterminer l'action de l'état suivant par optimisation exacte ou par approximation**
        2. **si** $s' \in \mathcal{B}_{\text{opt}}$ **alors**
            1. $a^*_{s'} \leftarrow \texttt{maximize}(q(s', \cdot; \boldsymbol{\theta}_n))$
            2. $\mathcal{D}_\pi \leftarrow \mathcal{D}_\pi \cup \{(s', a^*_{s'})\}$
        3. **sinon**
            1. $a^*_{s'} \leftarrow \pi_{\boldsymbol{w}_n}(s')$
        4. **// Calculer la cible de la fonction Q avec l'action choisie**
        5. $y_{s,a} \leftarrow r + \gamma q(s', a^*_{s'}; \boldsymbol{\theta}_n)$
        6. $\mathcal{D}_q \leftarrow \mathcal{D}_q \cup \{((s,a), y_{s,a})\}$
    4. **// Mettre à jour les deux approximateurs de fonctions**
    5. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_q)$
    6. $\boldsymbol{w}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_\pi)$
    7. **// Calculer les critères de convergence**
    8. $\delta_q \leftarrow \frac{1}{|\mathcal{D}_q|}\sum_{(s,a) \in \mathcal{D}_q} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    9. $\delta_\pi \leftarrow \frac{1}{|\mathcal{D}_\pi|}\sum_{(s,a^*) \in \mathcal{D}_\pi} \|a^* - \pi_{\boldsymbol{w}_{n+1}}(s)\|^2$
    10. $n \leftarrow n + 1$
4. **jusqu'à ce que** $\max(\delta_q, \delta_\pi) < \varepsilon$ ou $n \geq N$
5. **retourner** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$
```

Remarquons que la politique $\pi_{\boldsymbol{w}}$ est entraînée sur un jeu de données $\mathcal{D}_\pi$ contenant des actions optimales calculées par rapport à une fonction Q qui évolue. Plus précisément, à l'itération $n$, on recueille des paires $(s', a^*_{s'})$ où $a^*_{s'} = \arg\max_a q(s', a; \boldsymbol{\theta}_n)$. Cependant, après la mise à jour vers $\boldsymbol{\theta}_{n+1}$, ces actions peuvent ne plus être optimales par rapport à la nouvelle fonction Q.

Une façon naturelle de traiter cette péremption serait de ne conserver que les données d'optimisation les plus récentes. On pourrait modifier notre procédure pour garder une fenêtre glissante de $K$ itérations : à l'itération $n$, on n'utiliserait que les données des itérations $\max(0, n-K)$ à $n$. On réaliserait cela en ajoutant un horodatage à chaque entrée de $\mathcal{D}_\pi$ :

$$
\mathcal{D}_\pi^{(n)} = \{(s', a^*_{s'}, t) \mid t \in \{n-K,\ldots,n\}\}
$$

où $t$ indique l'itération à laquelle l'action optimale a été calculée. Lors de l'ajustement du réseau de politique, on n'utiliserait alors que les points de données datant d'au plus $K$ itérations :

$$
\boldsymbol{w}_{n+1} \leftarrow \texttt{fit}(\{(s', a^*_{s'}) \mid (s', a^*_{s'}, t) \in \mathcal{D}_\pi^{(n)}, n-K \leq t \leq n\})
$$

On introduit ainsi un compromis entre utiliser plus de données ($K$ plus grand) et utiliser des données plus récentes et plus exactes ($K$ plus petit). Le choix de $K$ dépendrait de la vitesse d'évolution de la fonction Q et du budget de calcul disponible pour calculer des actions optimales exactes.

La principale limite de cette approche, au-delà de la dérive hors distribution, est qu'elle exige de calculer des actions optimales exactes au moyen du solveur pour les états de $\mathcal{B}_{\text{opt}}$. Peut-on réduire ou éliminer ce coût de calcul? À mesure que la politique sélectionne de meilleures actions, on peut faire de l'autoamorçage à partir de ces choix de plus en plus bons. Amortir continuellement ces actions qui s'améliorent au fil du temps crée un cercle vertueux d'auto-amélioration vers la politique optimale. Ce processus d'autoamorçage exige toutefois une gestion soignée : avancer trop vite peut déstabiliser l'entraînement.


## Politiques paramétrées déterministes

Quelle classe de fonctions peut représenter une application directe des états vers
les actions pour la commande continue?

Dans cette section, on considère des politiques paramétrées déterministes de la forme $\pi_{\boldsymbol{w}}(s)$, qui renvoient directement une action pour un état donné. Cette approche diffère des politiques stochastiques, qui renvoient des distributions de probabilité sur les actions, et convient particulièrement aux problèmes de commande continue, où la politique optimale est souvent déterministe. Les méthodes de valeur Q ajustée s'étendent naturellement à l'apprentissage simultané de la fonction Q et d'une telle politique déterministe.

## Le problème de l'amortissement pour des actions continues

Quel objectif amène l'acteur paramétré à approcher l'action maximisante
du critique courant sur une distribution d'états?

Lorsque les actions sont continues, $a \in \mathbb{R}^d$, extraire une politique gloutonne d'une fonction Q devient coûteux en calcul. Considérons une tâche de commande d'un bras robotique où l'action est un vecteur de couples de dimension $d$. Pour agir de façon gloutonne étant donné la fonction Q $q(s,a; \boldsymbol{\theta})$, il faut résoudre :

$$
\pi(s) = \arg\max_{a \in \mathcal{A}} q(s, a; \boldsymbol{\theta}),
$$

où $\mathcal{A} \subset \mathbb{R}^d$ est un ensemble continu (souvent une boîte ou un polytope). Cela exige d'exécuter un algorithme d'optimisation à chaque pas de temps. Pour des fonctions Q représentées par des réseaux de neurones, cela signifie résoudre un programme non linéaire dont l'objectif fait intervenir des passes avant dans le réseau.

Une fois l'entraînement convergé, l'agent doit sélectionner ses actions en temps réel pendant le déploiement. Exécuter des méthodes de points intérieurs ou des optimiseurs à base de gradient à chaque décision crée une latence inacceptable, surtout en commande à haute fréquence, où les décisions surviennent à 100 Hz ou plus.

La solution consiste à **amortir** le coût de l'optimisation en apprenant un réseau de politique distinct $\pi_{\boldsymbol{w}}(s)$ qui renvoie directement des actions. Pendant l'entraînement, on optimise $\boldsymbol{w}$ de sorte que $\pi_{\boldsymbol{w}}(s) \approx \arg\max_a q(s,a; \boldsymbol{\theta})$ pour les états rencontrés. Au déploiement, la sélection d'une action se réduit à une seule passe avant dans le réseau de politique : $a = \pi_{\boldsymbol{w}}(s)$. Le coût de calcul de l'optimisation est payé pendant l'entraînement (où le temps est moins contraint) plutôt qu'à l'inférence.

On introduit ainsi une seconde approximation, en plus de celle de la fonction Q. On dispose maintenant de deux approximateurs de fonctions : un **critique** $q(s,a; \boldsymbol{\theta})$ qui estime les valeurs, et un **acteur** $\pi_{\boldsymbol{w}}(s)$ qui sélectionne les actions. Le critique est entraîné à l'aide de cibles de Bellman, comme dans l'itération Q ajustée standard. L'acteur est entraîné à maximiser le critique :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha \mathbb{E}_s \left[\nabla_{\boldsymbol{w}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta})\right],
$$

où l'espérance porte sur les états du jeu de données ou du tampon de rejeu. Cette montée de gradient pousse l'acteur vers les actions que le critique juge avantageuses. Par la dérivation des fonctions composées, ce gradient vaut $(\nabla_a q(s,a; \boldsymbol{\theta})|_{a=\pi_{\boldsymbol{w}}(s)}) \cdot (\nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s))$, ce qui se calcule efficacement par rétropropagation à travers la composition des deux réseaux.


## Itération Q ajustée neuronale pour actions continues (NFQCA)

Comment un algorithme Q ajusté par lots change-t-il lorsqu'un acteur fournit chaque
action maximisante continue dans la cible de Bellman?

NFQCA {cite:p}`Hafner2011` étend le gabarit NFQI du [chapitre précédent](fitted-q-iteration.md) aux espaces d'actions continus en remplaçant l'opérateur $\max_{a'} q(s',a'; \boldsymbol{\theta})$ de la cible de Bellman par une politique paramétrée $\pi_{\boldsymbol{w}}(s')$. L'itération Q ajustée devient ainsi une méthode acteur-critique : le critique $q(s,a; \boldsymbol{\theta})$ évalue les paires état-action par l'étape de régression standard, tandis que l'acteur $\pi_{\boldsymbol{w}}(s)$ fournit des actions en maximisant directement la fonction Q apprise.

L'algorithme conserve la structure à deux niveaux de NFQI : une boucle externe effectue une itération de valeur approchée en calculant des cibles de Bellman, et une boucle interne ajuste la fonction Q à ces cibles. NFQCA ajoute une troisième composante (l'amélioration de la politique) qui met à jour $\boldsymbol{w}$ pour maximiser la fonction Q sur des états tirés du jeu de données.

### Des actions discrètes aux actions continues

Rappelons, d'après le [chapitre sur l'itération Q ajustée](fitted-q-iteration.md), que NFQI calcule les cibles de Bellman avec le maximum strict :

$$
y_{s,a} = r + \gamma \max_{a' \in \mathcal{A}} q(s',a'; \boldsymbol{\theta}_n)
$$

Lorsque $\mathcal{A}$ est fini et petit, ce maximum se calcule par énumération. Lorsque $\mathcal{A}$ est continu ou de grande dimension, l'énumération est impraticable. NFQCA remplace le maximum par une politique paramétrée qui résout approximativement la maximisation :

$$
y_{s,a} = r + \gamma q(s', \pi_{\boldsymbol{w}_n}(s'); \boldsymbol{\theta}_n)
$$

La politique $\pi_{\boldsymbol{w}}(s)$ agit comme un **optimiseur amorti** : au lieu de résoudre $\arg\max_{a'} q(s',a')$ à partir de zéro en chaque état $s'$ lors du calcul des cibles, on entraîne un réseau de neurones à renvoyer directement des actions quasi optimales. Le terme « amorti » renvoie à l'étalement du coût de l'optimisation sur l'entraînement : on paie une fois pour apprendre $\pi_{\boldsymbol{w}}$, puis on la réutilise pour tous les calculs de cibles ultérieurs.

Pour entraîner la politique, on maximise la valeur Q espérée sous la distribution des états du jeu de données. Si l'on avait accès à la fonction Q optimale $q^*$, on résoudrait :

$$
\max_{\boldsymbol{w}} \mathbb{E}_{s \sim \hat{P}_{\mathcal{D}}}[q^*(s, \pi_{\boldsymbol{w}}(s))]
$$

où $\hat{P}_{\mathcal{D}}$ est la distribution empirique des états induite par le jeu de données hors ligne $\mathcal{D}$. En pratique, on utilise l'approximation courante de la fonction Q, $q(s,a; \boldsymbol{\theta}_{n+1})$, après son ajustement aux cibles les plus récentes. L'espérance est approchée par la moyenne empirique sur les états qui apparaissent dans $\mathcal{D}$ :

$$
\max_{\boldsymbol{w}} \frac{1}{|\mathcal{D}|} \sum_{(s,a,r,s') \in \mathcal{D}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}_{n+1})
$$

Cette étape d'amélioration de la politique s'exécute après la mise à jour de la fonction Q et se sert du critique fraîchement ajusté pour guider l'acteur vers des actions de plus grande valeur. L'ajustement de la fonction Q et l'amélioration de la politique emploient tous deux une optimisation à base de gradient sur leurs objectifs respectifs.

```{prf:algorithm} Itération Q ajustée neuronale pour actions continues (NFQCA)
:label: nfqca

**Entrée :** jeu de données $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$, fonction Q $q(s,a; \boldsymbol{\theta})$, politique $\pi_{\boldsymbol{w}}(s)$, facteur d'actualisation $\gamma$, taux d'apprentissage $\alpha_q, \alpha_\pi$, nombres de pas d'optimisation internes $K_q, K_\pi$

**Sortie :** paramètres de la fonction Q $\boldsymbol{\theta}$, paramètres de la politique $\boldsymbol{w}$

1. Initialiser $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$
2. $n \leftarrow 0$
3. **répéter** $\quad$ // **Boucle externe : évaluation et amélioration de la politique**
4. $\quad$ **// Construire le jeu de données de régression avec des cibles de Bellman fondées sur la politique**
5. $\quad$ $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
6. $\quad$ **pour** chaque $(s,a,r,s') \in \mathcal{D}$ **faire**
7. $\quad\quad$ $a' \leftarrow \pi_{\boldsymbol{w}_n}(s')$ $\quad$ // L'acteur sélectionne l'action (remplace le max)
8. $\quad\quad$ $y_{s,a} \leftarrow r + \gamma q(s', a'; \boldsymbol{\theta}_n)$ $\quad$ // Le critique évalue
9. $\quad\quad$ $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
10. $\quad$ **// Évaluation de la politique : ajuster la fonction Q aux cibles**
11. $\quad$ $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}_q(\mathcal{D}_n^{\text{fit}}, \boldsymbol{\theta}_n, K_q, \alpha_q)$
12. $\quad$ **// Amélioration de la politique : maximiser la fonction Q sur les états du jeu de données**
13. $\quad$ $\boldsymbol{w}_{n+1} \leftarrow \texttt{fit}_\pi(\mathcal{D}, \boldsymbol{w}_n, \boldsymbol{\theta}_{n+1}, K_\pi, \alpha_\pi)$
14. $\quad$ $n \leftarrow n + 1$
15. **jusqu'à** convergence ou $n \geq n_{\max}$
16. **retourner** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$

**où :**
- $\texttt{fit}_q(\mathcal{D}_n^{\text{fit}}, \boldsymbol{\theta}_n, K_q, \alpha_q)$ effectue $K_q$ pas de gradient qui minimisent $\frac{1}{|\mathcal{D}_n^{\text{fit}}|} \sum_{((s,a), y) \in \mathcal{D}_n^{\text{fit}}} (q(s,a; \boldsymbol{\theta}) - y)^2$, avec démarrage à chaud depuis $\boldsymbol{\theta}_n$
- $\texttt{fit}_\pi(\mathcal{D}, \boldsymbol{w}_n, \boldsymbol{\theta}_{n+1}, K_\pi, \alpha_\pi)$ effectue $K_\pi$ pas de gradient qui maximisent $\frac{1}{|\mathcal{D}|} \sum_{(s,a,r,s') \in \mathcal{D}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}_{n+1})$, avec démarrage à chaud depuis $\boldsymbol{w}_n$
```

La structure de l'algorithme reprend celle de NFQI ({prf:ref}`fitted-q-iteration-batch` dans le [chapitre sur l'itération Q ajustée](fitted-q-iteration.md)) avec deux extensions. D'abord, le calcul des cibles (lignes 7-8) remplace le maximum discret par un appel au réseau de politique $\pi_{\boldsymbol{w}_n}(s')$, ce qui rend l'opérateur de Bellman calculable pour des actions continues. Ensuite, après l'ajustement de la fonction Q (ligne 11), on ajoute une étape d'amélioration de la politique (ligne 13) qui met à jour $\boldsymbol{w}$ pour maximiser la fonction Q évaluée aux actions produites par la politique sur les états du jeu de données.

Les deux opérations `fit` emploient la descente de gradient avec démarrage à chaud, conformément au gabarit NFQI. La fonction Q minimise l'erreur de Bellman quadratique à l'aide de cibles calculées avec la politique courante. La politique maximise la fonction Q par montée de gradient sur la composition $q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}_{n+1})$, qui est différentiable de bout en bout lorsque les deux réseaux le sont. Le gradient par rapport à $\boldsymbol{w}$ est :

$$
\nabla_{\boldsymbol{w}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}) = \nabla_a q(s, a; \boldsymbol{\theta})\Big|_{a=\pi_{\boldsymbol{w}}(s)} \cdot \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s)
$$

calculé par la dérivation des fonctions composées (rétropropagation à travers l'acteur jusque dans le critique). Les bibliothèques modernes de différentiation automatique traitent cette composition automatiquement.

## Gradient de politique déterministe profond (DDPG)

Comment la construction acteur-critique par lots peut-elle recueillir de nouvelles transitions et mettre à jour
les deux réseaux en ligne?

On étend maintenant NFQCA au cadre en ligne avec des tampons de rejeu qui évoluent, de la même manière que DQN a étendu NFQI dans le [chapitre sur l'itération Q ajustée](fitted-q-iteration.md). Tout comme DQN laissait $\mathcal{B}_t$ et $\hat{P}_{\mathcal{B}_t}$ évoluer pendant l'apprentissage au lieu d'utiliser un jeu de données hors ligne fixe, DDPG {cite:p}`lillicrap2015continuous` recueille de nouvelles transitions pendant l'entraînement et les stocke dans un tampon de rejeu circulaire.

Comme DQN, DDPG utilise la structure aplatie de l'itération Q ajustée avec des réseaux cibles. Mais là où DQN maintient un seul réseau cible $\boldsymbol{\theta}_{\text{target}}$ pour la fonction Q, DDPG maintient **deux** réseaux cibles : un pour le critique, $\boldsymbol{\theta}_{\text{target}}$, et un pour l'acteur, $\boldsymbol{w}_{\text{target}}$. Les deux sont mis à jour périodiquement (tous les $K$ pas) pour marquer les frontières des itérations externes, selon la même transformation d'une structure imbriquée en une structure aplatie que celle présentée pour DQN.

Le réseau en ligne joue maintenant un triple rôle dans DDPG : (1) il porte les paramètres entraînés activement ($\boldsymbol{\theta}_t$ pour le critique, $\boldsymbol{w}_t$ pour l'acteur), (2) il est la politique utilisée pour recueillir de nouvelles données, et (3) il est la source du gradient pour l'amélioration de la politique. Les réseaux cibles ne servent qu'à une chose : calculer des cibles de Bellman stables.

### Exploration par bruit sur les actions

Puisque la politique $\pi_{\boldsymbol{w}}(s)$ est déterministe, l'exploration exige d'ajouter du bruit aux actions pendant la collecte des données :

$$
a = \pi_{\boldsymbol{w}_t}(s) + \eta_t
$$

où $\eta_t$ est un bruit d'exploration. L'article original de DDPG utilisait un processus d'Ornstein-Uhlenbeck (OU), qui engendre un bruit corrélé dans le temps au moyen de l'équation différentielle stochastique discrétisée :

$$
\eta_{t+1} = \eta_t + \theta(\mu - \eta_t)\Delta t + \sigma\sqrt{\Delta t}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,1)
$$

où $\mu$ est la moyenne à long terme (typiquement 0), $\theta$ règle l'intensité du rappel vers la moyenne, $\sigma$ met à l'échelle les fluctuations aléatoires, et $\Delta t$ est le pas de temps. Le terme $\theta(\mu - \eta_t)\Delta t$ agit comme un mouvement amorti dans un fluide visqueux : lorsque $\eta_t$ s'écarte de $\mu$, cette force le ramène en douceur, sans oscillation. Le terme aléatoire $\sigma\sqrt{\Delta t}\epsilon_t$ ajoute des perturbations et crée un bruit qui erre tout en étant doucement ramené vers $\mu$. Cette corrélation temporelle produit des trajectoires d'exploration plus lisses qu'un bruit gaussien indépendant.

Des travaux ultérieurs (dont TD3, présenté plus bas) ont toutefois constaté qu'un simple bruit gaussien non corrélé $\eta_t \sim \mathcal{N}(0, \sigma^2)$ fonctionne tout aussi bien et se règle plus facilement. Le mécanisme d'exploration est indépendant de la structure algorithmique de fond.

```{prf:algorithm} Gradient de politique déterministe profond (DDPG)
:label: ddpg

**Entrée :** processus de décision markovien $(S, \mathcal{A}, P, R, \gamma)$, réseau Q $q(s,a; \boldsymbol{\theta})$, réseau de politique $\pi_{\boldsymbol{w}}(s)$, taux d'apprentissage $\alpha_q, \alpha_\pi$, fréquence de mise à jour des cibles $K$, capacité du tampon de rejeu $B$, taille des mini-lots $b$, bruit d'exploration $\eta$

**Sortie :** paramètres de la fonction Q $\boldsymbol{\theta}$, paramètres de la politique $\boldsymbol{w}$

1. Initialiser $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$ aléatoirement
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$, $\boldsymbol{w}_{\text{target}} \leftarrow \boldsymbol{w}_0$
3. Initialiser le tampon de rejeu $\mathcal{B}$ de capacité $B$
4. Initialiser le processus de bruit d'exploration $\eta$ (par exemple un processus OU ou un bruit gaussien)
5. $t \leftarrow 0$
6. **tant que** l'entraînement se poursuit **faire**
    1. Observer l'état courant $s$
    2. **// Utiliser le réseau en ligne pour recueillir des données avec bruit d'exploration**
    3. Sélectionner l'action : $a \leftarrow \pi_{\boldsymbol{w}_t}(s) + \eta_t$
    4. Exécuter $a$, observer la récompense $r$ et l'état suivant $s'$
    5. Stocker $(s,a,r,s')$ dans $\mathcal{B}$, en remplaçant la plus ancienne transition si le tampon est plein
    6. Tirer de $\mathcal{B}$ un mini-lot de $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$
    7. **pour** chaque transition tirée $(s_i,a_i,r_i,s_i')$ **faire**
        1. $y_i \leftarrow r_i + \gamma q(s'_i, \pi_{\boldsymbol{w}_{\text{target}}}(s'_i); \boldsymbol{\theta}_{\text{target}})$ $\quad$ // L'acteur cible sélectionne, le critique cible évalue
    8. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b (q(s_i,a_i;\boldsymbol{\theta}_t) - y_i)^2$
    9. $\boldsymbol{w}_{t+1} \leftarrow \boldsymbol{w}_t + \alpha_\pi \frac{1}{b}\sum_{i=1}^b \nabla_a q(s_i,a;\boldsymbol{\theta}_{t+1})\Big|_{a=\pi_{\boldsymbol{w}_t}(s_i)} \cdot \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}_t}(s_i)$
    10. **si** $t \bmod K = 0$ **alors**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
        2. $\boldsymbol{w}_{\text{target}} \leftarrow \boldsymbol{w}_t$
    11. $t \leftarrow t + 1$
7. **retourner** $\boldsymbol{\theta}_t$, $\boldsymbol{w}_t$
```

La structure de l'algorithme est parallèle à celle de DQN ({prf:ref}`dqn` dans le [chapitre sur l'itération Q ajustée](fitted-q-iteration.md)), avec les extensions de NFQCA pour les actions continues. Les lignes 1-5 initialisent les deux réseaux et leurs cibles, selon le même schéma que DQN mais avec un réseau acteur supplémentaire. La ligne 3 utilise l'acteur en ligne avec bruit d'exploration pour la collecte des données, en remplacement de la sélection $\varepsilon$-gloutonne de DQN. La ligne 7 calcule les cibles avec les deux réseaux cibles : l'acteur cible $\pi_{\boldsymbol{w}_{\text{target}}}(s'_i)$ sélectionne l'action suivante, et le critique cible $q(\cdot; \boldsymbol{\theta}_{\text{target}})$ l'évalue. Cela remplace l'opérateur $\max_{a'}$ de DQN. Les lignes 8-9 mettent à jour les deux réseaux : le critique par minimisation de l'erreur de différence temporelle, l'acteur par gradient de politique à travers le critique mis à jour. La ligne 10 effectue des mises à jour strictes périodiques tous les $K$ pas, qui marquent les frontières des itérations externes.

Le gradient de politique de la ligne 9 utilise la dérivation des fonctions composées pour rétropropager à travers la composition acteur-critique :

$$
\nabla_{\boldsymbol{w}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}) = \nabla_a q(s,a; \boldsymbol{\theta})\Big|_{a=\pi_{\boldsymbol{w}}(s)} \cdot \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s)
$$

C'est le même gradient que celui de NFQCA, mais il est maintenant calculé sur des mini-lots tirés d'un tampon de rejeu qui évolue plutôt que d'un jeu de données hors ligne fixe. Le gradient du critique $\nabla_a q(s,a; \boldsymbol{\theta})$ à l'action produite par la politique donne la direction de plus forte montée dans l'espace des valeurs Q, pondérée par la sensibilité de la sortie de la politique à ses paramètres, $\nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s)$.

## Gradient de politique déterministe profond jumelé et retardé (TD3)

Quelles modifications du critique et des cibles empêchent un acteur déterministe
d'exploiter des erreurs étroites d'une fonction Q apprise?

DDPG hérite du biais de surestimation que DQN doit à l'opérateur max dans les cibles de Bellman. TD3 {cite:p}`fujimoto2018addressing` s'y attaque au moyen de trois modifications du gabarit DDPG, selon des principes semblables à ceux de Double DQN, mais adaptés aux actions continues et avec une approche plus prudente.

### Réseaux Q jumeaux et opérateur minimum

Rappelons, d'après le [chapitre sur les mises à jour de Bellman échantillonnées](monte-carlo-bellman-estimation.md), que la surestimation survient lorsqu'on utilise la même estimation bruitée à la fois pour choisir l'action qui semble la meilleure et pour évaluer cette action. Le Q-learning double rompt ce couplage en maintenant deux estimateurs indépendants, avec des termes de bruit $\varepsilon^{(1)}_a$ et $\varepsilon^{(2)}_a$ :

$$
a^\star = \arg\max_{a} \left\{r(s,a) + \gamma \hat{\mu}^{(1)}_N(s,a)\right\}, \quad Y = r(s,a^\star) + \gamma \hat{\mu}^{(2)}_N(s,a^\star).
$$

Lorsque $\varepsilon^{(1)}$ et $\varepsilon^{(2)}$ sont indépendants, la propriété de la tour de l'espérance conditionnelle donne $\mathbb{E}[\varepsilon^{(2)}_{a^\star} \mid a^\star] = \mathbb{E}[\varepsilon^{(2)}_{a^\star}] = 0$, car $a^\star$ (déterminée par $\varepsilon^{(1)}$) est indépendante de $\varepsilon^{(2)}$. Cela élimine le **biais d'évaluation** : on n'utilise plus le même bruit positif qui a fait choisir une action pour gonfler aussi sa valeur. En conditionnant sur l'action choisie, puis en prenant l'espérance sur le bruit d'évaluation indépendant, le biais du terme d'évaluation disparaît.

Double DQN ({prf:ref}`double-dqn`) applique ce principe dans le cadre des actions discrètes en utilisant le réseau en ligne $\boldsymbol{\theta}_t$ pour la sélection ($a^*_i \leftarrow \arg\max_{a'} q(s_i',a'; \boldsymbol{\theta}_t)$) et le réseau cible $\boldsymbol{\theta}_{\text{target}}$ pour l'évaluation ($y_i \leftarrow r_i + \gamma q(s_i',a^*_i; \boldsymbol{\theta}_{\text{target}})$). Comme ces réseaux subissent des bruits d'entraînement différents, leurs erreurs sont approximativement indépendantes, ce qui réalise la condition d'indépendance nécessaire pour éliminer le biais d'évaluation. Le **biais de sélection** demeure toutefois : l'argmax choisit encore des actions qui ont reçu un bruit positif dans le réseau de sélection, de sorte que $\mathbb{E}_{\varepsilon^{(1)}}[\mu(s,a^\star)] \ge \max_a \mu(s,a)$.

TD3 adopte une approche plus prudente. Au lieu de découpler la sélection de l'évaluation, TD3 maintient des **réseaux Q jumeaux** $q^A(s,a; \boldsymbol{\theta}^A)$ et $q^B(s,a; \boldsymbol{\theta}^B)$, entraînés sur les mêmes données avec des initialisations aléatoires différentes. Pour calculer les cibles, TD3 utilise la politique cible $\pi_{\boldsymbol{w}_{\text{target}}}(s')$ pour sélectionner les actions (aucune maximisation sur un ensemble discret), puis prend le minimum des évaluations des deux réseaux Q :

$$
y_i = r_i + \gamma \min\left(q^A(s'_i, \tilde{a}_i; \boldsymbol{\theta}^A_{\text{target}}), q^B(s'_i, \tilde{a}_i; \boldsymbol{\theta}^B_{\text{target}})\right)
$$

où $\tilde{a}_i = \pi_{\boldsymbol{w}_{\text{target}}}(s'_i)$. Cette opération de minimum fournit une estimation pessimiste : si les deux réseaux Q ont des erreurs indépendantes, $q^A(s',a) = q^*(s',a) + \varepsilon^A$ et $q^B(s',a) = q^*(s',a) + \varepsilon^B$, alors $\mathbb{E}[\min(q^A, q^B)] \le q^*(s',a)$, ce qui produit une sous-estimation systématique plutôt qu'une surestimation.

Le lien avec le cadre de l'indépendance conditionnelle est subtil mais important. Alors que Double DQN se sert de l'indépendance pour éliminer le biais en espérance (un réseau sélectionne, un autre évalue), TD3 s'en sert pour construire une borne inférieure délibérée. Les deux approches reposent sur le maintien de deux fonctions Q dont les erreurs sont partiellement décorrélées, ce qu'on obtient par des initialisations différentes et des gradients stochastiques pendant l'entraînement, mais elles agrègent ces fonctions différemment. Le découplage de Double DQN vise une estimation sans biais en rompant la corrélation entre le bruit de sélection et le bruit d'évaluation. L'opération de minimum de TD3 vise une estimation robuste en adoptant le point de vue le plus pessimiste lorsque les deux réseaux sont en désaccord.

Ce compromis entre biais et robustesse est délibéré. Dans les méthodes acteur-critique, le gradient de politique pousse vers les actions de grande valeur Q. La surestimation est particulièrement nuisible, car elle peut amener la politique à exploiter des régions de valeur élevée erronées. La sous-estimation est généralement plus sûre : la politique peut ignorer quelques bonnes actions, mais elle ne sera pas entraînée à poursuivre des actions qui ne semblent avantageuses qu'à cause de l'erreur d'approximation. L'opération de minimum met en œuvre un principe de « confiance au pessimiste » qui complète l'objectif d'optimisation de la politique.

TD3 introduit aussi deux modifications supplémentaires, au-delà du Q-learning double écrêté. D'abord, le lissage de la politique cible ajoute un bruit écrêté aux actions de la politique cible lors du calcul des cibles : $\tilde{a} = \pi_{\boldsymbol{w}_{\text{target}}}(s') + \text{clip}(\varepsilon, -c, c)$. Cette régularisation empêche la politique d'exploiter des pics étroits de l'erreur d'approximation de la fonction Q en moyennant sur les actions voisines. Ensuite, les mises à jour retardées de la politique changent la fréquence de mise à jour de l'acteur : l'acteur est mis à jour tous les $d$ pas plutôt qu'à chaque pas. Cela réduit l'erreur de chaque mise à jour en laissant les critiques converger avant que l'acteur ne s'y adapte.

TD3 remplace également les mises à jour strictes des cibles de DDPG par des mises à jour par **moyenne mobile exponentielle (EMA)**, selon le schéma de mise à jour en douceur de {prf:ref}`nfqi-flattened-ema` dans le [chapitre sur l'itération Q ajustée](fitted-q-iteration.md). Au lieu de copier $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$ tous les $K$ pas, la moyenne mobile exponentielle suit en douceur le réseau en ligne : $\boldsymbol{\theta}_{\text{target}} \leftarrow \tau \boldsymbol{\theta}_t + (1-\tau)\boldsymbol{\theta}_{\text{target}}$ à chaque mise à jour. Pour un petit $\tau \in [0.001, 0.01]$, la cible accuse un retard d'environ $1/\tau$ pas sur le réseau en ligne, ce qui donne une dynamique d'apprentissage plus lisse.

```{prf:algorithm} Gradient de politique déterministe profond jumelé et retardé (TD3)
:label: td3

**Entrée :** processus de décision markovien $(S, \mathcal{A}, P, R, \gamma)$, réseaux Q jumeaux $q^A(s,a; \boldsymbol{\theta}^A)$, $q^B(s,a; \boldsymbol{\theta}^B)$, réseau de politique $\pi_{\boldsymbol{w}}(s)$, taux d'apprentissage $\alpha_q, \alpha_\pi$, capacité du tampon de rejeu $B$, taille des mini-lots $b$, retard de la politique $d$, taux de la moyenne mobile exponentielle $\tau$, bruit des cibles $\sigma$, seuil d'écrêtage du bruit $c$, bruit d'exploration $\sigma_{\text{explore}}$

**Sortie :** paramètres des fonctions Q jumelles $\boldsymbol{\theta}^A, \boldsymbol{\theta}^B$, paramètres de la politique $\boldsymbol{w}$

1. Initialiser $\boldsymbol{\theta}^A_0$, $\boldsymbol{\theta}^B_0$, $\boldsymbol{w}_0$ aléatoirement
2. $\boldsymbol{\theta}^A_{\text{target}} \leftarrow \boldsymbol{\theta}^A_0$, $\boldsymbol{\theta}^B_{\text{target}} \leftarrow \boldsymbol{\theta}^B_0$, $\boldsymbol{w}_{\text{target}} \leftarrow \boldsymbol{w}_0$
3. Initialiser le tampon de rejeu $\mathcal{B}$ de capacité $B$
4. $t \leftarrow 0$
5. **tant que** l'entraînement se poursuit **faire**
    1. Observer l'état courant $s$
    2. **// Collecte des données avec bruit d'exploration**
    3. Sélectionner l'action : $a \leftarrow \pi_{\boldsymbol{w}_t}(s) + \varepsilon$, où $\varepsilon \sim \mathcal{N}(0, \sigma_{\text{explore}}^2)$
    4. Exécuter $a$, observer la récompense $r$ et l'état suivant $s'$
    5. Stocker $(s,a,r,s')$ dans $\mathcal{B}$, en remplaçant la plus ancienne transition si le tampon est plein
    6. Tirer de $\mathcal{B}$ un mini-lot de $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$
    7. **// Calculer les cibles par Q-learning double écrêté**
    8. **pour** chaque transition tirée $(s_i,a_i,r_i,s_i')$ **faire**
        1. $\tilde{a}_i \leftarrow \pi_{\boldsymbol{w}_{\text{target}}}(s'_i) + \text{clip}(\varepsilon_i, -c, c)$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ $\quad$ // Lissage de la politique cible
        2. $y_i \leftarrow r_i + \gamma \color{blue}{\min\big(q^A(s'_i, \tilde{a}_i; \boldsymbol{\theta}^A_{\text{target}}), q^B(s'_i, \tilde{a}_i; \boldsymbol{\theta}^B_{\text{target}})\big)}$ $\quad$ // Minimum des cibles jumelles
    9. **// Mettre à jour les deux réseaux Q vers les mêmes cibles**
    10. $\boldsymbol{\theta}^A_{t+1} \leftarrow \boldsymbol{\theta}^A_t - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b(q^A(s_i,a_i;\boldsymbol{\theta}^A_t) - y_i)^2$
    11. $\boldsymbol{\theta}^B_{t+1} \leftarrow \boldsymbol{\theta}^B_t - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b(q^B(s_i,a_i;\boldsymbol{\theta}^B_t) - y_i)^2$
    12. **// Mise à jour retardée de la politique et mises à jour des réseaux cibles**
    13. **si** $t \bmod d = 0$ **alors**
        1. $\boldsymbol{w}_{t+1} \leftarrow \boldsymbol{w}_t + \alpha_\pi \frac{1}{b}\sum_{i=1}^b \nabla_a q^A(s_i,a;\boldsymbol{\theta}^A_{t+1})\Big|_{a=\pi_{\boldsymbol{w}_t}(s_i)} \cdot \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}_t}(s_i)$
        2. $\boldsymbol{\theta}^A_{\text{target}} \leftarrow \tau\boldsymbol{\theta}^A_{t+1} + (1-\tau)\boldsymbol{\theta}^A_{\text{target}}$ (mise à jour par moyenne mobile exponentielle)
        3. $\boldsymbol{\theta}^B_{\text{target}} \leftarrow \tau\boldsymbol{\theta}^B_{t+1} + (1-\tau)\boldsymbol{\theta}^B_{\text{target}}$
        4. $\boldsymbol{w}_{\text{target}} \leftarrow \tau\boldsymbol{w}_{t+1} + (1-\tau)\boldsymbol{w}_{\text{target}}$
    14. $t \leftarrow t + 1$
6. **retourner** $\boldsymbol{\theta}^A_t$, $\boldsymbol{\theta}^B_t$, $\boldsymbol{w}_t$
```

La structure de l'algorithme est parallèle à celle de Double DQN, mais avec des actions continues. Les lignes 8.1-8.2 mettent en œuvre le Q-learning double écrêté : le lissage ajoute du bruit aux actions cibles (ce qui empêche l'exploitation d'artefacts de la fonction Q), et l'opération de minimum (en bleu) fournit des estimations de valeur pessimistes. Les deux critiques sont mis à jour vers la même cible partagée (lignes 10-11), mais leurs initialisations différentes et le bruit de leurs gradients stochastiques gardent leurs erreurs partiellement décorrélées, selon le même principe que l'hypothèse d'indépendance de Double DQN. La ligne 13 limite les mises à jour de la politique à une tous les $d$ pas (typiquement $d=2$), et les lignes 13.2-13.4 emploient des mises à jour par moyenne mobile exponentielle selon {prf:ref}`nfqi-flattened-ema`.

TD3 simplifie l'exploration en remplaçant le processus d'Ornstein-Uhlenbeck de DDPG par un bruit gaussien non corrélé $\varepsilon \sim \mathcal{N}(0, \sigma_{\text{explore}}^2)$ (ligne 5.3). On évite ainsi de régler plusieurs paramètres du processus OU tout en obtenant une exploration tout aussi efficace.

## Synthèse et perspectives

Les actions continues transforment la maximisation contenue dans chaque cible
Q ajustée en un problème d'optimisation non linéaire. NFQCA apprend un acteur
pour amortir cette recherche; DDPG transpose la même structure acteur-critique
en ligne; TD3 ajoute des critiques jumeaux, le lissage des cibles et des mises
à jour retardées de l'acteur pour réduire l'exploitation de l'erreur du
critique.

Les mises à jour de l'acteur présentées ci-dessus dérivent à travers un critique
appris. Quels estimateurs de gradient restent disponibles lorsque les actions ou
les trajectoires elles-mêmes sont aléatoires?
[L'estimation de gradient pour des objectifs stochastiques](gradient-estimation.md)
distingue les estimateurs par fonction de score et par reparamétrisation avant
qu'ils servent à l'apprentissage de politiques.

## Autoévaluation

:::{exercise} Qu'est-ce qui est amorti?
:label: ex-amortization-check-1

En commande à actions continues, quel calcul répété un acteur appris remplace-t-il?
:::

:::{solution} ex-amortization-check-1
:class: dropdown

L'acteur remplace la résolution répétée de $\arg\max_a q(s,a)$ en chaque état par une seule application directe apprise de l'état vers l'action.
:::
