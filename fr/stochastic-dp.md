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
# Programmation dynamique stochastique et processus de décision markoviens

La programmation dynamique à horizon fini choisit une action à partir de
chaque état et de chaque instant, puis suit un unique successeur
déterministe. Comment la récurrence doit-elle changer lorsqu'une action induit
une distribution de successeurs et que la règle de décision elle-même peut
être aléatoire ou dépendre de l'historique? La programmation dynamique
stochastique répond en moyennant les valeurs de continuation selon un noyau
de transition.

(decision-rules-and-policies)=
## Règles de décision et politiques

Quelles combinaisons d'état, d'historique et de tirage aléatoire une règle de
décision peut-elle utiliser pour choisir une action?

Avant d'aborder les systèmes stochastiques, il faut fixer la terminologie des différents types de stratégies qu'un décideur peut employer. Dans le cadre déterministe, nous avons implicitement utilisé des correcteurs à rétroaction de la forme $u(\mathbf{x}, t)$. Dans le cadre stochastique, il faut préciser davantage quelles informations les politiques peuvent utiliser et comment elles choisissent les actions.

Une **règle de décision** prescrit le choix de l'action dans chaque état à une époque de décision donnée. Ces règles peuvent être plus ou moins complexes selon deux critères principaux :

1. **Dépendance à l'historique** : markovienne ou dépendante de l'historique
2. **Mode de choix de l'action** : déterministe ou aléatoire

Les **règles de décision markoviennes** ne dépendent que de l'état courant, alors que les **règles dépendantes de l'historique** tiennent compte de toute la suite des états et des actions passés. Formellement, un historique $h_t$ au temps $t$ est :

$$h_t = (s_1, a_1, \ldots, s_{t-1}, a_{t-1}, s_t)$$

L'ensemble de tous les historiques possibles au temps $t$, noté $H_t$, croît exponentiellement avec $t$ :
- $H_1 = \mathcal{S}$ (seulement l'état initial)
- $H_2 = \mathcal{S} \times \mathcal{A} \times \mathcal{S}$
- $H_t = \mathcal{S} \times (\mathcal{A} \times \mathcal{S})^{t-1}$

Les **règles déterministes** choisissent une action avec certitude, tandis que les **règles aléatoires** précisent une distribution de probabilité sur l'espace des actions.

Ces classifications donnent quatre types de règles de décision :
1. **Markovienne déterministe (MD)** : $\pi_t: \mathcal{S} \rightarrow \mathcal{A}_s$
2. **Markovienne aléatoire (MR)** : $\pi_t: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A}_s)$
3. **Dépendante de l'historique et déterministe (HD)** : $\pi_t: H_t \rightarrow \mathcal{A}_s$
4. **Dépendante de l'historique et aléatoire (HR)** : $\pi_t: H_t \rightarrow \mathcal{P}(\mathcal{A}_s)$

où $\mathcal{P}(\mathcal{A}_s)$ désigne l'ensemble des distributions de probabilité sur $\mathcal{A}_s$.

Une **politique** $\boldsymbol{\pi}$ est une suite de règles de décision, une pour chaque époque de décision :

$$\boldsymbol{\pi} = (\pi_1, \pi_2, ..., \pi_{N-1})$$

L'ensemble de toutes les politiques de la classe $K$ (où $K \in \{HR, HD, MR, MD\}$) est noté $\Pi^K$. Ces classes de politiques forment une hiérarchie :

$$\Pi^{MD} \subset \Pi^{MR} \subset \Pi^{HR}, \quad \Pi^{MD} \subset \Pi^{HD} \subset \Pi^{HR}$$

Le plus grand ensemble, $\Pi^{HR}$, contient toutes les politiques possibles. Nous posons la question suivante : à quelles conditions peut-on se restreindre à l'ensemble beaucoup plus simple $\Pi^{MD}$ sans perdre l'optimalité?

```{admonition} Notation : règles et politiques
:class: tip
- **Règle de décision (noyau).** Une application qui associe à l'information une distribution sur les actions :
  - markovienne, déterministe :  $\pi_t:\mathcal{S}\to\mathcal{A}_s$
  - markovienne, aléatoire :     $\pi_t(\cdot\mid s)\in\Delta(\mathcal{A}_s)$
  - dépendante de l'historique :       $\pi_t(\cdot\mid h_t)\in\Delta(\mathcal{A}_{s_t})$
- **Politique (suite).** $\boldsymbol{\pi}=(\pi_1,\pi_2,\ldots)$.
- **Politique stationnaire.** $\boldsymbol{\pi}=\mathrm{const}(\pi)$ avec $\pi_t\equiv\pi \ \forall t$.  
  Par convention, on identifie $\pi$ à sa politique stationnaire $\mathrm{const}(\pi)$ lorsqu'il n'y a pas de confusion possible.
```

## Dynamique des systèmes stochastiques

Comment un noyau de transition remplace-t-il l'unique successeur déterministe
du modèle à horizon fini?

Dans le cadre stochastique, l'évolution du système prend la forme :

$$ \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t) $$

Ici, $\mathbf{w}_t$ représente une perturbation aléatoire, ou un terme de bruit, au temps $t$, qui traduit l'incertitude inhérente au comportement du système. La fonction de coût d'étape peut elle aussi intégrer des influences stochastiques :

$$ c_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t) $$

Dans ce contexte, l'objectif ne consiste plus à minimiser un coût déterministe, mais le coût total espéré :

$$ \mathbb{E}\left[c_\mathrm{T}(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)\right] $$

où l'espérance est prise par rapport aux distributions des variables aléatoires $\mathbf{w}_t$. Le principe d'optimalité tient toujours dans le cas stochastique, mais l'équation d'optimalité de Bellman fait désormais intervenir une espérance :

$$ J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \mathbb{E}_{\mathbf{w}_k}\left[c_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k))\right] $$

En pratique, on calcule souvent cette espérance en discrétisant la distribution de $\mathbf{w}_k$ lorsque l'ensemble des perturbations possibles est très grand, voire continu. Supposons que l'on approche la distribution par $K$ valeurs discrètes $\mathbf{w}_k^i$, chacune de probabilité $p_k^i$. L'équation de Bellman devient alors :

$$ J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \sum_{i=1}^K p_k^i \left(c_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k^i) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k^i))\right) $$

## Équations d'optimalité dans le cadre stochastique

Comment la valeur de continuation déterministe est-elle remplacée par une
espérance sur tous les états suivants possibles?

Face à des systèmes stochastiques, une question centrale se pose : quelles informations notre politique de commande doit-elle utiliser? Dans le cas le plus général, une politique peut utiliser tout l'historique des observations et des actions. Comme nous le verrons, la structure markovienne de nos problèmes permet toutefois des simplifications considérables.

Soit $h_t = (s_1, a_1, s_2, a_2, \ldots, s_{t-1}, a_{t-1}, s_t)$ l'historique complet jusqu'au temps $t$. Dans le cadre stochastique, les équations d'optimalité fondées sur l'historique deviennent :

$$
u_t(h_t) = \sup_{a\in A_{s_t}}\left\{ r_t(s_t,a) + \sum_{j\in S} p_t(j\mid s_t,a)\, u_{t+1}(h_t,a,j) \right\},\quad u_N(h_N)=r_N(s_N)
$$

où l'on utilise désormais explicitement les probabilités de transition $p_t(j|s_t,a)$ plutôt qu'une fonction de dynamique déterministe.

````{prf:theorem} Principe d'optimalité pour les systèmes stochastiques
:label: stoch-principle-opt

Soit $u_t^*$ le rendement espéré optimal à partir de l'époque $t$. Alors :

**a.** $u_t^*$ satisfait les équations d'optimalité :

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\}$$

avec la condition aux limites $u_N^*(h_N) = r_N(s_N)$.

**b.** Toute politique $\pi^*$ qui choisit, pour chaque historique, des actions atteignant le supremum (ou le maximum) dans l'équation ci-dessus est optimale.
````

**Intuition :** Ce résultat formalise le principe d'optimalité de Bellman : « Une politique optimale a la propriété suivante : quels que soient l'état initial et la décision initiale, les décisions restantes doivent constituer une politique optimale relativement à l'état qui résulte de la première décision. » La structure récursive signifie que des décisions locales optimales (choisir la meilleure action à chaque étape) mènent à l'optimalité globale, même lorsque l'incertitude est représentée par les probabilités de transition.

Une simplification apparaît lorsqu'on examine de plus près ces équations fondées sur l'historique. La propriété de Markov de la dynamique et des récompenses de notre système implique que le rendement optimal ne dépend en fait de l'historique qu'à travers l'état courant :

````{prf:proposition} Suffisance de l'état pour les processus de décision markoviens stochastiques
:label: stoch-state-suff

Dans les processus de décision markoviens stochastiques à horizon fini, dont la dynamique et les récompenses sont markoviennes, le rendement optimal $u_t^*(h_t)$ ne dépend de l'historique qu'à travers l'état courant $s_t$. On peut donc écrire $u_t^*(h_t) = v_t^*(s_t)$ pour une certaine fonction $v_t^*$ qui ne dépend que de l'état et du temps.
````

````{prf:proof}
D'après le théorème 4.4.2 de {cite:t}`Puterman1994`. On procède par induction arrière.

**Cas de base :** Au temps terminal $N$, on a $u_N^*(h_N) = r_N(s_N)$ par la condition aux limites. Comme la récompense terminale ne dépend que de l'état final $s_N$, et non de la façon dont on y est parvenu, $u_N^*(h_N) = u_N^*(s_N)$.

**Étape d'induction :** Supposons que $u_{t+1}^*(h_{t+1})$ ne dépende de $h_{t+1}$ qu'à travers $s_{t+1}$ pour tous les temps $t+1, \ldots, N$. L'équation d'optimalité donne alors :

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\}$$

Par l'hypothèse d'induction, $u_{t+1}^*(h_t, a, j)$ ne dépend que de l'état suivant $j$; donc :

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(j) \right\}$$

Comme l'expression entre accolades ne dépend de $h_t$ qu'à travers l'état courant $s_t$ (les récompenses et les probabilités de transition sont markoviennes), on conclut que $u_t^*(h_t) = u_t^*(s_t)$.
````

**Intuition :** La propriété de Markov signifie que l'état courant contient toute l'information nécessaire pour prédire l'évolution future. Le passé n'apporte aucune valeur supplémentaire à la prise de décision. Ce résultat permet de travailler avec des fonctions de valeur $v_t^*(s)$ indexées seulement par l'état et le temps, ce qui simplifie considérablement la théorie comme le calcul.

Ce résultat de suffisance de l'état, joint au fait que le tirage aléatoire n'aide jamais lorsqu'on maximise des rendements espérés, mène à une simplification considérable de l'espace des politiques :

````{prf:theorem} Réduction des politiques pour les processus de décision markoviens stochastiques
:label: stoch-policy-reduction

Pour les processus de décision markoviens stochastiques à horizon fini, dont les ensembles d'états et d'actions sont finis :

$$
\sup_{\pi \in \Pi^{\mathrm{HR}}} v_\pi(s,t) = \max_{\pi \in \Pi^{\mathrm{MD}}} v_\pi(s,t)
$$

Autrement dit, il existe une politique optimale à la fois déterministe et markovienne.
````

````{prf:proof}
Esquisse d'après le lemme 4.3.1 et le théorème 4.4.2 de {cite:t}`Puterman1994`. D'abord, le lemme 4.3.1 montre que, pour toute fonction $w$ et toute distribution $q$ sur les actions, $\sup_a w(a) \ge \sum_a q(a) w(a)$. Le tirage aléatoire ne peut donc pas améliorer la valeur espérée par rapport au choix d'une seule action maximisante. Ensuite, par la suffisance de l'état (proposition {ref}`stoch-state-suff` et {cite:t}`Puterman1994`, théorème 4.4.2(a)), le rendement optimal ne dépend de l'historique qu'à travers $(s_t,t)$. Par conséquent, choisir en chaque $(s_t,t)$ une action qui atteint le maximum donne une règle de décision markovienne déterministe, optimale dès que le maximum est atteint. Si seul un supremum existe, des sélecteurs $\varepsilon$-optimaux existent : il suffit de choisir des actions à moins de $\varepsilon$ du supremum (voir {cite:t}`Puterman1994`, théorème 4.3.4).
````

**Intuition :** Même dans les systèmes stochastiques, le tirage aléatoire dans la politique n'aide pas lorsqu'on maximise des rendements espérés : il faut toujours choisir l'action dont la valeur espérée est la plus élevée. Avec la suffisance de l'état, cela signifie que de simples applications des états vers les actions sont optimales.

Ces résultats justifient de se concentrer sur les politiques markoviennes déterministes et mènent à l'algorithme de récurrence arrière pour les systèmes stochastiques :

````{prf:algorithm} Récurrence arrière pour la programmation dynamique stochastique
:label: backward-recursion-stochastic

**Entrée :** fonction de coût terminal $c_\mathrm{T}(\cdot)$, fonctions de coût d'étape $c_t(\cdot, \cdot, \cdot)$, dynamique du système $\mathbf{f}_t(\cdot, \cdot, \cdot)$, horizon $\mathrm{T}$, distributions des perturbations

**Sortie :** fonctions de valeur optimales $J_t^\star(\cdot)$ et politiques de commande optimales $\pi_t^\star(\cdot)$ pour $t = 1, \ldots, T$

1. Initialiser $J_T^\star(\mathbf{x}) = c_\mathrm{T}(\mathbf{x})$ pour tout $\mathbf{x}$ de l'espace d'états
2. Pour $t = T-1, T-2, \ldots, 1$ :
   1. Pour chaque état $\mathbf{x}$ de l'espace d'états :
      1. Calculer $J_t^\star(\mathbf{x}) = \min_{\mathbf{u}} \mathbb{E}_{\mathbf{w}_t}\left[c_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t))\right]$
      2. Calculer $\pi_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u}} \mathbb{E}_{\mathbf{w}_t}\left[c_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t))\right]$
   2. Fin pour
3. Fin pour
4. Renvoyer $J_t^\star(\cdot)$, $\pi_t^\star(\cdot)$ pour $t = 1, \ldots, T$
````

Si la programmation dynamique stochastique fournit un cadre pour traiter l'incertitude, elle rend la malédiction de la dimension encore plus difficile à maîtriser en pratique. Il faut discrétiser à la fois l'espace d'états et l'espace des perturbations. Cela peut entraîner une explosion combinatoire du nombre de scénarios à évaluer à chaque étape.

Cependant, de même que nous avons surmonté les difficultés des espaces d'états continus par la discrétisation et l'interpolation, nous pouvons concevoir des méthodes efficaces pour gérer la complexité supplémentaire qu'apporte l'évaluation des espérances. Ce problème devient essentiellement un problème d'intégration numérique. Lorsque l'ensemble des perturbations est continu (comme c'est souvent le cas avec des espaces d'états continus), on entre dans un domaine où des méthodes de quadrature numérique pourraient s'appliquer. Mais ces méthodes passent mal à l'échelle lorsque le nombre de dimensions augmente. C'est là qu'interviennent des techniques plus efficaces, souvent issues des méthodes de Monte-Carlo. Deux ingrédients s'attaquent à la malédiction de la dimension :

1. l'approximation de fonctions (par discrétisation, interpolation, réseaux de neurones, etc.);
2. l'intégration de Monte-Carlo (simulation).

Ces deux éléments condensent essentiellement les ingrédients clés de l'apprentissage automatique, qui est la direction que nous explorerons dans ce cours.

## Exemple : gestion d'un réservoir soumis à des apports aléatoires

Comment la politique de gestion du réservoir change-t-elle lorsque le lâcher
de chaque semaine doit être choisi avant que l'apport de cette semaine soit
connu?

Le chapitre précédent exploitait un [réservoir hydroélectrique
saisonnier](finite-horizon-dp.md#example-seasonal-operation-of-a-hydroelectric-reservoir)
à partir d'une prévision des apports, qu'il traitait comme exacte. Sa
politique remplit de nouveau le réservoir après la crue printanière, le
maintient à pleine capacité de septembre à la fin novembre, et lâche
exactement l'apport prévu pendant chacune de ces semaines. Ce plan ne vaut que
ce que vaut la prévision : lors d'une semaine plus humide que prévu, le
réservoir est déjà plein et l'eau excédentaire passe par l'évacuateur de crues.
La programmation dynamique stochastique sert depuis longtemps à calculer des
politiques d'exploitation de réservoirs qui tiennent compte de ce risque
{cite:p}`Stedinger1984`. On rend l'apport aléatoire et on conserve tous les
autres éléments du modèle déterministe : 52 étapes hebdomadaires, les limites
de stockage et de lâcher, la hauteur de chute $h(x)$, le prix $p_t$ et la
valeur terminale.

Soit $\bar w_t$ la prévision du chapitre précédent. L'apport de la semaine $t$
est

$$
w_t = \xi_t\,\bar w_t,
\qquad
\xi_t = \begin{cases}
0.6 & \text{avec probabilité } 1/4, \\
1.0 & \text{avec probabilité } 1/2, \\
1.4 & \text{avec probabilité } 1/4,
\end{cases}
$$

où les facteurs $\xi_0, \ldots, \xi_{51}$ sont indépendants. L'apport moyen
est égal à la prévision, et une semaine sur quatre est 40 % plus humide que
prévu. L'apport joue le rôle de la perturbation $\mathbf{w}_t$ de la
dynamique stochastique ci-dessus. Comme les facteurs sont indépendants, les
apports passés n'apportent aucune information sur le suivant, et le stockage
seul reste un état suffisant, comme dans {prf:ref}`stoch-state-suff`.

L'exploitant fixe le lâcher $u_t$ au début de la semaine, avant l'arrivée de
l'apport. Lors d'une semaine sèche, le lâcher demandé peut dépasser l'eau
disponible au-dessus du stockage mort. Les turbines s'arrêtent alors quand le
niveau atteint le stockage mort, si bien que le volume effectivement lâché
est

$$
\tilde u_t = \min\{u_t,\ x_t + w_t - x_{\min}\}.
$$

Le stockage évolue selon $x_{t+1} = \min\{x_t + w_t - \tilde u_t,\ \bar x\}$, et
la semaine rapporte $p_t\,\kappa\,h(x_t)\,\tilde u_t$. Le successeur et le
revenu dépendent désormais tous deux de la perturbation, comme
$\mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t,
\mathbf{w}_t)$ et $c_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)$ dans la
formulation générale. En notant $\xi^k$ les trois facteurs et $q_k$ leurs
probabilités, l'équation de Bellman s'écrit

$$
J_t^\star(x) = \max_{u_{\min} \le u \le u_{\max}}
\sum_{k=1}^{3} q_k \Big[\, p_t\,\kappa\,h(x)\,\tilde u^k
+ J_{t+1}^\star\big(\min\{x + \xi^k \bar w_t - \tilde u^k,\ \bar x\}\big) \Big],
\qquad
\tilde u^k = \min\{u,\ x + \xi^k \bar w_t - x_{\min}\},
$$

avec la valeur terminale $J_T^\star(x) = p_T\,\kappa\,h(x)\,(x - x_{\min})$
comme auparavant. L'espérance se trouve à l'intérieur du maximum, car un seul
lâcher doit servir pour les trois apports.

Le code ci-dessous modifie le solveur déterministe en deux endroits. Le
crochet est moyenné sur les trois apports avant que l'on prenne le maximum sur
les lâchers. Et les successeurs $x + 0.6\,\bar w_t - u$ et $x + 1.4\,\bar w_t - u$
ne sont en général plus des multiples du pas de grille de 10 hm³; on évalue
donc $J_{t+1}^\star$ entre les points de la grille par interpolation linéaire,
comme dans [la section sur
l'interpolation](finite-horizon-dp.md#handling-continuous-spaces-with-interpolation)
du chapitre précédent. Avec l'unique facteur 1 de probabilité 1, la même
fonction résout le problème déterministe et renvoie ses 74.1 M\$ à partir de
1100 hm³; avec un apport connu, arrêter un lâcher au stockage mort revient à
l'interdire. C'est le problème d'*équivalent certain* : l'apport aléatoire est
remplacé par sa moyenne, et la politique obtenue est appliquée comme si la
prévision était exacte. Les dernières lignes exploitent les deux politiques
pendant 2 000 années simulées, sur les mêmes suites d'apports.

```{code-cell} python
import numpy as np

# The deterministic model of the previous chapter.
T = 52
weeks = np.arange(T)
w_bar = 60 + 340 * np.exp(-((weeks - 19) / 3.5) ** 2) + 60 * np.exp(-((weeks - 43) / 4) ** 2)
w_bar = 10 * np.round(w_bar / 10)                    # inflow forecast (hm³/week)
p = 60 + 30 * np.cos(2 * np.pi * (weeks - 5) / 52)   # price ($/MWh)
x_min, x_max = 200, 2000
u_min, u_max = 30, 250
kappa, p_T = 2.45, 50

def head(x):
    return 40 + 40 * np.sqrt(x / x_max)

def terminal_value(x):
    return p_T * kappa * head(x) * (x - x_min)

dx = 10
X = np.arange(x_min, x_max + dx, dx)
U = np.arange(u_min, u_max + dx, dx)

# Random inflow w_t = xi_t * w_bar_t, with xi_t drawn independently each week.
factors = np.array([0.6, 1.0, 1.4])
probs = np.array([0.25, 0.5, 0.25])

def solve(factors, probs):
    J = np.zeros((T + 1, len(X)))
    policy = np.zeros((T, len(X)))
    J[T] = terminal_value(X)
    x, u = X[:, None], U[None, :]                     # all (storage, release) pairs
    for t in reversed(range(T)):
        value = np.zeros((len(X), len(U)))
        for factor, prob in zip(factors, probs):      # expectation over the inflow...
            inflow = factor * w_bar[t]
            release = np.minimum(u, x + inflow - x_min)       # turbines stop at dead storage
            x_next = np.minimum(x + inflow - release, x_max)  # water above capacity spills
            value += prob * (p[t] * kappa * head(x) * release
                             + np.interp(x_next, X, J[t + 1]))
        J[t] = value.max(axis=1)                      # ...inside the maximum over releases
        policy[t] = U[value.argmax(axis=1)]
    return J, policy

def simulate(policy, years, x0):
    """Operate the reservoir for many years at once: one row of factors per year."""
    x = np.full(len(years), float(x0))
    storage, revenue, spill = [x], 0.0, 0.0
    for t in range(T):
        inflow = years[:, t] * w_bar[t]
        requested = policy[t, np.round((x - x_min) / dx).astype(int)]
        release = np.minimum(requested, x + inflow - x_min)
        revenue = revenue + p[t] * kappa * head(x) * release
        spill = spill + np.maximum(x + inflow - release - x_max, 0)
        x = np.minimum(x + inflow - release, x_max)
        storage.append(x)
    return np.array(storage), revenue + terminal_value(x), spill

J_ce, policy_ce = solve([1.0], [1.0])         # certainty equivalent: inflow = forecast
J_sto, policy_sto = solve(factors, probs)     # stochastic dynamic programming

rng = np.random.default_rng(0)
years = rng.choice(factors, size=(2000, T), p=probs)   # the same inflows for both policies
i0 = np.searchsorted(X, 1100)
results = {}
for name, J, policy in (("certainty equivalent", J_ce, policy_ce),
                        ("stochastic", J_sto, policy_sto)):
    storage, revenue, spill = simulate(policy, years, x0=1100)
    results[name] = (storage, revenue, spill)
    print(f"{name:<21} predicted {J[0, i0] / 1e6:5.2f} M$   simulated {revenue.mean() / 1e6:5.2f} M$"
          f"   spill {spill.mean():3.0f} hm³/year")
```

Les deux passes arrière prédisent des choses différentes. La passe
d'équivalent certain prédit 74.06 M\$, le revenu que sa politique obtiendrait
si l'apport de chaque semaine correspondait à la prévision, ce qui n'arrive
jamais. Sur les apports simulés, la même politique rapporte en moyenne
73.05 M\$, environ 1 M\$ de moins, et déverse de l'eau dans 96 % des années.
Avec une valeur de l'eau d'environ 11 k\$ par hm³, les 85 hm³ qu'elle déverse
chaque année expliquent l'essentiel du manque à gagner. La passe stochastique
prédit 73.75 M\$, et la simulation concorde à son erreur type de 0.06 M\$ près.
Cette concordance est attendue : $J_0^\star(1100)$ est le revenu espéré de la
politique stochastique sous le modèle d'apports, et la moyenne sur les années
simulées estime la même espérance.

Sur les mêmes apports, la politique stochastique rapporte 0.68 M\$ de plus par
année que la politique d'équivalent certain. La programmation stochastique
appelle cette différence la *valeur de la solution stochastique*
{cite:p}`Birge2011`. Elle est faible comparée au revenu total, mais l'essentiel
de ce revenu provient d'eau que n'importe quelle règle vendrait. Dans le
modèle déterministe, exploiter la rivière au fil de l'eau sans rien stocker
rapportait 66.8 M\$, et la gestion du stockage ajoutait 7.3 M\$. La perte due
au fait d'ignorer l'incertitude sur les apports représente environ le dixième
de ce montant.

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))

from hydro_reservoir_stochastic import (
    make_stochastic_comparison_figure,
    make_stochastic_policy_figure,
)

water_value_ce = np.gradient(J_ce, dx, axis=1)    # dJ_t/dx on the grid, $ per hm³
water_value_sto = np.gradient(J_sto, dx, axis=1)
storage_ce, revenue_ce, spill_ce = results["certainty equivalent"]
storage_sto, revenue_sto, spill_sto = results["stochastic"]
```

```{code-cell} python
:label: fig-hydro-stochastic-comparison
:caption: Les deux politiques exploitées sur les mêmes 2 000 années simulées à partir de $x_0 = 1100$ hm³. À gauche, le stockage d'août à décembre : médiane (trait plein) et 5e et 95e centiles (tireté). La politique d'équivalent certain (orange) maintient le réservoir à pleine capacité et déverse chaque fois qu'une semaine est plus humide que prévu; la politique stochastique (bleu) le maintient environ 50 hm³ plus bas. À droite, le revenu de la politique stochastique moins celui de la politique d'équivalent certain pour chaque année simulée.
:tags: [remove-input]

make_stochastic_comparison_figure(storage_ce, storage_sto, revenue_ce,
                                  revenue_sto, x_max, first_week=30)
```

Comme les deux politiques font face aux mêmes apports, le gain peut se mesurer
année par année, comme dans le panneau de droite de la
{numref}`fig-hydro-stochastic-comparison`. Sa moyenne a une erreur type de
0.01 M\$, contre 0.06 M\$ pour chacun des revenus moyens : une année humide ou
sèche fait monter ou baisser le revenu des deux politiques de la même façon,
et cette variation commune s'annule dans la différence. Évaluer des politiques
concurrentes sur les mêmes entrées aléatoires est une technique de réduction
de la variance appelée méthode des *nombres aléatoires communs*. La politique
stochastique rapporte davantage dans 87 % des années. Les autres années ont des
automnes secs, pendant lesquels la politique d'équivalent certain déverse peu
et la marge conservée par la politique stochastique s'avère inutile.

Le gain provient presque entièrement de l'automne, montré dans le panneau de
gauche. La politique d'équivalent certain remplit le réservoir à pleine
capacité en septembre et l'y maintient en lâchant l'apport prévu chaque
semaine; chaque semaine plus humide que prévu déverse donc. La politique
stochastique s'arrête environ 50 hm³ sous la capacité. Fin octobre, l'apport
prévu est de 120 hm³ par semaine, et la semaine la plus humide apporte
$0.4 \times 120 = 48$ hm³ de plus que prévu. La politique garde exactement
assez de place pour l'absorber.

La taille de cette marge découle de la comparaison qui expliquait la politique
déterministe au chapitre précédent : les turbines fonctionnent lorsque lâcher
un hm³ de plus, ce qui vaut $p_t\,\kappa\,h(x)$, rapporte davantage que le
garder, ce qui vaut la valeur de l'eau $\partial J_{t+1}^\star/\partial x$ au
stockage suivant. Avec un apport aléatoire, le second terme devient une
moyenne sur les trois successeurs $x_{t+1}^k$. Pour un apport qui remplit le
réservoir, un hm³ gardé cette semaine est déversé et n'ajoute rien la semaine
suivante; ces successeurs sortent donc de la moyenne. Tant que le lâcher n'est
pas interrompu au stockage mort, la pente $\partial Q_t/\partial u$ du
chapitre précédent devient

$$
p_t\,\kappa\,h(x) \;-\; \sum_{k:\ x_{t+1}^k < \bar x} q_k\,
\frac{\partial J_{t+1}^\star}{\partial x}\big(x_{t+1}^k\big).
$$

La règle devient : faire fonctionner les turbines lorsque le prix dépasse la
valeur espérée de l'eau gardée, en comptant pour nulle l'eau qui serait
déversée. Fin octobre, près de la capacité, un hm³ rapporte environ 10.3 k\$
s'il est lâché et a une valeur de l'eau d'environ 11.2 k\$ s'il est gardé.
Sous la marge, le hm³ gardé n'est jamais déversé; le garder vaut donc
davantage. Au-dessus de la marge, il est déversé lors de la semaine la plus
humide, avec une probabilité de 1/4, et sa valeur espérée tombe à
$0.75 \times 11.2 = 8.4$ k\$, moins que les 10.3 k\$ que rapporte son lâcher.
La politique maintient donc le niveau de stockage à partir duquel la semaine
la plus humide remplit tout juste le réservoir. Dans le modèle déterministe,
l'apport est connu; la politique peut donc remplir le réservoir à pleine
capacité sans déverser.

```{code-cell} python
:label: fig-hydro-stochastic-policy
:caption: Solution du problème stochastique. À gauche, le lâcher optimal $u_t^\star(x)$ sous la prévision des apports (marques foncées) et l'étendue de l'apport aléatoire (barres claires). Les courbes blanches sont 40 années d'exploitation simulées à partir de 1100 hm³. À droite, la valeur de l'eau au début de la crue. Avec un apport connu (orange), elle chute brusquement au niveau de stockage à partir duquel la crue remplit le réservoir; avec un apport aléatoire (bleu), la chute est étalée d'environ 1100 hm³ jusqu'à la capacité.
:tags: [remove-input]

make_stochastic_policy_figure(w_bar, factors, X, policy_sto, storage_sto[:, :40],
                              14, water_value_ce[14], water_value_sto[14])
```

Le même calcul de moyenne modifie la valeur de l'eau. Le panneau de droite de
la {numref}`fig-hydro-stochastic-policy` la montre à la semaine 14, début
avril, lorsque la crue commence. Dans le modèle d'équivalent certain, elle
chute brusquement à 1770 hm³ : à partir de ce niveau, la crue prévue remplit
le réservoir, et tout hm³ supplémentaire sera déversé à coup sûr. Dans le
modèle stochastique, la chute est étalée d'environ 1100 hm³ jusqu'à la
capacité. Un hm³ supplémentaire n'est déversé que si la crue est assez
abondante, et la probabilité d'une telle crue croît graduellement avec le
stockage. Plus généralement, chaque $J_t^\star$ moyenne $J_{t+1}^\star$ sur
trois successeurs, et cette moyenne adoucit les points anguleux de la fonction
de valeur déterministe à chaque étape de la récurrence.

Les courbes blanches du panneau de gauche sont 40 années simulées. Dans le
modèle déterministe, un seul chemin décrivait l'exploitation selon la
politique; ici, chaque année suit le sien. Les chemins s'écartent en janvier et
en février, pendant que la politique lâche le minimum et que le stockage suit
l'apport aléatoire. Ils se resserrent à la fin de l'abaissement hivernal,
quand chaque année atteint un faible stockage avant la crue, s'écartent de
nouveau pendant la crue, dont le volume varie d'une année à l'autre, et
convergent à l'automne vers le niveau sous la capacité où la politique les
maintient. Le résultat d'une politique sous incertitude est une distribution,
et l'estimer demande de nombreuses trajectoires simulées. Il en va de même
pour les méthodes d'apprentissage par renforcement des chapitres suivants :
une bonne trajectoire ne montre pas qu'une politique est bonne.

> **Question.** Si l'exploitant pouvait voir l'apport de chaque semaine avant
> de choisir le lâcher, le revenu espéré à partir de 1100 hm³ serait-il
> supérieur ou inférieur à 73.75 M\$? Quelles lignes de `solve` changeraient?

:::{dropdown} Réponse
Supérieur, ou au moins égal. Voir l'apport d'abord permet un lâcher différent
pour chacun des trois apports, et choisir le même lâcher pour les trois reste
possible; le maximum ne peut donc pas diminuer. Dans l'équation de Bellman, le
maximum passe à l'intérieur de l'espérance,
$J_t^\star(x) = \sum_k q_k \max_u [\,\cdots]$. Dans `solve`, chaque terme est
maximisé par rapport aux lâchers avant d'être ajouté,
`value += prob * (...).max(axis=1)`, avec `value` un vecteur indexé par les
niveaux de stockage, et `J[t] = value`. La politique dépend alors de l'apport
autant que du stockage. Le revenu espéré devient 74.01 M\$; connaître d'avance
l'apport de chaque semaine vaut donc 0.26 M\$ par année.
:::

> **Question.** Dans ce modèle, le facteur d'apport est tiré indépendamment
> chaque semaine; un mois d'avril humide ne dit donc rien du mois de mai. Les
> crues réelles tendent à être humides ou sèches pendant des semaines
> d'affilée, parce qu'elles proviennent de la fonte d'un même manteau neigeux.
> Que devrait contenir l'état pour que la récurrence arrière reste valide, et
> qu'est-ce que cela coûterait?

:::{dropdown} Réponse
Avec des apports corrélés, le stockage n'est plus un état suffisant : la
distribution de l'apport de la semaine suivante dépend des apports passés, et
l'argument de {prf:ref}`stoch-state-suff` ne s'applique plus au stockage
seul. Si le facteur suit une chaîne de Markov, ajouter à l'état le facteur de
la semaine précédente rétablit la suffisance. L'état devient
$(x_t, \xi_{t-1})$, avec $181 \times 3 = 543$ points de grille au lieu de 181,
et l'espérance utilise les probabilités de transition issues de
$\xi_{t-1}$. Un manteau neigeux mesuré serait une seconde variable d'état
continue, avec sa propre grille. Chaque variable de ce genre multiplie la
taille de la grille : c'est de nouveau la malédiction de la dimension. Une
autre solution courante consiste à résoudre de nouveau un problème
déterministe chaque semaine avec une prévision mise à jour, comme dans la
[commande à horizon glissant](receding-horizon-control.md).
:::

## Formulation en processus de décision markovien

Quels objets (états, actions, transitions, récompenses et horizon) définissent
la représentation usuelle d'un processus de décision markovien fini?

Plutôt que d'exprimer la stochasticité de notre système par un terme de perturbation passé en paramètre à une équation aux différences déterministe, on travaille souvent avec une autre représentation (plus courante en recherche opérationnelle), qui utilise la formulation en processus de décision markovien. L'idée est la suivante : lorsqu'on modélise ainsi le système, avec un terme de perturbation tiré indépendamment des étapes précédentes, les trajectoires induites sont celles d'une chaîne de Markov. On peut donc reformuler notre problème de commande dans ce langage, ce qui mène au cadre dit des processus de décision markoviens, dans lequel la dynamique du système s'exprime au moyen de probabilités de transition plutôt que d'équations d'état explicites. Dans ce cadre, on exprime la probabilité que le système se trouve dans un état donné à l'aide de la fonction de probabilité de transition :

$$ p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) $$

Cette fonction donne la probabilité de passer à l'état $\mathbf{x}_{t+1}$ au temps $t+1$, sachant que le système se trouve dans l'état $\mathbf{x}_t$ et que l'action $\mathbf{u}_t$ est prise au temps $t$. Ainsi, $p_t$ précise une distribution de probabilité conditionnelle sur les états suivants : la somme (pour des espaces d'états discrets) ou l'intégrale sur l'état suivant doit valoir 1.

À partir de la formulation de notre problème propre à la théorie de la commande, au moyen d'une fonction de dynamique déterministe et d'un terme de bruit, on peut dériver la fonction de probabilité de transition correspondante par la relation suivante :

$$
\begin{aligned}
p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) &= \mathbb{P}(\mathbf{W}_t \in \left\{\mathbf{w} \in \mathbf{W}: \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w})\right\}) \\
&= \sum_{\left\{\mathbf{w} \in \mathbf{W}: \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w})\right\}} q_t(\mathbf{w})
\end{aligned}
$$

Ici, $q_t(\mathbf{w})$ représente la densité de probabilité ou la fonction de masse de la perturbation $\mathbf{W}_t$ (en supposant des espaces d'états discrets). Pour des espaces continus, l'expression ci-dessus contient simplement une intégrale au lieu d'une somme.

Pour un système à dynamique déterministe et sans perturbation, les probabilités de transition deviennent beaucoup plus simples et peuvent s'exprimer à l'aide de la fonction indicatrice. Soit un système déterministe de dynamique :

$$ \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t) $$

La fonction de probabilité de transition peut s'écrire :

$$ p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) = \begin{cases}
1 & \text{si } \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t) \\
0 & \text{sinon}
\end{cases} $$

Avec cette fonction de probabilité de transition, on peut reformuler notre équation d'optimalité de Bellman :

$$ J_t^\star(\mathbf{x}_t) = \max_{\mathbf{u}_t \in \mathbf{U}} \left\{ c_t(\mathbf{x}_t, \mathbf{u}_t) + \sum_{\mathbf{x}_{t+1}} p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) J_{t+1}^\star(\mathbf{x}_{t+1}) \right\} $$

Ici, ${c}(\mathbf{x}_t, \mathbf{u}_t)$ représente la récompense immédiate espérée (ou l'opposé du coût) lorsqu'on se trouve dans l'état $\mathbf{x}_t$ et qu'on prend l'action $\mathbf{u}_t$ au temps $t$. Le terme de somme calcule la valeur optimale espérée des états futurs, pondérée par leurs probabilités de transition.

Cette formulation présente plusieurs avantages :

1. Elle rend explicite la nature markovienne du problème : l'état futur ne dépend que de l'état et de l'action courants, et non de l'historique des états et des actions.

2. Pour les problèmes à états discrets, toute la dynamique du système peut être décrite par un ensemble de matrices de transition, une pour chaque action possible.

3. Elle permet de faire le lien avec la riche panoplie de méthodes des modèles graphiques probabilistes et les techniques d'apprentissage automatique statistique pour la modélisation et l'analyse.

## Notation en recherche opérationnelle

Comment les conventions équivalentes pour les récompenses, les coûts, les
transitions et les politiques se correspondent-elles entre l'apprentissage
par renforcement et la recherche opérationnelle?

La présentation ci-dessus visait à faire le lien entre le point de vue de la théorie de la commande et le monde de la commande en boucle fermée, au moyen de l'idée de déterminer la fonction de valeur d'un problème de contrôle optimal paramétrique. Nous avons ensuite vu que la procédure d'induction arrière s'applique aux cas déterministe et stochastique, en prenant l'espérance par rapport à la variable de perturbation. Nous avons enfin indiqué que l'on peut aussi travailler avec une représentation du système où, au lieu d'écrire le modèle comme une fonction de dynamique déterministe recevant une perturbation en entrée, on passe directement par sa fonction de probabilité de transition, ce qui donne l'interprétation de notre système en simulation comme une chaîne de Markov.

Notons que la notation employée en théorie de la commande tend à différer de celle des communautés de recherche opérationnelle, où le domaine de la programmation dynamique s'est épanoui. Cette section résume ces différences (purement notationnelles).

En recherche opérationnelle, l'état du système à chaque époque de décision est habituellement noté $s \in \mathcal{S}$, où $S$ est l'ensemble des états possibles du système. Lorsque le système se trouve dans l'état $s$, le décideur peut choisir une action $a$ dans l'ensemble des actions admissibles $\mathcal{A}_s$. L'union de tous les ensembles d'actions est notée $\mathcal{A} = \bigcup_{s \in \mathcal{S}} \mathcal{A}_s$.

La dynamique du système est décrite par une fonction de probabilité de transition $p_t(j | s, a)$, qui représente la probabilité de passer à l'état $j \in \mathcal{S}$ au temps $t+1$, sachant que le système se trouve dans l'état $s$ au temps $t$ et que l'action $a \in \mathcal{A}_s$ est choisie. Cette fonction de probabilité de transition satisfait :

$$\sum_{j \in \mathcal{S}} p_t(j | s, a) = 1$$

Il convient de noter qu'en recherche opérationnelle, on maximise habituellement une récompense plutôt que de minimiser un coût, ce qui est plus courant en théorie de la commande. On passe toutefois facilement d'un point de vue à l'autre en changeant simplement le signe de la quantité. Autrement dit, maximiser une fonction de récompense équivaut à minimiser son opposé, que l'on appellerait alors une fonction de coût.

La fonction de récompense est notée $r_t(s, a)$; elle représente la récompense reçue au temps $t$ lorsque le système se trouve dans l'état $s$ et que l'action $a$ est prise. Dans certains cas, la récompense peut aussi dépendre de l'état suivant; on la note alors $r_t(s, a, j)$. La récompense espérée se calcule ensuite ainsi :

$$r_t(s, a) = \sum_{j \in \mathcal{S}} r_t(s, a, j) p_t(j | s, a)$$

Réunis, ces éléments précisent un processus de décision markovien, entièrement décrit par le n-uplet :

$$\{T, S, \mathcal{A}_s, p_t(\cdot | s, a), r_t(s, a)\}$$

où $\mathrm{T}$ représente l'ensemble des époques de décision (l'horizon).

## Qu'est-ce qu'une politique optimale?

À quelle classe de politiques concurrentes et à partir de quels états
initiaux faut-il comparer une politique avant de pouvoir la dire optimale?

Revenons au point de départ et définissons ce que signifie, pour une politique, être optimale dans un problème de décision markovien. Pour cela, nous considérerons différents espaces de recherche possibles (des classes de politiques) et comparerons les politiques selon l'ordre de leurs valeurs depuis tout état initial possible. La valeur d'une politique $\boldsymbol{\pi}$ (optimale ou non) est définie comme la récompense totale espérée obtenue en suivant cette politique à partir d'un état initial donné. Formellement, pour un processus de décision markovien à horizon fini comportant $N$ époques de décision, on définit la fonction de valeur $v^{\boldsymbol{\pi}}(s, t)$ par :

$$
v^{\boldsymbol{\pi}}(s, t) \triangleq \mathbb{E}\left[\sum_{k=t}^{N-1} r_t(S_k, A_k) + r_N(S_N) \mid S_t = s\right]
$$

où $S_t$ est l'état au temps $t$, $A_t$ l'action prise au temps $t$ et $r_t$ la fonction de récompense. Pour simplifier, on écrit $v^{\boldsymbol{\pi}}(s)$ pour désigner $v^{\boldsymbol{\pi}}(s, 1)$, la valeur obtenue en suivant la politique $\boldsymbol{\pi}$ depuis l'état $s$ à la première étape, sur tout l'horizon $N$.

Dans les processus de décision markoviens à horizon fini, notre but est de trouver une politique optimale, notée $\boldsymbol{\pi}^*$, qui maximise la récompense totale espérée sur l'horizon $N$. Plus précisément :

$$
v^{\boldsymbol{\pi}^*}(s) \geq v^{\boldsymbol{\pi}}(s), \quad \forall s \in \mathcal{S}, \quad \forall \boldsymbol{\pi} \in \Pi^{\text{HR}}
$$

On dit que $\boldsymbol{\pi}^*$ est une **politique optimale** parce qu'elle donne la plus grande valeur possible dans tous les états, parmi toutes les politiques de la classe $\Pi^{\text{HR}}$. On note $v^*$ la valeur maximale que peut atteindre une politique :

$$
v^*(s) = \max_{\boldsymbol{\pi} \in \Pi^{\text{HR}}} v^{\boldsymbol{\pi}}(s), \quad \forall s \in \mathcal{S}
$$

Dans la littérature sur l'apprentissage par renforcement, $v^*$ est habituellement appelée « fonction de valeur optimale », tandis que certaines références de recherche opérationnelle l'appellent « valeur d'un processus de décision markovien ». Une politique optimale $\boldsymbol{\pi}^*$ est une politique dont la fonction de valeur est égale à la fonction de valeur optimale :

$$
v^{\boldsymbol{\pi}^*}(s) = v^*(s), \quad \forall s \in \mathcal{S}
$$

Cette notion d'optimalité s'applique à chaque état. Les politiques optimales en ce sens sont parfois dites « uniformément optimales ». Une notion d'optimalité plus faible, souvent rencontrée dans la pratique de l'apprentissage par renforcement, est l'optimalité par rapport à une distribution initiale des états. On cherche alors une politique $\boldsymbol{\pi} \in \Pi^{\text{HR}}$ qui maximise :

$$
\sum_{s \in \mathcal{S}} v^{\boldsymbol{\pi}}(s) P_1(S_1 = s)
$$

où $P_1(S_1 = s)$ est la probabilité de partir de l'état $s$.

Les deux définitions classent les politiques, et ce classement repose sur deux
ingrédients qui sont des choix de modélisation plutôt que des conséquences de
la dynamique.

Le premier est l'ordre lui-même. L'optimalité uniforme compare les fonctions
de valeur état par état : $\boldsymbol{\pi}$ est au moins aussi bonne que
$\boldsymbol{\pi}'$ lorsque $v^{\boldsymbol{\pi}}(s) \ge v^{\boldsymbol{\pi}'}(s)$
pour tout $s \in \mathcal{S}$. C'est un ordre *partiel*. Deux politiques
peuvent être incomparables, chacune étant meilleure que l'autre à partir de
certains états initiaux. L'existence d'une politique qui domine toutes les
autres pour cet ordre est donc un résultat à démontrer, et non une hypothèse;
l'algorithme d'induction arrière ci-dessous en fournit la preuve pour les
horizons finis.

Le second est l'espérance. En présence de bruit, une politique induit, depuis
chaque état, une distribution des récompenses totales, et un classement exige
un seul nombre par distribution. Prendre la moyenne est la réponse classique
de la théorie de l'utilité espérée, qui remonte à von Neumann et
Morgenstern : des préférences sur des issues incertaines qui satisfont
quelques axiomes de cohérence peuvent être représentées par l'espérance d'une
utilité. Prendre la récompense elle-même comme utilité rend le décideur
*neutre au risque*. Deux politiques de même moyenne sont alors également
bonnes, si différentes que soient leurs dispersions ou leurs pires cas. Un
exploitant de réservoir qui doit éviter de déverser lors d'une année humide
peut refuser ce compromis. Les formulations sensibles au risque remplacent la
moyenne par une autre fonctionnelle de la distribution, par exemple la valeur
à risque conditionnelle (CVaR), la moyenne de la fraction $\alpha$ des pires
issues. De tels critères mènent à des équations de Bellman différentes, et
certaines d'entre elles ne se décomposent plus étape par étape. Ce livre
conserve l'espérance d'un bout à l'autre.

La valeur maximale peut être atteinte en cherchant dans l'espace des politiques markoviennes déterministes. Par conséquent :

$$ v^*(s) = \max_{\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}} v^{\boldsymbol{\pi}}(s) = \max _{\boldsymbol{\pi} \in \Pi^{M D}} v^{\boldsymbol{\pi}}(s), \quad s \in S$$

Cette égalité rend la recherche traitable : elle se décompose en $N$ sous-problèmes, dont chacun ne cherche que parmi les actions disponibles à une étape. C'est l'algorithme d'induction arrière, que nous présentons une seconde fois, en quittant cette fois la notation de la théorie de la commande pour le formalisme des processus de décision markoviens :

````{prf:algorithm} Induction arrière
:label: backward-induction

**Entrée :** espace d'états $S$, espace d'actions $A$, probabilités de transition $p_t$, fonction de récompense $r_t$, horizon $N$

**Sortie :** fonctions de valeur optimales $v^*$ et ensembles d'actions optimales $A^*_{s,t}$

1. Initialiser :
   - Poser $t = N$
   - Pour tout $s \in S$ :

     $$v^*(s, N) = r_N(s)$$

2. Pour $t = N-1$ jusqu'à $1$ :
   - Pour chaque $s \in S$ :
     a. Calculer la fonction de valeur optimale :

        $$v^*(s, t) = \max_{a \in A_s} \left\{r_t(s, a) + \mathbb{E}_{s' \sim p_t(\cdot | s, a)}\left[v^*(s', t+1)\right]\right\}$$
     
     b. Déterminer l'ensemble des actions optimales :

        $$A_{s,t}^* = \arg\max_{a \in A_s} \left\{r_t(s, a) + \mathbb{E}_{s' \sim p_t(\cdot | s, a)}\left[v^*(s', t+1)\right]\right\}$$

3. Renvoyer les fonctions de valeur optimales $v^*(s, t)$ et les ensembles d'actions optimales $A_{s,t}^*$ pour tous $t$ et $s$
````

Lorsque $S$ est fini, l'espérance est la somme $\sum_{s' \in S} p_t(s' | s, a)\, v^*(s', t+1)$. L'écrire comme une espérance rend explicite le lien avec le programme dynamique stochastique présenté plus tôt dans ce chapitre : $\mathbb{E}_{s' \sim p_t(\cdot|s,a)}$ joue le rôle de $\mathbb{E}_{\mathbf{w}_t}$, le bruit étant absorbé dans le noyau de transition.

Notons que la même procédure, légèrement modifiée, permet aussi de calculer la valeur d'une politique :

````{prf:algorithm} Évaluation d'une politique
:label: backward-policy-evaluation

**Entrée :** 
- espace d'états $S$
- espace d'actions $A$
- probabilités de transition $p_t$
- fonction de récompense $r_t$
- horizon $N$
- une politique markovienne déterministe $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_{N-1})$

**Sortie :** fonction de valeur $v^{\boldsymbol{\pi}}$ de la politique $\boldsymbol{\pi}$

1. Initialiser :
   - Poser $t = N$
   - Pour tout $s \in S$ :

     $$v^{\boldsymbol{\pi}}(s, N) = r_N(s)$$

2. Pour $t = N-1$ jusqu'à $1$ :
   - Pour chaque $s \in S$ :
     a. Calculer la fonction de valeur de la politique donnée :

        $$v^{\boldsymbol{\pi}}(s, t) = r_t(s, \pi_t(s)) + \mathbb{E}_{s' \sim p_t(\cdot | s, \pi_t(s))}\left[v^{\boldsymbol{\pi}}(s', t+1)\right]$$

3. Renvoyer la fonction de valeur $v^{\boldsymbol{\pi}}(s, t)$ pour tous $t$ et $s$
````

Ce code pourrait enfin être adapté aux politiques aléatoires à l'aide de :

$$v^{\boldsymbol{\pi}}(s_t, t) = \sum_{a_t \in \mathcal{A}_{s_t}} \pi_t(a_t \mid s_t) \left( r_t(s_t, a_t) + \sum_{j \in S} p_t(j | s_t, a_t) v^{\boldsymbol{\pi}}(j, t+1) \right)$$

(sec-antibiotic-steering)=
## Exemple : orienter la résistance aux antibiotiques

Une suite de médicaments peut-elle écarter une population bactérienne de sa
forme la plus résistante, et dans quelle mesure est-il utile d'examiner la
population avant de choisir chaque médicament?

Beaucoup de bactéries résistent aux médicaments apparentés à la pénicilline
(les β-lactamines) en produisant une enzyme, la β-lactamase, qui dégrade le
médicament. Quelques mutations ponctuelles du gène de l'enzyme modifient les
médicaments qu'elle peut dégrader.
{cite:t}`Mira2015` ont construit les 16 combinaisons de quatre de ces
mutations (M69L, E104K, G238S et N276D) dans l'enzyme TEM d'*E. coli* et ont
mesuré le taux de croissance de chaque souche sous 15 β-lactamines. Chaque
médicament définit donc un *paysage adaptatif* : un taux de croissance pour
chaque génotype. Les paysages ne concordent pas. Un génotype qui croît le
plus vite sous un médicament peut compter parmi les plus lents sous un autre;
un médicament administré maintenant modifie donc la direction que prendra
l'évolution sous le suivant.
{cite:t}`Nichol2015` s'en sont servis pour *orienter* l'évolution : administrer
un à trois médicaments d'orientation avant le traitement final, choisis de
sorte que la population ait peu de chances d'aboutir au génotype le plus
résistant au médicament final. Le problème est un processus de décision
markovien à horizon fini assez petit pour être résolu exactement par
induction arrière.

**États.** Un génotype est une chaîne $s\in\{0,1\}^4$, où un 1 marque la
présence d'une mutation; donc $|\mathcal S|=16$. Le type sauvage est $0000$.
Le modèle suppose que la population est dominée en tout temps par un seul
génotype, ce qui est le régime habituel lorsque les mutations sont rares et
la sélection forte.

**Actions.** À chacune des époques d'orientation $t=1,\ldots,N-1$, l'action
est l'un des 15 médicaments. L'époque $N$ applique un médicament final fixé
$d^\star$.

**Transitions.** Soit $f_a(s)$ le taux de croissance mesuré du génotype $s$
sous le médicament $a$, et soit $U_a(s)$ l'ensemble des génotypes qui
diffèrent de $s$ par une mutation et croissent plus vite. Sous le médicament
$a$, un seul mutant plus apte envahit la population à chaque substitution, et
chaque voisin plus apte a la même probabilité d'être celui-là :

$$
M_a(j\mid s)=
\begin{cases}
1/|U_a(s)| & j\in U_a(s),\\
1 & j=s \text{ et } U_a(s)=\emptyset,\\
0 & \text{sinon.}
\end{cases}
$$

Un génotype sans voisin plus apte est un pic local du paysage et absorbe la
chaîne. Comme chaque substitution augmente strictement le taux de croissance,
tout chemin atteint un pic en au plus 15 substitutions. Un médicament est
administré assez longtemps pour que cela se produise; le noyau de transition
d'une époque de décision est donc la limite de la chaîne,
$p(j\mid s,a)=\big[\lim_{m\to\infty}M_a^m\big]_{sj}=[M_a^{16}]_{sj}$.

**Objectif.** Soit $s^\star$ le génotype dont le taux de croissance est le
plus élevé sous $d^\star$. Il n'y a pas de coût d'étape. Le coût terminal est
la probabilité que le médicament final mène le génotype $s$ à $s^\star$,
$c_N(s)=p(s^\star\mid s,d^\star)$, et le but est de minimiser sa valeur
espérée. L'induction arrière ({prf:ref}`backward-induction`, avec un minimum
à la place du maximum) s'écrit

$$
v_N(s)=c_N(s),\qquad
v_t(s)=\min_{a}\sum_{j}p(j\mid s,a)\,v_{t+1}(j),\qquad t=N-1,\ldots,1 .
$$

La politique obtenue est markovienne déterministe : un médicament pour chaque
couple formé d'une époque et d'un génotype. L'utiliser suppose que le génotype
dominant est séquencé avant chaque médicament. {cite:t}`Nichol2015` supposent
au contraire que rien n'est observé et que le génotype initial est
uniformément distribué, le pire cas, où toute souche résistante peut déjà être
présente. Leurs règles de décision ignorent l'état; une politique se réduit
donc à une suite fixe de médicaments. Ce sont toutes deux des politiques au
sens de la [section sur les règles de décision](#decision-rules-and-policies),
et comme une suite fixe est une politique markovienne qui se trouve être
constante en $s$, l'induction arrière ne peut faire qu'aussi bien ou mieux.
Les deux classes diffèrent aussi par leur coût. La meilleure suite fixe de
$k$ médicaments d'orientation s'obtient en examinant les $15^k$ suites,
tandis que l'induction arrière coûte $k$ produits de 15 matrices de
transition par un vecteur de dimension 16.

:::{iframe} ../interactive/antibiotic-steering.html
:label: fig-antibiotic-steering
:width: 100%
:title: Orienter la résistance aux antibiotiques : suites fixes et politique d'induction arrière
:class: antibiotic-steering-replay
:placeholder: _static/antibiotic_steering/steering.svg

Orientation de l'évolution avant un traitement final à l'ampicilline (AMP),
avec un médicament d'orientation et des génotypes de départ tous également
probables. Les nœuds sont les 16 génotypes, du type sauvage à gauche au
mutant quadruple à droite, et les arêtes relient les génotypes qui diffèrent
d'une mutation. La couleur d'un nœud indique le taux de croissance sous le
médicament qui agit sur ce génotype, et l'aire du disque bleu est
proportionnelle à la probabilité du génotype. L'anneau rouge marque $1111$,
le génotype le plus résistant à l'ampicilline. En haut : le meilleur
médicament d'orientation fixe, le cefprozil (CPR). En bas : la politique
d'induction arrière, qui administre à chaque génotype le médicament inscrit à
côté de lui. Dans le livre en ligne, la figure rejoue chaque substitution, et
le lecteur peut changer le médicament final, le nombre de médicaments
d'orientation, la population de départ et la stratégie.
:::

Sans orientation, l'ampicilline mène 62 % des génotypes de départ à $1111$.
Le meilleur médicament d'orientation unique, le cefprozil, réduit cette
proportion à 25 %, la valeur rapportée par {cite:t}`Nichol2015`. Si le
génotype est d'abord observé, la politique d'induction arrière atteint 11 %
avec le même nombre de médicaments ({numref}`fig-antibiotic-steering`). Elle
utilise quatre médicaments différents, selon le génotype. Le cefprozil peut
laisser une population en $0101$, l'un de ses pics, d'où l'ampicilline grimpe
directement jusqu'à $1111$. La plupart des génotypes à partir desquels cela
peut se produire reçoivent plutôt du céfotaxime (CTX) ou de la ceftazidime
(CAZ).

Avec deux médicaments d'orientation, la suite fixe « ampicilline, puis
cefprozil » atteint déjà zéro, et la rétroaction n'a plus rien à ajouter.
Administrer d'abord le médicament final envoie chaque génotype vers l'un des
trois pics de l'ampicilline, $0011$, $0110$ et $1111$. Le cefprozil déplace
ensuite $1111$ vers $0110$ et laisse les deux autres en place. Quand
l'ampicilline est administrée de nouveau, la population se trouve sur l'un de
ses deux pics inférieurs et y reste. Le premier médicament a rendu l'état
prévisible sans le mesurer; c'est pourquoi une suite qui ignore l'état peut
faire aussi bien qu'une politique qui l'observe. Le tableau ci-dessous montre
que c'est l'exception. Pour la plupart des médicaments finaux, un écart
subsiste après trois médicaments d'orientation; pour le céfépime (FEP), la
meilleure suite fixe laisse une probabilité de 0.33 et la politique avec
rétroaction 0.14.

:::{include} artifacts/antibiotic_steering/results.md
:::

Deux lignes montrent les limites de l'orientation. L'ampicilline associée au
sulbactam (SAM) n'a qu'un seul pic, en $1111$; toute population y aboutit donc,
quoi qu'on administre auparavant. Pour la ceftizoxime (ZOX) et la
pipéracilline associée au tazobactam (TZP), trois médicaments d'orientation
n'éliminent qu'une fraction du risque, avec ou sans rétroaction. Une autre
différence entre les deux classes n'apparaît pas dans le tableau. La
politique d'induction arrière est calculée une seule fois et s'applique à
toute distribution de départ, tandis que la meilleure suite fixe doit être
recherchée de nouveau chaque fois que la loi a priori sur le génotype de
départ change. Choisir le type sauvage comme population de départ dans le
rejeu fait du céfotaxime le meilleur médicament d'orientation unique. Avec un
médicament d'orientation et un génotype de départ connu, il n'y a rien à
observer avant l'unique décision; la suite fixe et la politique donnent donc
la même probabilité, 0.15.

Le modèle est une idéalisation délibérée. Les populations réelles comportent
plusieurs génotypes à la fois, les médicaments sont administrés pendant des
durées fixes plutôt que jusqu'à ce que l'évolution s'arrête, et le taux de
croissance dans une culture de laboratoire n'est qu'un indicateur indirect de
l'aptitude chez un patient. Les matrices de transition et les paysages
mesurés sont néanmoins des données réelles, et l'exemple montre les deux
caractéristiques qui comptent pour la suite du chapitre : la décision optimale
dépend de l'état, et l'induction arrière la trouve à un coût qui croît
linéairement avec l'horizon.

- {download}`Modèle et solveur pour l'orientation de la résistance aux antibiotiques <code/antibiotic_steering.py>`
- {download}`Générateur de l'expérience d'orientation de la résistance aux antibiotiques <scripts/build_antibiotic_steering_artifacts.py>`
- {download}`Résultats de l'orientation de la résistance aux antibiotiques pour chaque médicament final <artifacts/antibiotic_steering/metrics.json>`

(sec-storm-surge-barrier)=
## Exemple : fermer une barrière anti-tempête

Combien de temps l'exploitant d'une barrière anti-tempête doit-il attendre
une meilleure prévision avant d'ordonner la fermeture de la barrière?

La barrière Maeslant (Maeslantkering) protège Rotterdam des ondes de tempête
qui remontent le Nieuwe Waterweg, le chenal de navigation qui relie le port à
la mer du Nord. Elle se compose de deux portes de 210 m de large chacune, qui
reposent sur les rives. Lorsque le niveau d'eau prévu à Rotterdam dépasse
300 cm au-dessus du NAP, le niveau de référence néerlandais, les portes sont
amenées dans le chenal, remplies d'eau et coulées jusqu'au fond, ce qui prend
environ deux heures {cite:p}`RWSMaeslant,TraceKleeberg2025`. Tant que la
barrière est fermée, la principale voie de navigation vers le port est
bloquée. Venise fait face au même compromis avec ses barrières MOSE : les
fermetures inutiles retardent les navires, et les fermetures manquées laissent
la ville être inondée {cite:p}`Umgiesser2021`. L'exploitant décide à partir
d'une prévision du niveau d'eau maximal, et cette prévision devient plus
précise à mesure que le pic approche : l'incertitude sur les maxima de
surcote prévus croît avec l'échéance de la prévision {cite:p}`Umgiesser2021`.
Attendre permet donc d'acquérir de l'information. Mais cela consomme aussi le
temps dont la procédure de fermeture a besoin. La seule décision est de savoir
quand cesser d'attendre, ce qui en fait un problème d'*arrêt optimal*, un
processus de décision markovien à horizon fini dans lequel une action met fin
au processus de décision.

Nous construisons un petit modèle de cette décision. Deux de ses ingrédients
sont publiés : le critère de fermeture de 300 cm au-dessus du NAP, que nous
utilisons aussi comme niveau au-delà duquel la ville est inondée, et la
probabilité de défaillance de 1 sur 100 par fermeture que le port de Rotterdam
indique pour la barrière {cite:p}`PortRotterdamFloodRisk`. Tous les autres
nombres ci-dessous sont des valeurs pédagogiques.

**Époques de décision.** On compte le temps en nombre d'heures $h$ restant
avant le pic prévu. Une tempête est annoncée à $h=H=24$, l'exploitant décide
une fois par heure à $h=24,23,\ldots,1$, et le pic survient à $h=0$.

**États.** L'état est $s=(f,b)$. La première composante est la prévision
courante $f$ du niveau d'eau maximal à Rotterdam, en cm au-dessus du NAP, sur
une grille de 1 cm allant de 150 cm à 420 cm. La seconde est le statut $b$ de
la barrière : ouverte; navires avertis depuis 1, 2 ou 3 heures; portes en
mouvement pendant la première ou la deuxième heure de l'opération de
fermeture; retardée par une panne; ou fermée. Cela donne 271 prévisions et
9 statuts.

**Actions.** Lorsque la barrière est ouverte, l'ensemble des actions est
$\mathcal A_{(f,\text{ouverte})}=\{\text{attendre},\text{fermer}\}$. Une fois la
fermeture ordonnée, elle ne peut plus être annulée; tous les autres statuts
n'ont donc qu'une seule action, attendre.

**Transitions.** La prévision et la barrière évoluent indépendamment,
$p(j\mid s,a)=K(f'\mid f)\,P(b'\mid b,a)$ pour $j=(f',b')$. La prévision varie
chaque heure d'un incrément gaussien,

$$
f' = f + w,\qquad w\sim\mathcal N(0,\sigma^2),\qquad \sigma = 4\ \text{cm},
$$

arrondi à la grille, de sorte que $K$ est une matrice stochastique
$271\times 271$. La prévision à $h=0$ est le pic réalisé $L$. Comme les
incréments sont de moyenne nulle, la prévision est une martingale : la
meilleure estimation du pic à l'échéance $h$ est la prévision courante, et
l'erreur restante a un écart type de $\sigma\sqrt h$, soit 20 cm un jour à
l'avance et 10 cm six heures à l'avance. Le statut suit une procédure fixe une
fois la fermeture ordonnée. Les navires sont avertis et le trafic s'arrête;
quatre heures après l'ordre, les portes commencent à bouger; deux heures plus
tard, elles sont fermées avec probabilité $1-q$, et avec probabilité $q=0.1$
une panne les retarde. Une fermeture retardée est achevée avec probabilité
$r=0.7$ à chacune des heures suivantes. Une fermeture ordonnée à l'échéance
$h$ n'est donc pas achevée au moment du pic avec probabilité

$$
\rho(h)=\begin{cases}1 & h<6,\\ q\,(1-r)^{h-6} & h\ge 6,\end{cases}
$$

soit 0.1 pour un ordre donné 6 heures à l'avance et 0.009 pour un ordre donné
8 heures à l'avance, ce qui est proche du 1 sur 100 publié.

**Coûts.** L'unité de coût est une heure d'arrêt de la navigation. Chaque
heure après l'ordre coûte 1, que les portes soient en mouvement ou fermées,
et la navigation reste arrêtée pendant $R=6$ heures de plus après le pic, le
temps que l'eau baisse et que les portes soient rouvertes. Si la barrière
n'est pas fermée lorsque le pic $L$ dépasse 300 cm, la ville est inondée, à un
coût $C$ que nous fixons à 1000 heures d'arrêt de la navigation. Les coûts
d'étape et terminal sont

$$
c(s,a)=\mathbb 1\{b\ne\text{ouverte ou }a=\text{fermer}\},\qquad
c_0(f,b)=R\,\mathbb 1\{b\ne\text{ouverte}\}+C\,\mathbb 1\{b\ne\text{fermée}\}\,\mathbb 1\{f>300\}.
$$

**Solution.** L'induction arrière ({prf:ref}`backward-induction`, avec un
minimum à la place du maximum) part du pic et s'en éloigne :

$$
v_0(s)=c_0(s),\qquad
v_h(s)=\min_{a\in\mathcal A_s}\Big\{c(s,a)+\sum_{j}p(j\mid s,a)\,v_{h-1}(j)\Big\},\qquad h=1,\ldots,H .
$$

Le solveur applique cette récurrence aux $271\times 9$ états à chacune des
24 époques, ce qui prend une fraction de seconde. Seul le statut ouvert
comporte un choix, et la récurrence y prend la forme classique de l'arrêt
optimal. Après l'ordre, le coût ne dépend plus de rien de ce que fait
l'exploitant. Il vaut $h+R$ heures d'arrêt de la navigation, plus le coût de
l'inondation si la fermeture est tardive et le pic élevé. En notant
$\pi_h(f)$ la probabilité que le pic dépasse 300 cm sachant la prévision $f$
à l'échéance $h$, le coût de l'arrêt et la valeur de la barrière ouverte sont

$$
g_h(f)=h+R+C\,\rho(h)\,\pi_h(f),\qquad
v_h(f,\text{ouverte})=\min\Big\{g_h(f),\ \sum_{f'}K(f'\mid f)\,v_{h-1}(f',\text{ouverte})\Big\}.
$$

La politique ordonne la fermeture à la première heure où s'arrêter coûte
moins que continuer. L'ensemble des états où cela se produit est la *région
d'arrêt*; pour ce modèle, elle a une forme simple à chaque échéance : la
fermeture est ordonnée lorsque la prévision atteint ou dépasse un niveau qui
dépend de $h$. Nous comparons la politique à deux règles du genre de celles
que pourrait contenir un manuel d'exploitation. La première applique le
critère de fermeture dès qu'il est atteint : ordonner la fermeture à la
première heure, entre 24 et 6 heures à l'avance, où la prévision atteint au
moins 300 cm. La seconde est la meilleure règle de la forme « ordonner la
fermeture dès que la prévision atteint un niveau $\bar f$, entre 6 et $\bar h$
heures à l'avance », trouvée en essayant chaque niveau de 260 cm à 310 cm
et chaque échéance limite $\bar h$. Toutes les politiques sont évaluées
exactement avec {prf:ref}`backward-policy-evaluation` sur des tempêtes dont
la prévision 24 heures à l'avance suit une loi normale de moyenne 280 cm et
d'écart type 20 cm; 15.3 % de ces tempêtes dépassent 300 cm.

:::{iframe} ../interactive/storm-surge-barrier.html
:label: fig-storm-surge-barrier
:width: 100%
:title: Fermer une barrière anti-tempête : région d'arrêt optimal et règles à seuil
:class: storm-surge-barrier-replay
:placeholder: _static/storm_surge_barrier/stopping_region.svg

Fermeture d'une barrière anti-tempête avec un coût d'inondation de 1000 heures
d'arrêt de la navigation. (a) Les cellules ombrées forment la région d'arrêt
de la politique d'induction arrière : à chaque échéance, les prévisions pour
lesquelles elle ordonne la fermeture. Le cadre tireté est la meilleure règle à
seuil, qui ordonne la fermeture lorsque la prévision atteint 282 cm entre 9 et
6 heures avant le pic. La ligne pointillée est le critère de fermeture de
300 cm, et la bande grise marque les 6 dernières heures, pendant lesquelles
une fermeture ne peut plus être achevée. Trois tempêtes échantillonnées sont
tracées avec leurs prévisions; un point marque l'heure à laquelle la politique
ordonne la fermeture, et un carré le pic réalisé. (b) Probabilité d'une
fermeture inutile en fonction de la probabilité d'une inondation manquée,
pour des coûts d'inondation de 200 à 5000 heures de navigation, pour la
politique et pour la meilleure règle à seuil à chaque coût. Le losange
applique le critère de 300 cm dès qu'il est atteint. Dans le livre en ligne,
la figure rejoue des tempêtes échantillonnées avec une coupe transversale du
chenal, la prévision avec son incertitude qui se resserre, et l'état courant
qui se déplace dans la région d'arrêt.
:::

La politique n'ordonne jamais la fermeture plus de 10 heures avant le pic
({numref}`fig-storm-surge-barrier`). Dix heures à l'avance, une fermeture
n'est pas achevée à temps avec probabilité $\rho(10)=0.0008$, et attendre une
heure de plus porte cette probabilité à $\rho(9)=0.0027$. Avec un coût
d'inondation de 1000, ce risque supplémentaire ne vaut une heure de
navigation que lorsque l'inondation est plus probable qu'improbable, et la
politique ne ferme effectivement à 10 heures que lorsque la prévision atteint
au moins 302 cm, niveau où la probabilité de dépasser 300 cm est de 0.55. À
mesure que l'échéance approche, chaque heure d'attente ajoute davantage de
risque, et le niveau de fermeture baisse : 290 cm à 9 heures, 285 cm à
8 heures, 282 cm à 7 heures et 279 cm à 6 heures. Six heures à l'avance est la
dernière occasion. Fermer à ce moment coûte 12 heures de navigation et
élimine 90 % du coût d'inondation espéré; la politique ferme donc dès que la
probabilité de dépasser 300 cm atteint environ 1.3 %. À 279 cm, elle est de
1.4 %.

:::{include} artifacts/storm_surge_barrier/results.md
:::

Le critère de fermeture appliqué dès qu'il est atteint donne de mauvais
résultats dans ce modèle, avec un coût espéré de 30.9 heures contre 9.1 pour
la politique optimale. Il ferme moins souvent que la politique optimale, mais
il ordonne des fermetures jusqu'à 24 heures à l'avance et arrête donc la
navigation pendant 9.1 heures en moyenne contre 7.9. Il manque aussi des
inondations. Une tempête dont la prévision n'atteint 300 cm qu'au cours des
6 dernières heures arrive trop tard pour lui, et une fermeture ordonnée au
dernier moment échoue chaque fois qu'une panne la retarde. Ensemble, ces
inondations manquées surviennent dans 2.17 % des tempêtes, soit 14 % des
tempêtes qui dépassent 300 cm. La politique optimale ferme, près de
l'échéance, pour des prévisions bien inférieures à 300 cm, ce qui coûte des
fermetures inutiles dans 30.3 % des tempêtes, mais ne laisse une inondation
manquée que dans 0.12 % d'entre elles. L'essentiel de ce gain provient de la
forme de la règle plutôt que de la pente de la frontière. La meilleure règle
à seuil, un niveau de 282 cm appliqué entre 9 et 6 heures à l'avance, coûte
9.4, environ 4 % de plus que l'optimum. Le coût d'inondation fixe l'équilibre
entre les deux types d'erreur. Le faire passer de 200 à 5000 fait passer la
politique de 19.7 % de fermetures inutiles et 0.60 % d'inondations manquées à
38.7 % et 0.03 %, et la région d'arrêt s'agrandit : avec un coût d'inondation
de 5000, la politique ferme dès 11 heures à l'avance, à 296 cm. Dans le
panneau (b), les deux courbes coïncident presque : une fois qu'une règle a la
bonne fenêtre et le bon niveau, laisser le niveau varier d'heure en heure
rapporte peu.

Le modèle laisse de côté une bonne partie de ce qui compte pour la barrière
réelle. L'erreur de prévision est une marche aléatoire gaussienne de variance
constante par heure. Les erreurs de prévision réelles sont plus grandes pour
les surcotes plus importantes; le modèle probabiliste qu'utilise
Rijkswaterstaat pour planifier l'entretien suppose un écart type de 10 cm plus
un dixième de la surcote au-delà de 50 cm {cite:p}`TraceKleeberg2025`. L'heure
du pic est supposée connue, le débit fluvial contre lequel protège le critère
de 290 cm à Dordrecht est ignoré, la fermeture ne peut pas être annulée, et
les coûts, les délais de fermeture et la distribution des tempêtes sont des
valeurs pédagogiques plutôt que des données. La barrière réelle se ferme
automatiquement lorsque la prévision atteint le critère, et elle a été fermée
pour la première fois lors d'une tempête en 2007 {cite:p}`RWSMaeslant`. Ce que
l'exemple montre bel et bien, c'est la structure d'un problème d'arrêt
optimal. L'information ne cesse de s'améliorer, l'échéance est fixe, et le bon
seuil dépend du temps qui reste. L'induction arrière trouve ce seuil pour
toutes les échéances à la fois.

- {download}`Modèle et solveur de la barrière anti-tempête <code/storm_surge_barrier.py>`
- {download}`Générateur de l'expérience de la barrière anti-tempête <scripts/build_storm_surge_barrier_artifacts.py>`
- {download}`Résultats de la barrière anti-tempête pour chaque coût d'inondation <artifacts/storm_surge_barrier/metrics.json>`

## Synthèse et perspectives

La programmation dynamique stochastique distingue les règles de décision selon
l'information et le tirage aléatoire qu'elles utilisent, puis évalue chaque
action en moyennant sa valeur de continuation selon le noyau de transition.
L'optimalité à horizon fini reste une récurrence arrière, parce que la date
terminale fournit la condition aux limites.

Qu'est-ce qui remplace cette condition aux limites lorsque les décisions se
poursuivent indéfiniment? [La formulation à horizon
infini](infinite-horizon-mdps.md) utilise l'actualisation pour obtenir des
fonctions de valeur bornées et des équations de point fixe dont les solutions
ne dépendent plus d'un temps terminal.
