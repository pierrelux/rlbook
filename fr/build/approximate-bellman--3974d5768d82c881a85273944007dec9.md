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
# Méthodes de valeur ajustée

Le [chapitre précédent](weighted-residual-methods.md) a remplacé la fonction de
valeur par une fonction paramétrique $v(\cdot;\theta)$ et s'est demandé quel
ensemble fini de conditions doit déterminer $\theta$. Son étape 4 a résolu ces
conditions par itération fonctionnelle, une mise à jour de Bellman en des états
choisis suivie d'un ajustement, avec deux ajustements particuliers :
l'interpolation aux nœuds d'une base, et les moindres carrés sur une base. Ce
chapitre abandonne la base. N'importe quel régresseur peut faire l'ajustement,
un interpolant sur une grille, un ensemble d'arbres ou un réseau de neurones,
et le résultat est l'**itération sur les valeurs ajustée**. Chaque itération
est un problème de régression ordinaire dont les cibles sont des mises à jour
de Bellman, et l'itération Q ajustée, NFQ, DQN et NFQCA, dans les chapitres à
venir, en sont des élaborations.

L'ajustement n'est pas gratuit. L'itération sur les valeurs converge parce que
$\Bellman$ est une contraction pour la norme sup
({prf:ref}`value-iteration-convergence`), et la pêcherie du chapitre précédent
a montré un ajustement qui défait cette propriété. Le chapitre fait donc suivre
l'algorithme de trois résultats. Si chaque ajustement est précis à
$\varepsilon$ près, les erreurs ne s'accumulent pas au-delà de
$\varepsilon/(1-\gamma)$, ce qui autorise un régresseur quelconque. Les
moindres carrés peuvent néanmoins faire croître les erreurs sans borne, sur un
problème à deux états dont ils représentent exactement la fonction de valeur.
Et deux types d'ajustement sont sûrs, ce qui se démontre : les moyenneurs, qui
comprennent l'interpolation sur une grille et les ensembles d'arbres du premier
algorithme d'itération Q ajustée, et les moindres carrés pondérés par la
distribution des états de la politique évaluée. Les réseaux de neurones
n'appartiennent à aucun des deux types. Ce qui garde l'itération Q ajustée
neuronale stable en pratique est un ensemble de choix de conception qu'examine
le [chapitre sur les données historiques et en ligne](fitted-q-iteration.md).
Nous passons ensuite à l'autre solveur du chapitre précédent, la méthode de
Newton, qui devient l'itération sur les politiques approchée et, pour des
caractéristiques linéaires, les algorithmes LSTD et LSPI. Le chapitre se
termine par les étapes qui séparent l'itération sur les valeurs ajustée de
NFQCA, et par le chapitre où chacune est franchie.

Un exemple emploie des données réelles. Le modèle avec lequel le U.S. Fish and
Wildlife Service fixe la réglementation de la chasse aux canards est assez
petit pour être résolu exactement, de sorte qu'on peut confronter les bornes
d'erreur à ce qui se produit réellement.

::::{admonition} Objectifs d'apprentissage
:class: note

Après la lecture de ce chapitre, vous devriez pouvoir :
- écrire l'itération sur les valeurs ajustée pour un régresseur quelconque, et reconnaître la collocation et la projection de Galerkin comme deux choix de la règle d'ajustement;
- borner l'erreur de l'itération sur les valeurs ajustée, et celle de sa politique gloutonne, par l'erreur de chaque ajustement;
- construire une itération sur les valeurs ajustée qui diverge, et énoncer deux conditions qui empêchent la divergence : les moyenneurs en norme sup, et la pondération sur politique pour l'évaluation de politique;
- expliquer l'itération sur les politiques approchée comme la méthode de Newton appliquée à l'équation de Bellman approchée, et LSTD et LSPI comme son cas linéaire;
- nommer les étapes qui mènent de l'itération sur les valeurs ajustée à FQI, NFQ et NFQCA, et le chapitre où chacune est franchie.

**Préalables :** le gabarit des résidus pondérés ([chapitre précédent](weighted-residual-methods.md)), la contraction et la méthode de Newton-Kantorovitch des [processus de décision markoviens à horizon infini](infinite-horizon-mdps.md), l'interpolation par fonctions cardinales en [programmation dynamique à horizon fini](finite-horizon-dp.md).
::::

(sec-abe-fvi)=
## Itération sur les valeurs ajustée

L'itération sur les valeurs calcule $\Bellman v_k$ en chaque état et stocke le
résultat. Avec une fonction paramétrique, nous ne pouvons faire ni l'un ni
l'autre, mais nous pouvons faire deux choses qui, ensemble, jouent le même
rôle. Nous pouvons évaluer l'opérateur de Bellman en un nombre fini d'états
$s_1, \ldots, s_m$,

$$
y_i = (\Bellman v_k)(s_i) = \max_{a \in \mathcal{A}_{s_i}} \Big\{ r(s_i,a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s_i,a)\, v(s';\theta_k) \Big\},
$$

et nous pouvons choisir $\theta_{k+1}$ de sorte que $v(\cdot;\theta_{k+1})$
s'ajuste aux paires $(s_i, y_i)$. La première étape exige le modèle et la
fonction courante. La seconde est un problème d'apprentissage supervisé qui
ignore tout des équations de Bellman.

```{prf:algorithm} Itération sur les valeurs ajustée
:label: fitted-value-iteration

**Entrées :** états $\{s_i\}_{i=1}^m$ où calculer les cibles, facteur d'actualisation $\gamma$, classe de fonctions $\{v(\cdot;\theta)\}$, procédure d'ajustement $\mathtt{fit}$, tolérance $\epsilon$

**Sortie :** fonction de valeur approchée $v(\cdot;\theta) \approx v^*$

1. Initialiser $\theta_0$ arbitrairement
2. Poser $k \leftarrow 0$
3. **répéter**
4. $\quad$ **pour** chaque état $s_i$ **faire**
5. $\quad\quad$ Calculer la cible : $y_i \leftarrow \displaystyle\max_{a \in \mathcal{A}_{s_i}} \Big\{ r(s_i, a) + \gamma \sum_{s'} p(s' \mid s_i, a)\, v(s';\theta_k) \Big\}$
6. $\quad$ **fin pour**
7. $\quad$ Ajuster : $\theta_{k+1} \leftarrow \mathtt{fit}\big(\{(s_i, y_i)\}_{i=1}^m\big)$
8. $\quad$ $k \leftarrow k+1$
9. **jusqu'à ce que** $\|v(\cdot;\theta_k) - v(\cdot;\theta_{k-1})\| < \epsilon$ (ou que le nombre maximal d'itérations soit atteint)
10. **retourner** $\theta_k$
```

Notons $\Proj$ l'application qui envoie une fonction, par ses valeurs en
$s_1, \ldots, s_m$, sur la fonction ajustée. L'itération sur les valeurs
ajustée s'écrit alors

$$
v_{k+1} = \Proj \Bellman v_k,
$$ (eq:fitted-vi)

soit l'itération sur les valeurs projetée {eq}`eq:projected-vi` du chapitre
précédent, où $\Proj$ est maintenant une règle d'ajustement quelconque. Deux
choix définissent l'algorithme : où les cibles sont calculées, et comment la
fonction leur est ajustée. Rien ne dépend du maximum. Remplacer $\Bellman$ par
l'opérateur d'évaluation de politique $\BellmanPi$ donne l'évaluation de
politique ajustée, et le remplacer par l'opérateur lisse $\mathrm{L}_\beta$ de
{ref}`sec-regularized-operators` donne l'itération sur les valeurs lisse
ajustée.

Les deux ajustements du chapitre précédent marquent les extrémités d'une
gamme. La collocation calcule des cibles en exactement $n$ états, autant qu'il
y a de fonctions de base, et les interpole : $\Proj$ est l'opérateur
d'interpolation, et avec des fonctions cardinales linéaires par morceaux sur
une grille, l'algorithme est la version stationnaire de la récurrence à
rebours interpolée {eq}`eq:interpolated-bellman` du chapitre sur l'horizon
fini. La projection de Galerkin calcule des cibles en autant d'états que la
quadrature l'exige et leur ajuste la base par moindres carrés pondérés :
$\Proj$ est la projection $w$-orthogonale sur l'espace engendré par la base,
et lorsque les états sont échantillonnés selon $w$, l'ajustement est une
régression ordinaire. La collocation calcule peu de cibles et les reproduit
exactement; la régression en calcule beaucoup et les reproduit en moyenne.
C'est la régression qui passe à l'échelle, parce qu'elle découple le nombre de
cibles du nombre de paramètres, et parce qu'elle s'applique sans changement à
tout modèle qu'on peut ajuster à des données.

La pêcherie de {ref}`sec-wr-fishery` a exécuté la version par collocation avec
les fonctions chapeau et avec des polynômes de Tchebychev, et son balayage sur
$n$ a montré le polynôme divergeant pour certains $n$. La cellule ci-dessous
mesure ce qui arrive aux valeurs ajustées aux nœuds d'une itération à
l'autre. La maximisation sur l'échappement et la quadrature sur le choc sont
faites par le module qui a produit les figures du chapitre précédent.

```{code-cell} python
:tags: [hide-input]

import sys
from pathlib import Path

import numpy as np

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))
from fishery_tipping_point import Bellman, ChebyshevBasis, HatBasis, solve_exact

bellman = Bellman()                  # backups on a fine grid of escapements, 21 quadrature nodes
exact = solve_exact(bellman)         # reference solution


def collocation(basis, max_iter=2000):
    theta = np.zeros(basis.n)
    steps = []
    for k in range(max_iter):
        v = basis.function(theta)
        targets = bellman.apply(v, basis.nodes)        # t_i = (L v_theta)(s_i)
        new = basis.fit(targets)                       # solve Phi theta = t
        steps.append(np.abs(basis.function(new)(basis.nodes) - v(basis.nodes)).max())
        theta = new
        if steps[-1] < 1e-10 or steps[-1] > 1e8:
            break
    return basis.function(theta), np.array(steps)


print(f"{'basis':<30}{'step 10':>10}{'step 100':>11}{'last step':>11}"
      f"{'iterations':>12}{'rate':>7}{'max error':>11}")
for basis in (HatBasis(12), ChebyshevBasis(12), ChebyshevBasis(10)):
    v, steps = collocation(basis)
    rate = np.median(steps[50:150] / steps[49:149])
    error = np.abs(v(bellman.x) - exact.v).max()
    label = f"{basis.name}, n = {basis.n}"
    print(f"{label:<30}{steps[9]:>10.2e}{steps[99]:>11.2e}{steps[-1]:>11.2e}"
          f"{len(steps):>12}{rate:>7.3f}{error:>11.3g}")
```

Avec 12 coefficients, les deux bases convergent, et la variation d'une
itération à l'autre diminue d'un facteur $\gamma = 0.95$, comme dans
l'itération sur les valeurs. Elles convergent vers des points fixes
différents, avec des erreurs d'environ $0.5$ et $1.7$. Avec 10 coefficients de
Tchebychev, la variation croît plutôt d'environ 5 % par itération, et les
coefficients partent vers l'infini. La mise à jour de Bellman est encore
contractante d'un facteur $\gamma$. C'est l'étape d'interpolation qui dilate,
d'un facteur d'au moins $1.049/0.95 \approx 1.10$, assez pour annuler la
contraction. La section {ref}`sec-abe-convergence` explique quelles règles
d'ajustement peuvent produire cet effet et lesquelles ne le peuvent pas.

### Ajuster un réseau de neurones : cibles fixes et semi-gradient

Pour un modèle $v(\cdot;\boldsymbol\theta)$ non linéaire en ses paramètres,
comme un réseau de neurones, l'ajustement est

$$
\boldsymbol{\theta}_{k+1} \in \arg\min_{\boldsymbol{\theta}} \sum_{i=1}^m \big(v(s_i;\boldsymbol{\theta}) - y_i\big)^2, \qquad y_i = (\Bellman v_k)(s_i),
$$

résolu par descente de gradient. Sa condition du premier ordre est de nouveau
une condition d'orthogonalité, avec des fonctions test qui se déplacent avec
les paramètres :

$$
\sum_{i=1}^m \big(v(s_i;\boldsymbol{\theta}) - y_i\big) \frac{\partial v(s_i;\boldsymbol{\theta})}{\partial \theta_j} = 0 \quad \text{pour tout } j.
$$

Les dérivées $\partial v/\partial\theta_j$ engendrent l'espace tangent du
modèle aux paramètres courants. Pour un modèle linéaire, ce sont les fonctions
de base, et la condition est la condition de Galerkin échantillonnée. Les
cibles $y_i$ sont maintenues fixes pendant l'ajustement, de sorte que le
gradient ne dérive que $v(s_i;\boldsymbol\theta)$ et jamais l'opérateur. C'est
le **semi-gradient** de l'apprentissage par différence temporelle. Le critère
des moindres carrés de {ref}`sec-wr-conditions`, la minimisation du résidu de
Bellman, minimiserait plutôt
$\sum_i \big(v(s_i;\boldsymbol\theta) - (\Bellman v(\cdot;\boldsymbol\theta))(s_i)\big)^2$
avec les mêmes paramètres des deux côtés. Ses fonctions test,
$\partial v/\partial\theta_j - \partial(\Bellman v)/\partial\theta_j$, forment
le **gradient complet**, et les calculer exige de dériver à travers le
maximum, par le théorème de l'enveloppe ({prf:ref}`thm-envelope`), ou à travers
le log-sum-exp de l'[opérateur de Bellman lisse](regularized-dp.md). Les deux
critères ont des solutions différentes. Tous les algorithmes des chapitres à
venir, de NFQ à NFQCA, s'ajustent à des cibles fixes, et le
[chapitre sur les données historiques et en ligne](fitted-q-iteration.md)
étudie ce qui se passe lorsque l'ajustement est initialisé à chaud et arrêté
après quelques pas de gradient au lieu d'être mené jusqu'à convergence.

(sec-abe-high-dim)=
### Exemple : un problème d'arrêt en huit dimensions

La pêcherie n'a qu'une variable d'état, et une grille de quelques milliers de
nœuds la résout assez exactement pour servir de référence. Pour voir pourquoi
le choix de l'ajustement compte, considérons un état à $D$ coordonnées. Un
agent observe un indice $s \in [0,1]^D$, par exemple les prix de $D$ actifs.
S'arrêter rapporte la moyenne des coordonnées,
$\bar{s} = \frac{1}{D}\sum_\ell s_\ell$, et met fin au problème. Continuer ne
rapporte rien, et chaque coordonnée se déplace alors vers un nouveau tirage
uniforme, $s'_\ell = \rho s_\ell + (1-\rho)\xi_\ell$ avec $\xi_\ell \sim U[0,1]$
indépendants et $\rho = 0.7$. L'équation de Bellman est

$$
v^*(s) = \max\Big\{ \bar{s},\; \gamma\, \mathbb{E}\big[v^*(\rho s + (1-\rho)\xi)\big] \Big\}, \qquad \gamma = 0.9.
$$

La moyenne $\bar s$ est elle-même une chaîne de Markov,
$\bar{s}' = \rho\bar{s} + (1-\rho)\bar{\xi}$, de sorte que $v^*$ ne dépend de
$s$ que par $\bar s$ et peut être calculée exactement par une itération sur
les valeurs en une dimension. L'algorithme l'ignore. Une grille de 11 nœuds
par axe compte $11^D$ nœuds : 121 pour $D = 2$, 14 641 pour $D = 4$ et
$2.1 \times 10^8$ pour $D = 8$, chacun exigeant à chaque itération une
espérance sur un successeur de dimension $D$. L'itération sur les valeurs
ajustée calcule plutôt des cibles en 2 000 états tirés uniformément dans le
cube, quel que soit $D$, et les ajuste avec un ensemble d'arbres extrêmement
aléatoires {cite}`ernst2005tree` ou avec un réseau de neurones à deux couches
cachées. Les trois méthodes approchent l'espérance par la moyenne sur les mêmes
32 tirages de $\xi$; elles résolvent donc le même problème de moyenne
empirique, et le [chapitre suivant](monte-carlo-bellman-estimation.md) étudie
cette approximation. Les erreurs sont mesurées par rapport au $v^*$ exact en
5 000 nouveaux états.

```{code-cell} python
:label: fig-fvi-eight-dimensions
:caption: Itération sur les valeurs ajustée sur le problème d'arrêt à état de dimension huit. Chaque point est un état de test, placé selon la moyenne de ses coordonnées et la valeur que le réseau lui attribue; la courbe noire est la fonction de valeur exacte, qui ne dépend de l'état que par cette moyenne. On ne l'a jamais indiqué au réseau. Il a été ajusté en 2 000 états échantillonnés, alors qu'une grille de 11 nœuds par axe en compterait $2.1 \times 10^8$.
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

gamma, rho = 0.9, 0.7       # discount, persistence of each coordinate
N, M, K = 2000, 32, 40      # sampled states, successor draws, iterations

def exact_value(D, rng):
    """v* depends on s only through the mean of its coordinates: 1-D value iteration."""
    m = np.linspace(0, 1, 1001)
    eta = rng.uniform(size=(400, D)).mean(axis=1)          # mean of the shocks
    m_next = rho * m[:, None] + (1 - rho) * eta[None, :]
    w = m.copy()
    for _ in range(500):
        w = np.maximum(m, gamma * np.interp(m_next, m, w).mean(axis=1))
    return lambda S: np.interp(S.mean(axis=1), m, w)

def fitted_value_iteration(fit, S, xi):
    """Targets max{stop, gamma * E v(s')} at the states S, then a fit."""
    D = S.shape[1]
    S_next = (rho * S[:, None, :] + (1 - rho) * xi[None, :, :]).reshape(-1, D)
    v = lambda X: np.zeros(len(X))
    for _ in range(K):
        continuation = v(S_next).reshape(len(S), -1).mean(axis=1)
        y = np.maximum(S.mean(axis=1), gamma * continuation)
        v = fit(S, y)
    return v

def fit_grid(nodes):
    def fit(S, y):            # S is the grid itself, y the values at its nodes
        D = S.shape[1]
        table = RegularGridInterpolator([nodes] * D, y.reshape([len(nodes)] * D))
        return lambda X: table(np.clip(X, 0, 1))
    return fit

def fit_trees(S, y):
    model = ExtraTreesRegressor(n_estimators=20, min_samples_leaf=3, random_state=0)
    return model.fit(S, y).predict

def fit_network(S, y):
    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=0)
    return model.fit(S, y).predict

rng = np.random.default_rng(0)
nodes = np.linspace(0, 1, 11)
rows = []
for D in [2, 4, 8]:
    v_star = exact_value(D, rng)
    test = rng.uniform(size=(5000, D))
    xi = rng.uniform(size=(M, D))
    S = rng.uniform(size=(N, D))
    errors = {}
    if len(nodes)**D <= 20000:
        grid = np.stack(np.meshgrid(*[nodes] * D, indexing="ij"), axis=-1).reshape(-1, D)
        v = fitted_value_iteration(fit_grid(nodes), grid, xi)
        errors["grid"] = np.abs(v(test) - v_star(test))
    for name, fit in [("trees", fit_trees), ("network", fit_network)]:
        v = fitted_value_iteration(fit, S, xi)
        errors[name] = np.abs(v(test) - v_star(test))
        if D == 8 and name == "network":
            scatter = (test.mean(axis=1), v(test), v_star(test))
    rows.append((D, len(nodes)**D, errors))

print("Mean absolute error (maximum in parentheses) against the exact value")
print(f"{'D':>2} {'grid nodes':>12}  {'grid':^16} {'trees, 2000 states':^20} {'network, 2000 states':^20}")
for D, n_nodes, e in rows:
    cell = lambda key: f"{e[key].mean():.3f} ({e[key].max():.3f})" if key in e else "out of reach"
    print(f"{D:>2} {n_nodes:>12,}  {cell('grid'):^16} {cell('trees'):^20} {cell('network'):^20}")

from book_figure_theme import INK, PALETTE, figure_style, show

mean_s, v_hat, v_true = scatter
order = np.argsort(mean_s)
with figure_style():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.scatter(mean_s, v_hat, s=4, alpha=0.25, color=PALETTE["blue"], edgecolor="none")
    ax.plot(mean_s[order], v_true[order], color=INK, lw=1.4)
    ax.text(0.2, 0.74, "network, $D = 8$", color=PALETTE["blue"], fontsize=8.5)
    ax.text(0.72, 0.6, "exact $v^*$", color=INK, fontsize=8.5)
    ax.set_xlabel(r"average of the coordinates, $\bar s$")
    ax.set_ylabel("value")
show(fig)
```

La grille est la méthode la plus précise tant qu'elle tient en mémoire, et
elle est hors de portée pour $D = 8$. Les arbres et le réseau emploient les
mêmes 2 000 états en toute dimension. L'erreur moyenne des arbres double à peu
près chaque fois que $D$ double, tandis que celle du réseau reste proche de
0.01; la figure montre que le réseau a trouvé l'unique direction le long de
laquelle la valeur change. Les deux ont des erreurs maximales plusieurs fois
supérieures à leur moyenne, aux coins du cube, où se trouvent peu d'états
échantillonnés. Cet écart donne un premier aperçu de la question qui domine
l'analyse ci-dessous et les méthodes par lots du
[chapitre sur les données historiques et en ligne](fitted-q-iteration.md) : un
ajustement n'est contrôlé que là où il y a des données.

(sec-abe-error)=
## Propagation des erreurs d'ajustement

Chaque ajustement n'est qu'approximativement juste. Cette section borne
l'écart entre $v^*$ et le point où aboutit l'itération sur les valeurs ajustée,
ainsi que la qualité de la politique renvoyée.

Le chapitre sur l'horizon fini a répondu à cette question pour la récurrence à
rebours avec interpolation : avec des poids non négatifs, l'erreur des valeurs
stockées est au plus la somme des erreurs d'interpolation des stades
ultérieurs ({prf:ref}`prop-interp-error-accumulation`). Sur un horizon infini,
la somme devient une série géométrique.

```{prf:theorem} Erreur de l'itération sur les valeurs approchée
:label: thm-approximate-vi

Soit $v_0$ bornée et soit une suite $v_k$ telle que $\|v_{k+1} - \Bellman v_k\|_\infty \leq \varepsilon$ pour tout $k$. Alors

$$
\|v_k - v^*\|_\infty \leq \gamma^k \|v_0 - v^*\|_\infty + \frac{\varepsilon}{1-\gamma}.
$$
```

```{prf:proof}
Puisque $v^* = \Bellman v^*$ et que $\Bellman$ est une $\gamma$-contraction,

$$
\|v_{k+1} - v^*\|_\infty \leq \|v_{k+1} - \Bellman v_k\|_\infty + \|\Bellman v_k - \Bellman v^*\|_\infty \leq \varepsilon + \gamma \|v_k - v^*\|_\infty.
$$

Dérouler la récurrence donne $\|v_k - v^*\|_\infty \leq \gamma^k\|v_0 - v^*\|_\infty + \varepsilon(1 + \gamma + \cdots + \gamma^{k-1})$.
```

La fonction de valeur est un instrument. Ce que nous employons est la
politique gloutonne, et sa perte est bornée par l'erreur de la fonction de
valeur pour laquelle elle est gloutonne {cite}`SinghYee1994`.

```{prf:proposition} Perte d'une politique gloutonne
:label: prop-greedy-loss

Si $\pi$ est gloutonne par rapport à $v$, c'est-à-dire $\BellmanPi v = \Bellman v$, alors

$$
\|v^* - v_\pi\|_\infty \leq \frac{2\gamma}{1-\gamma}\|v - v^*\|_\infty.
$$
```

```{prf:proof}
Soit $\pi^*$ une politique optimale. Alors $v^* - v_\pi = \mathrm{L}_{\pi^*} v^* - \BellmanPi v_\pi$, que nous décomposons ainsi :

$$
v^* - v_\pi = (\mathrm{L}_{\pi^*} v^* - \mathrm{L}_{\pi^*} v) + (\mathrm{L}_{\pi^*} v - \BellmanPi v) + (\BellmanPi v - \BellmanPi v^*) + (\BellmanPi v^* - \BellmanPi v_\pi).
$$

Le deuxième terme est non positif parce que $\pi$ est gloutonne pour $v$. Les autres valent $\gamma\mathbf{P}_{\pi^*}(v^* - v)$, $\gamma\mathbf{P}_\pi(v - v^*)$ et $\gamma\mathbf{P}_\pi(v^* - v_\pi)$, et les matrices stochastiques n'augmentent pas la norme sup. Puisque $v^* - v_\pi \geq 0$, prendre le maximum sur les états donne $\|v^* - v_\pi\|_\infty \leq 2\gamma\|v - v^*\|_\infty + \gamma\|v^* - v_\pi\|_\infty$.
```

En combinant les deux, les politiques gloutonnes de l'itération sur les
valeurs ajustée finissent par se trouver à au plus
$2\gamma\varepsilon/(1-\gamma)^2$ de l'optimum. La borne n'exige rien de
l'ajustement sauf sa précision : aucune linéarité, aucune base particulière,
aucune structure. C'est ce qui autorise un ensemble d'arbres ou un réseau de
neurones à la place d'une table. Elle indique aussi où se situe la difficulté,
en trois endroits que le reste de ce chapitre et les suivants abordent.

1. L'erreur est mesurée en norme sup, en chaque état. Une régression contrôle
   une erreur moyenne sur les états où les cibles ont été calculées, et rien
   aux états qu'elle n'a jamais vus. L'analyse de l'itération sur les valeurs
   ajustée par {cite:t}`Munos2007` et {cite:t}`MunosSzepesvari2008` remplace
   la norme sup par des normes pondérées. Le prix à payer est un coefficient de
   concentrabilité, qui mesure à quel point les distributions d'états
   qu'atteignent les politiques peuvent s'éloigner de la distribution des
   données. C'est la forme théorique du problème de couverture de
   l'apprentissage par renforcement par lots et hors ligne.
2. L'ajustement doit être réalisable. La borne exige un $\theta$ tel que
   $v(\cdot;\theta)$ soit proche de $\Bellman v_k$ pour chaque $v_k$ que
   produit l'algorithme. La classe de fonctions doit donc être presque fermée
   sous l'opérateur de Bellman, et non simplement capable de représenter
   $v^*$.
3. L'erreur doit rester bornée. Le théorème suppose un même $\varepsilon$ pour
   toutes les itérations. Si l'erreur d'ajustement croît avec $v_k$, la borne
   ne dit rien, et la section suivante montre que cela se produit.

(sec-abe-mallard)=
### Exemple : fixer la réglementation de la chasse aux canards

On peut mettre les bornes à l'épreuve sur un problème qu'un organisme
gouvernemental résout chaque année. Chaque été, le U.S. Fish and Wildlife
Service choisit la réglementation de la chasse aux canards pour les voies
migratoires du Mississippi et du Centre parmi quatre forfaits : une saison
fermée, ou une saison restrictive, modérée ou libérale, qui diffèrent par la
durée de la saison et la limite de prises. Depuis 1995, le choix pour le
canard colvert provient d'une politique calculée par programmation dynamique
stochastique, dans le cadre d'un programme appelé gestion adaptative des
prises {cite}`Johnson1997`. L'état se compose de deux nombres tirés du relevé
de mai : la population nicheuse $N$ de canards colverts du centre du continent
et le nombre $P$ d'étangs dans les Prairies canadiennes, tous deux en
millions. Les étangs suivent un processus autorégressif piloté par la météo.
La population réagit aux étangs et aux prises selon quatre modèles. Ceux-ci ne
s'entendent pas sur la question de savoir si les morts dues à la chasse
s'ajoutent aux morts naturelles ou les remplacent, ni sur l'intensité avec
laquelle la reproduction diminue à mesure que la population croît, et le
Service les pondère selon la qualité avec laquelle chacun a prédit les relevés
passés. La récompense est la récolte, dévaluée proportionnellement lorsque la
population devrait tomber sous l'objectif de 8.5 millions d'oiseaux, et la
saison doit être fermée sous 4.75 millions. Nous tirons ces équations et ces
nombres du rapport du Service pour la saison 2012 {cite}`USFWS2012AHM`, et
nous actualisons les récoltes futures avec $\gamma = 0.98$. Le module du modèle
énumère les quelques détails que le rapport laisse ouverts.

Le Service résout ce problème comme le chapitre sur l'horizon fini a résolu
celui du réservoir : sur une grille d'états, avec interpolation entre les
nœuds. Nous faisons de même avec des pas de 0.1 million d'oiseaux et d'étangs,
12 231 nœuds et une interpolation bilinéaire. L'interpolation bilinéaire est
un moyenneur; l'itération sur les valeurs sur la grille converge donc
({prf:ref}`santos-vigo-aguiar-stability` ci-dessous), et doubler le pas modifie
les valeurs de moins de 0.05. La {numref}`fig-mallard-policies` compare le
résultat à la table que le Service a publiée pour 2012. Les deux politiques
prescrivent le même forfait dans 126 des 150 cellules de la table et diffèrent
d'un forfait dans les autres. La politique publiée réagit plus fortement aux
étangs que la nôtre.

:::{figure} _static/mallard_harvest/policies.svg
:label: fig-mallard-policies

Forfaits de réglementation en fonction des étangs de mai et de la population
nicheuse. (a) La table publiée par le U.S. Fish and Wildlife Service pour 2012.
(b) La politique gloutonne de notre solution de référence, sur une grille à
pas de 0.1. (c) La politique gloutonne de l'itération sur les valeurs ajustée
avec un réseau de neurones entraîné sur 2 000 états échantillonnés. Sous la
ligne pointillée, la saison doit être fermée.
:::

La solution de référence nous permet de confronter
{prf:ref}`thm-approximate-vi` et {prf:ref}`prop-greedy-loss` à ce qui se
produit réellement. Nous exécutons l'itération sur les valeurs ajustée pendant
400 itérations avec cinq ajustements. Trois sont des interpolations bilinéaires
sur des grilles plus grossières, à pas de 2, 1 et 0.5. Les deux autres sont des
régresseurs entraînés, à chaque itération, sur des cibles aux mêmes 2 000 états
tirés uniformément au hasard : des arbres extrêmement aléatoires, qui sont des
moyenneurs, et un réseau à deux couches cachées de 64 unités, initialisé à
chaud à partir de l'itération précédente. Pour chaque ajustement, nous mesurons
$\varepsilon$, la plus grande erreur d'ajustement
$|v_{k+1}(s) - (\Bellman v_k)(s)|$ sur la grille de référence au cours des
dernières itérations, et nous comparons les deux bornes à l'erreur de valeur et
à la perte de la politique que nous calculons par rapport à la référence.

:::{include} artifacts/mallard_harvest/results.md
:::

Le tableau appuie trois observations.

1. Les deux bornes sont vérifiées, et toutes deux sont lâches. L'erreur de
   valeur est de 18 à 118 fois plus petite que $\varepsilon/(1-\gamma)$, et la
   perte de la politique est de quatre à cinq ordres de grandeur sous
   $2\gamma\varepsilon/(1-\gamma)^2$. Les bornes imputent l'erreur de chaque
   itération à son pire état et laissent les erreurs s'additionner dans la même
   direction pendant $1/(1-\gamma) = 50$ ans. Ni l'un ni l'autre ne se produit
   ici.
2. La politique est beaucoup plus précise que la fonction de valeur. La grille
   la plus grossière compte 45 nœuds et une erreur de valeur de 2.2 millions
   d'oiseaux, et pourtant elle choisit le forfait de référence dans 96.5 % des
   états, et sa pire perte de politique est de 0.10 million d'oiseaux, environ
   un quart de pour cent de la valeur. Une décision dépend des différences
   entre les valeurs des forfaits, et une grande partie de l'erreur
   d'ajustement est commune aux quatre forfaits, parce que leurs états
   successeurs sont proches.
3. Les régresseurs font à peu près aussi bien que la grille la plus grossière.
   Avec 2 000 cibles par itération, les arbres et le réseau ont des erreurs
   d'ajustement de $0.76$ et $1.47$, du même ordre que les $0.79$ de la grille
   à 45 nœuds. La politique gloutonne du réseau reproduit la carte de
   référence (panneau (c)) : elle choisit le forfait de référence dans 98 % des
   états et perd au plus $0.03$. Les arbres perdent davantage, $0.26$ au pire.
   Leur ajustement est constant par morceaux; il est donc plat là où la valeur
   a une pente, et le choix glouton entre deux forfaits dépend de cette pente.

Avec deux variables d'état, la grille du Service est le bon outil, et les
régresseurs n'apportent rien. Ils deviennent nécessaires lorsque l'état a de
nombreuses dimensions, comme dans le problème d'arrêt en huit dimensions, ou
lorsqu'il n'existe aucun modèle à partir duquel calculer les cibles, comme dans
l'exemple d'évitement de collisions du
[chapitre sur les données historiques et en ligne](fitted-q-iteration.md).

- {download}`Modèle de récolte du canard colvert et solveurs <code/mallard_harvest.py>`
- {download}`Générateur de l'expérience sur la récolte du canard colvert <scripts/build_mallard_harvest_artifacts.py>`
- {download}`Résultats sur la récolte du canard colvert <artifacts/mallard_harvest/metrics.json>`

(sec-abe-convergence)=
## Divergence et ajustements stables

Un ajustement qui représente exactement la solution peut tout de même faire
diverger l'itération sur les valeurs. Cette section montre comment, et quelles
règles d'ajustement l'excluent.

### Un contre-exemple à deux états

L'exemple suivant est dû à {cite:t}`TsitsiklisVanRoy1996`. Il y a deux états et
une action. L'état 1 passe à l'état 2, l'état 2 reste où il est, et toutes les
récompenses sont nulles, de sorte que $v^* = 0$. La fonction de valeur est
représentée par une seule caractéristique, $v(1;\theta) = \theta$ et
$v(2;\theta) = 2\theta$, ce qui contient $v^*$ pour $\theta = 0$. L'itération
sur les valeurs ajustée calcule les cibles aux deux états et les ajuste par
moindres carrés à poids égaux. Les deux cibles valent
$\gamma v(2;\theta_k) = 2\gamma\theta_k$, et l'ajustement minimise
$(\theta - 2\gamma\theta_k)^2 + (2\theta - 2\gamma\theta_k)^2$, ce qui donne

$$
\theta_{k+1} = \frac{6\gamma}{5}\theta_k.
$$

Pour $\gamma > 5/6$, les itérés divergent à partir de tout $\theta_0 \neq 0$.
L'opérateur de Bellman réduit la différence entre $v_k$ et $v^*$ d'un facteur
$\gamma$, et l'ajustement l'étire ensuite d'un facteur $6/5$ : la cible est la
constante $2\gamma\theta_k$, et la meilleure droite passant par l'origine la
dépasse à l'état 2, où elle prédit $\frac{12}{5}\gamma\theta_k$. L'erreur
d'ajustement croît proportionnellement à $\theta_k$, ce qui explique pourquoi
{prf:ref}`thm-approximate-vi` ne s'applique pas. {cite:t}`BoyanMoore1995` ont
observé le même échec avec la régression polynomiale et les réseaux de neurones
sur de petits problèmes de commande, à une époque où TD-Gammon avait donné
l'impression que les fonctions de valeur neuronales étaient faciles.

### Moyenneurs : des ajustements qui ne peuvent pas dilater

Le contre-exemple invite à se demander quelles règles d'ajustement ne peuvent
pas étirer les différences entre fonctions. La réponse a été trouvée de façon
indépendante en économie computationnelle
{cite}`Judd1992,Judd1996,McGrattan1997,SantosVigoAguiar1998`, en dynamique
économique {cite}`Stachurski2009` et en apprentissage par renforcement
{cite}`Gordon1995,Gordon1999`, et ces communautés sont arrivées à la même
condition sous des noms différents.

```{prf:definition} Moyenneur
:label: gordon-averager

Un opérateur linéaire $\Proj$ sur l'espace des fonctions réelles bornées sur $\mathcal{S}$ est un **moyenneur** s'il est monotone, c'est-à-dire que $f \leq g$ point par point implique $\Proj f \leq \Proj g$, et s'il préserve les constantes, $\Proj\mathbf{1} = \mathbf{1}$. Sur un espace d'états fini, $\Proj$ est un moyenneur exactement lorsque $\Proj v = Wv$ pour une matrice stochastique $W$ : $w_{ij} \geq 0$ et $\sum_j w_{ij} = 1$ pour chaque ligne $i$.
```

Le nom est de Gordon. La caractérisation dans le cas d'un nombre fini d'états
découle du fait qu'une application linéaire est monotone exactement lorsque sa
matrice a des entrées non négatives, et préserve les constantes exactement
lorsque ses lignes ont pour somme un. Un opérateur d'interpolation
$(\Proj f)(s) = \sum_j \ell_j(s) f(s_j)$ à fonctions cardinales $\ell_j$ est
un moyenneur exactement lorsque les poids sont non négatifs et ont pour somme
un, ce qui est la condition $\Lambda = 1$ du chapitre sur l'horizon fini et la
propriété qui a séparé les fonctions chapeau du polynôme dans la pêcherie.

```{prf:proposition} Les moyenneurs sont non expansifs
:label: monotone-nonexpansive

Si $\Proj$ est un moyenneur, alors $\|\Proj f - \Proj g\|_\infty \leq \|f - g\|_\infty$ pour toutes fonctions bornées $f, g$.
```

```{prf:proof}
Soit $M = \|f - g\|_\infty$. Alors $-M \leq f(s) - g(s) \leq M$ pour tout $s$, ce qui peut s'écrire $g - M\mathbf{1} \leq f \leq g + M\mathbf{1}$. Par monotonie, $\Proj(g - M\mathbf{1}) \leq \Proj f \leq \Proj(g + M\mathbf{1})$. Par linéarité et préservation des constantes, $\Proj g - M\mathbf{1} \leq \Proj f \leq \Proj g + M\mathbf{1}$, ce qui signifie $|\Proj f(s) - \Proj g(s)| \leq M$ pour tout $s$.
```

Un moyenneur agit comme une moyenne pondérée : il respecte l'ordre et ne peut
pas amplifier les différences. L'ajustement par moindres carrés du
contre-exemple n'en est pas un. Il envoie la fonction constante $(1, 1)$ sur
$(3/5, 6/5)$; il ne préserve donc pas les constantes, et il ne respecte pas non
plus l'ordre. L'interpolation polynomiale aux nœuds de Tchebychev préserve les
constantes mais n'est pas monotone : ses fonctions cardinales passent sous
zéro entre les nœuds, et sa constante de Lebesgue vaut environ $2.4$ pour dix
nœuds. C'est ainsi que le polynôme à dix coefficients de la pêcherie
({ref}`sec-abe-fvi`) a pu annuler la contraction de la mise à jour de Bellman,
alors que les fonctions chapeau, dont les fonctions cardinales sont non
négatives, ne le pouvaient pas.

```{prf:theorem} L'itération sur les valeurs ajustée avec un moyenneur converge
:label: santos-vigo-aguiar-stability

Soit $\Bellman$ une $\gamma$-contraction pour la norme sup sur les fonctions bornées sur $\mathcal{S}$, et soit $\Proj$ un moyenneur. Alors $\Proj \Bellman$ est une $\gamma$-contraction; l'itération sur les valeurs ajustée $v_{k+1} = \Proj \Bellman v_k$ converge donc à partir de tout $v_0$ vers l'unique point fixe $v_\Proj \in \text{Im}(\Proj)$, et

$$
\|v_\Proj - v^*\|_\infty \leq \frac{1}{1-\gamma} \|\Proj v^* - v^*\|_\infty .
$$
```

```{prf:proof}
Par {prf:ref}`monotone-nonexpansive`, $\Proj$ est non expansif. Puisque $\Bellman$ est une $\gamma$-contraction,

$$
\|\Proj \Bellman f - \Proj \Bellman g\|_\infty \leq \|\Bellman f - \Bellman g\|_\infty \leq \gamma\|f-g\|_\infty,
$$

de sorte que $\Proj \Bellman$ est une $\gamma$-contraction, et {prf:ref}`banach-fixed-point` donne l'unicité du point fixe et la convergence globale. Pour la borne d'erreur, on utilise $v_\Proj = \Proj\Bellman v_\Proj$ et $v^* = \Bellman v^*$ :

$$
\|v_\Proj - v^*\|_\infty \leq \|\Proj\Bellman v_\Proj - \Proj\Bellman v^*\|_\infty + \|\Proj \Bellman v^* - v^*\|_\infty \leq \gamma \|v_\Proj - v^*\|_\infty + \|\Proj v^* - v^*\|_\infty,
$$

et réarranger donne le résultat.
```

Le théorème est dû à {cite:t}`SantosVigoAguiar1998` pour les opérateurs
d'approximation monotones et à {cite:t}`Gordon1995` pour les moyenneurs; la
démonstration de la proposition suit {cite:t}`Stachurski2009`. L'erreur au
point fixe est l'erreur de meilleure approximation $\|\Proj v^* - v^*\|_\infty$
amplifiée par $(1-\gamma)^{-1}$. Pour un moyenneur, l'erreur par itération
reste bornée; le résultat est donc le cas particulier de
{prf:ref}`thm-approximate-vi` dans lequel la troisième difficulté ci-dessus ne
peut pas survenir.

La matrice stochastique donne à la moyenne une lecture en termes de chaîne de
Markov : $\Proj v$ est une espérance après une transition aléatoire, et les
espérances préservent l'ordre. Le chapitre sur l'horizon fini a fait la même
observation à propos de l'interpolation linéaire, qui remplace un successeur
déterministe par un nœud aléatoire de la grille, avec les poids
d'interpolation comme probabilités. Composer cette chaîne avec les transitions
propres au modèle fait de $\Proj\Bellman$ l'opérateur de Bellman d'un processus
de décision markovien à états finis, ce qui explique sa contraction.

Quels régresseurs sont des moyenneurs? L'agrégation d'états, les $k$ plus
proches voisins, le lissage par noyaux non négatifs normalisés pour avoir une
somme égale à un, et l'interpolation multilinéaire sur une grille prédisent
tous une combinaison convexe de cibles d'entraînement dont les poids ne
dépendent que des états. C'est aussi le cas d'un arbre de régression, qui
prédit la moyenne des cibles d'une feuille, pourvu que sa partition de l'espace
d'états ne dépende pas des cibles. {cite:t}`ernst2005tree` s'en sont servis
pour donner une garantie de convergence à l'itération Q ajustée à base
d'arbres. Leurs arbres extrêmement aléatoires choisissent les coupures à l'aide
des cibles, et figer la structure des arbres après un certain nombre
d'itérations rétablit la garantie. Les moindres carrés linéaires,
l'interpolation polynomiale de degré élevé et les réseaux de neurones ne sont
pas des moyenneurs.

| **Méthode** | **Moyenneur?** | **Remarques** |
|:-----------|:--------------|:----------|
| Interpolation linéaire par morceaux | Oui | Les poids barycentriques sont non négatifs et de somme un |
| Interpolation multilinéaire (grille) | Oui | Les poids sont des produits de fonctions chapeau unidimensionnelles |
| Splines préservant la forme (Schumaker) | Oui | Conçues pour maintenir la monotonie |
| Agrégation d'états, $k$ plus proches voisins | Oui | Poids égaux dans un groupe ou un voisinage |
| Lissage par noyaux (noyaux non négatifs) | Oui | Si les poids sont normalisés pour avoir une somme égale à un |
| Arbres et forêts de régression, partition fixe | Oui | Moyennes par feuille; garantie perdue si les coupures dépendent des cibles |
| Interpolation polynomiale de degré élevé | Non | Poids négatifs (phénomène de Runge) |
| Moindres carrés sur une base quelconque | Non | Le contre-exemple ci-dessus |
| Réseaux de neurones | Non | Aucune contrainte de signe ni de normalisation sur l'ajustement |

Le tableau situe les ajustements des deux chapitres. La collocation avec
interpolation linéaire par morceaux est un moyenneur, et la sagesse populaire
selon laquelle l'interpolation linéaire est sûre pour l'itération sur les
valeurs est un théorème. La collocation avec un polynôme de degré élevé n'en est
pas un, et sa convergence dans la pêcherie, lorsqu'elle s'est produite,
n'était pas garantie. La projection de Galerkin et les moindres carrés ne sont
pas des moyenneurs. L'ensemble d'arbres de l'exemple en huit dimensions se
situe à la frontière, et le réseau est à l'extérieur. Le solveur compte aussi :
la méthode de Newton n'itère jamais $\Proj\Bellman$, de sorte qu'elle n'a pas
besoin de la contraction; c'est pourquoi la stratégie hybride de
{ref}`sec-wr-solve` commence par un ajustement monotone de faible ordre et passe
ensuite à la méthode de Newton avec l'ajustement précis.

```{admonition} Exercices : divergence et moyenneurs
:class: hint dropdown

1. **Le contre-exemple avec un moyenneur.** Remplacez l'ajustement par moindres carrés de l'exemple à deux états par une agrégation d'états en un seul groupe. Écrivez $\Proj$ sous forme de matrice, vérifiez qu'il s'agit d'un moyenneur, et montrez que l'itération sur les valeurs ajustée converge pour tout $\gamma < 1$. Quel est son point fixe?

2. **Un exemple non monotone.** À l'aide de l'interpolation de Lagrange avec 4 nœuds équidistants sur $[0, 1]$, calculez les poids d'interpolation au point $x = 0.1$. Montrez que certains poids sont négatifs.

3. **Vérifier la monotonie.** Considérez l'interpolation linéaire par morceaux sur une grille de 5 points. Écrivez les poids d'interpolation pour un point situé entre les points 2 et 3 de la grille. Vérifiez que tous les poids sont non négatifs et ont pour somme un.

4. **Arbres.** Montrez qu'un arbre de régression à partition fixe est un moyenneur. Expliquez pourquoi un arbre qui choisit ses coupures d'après les cibles peut changer de partition d'une itération à l'autre, et quelle étape de la démonstration de {prf:ref}`santos-vigo-aguiar-stability` cela rompt.
```

### La pondération sur politique sauve l'évaluation de politique

Le contre-exemple se prête à une seconde lecture. La chaîne passe de l'état 1 à
l'état 2 et y reste; sa distribution stationnaire place donc toute sa masse
sur l'état 2. Si l'ajustement pondère chaque état par cette distribution, il
minimise $(2\theta - 2\gamma\theta_k)^2$ seul, et
$\theta_{k+1} = \gamma\theta_k$ converge. La pondération égale était le
problème : elle donne à l'état que le processus quitte aussitôt le même poids
qu'à l'état où il demeure.

Cette observation est un théorème pour l'évaluation de politique
{cite}`TsitsiklisVanRoy1997`, démontré pour expliquer la convergence de TD(0)
avec des caractéristiques linéaires. Prenons un espace d'états fini
$\mathcal{S} = \{s_1,\ldots,s_m\}$, une matrice de caractéristiques
$\boldsymbol{\Phi} \in \mathbb{R}^{m\times n}$ de lignes
$\boldsymbol\varphi(s)^\top$, et une distribution $\xi$ sur les états avec
$\boldsymbol{\Xi} = \text{diag}(\xi)$. Le produit scalaire, la norme et la
projection par moindres carrés sur l'espace engendré par les caractéristiques,
pondérés par $\xi$, sont

$$
\langle u, v \rangle_\xi = u^\top \boldsymbol{\Xi} v, \qquad \|v\|_\xi = \sqrt{v^\top \boldsymbol{\Xi} v}, \qquad \Proj = \boldsymbol{\Phi}(\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \boldsymbol{\Xi}.
$$ (eq:xi-projection)

L'évaluation de politique ajustée est $v_{k+1} = \Proj\BellmanPi v_k$ avec
$\BellmanPi v = \mathbf{r}_\pi + \gamma\mathbf{P}_\pi v$. En apprentissage par
renforcement, la pondération n'est pas un libre choix : les états présents
dans les données sont ceux que le processus visite, et si la chaîne sous $\pi$
est ergodique, leur distribution tend vers la distribution stationnaire $\xi$
telle que $\xi^\top\mathbf{P}_\pi = \xi^\top$. Supposons que ce soit la
pondération. Nous montrons que $\Proj$ est non expansif et que $\BellmanPi$ est
une $\gamma$-contraction, tous deux pour $\|\cdot\|_\xi$, de sorte que

$$
\|\Proj \BellmanPi v - \Proj \BellmanPi w\|_\xi \leq \|\BellmanPi v - \BellmanPi w\|_\xi \leq \gamma \|v - w\|_\xi.
$$

La projection orthogonale est non expansive. Pour tout vecteur $v$,
décomposons $v = \Proj v + (v - \Proj v)$, où $v - \Proj v$ est
$\xi$-orthogonal à l'espace engendré par $\boldsymbol\Phi$. Par le théorème de
Pythagore,

$$
\|v\|_\xi^2 = \|\Proj v\|_\xi^2 + \|v - \Proj v\|_\xi^2 \geq \|\Proj v\|_\xi^2.
$$

Aucune condition de signe sur les entrées de $\Proj$ n'a été nécessaire.
L'argument en norme sup échoue pour cette projection; l'argument en norme
$\xi$ ne voit pas du tout les poids négatifs.

$\BellmanPi$ est contractant pour $\|\cdot\|_\xi$. Puisque
$\BellmanPi v - \BellmanPi w = \gamma \mathbf{P}_\pi (v - w)$, il suffit de
montrer que $\|\mathbf{P}_\pi z\|_\xi \leq \|z\|_\xi$ pour tout $z$.

*Avant de poursuivre, essayez de le montrer vous-même. Indice : quelle
propriété de $\xi$ la relie à $\mathbf{P}_\pi$?*

La $s$-ième composante de $\mathbf{P}_\pi z$ est
$\sum_{s'} p(s'|s,\pi(s)) z(s')$, une moyenne des valeurs $z(s')$. Comme
$x \mapsto x^2$ est convexe, l'inégalité de Jensen donne
$\big[\sum_{s'} p(s'|s,\pi(s)) z(s')\big]^2 \leq \sum_{s'} p(s'|s,\pi(s)) z(s')^2$.
En pondérant par $\xi$ et en sommant sur $s$,

$$
\|\mathbf{P}_\pi z\|_\xi^2 \leq \sum_s \xi(s) \sum_{s'} p(s'|s,\pi(s)) z(s')^2 = \sum_{s'} z(s')^2 \sum_s \xi(s) p(s'|s,\pi(s)) = \sum_{s'} \xi(s') z(s')^2 = \|z\|_\xi^2,
$$

où l'avant-dernière étape est la stationnarité. Ainsi, $\Proj\BellmanPi$ est
une $\gamma$-contraction pour $\|\cdot\|_\xi$, et par
{prf:ref}`banach-fixed-point`, l'évaluation de politique ajustée converge à
partir de tout point de départ vers l'unique solution de l'**équation de
Bellman projetée**

$$
v = \Proj \BellmanPi v.
$$ (eq:projected-bellman-pi)

Sa solution est le point fixe vers lequel converge TD(0) avec des
caractéristiques linéaires. L'argument de
{prf:ref}`santos-vigo-aguiar-stability` borne sa distance à $v_\pi$ par
$(1-\gamma)^{-1}\|\Proj v_\pi - v_\pi\|_\xi$, et la structure de produit
scalaire améliore la constante en $(1-\gamma^2)^{-1/2}$
{cite}`TsitsiklisVanRoy1997`.

La convergence est venue de l'accord entre la pondération et l'opérateur, et
non d'une propriété de l'ajustement autre que l'orthogonalité : $\xi$ doit être
la distribution stationnaire de la matrice $\mathbf{P}_\pi$ même qui apparaît
dans $\BellmanPi$. Lorsque les données sont recueillies en suivant $\pi$, cela
est automatiquement vérifié : c'est l'apprentissage **sur politique**.
Lorsqu'elles proviennent d'une autre politique de comportement, comme dans la
pondération égale du contre-exemple ou dans l'apprentissage à partir d'un
tampon de rejeu rempli par des politiques plus anciennes, l'étape de Jensen
échoue, $\mathbf{P}_\pi$ n'est pas nécessairement non expansif pour
$\|\cdot\|_\xi$, et $\Proj\BellmanPi$ peut ne pas être contractant. Le
contre-exemple de Baird {cite}`Baird1995` en est la démonstration classique.

```{admonition} Exercices : la condition sur politique
:class: hint dropdown

1. **Pondérer le contre-exemple.** Dans l'exemple à deux états, faites l'ajustement avec les poids $(\xi_1, \xi_2)$. Montrez que $\theta_{k+1} = 2\gamma\theta_k(\xi_1 + 2\xi_2)/(\xi_1 + 4\xi_2)$ et trouvez l'ensemble des pondérations pour lesquelles l'itération converge pour tout $\gamma < 1$.

2. **La non-expansivité échoue hors politique.** Pour la chaîne à 3 états de {ref}`sec-abe-lstd` et la pondération $\xi' = (0.9, 0.05, 0.05)$, trouvez un vecteur $z$ tel que $\|\mathbf{P}_\pi z\|_{\xi'} > \|z\|_{\xi'}$.

3. **Perdre l'argument n'est pas perdre la convergence.** Implémentez l'évaluation de politique ajustée pour la même chaîne, avec les caractéristiques $\varphi(s) = (1, s)$ et la pondération $\xi'$. Elle converge encore : calculez le rayon spectral de $\gamma \Proj \mathbf{P}_\pi$ et vérifiez qu'il est égal à $\gamma$ pour cet ensemble de caractéristiques, quelle que soit la pondération. Ne gardez ensuite que la caractéristique $\varphi(s) = s$ et trouvez une pondération concentrée sur l'état 1 pour laquelle le rayon spectral dépasse un.
```

### L'opérateur d'optimalité et la triade mortelle

L'argument pondéré ne s'étend pas à l'opérateur d'optimalité
$\Bellman v = \max_a [\mathbf{r}_a + \gamma \mathbf{P}_a v]$. L'obstacle est
l'étape de Jensen. Pour l'opérateur d'optimalité, il faudrait borner
$\big[\max_{a} \sum_{s'} p(s'|s,a) z(s')\big]^2$, et un maximum de moyennes
n'est pas une moyenne. Il n'y a pas non plus de distribution stationnaire
unique à laquelle s'accorder, puisque la politique gloutonne, et avec elle la
distribution des états, change d'une itération à l'autre.

{cite:t}`SuttonBarto2018` appellent **triade mortelle** la combinaison de
l'approximation de fonctions, de l'autoamorçage (des cibles construites à partir
de l'estimation courante) et de données hors politique : deux de ces éléments,
quels qu'ils soient, sont gérables, mais les trois réunis peuvent diverger.
L'itération Q ajustée avec un réseau de neurones réunit les trois. Il n'existe
pour elle aucun théorème de convergence général, et on peut observer la
divergence sur de petits exemples. Pourtant, NFQ et DQN résolvent de nombreux
problèmes, grâce à des choix de conception qui limitent les dégâts : chaque
ajustement se fait par rapport à des cibles maintenues fixes, l'ajustement est
initialisé à chaud à partir des paramètres précédents, un réseau cible distinct
change lentement, et un grand tampon de rejeu empêche la distribution des
données de suivre chaque changement de politique. Le
[chapitre sur les données historiques et en ligne](fitted-q-iteration.md)
examine ces choix un à un.

(sec-abe-api)=
## Itération sur les politiques approchée

Dans le cadre exact, la méthode de Newton appliquée à l'équation de Bellman est
l'itération sur les politiques ({ref}`sec-newton-kantorovich-pi`) : par le
théorème de l'enveloppe, la jacobienne en $v_k$ fait intervenir la matrice de
transition de la politique gloutonne $\pi_k$, et le pas de Newton résout
l'équation d'évaluation de $\pi_k$. Le chapitre précédent a montré la même
chose pour la collocation, où le pas de Newton {eq}`eq:collocation-newton` est
l'équation d'évaluation de $\pi_k$ imposée aux nœuds et résolue dans l'espace
engendré par la base. Avec une fonction de valeur paramétrique quelconque, le
schéma est le même, sauf que l'évaluation est faite dans la classe de
fonctions par tous les moyens disponibles. L'algorithme alterne une
**évaluation approchée**, qui trouve $\theta$ tel que
$v(\cdot;\theta) \approx v_{\pi_k}$, et une **amélioration gloutonne**. C'est
l'itération sur les politiques approchée. Sa garantie a la même forme que pour
l'itération sur les valeurs ajustée : si chaque évaluation est précise à
$\varepsilon$ près en norme sup, les politiques finissent par se trouver à au
plus $2\gamma\varepsilon/(1-\gamma)^2$ de l'optimum
{cite}`BertsekasTsitsiklis1996`. Contrairement à l'itération sur les politiques
exacte, la suite des politiques ne s'améliore pas nécessairement de façon
monotone, et elle peut cycler entre quelques politiques. Avec l'opérateur de
Bellman lisse, la jacobienne est $\gamma\mathbf{P}_{\pi_k}$ pour la politique
softmax, sans égalités à départager ({ref}`sec-smooth-pi-newton`).

(sec-abe-lstd)=
### Le cas linéaire : LSTD et LSPI

Lorsque les caractéristiques interviennent linéairement et que l'espace d'états
est fini, l'étape d'évaluation approchée a une forme close. Le point fixe de
l'évaluation de politique ajustée résout l'équation de Bellman projetée
{eq}`eq:projected-bellman-pi`. Son résidu $\BellmanPi v - v$ est
$\xi$-orthogonal à chaque caractéristique, ce qui s'écrit sous forme matricielle

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \theta - \boldsymbol{\Phi} \theta) = \mathbf{0},
\qquad\text{c'est-à-dire}\qquad
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi}) \theta = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi.
$$ (eq:lstd-galerkin)

C'est la solution **LSTD (différence temporelle par moindres carrés)** de
{cite:t}`BradtkeBarto1996`, un système linéaire $n\times n$
$\mathbf{A}\theta = \mathbf{b}$, et le pendant de Galerkin du pas de
collocation {eq}`eq:collocation-newton`. L'algorithme pratique ne forme jamais
explicitement $\boldsymbol\Phi$ sur tous les états ni $\mathbf{P}_\pi$. Il
accumule
$\mathbf{A} = \mathbb{E}_{s \sim \xi}[\boldsymbol{\varphi}(s)(\boldsymbol{\varphi}(s) - \gamma \boldsymbol{\varphi}(s'))^\top]$
et $\mathbf{b} = \mathbb{E}_{s \sim \xi}[\boldsymbol{\varphi}(s) r]$ à partir
de transitions échantillonnées $(s, r, s')$, ce qui est possible parce que le
système est linéaire en l'inconnue. Sur une chaîne à 3 états sous une
politique fixe, avec
$\mathbf{P}_\pi = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\ 0.3 & 0.4 & 0.3 \\ 0.1 & 0.3 & 0.6 \end{pmatrix}$,
$\mathbf{r}_\pi = (1, 2, 0)$, les caractéristiques $\varphi(s) = (1, s)$ et des
poids uniformes :

```{code-cell} python
:tags: [hide-input]
import numpy as np

P_pi = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
r_pi = np.array([1.0, 2.0, 0.0])
gamma = 0.9

# Feature matrix: phi_1(s) = 1, phi_2(s) = s
states = np.array([1, 2, 3])
Phi = np.column_stack([np.ones(3), states])

# Uniform weighting
xi = np.ones(3) / 3
Xi = np.diag(xi)

# LSTD matrices
A = Phi.T @ Xi @ (Phi - gamma * P_pi @ Phi)
b = Phi.T @ Xi @ r_pi

theta_lstd = np.linalg.solve(A, b)
v_lstd = Phi @ theta_lstd
v_exact = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)

print(f"LSTD solution: theta = ({theta_lstd[0]:.4f}, {theta_lstd[1]:.4f})")
print(f"\n{'State':<8} {'Exact':<12} {'LSTD':<12} {'Error':<12}")
print("-" * 44)
for s in range(3):
    print(f"{s+1:<8} {v_exact[s]:<12.4f} {v_lstd[s]:<12.4f} {v_lstd[s] - v_exact[s]:<12.4f}")

# Verify orthogonality
residual = r_pi + gamma * P_pi @ v_lstd - v_lstd
print(f"\nGalerkin orthogonality: <residual, phi_1> = {np.sum(xi * residual * Phi[:,0]):.6f}")
```

Avec le maximum réintroduit, la méthode de Newton appliquée à l'équation
d'optimalité projetée
$\boldsymbol{\Phi}^\top \boldsymbol{\Xi}\big(\Bellman(\boldsymbol\Phi\theta) - \boldsymbol{\Phi} \theta\big) = \mathbf{0}$
emploie la jacobienne donnée par le théorème de l'enveloppe,
$\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi} - \boldsymbol{\Phi})$,
à la politique gloutonne $\pi_k$, et le pas de Newton est

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi})\, \theta^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_{\pi_k},
$$ (eq:lspi)

le système LSTD pour $\pi_k$. Alterner l'amélioration gloutonne et
l'évaluation LSTD donne **LSPI (itération sur les politiques par moindres
carrés)** {cite}`LagoudakisParr2003`. Comparons-la à l'itération sur les
valeurs ajustée dans les mêmes coordonnées,
$\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi}\, \theta^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_{\pi_k} + \gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi} \theta^{(k)})$ :
les deux emploient la politique gloutonne, mais l'itération sur les valeurs
ajustée applique son opérateur de Bellman une fois puis fait une régression,
tandis que LSPI résout pour obtenir sa valeur.

Les formes closes exigent que les caractéristiques interviennent
linéairement, et les caractéristiques linéaires sont rarement ce qu'emploie la
pratique. Avec un réseau de neurones, l'étape d'évaluation n'a pas de forme
close, et le remplacement naturel est quelques balayages d'évaluation de
politique ajustée, $q \leftarrow \Proj\BellmanPi q$, ce que fait le critique de
NFQCA.

```{admonition} Exercices : LSTD et LSPI
:class: hint dropdown

1. **Calculer la distribution stationnaire.** Pour la chaîne à 3 états ci-dessus, calculez la distribution stationnaire $\xi$ qui satisfait $\xi^\top \mathbf{P}_\pi = \xi^\top$.

2. **LSTD avec pondération stationnaire.** Recalculez la solution LSTD en employant la distribution stationnaire au lieu de la pondération uniforme. Comparez l'erreur d'approximation.

3. **Deux mises à jour.** Sur un petit processus de décision markovien à deux actions et à caractéristiques linéaires, implémentez l'itération sur les valeurs ajustée et LSPI {eq}`eq:lspi`. Comparez le nombre d'itérations, et trouvez des caractéristiques pour lesquelles LSPI cycle entre deux politiques.
```

(sec-abe-roadmap)=
## De l'itération sur les valeurs ajustée à FQI et NFQCA

L'itération sur les valeurs ajustée, telle qu'elle est énoncée ici, suppose
quatre choses que NFQCA ne suppose pas, et chacun des chapitres suivants lève
l'une d'elles.

Premièrement, elle apprend des valeurs d'états. Agir de façon gloutonne par
rapport à $v(\cdot;\theta)$ exige, à chaque décision, l'espérance
$r(s,a) + \gamma\,\mathbb{E}[v(s';\theta) \mid s,a]$ pour chaque action : le
modèle et une intégrale au moment de la décision. Le
[chapitre sur les mises à jour de Bellman échantillonnées](monte-carlo-bellman-estimation.md)
déplace ce travail vers l'entraînement en apprenant la valeur de chaque
action, $q(s,a;\theta)$, de sorte qu'agir se ramène à une maximisation sur les
valeurs stockées et n'exige aucun modèle. L'opérateur de Bellman sur les
valeurs d'action est de nouveau une $\gamma$-contraction, et tout ce qui
figure dans ce chapitre s'y applique sans changement. L'itération sur les
valeurs ajustée appliquée à $q$ est l'itération Q ajustée.

Deuxièmement, elle calcule les espérances exactement. Les cibles contiennent
$\mathbb{E}[\,\cdot \mid s,a]$, que l'exemple en huit dimensions remplaçait déjà
par une moyenne sur des tirages. Avec un simulateur, ou avec des données
historiques, une cible emploie généralement un seul successeur échantillonné.
Le [chapitre sur les mises à jour de Bellman échantillonnées](monte-carlo-bellman-estimation.md)
étudie ce que cela coûte : de la variance, et un biais qui provient de la
maximisation sur des estimations bruitées.

Troisièmement, elle choisit les états où les cibles sont calculées.
L'apprentissage par renforcement par lots reçoit plutôt un ensemble fixe de
transitions, et l'apprentissage par renforcement en ligne un tampon qui
change à mesure que l'agent agit. L'analyse d'erreur de ce chapitre a établi
qu'un ajustement n'est contrôlé que là où il y a des données. Le
[chapitre sur les données historiques et en ligne](fitted-q-iteration.md)
construit les algorithmes par lots sur ce gabarit, la FQI à base d'arbres de
{cite:t}`ernst2005tree` (un moyenneur) et NFQ {cite}`riedmiller2005neural` (un
réseau de neurones), les exécute sur un journal d'ordonnancement et sur les
alertes de collision de l'ESA, puis développe DQN, dont les choix de
conception répondent à la triade mortelle.

Quatrièmement, elle maximise sur un ensemble fini d'actions. Chaque cible
contient $\max_{a'} q(s',a';\theta)$. Lorsque l'action est continue, un couple
ou une dose, ce maximum est un programme non linéaire, à résoudre une fois pour
chaque cible à chaque itération. NFQCA {cite}`Hafner2011` entraîne un second
réseau, un acteur $\pi_{\boldsymbol w}(s) \approx \arg\max_a q(s,a;\theta)$,
et l'emploie dans la cible à la place du maximum. Le critique $q$ est ajusté
par régression sur des cibles fixes, exactement comme dans ce chapitre, et
l'acteur par montée de gradient sur le critique. Dans les termes de
{ref}`sec-abe-api`, NFQCA est l'itération sur les politiques approchée avec un
seul balayage d'évaluation ajustée par amélioration, c'est-à-dire une version
ajustée de l'itération sur les politiques modifiée
({prf:ref}`alg-modified-policy-iteration`) dans laquelle l'étape
d'amélioration est elle aussi apprise. Le
[chapitre sur l'optimisation amortie des actions](amortized-action-optimization.md)
la développe, avec ses descendants en ligne DDPG et TD3. L'opérateur de Bellman
lisse se heurte au même obstacle dans la fonction de partition sur un ensemble
d'actions continu ({ref}`sec-continuous-actions`), et l'acteur-critique souple
le résout de la même façon.

| Ce que suppose ce chapitre | Ce qui le remplace | Où |
|:--------------------------|:-----------------|:------|
| Valeurs d'états $v(s)$ | Valeurs d'action $q(s,a)$, de sorte qu'agir n'exige aucun modèle | [Échantillons](monte-carlo-bellman-estimation.md), [Données historiques et en ligne](fitted-q-iteration.md) |
| Espérance exacte dans la cible | Un successeur échantillonné par transition | [Échantillons](monte-carlo-bellman-estimation.md) |
| Cibles en des états choisis | Un lot fixe (FQI, NFQ) ou un tampon de rejeu (DQN) | [Données historiques et en ligne](fitted-q-iteration.md) |
| Maximum sur un ensemble fini d'actions | Un acteur appris dans la cible (NFQCA, DDPG, TD3) | [Optimisation amortie des actions](amortized-action-optimization.md) |

## Exercices

Ces exercices complètent les exercices plus courts insérés dans les sections.
Les premiers sont de courtes vérifications; {ref}`ex-abe-knn` réutilise le code
du problème d'arrêt en huit dimensions.

:::{exercise} Déplacement de la pondération
:label: ex-projection-check-3

Si la distribution d'ajustement ne place presque aucune masse sur une région importante de l'espace d'états, à quel échec faut-il s'attendre?
:::

:::{solution} ex-projection-check-3
:class: dropdown

L'ajustement peut être précis pour la norme pondérée tout en étant médiocre dans cette région négligée, ce qui mène à de mauvaises valeurs ou de mauvaises décisions à cet endroit. Dans la borne d'erreur de l'itération sur les valeurs ajustée, cela se traduit par une grande erreur en norme sup ou, dans l'analyse pondérée, par un grand coefficient de concentrabilité.
:::

:::{exercise} Des ajustements précis, une mauvaise itération
:label: ex-projection-check-4

Dans le contre-exemple à deux états, la classe de fonctions contient exactement $v^*$. Pourquoi l'itération sur les valeurs ajustée diverge-t-elle tout de même, et pourquoi {prf:ref}`thm-approximate-vi` ne l'empêche-t-il pas?
:::

:::{solution} ex-projection-check-4
:class: dropdown

L'ajustement par moindres carrés n'est pas un moyenneur : il étire la cible constante en une droite qui la dépasse à l'état 2, d'un facteur $6/5$, ce qui l'emporte sur la contraction $\gamma$ lorsque $\gamma > 5/6$. L'erreur d'ajustement à l'itération $k$ est proportionnelle à $\theta_k$; il n'existe donc aucun $\varepsilon$ unique qui la borne pour tout $k$, et l'hypothèse du théorème n'est pas satisfaite.
:::

:::{exercise} Où intervient l'acteur
:label: ex-projection-check-5

Quelle partie de l'itération Q ajustée l'acteur de NFQCA remplace-t-il, et pourquoi n'est-il nécessaire que pour des actions continues?
:::

:::{solution} ex-projection-check-5
:class: dropdown

Il remplace la maximisation sur les actions suivantes à l'intérieur de chaque cible de Bellman. Avec un ensemble fini d'actions, le maximum se calcule par énumération; avec un ensemble d'actions continu, c'est un programme non linéaire pour chaque cible, et l'acteur amortit ces programmes en un seul réseau entraîné.
:::

:::{exercise} Lire les bornes
:label: ex-abe-bounds

L'itération sur les valeurs ajustée avec $\gamma = 0.9$ part de $v_0 = 0$, et
$\|v^*\|_\infty = 10$. Chaque ajustement est précis à $\varepsilon = 0.1$ près
en norme sup.

1. Bornez $\|v_{50} - v^*\|_\infty$ à l'aide de {prf:ref}`thm-approximate-vi`,
   et séparez la partie que davantage d'itérations éliminent de celle qu'elles
   n'éliminent pas.
2. Bornez la perte de la politique gloutonne de $v_{50}$ à l'aide de
   {prf:ref}`prop-greedy-loss`.
3. Le tableau de la récolte des canards rapporte des pertes de politique
   inférieures de quatre à cinq ordres de grandeur à la seconde borne. Nommez
   les deux endroits des démonstrations où le pire cas est imputé, et dites
   pourquoi ni l'un ni l'autre ne se produit dans ce problème.
:::

:::{solution} ex-abe-bounds
:class: dropdown

1. $\|v_{50} - v^*\|_\infty \le 0.9^{50} \cdot 10 + 0.1/0.1 = 0.052 + 1$. Des itérations supplémentaires ne réduisent que le premier terme; le second, $\varepsilon/(1-\gamma) = 1$, est le prix des ajustements et demeure.
2. La perte gloutonne est d'au plus $\tfrac{2\gamma}{1-\gamma} \cdot 1.052 \approx 18.9$, près de deux fois $\|v^*\|_\infty$; la borne ne dit donc rien d'utile ici.
3. La borne sur la valeur impute chaque erreur d'ajustement à son pire état et laisse les erreurs s'additionner dans la même direction à chaque itération. La borne gloutonne impute deux fois la pire erreur de valeur, une fois pour chacune des deux actions comparées. Dans le problème des canards, les erreurs d'ajustement sont bien plus petites que $\varepsilon$ dans la plupart des états, elles ne s'alignent pas sur l'horizon de 50 ans, et une grande partie de l'erreur est commune aux quatre forfaits, de sorte qu'elle s'annule dans la comparaison qui décide de l'action.
:::

:::{exercise} Quels ajustements sont des moyenneurs
:label: ex-abe-which-averagers

Pour chaque ajustement, dites s'il s'agit d'un moyenneur au sens de
{prf:ref}`gordon-averager`, et justifiez votre réponse par les poids qu'il
attribue aux cibles.

1. Les $k$ plus proches voisins avec $k = 5$ et des poids égaux.
2. La régression à noyau $\hat v(s) = \sum_i K(s, s_i) y_i / \sum_i K(s, s_i)$ avec un noyau gaussien.
3. Les moindres carrés ordinaires sur les caractéristiques $(1, s)$ en une dimension.
4. Un arbre de régression dont les coupures sont choisies une fois, à partir des cibles de la première itération, puis figées.
5. Un réseau de neurones entraîné jusqu'à convergence sur l'erreur quadratique.
:::

:::{solution} ex-abe-which-averagers
:class: dropdown

1. Oui : des poids de $1/5$ sur les cinq états d'entraînement les plus proches, et nuls ailleurs.
2. Oui : les poids sont positifs et normalisés pour avoir une somme égale à un, et ils ne dépendent que des états.
3. Non. La prédiction en $s$ est $\sum_i w_i(s) y_i$ avec $w_i(s) = \tfrac1m + (s - \bar s)(s_i - \bar s)/\sum_j (s_j - \bar s)^2$, qui est négatif pour les états d'entraînement situés de l'autre côté de la moyenne lorsque $s$ est loin de $\bar s$. Les poids ont bien pour somme un, de sorte que les constantes sont préservées, mais la monotonie fait défaut.
4. Oui, à partir de l'itération où les coupures sont figées : chaque prédiction est la moyenne des cibles d'une feuille fixe.
5. Non. La prédiction n'est pas du tout une fonction linéaire fixe des cibles, et rien ne contraint son signe ni sa normalisation.
:::

:::{exercise} Ajuster la mise à jour ou minimiser le résidu
:label: ex-abe-brm

Revenons au contre-exemple à deux états : $v(1;\theta) = \theta$,
$v(2;\theta) = 2\theta$, toutes les récompenses nulles, l'état 1 passe à
l'état 2 et l'état 2 reste en place.

1. La minimisation du résidu de Bellman choisit $\theta$ de façon à minimiser
   $\sum_s \big(v(s;\theta) - (\Bellman v(\cdot;\theta))(s)\big)^2$ avec des
   poids égaux. Écrivez cet objectif et montrez que son minimiseur est
   $\theta = 0$ pour tout $\gamma$.
2. La descente de gradient sur cet objectif emploie le gradient complet.
   Montrez qu'elle converge pour tout $\gamma < 1$ avec un pas assez petit,
   tandis que l'itération sur les valeurs ajustée diverge pour $\gamma > 5/6$.
3. Pourquoi la minimisation du résidu de Bellman est-elle rarement employée
   malgré cela? Examinez ce qu'exige une estimation sans biais de
   $\big(v(s;\theta) - \mathbb{E}[r + \gamma v(s';\theta)]\big)^2$ lorsque le
   successeur est aléatoire, et reliez-le au paradigme d'un seul échantillon du
   chapitre suivant.
:::

:::{solution} ex-abe-brm
:class: dropdown

1. L'objectif est $(\theta - 2\gamma\theta)^2 + (2\theta - 2\gamma\theta)^2 = \theta^2\big[(1-2\gamma)^2 + 4(1-\gamma)^2\big]$, une parabole dont le coefficient est positif pour $\gamma < 1$. Son minimiseur est $\theta = 0$.
2. La descente de gradient sur une forme quadratique définie positive converge pour tout pas inférieur à $2/c$, où $c$ est sa courbure. L'itération sur les valeurs ajustée ne fait pas descendre cet objectif : elle maintient la cible fixe et s'y ajuste, ce qui est le semi-gradient, et l'application du semi-gradient $\theta \mapsto \tfrac65\gamma\theta$ est dilatante pour $\gamma > 5/6$.
3. Élever au carré le résidu d'un seul successeur échantillonné donne $\mathbb{E}\big[(v(s) - r - \gamma v(s'))^2\big]$, qui est égal au carré du résidu plus la variance de $r + \gamma v(s')$. Le minimiseur est biaisé vers les fonctions dont la valeur au successeur a une faible variance. Une estimation sans biais exige deux successeurs indépendants du même état, le problème du double échantillonnage, ce qu'un journal à un seul successeur par transition ne peut pas fournir.
:::

:::{exercise} Un moyenneur sur le problème en huit dimensions
:label: ex-abe-knn

Dans le code du problème d'arrêt en huit dimensions, ajoutez un ajustement par
les $k$ plus proches voisins (`sklearn.neighbors.KNeighborsRegressor`) avec
$k = 1$, $10$ et $50$, et enregistrez la plus grande variation
$\max_i |v_{k+1}(s_i) - v_k(s_i)|$ à chaque itération.

1. Vérifiez que les variations diminuent au moins du facteur $\gamma = 0.9$ par
   itération, comme le garantit {prf:ref}`santos-vigo-aguiar-stability`.
   Est-ce le cas pour les arbres et le réseau?
2. Comparez les erreurs moyennes et maximales à celles des arbres et du réseau
   pour $D = 8$. Quelle valeur de $k$ est la meilleure, et pourquoi un grand
   $k$ nuit-il surtout aux coins du cube?
:::

:::{solution} ex-abe-knn
:class: dropdown

Les variations diminuent géométriquement, d'un facteur d'environ $0.43$ à
$0.55$ par itération, plus vite que $\gamma$, parce que la récompense d'arrêt
plafonne les valeurs, et elles atteignent $10^{-14}$ au cours des 40 itérations.
Avec $k = 10$ pour $D = 8$, l'erreur moyenne est d'environ $0.022$ et la plus
grande d'environ $0.155$, entre les arbres ($0.031$) et le réseau ($0.013$) en
moyenne. Un $k$ plus grand fait la moyenne des cibles sur un voisinage plus
grand; à un coin du cube, tous les voisins se trouvent du même côté, de sorte
que la prédiction est tirée vers l'intérieur, et la plus grande erreur croît
jusqu'à environ $0.2$ pour $k = 50$. Pour les arbres et le réseau, la mesure
vaut la peine d'être faite, parce qu'aucun théorème n'en prédit le résultat :
les coupures des arbres dépendent des cibles, et le réseau n'a aucune
contrainte de signe ni de normalisation.
:::

:::{exercise} De l'itération sur les valeurs à l'itération sur les politiques, de façon approchée
:label: ex-abe-mpi

L'itération sur les politiques modifiée
({prf:ref}`alg-modified-policy-iteration`) applique $\BellmanPi$ $m$ fois
après chaque amélioration gloutonne.

1. Écrivez l'itération sur les politiques modifiée ajustée, dans laquelle
   chaque application de l'opérateur est suivie d'un ajustement. Quelles
   valeurs de $m$ donnent l'itération sur les valeurs ajustée et l'itération
   sur les politiques approchée?
2. NFQCA effectue un balayage d'évaluation ajustée par amélioration, avec un
   acteur appris à la place de l'étape gloutonne. De quel $m$ s'agit-il?
3. Dans le cadre exact, un $m$ plus grand signifie moins d'améliorations.
   Donnez une raison pour laquelle un grand $m$ peut être un mauvais choix
   lorsque chaque balayage comprend un ajustement.
:::

:::{solution} ex-abe-mpi
:class: dropdown

1. Pour $j = 1, \ldots, m$ : $q \leftarrow \Proj\mathrm{L}_{\pi_k} q$, puis $\pi_{k+1}$ gloutonne pour $q$. Avec $m = 1$, chaque balayage emploie l'opérateur de la politique gloutonne pour les valeurs courantes, ce qui est l'itération sur les valeurs ajustée; $m \to \infty$ donne l'itération sur les politiques approchée avec l'évaluation de politique ajustée comme étape d'évaluation.
2. $m = 1$ : NFQCA est l'extrémité de la gamme qui correspond à l'itération sur les valeurs ajustée, la maximisation dans la cible étant faite par l'acteur.
3. Chaque ajustement ajoute une erreur, et pour un ajustement non linéaire, rien ne garantit que l'évaluation ajustée répétée d'une politique fixe converge; avec des données hors politique, elle peut diverger, ce qui est de nouveau la triade mortelle. De nombreux balayages sur une politique sur le point de changer dépensent aussi du calcul pour une précision que la prochaine amélioration rejette.
:::

:::{exercise} Concevoir une itération ajustée convergente
:label: ex-abe-open

Choisissez un problème à état continu d'un chapitre antérieur : le réservoir
du chapitre sur l'horizon fini, la barrière anti-tempête ou la pêcherie.
Concevez pour ce problème une itération sur les valeurs ajustée dont la
convergence est garantie, exécutez-la et comparez sa politique gloutonne à la
politique exacte. Remplacez ensuite l'ajustement par un réseau de neurones et
indiquez ce qui change : le nombre d'itérations, la stabilité des itérés et la
perte de la politique. Expliquez les différences à l'aide des résultats de ce
chapitre.
:::

:::{solution} ex-abe-open
:class: dropdown

Tout moyenneur donne la garantie : l'interpolation multilinéaire sur une
grille, l'agrégation d'états ou les $k$ plus proches voisins sur des états
échantillonnés. Un rapport doit préciser l'ajustement, les états où les cibles
sont calculées, la tolérance et la perte de la politique mesurée par rapport à
une solution exacte ou sur grille fine. Avec le réseau, attendez-vous à ce que
les itérés errent d'une quantité comparable à l'erreur d'ajustement plutôt que
de converger, comme dans l'expérience sur la récolte des canards, et à une
politique dont les erreurs se concentrent là où les données sont rares ou là où
deux actions sont proches.
:::

## Synthèse et perspectives

Remplacer la table par une fonction paramétrique transforme l'itération sur
les valeurs en itération sur les valeurs ajustée : calculer des cibles de
Bellman en des états choisis, leur ajuster la fonction, recommencer. La
collocation et la projection de Galerkin sont deux règles d'ajustement, et la
régression avec des arbres ou des réseaux de neurones est celle qui passe à
l'échelle, comme l'a montré le problème d'arrêt en huit dimensions. Si chaque
ajustement est précis à $\varepsilon$ près en norme sup, l'erreur de valeur
reste inférieure à $\varepsilon/(1-\gamma)$ et la politique gloutonne perd au
plus $2\gamma\varepsilon/(1-\gamma)^2$. Sur le modèle de la récolte des
canards, les deux bornes ont été vérifiées avec une marge d'un à cinq ordres de
grandeur, et les politiques gloutonnes étaient bien plus précises que les
valeurs. L'ajustement peut cependant étirer les différences et faire diverger
l'itération, même lorsqu'il représente exactement la solution. Les moyenneurs
ne le peuvent pas, et pour l'évaluation de politique, les moindres carrés
pondérés par la distribution stationnaire de la politique ne le peuvent pas non
plus. Aucun de ces arguments ne couvre l'opérateur d'optimalité avec un réseau
de neurones, qui est le cadre de l'itération Q ajustée pratique. La méthode de
Newton devient l'itération sur les politiques approchée, dont LSTD et LSPI sont
le cas linéaire.

L'évaluation exacte des cibles suppose encore l'accès aux probabilités de
transition ou à une espérance exacte. Comment estimer la même mise à jour de
Bellman lorsque le modèle ne fournit que des échantillons? Le chapitre
[Mises à jour de Bellman à partir d'échantillons](monte-carlo-bellman-estimation.md)
remplace l'intégrale exacte par des moyennes d'échantillons et rend explicites
leur variance et leur biais de maximisation.
