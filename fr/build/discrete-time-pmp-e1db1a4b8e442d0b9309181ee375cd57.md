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
# États adjoints et principe de Pontryaguine en temps discret

Le contrôle optimal à horizon fini produit un programme non linéaire
structuré, et les conditions KKT caractérisent ses solutions locales. Comment
ces multiplicateurs s'organisent-ils lorsque les contraintes d'égalité forment
une récurrence dynamique en avant?

Si l'on prend la formulation de Bolza du PCOTD et qu'on y applique directement les conditions KKT, on obtient un système d'optimisation comportant de nombreux multiplicateurs et contraintes. Écrit sous forme brute, il ressemble à n'importe quel autre programme non linéaire. Mais en commande, cette structure a une longue histoire et un nom qui lui est propre : le **principe de Pontryaguine**. Sa version en temps discret est le système KKT obtenu lorsqu'on introduit un multiplicateur pour chaque contrainte de dynamique et qu'on regroupe les termes étape par étape.

On travaille avec le programme de Bolza

$$
\begin{aligned}
\min_{\{\mathbf{x}_t,\mathbf{u}_t\}} \quad & c_T(\mathbf{x}_T)\;+\;\sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{s.c.}\quad & \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t),\quad t=1,\dots,T-1,\\
& \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\le \mathbf{0},\quad \mathbf{u}_t\in \mathcal{U}_t,\\
& \mathbf{h}(\mathbf{x}_T)=\mathbf{0}\quad\text{(égalités terminales facultatives)}.
\end{aligned}
$$

On introduit des **états adjoints** $\boldsymbol{\lambda}_{t+1}\in\mathbb{R}^n$ pour la dynamique, des multiplicateurs $\boldsymbol{\mu}_t\ge \mathbf{0}$ pour les inégalités de chemin et $\boldsymbol{\nu}$ pour les égalités terminales. Le lagrangien est

$$
\mathcal{L}
= c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)
+ \sum_{t=1}^{T-1} \boldsymbol{\lambda}_{t+1}^\top\!\big(\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)-\mathbf{x}_{t+1}\big)
+ \sum_{t=1}^{T-1} \boldsymbol{\mu}_t^\top \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)
+ \boldsymbol{\nu}^\top \mathbf{h}(\mathbf{x}_T).
$$

Il est commode de regrouper les termes de chaque étape dans un **hamiltonien**

$$
H_t(\mathbf{x}_t,\mathbf{u}_t,\boldsymbol{\lambda}_{t+1},\boldsymbol{\mu}_t)
:= c_t(\mathbf{x}_t,\mathbf{u}_t)
+ \boldsymbol{\lambda}_{t+1}^\top \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)
+ \boldsymbol{\mu}_t^\top \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t).
$$

```{admonition} Vérifiez votre compréhension
:class: tip
Vérifiez que l'hamiltonien $H_t$ regroupe exactement les termes du lagrangien qui font intervenir l'étape $t$ : le coût d'étape, la contrainte de dynamique (pondérée par l'état adjoint suivant) et les contraintes de chemin (pondérées par leurs multiplicateurs). Pourquoi est-ce $\boldsymbol{\lambda}_{t+1}$ qui apparaît, et non $\boldsymbol{\lambda}_t$?
```

On a alors

$$
\mathcal{L} = c_T(\mathbf{x}_T)+\boldsymbol{\nu}^\top \mathbf{h}(\mathbf{x}_T)
+ \sum_{t=1}^{T-1}\Big[H_t(\mathbf{x}_t,\mathbf{u}_t,\boldsymbol{\lambda}_{t+1},\boldsymbol{\mu}_t)
- \boldsymbol{\lambda}_{t+1}^\top \mathbf{x}_{t+1}\Big].
$$

## Conditions nécessaires

Comment la faisabilité de l'état, la récurrence adjointe, la stationnarité de
la commande et la complémentarité émergent-elles du lagrangien écrit étape par
étape?

```{note} Convention pour les gradients
Dans toute cette section, on utilise la convention de **disposition en dénominateur** (disposition des gradients) :
- $\nabla_{\mathbf{x}} f(\mathbf{x})$ produit un **vecteur colonne** (le gradient)
- $\frac{\partial f}{\partial \mathbf{x}}$ produit la matrice jacobienne
- Pour les fonctions scalaires : $\nabla_{\mathbf{x}} f = \left(\frac{\partial f}{\partial \mathbf{x}}\right)^\top$
- Pour les applications vectorielles comme $\mathbf{f}_t$, on écrit $\nabla_{\mathbf{x}} \mathbf{f}_t$ pour la jacobienne $\frac{\partial \mathbf{f}_t}{\partial \mathbf{x}}$ (une ligne par composante de $\mathbf{f}_t$), de sorte que $[\nabla_{\mathbf{x}} \mathbf{f}_t]^\top \boldsymbol{\lambda}$ est un produit vecteur-jacobienne

C'est la convention habituelle en optimisation et en théorie de la commande.
```

En prenant les variations du premier ordre et en regroupant les termes, on obtient le système adjoint en temps discret, la stationnarité de la commande et la complémentarité. En un minimum local $\{\mathbf{x}_t^\star,\mathbf{u}_t^\star\}$ de multiplicateurs $\{\boldsymbol{\lambda}_t^\star,\boldsymbol{\mu}_t^\star,\boldsymbol{\nu}^\star\}$ :

**Dynamique de l'état (faisabilité primale)**

$$
\mathbf{x}_{t+1}^\star=\mathbf{f}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star),\quad t=1,\dots,T-1.
$$

**Récurrence adjointe (équation « adjointe » à rebours)**

$$
\boldsymbol{\lambda}_t^\star
= \nabla_{\mathbf{x}} H_t\big(\mathbf{x}_t^\star,\mathbf{u}_t^\star,\boldsymbol{\lambda}_{t+1}^\star,\boldsymbol{\mu}_t^\star\big)
= \nabla_{\mathbf{x}} c_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)
+ \big[\nabla_{\mathbf{x}} \mathbf{f}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)\big]^\top \boldsymbol{\lambda}_{t+1}^\star
+ \big[\nabla_{\mathbf{x}} \mathbf{g}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)\big]^\top \boldsymbol{\mu}_t^\star,
$$

avec la **condition terminale**

$$
\boldsymbol{\lambda}_T^\star
= \nabla_{\mathbf{x}} c_T(\mathbf{x}_T^\star) + \big[\nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}_T^\star)\big]^\top \boldsymbol{\nu}^\star
\quad\text{(et \(\boldsymbol{\nu}^\star=\mathbf{0}\) en l'absence d'égalités terminales).}
$$

**Stationnarité de la commande (optimalité du premier ordre en $\mathbf{u}_t$)**
Si $\mathcal{U}_t=\mathbb{R}^m$ (aucune contrainte d'ensemble explicite), alors

$$
\nabla_{\mathbf{u}} H_t\big(\mathbf{x}_t^\star,\mathbf{u}_t^\star,\boldsymbol{\lambda}_{t+1}^\star,\boldsymbol{\mu}_t^\star\big)=\mathbf{0}.
$$

Si $\mathcal{U}_t$ impose des bornes ou un ensemble convexe, la condition devient l'**inéquation variationnelle**

$$
\mathbf{0}\in \nabla_{\mathbf{u}} H_t(\cdot)\;+\;N_{\mathcal{U}_t}(\mathbf{u}_t^\star),
$$

où $N_{\mathcal{U}_t}(\cdot)$ est le cône normal à $\mathcal{U}_t$. Pour de simples bornes en boîte, elle se ramène aux conditions KKT habituelles de signe et de complémentarité sur les composantes de $\mathbf{u}_t^\star$.

**Multiplicateurs des contraintes de chemin (faisabilité primale et duale, complémentarité)**

$$
\mathbf{g}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)\le \mathbf{0},\quad
\boldsymbol{\mu}_t^\star\ge \mathbf{0},\quad
\mu_{t,i}^\star\, g_{t,i}(\mathbf{x}_t^\star,\mathbf{u}_t^\star)=0\quad \text{pour tous }i,t.
$$

**Égalités terminales (le cas échéant)**

$$
\mathbf{h}(\mathbf{x}_T^\star)=\mathbf{0}.
$$

Ensemble, l'équation d'état en avant, l'équation adjointe à rebours et la stationnarité de la commande forment le système d'Euler-Lagrange en temps discret pour les problèmes dont les contraintes sont une dynamique. C'est la même logique KKT qu'auparavant, mais organisée étape par étape au moyen de l'hamiltonien.

```{prf:proposition} Conditions nécessaires de Pontryaguine en temps discret (résumé)
En un minimum local du PCOTD

$$
\min_{\{\mathbf{x}_t,\mathbf{u}_t\}}\ c_T(\mathbf{x}_T)+\sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)
\quad\text{s.c.}\quad \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t),\ \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\le \mathbf{0},\ \mathbf{h}(\mathbf{x}_T)=\mathbf{0},
$$

il existe des multiplicateurs $\{\boldsymbol{\lambda}_{t+1}\}$, $\{\boldsymbol{\mu}_t\ge\mathbf{0}\}$ et (le cas échéant) $\boldsymbol{\nu}$ tels que, pour $t=1,\dots,T-1$ :

- Dynamique de l'état : $\ \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)$.
- Récurrence adjointe à rebours :

  $$
  \boldsymbol{\lambda}_t = \nabla_{\mathbf{x}} c_t(\mathbf{x}_t,\mathbf{u}_t)
  + \big[\nabla_{\mathbf{x}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1}
  + \big[\nabla_{\mathbf{x}} \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\mu}_t.
  $$
  
- Condition terminale : $\ \boldsymbol{\lambda}_T = \nabla_{\mathbf{x}} c_T(\mathbf{x}_T) + \big[\nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}_T)\big]^\top \boldsymbol{\nu}$.
- Stationnarité de la commande (commande non contrainte) : $\ \nabla_{\mathbf{u}} H_t(\cdot)=\mathbf{0}$; avec un ensemble de commandes convexe $\mathcal{U}_t$, $\ \mathbf{0}\in \nabla_{\mathbf{u}} H_t(\cdot)+N_{\mathcal{U}_t}(\mathbf{u}_t)$.
- Inégalités de chemin : $\ \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\le \mathbf{0}$, $\ \boldsymbol{\mu}_t\ge\mathbf{0}$, et la complémentarité $\ \mu_{t,i}\,g_{t,i}(\mathbf{x}_t,\mathbf{u}_t)=0$ pour tout $i$.
- Égalités terminales (le cas échéant) : $\ \mathbf{h}(\mathbf{x}_T)=\mathbf{0}$.

Ici, $H_t(\mathbf{x}_t,\mathbf{u}_t,\boldsymbol{\lambda}_{t+1},\boldsymbol{\mu}_t):=c_t(\mathbf{x}_t,\mathbf{u}_t)+\boldsymbol{\lambda}_{t+1}^\top\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)+\boldsymbol{\mu}_t^\top\mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)$ est l'hamiltonien d'étape.
```

**Récapitulation.** Le principe de Pontryaguine en temps discret est le système KKT de l'optimisation de trajectoires, organisé pour exploiter la structure temporelle. Il se décompose en un passage en avant et un passage à rebours : les états se propagent vers l'avant par la dynamique, tandis que les états adjoints se propagent vers l'arrière par l'équation adjointe. L'hamiltonien $H_t$ regroupe le coût d'étape, la dynamique (pondérée par l'état adjoint suivant) et les éventuelles contraintes de chemin (pondérées par leurs multiplicateurs). La stationnarité de la commande dit que les commandes optimales sont des points stationnaires de l'hamiltonien à chaque étape (gradient nul, ou inéquation variationnelle lorsque $\mathcal{U}_t$ est contraint); lorsque $H_t$ est convexe en $\mathbf{u}_t$, comme dans le cadre linéaire quadratique, la stationnarité se renforce en minimisation de l'hamiltonien. La complémentarité garantit que seules les contraintes saturées portent des multiplicateurs non nuls. Cette structure sous-tend à la fois des méthodes de résolution analytiques (comme la commande linéaire quadratique) et des algorithmes numériques (comme la méthode adjointe de calcul des gradients).

## L'équation adjointe comme accumulation inverse

Les équations adjointes caractérisent la stationnarité, mais la même
récurrence à rebours peut-elle calculer toutes les dérivées par rapport aux
commandes en un seul balayage inverse?

L'optimisation a besoin de sensibilités. Dans les problèmes de trajectoire, on ajuste des décisions (commandes ou paramètres) pour réduire un objectif tout en respectant la dynamique et les contraintes. Les méthodes du premier ordre sans contraintes (par exemple la descente de gradient, L-BFGS ou Adam) exigent le gradient de l'objectif par rapport à toutes les commandes, et les méthodes sous contraintes (PQS, points intérieurs) exigent les gradients du lagrangien, c'est-à-dire des coûts et des contraintes. Les équations adjointes en temps discret fournissent ces dérivées d'une manière qui passe à l'échelle pour de longs horizons et de nombreuses variables de décision.

Considérons

$$
J = c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t),
\qquad \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t).
$$

Un seul déroulement en avant calcule et stocke la trajectoire $\mathbf{x}_{1:T}$. Un seul balayage à rebours applique ensuite la dérivation des fonctions composées en mode inverse, étape par étape.

```{admonition} Avant de poursuivre
:class: tip
Essayez d'appliquer la dérivation des fonctions composées pour calculer comment une perturbation de l'état au temps $t$ modifie le coût total $J$. Partez de $\partial J / \partial \mathbf{x}_T$ et remontez le temps. Quel motif apparaît?
```

Définir l'état adjoint par

$$
\boldsymbol{\lambda}_T = \nabla_{\mathbf{x}} c_T(\mathbf{x}_T),\qquad
\boldsymbol{\lambda}_t = \nabla_{\mathbf{x}} c_t(\mathbf{x}_t,\mathbf{u}_t) + \big[\nabla_{\mathbf{x}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1},\quad t=T-1,\dots,1,
$$

donne exactement la récurrence adjointe en temps discret (celle du principe de Pontryaguine).

```{admonition} Vérifiez votre compréhension
:class: tip
Quels termes vous attendez-vous à voir dans le gradient $\nabla_{\mathbf{u}_t} J$? La commande $\mathbf{u}_t$ influe sur $J$ de deux façons : directement par le coût d'étape $c_t$, et indirectement en modifiant l'état suivant $\mathbf{x}_{t+1}$. Comment ces contributions doivent-elles se combiner?
```

Le gradient par rapport à chaque commande découle du même passage inverse :

$$
\nabla_{\mathbf{u}_t} J = \nabla_{\mathbf{u}} c_t(\mathbf{x}_t,\mathbf{u}_t) + \big[\nabla_{\mathbf{u}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1}.
$$

Cette accumulation inverse produit tous les gradients par rapport aux
commandes avec un seul déroulement en avant et un seul passage adjoint à
rebours. L'état adjoint $\boldsymbol{\lambda}_t$ mesure l'effet marginal sur
l'objectif total d'une perturbation de l'état au temps $t$. Chaque gradient
par rapport à une commande combine une contribution directe de $c_t$ et une
contribution indirecte passant par l'état suivant. La rétropropagation à
travers un système dynamique déroulé effectue le même calcul.

Les différences finies, au contraire, perturbent une décision à la fois et
relancent le système. Elles exigent de l'ordre de $p$ déroulements pour
$p=(T-1)m$ variables de commande et introduisent un pas de différence finie.
Les sensibilités en mode direct propagent un produit jacobienne-vecteur
distinct pour chaque direction de paramètre; leur travail croît donc aussi
avec $p$. Le mode inverse propage un seul vecteur adjoint à rebours et lit
toutes les dérivées partielles dans ce balayage. Pour un objectif scalaire,
cela remplace un déroulement par paramètre par un seul passage en avant et à
rebours, au prix du stockage ou de la mise en points de reprise de la
trajectoire d'état.

La récurrence adjointe est donc la dérivée en mode inverse de l'objectif de
trajectoire. Les états portent la trajectoire nominale vers l'avant, les états
adjoints portent sa sensibilité vers l'arrière, et les dérivées locales par
rapport aux commandes combinent les deux à chaque étape.

## Synthèse et perspectives

Les conditions KKT organisent l'optimalité locale en faisabilité primale,
stationnarité, faisabilité duale et complémentarité. Appliquées le long d'une
trajectoire, elles donnent le principe de Pontryaguine en temps discret. Les
états se propagent vers l'avant, les états adjoints vers l'arrière, et
l'hamiltonien fournit la condition locale de stationnarité de la commande. La
même récurrence à rebours calcule tous les gradients par rapport aux commandes
en un seul passage inverse.

Les conditions nécessaires ne disent pas comment présenter les états et les
actions à un solveur numérique. Les états doivent-ils rester des variables de
décision, être éliminés par simulation en avant, ou n'apparaître qu'aux
frontières de segments? Le chapitre [Optimisation numérique de
trajectoires](numerical-trajectory-optimization.md) compare ces formulations.

## Exercices

:::{exercise} La récurrence adjointe comme sensibilité
:label: ex-trajectories-costate

Considérons un PCOTD sans contraintes d'objectif $J = c_T(x_T) + \sum_{t=1}^{T-1} c_t(x_t, u_t)$ et de dynamique $x_{t+1} = f_t(x_t, u_t)$.

**(a)** Définissons le coût restant à partir du temps $t$ par $V_t(x_t) = \min_{u_{t:T-1}} \left[ c_T(x_T) + \sum_{s=t}^{T-1} c_s(x_s, u_s) \right]$. Montrez que, le long d'une trajectoire optimale, $\nabla_{x_t} V_t(x_t^\star) = \lambda_t^\star$, où $\lambda_t$ est l'état adjoint.

**(b)** Interprétez économiquement l'état adjoint : que mesure $\lambda_t$?

:::

:::{solution} ex-trajectories-costate
:class: dropdown

Par définition, $V_t(x_t)$ est le coût futur minimal à partir de $x_t$. Le long de la trajectoire optimale, le théorème de l'enveloppe donne $\nabla_{x_t} V_t = \lambda_t$, la valeur marginale de l'état. Économiquement, $\lambda_t$ est le taux de variation du coût restant optimal sous une petite perturbation de l'état $x_t$ (le coût augmente d'environ $\lambda_t^\top \delta x_t$ pour une perturbation $\delta x_t$). C'est le « prix implicite » de l'état au temps $t$.
:::

---

:::{exercise} Vérification du gradient adjoint
:label: ex-trajectories-adjoint

Pour le système $x_{t+1} = x_t^2 + u_t$ avec $x_1 = 0.5$, $T = 4$ et l'objectif $J = x_4$ :

**(a)** Calculez le gradient $\nabla_{u} J$ par différences finies (perturbez chaque $u_t$ de $\epsilon = 10^{-5}$).

**(b)** Calculez le gradient par la méthode adjointe : simulez d'abord en avant pour obtenir $x_{1:4}$, puis propagez les états adjoints à rebours avec $\lambda_4 = 1$ et $\lambda_t = 2x_t \lambda_{t+1}$.

**(c)** Vérifiez que les deux méthodes donnent la même réponse. Laquelle est la plus efficace pour un grand $T$?

:::

:::{solution} ex-trajectories-adjoint
:class: dropdown

Pour les commandes $u = [0, 0, 0]$, la simulation en avant donne $x_1 = 0.5$, $x_2 = 0.25$, $x_3 = 0.0625$, $x_4 = 0.00390625$.

Méthode adjointe : $\lambda_4 = 1$, $\lambda_3 = 2(0.0625)(1) = 0.125$, $\lambda_2 = 2(0.25)(0.125) = 0.0625$, $\lambda_1 = 2(0.5)(0.0625) = 0.0625$.

Gradients par rapport aux commandes : $\nabla_{u_t} J = \lambda_{t+1}$, donc $\nabla_u J = [\lambda_2, \lambda_3, \lambda_4] = [0.0625, 0.125, 1.0]$.

Les différences finies devraient concorder. La méthode adjointe demande un travail en $O(T)$ (un déroulement en avant et un passage à rebours), quel que soit le nombre de commandes; les différences finies exigent un déroulement par variable de commande, soit $O(T m)$ déroulements de coût $O(T)$ chacun, pour un travail total en $O(T^2 m)$ avec une commande de dimension $m$.
:::

---
