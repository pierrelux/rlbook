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

# Mises à jour de Bellman à partir d'échantillons

Les deux chapitres précédents supposaient qu'on pouvait évaluer exactement
l'opérateur de Bellman en tout état. Chaque cible $(\Bellman v_k)(s_i)$
contient une espérance sur l'état successeur,

$$
(\Bellman v)(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \int v(s')\,p(ds'|s,a)\right\},
$$

et la pêcherie la calculait avec une règle de quadrature à 21 nœuds, alors que
le problème d'arrêt en dimension huit la remplaçait déjà par une moyenne sur
32 tirages aléatoires. Ce chapitre se demande ce que coûte ce remplacement et
ce qu'il rend possible. Lorsque la densité de transition n'est pas disponible,
parce que le modèle est un simulateur ou un journal de transitions
enregistrées, l'échantillonnage est le seul moyen d'évaluer l'opérateur, et la
moyenne empirique en est l'estimateur. Son erreur est aléatoire plutôt que
systématique, sa vitesse ne dépend pas de la dimension de l'état, et elle
s'ajoute à l'erreur par itération que le
[chapitre précédent](approximate-bellman-equations.md) a bornée. Remplacer les
espérances exactes par des échantillons, c'est ce qui transforme la
programmation dynamique approchée des chapitres précédents en ce que la
communauté de la recherche opérationnelle appelle la programmation dynamique
par simulation et la communauté de l'apprentissage automatique,
l'apprentissage par renforcement.

Deux conséquences de l'échantillonnage organisent le chapitre. Agir à partir
d'une fonction de valeur $v$ exige l'espérance pour chaque action à chaque
décision; avec des échantillons, il faut donc simuler au moment de la
décision. Stocker plutôt la valeur de chaque action, la **fonction de valeur
d'action** $q(s,a)$, déplace ce travail vers l'entraînement. Par ailleurs, le
maximum d'estimations bruitées est biaisé vers le haut, même si chaque
estimation est sans biais; les cibles de toute mise à jour de Bellman
échantillonnée surestiment donc, et le chapitre se termine sur les moyens de
corriger ce biais.

::::{admonition} Objectifs d'apprentissage
:class: note

Après l'étude de ce chapitre, vous devriez pouvoir :
- Repérer les deux espérances d'une mise à jour de Bellman ajustée, à l'intérieur de l'opérateur et sur les états, et borner l'effet de l'estimation de l'une ou l'autre sur le point fixe
- Remplacer une espérance par une moyenne empirique, donner le biais et la variance du résultat, et expliquer pourquoi sa vitesse est indépendante de la dimension
- Définir la fonction de valeur d'action et son opérateur de Bellman, et expliquer ce qu'elle épargne au moment de la décision
- Dériver le biais vers le haut d'un maximum d'estimations bruitées, et expliquer comment découpler la sélection de l'évaluation en élimine une partie

**Préalables :** l'itération sur les valeurs ajustée et sa borne d'erreur ([chapitre précédent](approximate-bellman-equations.md)), les probabilités de base (espérance, variance, théorème central limite, inégalité de Jensen).
::::

(sec-mc-two-expectations)=
## Deux espérances dans chaque mise à jour ajustée

Des intégrales apparaissent à deux endroits dans une mise à jour de Bellman ajustée. La première se trouve à l'intérieur de l'opérateur : l'espérance de
transition $\int v(s')\,p(ds'|s,a)$ que contient chaque cible. La seconde
porte sur les états : les conditions de Galerkin et des moindres carrés
intègrent le résidu contre des fonctions test, et une régression fait la
moyenne de l'erreur d'ajustement sur les états où les cibles ont été
calculées. La collocation ne comporte que la première intégrale, ce qui est
l'une des raisons pour lesquelles c'était le cas le plus simple.

Notons $\BellmanQuad$ l'opérateur de Bellman dont l'espérance de transition
est remplacée par une règle numérique, déterministe ou aléatoire, et faisons
tourner l'itération sur les valeurs ajustée avec cet opérateur,
$v_{k+1} = \Proj\,\BellmanQuad v_k$. Par l'inégalité triangulaire, l'erreur
après une itération se décompose en trois morceaux. Avec
$v^\star = \Bellman v^\star$,

$$
\begin{aligned}
\|v^\star - v_{k+1}\| &= \|v^\star - \Proj \BellmanQuad v_k\| \\
&\le \underbrace{\|v^\star - \Proj \Bellman v^\star\|}_{\text{meilleure approximation}} + \underbrace{\|\Proj \Bellman v^\star - \Proj \Bellman v_k\|}_{\text{contraction de l'erreur courante}} + \underbrace{\|\Proj \Bellman v_k - \Proj \BellmanQuad v_k\|}_{\text{erreur d'intégration}} .
\end{aligned}
$$

Le premier terme est le mieux que puisse faire la base, le deuxième diminue
au fil des itérations lorsque $\Proj\Bellman$ est une contraction, et le
troisième est l'erreur de la règle d'intégration, qui ne diminue pas au fil
des itérations. Dans le langage du {prf:ref}`thm-approximate-vi`, l'erreur
d'intégration est une contribution de plus à l'erreur par itération
$\varepsilon$, et elle est amplifiée par le même facteur $1/(1-\gamma)$. C'est
pourquoi la précision de la règle d'intégration doit correspondre à celle de
la base. Une base de Tchebychev de degré 10 représente une fonction de valeur
lisse à environ $10^{-8}$ près, mais avec une règle des rectangles pour
l'espérance, l'erreur d'intégration, d'ordre $h^2$, domine la somme, et la
base est gaspillée.

Pour les problèmes de faible dimension dont la densité de transition est
lisse et explicite, la quadrature déterministe est le bon outil, et la règle
est choisie en fonction de la base : nœuds de Gauss–Tchebychev avec des
polynômes de Tchebychev, Gauss–Legendre avec Legendre, Gauss–Hermite avec des
polynômes d'Hermite et des chocs gaussiens, règles du point milieu ou des
trapèzes avec des éléments constants ou linéaires par morceaux. Avec la
collocation orthogonale, les nœuds de la base servent aussi de nœuds de
quadrature, et les espérances des fonctions de base aux successeurs de chaque
nœud peuvent être précalculées une fois pour toutes dans une matrice, de sorte
que chaque itération coûte un produit matriciel et une maximisation. La
pêcherie utilisait ainsi une quadrature de Gauss–Hermite sur son choc
log-normal. Le même principe d'appariement gouvernait l'optimisation de
trajectoires en temps continu, où l'intégrale du coût et la dynamique étaient
discrétisées sur les mêmes nœuds.

La quadrature déterministe exige trois choses : un état de faible dimension,
une densité qu'on peut évaluer aux nœuds et un modèle peu coûteux. Les
problèmes vers lesquels se dirige cette partie du livre n'en ont aucune.
L'état du problème d'arrêt en dimension huit exige $N^8$ nœuds de quadrature
pour une règle produit à $N$ nœuds par axe. Un simulateur fournit des tirages
de $s'$ et aucune densité. Et une transition enregistrée fournit exactement un
tirage, qu'on ne peut pas choisir. L'intégration de Monte-Carlo répond aux
trois difficultés.

## Intégration de Monte-Carlo

L'intégration de Monte-Carlo approche une espérance par la moyenne de tirages
indépendants. Pour $J = \mathbb{E}[f(X)]$ avec $X \sim p$, on tire
$X^{(1)}, \ldots, X^{(N)} \sim p$ et on forme

$$
\hat{J}_N = \frac{1}{N}\sum_{n=1}^N f\big(X^{(n)}\big).
$$

Deux propriétés de cet estimateur portent tout le chapitre. Il est sans
biais, $\mathbb{E}[\hat{J}_N] = J$, par linéarité de l'espérance. Et sa
variance est $\sigma^2/N$ avec $\sigma^2 = \mathrm{Var}[f(X)]$, par
indépendance des tirages; par le théorème central limite,

$$
\sqrt{N}\,(\hat{J}_N - J) \Longrightarrow \mathcal{N}(0,\sigma^2),
$$

et l'erreur décroît comme $N^{-1/2}$. Cette vitesse est lente par rapport à
une quadrature d'ordre élevé en dimension un, mais elle est la même en toute
dimension : Monte-Carlo paie en variance, jamais en un nombre exponentiel de
nœuds. C'est aussi une règle de quadrature, à nœuds aléatoires et à poids
égaux $1/N$. L'échantillonnage préférentiel modifie la loi des nœuds et
compense par des poids inégaux, et la vision de Monte-Carlo comme quadrature
randomisée le couvre aussi.

La cellule ci-dessous vérifie les deux propriétés sur une gaussienne de
moyenne $2$ et d'écart type $3$, avec 4 000 répétitions pour chaque taille
d'échantillon. Avant de regarder la figure, prédisez l'effet sur la racine de
l'erreur quadratique moyenne d'une multiplication de $N$ par quatre.

```{code-cell} python
:tags: [hide-input]
:label: fig-mc-rate
:caption: La moyenne empirique de $N$ tirages d'une gaussienne de moyenne 2 et d'écart type 3, sur 4 000 répétitions pour chaque $N$. (a) L'estimation moyenne reste égale à la vraie moyenne quel que soit $N$ : l'estimateur est sans biais. (b) La racine de son erreur quadratique moyenne (points) suit $\sigma/\sqrt{N}$ (droite), une pente de $-1/2$ en axes log-log; quatre fois plus de tirages divisent donc l'erreur par deux.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))
from book_figure_theme import INK, MUTED, PALETTE, figure_style, show

rng = np.random.default_rng(2026)
true_mean, sigma, trials = 2.0, 3.0, 4000
sample_sizes = 2 ** np.arange(0, 11)
means, rmse = [], []
for n in sample_sizes:
    estimates = rng.normal(true_mean, sigma, size=(trials, n)).mean(axis=1)
    means.append(estimates.mean())
    rmse.append(np.sqrt(np.mean((estimates - true_mean) ** 2)))

with figure_style():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax.axhline(true_mean, color=MUTED, lw=.8, ls="--")
    ax.plot(sample_sizes, means, "o-", color=PALETTE["blue"], ms=3.5, lw=1)
    ax.text(1.2, true_mean + .045, "true mean", color=MUTED, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
    ax.set_ylim(true_mean - .15, true_mean + .15)
    ax.set_xlabel("draws $N$")
    ax.set_ylabel("average estimate")
    ax.set_title("(a) No bias at any $N$")

    bx.plot(sample_sizes, sigma / np.sqrt(sample_sizes), color=INK, lw=1)
    bx.plot(sample_sizes, rmse, "o", color=PALETTE["blue"], ms=3.5)
    bx.text(40, 1.0, r"$\sigma/\sqrt{N}$", color=INK, fontsize=9)
    bx.set_xscale("log", base=2)
    bx.set_yscale("log")
    bx.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
    bx.set_xlabel("draws $N$")
    bx.set_ylabel("root mean squared error")
    bx.set_title(r"(b) Error falls as $N^{-1/2}$")
    fig.tight_layout(w_pad=2)
show(fig)
```

### L'opérateur de Bellman échantillonné

Pour une fonction de valeur $v$ fixée et un couple état-action $(s,a)$, la
partie transition de l'opérateur de Bellman est l'espérance
$\mathbb{E}[v(S') \mid S = s, A = a]$. Si l'on peut tirer des états suivants
$S'^{(1)}, \ldots, S'^{(N)}$ selon $p(\cdot \mid s,a)$, en appelant un
simulateur ou en agissant dans le système réel, la moyenne empirique
l'estime. Lorsque la récompense est elle aussi aléatoire, les tirages sont des
paires $(r^{(n)}, S'^{(n)})$, et le rendement à un pas est estimé par

$$
\widehat{G}_N(s,a) = \frac{1}{N}\sum_{n=1}^N \Big[r^{(n)} + \gamma\, v\big(S'^{(n)}\big)\Big],
$$

qui est sans biais pour $\mathbb{E}[r + \gamma v(S') \mid s, a]$ tant que $v$
est fixée. En le substituant dans l'opérateur, on obtient l'**opérateur de
Bellman de Monte-Carlo**

$$
(\widehat{\Bellman}_N v)(s) = \max_{a \in \mathcal{A}_s} \widehat{G}_N(s,a).
$$

Ni l'approximation ni la projection ne sont encore intervenues. Pour un $v$
fixé, Monte-Carlo fournit des évaluations sans biais et bruitées de
$(\Bellman v)(s)$ en tout état, à partir de tirages seulement. Le maximum des
estimations bruitées, en revanche, n'est pas sans biais, et c'est l'objet de
la dernière section du chapitre.

### Échantillonner les états

La seconde espérance, sur les états, s'estime de la même façon. Les
conditions de Galerkin et des moindres carrés sont de la forme
$\int R(s;\theta)\, p_i(s)\, \xi(ds) = 0$ pour une loi $\xi$ sur les états, et
la norme pondérée du résidu est $\int R(s;\theta)^2\,\xi(ds)$. Le poids $w$
des chapitres précédents pouvait être n'importe quelle fonction positive;
pour échantillonner, il doit s'agir d'une loi de probabilité selon laquelle
on peut tirer, soit $\xi$ après normalisation, soit toute autre loi munie des
poids d'importance $w/\xi$. Si des états $S^{(1)}, \ldots, S^{(M)}$ sont tirés
selon $\xi$, chaque condition devient une moyenne empirique,

$$
\int R(s;\theta)\, p_i(s)\, \xi(ds) \approx \frac{1}{M}\sum_{m=1}^M R\big(S^{(m)};\theta\big)\, p_i\big(S^{(m)}\big),
$$

et l'objectif des moindres carrés devient le risque empirique
$\frac{1}{M}\sum_m R(S^{(m)};\theta)^2$. C'est la régression du chapitre
précédent : des cibles en des états échantillonnés, et un ajustement qui
minimise l'erreur moyenne sur ces états. En apprentissage par renforcement,
$\xi$ est rarement choisie; c'est la loi des états que les données se
trouvent contenir, dont le chapitre précédent a montré l'importance pour la
convergence et dont le [chapitre suivant](fitted-q-iteration.md) montre
l'importance pour la couverture.

Lorsque les deux espérances sont échantillonnées, la mise à jour ajustée est

$$
v_{k+1} = \Proj_M\, \widehat{\Bellman}_N\, v_k,
$$

un ajustement, en $M$ états échantillonnés, de cibles qui font chacune la
moyenne de $N$ successeurs échantillonnés. Elle n'exige ni densité de
transition ni loi des états sous forme explicite, seulement la possibilité
d'en tirer des échantillons. Dans le cadre typique de l'apprentissage par
renforcement, $N = 1$ : chaque transition enregistrée
$(s_i, a_i, r_i, s'_i)$ fournit un successeur, et la cible
$r_i + \gamma v(s'_i)$ porte toute la variance d'un tirage unique. La
variance est alors réduite par l'ajustement, qui met en commun l'information
d'états voisins, et par le nombre de transitions, plutôt que par une moyenne
en un $(s,a)$ fixé. La section après la suivante revient sur ce choix.

(sec-mc-q-functions)=
## Valeurs d'action

La politique gloutonne d'une fonction de valeur $v$ est

$$
\pi(s) = \arg\max_{a \in \mathcal{A}_s} \Big\{ r(s,a) + \gamma \int v(s')\,p(ds'|s,a) \Big\},
$$

soit une espérance de transition par action à chaque décision. Avec un modèle
et une règle de quadrature, c'est un calcul; avec un simulateur, c'est une
estimation de Monte-Carlo par action, répétée chaque fois que l'état est
visité; avec un journal, c'est impossible, car on ne peut pas tirer de
nouveaux successeurs au moment de la décision. On peut plutôt calculer
l'espérance une fois, pendant l'entraînement, et la stocker. La **fonction de
valeur d'action** de $v$,

$$
q(s,a) = r(s,a) + \gamma \int v(s')\,p(ds'|s,a),
$$

met en cache la mise à jour de Bellman de chaque action, et la politique gloutonne devient
$\pi(s) = \arg\max_{a} q(s,a)$, une maximisation sur des valeurs stockées,
sans intégrale ni modèle. La fonction de valeur d'action optimale $q^\star$
satisfait sa propre équation de Bellman, obtenue en substituant
$v^\star(s') = \max_{a'} q^\star(s',a')$ :

$$
q^\star(s,a) = r(s,a) + \gamma \int p(ds'|s,a)\, \max_{a' \in \mathcal{A}_{s'}} q^\star(s',a').
$$

Le membre de droite définit l'opérateur de Bellman sur les fonctions de
valeur d'action,
$(\Bellman q)(s,a) = r(s,a) + \gamma\,\mathbb{E}[\max_{a'} q(s',a') \mid s,a]$,
et c'est une $\gamma$-contraction pour la norme du sup, par le même argument
que pour $v$ : le maximum sur $a'$ n'augmente pas les distances en norme du
sup, et l'espérance non plus. Tout ce qui figure au chapitre précédent se
transpose. L'itération sur les valeurs ajustée appliquée à $q$ est
l'**itération Q ajustée**, ses bornes d'erreur valent avec $(s,a)$ à la place
de $s$, les moyenneurs restent sûrs, et la classe de fonctions prend
maintenant l'action en entrée ou possède une sortie par action. Ce qui change,
c'est la comptabilité des coûts : les espérances sont évaluées pendant les
itérations, aux couples état-action où les cibles sont calculées, et agir
ensuite coûte peu.

```{prf:algorithm} Itération Q ajustée avec un modèle
:label: simadp-parametric-q-value-iteration

**Entrée** Un processus de décision markovien $(S, A, P, R, \gamma)$, des points de base $\mathcal{B} \subset S$, une classe d'approximateurs de fonctions $q(s,a; \boldsymbol{\theta})$, un nombre maximal d'itérations $N$, une tolérance $\varepsilon > 0$, des paramètres initiaux $\boldsymbol{\theta}_0$

**Sortie** Les paramètres $\boldsymbol{\theta}$ de l'approximation de la fonction Q

1. Initialiser $\boldsymbol{\theta}_0$ (p. ex. aléatoirement)
2. $n \leftarrow 0$
3. **répéter**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. Pour chaque $(s,a) \in \mathcal{B} \times A$ :
        1. $y_{s,a} \leftarrow r(s,a) + \gamma \int p(ds'|s,a)\max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}, \boldsymbol{\theta}_{\text{init}}=\boldsymbol{\theta}_0, K=\infty)$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}||A|}\sum_{(s,a) \in \mathcal{D} \times A} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **jusqu'à ce que** ($\delta < \varepsilon$ ou $n \geq N$)
5. **retourner** $\boldsymbol{\theta}_n$
```

L'espérance de la cible peut être calculée par quadrature lorsque le modèle
le permet, ou estimée par $N$ tirages du successeur selon $p(\cdot|s,a)$, en
faisant la moyenne de $\max_{a'} q(s'_i, a')$ sur ces tirages. L'étape
d'ajustement est celle du chapitre précédent; son initialisation
$\boldsymbol\theta_{\text{init}}$ et son budget d'optimisation $K$ sont écrits
explicitement parce que le [chapitre suivant](fitted-q-iteration.md) fait
varier les deux.

### Le paradigme de l'échantillon unique

La plupart des algorithmes d'apprentissage par renforcement ne tirent pas $N$
successeurs par couple état-action. Ils travaillent avec un jeu de données de
**transitions**
$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^M$, chacune enregistrant
l'unique successeur observé lorsque $a_i$ a été exécutée en $s_i$, et la cible
d'une transition est $y_i = r_i + \gamma \max_{a'} q(s'_i, a')$, avec $N = 1$.

Cette conception a deux origines. Dans certains contextes, c'est une
contrainte. Un robot qui exécute une action observe une seule issue et ne
peut pas revenir au même état pour en échantillonner une autre; il en va de
même pour un essai clinique, un procédé de fabrication ou toute interaction
qui coûte quelque chose. Beaucoup de contextes où l'on applique
l'apprentissage par renforcement n'ont pas cette contrainte. Les émulateurs de
jeux sauvegardent et restaurent des états, les simulateurs physiques se
réinitialisent dans n'importe quelle configuration, et les environnements
logiciels se clonent gratuitement. La seconde origine est historique.
L'apprentissage par renforcement est issu de modèles de l'apprentissage
animal, où un organisme vit une conséquence par action et apprend d'un flux
d'expérience, et l'apprentissage par différence temporelle, le Q-learning et
les méthodes acteur-critique ont été conçus pour ce flux.

La convention a persisté là où la contrainte n'existait pas, et elle a
façonné les algorithmes. Ceux-ci stockent et manipulent des transitions
plutôt que des couples état-action dotés de plusieurs successeurs. Ils
réduisent la variance au moyen de l'approximateur de fonctions, qui met en
commun les cibles d'états voisins, et au moyen du nombre de transitions,
plutôt que par une moyenne en un $(s,a)$ fixé. Et ils traitent les deux
espérances de façon asymétrique : l'espérance extérieure, sur les états, est
échantillonnée avec le budget qui convient, un jeu de données entier, un
tampon de rejeu d'un million de transitions ou une transition par pas, tandis
que l'espérance intérieure est échantillonnée une seule fois. Dans un domaine
doté d'un simulateur peu coûteux et réinitialisable et de cibles à forte
variance, faire la moyenne de plusieurs successeurs par $(s,a)$ est une
réduction directe de variance que la pratique actuelle utilise rarement.

(sec-mc-overestimation)=
## Biais de maximisation

Chaque estimation $\widehat{G}_N(s,a)$ est sans biais, mais leur maximum ne
l'est pas. Fixons un état $s$ et une fonction de valeur $v$, et écrivons la
vraie valeur de continuation et son estimation

$$
\mu(s,a) = \int v(s')\,p(ds' \mid s,a), \qquad \hat{\mu}_N(s,a) = \frac{1}{N}\sum_{i=1}^N v(s'_i), \quad s'_i \sim p(\cdot|s,a),
$$

de sorte que $(\widehat{\Bellman}_N v)(s) = \max_a \{r(s,a) + \gamma \hat{\mu}_N(s,a)\}$.
Soit $a^\star$ l'action qui maximise la vraie mise à jour de Bellman. Le maximum d'une
collection est au moins aussi grand que chacun de ses éléments, donc

$$
\mathbb{E}\big[\max_a \{r(s,a) + \gamma \hat{\mu}_N(s,a)\}\big] \ge \mathbb{E}\big[r(s,a^\star) + \gamma \hat{\mu}_N(s,a^\star)\big] = r(s,a^\star) + \gamma \mu(s,a^\star) = (\Bellman v)(s).
$$

L'inégalité est stricte dès que plus d'une action a une variance non nulle :
la maximisation retient l'action qui a reçu un bruit positif, et cette
estimation gonflée devient la cible. C'est l'inégalité de Jensen pour le
maximum, $\mathbb{E}[\max_a Y_a] \ge \max_a \mathbb{E}[Y_a]$, puisque le
maximum est convexe. La même inégalité explique pourquoi les politiques
déterministes sont optimales dans les processus de décision markoviens
({prf:ref}`stoch-policy-reduction`), où elle dit que randomiser sur les
actions ne peut pas battre la meilleure d'entre elles; ici, elle dit que
maximiser sur des estimations bruitées ne peut pas être sans biais.

Le biais ne disparaît pas en moyenne avec davantage de tirages de la même
estimation, et il s'accumule. L'itération sur les valeurs de Monte-Carlo
$v_{k+1} = \widehat{\Bellman}_N v_k$ construit ses cibles à l'itération $k+1$
à partir d'un $v_k$ déjà biaisé vers le haut, de sorte que
$\mathbb{E}[v_k] \ge \mathbb{E}[v_{k-1}] \ge \cdots \ge v^\star$, et les
valeurs gonflées sont réinjectées dans la maximisation suivante. La cellule
ci-dessous fixe toutes les vraies valeurs d'action à zéro, tire cinq
échantillons bruités par action et mesure le biais du maximum à mesure que le
nombre d'actions augmente, à côté de l'estimateur de la sous-section
suivante.

```{code-cell} python
:tags: [hide-input]
:label: fig-mc-max-bias
:caption: Biais de deux estimateurs de $\max_a \mu_a$ lorsque chaque vraie valeur d'action $\mu_a$ est nulle et que chacune est estimée à partir de cinq tirages gaussiens centrés réduits, sur 6 000 répétitions. Le maximum des estimations (vermillon) est biaisé vers le haut, et d'autant plus que les actions sont nombreuses. Sélectionner l'action avec un ensemble de tirages et l'évaluer avec un ensemble indépendant (bleu) élimine le biais dans ce cas, où toutes les actions sont également bonnes. Deux erreurs types de chaque estimation sont plus petites que les points.

rng = np.random.default_rng(2026)
replications, samples_per_action = 6000, 5
action_counts = [2, 4, 8, 16, 32]
bias = {"maximum": [], "double": []}
se = {"maximum": [], "double": []}
for actions in action_counts:
    selection = rng.normal(size=(replications, actions, samples_per_action)).mean(axis=2)
    evaluation = rng.normal(size=(replications, actions, samples_per_action)).mean(axis=2)
    chosen = selection.argmax(axis=1)
    for name, values in (("maximum", selection.max(axis=1)),
                         ("double", evaluation[np.arange(replications), chosen])):
        bias[name].append(values.mean())
        se[name].append(values.std(ddof=1) / np.sqrt(replications))

with figure_style():
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.axhline(0, color=MUTED, lw=.8)
    for name, color in (("maximum", PALETTE["vermilion"]), ("double", PALETTE["blue"])):
        ax.errorbar(action_counts, bias[name], yerr=2 * np.array(se[name]), fmt="o-", color=color,
                    ms=3.5, lw=1, capsize=0, elinewidth=1)
    ax.text(32, bias["maximum"][-1] - .06, "maximum of the\nestimates", color=PALETTE["vermilion"],
            fontsize=8.5, ha="right", va="top")
    ax.text(32, .06, "select with one estimate,\nevaluate with the other", color=PALETTE["blue"],
            fontsize=8.5, ha="right", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(action_counts, [str(a) for a in action_counts])
    ax.set_xlabel("number of actions")
    ax.set_ylabel("bias")
show(fig)
```

### Découpler la sélection de l'évaluation

Le biais vient de ce qu'une même estimation bruitée sert à deux tâches.
Décomposons l'estimation en sa moyenne et son bruit,
$\hat{\mu}_N(s,a) = \mu(s,a) + \varepsilon_a$ avec
$\mathbb{E}[\varepsilon_a] = 0$. La cible usuelle est

$$
Y = r(s,\hat a) + \gamma \mu(s,\hat a) + \gamma \varepsilon_{\hat a}, \qquad \hat a = \arg\max_a \{r(s,a) + \gamma \mu(s,a) + \gamma \varepsilon_a\},
$$

et le même $\varepsilon_{\hat a}$ qui a fait gagner $\hat a$ est ajouté à sa
valeur. Le double Q-learning {cite}`van2016deep` rompt ce couplage au moyen de
deux estimations indépendantes de chaque action,
$\hat{\mu}^{(1)}_N(s,a) = \mu(s,a) + \varepsilon^{(1)}_a$ et
$\hat{\mu}^{(2)}_N(s,a) = \mu(s,a) + \varepsilon^{(2)}_a$, où
$\varepsilon^{(1)}$ et $\varepsilon^{(2)}$ sont indépendants. La première
sélectionne et la seconde évalue :

$$
\hat a = \arg\max_{a} \big\{r(s,a) + \gamma \hat{\mu}^{(1)}_N(s,a)\big\}, \qquad Y = r(s,\hat a) + \gamma \hat{\mu}^{(2)}_N(s,\hat a).
$$

L'action sélectionnée $\hat a$ est aléatoire, fonction de $\varepsilon^{(1)}$
seulement. En conditionnant sur elle,

$$
\mathbb{E}[Y] = \mathbb{E}_{\varepsilon^{(1)}}\Big[\mathbb{E}_{\varepsilon^{(2)}}\big[r(s,\hat a) + \gamma \mu(s,\hat a) + \gamma \varepsilon^{(2)}_{\hat a} \mid \hat a\big]\Big] = \mathbb{E}_{\varepsilon^{(1)}}\big[r(s,\hat a) + \gamma \mu(s,\hat a)\big],
$$

car $\varepsilon^{(2)}$ est indépendant de $\hat a$, de sorte que
$\mathbb{E}[\varepsilon^{(2)}_{\hat a} \mid \hat a] = \mathbb{E}[\varepsilon^{(2)}_{\hat a}] = 0$.
Le bruit d'évaluation ne contribue en rien en moyenne. Il reste le **biais de
sélection** : $\hat a$ est plus souvent une action dont le
$\varepsilon^{(1)}$ était positif qu'une action dont la vraie valeur est la
plus élevée, donc $\mathbb{E}[\mu(s,\hat a)] \le \max_a \mu(s,a)$, et le
double estimateur est biaisé vers le bas. Dans {numref}`fig-mc-max-bias`,
l'estimateur du maximum devient plus optimiste à mesure que le nombre
d'actions augmente, tandis que le double estimateur n'a aucun biais : toutes
les actions y ont la même vraie valeur, si bien que choisir la mauvaise ne
coûte rien et que le biais de sélection est nul. Lorsque les vraies valeurs
diffèrent, le double estimateur est plutôt biaisé vers le bas, beaucoup moins
que le maximum ne l'est vers le haut. Éliminer le **biais d'évaluation**, la
réutilisation du bruit gagnant, élimine l'essentiel de la surestimation.

```{prf:remark} Deux fonctions Q au lieu de deux ensembles de tirages
:class: dropdown

L'argument suppose deux ensembles indépendants de tirages selon $p(\cdot|s,a)$ pour chaque couple état-action, ce qu'un journal ne fournit jamais et qu'un simulateur ne fournit qu'en se réinitialisant. La mise en œuvre pratique conserve plutôt deux fonctions Q, entraînées sur des données différentes ou mises à jour à des moments différents, par exemple un réseau en ligne et une copie de celui-ci mise à jour lentement. Leurs erreurs sont alors moins corrélées que les erreurs d'un seul réseau utilisé deux fois, ce qui fournit une partie de l'indépendance qu'exige l'argument. C'est le Double DQN, au [chapitre suivant](fitted-q-iteration.md) ({ref}`sec-fqi-double-dqn`).
```

L'algorithme ci-dessous écrit le principe en toutes lettres avec des cibles
de Monte-Carlo. Deux fonctions Q alternent les rôles : à la ligne 9, la
première sélectionne l'action et, à la ligne 12, la seconde l'évalue; les
lignes 10 et 13 font de même avec les rôles inversés. Le mécanisme est le
même que les fonctions Q soient des tables ou des fonctions ajustées et,
avec des données à échantillon unique ($N = 1$), les deux fonctions Q sont
entraînées sur des sous-ensembles différents ou à des fréquences différentes
pour approcher l'indépendance.

```{prf:algorithm} Itération Q double sur les valeurs avec Monte-Carlo
:label: double-q-value-iteration

**Entrée :** processus de décision markovien, échantillon d'états $\mathcal{S}$, taille d'échantillon de Monte-Carlo $N$, nombre maximal d'itérations $K$

**Sortie :** deux fonctions Q $q^{(1)}$, $q^{(2)}$

1. Initialiser $q^{(1)}_0(s,a)$ et $q^{(2)}_0(s,a)$ pour tous $s,a$
2. $k \leftarrow 0$
3. **répéter**
4. $\quad$ **pour** chaque $s \in \mathcal{S}$ **faire**
5. $\quad\quad$ **pour** chaque $a \in \mathcal{A}(s)$ **faire**
6. $\quad\quad\quad$ Tirer $N$ états suivants : $s'_1, \ldots, s'_N \sim p(\cdot \mid s,a)$
7. $\quad\quad\quad$ **// Calculer les cibles avec sélection et évaluation découplées**
8. $\quad\quad\quad$ **pour** $i = 1, \ldots, N$ **faire**
9. $\quad\quad\quad\quad$ $a^{(1)}_i \leftarrow \arg\max_{a'} q^{(1)}_k(s'_i, a')$
10. $\quad\quad\quad\quad$ $a^{(2)}_i \leftarrow \arg\max_{a'} q^{(2)}_k(s'_i, a')$
11. $\quad\quad\quad$ **fin pour**
12. $\quad\quad\quad$ $q^{(1)}_{k+1}(s,a) \leftarrow r(s,a) + \frac{\gamma}{N}\sum_{i=1}^N q^{(2)}_k(s'_i, a^{(1)}_i)$
13. $\quad\quad\quad$ $q^{(2)}_{k+1}(s,a) \leftarrow r(s,a) + \frac{\gamma}{N}\sum_{i=1}^N q^{(1)}_k(s'_i, a^{(2)}_i)$
14. $\quad\quad$ **fin pour**
15. $\quad$ **fin pour**
16. $\quad$ $k \leftarrow k+1$
17. **jusqu'à** convergence ou $k \geq K$
18. **retourner** $q^{(1)}_k$, $q^{(2)}_k$
```

### Autres corrections

Trois autres approches conservent une seule estimation et corrigent le
maximum lui-même. Elles diffèrent par ce qu'elles supposent du bruit.

La correction peut être apprise. {cite:t}`Keane1994` traitaient le biais
$\delta(s) = \mathbb{E}[(\widehat{\Bellman}_N v)(s)] - (\Bellman v)(s) \ge 0$
comme une quantité à estimer et à soustraire. On ne peut pas le calculer
directement, mais il dépend de caractéristiques observables de l'état : le
nombre d'actions, la taille d'échantillon et la dispersion des valeurs
d'action, puisque le biais est grand lorsque plusieurs actions sont proches et
petit lorsqu'une seule domine. En un petit ensemble d'états, ils effectuaient
à la fois une évaluation haute fidélité de la mise à jour de Bellman, avec de nombreux
tirages, et l'estimation à $N$ tirages utilisée par l'itération, puis
régressaient l'écart sur ces caractéristiques; pendant l'itération sur les
valeurs, le biais prédit est soustrait du maximum brut. Ce schéma, simuler sur
un sous-ensemble et interpoler partout, est courant en programmation
dynamique économétrique. Il a peu servi en apprentissage par renforcement,
parce que les évaluations haute fidélité coûtent cher et qu'une régression
mal spécifiée réinjecte sa propre erreur dans les valeurs.

La correction peut être dérivée, sous une hypothèse de loi.
{cite:t}`lee2013bias,lee2019biascorrected` supposaient un bruit gaussien
d'écart type $\sigma$ sur les estimations de $M$ actions; l'espérance du
maximum des bruits découle alors de la théorie des valeurs extrêmes,

$$
\mathbb{E}\Big[\max_{a} \varepsilon_a\Big] \approx \sigma \left(\frac{C_{\mathrm{E}}}{b_M} + b_M\right), \qquad b_M = \sqrt{2\log M - \log\log M - \log 4\pi},
$$

où $C_{\mathrm{E}} \approx 0.5772$ est la constante d'Euler–Mascheroni, et la
cible est corrigée en soustrayant cette quantité, avec $\sigma$ estimé à
partir de récompenses répétées au même $(s,a)$. La formule est exacte pour
un problème à un seul état dont le seul bruit est dans les récompenses, et
c'est une correction du premier ordre dans les autres cas; avec des queues
plus lourdes que la gaussienne, elle corrige trop peu. Elle remplace la
régression de la correction apprise par une forme explicite, au prix de
l'hypothèse gaussienne.

Le maximum peut être remplacé. {cite:t}`deramo2016estimating` pondèrent
chaque action par la probabilité qu'elle soit la meilleure sous un modèle
gaussien des estimations,
$w_a = \mathbb{P}\big(a \in \arg\max_{a'} \hat{\mu}_{a'}\big)$, et utilisent
$\sum_a w_a \hat{\mu}_a$ à la place de $\max_a \hat{\mu}_a$. L'estimateur
pondéré se situe entre l'optimisme du maximum et le pessimisme du double
estimateur, et s'adapte aux écarts entre actions : lorsqu'une action domine,
son poids est proche de un et l'estimateur est le maximum; lorsque plusieurs
sont proches, on fait la moyenne de leurs estimations. Il exige une
estimation de variance et une intégrale par action, ce qui le limite aux
petits ensembles d'actions. L'opérateur de Bellman lissé du
[chapitre sur la régularisation](regularized-dp.md) remplace lui aussi le
maximum strict par une moyenne, avec des poids fixés par une température
plutôt que par une incertitude estimée. Une dernière option ne modifie ni la
cible ni l'opérateur, mais la perte de l'ajustement, de sorte que les
surestimations soient plus pénalisées que les sous-estimations; le
[chapitre suivant](fitted-q-iteration.md) l'aborde avec la régression de
Gumbel.

## Exercices

Les exercices passent des estimateurs isolés à leur effet à l'intérieur de
l'itération. Les premiers sont de courtes vérifications;
{ref}`ex-mc-compounding` est une courte simulation.

:::{exercise} Le signe avant la simulation
:label: ex-montecarlo-check-2

Plusieurs actions ont des vraies valeurs égales et des estimations bruitées indépendantes. Quel signe attendez-vous pour le biais de leur maximum?
:::

:::{solution} ex-montecarlo-check-2
:class: dropdown

Positif. La maximisation retient de préférence les estimations dont le bruit est favorable, de sorte que l'espérance du maximum dépasse la vraie valeur commune.
:::

:::{exercise} Double estimation
:label: ex-montecarlo-check-3

Quelle dépendance est rompue lorsqu'une estimation sélectionne l'action maximisante et qu'une estimation indépendante l'évalue, et quel biais subsiste?
:::

:::{solution} ex-montecarlo-check-3
:class: dropdown

Le bruit d'évaluation devient indépendant du bruit chanceux qui a causé la sélection, ce qui élimine la réutilisation de la même erreur positive pour les deux tâches. Le biais de sélection subsiste : l'action choisie est encore plus souvent une action dont la première estimation a été chanceuse, de sorte que le double estimateur est légèrement biaisé vers le bas.
:::

:::{exercise} Tailles d'échantillon
:label: ex-mc-sample-size

Un simulateur renvoie des valeurs de successeurs d'écart type $\sigma = 3$.

1. Combien de tirages donnent une racine de l'erreur quadratique moyenne de
   $0.01$ pour une espérance?
2. Une itération ajustée exige une telle espérance pour chacun de 2 000 états
   et 4 actions. Combien d'appels au simulateur coûte une itération, et
   comment ce coût changerait-il si l'état avait deux fois plus de
   coordonnées?
3. D'après le {prf:ref}`thm-approximate-vi`, que contribue une erreur
   d'intégration de $0.01$ par itération à l'erreur finale sur la valeur avec
   $\gamma = 0.95$?
:::

:::{solution} ex-mc-sample-size
:class: dropdown

1. $\sigma/\sqrt N = 0.01$ donne $N = (3/0.01)^2 = 90\,000$.
2. $2\,000 \times 4 \times 90\,000 = 7.2 \times 10^8$ appels par itération. La vitesse de Monte-Carlo ne dépend pas de la dimension, de sorte que ce nombre ne change pas à moins que $\sigma$ ne change; une règle de quadrature produit exigerait plutôt $N^D$ nœuds.
3. Au plus $0.01/(1 - 0.95) = 0.2$, qui s'ajoute à l'erreur de l'ajustement.
:::

:::{exercise} Le biais d'un maximum
:label: ex-mc-max-two

Deux actions ont la même vraie valeur $0$, et chacune est estimée par la
moyenne de cinq tirages d'écart type $1$.

1. Montrer que l'espérance du maximum des deux estimations est
   $\sigma_{\text{moy}}/\sqrt\pi$, où $\sigma_{\text{moy}} = 1/\sqrt5$ est
   l'écart type de chaque moyenne, et l'évaluer. Indice : le maximum de deux
   nombres est leur moyenne plus la moitié de leur différence absolue.
2. Comparer avec le premier point de {numref}`fig-mc-max-bias`.
3. Comment le biais varie-t-il avec $N$, le nombre de tirages par action?
:::

:::{solution} ex-mc-max-two
:class: dropdown

1. $\max(X_1, X_2) = \tfrac12(X_1 + X_2) + \tfrac12|X_1 - X_2|$. Le premier terme est de moyenne $0$. La différence est gaussienne d'écart type $\sqrt2\,\sigma_{\text{moy}}$, et $\mathbb{E}|Z| = \sqrt{2/\pi}$ pour une gaussienne centrée réduite, de sorte que l'espérance vaut $\tfrac12\sqrt2\,\sigma_{\text{moy}}\sqrt{2/\pi} = \sigma_{\text{moy}}/\sqrt\pi = 1/\sqrt{5\pi} \approx 0.252$.
2. La figure indique $0.25$ pour deux actions.
3. Il est proportionnel à $\sigma_{\text{moy}} = 1/\sqrt N$ : quatre fois plus de tirages le divisent par deux. Il ne s'annule jamais pour $N$ fini.
:::

:::{exercise} Quand le double estimateur fait moins bien
:label: ex-mc-double-worse

Deux actions ont les vraies valeurs $\mu_1 = 0$ et $\mu_2 = -1$, et chaque
estimation est entachée d'un bruit gaussien d'écart type $1$.

1. Pour l'estimateur du maximum, utiliser
   $\mathbb{E}[\max(X_1, X_2)] = \mu_1\Phi(\delta/\vartheta) + \mu_2\Phi(-\delta/\vartheta) + \vartheta\,\phi(\delta/\vartheta)$
   avec $\delta = \mu_1 - \mu_2$ et $\vartheta = \sqrt2$, où $\Phi$ et $\phi$
   sont la fonction de répartition et la densité gaussiennes centrées
   réduites, pour calculer son biais.
2. Pour le double estimateur, l'action sélectionnée est la mauvaise avec
   probabilité $\Phi(-\delta/\vartheta)$. Calculer son biais.
3. Quel estimateur est le plus proche de la vérité ici? Relier la réponse aux
   deux sources de biais de {ref}`sec-mc-overestimation`, et au cas des
   valeurs égales de {numref}`fig-mc-max-bias`.
:::

:::{solution} ex-mc-double-worse
:class: dropdown

1. $\Phi(-1/\sqrt2) = 0.240$ et $\phi(1/\sqrt2) = 0.311$, donc l'espérance du maximum vaut $-0.240 + \sqrt2 \times 0.311 = 0.200$, soit un biais de $+0.20$.
2. Le double estimateur est sans biais pour la valeur de l'action qu'il sélectionne; son espérance est donc $-1 \times 0.240 = -0.240$, soit un biais de $-0.24$. Une simulation avec $2 \times 10^6$ tirages donne $0.200$ et $-0.238$.
3. L'estimateur du maximum est ici un peu plus proche. Avec des valeurs égales, la sélection ne coûte rien et le double estimateur est sans biais; avec un écart de l'ordre du bruit, le biais de sélection domine, et il est négatif. Éliminer le biais d'évaluation n'élimine pas le biais de sélection, et c'est pourquoi le double estimateur est en général pessimiste plutôt que sans biais.
:::

:::{exercise} Un biais qui s'accumule
:label: ex-mc-compounding

On considère un état et $M$ actions, chacune de récompense $0$ observée à
travers un bruit gaussien d'écart type $1$, et $\gamma = 0.9$. L'itération
sur les valeurs de Monte-Carlo pose $v_{k+1} = \max_a \hat r_a + \gamma v_k$,
où chaque $\hat r_a$ est une nouvelle moyenne de $N$ tirages.

1. Montrer que $\mathbb{E}[v_k] \to b/(1-\gamma)$, où $b$ est l'espérance du
   maximum de $M$ moyennes indépendantes, alors que la vraie valeur est $0$.
2. Le simuler pour $M = 4$ et $N = 5$, et comparer avec la valeur
   $b \approx 0.46$ lue sur {numref}`fig-mc-max-bias`.
3. Remplacer le maximum par le double estimateur, avec deux ensembles
   indépendants de tirages, et simuler de nouveau.
:::

:::{solution} ex-mc-compounding
:class: dropdown

1. En prenant l'espérance, $\mathbb{E}[v_{k+1}] = b + \gamma\,\mathbb{E}[v_k]$, une contraction de point fixe $b/(1-\gamma)$.
2. Avec $b \approx 0.460$, la limite vaut $4.6$, soit dix fois le biais par pas. Une simulation comme

```python
import numpy as np
rng = np.random.default_rng(0)
M, N, gamma, runs, iterations = 4, 5, 0.9, 20000, 200
v = np.zeros(runs)
for _ in range(iterations):
    v = rng.normal(size=(runs, M, N)).mean(axis=2).max(axis=1) + gamma * v
print(v.mean())
```

affiche une valeur proche de $4.6$.
3. Sélectionner avec un ensemble de moyennes et évaluer avec l'autre donne une espérance de $0$ à chaque pas, donc $\mathbb{E}[v_k] \to 0$ : avec des vraies valeurs égales, le double estimateur élimine complètement l'accumulation.
:::

:::{exercise} Fonctions Q à partir de l'opérateur de Bellman sur les valeurs
:label: ex-mc-q-contraction

1. Montrer que $(\Bellman q)(s,a) = r(s,a) + \gamma\,\mathbb{E}[\max_{a'} q(s',a') \mid s,a]$
   est une $\gamma$-contraction pour la norme du sup. Utiliser l'inégalité
   $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$, que le chapitre
   sur l'horizon infini a utilisée pour l'opérateur sur les valeurs.
2. Étant donné $q^\star$, écrire la politique optimale et $v^\star$. Laquelle
   des deux exige le modèle de transition?
3. L'itération sur les valeurs ajustée et l'itération Q ajustée convergent
   toutes deux avec un moyenneur. Laquelle exige le plus de cibles par
   itération, et laquelle coûte le moins cher pour agir?
:::

:::{solution} ex-mc-q-contraction
:class: dropdown

1. Pour chaque $(s,a)$, $|(\Bellman q)(s,a) - (\Bellman q')(s,a)| \le \gamma\,\mathbb{E}\big[|\max_{a'} q(s',a') - \max_{a'} q'(s',a')|\big] \le \gamma\|q - q'\|_\infty$.
2. $\pi^\star(s) \in \arg\max_a q^\star(s,a)$ et $v^\star(s) = \max_a q^\star(s,a)$. Ni l'une ni l'autre n'exige le modèle, et c'est tout l'intérêt de stocker $q$.
3. L'itération Q ajustée calcule des cibles pour chaque couple état-action, soit $|\mathcal{A}|$ fois plus que l'itération sur les valeurs ajustée avec un modèle, mais agir avec elle revient à maximiser sur des valeurs stockées, sans espérance à estimer.
:::

:::{exercise} Beaucoup de successeurs ou beaucoup de transitions
:label: ex-mc-budget

Un simulateur réinitialisable peut renvoyer plusieurs successeurs d'un même
état et d'une même action. Avec un budget fixe d'appels au simulateur par
itération, comparer deux usages sur le problème d'arrêt en dimension huit du
chapitre précédent : beaucoup d'états avec un successeur chacun, ou quatre
fois moins d'états avec quatre successeurs chacun. Mesurer l'erreur de la
fonction de valeur ajustée et de sa politique gloutonne, et expliquer le
résultat à l'aide des deux espérances de {ref}`sec-mc-two-expectations` :
quelle erreur chaque choix réduit-il?
:::

:::{solution} ex-mc-budget
:class: dropdown

Cet exercice est ouvert. Davantage de successeurs par état réduisent la
variance de chaque cible, qu'une régression devrait sinon compenser en faisant
la moyenne sur des états voisins. Davantage d'états réduisent l'erreur de
l'ajustement lui-même et étendent la couverture. Le gagnant dépend du bruit
des cibles par rapport à la régularité de la valeur : un problème aux
transitions très bruitées et à la valeur lisse favorise davantage de
successeurs, et un problème dont la valeur varie beaucoup dans l'espace
d'états favorise davantage d'états.
:::

## Synthèse et perspectives

Les mises à jour de Bellman ajustées contiennent deux espérances, à
l'intérieur de l'opérateur et sur les états, et l'une comme l'autre peut être
estimée à partir d'échantillons. La moyenne empirique est sans biais, son
erreur diminue comme $N^{-1/2}$ quelle que soit la dimension, et elle ajoute
une erreur par itération que la borne du chapitre précédent absorbe.
L'échantillonnage rend le modèle facultatif : un simulateur ou un journal de
transitions suffit, et c'est le cadre de l'apprentissage par renforcement.
Agir à partir d'une fonction de valeur échantillonnée exigerait
d'échantillonner au moment de la décision; la fonction de valeur d'action
$q(s,a)$ stocke donc plutôt la mise à jour de Bellman de chaque action, et l'itération Q
ajustée est l'itération sur les valeurs ajustée appliquée à $q$. Le maximum
d'estimations bruitées est biaisé vers le haut, et ce biais s'accumule au fil
des itérations; utiliser une estimation pour sélectionner l'action et une
estimation indépendante pour l'évaluer élimine la plus grande partie du
biais, et les corrections apprises, analytiques ou pondérées sont les
autres options.

Les algorithmes de ce chapitre choisissent encore les états où les cibles
sont calculées, et tirent encore autant de successeurs qu'ils le veulent.
[L'apprentissage à partir de données historiques et en ligne](fitted-q-iteration.md)
prend les données telles qu'elles sont : un journal fixe de transitions à un
successeur chacune, ou un tampon qui se remplit à mesure que l'agent agit, et
un ajustement démarré à chaud et arrêté de façon anticipée. Les autres choix
de conception des algorithmes pratiques sont des choix portant sur ces
données et sur cet ajustement.
