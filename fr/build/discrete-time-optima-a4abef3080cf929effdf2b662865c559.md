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
# Problèmes de contrôle optimal à horizon fini

Les modèles présentés au chapitre précédent prédisent comment les actions
modifient l'état d'un système. L'optimisation de trajectoires y ajoute un
objectif et des contraintes, puis choisit les actions qui produisent une suite
d'états souhaitable. Pour un horizon discret fixé, les états et les commandes
forment un vecteur de dimension finie; le problème de planification peut donc
s'écrire comme un programme non linéaire.

Quel problème d'optimisation de dimension finie choisit la meilleure
trajectoire admissible à partir d'une condition initiale connue?

Une **trajectoire** est la suite indexée par le temps des états
$(\mathbf{x}_1,\ldots,\mathbf{x}_T)$ et des commandes
$(\mathbf{u}_1,\ldots,\mathbf{u}_{T-1})$. Ce chapitre calcule d'abord la suite
complète des commandes à partir d'un état initial connu. Cette suite est en
**boucle ouverte** : une fois l'exécution commencée, l'action au temps $t$ ne
change pas en fonction de l'état mesuré.

L'exemple des trois satellites ci-dessous montre à la fois l'intérêt et la
limite de la planification en boucle ouverte. Un modèle linéaire produit un
plan creux et réalisable, mais ce même plan immuable manque sa cible lorsqu'on
le rejoue dans un modèle non linéaire. Les chapitres suivants remplaceront la
suite fixe par une rétroaction, au moyen de la commande à horizon glissant et
des politiques.

```{admonition} Objectifs d'apprentissage
:class: note

Après l'étude de ce chapitre, vous devriez pouvoir :

1. Formuler un problème de contrôle optimal en temps discret (PCOTD) sous les formes de Bolza, de Lagrange et de Mayer, et passer de l'une à l'autre.
2. Énoncer les conditions KKT d'un programme non linéaire (PNL) sous contraintes et expliquer le rôle des qualifications de contraintes.
3. Dériver le principe de Pontryaguine en temps discret à partir du lagrangien d'un PCOTD.
4. Implémenter les méthodes de tir simple et de tir multiple pour l'optimisation de trajectoires.
5. Expliquer les compromis entre méthodes simultanées (transcription directe) et séquentielles (tir).
6. Calculer les gradients de l'objectif par rapport aux commandes à l'aide de la récurrence adjointe (des états adjoints).
7. Auditer un plan immuable en boucle ouverte sous un écart de modèle précisé.
```

```{admonition} Préalables
:class: tip

Ce chapitre utilise les gradients, les jacobiennes, la dérivation des
fonctions composées, les produits matrice-vecteur et les systèmes linéaires.
L'annexe [](appendix_nlp.md) revoit les objets de programmation non linéaire
utilisés dans la dérivation des conditions KKT. Les modèles d'état et de
transition proviennent de [](modeling-controlled-systems.md).
```

## Un exemple motivant : mise en phase de trois satellites par traînée différentielle

Que gagne un plan en boucle ouverte à reposer sur un modèle transparent à
horizon fini, et quelle partie de sa garantie disparaît lorsque le modèle de
rejeu change?

Trois petits satellites sont largués sur presque la même orbite circulaire,
mais la mission exige qu'ils occupent des créneaux séparés de $120^\circ$. Ils
n'ont aucune propulsion. Chaque satellite peut plutôt pivoter entre une
attitude à faible traînée et une attitude à forte traînée. Une forte traînée
abaisse légèrement son orbite; le satellite plus bas va alors plus vite et
accumule de la phase par rapport aux autres. La capacité de commande est
faible, lente et irréversible, car chaque manœuvre consomme de l'altitude.

La traînée différentielle a servi à mettre en phase en orbite des
constellations de satellites sans propulsion
{cite:p}`Foster2018ConstellationPhasing`. Les programmes linéaires offrent un
modèle de planification utile pour ce mécanisme de commande
{cite:p}`Sin2018DifferentialDrag`. L'exemple ci-dessous est une reconstitution
à l'échelle pédagogique inspirée de cette littérature. Ce n'est pas la
reconstitution d'une campagne de vol particulière.

On planifie sur $N=180$ intervalles quotidiens. Les trois satellites partent
d'une altitude circulaire de $475$ km, avec les décalages de phase

$$
\varphi_{0}
=
\begin{bmatrix}-0.5&0&0.5\end{bmatrix}^{\mathsf T}\text{ degrés}
$$

et des vitesses angulaires relatives nulles. L'action

$$
0\leq u_{i,k}\leq1
$$

est la fraction du jour $k$ que le satellite $i$ passe dans son attitude à
forte traînée. Pour le modèle quotidien nominal, son état est

$$
x_{i,k}
=
\begin{bmatrix}
\varphi_{i,k}\\
\omega_{i,k}\\
\ell_{i,k}
\end{bmatrix},
$$

où $\varphi$ est la phase en degrés, $\omega$ la vitesse angulaire relative en
degrés par jour, et $\ell$ la perte d'altitude supplémentaire en kilomètres par
rapport au maintien dans l'attitude à faible traînée.

### De la physique de la traînée à un modèle linéaire quotidien

On utilise les unités SI dans la dérivation. Avec $h_0=475\text{ km}$, on pose

$$
a_0=(R_\mathrm{E}+h_0)\frac{10^3\text{ m}}{1\text{ km}}
=6\,853\,137\text{ m},
\qquad
n_0=\sqrt{\mu/a_0^3}.
$$

À la densité de référence $\rho_0$, faire passer le coefficient balistique de
$B_\mathrm{low}$ à $B_\mathrm{high}$ modifie le facteur surface-masse de

$$
\Delta\sigma
=\frac{1}{B_\mathrm{high}}-\frac{1}{B_\mathrm{low}}.
$$

En linéarisant les variations du demi-grand axe et du moyen mouvement sur une
journée, on obtient

$$
\begin{aligned}
d
&=\rho_0\Delta\sigma\sqrt{\mu a_0}\,\Delta t
\left(\frac{1\ {\rm km}}{10^3\ {\rm m}}\right),\\
\alpha
&=\frac{3n_0}{2a_0}
\rho_0\Delta\sigma\sqrt{\mu a_0}\,\Delta t^2
\left(\frac{180^\circ}{\pi\ {\rm rad}}\right),
\end{aligned}
$$

où $\Delta t=86400$ s. Pour

$$
\begin{gathered}
\rho_0=3\times10^{-13}\ {\rm kg/m^3},\qquad
B_\mathrm{low}=60\ {\rm kg/m^2},\qquad
B_\mathrm{high}=20\ {\rm kg/m^2},\\
R_\mathrm{E}=6378.137\ {\rm km},\qquad
\mu=3.986004418\times10^{14}\ {\rm m^3/s^2},
\end{gathered}
$$

les coefficients valent

$$
\alpha=0.0544503\ {\rm deg/jour^2},
\qquad
d=0.0451572\ {\rm km/jour}.
$$

La dynamique quotidienne est donc

$$
x_{i,k+1}
=
\underbrace{
\begin{bmatrix}
1&1&0\\
0&1&0\\
0&0&1
\end{bmatrix}}_{A}
x_{i,k}
+
\underbrace{
\begin{bmatrix}
\alpha/2\\
\alpha\\
d
\end{bmatrix}}_{B}
u_{i,k}.
$$

La phase reçoit la moitié de la nouvelle vitesse quotidienne pendant
l'intervalle, la vitesse cumule l'accélération induite par la traînée, et la
perte d'altitude supplémentaire s'accumule de façon monotone.

### Un plan fini en boucle ouverte

On définit la matrice de différences cycliques

$$
G=
\begin{bmatrix}
-1&1&0\\
0&-1&1\\
1&0&-1
\end{bmatrix}.
$$

La cible terminale est

$$
G\varphi_N
\approx
\begin{bmatrix}120\\120\\-240\end{bmatrix}
\text{ degrés},
\qquad
G\omega_N\approx0,
$$

avec des tolérances de $0.1$ degré et de $0.002$ degré par jour. Ce sont des
différences orientées non enroulées. Comme les lignes de $G$ ont une somme
nulle, la troisième cible est $-240^\circ$, ce qui représente la même
séparation circulaire que $+120^\circ$ modulo $360^\circ$ tout en conservant un
système de coordonnées non enroulées cohérent.

Le programme linéaire principal minimise la pire perte d'altitude
supplémentaire finale :

$$
\begin{aligned}
\underset{x,u,z}{\operatorname{minimiser}}\quad&z\\
\text{sous les contraintes}\quad&
x_{i,k+1}=Ax_{i,k}+Bu_{i,k},\\
&\ell_{i,N}\leq z,\qquad 0\leq u_{i,k}\leq1,\\
&\left\lVert G\varphi_N-
\begin{bmatrix}120&120&-240\end{bmatrix}^{\mathsf T}
\right\rVert_\infty\leq0.1,\\
&\lVert G\omega_N\rVert_\infty\leq0.002.
\end{aligned}
$$

Un second programme linéaire maintient $z$ à son optimum principal, à la
tolérance numérique près, et minimise

$$
\sum_{i=1}^{3}\sum_{k=0}^{N-2}
\left|u_{i,k+1}-u_{i,k}\right|.
$$

Cette étape lexicographique choisit, dans l'ensemble des solutions optimales
du programme principal, un élément à faible variation, sans modifier l'objectif
de pire perte au-delà de la tolérance numérique.

### Rejeu nominal et non linéaire

Le plan linéaire est d'abord déroulé dans le modèle utilisé par l'optimiseur.
Exactement les mêmes $u_{i,k}$ sont ensuite rejoués, sans réoptimisation, dans
le modèle orbital non linéaire

$$
\dot a_i
=-\rho(a_i,t)\,\sigma(u_i)\sqrt{\mu a_i},
\qquad
\dot\theta_i
=\sqrt{\frac{\mu}{a_i^3}},
$$

où

$$
\begin{aligned}
\sigma(u)
&=\frac{1-u}{B_\mathrm{low}}+\frac{u}{B_\mathrm{high}},\\
\rho(a,t)
&=\rho_0
\left[
0.90+0.15\sin\left(\frac{2\pi t}{26\text{ jours}}+\frac{\pi}{6}\right)
\right]
\exp\left(\frac{h_0-h_i}{60\text{ km}}\right),
\qquad
h_i=\frac{a_i-R_\mathrm{E}}{10^3}\text{ km}.
\end{aligned}
$$

Ici, $a_i$ et $R_\mathrm{E}$ sont en mètres dans les équations orbitales, si
bien que $h_i$ est l'altitude correspondante en kilomètres. Dans l'affichage,
la courbe d'altitude nominale est l'approximation $475-\ell_i$ km du modèle de
planification et omet donc la décroissance commune en faible traînée; la
courbe non linéaire indique l'altitude orbitale absolue.

La trajectoire non linéaire est intégrée par RK4 à pas horaire et vérifiée par
un rejeu à 30 minutes. Ce modèle à densité variable est un test de résistance
pédagogique déterministe, et non une reconstitution de dynamique de vol. La
suite complète des commandes est connue dès le jour zéro; les courbes d'état
ci-dessous ne sont révélées que jusqu'à la tête de lecture.

```{code-cell} python
:tags: [remove-input]
:label: fig-cubesat-differential-drag
:caption: Un même plan de traînée différentielle en boucle ouverte est évalué par deux modèles du procédé. Le déroulement linéaire nominal atteint les tolérances de créneau cyclique et de vitesse relative. Le rejeu non linéaire à densité variable utilise le plan inchangé et révèle l'écart de phase accumulé. Les diagrammes d'orbite utilisent un rayon fixe; les écarts d'altitude sont donnés numériquement plutôt qu'exagérés géométriquement.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from cubesat_replay import render_cubesat_replay

display(HTML(render_cubesat_replay(
    Path("artifacts/cubesat/textbook_results.json"),
    replay_id="cubesat-differential-drag-replay",
    fallback_id="fig-cubesat-differential-drag-fallback",
)))
```

:::{figure} _static/cubesat/differential-drag.svg
:label: fig-cubesat-differential-drag-fallback
:class: pdf-fallback
:alt: Écarts finaux de la constellation, perte d'altitude, historiques d'erreur de phase en nominal et en non linéaire, et plan commun de traînée en boucle ouverte pour trois satellites.

Audit statique du plan immuable de traînée différentielle. Le livre en ligne
ajoute une lecture synchronisée et le défilement, tout en gardant visible dès
le départ la carte thermique complète des commandes planifiées.
:::

```{include} artifacts/cubesat/results.md
```

{download}`Télécharger le plan en boucle ouverte (CSV) <artifacts/cubesat/open_loop_plan.csv>`

{download}`Télécharger les mesures de l'audit (CSV) <artifacts/cubesat/metrics.csv>`

Le déroulement nominal établit la faisabilité pour le modèle d'optimisation.
Le rejeu non linéaire teste le même plan sous le changement de modèle déclaré,
et les contraintes de phase terminales sont violées. [Fermer la boucle par
replanification](receding-horizon-control.md#closing-the-loop-by-replanning)
remplacera le programme immuable par des commandes qui peuvent changer à
l'arrivée de nouvelles mesures de l'état.

L'exemple contient déjà les ingrédients d'un **problème de contrôle optimal
en temps discret** (PCOTD) : un état $x_{i,k}$, une commande bornée $u_{i,k}$,
une application de transition, des contraintes terminales et un objectif
cumulé sur un horizon fini. Formalisons maintenant cette structure.

## Problèmes de contrôle optimal en temps discret (PCOTD)

Quels variables, coûts, dynamique et contraintes placent le planificateur de
satellites dans un gabarit réutilisable de commande à horizon fini?

Considérons un système décrit par un **état** $\mathbf{x}_t \in \mathbb{R}^n$, qui résume tout ce qu'il faut pour prédire son évolution. À chaque étape $t$, on peut agir sur le système par une **entrée de commande** $\mathbf{u}_t \in \mathbb{R}^m$. La dynamique précise comment l'état évolue :

$$
\mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t),
$$

où $\mathbf{f}_t$ peut être non linéaire ou variable dans le temps. On suppose connu l'état initial $\mathbf{x}_1$.

Le but est de choisir une suite de commandes $\mathbf{u}_1,\dots,\mathbf{u}_{T-1}$ qui rende la trajectoire souhaitable. Mais souhaitable en quel sens? Cela dépend d'une **fonction objectif**, qui comprend souvent deux composantes :

$$
\text{(i) coût d'étape : } c_t(\mathbf{x}_t,\mathbf{u}_t), \qquad \text{(ii) coût terminal : } c_T(\mathbf{x}_T).
$$

Le coût d'étape traduit des pénalités continues comme l'énergie, le délai ou le risque. Le coût terminal mesure la valeur (ou le coût) de terminer dans un état donné. Ensemble, ils forment un problème de Bolza en temps discret avec contraintes de chemin et bornes :

$$
\begin{aligned}
    \text{minimiser} \quad & c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
    \text{sous les contraintes} \quad & \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) \\
                            & \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t) \leq \mathbf{0} \\
                            & \mathbf{x}_{\text{min}} \leq \mathbf{x}_t \leq \mathbf{x}_{\text{max}} \\
                            & \mathbf{u}_{\text{min}} \leq \mathbf{u}_t \leq \mathbf{u}_{\text{max}} \\
    \text{étant donné} \quad & \mathbf{x}_1 = \mathbf{x}_0 \enspace .
\end{aligned}
$$

Dans l'exemple des satellites, $\mathbf{x}_t$ empile les trois états de phase,
de vitesse relative et de perte d'altitude; $\mathbf{u}_t$ empile les trois
fractions quotidiennes de traînée; et $\mathbf{f}_t$ applique les copies
bloc-diagonales de l'application quotidienne. Les inégalités terminales
imposent les tolérances de créneau cyclique et de vitesse, tandis que la
variable d'épigraphe $z$ représente la pire perte d'altitude. Cette
correspondance montre aussi pourquoi les valeurs d'état peuvent apparaître
explicitement comme variables de décision, même lorsqu'un déroulement
déterministe pourrait les reconstruire à partir des commandes.

Écrit de cette façon, il peut sembler évident que les variables de décision sont les commandes $\mathbf{u}_t$. Après tout, dans la plupart des descriptions intuitives de la commande, on pense à choisir des entrées pour agir sur le système. Remarquez pourtant que, dans le programme ci-dessus, toute la trajectoire d'état apparaît aussi comme un ensemble de variables, reliées aux commandes par les contraintes de dynamique. C'est voulu : cela reflète une manière d'écrire le problème qui rend les contraintes explicites.

Pourquoi introduire les $\mathbf{x}_t$ comme variables de décision si on peut les simuler en avant à partir des commandes? Beaucoup de lecteurs hésitent ici, et la question est naturelle : *si le modèle est déterministe et que $\mathbf{x}_1$ est connu, pourquoi ne pas choisir $\mathbf{u}_{1:T-1}$ et calculer $\mathbf{x}_{2:T}$ au fur et à mesure?* Cette intuition mène au **tir simple**, une méthode sur laquelle nous reviendrons bientôt.

Dès cette formulation, toutefois, **la structure du problème compte**. L'ignorer peut nous compliquer beaucoup la vie, pour deux raisons :

* **La dimension croît avec l'horizon.** Pour un horizon de longueur $T$, le programme compte environ $(T-1)(m+n)$ variables de décision.
* **Le couplage temporel.** Chaque commande influe sur tous les états et les coûts futurs. L'ensemble réalisable n'est pas une simple boîte, mais une variété étroite définie par la dynamique.

Ensemble, ces caractéristiques expliquent pourquoi des méthodes spécialisées existent et pourquoi la façon d'écrire le problème influe sur les algorithmes utilisables. Garder les états explicites ou les éliminer par simulation en avant détermine la taille du problème, son conditionnement et les compromis entre robustesse et effort de calcul.

## Existence de solutions et conditions d'optimalité

Écrire le problème comme un programme non linéaire ne garantit ni qu'un
minimiseur existe ni qu'un candidat est localement optimal. Quelles conditions
fournissent ces deux affirmations?

Maintenant que le problème d'optimisation est posé, on peut se demander : a-t-il toujours une solution? Et si oui, comment la reconnaître? Ces questions mènent aux conditions de faisabilité et d'optimalité.


### Existence de solutions

Remarquons d'abord que rien dans l'énoncé du problème n'exigeait que la dynamique

$$
\mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)
$$

soit stable. En fait, beaucoup de problèmes intéressants portent sur des systèmes instables; pensons à l'équilibre d'un pendule inversé ou au pilotage d'un engin spatial. Ce qui compte, c'est que la dynamique soit **bien définie** : étant donné un couple état-commande, la règle $\mathbf{f}_t$ produit un état suivant valide.

En temps continu, on exige habituellement que $\mathbf{f}$ soit continue (souvent lipschitzienne) en $\mathbf{x}$ pour que l'EDO ait une solution unique sur l'horizon considéré. En temps discret, l'exigence est plus légère : il suffit que l'application de mise à jour soit bien posée.

L'existence dépend aussi de la **faisabilité**. Une suite de commandes candidate doit produire une trajectoire qui respecte toutes les contraintes : la dynamique, les éventuelles bornes sur l'état et la commande, et les exigences terminales. S'il n'existe aucune telle suite, l'ensemble réalisable est vide et le problème n'a pas de solution. Cela peut arriver si les contraintes sont trop strictes, ou si le système n'est pas commandable à partir de la condition initiale donnée.


### Conditions d'optimalité

Supposons l'ensemble réalisable non vide. Des conditions nécessaires de
minimum local découlent du théorème des multiplicateurs de Lagrange. Elles
fournissent des équations pour les solutions candidates, même si les
satisfaire ne suffit pas, à lui seul, à établir un minimum. Pour un problème
lisse

$$
\begin{aligned}
\min_{\mathbf{z}}\quad & F(\mathbf{z})\\
\text{s.c.}\quad & H(\mathbf{z})=\mathbf{0},\\
& G(\mathbf{z})\le \mathbf{0},
\end{aligned}
$$

où $G$ empile des inégalités de la forme $g_i\leq0$ et $H$ des égalités de la
forme $h_i=0$, on définit le **lagrangien**

$$
\mathcal{L}(\mathbf{z},\boldsymbol{\lambda},\boldsymbol{\mu})
= F(\mathbf{z})+\boldsymbol{\lambda}^{\top}H(\mathbf{z})+\boldsymbol{\mu}^{\top}G(\mathbf{z}),\qquad \boldsymbol{\mu}\ge \mathbf{0}.
$$

Le signe positif devant le terme d'inégalité découle de la convention
$G(\mathbf z)\le \mathbf0$ : une violation $G_i(\mathbf z)>0$ augmente le
lagrangien lorsque $\mu_i>0$. Les multiplicateurs d'égalité ne sont pas
contraints. Pour un point candidat $\mathbf z$, l'**ensemble actif** est

$$
\mathcal{A}(\mathbf{z}) \;=\; \{\, i \;:\; G_i(\mathbf{z})=0 \,\},
$$

tandis que les indices tels que $G_i(\mathbf{z})<0$ sont **inactifs**. Seules les inégalités actives peuvent porter des multiplicateurs positifs.

Une **qualification de contraintes** est une hypothèse de régularité sous
laquelle un minimum local admet des multiplicateurs de Lagrange. Un choix
suffisant est la **qualification d'indépendance linéaire des contraintes
(LICQ)** : au point candidat, les gradients des égalités et ceux des
inégalités actives sont linéairement indépendants. Si $J_H$ et
$J_{G_{\mathcal A}}$ ont ces gradients pour lignes, la LICQ exige que

$$
\begin{bmatrix}J_H(\mathbf z^\star)\\J_{G_{\mathcal A}}(\mathbf z^\star)\end{bmatrix}
\quad\text{soit de rang plein en lignes.}
$$

Cette hypothèse garantit l'existence de multiplicateurs en un minimum local.
Elle ne rend pas à elle seule inversible le système de Newton des conditions
d'optimalité; cela dépend aussi de la courbure de l'objectif et des
contraintes.

Sous une telle qualification de contraintes, tout minimiseur local $\mathbf{z}^\star$ admet des multiplicateurs $(\boldsymbol{\lambda}^\star,\boldsymbol{\mu}^\star)$ qui satisfont les **conditions de Karush-Kuhn-Tucker (KKT)** :

$$
\begin{aligned}
&\text{stationnarité :} && \nabla_{\mathbf{z}}\mathcal{L}(\mathbf{z}^\star,\boldsymbol{\lambda}^\star,\boldsymbol{\mu}^\star)=\mathbf{0},\\
&\text{faisabilité primale :} && H(\mathbf{z}^\star)=\mathbf{0},\quad G(\mathbf{z}^\star)\le \mathbf{0},\\
&\text{faisabilité duale :} && \boldsymbol{\mu}^\star\ge \mathbf{0},\\
&\text{complémentarité :} && \mu_i^\star\,G_i(\mathbf{z}^\star)=0\quad \text{pour tout } i.
\end{aligned}
$$

La complémentarité exige $\mu_i^\star=0$ lorsque $G_i(\mathbf z^\star)<0$. Pour
une contrainte saturée, le multiplicateur peut être positif ou nul. Sous des
hypothèses de sensibilité convenables, les multiplicateurs quantifient les
variations de la valeur optimale lorsque les seconds membres des contraintes
changent, avec des signes fixés par la convention de perturbation choisie.

Dans les problèmes de trajectoire, $\mathbf z$ empile les états et les
commandes, $H$ impose la dynamique, et $G$ regroupe les bornes et les
contraintes de chemin. Les multiplicateurs d'égalité associés à la dynamique
deviennent les états adjoints de Pontryaguine lorsque le résidu s'écrit
$\mathbf f_t-\mathbf x_{t+1}$. Écrire le résidu dans l'ordre inverse change le
signe du multiplicateur. Les multiplicateurs des égalités terminales sont
notés séparément $\boldsymbol\nu$ dans cette dérivation. Comme une dynamique
non linéaire rend en général ces programmes non convexes, KKT est un test
nécessaire du premier ordre sous hypothèse de régularité. Il n'implique en
général ni un minimum local du lagrangien ni un point de selle global.

Si la LICQ n'est pas satisfaite, une autre qualification de contraintes peut
encore s'appliquer. Sans qualification applicable, les multiplicateurs peuvent
ne pas exister ou ne pas être uniques. Des contraintes de dynamique ou de
chemin redondantes peuvent aussi rendre les systèmes numériques singuliers ou
mal conditionnés.

### Des conditions KKT aux algorithmes

Le lagrangien donne une représentation min-max exacte du problème sous
contraintes, que le problème soit convexe ou non :

$$
p^\star=\inf_{\mathbf z}
\sup_{\boldsymbol\lambda,\,\boldsymbol\mu\ge\mathbf0}
\mathcal L(\mathbf z,\boldsymbol\lambda,\boldsymbol\mu),
\qquad
\mathcal L=F+\boldsymbol\lambda^\top H+\boldsymbol\mu^\top G.
$$

Ici, $\boldsymbol\lambda$ n'est pas contraint. Pour un $\mathbf z$ réalisable,
le supremum intérieur vaut $F(\mathbf z)$. Si une égalité est violée, son
multiplicateur peut choisir un signe et une amplitude qui rendent le
lagrangien arbitrairement grand. Si $G_i(\mathbf z)>0$, faire tendre $\mu_i$
vers l'infini a le même effet. La réponse du joueur des multiplicateurs
attribue donc $+\infty$ à tout choix non réalisable.

Inverser l'ordre donne la valeur duale, une borne inférieure de la valeur
primale :

$$
d^\star=\sup_{\boldsymbol\lambda,\,\boldsymbol\mu\ge\mathbf0}
\inf_{\mathbf z}\mathcal L(\mathbf z,\boldsymbol\lambda,\boldsymbol\mu)
\le p^\star.
$$

La distinction tient à ce que les multiplicateurs puissent répondre à chaque
candidat ou doivent être fixés avant le choix du candidat. Un **point de
selle** exige un couple de multiplicateurs fixe pour lequel $\mathbf z^\star$
minimise globalement le lagrangien, ainsi que des multiplicateurs optimaux
face à ce $\mathbf z^\star$ fixé. Un tel couple vérifie

$$
\mathcal L(\mathbf z^\star,\boldsymbol\lambda,\boldsymbol\mu)
\le\mathcal L(\mathbf z^\star,\boldsymbol\lambda^\star,\boldsymbol\mu^\star)
\le\mathcal L(\mathbf z,\boldsymbol\lambda^\star,\boldsymbol\mu^\star)
$$

pour tout $\mathbf z$, tout $\boldsymbol\lambda$ et tout
$\boldsymbol\mu\ge\mathbf0$. Résoudre seulement le problème min-max primal ne
garantit pas cette propriété. Par exemple, minimiser $-z^2$ sous la contrainte
$z=0$ a pour unique optimum réalisable $z^\star=0$ et pour multiplicateur KKT
$\lambda^\star=0$, et pourtant $\mathcal L(z,0)=-z^2$ atteint son maximum en
zéro.

Lorsque $F$ est convexe, que chaque $G_i$ est convexe et que $H$ est affine,
le lagrangien est convexe en $\mathbf z$. La stationnarité KKT garantit alors
son minimum global; la faisabilité et la complémentarité donnent l'autre
inégalité du point de selle. Un point KKT, s'il existe, suffit donc à obtenir
un point de selle global. La condition de Slater, soit l'existence d'un point
où $H=0$ et $G_i<0$ pour toute inégalité, garantit l'existence de
multiplicateurs convenables en un optimum fini atteint dans ce cadre convexe.
L'[annexe de programmation non linéaire](appendix_nlp.md#the-lagrangian-duality-and-optimality-conditions)
dérive ces énoncés et les distingue des conditions nécessaires utilisées en
optimisation de trajectoires non convexe.

#### Dynamique primale-duale par gradient (Arrow-Hurwicz)

Une méthode du premier ordre fait un pas de descente sur les variables
primales et un pas de montée sur les multiplicateurs, en projetant les
multiplicateurs d'inégalité sur leur ensemble admissible :

$$
\begin{aligned}
\mathbf{z}^{k+1} &= \mathbf{z}^{k}-\alpha_k\big(\nabla F(\mathbf{z}^{k})+J_H(\mathbf{z}^{k})^{\top}\boldsymbol{\lambda}^{k}+J_G(\mathbf{z}^{k})^{\top}\boldsymbol{\mu}^{k}\big),\\[2mm]
\boldsymbol{\lambda}^{k+1} &= \boldsymbol{\lambda}^{k}+\beta_k\,H(\mathbf{z}^{k}),\\[1mm]
\boldsymbol{\mu}^{k+1} &= \Pi_{\ge 0}\!\big(\boldsymbol{\mu}^{k}+\beta_k\,G(\mathbf{z}^{k})\big).
\end{aligned}
$$

Ici, $\Pi_{\ge0}$ ramène à zéro chaque composante négative. Avec la convention
$G\le0$, une inégalité violée augmente son multiplicateur. Ces mises à jour
finies n'effectuent pas le supremum intérieur de la représentation primale
exacte, et l'existence d'un point de selle ne garantit pas à elle seule leur
convergence. Même le cas bilinéaire convexe $\mathcal L(z,\lambda)=\lambda z$
donne, pour des pas positifs fixes, des mises à jour simultanées de valeurs
propres $1\pm i\sqrt{\alpha\beta}$, dont le module dépasse un.

La convergence exige des hypothèses de courbure supplémentaires ou des
modifications de l'algorithme. Les méthodes de pénalité et de lagrangien
augmenté ajoutent des termes qui découragent les violations. On peut par
exemple utiliser le lagrangien pénalisé

$$
\mathcal{L}_\rho(\mathbf{z},\boldsymbol{\lambda},\boldsymbol{\mu})
= \mathcal{L}(\mathbf{z},\boldsymbol{\lambda},\boldsymbol{\mu})
+\tfrac{\rho}{2}\|H(\mathbf{z})\|^2
+\tfrac{\rho}{2}\|\max\{0,G(\mathbf{z})\}\|^2,
$$

où $\rho>0$ pondère les carrés des résidus d'égalité et des violations
d'inégalité. Ces termes modifient la courbure loin de la faisabilité. Leur
emploi exige encore une stratégie convenable de choix du pas ou de
sous-problème; les ajouter ne garantit pas à lui seul la convergence.

(sec-sqp-newton)=
#### La PQS comme méthode de Newton sur le système KKT (cas des égalités)

Avec **seulement des contraintes d'égalité** $H(\mathbf{z})=\mathbf{0}$, on écrit les conditions du premier ordre

$$
\nabla_{\mathbf{z}}\mathcal{L}(\mathbf{z},\boldsymbol{\lambda})=\mathbf{0},
\qquad
H(\mathbf{z})=\mathbf{0},
\quad \text{où }\mathcal{L}=F+\boldsymbol{\lambda}^{\top}H.
$$

Appliquer la méthode de Newton à ce système donne la résolution du système KKT linéaire

$$
\begin{bmatrix}
\nabla_{\mathbf{z}\mathbf{z}}^2\mathcal{L}(\mathbf{z}^k,\boldsymbol{\lambda}^k) & J_H(\mathbf{z}^k)^{\top}\\
J_H(\mathbf{z}^k) & 0
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{z}\\ \Delta \boldsymbol{\lambda}
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla_{\mathbf{z}}\mathcal{L}(\mathbf{z}^k,\boldsymbol{\lambda}^k)\\
H(\mathbf{z}^k)
\end{bmatrix}.
$$

Le même pas a une interprétation en optimisation. Avec
$B_k=\nabla^2_{\mathbf z\mathbf z}\mathcal L(\mathbf z^k,\boldsymbol\lambda^k)$,
la **programmation quadratique successive (PQS)** choisit un déplacement
$\Delta\mathbf z$ en résolvant le programme quadratique

$$
\boxed{
\begin{aligned}
\underset{\Delta\mathbf z}{\operatorname{minimiser}}\quad&
\nabla F(\mathbf z^k)^\top\Delta\mathbf z
+\frac12\Delta\mathbf z^\top B_k\Delta\mathbf z\\
\text{sous la contrainte}\quad&
H(\mathbf z^k)+J_H(\mathbf z^k)\Delta\mathbf z=\mathbf0.
\end{aligned}}
$$

L'objectif quadratique conserve une pente, $\nabla F(\mathbf z^k)$, et une
matrice de courbure, $B_k$. Son gradient varie linéairement avec le
déplacement proposé : $\nabla F(\mathbf z^k)+B_k\Delta\mathbf z$. Le modèle
peut ainsi prédire comment le bénéfice d'un mouvement dans une direction
évolue à mesure qu'on s'éloigne. La constante $F(\mathbf z^k)$ peut être omise,
car elle n'influe pas sur le pas qui minimise le modèle.

Linéariser une contrainte consiste à remplacer sa valeur au point d'essai par
sa valeur courante plus sa variation prédite au premier ordre. Le programme
quadratique demande donc à $J_H(\mathbf z^k)\Delta\mathbf z$ d'annuler le
résidu courant $H(\mathbf z^k)$. En un itéré réalisable, cela devient
$J_H(\mathbf z^k)\Delta\mathbf z=\mathbf0$ : le pas appartient à l'espace
tangent aux contraintes. Ce sont des contraintes *affines* sur le pas,
c'est-à-dire des expressions linéaires plus des constantes.

Pourquoi utiliser la hessienne du lagrangien dans l'objectif quadratique? Pour
des contraintes d'égalité, elle vaut

$$
B_k=\nabla^2 F(\mathbf z^k)
+\sum_i\lambda_i^k\nabla^2 H_i(\mathbf z^k).
$$

Les termes supplémentaires portent l'information sur la courbure des
contraintes, pondérée par leurs multiplicateurs. Le programme quadratique
utilise donc un modèle quadratique conçu pour le problème sous contraintes; il
diffère en général du développement de Taylor d'ordre deux de $F$ seule. Son
lien avec la méthode de Newton explique ce choix. Si
$\boldsymbol\lambda_{\mathrm{PQ}}$ désigne le multiplicateur de l'égalité du
programme quadratique, ses conditions de stationnarité et de faisabilité sont

$$
\begin{bmatrix}
B_k & J_H(\mathbf z^k)^\top\\
J_H(\mathbf z^k) & 0
\end{bmatrix}
\begin{bmatrix}
\Delta\mathbf z\\ \boldsymbol\lambda_{\mathrm{PQ}}
\end{bmatrix}
=-
\begin{bmatrix}
\nabla F(\mathbf z^k)\\ H(\mathbf z^k)
\end{bmatrix}.
$$

Substituons
$\boldsymbol\lambda_{\mathrm{PQ}}=\boldsymbol\lambda^k+
\Delta\boldsymbol\lambda$ et passons
$J_H(\mathbf z^k)^\top\boldsymbol\lambda^k$ dans le membre de droite. On
retrouve exactement le système de Newton ci-dessus. Le multiplicateur du
programme quadratique est donc la nouvelle estimation du multiplicateur pour
un pas complet; son écart à l'ancienne estimation est l'incrément de Newton
du multiplicateur.

Avec des inégalités $G(\mathbf z)\le\mathbf0$, le programme quadratique
contient aussi

$$
G(\mathbf z^k)+J_G(\mathbf z^k)\Delta\mathbf z\le\mathbf0.
$$

La hessienne $B_k$ inclut alors $\sum_i\mu_i^k\nabla^2G_i(\mathbf z^k)$, et le
programme quadratique renvoie aussi des multiplicateurs d'inégalité
$\boldsymbol\mu_{\mathrm{PQ}}\ge\mathbf0$.

L'avantage de calcul vient de ce que ces matrices restent fixes pendant chaque
résolution du programme quadratique. Le solveur travaille avec une fonction
quadratique et des frontières de contraintes planes, sans redériver sans cesse
les fonctions non linéaires d'origine. Dans le cas des égalités, une seule
résolution du système KKT linéaire donne l'unique minimiseur du programme
quadratique lorsque $J_H$ est de rang plein en lignes et que $B_k$ est définie
positive sur les directions non nulles $\mathbf v$ telles que
$J_H\mathbf v=\mathbf0$. Avec des inégalités, une **méthode d'ensemble actif**
choisit des contraintes candidates saturées et résout des programmes
quadratiques sous contraintes d'égalité, en mettant cette sélection à jour au
besoin. Une **méthode de points intérieurs** traite les inégalités au moyen de
variables d'écart et d'équations de complémentarité perturbées, en résolvant
des systèmes linéaires apparentés. Les deux approches peuvent tirer parti de
factorisations de matrices creuses. Le
[guide NEOS des algorithmes de programmation quadratique](https://neos-guide.org/guide/algorithms/qp/)
développe ces deux approches.

Des sous-problèmes consécutifs partagent souvent la même structure de
coefficients non nuls; les solveurs peuvent donc réutiliser l'ordre
d'élimination creuse et partir des estimations primales et duales
précédentes. Une factorisation numérique peut aussi être réutilisée lorsque sa
matrice de coefficients ne change pas; modifier la hessienne ou les
jacobiennes des contraintes exige en général de la mettre à jour ou de la
recalculer. [OSQP](https://osqp.org/docs/) offre une implémentation de la
réutilisation de factorisations et du démarrage à chaud pour les programmes
quadratiques convexes. En optimisation de trajectoires, chaque contrainte de
dynamique couple des pas de temps voisins, ce qui donne aux matrices une
structure creuse par blocs. Avec des coûts quadratiques par étape, une
dynamique linéarisée et aucune inégalité supplémentaire, des programmes
quadratiques convenables se résolvent par une récurrence de Riccati, comme pour
la commande linéaire quadratique.

Loin d'une solution, la hessienne exacte du lagrangien ne donne pas
forcément un programme quadratique convexe. Les méthodes de PQS pratiques la
modifient souvent ou utilisent une approximation définie positive. Pour un
objectif de moindres carrés, une approximation de Gauss-Newton fournit une
courbure semi-définie positive; lui ajouter un multiple positif de l'identité
la rend définie positive. Ces choix donnent des pas de Newton modifiés. Enfin,
satisfaire les contraintes linéarisées ne garantit pas la faisabilité non
linéaire en $\mathbf z^k+\Delta\mathbf z$. Une recherche linéaire ou une
région de confiance vérifie le progrès à l'aide de l'objectif et des
contraintes d'origine et règle la longueur du déplacement avant de construire
le programme quadratique suivant.


## Autres sources de problèmes de contrôle optimal en temps discret

Au-delà de la dynamique physique échantillonnée, quels calculs et quels
modèles en temps continu produisent la même structure d'optimisation couplée
dans le temps?

Le planificateur de satellites part d'un modèle quotidien volontairement
discrétisé. D'autres problèmes sont discrets parce que les décisions
surviennent naturellement par étapes, et d'autres encore héritent d'une
transition discrète de l'intégration numérique ou de l'exécution d'un
programme. Les deux constructions suivantes rendent ces derniers liens
explicites.

### PCOTD issus de la discrétisation de problèmes de contrôle optimal en temps continu

Bien que beaucoup d'applications soient discrètes par nature, il est aussi courant d'obtenir un PCOTD en discrétisant une formulation en temps continu. Considérons un système sur $[0, T_c]$ donné par

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t),t), \qquad \mathbf{x}(0) = \mathbf{x}_0.
$$

Choisissons un pas $\Delta > 0$ et une grille $t_k = k\,\Delta$. Un schéma d'intégration à un pas induit une application discrète $\mathbf{F}_\Delta$ telle que

$$
\mathbf{x}_{k+1} = \mathbf{F}_\Delta(\mathbf{x}_k, \mathbf{u}_k, t_k),\qquad k=0,\dots, T-1,
$$

où, par exemple, le schéma d'Euler explicite donne $\mathbf{F}_\Delta(\mathbf{x},\mathbf{u},t) = \mathbf{x} + \Delta\,\mathbf{f}(\mathbf{x},\mathbf{u},t)$. Le problème de contrôle optimal en temps discret qui en résulte prend la forme de Bolza avec cette dynamique induite :

$$
\begin{aligned}
\min_{\{\mathbf{x}_k,\mathbf{u}_k\}}\; & c_T(\mathbf{x}_T) + \sum_{k=0}^{T-1} c_k(\mathbf{x}_k,\mathbf{u}_k) \\
\text{s.c.}\; & \mathbf{x}_{k+1} - \mathbf{F}_\Delta(\mathbf{x}_k,\mathbf{u}_k, t_k) = 0,\quad k=0,\dots,T-1, \\
& \mathbf{x}_0 = \mathbf{x}_\mathrm{init}.
\end{aligned}
$$

### Programmes informatiques comme PCOTD et programmation différentiable

Il est souvent utile de voir un programme informatique lui-même comme un système dynamique en temps discret. Que l'**état du programme** regroupe la mémoire, les tampons et les variables intermédiaires, et que la **commande** représente les entrées ou les décisions réglables à chaque pas. Un pas d'exécution définit une application de transition

$$
\mathbf{x}_{k+1}=\Phi_k(\mathbf{x}_k,\mathbf{u}_k),
$$

et un objectif scalaire (par exemple une perte, une erreur, un temps d'exécution ou une énergie) donne un PCOTD :

$$
\min_{\{\mathbf{u}_k\}} \; c_T(\mathbf{x}_T)+\sum_{k=0}^{T-1} c_k(\mathbf{x}_k,\mathbf{u}_k)
\quad\text{s.c.}\quad \mathbf{x}_{k+1}=\Phi_k(\mathbf{x}_k,\mathbf{u}_k).
$$

En programmation différentiable (par exemple avec JAX ou PyTorch), l'application composée $\Phi_{T-1}\circ\cdots\circ\Phi_0$ est différentiable, ce qui permet la différentiation automatique en mode inverse et une optimisation de trajectoires efficace par gradient. Lorsque certaines parties du programme ne sont pas différentiables (branchements discrets, simulateurs à événements), on peut encore résoudre les PCOTD par des méthodes sans dérivées ou à gradient faible (par exemple différences finies, SPSA, Nelder-Mead, CMA-ES ou stratégies d'évolution), éventuellement combinées à un lissage, à des relaxations ou à des estimateurs stochastiques pour franchir les régions non lisses.

(example-offline-frequency-planning-for-inference)=
#### Exemple : planification hors ligne des fréquences pour l'inférence

Le service d'inférence du [chapitre sur les interfaces de modèle](model-interfaces.md#inference-serving-as-a-controlled-system)
est aussi un programme dont la vitesse d'exécution est commandable. Ses
courbes de débit de service et de puissance par phase proviennent d'un profil
mesuré sur un NVIDIA L4. Les trajectoires de requêtes et de files ci-dessous
proviennent d'un simulateur étalonné avec ces courbes, et non de l'exécution
de chaque correcteur sur le GPU. La règle d'ordonnancement est fixée. Chaque
pas de simulation de 0.1 seconde donne la priorité au décodage ou commence par
un bloc de préremplissage d'au plus 512 jetons. Si ce bloc se termine plus
tôt, le budget de service restant peut retourner au décodage. Le décodage
actif reçoit la priorité un pas sur deux ou lorsque le cache est sollicité. Ce
modèle d'entrelacement réduit n'est pas une reproduction de l'ordonnanceur de
vLLM utilisé pour le profilage. La suite de commandes précise une fréquence
GPU normalisée pour chaque seconde d'un horizon de 60 secondes :

$$
u_k=\frac{f_k-f_{\min}}{f_{\max}-f_{\min}}\in[0,1].
$$

Un état agrégé regroupe le travail de préremplissage en file, le travail de
décodage actif, la température et la fréquence précédente :

$$
x_k=(p_k,d_k,T_k,f_{k-1}),
\qquad
x_{k+1}=F_k(x_k,u_k,w_k).
$$

La perturbation $w_k$ contient les instants d'arrivée et les longueurs
d'invite prévus pour la seconde $k$. Les longueurs de sortie futures restent
cachées. Le planificateur les remplace par la longueur de sortie espérée de la
loi de la trace, tandis que le rejeu au niveau des requêtes n'utilise chaque
longueur réalisée que comme perturbation. Le service et la puissance dans
$F_k$ interpolent le profil versionné, dont la provenance est affichée avec le
résultat. Le simulateur au niveau des requêtes reste hors de l'optimiseur et
valide le programme obtenu après la résolution.

La charge de travail de 60 secondes est choisie avant l'optimisation par une
règle déterministe filtrée selon la capacité. La règle parcourt les fenêtres
alignées sur dix secondes après normalisation de la charge, écarte toute
fenêtre dont le travail prévu dépasse la capacité de service de l'horizon à
fréquence maximale, et exige une rafale occupée qu'on puisse avancer de vingt
secondes. Parmi les fenêtres restantes, elle choisit la rafale admissible au
décalage dont le travail prévu est le plus grand; le travail total de la
fenêtre, puis l'instant source le plus précoce, départagent les égalités.
L'intervalle source retenu est $[890,950)$ secondes dans la trace normalisée,
et il est recalé pour commencer à zéro. Ses 48 requêtes demandent 51.936
secondes de travail prévu à fréquence maximale, soit 86.56 % de la capacité de
l'horizon. La rafale choisie occupe $[40,50)$ secondes après recalage et
contient 31 requêtes. Le rejeu décalé déplace ces requêtes vers $[20,30)$
secondes.

Le problème hors ligne utilise la prévision nominale complète des arrivées. Il
introduit une décision de service normalisée $\nu_k\in[\nu_{\min},1]$ et une
variable d'arriéré non négative $B_{k+1}$. Si $W_k$ est le travail qui arrive,
exprimé en secondes de service à la fréquence la plus élevée, leur bilan
fluide est relâché en

$$
\begin{aligned}
\underset{\nu_0,\ldots,\nu_{59},B_1,\ldots,B_{60}}
{\operatorname{minimiser}}\quad
&\sum_{k=0}^{59}\left[\alpha_P\nu_k+20B_{k+1}\right]
+20B_{60}\\
\text{sous les contraintes}\quad
&B_{k+1}\geq B_k+W_k-\Delta t\,\nu_k,\qquad B_0=0,\\
&B_{k+1}\geq0,\qquad \nu_{\min}\leq\nu_k\leq1.
\end{aligned}
$$

Le coefficient $\alpha_P$ est la pente d'une interpolation linéaire entre les
puissances normalisées aux fréquences la plus basse et la plus élevée. Le
terme d'arriéré tarife l'attente sur tout l'horizon, tandis que le terme
terminal supplémentaire décourage le report du travail au-delà de la seconde
60. C'est un programme linéaire, résolu avec HiGHS. Les décisions de service
sont converties en fréquences continues par la courbe de service profilée.
L'exécution arrondit chaque fréquence vers le bas à la fréquence demandée
profilée la plus proche, de sorte que la validation au niveau des requêtes
tient compte de l'ensemble fini d'actions de l'actionneur. Le rejeu indique
séparément la fréquence réalisée médiane mesurée correspondante; un niveau
demandé et sa fréquence réalisée ne coïncident pas forcément sous le plafond
de puissance expérimental.

Ce modèle de planification omet volontairement l'identité des requêtes, les
files propres à chaque phase, la vitesse de variation de l'horloge et l'état
thermique non linéaire. Le rejeu détaillé rétablit ces variables et indique
les violations de puissance, de température, de latence et de mémoire.
L'optimiseur fournit donc un plan hors ligne tiré d'un modèle agrégé
traitable, tandis que le rejeu audite les hypothèses qui ont servi à
l'obtenir.

L'expérience se demande ce qu'un programme de fréquences calculé une seule
fois gagne à disposer d'une prévision nominale parfaite des arrivées et des
longueurs d'invite, et ce qu'il perd lorsque cette prévision est fausse. Les
longueurs de sortie futures restent incertaines dans les deux cas. Le même
plan est rejoué deux fois. Le rejeu nominal utilise la prévision fournie à
l'optimiseur. Le rejeu décalé utilise les instants d'arrivée avancés définis
ci-dessus. Aucune réoptimisation n'a lieu après le début de l'un ou l'autre
rejeu.

```{code-cell} python
:tags: [remove-input]
:label: fig-inference-open-loop
:caption: Le programme de fréquences optimisé est calculé une seule fois à partir de la prévision nominale des requêtes sur 60 secondes. Le rejeu décalé avance de vingt secondes la rafale de travail retenue, en gardant fixes les fréquences planifiées. Les deux trajectoires au niveau des requêtes sont des simulations étalonnées par des courbes mesurées de débit de service et de puissance par phase sur un NVIDIA L4. La tête de lecture ne révèle que le début de trajectoire exécuté; le programme en pointillés est le plan disponible au temps zéro.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="open_loop",
)))
```

:::{figure} _static/inference_serving/open-loop.svg
:label: fig-inference-open-loop-fallback
:class: pdf-fallback
:alt: Comparaison statique du plan de fréquences fixe hors ligne sous des arrivées de requêtes nominales et décalées.

Comparaison statique des rejeux nominal et à rafale décalée. Le livre en ligne
ajoute la lecture et un sélecteur de correcteur.
:::

Le tableau donne les résultats au niveau des requêtes pour le programme fixe
hors ligne, sous sa prévision nominale et sous la perturbation de rafale
décalée. Les deux colonnes utilisent les mêmes requêtes, les mêmes paramètres
de correcteur et le même étalonnage du profil mesuré.

```{code-cell} python
:tags: [remove-input]

import pandas as pd

open_loop_metrics = pd.read_csv(
    "artifacts/inference_serving/metrics_open_loop.csv"
).set_index("controller")
open_loop_metrics[
    [
        "mean_ttft_s",
        "p95_ttft_s",
        "matched_moved_burst_mean_ttft_s",
        "matched_moved_burst_p95_ttft_s",
        "energy_j",
        "peak_queued_requests_at_minimum_clock",
        "queued_requests_at_30_s",
        "power_violation_w",
        "thermal_violation_c",
    ]
].T.rename_axis("metric").round(3)
```

{download}`Télécharger toutes les mesures en boucle ouverte (CSV) <artifacts/inference_serving/metrics_open_loop.csv>`

HiGHS déclare une solution optimale pour le programme linéaire énoncé, avec un
objectif de 11 205.68 dans les unités pondérées de son modèle. Sur l'ensemble
des 48 requêtes, le délai moyen avant le premier jeton passe de 16.35 secondes
avec les instants d'arrivée nominaux à 23.23 secondes après le décalage. Le
95e centile passe de 28.09 à 31.28 secondes. Pour les 31 requêtes déplacées, la
moyenne passe de 15.22 à 22.73 secondes et le 95e centile de 23.74 à 32.04
secondes. Ces variations sont des hausses de 7.52 et de 8.30 secondes,
respectivement.

À 30 secondes, le rejeu nominal n'a aucune requête en file, tandis que le
rejeu décalé en a 29. La file maximale pendant que l'horloge est à son minimum
passe aussi de zéro à 29 requêtes. L'énergie baisse de 3 719.7 à 3 646.8
joules malgré des délais plus longs. Déplacer la rafale change les requêtes qui
se chevauchent et la durée que le système passe dans chaque phase; une baisse
d'énergie n'implique donc pas une meilleure trajectoire de service.

Toutes les requêtes finissent par se terminer pendant la vidange qui suit
l'horizon. À l'horizon de 60 secondes, 27 requêtes nominales et 22 requêtes
décalées restent inachevées. Les deux simulations atteignent une puissance de
phase modélisée de 64.852 W et dépassent de 0.052 W la limite de puissance
configurée de 64.800 W. Aucune ne consigne de violation thermique ni de
capacité du cache clé-valeur.

Le tableau affiché porte sur la latence, l'énergie, les files et les
violations de contraintes. Le fichier CSV téléchargeable donne aussi l'énergie
par jeton de sortie, le temps par jeton de sortie, le travail inachevé et
l'ensemble complet des diagnostics enregistrés. L'essai nominal teste la
trajectoire optimisée sous ses propres hypothèses. L'essai décalé teste la
sensibilité à une erreur de prévision explicite. Il n'établit pas la
robustesse à des arrivées arbitraires, à une erreur de modèle ou à une
limitation matérielle de la fréquence. Combler cet écart exige qu'une nouvelle
information puisse modifier les commandes futures, ce qui est le rôle de la
rétroaction et de l'optimisation à horizon glissant. L'étalonnage du profil
est une mesure matérielle, tandis que la comparaison des correcteurs est une
simulation du modèle étalonné. Les résultats de latence, d'énergie et de
contraintes ne sont pas des mesures directes obtenues en rejouant cette trace
dans vLLM.

:::{dropdown} Examiner l'optimisation hors ligne des fréquences
```{literalinclude} code/inference_control.py
:language: python
:start-at: def optimize_open_loop
:end-before: def open_loop_clock_controller
:linenos:
```

{download}`Télécharger l'implémentation complète de la commande d'inférence <code/inference_control.py>`
:::



#### Exemple : la descente de gradient avec moment comme PCOTD

Pour relier ce point de vue à une pratique familière, y compris l'optimisation des hyperparamètres, traitons le pas d'apprentissage et le moment (ou leurs calendriers) comme des commandes. Au lieu de les fixer a priori, on peut les optimiser dans le cadre d'une optimisation de trajectoire. L'optimiseur lui-même devient le système dynamique dont on façonne l'exécution pour minimiser la perte finale.

Programme : descente de gradient avec moment sur une perte quadratique. On ajuste $\boldsymbol{\theta}\in\mathbb{R}^p$ aux données $(\mathbf{A},\mathbf{b})$ en minimisant

$$
\ell(\boldsymbol{\theta})=\tfrac{1}{2}\,\lVert\mathbf{A}\boldsymbol{\theta}-\mathbf{b}\rVert_2^2.
$$

Le programme maintient des paramètres $\boldsymbol{\theta}_k$ et un moment $\mathbf{m}_k$. Chaque itération :

1. calcule le gradient $ \mathbf{g}_k=\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_k)=\mathbf{A}^\top(\mathbf{A}\boldsymbol{\theta}_k-\mathbf{b})$
2. met à jour le moment $ \mathbf{m}_{k+1}=\beta_k \, \mathbf{m}_k + \mathbf{g}_k$
3. met à jour les paramètres $ \boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_k - \alpha_k \, \mathbf{m}_{k+1}$

État, commande et transition. On définit l'état $\mathbf{x}_k=\begin{bmatrix}\boldsymbol{\theta}_k\\ \mathbf{m}_k\end{bmatrix}\in\mathbb{R}^{2p}$ et la commande $\mathbf{u}_k=\begin{bmatrix}\alpha_k\\ \beta_k\end{bmatrix}$. Un pas du programme est

$$
\Phi_k(\mathbf{x}_k,\mathbf{u}_k)=
\begin{bmatrix}
\boldsymbol{\theta}_k - \alpha_k\!\left(\beta_k \, \mathbf{m}_k + \mathbf{A}^\top(\mathbf{A}\boldsymbol{\theta}_k-\mathbf{b})\right)\\[2mm]
\beta_k \, \mathbf{m}_k + \mathbf{A}^\top(\mathbf{A}\boldsymbol{\theta}_k-\mathbf{b})
\end{bmatrix}.
$$

Exécuter le programme pendant $T$ itérations donne la trajectoire

$$
\mathbf{x}_{k+1}=\Phi_k(\mathbf{x}_k,\mathbf{u}_k),\quad k=0,\dots,T-1,\qquad
\mathbf{x}_0=\begin{bmatrix}\boldsymbol{\theta}_0\\ \mathbf{m}_0\end{bmatrix}.
$$

L'objectif comme PCOTD. On choisit le coût terminal $c_T(\mathbf{x}_T)=\ell(\boldsymbol{\theta}_T)$ et, facultativement, des coûts d'étape $c_k(\mathbf{x}_k,\mathbf{u}_k)=\rho_\alpha \, \alpha_k^2+\rho_\beta\,(\beta_k- \bar\beta)^2$. Le problème du programme comme commande s'écrit

$$
\min_{\{\alpha_k,\beta_k\}} \; \ell(\boldsymbol{\theta}_T)+\sum_{k=0}^{T-1}\big(\rho_\alpha \, \alpha_k^2+\rho_\beta\,(\beta_k-\bar\beta)^2\big)
\quad\text{s.c.}\quad \mathbf{x}_{k+1}=\Phi_k(\mathbf{x}_k,\mathbf{u}_k).
$$

Rétropropagation = récurrence adjointe à rebours. Comme $\Phi_k$ est différentiable, la différentiation automatique en mode inverse calcule $\nabla_{\mathbf{u}_{0:T-1}} \big(c_T+\sum c_k\big)$ en propageant à rebours un état adjoint $\boldsymbol{\lambda}_k=\partial \mathcal{J}/\partial \mathbf{x}_k$ :

$$
\boldsymbol{\lambda}_T=\nabla_{\mathbf{x}_T} c_T,\qquad
\boldsymbol{\lambda}_k=\nabla_{\mathbf{x}_k} c_k + \left(\nabla_{\mathbf{x}_k}\Phi_k\right)^\top \boldsymbol{\lambda}_{k+1},
$$

où $\nabla_{\mathbf{x}_k}\Phi_k$ désigne la jacobienne de l'application vectorielle $\Phi_k$ par rapport à $\mathbf{x}_k$, de sorte que $\left(\nabla_{\mathbf{x}_k}\Phi_k\right)^\top \boldsymbol{\lambda}_{k+1}$ est un produit vecteur-jacobienne; les gradients par rapport aux commandes sont

$$
\nabla_{\mathbf{u}_k}\mathcal{J}=\nabla_{\mathbf{u}_k} c_k + \left(\nabla_{\mathbf{u}_k}\Phi_k\right)^\top \boldsymbol{\lambda}_{k+1}.
$$

Déroulons un tout petit horizon ($T=3$) pour voir la composition :

$$
\begin{aligned}
\mathbf{x}_1&=\Phi_0(\mathbf{x}_0,\mathbf{u}_0),\\
\mathbf{x}_2&=\Phi_1(\mathbf{x}_1,\mathbf{u}_1),\\
\mathbf{x}_3&=\Phi_2(\mathbf{x}_2,\mathbf{u}_2),\qquad
\mathcal{J}=c_T(\mathbf{x}_3)+\sum_{k=0}^{2} c_k(\mathbf{x}_k,\mathbf{u}_k).
\end{aligned}
$$

Et si le programme comporte un branchement? Supposons qu'on insère un branchement qui « saute les petits gradients »

$$
\boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_k - \alpha_k\,\mathbf{m}_{k+1}\,\mathbf{1}\{ \lVert\mathbf{g}_k\rVert>\tau\},
$$

qui n'est pas différentiable à cause de l'indicatrice. Le point de vue PCOTD s'applique toujours, mais les gradients ne sont pas fiables. Deux voies pratiques : lisser le branchement (par exemple remplacer $\mathbf{1}\{\cdot\}$ par $\sigma((\lVert\mathbf{g}_k\rVert-\tau)/\epsilon)$ pour un petit $\epsilon$) et utiliser la différentiation automatique; ou passer à une méthode sans dérivées sur $\{\alpha_k,\beta_k,\tau\}$ (par exemple SPSA ou CMA-ES) en gardant exacte la dynamique interne.

## Variantes : problèmes de Lagrange et de Mayer

Comment le transfert du coût entre termes courants et termes terminaux
change-t-il la représentation sans changer le problème de commande
sous-jacent?

La forme de Bolza est assez générale pour couvrir la plupart des situations, mais deux cas particuliers courants méritent d'être mentionnés :

* **Problème de Lagrange (sans coût terminal)**
  Si l'objectif ne fait que cumuler des coûts d'étape :

$$
\min_{\mathbf{u}_{1:T-1}} \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t).
$$

Exemple : *minimisation de l'énergie d'un drone de livraison*. Ce qui compte est la consommation totale de la batterie, quelle que soit la position finale.

* **Problème de Mayer (coût terminal seulement)**
  Si l'objectif ne dépend que de l'état final :

$$
\min_{\mathbf{u}_{1:T-1}} c_T(\mathbf{x}_T).
$$

Exemple : *transfert orbital d'un satellite*. Le seul but est d'atteindre une orbite donnée, quel que soit le carburant dépensé en route.

Ces distinctions comptent lorsqu'on dérive les conditions d'optimalité, mais elles s'inscrivent conceptuellement dans le même cadre : le système évolue dans le temps, et on choisit des commandes pour façonner la trajectoire.

### Réduction à la forme de Mayer par augmentation de l'état

Bien que les problèmes de Bolza, de Lagrange et de Mayer paraissent différents, ils ont le même pouvoir d'expression. Tout problème à coûts courants peut se réécrire comme un problème de Mayer (dont l'objectif ne dépend que de l'état final) grâce à une astuce simple : **augmenter l'état d'une somme courante des coûts**.

L'idée est directe. On introduit une nouvelle variable, $y_t$, qui garde la trace du coût cumulé jusque-là. À chaque pas, on met à jour cette somme courante en même temps que l'état du système :

$$
\tilde{\mathbf{x}}_{t+1} =
\begin{pmatrix}
\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) \\
y_t + c_t(\mathbf{x}_t,\mathbf{u}_t)
\end{pmatrix},
$$

où $\tilde{\mathbf{x}}_t = (\mathbf{x}_t, y_t)$. Le coût terminal devient alors :

$$
\tilde{c}_T(\tilde{\mathbf{x}}_T) = c_T(\mathbf{x}_T) + y_T.
$$

Au total, la somme explicite $\sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)$ disparaît de l'objectif et se trouve prise en compte implicitement par l'état augmenté. On peut ainsi écrire tout problème de contrôle optimal sous forme de Mayer.

Cette réduction sert deux buts. D'abord, elle simplifie souvent les dérivations mathématiques, comme on le verra plus loin en dérivant les conditions nécessaires. Ensuite, elle peut simplifier l'implémentation des algorithmes : au lieu d'écrire des chemins de code distincts pour les problèmes de Mayer, de Lagrange et de Bolza, on peut tout ramener à une forme canonique unique. Cela dit, cette approche unifiée n'est pas toujours la meilleure en pratique. Des formulations spécialisées sont parfois plus efficaces en calcul, surtout lorsque le coût courant a une structure simple.


Le fil conducteur est qu'un PCOTD peut ressembler sur papier à un PNL générique, mais que sa structure compte. L'ignorer mène souvent à des solutions impraticables, alors que des formulations qui exposent le caractère creux et respectent le couplage temporel permettent aux solveurs modernes de passer efficacement à l'échelle. Dans les sections suivantes, nous verrons comment ces choix se traduisent en pratique avec le tir simple, le tir multiple et la collocation, et pourquoi différentes formulations font des compromis différents entre robustesse et effort de calcul.

## Synthèse et perspectives

Un horizon fini transforme un modèle dynamique commandé en programme non
linéaire sur les états et les actions. Les exemples de traînée différentielle
et d'inférence délimitent aussi sa garantie : la faisabilité et la performance
valent pour la condition initiale, la dynamique et la prévision de
perturbation énoncées. Les formes de Bolza, de Lagrange et de Mayer changent
la comptabilité sans changer les suites d'actions admissibles.

Sous une qualification de contraintes, les conditions KKT sont nécessaires en
une solution locale du programme non linéaire obtenu, mais leur forme brute
masque le sens du temps. Peut-on organiser les multiplicateurs en une
récurrence à rebours qui corresponde à la dynamique d'état en avant? Le
chapitre [États adjoints et principe de Pontryaguine en temps
discret](discrete-time-pmp.md) fournit cette organisation.

## Exercices

:::{exercise} Conditions KKT d'un PCOTD simple
:label: ex-trajectories-kkt

Considérons un problème de contrôle optimal à deux étapes, avec état et commande scalaires :

$$
\min_{u_1, u_2} \quad x_3^2 + u_1^2 + u_2^2
$$

sous les contraintes :

$$
x_2 = x_1 + u_1, \quad x_3 = x_2 + u_2, \quad x_1 = 1.
$$

**(a)** Écrivez le lagrangien avec les multiplicateurs $\lambda_2, \lambda_3$ des contraintes de dynamique.

**(b)** Dérivez les conditions KKT (stationnarité par rapport à $x_2, x_3, u_1, u_2$).

**(c)** Résolvez le système pour trouver les commandes optimales $u_1^\star, u_2^\star$ et le coût optimal.

:::

:::{solution} ex-trajectories-kkt
:class: dropdown

Le lagrangien est $\mathcal{L} = x_3^2 + u_1^2 + u_2^2 + \lambda_2(x_1 + u_1 - x_2) + \lambda_3(x_2 + u_2 - x_3)$.

Conditions de stationnarité :
- $\partial \mathcal{L}/\partial x_2 = -\lambda_2 + \lambda_3 = 0 \Rightarrow \lambda_2 = \lambda_3$
- $\partial \mathcal{L}/\partial x_3 = 2x_3 - \lambda_3 = 0 \Rightarrow \lambda_3 = 2x_3$
- $\partial \mathcal{L}/\partial u_1 = 2u_1 + \lambda_2 = 0 \Rightarrow u_1 = -\lambda_2/2$
- $\partial \mathcal{L}/\partial u_2 = 2u_2 + \lambda_3 = 0 \Rightarrow u_2 = -\lambda_3/2$

En substituant et en utilisant la dynamique : $x_3 = 1 + u_1 + u_2 = 1 - \lambda_2/2 - \lambda_3/2 = 1 - \lambda_3$. Avec $\lambda_3 = 2x_3$, on obtient $x_3 = 1 - 2x_3$, donc $x_3^\star = 1/3$, $\lambda_3^\star = 2/3$, $u_1^\star = u_2^\star = -1/3$. Le coût optimal vaut $(1/3)^2 + 2(1/3)^2 = 1/3$.
:::

---

:::{exercise} Passage de la forme de Lagrange à la forme de Mayer
:label: ex-trajectories-mayer

Considérons le problème de Lagrange :

$$
\min_{u_{1:T-1}} \sum_{t=1}^{T-1} c_t(x_t, u_t) \quad \text{s.c.} \quad x_{t+1} = f_t(x_t, u_t), \quad x_1 = x_0.
$$

**(a)** Introduisez un état auxiliaire $y_t$ qui cumule le coût courant. Écrivez la dynamique augmentée et le problème de Mayer équivalent.

**(b)** Quelle est la condition initiale de $y_1$? Quel est le coût terminal sous forme de Mayer?

**(c)** Vérifiez que les deux formulations ont la même suite de commandes optimale.

:::

:::{solution} ex-trajectories-mayer
:class: dropdown

On définit l'état augmenté $\tilde{x}_t = (x_t, y_t)$ avec la dynamique :

$$
\tilde{x}_{t+1} = \begin{pmatrix} f_t(x_t, u_t) \\ y_t + c_t(x_t, u_t) \end{pmatrix}.
$$

Condition initiale : $y_1 = 0$. Coût terminal : $\tilde{c}_T(\tilde{x}_T) = y_T$.

L'objectif $y_T = \sum_{t=1}^{T-1} c_t(x_t, u_t)$ est identique à l'objectif de Lagrange; les commandes optimales sont donc les mêmes.
:::

---

:::{exercise} Quand la LICQ n'est pas satisfaite
:label: ex-trajectories-licq

Considérons un PCOTD où l'état doit satisfaire $x_T = 0$ et aussi $x_T \leq 0$ au temps terminal.

**(a)** En un point réalisable où $x_T = 0$, quelles contraintes sont actives?

**(b)** Écrivez les gradients des contraintes actives par rapport à $x_T$. Sont-ils linéairement indépendants?

**(c)** Expliquez pourquoi cela viole la LICQ et quelles conséquences cela peut avoir pour les multiplicateurs KKT.

:::

:::{solution} ex-trajectories-licq
:class: dropdown

Les deux contraintes $h(x_T) = x_T = 0$ et $g(x_T) = x_T \leq 0$ sont actives lorsque $x_T = 0$. Les gradients sont $\nabla h = 1$ et $\nabla g = 1$, qui sont parallèles (linéairement dépendants). La LICQ n'est pas satisfaite parce que les gradients des contraintes n'engendrent pas des directions indépendantes. Conséquence : les multiplicateurs $\lambda$ (pour l'égalité) et $\mu \geq 0$ (pour l'inégalité) peuvent ne pas être uniques; seule leur somme intervient dans la condition de stationnarité, si bien que toute combinaison vérifiant $\lambda + \mu = c$ pour un $c$ fixé pourrait convenir. Cela entraîne des difficultés numériques dans les algorithmes d'optimisation.
:::

---

:::{exercise} Dérivation de la traînée différentielle et audit du modèle
:label: ex-trajectories-differential-drag

Revenons au planificateur à trois satellites. Soit
$\Delta\sigma=1/B_\mathrm{high}-1/B_\mathrm{low}$ et
$a_0=R_\mathrm{E}+h_0$, toutes les grandeurs étant converties en unités SI
cohérentes avant substitution.

**(a)** À partir de

$$
\dot a=-\rho_0\sigma\sqrt{\mu a},
\qquad
n(a)=\sqrt{\frac{\mu}{a^3}},
$$

dérivez la perte d'altitude quotidienne supplémentaire $d$ et l'accélération
angulaire $\alpha$ causées par une forte traînée permanente, après
linéarisation en $a_0$.

**(b)** Déroulez le modèle quotidien et montrez que

$$
\omega_{i,N}=\alpha\sum_{k=0}^{N-1}u_{i,k},
\qquad
\varphi_{i,N}
=\varphi_{i,0}
+\alpha\sum_{k=0}^{N-1}
\left(N-k-\frac12\right)u_{i,k}.
$$

Expliquez pourquoi une exposition totale égale à la forte traînée peut
produire des vitesses terminales égales, alors qu'un calendrier de commandes
différent produit quand même des phases terminales différentes.

**(c)** Le résidu du programme linéaire nominal est inférieur à $10^{-10}$,
mais le rejeu non linéaire manque un écart cyclique de plus de $10^\circ$.
Précisez quelle affirmation chaque nombre appuie. Pourquoi resserrer la
tolérance du programme linéaire ne corrigerait-il pas l'écart non linéaire?

**(d)** Proposez une expérience de validation ou de rétroaction qui distingue
l'erreur d'intégration de l'écart de modèle sans modifier la partie déjà
exécutée de la commande.

:::

:::{solution} ex-trajectories-differential-drag
:class: dropdown

La vitesse supplémentaire de variation du demi-grand axe est

$$
\Delta\dot a
=-\rho_0\Delta\sigma\sqrt{\mu a_0}.
$$

Ainsi $d=-\Delta\dot a\,\Delta t$, avec une conversion des mètres en
kilomètres. Comme $dn/da=-3n/(2a)$,

$$
\dot n
=\frac{3n_0}{2a_0}
\rho_0\Delta\sigma\sqrt{\mu a_0},
$$

et $\alpha=\dot n\,\Delta t^2$ après conversion des radians en degrés. Des
substitutions répétées dans l'équation de la vitesse donnent la première
expression de la partie (b). Chaque entrée apporte la moitié de sa nouvelle
vitesse le jour même et sa vitesse complète chacun des jours suivants, ce qui
donne le poids de phase $N-k-\tfrac12$.

Des sommes égales $\sum_k u_{i,k}$ donnent des variations finales de vitesse
égales et des pertes d'altitude supplémentaires nominales égales. Avancer une
même quantité de forte traînée lui donne un poids de phase plus grand; le
calendrier peut donc séparer les satellites sans laisser de grand écart de
vitesse terminale.

Le faible résidu du programme linéaire certifie que la solution numérique
satisfait le programme linéaire. L'écart cyclique manqué mesure un échec de
prédiction sous le procédé non linéaire déclaré. La tolérance du solveur ne
peut pas supprimer la dépendance omise à la densité et à l'altitude. Diviser
d'abord par deux le pas de RK4 permet d'auditer l'erreur d'intégration. Pour
traiter l'écart de modèle en cours d'exploitation, on observe l'état orbital
courant et on replanifie seulement la commande restante, comme en commande à
horizon glissant.
:::

## Autoévaluation

:::{exercise} Prédire la borne active
:label: ex-trajectories-check-3

Dans un problème de freinage en temps minimal avec décélération bornée, quelle borne de commande vous attendez-vous à voir active avant l'arrêt du véhicule?
:::

:::{solution} ex-trajectories-check-3
:class: dropdown

La borne de freinage maximal devrait être active presque partout : retarder ou réduire le freinage ne peut pas raccourcir le temps d'arrêt lorsque la position terminale et la vitesse nulle sont fixées.
:::

:::{exercise} Décalage de la prévision
:label: ex-trajectories-check-4

Le correcteur d'inférence hors ligne reçoit le travail total exact sur 60
secondes, mais la plus grosse rafale arrive vingt secondes plus tôt que prévu.
Expliquez pourquoi respecter le travail total ne préserve pas la trajectoire
planifiée de la file et de la température.
:::

:::{solution} ex-trajectories-check-4
:class: dropdown

L'état dépend du moment où le travail arrive. Un travail plus précoce augmente
la file et la demande de puissance avant que les fréquences choisies pour ce
travail soient programmées, et l'état thermique reporte cet écart de
calendrier sur les secondes suivantes.
:::
