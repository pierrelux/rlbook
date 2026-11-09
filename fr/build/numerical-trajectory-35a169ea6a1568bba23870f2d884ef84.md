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
# Optimisation numérique de trajectoires

La récurrence de Pontryaguine met au jour la structure temporelle de
l'optimalité du premier ordre, mais un solveur a tout de même besoin d'un
vecteur fini de variables et de contraintes. Les états doivent-ils rester
explicites, être éliminés par simulation, ou n'apparaître qu'aux frontières de
segments choisis?

Chacun de ces choix reformule le problème de contrôle optimal en temps discret
comme un programme non linéaire (PNL) standard. On rassemble toutes les
variables de décision (états, commandes et éventuelles variables auxiliaires)
dans un seul vecteur $\mathbf{z}\in\mathbb{R}^{n_z}$ et on écrit

$$
\begin{aligned}
\min_{\mathbf{z}\in\mathbb{R}^{n_z}} \quad & F(\mathbf{z}) \\
\text{s.c.} \quad & H(\mathbf{z}) = 0, \\
& G(\mathbf{z}) \le 0,
\end{aligned}
$$

avec les applications $F:\mathbb{R}^{n_z}\to\mathbb{R}$, $H:\mathbb{R}^{n_z}\to\mathbb{R}^{r_e}$ et $G:\mathbb{R}^{n_z}\to\mathbb{R}^{r_i}$. En contrôle optimal, $H$ encode habituellement la dynamique et les conditions aux limites, tandis que $G$ représente les contraintes de chemin et les contraintes de bornes.

Ainsi, la majuscule $G$ empile les inégalités $g_i\leq0$ et la majuscule $H$ empile les égalités $h_i=0$, conformément aux conventions de la formulation du contrôle optimal et de l'annexe sur la programmation non linéaire.

Il existe plusieurs façons d'aboutir à ce PNL (et d'en tirer parti) :

* Méthodes simultanées (transcription directe / discrétisation complète) : on conserve tous les états et toutes les commandes comme variables et on impose la dynamique sous forme de contraintes d'égalité. Cette approche est directe et expose la structure creuse, mais le problème peut devenir volumineux à moins d'exploiter des techniques du côté du solveur (p. ex. la condensation).
* Méthodes séquentielles (élimination récursive / tir simple) : on élimine les états par propagation vers l'avant à partir de la condition initiale, si bien que les commandes restent les principales variables de décision. On réduit ainsi la dimension et le nombre de contraintes, mais la méthode peut être sensible à l'initialisation et aux horizons plus longs.
* Tir multiple : on introduit des variables d'état aux frontières des segments et on impose la continuité entre des segments simulés séparément. Cette méthode réalise un compromis entre taille et conditionnement et s'avère souvent plus robuste que le tir simple pur.

## Objectifs d'apprentissage

Après avoir étudié les exemples et les exercices, vous devriez pouvoir :

- Formuler des programmes simultanés, de tir simple et de tir multiple.
- Comparer leurs variables, leurs contraintes, leur structure creuse et leurs chemins de sensibilité.
- Dériver un déroulement en mode inverse pour mesurer l'effet de chaque
  commande sur l'état terminal, et relier cette sensibilité au conditionnement
  d'un programme de tir.
- Construire les défauts de continuité entre des segments simulés séparément
  et vérifier qu'une trajectoire calculée respecte sa dynamique et ses
  contraintes.

## Préalables

Les formulations reposent sur le [problème de contrôle optimal à horizon fini](discrete-time-optimal-control.md)
et sur sa [récurrence de Pontryaguine](discrete-time-pmp.md).
L'[annexe sur la programmation non linéaire](appendix_nlp.md) revoit
l'optimisation sous contraintes, les dérivées et les solveurs numériques.

## Méthodes simultanées

Quelle structure numérique apparaît lorsque chaque état et chaque action
restent des variables d'optimisation et que chaque transition devient une
contrainte d'égalité?

Dans l'approche simultanée (aussi appelée transcription directe ou discrétisation complète), on garde toute la trajectoire explicite et on impose la dynamique sous forme de contraintes d'égalité. En partant du PCOTD de Bolza,

$$
\min_{\{\mathbf{x}_t,\mathbf{u}_t\}}\; c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)
\quad\text{s.c.}\quad \mathbf{x}_{t+1} - \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) = 0,\; t=1,\dots,T-1,
$$

on rassemble toutes les variables dans un seul vecteur

$$
\mathbf{z} := \begin{bmatrix}
\mathbf{x}_1^\top & \cdots & \mathbf{x}_T^\top & \mathbf{u}_1^\top & \cdots & \mathbf{u}_{T-1}^\top
\end{bmatrix}^\top \in \mathbb{R}^{n_z}.
$$

Le résidu de la dynamique s'écrit ici $\mathbf{x}_{t+1}-\mathbf f_t$. Son multiplicateur est l'opposé de l'état adjoint de Pontryaguine, qui multiplie $\mathbf f_t-\mathbf{x}_{t+1}$; inverser le signe d'un résidu d'égalité change le signe de son multiplicateur, mais laisse inchangées les trajectoires admissibles.

Les contraintes de chemin ne s'appliquent habituellement qu'à certains instants. Soit $\mathscr{E}$ l'ensemble d'indices des contraintes d'égalité supplémentaires $h_i$ et $\mathscr{I}$ celui des contraintes d'inégalité $g_i$. Pour chaque contrainte $i$, on définit l'ensemble des indices temporels $K_i \subseteq \{1,\dots,T\}$ où elle est imposée (p. ex. les contraintes terminales utilisent $K_i = \{T\}$). La transcription simultanée est le PNL

$$
\begin{aligned}
\min_{\mathbf{z}}\quad & F(\mathbf{z}) := c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{s.c.}\quad & H(\mathbf{z}) = \begin{bmatrix}
\big[\, h_i(\mathbf{x}_k,\mathbf{u}_k) \big]_{i\in\mathscr{E},\, k\in K_i} \\
\big[\, \mathbf{x}_{t+1} - \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) \big]_{t=1: T-1} \\
\mathbf{x}_1 - \mathbf{x}_\mathrm{init}
\end{bmatrix} = \mathbf{0}, \\
& G(\mathbf{z}) = \big[\, g_i(\mathbf{x}_k,\mathbf{u}_k) \big]_{i\in\mathscr{I},\, k\in K_i} \; \le \; \mathbf{0},
\end{aligned}
$$

éventuellement avec des bornes simples $\mathbf{x}_{\mathrm{lb}} \le \mathbf{x}_t \le \mathbf{x}_{\mathrm{ub}}$ et $\mathbf{u}_{\mathrm{lb}} \le \mathbf{u}_t \le \mathbf{u}_{\mathrm{ub}}$ intégrées à $G$ ou fournies séparément au solveur. Pour alléger la notation, certaines contraintes peuvent ne pas dépendre de $\mathbf{u}_k$ aux instants de $K_i$; l'indexation aide tout de même à préciser quand chaque condition est active.

Cette transcription directe est intéressante parce qu'elle reste fidèle au modèle et expose la structure creuse. La jacobienne de $H$ a une structure bidiagonale par blocs induite par la dynamique, et la matrice KKT est creuse et structurée. Les méthodes de points intérieurs et de PQS exploitent ces propriétés. Le prix à payer est la taille : avec un état de dimension $n$ et une commande de dimension $m$, le vecteur de décision compte $(T\!\cdot\!n) + ((T\!-
1)\cdot m)$ composantes, et il y a environ $(T\!-
1)\cdot n$ égalités dynamiques, auxquelles s'ajoutent les conditions de chemin et aux limites. Des techniques comme la condensation partielle ou complète éliminent des variables d'état pour réduire l'ensemble des égalités (au prix de matrices plus denses), tandis que garder les états explicites préserve la structure creuse et améliore souvent la robustesse sur de longs horizons et en présence de contraintes d'état.

Par rapport aux autres approches, les méthodes simultanées évitent les longues chaînes de dépendances non linéaires du tir simple et facilitent l'imposition des contraintes d'état et de chemin. Elles peuvent toutefois exiger davantage de mémoire et d'algèbre linéaire à chaque itération; leur performance pratique dépend donc de l'exploitation de la structure creuse et d'une bonne initialisation.

La même logique s'applique au choix d'un optimiseur. Pour les problèmes de petite taille, on s'appuie couramment sur des routines générales comme celles de `scipy.optimize.minimize`. Les méthodes sans dérivées comme celle de Nelder–Mead n'ont besoin d'aucun gradient, mais s'adaptent mal à l'augmentation de la dimension. Les méthodes quasi-newtoniennes comme BFGS fonctionnent bien en dimension modérée et peuvent approcher les gradients par différences finies, tandis que l'optimisation de trajectoires à grande échelle fait souvent appel à des solveurs sous contraintes fondés sur le gradient, comme les méthodes de points intérieurs ou de programmation quadratique successive, qui peuvent exploiter des jacobiennes creuses et profiter de la différentiation automatique. Des techniques stochastiques, dont les algorithmes génétiques, le recuit simulé ou l'optimisation par essaims particulaires, apparaissent à l'occasion quand les gradients ne sont pas disponibles, mais leur coût croît rapidement avec la dimension et elles sont rarement compétitives pour les problèmes structurés de contrôle optimal.

### Exemple : redressement non linéaire du chariot-pendule

Un chariot porte un pendule rigide dont l'angle est mesuré à partir de la verticale vers le haut. Le chariot peut accélérer horizontalement, mais aucun actionneur n'applique directement de couple à l'articulation du pendule. À partir de la configuration stable vers le bas, la tâche consiste à déplacer la base de sorte que le pendule arrive en position verticale haute pendant que le chariot revient près du centre d'un rail de longueur finie.

Soit l'état $\mathbf{x}=(p,v,\theta,\omega)$, où $p$ et $v$ sont la position et la vitesse du chariot, et $\theta$ et $\omega$ sont l'angle et la vitesse angulaire du pendule. Une accélération horizontale commandée $u$ produit la dynamique non linéaire

$$
\dot p = v, \qquad
\dot v = u, \qquad
\dot\theta = \omega, \qquad
\dot\omega = \frac{g}{\ell}\sin\theta
                 - \frac{u}{\ell}\cos\theta
                 - b\omega.
$$

Le facteur $-u\cos\theta/\ell$ montre comment l'accélération commandée $u$ du chariot agit sur le pendule. Le mouvement horizontal de la base se couple à l'accélération angulaire, et son signe comme son amplitude dépendent de la configuration courante. Un optimiseur boîte noire pourrait évaluer ces équations sans examiner ce terme, mais celui-ci explique pourquoi le chariot doit d'abord s'éloigner de sa position de repos finale pour accumuler de l'énergie dans le pendule.

L'expérience numérique utilise un horizon de $4.5$ s avec $N=30$ commandes maintenues par un bloqueur d'ordre zéro et un pas $h=0.15$ s. Une intégration de Runge--Kutta d'ordre quatre définit l'application discrète $\mathbf{x}_{k+1}=F_h(\mathbf{x}_k,u_k)$. Les deux formulations numériques résolvent le même problème :

$$
\begin{aligned}
\min_{\mathbf{x}_{0:N},u_{0:N-1}}\quad
& h\sum_{k=0}^{N-1}\Bigl[
0.05p_k^2+0.01v_k^2+0.25(1-\cos\theta_k)
+0.01\omega_k^2+0.004u_k^2\Bigr] \\
& {}+20p_N^2+5v_N^2+120(1-\cos\theta_N)+12\omega_N^2 \\
\text{sous les contraintes}\quad
& \mathbf{x}_{k+1}=F_h(\mathbf{x}_k,u_k), \\
& \mathbf{x}_0=(0,0,\pi,0), \\
& |p_k|\leq 2.4,\quad |v_k|\leq 4,\quad
  |\omega_k|\leq 12,\quad |u_k|\leq 8.
\end{aligned}
$$

La pénalité périodique $1-\cos\theta$ attribue le même coût terminal à des angles qui diffèrent d'un tour complet. Les pénalités sur la position et la vitesse exigent tout de même que le chariot termine près du repos; faire passer le pendule par le haut ne suffit donc pas à lui seul.

La transcription directe conserve les $31$ états et les $30$ commandes. Elle optimise donc sur $154$ variables scalaires et impose $124$ égalités scalaires, dont la condition initiale et une équation de dynamique à quatre dimensions par pas. La jacobienne des égalités est en bande par blocs, car le résidu au pas $k$ ne dépend que de $(\mathbf{x}_k,u_k,\mathbf{x}_{k+1})$.

La petite démonstration ci-dessous transmet cette jacobienne à SLSQP sous forme de tableau dense. Un solveur direct à grande échelle stockerait et factoriserait plutôt ce même motif en bande par blocs sous forme creuse. La formulation expose la structure creuse, mais l'exploiter relève d'un choix d'implémentation distinct.

```{admonition} Prédiction avant le calcul
:class: tip

Le tir simple réduira le vecteur de décision de $154$ variables à $30$. Avant d'examiner les exécutions appariées ci-dessous, prédisez si le programme plus petit est nécessairement plus facile à optimiser. Repérez les contraintes qui deviennent moins explicites une fois les états éliminés.
```


## Méthodes séquentielles

L'élimination des états peut-elle réduire le programme non linéaire sans que
la longue chaîne de simulation qui en résulte devienne trop sensible aux
premières actions?

La section précédente a montré comment résoudre un problème de contrôle optimal en temps discret en traitant tous les états et toutes les commandes comme variables de décision et en imposant la dynamique sous forme de contraintes d'égalité. On obtient ainsi un programme non linéaire que l'on peut confier à des solveurs comme `scipy.optimize.minimize` avec la méthode SLSQP. Pour de courts horizons, cette approche est simple et fonctionne bien; le code reste proche de la formulation mathématique.

Elle présente aussi un avantage réel : en gardant les états explicites et en imposant la dynamique par des contraintes, on ancre la trajectoire en plusieurs points. Cette structure supplémentaire aide à stabiliser l'optimisation, surtout sur de longs horizons, où de petits écarts aux premiers pas pourraient autrement se propager et faire dériver ou diverger l'optimiseur. En ce sens, cette formulation est mieux conditionnée et plus robuste que les approches qui traitent la dynamique de manière implicite.

L'inconvénient est l'échelle. À mesure que l'horizon s'allonge, le nombre de variables et de contraintes croît avec lui, et toutes sont couplées par la dynamique. Chaque itération d'une méthode de programmation quadratique successive (PQS) ou de points intérieurs exige de construire et de factoriser de grandes jacobiennes et hessiennes. Ces méthodes ont été intégrées à des chaînes de traitement d'apprentissage par renforcement et de programmation différentiable, au moyen de couches implicites ou de solveurs convexes différentiables, mais leur coût est important. Elles restent séquentielles, reposent sur des factorisations répétées d'algèbre linéaire et sont difficiles à paralléliser efficacement. Lorsqu'il faut résoudre des milliers de ces problèmes à l'intérieur d'une boucle d'apprentissage, le surcoût devient prohibitif.

Cela motive une autre approche, mieux accordée au modèle de calcul de l'apprentissage automatique. Pour une dynamique déterministe, on peut éliminer les contraintes d'égalité en rendant les états implicites. Au lieu de résoudre à la fois pour les états et les commandes, on fixe l'état initial et on fait évoluer le système vers l'avant sous une suite de commandes candidate. Les contraintes d'état peuvent subsister, mais elles deviennent des fonctions non linéaires de toute la suite de commandes qui précède. C'est l'essence du **tir simple**.

Le terme « tir » vient de l'idée de *viser et tirer* une trajectoire à partir de l'état initial : on choisit une suite de commandes, on intègre le système (ou on le fait avancer pas à pas) et on regarde où il aboutit. Si l'état final manque la cible, on ajuste les commandes et on recommence, comme on ajuste l'angle d'un tir jusqu'à atteindre le but. On parle de tir **simple** parce qu'on calcule toute la trajectoire en une seule passe à partir du point de départ, sans la découper en segments. Plus loin, on le comparera au **tir multiple**, où l'horizon est divisé en arcs plus courts optimisés conjointement pour améliorer la stabilité et le conditionnement.

L'analogie avec l'apprentissage profond est elle aussi immédiate : la suite de commandes joue le rôle des paramètres, le déroulement est une passe avant et le coût est une perte scalaire. On peut obtenir les gradients par différentiation automatique en mode inverse. Dans la formulation du PCOTD par tir simple, le programme sous contraintes

$$
\min_{\mathbf{x}_{1:T},\,\mathbf{u}_{1:T-1}} J(\mathbf{x}_{1:T},\mathbf{u}_{1:T-1})
\quad\text{s.c.}\quad
\mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)
$$

se réduit à

$$
\min_{\mathbf{u}_{1:T-1}}\;
c_T\!\bigl(\boldsymbol{\phi}_{T}(\mathbf{u}, \mathbf{x}_1)\bigr)
+\sum_{t=1}^{T-1} c_t\!\bigl(\boldsymbol{\phi}_{t}(\mathbf{u}, \mathbf{x}_1), \mathbf{u}_t\bigr),
\quad\text{s.c.}\quad
\mathbf{g}_t\!\bigl(\boldsymbol{\phi}_{t}(\mathbf{u},\mathbf{x}_1),\mathbf{u}_t\bigr)\leq 0,
\quad
\mathbf{u}_{\mathrm{lb}}\le\mathbf{u}_{t}\le\mathbf{u}_{\mathrm{ub}}.
$$

Ici, $\boldsymbol{\phi}_t$ désigne l'état atteint au temps $t$ en appliquant récursivement la dynamique à l'état précédent et à la commande courante. Cette récurrence s'écrit

$$
\boldsymbol{\phi}_{t+1}(\mathbf{u},\mathbf{x}_1)=
\mathbf{f}_{t}\!\bigl(\boldsymbol{\phi}_{t}(\mathbf{u},\mathbf{x}_1),\mathbf{u}_t\bigr),\qquad
\boldsymbol{\phi}_{1}=\mathbf{x}_1.
$$

Concrètement, voici un pseudocode dans le style de JAX qui définit `phi(u, x_0, t)` au moyen de `jax.lax.scan`, avec un indice temporel commençant à zéro :

```python
def phi(u_seq, x0, t):
    """Return \phi_t(u, x0) with 0-based t (\phi_0 = x0).

    u_seq: controls of length T (or T-1); only first t entries are used
    x0: initial state at time 0
    t: integer >= 0
    """
    if t <= 0:
        return x0

    def step(carry, u):
        x, t_idx = carry
        x_next = f(x, u, t_idx)
        return (x_next, t_idx + 1), None

    (x_t, _), _ = lax.scan(step, (x0, 0), u_seq[:t])
    return x_t
```


Ce schéma reproduit le déroulement d'un réseau de neurones récurrent : à partir d'un état initial ($\mathbf{x}^\star_1$) et d'une suite de commandes ($\mathbf{u}^*_{1:T-1}$), on propage vers l'avant à travers la dynamique, en mettant l'état à jour à chaque pas et en accumulant le coût en chemin. Cette similitude structurelle explique pourquoi le tir simple paraît souvent naturel aux praticiens issus de l'apprentissage profond : le déroulement est une passe avant, et les gradients se propagent à rebours dans le temps exactement comme dans la rétropropagation à travers un réseau récurrent.

Sous forme d'algorithme :

```{prf:algorithm} Tir simple : déroulement vers l'avant
:label: single-shooting-forward-unroll

**Entrées** : état initial $\mathbf{x}_1$, horizon $T$, bornes sur les commandes $\mathbf{u}_{\mathrm{lb}}, \mathbf{u}_{\mathrm{ub}}$, dynamique $\mathbf{f}_t$, coûts $c_t$

**Sortie** : suite de commandes optimale $\mathbf{u}^*_{1:T-1}$

1. Initialiser $\mathbf{u}_{1:T-1}$ à l'intérieur des bornes
2. Définir `ComputeTrajectoryAndCost`($\mathbf{u}, \mathbf{x}_1$) :
    - $\mathbf{x} \leftarrow \mathbf{x}_1$, $J \leftarrow 0$
    - Pour $t = 1$ à $T-1$ :
        - $J \leftarrow J + c_t(\mathbf{x}, \mathbf{u}_t)$
        - $\mathbf{x} \leftarrow \mathbf{f}_t(\mathbf{x}, \mathbf{u}_t)$
    - $J \leftarrow J + c_T(\mathbf{x})$
    - Retourner $J$
3. Résoudre $\min_{\mathbf{u}} J(\mathbf{u})$ sous les bornes sur les commandes et les éventuelles contraintes d'état évaluées le long du déroulement
4. Retourner $\mathbf{u}^*_{1:T-1}$
```

Avec JAX ou PyTorch, cette boucle peut être compilée et différentiée automatiquement. La suite de commandes joue le rôle de paramètres entraînables, tandis que la trajectoire simulée constitue le calcul vers l'avant. La différentiation de ce calcul en mode inverse donne $\nabla J(\mathbf{u})$.

Le tir simple séduit par sa simplicité et sa compatibilité avec la programmation différentiable, mais il a ses limites. Les premières commandes influencent chaque état ultérieur par l'intermédiaire d'un long produit de jacobiennes de la dynamique, ce qui peut rendre les gradients mal conditionnés sur de longs horizons. Les contraintes d'état perdent aussi leur représentation locale et creuse, puisque chaque état contraint dépend de toutes les commandes antérieures. Les formulations qui gardent certains états explicites, comme le tir multiple ou la collocation, raccourcissent ces chaînes de dépendances.

### Comparaison appariée du redressement

Les implémentations par transcription directe et par tir simple ci-dessous utilisent le problème du chariot-pendule énoncé plus haut sans modifier le modèle, le coût, les limites, l'horizon, la commande initiale ni le solveur de programmation non linéaire. Seules les variables de décision et la représentation de la dynamique diffèrent.

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))

from cartpole_control import (
    SwingUpScenario,
    format_swingup_metrics,
    make_open_loop_perturbation_figure,
    make_swingup_animation,
    make_swingup_figure,
    replay_open_loop_with_disturbance,
    solve_swingup_comparison,
)

swingup_scenario = SwingUpScenario()
swingup_results = solve_swingup_comparison(swingup_scenario)
```

```{code-cell} python
:label: fig-cartpole-formulations
:caption: La transcription directe et le tir simple résolvent le même problème non linéaire du chariot-pendule à partir de la même initialisation. Les deux atteignent la configuration verticale haute et respectent les mêmes limites, mais convergent vers des solutions locales différentes. La transcription directe conserve 154 variables scalaires et 124 égalités locales de dynamique; le tir simple ne garde que les 30 commandes et reconstruit chaque état par simulation vers l'avant.
:tags: [remove-input]

print(format_swingup_metrics(swingup_results))
make_swingup_figure(swingup_results, swingup_scenario)
```

Les deux solveurs produisent un redressement en boucle ouverte réussi. Dans cette exécution précise, la formulation directe atteint un objectif plus bas, tandis que le tir simple utilise un vecteur de décision beaucoup plus petit. Ce résultat numérique n'établit pas que la transcription directe trouve toujours de meilleures solutions. Il expose un compromis concret : éliminer des variables raccourcit le programme, mais allonge la dépendance entre une commande précoce et le coût terminal ainsi que les contraintes ultérieures.

```{code-cell} python
:label: anim-cartpole-formulations
:caption: Les deux trajectoires sont générées par le même procédé non linéaire intégré par RK4. Le pendule part vers le bas et atteint la configuration verticale haute pendant que le chariot reste à l'intérieur des limites du rail de 2.4 m. Les images de l'animation sont calculées à partir des trajectoires d'état optimisées; aucun simulateur n'est exécuté dans le navigateur.
:tags: [remove-input]

from IPython.display import HTML, display
import matplotlib.pyplot as plt

swingup_animation = make_swingup_animation(swingup_results, swingup_scenario)
display(HTML(swingup_animation.to_jshtml()))
plt.close(swingup_animation._fig)
```

La comparaison sépare aussi l'optimisation de la rétroaction. Chaque optimiseur renvoie une seule suite de commandes fixe pour un seul état initial supposé. Pour vérifier ce que cet objet peut et ne peut pas faire, le rejeu suivant applique deux fois les commandes de la transcription directe. Une réalisation suit le modèle nominal. L'autre reçoit une accélération supplémentaire du chariot de $1\;\mathrm{m\,s^{-2}}$ pendant un pas de $0.15$ s à $t=2.1$ s, après quoi les deux réalisations reçoivent les mêmes commandes restantes.

```{code-cell} python
:label: fig-cartpole-open-loop-perturbation
:caption: Une accélération non modélisée d'un seul pas sépare deux réalisations pilotées par les mêmes commandes en boucle ouverte. La trajectoire nominale atteint la hauteur normalisée du pendule $\cos\theta=1$; la trajectoire perturbée termine sous l'horizontale. L'optimiseur a produit un plan, et non une règle qui réagit à l'état observé.
:tags: [remove-input]

open_loop_replay = replay_open_loop_with_disturbance(
    swingup_results["direct"],
    swingup_scenario,
)
make_open_loop_perturbation_figure(open_loop_replay)
```

La rétroaction change la nature de l'objet calculé. Un correcteur à rétroaction associe à l'état observé après la perturbation une nouvelle action. La commande prédictive obtiendra une telle application en résolvant de manière répétée des problèmes de trajectoire, tandis que la programmation dynamique construira des décisions conditionnelles à l'état au moyen de la fonction de valeur.

:::{dropdown} Examiner la dynamique non linéaire commune
```{literalinclude} code/cartpole_control.py
:language: python
:start-at: def cartpole_dynamics
:end-before: def rk4_step
:linenos:
```
:::

{download}`Télécharger le code source complet d'optimisation de trajectoires et de commande du chariot-pendule <code/cartpole_control.py>`.



(sec-truck-alley-dock)=
### Exemple : mise à quai d'un semi-remorque en marche arrière

La comparaison du chariot-pendule gardait le solveur fixe et changeait les
variables de décision. Cet exemple s'en tient au tir simple et examine ce que
son unique long déroulement fait aux dérivées que reçoit l'optimiseur. Un
tracteur semi-remorque qui recule vers un quai de chargement constitue un bon
test, car cet attelage présente une instabilité bien connue : en marche
arrière, une petite erreur de direction s'amplifie jusqu'à la mise en
portefeuille si elle n'est pas corrigée. Cette amplification correspond au
long produit de jacobiennes de la dynamique mentionné plus haut, et on peut la
mesurer sur la solution.

**Le modèle.** L'état comprend la position de l'essieu arrière du tracteur
$(p_x, p_y)$, le cap du tracteur $\theta_0$ et le cap de la remorque
$\theta_1$. La sellette d'attelage se trouve au-dessus de l'essieu arrière du
tracteur, et l'essieu de la remorque la suit à $L_1 = 10\ \mathrm m$ le long
de l'axe de la remorque. Les commandes sont la vitesse du tracteur $v$ et
l'angle de braquage $\delta$, qui agissent au moyen du modèle cinématique

$$
\dot p_x = v\cos\theta_0,\qquad
\dot p_y = v\sin\theta_0,\qquad
\dot\theta_0 = \frac{v}{L_0}\tan\delta,\qquad
\dot\theta_1 = \frac{v}{L_1}\sin(\theta_0-\theta_1),
$$

avec un empattement du tracteur $L_0 = 4.5\ \mathrm m$. L'angle d'attelage
$\beta = \theta_0 - \theta_1$ est nul lorsque l'attelage est droit. Les
commandes sont maintenues constantes sur des intervalles de
$0.25\ \mathrm s$, chacun intégré par un pas de RK4, et $T = 160$ intervalles
couvrent une manœuvre de durée fixe de $40\ \mathrm s$. La vitesse est bornée
par $\pm 1.5\ \mathrm{m/s}$ et le braquage par $\pm 30^\circ$.

**La manœuvre.** La face du quai est la droite $p_y = 0$ et la baie est
centrée en $p_x = 0$. L'attelage part d'une voie parallèle au quai, à
$19\ \mathrm m$ de celui-ci, orienté selon $+x$ avec la remorque derrière. Il
doit terminer droit et orienté à l'opposé du quai, avec le pare-chocs arrière
de la remorque à $0.5\ \mathrm m$ de la face. Un second scénario échange les
extrémités : l'attelage part à quai et doit sortir jusqu'à la position dans
la voie.

**Le programme.** Le tir simple ne garde comme variables de décision que les
$2T = 320$ valeurs de commande. L'objectif est le coût du déroulement qu'elles
génèrent,

$$
J(\mathbf u) = \sum_{t=0}^{T-1} c\bigl(\boldsymbol\phi_t(\mathbf u,\mathbf x_0),\mathbf u_t\bigr) + r(\mathbf u) + c_T\bigl(\boldsymbol\phi_T(\mathbf u,\mathbf x_0)\bigr).
$$

Le coût d'étape pénalise la vitesse, le braquage et la distance entre l'essieu
de la remorque et sa cible, et ajoute deux pénalités lisses évaluées sur
l'état déroulé : l'une pour tout coin de la carrosserie situé à moins de
$0.2\ \mathrm m$ de la face du quai, l'autre pour les angles d'attelage
supérieurs à $60^\circ$. Ce sont des contraintes d'état du type décrit plus
haut, et dans cette formulation chacune devient une fonction de toutes les
commandes antérieures. Le terme $r(\mathbf u)$ ne dépend que des commandes. Il
pénalise les variations entre intervalles consécutifs et un dernier intervalle
en mouvement. Le coût terminal pénalise l'erreur de position de l'essieu de la
remorque, l'erreur de cap de la remorque et l'angle d'attelage. Les bornes sur
les commandes restent explicites et sont traitées par le solveur.

L'implémentation suit {prf:ref}`single-shooting-forward-unroll`. Un
déroulement par `jax.lax.scan` produit les états, le coût est accumulé le long
de celui-ci, et `jax.value_and_grad` renvoie $J(\mathbf u)$ et son gradient en
mode inverse. L-BFGS-B de SciPy reçoit les deux, ainsi que la boîte associée à
chaque commande, et s'arrête lorsque le gradient projeté passe sous $10^{-5}$
ou que la diminution relative de $J$ passe sous $10^{-10}$. L'amorce est un
avancement lent en ligne droite à $0.5\ \mathrm{m/s}$ dans la direction
pertinente : en marche arrière pour le scénario de mise à quai et en marche
avant pour la sortie.

:::{figure} _static/truck_parking/iterations.svg
:label: fig-truck-iterations
:width: 100%

Quatre itérés de L-BFGS-B pour la résolution en marche arrière. Chaque panneau
est un déroulement complet de la suite de commandes courante, avec le chemin
de l'essieu de la remorque en bleu et le contour de l'attelage toutes les dix
secondes; le contour pointillé est la cible. L'amorce recule en ligne droite
le long de la voie. À l'itération 10, la remorque pivote vers la baie, et le
plan final la met à quai.
:::

:::{iframe} ../interactive/truck-parking.html
:label: fig-truck-paths
:width: 100%
:title: Mise à quai d'un semi-remorque par tir simple
:class: truck-parking-replay
:placeholder: _static/truck_parking/paths.svg

Plans finaux pour les deux scénarios. La courbe bleue est l'essieu de la
remorque, la courbe orange tiretée l'essieu du tracteur, et la droite grise
pointillée l'amorce en ligne droite. Les contours montrent l'attelage toutes
les cinq secondes. Dans la version en ligne du livre, cette figure est un
rejeu : choisissez un scénario et un point de contrôle de l'optimiseur, puis
faites défiler ou lancez le temps de simulation à l'intérieur de ce plan fixe.
Le chemin à venir et les contours translucides montrent les états restants du
plan; les tuiles et les graphiques correspondent au point de contrôle choisi,
y compris les premiers, qui n'atteignent pas le quai.
:::

:::{include} artifacts/truck_parking/results.md
:::

Les deux résolutions s'arrêtent sur une faible diminution relative de
l'objectif et réussissent les vérifications d'arrivée, qui rejouent les
commandes finales avec quatre sous-pas de RK4 par intervalle. La marche
arrière a demandé 821 itérations et 1750 évaluations de l'objectif, la sortie
510 et 1044. Selon le plan de mise à quai, l'attelage recule à la vitesse
maximale pendant les 17 premières secondes. Son braquage commence à
$+21^\circ$, ce qui plie l'attelage, passe par zéro et atteint $-29^\circ$ à
17 secondes pendant que la remorque pivote vers la baie; la vitesse et le
braquage se relâchent ensuite ensemble à mesure que l'attelage se redresse.
L'angle d'attelage culmine près de $45^\circ$, sous la limite de $60^\circ$
({numref}`fig-truck-plan`). Dans le plan de sortie, l'attelage dépasse
légèrement la position dans la voie et recule pour se redresser, ce qui
explique que sa distance soit plus longue.

:::{figure} _static/truck_parking/plan.svg
:label: fig-truck-plan
:width: 100%

À gauche : le plan final de marche arrière, c'est-à-dire les 320 nombres que
renvoie l'optimiseur. La vitesse sature à la limite de marche arrière et le
braquage s'approche de sa limite de $30^\circ$ avant que les deux ne se
relâchent à mesure que l'attelage se redresse. À droite : l'angle d'attelage
le long des deux plans finaux; les lignes pointillées marquent le seuil de
pénalité de $60^\circ$.
:::

**Mesurer la chaîne de sensibilité.** En soustrayant les deux équations de
cap, on obtient une équation fermée pour l'angle d'attelage,

$$
\dot\beta = \frac{v}{L_0}\tan\delta - \frac{v}{L_1}\sin\beta .
$$

Le long d'un plan donné, une petite perturbation d'attelage $\Delta\beta$
évolue selon $\Delta\dot\beta = -(v\cos\beta/L_1)\,\Delta\beta$. En marche
avant, $v>0$, le coefficient est négatif et la perturbation s'atténue. En
marche arrière, $v<0$, il est positif et la perturbation croît : c'est
l'instabilité de mise en portefeuille. Sur un intervalle de commande de
longueur $\Delta t$, la perturbation est multipliée par
$\exp(-v\cos\beta\,\Delta t/L_1)$, et l'effet d'une modification du braquage
à l'intervalle $t$ sur l'angle d'attelage terminal est le produit de ces
facteurs sur tous les intervalles suivants. Pour un attelage droit, ce produit
vaut $\exp(d_t/L_1)$, où $d_t$ est la distance qu'il reste à parcourir en
marche arrière après l'intervalle $t$ : chaque tranche de dix mètres de marche
arrière multiplie par $e$ une erreur d'attelage non corrigée.

C'est ce produit que calcule la différentiation en mode inverse à travers le
déroulement. La {numref}`fig-truck-conditioning` compare les deux. Les points
sont $\partial\beta_T/\partial\delta_t$ obtenus avec `jax.jacrev`, divisés par
la variation de cap du tracteur que ce même braquage provoque à l'intérieur de
l'intervalle $t$, de sorte qu'une valeur de un signifie que le reste du
déroulement n'amplifie ni n'atténue l'erreur. Les courbes sont les produits
des facteurs par intervalle le long du même plan. Les deux concordent à un
demi pour cent près. Pour le plan de marche arrière, une erreur de braquage au
début de la manœuvre est amplifiée quatorze fois lorsque la remorque atteint
le quai. Pour le même chemin parcouru en marche avant, obtenu en inversant la
suite de commandes et le signe de la vitesse, une erreur de braquage précoce
est atténuée à $0.07$ de son effet immédiat, car la remorque suit le tracteur
et oublie.

:::{figure} _static/truck_parking/conditioning.svg
:label: fig-truck-conditioning
:width: 100%

À gauche : amplification d'une perturbation du braquage dans l'angle
d'attelage terminal, par intervalle de commande, pour le plan final de marche
arrière et pour le même chemin parcouru en marche avant. Les points
proviennent de la différentiation du déroulement en mode inverse; les courbes
sont des produits des jacobiennes d'attelage linéarisées par intervalle. Les
intervalles où la vitesse est inférieure à $0.1\ \mathrm{m/s}$ sont omis, car
un tracteur immobile ne peut pas braquer. À droite : valeur de l'objectif à
chaque itération de L-BFGS-B pour les deux scénarios, qui partent du même coût
d'amorce.
:::

La conséquence pour l'optimiseur apparaît dans le panneau de droite. À
l'amorce, le gradient de $J$ par rapport au premier angle de braquage vaut
environ 800 dans le scénario de marche arrière et environ 200 pour la sortie,
tandis que le gradient par rapport au dernier angle de braquage vaut 0.04 dans
les deux cas. La courbure de $J$ varie de la même façon d'une composante à
l'autre du vecteur de décision, plus fortement pour la marche arrière. La
résolution en marche arrière descend plus lentement, s'arrête plus tard et se
termine avec un gradient projeté environ trois fois plus grand que celui de la
sortie. Deux résolutions locales n'établissent pas de règle générale, mais
elles montrent le mécanisme : l'élimination des états n'a pas retiré la
dynamique du problème; elle l'a déplacée dans les dérivées. Une distance de
marche arrière plus longue multiplie l'amplification par $e$ tous les dix
mètres, et la formulation de la section suivante raccourcit cette chaîne en
gardant certains états comme variables.

:::{dropdown} Examiner l'objectif réduit et le pilote de L-BFGS-B
```{literalinclude} code/truck_parking.py
:language: python
:start-at: def shooting_objective
:end-before: def terminal_sensitivity
:linenos:
```
:::

- {download}`Modèle du camion, coûts, solveur et sensibilité <code/truck_parking.py>`
- {download}`Expérience et construction des figures <scripts/build_truck_parking_artifacts.py>`
- {download}`Diagnostics numériques et empreintes des sources <artifacts/truck_parking/metrics.json>`

Exécutez `uv run python scripts/build_truck_parking_artifacts.py` à partir de
la racine du dépôt pour reproduire les résolutions, les figures, les tableaux
et les données de rejeu. Les constructions habituelles du livre lisent ces
artefacts.


## Entre le séquentiel et le simultané

Des états de frontière choisis peuvent-ils raccourcir ces chemins de
sensibilité tout en conservant une simulation locale à l'intérieur de chaque
segment?

```{admonition} Avant de poursuivre
:class: tip
Réfléchissez à ce qui arrive au tir simple lorsque l'horizon $T$ est très grand. Quelles difficultés numériques pourraient survenir? Pensez à la façon dont de petites erreurs sur les premières commandes pourraient se propager à travers la dynamique.
```

Les deux formulations vues jusqu'ici se situent aux deux extrémités. L'approche par **discrétisation complète** garde chaque état explicite et impose la dynamique par des contraintes d'égalité, ce qui rend la structure claire mais mène à un gros problème d'optimisation. À l'autre extrémité, le **tir simple** supprime ces contraintes en simulant vers l'avant à partir de l'état initial, si bien que seules les commandes restent des variables de décision. Le problème devient plus petit, mais on introduit aussi une dépendance longue et fortement non linéaire entre la première commande et le dernier état.

Le **tir multiple** se situe entre les deux. Au lieu de simuler tout l'horizon d'un seul coup, on le divise en segments plus courts. Pour chaque segment, on garde son état de départ comme variable de décision et on propage vers l'avant au moyen de la dynamique de ce segment. À la fin, on impose la continuité en exigeant que l'état final simulé d'un segment coïncide avec la variable de décision du segment suivant.

Formellement, supposons que l'horizon de $T$ pas soit divisé en $K$ segments de longueur $L$ (avec $T = K \cdot L$ pour simplifier). On introduit :

* Les commandes de chaque pas : $\mathbf{u}_{1:T-1}$.
* L'état au début de chaque segment : $\mathbf{x}_1,\dots,\mathbf{x}_K$.

Étant donné $\mathbf{x}_k$ et les commandes de son segment, on calcule l'état terminal prédit en simulant vers l'avant :

$$
\hat{\mathbf{x}}_{k+1} = \Phi(\mathbf{x}_k,\mathbf{u}_{\text{segment }k}),
$$

où $\Phi$ représente $L$ applications de la dynamique. Les contraintes de continuité imposent :

$$
\mathbf{x}_{k+1} - \hat{\mathbf{x}}_{k+1} = 0, \qquad k=1,\dots,K-1.
$$

Le programme non linéaire qui en résulte prend la forme suivante :

$$
\begin{aligned}
\min_{\{\mathbf{x}_k,\mathbf{u}_t\}} \quad &
c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{sous les contraintes} \quad &
\mathbf{x}_{k+1} - \Phi(\mathbf{x}_k,\mathbf{u}_{\text{segment }k}) = 0,\quad k = 1,\dots,K-1, \\
& \mathbf{u}_{\mathrm{lb}} \le \mathbf{u}_t \le \mathbf{u}_{\mathrm{ub}}, \\
& \text{conditions aux limites sur } \mathbf{x}_1 \text{ et } \mathbf{x}_K.
\end{aligned}
$$

Par rapport au PNL complet, on n'introduit plus chaque état intermédiaire comme variable, seulement les points d'ancrage aux frontières des segments. À l'intérieur de chaque segment, les états sont reconstruits par simulation. Par rapport au tir simple, ces points d'ancrage rompent la longue chaîne de dépendances qui rend l'optimisation instable : les gradients n'ont à traverser que $L$ pas avant d'atteindre une variable de décision, plutôt que l'horizon entier. C'est la même raison pour laquelle des gradients explosifs ou évanescents apparaissent dans les réseaux récurrents profonds : quand la chaîne est trop longue, l'information s'éteint ou explose. Le tir multiple raccourcit la chaîne et améliore le conditionnement.

En ajustant le nombre de segments $K$, on peut interpoler entre les deux extrêmes : $K = 1$ donne le tir simple, tandis que $K = T$ redonne le PNL direct complet. En pratique, un nombre modéré de segments offre souvent un bon équilibre entre robustesse et complexité.


```{code-cell} python
:tags: [hide-input]

#  label: fig-ocp-multiple-shooting
#  caption: Multiple shooting ballistic BVP: the code produces an animation (and optional static plot) that shows how segment defects shrink while steering the projectile to the target.

%config InlineBackend.figure_format = 'retina'
"""
Multiple Shooting as a Boundary-Value Problem (BVP) for a Ballistic Trajectory
-----------------------------------------------------------------------------
We solve for the initial velocities (and total flight time) so that the terminal
position hits a target, enforcing continuity between shooting segments.
"""

import numpy as np
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from IPython.display import HTML, display

# -----------------------------
# Physical parameters
# -----------------------------
g = 9.81          # gravity (m/s^2)
m = 1.0           # mass (kg)
drag_coeff = 0.1  # quadratic drag coefficient


def dynamics(t, state):
    """Ballistic dynamics with quadratic drag. state = [x, y, vx, vy]."""
    x, y, vx, vy = state
    v = np.hypot(vx, vy)
    drag_x = -drag_coeff * v * vx / m if v > 0 else 0.0
    drag_y = -drag_coeff * v * vy / m if v > 0 else 0.0
    dx  = vx
    dy  = vy
    dvx = drag_x
    dvy = drag_y - g
    return np.array([dx, dy, dvx, dvy])


def flow(y0, h):
    """One-segment flow map Φ(y0; h): integrate dynamics over duration h."""
    sol = solve_ivp(dynamics, (0.0, h), y0, method="RK45", rtol=1e-7, atol=1e-9)
    return sol.y[:, -1], sol

# -----------------------------
# Multiple-shooting BVP residuals
# -----------------------------

def residuals(z, K, x_init, x_target):
    """
    Unknowns z = [vx0, vy0, H, y1(4), y2(4), ..., y_{K-1}(4)]  (total len = 3 + 4*(K-1))
    We define y0 from x_init and (vx0, vy0). Each segment has duration h = H/K.
    Residual vector stacks:
      - initial position constraints: y0[:2] - x_init[:2]
      - continuity: y_{k+1} - Φ(y_k; h) for k=0..K-2
      - terminal position constraint at end of last segment: Φ(y_{K-1}; h)[:2] - x_target[:2]
    """
    n = 4
    vx0, vy0, H = z[0], z[1], z[2]
    if H <= 0:
        # Strongly penalize nonpositive durations to keep solver away
        return 1e6 * np.ones(2 + 4*(K-1) + 2)

    h = H / K

    # Build list of segment initial states y_0..y_{K-1}
    ys = []
    y0 = np.array([x_init[0], x_init[1], vx0, vy0], dtype=float)
    ys.append(y0)
    if K > 1:
        rest = z[3:]
        y_internals = rest.reshape(K-1, n)
        ys.extend(list(y_internals))  # y1..y_{K-1}

    res = []

    # Initial position must match exactly
    res.extend(ys[0][:2] - x_init[:2])

    # Continuity across segments
    for k in range(K-1):
        yk = ys[k]
        yk1_pred, _ = flow(yk, h)
        res.extend(ys[k+1] - yk1_pred)

    # Terminal position at the end of last segment equals target
    y_last_end, _ = flow(ys[-1], h)
    res.extend(y_last_end[:2] - x_target[:2])

    # Optional soft "stay above ground" at knots (kept gentle)
    # res.extend(np.minimum(0.0, np.array([y[1] for y in ys])).ravel())

    return np.asarray(res)

# -----------------------------
# Solve BVP via optimization on 0.5*||residuals||^2
# -----------------------------

def solve_bvp_multiple_shooting(K=5, x_init=np.array([0., 0.]), x_target=np.array([10., 0.])):
    """
    K: number of shooting segments.
    x_init: initial position (x0, y0). Initial velocities are unknown.
    x_target: desired terminal position (xT, yT) at time H (unknown).
    """
    # Heuristic initial guesses:
    dx = x_target[0] - x_init[0]
    dy = x_target[1] - x_init[1]
    H0 = max(0.5, dx / 5.0)  # guess ~ 5 m/s horizontal
    vx0_0 = dx / H0
    vy0_0 = (dy + 0.5 * g * H0**2) / H0  # vacuum guess

    # Intentionally disconnected internal knots to visualize defect shrinkage
    internals = []
    for k in range(1, K):  # y1..y_{K-1}
        xk = x_init[0] + (dx * k) / K
        yk = x_init[1] + (dy * k) / K + 2.0  # offset to create mismatch
        internals.append(np.array([xk, yk, 0.0, 0.0]))
    internals = np.array(internals) if K > 1 else np.array([])

    z0 = np.concatenate(([vx0_0, vy0_0, H0], internals.ravel()))

    # Variable bounds: H > 0, keep velocities within a reasonable range
    # Use wide bounds to let the solver work; tune if needed.
    lb = np.full_like(z0, -np.inf, dtype=float)
    ub = np.full_like(z0,  np.inf, dtype=float)
    lb[2] = 1e-2  # H lower bound
    # Optional velocity bounds
    lb[0], ub[0] = -50.0, 50.0
    lb[1], ub[1] = -50.0, 50.0

    # Objective and callback for L-BFGS-B
    def objective(z):
        r = residuals(z, K,
                      np.array([x_init[0], x_init[1], 0., 0.]),
                      np.array([x_target[0], x_target[1], 0., 0.]))
        return 0.5 * np.dot(r, r)

    iterate_history = []
    def cb(z):
        iterate_history.append(z.copy())

    bounds = list(zip(lb.tolist(), ub.tolist()))
    sol = minimize(objective, z0, method='L-BFGS-B', bounds=bounds,
                   callback=cb, options={'maxiter': 300, 'ftol': 1e-12})

    return sol, iterate_history

# -----------------------------
# Reconstruct and plot (optional static figure)
# -----------------------------

def reconstruct_and_plot(sol, K, x_init, x_target):
    n = 4
    vx0, vy0, H = sol.x[0], sol.x[1], sol.x[2]
    h = H / K

    ys = []
    y0 = np.array([x_init[0], x_init[1], vx0, vy0])
    ys.append(y0)
    if K > 1:
        internals = sol.x[3:].reshape(K-1, n)
        ys.extend(list(internals))

    # Integrate each segment and stitch
    traj_x, traj_y = [], []
    for k in range(K):
        yk = ys[k]
        yend, seg = flow(yk, h)
        traj_x.extend(seg.y[0, :].tolist() if k == 0 else seg.y[0, 1:].tolist())
        traj_y.extend(seg.y[1, :].tolist() if k == 0 else seg.y[1, 1:].tolist())

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(traj_x, traj_y, '-', label='Multiple-shooting solution')
    ax.plot([x_init[0]], [x_init[1]], 'go', label='Start')
    ax.plot([x_target[0]], [x_target[1]], 'r*', ms=12, label='Target')
    total_pts = len(traj_x)
    for k in range(1, K):
        idx = int(k * total_pts / K)
        ax.axvline(traj_x[idx], color='k', ls='--', alpha=0.3, lw=1)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Multiple Shooting BVP (K={K})   H={H:.3f}s   v0=({vx0:.2f},{vy0:.2f}) m/s')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()

    # Report residual norms
    res = residuals(sol.x, K, np.array([x_init[0], x_init[1], 0., 0.]), np.array([x_target[0], x_target[1], 0., 0.]))
    print(f"\nFinal residual norm: {np.linalg.norm(res):.3e}")
    print(f"vx0={vx0:.4f} m/s, vy0={vy0:.4f} m/s, H={H:.4f} s")

# -----------------------------
# Create JS animation for notebooks
# -----------------------------

def create_animation_progress(iter_history, K, x_init, x_target):
    """Return a JS animation (to_jshtml) showing defect shrinkage across segments."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Apply book style
    try:
        import scienceplots
        plt.style.use(['science', 'notebook'])
    except (ImportError, OSError):
        pass  # Use matplotlib defaults

    n = 4

    def unpack(z):
        vx0, vy0, H = z[0], z[1], z[2]
        ys = [np.array([x_init[0], x_init[1], vx0, vy0])]
        if K > 1 and len(z) > 3:
            internals = z[3:].reshape(K-1, n)
            ys.extend(list(internals))
        return H, ys

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_xlabel('Segment index (normalized time)')
    ax.set_ylabel('y (m)')
    ax.set_title('Multiple Shooting: Defect Shrinkage (Fixed Boundaries)')
    ax.grid(True, alpha=0.3)

    # Start/target markers at fixed indices
    ax.plot([0], [x_init[1]], 'go', label='Start')
    ax.plot([K], [x_target[1]], 'r*', ms=12, label='Target')
    # Vertical dashed lines at boundaries
    for k in range(1, K):
        ax.axvline(k, color='k', ls='--', alpha=0.35, lw=1)
    ax.legend(loc='best')

    # Pre-create line artists
    colors = plt.cm.plasma(np.linspace(0, 1, K))
    segment_lines = [ax.plot([], [], '-', color=colors[k], lw=2, alpha=0.9)[0] for k in range(K)]
    connector_lines = [ax.plot([], [], 'r-', lw=1.4, alpha=0.75)[0] for _ in range(K-1)]

    text_iter = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def animate(i):
        idx = min(i, len(iter_history)-1)
        z = iter_history[idx]
        H, ys = unpack(z)
        h = H / K

        all_y = [x_init[1], x_target[1]]
        total_defect = 0.0
        for k in range(K):
            yk = ys[k]
            yend, seg = flow(yk, h)
            # Map local time to [k, k+1]
            t_local = seg.t
            x_vals = k + (t_local / t_local[-1])
            y_vals = seg.y[1, :]
            segment_lines[k].set_data(x_vals, y_vals)
            all_y.extend(y_vals.tolist())
            if k < K-1:
                y_next = ys[k+1]
                # Vertical connector at boundary x=k+1
                connector_lines[k].set_data([k+1, k+1], [yend[1], y_next[1]])
                total_defect += abs(y_next[1] - yend[1])

        # Fixed x-limits in index space
        ax.set_xlim(-0.1, K + 0.1)
        ymin, ymax = min(all_y), max(all_y)
        margin_y = 0.10 * max(1.0, ymax - ymin)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)

        text_iter.set_text(f'Iterate {idx+1}/{len(iter_history)}  |  Sum vertical defect: {total_defect:.3e}')
        return segment_lines + connector_lines + [text_iter]

    anim = FuncAnimation(fig, animate, frames=len(iter_history), interval=600, blit=False, repeat=True)
    plt.tight_layout()
    js_anim = anim.to_jshtml()
    plt.close(fig)
    return js_anim


def main():
    # Problem definition
    x_init = np.array([0.0, 0.0])      # start at origin
    x_target = np.array([10.0, 0.0])   # hit ground at x=10 m
    K = 6                               # number of shooting segments

    sol, iter_hist = solve_bvp_multiple_shooting(K=K, x_init=x_init, x_target=x_target)
    # Optionally show static reconstruction (commented for docs cleanliness)
    # reconstruct_and_plot(sol, K, x_init, x_target)

    # Animate progression (defect shrinkage across segments) and display as JS
    js_anim = create_animation_progress(iter_hist, K, x_init, x_target)
    display(HTML(js_anim))


if __name__ == "__main__":
    main()
```





### Exemple : planification d'une cascade hydroélectrique avec routage physique

Le problème aux limites balistique couple des segments consécutifs d'une même trajectoire. Une cascade hydroélectrique ajoute une seconde forme de couplage : les actions prises en amont modifient, après un délai de propagation, les apports observés en aval. Le tir multiple expose ces deux formes au moyen d'intégrations locales d'EDO, de défauts de continuité temporelle et de contraintes de routage entre biefs.

L'[exemple du réservoir saisonnier](finite-horizon-dp.md), plus loin dans le livre, traite un seul barrage avec un bilan hydrique hebdomadaire et une prévision d'apports connue. Cette abstraction est utile pour l'apprentissage et la conception de la commande, mais elle omet une bonne partie du comportement physique des rivières et des barrages. On utilise ici un cadre plus détaillé, inspiré de {cite:p}`Savorgnan2011`. On considère une série de barrages disposés en cascade, où les actions prises en amont influencent les niveaux en aval avec un certain délai. La puissance produite dépend du débit d'eau à travers les turbines et de la hauteur de chute (la distance verticale entre la surface du réservoir et la sortie de la turbine). Plus la hauteur de chute est grande, plus l'énergie potentielle disponible pour la conversion en électricité est importante, et plus la puissance produite est élevée.

Pour représenter ces effets, on suit une approche de modélisation inspirée des équations de Saint-Venant, qui décrivent l'évolution des niveaux et des débits d'eau dans les canaux à surface libre. Au lieu de résoudre les équations aux dérivées partielles complètes, on utilise un modèle réduit qui approche chaque tronçon de rivière retenu par un barrage (appelé bief) par un système à paramètres localisés régi par une équation différentielle ordinaire. La principale variable d'intérêt est le niveau d'eau $h_r(t)$, qui évolue dans le temps selon la quantité d'eau qui entre, la quantité turbinée $q_r(t)$ et la quantité déversée $s_r(t)$. Le bilan de masse du bief $r$ s'écrit :

$$
\frac{d h_r(t)}{dt} = \frac{1}{A_r} \left( z_r(t) - q_r(t) - s_r(t) \right),
$$

où $A_r$ est la superficie du réservoir, supposée constante. L'apport $z_r(t)$ d'un bief provient soit de la nature (pour le premier barrage), soit des débits turbiné et déversé en amont, retardés d'un temps de propagation $\tau_{r-1}$ :

$$
z_1(t) = \text{apport}(t), \qquad
z_r(t) = q_{r-1}(t - \tau_{r-1}) + s_{r-1}(t - \tau_{r-1}), \quad \text{pour } r > 1.
$$

La production d'électricité de chaque bief dépend de la quantité d'eau turbinée et de la hauteur de chute disponible :

$$
P_r(t) = \rho g \eta \, q_r(t) \, H_r(h_r(t)),
$$

où $\rho$ est la masse volumique de l'eau, $g$ l'accélération gravitationnelle, $\eta$ le rendement de la turbine, et $H_r(h_r(t))$ la hauteur de chute en fonction du niveau d'eau. Dans certains modèles, la hauteur de chute est approchée par la différence entre le niveau courant et une hauteur fixe du niveau aval (le niveau de l'eau en aval du barrage, après son passage dans la turbine).

L'objectif de l'exploitant est de suivre un profil de production cible $P^\text{ref}(t)$, par exemple dicté par la répartition d'un marché ou par une contrainte de suivi de charge. On obtient ainsi un objectif qui minimise l'écart à la cible sur tout l'horizon :

$$
\min_{\{q_r(t), s_r(t)\}} \int_0^T \left( \sum_{r=1}^R P_r(t) - P^\text{ref}(t) \right)^2 dt.
$$

En pratique, on y combine des contraintes d'exploitation : la capacité des turbines $0 \le q_r(t) \le \bar{q}_r$, les limites des évacuateurs de crue $0 \le s_r(t) \le \bar{s}_r$ et des bornes de niveau sécuritaires $h_r^{\min} \le h_r(t) \le h_r^{\max}$. Selon l'usage, on peut aussi pénaliser le déversement pour favoriser la conservation de l'eau, ou pénaliser les variations rapides de niveau pour des raisons écologiques.

Les biefs sont couplés dans l'espace et dans le temps. Un bief en amont ne peut pas simplement agir isolément : si l'exploitant veut que le bief $r$ produise de l'électricité à un moment précis, l'eau doit être lâchée par le bief $r-1$ suffisamment à l'avance. Cette coordination est encore compliquée par les délais, par les non-linéarités de la puissance dépendant de la hauteur de chute et par la capacité de stockage limitée.

On résout le problème par **tir multiple**. Chaque bief est divisé en segments de simulation locaux sur de courtes fenêtres temporelles. À l'intérieur de chaque segment, la dynamique est intégrée vers l'avant au moyen des EDO, et des contraintes de continuité sont ajoutées pour garantir que les niveaux d'eau concordent aux frontières des segments. En même temps, les apports transmis par les biefs en amont doivent arriver au bon moment et être cohérents avec les décisions précédentes. En temps discret, on obtient un ensemble d'équations de mise à jour de l'état :

$$
h_r^{k+1} = h_r^k + \Delta t \cdot \frac{1}{A_r}(z_r^k - q_r^k - s_r^k),
$$

où les délais sont pris en compte en décalant $z_r^k$ selon le temps de propagation approprié. Ces contraintes sont imposées dans un programme non linéaire, avec l'objectif de suivi de puissance et les bornes sur les commandes.

Par rapport à un modèle d'entrées-sorties à réservoir unique, la cascade ajoute des contraintes de couplage retardé. Les réservoirs en amont peuvent stocker de l'eau en prévision des besoins futurs, tandis que les barrages en aval ajustent leur production pour s'accorder aux arrivées et éviter les débordements. La planification obtenue coordonne tout le système en fonction du profil de demande.

```{code-cell} python
:tags: [hide-input]
:label: fig-trajectories-hydro-multiple-shooting
:caption: Le tir multiple coordonne les niveaux des réservoirs, les débits turbinés, les apports routés et la production totale dans une cascade hydroélectrique à trois biefs.

%config InlineBackend.figure_format = 'retina'
# Instrumented MSD hydro demo with heterogeneity + diagnostics
# - Breaks symmetry to avoid trivial identical plots
# - Adds rich diagnostics to explain flat levels and equalities
#
# This cell runs end-to-end and shows plots + tables.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize
from math import sqrt
import warnings

# ---------- Model ----------

g = 9.81  # m/s^2

@dataclass
class ReachParams:
    L: float
    W: float
    k_b: float
    S_b: float
    k_t: float
    @property
    def A_surf(self) -> float:
        return self.L * self.W

def smooth_relu(x, eps=1e-9):
    return 0.5*(x + np.sqrt(x*x + eps))

def q_bypass(H, rp: ReachParams):
    H_eff = smooth_relu(H)
    return rp.k_b * rp.S_b * np.sqrt(2*g*H_eff)

def muskingum_coeffs(K: float, X: float, dt: float) -> Tuple[float, float, float]:
    D  = 2.0*K*(1.0 - X) + dt
    C0 = (dt - 2.0*K*X) / D
    C1 = (dt + 2.0*K*X) / D
    C2 = (2.0*K*(1.0 - X) - dt) / D
    return C0, C1, C2

def integrate_interval(H0, u, z, dt, nsub, rp: ReachParams):
    """Forward Euler. Returns Hend, avg_qout."""
    h = dt/nsub
    H = H0
    qsum = 0.0
    for _ in range(nsub):
        qb = q_bypass(H, rp)
        qout = u + qb
        dHdt = (z - qout) / rp.A_surf
        H += h*dHdt
        qsum += qout
    return H, qsum/nsub

def shapes(M,N): return (M*(N+1), M*N, M*N)

def unpack(x, M, N):
    nH, nu, nz = shapes(M,N)
    H = x[:nH].reshape(M,N+1)
    u = x[nH:nH+nu].reshape(M,N)
    z = x[nH+nu:nH+nu+nz].reshape(M,N)
    return H,u,z

def pack(H,u,z): return np.concatenate([H.ravel(), u.ravel(), z.ravel()])

# ---------- Problem builder ----------

def make_params_hetero(M):
    """Heterogeneous reaches to break symmetry."""
    # Widths, spillway areas, and power coeffs vary by reach
    W_list = np.linspace(80, 140, M)         # m
    L_list = np.full(M, 4000.0)              # m
    S_b_list = np.linspace(14.0, 20.0, M)    # m^2
    k_t_list = np.linspace(7.5, 8.5, M)      # power coeff
    k_b_list = np.linspace(0.55, 0.65, M)    # spill coeff
    return [ReachParams(L=float(L_list[i]), W=float(W_list[i]),
                        k_b=float(k_b_list[i]), S_b=float(S_b_list[i]),
                        k_t=float(k_t_list[i])) for i in range(M)]

def build_demo(M=3, N=12, dt=900.0, seed=0, hetero=True):
    rng = np.random.default_rng(seed)
    params = make_params_hetero(M) if hetero else [ReachParams(4000.0, 100.0, 0.6, 18.26, 8.0) for _ in range(M)]

    # initial levels (heterogeneous)
    H0 = np.array([17.0, 16.7, 17.3][:M])

    H_ref = np.array([17.0, 16.9, 17.1][:M]) if hetero else np.full(M, 17.0)
    H_bounds = (16.0, 18.5)
    u_bounds = (40.0, 160.0)

    Qin_base = 300.0
    Qin_ext = Qin_base + 30.0*np.sin(2*np.pi*np.arange(N)/N)  # stronger swing

    Pref_raw = 60.0 + 15.0*np.sin(2*np.pi*(np.arange(N)-2)/N)

    # default Muskingum parameters per link (M-1 links)
    if M > 1:
        K_list = list(np.linspace(1800.0, 2700.0, M-1))
        X_list = [0.2]*(M-1)
    else:
        K_list = []
        X_list = []

    return dict(params=params, H0=H0, H_ref=H_ref, H_bounds=H_bounds,
                u_bounds=u_bounds, Qin_ext=Qin_ext, Pref_raw=Pref_raw,
                dt=dt, N=N, M=M, nsub=10,
                muskingum=dict(K=K_list, X=X_list))

# ---------- Objective / constraints / helpers ----------

def compute_total_power(H,u,params):
    M,N = u.shape
    Pn = np.zeros(N)
    for n in range(N):
        for i in range(M):
            Pn[n] += params[i].k_t * u[i,n] * H[i,n]
    return Pn

def decompose_objective(x, data, Pref, wP, wH, wDu):
    H,u,z = unpack(x, data["M"], data["N"])
    params, H_ref = data["params"], data["H_ref"]
    track = np.sum((compute_total_power(H,u,params)-Pref)**2)
    lvl   = np.sum((H[:,:-1]-H_ref[:,None])**2)
    du    = np.sum((u[:,1:]-u[:,:-1])**2)
    return dict(track=wP*track, lvl=wH*lvl, du=wDu*du, raw=dict(track=track,lvl=lvl,du=du))

def make_objective(data, Pref, wP=8.0, wH=0.02, wDu=1e-4):
    params, H_ref, N, M = data["params"], data["H_ref"], data["N"], data["M"]
    def obj(x):
        H,u,z = unpack(x,M,N)
        return (
            wP*np.sum((compute_total_power(H,u,params)-Pref)**2)
            + wH*np.sum((H[:,:-1]-H_ref[:,None])**2)
            + wDu*np.sum((u[:,1:]-u[:,:-1])**2)
        )
    return obj, dict(wP=wP,wH=wH,wDu=wDu)

def make_constraints(data):
    params, H0, Qin_ext, dt, N, M, nsub = (
        data["params"], data["H0"], data["Qin_ext"], data["dt"], data["N"], data["M"], data["nsub"]
    )
    cons = []
    def init_fun(x):
        H,u,z = unpack(x,M,N); return H[:,0]-H0
    cons.append({'type':'eq','fun':init_fun})
    def dyn_fun(x):
        H,u,z = unpack(x,M,N)
        res=[]
        for i in range(M):
            for n in range(N):
                Hend, _ = integrate_interval(H[i,n], u[i,n], z[i,n], dt, nsub, params[i])
                res.append(H[i,n+1]-Hend)
        return np.array(res)
    cons.append({'type':'eq','fun':dyn_fun})
    def coup_fun(x):
        H,u,z = unpack(x,M,N)
        res=[]
        # First reach is exogenous inflow per interval
        for n in range(N):
            res.append(z[0,n]-Qin_ext[n])
        # Downstream links: Muskingum routing
        K_list = data.get("muskingum", {}).get("K", [])
        X_list = data.get("muskingum", {}).get("X", [])
        for i in range(1,M):
            # Seed condition for z[i,0]
            _, I0 = integrate_interval(H[i-1,0], u[i-1,0], z[i-1,0], dt, nsub, params[i-1])
            res.append(z[i,0] - I0)
            # Coefficients
            Ki = K_list[i-1] if i-1 < len(K_list) else 1800.0
            Xi = X_list[i-1] if i-1 < len(X_list) else 0.2
            C0, C1, C2 = muskingum_coeffs(Ki, Xi, dt)
            # Recursion over intervals
            for n in range(N-1):
                # upstream interval-average outflows for n and n+1
                _, I_n   = integrate_interval(H[i-1,n],   u[i-1,n],   z[i-1,n],   dt, nsub, params[i-1])
                _, I_np1 = integrate_interval(H[i-1,n+1], u[i-1,n+1], z[i-1,n+1], dt, nsub, params[i-1])
                res.append(z[i,n+1] - (C0*I_np1 + C1*I_n + C2*z[i,n]))
        return np.array(res)
    cons.append({'type':'eq','fun':coup_fun})
    return cons

def make_bounds(data):
    Hmin,Hmax = data["H_bounds"]
    umin,umax = data["u_bounds"]
    M,N = data["M"], data["N"]
    nH,nu,nz = shapes(M,N)
    lb = np.empty(nH+nu+nz); ub = np.empty_like(lb)
    lb[:nH]=Hmin; ub[:nH]=Hmax
    lb[nH:nH+nu]=umin; ub[nH:nH+nu]=umax
    lb[nH+nu:]=0.0; ub[nH+nu:]=2000.0
    return list(zip(lb,ub))

def residuals(x, data):
    params, H0, Qin_ext, dt, N, M, nsub = (
        data["params"], data["H0"], data["Qin_ext"], data["dt"], data["N"], data["M"], data["nsub"]
    )
    H,u,z = unpack(x, M, N)
    dyn = np.zeros((M,N)); coup = np.zeros((M,N))
    for i in range(M):
        for n in range(N):
            Hend, qavg = integrate_interval(H[i,n], u[i,n], z[i,n], dt, nsub, params[i])
            dyn[i,n] = H[i,n+1] - Hend
            if i == 0:
                coup[i,n] = z[i,n] - Qin_ext[n]
            else:
                # Muskingum residual, align on current index using n and n-1
                Ki = data.get("muskingum", {}).get("K", [1800.0]*(M-1))[i-1]
                Xi = data.get("muskingum", {}).get("X", [0.2]*(M-1))[i-1]
                C0, C1, C2 = muskingum_coeffs(Ki, Xi, dt)
                if n == 0:
                    coup[i,n] = 0.0
                else:
                    _, I_nm1 = integrate_interval(H[i-1,n-1], u[i-1,n-1], z[i-1,n-1], dt, nsub, params[i-1])
                    _, I_n   = integrate_interval(H[i-1,n],   u[i-1,n],   z[i-1,n],   dt, nsub, params[i-1])
                    coup[i,n] = z[i,n] - (C0*I_n + C1*I_nm1 + C2*z[i,n-1])
    return dyn, coup

# ---------- Feasible initial guess with hetero controls ----------

def feasible_initial_guess(data):
    """Feasible x0 with nontrivial u by setting u at mid + per-reach pattern, then integrating to define H,z."""
    M,N,dt,nsub = data["M"], data["N"], data["dt"], data["nsub"]
    params = data["params"]
    umin,umax = data["u_bounds"]
    Qin_ext = data["Qin_ext"]

    # pattern to break symmetry
    base = 0.5*(umin+umax)
    phase = np.linspace(0, np.pi/2, M)
    tgrid = np.arange(N)
    u_pattern = np.array([base + 25*np.sin(2*np.pi*(tgrid/N) + ph) for ph in phase])
    u_pattern = np.clip(u_pattern, umin, umax)

    H = np.zeros((M, N+1)); u = np.zeros((M, N)); z = np.zeros((M, N))
    H[:,0] = data["H0"]
    # Set controls from pattern first
    for i in range(M):
        u[i,:] = u_pattern[i,:]

    # First reach: exogenous inflow, integrate forward and record outflow averages
    qavg_up = np.zeros((M, N))
    for n in range(N):
        z[0,n] = Qin_ext[n]
        Hend, qavg = integrate_interval(H[0,n], u[0,n], z[0,n], dt, nsub, params[0])
        H[0,n+1] = Hend
        qavg_up[0,n] = qavg

    # Downstream reaches with Muskingum routing
    K_list = data.get("muskingum", {}).get("K", [1800.0]*(M-1))
    X_list = data.get("muskingum", {}).get("X", [0.2]*(M-1))
    for i in range(1,M):
        Ki = K_list[i-1] if i-1 < len(K_list) else 1800.0
        Xi = X_list[i-1] if i-1 < len(X_list) else 0.2
        C0, C1, C2 = muskingum_coeffs(Ki, Xi, dt)
        I = qavg_up[i-1,:]
        # seed
        z[i,0] = I[0]
        # propagate recursively over time
        for n in range(N-1):
            z[i,n+1] = C0*I[n+1] + C1*I[n] + C2*z[i,n]
        # integrate levels for reach i using routed inflow
        for n in range(N):
            Hend, qavg = integrate_interval(H[i,n], u[i,n], z[i,n], dt, nsub, params[i])
            H[i,n+1] = Hend
            qavg_up[i,n] = qavg
    return pack(H,u,z)

def scale_pref(Pref_raw, x0, data):
    H,u,z = unpack(x0, data["M"], data["N"])
    P0 = compute_total_power(H,u,data["params"])
    s = max(np.mean(P0),1e-6)/max(np.mean(Pref_raw),1e-6)
    return Pref_raw*s, P0

def run_demo(show: bool = True, save_path: str | None = 'hydro.png', verbose: bool = False):
    """Build, solve, and render the hydro demo.

    Parameters
    ----------
    show : bool
        If True, displays the matplotlib figure via plt.show().
    save_path : str | None
        If provided, saves the figure to this path.
    verbose : bool
        If True, prints diagnostic information.

    Returns
    -------
    matplotlib.figure.Figure | None
        Returns the Figure when show is False; otherwise returns None.
    """
    # ---------- Solve ----------
    data = build_demo(M=3, N=16, dt=900.0, hetero=True)
    x0 = feasible_initial_guess(data)
    Pref, P0 = scale_pref(data["Pref_raw"], x0, data)

    objective, weights = make_objective(data, Pref, wP=8.0, wH=0.02, wDu=5e-4)
    # Suppress noisy SciPy warning about delta_grad during quasi-Newton updates
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"delta_grad == 0.0",
            category=UserWarning,
            module=r"scipy\.optimize\.\_differentiable_functions",
        )
        res = minimize(
            fun=objective,
            x0=x0,
            method='trust-constr',
            bounds=make_bounds(data),
            constraints=make_constraints(data),
            options=dict(maxiter=1000, disp=verbose),
        )

    H,u,z = unpack(res.x, data["M"], data["N"])
    P = compute_total_power(H,u,data["params"])
    dyn_res, coup_res = residuals(res.x, data)

    # ---------- Diagnostics ----------
    if verbose:
        terms = decompose_objective(res.x, data, Pref, **weights)
        print("\n=== Objective decomposition ===")
        print({k: float(v) if not isinstance(v, dict) else {kk: float(vv) for kk,vv in v.items()} for k,v in terms.items()})

        print("\n=== Constraint residuals (max |.|) ===")
        print("dyn:", float(np.max(np.abs(dyn_res)))), print("coup:", float(np.max(np.abs(coup_res))))

        # Muskingum coefficient sanity and residuals
        if data.get("M", 1) > 1:
            K_list = data.get("muskingum", {}).get("K", [])
            X_list = data.get("muskingum", {}).get("X", [])
            coef_checks = []
            mean_abs_res = []
            for i in range(1, data["M"]):
                Ki = K_list[i-1] if i-1 < len(K_list) else 1800.0
                Xi = X_list[i-1] if i-1 < len(X_list) else 0.2
                C0, C1, C2 = muskingum_coeffs(Ki, Xi, data["dt"])
                coef_checks.append(dict(link=i, sum=float(C0+C1+C2), min_coef=float(min(C0,C1,C2))))
                # compute mean abs residual for this link
                res_vals = []
                for n in range(data["N"]-1):
                    _, I_n   = integrate_interval(H[i-1,n],   u[i-1,n],   z[i-1,n],   data["dt"], data["nsub"], data["params"][i-1])
                    _, I_np1 = integrate_interval(H[i-1,n+1], u[i-1,n+1], z[i-1,n+1], data["dt"], data["nsub"], data["params"][i-1])
                    res_vals.append(float(abs(z[i,n+1] - (C0*I_np1 + C1*I_n + C2*z[i,n]))))
                mean_abs_res.append(dict(link=i, mean_abs=float(np.mean(res_vals))))
            print("\n=== Muskingum coeff checks (sum, min_coef) ===")
            print(coef_checks)
            print("=== Muskingum mean |residual| per link ===")
            print(mean_abs_res)

    # Per-interval diagnostic table for each reach (kept for debugging but unused here)
    def interval_table(i):
        rp = data["params"][i]
        rows = []
        for n in range(data["N"]):
            qb = q_bypass(H[i,n], rp)
            net = z[i,n] - (u[i,n] + qb)
            dH = data["dt"]*net/rp.A_surf
            rows.append(dict(interval=n, Hn=H[i,n], Hn1=H[i,n+1], u=u[i,n], z=z[i,n], qb=qb, net_flow=net, dH_pred=dH))
        return pd.DataFrame(rows)

    # summary and tables available to callers if needed
    tables = [interval_table(i) for i in range(data["M"])]
    summary = pd.DataFrame([
        dict(reach=i+1,
             H_mean=float(np.mean(H[i])), H_std=float(np.std(H[i])),
             u_mean=float(np.mean(u[i])), u_std=float(np.std(u[i])),
             z_mean=float(np.mean(z[i])), z_std=float(np.std(z[i])))
        for i in range(data["M"])
    ])

    # ---------- Plots ----------
    M,N = data["M"], data["N"]
    t_nodes = np.arange(N+1)
    t = np.arange(N)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hydroelectric System Optimization Results', fontsize=16)

    ax1 = axes[0, 0]
    for i in range(M):
        ax1.plot(t_nodes, H[i], marker='o', label=f'Reach {i+1}')
    ax1.set_xlabel("Node n"); ax1.set_ylabel("H [m]"); ax1.set_title("Water Levels")
    ax1.grid(True); ax1.legend()

    ax2 = axes[0, 1]
    for i in range(M):
        ax2.step(t, u[i], where='post', label=f'Reach {i+1}')
    ax2.set_xlabel("Interval n"); ax2.set_ylabel("u [m³/s]"); ax2.set_title("Turbine Discharge")
    ax2.grid(True); ax2.legend()

    ax3 = axes[1, 0]
    for i in range(M):
        ax3.step(t, z[i], where='post', label=f'Reach {i+1}')
    ax3.set_xlabel("Interval n"); ax3.set_ylabel("z [m³/s]"); ax3.set_title("Inflow (Coupling)")
    ax3.grid(True); ax3.legend()

    ax4 = axes[1, 1]
    ax4.plot(t, P0, marker='s', label="Power @ x0")
    ax4.plot(t, P, marker='o', label="Power @ optimum")
    ax4.plot(t, Pref, marker='x', label="Scaled Pref")
    ax4.set_xlabel("Interval n"); ax4.set_ylabel("Power units"); ax4.set_title("Power Tracking")
    ax4.legend(); ax4.grid(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        return None
    return fig


# Run the demo directly when loaded in a notebook cell
run_demo(show=True, save_path=None, verbose=False)

```


La figure présente le résultat d'une optimisation par tir multiple appliquée à une cascade hydroélectrique à trois biefs. L'horizon temporel est discrétisé en 16 intervalles, et le solveur `trust-constr` de SciPy sert à trouver une suite de commandes admissible qui respecte le bilan de masse, les limites des turbines et des évacuateurs de crue, ainsi qu'une dynamique de routage de type Muskingum. Le solveur intègre une EDO locale distincte pour chaque bief. Les défauts de tir relient les niveaux des réservoirs dans le temps, tandis que des contraintes de Muskingum distinctes relient les débits routés entre les biefs.

Le panneau supérieur gauche montre les niveaux d'eau de chaque réservoir. On observe que les réservoirs en amont tendent à élever leur niveau avant les épisodes de lâcher, accumulant de l'énergie potentielle avant de libérer l'eau vers l'aval. Le panneau supérieur droit montre les débits turbinés de chaque bief. Ils varient de façon lisse et sont coordonnés dans le temps à l'échelle du système. Le panneau inférieur droit compare la production totale à un profil de demande synthétique, généré par une somme de sigmoïdes décalées dans le temps et normalisé pour être réalisable compte tenu des capacités des turbines. La planification optimisée (orange) suit étroitement cette demande, tandis que l'estimation initiale (bleu) accuse un retard. Le panneau inférieur gauche trace les apports routés entre les biefs, qui présentent le retard et le lissage attendus du routage de Muskingum. L'interaction entre ces graphiques montre comment le système anticipe, stocke et achemine l'eau pour atteindre des cibles de production variables dans le temps à l'intérieur des limites physiques et d'exploitation.

Les exemples balistique et hydroélectrique utilisent la même structure
numérique à des échelles différentes : intégrer localement, exposer les états
aux frontières des segments et ramener chaque défaut de continuité à zéro.
L'introduction de ces états de frontière modifie les variables d'optimisation
et leur couplage, que le modèle physique sous-jacent comporte ou non des
perturbations aléatoires.

## Résumé et perspectives

La transcription directe garde les états et les commandes explicites et expose
des contraintes de dynamique creuses. Le tir simple élimine les états, mais
couple les premières actions à toutes les quantités ultérieures par un seul
déroulement. L'exemple de mise à quai du camion a mesuré ce couplage : les
dérivées en mode inverse à travers le déroulement reproduisent le produit des
jacobiennes d'attelage par intervalle, et la marche arrière amplifie une
erreur de braquage précoce qui s'atténue lorsque l'attelage roule en marche
avant. Le tir multiple garde certains états de frontière, échangeant des
variables supplémentaires contre des chemins de sensibilité plus courts et des
défauts de continuité creux.

Les formulations de transcription utilisent une application de transition
discrète. Le chapitre suivant,
[transcription en temps continu et collocation](continuous-time-collocation.md),
développe des représentations polynomiales nodales directement à partir des
équations différentielles. Le [chapitre sur MPPI](model-predictive-path-integral-control.md)
qui suit revient au déroulement du tir simple, en échantillonnant des suites
de commandes candidates et en les combinant selon leurs coûts. Pour les
modèles à perturbations incertaines, plusieurs futurs simulés estiment le coût
espéré de chaque candidate.

## Exercices

:::{exercise} Implémentation du tir simple
:label: ex-trajectories-single-shooting

On considère le système scalaire $x_{t+1} = x_t + u_t$ avec $x_1 = 1$ et l'objectif :

$$
J = x_T^2 + \sum_{t=1}^{T-1} u_t^2.
$$

**(a)** Exprimez $x_T$ en fonction des commandes $u_1, \ldots, u_{T-1}$.

**(b)** Substituez dans $J$ pour obtenir un objectif sans contraintes qui ne dépend que des commandes.

**(c)** Implémentez le tir simple en Python/JAX pour minimiser $J$ avec $T = 10$. Utilisez la descente de gradient avec un taux d'apprentissage de 0.01 pendant 100 itérations. Rapportez les commandes optimales et le coût final.

:::

:::{solution} ex-trajectories-single-shooting
:class: dropdown

**(a)** $x_T = 1 + \sum_{t=1}^{T-1} u_t$.

**(b)** $J(u) = \left(1 + \sum_{t=1}^{T-1} u_t\right)^2 + \sum_{t=1}^{T-1} u_t^2$.

**(c)** Exemple de code :
```python
import jax.numpy as jnp
from jax import grad

def objective(u):
    x_T = 1.0 + jnp.sum(u)
    return x_T**2 + jnp.sum(u**2)

T = 10
u = jnp.zeros(T - 1)
for _ in range(100):
    u = u - 0.01 * grad(objective)(u)

print(f"Optimal u: {u}, Cost: {objective(u):.4f}")
```
Par symétrie, les commandes optimales sont égales, $u_t = -x_1/T = -0.1$, ce qui donne $x_T = 0.1$ et un coût total $J = x_1^2/T = 0.1$ : la pénalité quadratique sur la commande rend moins coûteux de répartir la correction sur toutes les étapes que d'annuler exactement $x_1$. (La plus grande valeur propre de la hessienne de $J$ vaut $2T = 20$; un taux d'apprentissage de $0.1$ oscillerait donc sans converger, alors que $0.01$ converge en bien moins de 100 itérations.)
:::

---

:::{exercise} Segments de tir multiple
:label: ex-trajectories-multiple-shooting

À partir du problème de @ex-trajectories-single-shooting, implémentez le tir multiple avec $K = 3$ segments.

**(a)** Définissez les frontières des segments et les défauts de continuité.

**(b)** Formulez le PNL avec les variables de décision $[x_1, x_4, x_7, u_1, \ldots, u_9]$ (pour $T=10$, avec des segments de longueur 3).

**(c)** Comparez le comportement de convergence à celui du tir simple. Le tir multiple exige-t-il moins d'itérations pour atteindre la même tolérance?

:::

:::{solution} ex-trajectories-multiple-shooting
:class: dropdown

Les défauts sont $d_k = x_{k+1}^{\text{simulé}} - x_{k+1}^{\text{variable}}$ aux frontières des segments. On peut minimiser $J + \rho \sum_k \|d_k\|^2$ pour un grand $\rho$ (méthode de pénalité) ou utiliser un solveur sous contraintes. Le tir multiple converge habituellement plus vite sur de longs horizons, car le paysage d'optimisation est mieux conditionné.
:::

---

:::{exercise} Stationnement du camion avec un optimiseur du premier ordre
:label: ex-trajectories-truck-first-order

L'exemple de mise à quai de @sec-truck-alley-dock minimise $J(\mathbf u)$ avec
L-BFGS-B. À l'aide du modèle téléchargeable, remplacez le solveur par une
descente de gradient projetée à pas fixe, en projetant chaque itéré sur la
boîte des commandes après chaque mise à jour.

**(a)** À l'amorce, le gradient par rapport au premier angle de braquage vaut
environ $800$, et celui par rapport au dernier environ $0.04$. À l'aide du
profil d'amplification de la {numref}`fig-truck-conditioning`, expliquez
pourquoi un pas qui est stable pour les premières commandes de braquage ne
fait presque aucun progrès sur les dernières.

**(b)** Exécutez la descente de gradient projetée à partir de la même amorce
en ligne droite pour le même nombre d'évaluations de l'objectif que celui
utilisé par L-BFGS-B, et comparez le coût atteint ainsi que les vérifications
d'arrivée rapportées par `audit`.

**(c)** Recommencez avec Adam, qui remet chaque coordonnée à l'échelle par une
estimation courante de l'amplitude de son gradient. La mise à l'échelle par
coordonnée élimine-t-elle la difficulté, ou seulement une partie de celle-ci?

:::

:::{solution} ex-trajectories-truck-first-order
:class: dropdown

**(a)** Le pas stable de la descente de gradient est borné par la plus grande
courbure de $J$, qui est fixée par les premières commandes de braquage,
puisque leur effet sur le coût terminal est amplifié tout au long du
déroulement en marche arrière. Ce même pas, multiplié par une composante du
gradient de $0.04$, ne déplace les dernières commandes que d'une quantité
négligeable; progresser dans ces directions demande donc des milliers
d'itérations.

**(b)** À budget d'évaluations égal, la descente de gradient projetée
s'arrête habituellement loin du coût obtenu par L-BFGS-B, et les
vérifications d'arrivée échouent sur le cap ou sur l'angle d'attelage.
Augmenter le pas pour accélérer les dernières commandes fait osciller les
premières.

**(c)** Adam remet chaque coordonnée à l'échelle et atteint donc un plan de
mise à quai en quelques milliers d'itérations, mais le couplage entre les
premières et les dernières commandes est hors diagonale et subsiste : son coût
après plusieurs milliers d'itérations reste supérieur à la valeur obtenue par
L-BFGS-B en moins d'un millier.
:::

## Autoévaluation

:::{exercise} Compter les décisions
:label: ex-trajectories-check-1

Un problème de tir simple a un horizon $T$ et des commandes scalaires. Combien de variables d'optimisation restent après l'élimination des états, et quel calcul couple une commande précoce au coût terminal?
:::

:::{solution} ex-trajectories-check-1
:class: dropdown

Avec les états $x_1,\ldots,x_T$ du chapitre, il y a $T-1$ variables de
commande. La simulation vers l'avant couple chaque commande précoce à tous les
états ultérieurs, et donc au coût terminal.
:::

:::{exercise} Compromis du tir
:label: ex-trajectories-check-2

Pourquoi le tir multiple peut-il être plus facile à optimiser que le tir simple, même s'il introduit davantage de variables de décision?
:::

:::{solution} ex-trajectories-check-2
:class: dropdown

Les états intermédiaires découpent un long déroulement sensible en segments plus courts. Les contraintes de continuité qui en résultent sont creuses, et les dérivées n'ont pas à se propager sur tout l'horizon en une seule chaîne.
:::

:::{exercise} Sens de la marche
:label: ex-trajectories-check-direction

Dans l'exemple du camion, le même chemin est parcouru en marche arrière dans un plan et en marche avant dans l'autre. Pourquoi une erreur de braquage précoce compte-t-elle quatorze fois plus dans un sens et quatorze fois moins dans l'autre?
:::

:::{solution} ex-trajectories-check-direction
:class: dropdown

L'angle d'attelage obéit à $\dot\beta = (v/L_0)\tan\delta - (v/L_1)\sin\beta$.
Sa linéarisation a pour coefficient $-v\cos\beta/L_1$ : négatif en marche
avant, de sorte que les écarts s'atténuent, et positif en marche arrière, de
sorte qu'ils croissent. La sensibilité terminale à une modification précoce du
braquage est le produit de ces facteurs par intervalle, que la différentiation
du déroulement en mode inverse calcule exactement.
:::
