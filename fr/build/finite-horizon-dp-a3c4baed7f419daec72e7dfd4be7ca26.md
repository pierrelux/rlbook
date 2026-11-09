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
# Programmation dynamique à horizon fini

La commande prédictive produit une rétroaction en résolvant un nouveau problème
de trajectoire à partir de chaque état mesuré. Les décisions pour toute une
famille d'états peuvent-elles plutôt être calculées à l'avance? La
programmation dynamique décompose un problème de commande à horizon fini en
fonctions de coût restant indexées par le temps et par l'état.

Le problème de contrôle optimal déterministe en temps discret sert de point de
départ. Les successeurs stochastiques entreront au chapitre suivant par
l'intermédiaire d'espérances conditionnelles.

Considérons un PCOTD typique de type Bolza :

$$
\begin{align*}
\text{minimiser} \quad & J \triangleq c_\mathrm{T}(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t) \\
\text{sous les contraintes} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), \quad t = 1, \ldots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, \quad t = 1, \ldots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, \quad t = 1, \ldots, T, \\
\text{étant donné} \quad & \mathbf{x}_1
\end{align*}
$$

Plutôt que de considérer seulement le coût total de l'instant initial à l'instant final, la programmation dynamique introduit la notion de coût à partir d'un instant quelconque jusqu'à la fin. On aboutit ainsi à la définition du « coût restant », ou « fonction de valeur », $J_k(\mathbf{x}_k)$ :

$$
J_k(\mathbf{x}_k) \triangleq c_\mathrm{T}(\mathbf{x}_T) + \sum_{t=k}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t)
$$

Cette fonction représente le coût total encouru de l'étape $k$ jusqu'à la fin de l'horizon, sachant que le système est initialisé dans l'état $\mathbf{x}_k$ à l'étape $k$. Supposons que le problème ait été résolu de l'étape $k+1$ jusqu'à la fin, ce qui donne le coût restant optimal $J_{k+1}^\star(\mathbf{x}_{k+1})$ pour tout état $\mathbf{x}_{k+1}$ à l'étape $k+1$. La question devient alors : comment cette information éclaire-t-elle la décision à l'étape $k$?

Connaissant le comportement optimal à partir de $k+1$, la tâche se réduit à déterminer l'action optimale $\mathbf{u}_k$ à l'étape $k$. Cette commande doit minimiser la somme du coût immédiat $c_k(\mathbf{x}_k, \mathbf{u}_k)$ et du coût futur optimal $J_{k+1}^\star(\mathbf{x}_{k+1})$, où $\mathbf{x}_{k+1}$ est l'état obtenu après l'application de l'action $\mathbf{u}_k$. Mathématiquement, cela s'écrit :

$$
J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \left[ c_k(\mathbf{x}_k, \mathbf{u}_k) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k)) \right]
$$

Cette équation est connue sous le nom d'équation de Bellman, d'après Richard Bellman, qui a formulé le principe d'optimalité :

> Une politique optimale a la propriété que, quels que soient l'état et la décision précédents, les décisions restantes doivent constituer une politique optimale par rapport à l'état qui résulte de la décision précédente.

Autrement dit, tout sous-chemin d'un chemin optimal, de n'importe quel point intermédiaire jusqu'à la fin, doit lui-même être optimal. Ce principe est le fondement de la procédure d'induction arrière, qui calcule la fonction de valeur optimale et fournit une commande en boucle fermée sans recourir à un solveur explicite de PNL. 

La programmation dynamique traite naturellement les systèmes non linéaires et les fonctions de coût non quadratiques. Elle fournit une solution optimale globale, lorsqu'il en existe une, et peut intégrer assez facilement des contraintes sur l'état et sur la commande. Ces avantages proviennent d'un changement dans ce que l'on calcule. L'[optimisation de trajectoires](numerical-trajectory-optimization.md) prend l'état initial $\mathbf{x}_1$ comme donné et renvoie une seule suite d'états et de commandes, habituellement un optimum local. Avec un état de dimension $n$ et une commande de dimension $m$, le programme non linéaire obtenu compte de l'ordre de $T(n+m)$ variables de décision; sa taille ne croît donc que linéairement avec l'horizon et la dimension. L'[optimisation paramétrique](parametric-controllers.md) a élargi la question à une famille de problèmes indexée par l'état initial, mais y a encore répondu localement : les mises à jour par sensibilité transportent la solution d'une instance vers une instance voisine, et les correcteurs appris ne sont ajustés que sur des états échantillonnés dans le domaine d'exploitation. La récurrence arrière demande le coût restant optimal et la décision optimale à partir de chaque état et à chaque étape. La réponse est une fonction sur tout l'espace d'états plutôt qu'un vecteur de variables de décision.

C'est la représentation d'une telle fonction qui devient coûteuse à mesure que la dimension de l'espace d'états augmente, une difficulté que Bellman a appelée le « fléau de la dimension ». Si chacune des $n$ coordonnées de l'état est discrétisée en $N$ niveaux, la table qui stocke $J_t^\star$ compte $N^n$ entrées, et chaque entrée exige sa propre minimisation sur les commandes. Le calcul et la mémoire croissent donc exponentiellement avec la dimension de l'état, ce qui rend l'application directe de la programmation dynamique impraticable pour les problèmes de grande dimension. Ce fléau est le prix à payer pour demander une solution partout. L'optimisation de trajectoires renonce à la réponse globale en boucle fermée pour garder un coût polynomial en la dimension; la programmation dynamique obtient cette réponse et la paie par une croissance exponentielle.

Heureusement, les méthodes fondées sur l'apprentissage offrent des outils efficaces pour combattre le fléau de la dimension sur deux fronts : l'approximation de fonctions (par exemple, des réseaux de neurones) pour éviter une discrétisation explicite, et la randomisation au moyen des méthodes de Monte-Carlo inhérentes au paradigme de l'apprentissage. La majeure partie de ce cours est consacrée à ces idées.

## Récurrence arrière

Comment le coût terminal se propage-t-il vers l'arrière pour donner une action
et une valeur optimales pour chaque état antérieur?

Le principe d'optimalité fournit une méthode de résolution des problèmes de contrôle optimal. En partant de la fin de l'horizon et en remontant le temps, on résout à chaque étape le problème d'optimisation local donné par l'équation de Bellman. Ce processus, appelé récurrence arrière ou induction arrière, construit la fonction de valeur optimale étape par étape.

````{prf:algorithm} Récurrence arrière pour la programmation dynamique
:label: backward-recursion-deterministic

**Entrée :** fonction de coût terminal $c_\mathrm{T}(\cdot)$, fonctions de coût d'étape $c_t(\cdot, \cdot)$, dynamique du système $f_t(\cdot, \cdot)$, horizon $\mathrm{T}$

**Sortie :** fonctions de valeur optimales $J_t^\star(\cdot)$ et politiques de commande optimales $\pi_t^\star(\cdot)$ pour $t = 1, \ldots, T$

1. Initialiser $J_T^\star(\mathbf{x}) = c_\mathrm{T}(\mathbf{x})$ pour tout $\mathbf{x}$ de l'espace d'états
2. Pour $t = T-1, T-2, \ldots, 1$ :
   1. Pour chaque état $\mathbf{x}$ de l'espace d'états :
      1. Calculer $J_t^\star(\mathbf{x}) = \min_{\mathbf{u}} \left[ c_t(\mathbf{x}, \mathbf{u}) + J_{t+1}^\star(f_t(\mathbf{x}, \mathbf{u})) \right]$
      2. Calculer $\pi_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u}} \left[ c_t(\mathbf{x}, \mathbf{u}) + J_{t+1}^\star(f_t(\mathbf{x}, \mathbf{u})) \right]$
   2. Fin pour
3. Fin pour
4. Renvoyer $J_t^\star(\cdot)$, $\pi_t^\star(\cdot)$ pour $t = 1, \ldots, T$
````


À la fin de cette passe arrière, on dispose de la commande optimale à appliquer à n'importe quelle étape et dans n'importe quel état. De plus, on peut simuler des trajectoires optimales à partir de n'importe quel état initial en appliquant la politique optimale à chaque étape pour générer la trajectoire optimale.



```{prf:theorem} L'induction arrière résout le PCOTD déterministe de Bolza
:label: thm-bolza-backward

**Cadre.** Soit $\mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)$ pour $t=1,\dots,T-1$, avec des ensembles d'actions admissibles $\mathcal{U}_t(\mathbf{x})\neq\varnothing$. Supposons que les coûts d'étape $c_t(\mathbf{x},\mathbf{u})$ et le coût terminal $c_\mathrm{T}(\mathbf{x})$ soient à valeurs réelles et bornés inférieurement. Supposons que, pour tout $(t,\mathbf{x})$, le problème à un pas

$$
\min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x})}\big\{c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star(\mathbf{f}_t(\mathbf{x},\mathbf{u}))\big\}
$$
admette un minimiseur (par exemple, un ensemble $\mathcal{U}_t(\mathbf{x})$ compact et la continuité suffisent).

Définissons $J_T^\star(\mathbf{x}) \equiv c_\mathrm{T}(\mathbf{x})$ et, pour $t=T-1,\dots,1$,

$$
J_t^\star(\mathbf{x}) \;\triangleq\; \min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x})}
\Big[c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star\big(\mathbf{f}_t(\mathbf{x},\mathbf{u})\big)\Big],
$$
et soit $\boldsymbol{\pi}_t^\star(\mathbf{x})$ un minimiseur quelconque. La fonction $\boldsymbol{\pi}_t^\star$, qui renvoie une commande minimisante pour chaque état, est la **règle de décision** à l'étape $t$.

**Énoncé.** Pour tout état initial $\mathbf{x}_1$, la suite de commandes
$\boldsymbol{\pi}_1^\star(\mathbf{x}_1),\dots,\boldsymbol{\pi}_{T-1}^\star(\mathbf{x}_{T-1})$
générée par ces règles de décision est optimale pour le problème de Bolza, et
$J_1^\star(\mathbf{x}_1)$ est égal au coût optimal. De plus, $J_t^\star(\cdot)$ est le coût restant optimal à partir de l'étape $t$ pour tout état; autrement dit, l'induction arrière retrouve la fonction de valeur tout entière.
```

La preuve utilise un argument de « couper-coller ». Toute suite de commandes
peut être coupée après son premier pas, et sa queue remplacée par une
continuation optimale sans que son coût augmente. Comparer les premiers pas,
chacun suivi de sa continuation optimale, donne alors une suite qu'aucune autre
suite ne peut battre. La {numref}`fig-cut-and-paste` suit ces deux opérations
avec des coûts concrets.

:::{figure} _static/cut-and-paste.svg
:label: fig-cut-and-paste
:alt: Deux panneaux couvrant les étapes t à T. À gauche, une suite partant de x fait un premier pas de coût 2 vers x_{t+1}, puis une queue en pointillés de coût 9; une queue optimale à partir du même x_{t+1} coûte 7. À droite, trois premiers pas à partir de x coûtent 2, 3 et 1 et mènent à des états dont les queues optimales coûtent 7, 4 et 8; le pas du milieu, suivi de sa queue, coûte 7.

Une étape d'induction de la preuve du {prf:ref}`thm-bolza-backward` avec des
coûts concrets. **Étape 1 (couper) :** une suite quelconque partant de
$\mathbf{x}$ coûte $2 + 9 = 11$. Garder son premier pas et remplacer sa queue
par la queue optimale à partir de $\mathbf{x}_{t+1}$ donne $2 + 7 = 9$; la
suite coûte donc au moins $c_t + J^\star_{t+1}(\mathbf{x}_{t+1})$. **Étape 2
(coller) :** le candidat du haut est le premier pas de l'étape 1. Parmi les
trois premiers pas, chacun suivi de sa queue optimale, celui du milieu est le
moins coûteux, et son total $3 + 4 = 7$ est $J^\star_t(\mathbf{x})$. La suite
de l'étape 1 coûte au moins autant ($11 \geq 9 \geq 7$), et coller la queue
optimale à $\boldsymbol{\pi}^\star_t(\mathbf{x})$ atteint ce coût.
:::

```{prf:proof}
La preuve procède par induction arrière sur $t$. L'énoncé transmis d'une étape
à la précédente est :

> **($H_t$)** Pour tout état $\mathbf{y}$, le plus faible coût atteignable à
> partir de l'étape $t$ en partant de $\mathbf{y}$ est égal à
> $J_t^\star(\mathbf{y})$, et les règles de décision
> $\boldsymbol{\pi}_t^\star,\dots,\boldsymbol{\pi}_{T-1}^\star$ génèrent une
> suite qui l'atteint.

Ici, $J_t^\star$ est la fonction produite par la récurrence. C'est son égalité
avec le coût restant optimal qu'il faut démontrer.

**Cas de base ($t=T$).** Il ne reste aucune commande à choisir; le seul coût à
partir de $\mathbf{y}$ est donc $c_\mathrm{T}(\mathbf{y})$, qui vaut
$J_T^\star(\mathbf{y})$ par définition.

**Étape d'induction.** Supposons que $H_{t+1}$ soit vraie et fixons un état
$\mathbf{x}$ à l'étape $t$.

*Étape 1 (couper : une borne inférieure pour toute suite).* Prenons une suite
admissible quelconque $\mathbf{u}_t,\dots,\mathbf{u}_{T-1}$ à partir de
$\mathbf{x}_t=\mathbf{x}$, avec les états induits
$\mathbf{x}_{k+1}=\mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k)$. Coupons-la après
le premier pas. Son coût se décompose en coût de ce pas et en coût des
commandes restantes $\mathbf{u}_{t+1},\dots,\mathbf{u}_{T-1}$, qui forment une
suite partant de $\mathbf{x}_{t+1}$ à l'étape $t+1$ :

$$
\underbrace{c_t(\mathbf{x},\mathbf{u}_t)}_{\text{premier pas}}
\;+\;
\underbrace{\sum_{k=t+1}^{T-1}c_k(\mathbf{x}_k,\mathbf{u}_k)+c_\mathrm{T}(\mathbf{x}_T)}_{\text{queue à partir de }\mathbf{x}_{t+1}}.
$$

Par $H_{t+1}$, aucune suite partant de $\mathbf{x}_{t+1}$ à l'étape $t+1$ ne
coûte moins que $J_{t+1}^\star(\mathbf{x}_{t+1})$, et cela vaut en particulier
pour la queue. Remplacer le coût de la queue par
$J_{t+1}^\star(\mathbf{x}_{t+1})$ ne peut donc pas augmenter le total. Le
premier pas $\mathbf{u}_t$ est l'un des candidats du minimum qui définit
$J_t^\star(\mathbf{x})$, donc

$$
\begin{aligned}
\text{coût de la suite}
&\;\ge\; c_t(\mathbf{x},\mathbf{u}_t)+J_{t+1}^\star\big(\mathbf{f}_t(\mathbf{x},\mathbf{u}_t)\big) \\
&\;\ge\; \min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x})}\Big[c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star\big(\mathbf{f}_t(\mathbf{x},\mathbf{u})\big)\Big]
\;=\; J_t^\star(\mathbf{x}).
\end{aligned}
$$ (eq:cut-lower-bound)

La suite était quelconque; aucune suite admissible à partir de
$(t,\mathbf{x})$ ne coûte donc moins que $J_t^\star(\mathbf{x})$. Dans la
{numref}`fig-cut-and-paste`, les deux inégalités sont $11 \ge 9$ et $9 \ge 7$.

*Étape 2 (coller : une suite qui atteint la borne).* Prenons le premier pas
$\boldsymbol{\pi}_t^\star(\mathbf{x})$, qui atteint le minimum dans {eq}`eq:cut-lower-bound`.
Il mène à $\mathbf{x}_{t+1}^\star=\mathbf{f}_t\big(\mathbf{x},\boldsymbol{\pi}_t^\star(\mathbf{x})\big)$,
et $H_{t+1}$ fournit une queue à partir de $\mathbf{x}_{t+1}^\star$, générée par
$\boldsymbol{\pi}_{t+1}^\star,\dots,\boldsymbol{\pi}_{T-1}^\star$, dont le coût est
$J_{t+1}^\star(\mathbf{x}_{t+1}^\star)$. Collons cette queue après le premier pas.
La suite obtenue coûte

$$
c_t\big(\mathbf{x},\boldsymbol{\pi}_t^\star(\mathbf{x})\big)
+J_{t+1}^\star(\mathbf{x}_{t+1}^\star)
=J_t^\star(\mathbf{x}).
$$

Combinons ce résultat avec l'étape 1. Toute suite admissible à partir de
$(t,\mathbf{x})$ coûte au moins $J_t^\star(\mathbf{x})$, et la suite collée est
l'une d'elles et coûte exactement $J_t^\star(\mathbf{x})$. Aucune suite ne peut
donc coûter moins que la suite collée; celle-ci est donc optimale, et le plus
faible coût atteignable à partir de $(t,\mathbf{x})$ est $J_t^\star(\mathbf{x})$.
Dans la {numref}`fig-cut-and-paste`, toute suite coûte au moins 7 et la suite
collée coûte 7. La suite collée est générée par les règles de décision à partir
de l'étape $t$; $H_t$ est donc vraie.

Par induction, $H_t$ est vraie pour tout $t$, en particulier pour $t=1$ et pour
tout état initial $\mathbf{x}_1$. La récurrence arrière **certifie** donc
l'optimalité (étape 1) et **construit** une politique optimale (étape 2), tout
en retrouvant la famille complète $\{J_t^\star\}_{t=1}^T$.
```

```{prf:remark} Nul besoin d'un « grand PNL »
Le PCOTD de Bolza sur tout l'horizon couple toutes les commandes par l'intermédiaire de la dynamique et se pose habituellement comme un seul grand programme non linéaire. La preuve montre qu'on peut plutôt résoudre **$T-1$ suites de problèmes à un pas** : en chaque $(t,\mathbf{x})$, minimiser

$$
\mathbf{u}\mapsto c_t(\mathbf{x},\mathbf{u}) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x},\mathbf{u})).
$$

Dans les espaces d'états et d'actions finis, cela se réduit à une simple consultation de table et à un argmin. Dans les espaces continus, on résout encore des minimisations locales à un pas, mais on évite un PNL monolithique couplé sur tout l'horizon.
```

```{prf:remark} Interprétation en graphe (intuition facultative)
Déroulons le temps pour former un graphe orienté acyclique dont les nœuds sont les couples $(t,\mathbf{x})$ et dont les arcs correspondent aux commandes admissibles, avec un poids d'arc $c_t(\mathbf{x},\mathbf{u})$. Le coût des nœuds terminaux est $c_\mathrm{T}(\cdot)$. Le problème de Bolza est un problème de plus court chemin sur ce graphe. L'équation

$$
J_t^\star(\mathbf{x})=\min_{\mathbf{u}}\{c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star(\mathbf{f}_t(\mathbf{x},\mathbf{u}))\}
$$

est exactement la récurrence de programmation dynamique pour les plus courts chemins dans les graphes acycliques; l'induction arrière est donc optimale.
```

<!-- ```{prf:remark} Si les minimiseurs peuvent ne pas exister
Remplacer chaque « min » par « inf » dans les définitions et énoncer que, pour tout $\varepsilon>0$, il existe des règles de décision $\varepsilon$-optimales $\boldsymbol{\pi}_t^\varepsilon(\cdot)$ dont le coût est à moins de $\varepsilon$ de $J_t^\star(\cdot)$. Le même argument de couper-coller et la même induction s'appliquent.
``` -->



(example-seasonal-operation-of-a-hydroelectric-reservoir)=
### Exemple : exploitation saisonnière d'un réservoir hydroélectrique

Sur une rivière nordique, la majeure partie de l'eau arrive en quelques
semaines au printemps, lorsque le manteau neigeux fond. C'est la *crue
printanière*. L'électricité vaut le plus cher en hiver, quand la demande de
chauffage culmine. Un réservoir derrière un barrage déplace l'eau d'une saison
à l'autre. Chaque semaine, l'exploitant décide de la quantité d'eau à faire
passer dans les turbines. L'eau lâchée maintenant est vendue au prix de cette
semaine. L'eau gardée dans le réservoir peut être vendue plus tard, et elle
élève aussi le niveau de l'eau, de sorte que chaque mètre cube lâché par la
suite tombe de plus haut et produit plus d'énergie. Garder l'eau comporte aussi
un risque. Lorsque la crue printanière atteint un réservoir plein, l'excédent
passe par l'évacuateur de crues et ne rapporte rien.

La [cascade hydroélectrique](numerical-trajectory-optimization.md) du chapitre
sur l'optimisation de trajectoires calculait un seul plan à partir d'un seul
état initial. Un exploitant a besoin de plus que cela. Les apports ne suivent
jamais exactement la prévision; le réservoir est donc rarement au niveau
supposé par le plan, et l'exploitant a besoin d'un débit lâché pour n'importe
quel niveau. La programmation dynamique le fournit : une seule passe arrière
donne un débit lâché pour chaque semaine et chaque niveau de stockage. Les
exploitants de réservoirs appellent les courbes obtenues des *courbes guides*,
et la programmation dynamique sert à les calculer depuis des décennies
{cite:p}`Yeh1985`.

**Modèle.** L'étape est d'une semaine, avec $T=52$ semaines à partir de
janvier. Les volumes sont mesurés en hectomètres cubes
($1\,\mathrm{hm}^3 = 10^6\,\mathrm{m}^3$).

| Symbole | Signification | Valeur |
|---|---|---|
| $x_t$ | stockage au début de la semaine $t$ (état) | $x_{\min} \le x_t \le \bar x$ |
| $u_t$ | eau lâchée dans les turbines (commande) | $u_{\min} \le u_t \le u_{\max}$ |
| $w_t$ | prévision des apports pour la semaine $t$ | 60 à 400 hm³/semaine |
| $p_t$ | prix de l'électricité pendant la semaine $t$ | 30 à 90 \$/MWh |
| $x_{\min},\ \bar x$ | réserve morte (sous la prise d'eau) et capacité | 200 et 2000 hm³ |
| $u_{\min},\ u_{\max}$ | débit écologique minimal et capacité des turbines | 30 et 250 hm³/semaine |
| $h(x)$ | hauteur de chute : hauteur de laquelle l'eau tombe | 40 à 80 m |
| $\kappa$ | énergie par hm³ et par mètre de chute | 2.45 MWh |

Les apports et les prix sont des courbes stylisées, et non des mesures : un
débit de base hivernal de 100 m³/s, une crue printanière qui culmine à la
mi-mai à environ quatre fois le débit moyen, une hausse automnale due à la
pluie, et un prix maximal au début de février et minimal en août. L'eau
au-delà de la capacité est déversée; le stockage évolue donc selon

$$
x_{t+1} = \min\{\,x_t + w_t - u_t,\ \bar x\,\},
\qquad \text{sous la contrainte } x_t + w_t - u_t \ge x_{\min}.
$$

Lâcher $u$ avec une hauteur de chute $h(x)$ produit $\kappa\,h(x)\,u$
mégawattheures, où $\kappa = \rho g \eta = 1000 \cdot 9.81 \cdot 0.9$ joules
par mètre cube et par mètre, converti en MWh par hm³. La hauteur de chute croît
avec le stockage, et c'est ce qui donne à l'eau gardée une valeur au-delà de
son prix de vente ultérieur. L'eau restante à la fin de l'année est créditée à
$p_T = 50$ \$/MWh, de sorte que la valeur terminale est
$J_T^\star(x) = p_T\,\kappa\,h(x)\,(x - x_{\min})$. Comme l'objectif est un
revenu à maximiser plutôt qu'un coût, l'équation de Bellman prend un maximum :

$$
J_t^\star(x) = \max_{u_{\min} \le u \le u_{\max}}
\Big[\, p_t\,\kappa\,h(x)\,u \;+\; J_{t+1}^\star\big(\min\{x + w_t - u,\ \bar x\}\big) \Big],
\qquad x + w_t - u \ge x_{\min}.
$$

Appelons l'expression à l'intérieur du maximum

$$
Q_t(x, u) = p_t\,\kappa\,h(x)\,u \;+\; J_{t+1}^\star\big(\min\{x + w_t - u,\ \bar x\}\big),
$$

le revenu total à partir du stockage $x$ pendant la semaine $t$ si l'on lâche
$u$ maintenant et qu'on agit de façon optimale par la suite. Elle met en
balance le revenu obtenu en lâchant $u$ cette semaine et le revenu encore
réalisable avec le stockage qui reste, et l'équation de Bellman s'écrit
$J_t^\star(x) = \max_u Q_t(x, u)$.

**Code.** La grille de stockage a un pas de 10 hm³, et les apports sont
arrondis à des multiples de 10 hm³. Chaque successeur $x + w_t - u$ est donc un
point de la grille, et l'{prf:ref}`backward-recursion-deterministic`
s'applique telle quelle : $J_{t+1}^\star$ est consulté dans la table, jamais
interpolé. La section suivante retire cette commodité. La boucle de
l'algorithme sur les états et les actions est écrite avec la diffusion
(*broadcasting*) de NumPy : `x` est une colonne de tous les niveaux de stockage
et `u` une ligne de tous les débits lâchés, de sorte que chaque ligne évalue
toutes les paires (stockage, débit lâché) à la fois.

```{code-cell} python
import numpy as np

# Calendar: 52 weekly decisions; week 0 is the first week of January.
T = 52
weeks = np.arange(T)

# Inflow forecast w_t (hm³ per week): winter base flow, spring freshet, autumn rain.
w = 60 + 340 * np.exp(-((weeks - 19) / 3.5) ** 2) + 60 * np.exp(-((weeks - 43) / 4) ** 2)
w = 10 * np.round(w / 10)            # multiples of the grid step

# Price p_t ($/MWh): winter peak, summer trough.
p = 60 + 30 * np.cos(2 * np.pi * (weeks - 5) / 52)

# Reservoir and plant.
x_min, x_max = 200, 2000             # dead storage and capacity (hm³)
u_min, u_max = 30, 250               # ecological minimum and turbine capacity (hm³/week)
kappa = 2.45                         # MWh per hm³ per metre of head
p_T = 50                             # $/MWh credited to water left at the end of the year

def head(x):                         # metres of head; grows with storage
    return 40 + 40 * np.sqrt(x / x_max)

def terminal_value(x):
    return p_T * kappa * head(x) * (x - x_min)

dx = 10
X = np.arange(x_min, x_max + dx, dx)   # storage grid: 181 levels
U = np.arange(u_min, u_max + dx, dx)   # release grid: 23 levels

def index(x):                          # position of an on-grid storage level
    return np.round((x - x_min) / dx).astype(int)

J = np.zeros((T + 1, len(X)))
policy = np.zeros((T, len(X)))
J[T] = terminal_value(X)

for t in reversed(range(T)):
    x, u = X[:, None], U[None, :]                  # all (storage, release) pairs
    x_next = np.minimum(x + w[t] - u, x_max)       # water above capacity spills
    feasible = x + w[t] - u >= x_min               # cannot draw below dead storage
    value = p[t] * kappa * head(x) * u + J[t + 1][index(np.maximum(x_next, x_min))]
    value = np.where(feasible, value, -np.inf)
    J[t] = value.max(axis=1)
    policy[t] = U[value.argmax(axis=1)]

def simulate(x0):
    storage = [x0]
    for t in range(T):
        u = policy[t, index(storage[-1])]
        storage.append(min(storage[-1] + w[t] - u, x_max))
    return np.array(storage)

print(f"optimal revenue from x0 = 1100 hm³: {J[0, index(1100)] / 1e6:.1f} M$")
```

La passe arrière évalue $52 \times 181 \times 23 \approx 2.2 \times 10^5$
triplets (semaine, stockage, débit lâché) et prend une fraction de seconde. Sa
sortie n'est pas une trajectoire, mais deux tables, $J_t^\star(x)$ et
$u_t^\star(x)$, avec une entrée pour chaque semaine et chaque niveau de
stockage.

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))

from hydro_reservoir import (
    format_comparison,
    make_interpolation_figure,
    make_policy_figure,
    make_threshold_figure,
    simulate_rule,
)

water_value = np.gradient(J, dx, axis=1)       # dJ_t/dx on the grid, $ per hm³
rollouts = [simulate(x0) for x0 in (300, 700, 1100, 1500, 1900)]
reservoir = dict(w=w, p=p, head=head, terminal_value=terminal_value, x_min=x_min,
                 x_max=x_max, u_min=u_min, u_max=u_max, kappa=kappa)
```

```{code-cell} python
:label: fig-hydro-policy
:caption: Sortie d'une passe arrière. À gauche, le débit lâché optimal $u_t^\star(x)$ pour chaque semaine et chaque niveau de stockage, sous la prévision des apports. À droite, la valeur de l'eau $\partial J_t^\star/\partial x$, le revenu qu'ajouterait un hm³ de plus en réserve, sous le prix. Les courbes blanches sont les trajectoires de stockage obtenues en suivant la politique à partir de cinq niveaux initiaux; après quelques semaines, elles se confondent.
:tags: [remove-input]

make_policy_figure(w, p, X, policy, water_value[:T], rollouts)
```

La carte de gauche est la politique. Elle comporte deux régions : lâcher à la
capacité des turbines ou lâcher le minimum écologique, avec une mince frontière
entre les deux. Les trajectoires de stockage blanches montrent la politique
telle qu'un exploitant la lirait. Quel que soit le niveau en janvier, le
réservoir est abaissé jusqu'à la réserve morte vers la fin mars, rempli par la
crue printanière puis plus lentement pendant l'été jusqu'à être plein à la
mi-septembre, maintenu plein jusqu'à la fin novembre, puis abaissé à la
capacité des turbines en décembre, lorsque les prix sont de nouveau élevés. Le
stockage initial n'influe que sur les premières semaines, car toutes les
trajectoires rejoignent la même courbe guide avant la crue. Un réservoir plus
plein en janvier signifie simplement plus d'eau à vendre pendant l'abaissement
hivernal.

La carte de droite est la dérivée de la fonction de valeur par rapport au
stockage, que les exploitants hydroélectriques appellent la *valeur de l'eau*.
Elle est la plus faible, proche de zéro, à stockage plein pendant la crue : le
réservoir ne peut pas contenir davantage, et un hm³ supplémentaire serait
déversé sans rien rapporter. Elle est la plus élevée en hiver à faible
stockage, lorsque chaque hm³ peut être vendu au prix hivernal et qu'aucun ne
risque d'être déversé.

**Pourquoi la politique est tout ou rien.** Fixons la semaine $t$ et le
stockage $x$, et regardons $Q_t(x, u)$ comme une fonction du seul débit lâché
$u$. Elle comporte deux termes, et $u$ entre une seule fois dans chacun.

Le premier terme, $p_t\,\kappa\,h(x)\,u$, est linéaire en $u$. La hauteur de
chute $h(x)$ dépend du stockage au début de la semaine, qui est fixé; chaque
hm³ supplémentaire lâché rapporte donc le même montant, $p_t\,\kappa\,h(x)$.

Le second terme ne dépend de $u$ que par le stockage de la semaine suivante
$x_{t+1} = x + w_t - u$. Lâcher un hm³ de plus abaisse $x_{t+1}$ d'un hm³;
par la dérivation des fonctions composées, ce terme varie donc au taux
$-\partial J_{t+1}^\star/\partial x$ évalué en $x_{t+1}$, soit moins la valeur
de l'eau. (Si le réservoir déborde, $x_{t+1} = \bar x$ ne dépend pas de $u$, et
ce taux est nul : le hm³ supplémentaire aurait de toute façon été déversé.)

L'addition des deux taux donne la pente de $Q_t$ en $u$ :

$$
\frac{\partial Q_t}{\partial u}(x, u)
\;=\; \underbrace{p_t\,\kappa\,h(x)}_{\text{valeur du lâcher d'un hm³}}
\;-\; \underbrace{\frac{\partial J_{t+1}^\star}{\partial x}(x + w_t - u)}_{\text{valeur de sa conservation}} .
$$

On maximise $Q_t$ sur l'intervalle $[u_{\min}, u_{\max}]$, et le signe de
cette pente indique où se trouve le maximum :

- Si la pente est positive pour tout $u$ de l'intervalle, $Q_t$ croît d'une
  extrémité à l'autre, et le maximum est en $u = u_{\max}$.
- Si elle est négative pour tout $u$, $Q_t$ décroît, et le maximum est en
  $u = u_{\min}$.
- Ce n'est que si la pente change de signe à l'intérieur de l'intervalle que le
  maximum peut se trouver strictement entre les bornes, en un débit lâché où
  les deux valeurs sont égales.

Le premier terme de la pente ne dépend pas du tout de $u$. Le second ne dépend
de $u$ que par la valeur de l'eau, et, sur les niveaux de stockage atteignables
en une semaine, la valeur de l'eau varie peu comparativement à l'écart qui la
sépare du prix. Pour la plupart des $(t, x)$, la pente garde donc un seul signe
sur tout l'intervalle, et le débit lâché est à une borne. C'est la carte à deux
régions de la {numref}`fig-hydro-policy`. Le cas intérieur est la mince
frontière entre les régions : la politique y lâche exactement assez d'eau pour
amener $x_{t+1}$ au niveau où la valeur de l'eau est égale au prix.

Deux limites physiques produisent aussi des débits lâchés entre $u_{\min}$ et
$u_{\max}$, et toutes deux apparaissent dans les figures. À faible stockage, la
contrainte $x + w_t - u \ge x_{\min}$ plafonne le débit lâché sous la capacité
des turbines, de sorte que « lâcher autant que possible » signifie vider le
réservoir jusqu'à la réserve morte. À stockage plein, la pente saute au débit
lâché $u = x + w_t - \bar x$ : en deçà, l'eau supplémentaire serait déversée,
le second terme est nul et la pente est positive; au-delà, la pente est le prix
moins la valeur de l'eau en $\bar x$. Lorsque celle-ci est négative, le
meilleur débit lâché est le coude lui-même, qui maintient le réservoir
exactement plein et lâche l'apport. C'est ce qui se produit de septembre à
novembre dans la {numref}`fig-hydro-threshold`.

La règle de décision est donc une comparaison de prix : **faire tourner les
turbines lorsque le prix dépasse la valeur de l'eau dans le réservoir**, les
deux étant exprimés par hm³. Dans les réseaux électriques à dominante
hydroélectrique, comme celui de la Norvège, les producteurs calculent les
valeurs de l'eau par programmation dynamique et s'en servent pour fixer les
prix auxquels ils offrent leur énergie.

La même structure apparaît en temps continu, où la pente s'appelle une
*fonction de commutation* et où une commande qui reste à ses bornes est dite
*bang-bang*. Chaque fois que la récompense est linéaire en une commande bornée,
l'optimum est à une borne, sauf là où la fonction de commutation est nulle, le
cas *singulier*.

```{code-cell} python
:label: fig-hydro-threshold
:caption: Les deux termes de la dérivée le long de la trajectoire optimale à partir de $x_0 = 1100$ hm³. Les turbines tournent à pleine capacité (semaines ombrées) lorsque le revenu tiré du lâcher immédiat d'un hm³ (orange) dépasse la valeur de sa conservation (bleu), et au minimum écologique lorsqu'il est inférieur. De septembre à novembre, le réservoir est plein et toute eau conservée serait déversée; le débit lâché est donc égal à l'apport, même si conserver l'eau paraît plus avantageux.
:tags: [remove-input]

x0 = 1100
comparison = {
    "dynamic programming": simulate_rule(lambda t, x: policy[t, index(x)], x0, **reservoir),
    "run of river (release = inflow)": simulate_rule(lambda t, x: w[t], x0, **reservoir),
    "full release when price > mean": simulate_rule(
        lambda t, x: u_max if p[t] > p.mean() else u_min, x0, **reservoir),
}
print(format_comparison(comparison))

optimal = comparison["dynamic programming"]
marginal_release = p * kappa * head(optimal["storage"][:-1])
marginal_storage = np.array([np.interp(optimal["storage"][t + 1], X, water_value[t + 1])
                             for t in range(T)])
make_threshold_figure(p, optimal["storage"], optimal["release"], marginal_release,
                      marginal_storage, u_min, u_max)
```

Le tableau au-dessus de la figure compare la politique à deux règles
d'exploitation plus simples sur la même année. L'exploitation au fil de l'eau,
qui lâche chaque semaine l'apport de la semaine, rapporte environ 10 % de moins,
car elle vend la majeure partie de l'eau pendant la crue printanière, lorsque
les prix sont bas. Lâcher au maximum chaque fois que le prix dépasse sa moyenne
annuelle rapporte encore moins et déverse 900 hm³, car cette règle ignore le
niveau de stockage. Elle vide le réservoir dès le début de février et n'a plus
que le faible apport hivernal à vendre pendant le reste de la saison des prix
élevés. En mai, lorsque le prix passe sous sa moyenne, elle passe au débit
minimal en pleine crue printanière. Le réservoir est plein vers la fin juin, et
de juillet à novembre, chaque hm³ au-delà du débit minimal est déversé. La
programmation dynamique tient compte à la fois du prix, de la hauteur de chute
et de la capacité restante, au moyen d'une seule fonction du stockage.

**Explorer la solution.** La page ci-dessous résout de nouveau la même
récurrence arrière dans votre navigateur chaque fois qu'un paramètre change.
*Passe arrière* anime la récurrence : les cartes se remplissent de la semaine 52
à la semaine 1, une colonne par mise à jour de Bellman, et le panneau à côté de
la carte montre la fonction de valeur de la colonne en cours de calcul. *Jouer
l'année* exploite ensuite le réservoir vers l'avant à partir du stockage
initial choisi. La page compare les deux termes de la dérivée par MWh plutôt
que par hm³ : diviser les deux par $\kappa\,h(x_t)$ transforme le premier en
prix $p_t$ et le second en valeur de l'eau en \$/MWh.

:::{iframe} ../interactive/hydro-reservoir.html
:label: fig-hydro-interactive
:width: 100%
:title: Réservoir hydroélectrique saisonnier résolu par récurrence arrière
:class: hydro-reservoir-replay

Le problème du réservoir résolu de nouveau dans le navigateur. La carte montre
le débit lâché, la valeur de l'eau ou la fonction de valeur sur (semaine,
stockage); le panneau latéral montre $J_t^\star(x) - J_t^\star(x_{\min})$ pour
la semaine courante, dont la pente est la valeur de l'eau. Les bandes comparent
le prix à la valeur de l'eau le long de l'année simulée et montrent l'apport, le
débit lâché et le déversement.
:::

> **Question.** Fixez à zéro le prix de l'eau restante à la fin de l'année.
> Que fait la politique en novembre et en décembre? Est-ce ce que devrait faire
> un exploitant qui exploitera encore le barrage l'année suivante?

:::{dropdown} Réponse
Elle vide le réservoir jusqu'à la réserve morte d'ici la fin de décembre, car
le modèle n'accorde aucune valeur à l'eau qui reste après la semaine 52. Un
exploitant réel commencerait janvier avec un réservoir vide et aucune eau à
vendre pendant les semaines de prix élevés. Cet effet de fin d'horizon est un
artefact de la troncature du problème à une année. La valeur terminale
$J_T^\star$ tient lieu des années suivantes; les formulations à horizon infini
des chapitres ultérieurs suppriment la nécessité de la choisir.
:::

> **Question.** À partir de 1100 hm³, la politique maintient le débit minimal
> pendant tout janvier, bien que le prix dépasse déjà 80 \$/MWh, et lâche à
> pleine capacité de la mi-février à la mi-mars. Pourquoi attend-elle?

:::{dropdown} Réponse
Le réservoir doit être presque vide avant le début de la crue printanière en
avril; la quantité d'eau à vendre d'ici là est donc fixée. Deux effets rendent
préférable de la vendre tard plutôt que tôt. Le prix continue de monter jusqu'au
début de février. Et l'eau lâchée plus tard tombe d'un réservoir plus plein :
la retenir maintient la hauteur de chute élevée, de sorte que chaque hm³ produit
plus d'énergie. Les turbines lâchent au plus 250 hm³ par semaine, soit environ
190 hm³ net de l'apport hivernal; vider le réservoir prend donc environ six
semaines, et la politique commence l'abaissement juste assez tôt pour le
terminer avant la crue.
:::

> **Question.** Supposons que la rivière comporte trois réservoirs semblables
> en série, chacun discrétisé avec les mêmes 181 niveaux de stockage et 23
> débits lâchés. Combien de combinaisons (stockage, débit lâché) une mise à jour
> de Bellman devrait-elle évaluer chaque semaine?

:::{dropdown} Réponse
L'état est le vecteur des trois stockages, avec $181^3 \approx 5.9 \times
10^6$ points de grille, et chacun a $23^3 = 12\,167$ débits lâchés conjoints,
soit environ $7 \times 10^{10}$ évaluations par semaine au lieu de $4\,163$.
C'est le fléau de la dimension évoqué au début du chapitre, et la raison pour
laquelle la cascade a été résolue par optimisation de trajectoires à partir
d'un seul état initial.
:::


(handling-continuous-spaces-with-interpolation)=
## Traitement des espaces continus par interpolation

Lorsqu'un successeur tombe entre les états stockés dans la table, quelle valeur
la récurrence doit-elle lui attribuer, et quel problème la récurrence
résout-elle alors?

Le {prf:ref}`thm-bolza-backward` n'impose aucune restriction à l'espace
d'états. Un programme, en revanche, doit stocker $J_{t+1}^\star$ sous la forme
d'un nombre fini de valeurs. Le code du réservoir stocke une valeur par tranche
de 10 hm³ de stockage, et il a évité la question par construction : ses apports
étaient arrondis à des multiples du pas de la grille, de sorte que chaque
successeur $x + w_t - u$ tombait sur un point de la grille. Les apports mesurés
ne sont des multiples de rien. Pendant une semaine où l'apport est de 63.7 hm³
et le débit lâché de 30 hm³, un réservoir à 1000 hm³ passe à 1033.7 hm³, entre
deux points de la grille, et l'équation de Bellman demande alors
$J_{t+1}^\star(1033.7)$, que la table ne contient pas. Choisir une valeur pour
cette quantité est un problème de représentation que le
[chapitre sur la collocation](continuous-time-collocation.md) a déjà traité
pour des fonctions du temps.

### Représentation nodale du coût restant

En collocation, un morceau de trajectoire était stocké par ses valeurs $y_j$ en
des nœuds d'appui $\sigma_j$ et reconstruit entre eux par des fonctions
cardinales, $p(\tau)=\sum_j y_j\ell_j(\tau)$ avec $\ell_j(\sigma_i)=\delta_{ij}$.
Une table de coût restant est le même genre d'objet, les nœuds étant placés
dans l'espace d'états plutôt que sur l'axe du temps. Choisissons des états de
grille $\mathbf{x}^{(1)},\dots,\mathbf{x}^{(N)}$ et notons $v_{t,i}$ la valeur
stockée pour $\mathbf{x}^{(i)}$ à l'étape $t$. Une famille de fonctions
cardinales sur la grille transforme ces nombres en une fonction de l'état :

$$
\hat J_t(\mathbf{x})=\sum_{i=1}^{N}v_{t,i}\,\ell_i(\mathbf{x}),
\qquad
\ell_i\big(\mathbf{x}^{(j)}\big)=\delta_{ij}.
$$ (eq:nodal-cost-to-go)

Le tableau `J` du code du réservoir contient les coordonnées nodales
$v_{t,i}$. Une règle d'interpolation est un choix de fonctions cardinales, que
le tableau seul ne précise pas.

Lorsque le chapitre sur la collocation
[a choisi comment représenter une commande](continuous-time-collocation.md#choosing-a-polynomial-for-the-control)
entre ses valeurs d'appui, il a énuméré trois options : une commande constante
sur chaque intervalle, une droite entre les commandes aux extrémités, ou un
polynôme de degré plus élevé. Les trois mêmes options s'offrent pour le coût
restant. Sur une grille unidimensionnelle $x^{(1)}<\dots<x^{(N)}$, comme les
niveaux de stockage du réservoir, ce sont :

1. **L'arrondi** (interpolation au plus proche voisin). La fonction cardinale
   $\ell_i$ vaut un sur les états plus proches de $x^{(i)}$ que de tout autre
   nœud et zéro ailleurs; $\hat J_t(x)$ est donc la valeur stockée au nœud le
   plus proche. Elle est constante sur chaque cellule, comme une commande
   bloquée d'ordre zéro, à ceci près que la cellule est centrée sur son nœud.
2. **L'interpolation linéaire.** Chaque $\ell_i$ est un « chapeau » qui monte
   de zéro en $x^{(i-1)}$ à un en $x^{(i)}$ et redescend à zéro en
   $x^{(i+1)}$. Entre deux nœuds adjacents, seuls deux chapeaux sont non nuls,
   et $\hat J_t(x)=(1-\lambda)\,v_{t,i}+\lambda\,v_{t,i+1}$ avec
   $\lambda=(x-x^{(i)})/(x^{(i+1)}-x^{(i)})\in[0,1]$. Ce sont les commandes
   continues affines par morceaux, à valeurs partagées aux extrémités, de la
   [transcription trapézoïdale](continuous-time-collocation.md#endpoint-slope-values-trapezoidal-transcription).
3. **La spline cubique.** Des morceaux cubiques sont raccordés avec des
   dérivées première et seconde continues. Chaque fonction cardinale est alors
   non nulle sur toute la grille, avec des lobes de signes alternés qui
   s'amortissent en s'éloignant de son nœud.

Dans les trois cas, les poids $\ell_i(x)$ ont une somme égale à un, car chaque
règle reproduit exactement une fonction constante. Ils diffèrent par leur
signe. L'arrondi et l'interpolation linéaire ont des poids positifs ou nuls;
$\hat J_t(x)$ est donc une combinaison convexe des valeurs stockées et reste
dans leur plage. Le chapitre sur la collocation a utilisé la même propriété
pour borner une commande affine par morceaux en bornant ses valeurs d'appui. La
spline a des poids négatifs, comme la base de Lagrange quadratique de
l'[exercice en classe](#ex-classroom-lagrange-quadratic), dont les poids en
$\tau=1/4$ valaient $(3/8,3/4,-1/8)$. Elle peut donc sortir de la plage des
valeurs stockées, comme le montre la rangée du bas de la
{numref}`fig-dp-cardinal-functions` sur des données qui présentent un saut. En
contrepartie, elle est bien plus précise que l'interpolation linéaire sur les
fonctions lisses.

```{code-cell} python
:label: fig-dp-cardinal-functions
:caption: Trois règles d'interpolation sur dix états de grille. Rangée du haut, la fonction cardinale d'un nœud (bleu) et de ses deux voisins (gris); chacune vaut un à son propre nœud et zéro à tous les autres nœuds. Rangée du bas, les mêmes valeurs stockées reconstruites par chaque règle. Les valeurs sautent de zéro à un entre deux nœuds, comme le fait une table de débits lâchés tout ou rien à son stockage de commutation. La bande ombrée est la plage des valeurs stockées. L'arrondi et l'interpolation linéaire y restent, car leurs poids sont positifs ou nuls; les lobes négatifs des fonctions cardinales de la spline la font dépasser des deux côtés du saut.
:tags: [remove-input]

from finite_horizon_dp_figures import make_cardinal_functions_figure

make_cardinal_functions_figure()
```

Substituer {eq}`eq:nodal-cost-to-go` dans l'équation de Bellman et ne
l'évaluer qu'aux états de la grille donne une récurrence sur $N$ nombres par
étape :

$$
v_{T,i}=c_\mathrm{T}\big(\mathbf{x}^{(i)}\big),
\qquad
v_{t,i}=\min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x}^{(i)})}
\Big[c_t\big(\mathbf{x}^{(i)},\mathbf{u}\big)
+\sum_{j=1}^{N}\ell_j\big(\mathbf{f}_t(\mathbf{x}^{(i)},\mathbf{u})\big)\,v_{t+1,j}\Big],
\qquad i=1,\dots,N.
$$ (eq:interpolated-bellman)

Par rapport à l'{prf:ref}`backward-recursion-deterministic`, la boucle sur les
états devient une boucle sur les états de la grille, et la consultation de
$J_{t+1}^\star$ au successeur devient une évaluation de $\hat J_{t+1}$. Chaque
étape se compose encore de problèmes à un pas indépendants, un par état de
grille. Pour le réservoir, qui maximise un revenu, le minimum devient un
maximum. Les nombres $v_{t,i}$ sont des approximations de
$J_t^\star(\mathbf{x}^{(i)})$, et les fonctions cardinales déterminent leur
qualité.

````{prf:algorithm} Récurrence arrière avec interpolation
:label: backward-recursion-interp

**Entrée :** coût terminal $c_\mathrm{T}$, coûts d'étape $c_t$, dynamique
$\mathbf{f}_t$, horizon $T$, états de grille $\mathbf{x}^{(1)},\dots,\mathbf{x}^{(N)}$,
fonctions cardinales $\ell_1,\dots,\ell_N$

**Sortie :** valeurs nodales $v_{t,i}$ pour $t=1,\ldots,T$ et décisions nodales
$\boldsymbol{\pi}_{t,i}$ pour $t=1,\ldots,T-1$

1. Poser $v_{T,i}=c_\mathrm{T}(\mathbf{x}^{(i)})$ pour $i=1,\ldots,N$
2. Pour $t=T-1,T-2,\ldots,1$ :
   1. Pour chaque état de grille $\mathbf{x}^{(i)}$ :
      1. Pour chaque candidat $\mathbf{u}$, calculer le successeur
         $\mathbf{y}=\mathbf{f}_t(\mathbf{x}^{(i)},\mathbf{u})$ et sa valeur
         interpolée $\hat J_{t+1}(\mathbf{y})=\sum_j \ell_j(\mathbf{y})\,v_{t+1,j}$
      2. Poser $v_{t,i}=\min_{\mathbf{u}}\big[c_t(\mathbf{x}^{(i)},\mathbf{u})+\hat J_{t+1}(\mathbf{f}_t(\mathbf{x}^{(i)},\mathbf{u}))\big]$
         et prendre pour $\boldsymbol{\pi}_{t,i}$ un minimiseur
   2. Fin pour
3. Fin pour
4. Renvoyer $\{v_{t,i}\}$ et $\{\boldsymbol{\pi}_{t,i}\}$
````

### Modèles de grille sous-jacents à la récurrence interpolée

Les poids $\ell_j(\mathbf{f}_t(\mathbf{x}^{(i)},\mathbf{u}))$ de
{eq}`eq:interpolated-bellman` ne dépendent que de l'endroit où le successeur
tombe par rapport à la grille. Lus comme une description de l'endroit où va
ensuite le système, ils montrent quel problème la récurrence interpolée résout.

Avec l'arrondi, exactement un poids vaut un, celui du nœud le plus proche du
successeur. La récurrence est alors la récurrence arrière du
{prf:ref}`thm-bolza-backward` pour un autre système déterministe, qui vit sur
la grille et se déplace vers le nœud le plus proche de
$\mathbf{f}_t(\mathbf{x},\mathbf{u})$. Le théorème s'applique sans changement à
ce système arrondi; la récurrence renvoie donc ses valeurs et ses décisions
optimales. L'erreur d'approximation réside entièrement dans le remplacement du
modèle, et ce remplacement peut être qualitativement faux. Tout changement
inférieur à un demi-pas de grille disparaît dans le système arrondi. Dans
l'exemple ci-dessous, le gain hebdomadaire de stockage en été est un tel
changement, et le réservoir arrondi ne peut pas se remplir.

Avec l'interpolation linéaire, le poids du successeur est réparti entre les
deux nœuds qui l'encadrent, sous la forme $1-\lambda$ et $\lambda$. Ces nombres
sont positifs ou nuls et ont une somme égale à un; on peut donc les lire comme
des probabilités. Le crochet de {eq}`eq:interpolated-bellman` devient le coût
d'étape plus l'espérance de la valeur stockée en un nœud suivant aléatoire, et
la récurrence interpolée est la récurrence arrière exacte d'un système à états
finis dont les successeurs sont aléatoires : à partir de $x^{(i)}$ sous $u$, le
système passe en $x^{(j)}$ avec probabilité $\ell_j(f_t(x^{(i)},u))$. Pour le
réservoir, un successeur de 1033.7 hm³ sur la grille de 10 hm³ devient 1030 hm³
avec probabilité $0.63$ et 1040 hm³ avec probabilité $0.37$. Les fonctions
chapeaux reproduisent les fonctions linéaires,
$\sum_j\ell_j(x)\,x^{(j)}=x$, de sorte que le nœud suivant aléatoire est égal
en moyenne au vrai successeur. L'interpolation linéaire remplace donc le modèle
déterministe par une chaîne de Markov sur la grille. Faire correspondre le
mouvement moyen d'une telle chaîne à la dynamique continue est la condition de
cohérence locale à la base de la méthode d'approximation par chaînes de Markov
de {cite:t}`KushnerDupuis2001` pour la commande stochastique en temps continu.
En $n$ dimensions, l'interpolation multilinéaire sur une grille rectangulaire
multiplie les poids chapeaux unidimensionnels. Ses $2^n$ poids aux coins de la
cellule englobante sont encore positifs ou nuls et ont une somme égale à un; la
même lecture s'applique donc. Le [chapitre suivant](stochastic-dp.md) rend
l'apport lui-même aléatoire et interpole linéairement chacun de ses
successeurs. Les probabilités des apports et les poids d'interpolation se
multiplient alors, et le résultat est encore une loi sur les états de la
grille.

Une spline cubique place des poids négatifs sur certains nœuds, et des nombres
négatifs ne peuvent pas être lus comme des probabilités. Aucun problème à
états finis n'a {eq}`eq:interpolated-bellman` pour récurrence exacte. La valeur
interpolée $\hat J_{t+1}$ peut aussi sortir de la plage des valeurs stockées,
et la minimisation sur $\mathbf{u}$ est libre de s'orienter vers les
successeurs où cela se produit.

### Propagation de l'erreur d'une étape à l'autre

Chaque étape de {eq}`eq:interpolated-bellman` utilise des valeurs stockées qui
contiennent déjà l'erreur des étapes ultérieures. Le signe des poids décide si
cette erreur est transmise telle quelle ou amplifiée. Pour préciser cela,
définissons l'erreur des valeurs nodales et l'erreur d'interpolation du coût
restant exact,

$$
e_t=\max_i\big|v_{t,i}-J_t^\star\big(\mathbf{x}^{(i)}\big)\big|,
\qquad
\delta_t=\sup_{\mathbf{y}}\Big|\sum_j\ell_j(\mathbf{y})\,J_t^\star\big(\mathbf{x}^{(j)}\big)-J_t^\star(\mathbf{y})\Big|,
$$

où la borne supérieure porte sur les successeurs atteints à partir de la
grille. La constante $\Lambda=\sup_{\mathbf{y}}\sum_j|\ell_j(\mathbf{y})|$
borne l'amplification possible, par l'interpolant, des erreurs sur les valeurs
stockées. Comme les poids ont une somme égale à un, $\Lambda\ge 1$, avec
égalité exactement lorsqu'ils sont positifs ou nuls.

```{prf:proposition} Accumulation de l'erreur dans la récurrence arrière interpolée
:label: prop-interp-error-accumulation

Sous les hypothèses du {prf:ref}`thm-bolza-backward`, les valeurs nodales
calculées par {eq}`eq:interpolated-bellman` satisfont $e_T=0$ et

$$
e_t\le\Lambda\,e_{t+1}+\delta_{t+1},\qquad t=T-1,\ldots,1.
$$

Avec des poids positifs ou nuls, $\Lambda=1$ et
$e_1\le\delta_2+\delta_3+\cdots+\delta_T$.
```

```{prf:proof}
Le cas de base est vrai, car $v_{T,i}=c_\mathrm{T}(\mathbf{x}^{(i)})=J_T^\star(\mathbf{x}^{(i)})$.
Pour $t<T$, fixons un état de grille $\mathbf{x}^{(i)}$ et une commande
$\mathbf{u}$, et posons $\mathbf{y}=\mathbf{f}_t(\mathbf{x}^{(i)},\mathbf{u})$.
Le crochet de {eq}`eq:interpolated-bellman` et celui de l'équation de Bellman
ont le même coût d'étape; ils diffèrent donc de

$$
\sum_j\ell_j(\mathbf{y})\big(v_{t+1,j}-J_{t+1}^\star(\mathbf{x}^{(j)})\big)
+\Big(\sum_j\ell_j(\mathbf{y})\,J_{t+1}^\star(\mathbf{x}^{(j)})-J_{t+1}^\star(\mathbf{y})\Big).
$$

Le premier terme vaut au plus $\sum_j|\ell_j(\mathbf{y})|\,e_{t+1}\le\Lambda e_{t+1}$
en valeur absolue, et le second au plus $\delta_{t+1}$. Les nombres $v_{t,i}$
et $J_t^\star(\mathbf{x}^{(i)})$ sont les minimums des deux crochets sur les
mêmes commandes admissibles. Si deux fonctions $a$ et $b$ diffèrent d'au plus
$\varepsilon$ partout, alors $\min a\le a(\mathbf{u}_b)\le b(\mathbf{u}_b)+\varepsilon=\min b+\varepsilon$,
où $\mathbf{u}_b$ minimise $b$, et symétriquement. Par conséquent,
$|v_{t,i}-J_t^\star(\mathbf{x}^{(i)})|\le\Lambda e_{t+1}+\delta_{t+1}$ pour tout
$i$. Dérouler l'inégalité avec $\Lambda=1$ donne la somme.
```

Avec des poids positifs ou nuls, chaque étape ajoute sa propre erreur
d'interpolation et aucune erreur antérieure n'est amplifiée; l'erreur croît
donc au plus linéairement avec l'horizon. Avec des poids négatifs, la borne
permet que chaque erreur antérieure soit multipliée par $\Lambda>1$ à chaque
étape subséquente. La taille de $\delta_t$ dépend de la règle et de
l'espacement $\Delta$ de la grille. Pour un coût restant à dérivées bornées sur
une grille unidimensionnelle, l'arrondi donne
$\delta\le\tfrac{\Delta}{2}\max|J'|$ et l'interpolation linéaire donne
$\delta\le\tfrac{\Delta^2}{8}\max|J''|$; diviser l'espacement par deux divise
donc la première borne par deux et la seconde par quatre. Une spline cubique
converge encore plus vite sur les fonctions lisses, ce qui explique qu'elle
puisse être la plus précise des trois en pratique, même si son $\Lambda$
dépasse un.

La preuve compare les deux minimums au moyen du plus grand écart entre leurs
crochets, et non d'un écart moyen, et l'exemple ci-dessous montre pourquoi
c'est la bonne comparaison. Une erreur d'interpolation qui paraît sans biais
d'une commande à l'autre devient biaisée dès qu'une commande est sélectionnée,
car la minimisation privilégie les commandes dont les successeurs se trouvent
évalués favorablement. Sur la grille grossière du réservoir, l'arrondi déplace
les successeurs réalisables de $-0.3$ hm³ en moyenne, sur l'ensemble des
semaines, des états de grille et des débits lâchés. Les successeurs que
sélectionne la maximisation sont déplacés de $+45$ hm³ en moyenne, sur un
maximum possible de $50$, et $94\%$ d'entre eux sont arrondis vers le haut. Le
même effet de sélection réapparaît sous la forme du biais de surestimation dans
l'[itération Q ajustée](fitted-q-iteration.md#sec-fqi-double-dqn).

Ces bornes ne suppriment pas le fléau de la dimension évoqué au début de ce
chapitre. Une grille à $N$ niveaux par coordonnée compte toujours $N^n$ nœuds,
chacun avec son propre problème à un pas, et l'interpolation multilinéaire fait
intervenir $2^n$ valeurs stockées par évaluation. Une règle plus précise réduit
le nombre de niveaux nécessaires pour une erreur donnée, mais non la dépendance
exponentielle en $n$. Les [méthodes de valeur approchée](approximate-bellman-equations.md)
remplacent la grille par des approximateurs paramétriques et font face à la
même question au sujet des poids. L'interpolation à poids positifs ou nuls de
somme égale à un est un {prf:ref}`moyenneur <gordon-averager>` au sens de
Gordon et maintient la stabilité de l'itération sur les valeurs, tandis que les
ajustements par moindres carrés et les réseaux de neurones n'offrent aucune
garantie de ce genre.

### Décisions entre les points de la grille

La récurrence ne renvoie des décisions qu'aux états de la grille, mais le
réservoir sera mesuré entre eux. Une option consiste à interpoler la table des
décisions $\boldsymbol{\pi}_{t,i}$. La table des débits lâchés du réservoir est
tout ou rien, comme le montre la {numref}`fig-hydro-policy`; chaque semaine,
elle saute donc du minimum écologique à la capacité des turbines à un stockage
de commutation. L'interpolation linéaire de cette table reste dans les limites
de débit lâché, mais renvoie des débits intermédiaires que la récurrence n'a
jamais évalués. Une spline cubique qui passe par cette table dépasse au saut,
comme dans la rangée du bas de la {numref}`fig-dp-cardinal-functions`, et
commande des débits lâchés inférieurs au minimum écologique et supérieurs à la
capacité des turbines.

L'autre option garde la décision à l'intérieur d'une optimisation. À l'état
mesuré, on résout le problème à un pas en laissant l'interpolant tenir lieu de
futur :

$$
\hat{\boldsymbol{\pi}}_t(\mathbf{x})\in\arg\min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x})}
\Big[c_t(\mathbf{x},\mathbf{u})+\hat J_{t+1}\big(\mathbf{f}_t(\mathbf{x},\mathbf{u})\big)\Big].
$$ (eq:one-step-lookahead)

À un état de la grille, cette règle reproduit la décision stockée. Entre les
états de la grille, elle renvoie une commande admissible, cohérente avec les
valeurs calculées par la récurrence. La règle {eq}`eq:one-step-lookahead` est
un correcteur à horizon glissant dont l'horizon de prédiction est d'un pas et
dont le coût terminal est $\hat J_{t+1}$. Dans le
[chapitre sur la commande à horizon glissant](receding-horizon-control.md), le
coût terminal tenait lieu du coût restant au-delà de l'horizon de prédiction et
devait être fourni par le concepteur. Ici, c'est la récurrence arrière
interpolée qui le calcule. Cela précise la réponse à la question qui ouvrait le
chapitre. Ce qui peut être calculé à l'avance pour toute une famille d'états,
c'est le coût restant, stocké aux états de la grille. La décision en un état
particulier provient encore d'une minimisation en ligne, désormais sur un seul
pas. Un horizon en ligne plus long rend la décision moins dépendante de
$\hat J$, au prix d'un problème en ligne plus grand.

### Exemple : exploitation du réservoir avec des apports non arrondis

Revenons au réservoir en laissant la prévision des apports non arrondie, de
sorte que les successeurs tombent entre les points de la grille, et
résolvons-le sur une grille grossière de 19 niveaux de stockage espacés de
100 hm³. Deux versions de l'{prf:ref}`backward-recursion-interp` ne diffèrent
que par la règle d'interpolation qui leur est fournie : plus proche voisin ou
linéaire (`np.interp`). Une solution de référence utilise l'interpolation
linéaire sur une grille vingt fois plus fine. Chaque fonction de valeur est
ensuite utilisée comme le ferait un exploitant : le réservoir est simulé avec
la dynamique exacte, non arrondie, et le débit lâché de chaque semaine est
choisi par l'anticipation à un pas {eq}`eq:one-step-lookahead` avec la valeur
interpolée $\hat J_{t+1}$.

```{code-cell} python
# Unrounded inflows: successors now fall between grid points.
w_raw = 60 + 340 * np.exp(-((weeks - 19) / 3.5) ** 2) + 60 * np.exp(-((weeks - 43) / 4) ** 2)
U_fine = np.linspace(u_min, u_max, 45)

def nearest(x_query, grid, values):
    step = grid[1] - grid[0]
    i = np.clip(np.round((x_query - grid[0]) / step).astype(int), 0, len(grid) - 1)
    return values[i]

def linear(x_query, grid, values):
    return np.interp(x_query, grid, values)

def backward_recursion(grid, interpolate):
    J = np.zeros((T + 1, len(grid)))
    J[T] = terminal_value(grid)
    for t in reversed(range(T)):
        x, u = grid[:, None], U_fine[None, :]
        x_next = np.clip(x + w_raw[t] - u, x_min, x_max)
        value = p[t] * kappa * head(x) * u + interpolate(x_next, grid, J[t + 1])
        J[t] = np.where(x + w_raw[t] - u >= x_min, value, -np.inf).max(axis=1)
    return J

def operate(grid, J, interpolate, x0):
    """Simulate the true dynamics, choosing each release by one Bellman step."""
    storage, revenue = [x0], 0.0
    for t in range(T):
        x = storage[-1]
        x_next = np.clip(x + w_raw[t] - U_fine, x_min, x_max)
        value = p[t] * kappa * head(x) * U_fine + interpolate(x_next, grid, J[t + 1])
        u = U_fine[np.argmax(np.where(x + w_raw[t] - U_fine >= x_min, value, -np.inf))]
        revenue += p[t] * kappa * head(x) * u
        storage.append(min(x + w_raw[t] - u, x_max))
    return np.array(storage), revenue + terminal_value(storage[-1])

coarse = np.arange(x_min, x_max + 100, 100)   # 19 levels
fine = np.arange(x_min, x_max + 5, 5)         # 361 levels
schemes = {
    "fine grid, linear (reference)": (fine, linear),
    "coarse grid, nearest neighbor": (coarse, nearest),
    "coarse grid, linear": (coarse, linear),
}
outcomes = {}
for name, (grid, interpolate) in schemes.items():
    J_scheme = backward_recursion(grid, interpolate)
    storage, revenue = operate(grid, J_scheme, interpolate, 1100)
    outcomes[name] = (grid, J_scheme[0], storage)
    print(f"{name:<32} predicted {linear(1100, grid, J_scheme[0]) / 1e6:6.2f} M$"
          f"   realized {revenue / 1e6:6.2f} M$")
```

```{code-cell} python
:label: fig-hydro-interpolation
:caption: Solutions sur grille grossière du problème du réservoir avec des apports non arrondis. À gauche, la fonction de valeur au début de l'année. L'arrondi au plus proche voisin (pointillés) prédit un revenu bien supérieur à la référence; l'interpolation linéaire sur les mêmes 19 points (bleu) se superpose à la référence (gris). À droite, le stockage lorsque chaque fonction de valeur sert à exploiter le réservoir à partir de 1100 hm³. La politique au plus proche voisin ne remplit jamais le réservoir en été.
:tags: [remove-input]

make_interpolation_figure(T, outcomes, "fine grid, linear (reference)")
```

L'interpolation linéaire sur 19 points prédit un revenu à moins de 0.2 % de la
référence à 361 points et en réalise un à moins de 0.03 %. L'arrondi sur la
même grille échoue des deux façons annoncées plus haut, et les deux découlent
de la taille d'une variation hebdomadaire typique. Son modèle de grille est
faux en été. Le réservoir gagne environ 30 hm³ par semaine au débit lâché
minimal, moins de la moitié du pas de la grille; le successeur arrondi est donc
égal au point de grille courant. Dans le modèle arrondi, stocker de l'eau ne
rapporte rien; la politique lâche donc l'apport estival aux prix de l'été au
lieu de le garder pour l'hiver, et le réservoir ne se remplit jamais. Sa
fonction de valeur est aussi trop optimiste, d'environ un tiers, car la
maximisation sélectionne les débits lâchés dont les successeurs sont arrondis
vers le haut, ce qui ajoute de l'eau qui n'existe pas. L'interpolation linéaire
conserve la partie fractionnaire de chaque pas, de sorte que les petites
variations s'accumulent correctement sur l'année.

> **Question.** Remplacez `linear` par une spline cubique,
> `lambda x_query, grid, values: CubicSpline(grid, values)(x_query)` avec
> `CubicSpline` tiré de `scipy.interpolate`, et relancez la récurrence sur la
> grille grossière. La fonction de valeur est-elle plus ou moins précise
> qu'avec l'interpolation linéaire? La réponse contredit-elle la
> {prf:ref}`prop-interp-error-accumulation`?

:::{dropdown} Réponse
Elle est plus précise. Sur les 19 états de la grille, la plus grande erreur sur
$v_0$ passe de 0.17 M\$ avec l'interpolation linéaire à 0.02 M\$, et la spline
prédit et réalise 74.16 M\$ à partir de 1100 hm³, comme la référence. La
proposition n'est pas contredite. Elle borne l'erreur par des erreurs
d'interpolation $\delta_t$ qui sont beaucoup plus petites pour la spline sur
cette fonction de valeur, et son facteur $\Lambda>1$ permet l'amplification
des erreurs antérieures sans l'imposer. Les poids négatifs causent des
difficultés là où la fonction interpolée présente un saut, ce qu'examine la
question suivante.
:::

> **Question.** Au lieu de résoudre le problème à un pas au stockage mesuré, un
> exploitant interpole la table des débits lâchés de la grille grossière entre
> les états de la grille. Enregistrez le débit lâché maximisant dans
> `backward_recursion` pour la règle linéaire, puis exploitez le réservoir à
> partir de 1100 hm³ en interpolant cette table, d'abord linéairement, puis
> avec une spline cubique. Qu'est-ce qui ne va pas dans chaque cas?

:::{dropdown} Réponse
La table est tout ou rien; chaque semaine où elle commute, elle saute donc de
30 à 250 hm³/semaine entre deux états de la grille. Une spline qui passe par
ces sauts les dépasse, comme dans la rangée du bas de la
{numref}`fig-dp-cardinal-functions`. Au fil de l'année, elle commande des
débits lâchés allant de 22 hm³/semaine, sous le minimum écologique, à
262 hm³/semaine, au-dessus de la capacité des turbines, et, sur l'ensemble des
semaines et des niveaux de stockage, sa plage va de $-38$ à 279 hm³/semaine.
L'interpolation linéaire de la table reste dans les limites, car ses poids sont
positifs ou nuls. Ses débits lâchés intermédiaires sont admissibles, mais n'ont
jamais été évalués par la récurrence, et elle réalise 74.05 M\$, contre
74.15 M\$ pour l'anticipation à un pas avec la même fonction de valeur.
:::


(sec-lqr-dp)=
## Commande linéaire quadratique par programmation dynamique

Quelle forme prennent la fonction de valeur et la loi de rétroaction lorsque la
dynamique est linéaire et que tous les coûts sont quadratiques?

Une dynamique linéaire et des coûts quadratiques donnent une récurrence arrière qui se résout sous forme close. La fonction de valeur reste quadratique à chaque étape, et la politique optimale est une loi de rétroaction linéaire. Aucune grille d'états, aucun schéma d'interpolation ni aucun approximateur de fonctions général n'est nécessaire. La récurrence suit une suite finie de matrices.

Considérons un système linéaire en temps discret :

$$
\mathbf{x}_{t+1} = A_t\mathbf{x}_t + B_t\mathbf{u}_t
$$

où $\mathbf{x}_t \in \mathbb{R}^n$ est l'état et $\mathbf{u}_t \in \mathbb{R}^m$ l'entrée de commande. Les matrices $A_t \in \mathbb{R}^{n \times n}$ et $B_t \in \mathbb{R}^{n \times m}$ décrivent la dynamique du système au temps $t$.

La fonction de coût à minimiser est quadratique :

$$
J = \frac{1}{2}\mathbf{x}_T^\top Q_T \mathbf{x}_T + \frac{1}{2}\sum_{t=0}^{T-1} \left(\mathbf{x}_t^\top Q_t \mathbf{x}_t + \mathbf{u}_t^\top R_t \mathbf{u}_t\right)
$$

où $Q_T \succeq 0$ (semi-définie positive), $Q_t \succeq 0$ et $R_t \succ 0$ (définie positive) sont des matrices symétriques de dimensions appropriées. Le caractère défini positif de $R_t$ garantit que le problème de minimisation est bien posé.

Un coût terminal quadratique entraîne une fonction de valeur quadratique à chaque étape antérieure. Supposons que le coût restant optimal à l'étape $t$ soit de la forme

$$
J_t^\star(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top P_t \mathbf{x}_t
$$

pour une certaine matrice semi-définie positive $P_t$. Au temps terminal, c'est vrai par définition : $P_T = Q_T$.

L'induction arrière vérifie cette hypothèse. Supposons que $J_{t+1}^\star(\mathbf{x}_{t+1}) = \frac{1}{2}\mathbf{x}_{t+1}^\top P_{t+1} \mathbf{x}_{t+1}$. L'équation de Bellman à l'étape $t$ s'écrit

$$
J_t^\star(\mathbf{x}_t) = \min_{\mathbf{u}_t} \left[ \frac{1}{2}\mathbf{x}_t^\top Q_t \mathbf{x}_t + \frac{1}{2}\mathbf{u}_t^\top R_t \mathbf{u}_t + J_{t+1}^\star(\mathbf{x}_{t+1}) \right]
$$

En substituant la dynamique $\mathbf{x}_{t+1} = A_t\mathbf{x}_t + B_t\mathbf{u}_t$ et la forme quadratique de $J_{t+1}^\star$ :

$$
J_t^\star(\mathbf{x}_t) = \min_{\mathbf{u}_t} \left[ \frac{1}{2}\mathbf{x}_t^\top Q_t \mathbf{x}_t + \frac{1}{2}\mathbf{u}_t^\top R_t \mathbf{u}_t + \frac{1}{2}(A_t\mathbf{x}_t + B_t\mathbf{u}_t)^\top P_{t+1} (A_t\mathbf{x}_t + B_t\mathbf{u}_t) \right]
$$

En développant le dernier terme :

$$
(A_t\mathbf{x}_t + B_t\mathbf{u}_t)^\top P_{t+1} (A_t\mathbf{x}_t + B_t\mathbf{u}_t) = \mathbf{x}_t^\top A_t^\top P_{t+1} A_t \mathbf{x}_t + 2\mathbf{x}_t^\top A_t^\top P_{t+1} B_t \mathbf{u}_t + \mathbf{u}_t^\top B_t^\top P_{t+1} B_t \mathbf{u}_t
$$

L'expression à l'intérieur de la minimisation devient :

$$
\frac{1}{2}\mathbf{x}_t^\top Q_t \mathbf{x}_t + \frac{1}{2}\mathbf{u}_t^\top R_t \mathbf{u}_t + \frac{1}{2}\mathbf{x}_t^\top A_t^\top P_{t+1} A_t \mathbf{x}_t + \mathbf{x}_t^\top A_t^\top P_{t+1} B_t \mathbf{u}_t + \frac{1}{2}\mathbf{u}_t^\top B_t^\top P_{t+1} B_t \mathbf{u}_t
$$

En regroupant les termes qui font intervenir $\mathbf{u}_t$ et en appelant le résultat $G_t$ :

$$
G_t(\mathbf{x}_t,\mathbf{u}_t) = \frac{1}{2}\mathbf{x}_t^\top (Q_t + A_t^\top P_{t+1} A_t) \mathbf{x}_t + \mathbf{x}_t^\top A_t^\top P_{t+1} B_t \mathbf{u}_t + \frac{1}{2}\mathbf{u}_t^\top (R_t + B_t^\top P_{t+1} B_t) \mathbf{u}_t
$$

C'est une fonction quadratique de $\mathbf{u}_t$, et $J_t^\star(\mathbf{x}_t) = \min_{\mathbf{u}_t} G_t(\mathbf{x}_t,\mathbf{u}_t)$. Pour trouver le minimiseur, on annule le gradient de $G_t$ par rapport à $\mathbf{u}_t$ :

$$
\nabla_{\mathbf{u}_t} G_t(\mathbf{x}_t,\mathbf{u}_t) = (R_t + B_t^\top P_{t+1} B_t) \mathbf{u}_t + B_t^\top P_{t+1} A_t \mathbf{x}_t = 0
$$

Comme $R_t + B_t^\top P_{t+1} B_t$ est définie positive ($R_t$ et $P_{t+1}$ sont toutes deux semi-définies positives, $R_t$ étant strictement positive), on peut résoudre pour obtenir la commande optimale :

$$
\mathbf{u}_t^\star = -(R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t \mathbf{x}_t
$$

Définissons la matrice de gain :

$$
K_t = (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t
$$

de sorte que $\mathbf{u}_t^\star = -K_t\mathbf{x}_t$. C'est une **politique de rétroaction linéaire** : la commande optimale est simplement une fonction linéaire de l'état courant.

En substituant $\mathbf{u}_t^\star$ dans l'expression du coût restant et en simplifiant (par complétion du carré), on obtient :

$$
J_t^\star(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top P_t \mathbf{x}_t
$$

où $P_t$ satisfait l'**équation de Riccati en temps discret** :

$$
P_t = Q_t + A_t^\top P_{t+1} A_t - A_t^\top P_{t+1} B_t (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t
$$


La récurrence arrière obtenue est :


````{prf:algorithm} Récurrence arrière pour la commande linéaire quadratique
:label: backward-recursion-lqr

**Entrée :** matrices du système $A_t, B_t$, matrices de coût $Q_t, R_t, Q_T$, horizon $T$

**Sortie :** matrices de coût $P_t$ et matrices de gain $K_t$ pour $t = 0, \ldots, T-1$

1. **Initialiser :** $P_T = Q_T$

2. **Pour** $t = T-1, T-2, \ldots, 0$ :
   1. Calculer la matrice de gain :

      $$K_t = (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t$$

   2. Calculer la matrice de coût par l'équation de Riccati :

      $$P_t = Q_t + A_t^\top P_{t+1} A_t - A_t^\top P_{t+1} B_t (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t$$

3. **Fin pour**

4. **Renvoyer :** $\{P_0, \ldots, P_T\}$ et $\{K_0, \ldots, K_{T-1}\}$

**Politique optimale :** $\mathbf{u}_t^\star = -K_t\mathbf{x}_t$

**Coût restant optimal :** $J_t^\star(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top P_t \mathbf{x}_t$
````

Le [chapitre sur iLQR et DDP](iterative-trajectory-optimization.md) applique
cette récurrence à des problèmes non linéaires. Linéariser la dynamique et
approcher les coûts par des formes quadratiques le long d'une trajectoire
nominale donne un problème linéaire quadratique local, dont la récurrence de
Riccati fournit une correction de la trajectoire. Répéter cette étape donne
iLQR et la programmation dynamique différentielle.

### Stabilisation locale du chariot-pendule

Le chariot-pendule du chapitre sur l'optimisation de trajectoires partait de la configuration pendante et exigeait une grande manœuvre non linéaire pour atteindre le sommet. Une fois qu'il est près de l'équilibre vertical, il reste un problème plus petit : rejeter les écarts locaux en déplaçant le chariot en réponse à l'état mesuré. L'état et l'entrée non linéaires sont les mêmes qu'auparavant,

$$
\mathbf{x}=(p,v,\theta,\omega), \qquad
u=\text{accélération horizontale du chariot},
$$

avec $\theta=0$ en position verticale. Un pas d'intégration de Runge-Kutta d'ordre quatre avec un pas $h$ définit l'application discrète $\mathbf{x}_{k+1}=F_h(\mathbf{x}_k,u_k)$, comme dans le chapitre sur l'optimisation de trajectoires. La configuration verticale avec une entrée nulle est un point fixe de cette application : $F_h(\mathbf{x}^\star,u^\star)=\mathbf{x}^\star$ avec $(\mathbf{x}^\star,u^\star)=(0,0)$.

Écrivons l'état et l'entrée comme des écarts par rapport à ce point, $\mathbf{x}_k=\mathbf{x}^\star+\delta\mathbf{x}_k$ et $u_k=u^\star+\delta u_k$. Un développement de Taylor au premier ordre de $F_h$ autour de $(\mathbf{x}^\star,u^\star)$ donne

$$
F_h(\mathbf{x}^\star+\delta\mathbf{x}_k,\,u^\star+\delta u_k)
= F_h(\mathbf{x}^\star,u^\star)
+ A\,\delta\mathbf{x}_k + B\,\delta u_k
+ O\!\left(\|\delta\mathbf{x}_k\|^2+|\delta u_k|^2\right),
$$

où $A$ et $B$ sont les jacobiennes de l'application discrète à l'équilibre :

$$
A=\left.\frac{\partial F_h}{\partial\mathbf{x}}\right|_{(\mathbf{x}^\star,u^\star)}\in\mathbb{R}^{4\times4},
\qquad
B=\left.\frac{\partial F_h}{\partial u}\right|_{(\mathbf{x}^\star,u^\star)}\in\mathbb{R}^{4\times1}.
$$

Comme $F_h(\mathbf{x}^\star,u^\star)=\mathbf{x}^\star$, soustraire $\mathbf{x}^\star$ des deux membres et négliger le reste du second ordre laisse la dynamique linéaire des écarts

$$
\delta\mathbf{x}_{k+1}\approx A\,\delta\mathbf{x}_k+B\,\delta u_k .
$$

L'approximation n'est bonne que tant que $\delta\mathbf{x}_k$ et $\delta u_k$ restent petits, ce que teste le troisième déroulement ci-dessous. On dérive le pas RK4 lui-même, et non le champ de vecteurs en temps continu; $A$ et $B$ tiennent donc déjà compte de l'échantillonnage au pas $h$ et ne nécessitent aucune discrétisation séparée. Le code calcule les deux jacobiennes par différences finies centrées de `rk4_step`.

L'expérience utilise $h=0.02$ s, $Q=\operatorname{diag}(2,0.2,80,3)$ et $R=0.15$. Dans ce cas invariant dans le temps et à horizon infini, la récurrence de Riccati converge vers une matrice fixe $P$ qui satisfait l'équation algébrique de Riccati discrète. La politique correspondante est $u_k=-K\delta\mathbf{x}_k$.

La boucle fermée linéaire sans contraintes est asymptotiquement stable lorsque toutes les valeurs propres de $A-BK$ se trouvent à l'intérieur du disque unité. L'implémentation physique ajoute deux contraintes absentes de ce calcul de valeurs propres : l'accélération est écrêtée à $|u|\leq 8\;\mathrm{m\,s^{-2}}$, et le chariot doit rester sur un rail de $2.4$ m. Trois déroulements non linéaires déterministes départagent les affirmations que la linéarisation permet d'étayer :

1. Un pendule non commandé part à $5^\circ$ de la verticale.
2. Le correcteur linéaire quadratique part du même écart de $5^\circ$.
3. Le même correcteur part à $45^\circ$ de la verticale, avec les mêmes limites d'actionneur et de rail.

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))

from cartpole_control import (
    CartPoleParameters,
    design_lqr,
    format_lqr_metrics,
    make_lqr_animation,
    make_lqr_figure,
    run_lqr_cases,
)

cartpole_parameters = CartPoleParameters()
lqr_design = design_lqr(cartpole_parameters)
lqr_cases = run_lqr_cases(cartpole_parameters)
```

```{code-cell} python
:label: fig-cartpole-lqr-limits
:caption: La boucle fermée linéaire discrète est stable, et le correcteur non linéaire se rétablit d'une perturbation de 5 degrés. Sans commande, le même écart initial s'amplifie. À partir de 45 degrés, l'accélération commandée sature et le chariot atteint la limite de son rail; le correcteur local ne mène donc pas le rétablissement à terme. Toutes les courbes utilisent le même procédé non linéaire; seuls l'état initial et le correcteur diffèrent.
:tags: [remove-input]

print(format_lqr_metrics(lqr_cases, lqr_design))
make_lqr_figure(lqr_cases, cartpole_parameters)
```

Les valeurs propres en boucle fermée certifient la stabilité asymptotique du modèle linéarisé sans contraintes, et non celle de chaque trajectoire du procédé non linéaire sous contraintes. Le déroulement à $5^\circ$ reste dans une région où l'approximation linéaire fournit des actions utiles. Dans le déroulement à $45^\circ$, le correcteur commande immédiatement plus d'accélération que l'actionneur ne peut en fournir, puis le chariot atteint le bout du rail. Augmenter les coefficients de $Q$ ne peut pas supprimer ces limites physiques.

```{code-cell} python
:label: anim-cartpole-lqr
:caption: Validation non linéaire du correcteur linéaire quadratique local. Le panneau de gauche montre la chute non commandée à partir de 5 degrés, le panneau central le rétablissement à partir du même état, et le panneau de droite le déroulement à 45 degrés qui se termine à la limite du rail. Python génère chaque image à partir des trajectoires enregistrées.
:tags: [remove-input]

from IPython.display import HTML, display
import matplotlib.pyplot as plt

lqr_animation = make_lqr_animation(lqr_cases, cartpole_parameters)
display(HTML(lqr_animation.to_jshtml()))
plt.close(lqr_animation._fig)
```

Tenir un stylo en équilibre sur un doigt motive le choix de l'accélération du chariot comme entrée, car le doigt stabilise l'objet en déplaçant sa base. Le modèle du chariot-pendule remplace le contact par une articulation plane sans frottement. Un vrai stylo peut glisser, se détacher, fléchir et tourner hors du plan, tandis que la perception et le mouvement de la main introduisent des retards. Le calcul établit une stabilisation locale pour le modèle de corps rigide énoncé; la démonstration en classe partage son instabilité et son mécanisme de rétroaction, mais pas toutes ses équations.

:::{dropdown} Examiner la linéarisation et la conception de la commande linéaire quadratique
```{literalinclude} code/cartpole_control.py
:language: python
:start-at: def linearize_upright
:end-before: def simulate_lqr
:linenos:
```
:::

{download}`Télécharger le code source partagé du chariot-pendule non linéaire et de la commande linéaire quadratique <code/cartpole_control.py>`.

## Synthèse et perspectives

La récurrence arrière remplace une optimisation de trajectoire unique par une
suite de sous-problèmes indexés par l'état. Stocker le coût restant aux états
de la grille et choisir des fonctions cardinales pour interpoler entre eux fait
de la récurrence un calcul fini. Avec des poids positifs ou nuls, la récurrence
interpolée est la récurrence exacte d'un modèle de grille, déterministe pour
l'arrondi et stochastique pour l'interpolation linéaire, et ses erreurs
s'additionnent d'une étape à l'autre au lieu de se multiplier. Une décision
entre les états de la grille provient d'une anticipation à un pas par rapport
au coût restant interpolé. Le cas linéaire quadratique ne nécessite aucune
grille : la fonction de valeur reste quadratique, la récurrence devient la
récurrence de Riccati, et la politique est une rétroaction linéaire sur l'état.

Le [chapitre sur iLQR et DDP](iterative-trajectory-optimization.md) utilise
cette récurrence de Riccati comme étape locale de l'optimisation de
trajectoires non linéaire. Il linéarise la dynamique autour d'une trajectoire
nominale, résout vers l'arrière le problème linéaire quadratique obtenu, et
teste la correction au moyen du simulateur non linéaire.

L'interpolation linéaire a déjà remplacé chaque successeur déterministe par
une loi sur deux états de la grille. Comment la valeur et la politique
changent-elles lorsque le modèle lui-même prescrit une loi sur les successeurs
possibles? La [programmation dynamique stochastique](stochastic-dp.md)
remplace la valeur de continuation déterministe par une espérance
conditionnelle.
