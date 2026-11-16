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
# Variantes de la commande prédictive et fonctionnement fiable

La commande à horizon glissant fournit une rétroaction en résolvant à répétition
un même problème nominal à horizon fini. Comment ce problème, et le correcteur
qui l'entoure, doivent-ils changer lorsque la tâche consiste à suivre une
référence ou à maximiser un rendement économique, lorsque le problème comporte
de l'incertitude ou des modes discrets, ou lorsque le solveur manque son
échéance?

Une fois l'idée de base de la commande à horizon glissant bien comprise, il est utile de voir comment la même ossature accueille de nombreuses variantes. Dans tous les cas, on transcrit le problème de contrôle optimal en temps continu en un programme non linéaire ayant la structure du [sous-problème à horizon glissant](receding-horizon-control.md#the-receding-horizon-principle) :

$$
\begin{aligned}
    \text{minimiser} \quad & c_N(\mathbf{x}_N) + \sum_{k=0}^{N-1} c_k(\mathbf{x}_k, \mathbf{u}_k) \\
    \text{sous les contraintes} \quad & \mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k) \\
                            & \mathbf{g}_k(\mathbf{x}_k, \mathbf{u}_k) \leq \mathbf{0} \\
                            & \mathbf{x}_{\min} \leq \mathbf{x}_k \leq \mathbf{x}_{\max} \\
                            & \mathbf{u}_{\min} \leq \mathbf{u}_k \leq \mathbf{u}_{\max} \\
    \text{étant donné} \quad & \mathbf{x}_0 = \mathbf{x}_{\text{courant}} \enspace .
\end{aligned}
$$

Les composantes de ce PNL proviennent de la discrétisation du problème en temps continu sur un horizon de prédiction de $N$ pas de longueur $\Delta t$, à partir de l'instant courant. La transition discrète $\mathbf{f}_k$ est déterminée par le schéma d'intégration, et chaque coût d'étape $c_k$ comprend le poids de quadrature de son intervalle, comme dans les [transcriptions par collocation](continuous-time-collocation.md). Une fois ce canevas en place, le reste est affaire d'interprétation : comment on définit le coût, comment on traite l'incertitude, comment on traite les contraintes et quelle structure on exploite.

## Commande prédictive de poursuite

Comment l'objectif à horizon fini doit-il pénaliser l'écart à une trajectoire
de référence variable dans le temps?

La configuration la plus courante est la poursuite d'une référence. On dispose alors de trajectoires cibles variables dans le temps $(\mathbf{x}_k^{\mathrm{ref}}, \mathbf{u}_k^{\mathrm{ref}})$, et le rôle du correcteur est de maintenir le système près de celles-ci. Le coût est habituellement quadratique :

$$
\begin{aligned}
    c_k(\mathbf{x}_k, \mathbf{u}_k) &= \| \mathbf{x}_k - \mathbf{x}_k^{\mathrm{ref}} \|_{Q}^2 + \| \mathbf{u}_k - \mathbf{u}_k^{\mathrm{ref}} \|_{R}^2 \\
    c_N(\mathbf{x}_N) &= \| \mathbf{x}_N - \mathbf{x}_N^{\mathrm{ref}} \|_{P}^2 \enspace .
\end{aligned}
$$

Lorsque la dynamique est linéaire et que les contraintes sont polyédrales, on obtient un programme quadratique convexe à chaque pas de temps.

## Commande prédictive de régulation

Qu'est-ce qui change lorsque la cible est un équilibre fixe plutôt qu'une
référence mobile?

Dans les tâches de régulation, on cherche à ramener le système vers un point d'équilibre $(\mathbf{x}_{\mathrm{eq}}, \mathbf{u}_{\mathrm{eq}})$, habituellement en présence de perturbations. Il s'agit simplement d'une commande prédictive de poursuite à références constantes, de sorte que le coût d'étape ne dépend plus de $k$ :

$$
\begin{aligned}
    c(\mathbf{x}_k, \mathbf{u}_k) &= \| \mathbf{x}_k - \mathbf{x}_{\mathrm{eq}} \|_{Q}^2 + \| \mathbf{u}_k - \mathbf{u}_{\mathrm{eq}} \|_{R}^2 \\
    c_N(\mathbf{x}_N) &= \| \mathbf{x}_N - \mathbf{x}_{\mathrm{eq}} \|_{P}^2 \enspace .
\end{aligned}
$$

Pour garantir la stabilité, il est courant d'ajouter une contrainte terminale $\mathbf{x}_N \in \mathcal{X}_f$, où $\mathcal{X}_f$ est un ensemble invariant par commande sous une loi de rétroaction connue.


## Commande prédictive économique

Le correcteur peut-il optimiser directement la production ou le revenu lorsque
le meilleur fonctionnement n'a pas à rester près d'une consigne prescrite?

Les commandes prédictives de poursuite et de régulation pénalisent l'écart à une référence prescrite. Un correcteur économique optimise plutôt la quantité produite par l'exploitation du système. L'état optimal peut varier continuellement, puisqu'il n'y a pas de trajectoire de référence fixe.

### Captage de l'énergie des vagues

Un volet articulé extrait l'énergie de l'eau oscillante au moyen d'un dispositif de prélèvement de puissance (noté PTO dans les formules). Un seul mode de rotation donne le modèle

$$
I\ddot q+b\dot q+kq=\tau_{\mathrm{wave}}(t)-\rho(t)\dot q,
$$

où $q$ est l'angle du volet et $\rho$ l'amortissement commandé du prélèvement. Le couple de prélèvement et la puissance captée sont

$$
\tau_{\mathrm{PTO}}=-\rho\dot q,
\qquad
P_{\mathrm{capture}}=-\tau_{\mathrm{PTO}}\dot q=\rho\dot q^2.
$$

Imposer $\rho\geq0$ maintient le prélèvement passif, car $\tau_{\mathrm{PTO}}\dot q=-\rho\dot q^2\leq0$. L'actionneur peut retirer de l'énergie mécanique au volet, mais il ne peut pas lui en injecter.

Un calcul à état figé suggère d'utiliser le plus d'amortissement possible. À $\dot q=0.4$ rad/s, des amortissements de $900$ et de $1900$ N m s/rad donnent des puissances captées instantanées de $144$ et de $304$ W. Ce calcul tient la vitesse fixe. Dans le système dynamique, un amortissement plus fort réduit la vitesse future, de sorte que l'énergie captée n'augmente pas nécessairement de façon monotone avec $\rho$. Le balayage à amortissement constant ci-dessous mesure cet effet différé.

L'expérience utilise une somme déterministe de trois composantes de couple de houle pendant $45$ s. Le correcteur se met à jour toutes les $h=0.12$ s et prédit $N=18$ pas, soit $2.16$ s, dans le futur. Son problème à horizon fini est

$$
\begin{aligned}
\max_{\rho_{0:N-1}}\quad &
h\sum_{j=0}^{N-1}\rho_j\omega_j^2
-\lambda\sum_{j=0}^{N-1}
\left(\frac{\rho_j-\rho_{j-1}}{\rho_{\max}}\right)^2 \\
\text{sous les contraintes}\quad &
\mathbf{x}_{j+1}=F_h(\mathbf{x}_j,\rho_j,\tau_{\mathrm{wave}}), \\
&0\leq\rho_j\leq2400\ \mathrm{N\,m\,s/rad},\\
&|q_j|\leq0.55\ \mathrm{rad},\\
&|\rho_j\omega_j|\leq2800\ \mathrm{N\,m},\\
&|\rho_j-\rho_{j-1}|/h\leq5000\ \mathrm{N\,m\,s^{-2}/rad}.
\end{aligned}
$$

Ici, $F_h$ est un pas de Runge-Kutta du modèle du volet et $\rho_{-1}$ est l'amortissement appliqué pendant l'intervalle de commande précédent. Seul $\rho_0$ est appliqué. L'état du volet est mesuré de nouveau, l'horizon avance, et le programme non linéaire est résolu à partir de la solution précédente décalée.

Deux correcteurs passifs servent de références comparables. L'amortissement constant demande $\rho=900$ N m s/rad. Une heuristique sensible à la phase augmente l'amortissement lorsque le produit mesuré $\tau_{\mathrm{wave}}\dot q$ est positif et le diminue dans le cas contraire. Les trois demandes passent par la même projection sur les bornes d'amortissement, de couple et de vitesse de variation de l'amortissement avant d'atteindre le procédé. La contrainte de course est différente, car elle dépend de l'évolution prédite de l'état plutôt que de la commande courante de l'actionneur.

Avant d'examiner les trajectoires, prédisez si l'amortissement constant devrait maximiser l'énergie captée totale. Prédisez aussi quel correcteur devrait faire varier le plus l'actionneur. L'objectif économique peut favoriser un fonctionnement proche des limites d'état et de couple.

```{code-cell} python
:tags: [remove-cell]

import sys
sys.path.insert(0, "code")

import pandas as pd
from IPython.display import HTML, display
import matplotlib.pyplot as plt

from wave_energy import (
    WaveParameters,
    create_animation as create_wave_animation,
    make_closed_loop_figure,
    make_tradeoff_figure,
    metrics_table as wave_metrics_table,
    run_comparison as run_wave_comparison,
    run_damping_sweep,
)

wave_parameters = WaveParameters()
wave_results = run_wave_comparison(wave_parameters, duration=45.0)
wave_sweep = run_damping_sweep(wave_parameters, duration=45.0)
```

```{code-cell} python
:tags: [remove-input]
:label: fig-wave-economic-mpc-trajectories
:caption: Réponse en boucle fermée sous le même forçage déterministe à trois fréquences. La commande prédictive économique capte plus d'énergie en faisant varier l'amortissement passif et en fonctionnant près des limites de course et de couple de prélèvement. L'amortissement constant franchit la région ombrée où la course est admissible, tandis que la référence sensible à la phase reste plus loin à l'intérieur. Chaque courbe est produite par la simulation du procédé non linéaire.

wave_summary_figure = make_closed_loop_figure(wave_results, wave_parameters)
display(wave_summary_figure)
plt.close(wave_summary_figure)
```

La commande prédictive économique capte environ $47.3$ kJ, contre $39.9$ kJ pour l'amortissement sensible à la phase et $39.3$ kJ pour l'amortissement constant. Elle atteint la limite de course de $0.55$ rad et la limite de couple de $2.8$ kN m sans dépasser l'une ou l'autre au-delà de la tolérance numérique. L'amortissement constant dépasse la limite de course d'environ $0.084$ rad. La loi sensible à la phase reste réalisable et fait moins varier l'amortissement, mais elle laisse de l'énergie non récoltée.

L'un des 375 programmes non linéaires se termine sans satisfaire l'indicateur de convergence du solveur. L'implémentation applique alors la suite réalisable décalée, en la faisant passer par la projection commune de l'actionneur. La trajectoire obtenue reste dans les limites enregistrées, mais ce repli fait partie du résultat plutôt que d'être une preuve de faisabilité récursive.

```{code-cell} python
:tags: [remove-input]

wave_table = pd.DataFrame(wave_metrics_table(wave_results))
for column in (
    "energy_kj",
    "peak_stroke_deg",
    "peak_torque_knm",
    "damping_variation_per_s",
    "stroke_violation_deg",
    "torque_violation_nm",
):
    wave_table[column] = wave_table[column].map(lambda x: f"{x:.3f}")
wave_table.rename(
    columns={
        "controller": "controller",
        "energy_kj": "captured energy (kJ)",
        "peak_stroke_deg": "peak stroke (deg)",
        "peak_torque_knm": "peak PTO torque (kN m)",
        "damping_variation_per_s": "damping variation per second",
        "stroke_violation_deg": "stroke violation (deg)",
        "torque_violation_nm": "torque violation (N m)",
    }
)
```

Le balayage à amortissement constant place les trois correcteurs sur un même diagramme énergie-mouvement. Un amortissement constant plus grand améliore d'abord le captage, puis étouffe le mouvement qui produit la puissance. La commande prédictive économique se situe au-dessus de ce compromis statique, car elle modifie l'amortissement selon la phase prédite tout en respectant la borne de course.

```{code-cell} python
:tags: [remove-input]
:label: fig-wave-economic-mpc-tradeoff
:caption: Énergie captée en fonction de la course maximale du volet. Les cercles gris balaient un amortissement passif constant de 100 à 2800 N m s/rad. La ligne pointillée verticale est la contrainte de course de 0.55 rad. La commande prédictive économique obtient plus d'énergie que le balayage à amortissement fixe en planifiant l'amortissement sur le cycle de houle prédit. La référence sensible à la phase est réalisable, mais n'atteint pas l'énergie de la commande prédictive.

wave_tradeoff_figure = make_tradeoff_figure(
    wave_results,
    wave_sweep,
    wave_parameters,
)
display(wave_tradeoff_figure)
plt.close(wave_tradeoff_figure)
```

L'animation montre la trajectoire de la commande prédictive économique et l'énergie accumulée, tirées de la même simulation exécutée. L'amortissement affiché est l'action appliquée après la projection de l'actionneur.

```{code-cell} python
:tags: [remove-input]
:label: fig-wave-economic-mpc-animation
:caption: Rejeu en temps compressé du mouvement du volet et de l'énergie captée sous la commande prédictive économique. Le correcteur résout de nouveau un programme non linéaire à 18 pas toutes les 0.12 secondes et applique la première action d'amortissement passif.

wave_animation = create_wave_animation(wave_results, wave_parameters, frame_stride=3)
wave_html = wave_animation.to_jshtml(fps=25)
plt.close(wave_animation._fig)
display(HTML(wave_html))
```

:::{dropdown} Examiner la résolution de la commande prédictive économique

```{literalinclude} code/wave_energy.py
:language: python
:start-at: def solve_economic_mpc
:end-before: def _metrics
:linenos:
```

:::

{download}`Télécharger l'expérience complète sur l'énergie des vagues <code/wave_energy.py>`

Ce modèle suppose une prévision exacte de la houle à court terme et un seul mode rigide du volet. Les dispositifs réels comportent des effets de mémoire de radiation, d'autres modes structurels, des pertes, des erreurs de mesure et une incertitude de prévision. Ces omissions touchent à la fois le captage prédit et les marges sur les contraintes. La commande prédictive robuste ou stochastique peut représenter une partie de cette incertitude, tandis que des données peuvent servir à estimer la dynamique résiduelle non modélisée.

(robust-mpc)=
## Commande prédictive robuste

Comment protéger la faisabilité et la performance contre toute perturbation
appartenant à un ensemble d'incertitude borné?

Certains systèmes sont exposés à des perturbations externes ou à de petites erreurs de modèle. On veut alors que le correcteur prenne des décisions qui fonctionneront quoi qu'il arrive, tant que les perturbations restent dans des bornes connues. C'est l'idée de la **commande prédictive robuste**.

Plutôt que de planifier une seule trajectoire, le correcteur planifie un chemin « nominal » (ce qui se produirait en l'absence de toute perturbation), puis ajoute une correction par rétroaction pour réagir aux perturbations qui surviennent réellement. Cela s'écrit :

$$
\mathbf{u}_k = \bar{\mathbf{u}}_k + K (\mathbf{x}_k - \bar{\mathbf{x}}_k) \enspace ,
$$

où $\bar{\mathbf{u}}_k$ est l'entrée planifiée et $K$ un gain de rétroaction qui ramène le système vers le chemin nominal s'il s'en écarte.

Comme on connaît la taille de la perturbation dans le pire cas, on peut estimer jusqu'où l'état réel peut dériver par rapport au plan, et « resserrer » les contraintes en conséquence. Le plan nominal est ainsi tenu à bonne distance des frontières des contraintes, de sorte que, même si le système est bousculé, il reste dans les limites. On parle souvent de **commande prédictive par tube**, car la trajectoire réelle reste dans un tube autour de la trajectoire nominale.

Le principal avantage est qu'on peut traiter l'incertitude sans résoudre une optimisation compliquée dans le pire cas à chaque pas de temps. Toute l'incertitude est prise en compte dans la conception de la rétroaction $K$ et des contraintes resserrées.


## Commande prédictive stochastique

Que devient-il possible de faire lorsque l'incertitude est représentée par des
probabilités plutôt que seulement par des bornes dans le pire cas?

Si les perturbations sont aléatoires plutôt qu'adverses, un objectif naturel consiste à optimiser le coût espéré tout en imposant les contraintes de façon probabiliste. On obtient ainsi la **commande prédictive stochastique**, dans laquelle :

* Le coût devient une espérance :

  $$
  \mathbb{E} \left[ c_N(\mathbf{x}_N) + \sum_{k=0}^{N-1} c_k(\mathbf{x}_k, \mathbf{u}_k) \right]
  $$
* Les contraintes peuvent être violées avec une faible probabilité :

  $$
  \mathbb{P}[\mathbf{g}_k(\mathbf{x}_k, \mathbf{u}_k) \leq \mathbf{0}] \geq 1 - \varepsilon
  $$

En pratique, les espérances sont approchées au moyen d'un ensemble fini de scénarios de perturbation tirés à l'avance. Pour chaque scénario, la dynamique du système est simulée vers l'avant avec les mêmes entrées de commande $\mathbf{u}_k$, qui sont partagées entre tous les scénarios afin de respecter la non-anticipativité. On obtient un seul problème d'optimisation déterministe comportant plusieurs copies parallèles de la dynamique, une par futur échantillonné. Cette approche conserve la structure habituelle de la commande prédictive, avec une croissance seulement modérée de la taille du problème.

Malgré les apparences, ce n'est pas de la programmation dynamique. Il n'y a ni fonction de valeur ni arbre de tous les chemins possibles. Il n'y a qu'un ensemble fini de futurs choisis a priori, sur lesquels on optimise directement. Cette approche par scénarios est courante dans les systèmes énergétiques comme la planification hydroélectrique, où les apports sont incertains mais où l'on peut générer des trajectoires échantillonnées à partir de prévisions.

Les contraintes de risque sont habituellement imposées sur tous les scénarios ou encodées au moyen de mesures de risque comme la CVaR. Par exemple, on peut pénaliser les violations qui surviennent dans les pires $(1 - \alpha)\%$ des échantillons, tout en optimisant la performance espérée dans l'ensemble.

## Commande prédictive hybride et en nombres entiers mixtes

Comment les changements de mode, les dispositifs tout-ou-rien et les décisions
logiques peuvent-ils entrer dans la même optimisation à horizon fini?

Lorsque les systèmes comportent des commutations discrètes (par exemple des vannes tout-ou-rien, une sélection de mode ou une logique combinatoire), le problème de commande prédictive doit inclure des variables entières ou binaires. Elles apparaissent dans des contraintes comme

$$
\boldsymbol{\delta}_k \in \{0,1\}^{n_\delta}, \qquad \mathbf{u}_k \in \mathcal{U}(\boldsymbol{\delta}_k)
$$

où $n_\delta$ compte les décisions binaires à chaque pas, en plus d'une dynamique et de coûts qui dépendent du mode. La formulation obtenue est un **programme non linéaire en nombres entiers mixtes** (PNLNEM). L'idée de l'horizon glissant reste la même, mais chaque résolution coûte plus cher en raison de la nature combinatoire de l'espace des décisions.

## Commande prédictive répartie et décentralisée

Lorsqu'une seule optimisation est trop grande ou que la responsabilité est
répartie, quelle information les correcteurs locaux doivent-ils échanger pour
coordonner leurs plans?

Les systèmes de grande taille se composent souvent de sous-systèmes en interaction. La commande prédictive répartie décompose le PNL global en problèmes plus petits résolus en parallèle, avec des contraintes de coordination qui imposent la cohérence des variables partagées :

$$
\sum_{i} H^i \mathbf{z}^i_k = \mathbf{0} \qquad \text{(contrainte de couplage)}
$$

Chaque sous-système résout un problème local portant sur ses propres variables d'état et d'entrée, puis échange de l'information avec ses voisins. La coordination peut se faire par des méthodes primales-duales, par ADMM ou par des schémas de consensus, mais chaque bloc local ressemble à un problème de commande prédictive standard.


## Commande prédictive adaptative et fondée sur l'apprentissage

Comment le modèle de prédiction peut-il s'améliorer à partir de données sans
abandonner les contraintes explicites qu'impose la commande prédictive?

En pratique, on ne connaît pas toujours précisément le vrai modèle $\mathbf{f}_k$ ni la fonction de coût $c_k$. En **commande prédictive adaptative**, ceux-ci sont mis à jour en ligne à partir de données :

$$
\mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k; \boldsymbol{\theta}_t), \qquad
c_k(\mathbf{x}_k, \mathbf{u}_k) = c_k(\mathbf{x}_k, \mathbf{u}_k; \boldsymbol{\psi}_t)
$$

Les paramètres $\boldsymbol{\theta}_t$ et $\boldsymbol{\psi}_t$ sont appris en temps réel, et le sous-problème résolu au temps $t$ utilise leurs estimations courantes. Combinée à la distillation de politiques, à l'approximation de la valeur ou à l'imitation de trajectoires, cette approche rejoint l'apprentissage par renforcement : les solutions de la commande prédictive servent alors de supervision pour une politique réactive.


## Robustesse et traitement des défaillances

Que doit faire le système en boucle fermée lorsque le problème de commande
prédictive privilégié est irréalisable, imprécis ou inachevé à l'échéance de
commande?

L'exemple de l'énergie des vagues suppose que chaque optimisation se termine avant la mise à jour suivante de la commande et que son modèle de prédiction reste exact sur l'horizon. Un correcteur prédictif en exploitation doit aussi gérer des contraintes incompatibles, des erreurs de modélisation et des échéances de calcul manquées.

Une perturbation peut amener l'état mesuré en un point d'où la cible demandée est inatteignable dans l'horizon. Un écart de modèle peut rendre une trajectoire réalisable en prédiction et irréalisable sur le procédé. Un solveur peut aussi s'arrêter à cause d'un modèle local mal conditionné ou d'un ensemble actif qui change. Ces cas exigent une hiérarchie explicite de contraintes dures, d'objectifs souples et d'actions de repli. Les mécanismes suivants modifient le problème à horizon fini ou son déploiement sans changer le principe de l'horizon glissant.
## Assouplir les contraintes au moyen de variables d'écart

Quelles contraintes peuvent être violées à un prix quantifié pour que
l'optimiseur puisse tout de même renvoyer une action utilisable?

La première approche du traitement de l'irréalisabilité reconnaît que toutes les contraintes n'ont pas la même importance. La température d'un réacteur chimique ne doit jamais dépasser le seuil d'emballement : c'est une contrainte dure qui ne peut pas être violée. En revanche, maintenir la température dans une plage d'efficacité optimale est seulement souhaitable. On peut traiter cette exigence comme une contrainte souple, qu'on préfère satisfaire mais qu'on peut relâcher au besoin.

Cette hiérarchie motive la reformulation du problème d'optimisation au moyen de **variables d'écart** :

$$
\begin{aligned}
\min_{\mathbf{u}, \boldsymbol{\epsilon}} \quad & \sum_{k=0}^{N-1} \left( \|\mathbf{x}_k - \mathbf{x}_k^{\mathrm{ref}}\|_{Q}^2 + \|\mathbf{u}_k\|_{R}^2 + \mathbf{w}^\top \boldsymbol{\epsilon}_k \right) \\
\text{s.c.} \quad & \mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k) \\
& \mathbf{g}_{\text{dure}}(\mathbf{x}_k, \mathbf{u}_k) \leq \mathbf{0} \\
& \mathbf{g}_{\text{souple}}(\mathbf{x}_k, \mathbf{u}_k) \leq \boldsymbol{\epsilon}_k \\
& \boldsymbol{\epsilon}_k \geq \mathbf{0}
\end{aligned}
$$

Les poids de pénalité $\mathbf{w}$ encodent nos priorités. Les contraintes de sécurité peuvent utiliser $w_j = 10^6$, tandis que les contraintes de confort utilisent $w_j = 1$. Ce problème reformulé est toujours réalisable dès que les contraintes dures, à elles seules, admettent une solution. Autrement dit, on peut toujours rendre les variables d'écart $\boldsymbol{\epsilon}$ assez grandes pour satisfaire les contraintes souples.

Plutôt que de classer les contraintes en deux catégories, dures ou souples, on peut établir une **hiérarchie de contraintes** qui permet une dégradation progressive :

$$
\begin{aligned}
\text{Sécurité :} \quad & T_{\text{réacteur}} \leq T_{\text{emballement}} - 10 \quad & w = \infty \text{ (dure)} \\
\text{Équipement :} \quad & 0 \leq u_{\text{vanne}} \leq 100 \quad & w = 10^4 \\
\text{Efficacité :} \quad & T_{\text{optimale}} - 5 \leq T \leq T_{\text{optimale}} + 5 \quad & w = 10^2 \\
\text{Confort :} \quad & |T - T_{\text{consigne}}| \leq 1 \quad & w = 1
\end{aligned}
$$

À mesure que les conditions se dégradent, le correcteur abandonne les objectifs dans l'ordre inverse des priorités, en maintenant la sécurité même lorsque l'optimalité devient impossible.

## Restauration de la faisabilité

Si le problème principal échoue, une optimisation secondaire peut-elle
retrouver un état à partir duquel les contraintes nominales redeviennent
satisfaisables?

Lorsque même les contraintes souples se révèlent insuffisantes (par exemple à cause d'une défaillance catastrophique du solveur ou d'une structure de problème corrompue), il faut une **restauration de la faisabilité** qui trouve n'importe quel point réalisable, sans égard à l'optimalité :

$$
\begin{aligned}
\min_{\mathbf{u}, \mathbf{s}} \quad & \|\mathbf{s}\|_1 \\
\text{s.c.} \quad & \mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k) + \mathbf{s}_k \\
& \mathbf{x}_{\min} - \mathbf{s}_{x,k} \leq \mathbf{x}_k \leq \mathbf{x}_{\max} + \mathbf{s}_{x,k} \\
& \mathbf{u}_{\min} \leq \mathbf{u}_k \leq \mathbf{u}_{\max} \\
& \mathbf{s} \geq \mathbf{0}
\end{aligned}
$$

Cette formulation relâche temporairement jusqu'aux contraintes de dynamique, et trouve la solution « la moins irréalisable ». Elle répond à la question : s'il faut violer quelque chose, quelle est la violation minimale nécessaire? Une fois la faisabilité restaurée, on peut démarrer à chaud le problème d'origine à partir de ce point.

## Gouverneurs de référence

Une référence dangereuse peut-elle être modifiée avant d'atteindre un
correcteur interne par ailleurs fiable?

Plutôt que de réagir à l'irréalisabilité après coup, on peut la prévenir en filtrant les références au moyen d'un **gouverneur de référence**. Considérons un aéronef qui suit des points de passage. Au lieu de transmettre directement les points de passage à la commande prédictive, le gouverneur se demande : quelle est la référence atteignable la plus proche à partir de l'état courant?

$$
\begin{aligned}
\kappa^\star &= \max\left\{\kappa \in [0,1] : \text{commande prédictive}(\mathbf{x}_{\text{courant}}, \kappa \mathbf{r}_{\text{désirée}} + (1-\kappa)\mathbf{x}_{\text{courant}}) \text{ est réalisable}\right\}, \\
\mathbf{r}_{\text{filtrée}} &= \kappa^\star \mathbf{r}_{\text{désirée}} + (1-\kappa^\star)\mathbf{x}_{\text{courant}}
\end{aligned}
$$

Le gouverneur effectue une recherche linéaire entre l'état courant (toujours réalisable, puisque rester sur place n'exige aucune action) et la référence désirée (potentiellement irréalisable). Cela garantit que la commande prédictive reçoit toujours des problèmes réalisables, tout en progressant le plus possible vers le but.

Pour l'efficacité du calcul, on peut précalculer l'**ensemble admissible maximal de sortie** :

$$
\mathcal{O}_\infty = \{\mathbf{r} : \exists \text{ une trajectoire réalisable de } \mathbf{x} \text{ à } \mathbf{r} \text{ qui respecte toutes les contraintes}\}
$$

En ligne, le gouverneur se contente de projeter la référence désirée sur $\mathcal{O}_\infty$.

## Correcteurs de secours

Quelle action validée de façon indépendante doit prendre le relais lorsqu'on
ne peut pas se fier à l'optimiseur ou à son modèle?

Lorsque la commande prédictive échoue complètement (à cause d'un plantage du solveur, d'un dépassement de délai ou d'une défaillance numérique), il faut des correcteurs de secours qui demandent un calcul minimal tout en garantissant la stabilité et en tenant le système à l'écart des régions dangereuses.

L'approche standard utilise un **correcteur linéaire quadratique local** précalculé autour de l'équilibre :

$$
K_{\text{LQR}}, P = \text{LQR}(A, B, Q, R)
$$

où $(A, B)$ est la dynamique linéarisée à l'équilibre. Lorsque la commande prédictive échoue :

$$
\mathbf{u}_{\text{secours}} = \begin{cases}
K_{\text{LQR}}(\mathbf{x} - \mathbf{x}_{\text{eq}}) & \text{si } \mathbf{x} \in \mathcal{X}_{\text{LQR}} \\
\mathbf{u}_{\text{sûre}} & \text{sinon}
\end{cases}
$$

La région $\mathcal{X}_{\text{LQR}} = \{\mathbf{x} : (\mathbf{x} - \mathbf{x}_{\text{eq}})^\top P (\mathbf{x} - \mathbf{x}_{\text{eq}}) \leq \alpha\}$ représente le plus grand ensemble invariant dans lequel la commande linéaire quadratique fonctionne à coup sûr.

## Architectures en cascade

Comment superposer une optimisation plus lente et une rétroaction locale plus
rapide sans donner aux deux boucles une autorité conflictuelle?

Les systèmes de commande prédictive en production reposent rarement sur un seul solveur. Ils mettent plutôt en œuvre une **cascade de correcteurs de plus en plus prudents** qui échangent de l'optimalité contre de la fiabilité :

```python
def get_control(self, x, time_budget):
    """
    Multi-level cascade for robust real-time control
    """
    time_remaining = time_budget
    
    # Level 1: Full nonlinear MPC
    if time_remaining > 5e-3:  # 5ms minimum
        try:
            u, solve_time = self.solve_nmpc(x, time_remaining)
            if converged:
                return u
        except:
            pass
        time_remaining -= solve_time
    
    # Level 2: Simplified linear MPC
    if time_remaining > 1e-3:  # 1ms minimum
        try:
            # Linearize around current state
            A, B = self.linearize_dynamics(x)
            u, solve_time = self.solve_lmpc(x, A, B, time_remaining)
            return u
        except:
            pass
        time_remaining -= solve_time
    
    # Level 3: Explicit MPC lookup
    if time_remaining > 1e-4:  # 0.1ms minimum
        region = self.find_critical_region(x)
        if region is not None:
            return self.explicit_control_law[region](x)
    
    # Level 4: LQR backup
    if self.in_lqr_region(x):
        return self.K_lqr @ (x - self.x_eq)
    
    # Level 5: Emergency safe mode
    return self.emergency_stop(x)
```

Chaque niveau échange de l'optimalité contre de la fiabilité : le niveau 1 fournit une commande optimale mais coûteuse en calcul, le niveau 2 offre des solutions sous-optimales mais plus rapides, le niveau 3 fournit une évaluation instantanée précalculée, le niveau 4 assure une commande stabilisante sans poursuite, et le niveau 5 met en œuvre un arrêt sécuritaire.

Même lorsqu'on utilise des correcteurs de secours, on peut maintenir la continuité des solutions grâce à un **démarrage à chaud persistant** :

$$
\begin{aligned}
\mathbf{z}_{\text{chaud}}^{(t+1)} = \begin{cases}
\text{décaler}(\mathbf{z}^{(t)}) & \text{si la commande prédictive a réussi au temps } t \\
\text{étendre}(\mathbf{u}_{\text{secours}}^{(t)}) & \text{si le correcteur de secours a été utilisé} \\
\text{propager}(\mathbf{z}_{\text{chaud}}^{(t)}) & \text{si l'on maintient une solution virtuelle}
\end{cases}
\end{aligned}
$$

Ici, l'exposant $(t)$ désigne la solution calculée au temps $t$, et les indices restent des pas de prédiction. L'opération **décaler** prend une solution réussie de la commande prédictive, l'avance d'un pas de temps et y ajoute une action terminale : $[\mathbf{u}_1^{(t)}, \mathbf{u}_2^{(t)}, \ldots, \mathbf{u}_{N-1}^{(t)}, \pi_f(\mathbf{x}_N^{(t)})]$. Cette suite décalée assure une continuité temporelle naturelle pour l'optimisation suivante.

Lorsque la commande prédictive échoue et que la commande de secours est appliquée, l'opération **étendre** prolonge l'action de secours unique $\mathbf{u}_{\text{secours}}^{(t)}$ en une suite couvrant tout l'horizon, soit par répétition, soit en simulant le correcteur de secours vers l'avant. On obtient ainsi une estimation initiale raisonnable pour le démarrage à chaud à partir d'une information limitée.

L'opération **propager** maintient une trajectoire « virtuelle » en continuant de faire évoluer la solution précédente comme si elle était encore exécutée, même lorsque le système réel suit la commande de secours. Cette simulation vers l'avant garde le démarrage à chaud aligné dans le temps et pertinent pour le moment où la commande prédictive reprend.

## Service d'inférence sous un budget de latence et de puissance

Comment les échéances, les limites thermiques, l'erreur de prévision et le
temps de résolution interagissent-ils dans un correcteur à horizon glissant
appliqué à une véritable charge de travail informatique?

Le programme de fréquences hors ligne de [Optimisation de trajectoires](discrete-time-optimal-control.md#example-offline-frequency-planning-for-inference)
ne peut pas réagir lorsqu'une rafale arrive plus tôt que prévu. La commande à
horizon glissant utilise le même modèle agrégé, mais remplace la prévision
initiale après chaque seconde. Des courbes mesurées sur un NVIDIA L4 de débit
de service et de puissance par phase étalonnent le modèle, tandis que les
trajectoires de requêtes et de files restent des sorties de simulation. La
règle d'ordonnancement reste fixée au modèle réduit de préremplissage par blocs
entrelacés de 512 jetons du chapitre sur la modélisation, de sorte que la
comparaison isole la rétroaction exercée par l'horloge du GPU.

Au temps de commande $t$, l'état

$$
x_t=(p_t,d_t,T_t,f_{t-1})
$$

contient le travail de préremplissage en file, le travail de décodage inachevé,
la température et la dernière fréquence appliquée. Le correcteur prévoit dix
transitions d'une seconde. Les arrivées sont prédites à partir du taux observé
sur les 30 dernières secondes, tandis que le travail de sortie espéré utilise
la loi empirique des longueurs de sortie de la trace. Le procédé au niveau des
requêtes reçoit plutôt les requêtes et les longueurs de sortie réalisées.

Le problème à horizon fini utilise l'objectif normalisé

$$
J_t=\sum_{k=0}^{9}\left[
\frac{E_k}{E_{\max}}
+20\delta_{k,\mathrm{TTFT}}^2
+10\delta_{k,\mathrm{TPOT}}^2
+0.05\Delta f_k^2
+1000(s_{P,k}^2+s_{T,k}^2)
\right]+20B_{10}^2,
$$

où, comme dans le sous-problème à horizon glissant, $k$ compte les secondes
écoulées après le temps de commande $t$. Les variables $\delta_{k,\mathrm{TTFT}}$
et $\delta_{k,\mathrm{TPOT}}$ sont des dépassements normalisés de deux
estimations agrégées de délai. La première divise le nombre prédit de jetons de
préremplissage en file par le débit de préremplissage profilé. La seconde
divise le nombre prédit de requêtes de décodage actives par le débit de
décodage profilé. Ces estimations entrent dans l'optimisation, mais ne sont pas
égales au délai avant le premier jeton (TTFT) ni au temps par jeton de sortie
(TPOT) réalisés au niveau des requêtes; c'est la simulation détaillée qui
calcule ces derniers. La différence de fréquence $\Delta f_k$ est normalisée
par la plage d'horloge profilée.
Les variables d'écart $s_{P,k}$ et $s_{T,k}$ pénalisent les violations des
limites expérimentales de puissance et de température, et $B_{10}$ pénalise le
travail inachevé à la fin de l'horizon. SLSQP dispose d'au plus 50 itérations
et d'une tolérance de $10^{-6}$. Seule la première fréquence est appliquée. Le
résultat continu est arrondi vers le bas à la fréquence demandée profilée la
plus proche avant l'observation de l'état suivant. Le rejeu indique séparément
la fréquence réalisée médiane mesurée correspondante, puisque les valeurs
demandée et réalisée peuvent différer sous le plafond de puissance
expérimental.

Un résultat du solveur est rejeté s'il est irréalisable, non fini, ou s'il
arrive après le budget de commande de 0.8 seconde. Le repli est un gestionnaire
réactif à hystérésis : il augmente la fréquence d'un niveau du profil lorsque
la pression pondérée sur la file atteint huit ou que l'invite la plus ancienne
attend depuis une seconde, et la diminue d'un niveau lorsque la température
atteint 72 degrés C ou que la pression retombe à un. Chaque repli apparaît dans
la trajectoire enregistrée plutôt que d'être retiré des mesures agrégées.

La comparaison cherche à savoir si la replanification améliore la réponse à la
rafale décalée sans masquer son coût énergétique. Quatre correcteurs reçoivent
des requêtes et des conditions initiales identiques :

1. la fréquence maximale pendant toute l'expérience;
2. le gestionnaire réactif utilisé comme repli de la commande prédictive;
3. le programme en boucle ouverte optimisé pour la trace nominale;
4. la commande prédictive à horizon glissant.

```{code-cell} python
:tags: [remove-input]
:label: fig-inference-mpc
:caption: Commande de fréquence à fréquence maximale, réactive, en boucle ouverte et prédictive, sous la même trace de requêtes décalée et le même modèle réduit fixe de préremplissage par blocs entrelacés. Des courbes mesurées sur un NVIDIA L4 étalonnent chaque trajectoire simulée. À chaque seconde, le segment en pointillés de la commande prédictive est le plan courant sur dix secondes et le segment plein est l'historique des actions. Les plans ultérieurs ne sont pas révélés avant d'être calculés.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="mpc",
)))
```

:::{figure} _static/inference_serving/mpc.svg
:label: fig-inference-mpc-fallback
:class: pdf-fallback
:alt: Comparaison statique de la commande de fréquence à fréquence maximale, réactive, en boucle ouverte et à horizon glissant.

Vue statique de la comparaison en boucle fermée. Le livre en ligne ajoute la
lecture et montre le plan de la commande prédictive disponible sur l'horizon à
chaque temps de commande.
:::

Le tableau compare les quatre correcteurs de fréquence sous des arrivées
décalées identiques et le même modèle réduit fixe de préremplissage par blocs
entrelacés. Le nombre de replis compte chaque pas de la commande prédictive
qui n'a pas satisfait son exigence de faisabilité, de finitude ou de délai.

```{code-cell} python
:tags: [remove-input]

import pandas as pd

mpc_metrics = pd.read_csv(
    "artifacts/inference_serving/metrics_mpc.csv"
).set_index("controller")
mpc_metrics[
    [
        "mean_ttft_s",
        "p95_ttft_s",
        "mean_tpot_s",
        "energy_j",
        "peak_queued_requests",
        "ttft_violation_rate",
        "tpot_violation_rate",
        "power_violation_w",
        "thermal_violation_c",
        "fallback_count",
    ]
].T.rename_axis("metric").round(3)
```

{download}`Télécharger toutes les mesures en boucle fermée (CSV) <artifacts/inference_serving/metrics_mpc.csv>`

Sur la trace décalée, la fréquence maximale donne le plus faible délai moyen
avant le premier jeton, soit 10.93 secondes, pour 3 506.0 joules. Son 95e
centile est de 18.41 secondes. La commande prédictive réduit l'énergie à
3 341.8 joules, la plus faible des quatre correcteurs, tandis que la moyenne et
le 95e centile de son délai avant le premier jeton sont de 11.53 et de 19.01
secondes. Le gestionnaire réactif enregistre une moyenne de 12.51 secondes, un
95e centile de 20.01 secondes et 3 363.3 joules. Le plan hors ligne inchangé
enregistre 23.23 secondes, un 95e centile de 31.28 secondes et 3 646.8 joules.

La fréquence maximale et la commande prédictive ont toutes deux un temps moyen
par jeton de sortie de 0.0417 seconde. Le gestionnaire réactif enregistre
0.0419 seconde, et le plan hors ligne 0.0466 seconde. La file maximale compte
21 requêtes à fréquence maximale et 22 sous la commande prédictive. Elle monte
à 23 sous le gestionnaire réactif et à 30 sous le plan hors ligne. À l'horizon
de compte rendu de 60 secondes, la fréquence maximale et la commande
prédictive laissent chacune neuf requêtes inachevées. Le gestionnaire réactif
en laisse 11, et le plan hors ligne 22. Les 48 requêtes finissent toutes par
se terminer pendant la vidange qui suit l'horizon, dont l'énergie est incluse
dans les totaux rapportés.

Chaque correcteur viole la limite expérimentale du délai avant le premier jeton
pour chaque requête, et chacun a un taux de violation de 12.5 % pour le temps
par jeton de sortie. La commande prédictive accepte 67 résolutions sur tout le
déroulement et n'invoque aucun repli. Chaque résolution acceptée respecte
l'échéance de 0.8 seconde. Sur cette trace, la commande prédictive améliore à
la fois le délai moyen avant le premier jeton et l'énergie par rapport au
gestionnaire réactif et au plan hors ligne. La fréquence maximale reste plus
rapide de 0.60 seconde pour le délai moyen avant le premier jeton, mais
consomme 164.2 joules de plus que la commande prédictive.

Le tableau affiché porte sur la latence, l'énergie, les files d'attente, les
violations de contraintes et le nombre de replis. Le fichier CSV téléchargeable
indique aussi la latence de bout en bout, le débit, l'énergie par jeton de
sortie, la puissance et la température maximales, ainsi que le travail
inachevé. La comparaison distingue donc une réponse plus rapide d'une réponse
moins énergivore, au lieu d'attribuer un score unique à chaque correcteur.

Le profil mesuré fournit l'étalonnage du débit de service et de la puissance
par phase; les trajectoires des correcteurs sont des simulations d'un modèle de
service sur un seul GPU. Les quatre simulations atteignent une puissance de
phase modélisée de 64.852 W à la fréquence demandée la plus élevée. Cette
valeur provient de la moyenne mesurée de la phase de décodage à ce niveau du
profil et dépasse de 0.052 W le plafond configuré de 64.800 W. Ce plafond était
un réglage expérimental plutôt qu'une garantie stricte échantillon par
échantillon. Aucune des quatre simulations n'enregistre de violation thermique
ou de capacité du cache KV.

### Comparaison en direct à des cibles de latence communes

L'expérience en direct cherche quelle politique de fréquence consomme le moins
d'énergie GPU mesurée tout en respectant des limites de latence communes au
niveau des requêtes. On utilise des mesures de développement pour choisir une
fréquence fixe et pour fixer les limites sur le 95e centile du délai avant le
premier jeton et de la latence moyenne entre jetons par requête. La
confirmation prévue compare cette fréquence fixe, la fréquence maximale, le
gestionnaire et la commande prédictive avec son repli. Une fréquence constante
ou le gestionnaire peut être la politique à privilégier s'il respecte les deux
limites avec une énergie mesurée plus faible.

Le développement et la confirmation utilisent chacun 32 requêtes tirées de
portions disjointes de la trace enregistrée, selon des calendriers d'arrivée
réguliers et en rafales. Chaque fenêtre d'arrivée couvre au moins 240
secondes. Le GPU a un plafond de puissance de carte de 40 W et une fréquence
mémoire demandée de 6 251 MHz. Les débits de service et la puissance au repos
exigent de nouvelles mesures à ce plafond; l'étalonnage d'origine à 64.8 W
reste archivé. L'état en direct distingue les requêtes qui attendent leur
première sortie de celles qui sont déjà en cours de diffusion. Un modèle de
transition ajusté sur des données de service mixte doit réussir une validation
sur des exécutions complètes avant de pouvoir servir à la commande prédictive.
Ces mesures testent la vidange des files sous la charge de travail que le
correcteur rencontrera réellement.

Pour la confirmation, on prévoit cinq répétitions appariées par calendrier,
réparties sur trois acquisitions indépendantes, avec un ordre des méthodes
contrebalancé. Chaque paire utilise des requêtes et des instants d'arrivée
identiques. L'énergie comprend la vidange finale des requêtes; une affirmation
d'efficacité exige les cinq paires valides et que les deux méthodes respectent
les deux limites de latence figées. Les essais invalides et les décisions de
repli restent consignés. Une qualification de développement et un protocole de
comparaison figé sont encore nécessaires avant que la confirmation puisse
commencer.

Lors de deux acquisitions de développement antérieures, le GPU a traité toute
la charge régulière à 210 MHz, en consommant 8.55 et 8.30 kJ, mais n'a pas pu
revenir à ses températures de référence de démarrage avant l'expiration du
délai de refroidissement. Une qualification ultérieure réchauffe le GPU avec
toute la charge régulière avant de choisir sa référence dans une fenêtre de
refroidissement stable de 300 secondes. La moyenne de ses 60 dernières secondes
fixe la référence pour les essais restants; ce critère de stabilité n'établit
pas l'équilibre thermique.

Cette acquisition avec référence à chaud a qualifié les deux réglages demandés,
210 et 660 MHz, sur les deux profils d'arrivée désignés. Les 32 requêtes ont
été traitées de façon valide dans chacun des quatre essais. Le réchauffement
séparé ne compte pas pour la qualification. L'énergie a été de 7.65 et de
8.29 kJ pour les essais à 210 MHz en rafales et réguliers, et de 7.09 et de
8.43 kJ pour les essais correspondants à 660 MHz. Les étiquettes de fréquence
désignent des réglages demandés : la demande de 660 MHz a produit des
fréquences mesurées plus basses pendant l'inférence active, avec des médianes
de 360 MHz pour les arrivées en rafales et de 300 MHz pour les arrivées
régulières. Ces vérifications des points de fonctionnement n'établissent ni
une cible de latence commune ni un gagnant énergétique. Le développement du
correcteur et un protocole de comparaison figé restent nécessaires avant la
confirmation.

{download}`Télécharger le protocole de l'expérience en direct et les instructions d'exécution <artifacts/inference_serving/live_control_development/README.md>`

### Validation de la contrainte thermique

Dans le rejeu enregistré, le correcteur contraint une température de jonction
prédite par un modèle thermique à un état. Aucun des correcteurs simulés ne
viole cette contrainte, mais cela n'établit la faisabilité que pour le modèle
utilisé dans l'optimiseur.

Une paire de charges de travail de même puissance met en évidence le problème
de modélisation. Pendant l'expérience réservée à la validation, une impulsion
de décodage de 55 W et une impulsion de préremplissage de 55 W ont consommé
presque la même énergie électrique, avec une différence de 0.2 pour cent.
L'élévation maximale de température mesurée pendant le préremplissage a
néanmoins dépassé de 2.00 degrés C celle mesurée pendant le décodage. Un
modèle à un état piloté uniquement par la puissance de la carte reçoit presque
la même entrée pour les deux impulsions et n'a aucune variable pour représenter
cette différence.

Le balayage des fréquences utilisé pour étalonner le modèle de service a été
conçu pour mesurer le débit de service et la puissance. Son ajustement
thermique à un état avait $R^2=0.0020$; une expérience distincte a donc vérifié
si la phase de la charge de travail était nécessaire dans l'entrée thermique.
Les deux modèles candidats utilisaient

$$
\dot T(t)
=
-\frac{T(t)-T_{\mathrm{amb}}}{\tau}
+\frac{R}{\tau}P_{\mathrm{eff}}(t).
$$

Ici, $T_{\mathrm{amb}}$ est une température ambiante effective, $\tau$ la
constante de temps thermique et $R$ la résistance thermique en régime
permanent. Les deux modèles d'entrée diffèrent d'un terme :

$$
P_{\mathrm{eff}}(t)
=
\begin{cases}
P(t), & \text{modèle à puissance seule},\\
P(t)\bigl(1+\beta I_{\mathrm{prefill}}(t)\bigr),
& \text{modèle à gain de phase}.
\end{cases}
$$

L'indicatrice $I_{\mathrm{prefill}}$ vaut un pendant le préremplissage et zéro
pendant le décodage. Le modèle à gain de phase peut donc attribuer des entrées
thermiques effectives différentes à deux charges de travail de même puissance
de carte mesurée. La dimension de son état et son entrée mesurée restent
inchangées.

La séparation entre entraînement et validation utilisait des impulsions de
charge distinctes :

| Partition des données | Charges de travail et durée | Usage |
|---|---|---|
| Entraînement | Décodage et préremplissage à 46 W pendant 75 s et à 61 W pendant 45 s | Ajuster les deux modèles candidats |
| Validation | Décodage et préremplissage à 55 W pendant 60 s | Évaluer une seule fois les modèles figés |

L'ordre des quatre impulsions d'entraînement était contrebalancé. Chaque
impulsion commençait à partir d'une température vérifiée après le reverrouillage,
située dans une bande d'un degré Celsius. L'acquisition conservait un seuil de
mise en sécurité à 77 degrés C et un seuil d'interruption à 79 degrés C. Le fait
de faire correspondre la température de jonction mesurée n'établit pas que la
température non observée du dissipateur thermique correspondait aussi.

:::{figure} _static/inference_serving/thermal-phase-validation.svg
:label: fig-inference-thermal-validation
:alt: Élévations de température mesurées en décodage et en préremplissage sur un NVIDIA L4, comparées aux modèles thermiques figés à un état, à puissance seule et à gain de phase, suivies de trois erreurs de validation par rapport à une limite d'acceptation d'un degré C.

Le modèle à puissance seule prédit des réponses semblables pour les impulsions
de même puissance. Le modèle à gain de phase les sépare et réduit la RMSE sur
les données réservées, mais ses plus grandes erreurs de trajectoire et de
contraste entre phases restent au-dessus de la limite d'acceptation fixée à un
degré Celsius. Les données de validation n'ont servi à ajuster aucun des deux
modèles.
:::

:::{include} artifacts/inference_serving/thermal_phase_result.md
:::

Le gain ajusté attribue au préremplissage une entrée thermique effective 1.147
fois celle du décodage. Ce paramètre est reproductible sur les dix départs de
l'optimisation. La jacobienne d'entraînement est aussi de rang numérique plein,
avec un conditionnement de 384. Ces vérifications réduisent la crainte d'un
ajustement de paramètre ambigu, mais elles n'établissent pas qu'un seul état
thermique suffit. Les résidus de validation ratent l'élévation initiale rapide
de la température, puis changent de signe. L'état thermique hérité, la
synchronisation et le filtrage du capteur et un échauffement dépendant de la
phase restent des explications possibles; ces mesures par impulsions n'isolent
pas leurs contributions.

#### Ajustement sur l'historique thermique complet

La comparaison de développement élargie retient un modèle thermique à deux
états. Elle ajoute aux cinq acquisitions historiques le projet pilote terminé
du 4 septembre, y compris son essai de service soutenu qui a échoué. Les six
acquisitions, y compris les données de validation antérieures, sont des
données de développement pour cette comparaison.

Chaque acquisition conserve son historique de température continu pendant le
refroidissement, le reverrouillage de la mémoire et les segments de charge. Le
modèle à deux états transporte une température non observée du dissipateur à
travers ces transitions, au lieu de la réinitialiser à la température de
jonction mesurée à chaque impulsion. L'ajustement compare les prédictions aux
intervalles d'un degré Celsius du capteur. Chaque évaluation réserve une
acquisition complète; le tableau fait la moyenne de sa RMSE à dix secondes à
parts égales sur les six acquisitions. Ces prédictions utilisent la puissance
de carte future mesurée; elles testent donc le modèle de température lorsque
son entrée future est fournie.

| Modèle thermique | RMSE moyenne à dix secondes pondérée par exécution (degrés C) |
|---|---:|
| Un état | 0.726 |
| Un état avec gain de phase, diagnostic seulement | 0.722 |
| Deux états | 0.540 |
| Deux états avec gain de phase, diagnostic seulement | 0.518 |

La règle de sélection inchangée exige une amélioration d'au moins 0.1 degré C
et de dix pour cent pour ajouter un état, sans détériorer sensiblement
l'erreur à une seconde ni la sous-prédiction à dix secondes. Le modèle à deux
états améliore la RMSE moyenne de 0.186 degré C, soit 25.58 pour cent, et
satisfait ces garde-fous. Ses modes ajustés sont d'environ 1.72 et 154
secondes. Ils décrivent la réponse du modèle sous l'alignement enregistré du
capteur; ils n'établissent pas des constantes de temps physiques identifiées
de façon unique. Les modèles à gain de phase restent diagnostiques, car le
service mixte ne fournit aucun signal d'activité de phase à l'exécution.

Le gain est plus faible dans l'essai soutenu qui a échoué : la RMSE à dix
secondes passe de 0.907 à 0.825 degré C, sur seulement neuf origines de
prévision. Le maximum brut du capteur est de 77 degrés C, tandis que les
observations conservées à une seconde culminent à 76. Le modèle retenu a été
figé avant l'acquisition à 40 W. Une évaluation sur de nouvelles données exige
encore des prévisions de puissance disponibles au moment où le correcteur agit.
Le modèle n'est toujours pas accepté pour des prévisions consultatives en ligne
ni pour des contraintes thermiques dures; des chiens de garde fondés sur la
température mesurée fonctionnent de façon indépendante.

{download}`Télécharger le rapport de développement thermique sur six acquisitions (JSON) <artifacts/inference_serving/live_control_development/thermal_history_development_live_v2.json>`

L'analyse d'origine sur cinq acquisitions conservait un seul état, car ajouter
un second état n'améliorait la RMSE que de 0.069 degré C, sous le même seuil
absolu. Ses données et son rapport restent inchangés.

{download}`Télécharger le rapport d'origine sur cinq acquisitions (JSON) <artifacts/inference_serving/live_control_development/thermal_history_development.json>`

Pour la simulation enregistrée de la commande prédictive, l'état de
température et l'inégalité restent un modèle pédagogique. L'absence de
violation thermique simulée n'est pas un certificat de sécurité matérielle. La
prévision des arrivées est volontairement simple, la longueur de sortie n'est
observée qu'à la fin de la requête, et l'ordonnanceur est maintenu fixe. Le
délai réseau, la segmentation en jetons, les communications entre plusieurs GPU
et les effets sur la qualité du modèle sont omis. La rafale décalée est une
seule perturbation contrôlée; la performance relative sur cette trace
n'établit donc pas une dominance sous d'autres charges de travail.

{download}`Télécharger les trajectoires réservées à une seconde et les prédictions des modèles figés (CSV) <artifacts/inference_serving/thermal_phase_validation.csv>`

{download}`Télécharger le rapport complet d'ajustement thermique et d'acceptation (JSON) <data/inference_serving/thermal-phase-identification-20260903T131518Z/thermal_phase_fit_report.json>`

{download}`Télécharger la télémétrie de validation mesurée (CSV) <data/inference_serving/thermal-phase-identification-20260903T131518Z/l4_thermal_phase_telemetry.csv>`

:::{dropdown} Examiner le correcteur à horizon glissant
```{literalinclude} code/inference_control.py
:language: python
:start-at: def _solve(
:end-before: def __call__(
:linenos:
```

```{literalinclude} code/inference_control.py
:language: python
:start-at: def run_mpc
:end-before: def shift_largest_burst
:linenos:
```

{download}`Télécharger l'implémentation complète de la commande d'inférence <code/inference_control.py>`
:::

## Synthèse et perspectives

La poursuite, la régulation, la commande économique, la robustesse, les
prédictions stochastiques et les décisions hybrides conservent toutes la même
ossature à horizon glissant, tout en modifiant l'objectif, le modèle
d'incertitude ou l'ensemble réalisable. Les variables d'écart, les gouverneurs
de référence, les correcteurs de secours et les cascades déterminent le
comportement du correcteur lorsque le problème d'optimisation privilégié est
en retard ou irréalisable.

Une exécution fiable exige tout de même de résoudre un problème d'optimisation
très semblable à chaque pas. La structure répétée peut-elle être réutilisée,
voire approchée par une application apprise des paramètres vers les actions?
Le chapitre [Correcteurs paramétriques et
approchés](parametric-controllers.md) développe ces solutions de rechange.

## Autoévaluation

:::{exercise} Échéance du solveur
:label: ex-mpc-check-4

Pourquoi la commande prédictive de l'inférence rejette-t-elle une solution
réalisable renvoyée après son échéance de 0.8 seconde?
:::

:::{solution} ex-mpc-check-4
:class: dropdown

La décision est calculée pour l'état au début d'un intervalle d'une seconde.
Après l'échéance, il reste peu de temps pour l'appliquer, et la file des
requêtes peut déjà avoir changé. Le repli réactif fournit une action à temps,
dont les bornes sont connues.
:::

:::{exercise} Validation du modèle thermique
:label: ex-mpc-check-5

Le modèle thermique à gain de phase s'ajuste de façon reproductible à ses
quatre impulsions d'entraînement et prédit les trajectoires réservées mieux que
le modèle à puissance seule. Il échoue pourtant à l'un des trois critères de
validation fixés à l'avance. Sa température prédite peut-elle servir de
contrainte dure de sécurité matérielle? Qu'est-ce qui reste utile dans
l'expérience?
:::

:::{solution} ex-mpc-check-5
:class: dropdown

Non. Une estimation reproductible d'un paramètre n'établit pas que la classe de
modèles capture la dynamique pertinente du procédé. Le critère réservé qui a
échoué empêche la contrainte de servir de certificat de sécurité. L'expérience
reste utile, car elle mesure un écart entre des charges de travail de même
puissance et quantifie l'amélioration des prédictions qu'apporte une entrée
dépendant de la phase. Elle n'identifie pas la cause physique : l'état
thermique hérité ainsi que la synchronisation ou le filtrage du capteur
restent des explications concurrentes à tester avec des historiques
d'acquisition complets.
:::
