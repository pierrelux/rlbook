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
# Correcteurs paramétriques et approchés

La commande prédictive en temps réel impose des limites strictes au calcul. Dans des applications comme l'optique adaptative, le correcteur doit fonctionner à des cadences de l'ordre du kilohertz. Une fréquence d'échantillonnage de 1000 Hz ne laisse qu'une milliseconde par pas pour calculer et appliquer une entrée de commande. L'efficacité devient donc une préoccupation de premier plan.

La structure de la commande prédictive se prête naturellement à la réutilisation de l'optimisation. Chaque pas de temps exige la résolution d'un problème ayant la même dynamique et les mêmes contraintes. Seuls l'état initial, les prévisions ou les signaux de référence changent. Plutôt que de traiter chaque instance comme un nouveau problème, on peut formuler la commande prédictive comme un *problème d'optimisation paramétrique* et s'intéresser à la façon dont la solution évolue avec le paramètre.

La solution précédente, sa sensibilité locale ou une approximation apprise
peuvent-elles remplacer une partie de l'optimisation en ligne suivante sans
masquer le moment où cette approximation cesse d'être valide?

## Cadre général : l'optimisation paramétrique

Comment l'optimum et la valeur optimale changent-ils lorsque le paramètre
d'état initial, de prévision ou de référence se déplace?

On part d'un problème d'optimisation général indexé par un paramètre $\boldsymbol{\theta} \in \Theta \subset \mathbb{R}^p$ :

$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} \quad & f(\mathbf{x}; \boldsymbol{\theta}) \\
\text{s.c.} \quad & \mathbf{g}(\mathbf{x}; \boldsymbol{\theta}) \le \mathbf{0}, \\
& \mathbf{h}(\mathbf{x}; \boldsymbol{\theta}) = \mathbf{0}.
\end{aligned}
$$

Pour chaque valeur de $\boldsymbol{\theta}$, on obtient un problème d'optimisation concret. Le but est de comprendre comment l'optimum $\mathbf{x}^\star(\boldsymbol{\theta})$ et la fonction de valeur

$$
v(\boldsymbol{\theta}) := \inf\{\, f(\mathbf{x}; \boldsymbol{\theta}) : \mathbf{x} \text{ réalisable en } \boldsymbol{\theta}\,\}
$$

dépendent de $\boldsymbol{\theta}$.

Lorsque le problème est lisse et régulier, les conditions de Karush-Kuhn-Tucker (KKT) caractérisent l'optimalité :

$$
\begin{aligned}
\nabla_{\mathbf{x}} f(\mathbf{x}; \boldsymbol{\theta})
+ \nabla_{\mathbf{x}} \mathbf{g}(\mathbf{x}; \boldsymbol{\theta})^\top \boldsymbol{\lambda}
+ \nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}; \boldsymbol{\theta})^\top \boldsymbol{\nu} &= 0, \\
\mathbf{g}(\mathbf{x}; \boldsymbol{\theta}) \le 0, \quad
\boldsymbol{\lambda} \ge 0, \quad
\lambda_i g_i(\mathbf{x}; \boldsymbol{\theta}) &= 0, \\
\mathbf{h}(\mathbf{x}; \boldsymbol{\theta}) &= 0.
\end{aligned}
$$

Si l'ensemble actif reste fixe lorsque $\boldsymbol{\theta}$ varie, le théorème des fonctions implicites garantit que les applications

$$
\boldsymbol{\theta} \mapsto \mathbf{x}^\star(\boldsymbol{\theta}), \quad
\boldsymbol{\theta} \mapsto \boldsymbol{\lambda}^\star(\boldsymbol{\theta}), \quad
\boldsymbol{\theta} \mapsto \boldsymbol{\nu}^\star(\boldsymbol{\theta})
$$

sont différentiables.

En programmation linéaire et quadratique, cette structure devient encore plus facile à exploiter. Considérons un programme linéaire qui dépend de façon affine de $\boldsymbol{\theta}$ :

$$
\min_{\mathbf{x}} \ \mathbf{c}(\boldsymbol{\theta})^\top \mathbf{x}
\quad \text{s.c.} \quad \mathbf{A}(\boldsymbol{\theta})\mathbf{x} \le \mathbf{b}(\boldsymbol{\theta}).
$$

Chaque ensemble actif détermine une base, et donc une région de $\Theta$ dans laquelle la solution est affine en $\boldsymbol{\theta}$. L'espace des paramètres réalisables est partitionné en régions polyédrales, chacune munie de sa propre loi affine.

De même, dans les programmes quadratiques strictement convexes

$$
\min_{\mathbf{x}} \ \tfrac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x} + \mathbf{q}(\boldsymbol{\theta})^\top \mathbf{x}
\quad \text{s.c.} \quad \mathbf{A}\mathbf{x} \le \mathbf{b}(\boldsymbol{\theta}), \qquad \mathbf{H} \succ 0,
$$

chaque ensemble actif conduit encore à un optimum affine, avec une structure globale affine par morceaux et une fonction de valeur quadratique par morceaux.

La programmation paramétrique s'intéresse à la structure de l'application $\boldsymbol{\theta} \mapsto \mathbf{x}^\star(\boldsymbol{\theta})$ et aux régions sur lesquelles cette application prend une forme simple.

### Sensibilité de la solution par le théorème des fonctions implicites

On rencontre souvent des équations de la forme

$$
F(y,\boldsymbol{\theta})=0,
$$

où $y\in\mathbb{R}^m$ sont les inconnues et $\boldsymbol{\theta}\in\mathbb{R}^p$ les paramètres. Le **théorème des fonctions implicites** affirme que, si $F$ est lisse et que la jacobienne par rapport à $y$,

$$
\frac{\partial F}{\partial y}(y^\star,\boldsymbol{\theta}^\star),
$$

est inversible en une solution $(y^\star,\boldsymbol{\theta}^\star)$, alors il existe, dans un voisinage de $\boldsymbol{\theta}^\star$, une unique application lisse $y(\boldsymbol{\theta})$ telle que $F(y(\boldsymbol{\theta}),\boldsymbol{\theta})=0$ et $y(\boldsymbol{\theta}^\star)=y^\star$. De plus, sa dérivée est

$$
\frac{d y}{d\boldsymbol{\theta}}(\boldsymbol{\theta}^\star)
\;=\;
-\Big(\tfrac{\partial F}{\partial y}(y^\star,\boldsymbol{\theta}^\star)\Big)^{-1}
\;\tfrac{\partial F}{\partial \boldsymbol{\theta}}(y^\star,\boldsymbol{\theta}^\star).
$$

Autrement dit : si la jacobienne carrée en $y$ est non singulière, la solution varie de façon lisse avec le paramètre, et on peut la dériver en résolvant un seul système linéaire.

Revenons à $(P_{\theta})$ et à son système KKT. Regroupons les variables primales et duales en

$$
y \;:=\; (\mathbf{x},\,\boldsymbol{\lambda},\,\boldsymbol{\nu}),
$$

et écrivons les équations KKT comme un seul résidu

$$
F(y,\boldsymbol{\theta}) \;=\; 
\begin{bmatrix}
\nabla_{\mathbf{x}} f(\mathbf{x};\boldsymbol{\theta})
+ \nabla_{\mathbf{x}} \mathbf{g}(\mathbf{x};\boldsymbol{\theta})^\top \boldsymbol{\lambda}
+ \nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x};\boldsymbol{\theta})^\top \boldsymbol{\nu} \\
\mathbf{h}(\mathbf{x};\boldsymbol{\theta}) \\
\mathbf{g}_\mathcal{A}(\mathbf{x};\boldsymbol{\theta})
\end{bmatrix}
\;=\; \mathbf{0}.
$$

Ici, $\mathcal{A}$ désigne l'ensemble des contraintes d'inégalité actives à la solution (la partie de complémentarité est encodée en gardant $\mathcal{A}$ fixe; voir plus bas).

Pour invoquer le théorème des fonctions implicites, il faut que la jacobienne $\partial F/\partial y$ soit inversible en $(y^\star,\boldsymbol{\theta}^\star)$. Les conditions de régularité habituelles qui l'assurent sont :

* **LICQ (qualification des contraintes par indépendance linéaire)** en $(\mathbf{x}^\star,\boldsymbol{\theta}^\star)$ : les gradients de toutes les contraintes actives sont linéairement indépendants.
* **Condition suffisante du second ordre** sur le cône critique (le hessien du lagrangien est défini positif sur les directions réalisables).
* **Complémentarité stricte** (facultative mais commode) : chaque inégalité active a un multiplicateur strictement positif.

Sous ces conditions, la **matrice KKT**,

$$
K \;=\;
\frac{\partial F}{\partial y}(y^\star,\boldsymbol{\theta}^\star)
\;=\;
\begin{bmatrix}
\nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L}(\mathbf{x}^\star,\boldsymbol{\lambda}^\star,\boldsymbol{\nu}^\star;\boldsymbol{\theta}^\star)
& \nabla_{\mathbf{x}} \mathbf{g}_\mathcal{A}(\mathbf{x}^\star;\boldsymbol{\theta}^\star)^\top
& \nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}^\star;\boldsymbol{\theta}^\star)^\top \\
\nabla_{\mathbf{x}} \mathbf{g}_\mathcal{A}(\mathbf{x}^\star;\boldsymbol{\theta}^\star) & 0 & 0 \\
\nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}^\star;\boldsymbol{\theta}^\star) & 0 & 0
\end{bmatrix},
$$

est non singulière. Ici, $\mathcal{L}=f+\boldsymbol{\lambda}^\top \mathbf{g}+\boldsymbol{\nu}^\top \mathbf{h}$.

La sensibilité du membre de droite aux paramètres est

$$
G \;=\; \frac{\partial F}{\partial \boldsymbol{\theta}}(y^\star,\boldsymbol{\theta}^\star)
\;=\;
\begin{bmatrix}
\nabla_{\boldsymbol{\theta}}\nabla_{\mathbf{x}} f
+ \sum_{i\in\mathcal{A}} \lambda_i^\star \nabla_{\boldsymbol{\theta}}\nabla_{\mathbf{x}} g_i
+ \sum_j \nu_j^\star \nabla_{\boldsymbol{\theta}}\nabla_{\mathbf{x}} h_j \\
\nabla_{\boldsymbol{\theta}} \mathbf{h} \\
\nabla_{\boldsymbol{\theta}} \mathbf{g}_\mathcal{A}
\end{bmatrix}_{(\mathbf{x}^\star,\boldsymbol{\theta}^\star)} .
$$

Le théorème des fonctions implicites donne alors la **différentiabilité locale de l'optimum et des multiplicateurs** :

$$
\frac{d y^\star}{d\boldsymbol{\theta}}(\boldsymbol{\theta}^\star)
\;=\; -\,K^{-1} G.
$$

La formule ci-dessus est valide **tant que l'ensemble actif $\mathcal{A}$ ne change pas**. Si une contrainte passe d'active à inactive ou l'inverse, l'application reste lisse par morceaux, mais la dérivée peut sauter. En commande prédictive, c'est précisément pourquoi les démarrages à chaud sont très efficaces la plupart du temps et exigent à l'occasion une nouvelle factorisation lorsque l'ensemble actif bascule.

En commande prédictive paramétrique, $\boldsymbol{\theta}$ regroupe l'état courant, les références et les prévisions. Le théorème des fonctions implicites nous dit que, sous des hypothèses de régularité et avec un ensemble actif stable, la trajectoire optimale et la première entrée varient de façon lisse avec $\boldsymbol{\theta}$. L'application linéaire $-K^{-1}G$ est exactement l'objet utilisé dans les démarrages à chaud fondés sur la sensibilité et dans les itérations en temps réel : de petites variations de $\boldsymbol{\theta}$ peuvent être propagées au moyen d'une seule résolution KKT pour mettre à jour l'estimation primale-duale avant d'effectuer un ou deux pas de Newton ou de PQS.

### Commande prédictive par prédiction-correction

On part d'un problème lisse de recherche de racine

$$
F(y)=0,\qquad F:\mathbb{R}^m\to\mathbb{R}^m.
$$

La **méthode de Newton** itère

$$
y^{(t+1)} \;=\; y^{(t)} - \big[\nabla F(y^{(t)})\big]^{-1} F\big(y^{(t)}\big),
$$

ou, de façon équivalente, résout le système linéarisé

$$
\nabla F(y^{(t)})\,\Delta y^{(t)} = -F\big(y^{(t)}\big),\qquad y^{(t+1)}=y^{(t)}+\Delta y^{(t)}.
$$

La convergence est locale et rapide lorsque la jacobienne est non singulière et que l'estimation initiale est proche.

Supposons maintenant que la racine dépende d'un paramètre :

$$
F\big(y,\theta\big)=0,\qquad \theta\in\mathbb{R}.
$$

On veut le chemin des solutions $\theta\mapsto y^\star(\theta)$. La **continuation numérique** fait avancer $\theta$ par petits pas et utilise la solution précédente comme démarrage à chaud pour la résolution de Newton suivante. C'est la façon la plus simple et la plus efficace de « suivre » les solutions de systèmes paramétriques.

En une solution connue $(y^\star,\theta^\star)$, on dérive $F(y^\star(\theta),\theta)=0$ par rapport à $\theta$ :

$$
\nabla_y F(y^\star,\theta^\star)\,\frac{dy^\star}{d\theta}(\theta^\star) \;+\; \nabla_\theta F(y^\star,\theta^\star) \;=\; 0.
$$

Si $\nabla_y F$ est inversible (conditions du théorème des fonctions implicites), la **tangente** est

$$
\frac{dy^\star}{d\theta}(\theta^\star) \;=\; -\big[\nabla_y F(y^\star,\theta^\star)\big]^{-1}\,\nabla_\theta F(y^\star,\theta^\star).
$$

C'est exactement la formule de **dérivation implicite**. La continuation l'utilise comme **prédicteur** :

$$
y_{\text{préd}} \;=\; y^\star(\theta^\star) \;+\; \Delta\theta\;\frac{dy^\star}{d\theta}(\theta^\star).
$$

Quelques pas **correcteurs** appliquent ensuite la méthode de Newton à $F(\,\cdot\,,\theta^\star+\Delta\theta)=0$ à partir de $y_{\text{préd}}$. Si la méthode de Newton converge rapidement, le pas $\Delta\theta$ était approprié; sinon, on réduit $\Delta\theta$ et on recommence.

Pour les systèmes KKT paramétriques, on pose $y=(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\nu})$, où $\mathbf{x}$ empile les variables de décision primales (états et entrées), et $F(y,\theta)=0$ le résidu KKT, où $\theta$ regroupe l'état, les références et les prévisions. La **matrice KKT** $K=\partial F/\partial y$ et la **sensibilité aux paramètres** $G=\partial F/\partial \theta$ donnent la tangente

$$
\frac{dy^\star}{d\theta} \;=\; -\,K^{-1}G.
$$

La continuation devient alors :

1. **Prédicteur** : $y_{\text{préd}} = y^\star + (\Delta\theta)\,(-K^{-1}G)$.
2. **Correcteur** : quelques pas de Newton ou de PQS sur les équations KKT au nouveau $\theta$.

En commande prédictive, on obtient ainsi des **démarrages à chaud** efficaces d'un instant à l'autre. Lorsque le paramètre $\theta_t$ (état courant et références) change légèrement, on prédit le nouveau point primal-dual et on le corrige en 1 ou 2 itérations, ce qui suffit souvent à atteindre la tolérance en temps réel.



## Optimisation amortie et approximation neuronale des correcteurs

Une fonction entraînée peut-elle approcher l'optimiseur paramétrique d'assez
près pour remplacer la plupart des résolutions en ligne?

L'idée de réutiliser la structure entre des problèmes d'optimisation semblables n'est pas propre à la programmation paramétrique. En apprentissage automatique, un concept apparenté, l'**optimisation amortie**, vise à réduire le coût d'inférences répétées en remplaçant l'optimisation explicite par une fonction *apprise* pour approcher l'application solution. Cette approche déplace la charge de calcul de la résolution en ligne vers l'entraînement hors ligne.

Le but est de construire une fonction $\hat{\pi}_{\phi}(\boldsymbol{\theta})$, habituellement paramétrée par un réseau de neurones, qui envoie l'entrée $\boldsymbol{\theta}$ vers une solution approchée $\hat{z}^\star(\boldsymbol{\theta})$ ou une action de commande $\hat{\mathbf{u}}_0^\star(\boldsymbol{\theta})$. Une fois entraînée, cette application peut être évaluée rapidement à l'exécution, sans qu'il soit nécessaire de résoudre explicitement un problème d'optimisation.

L'optimisation amortie est apparue dans plusieurs contextes :

* En **inférence probabiliste**, où les autoencodeurs variationnels amortissent le calcul des lois a posteriori sur un jeu de données.
* En **méta-apprentissage**, où l'objectif est d'apprendre un modèle qui se généralise d'une tâche à l'autre en intériorisant la manière de s'adapter.
* En **optimisation des hyperparamètres**, où l'apprentissage d'un modèle de substitution peut guider efficacement la recherche dans l'espace des configurations.

Ce point de vue commence aussi à influencer la commande. Des travaux récents étudient comment **amortir les politiques de commande prédictive non linéaire** dans des réseaux de neurones. Les données d'entraînement proviennent de la résolution hors ligne de nombreuses instances du problème de contrôle optimal sous-jacent. La politique neuronale obtenue $\hat{\pi}_\phi$ agit comme un correcteur différentiable à faible latence, capable de se généraliser à de nouvelles situations à l'intérieur de la distribution d'entraînement.

Par rapport à la commande prédictive explicite, qui partitionne l'espace des paramètres et stocke les solutions exactes région par région, la commande amortie lisse le domaine en apprenant globalement une politique approchée. Elle est moins précise, mais elle passe à l'échelle de problèmes de grande dimension où l'énumération des régions est impossible.

L'amortissement par réseau de neurones est avantageux grâce à l'expressivité de ces modèles. La difficulté consiste toutefois à assurer le **respect des contraintes et la sécurité**, difficiles à garantir avec des approximateurs neuronaux sans contraintes. Des approches hybrides tentent de résoudre ce problème en combinant une politique neuronale de démarrage à chaud avec une étape finale de projection, ou en intégrant le réseau dans une couche d'optimisation sous contraintes. D'autres stratégies consistent à apprendre des architectures structurées qui respectent la physique connue ou les symétries de la commande.


## Cadre de l'apprentissage par imitation

Quelle distribution d'états et quelle cible supervisée permettent à un
correcteur approché de reproduire l'optimiseur là où il sera réellement
déployé?
Considérons un horizon fixe $N$ et un vecteur de paramètres $\boldsymbol{\theta}$ qui encode l'état courant, les références et les prévisions. Le correcteur prédictif oracle résout

$$
\begin{aligned}
z^\star(\boldsymbol{\theta}) \in \arg\min_{z=(\mathbf{x}_{0:N},\mathbf{u}_{0:N-1})}
&\; J(z;\boldsymbol{\theta})\\
\text{s.c. }& \mathbf{x}_{k+1}=f(\mathbf{x}_k,\mathbf{u}_k;\boldsymbol{\theta}),\quad k=0..N-1,\\
& g(\mathbf{x}_k,\mathbf{u}_k;\boldsymbol{\theta})\le 0,\; h(\mathbf{x}_N;\boldsymbol{\theta})=0.
\end{aligned}
$$

L'action appliquée est $\pi^\star(\boldsymbol{\theta}) := \mathbf{u}_0^\star(\boldsymbol{\theta})$. Notre but est d'apprendre une application de substitution rapide $\hat{\pi}_\phi:\boldsymbol{\theta}\mapsto \hat{\mathbf{u}}_0 \approx \pi^\star(\boldsymbol{\theta})$, évaluable en quelques microsecondes et éventuellement suivie d'une couche de projection de sécurité.

**Apprentissage supervisé à partir des solutions de l'oracle.**
On échantillonne d'abord des paramètres $\boldsymbol{\theta}^{(i)}$ dans le domaine d'exploitation et on résout hors ligne les problèmes de commande prédictive non linéaire correspondants. Le jeu de données obtenu

$$
\mathcal{D} = \{ (\boldsymbol{\theta}^{(i)},\, \mathbf{u}_0^\star(\boldsymbol{\theta}^{(i)})) \}_{i=1}^M
$$

sert ensuite à entraîner un réseau de neurones $\hat{\pi}_\phi$ en minimisant

$$
\min_\phi \; \frac{1}{M}\sum_{i=1}^M \big\|\hat{\pi}_\phi(\boldsymbol{\theta}^{(i)}) - \mathbf{u}_0^\star(\boldsymbol{\theta}^{(i)})\big\|^2 .
$$

Une fois entraîné, le réseau sert de substitut à l'optimiseur et fournit des évaluations instantanées qui approchent la loi de la commande prédictive.


## Exemple : commande de la perfusion de propofol

Comment un correcteur paramétrique à horizon glissant traduit-il l'état mesuré
d'un patient en une décision de perfusion sous contraintes?

Ce problème porte sur la commande de la perfusion de propofol en anesthésie intraveineuse totale. Notre présentation suit la formulation du problème proposée par {cite:t}`Sawaguchi2008`. L'objectif principal est de maintenir le niveau d'inconscience souhaité tout en réduisant au minimum les réactions indésirables et en assurant un réveil rapide après la chirurgie. 

Le niveau d'inconscience est mesuré par l'indice bispectral (BIS), obtenu au moyen d'un appareil d'électroencéphalographie (EEG). Le BIS va de $0$ (suppression complète de l'activité cérébrale) à $100$ (éveil complet), la plage cible pour l'anesthésie générale se situant habituellement entre $40$ et $60$.

Le but est de concevoir un système de commande qui règle le débit de perfusion du propofol pour maintenir le BIS dans la plage cible. On peut le formuler comme un problème de contrôle optimal :

$$
\begin{align*}
\min_{u(t)} & \int_{0}^{T} \left( BIS(t) - BIS_{\text{cible}} \right)^2 + \lambda\, u(t)^2 \, dt \\
\text{sous les contraintes :} \\
\dot{x}_1 &= -(k_{10} + k_{12} + k_{13})x_1 + k_{21}x_2 + k_{31}x_3 + \frac{u(t)}{V_1} \\
\dot{x}_2 &= k_{12}x_1 - k_{21}x_2 \\
\dot{x}_3 &= k_{13}x_1 - k_{31}x_3 \\
\dot{x}_e &= k_{e0}(x_1 - x_e) \\
BIS(t) &= E_0 - E_{\text{max}}\frac{x_e^\gamma}{x_e^\gamma + EC_{50}^\gamma}
\end{align*}
$$

où :
- $u(t)$ est le débit de perfusion du propofol (mg/kg/h)
- $x_1$, $x_2$ et $x_3$ sont les concentrations du médicament dans différents compartiments du corps
- $x_e$ est la concentration au site d'effet
- les $k_{ij}$ sont les constantes de vitesse du transfert du médicament entre compartiments
- $BIS(t)$ est l'indice bispectral
- $\lambda$ est un paramètre de régularisation qui pénalise l'usage excessif du médicament
- $E_0$, $E_{\text{max}}$, $EC_{50}$ et $\gamma$ sont des paramètres du modèle pharmacodynamique

Le modèle de dynamique utilisé dans ce problème est le « modèle pharmacocinétique-pharmacodynamique », qui comprend trois composantes principales :

1. Le **modèle pharmacocinétique**, qui décrit comment le médicament se distribue dans le corps au fil du temps. Il repose sur un modèle à trois compartiments :
   - le compartiment central (le sang et les organes bien perfusés)
   - le compartiment périphérique superficiel (les muscles et d'autres tissus)
   - le compartiment périphérique profond (la graisse)

2. Le **modèle du site d'effet**, qui représente le retard entre la concentration du médicament dans le sang et son effet sur le cerveau.

3. Le **modèle pharmacodynamique**, qui relie la concentration au site d'effet au BIS observé.

Le problème de la commande de la perfusion de propofol présente plusieurs difficultés intéressantes du point de vue de la recherche. 
D'abord, le médicament met un certain temps à atteindre les différents compartiments, et les mesures du BIS peuvent aussi être en retard. Cela pourrait mener à de l'instabilité si la conception de la commande n'en tient pas compte adéquatement. 

De plus, chaque patient est différent des autres. On ne peut donc pas simplement apprendre hors ligne un seul correcteur en espérant qu'il se généralise à toute une population de patients. On tiendra compte de cette variabilité au moyen de la commande prédictive et on s'adaptera dynamiquement à l'écart de modèle par la replanification. La réaction d'un patient à une dose donnée de médicament varie aussi et doit être soigneusement maîtrisée pour éviter les surdoses. Cela ajoute un niveau de complexité, puisqu'il faut intégrer des contraintes de sécurité. Enfin, l'état du patient peut changer soudainement, par exemple à cause de stimulations chirurgicales, et le correcteur doit pouvoir s'adapter rapidement pour compenser la perturbation du système.

```{code-cell} python
:tags: [hide-input]


#  label: fig-mpc-propofol
#  caption: Closed-loop MPC for propofol infusion keeps the Bispectral Index near the target (top), regulates infusion rates (middle), and tracks the effect-site concentration (bottom).

%config InlineBackend.figure_format = 'retina'
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults

class Patient:
    def __init__(self, age, weight):
        self.age = age
        self.weight = weight
        self.set_pk_params()
        self.set_pd_params()

    def set_pk_params(self):
        self.v1 = 4.27 * (self.weight / 70) ** 0.71 * (self.age / 30) ** (-0.39)
        self.v2 = 18.9 * (self.weight / 70) ** 0.64 * (self.age / 30) ** (-0.62)
        self.v3 = 238 * (self.weight / 70) ** 0.95
        self.cl1 = 1.89 * (self.weight / 70) ** 0.75 * (self.age / 30) ** (-0.25)
        self.cl2 = 1.29 * (self.weight / 70) ** 0.62
        self.cl3 = 0.836 * (self.weight / 70) ** 0.77
        self.k10 = self.cl1 / self.v1
        self.k12 = self.cl2 / self.v1
        self.k13 = self.cl3 / self.v1
        self.k21 = self.cl2 / self.v2
        self.k31 = self.cl3 / self.v3
        self.ke0 = 0.456

    def set_pd_params(self):
        self.E0 = 100
        self.Emax = 100
        self.EC50 = 3.4
        self.gamma = 3

def pk_model(x, u, patient):
    x1, x2, x3, xe = x
    dx1 = -(patient.k10 + patient.k12 + patient.k13) * x1 + patient.k21 * x2 + patient.k31 * x3 + u / patient.v1
    dx2 = patient.k12 * x1 - patient.k21 * x2
    dx3 = patient.k13 * x1 - patient.k31 * x3
    dxe = patient.ke0 * (x1 - xe)
    return np.array([dx1, dx2, dx3, dxe])

def pd_model(ce, patient):
    return patient.E0 - patient.Emax * (ce ** patient.gamma) / (ce ** patient.gamma + patient.EC50 ** patient.gamma)

def simulate_step(x, u, patient, dt):
    x_next = x + dt * pk_model(x, u, patient)
    bis = pd_model(x_next[3], patient)
    return x_next, bis

def objective(u, x0, patient, dt, N, target_bis):
    x = x0.copy()
    total_cost = 0
    for i in range(N):
        x, bis = simulate_step(x, u[i], patient, dt)
        total_cost += (bis - target_bis)**2 + 0.1 * u[i]**2
    return total_cost

def mpc_step(x0, patient, dt, N, target_bis):
    u0 = 10 * np.ones(N)  # Initial guess
    bounds = [(0, 20)] * N  # Infusion rate between 0 and 20 mg/kg/h
    
    result = minimize(objective, u0, args=(x0, patient, dt, N, target_bis),
                      method='SLSQP', bounds=bounds)
    
    return result.x[0]  # Return only the first control input

def run_mpc_simulation(patient, T, dt, N, target_bis):
    rng = np.random.default_rng(2026)
    steps = int(T / dt)
    x = np.zeros((steps+1, 4))
    bis = np.zeros(steps+1)
    u = np.zeros(steps)
    
    for i in range(steps):
        # Add noise to the current state to simulate real-world uncertainty
        x_noisy = x[i] + rng.normal(0, 0.01, size=4)
        
        # Use noisy state for MPC planning
        u[i] = mpc_step(x_noisy, patient, dt, N, target_bis)
        
        # Evolve the true state using the deterministic model
        x[i+1], bis[i] = simulate_step(x[i], u[i], patient, dt)
    
    bis[-1] = pd_model(x[-1, 3], patient)
    return x, bis, u

# Set up the problem
patient = Patient(age=40, weight=70)
T = 120  # Total time in minutes
dt = 0.5  # Time step in minutes
N = 20  # Prediction horizon
target_bis = 50  # Target BIS value

# Run MPC simulation
x, bis, u = run_mpc_simulation(patient, T, dt, N, target_bis)

# Plot results
t = np.arange(0, T+dt, dt)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax1.plot(t, bis)
ax1.set_ylabel('BIS')
ax1.set_ylim(0, 100)
ax1.axhline(y=target_bis, color='r', linestyle='--')

ax2.plot(t[:-1], u)
ax2.set_ylabel('Infusion Rate (mg/kg/h)')

ax3.plot(t, x[:, 3])
ax3.set_ylabel('Effect-site Concentration (µg/mL)')
ax3.set_xlabel('Time (min)')

plt.tight_layout()

print(f"Initial BIS: {bis[0]:.2f}")
print(f"Final BIS: {bis[-1]:.2f}")
print(f"Mean infusion rate: {np.mean(u):.2f} mg/kg/h")
print(f"Final effect-site concentration: {x[-1, 3]:.2f} µg/mL")
```

## Synthèse et perspectives

L'optimisation paramétrique traite les problèmes successifs de commande
prédictive comme les membres d'une même famille plutôt que comme des programmes
non linéaires sans lien entre eux. Les mises à jour par sensibilité, les pas de
prédiction-correction et les approximations apprises du correcteur échangent du
temps d'optimisation en ligne contre de l'erreur d'approximation et la
nécessité de détecter un changement d'ensemble actif ou de régime de
fonctionnement.

La commande prédictive a maintenant produit un comportement en boucle fermée en
résolvant un nouveau problème à horizon fini après chaque observation. Des
décisions conditionnelles à l'état peuvent-elles plutôt être calculées pour
toute une famille d'états possibles avant que l'état suivant soit connu?
La [programmation dynamique à horizon fini](finite-horizon-dp.md) amorce cette
seconde construction de la rétroaction.
