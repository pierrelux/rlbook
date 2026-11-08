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

# Transcription en temps continu et collocation

Les chapitres précédents ont formulé et optimisé des trajectoires au moyen
d'applications de transition discrètes. Les modèles physiques sont toutefois
souvent donnés par des équations différentielles ordinaires, de sorte que
leurs états et leurs commandes sont des fonctions du temps. Un optimiseur de
dimension finie ne peut pas choisir directement une fonction entière.
L'optimisation de trajectoires en temps continu commence donc par remplacer
ces fonctions et leurs équations différentielles par un nombre fini de
variables et de contraintes sur ces variables.

Comment ce remplacement peut-il préserver assez de l'équation différentielle
pour que la trajectoire obtenue ait un sens entre les instants stockés?

Ce remplacement s'appelle la **transcription**, et le problème d'optimisation
fini qui en résulte est un programme non linéaire (PNL).

La formulation par collocation directe utilisée ici stocke les valeurs de
l'état et de la commande à des instants choisis et les interpole entre ces
instants par des polynômes de bas degré. L'équation différentielle est imposée
en des points choisis de chaque intervalle de temps. Ce chapitre développe
cette construction à partir d'un exemple simple, en passant par les
transcriptions d'Euler, des trapèzes et de Hermite--Simpson, puis l'applique
au mouvement d'un pont roulant.

:::{admonition} Objectifs d'apprentissage
:class: note

Après l'étude de ce chapitre, vous devriez pouvoir :

1. Expliquer pourquoi le contrôle optimal en temps continu exige une transcription avant de pouvoir être confié à un solveur de PNL.
2. Représenter un polynôme soit par ses coefficients dans une base, soit par ses valeurs en des nœuds distincts.
3. Distinguer l'interpolation polynomiale exacte de la régression par moindres carrés.
4. Construire des opérateurs de dérivation et de quadrature à partir des fonctions cardinales de Lagrange.
5. Assembler les problèmes d'optimisation complets d'Euler, des trapèzes et de Hermite--Simpson.
6. Choisir séparément les représentations de l'état et de la commande, et identifier leurs variables de décision.
7. Générer des coefficients de collocation à partir de nœuds et s'en servir pour assembler un PNL creux.
:::

:::{admonition} Préalables
:class: tip

Ce chapitre utilise les équations différentielles ordinaires, les intégrales
définies et la forme de base d'un programme non linéaire avec contraintes
d'égalité et d'inégalité. [](discrete-time-optimal-control.md) présente ce
problème d'optimisation, et [](numerical-trajectory-optimization.md)
développe les formulations par tir et simultanées. [](appendix_ivps.md)
revoit les intégrateurs séquentiels utilisés à l'intérieur du tir.
:::

:::{admonition} Démonstration d'accompagnement
:class: tip

La démonstration [Collocation directe pour le redressement d'un pendule](lab/notebooks/08_collocation_from_nodes.ipynb)
applique pas à pas la construction de ce chapitre à un problème de commande
non linéaire. Elle calcule les coefficients de collocation à la main et dans
le code, assemble et résout le programme non linéaire, et explique chaque
ligne du code Python dans la notation de ce chapitre. On peut la lire après
@alg-collocation-from-nodes ou en parallèle avec le reste du chapitre, et elle
s'exécute dans Colab.
:::

## Un exemple sur un seul intervalle

Comment une contrainte portant sur quatre valeurs aux extrémités peut-elle
garantir que la commande choisie produit le déplacement requis?

Considérons un point qui doit se déplacer de $x(0)=0$ à $x(1)=1$. Sa vitesse
est la commande, de sorte que

$$
\dot x(t)=u(t),
$$

et l'objectif pénalise le carré de l'effort de commande,

$$
\underset{x(\cdot),u(\cdot)}{\operatorname{minimiser}}
\quad \int_0^1 u(t)^2\,dt.
$$

$x$ et $u$ sont tous deux des fonctions inconnues. Comme première
approximation finie, on ne retient que leurs valeurs aux extrémités
$x_0,x_1,u_0,u_1$ et on laisse la commande varier linéairement entre $u_0$ et
$u_1$. Intégrer $\dot x=u$ signifie que la variation de l'état est égale à
l'aire sous cette commande. La région ombrée ci-dessous est un trapèze de
largeur unitaire. Son aire est la moyenne de ses deux hauteurs aux extrémités,
$u_0$ et $u_1$, multipliée par la largeur.

```{code-cell} python
:tags: [remove-input]
:label: fig-linear-control-trapezoid
:caption: Faites glisser les commandes aux extrémités ou lancez l'accumulation de gauche à droite. Le rectangle et le triangle ombrés ont pour somme la variation d'état impliquée par $\dot x=u$. Avec une largeur d'intervalle $h=1$, on obtient la relation de variation d'état $x_1-x_0=(u_0+u_1)/2$.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from collocation_widgets import render_linear_control_area

control_area = render_linear_control_area()
display(HTML(control_area))
```

:::{figure} _static/collocation/linear-control-area.svg
:label: fig-linear-control-trapezoid-fallback
:class: pdf-fallback
:alt: Une commande linéaire entre deux valeurs aux extrémités forme un trapèze dont les aires du rectangle et du triangle ont pour somme la variation de l'état.

Une commande linéaire relie $u_0$ et $u_1$ sur un intervalle de largeur $h$.
Le rectangle $hu_0$ et le triangle $h(u_1-u_0)/2$ ont pour somme
$h(u_0+u_1)/2$. Comme $\dot x=u$, cette aire est égale à $x_1-x_0$.
:::

L'optimiseur doit choisir des états et des commandes aux extrémités qui
s'accordent avec ce calcul d'aire. Sinon, il pourrait choisir $x_0=0$ et
$x_1=1$ tout en mettant les deux commandes à zéro : les conditions aux
extrémités seraient respectées, mais une vitesse nulle ne pourrait pas
produire le déplacement. Pour exclure de tels choix, la variation d'état
stockée $x_1-x_0$ doit être égale à la variation $(u_0+u_1)/2$ prédite par
l'intégration de la commande. En faisant passer les deux quantités à gauche,
on obtient la contrainte

$$
x_1-x_0-\frac{1}{2}(u_0+u_1)=0.
$$

Le membre de gauche mesure l'écart entre ces deux variations et s'appelle le
**défaut**. Exiger que cet écart s'annule donne une **contrainte de défaut**,
qui impose la dynamique dans le cadre de l'approximation choisie. L'équation
est **algébrique**, car elle relie les quatre nombres $x_0,x_1,u_0,u_1$ par
des opérations arithmétiques; son évaluation ne demande de dériver ni
d'intégrer aucune fonction inconnue. Un optimiseur peut donc vérifier
directement la contrainte pour chaque ensemble candidat de valeurs aux
extrémités.

En approchant l'intégrale du coût courant par la moyenne de ses valeurs aux
extrémités, on obtient le programme non linéaire fini

$$
\begin{aligned}
\underset{x_0,x_1,u_0,u_1}{\operatorname{minimiser}}
\quad&\frac{1}{2}(u_0^2+u_1^2)\\
\text{sous les contraintes}\quad&x_0=0,\qquad x_1=1,\\
&x_1-x_0-\frac{1}{2}(u_0+u_1)=0.
\end{aligned}
$$

Le défaut intègre exactement la commande linéaire. L'objectif utilise une
autre approximation : $u(t)^2$ est en général quadratique, si bien que la
moyenne de ses valeurs aux extrémités ne donne pas forcément son intégrale
exacte. Les deux calculs deviennent exacts pour une commande constante.

Les conditions aux limites réduisent le défaut à $u_0+u_1=2$. En substituant
$u_1=2-u_0$ dans l'objectif, on obtient

$$
\frac12\left(u_0^2+(2-u_0)^2\right)
=(u_0-1)^2+1.
$$

Le terme au carré est minimal en $u_0=1$, ce qui donne aussi $u_1=1$.
L'interpolation obtenue est la solution exacte $u(t)=1$ et $x(t)=t$.

Cet exemple contient déjà les principaux ingrédients de la collocation
directe. Les valeurs des fonctions sont devenues des variables d'optimisation,
l'intégration est devenue une somme pondérée, et l'équation différentielle est
devenue une contrainte d'égalité. Les sections suivantes construisent ces
opérations de façon systématique pour des dynamiques vectorielles non
linéaires et des polynômes de degré plus élevé.

## D'un problème continu à un PNL fini

Quelles quantités et quelles contraintes une transcription doit-elle retenir
du problème de commande en temps continu?

On conserve la notation précédente : $\mathbf x\in\mathbb R^n$ pour l'état,
$\mathbf u\in\mathbb R^m$ pour la commande et $c$ pour le coût. Ici, $t$ est
le temps physique continu, tandis que $k$ indexera les intervalles du
maillage. Le coût courant $c(\mathbf x,\mathbf u,t)$ est un coût par unité de
temps; son intégrale sur un intervalle donne l'équivalent du coût d'étape
discret $c_k$. Le coût terminal s'écrit $c_f$, car le temps physique final est
$t_f$. Les exemples scalaires omettent le gras.

Avec les inégalités $\mathbf g\leq\mathbf0$ et les égalités
$\mathbf h=\mathbf0$, le problème de Bolza en temps continu s'écrit

$$
\begin{aligned}
\underset{\mathbf{x}(\cdot),\mathbf{u}(\cdot),t_f}{\operatorname{minimiser}}
\quad&
c_f(\mathbf{x}(t_f),t_f)
+\int_{t_0}^{t_f} c(\mathbf{x}(t),\mathbf{u}(t),t)\,dt\\
\text{sous les contraintes}\quad&
\dot{\mathbf{x}}(t)=\mathbf{f}(\mathbf{x}(t),\mathbf{u}(t),t),\\
&\mathbf{h}(\mathbf{x}(t_0),\mathbf{x}(t_f),t_f)=\mathbf0,\\
&\mathbf{g}(\mathbf{x}(t),\mathbf{u}(t),t)\leq\mathbf0.
\end{aligned}
$$

L'égalité $\mathbf h=\mathbf0$ impose des conditions aux extrémités, tandis que
$\mathbf g\leq\mathbf0$ impose des contraintes le long du chemin, comme les
limites des actionneurs ou une borne sur l'état. Un temps final fixe est
fourni comme donnée et omis des variables de décision. Poser $c=0$ donne le
cas particulier de Mayer, tandis que poser $c_f=0$ donne le cas particulier de
Lagrange. Les trois formes utilisent la même transcription.

Deux stratégies de transcription diffèrent par les valeurs qui deviennent des
variables de décision et par la manière d'imposer l'équation différentielle :

| Stratégie | Variables de décision finies | Traitement de l'EDO |
|---|---|---|
| Tir | Paramètres de commande et, pour le tir multiple, certains états de frontière | Un intégrateur temporel fait avancer l'état séquentiellement à l'intérieur de chaque intervalle de tir |
| Collocation directe | Valeurs de l'état et de la commande en des nœuds choisis | Des équations de défaut algébriques imposent l'EDO simultanément |

Le tir et la collocation sont donc des stratégies de transcription distinctes.
Une méthode de collocation peut reproduire une formule d'intégration connue,
mais elle expose les valeurs de l'état au PNL plutôt que de cacher chaque mise
à jour de l'état à l'intérieur d'une simulation. Le terme **directe** indique
que le problème continu est converti directement en un problème
d'optimisation, sans dériver au préalable des conditions nécessaires comme les
équations de Pontryaguine.

Pour appliquer une approximation locale de façon répétée sur l'horizon, on
divise le temps en intervalles d'extrémités

$$
t_0<t_1<\cdots<t_N=t_f,\qquad h_k=t_{k+1}-t_k.
$$

Cette suite est un **maillage**, dont les intervalles sont indexés par
$k=0,\ldots,N-1$. On écrit $\mathbf x_k$ et $\mathbf u_k$ pour les valeurs
au temps de maillage $t_k$, comme dans les chapitres précédents en temps
discret; l'indice est un rang, non un temps physique. Sur chaque intervalle, la
coordonnée normalisée est $\tau=(t-t_k)/h_k$, ou de manière équivalente

$$
t=t_k+h_k\tau,\qquad 0\leq\tau\leq1.
$$

Chaque intervalle physique est ainsi ramené au même intervalle de référence
$[0,1]$. Les formules de dérivation et d'intégration peuvent être construites
une seule fois sur cet intervalle de référence; la longueur $h_k$ fournit
ensuite la mise à l'échelle en temps physique. Si le temps final est aussi
optimisé, les longueurs des intervalles deviennent des variables ou des
fractions fixes de l'horizon variable; les opérateurs de l'intervalle de
référence restent inchangés.

### Morceaux polynomiaux sur un maillage temporel

Un seul polynôme peut représenter une trajectoire sur tout l'horizon; la
solution $x(t)=t$ de l'exemple d'ouverture le fait déjà. Pourquoi introduire
plusieurs morceaux? Considérons la même dynamique scalaire $\dot x=u$, à
partir de $x(0)=0$, mais avec un déplacement vers la droite à vitesse unitaire
jusqu'à $t=1/2$, puis vers la gauche à vitesse unitaire :

$$
u(t)=\begin{cases}
1,&0\leq t<\tfrac12,\\
-1,&\tfrac12<t\leq1,
\end{cases}
\qquad
x(t)=\begin{cases}
t,&0\leq t\leq\tfrac12,\\
1-t,&\tfrac12\leq t\leq1.
\end{cases}
$$

L'état est continu, mais sa pente change brusquement à l'instant de
commutation. Deux segments de droite le représentent exactement. Un polynôme
unique a une dérivée continue; il ne peut donc pas reproduire exactement ce
coin, même s'il peut approcher la trajectoire. Ici, l'EDO est vérifiée de part
et d'autre de la commutation; la valeur de la commande en ce seul instant
n'affecte pas son intégrale.

La comparaison ci-dessous utilise le polynôme quadratique $q(t)=2t(1-t)$, qui
passe par les mêmes états en $t=0,1/2,1$. Faire concorder ces trois valeurs ne
reproduit pas le mouvement entre elles : le polynôme quadratique a une pente
qui varie continûment, alors que la vitesse prescrite passe de $+1$ à $-1$.

:::{figure} _static/collocation/switching-trajectory.svg
:label: fig-switching-trajectory
:alt: La trajectoire d'état triangulaire est formée de deux segments de droite qui se rejoignent au temps un demi. Un polynôme quadratique en tirets passe par les trois mêmes points, mais s'incurve au-dessus des segments. En dessous, la pente exacte saute de plus un à moins un, alors que la pente du polynôme quadratique décroît continûment de plus deux à moins deux.

Deux morceaux linéaires représentent exactement le mouvement prescrit. Le
polynôme quadratique orange en tirets $q(t)=2t(1-t)$ passe par les trois
points d'échantillonnage noirs, mais s'en écarte entre eux. Le panneau
inférieur compare leurs dérivées en temps physique. Les cercles vides
indiquent les deux pentes unilatérales à la commutation, où l'état exact n'a
pas de dérivée. Des polynômes globaux de degré plus élevé peuvent améliorer
l'approximation, mais tout polynôme unique a une dérivée continue.
:::

Même lorsque la trajectoire est lisse, certaines portions peuvent varier
beaucoup plus vite que d'autres. Des morceaux polynomiaux séparés permettent
de raccourcir les intervalles du maillage près d'une variation rapide tout en
conservant des intervalles plus longs ailleurs. Augmenter le degré d'un
polynôme global ajoute de la flexibilité sur tout l'horizon. La représentation
locale donne aussi à l'optimiseur une structure utile : les contraintes de
dynamique de chaque intervalle ne font intervenir que des valeurs de cet
intervalle et les états aux extrémités voisines. Un polynôme global couple les
valeurs sur tout l'horizon par l'intermédiaire de sa matrice de dérivation.

Pour une trajectoire lisse sur tout l'horizon, un polynôme unique de degré
suffisamment élevé peut être un choix efficace. Les polynômes par morceaux
donnent un contrôle local sur la résolution, permettent des changements de
pente aux jonctions et gardent creuses les dérivées des contraintes. Ils
exigent de raccorder explicitement les morceaux adjacents pour que l'état
reste continu.

On représente donc la trajectoire par un polynôme distinct sur chaque
intervalle du maillage $[t_k,t_{k+1}]$. L'approximation complète
$\mathbf{x}_h$ est **polynomiale par morceaux**. Pour construire le morceau de
l'intervalle $k$, on utilise sa coordonnée locale $\tau=(t-t_k)/h_k$, où
$h_k=t_{k+1}-t_k$, et on écrit

$$
\mathbf{p}_k(\tau):=\mathbf{x}_h(t_k+h_k\tau),\qquad 0\le\tau\le1.
$$

Ainsi, $\mathbf{p}_k(0)$ est l'état au début de cet intervalle et
$\mathbf{p}_k(1)$ l'état à sa fin. Sur l'intervalle suivant, la coordonnée
locale repart de zéro, et un autre polynôme $\mathbf{p}_{k+1}$ décrit la
trajectoire. L'intervalle de référence et ses poids d'intégration peuvent être
réutilisés même lorsque les intervalles physiques ont des longueurs
différentes.

:::{figure} _static/collocation/local-time-pieces.svg
:label: fig-local-time-pieces
:alt: Une trajectoire scalaire continue comporte trois morceaux polynomiaux sur les intervalles physiques de zéro à un, de un à trois et de trois à quatre. En dessous, chacun de ces morceaux occupe un axe de temps local de zéro à un. Les morceaux adjacents se rejoignent aux états d'extrémité communs x1 et x2.

Un exemple scalaire de la représentation par morceaux. Le panneau supérieur
raccorde trois morceaux quadratiques sur des intervalles physiques de
longueurs $1$, $2$ et $1$. Les panneaux inférieurs montrent ces mêmes morceaux
dans leurs propres coordonnées $0\leq\tau\leq1$, en conservant leurs couleurs
et leurs styles de trait. Le morceau du milieu utilise $t=1+2\tau$, si bien
que deux unités de temps physique occupent une unité de temps local. Les
points noirs marquent les états aux extrémités; faire coïncider la valeur de
droite d'un morceau avec la valeur de gauche du suivant garde la trajectoire
continue. Les pentes n'ont pas à concorder aux jonctions.
:::

## Un polynôme, deux systèmes de coordonnées

Faut-il stocker une trajectoire polynomiale par ses coefficients ou par ses
valeurs en des points choisis, et comment ces deux descriptions sont-elles
liées?

L'exemple d'ouverture stockait une commande linéaire par ses deux valeurs aux
extrémités. Chaque morceau d'une trajectoire de collocation demande une
reconstruction semblable : étant donné les valeurs que stocke l'optimiseur,
quelle courbe définissent-elles entre les nœuds? Commençons par un polynôme
scalaire $p$ sur l'intervalle de référence $[0,1]$. La construction
s'applique à chaque composante de l'état et de la commande.

Supposons que $p(0)=y_0$ et $p(1)=y_1$, et restreignons $p$ à un degré au
plus un. Une droite a la forme $p(\tau)=a_0+a_1\tau$. La première condition
aux extrémités donne $a_0=y_0$, et la seconde $a_0+a_1=y_1$, d'où
$a_1=y_1-y_0$. Par substitution et regroupement, on obtient

$$
p(\tau)=y_0(1-\tau)+y_1\tau.
$$

Les poids $1-\tau$ et $\tau$ imposent les deux conditions aux extrémités : en
$0$, la formule renvoie $y_0$, et en $1$, elle renvoie $y_1$. Entre les
extrémités, les poids varient continûment; en $\tau=1/4$, par exemple, la
valeur est $3y_0/4+y_1/4$. C'est l'unique droite qui passe par les deux
valeurs prescrites. Sans la restriction sur le degré, d'autres courbes
pourraient y passer. Même pour cette droite, $y_0+(y_1-y_0)\tau$ en est une
représentation équivalente. La forme pondérée est commode, car chaque valeur
stockée a sa propre fonction, qui détermine sa contribution sur tout
l'intervalle. Les fonctions $1-\tau$ et $\tau$ forment une base des polynômes
de degré un, tout comme $1$ et $\tau$. Dans la première base, les
coefficients sont les valeurs aux extrémités $y_0,y_1$; dans la base des
monômes, ce sont $a_0=y_0$ et $a_1=y_1-y_0$.

Les morceaux de degré plus élevé reposent sur la même idée avec davantage de
valeurs stockées. Pour préciser quelles courbes ces valeurs peuvent définir,
on restreint le degré à au plus $r$. L'espace de polynômes obtenu est

$$
\mathcal P_r=\{p:\deg p\leq r\}.
$$

Cet espace, noté $\mathcal P_r$, est de dimension $r+1$. Le choix d'une base
$\{\phi_0,\ldots,\phi_r\}$ donne des coordonnées en coefficients

$$
p(\tau)=\sum_{j=0}^{r}a_j\phi_j(\tau).
$$

Le choix des monômes $\phi_j(\tau)=\tau^j$ donne l'expression familière
$a_0+a_1\tau+\cdots+a_r\tau^r$. Une autre base change les coefficients qui
servent à décrire la même fonction.

On peut aussi identifier le même polynôme par ses valeurs. On choisit $r+1$
points distincts

$$
\sigma_0,\ldots,\sigma_r\in[0,1],
$$

et on enregistre

$$
y_i=p(\sigma_i).
$$

Ces points s'appellent des **nœuds de support**, ou **nœuds
d'interpolation** : une fois le degré restreint à au plus $r$, les $r+1$
valeurs stockées déterminent le polynôme entier. Stocker des valeurs est utile
en optimisation de trajectoires, car une borne sur l'état en un nœud devient
alors une borne sur une variable de décision. Pour évaluer la dynamique entre
les nœuds, il faut toutefois une formule qui reconstruit le polynôme à partir
de ces valeurs.

Avec davantage de nœuds, la même construction demande une fonction de poids
$\ell_j$ par valeur stockée $y_j$. Pour retrouver $y_j$ en son propre nœud
sans modifier les valeurs aux autres nœuds, $\ell_j$ doit valoir un en
$\sigma_j$ et zéro en chaque autre nœud de support. Un polynôme qui a ces
zéros contient les facteurs $(\tau-\sigma_m)$ pour tous les $m\ne j$. En
divisant leur produit par sa valeur en $\sigma_j$, on rend la valeur en ce
point égale à un. On construit ainsi la **fonction de base de Lagrange**,
aussi appelée **fonction cardinale**,

$$
\ell_j(\tau)
=\prod_{\substack{m=0\\m\neq j}}^r
\frac{\tau-\sigma_m}{\sigma_j-\sigma_m}.
$$

En un nœud de support $\sigma_i$, l'un des facteurs du numérateur est nul,
sauf si $i=j$. Lorsque $i=j$, chaque numérateur est égal au dénominateur
correspondant. Par conséquent,

$$
\ell_j(\sigma_i)=\delta_{ij},
$$

où le symbole de Kronecker $\delta_{ij}$ vaut un lorsque $i=j$ et zéro sinon.
Cette identité ne précise que les valeurs aux nœuds. En tout autre $\tau$, la
formule du produit donne un poids polynomial qui varie de façon lisse et qui
peut être négatif ou dépasser un pour des degrés plus élevés. Ainsi,
contrairement à une droite entre deux valeurs aux extrémités, un interpolant
de degré plus élevé peut prendre des valeurs hors de l'étendue de ses valeurs
stockées. Les fonctions cardinales reconstruisent le polynôme sur tout
l'intervalle par

$$
\boxed{
p(\tau)=\sum_{j=0}^{r}y_j\ell_j(\tau),
\qquad y_j=p(\sigma_j).
}
$$

En un nœud $\sigma_i$, tous les termes sauf $y_i\ell_i(\sigma_i)=y_i$
s'annulent. Entre les nœuds, la somme pondérée fournit les valeurs
intermédiaires. Cette expression est la **forme de Lagrange** de $p$. Lorsque
les valeurs $y_j$ sont prescrites et que $p$ est construit pour passer par
elles, $p$ s'appelle le **polynôme d'interpolation de Lagrange**. Chaque
$\ell_j$ est une fonction de base; $p$ est leur somme pondérée. Modifier une
valeur stockée $y_j$ de $\Delta y_j$ modifie la courbe de
$\Delta y_j\ell_j(\tau)$; la même fonction décrit donc l'effet de cette
variable sur le polynôme entier.

L'unicité de cette reconstruction découle d'un argument élémentaire de
dénombrement des racines. Si deux polynômes de $\mathcal P_r$ ont les mêmes
$r+1$ valeurs nodales, leur différence a $r+1$ racines distinctes. Un polynôme
non nul de degré au plus $r$ ne peut pas avoir autant de racines; les deux
polynômes sont donc identiques.

Par conséquent, tout polynôme de $\mathcal P_r$ a un développement unique en
$\ell_0,\ldots,\ell_r$. Ces fonctions forment la **base de Lagrange** associée
aux nœuds de support choisis. Ses coefficients sont exactement les valeurs
nodales $y_j=p(\sigma_j)$. Stocker un polynôme par ses valeurs est donc aussi
une représentation dans une base, choisie de sorte que ses coefficients
s'interprètent directement comme des valeurs de la fonction.

Pour un polynôme scalaire, on rassemble les coefficients et les valeurs dans
des vecteurs colonnes $\mathbf a=(a_0,\ldots,a_r)^\top$ et
$\mathbf y=(y_0,\ldots,y_r)^\top$. Les descriptions par coefficients et par
valeurs nodales sont liées par la matrice d'évaluation

$$
V_{ij}=\phi_j(\sigma_i),
\qquad \mathbf{y}=V\mathbf{a}.
$$

Des nœuds de support distincts rendent $V$ inversible, par le même argument de
dénombrement des racines. Ainsi, $\mathbf{a}$ et $\mathbf{y}$ sont deux
vecteurs de coordonnées d'un même polynôme, et non deux approximations
différentes.

Par exemple, prenons $p(\tau)=1+2\tau-\tau^2$ et les nœuds $0,\tfrac12,1$.
En coordonnées monomiales,

$$
\mathbf{a}=
\begin{bmatrix}1\\2\\-1\end{bmatrix},
\qquad
V=
\begin{bmatrix}
1&0&0\\
1&\tfrac12&\tfrac14\\
1&1&1
\end{bmatrix},
$$

alors que les coordonnées nodales sont

$$
\mathbf{y}=V\mathbf{a}=
\begin{bmatrix}1\\\tfrac74\\2\end{bmatrix}.
$$

Dans la base de Lagrange associée à ces trois nœuds, le même polynôme
quadratique s'écrit

$$
p(\tau)=\ell_0(\tau)+\tfrac74\ell_1(\tau)+2\ell_2(\tau).
$$

Ici, $\ell_0,\ell_1,\ell_2$ sélectionnent respectivement les nœuds
$0,\tfrac12,1$. Les deux vecteurs de coefficients décrivent le même polynôme
quadratique. La transcription utilisée ici choisit des coordonnées comme
$\mathbf y$ pour l'état et la commande : des valeurs que l'on peut insérer
directement dans la dynamique, le coût et les bornes. Les coefficients
monomiaux restent utiles pour dériver les formules qui agissent sur ces
valeurs.

(polynomial-space-basis-and-nodes-are-different-choices)=
### Espaces de polynômes, bases et nœuds

La construction sépare trois décisions faciles à confondre :

- L'**espace de polynômes** $\mathcal P_r$ précise quelles fonctions sont disponibles.
- La **base** précise des coordonnées pour un élément de cet espace. Les bases des monômes et de Lagrange engendrent le même $\mathcal P_r$.
- Les **nœuds** précisent où l'on évalue les valeurs, les résidus ou les intégrales.

Changer de base ne change pas l'espace de polynômes exact, mais peut modifier
le conditionnement numérique. Changer les nœuds modifie l'interpolation et les
opérateurs construits à partir d'elle. Les familles de nœuds présentées plus
loin peuvent tout de même utiliser des coordonnées nodales de Lagrange dans le
PNL.

## Interpolation polynomiale et régression par moindres carrés

Quand un polynôme doit-il passer par chaque valeur fournie, et quand doit-il
plutôt approcher ces valeurs avec un plus petit nombre de coefficients?

L'interpolation et la régression polynomiale imposent des exigences
différentes aux mêmes données. Six valeurs en six nœuds distincts déterminent
un polynôme de degré au plus cinq qui passe par chaque point. Si les valeurs
sont des observations bruitées et que l'on cherche à estimer une tendance
quadratique, un polynôme quadratique ne pourra en général pas passer par les
six. La régression par moindres carrés choisit alors ses trois coefficients de
façon à minimiser la somme des carrés des écarts. La figure ci-dessous
applique les deux choix aux six mêmes points. On note $V$ la matrice
d'évaluation de la base, comme dans $\mathbf y=V\mathbf a$. Elle a six lignes
dans les deux cas, mais six colonnes pour l'interpolant de degré cinq et trois
pour l'ajustement quadratique.

| | Interpolation polynomiale | Régression par moindres carrés |
|---|---|---|
| Entrée | Conditions exactes sur les valeurs | Observations habituellement bruitées ou surdéterminées |
| Problème algébrique | Satisfaire exactement $V\mathbf{a}=\mathbf{y}$ | Minimiser $\lVert V\mathbf{a}-\mathbf{y}\rVert_2^2$ |
| Résidu | Nul lorsque les conditions sur les valeurs déterminent un polynôme unique | En général non nul |
| Usage typique | Représenter une fonction à partir de données nodales exactes | Estimer une tendance ou une moyenne conditionnelle |

Si $V$ est carrée et inversible, minimiser le carré du résidu redonne aussi
l'interpolant exact. Les deux procédures coïncident alors, car l'espace de
polynômes choisi peut satisfaire chaque condition sur les valeurs.

```{code-cell} python
:tags: [remove-input]
:label: fig-interpolation-versus-regression
:caption: Deux polynômes ajustés aux six mêmes points (points noirs). L'interpolant de degré cinq, en bleu plein, passe par chaque point. La régression quadratique, en orange tireté, minimise la somme des carrés des résidus, représentés par des segments verticaux gris. Les degrés des polynômes diffèrent, car un polynôme quadratique ne peut en général pas satisfaire exactement les six conditions sur les valeurs.

import numpy as np
import matplotlib.pyplot as plt

observation_nodes = np.linspace(0.0, 1.0, 6)
observation_values = (
    0.25
    + 0.95 * observation_nodes
    - 0.25 * observation_nodes**2
    + np.array([0.00, 0.10, -0.07, 0.08, -0.09, 0.02])
)
plot_nodes = np.linspace(0.0, 1.0, 401)

def evaluate_lagrange(nodes, values, points):
    result = np.zeros_like(points)
    for j, node in enumerate(nodes):
        cardinal = np.ones_like(points)
        for m, other_node in enumerate(nodes):
            if m != j:
                cardinal *= (points - other_node) / (node - other_node)
        result += values[j] * cardinal
    return result

interpolated_values = evaluate_lagrange(
    observation_nodes, observation_values, plot_nodes
)
regression_matrix = np.vander(observation_nodes, 3, increasing=True)
regression_coefficients, *_ = np.linalg.lstsq(
    regression_matrix, observation_values, rcond=None
)
regression_curve = np.vander(plot_nodes, 3, increasing=True) @ regression_coefficients
regression_at_nodes = regression_matrix @ regression_coefficients

with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}):
    figure, axis = plt.subplots(figsize=(5.5, 2.8), constrained_layout=True)
    blue = "#0072B2"
    orange = "#D55E00"
    gray = "#4D4D4D"

    axis.plot(plot_nodes, interpolated_values, color=blue, linewidth=1.8,
              label="Degree-five interpolation")
    axis.plot(plot_nodes, regression_curve, color=orange, linewidth=1.8,
              linestyle="--", label="Quadratic least squares")
    axis.vlines(observation_nodes, regression_at_nodes, observation_values,
                color="#777777", linewidth=1.2, zorder=2)
    axis.scatter(observation_nodes, observation_values, color=gray,
                 edgecolor="white", linewidth=0.6, s=30, zorder=3,
                 label="Same six points")
    axis.set_xlabel(r"$\tau$")
    axis.set_ylabel("value")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.set_xlim(-0.025, 1.025)
    axis.set_ylim(0.15, 1.25)
    axis.legend(loc="upper left", frameon=False, fontsize=8)

display(figure)
plt.close(figure)
```

Dans une représentation nodale de l'état en collocation directe, les états
stockés sont des variables de décision inconnues plutôt que des observations.
Chaque vecteur candidat de valeurs de support définit exactement un polynôme
d'interpolation. L'optimiseur choisit un candidat dont le résidu de l'EDO
s'annule aux nœuds de collocation. Un solveur numérique peut travailler
temporairement avec une mesure scalaire de la violation des contraintes, mais
les conditions de collocation restent des contraintes d'égalité, et non une
perte de régression statistique.

## Opérateurs fixes à partir des valeurs nodales

Comment les valeurs stockées peuvent-elles nous donner à la fois les pentes et
l'intégrale de chaque polynôme candidat?

La représentation nodale transforme aussi la dérivation et l'intégration en
produits de matrices et de vecteurs. Une fois les nœuds choisis, les tableaux
nécessaires peuvent être calculés une seule fois et réutilisés pour chaque
trajectoire candidate. L'optimiseur peut alors évaluer les pentes et les
intégrales à mesure qu'il modifie les valeurs nodales, sans recalculer ces
tableaux. Chaque fonction cardinale est un polynôme connu; sa dérivée et sa
primitive peuvent donc être calculées sous forme close en modifiant ses
coefficients. La construction des tableaux n'exige donc ni approximation par
différences finies ni quadrature numérique des fonctions de base. Les
opérateurs obtenus sont exacts pour le polynôme représenté, à l'erreur
d'arrondi près dans leur construction et leur application.

C'est possible parce que les fonctions cardinales restent fixes et que seuls
leurs coefficients $y_j$ changent. Dans la représentation

$$
p(\tau)=\sum_{j=0}^r y_j\ell_j(\tau),
$$

chaque $y_j$ est constant par rapport à $\tau$; la dérivation n'agit donc que
sur $\ell_j$. Pour obtenir les pentes dont l'EDO a besoin, on dérive puis on
évalue en un nœud $\sigma_i$ :

$$
p'(\sigma_i)=\sum_{j=0}^r
\underbrace{\ell_j'(\sigma_i)}_{D_{ij}}y_j,
$$

tandis que l'intégration sur l'intervalle de référence donne

$$
\int_0^1p(\tau)\,d\tau
=\sum_{j=0}^r
\underbrace{\left(\int_0^1\ell_j(\tau)\,d\tau\right)}_{w_j}y_j.
$$

La matrice de dérivation $D$ et les poids d'intégration $w$ obtenus ne
dépendent que des nœuds choisis. Leurs composantes enregistrent les dérivées
et les aires des fonctions cardinales fixes. La multiplication par les valeurs
nodales courantes combine ces contributions en la dérivée ou l'intégrale du
polynôme courant. C'est ainsi que la transcription fournit le calcul
différentiel et intégral dont ont besoin la dynamique et le coût, au moyen
d'opérations arithmétiques sur les variables de décision.

Pour une construction concrète, supposons qu'un polynôme quadratique soit
stocké par ses valeurs $y_0,y_m,y_1$ en $0,\tfrac12,1$, où $m$ désigne le
milieu. Les sorties recherchées sont sa pente en chacun de ces nœuds et son
intégrale sur $[0,1]$. Pour construire les tableaux qui renvoient ces sorties
pour tout choix des trois valeurs, on construit d'abord les trois fonctions
cardinales qui les multiplient :

$$
\begin{aligned}
\ell_0(\tau)
&=\frac{(\tau-\frac12)(\tau-1)}{(0-\frac12)(0-1)}
=2\tau^2-3\tau+1,\\
\ell_m(\tau)
&=\frac{\tau(\tau-1)}{(\frac12-0)(\frac12-1)}
=4\tau(1-\tau),\\
\ell_1(\tau)
&=\frac{\tau(\tau-\frac12)}{(1-0)(1-\frac12)}
=2\tau^2-\tau.
\end{aligned}
$$

Ce sont les trois contributions dans
$p(\tau)=y_0\ell_0(\tau)+y_m\ell_m(\tau)+y_1\ell_1(\tau)$.
Leurs dérivées sont $4\tau-3$, $4-8\tau$ et $4\tau-1$. À l'extrémité gauche,
elles donnent les poids $-3,4,-1$, d'où $p'(0)=-3y_0+4y_m-y_1$. L'évaluation
au milieu et à l'extrémité droite fournit les deux autres lignes de la matrice
de dérivation :

$$
\begin{bmatrix}
p'(0)\\
p'(\tfrac12)\\
p'(1)
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
-3&4&-1\\
-1&0&1\\
1&-4&3
\end{bmatrix}}_{D}
\begin{bmatrix}
y_0\\y_m\\y_1
\end{bmatrix}.
$$

Ainsi, un produit matrice-vecteur transforme les trois valeurs stockées en
les trois dérivées nodales. L'intégration des mêmes fonctions cardinales donne

$$
\int_0^1 p(\tau)\,d\tau
=
\underbrace{
\begin{bmatrix}
\tfrac16&\tfrac46&\tfrac16
\end{bmatrix}}_{w^\mathsf T}
\begin{bmatrix}
y_0\\y_m\\y_1
\end{bmatrix}.
$$

La ligne $w^\mathsf T$ transforme les trois mêmes valeurs en l'intégrale
exacte de leur interpolant quadratique. La matrice $D$ dérive exactement tout
polynôme de $\mathcal P_2$, et $w$ intègre exactement tout polynôme de cet
espace. « Opérateur fixe » signifie que ces tableaux restent constants pendant
que le PNL modifie les valeurs nodales.

Pour le polynôme précédent $p(\tau)=1+2\tau-\tau^2$, les valeurs stockées sont
$\mathbf y=(1,\tfrac74,2)^\top$. L'application des tableaux donne

$$
D\mathbf y=\begin{bmatrix}2\\1\\0\end{bmatrix},
\qquad
w^\top\mathbf y
=\frac16+\frac46\frac74+\frac16\,2=\frac53.
$$

Le premier résultat concorde avec $p'(\tau)=2-2\tau$ aux trois nœuds; le
second concorde avec l'intégrale de $1+2\tau-\tau^2$ sur $[0,1]$. La figure
rassemble le changement de coordonnées et ces deux opérations pour ce même
polynôme.

:::{figure} _static/collocation/polynomial-coordinate-operators.svg
:label: fig-polynomial-coordinate-operators
:alt: Le polynôme quadratique p de tau égal à un plus deux tau moins tau au carré est représenté par son vecteur de coefficients monomiaux et par ses valeurs aux nœuds de support zéro, un demi et un. Des opérateurs fixes de dérivation et de quadrature agissent sur le vecteur nodal.

Un polynôme quadratique, deux systèmes de coordonnées. L'évaluation des
coefficients monomiaux $\mathbf{a}$ aux nœuds de support donne
$\mathbf{y}=V\mathbf{a}$. La collocation directe stocke le vecteur nodal
$\mathbf{y}$; les opérateurs fixes $D$ et $w$ renvoient alors ses dérivées
nodales et son intégrale exacte sans ajouter de variables d'optimisation.
:::

Une transcription peut utiliser des nœuds à trois fins distinctes :

| Rôle du nœud | Ce qu'il fait |
|---|---|
| Nœud de support | Fournit une valeur servant à définir un polynôme particulier par interpolation |
| Nœud de collocation | Précise où l'état reconstruit doit satisfaire l'EDO |
| Nœud de quadrature | Fournit un point servant à approcher une intégrale |

Une méthode réutilise souvent un même ensemble de points pour deux ou trois
rôles. Les nœuds de support appartiennent à un polynôme particulier : les
polynômes de l'état, de la commande et de la pente n'ont pas à utiliser les
mêmes nœuds d'interpolation. Par exemple, un polynôme d'état peut s'appuyer
sur un ensemble de nœuds et être dérivé en d'autres nœuds de collocation. Dans
la construction par intégration ci-dessous, les nœuds de collocation servent
aussi de nœuds de support pour le polynôme des pentes. Les formes par
intégration et par dérivation ont toutes deux des nœuds de collocation. Leurs
résidus algébriques diffèrent : l'une fait concorder des valeurs d'état après
avoir intégré des pentes, l'autre fait concorder des pentes après avoir
interpolé des valeurs d'état.

(sec-collocation-nodal-slopes)=
## Des pentes nodales aux contraintes de collocation

Comment ces applications fixes transforment-elles les équations
différentielles en contraintes algébriques aux nœuds de collocation choisis?

Les opérateurs précédents s'appliquent à tout polynôme. Une trajectoire doit
en outre satisfaire la dynamique. Sur chaque intervalle, la construction
interpolera les pentes fournies par l'EDO et les intégrera à partir de l'état
gauche stocké. Des contraintes exigeront ensuite que les états stockés
s'accordent avec le polynôme obtenu.

L'intervalle $k$ s'étend du temps physique $t_k$ à $t_{k+1}$ et contient un
morceau polynomial. Sa coordonnée de temps locale va de $\tau=0$ à
l'extrémité gauche à $\tau=1$ à l'extrémité droite. On choisit $s$ nœuds de
collocation distincts $\tau_1,\ldots,\tau_s$ sur $[0,1]$. Un nœud $\tau_j$
précise un instant de cet intervalle, $t_k+h_k\tau_j$; par exemple,
$\tau_j=1/2$ en désigne le milieu. Les valeurs de l'état et de la commande à
cet instant sont $\mathbf{x}_{k,j}$ et $\mathbf{u}_{k,j}$.

Ce point d'évaluation, avec ses valeurs d'état et de commande, s'appelle un
**étage**. Un étage est associé à un seul instant; il n'a donc ni début ni fin
distincts. L'indice $k$ désigne l'intervalle et $j$ un étage de cet
intervalle. Selon la méthode, les étages peuvent être des points intérieurs ou
coïncider avec une extrémité : $\tau_j=0$ place un étage au début et
$\tau_j=1$ en place un à la fin. À chaque étage, l'équation différentielle
prescrit la pente en temps physique

$$
\mathbf{f}_{k,j}
=\mathbf{f}(\mathbf{x}_{k,j},\mathbf{u}_{k,j},t_k+h_k\tau_j)
$$

pour $j=1,\ldots,s$. La figure ci-dessous montre une telle évaluation pour la
dynamique scalaire $\dot x=u$. L'état et la commande sont lus au même temps
physique; l'EDO donne alors la pente à laquelle le polynôme d'état doit s'y
conformer.

:::{figure} _static/collocation/collocation-stage.svg
:label: fig-collocation-stage
:alt: Courbes de l'état et de la commande sur l'intervalle physique allant du temps deux à quatre. Une ligne verticale orange au temps trois marque le nœud de collocation tau j égal à un demi. L'état de l'étage vaut deux et la commande de l'étage trois demis. Une tangente en tirets au point d'état a une pente en temps physique de trois demis, comme l'exige x point égal à u.

Un étage combine un instant, une valeur d'état et une valeur de commande. Ici,
$t_k=2$, $h_k=2$ et $\tau_j=1/2$; l'étage se situe donc au temps physique
$3$. Les points orange marquent $x_{k,j}=2$ et $u_{k,j}=3/2$. Pour
$\dot x=u$, la pente prescrite est $f_{k,j}=3/2$, tracée en tangente tiretée
dans le panneau supérieur. Les courbes sont $u_h(t)=1/2+(t-2)$ et
$x_h(t)=1+(t-2)/2+(t-2)^2/2$. Les deux panneaux utilisent le temps physique;
en coordonnées locales, la dérivée est $p_k'(\tau_j)=h_kf_{k,j}=3$.
:::

Comme précédemment, $\mathbf p_k(\tau)=\mathbf x_h(t_k+h_k\tau)$ désigne le
morceau d'état en temps normalisé. Ici, $\ell_j$ est la fonction cardinale de
Lagrange associée aux nœuds de pente $\tau_1,\ldots,\tau_s$; les $s$ valeurs
de pente déterminent un polynôme de degré au plus $s-1$. En les interpolant,
on obtient la dérivée en temps physique sur cet intervalle :

$$
\dot{\mathbf{x}}_h(t_k+h_k\tau)
=\sum_{j=1}^{s}\mathbf{f}_{k,j}\ell_j(\tau).
$$

Ce polynôme concorde avec chaque pente échantillonnée $\mathbf{f}_{k,j}$ en
son nœud de collocation. Chaque échantillon a été évalué en un état d'étage
stocké $\mathbf x_{k,j}$; la courbe d'état retrouvée par intégration n'a pas
encore à passer par ces états stockés. Les équations d'étage ci-dessous
imposeront cette concordance. Pour retrouver les valeurs de l'état à partir de
ces pentes, il faut inverser la dérivation. Par exemple, $\tau+2\tau^2$ et
$1+\tau+2\tau^2$ ont toutes deux pour dérivée $1+4\tau$. Chacune est une
**primitive** de $1+4\tau$ : sa dérivée redonne cette fonction. Ajouter une
constante quelconque donne une autre primitive, puisque la dérivée d'une
constante est nulle. Les pentes déterminent donc les variations de l'état,
mais il faut une valeur de départ pour déterminer l'état lui-même. Cette
valeur de départ fixe la **constante d'intégration**.

Le **théorème fondamental de l'analyse** relie les primitives aux intégrales
définies. Pour une fonction scalaire $p$ à dérivée continue, il énonce
qu'intégrer la dérivée entre deux points donne la différence des valeurs de
la fonction :

$$
\int_0^\tau p'(\eta)\,d\eta=p(\tau)-p(0),
\qquad
p(\tau)=p(0)+\int_0^\tau p'(\eta)\,d\eta.
$$

Ici, $\eta$ est une variable muette qui parcourt l'intervalle d'intégration;
$\tau$ précise où cet intervalle se termine. Sur un court intervalle de
largeur $\Delta\eta$, la variation de $p$ vaut approximativement
$p'(\eta)\Delta\eta$. L'intégrale définie est la limite des sommes de ces
variations signées lorsque les largeurs des intervalles tendent vers zéro. En
ajoutant $p(0)$ à cette variation accumulée, on retrouve la valeur en $\tau$.
Les dérivées des polynômes sont continues; le théorème s'applique donc aux
morceaux utilisés ici. Pour un état vectoriel, la même identité s'applique à
chaque composante.

Les pentes échantillonnées $\mathbf f_{k,j}$ sont des dérivées par rapport au
temps physique, alors que $\mathbf p_k$ est exprimé en temps local. Comme
$t=t_k+h_k\tau$, la dérivation des fonctions composées donne

$$
\mathbf p_k'(\tau)
=h_k\dot{\mathbf x}_h(t_k+h_k\tau)
=h_k\sum_{j=1}^{s}\mathbf f_{k,j}\ell_j(\tau).
$$

Le facteur $h_k$ tient compte de la durée de l'intervalle physique : une
variation de $\Delta\tau$ en temps local correspond à $h_k\Delta\tau$ unités
de temps physique. Par exemple, une pente physique constante $\mathbf f$ qui
agit de $0$ à $\tau_i$ modifie l'état de $h_k\tau_i\mathbf f$.

En appliquant le théorème fondamental avec la condition à l'extrémité gauche
$\mathbf p_k(0)=\mathbf x_k$, on obtient maintenant le morceau d'état :

$$
\begin{aligned}
\mathbf{p}_k(\tau)
&=\mathbf{x}_k+\int_0^\tau\mathbf p_k'(\eta)\,d\eta\\
&=\mathbf{x}_k+h_k\sum_{j=1}^{s}
\left(\int_0^\tau\ell_j(\eta)\,d\eta\right)\mathbf{f}_{k,j}.
\end{aligned}
$$

La dernière expression s'obtient en substituant le polynôme des pentes et en
intégrant ses termes séparément; les pentes d'étage $\mathbf f_{k,j}$ sont
constantes par rapport à $\eta$. En $\tau=0$, chaque intégrale s'annule, et il
reste $\mathbf p_k(0)=\mathbf x_k$. La dérivée de l'expression redonne le
polynôme des pentes prescrit. Ainsi, l'état gauche stocké sélectionne
l'unique primitive qui a cette valeur de départ. L'intégration d'un polynôme
de degré au plus $s-1$ donne un morceau d'état de degré au plus $s$.

Pour l'exemple scalaire ci-dessus, $h_k=2$ et la pente en temps physique est
$1/2+2\tau$, de sorte que $p_k'(\tau)=1+4\tau$. À partir de $x_k=1$, on
obtient

$$
p_k(\tau)=1+\int_0^\tau(1+4\eta)\,d\eta
=1+\left[\eta+2\eta^2\right]_0^\tau
=1+\tau+2\tau^2.
$$

Les crochets signifient que l'on évalue la primitive à la borne supérieure et
que l'on soustrait sa valeur à la borne inférieure. En $\tau_i=1/2$, la
variation accumulée vaut $1$, d'où $p_k(1/2)=2$. En $\tau=1$, la variation
accumulée vaut $3$, d'où $p_k(1)=4$. Ce sont deux évaluations du même
polynôme : les deux intégrales commencent à l'extrémité gauche, mais l'une
s'arrête à un étage et l'autre à l'extrémité droite.

Cette intégration est exacte pour le polynôme des pentes interpolé. Elle ne
donne pas en général la solution exacte de l'EDO d'origine : entre les nœuds
de collocation, la pente interpolée peut différer de
$\mathbf f(\mathbf p_k(\tau),\mathbf u_h(t_k+h_k\tau),t_k+h_k\tau)$.

:::{figure} _static/collocation/stage-and-endpoint.svg
:label: fig-stage-and-endpoint
:alt: Un morceau d'état quadratique s'étend du temps local zéro à un, ce qui correspond au temps physique deux à quatre. Un point d'étage circulaire au temps local un demi a pour état deux; des extrémités carrées ont pour états un et quatre. Une flèche bleue de zéro à un demi montre l'étendue d'intégration d'une équation d'étage. Une flèche orange de zéro à un montre l'étendue d'intégration de l'équation d'extrémité. Les deux flèches partent de la même extrémité gauche.

Deux bornes d'intégration sur un même morceau polynomial. Ici, $t_k=2$,
$h_k=2$ et $p_k(\tau)=1+\tau+2\tau^2$. Le cercle marque l'étage $i$ en
$\tau_i=1/2$, ce qui correspond au temps physique $3$; les carrés marquent
les extrémités de l'intervalle. L'étendue bleue accumule une variation d'état
de $1$ de $\tau=0$ à $\tau_i$, ce qui donne $x_{k,i}=2$. L'étendue orange
accumule une variation d'état de $3$ de $\tau=0$ à $1$, ce qui donne
$x_{k+1}=4$. Les flèches marquent des étendues d'intégration, et non des pas
successifs d'un solveur numérique. Les variations proviennent de
l'intégration de la dérivée de la courbe d'état représentée. Seul l'étage $i$
est mis en évidence; chaque somme utilise tous les étages de pente de
l'intervalle.
:::

La construction par intégration détermine déjà la dérivée en chaque nœud de
collocation. Comme $\ell_j(\tau_i)=\delta_{ij}$, toutes valeurs candidates
des étages donnent

$$
\begin{gathered}
\frac{1}{h_k}\mathbf p_k'(\tau_i)
=\mathbf f_{k,i}
=\mathbf f(\mathbf x_{k,i},\mathbf u_{k,i},t_k+h_k\tau_i),\\
\forall\,k=0,\ldots,N-1,\quad i=1,\ldots,s.
\end{gathered}
$$

Cette identité est vraie par construction, avant même que les contraintes du
PNL soient satisfaites. Elle ne garantit pas encore que la courbe reconstruite
satisfait l'EDO : la dynamique du membre de droite est évaluée en l'état
stocké $\mathbf x_{k,i}$, qui peut différer de la valeur de la courbe
$\mathbf p_k(\tau_i)$.

La condition restante est l'accord entre ces deux valeurs d'état. Le point
$(\tau_i,\mathbf{x}_{k,i})$ doit se trouver sur le graphe de $\mathbf{p}_k$.
En imposant $\mathbf{x}_{k,i}=\mathbf{p}_k(\tau_i)$ à chaque étage de chaque
intervalle, on obtient les **équations d'étage**

$$
\mathbf{x}_{k,i}
=\mathbf{x}_k+h_k\sum_{j=1}^{s}A_{ij}\mathbf{f}_{k,j},
\qquad \forall\,k=0,\ldots,N-1,\quad i=1,\ldots,s.
$$

Leurs coefficients sont

$$
A_{ij}=\int_0^{\tau_i}\ell_j(\tau)\,d\tau,
\qquad i,j=1,\ldots,s.
$$

Chaque équation d'étage annule donc un résidu de concordance des états :

$$
\mathbf r^{\mathrm{int}}_{k,i}
=\mathbf x_{k,i}-\mathbf p_k(\tau_i)=\mathbf0,
\qquad \forall\,k=0,\ldots,N-1,\quad i=1,\ldots,s.
$$

Une fois ce résidu annulé, l'état stocké qui sert à évaluer $\mathbf f_{k,i}$
est l'état sur la courbe reconstruite. En substituant
$\mathbf x_{k,i}=\mathbf p_k(\tau_i)$ dans l'identité sur la dérivée
ci-dessus, on obtient alors l'EDO en ce nœud. Ainsi, la forme par intégration
impose l'EDO aux nœuds de collocation, même si ses résidus d'étage explicites
comparent des valeurs d'état plutôt que des pentes.

Par exemple, considérons $\dot x=x$ sur un intervalle de longueur $h_k=1$, à
partir de $x_k=1$, avec un seul nœud de collocation en $\tau_1=1/2$. Une
valeur d'étage d'essai $x_{k,1}=3$ fournit la pente $f_{k,1}=3$, et donc le
morceau d'état $p_k(\tau)=1+3\tau$. Sa valeur au milieu est $5/2$, et non la
valeur stockée $3$. La pente du polynôme vaut $3$, mais l'EDO évaluée sur la
courbe au milieu donne $5/2$. L'équation d'état $x_{k,1}=1+\tfrac12 x_{k,1}$
donne plutôt $x_{k,1}=2$. Le polynôme reconstruit est alors
$p_k(\tau)=1+2\tau$, dont l'état et la pente au milieu valent tous deux $2$.
La contrainte de concordance des états a rendu l'EDO vraie au nœud de
collocation.

La ligne $i$ de $A$ intègre chaque fonction de base des pentes de $0$ à
$\tau_i$. Ici, $i$ sélectionne l'étage de destination et $j$ parcourt tous les
étages de pente qui contribuent au polynôme. Toutes ces intégrales commencent
en $0$, y compris lorsque $i>1$; elles ne commencent pas à l'étage précédent.
Les équations d'étage sont des contraintes simultanées, puisque les pentes du
membre de droite dépendent des variables d'état et de commande en cours de
choix.

Par exemple, prenons trois étages de pente en
$(\tau_1,\tau_2,\tau_3)=(0,\tfrac12,1)$ et fixons la destination $i=2$.
Chaque intégrale de la ligne 2 de $A$ se termine alors en
$\tau_2=\tfrac12$. Faire varier $j$ sélectionne la fonction de base intégrée :
$\ell_1$, $\ell_2$ ou $\ell_3$.
@fig-collocation-destination-and-slopes montre ces trois aires signées. Même
si la troisième pente est évaluée à l'extrémité droite $\tau_3=1$, sa fonction
de base $\ell_3$ n'est pas nulle entre $0$ et $\tfrac12$. Cette pente
contribue donc aussi à l'état au milieu. La somme porte sur tous les
$j=1,\ldots,s$, et non seulement sur les étages tels que $j\leq i$.

:::{figure} _static/collocation/destination-and-slope-stages.svg
:label: fig-collocation-destination-and-slopes
:alt: Trois panneaux montrent les fonctions de base cardinales des pentes pour les nœuds zéro, un demi et un. La destination est fixée à l'étage i égal à deux, marqué par la même ligne verticale tiretée en un demi dans chaque panneau. Les aires signées ombrées de zéro à un demi donnent A21 égal à cinq vingt-quatrièmes, A22 égal à un tiers et A23 égal à moins un vingt-quatrième. Les trois pentes entrent dans l'équation d'état au milieu.

Fixer $i$ choisit la borne d'intégration; faire varier $j$ inclut chaque
fonction de base des pentes. Pour les nœuds $(0,\tfrac12,1)$, les courbes
bleues sont $\ell_1(\tau)=2\tau^2-3\tau+1$, $\ell_2(\tau)=4\tau(1-\tau)$ et
$\ell_3(\tau)=2\tau^2-\tau$. Chaque point bleu marque le nœud où sa fonction
de base vaut un. La ligne orange tiretée marque la borne supérieure commune
$\tau_i=\tau_2=\tfrac12$. Les aires signées ombrées forment la ligne 2 de
$A$ : $(A_{21},A_{22},A_{23})=(\tfrac5{24},\tfrac13,-\tfrac1{24})$. La
troisième aire est négative parce que $\ell_3$ est sous zéro sur cette
étendue; elle n'est pas absente du seul fait que l'étage 3 se situe après la
destination. L'équation du bas développe la somme pour un état scalaire; les
mêmes coefficients multiplient des pentes vectorielles dans le cas général.
:::

L'extrémité droite a pour coordonnée locale $1$. L'évaluation du même
polynôme en ce point utilise l'étendue d'intégration complète $[0,1]$, au lieu
de l'étendue $[0,\tau_i]$ utilisée pour l'étage $i$. En exigeant que cette
valeur $\mathbf{p}_k(1)$ soit égale à l'état d'extrémité stocké
$\mathbf{x}_{k+1}$, on obtient

$$
\boxed{
\mathbf{x}_{k+1}
=\mathbf{x}_k+h_k\sum_{j=1}^{s}b_j\mathbf{f}_{k,j},
\qquad
b_j=\int_0^1\ell_j(\tau)\,d\tau.
}
$$

Les coefficients $b_j$ intègrent les fonctions cardinales sur tout
l'intervalle de référence. Dans le code, $A_{ij}$ et $b_j$ s'obtiennent tous
deux en formant un polynôme primitive et en l'évaluant aux bornes
d'intégration. Le calcul de ces coefficients ne fait intervenir aucune
intégration répétée de la dynamique non linéaire. Le calcul détaillé de
@sec-collocation-coefficient-generator montre l'arithmétique des
coefficients. Si un étage se trouve en $\tau_i=1$, alors $A_{ij}=b_j$ pour
tout $j$, de sorte que son équation d'étage donne la même valeur du polynôme
que l'équation d'extrémité. Son état doit donc être égal à $\mathbf{x}_{k+1}$;
une implémentation peut partager cette variable.

L'équation d'extrémité est une contrainte de défaut : elle exige que
l'extrémité stockée coïncide avec l'extrémité obtenue en intégrant les pentes
du morceau $k$. Le morceau suivant part de cette même valeur stockée; les deux
morceaux se rejoignent donc :

$$
\mathbf{p}_k(1)=\mathbf{x}_{k+1}=\mathbf{p}_{k+1}(0).
$$

Cette variable partagée impose la continuité de l'état à la jonction. Si une
implémentation stocke des variables d'extrémité distinctes pour les deux
morceaux, elle doit imposer une égalité entre elles. Choisir une extrémité
comme nœud de collocation ne raccorde pas à lui seul les morceaux. La
continuité de l'état n'exige pas non plus la continuité de sa dérivée; des
morceaux voisins peuvent se rejoindre avec des pentes différentes.

:::{figure} _static/collocation/piecewise-trajectory.svg
:label: fig-piecewise-collocation-trajectory
:alt: Trois segments d'état quadratiques se rejoignent en des états d'extrémité communs en temps physique. Le segment du milieu est repris sur un axe de temps normalisé de zéro à un, avec deux valeurs d'étage intérieures et leurs pentes marquées.

Une trajectoire scalaire est formée de morceaux polynomiaux raccordés en des
états d'extrémité communs (carrés). Le morceau du milieu $p_k$ apparaît en
bleu dans les deux panneaux; seule sa coordonnée temporelle change. Les
valeurs d'étage (points orange) se situent en $\tau_1=1/3$ et $\tau_2=2/3$.
Les traits tangents du panneau inférieur ont pour pentes
$p_k'(\tau_j)=h_kf_{k,j}$, car cet axe utilise le temps normalisé. Les
équations d'étage placent les points sur le morceau, et le défaut
d'extrémité lui fait atteindre $x_{k+1}$, où commence le morceau suivant. Ce
schéma utilise des morceaux d'état quadratiques et des interpolants linéaires
des pentes.
:::

L'intégrale du coût courant peut être traitée par exactement la même
construction. Comme dans la réduction précédente de la forme de Bolza à celle
de Mayer, on introduit un état scalaire $y(t)$ qui enregistre le coût
accumulé depuis $t_0$ :

$$
y(t)=\int_{t_0}^{t}c(\mathbf{x}(s),\mathbf{u}(s),s)\,ds,
\qquad
\dot y(t)=c(\mathbf{x}(t),\mathbf{u}(t),t),\qquad y(t_0)=0.
$$

L'objectif d'origine devient $c_f(\mathbf{x}(t_f),t_f)+y(t_f)$. L'état
augmenté $(\mathbf{x}^{\top},y)^{\top}$ a deux taux de variation : $\mathbf{f}$
fournit le taux de l'état physique et $c$ le taux du coût. L'application de la
même méthode de collocation à cette EDO augmentée évalue les deux taux en
l'état, la commande et l'instant de chaque étage. En particulier, le taux de
coût à l'étage $j$ de l'intervalle $k$ est

$$
c_{k,j}:=c(\mathbf{x}_{k,j},\mathbf{u}_{k,j},t_k+h_k\tau_j).
$$

Soient $y_k$ et $y_{k,i}$ les valeurs du coût accumulé à l'extrémité gauche
de l'intervalle et à l'étage $i$. En intégrant l'interpolant du taux de coût
jusqu'à un étage et jusqu'à l'extrémité droite, on obtient

$$
y_{k,i}=y_k+h_k\sum_{j=1}^{s}A_{ij}c_{k,j},
\qquad
y_{k+1}=y_k+h_k\sum_{j=1}^{s}b_jc_{k,j}.
$$

Ce sont les mêmes équations d'étage et d'extrémité que pour $\mathbf{x}$, avec
$c_{k,j}$ à la place de $\mathbf{f}_{k,j}$. L'augmentation du coût accumulé
sur l'intervalle $k$ fournit donc sa contribution à l'objectif discret :

$$
c_k:=y_{k+1}-y_k
=h_k\sum_{j=1}^{s}b_jc_{k,j}
\approx\int_{t_k}^{t_{k+1}}c(\mathbf{x}_h(t),\mathbf{u}_h(t),t)\,dt.
$$

Ici, $\mathbf{u}_h$ est l'approximation par morceaux de la commande. Le $c_k$
à un seul indice est un coût d'intervalle, alors que $c_{k,j}$ est un taux de
coût échantillonné; le facteur $h_k$ leur donne des unités différentes. La
somme pondérée intègre exactement le polynôme qui interpole les taux de coût
échantillonnés. Elle approche l'intégrale du coût courant, car la fonction
composée $c(\mathbf{x}_h(t),\mathbf{u}_h(t),t)$ n'est pas forcément elle-même
ce polynôme. Une telle approximation par somme pondérée s'appelle une **règle
de quadrature**.

Comme $y_0=0$, la somme des incréments par intervalle donne
$y_N=\sum_{k=0}^{N-1}c_k$. On peut donc éliminer les variables de coût
accumulé et utiliser directement la somme dans l'objectif. L'introduction de
$y$ explique pourquoi l'application d'un même schéma de collocation aux deux
taux utilise les mêmes nœuds et les mêmes poids; elle n'exige pas d'ajouter
des variables de coût à l'implémentation. Une autre règle de quadrature est
aussi possible, les polynômes de l'état et de la commande étant alors évalués
aux nœuds de cette règle.

(choosing-a-polynomial-for-the-control)=
### Choisir un polynôme pour la commande

L'état et la commande jouent des rôles différents. L'EDO contraint la dérivée
de l'état; l'intégration d'un interpolant de $s$ pentes donne donc un
polynôme d'état de degré au plus $s$. Il n'existe dans ce problème aucune EDO
correspondante pour la commande. Sa représentation est un choix distinct : une
commande constante sur chaque intervalle, une droite entre des commandes aux
extrémités, ou un polynôme de degré plus élevé.

Par exemple, avec deux pentes aux extrémités, le polynôme d'état est
quadratique. La commande peut tout de même être la droite

$$
\mathbf u_h(t_k+h_k\tau)
=(1-\tau)\mathbf u_k+\tau\mathbf u_{k+1}.
$$

Ces degrés n'ont pas à concorder. De plus, substituer des courbes polynomiales
d'état et de commande dans une fonction $\mathbf f$ non linéaire ne produit
pas forcément un polynôme. La collocation fait concorder la dérivée de l'état
avec $\mathbf f$ aux nœuds choisis, au lieu d'exiger que l'EDO soit vérifiée
identiquement entre eux.

Pour rendre explicite le choix de la commande, on choisit des nœuds de support
de commande distincts $\rho_0,\ldots,\rho_{d_u}\in[0,1]$ et on construit leurs
fonctions cardinales $\psi_0,\ldots,\psi_{d_u}$. Les variables de décision
$\widehat{\mathbf u}_{k,r}$ sont les valeurs de la commande en ces nœuds de
support :

$$
\mathbf u_h(t_k+h_k\tau)
=\sum_{r=0}^{d_u}\widehat{\mathbf u}_{k,r}\psi_r(\tau),
\qquad
\mathbf u_{k,j}
=\sum_{r=0}^{d_u}B_{jr}\widehat{\mathbf u}_{k,r},
\quad B_{jr}:=\psi_r(\tau_j).
$$

Ainsi, les commandes d'étage $\mathbf u_{k,j}$ sont calculées à partir des
variables de commande; ce ne sont des variables indépendantes que si la
représentation admet des valeurs indépendantes à tous ces étages. Pour une
commande constante, $d_u=0$ et chaque composante de $B$ vaut un. Pour une
commande linéaire s'appuyant sur $0,1$, la ligne $j$ de $B$ est
$(1-\tau_j,\tau_j)$.

Utiliser plus de valeurs de support de commande que d'étages de collocation
rend certaines variations de la commande invisibles pour ces échantillons
d'étage. Les exemples ci-dessous utilisent $d_u+1\leq s$; une représentation
plus riche de la commande exige un échantillonnage supplémentaire ou d'autres
contraintes pour tenir compte de ces variations.

L'EDO exige des états continus, mais elle peut admettre des commandes qui
sautent d'un intervalle à l'autre. Des polynômes de commande indépendants
n'ont donc besoin d'aucune contrainte de continuité, à moins que le modèle ou
la paramétrisation choisie n'en exige une. Partager les commandes aux
extrémités entre des morceaux linéaires voisins impose la continuité. Avec
des variables locales distinctes, l'égalité équivalente est

$$
\sum_{r=0}^{d_u}\psi_r(1)\widehat{\mathbf u}_{k,r}
=\sum_{r=0}^{d_u}\psi_r(0)\widehat{\mathbf u}_{k+1,r}.
$$

À un saut, les étages aux extrémités utilisent la commande propre à leur
intervalle. Enfin, des bornes sur les valeurs de support garantissent des
bornes sur tout un segment de commande constant ou linéaire, puisque ses
valeurs sont des combinaisons convexes de ses extrémités. Un polynôme de degré
plus élevé peut dépasser ces valeurs; les bornes sur les valeurs de support
ne suffisent alors plus à donner cette garantie.

### Assemblage du programme non linéaire

Les représentations de l'état et de la commande précisent maintenant quelles
quantités l'optimiseur choisit et lesquelles il évalue à partir de ces choix.
Les états du maillage $\mathbf x_k,\mathbf x_{k+1}$ sont des variables de
décision partagées aux extrémités de l'intervalle $k$. Les états d'étage
$\mathbf x_{k,j}$ sont des variables de décision aux nœuds de collocation; un
étage situé à une extrémité peut utiliser l'état du maillage lui-même. Pour la
commande, l'optimiseur choisit les valeurs de support
$\widehat{\mathbf u}_{k,r}$ en $\rho_r$. L'évaluation du polynôme de commande
en $\tau_j$ donne $\mathbf u_{k,j}$. Avec $\mathbf x_{k,j}$ et l'instant
$t_k+h_k\tau_j$, cette commande évaluée fournit les entrées de $\mathbf f$ et
de $c$.

:::{figure} _static/collocation/nlp-quantities.svg
:label: fig-collocation-nlp-quantities
:alt: Les courbes de l'état et de la commande partagent un intervalle avec des nœuds de collocation en un tiers et deux tiers. Des carrés bleus pleins marquent les variables d'état du maillage partagées, des cercles bleus pleins les variables d'état d'étage, et des losanges bleus pleins les variables de support de commande en zéro et en un. Un cercle orange vide marque la commande d'étage évaluée au moyen de B. L'état d'étage sélectionné, la commande évaluée et l'instant déterminent les taux de dynamique et de coût à droite.

Variables de décision et quantités évaluées sur un intervalle. Les courbes
montrent un exemple scalaire avec deux nœuds de collocation, $\tau_1=1/3$ et
$\tau_2=2/3$; l'étage mis en évidence est $j=2$. Les carrés marquent les états
du maillage partagés avec les intervalles voisins, et les cercles pleins les
états d'étage. Les losanges marquent les valeurs de commande indépendantes en
$\rho_0=0$ et $\rho_1=1$. Pour cette commande linéaire,
$B_{j,:}=(1-\tau_j,\tau_j)$ évalue le cercle vide $\mathbf u_{k,j}$ à partir
des deux losanges. Les quantités $\mathbf f_{k,j}$ et $c_{k,j}$ sont ensuite
évaluées en l'état, la commande et le temps physique de l'étage. Les marqueurs
bleus pleins sont des variables de décision; la valeur de commande et les taux
en orange sont calculés à partir d'elles. Les courbes représentent un choix
admissible : les contraintes d'étage et d'extrémité placent les variables
d'état sur le polynôme intégré. Si un nœud de collocation est une extrémité,
son état d'étage peut être identifié à la variable de maillage partagée.
:::

On rassemble les valeurs indépendantes de l'état et de la commande sur tous
les intervalles dans le vecteur de décision

$$
\mathbf z=\operatorname{col}\!\left(
\{\mathbf x_k\}_{k=0}^{N},
\{\mathbf x_{k,j}\}_{k=0,\,j=1}^{N-1,\,s},
\{\widehat{\mathbf u}_{k,r}\}_{k=0,\,r=0}^{N-1,\,d_u}
\right).
$$

Ici, $\operatorname{col}$ empile les vecteurs énumérés en une seule colonne, en
omettant les variables en double lorsqu'un étage partage une extrémité du
maillage. Les commandes d'étage sont évaluées au moyen de $B$, et les
éventuelles égalités de continuité de la commande retenues sont ajoutées aux
contraintes. Si $t_f$ est optimisé, il est aussi inclus dans $\mathbf z$. Le
problème d'optimisation fini obtenu a la forme

$$
\begin{aligned}
\underset{\mathbf{z}}{\operatorname{minimiser}}\quad&
c_f(\mathbf{x}_N,t_f)+\sum_{k=0}^{N-1}c_k\\
\text{sous les contraintes}\quad&
\text{équations d'étage},\\
&\text{défauts d'extrémité et continuité},\\
&\text{contraintes aux limites, de chemin et de bornes}.
\end{aligned}
$$

Dans la notation des PNL du chapitre précédent, $F(\mathbf z)$ est cet
objectif scalaire, $H(\mathbf z)=\mathbf0$ empile les égalités d'étage,
d'extrémité et aux limites, et $G(\mathbf z)\leq\mathbf0$ empile les
inégalités échantillonnées de chemin et de bornes. La pente
$\mathbf f_{k,j}$ est un vecteur, distinct de $F$.

Les contraintes de chemin et de bornes d'origine s'appliquent à chaque instant
continu. Ce PNL les impose en des points choisis, habituellement ses nœuds de
support ou de collocation. L'admissibilité en ces nœuds n'exclut pas une
violation entre eux. Vérifier le polynôme entre les nœuds et simuler la
commande choisie fournissent des diagnostics complémentaires, développés plus
loin.

Chaque contrainte d'intervalle ne touche que des variables d'étage locales et
les états des extrémités voisines. Par conséquent, la plupart des dérivées des
contraintes par rapport aux variables de décision sont nulles. Lorsque les
variables sont ordonnées dans le temps, les blocs non nuls se situent près de
la diagonale de la jacobienne, la matrice des dérivées des contraintes. Les
solveurs de PNL peuvent exploiter ce motif creux en bande par blocs. Les
tableaux $A$ et $b$ sont des constantes numériques calculées avant
l'optimisation.

### Forme équivalente par dérivation

La forme par dérivation utilise les mêmes nœuds de collocation $\tau_i$ que la
forme par intégration. Elle change la condition de cohérence qui est intégrée
à la représentation et celle qui doit être imposée par un résidu. La forme
par intégration construit une courbe dont les dérivées nodales concordent avec
les pentes d'étage évaluées, puis contraint ses valeurs à concorder avec les
états d'étage stockés. La forme par dérivation construit une courbe à partir
de valeurs de support de l'état, évalue ses états d'étage directement sur
cette courbe, et contraint ses dérivées à concorder avec l'EDO.

Pour représenter les mêmes polynômes d'état, on conserve le même degré.
L'intégrale du polynôme des pentes de degré $s-1$ est de degré au plus $s$;
on prend donc $d=s$ et on choisit $d+1$ nœuds de support
$\sigma_0,\ldots,\sigma_d$ pour le polynôme d'état. On note
$\ell_r^{\mathrm{state}}$ leurs fonctions cardinales pour distinguer cette
base des nœuds de support de la base des nœuds de pente $\ell_j$. On note
$\widehat{\mathbf x}_{k,r}$ les valeurs de support; elles ne sont pas
forcément les valeurs d'étage $\mathbf x_{k,j}$ en $\tau_j$ :

$$
\mathbf{x}_h(t_k+h_k\tau)
=\sum_{r=0}^{d}\widehat{\mathbf x}_{k,r}\ell_r^{\mathrm{state}}(\tau).
$$

En un nœud de collocation $\tau_i$, l'état est une somme pondérée de ses
valeurs de support, et sa dérivée par rapport à $\tau$ est une autre somme
pondérée. Les tableaux fixes qui contiennent ces poids sont

$$
E_{ir}=\ell_r^{\mathrm{state}}(\tau_i),
\qquad
D_{ir}=(\ell_r^{\mathrm{state}})'(\tau_i).
$$

La $i$-ième ligne de $E$ calcule $\mathbf p_k(\tau_i)$, le point du polynôme
d'état que l'on fournit à la fonction de dynamique. La $i$-ième ligne de $D$
calcule $\mathbf p_k'(\tau_i)$, la pente de la tangente mesurée par unité de
temps local $\tau$. Les deux calculs utilisent les mêmes états de support
stockés $\widehat{\mathbf x}_{k,r}$; ils n'introduisent pas de nouvelles
variables de décision. En particulier, l'état d'étage est défini par
$\mathbf x_{k,i}=\mathbf p_k(\tau_i)=\sum_r E_{ir}\widehat{\mathbf x}_{k,r}$.
La cohérence des valeurs d'état est donc vérifiée pour tout tableau de support
candidat; il n'y a pas de résidu distinct de concordance des états à un étage.

Pour comparer cette pente de la tangente au taux renvoyé par $\mathbf f$, il
faut mesurer les deux en temps physique. Un incrément de temps local
$\Delta\tau$ correspond à un incrément de temps physique
$\Delta t=h_k\Delta\tau$. On divise donc la pente en temps local par $h_k$
pour obtenir la pente en temps physique : c'est la dérivation des fonctions
composées $d/dt=(1/h_k)d/d\tau$ utilisée plus haut. La contrainte restante
annule le résidu de pente en temps physique :

$$
\begin{gathered}
\mathbf r^{\mathrm{diff}}_{k,i}
=\frac{1}{h_k}\mathbf p_k'(\tau_i)
-\mathbf f\!\left(\mathbf p_k(\tau_i),\mathbf u_{k,i},t_k+h_k\tau_i\right)
=\mathbf0,\\
\forall\,k=0,\ldots,N-1,\quad i=1,\ldots,s.
\end{gathered}
$$

Les deux pentes sont évaluées au même point de la courbe d'état.
Contrairement à la cohérence des valeurs d'état, leur égalité n'est pas
intégrée à l'interpolant de l'état. En écrivant ces évaluations au moyen de
$E,D$ et en multipliant par $h_k$, on obtient

$$
\begin{gathered}
\boxed{
\sum_{r=0}^{d}D_{ir}\widehat{\mathbf x}_{k,r}
=h_k \mathbf{f}\left(
\sum_{r=0}^{d}E_{ir}\widehat{\mathbf x}_{k,r},
\mathbf{u}_{k,i},
t_k+h_k\tau_i
\right).
}\\
\forall\,k=0,\ldots,N-1,\quad i=1,\ldots,s.
\end{gathered}
$$

Le membre de gauche est la pente du polynôme par rapport à $\tau$; le membre
de droite est le taux en temps physique de l'EDO multiplié par $h_k$. Le
polynôme quadratique scalaire de @fig-collocation-evaluation-differentiation
montre les deux calculs en un nœud de collocation qui n'est pas un nœud de
support.

:::{figure} _static/collocation/evaluation-and-differentiation.svg
:label: fig-collocation-evaluation-differentiation
:alt: Trois états de support carrés définissent un polynôme quadratique. En un nœud de collocation situé entre les nœuds de support, E calcule la valeur indiquée par un cercle vide et D calcule la pente de la tangente tiretée. En divisant cette pente par la longueur de l'intervalle, on obtient le taux en temps physique qui doit concorder avec l'EDO.

Évaluation et dérivation d'un même polynôme d'état. Les états de support
$(1,2,4)$ en $\sigma=(0,\tfrac12,1)$ définissent $p_k(\tau)=1+\tau+2\tau^2$.
Au nœud de collocation mis en évidence $\tau_i=\tfrac14$, $E$ calcule l'état
$p_k(\tau_i)=\tfrac{11}{8}$, tandis que $D$ calcule la pente de la tangente
$p_k'(\tau_i)=2$. Avec $h_k=2$, la pente en temps physique vaut $2/2=1$. Pour
la dynamique $\dot x=u$ et la commande d'étage $u_{k,i}=1$, l'EDO renvoie
aussi $1$; la contrainte de collocation est donc vérifiée en ce nœud. En
prenant $t_k=2$, le nœud se situe au temps physique
$t_k+h_k\tau_i=\tfrac52$.
:::

En notation matricielle abrégée, toutes les contraintes nodales s'écrivent
$D\mathsf X_k=h_k\mathsf F_k$. Ici, $\mathsf X_k\in\mathbb R^{(d+1)\times n}$
a pour lignes les états de support, et $\mathsf F_k\in\mathbb R^{s\times n}$
a pour lignes les pentes d'étage; aucun des deux n'est un vecteur d'état
unique. L'évaluation aux deux extrémités relie ce polynôme aux états du
maillage partagés :

$$
\mathbf{x}_{k}=\sum_{r=0}^{d}\ell_r^{\mathrm{state}}(0)\widehat{\mathbf x}_{k,r},
\qquad
\mathbf{x}_{k+1}=\sum_{r=0}^{d}\ell_r^{\mathrm{state}}(1)\widehat{\mathbf x}_{k,r}.
$$

Lorsque les nœuds de support et de collocation coïncident, la propriété
cardinale fait que $E$ sélectionne directement l'état stocké correspondant.
Les formes par valeurs de pente et par valeurs d'état décrivent la même
construction polynomiale; l'une intègre des pentes nodales, l'autre dérive
des états nodaux.

Ce schéma d'implémentation est courant en collocation directe {cite:p}`Kelly2017DirectCollocation,Andersson2019CasADi`. L'[exemple officiel de collocation directe de CasADi](https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py) construit des tableaux fixes de dérivation, d'extrémité et de quadrature à partir de polynômes de Lagrange, tandis que les variables du PNL restent des valeurs d'état et de commande.

Dans la forme par intégration, la concordance avec les pentes d'étage évaluées
est intégrée à la construction du polynôme, tandis que la concordance avec les
états d'étage exige des contraintes. Dans la forme par dérivation, la
concordance des états d'étage avec la courbe est intégrée à l'évaluation,
tandis que la concordance avec les pentes de l'EDO exige des contraintes. Les
deux formes relient aussi le polynôme aux extrémités du maillage partagées.
Avec les mêmes nœuds de collocation, le même degré d'état et la même
représentation de la commande, leurs résidus nuls décrivent les mêmes
trajectoires de collocation, même si leurs valeurs de résidu peuvent différer
pour un candidat non admissible. La recette qui mène des nœuds au PNL pour la
forme par dérivation est développée dans
@sec-collocation-differentiation-from-nodes.

## Transcriptions d'ordre bas

Quelles règles d'intégration familières apparaissent lorsque les ensembles de
polynômes et de nœuds sont réduits à leurs choix d'ordre le plus bas?

La construction générale devient concrète lorsqu'on ne retient qu'une ou deux
valeurs de pente sur chaque intervalle. Ces cas redonnent des formules
d'intégration familières, mais les formules apparaissent maintenant comme des
contraintes à l'intérieur d'un PNL. Pour les problèmes concrets ci-dessous,
on fixe le maillage et le temps final. L'application aux limites
$\mathbf h(\mathbf x_0,\mathbf x_N,t_N)=\mathbf0$ comprend l'état initial
prescrit et les éventuelles conditions terminales; $\mathbf g\leq\mathbf0$
comprend les contraintes de chemin et les bornes sur l'état et la commande.
Chaque programme présenté précise les points où ces inégalités sont imposées.
On peut ajouter des points de vérification supplémentaires, et un rejeu en
temps continu vérifie tout de même le résultat entre les nœuds.

### Une seule valeur de pente : Euler explicite et implicite

Une seule valeur de pente détermine une dérivée constante sur chaque
intervalle. On choisit aussi une commande constante $\mathbf u_{k,1}$ sur cet
intervalle : $\mathbf u_h(t_k+h_k\tau)=\mathbf u_{k,1}$. Des intervalles
différents peuvent utiliser des constantes différentes. L'état est linéaire,
même si la commande est constante. L'exercice suivant applique la
construction précédente pour retrouver les deux méthodes d'Euler. La solution
donne une recette qui s'étend aussi à plusieurs nœuds de pente.

````{exercise} Dériver une transcription à partir d'une seule pente
:label: ex-collocation-one-slope-recipe

On considère $\dot{\mathbf x}=\mathbf{f}(\mathbf x,\mathbf u,t)$ sur $[t_k,t_{k+1}]$, de durée $h_k$ et de
temps normalisé $\tau=(t-t_k)/h_k$. Approchez la dérivée en temps physique par
une constante qui concorde avec l'EDO en un seul nœud de collocation $\tau_1$.

Effectuez la construction pour $\tau_1=0$, puis pour $\tau_1=1$. Pour chaque choix :

1. Construisez la fonction cardinale et identifiez la pente de l'EDO qu'elle multiplie.
2. Intégrez à partir de $\mathbf{x}_k$ pour obtenir le polynôme d'état $\mathbf{p}_k(\tau)$, puis
   imposez qu'il concorde avec l'extrémité droite stockée $\mathbf{x}_{k+1}$.
3. Dérivez la contribution de l'intervalle au coût en appliquant la même
   construction à $\dot y=c(\mathbf x,\mathbf u,t)$.
4. Identifiez la méthode, donnez le degré de son polynôme d'état et
   expliquez si son équation d'extrémité peut être évaluée directement dans
   une simulation vers l'avant avec des commandes prescrites.
````

````{solution} ex-collocation-one-slope-recipe
:class: dropdown

La dérivation suit cinq étapes : choisir où l'EDO est évaluée, interpoler sa
pente, intégrer cet interpolant, faire concorder l'extrémité stockée, et
intégrer le taux de coût avec les mêmes poids.

1. **Choisir l'étage et évaluer sa pente.** Le nœud de gauche utilise l'état,
   la commande et l'instant au début de l'intervalle; le nœud de droite
   utilise ceux de sa fin :

   $$
   \mathbf{f}_{k,1}=\begin{cases}
   \mathbf{f}(\mathbf{x}_k,\mathbf u_{k,1},t_k),&\tau_1=0,\\
   \mathbf{f}(\mathbf{x}_{k+1},\mathbf u_{k,1},t_{k+1}),&\tau_1=1.
   \end{cases}
   $$

2. **Interpoler la dérivée en temps physique.** Avec un seul nœud, la
   fonction cardinale est la constante $\ell_1(\tau)=1$, puisqu'elle doit
   valoir un en ce nœud. Par conséquent, $\dot{\mathbf{x}}_h(t_k+h_k\tau)=\mathbf{f}_{k,1}$ sur tout l'intervalle.
   Son poids d'intégration sur l'intervalle complet est
   $b_1=\int_0^1\ell_1(\tau)\,d\tau=1$.

3. **Intégrer à partir de l'état gauche.** La dérivée en temps de référence
   est $\mathbf{p}_k'(\tau)=h_k\mathbf{f}_{k,1}$. En intégrant et en imposant $\mathbf{p}_k(0)=\mathbf{x}_k$, on obtient

   $$
   \mathbf{p}_k(\tau)=\mathbf{x}_k+h_k\tau \mathbf{f}_{k,1}.
   $$

   Le polynôme d'état est donc de degré au plus un. Le facteur $h_k$ tient
   compte de la durée physique de l'intervalle.

4. **Faire concorder l'extrémité droite stockée.** En posant $\mathbf{p}_k(1)=\mathbf{x}_{k+1}$, on obtient le
   défaut $\mathbf{x}_{k+1}-\mathbf{x}_k-h_k\mathbf{f}_{k,1}=\mathbf0$. En substituant les deux choix de pente, on obtient

   $$
   \begin{aligned}
   \mathbf{x}_{k+1}-\mathbf{x}_k-h_k\mathbf{f}(\mathbf{x}_k,\mathbf u_{k,1},t_k)&=\mathbf0
   &&\text{(Euler explicite)},\\
   \mathbf{x}_{k+1}-\mathbf{x}_k-h_k\mathbf{f}(\mathbf{x}_{k+1},\mathbf u_{k,1},t_{k+1})&=\mathbf0
   &&\text{(Euler implicite)}.
   \end{aligned}
   $$

   En simulation vers l'avant, Euler explicite calcule l'état suivant à
   partir de données connues à l'extrémité gauche. Euler implicite place
   l'état suivant inconnu à l'intérieur de $\mathbf{f}$; il exige donc en
   général la résolution d'une équation, non linéaire lorsque la dynamique
   dépend de cet état de façon non linéaire. En transcription directe, les
   deux extrémités sont des variables d'optimisation et l'un ou l'autre défaut
   est imposé comme contrainte d'égalité.

5. **Appliquer la même règle d'intégration au taux de coût.** L'unique taux
   de coût échantillonné est constant dans l'approximation; son intégrale est
   donc ce taux multiplié par $h_k$ :

   $$
   c_k=\begin{cases}
   h_kc(\mathbf{x}_k,\mathbf u_{k,1},t_k),&\tau_1=0,\\
   h_kc(\mathbf{x}_{k+1},\mathbf u_{k,1},t_{k+1}),&\tau_1=1.
   \end{cases}
   $$

   Ce sont les approximations par l'extrémité gauche et par l'extrémité
   droite de l'intégrale du coût courant sur l'intervalle. Elles utilisent le
   même étage que la contrainte de dynamique correspondante.

Avec davantage de nœuds de pente, la démarche reste la même. La fonction
cardinale constante est remplacée par plusieurs polynômes cardinaux, et leurs
intégrales fournissent les poids des équations d'étage, du défaut d'extrémité
et de la contribution au coût.
````

### Le PNL d'Euler complet

Soit $\theta=0$ pour Euler explicite et $\theta=1$ pour Euler implicite. Les
deux méthodes optimisent les états du maillage $\mathbf x_0,\ldots,\mathbf x_N$
et les $N$ commandes d'intervalle $\mathbf u_{k,1}$. Il n'y a pas de variables
d'état d'étage supplémentaires une fois l'unique étage identifié à son
extrémité :

$$
\begin{aligned}
\min_{\{\mathbf x_k\}_{k=0}^N,\,\{\mathbf u_{k,1}\}_{k=0}^{N-1}}
\quad &c_f(\mathbf x_N,t_N)
+\sum_{k=0}^{N-1}h_kc(\mathbf x_{k+\theta},\mathbf u_{k,1},t_{k+\theta})\\
\text{sous les contraintes}\quad
&\mathbf x_{k+1}-\mathbf x_k
-h_k\mathbf f(\mathbf x_{k+\theta},\mathbf u_{k,1},t_{k+\theta})=\mathbf0,\\
&\mathbf g(\mathbf x_{k+\theta},\mathbf u_{k,1},t_{k+\theta})\leq\mathbf0,
\quad k=0,\ldots,N-1,\\
&\mathbf h(\mathbf x_0,\mathbf x_N,t_N)=\mathbf0.
\end{aligned}
$$

Ici, $k+\theta$ sélectionne une extrémité, et non un instant intermédiaire.
Les contraintes de chemin utilisent la même extrémité que la dynamique; les
bornes requises à l'autre extrémité peuvent aussi y être imposées. Les deux
PNL ne diffèrent que par l'endroit où la pente et le taux de coût sont
évalués. Tous deux sont résolus simultanément pour la trajectoire entière.

(endpoint-slope-values-trapezoidal-transcription)=
### Valeurs de pente aux extrémités : transcription des trapèzes

On utilise des commandes continues et linéaires par morceaux, avec des
variables d'extrémité partagées $\mathbf u_0,\ldots,\mathbf u_N$. On choisit
les nœuds de collocation aux extrémités $\tau_1=0$ et $\tau_2=1$, et on abrège
les deux pentes de l'EDO par

$$
\mathbf{f}_k=\mathbf{f}(\mathbf{x}_k,\mathbf{u}_k,t_k),\qquad
\mathbf{f}_{k+1}=\mathbf{f}(\mathbf{x}_{k+1},\mathbf{u}_{k+1},t_{k+1}).
$$

Pour cette formule aux extrémités, on désigne les fonctions cardinales par
leurs positions, $0$ et $1$, plutôt que par les numéros d'étage $1$ et $2$ :

$$
\ell_0(\tau)=1-\tau,\qquad
\ell_1(\tau)=\tau.
$$

L'interpolant de la dérivée est la droite qui relie les deux pentes :

$$
\dot{\mathbf{x}}_h(t_k+h_k\tau)
=(1-\tau)\mathbf{f}_k+\tau \mathbf{f}_{k+1}.
$$

Il vaut $\mathbf{f}_k$ en $\tau=0$ et $\mathbf{f}_{k+1}$ en $\tau=1$. En
l'intégrant à partir de l'état gauche stocké, on obtient l'approximation
continue de l'état

$$
\mathbf{x}_h(t_k+h_k\tau)
=\mathbf{x}_k+h_k\left[
\left(\tau-\frac{\tau^2}{2}\right)\mathbf{f}_k
+\frac{\tau^2}{2}\mathbf{f}_{k+1}
\right].
$$

L'approximation de l'état est quadratique, même si une implémentation ne
stocke parfois que ses états et ses pentes aux extrémités. En évaluant ce
polynôme en $\tau=1$ et en l'égalant à l'extrémité stockée, on obtient

$$
\boxed{
\mathbf{x}_{k+1}-\mathbf{x}_k
-\frac{h_k}{2}
\left[
\mathbf{f}(\mathbf{x}_k,\mathbf{u}_k,t_k)
+\mathbf{f}(\mathbf{x}_{k+1},\mathbf{u}_{k+1},t_{k+1})
\right]
=\mathbf0.
}
$$

Les coefficients $1/2$ sont les intégrales des deux fonctions cardinales
linéaires. C'est le défaut des trapèzes dérivé dans l'exemple d'ouverture. En
appliquant les mêmes poids aux extrémités au coût courant, on obtient

$$
c_k
=
\frac{h_k}{2}
\left[
c(\mathbf{x}_k,\mathbf{u}_k,t_k)
+c(\mathbf{x}_{k+1},\mathbf{u}_{k+1},t_{k+1})
\right].
$$

On fait la moyenne des deux coûts aux extrémités, puis on la multiplie par la
longueur de l'intervalle. L'intégration augmente le degré : un interpolant
linéaire de la dérivée produit un interpolant quadratique de l'état. Cette
construction apparaît aussi chez {cite:t}`Kelly2017DirectCollocation`.

### Le PNL des trapèzes complet

Les variables de décision sont les états et les commandes du maillage. On
écrit $\mathbf f_k=\mathbf f(\mathbf x_k,\mathbf u_k,t_k)$ et
$c_k^{\mathrm{rate}}=c(\mathbf x_k,\mathbf u_k,t_k)$ pour des quantités
évaluées, et non pour des variables d'optimisation supplémentaires. Le
problème transcrit est alors

$$
\begin{aligned}
\min_{\{\mathbf x_k,\mathbf u_k\}_{k=0}^{N}}\quad
&c_f(\mathbf x_N,t_N)
+\sum_{k=0}^{N-1}\frac{h_k}{2}
\left(c_k^{\mathrm{rate}}+c_{k+1}^{\mathrm{rate}}\right)\\
\text{sous les contraintes}\quad
&\mathbf x_{k+1}-\mathbf x_k
-\frac{h_k}{2}(\mathbf f_k+\mathbf f_{k+1})=\mathbf0,
\quad k=0,\ldots,N-1,\\
&\mathbf g(\mathbf x_k,\mathbf u_k,t_k)\leq\mathbf0,
\quad k=0,\ldots,N,\\
&\mathbf h(\mathbf x_0,\mathbf x_N,t_N)=\mathbf0.
\end{aligned}
$$

Entre les extrémités, l'état est le polynôme quadratique obtenu plus haut,
tandis que la commande est linéaire. Les états et les commandes du maillage
partagés relient les morceaux adjacents; cette représentation ne demande
aucune équation de continuité distincte.

## Transcription de Hermite--Simpson

L'information sur l'état et la pente au milieu peut-elle augmenter l'ordre de
la transcription sans exiger un polynôme de degré élevé sur tout l'horizon?

Hermite--Simpson prolonge la construction des trapèzes en ajoutant l'état au
milieu $\mathbf{x}_{k+\frac12}$. On conserve la même commande linéaire par
morceaux; sa valeur au milieu est donc déjà déterminée :

$$
\mathbf u_{k+\frac12}=\frac{\mathbf u_k+\mathbf u_{k+1}}{2}.
$$

On évalue la pente de l'EDO au milieu au moyen de cette commande :

$$
\mathbf{f}_{k+\frac12}=\mathbf{f}\left(
\mathbf{x}_{k+\frac12},\mathbf{u}_{k+\frac12},t_k+\frac{h_k}{2}
\right).
$$

Les notations $\mathbf x_{k+\frac12}$ et $\mathbf u_{k+\frac12}$ désignent des
valeurs au temps physique $t_k+h_k/2$, et non une autre extrémité du maillage.

Les trois nœuds d'étage sont $\tau_1=0$, $\tau_2=\tfrac12$ et $\tau_3=1$. On désigne leurs fonctions cardinales par $0,m,1$ pour l'extrémité gauche, le milieu et l'extrémité droite. L'interpolant de la dérivée est le
polynôme quadratique construit à partir des trois fonctions cardinales
suivantes :

$$
\ell_0(\tau)=2\tau^2-3\tau+1,\qquad
\ell_m(\tau)=4\tau(1-\tau),\qquad
\ell_1(\tau)=2\tau^2-\tau.
$$

L'intégration de chaque fonction sur tout l'intervalle de référence donne

$$
b_0=\frac16,\qquad
b_m=\frac46,\qquad
b_1=\frac16.
$$

Ce sont les poids de quadrature familiers de Simpson. En les substituant dans
l'équation d'extrémité générale, on obtient le défaut de Simpson :

$$
\boxed{
\mathbf{x}_{k+1}-\mathbf{x}_k
-\frac{h_k}{6}
\left(\mathbf{f}_k+4\mathbf{f}_{k+\frac12}+\mathbf{f}_{k+1}\right)
=\mathbf0.
}
$$

L'état au milieu doit aussi se trouver sur le polynôme cubique obtenu en
intégrant la dérivée quadratique. L'intégration jusqu'à $\tau=\tfrac12$
seulement donne

$$
\mathbf{x}_{k+\frac12}
=\mathbf{x}_k+\frac{h_k}{24}
\left(5\mathbf{f}_k+8\mathbf{f}_{k+\frac12}-\mathbf{f}_{k+1}\right).
$$

Cette forme contient encore la pente au milieu. Le défaut d'extrémité donne

$$
4\mathbf{f}_{k+\frac12}
=\frac{6}{h_k}(\mathbf{x}_{k+1}-\mathbf{x}_k)-\mathbf{f}_k-\mathbf{f}_{k+1}.
$$

En substituant cette expression dans l'équation du milieu et en regroupant les
états et les pentes aux extrémités, on obtient

$$
\boxed{
\mathbf{x}_{k+\frac12}
=\frac{\mathbf{x}_k+\mathbf{x}_{k+1}}{2}
+\frac{h_k}{8}\left(\mathbf{f}_k-\mathbf{f}_{k+1}\right).
}
$$

La relation sur l'état au milieu et la définition de $\mathbf{f}_{k+\frac12}$
imposent ensemble l'EDO au milieu. L'interpolant de l'état obtenu est cubique,
car il intègre une dérivée quadratique. Le nom reflète ses deux ingrédients :
l'état est un interpolant cubique de Hermite déterminé par l'information sur
l'état et la pente, et son défaut d'extrémité utilise les poids de Simpson.

### Le PNL de Hermite--Simpson complet

Avec des commandes linéaires, on optimise les états du maillage, les commandes
du maillage et les états au milieu. Pour $q=k$ ou $k+\tfrac12$, on abrège
$\mathbf f_q=\mathbf f(\mathbf x_q,\mathbf u_q,t_q)$ et
$c_q^{\mathrm{rate}}=c(\mathbf x_q,\mathbf u_q,t_q)$, avec
$t_{k+\frac12}=t_k+h_k/2$ et la commande au milieu donnée par la moyenne aux
extrémités. Le programme complet est

$$
\begin{aligned}
\min_{\{\mathbf x_k,\mathbf u_k\}_{k=0}^{N},\,
\{\mathbf x_{k+\frac12}\}_{k=0}^{N-1}}\quad
&c_f(\mathbf x_N,t_N)
+\sum_{k=0}^{N-1}\frac{h_k}{6}
\left(c_k^{\mathrm{rate}}+4c_{k+\frac12}^{\mathrm{rate}}+c_{k+1}^{\mathrm{rate}}\right)\\
\text{sous les contraintes}\quad
&\mathbf x_{k+1}-\mathbf x_k
-\frac{h_k}{6}(\mathbf f_k+4\mathbf f_{k+\frac12}+\mathbf f_{k+1})=\mathbf0,\\
&\mathbf x_{k+\frac12}-\frac{\mathbf x_k+\mathbf x_{k+1}}{2}
-\frac{h_k}{8}(\mathbf f_k-\mathbf f_{k+1})=\mathbf0,
\quad k=0,\ldots,N-1,\\
&\mathbf g(\mathbf x_q,\mathbf u_q,t_q)\leq\mathbf0,
\quad q\in\{0,\tfrac12,1,\tfrac32,\ldots,N\},\\
&\mathbf h(\mathbf x_0,\mathbf x_N,t_N)=\mathbf0.
\end{aligned}
$$

Une autre possibilité consiste à optimiser la commande au milieu de façon
indépendante. Les trois valeurs de commande définissent alors un polynôme
quadratique sur chaque intervalle; la relation de moyenne au milieu est
supprimée, et $\mathbf u_{k+\frac12}$ s'ajoute au vecteur de décision. Les
deux équations d'état et les poids de coût de Simpson restent les mêmes. Ce
choix modifie les commandes admissibles; il peut donc modifier la solution de
l'optimiseur.

Les représentations obtenues peuvent être comparées directement. Tous les
degrés sont des bornes supérieures sur chaque intervalle; ils ne disent rien
de l'ordre de convergence.

| Schéma utilisé ici | Degré de l'état | Degré de la commande | Valeurs indépendantes au-delà des états du maillage |
|---|---:|---:|---|
| Euler explicite ou implicite | 1 | 0 | Une commande par intervalle |
| Trapèzes | 2 | 1 | Commandes aux extrémités partagées |
| Hermite--Simpson, commande linéaire | 3 | 1 | Commandes aux extrémités partagées et états au milieu |
| Hermite--Simpson, commande quadratique | 3 | 2 | Commandes aux extrémités partagées, états au milieu et commandes au milieu |

## Construire un schéma à partir de ses nœuds

Comment un programme peut-il générer ces transcriptions à partir de choix de
nœuds, sans dériver à la main un nouvel ensemble d'équations de défaut à
chaque fois?

La même construction génère chacun des schémas du tableau précédent. Ses
entrées sont les nœuds de collocation, les nœuds de support de la commande et
un choix quant à la continuité de la commande. Le nombre de nœuds de
collocation détermine le degré de l'interpolant des pentes, et donc celui du
polynôme d'état. Les nœuds de support de la commande déterminent le polynôme
de commande de façon indépendante.

Cette recette couvre la collocation polynomiale obtenue en interpolant des
pentes en des nœuds distincts et en les intégrant. Elle inclut la collocation
de Gauss, de Radau et de Lobatto, ainsi que les exemples d'ordre bas. Un
tableau de Runge--Kutta quelconque ne provient pas forcément de tels nœuds;
préciser des nœuds ne génère donc pas toutes les méthodes d'intégration
possibles.

La première recette utilise la forme par intégration. Son équivalent sous
forme par dérivation, à @sec-collocation-differentiation-from-nodes, prend
aussi en entrée des nœuds de support de l'état, car il stocke des valeurs du
polynôme d'état plutôt que de construire ce polynôme à partir de ses pentes.

````{prf:algorithm} Assembler un PNL de collocation sous forme par intégration
:label: alg-collocation-from-nodes

**Entrée :** un maillage $t_0<\cdots<t_N$; la dynamique $\mathbf f$, le coût
courant $c$, le coût terminal $c_f$, les égalités aux limites $\mathbf h$ et
les inégalités de chemin $\mathbf g$; des nœuds de collocation distincts
$\tau_1,\ldots,\tau_s$ et des nœuds de support de la commande
$\rho_0,\ldots,\rho_{d_u}$ dans $[0,1]$. Préciser si les commandes doivent
être continues d'un intervalle à l'autre.

**Sortie :** un vecteur de décision fini $\mathbf z$, un objectif
$F(\mathbf z)$, des inégalités $G(\mathbf z)\leq\mathbf0$ et des égalités
$H(\mathbf z)=\mathbf0$.

1. **Calculer la règle une seule fois.** Construire les polynômes cardinaux
   $\ell_j$ pour les nœuds de collocation et $\psi_r$ pour les nœuds de
   support de la commande. Les intégrer et les évaluer pour obtenir

   $$
   A_{ij}=\int_0^{\tau_i}\ell_j(\tau)\,d\tau,\qquad
   b_j=\int_0^1\ell_j(\tau)\,d\tau,\qquad
   B_{jr}=\psi_r(\tau_j).
   $$

2. **Allouer les variables.** Créer les états du maillage $\mathbf x_k$, les
   états d'étage $\mathbf x_{k,j}$ et les valeurs de support de la commande
   $\widehat{\mathbf u}_{k,r}$. Initialiser $F=c_f(\mathbf x_N,t_N)$, ajouter
   le résidu aux limites $\mathbf h(\mathbf x_0,\mathbf x_N,t_N)$ à $H$ et
   initialiser $G$ vide.

3. **Assembler chaque intervalle.** Pour $k=0,\ldots,N-1$, poser
   $h_k=t_{k+1}-t_k$ et calculer les expressions

   $$
   \begin{aligned}
   \mathbf u_{k,j}&=\sum_rB_{jr}\widehat{\mathbf u}_{k,r},\\
   \mathbf f_{k,j}&=\mathbf f(\mathbf x_{k,j},\mathbf u_{k,j},t_k+h_k\tau_j),\\
   c_{k,j}&=c(\mathbf x_{k,j},\mathbf u_{k,j},t_k+h_k\tau_j).
   \end{aligned}
   $$

   Ajouter $h_k\sum_jb_jc_{k,j}$ à $F$. Ajouter à $H$ les résidus d'étage et
   d'extrémité

   $$
   \mathbf x_{k,i}-\mathbf x_k-h_k\sum_jA_{ij}\mathbf f_{k,j},
   \qquad
   \mathbf x_{k+1}-\mathbf x_k-h_k\sum_jb_j\mathbf f_{k,j}.
   $$

   Ajouter $\mathbf g(\mathbf x_{k,j},\mathbf u_{k,j},t_k+h_k\tau_j)$ à $G$ à
   chaque étage. Ajouter les éventuelles contraintes échantillonnées
   supplémentaires ou bornes sur les variables exigées par le modèle.

4. **Raccorder les commandes au besoin.** Soit partager les variables de
   commande aux extrémités, soit ajouter l'égalité entre la valeur de commande
   de droite du morceau $k$ et la valeur de commande de gauche du morceau
   $k+1$. Les états du maillage sont déjà partagés.

5. **Résoudre et reconstruire.** Transmettre $F,G,H$ et leurs dérivées à un
   solveur de PNL. Retrouver chaque morceau d'état en intégrant son
   interpolant des pentes à partir de $\mathbf x_k$, et chaque morceau de
   commande à partir de ses valeurs de support. Vérifier le résidu de l'EDO
   et les contraintes entre les nœuds; raffiner le maillage ou le degré là où
   c'est nécessaire et résoudre de nouveau.
````

Les commandes, les pentes et les taux de coût d'étage de l'étape 3 sont des
expressions des variables de décision. Ils sont recalculés à mesure que
l'optimiseur modifie ces variables; $A,b,B$ restent fixes. La différentiation
automatique peut fournir les dérivées des expressions assemblées. Si les
étages aux extrémités sont conservés comme variables distinctes, les
équations d'étage et d'extrémité les relient aux états du maillage. Éliminer
ces copies produit les programmes d'ordre bas plus petits vus plus haut;
conserver à la fois les copies et leurs équations de définition est aussi
valide. Identifier une copie à son extrémité exige de supprimer l'équation en
double ou identiquement nulle qui en résulte.

La démonstration
[Collocation directe pour le redressement d'un pendule](lab/notebooks/08_collocation_from_nodes.ipynb#step-1-compute-the-rule-once)
exécute ces cinq étapes pour un problème non linéaire avec les nœuds de Radau
à deux étages. Elle calcule $A$ et $b$ à la main, les reproduit dans le code,
assemble $F$ et $H$, résout le PNL obtenu avec SciPy et vérifie la réponse par
simulation. Chaque étape est accompagnée des lignes de Python
correspondantes.

(sec-collocation-coefficient-generator)=
### Un générateur de coefficients et un évaluateur de résidus réutilisables

Le [code source Python complet sur GitHub](https://github.com/pierrelux/rlbook/blob/main/code/collocation_transcription.py)
contient le générateur de coefficients et l'évaluateur de résidus du PNL. Son
étape d'intégration utilise le calcul polynomial sous forme close plutôt
qu'une routine de quadrature numérique qui échantillonne une fonction pour en
estimer l'intégrale. Pour voir ce que calcule le code, revenons à la
troisième fonction de base des pentes pour les nœuds $(0,\tfrac12,1)$ de
@fig-collocation-destination-and-slopes :

$$
\ell_3(\tau)=2\tau^2-\tau.
$$

Intégrer un terme $a\tau^m$ augmente son exposant de un et divise son
coefficient par ce nouvel exposant. En appliquant cette règle aux deux
termes, on obtient une primitive

$$
L_3(\tau)=\frac23\tau^3-\frac12\tau^2.
$$

La dérivée de $L_3$ redonne $\ell_3$. On a choisi une constante
d'intégration nulle; toute autre constante s'annulerait lors de l'évaluation
d'une intégrale définie. Le théorème fondamental de l'analyse fournit
maintenant à la fois le poids sur l'intervalle complet et le poids au
milieu :

$$
\begin{aligned}
b_3
&=L_3(1)-L_3(0)=\frac23-\frac12=\frac16,\\
A_{23}
&=L_3(\tfrac12)-L_3(0)=\frac1{12}-\frac18=-\frac1{24}.
\end{aligned}
$$

Seule la borne supérieure change d'un calcul à l'autre. Le poids négatif au
milieu concorde avec l'aire signée sous $\ell_3$ dans la figure. Le calcul de
l'un ou l'autre poids n'exige donc qu'une primitive et deux évaluations de
polynôme, sans approcher l'aire par de petits pas de temps.

L'implémentation représente un polynôme par ses coefficients classés par
puissances croissantes. Pour $\ell_3$, le tableau est $[0,-1,2]$ : les
composantes multiplient $1,\tau,\tau^2$. L'intégration et la dérivation
transforment ce tableau en tableaux de coefficients de nouveaux polynômes :

$$
\begin{aligned}
[0,-1,2]&\xrightarrow{\text{intégrer}}
[0,0,-\tfrac12,\tfrac23],\\
[0,-1,2]&\xrightarrow{\text{dériver}}[-1,4].
\end{aligned}
$$

Le premier résultat représente $L_3$; le second représente
$\ell_3'(\tau)=4\tau-1$. Plus généralement, l'intégration d'un tableau de
coefficients $[a_0,a_1,\ldots,a_d]$ produit
$[C,a_0,a_1/2,\ldots,a_d/(d+1)]$, où $C$ est la constante d'intégration. Sa
dérivation produit $[a_1,2a_2,\ldots,d a_d]$. Ces opérations n'exigent que de
l'arithmétique sur les coefficients stockés, et non un système de calcul
formel. La méthode
d'[intégration de polynômes](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.integ.html)
de NumPy implémente l'opération de primitive :

```python
from numpy.polynomial import Polynomial

ell3 = Polynomial([0, -1, 2])  # 2*tau**2 - tau
L3 = ell3.integ()             # (2/3)*tau**3 - (1/2)*tau**2
dell3 = ell3.deriv()          # 4*tau - 1

b3 = L3(1.0) - L3(0.0)       # 1/6, up to floating-point error
A23 = L3(0.5) - L3(0.0)      # -1/24, up to floating-point error
```

Le générateur ci-dessous applique cette procédure à chaque fonction de base
des pentes. Il construit d'abord chaque polynôme cardinal en multipliant ses
facteurs linéaires. L'appel à `.integ()` construit ensuite sa primitive;
l'évaluation de ce polynôme aux nœuds d'étage, moins sa valeur en zéro,
fournit une colonne de $A$. L'évaluation en $1$ fournit la composante
correspondante de $b$. L'évaluation directe des fonctions de base de la
commande fournit $B$. La même règle de dérivation des coefficients, suivie
d'une évaluation aux nœuds de collocation, construit $D$ à partir de la base
de l'état dans la forme par dérivation. Elle n'estime pas une dérivée au moyen
d'un petit pas de différences finies.

Ces calculs ont lieu une seule fois pour les ensembles de nœuds choisis. Lors
d'une évaluation du PNL, les états et les commandes courants fournissent de
nouvelles pentes $\mathbf f_{k,j}$, mais les coefficients restent fixes. À ce
stade, « intégrer les pentes interpolées » signifie calculer les sommes
pondérées $h_k\sum_j A_{ij}\mathbf f_{k,j}$ ou
$h_k\sum_j b_j\mathbf f_{k,j}$. L'intégrale du polynôme est exacte en
arithmétique exacte; le calcul en virgule flottante introduit des erreurs
d'arrondi, et le remplacement de l'EDO non linéaire par un interpolant des
pentes introduit tout de même une erreur de discrétisation distincte.
L'intégration sous forme close de l'interpolant ne donne pas une solution sous
forme close de l'EDO d'origine.

La fonction suivante calcule $A,b,B$ pour tous nœuds de collocation et de
support de la commande distincts :

```{literalinclude} code/collocation_transcription.py
:language: python
:start-at: def make_rule
:end-before: def transcribed_problem
```

Par exemple, les choix ci-dessous redonnent les quatre constructions d'ordre
bas :

```python
from collocation_transcription import make_rule

explicit_euler = make_rule([0.0], [0.0])
implicit_euler = make_rule([1.0], [0.0])
trapezoidal = make_rule([0.0, 1.0], [0.0, 1.0])
hermite_simpson = make_rule([0.0, 0.5, 1.0], [0.0, 1.0])
# An independent midpoint control changes B, not A or b:
hermite_simpson_quadratic_u = make_rule([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
```

Le nœud de support de la commande d'Euler implicite peut valoir $0$ même si
son nœud de collocation vaut $1$ : un polynôme constant a la même valeur aux
deux points. Pour Hermite--Simpson avec des commandes linéaires, les tableaux
générés sont

$$
A=\begin{bmatrix}
0&0&0\\
5/24&1/3&-1/24\\
1/6&2/3&1/6
\end{bmatrix},\qquad
b=\begin{bmatrix}1/6\\2/3\\1/6\end{bmatrix},\qquad
B=\begin{bmatrix}1&0\\1/2&1/2\\0&1\end{bmatrix}.
$$

La ligne du milieu de $A$ donne l'équation d'étage au milieu avant
élimination; la ligne du milieu de $B$ donne la commande égale à la moyenne
aux extrémités. Fournir plutôt trois nœuds de support de la commande fait de
$B$ l'identité, ce qui permet une commande au milieu indépendante tout en
laissant la construction de l'état inchangée. Les
[expériences de la démonstration du pendule](lab/notebooks/08_collocation_from_nodes.ipynb#experiments)
modifient les nœuds de collocation de la même façon et comparent la précision
des schémas obtenus sur un problème non linéaire.

L'[évaluateur de résidus](https://github.com/pierrelux/rlbook/blob/main/code/collocation_transcription.py#L68)
du même fichier, `transcribed_problem`, évalue $F,G,H$ à partir de tableaux
d'états du maillage, d'états d'étage et de valeurs de support de la commande.
Par défaut, il admet des sauts de commande; `continuous_control=True` ajoute
les équations de concordance aux extrémités utilisées par les exemples à
commande linéaire. Une enveloppe du solveur aplatit ces tableaux en
$\mathbf z$ et redonne sa forme à chaque candidat avant l'évaluation.
L'évaluateur NumPy peut être utilisé avec des différences finies; une
implémentation par différentiation automatique utilise les mêmes opérations
sur les tableaux dans le moteur de calcul choisi. Les résidus d'inégalité
renvoyés suivent la convention $G\leq0$; il faut en changer le signe pour une
interface de solveur qui attend des résidus non négatifs.

{download}`Télécharger le générateur de coefficients et l'évaluateur du PNL <code/collocation_transcription.py>`

Des coefficients polynomiaux explicites gardent cette implémentation lisible
pour des degrés modestes. Pour des degrés élevés, on utilise des routines
d'évaluation dans une base et d'intégration numériquement stables;
l'assemblage du PNL reste le même.

(sec-collocation-differentiation-from-nodes)=
### Construire la forme par dérivation à partir de ses nœuds

Les deux recettes retiennent des nœuds de collocation où l'état reconstruit
doit satisfaire l'EDO. La recette sous forme par intégration interpole les
pentes d'étage évaluées et les intègre; ses résidus font ensuite concorder
les valeurs d'état obtenues avec les états d'étage stockés. La forme par
dérivation interpole des valeurs de support de l'état stockées et évalue les
états d'étage sur cette courbe; ses résidus font ensuite concorder les
dérivées de la courbe avec les pentes de l'EDO. Choisir une forme modifie donc
la représentation et les résidus, et non la présence de nœuds de collocation
dans le schéma.

Pour un exemple concret, on retient les deux nœuds de collocation de la règle
des trapèzes, $\tau_1=0$ et $\tau_2=1$. Son interpolant des pentes est
linéaire; son polynôme d'état est donc quadratique. Ne stocker que les deux
états aux extrémités et les interpoler linéairement donnerait une dérivée
constante. Imposer cette dérivée aux deux extrémités exigerait que les deux
pentes de l'EDO soient égales, ce que la règle des trapèzes n'exige pas. Pour
représenter le même polynôme quadratique que la forme par intégration, on
choisit trois nœuds de support de l'état, par exemple
$\sigma=(0,\tfrac12,1)$.

Les polynômes cardinaux associés à ces nœuds de support sont
$2\tau^2-3\tau+1$, $4\tau-4\tau^2$ et $2\tau^2-\tau$. En les évaluant, ainsi
que leurs dérivées, aux deux nœuds de collocation, on obtient

$$
E=\begin{bmatrix}1&0&0\\0&0&1\end{bmatrix},
\qquad
D=\begin{bmatrix}-3&4&-1\\1&-4&3\end{bmatrix}.
$$

Il y a trois colonnes parce que chaque opération polynomiale utilise trois
valeurs de support, mais seulement deux lignes parce que l'EDO est imposée en
deux nœuds de collocation. On identifie les valeurs de support aux
extrémités à $\mathbf x_k$ et $\mathbf x_{k+1}$, et on appelle
$\widehat{\mathbf x}_{k,1}$ la valeur de support au milieu. Les deux
contraintes de dérivation sont

$$
\begin{aligned}
-3\mathbf x_k+4\widehat{\mathbf x}_{k,1}-\mathbf x_{k+1}
&=h_k\mathbf f_{k,1},\\
\mathbf x_k-4\widehat{\mathbf x}_{k,1}+3\mathbf x_{k+1}
&=h_k\mathbf f_{k,2}.
\end{aligned}
$$

Les pentes du membre de droite sont évaluées respectivement à l'extrémité
gauche et à l'extrémité droite. L'addition des équations élimine la valeur de
support au milieu. En divisant par deux, on retrouve la contrainte
d'extrémité des trapèzes :

$$
\mathbf x_{k+1}-\mathbf x_k
=\frac{h_k}{2}(\mathbf f_{k,1}+\mathbf f_{k,2}).
$$

En substituant cette relation dans l'une ou l'autre contrainte de dérivation,
on détermine aussi la valeur de support au milieu :

$$
\widehat{\mathbf x}_{k,1}
=\frac{\mathbf x_k+\mathbf x_{k+1}}{2}
+\frac{h_k}{8}(\mathbf f_{k,1}-\mathbf f_{k,2}).
$$

La valeur au milieu enregistre la courbure du polynôme quadratique. La
stocker n'ajoute pas de contrainte d'EDO au milieu. Un nœud de support précise
où une valeur du polynôme est stockée; un nœud de collocation précise où sa
dérivée doit concorder avec la dynamique.

Le même décompte des degrés s'applique à un nombre quelconque d'étages. Avec
$s$ nœuds de collocation distincts, la forme par intégration construit un
polynôme d'état de degré au plus $s$. Son équivalent sous forme par dérivation
utilise donc $s+1$ nœuds de support de l'état distincts
$\sigma_0,\ldots,\sigma_s$. Ils n'ont pas à coïncider avec les nœuds de
collocation ni à inclure les extrémités de l'intervalle. Changer ces
emplacements de support change les coordonnées servant à décrire le polynôme,
et non l'espace de polynômes. Les nœuds de support de la commande précisent
toujours un polynôme distinct et peuvent être choisis indépendamment.

````{prf:algorithm} Assembler un PNL de collocation sous forme par dérivation
:label: alg-collocation-differentiation-from-nodes

**Entrée :** un maillage $t_0<\cdots<t_N$; la dynamique $\mathbf f$, les
coûts $c,c_f$, les égalités aux limites $\mathbf h$ et les inégalités de
chemin $\mathbf g$; des nœuds de collocation $\tau_1,\ldots,\tau_s$, des
nœuds de support de l'état $\sigma_0,\ldots,\sigma_s$ et des nœuds de support
de la commande $\rho_0,\ldots,\rho_{d_u}$ dans $[0,1]$, distincts à
l'intérieur de chaque ensemble. Préciser si les commandes doivent être
continues d'un intervalle à l'autre.

**Sortie :** un vecteur de décision $\mathbf z$ et des fonctions
$F(\mathbf z)$, $G(\mathbf z)\leq\mathbf0$ et $H(\mathbf z)=\mathbf0$.

1. **Calculer les opérateurs une seule fois.** Construire les polynômes
   cardinaux $\ell_r^{\mathrm{state}}$ pour les nœuds de support de l'état,
   $\ell_i$ pour les nœuds de collocation et $\psi_q$ pour les nœuds de
   support de la commande. Former

   $$
   \begin{aligned}
   E_{ir}&=\ell_r^{\mathrm{state}}(\tau_i),&
   D_{ir}&=(\ell_r^{\mathrm{state}})'(\tau_i),\\
   e^L_r&=\ell_r^{\mathrm{state}}(0),&
   e^R_r&=\ell_r^{\mathrm{state}}(1),\\
   B_{iq}&=\psi_q(\tau_i),&
   b_i&=\int_0^1\ell_i(\tau)\,d\tau.
   \end{aligned}
   $$

   Ici, $i=1,\ldots,s$, $r=0,\ldots,s$ et $q=0,\ldots,d_u$. Les vecteurs
   $e^L,e^R$ évaluent le polynôme d'état à ses extrémités gauche et droite.
   $E$ et $D$ sont tous deux de forme $s\times(s+1)$.

2. **Allouer les variables.** Créer les états du maillage partagés
   $\mathbf x_k$, les valeurs de support de l'état
   $\widehat{\mathbf x}_{k,r}$ et les valeurs de support de la commande
   $\widehat{\mathbf u}_{k,q}$. Ne pas allouer d'états d'étage distincts : ils
   seront évalués à partir des valeurs de support. Initialiser
   $F=c_f(\mathbf x_N,t_N)$, ajouter $\mathbf h(\mathbf x_0,\mathbf x_N,t_N)$
   à $H$ et initialiser $G$ vide.

3. **Assembler chaque intervalle.** Pour chaque $k=0,\ldots,N-1$, poser
   $h_k=t_{k+1}-t_k$. À chaque étage $i=1,\ldots,s$, calculer

   $$
   \begin{aligned}
   \mathbf x_{k,i}&=\sum_{r=0}^{s}E_{ir}\widehat{\mathbf x}_{k,r},\\
   \mathbf u_{k,i}&=\sum_{q=0}^{d_u}B_{iq}\widehat{\mathbf u}_{k,q},\\
   \mathbf f_{k,i}&=\mathbf f(\mathbf x_{k,i},\mathbf u_{k,i},t_k+h_k\tau_i),\\
   c_{k,i}&=c(\mathbf x_{k,i},\mathbf u_{k,i},t_k+h_k\tau_i).
   \end{aligned}
   $$

   Ajouter les résidus de pente à $H$ :

   $$
   \sum_{r=0}^{s}D_{ir}\widehat{\mathbf x}_{k,r}
   -h_k\mathbf f_{k,i},\qquad i=1,\ldots,s.
   $$

   Ajouter les deux résidus d'extrémité pour relier le polynôme au maillage :

   $$
   \sum_{r=0}^{s}e^L_r\widehat{\mathbf x}_{k,r}-\mathbf x_k,
   \qquad
   \sum_{r=0}^{s}e^R_r\widehat{\mathbf x}_{k,r}-\mathbf x_{k+1}.
   $$

   Ajouter $h_k\sum_{i=1}^{s}b_i c_{k,i}$ à $F$. Ajouter
   $\mathbf g(\mathbf x_{k,i},\mathbf u_{k,i},t_k+h_k\tau_i)$ à $G$ à chaque
   étage, ainsi que les éventuelles contraintes échantillonnées
   supplémentaires ou bornes sur les variables exigées par le modèle.

4. **Raccorder les commandes au besoin.** Partager les variables de commande
   aux extrémités, ou égaler l'évaluation à l'extrémité droite de chaque
   polynôme de commande à l'évaluation à l'extrémité gauche du suivant. Les
   états du maillage partagés et les équations d'extrémité de l'étape 3
   raccordent déjà les polynômes d'état d'un intervalle à l'autre.

5. **Résoudre et reconstruire.** Transmettre $F,G,H$ et leurs dérivées à un
   solveur de PNL. Reconstruire directement l'état par
   $\mathbf p_k(\tau)=\sum_{r=0}^{s}\widehat{\mathbf x}_{k,r}
   \ell_r^{\mathrm{state}}(\tau)$, et reconstruire la commande à partir de ses
   valeurs de support. Vérifier les résidus entre les nœuds et effectuer un
   rejeu continu distinct comme à @sec-collocation-validation; raffiner et
   résoudre de nouveau au besoin.
````

Le facteur $h_k$ du résidu de pente convertit le taux en temps physique de
l'EDO en une dérivée en temps local, comme dans
@fig-collocation-evaluation-differentiation. Les tableaux sont calculés pour
$\tau\in[0,1]$; les mêmes tableaux conviennent donc à des intervalles de
longueurs différentes. Le raffinement du maillage modifie les longueurs des
intervalles et les variables, mais n'exige pas de nouveaux tableaux à moins
que les ensembles de nœuds locaux ne changent. La dernière section de la
démonstration du pendule,
[La même construction avec des dérivées](lab/notebooks/08_collocation_from_nodes.ipynb#the-same-construction-with-derivatives),
exécute cet algorithme avec les nœuds de Radau à deux étages et montre qu'il
renvoie la même trajectoire que la forme par intégration.

Cet algorithme garde les états du maillage et les états de support comme
variables distinctes et les relie par des équations. Si un nœud de support est
une extrémité, sa valeur peut plutôt être identifiée à l'état du maillage
correspondant, comme dans l'exemple quadratique. On omet alors son égalité
d'extrémité : après identification, cette égalité dirait seulement
$\mathbf x_k-\mathbf x_k=\mathbf0$. Si aucune extrémité n'est un nœud de
support, on conserve les deux évaluations aux extrémités.

Dériver l'état ne supprime pas l'intégration de l'objectif. Le coût courant
est toujours échantillonné aux $s$ nœuds de collocation et intégré avec les
mêmes poids $b$ que dans la forme par intégration. Ces poids intègrent la base
des nœuds de collocation $\ell_i$, et non la base des nœuds de support de
l'état $\ell_r^{\mathrm{state}}$. Pour l'exemple quadratique, la contribution
au coût est l'expression des trapèzes $h_k(c_{k,1}+c_{k,2})/2$, même si trois
valeurs représentent l'état.

#### Équivalence avec la construction par intégration

En un point admissible, les contraintes de dérivation précisent les valeurs de
$\mathbf p_k'$ aux $s$ nœuds de collocation. Comme cette dérivée est de degré
au plus $s-1$, ses valeurs en ces nœuds la déterminent partout :

$$
\mathbf p_k'(\tau)=h_k\sum_{j=1}^{s}\mathbf f_{k,j}\ell_j(\tau),
\qquad \forall\tau\in[0,1].
$$

L'équation d'extrémité gauche fournit $\mathbf p_k(0)=\mathbf x_k$.
L'application du théorème fondamental de l'analyse donne donc

$$
\mathbf p_k(\tau)=\mathbf x_k
+h_k\sum_{j=1}^{s}\mathbf f_{k,j}\int_0^\tau\ell_j(\xi)\,d\xi,
\qquad \forall\tau\in[0,1].
$$

L'évaluation en $\tau_i$ redonne l'équation d'étage de la forme par
intégration; l'évaluation en $1$, avec l'équation d'extrémité droite, redonne
son défaut d'extrémité. Aucun défaut intégral supplémentaire n'est nécessaire
dans le PNL sous forme par dérivation. Réciproquement, le polynôme d'état
construit par la forme par intégration a ces dérivées nodales et peut être
représenté par ses valeurs en $s+1$ nœuds de support de l'état distincts
quelconques.

Ainsi, les deux formulations décrivent les mêmes trajectoires polynomiales
admissibles lorsqu'elles utilisent les mêmes nœuds de collocation, le même
degré et le même espace de commandes. Utiliser les mêmes poids de coût et les
mêmes points d'échantillonnage des contraintes donne aussi le même objectif
discret et les mêmes contraintes de chemin. Leurs tableaux de décision et
leurs résidus diffèrent; leur conditionnement numérique n'est donc pas
forcément identique.

#### Un générateur de coefficients et un évaluateur de résidus sous forme par dérivation

La construction de $D$ utilise la dérivation des coefficients présentée à
@sec-collocation-coefficient-generator. Par exemple, la première fonction de
base de l'état de l'exemple quadratique est
$\ell_0^{\mathrm{state}}(\tau)=1-3\tau+2\tau^2$. La dérivation transforme son
tableau de coefficients $[1,-3,2]$ en $[-3,4]$. L'évaluation du polynôme
obtenu $-3+4\tau$ en $(0,1)$ donne $(-3,1)$, la première colonne de $D$. La
[méthode de dérivation de polynômes](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.deriv.html)
de NumPy effectue cette transformation des coefficients. Aucun petit
incrément de temps ni aucune estimation par différences finies n'intervient.

Le générateur suivant utilise la même fonction auxiliaire
`cardinal_polynomials` que `make_rule`. Il évalue chaque fonction de base de
l'état pour former une colonne de $E$, et évalue sa dérivée pour former la
colonne correspondante de $D$. Il n'intègre la base des nœuds de collocation
que pour obtenir les poids du coût :

```{literalinclude} code/collocation_transcription.py
:language: python
:start-at: def make_differentiation_rule
:end-before: def differentiation_problem
```

Les quatre schémas d'ordre bas peuvent maintenant être générés sous forme par
dérivation :

```python
from collocation_transcription import make_differentiation_rule

explicit_euler = make_differentiation_rule([0], [0, 1], [0])
implicit_euler = make_differentiation_rule([1], [0, 1], [0])
trapezoidal = make_differentiation_rule([0, 1], [0, 0.5, 1], [0, 1])
hermite_simpson = make_differentiation_rule(
    [0, 0.5, 1], [0, 1/3, 2/3, 1], [0, 1]
)
```

L'ordre des arguments est : nœuds de collocation, nœuds de support de l'état
et nœuds de support de la commande. Dans l'appel pour Hermite--Simpson, quatre
valeurs de support de l'état représentent un polynôme cubique. L'état d'étage
au milieu est évalué par $E$ à partir de ces valeurs, puisque $1/2$ n'est pas
un nœud de support de l'état dans ce choix. N'utiliser que les trois nœuds de
collocation comme supports de l'état restreindrait l'état à un polynôme
quadratique et ne reproduirait pas la même méthode.

Le théorème fondamental de l'analyse fournit aussi des vérifications sur les
tableaux fixes, avant qu'intervienne une fonction de dynamique ou un solveur
de PNL. Soustraire l'évaluation à l'extrémité gauche de chaque ligne de $E$
doit donner le même opérateur que l'intégration de la dérivée avec $A$. De
même, la différence des évaluations aux extrémités doit être égale à
l'intégration sur l'intervalle complet avec $b$ :

```python
import numpy as np
from collocation_transcription import make_rule

rule = hermite_simpson
integral = make_rule(rule.nodes, rule.control_nodes)
np.testing.assert_allclose(
    rule.E - rule.state_left, integral.A @ rule.D, atol=1e-12
)
np.testing.assert_allclose(
    rule.state_right - rule.state_left, rule.b @ rule.D, atol=1e-12
)
```

Dans la première vérification, NumPy soustrait `state_left` de chaque ligne de
`E`. Ces identités sont vraies pour tout polynôme d'état de degré $s$, et non
seulement pour un polynôme qui satisfait l'EDO. Les tolérances tiennent compte
des erreurs d'arrondi dans la construction des coefficients.

L'évaluateur de résidus `differentiation_problem` prend des états du maillage
de forme $(N+1,n)$, des valeurs de support de l'état de forme $(N,s+1,n)$ et
des valeurs de support de la commande de forme $(N,d_u+1,m)$. Pour chaque
vecteur de décision candidat, `E @ x_support[k]` fournit les états d'étage et
`D @ x_support[k] - width * slopes` fournit les résidus de pente. Les deux
évaluations aux extrémités relient le polynôme aux états du maillage. Les
conventions pour le coût, les conditions aux limites, les contraintes de
chemin et la continuité facultative de la commande sont les mêmes que pour
`transcribed_problem`.

:::{dropdown} Examiner le module complet et les deux évaluateurs de résidus

```{literalinclude} code/collocation_transcription.py
:language: python
```

:::

La méthode `.deriv()` du générateur dérive un polynôme de base connu par
rapport au temps local $\tau$. Un solveur de PNL a aussi besoin de dérivées,
mais par rapport au vecteur de décision $\mathbf z$ : celles-ci dérivent
l'objectif et les contraintes assemblés, y compris la fonction de dynamique.
La différentiation automatique peut calculer ces dernières dans une
implémentation compatible; elle ne remplace ni ne modifie la matrice fixe
$D$.

(sec-collocation-validation)=
### Vérifier la trajectoire continue

Un faible résidu du PNL confirme que les valeurs stockées satisfont les
contraintes finies. Pour vérifier l'approximation entre les nœuds, on évalue
d'abord le polynôme d'état et sa dérivée à des instants supplémentaires. Sur
l'intervalle $k$, leur résidu de l'EDO est

$$
\mathbf r_k(\tau)=
\frac{1}{h_k}\mathbf p_k'(\tau)
-\mathbf f\!\left(
\mathbf p_k(\tau),\mathbf u_h(t_k+h_k\tau),t_k+h_k\tau
\right).
$$

Ce résidu compare la pente de la courbe reconstruite à la pente prescrite par
le modèle aux mêmes état, commande et instant. Les équations de collocation
l'annulent aux étages, à la tolérance du solveur près. Une valeur élevée
ailleurs signale un intervalle dont le polynôme ne résout pas la dynamique. On
évalue aussi les contraintes de chemin sur cette courbe. Cet usage de
l'interpolant propre à la méthode pour évaluer l'erreur est décrit par
{cite:t}`Kelly2017DirectCollocation`.

Une vérification distincte fixe la commande reconstruite $\mathbf u_h(t)$ et
intègre l'EDO d'origine à partir de l'état initial prescrit au moyen d'un
intégrateur temporel précis. On compare cet état simulé à $\mathbf x_h(t)$, et
on vérifie ses conditions terminales et ses contraintes de chemin. La
simulation mesure ce que la commande choisie produit dans le modèle;
$\mathbf r_k$ mesure à quel point le polynôme lui-même satisfait ce modèle.
L'[étape 5 de la démonstration du pendule](lab/notebooks/08_collocation_from_nodes.ipynb#step-5-solve-and-check)
effectue ce rejeu et distingue le résidu du solveur de l'erreur de
discrétisation.

Si les deux trajectoires divergent ou si le résidu du polynôme est élevé, on
subdivise les intervalles concernés ou on augmente leur degré polynomial, puis
on résout de nouveau. On compare l'objectif et les trajectoires d'un
raffinement à l'autre. Une vérification dense fournit des indices sur la
résolution, mais ne prouve pas que chaque point respecte une borne. Resserrer
à elle seule une tolérance du PNL déjà faible ne peut pas éliminer l'erreur
introduite par une transcription grossière.

(optional-orientation-gauss-radau-and-lobatto-nodes)=
### Collocation de Gauss--Legendre et de Radau

Le générateur de coefficients accepte n'importe quels nœuds distincts, mais
leur emplacement influe sur la précision de la règle d'intégration obtenue.
Pour voir ce que permet le choix des nœuds, on conserve deux valeurs de pente
par intervalle, comme dans la construction des trapèzes, et on laisse leurs
emplacements varier. L'interpolant des pentes reste linéaire et son intégrale
reste un morceau d'état quadratique. Le choix des nœuds modifie l'endroit où
l'EDO est imposée et les poids servant à combiner ses pentes.

Pour la **collocation de Gauss--Legendre à deux étages**, on choisit les nœuds
intérieurs

$$
\tau_1=\frac12-\frac{\sqrt3}{6},\qquad
\tau_2=\frac12+\frac{\sqrt3}{6}.
$$

Ils valent environ $0.2113$ et $0.7887$. En construisant les deux fonctions de
base de Lagrange en ces nœuds et en les intégrant avec `make_rule`, on obtient

$$
A=\begin{bmatrix}
\tfrac14&\tfrac14-\tfrac{\sqrt3}{6}\\
\tfrac14+\tfrac{\sqrt3}{6}&\tfrac14
\end{bmatrix},\qquad
b=\begin{bmatrix}\tfrac12\\\tfrac12\end{bmatrix}.
$$

Les lignes de $A$ déterminent les deux états d'étage intérieurs. Les poids $b$
donnent ensuite la contrainte d'extrémité

$$
\mathbf x_{k+1}-\mathbf x_k
-\frac{h_k}{2}(\mathbf f_{k,1}+\mathbf f_{k,2})=0.
$$

Les poids ressemblent à ceux de la règle des trapèzes, mais les pentes sont
évaluées aux deux étages intérieurs. Ce déplacement améliore la précision de
la quadrature : pour un intégrande scalaire $q(\tau)=\tau^2$, la moyenne des
valeurs aux extrémités donne $1/2$, alors que ces nœuds donnent

$$
\frac{\tau_1^2+\tau_2^2}{2}=\frac13
=\int_0^1\tau^2\,d\tau.
$$

La même règle de Gauss à deux nœuds intègre exactement tout polynôme de degré
au plus trois. Cela ne rend pas le morceau d'état cubique : son degré reste au
plus deux. L'exactitude de la quadrature décrit les intégrandes qu'une somme
pondérée intègre exactement; le degré de l'état décrit la courbe représentée
sur un intervalle.

Le nom provient des **polynômes de Legendre** $P_s$ sur $[-1,1]$, qui sont
orthogonaux à tous les polynômes de degré inférieur pour l'intégration sur cet
intervalle et sont normalisés par $P_s(1)=1$. Par exemple, $P_0(\xi)=1$,
$P_1(\xi)=\xi$ et $P_2(\xi)=(3\xi^2-1)/2$. Les racines de $P_2$ sont
$\pm1/\sqrt3$; en les ramenant à $[0,1]$ par $\tau=(1+\xi)/2$, on obtient les
deux nœuds ci-dessus. Avec $s$ étages, la même construction utilise les $s$
racines de $P_s$ et donne une règle de quadrature exacte jusqu'au degré
$2s-1$. La [quadrature de Gauss--Legendre](https://dlmf.nist.gov/3.5#v)
fournit ces propriétés des nœuds et d'exactitude. La trajectoire est toujours
représentée dans une base de Lagrange : les polynômes de Legendre déterminent
les nœuds, et les fonctions de base de Lagrange reconstruisent les valeurs à
partir de ces nœuds.

Pour la **collocation de Radau à droite**, on exige que le dernier nœud soit
en $1$. Avec deux étages, les nœuds sont $\tau_1=1/3$ et $\tau_2=1$. Leurs
fonctions de base de Lagrange sont $\ell_1(\tau)=\tfrac32(1-\tau)$ et
$\ell_2(\tau)=\tfrac32\tau-\tfrac12$. En les intégrant, on obtient

$$
A=\begin{bmatrix}
\tfrac5{12}&-\tfrac1{12}\\
\tfrac34&\tfrac14
\end{bmatrix},\qquad
b=\begin{bmatrix}\tfrac34\\\tfrac14\end{bmatrix}.
$$

La dernière ligne de $A$ est égale à $b^\top$, car cet étage est l'extrémité
droite. Les équations d'étage et d'extrémité impliquent donc
$\mathbf x_{k,2}=\mathbf x_{k+1}$; ces variables peuvent être identifiées et
l'équation en double supprimée. La règle intègre exactement les polynômes
quadratiques, puisque $\tfrac34(\tfrac13)^2+\tfrac14=\tfrac13$, mais elle
n'intègre pas exactement tout polynôme cubique. Elle sacrifie une partie de
l'exactitude de la quadrature pour inclure l'extrémité. Pour $s$ étages, les
nœuds de Radau à droite sont les racines décalées de $P_s(\xi)-P_{s-1}(\xi)$,
dont $\xi=1$. Avec un seul étage, Gauss--Legendre redonne la méthode
implicite du point milieu, et Radau à droite redonne Euler implicite. La
démonstration du pendule utilise ces tableaux de Radau à deux étages; son
[étape 1](lab/notebooks/08_collocation_from_nodes.ipynb#step-1-compute-the-rule-once)
les calcule à la main et dans le code et représente chaque composante comme
une aire signée sous une fonction de base.

CasADi fournit les deux familles de nœuds au moyen de
[`collocation_points`](https://web.casadi.org/python-api/#collocation_points).
On peut les transmettre directement au générateur de coefficients fondé sur
les pentes :

```python
import casadi as ca
from collocation_transcription import make_rule

s = 2
control_nodes = [0.0, 1.0]  # Linear control, chosen separately.
gauss_rule = make_rule(ca.collocation_points(s, "legendre"), control_nodes)
radau_rule = make_rule(ca.collocation_points(s, "radau"), control_nodes)
```

Ici, $s$ compte les étages de pente, ce qui donne un degré d'état d'au plus
$s$. Les deux appels conservent une commande linéaire, avec les valeurs
d'étage $\mathbf u_{k,j}=(1-\tau_j)\widehat{\mathbf u}_{k,0}
+\tau_j\widehat{\mathbf u}_{k,1}$. La même fonction `transcribed_problem`
assemble ensuite les contraintes d'étage, les contraintes d'extrémité et le
coût avec l'une ou l'autre règle. Ce générateur n'a besoin d'aucun nœud
supplémentaire en zéro : l'état gauche intervient déjà comme constante
d'intégration $\mathbf x_k$. L'[exemple de collocation directe de CasADi](https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py)
interpole plutôt des valeurs d'état; il inclut donc l'extrémité gauche parmi
les nœuds de support de l'état avant de dériver ce polynôme.

Les nœuds de Gauss n'incluent aucune extrémité, ceux de Radau à droite
incluent l'extrémité droite, et ceux de Lobatto incluent les deux, comme dans
les exemples des trapèzes et de Hermite--Simpson. Pour chaque famille, les
morceaux d'état adjacents doivent tout de même partager un état d'extrémité
ou être reliés par une contrainte d'égalité. Les contraintes exigées aux
extrémités doivent aussi y être imposées, même lorsque ces extrémités ne sont
pas des étages de collocation. Une quadrature plus précise ne garantit pas à
elle seule que l'optimisation de trajectoire complète est bien résolue; la
représentation de la commande, le maillage et les vérifications entre les
nœuds font toujours partie de la construction.

## Exemple détaillé : déplacer un pont roulant en limitant le balancement résiduel

Comment les variables nodales et les défauts des trapèzes se comportent-ils
dans un problème de mouvement sous contraintes où la charge doit terminer sans
oscillation résiduelle?

Un pont roulant déplace un chariot pendant qu'une charge est suspendue à un
câble. Une commande du chariot calculée à l'avance peut accomplir le
déplacement tout en laissant la charge osciller. La comparaison utilise trois
commandes d'accélération calculées à l'avance. La commande de référence non
façonnée déplace le chariot sans tenir compte de la charge. Un façonneur
d'entrée à vibration nulle modifie cette commande de référence pour annuler
l'oscillation nominale. La collocation directe choisit plutôt une commande à
partir de la dynamique non linéaire de la charge et des contraintes de
mouvement. Les trois commandes sont en boucle ouverte : elles sont fixées
avant le déplacement et ne réagissent à aucune mesure pendant l'exécution.

L'état est $\mathbf{x}=(p,v,\theta,\omega)^\top$, où $p$ et $v$ sont la
position et la vitesse du chariot, et $\theta$ et $\omega$ l'angle et la
vitesse angulaire de la charge. La commande scalaire est l'accélération
commandée du chariot $u=a$. Elle intervient dans la dynamique non linéaire
sous la forme

$$
\dot p=v,\qquad
\dot v=a,\qquad
\dot\theta=\omega,\qquad
\dot\omega=-\frac{g}{\ell}\sin\theta-\frac{a}{\ell}\cos\theta-\gamma\omega.
$$

Ici, $g$ est l'accélération gravitationnelle, $\ell$ la longueur du câble et $\gamma$ le coefficient d'amortissement. Les deux premières équations décrivent le mouvement du chariot, tandis que les deux
dernières décrivent un pendule amorti excité à son point de suspension. Une
accélération positive du chariot fait traîner la charge vers l'arrière, ce qui
explique le signe moins qui multiplie $a$. Le modèle traite le câble comme une
liaison rigide sans masse et suppose que l'accélération du chariot peut être
commandée directement.

Le chariot et la charge partent du repos. Le but est de déplacer le chariot de
$4$ m, de l'arrêter et de laisser la charge suspendue à la verticale sans
balancement. La longueur nominale du câble est $\ell=1.20$ m, avec
$g=9.81$ m/s$^2$ et $\gamma=0.035$ s$^{-1}$. Chaque commande est limitée à
$|a|\leq 1.60$ m/s$^2$, et chaque commande est rejouée sur le même procédé non
linéaire en temps continu avec une période d'échantillonnage de $0.02$ s. Un
second rejeu augmente la longueur du câble de $10\%$ sans recalculer aucune
commande. Ce second procédé teste la sensibilité à un écart de modèle simple.

### Deux commandes de référence en boucle ouverte

La commande de référence non façonnée utilise un profil de vitesse
trapézoïdal symétrique, produit par une accélération à $0.80$ m/s$^2$ jusqu'à
$1.00$ m/s, une phase à vitesse constante et une décélération symétrique. Ces
phases durent respectivement $1.25$, $2.75$ et $1.25$ s. Ses phases
d'accélération et de décélération excitent l'oscillation de la charge, car
leur durée ne tient pas compte de la période du pendule.

Un **façonneur d'entrée** filtre une commande en copies pondérées et
retardées. Les retards sont choisis de sorte que les vibrations excitées par
les copies s'annulent mutuellement. Pour un façonneur à vibration nulle (ZV),
on commence par approcher $\sin\theta\approx\theta$ et $\cos\theta\approx1$
près de l'équilibre suspendu. L'équation de la charge devient alors

$$
\ddot\theta+2\zeta\omega_n\dot\theta+\omega_n^2\theta=-\frac{a}{\ell},
\qquad
\omega_n=\sqrt{\frac{g}{\ell}},
\qquad
\zeta=\frac{\gamma}{2\omega_n}.
$$

Ici, $\omega_n$ est la pulsation propre non amortie et $\zeta$ le taux
d'amortissement. Pour les paramètres nominaux, $\omega_n=2.86$ rad/s et
$\zeta=0.0061$. Le façonneur divise la commande de référence $a_0$ en deux
copies séparées par une demi-période d'oscillation amortie :

$$
a_{\mathrm{ZV}}(t)=A_1a_0(t)+A_2a_0(t-T_d),
\qquad
T_d=\frac{\pi}{\omega_n\sqrt{1-\zeta^2}}\,.
$$

Le retard fait en sorte que la vibration causée par la seconde copie s'oppose
à la vibration qui subsiste de la première. En tenant compte de la décroissance
pendant le retard, on obtient $A_1=1/(1+K)$, $A_2=K/(1+K)$ et
$K=\exp[-\zeta\pi/\sqrt{1-\zeta^2}]$. Ici, $T_d=1.10$ s et les deux poids
valent environ $0.505$ et $0.495$.

### Variables de décision nodales et défauts des trapèzes

La commande façonnée se termine après $5.25+T_d\approx6.35$ s. On donne à la
commande de collocation le même horizon fixe, et on évalue le balancement
résiduel pendant les deux secondes suivantes avec une accélération commandée
nulle.

La troisième commande s'obtient en résolvant le problème d'optimisation de
trajectoire. On divise $[0,5.25+T_d]$ en $N=28$ intervalles égaux, de sorte
que $h=(5.25+T_d)/28$. Le vecteur de décision du PNL contient l'état

$$
\mathbf{x}_k=(p_k,v_k,\theta_k,\omega_k)^\top
$$

et l'accélération $a_k$ en chaque nœud du maillage. Ce sont des valeurs du
polynôme, et non des coefficients monomiaux. Chaque intervalle contribue le
défaut des trapèzes

$$
\mathbf{x}_{k+1}-\mathbf{x}_k
-\frac{h}{2}\left[
\mathbf{f}(\mathbf{x}_k,a_k)+\mathbf{f}(\mathbf{x}_{k+1},a_{k+1})
\right]=\mathbf0.
$$

L'objectif arbitre entre le mouvement de la charge, l'amplitude de
l'accélération et les variations rapides de l'accélération. Les pénalités sur
l'état et l'accélération sont rassemblées dans la quantité nodale

$$
c_k^{\mathrm{rate}}=6\theta_k^2+0.15\omega_k^2+0.035a_k^2.
$$

Le coefficient de $\theta_k^2$ pénalise le plus fortement le balancement,
tandis que les coefficients plus petits pénalisent la vitesse angulaire et
l'accélération. Le coût lisse sur l'état et la commande utilise les poids des
trapèzes aux extrémités. Le dernier terme pénalise la **vitesse de variation
de l'accélération**, c'est-à-dire le rythme auquel la commande d'accélération
change. Comme la commande est linéaire par morceaux, ce rythme est constant
sur chaque intervalle, ce qui donne

$$
\mathcal J
=h\left(\frac12c_0^{\mathrm{rate}}+\sum_{k=1}^{N-1}c_k^{\mathrm{rate}}+\frac12c_N^{\mathrm{rate}}\right)
+0.002h\sum_{k=0}^{N-1}
\left(\frac{a_{k+1}-a_k}{h}\right)^2.
$$

Les conditions aux limites imposent $\mathbf{x}_0=(0,0,0,0)^\top$ et
$\mathbf{x}_N=(4,0,0,0)^\top$, de sorte que le chariot et la charge terminent
tous deux au repos. Les bornes nodales imposent $|a_k|\leq1.60$ m/s$^2$,
$|v_k|\leq1.50$ m/s et $|\theta_k|\leq15^\circ$. L'interpolation linéaire
préserve la borne sur l'accélération entre ses valeurs aux extrémités. Les
bornes sur l'état exigent tout de même des vérifications entre les nœuds et
sur la trajectoire simulée indépendamment.

Le façonneur ZV est adapté à une seule fréquence; il devrait donc laisser le
plus faible balancement résiduel lorsque la longueur du câble correspond à son
modèle de conception. La solution de collocation utilise le modèle nominal non
linéaire complet et traite toutes les contraintes à la fois, mais elle est
elle aussi en boucle ouverte et n'offre aucune garantie explicite de
robustesse. Le test sur la longueur du câble mesure donc la sensibilité à un
seul écart de modèle; il n'établit pas que l'une ou l'autre méthode est
robuste en général.

```{code-cell} python
:tags: [remove-cell]

import sys
sys.path.insert(0, "code")

import pandas as pd
from IPython.display import HTML, display
import matplotlib.pyplot as plt

from crane_control import (
    CraneParameters,
    create_animation as create_crane_animation,
    make_summary_figure as make_crane_summary_figure,
    metrics_table as crane_metrics_table,
    run_comparison as run_crane_comparison,
)

crane_parameters = CraneParameters()
crane_comparison = run_crane_comparison(
    crane_parameters,
    intervals=28,
    sample_period=0.02,
)
```

```{code-cell} python
:tags: [remove-input]
:label: fig-crane-collocation-comparison
:caption: Les trois commandes font accomplir au chariot la même tâche de quatre mètres sur le procédé non linéaire. La commande non façonnée laisse une forte oscillation. Le façonneur ZV annule presque le mode nominal, tandis que la collocation directe atteint l'état terminal avec une commande plus lisse et demandant moins d'effort. Le panneau inférieur rejoue les commandes inchangées après une augmentation de 10 pour cent de la longueur du câble. Les barres hachurées désignent le procédé avec écart de modèle.

crane_summary_figure = make_crane_summary_figure(crane_comparison)
display(crane_summary_figure)
plt.close(crane_summary_figure)
```

Le façonneur ZV produit le plus faible balancement résiduel nominal, car le
procédé simulé correspond étroitement au mode oscillatoire unique qui a servi
à le concevoir. La commande de collocation utilise moins d'accélération au
carré que l'une ou l'autre des commandes de référence et maintient le
balancement résiduel sous un degré dans les deux rejeux. La commande non
façonnée accomplit le déplacement du chariot, mais laisse plusieurs degrés
d'oscillation. L'écart sur la longueur du câble augmente considérablement le
résidu du façonneur ZV, tandis que la commande de collocation se dégrade plus
graduellement pour cette perturbation particulière.

Le tableau rapporte des mesures sur le procédé continu plutôt que des valeurs
nodales du programme non linéaire. Le balancement résiduel est le plus grand
angle absolu après l'horizon de commande commun.

```{code-cell} python
:tags: [remove-input]

crane_table = pd.DataFrame(crane_metrics_table(crane_comparison))
crane_table["residual_sway_deg"] = crane_table["residual_sway_deg"].map(lambda x: f"{x:.3f}")
crane_table["peak_sway_deg"] = crane_table["peak_sway_deg"].map(lambda x: f"{x:.2f}")
crane_table["position_error_mm"] = crane_table["position_error_mm"].map(lambda x: f"{x:.2f}")
crane_table["effort"] = crane_table["effort"].map(lambda x: f"{x:.3f}")
crane_table.rename(
    columns={
        "scenario": "plant",
        "controller": "command",
        "residual_sway_deg": "residual sway (deg)",
        "peak_sway_deg": "peak sway (deg)",
        "position_error_mm": "final position error (mm)",
        "effort": "integral a^2 dt",
    }
)
```

L'animation utilise les mêmes trajectoires de validation non linéaires de
haute précision que la figure. Elle ne rejoue pas le polynôme de collocation
lui-même.

```{code-cell} python
:tags: [remove-input]
:label: fig-crane-collocation-animation
:caption: Rejeu non linéaire continu des commandes non façonnée, façonnée à vibration nulle et obtenue par collocation directe. Tous les panneaux utilisent les mêmes échelles spatiales et temporelles.

crane_animation = create_crane_animation(crane_comparison, frame_stride=5)
crane_html = crane_animation.to_jshtml(fps=25)
plt.close(crane_animation._fig)
display(HTML(crane_html))
```

:::{dropdown} Examiner l'implémentation par transcription directe

```{literalinclude} code/crane_control.py
:language: python
:start-at: def solve_direct_collocation
:end-before: def _compute_metrics
:linenos:
```

:::

{download}`Télécharger l'expérience complète du pont roulant <code/crane_control.py>`

L'optimisation vérifie les défauts algébriques, les bornes et les conditions
aux extrémités aux nœuds. Le rejeu non linéaire distinct vérifie ce qui se
passe entre ces nœuds. Un faible défaut nodal indique que le PNL discret a été
résolu avec précision; il n'indique pas, à lui seul, que le maillage résout la
dynamique continue.

La comparaison n'établit pas que la collocation surpasse toujours le
façonnage d'entrée. Le calcul du façonneur ZV est peu coûteux et peut presque
annuler la vibration résiduelle lorsqu'un mode faiblement amorti est connu
avec précision. La collocation directe devient utile lorsque plusieurs
contraintes sur l'état et les actionneurs doivent être traitées ensemble. Ici,
les trois commandes restent en boucle ouverte. Une rétroaction ou une
replanification à horizon glissant serait nécessaire pour réagir à des
perturbations non mesurées pendant le déplacement.

## Exercices

(classroom-lagrange-practice)=
### Exercices sur l'interpolation de Lagrange

L'exemple détaillé et les quatre premiers exercices forment une séquence de
classe d'environ 30 à 40 minutes. Ils progressent de la construction d'un
polynôme passant par des valeurs prescrites à l'intégration d'une commande sur
un intervalle de temps physique. Le dernier exercice présente l'évaluation
barycentrique comme prolongement facultatif. Chaque solution est repliée pour
que l'on puisse d'abord tenter de résoudre les problèmes.

#### Exemple détaillé : retrouver un polynôme quadratique

Supposons que l'on ne connaisse les valeurs de $f(t)=t^2$ qu'en $t=1,2,3$ et
que l'on cherche un polynôme de degré au plus deux qui passe par elles. Ce
sont des coordonnées physiques pour cet exemple; elles n'ont pas à se trouver
dans l'intervalle normalisé $[0,1]$. Les nœuds et les valeurs sont

$$
(\sigma_0,\sigma_1,\sigma_2)=(1,2,3),\qquad
(y_0,y_1,y_2)=(1,4,9).
$$

Pour associer une fonction de base à $y_0$, on place des zéros aux deux autres
nœuds et on normalise sa valeur en $\sigma_0=1$. On répète pour les nœuds
restants :

$$
\begin{aligned}
\ell_0(t)&=\frac{(t-2)(t-3)}{(1-2)(1-3)}
=\frac{(t-2)(t-3)}2,\\
\ell_1(t)&=\frac{(t-1)(t-3)}{(2-1)(2-3)}
=-(t-1)(t-3),\\
\ell_2(t)&=\frac{(t-1)(t-2)}{(3-1)(3-2)}
=\frac{(t-1)(t-2)}2.
\end{aligned}
$$

Par exemple, en $t=2$, les trois valeurs de base sont $(0,1,0)$; leur somme
pondérée redonne donc $y_1=4$. À tout autre instant, on évalue les mêmes
fonctions et on les combine avec les mêmes valeurs stockées. En développant,
on obtient

$$
\begin{aligned}
p(t)&=\ell_0(t)+4\ell_1(t)+9\ell_2(t)\\
&=\left(\tfrac12t^2-\tfrac52t+3\right)
+\left(-4t^2+16t-12\right)
+\left(\tfrac92t^2-\tfrac{27}2t+9\right)\\
&=t^2.
\end{aligned}
$$

La reconstruction concorde partout avec $f$, car $f$ est elle-même un
polynôme de degré deux. Trois échantillons d'une fonction quelconque
détermineraient son interpolant quadratique, mais ne détermineraient pas
cette fonction entre les échantillons. Ici, par exemple,
$t^2+(t-1)(t-2)(t-3)$ a les trois mêmes valeurs échantillonnées et en diffère
entre elles.

````{exercise} Une droite à partir de deux valeurs stockées
:label: ex-classroom-lagrange-line

Sur $[0,1]$, on prescrit $p(0)=2$ et $p(1)=-1$ et on restreint $p$ à un degré
au plus un.

1. Construisez les deux fonctions de base de Lagrange au moyen de la formule du produit.
2. Écrivez $p$ sous forme de Lagrange, puis développez-le dans la base des monômes.
3. Évaluez $p(1/4)$. Quelle paire de nombres constitue les coefficients dans chaque base?
4. Si l'on supprime la restriction sur le degré, donnez un autre polynôme qui
   a les mêmes valeurs aux extrémités.
````

````{solution} ex-classroom-lagrange-line
:class: dropdown

Les fonctions de base sont $\ell_0(\tau)=(\tau-1)/(0-1)=1-\tau$ et
$\ell_1(\tau)=(\tau-0)/(1-0)=\tau$. En combinant les valeurs stockées, on
obtient

$$
p(\tau)=2(1-\tau)-\tau=2-3\tau,\qquad p(1/4)=5/4.
$$

Ses coefficients de Lagrange sont $(2,-1)$, les valeurs aux extrémités. Ses
coefficients monomiaux sont $(2,-3)$, l'ordonnée à l'origine et la pente. Le
polynôme $2-3\tau+\tau(\tau-1)$ a les mêmes valeurs aux extrémités, car le
terme ajouté s'annule aux deux extrémités. Il est quadratique; il sort donc de
la restriction de degré d'origine.
````

````{exercise} Un polynôme quadratique à partir de trois valeurs stockées
:label: ex-classroom-lagrange-quadratic

Utilisez les nœuds $0,\tfrac12,1$ et les valeurs $y_0=1$, $y_m=2$, $y_1=0$,
où $m$ désigne le milieu. On cherche un polynôme de degré au plus deux.

1. Construisez $\ell_0,\ell_m,\ell_1$ sous forme factorisée, puis développez-les.
2. Vérifiez leurs valeurs aux trois nœuds et vérifiez que
   $\ell_0(\tau)+\ell_m(\tau)+\ell_1(\tau)=1$.
3. Formez $p(\tau)=y_0\ell_0(\tau)+y_m\ell_m(\tau)+y_1\ell_1(\tau)$.
   Simplifiez-le et évaluez $p(1/4)$.
4. Évaluez les trois fonctions de base en $1/4$. Ces poids sont-ils tous
   non négatifs, comme ils l'étaient pour l'interpolation linéaire sur $[0,1]$?
````

````{solution} ex-classroom-lagrange-quadratic
:class: dropdown

En plaçant des zéros aux deux autres nœuds et en normalisant au nœud
sélectionné, on obtient

$$
\begin{aligned}
\ell_0(\tau)&=2(\tau-\tfrac12)(\tau-1)=2\tau^2-3\tau+1,\\
\ell_m(\tau)&=-4\tau(\tau-1)=4\tau-4\tau^2,\\
\ell_1(\tau)&=2\tau(\tau-\tfrac12)=2\tau^2-\tau.
\end{aligned}
$$

Les lignes de leurs valeurs en $0,\tfrac12,1$ sont $(1,0,0)$, $(0,1,0)$ et
$(0,0,1)$. L'addition des expressions développées annule les termes
quadratiques et linéaires et laisse un. L'interpolant est

$$
p(\tau)=\ell_0(\tau)+2\ell_m(\tau)
=1+5\tau-6\tau^2,\qquad p(1/4)=15/8.
$$

En $1/4$, les valeurs des fonctions de base sont $(3/8,3/4,-1/8)$. Leur somme
vaut un, mais elles comprennent un poids négatif. L'interpolant ne reste donc
pas forcément dans l'étendue de ses valeurs nodales.
````

````{exercise} Modifier une valeur et vérifier une borne
:label: ex-classroom-lagrange-perturbation

Conservez les nœuds et le polynôme de l'exercice précédent.

1. Modifiez seulement la valeur au milieu, de $2$ à $3$. Écrivez le nouveau
   polynôme $\widetilde p$ à l'aide de l'ancien $p$ et d'une seule fonction de
   base. De combien sa valeur en $1/4$ change-t-elle? Que se passe-t-il aux
   extrémités?
2. Pour le polynôme d'origine $p$, les trois valeurs stockées satisfont toutes
   $p\leq2$. Trouvez son maximum sur $[0,1]$. La borne est-elle respectée
   partout?
3. Expliquez ce que ces calculs impliquent lorsque les valeurs nodales sont
   des variables de décision représentant une trajectoire d'état.
````

````{solution} ex-classroom-lagrange-perturbation
:class: dropdown

Seul le coefficient de $\ell_m$ change, d'où

$$
\widetilde p(\tau)=p(\tau)+\ell_m(\tau)=1+9\tau-10\tau^2.
$$

La variation en $1/4$ vaut $\ell_m(1/4)=3/4$, ce qui donne
$\widetilde p(1/4)=21/8$. Les extrémités sont inchangées, car $\ell_m$ s'y
annule.

Pour le polynôme d'origine, $p'(\tau)=5-12\tau$ s'annule en $5/12$. Comme
$p''=-12<0$, ce point est son maximum, avec

$$
p(5/12)=\frac{49}{24}=2+\frac1{24}>2.
$$

Modifier une variable de décision nodale modifie la courbe entre les nœuds par
l'intermédiaire de sa fonction de base. De même, des bornes sur les seules
variables nodales ne garantissent pas des bornes sur une trajectoire d'état
quadratique entre les nœuds. La transcription exige des vérifications
supplémentaires à ces endroits, comme le décrit
[Vérifier la trajectoire continue](#sec-collocation-validation).
````

````{exercise} Intégrer une commande en une trajectoire d'état
:label: ex-classroom-lagrange-state-change

Sur l'intervalle physique $2\leq t\leq4$, on considère $\dot x(t)=u(t)$ et
$x(2)=3$. On représente la commande par le polynôme quadratique qui passe par
$u(2)=0$, $u(3)=1$ et $u(4)=0$. Soit $\tau=(t-2)/2$.

1. Exprimez $u(2+2\tau)$ au moyen de la base de l'exercice à trois nœuds.
2. Intégrez chaque fonction de base sur $[0,1]$ pour obtenir les poids
   $w_0,w_m,w_1$. Servez-vous-en pour calculer $x(4)-x(2)$, en tenant compte
   de la longueur de l'intervalle physique.
3. Trouvez le morceau d'état complet $p(\tau)=x(2+2\tau)$ et donnez son degré.
   Vérifiez que $p'(\tau)/2=u(2+2\tau)$.
4. Si $x_k$ et $x_{k+1}$ sont des états d'extrémité stockés, écrivez la
   contrainte de défaut d'extrémité pour des valeurs de commande quelconques
   $u_0,u_m,u_1$ aux trois mêmes instants. Quel état d'extrémité est exigé
   pour les données fournies?
````

````{solution} ex-classroom-lagrange-state-change
:class: dropdown

La seule valeur de commande non nulle est au milieu, d'où
$u(2+2\tau)=\ell_m(\tau)=4\tau(1-\tau)$. L'intégration des fonctions de base
développées donne

$$
\begin{aligned}
w_0&=\tfrac23-\tfrac32+1=\tfrac16,\\
w_m&=2-\tfrac43=\tfrac23,\\
w_1&=\tfrac23-\tfrac12=\tfrac16.
\end{aligned}
$$

Comme $dt=2\,d\tau$, le déplacement vaut
$2(w_0\cdot0+w_m\cdot1+w_1\cdot0)=4/3$. En intégrant seulement jusqu'au
$\tau$ courant, on obtient le morceau d'état

$$
p(\tau)=3+2\int_0^\tau4\eta(1-\eta)\,d\eta
=3+4\tau^2-\frac83\tau^3.
$$

Il est cubique, avec $p'(\tau)/2=4\tau-4\tau^2$, comme il se doit. La
contrainte d'extrémité pour des valeurs stockées quelconques est

$$
x_{k+1}-x_k-2\left(\frac16u_0+\frac23u_m+\frac16u_1\right)=0.
$$

Pour $x_k=3$ et les commandes prescrites, elle exige $x_{k+1}=13/3$. Cette
intégration est exacte pour la commande quadratique choisie et la dynamique
$\dot x=u$; aucune affirmation d'exactitude pour une dynamique non linéaire
générale n'est nécessaire.
````

````{exercise} Facultatif : évaluation barycentrique
:label: ex-classroom-lagrange-barycentric

Revenez à l'exemple détaillé avec les nœuds $\sigma_j=1,2,3$ et les valeurs
$y_j=1,4,9$. On peut évaluer un polynôme sans développer ses fonctions de base
en précalculant les **poids barycentriques**

$$
\beta_j=\frac{1}{\prod_{m\ne j}(\sigma_j-\sigma_m)}.
$$

Ces poids décrivent une formule d'évaluation; ils sont distincts des poids
d'intégration $w_j$ utilisés plus haut. Hors des nœuds, le même polynôme
d'interpolation est donné par

$$
p(t)=\frac{\displaystyle\sum_{j=0}^2\frac{\beta_jy_j}{t-\sigma_j}}
{\displaystyle\sum_{j=0}^2\frac{\beta_j}{t-\sigma_j}}.
$$

1. Calculez $\beta_0,\beta_1,\beta_2$ et appliquez la formule en $t=3/2$.
2. Multipliez les trois poids par $2$. Le résultat change-t-il?
3. Que devrait renvoyer une implémentation lorsque la requête est exactement
   $t=2$? Pourquoi devrait-elle éviter d'y évaluer directement les quotients
   affichés?
````

````{solution} ex-classroom-lagrange-barycentric
:class: dropdown

Les produits des différences entre nœuds donnent $(\beta_0,\beta_1,\beta_2)
=(1/2,-1,1/2)$. En $t=3/2$, le numérateur et le dénominateur valent

$$
\frac{1/2}{1/2}+\frac{-4}{-1/2}+\frac{9/2}{-3/2}=6,
\qquad
\frac{1/2}{1/2}+\frac{-1}{-1/2}+\frac{1/2}{-3/2}=\frac83.
$$

Leur rapport vaut $9/4$, ce qui concorde avec $(3/2)^2$. Multiplier chaque
poids par un même nombre non nul multiplie les deux sommes par le même
facteur et laisse leur rapport inchangé. Pour une requête égale à un nœud, on
renvoie directement la valeur stockée : $p(2)=4$. Le polynôme y est défini,
mais les quotients individuels de cette formule d'évaluation y divisent par
zéro.
````

### Exercices supplémentaires

Les exercices suivants reviennent sur les coordonnées polynomiales et les
relient aux schémas de collocation complets et aux vérifications de
trajectoire développés dans le chapitre.

````{exercise}
:label: ex-collocation-coordinates

Soit $p(\tau)=2-\tau+2\tau^2$; on choisit les nœuds de support $0,\tfrac12,1$.

1. Calculez son vecteur de coordonnées nodales $\mathbf{y}$.
2. Écrivez la matrice d'évaluation $V$ pour la base des monômes.
3. Retrouvez le vecteur des coefficients monomiaux à partir de $V\mathbf{a}=\mathbf{y}$.
````

````{solution} ex-collocation-coordinates
:class: dropdown

Les valeurs nodales sont

$$
\mathbf{y}=
\begin{bmatrix}
p(0)\\p(\tfrac12)\\p(1)
\end{bmatrix}
=
\begin{bmatrix}
2\\2\\3
\end{bmatrix}.
$$

Pour la base des monômes,

$$
V=
\begin{bmatrix}
1&0&0\\
1&\tfrac12&\tfrac14\\
1&1&1
\end{bmatrix}.
$$

La résolution de $V\mathbf{a}=\mathbf{y}$ donne $\mathbf{a}=(2,-1,2)^\mathsf T$. Cette résolution change de coordonnées; elle ne construit pas un polynôme différent.
````

````{exercise}
:label: ex-collocation-interpolation-regression

Classez chaque problème comme interpolation, régression par moindres carrés, ou ni l'une ni l'autre.

1. Trouver un polynôme cubique qui passe par quatre valeurs exactes distinctes.
2. Ajuster une tendance cubique à vingt mesures de température bruitées.
3. Choisir des états nodaux inconnus de sorte qu'un résidu d'EDO s'annule en quatre nœuds de collocation.
4. Ajuster une droite passant par deux valeurs exactes distinctes en minimisant l'erreur quadratique.
````

````{solution} ex-collocation-interpolation-regression
:class: dropdown

Les problèmes 1 et 4 sont des problèmes d'interpolation; la formulation par moindres carrés du problème 4 a une solution d'interpolation à résidu nul. Le problème 2 est une régression. Le problème 3 n'est ni une régression statistique ni une interpolation d'observations : les valeurs nodales inconnues définissent un polynôme d'interpolation, et la collocation ajoute des contraintes d'égalité qui sélectionnent ces valeurs.
````

````{exercise}
:label: ex-collocation-operators

Pour les nœuds $0,\tfrac12,1$ :

1. Construisez les trois fonctions cardinales de Lagrange.
2. Évaluez leurs dérivées aux trois nœuds pour obtenir $D$.
3. Intégrez-les sur $[0,1]$ pour obtenir $w$.
4. Vérifiez que $D$ dérive exactement $p(\tau)=1+2\tau-\tau^2$ aux nœuds.
````

````{solution} ex-collocation-operators
:class: dropdown

Les fonctions cardinales sont

$$
\ell_0=2\tau^2-3\tau+1,\qquad
\ell_m=4\tau(1-\tau),\qquad
\ell_1=2\tau^2-\tau.
$$

Elles donnent

$$
D=
\begin{bmatrix}
-3&4&-1\\
-1&0&1\\
1&-4&3
\end{bmatrix},
\qquad
w=\frac16
\begin{bmatrix}1\\4\\1\end{bmatrix}.
$$

Pour $\mathbf{y}=(1,\tfrac74,2)^\mathsf T$, $D\mathbf{y}=(2,1,0)^\mathsf T$, ce qui est égal à $p'(\tau)=2-2\tau$ aux trois nœuds.
````

````{exercise}
:label: ex-collocation-euler-trapezoid

En partant de

$$
\dot{\mathbf{x}}_h(t_k+h_k\tau)=\sum_j\mathbf{f}_{k,j}\ell_j(\tau),
$$

dérivez :

1. Euler explicite à partir de l'unique nœud de pente $\tau_1=0$;
2. Euler implicite à partir de l'unique nœud de pente $\tau_1=1$;
3. le défaut des trapèzes à partir des nœuds de pente $\tau_1=0$ et $\tau_2=1$.

Donnez dans chaque cas le degré de l'approximation de l'état obtenue.
````

````{solution} ex-collocation-euler-trapezoid
:class: dropdown

Un seul nœud donne l'interpolant constant de la dérivée $\mathbf{f}_k$ ou $\mathbf{f}_{k+1}$. L'intégration donne le défaut d'Euler explicite ou implicite et un état de degré un. Avec les nœuds aux extrémités, $\dot{\mathbf{x}}_h=(1-\tau)\mathbf{f}_k+\tau \mathbf{f}_{k+1}$. Son intégrale en $\tau=1$ vaut $\tfrac12(\mathbf{f}_k+\mathbf{f}_{k+1})$, ce qui donne le défaut des trapèzes. L'intégration de la dérivée linéaire donne un état de degré deux.
````

````{exercise}
:label: ex-collocation-hermite-simpson-degree

Expliquez pourquoi Hermite--Simpson a un interpolant d'état cubique alors que sa dérivée n'est représentée qu'en trois nœuds. Dérivez ensuite les poids d'extrémité de Simpson et la relation au milieu.
````

````{solution} ex-collocation-hermite-simpson-degree
:class: dropdown

Trois valeurs de pente distinctes définissent un interpolant quadratique de la dérivée. L'intégration de ce polynôme quadratique ajoute un degré; l'état est donc cubique. L'intégration des fonctions cardinales sur $[0,1]$ donne $(1,4,1)/6$, et donc le défaut de Simpson. L'intégration jusqu'à $\tau=\tfrac12$ donne $(5,8,-1)/24$; en éliminant la pente au milieu au moyen du défaut d'extrémité, on obtient

$$
\mathbf{x}_{k+\frac12}
=\frac12(\mathbf{x}_k+\mathbf{x}_{k+1})
+\frac{h_k}{8}(\mathbf{f}_k-\mathbf{f}_{k+1}).
$$
````

````{exercise} Générer une transcription du point milieu
:label: ex-collocation-generated-midpoint

Choisissez l'unique nœud de collocation $\tau_1=1/2$ et une commande constante
sur chaque intervalle.

1. Utilisez la fonction cardinale pour calculer $A$, $b$ et $B$, puis vérifiez
   les tableaux avec `make_rule([0.5], [0.0])`.
2. Éliminez l'état d'étage des équations d'étage et d'extrémité. Donnez le
   défaut obtenu et expliquez pourquoi il s'agit d'une règle implicite du
   point milieu.
3. Pour $\dot x=u$, minimisez $\int_0^1u(t)^2\,dt$ avec $x(0)=0$, $x(1)=1$ et
   $|u|\leq2$. Écrivez le PNL complet sur le maillage $0,1/2,1$, y compris ses
   variables de décision, et résolvez-le.
````

````{solution} ex-collocation-generated-midpoint
:class: dropdown

La fonction cardinale est $\ell_1=1$, d'où $A=[1/2]$, $b=[1]$ et $B=[1]$.
L'équation d'étage donne $\mathbf x_{k,1}=\mathbf x_k+(h_k/2)\mathbf f_{k,1}$,
tandis que l'équation d'extrémité donne
$\mathbf x_{k+1}=\mathbf x_k+h_k\mathbf f_{k,1}$. En éliminant la pente entre
ces équations, on obtient $\mathbf x_{k,1}=(\mathbf x_k+\mathbf x_{k+1})/2$,
et donc

$$
\mathbf x_{k+1}-\mathbf x_k
-h_k\mathbf f\!\left(
\frac{\mathbf x_k+\mathbf x_{k+1}}2,\mathbf u_{k,1},t_k+\frac{h_k}2
\right)=\mathbf0.
$$

L'état droit inconnu apparaît à l'intérieur de l'évaluation de la dynamique,
ce qui rend la règle implicite. Le polynôme d'état est linéaire et la commande
constante. Pour le problème scalaire avec deux intervalles de longueur un
demi, le PNL complet est

$$
\begin{aligned}
\min_{x_0,x_1,x_2,u_{0,1},u_{1,1}}\quad&
\frac12(u_{0,1}^2+u_{1,1}^2)\\
\text{sous les contraintes}\quad&
x_{k+1}-x_k-\frac12u_{k,1}=0,\quad k=0,1,\\
&-2\leq u_{k,1}\leq2,\quad k=0,1,\\
&x_0=0,\qquad x_2=1.
\end{aligned}
$$

Les défauts impliquent $u_{0,1}+u_{1,1}=2$. La substitution dans l'objectif
donne $(u_{0,1}-1)^2+1$; les deux commandes valent donc un, $x_1=1/2$, et le
coût minimal vaut un. On retrouve la trajectoire d'ouverture sur un maillage
plus fin.
````

````{exercise} Variation de la commande entre les étages échantillonnés
:label: ex-collocation-control-sampling

Sur $[0,1]$, on considère $\dot x=u$ avec $x(0)=0$ et des étages de
collocation aux extrémités seulement. Comparez les commandes $u(t)=0$ et
$u(t)=4t(1-t)$.

1. Quelles valeurs de commande entrent dans la transcription aux extrémités?
   Son défaut des trapèzes peut-il distinguer les deux commandes?
2. Intégrez chaque commande pour trouver l'état en $t=1$. Expliquez pourquoi
   une commande quadratique peut poser problème lorsqu'on n'utilise que des
   étages aux extrémités.
3. Proposez une modification de la représentation de la commande ou des nœuds
   d'étage qui distingue ces deux commandes.
````

````{solution} ex-collocation-control-sampling
:class: dropdown

Les deux commandes sont nulles aux deux extrémités. Le défaut des trapèzes
prédit donc $x(1)-x(0)=0$ pour l'une comme pour l'autre. Leurs déplacements
réels valent zéro et

$$
\int_0^1 4t(1-t)\,dt
=4\left(\frac12-\frac13\right)=\frac23.
$$

Des échantillons aux extrémités ne peuvent pas détecter la variation
intérieure de la commande quadratique. Restreindre la commande à une droite
fait que deux valeurs nulles aux extrémités déterminent la commande nulle. Un
étage au milieu, autre possibilité, échantillonne la seconde commande à la
valeur un, si bien que les deux commandes produisent des pentes d'étage
différentes. Avec les trois étages, les poids de Simpson intègrent exactement
cette commande quadratique.
````

````{exercise}
:label: ex-collocation-between-node-residual

Un PNL rapporte un défaut de collocation maximal de $10^{-10}$. Le polynôme
reconstruit présente un résidu d'EDO élevé à mi-chemin entre deux nœuds, et
une simulation indépendante de la commande choisie viole une borne sur l'état.

1. Pourquoi ces observations sont-elles compatibles?
2. Quel diagnostic faut-il calculer?
3. Quelle modification de la transcription constitue la première réponse naturelle?
````

````{solution} ex-collocation-between-node-residual
:class: dropdown

Le défaut du PNL mesure ses contraintes d'égalité aux nœuds choisis. Un
polynôme grossier peut satisfaire ces contraintes sans résoudre le
comportement entre elles. On évalue $\mathbf r_k$ et les contraintes de chemin
sur le polynôme à des instants supplémentaires. Séparément, on intègre l'EDO
d'origine avec la commande choisie et on compare les états simulés au
polynôme. On raffine les intervalles en cause ou on augmente le degré local,
puis on résout et on valide de nouveau. Resserrer une tolérance du PNL déjà
faible ne corrige pas l'erreur de représentation.
````

````{exercise}
:label: ex-collocation-crane-mismatch

Téléchargez `code/crane_control.py`. Rejouez les commandes fixes du façonneur ZV et de la collocation pour des longueurs de câble comprises entre $0.9\ell$ et $1.1\ell$, sans recalculer l'une ou l'autre commande. Tracez le balancement résiduel en fonction de la longueur du câble et expliquez la courbe à l'aide de la pulsation propre $\sqrt{g/\ell}$.
````

## Résumé et perspectives

La collocation directe transforme une optimisation sur des fonctions entières
en une optimisation sur un nombre fini de valeurs d'état et de commande. Ces
valeurs nodales définissent des segments polynomiaux. Des contraintes de
défaut algébriques exigent ensuite que chaque segment s'accorde avec l'EDO en
des points choisis.

Chaque polynôme admet deux descriptions équivalentes : des coefficients dans
une base ou des valeurs en des nœuds,

$$
p(\tau)=\sum_j a_j\phi_j(\tau)
=\sum_j y_j\ell_j(\tau),
\qquad \mathbf y=V\mathbf a.
$$

La collocation directe utilise la description nodale, car les valeurs
stockées ont une signification physique immédiate. Les fonctions cardinales de
Lagrange les transforment en opérateurs fixes qui calculent des dérivées, des
valeurs aux extrémités et des intégrales. Même si cette construction
ressemble à un ajustement polynomial, les valeurs nodales sont des variables
d'optimisation plutôt que des observations bruitées, et le résidu de l'EDO est
une contrainte d'égalité plutôt qu'une perte de régression.

Les formes par intégration et par dérivation imposent toutes deux la
collocation en des nœuds choisis. La forme par intégration interpole les
pentes évaluées de l'EDO et utilise des résidus de concordance des états pour
que leur intégrale s'accorde avec les états stockés. La forme par dérivation
interpole des valeurs d'état et utilise des résidus de concordance des pentes
pour que leur dérivée s'accorde avec l'EDO. Avec les mêmes choix de nœuds, le
même degré et le même espace de commandes, les deux décrivent les mêmes
trajectoires polynomiales admissibles.

L'interpolation d'une seule pente, à gauche ou à droite, donne Euler explicite
ou implicite et une approximation linéaire de l'état. L'interpolation des deux
pentes aux extrémités donne le défaut des trapèzes et une approximation
quadratique de l'état. L'ajout de la pente au milieu donne les poids de
Simpson, la relation au milieu de Hermite--Simpson et une approximation
cubique de l'état.

Le degré de la commande se choisit séparément. Ses valeurs de support
déterminent les commandes d'étage au moyen de la matrice d'évaluation $B$. Un
programme peut construire $A$, $b$ et $B$ à partir des nœuds choisis, puis
assembler les mêmes expressions d'étage, d'extrémité et de coût pour tous ces
schémas. Changer les nœuds ou la représentation de la commande modifie
l'approximation finie sans exiger une nouvelle dérivation de ces expressions.

Sur de nombreux intervalles, chaque défaut ne fait intervenir que les
extrémités voisines et des valeurs d'étage locales. La plupart des composantes
de la jacobienne des contraintes sont donc nulles, ce qui permet à un solveur
de PNL d'exploiter une structure creuse en bande par blocs. L'exemple du pont
roulant montre aussi pourquoi la résolution du PNL n'est pas la vérification
finale : des contraintes respectées aux nœuds peuvent tout de même être
violées entre eux. Les résidus polynomiaux évaluent l'approximation locale, et
une simulation indépendante vérifie la trajectoire produite par la commande
choisie.

Comment améliorer une suite de commandes lorsqu'un simulateur en évalue le
coût sans fournir de dérivées?
[La commande prédictive par intégrale de chemin (MPPI)](model-predictive-path-integral-control.md)
échantillonne des suites de commandes candidates et les combine au moyen de
poids qui dépendent du coût. Lorsque des perturbations physiques rendent le
déroulement incertain, plusieurs futurs simulés estiment le coût espéré de
chaque candidate. Le [chapitre sur la commande à horizon glissant](receding-horizon-control.md)
qui suit étudie comment la replanification à partir de mesures fournit une
rétroaction à un optimiseur à horizon fini.
