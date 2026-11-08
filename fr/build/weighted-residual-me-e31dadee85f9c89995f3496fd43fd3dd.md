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
# Fonctions de valeur paramétriques

:::{note} Traduction à mettre à jour
La version anglaise de ce chapitre a été modifiée depuis cette traduction.
Le menu « English » mène à la version à jour.
:::

Les chapitres de programmation dynamique ont résolu les équations de Bellman en
stockant un nombre par état, ou par état de la grille. Cela fonctionne pour le
réservoir, le traitement antibiotique et une rivière de huit tronçons, et cela
échoue peu après. Cette partie du livre remplace la table par une fonction à
nombre fixe de paramètres, et elle progresse vers les algorithmes employés en
pratique : l'itération Q ajustée avec des arbres ou des réseaux de neurones, sa
forme en ligne DQN et sa forme à actions continues NFQCA. Ce chapitre commence
par expliquer pourquoi ce remplacement est nécessaire et ce qu'il doit
fournir. Il pose ensuite la première question que soulève ce remplacement. Une
fonction à nombre fini de paramètres ne peut pas satisfaire l'équation de
Bellman en chaque état; quelles équations doivent donc déterminer ses
paramètres? L'analyse numérique répond à cette question pour les équations
d'opérateurs en général avec la méthode des résidus pondérés, un gabarit en
cinq étapes. Nous présentons ce gabarit directement sur l'équation de Bellman.
Un seul problème, une pêcherie proche d'un point de bascule, le motive puis le
suit à chaque étape, dans des encadrés dont la méthode ne dépend pas.

::::{admonition} Objectifs d'apprentissage
:class: note

Après la lecture de ce chapitre, vous devriez pouvoir :
- expliquer pourquoi la programmation dynamique tabulaire échoue pour les espaces d'états grands ou continus, et ce qu'un approximateur de fonctions doit fournir à sa place;
- écrire le résidu de Bellman d'une fonction de valeur paramétrique, et dériver les conditions de collocation, de Galerkin et des moindres carrés qui déterminent ses paramètres;
- résoudre les équations obtenues par itération fonctionnelle ou par la méthode de Newton, et énoncer ce que chacune exige du problème;
- juger une fonction de valeur approchée par son résidu, son erreur de valeur et la perte de sa politique gloutonne, et expliquer pourquoi ces trois mesures peuvent être en désaccord.

**Préalables :** opérateur de Bellman et propriété de contraction ([processus de décision markoviens à horizon infini](infinite-horizon-mdps.md)), interpolation par fonctions cardinales ([programmation dynamique à horizon fini](finite-horizon-dp.md)), quadrature numérique de base, algèbre linéaire (inverses de matrices, projection orthogonale).
::::

(sec-why-approximate)=
## Limites de la programmation dynamique tabulaire

On ne peut pas simplement agrandir les tables des chapitres précédents, pour
trois raisons, chacune déjà visible dans un exemple que nous avons résolu.

La première est que le nombre d'états croît exponentiellement avec le nombre de
variables d'état. La rivière envahie par le tamaris de {ref}`sec-tamarisk-river`
compte un site par tronçon et trois conditions par site, de sorte qu'une rivière
de $n$ tronçons a $3^n$ états. L'itération sur les valeurs a demandé environ
150 balayages quelle que soit la taille de la rivière, mais chaque balayage
touche tous les états, et une table de transition dense pour une rivière de
sept tronçons sans le budget de traitement occuperait 84 Go. Les rivières
auxquelles font face les gestionnaires du territoire comptent des centaines de
tronçons. Les états continus se heurtent au même mur par l'entremise de la
grille. Le réservoir du
[chapitre sur l'horizon fini](finite-horizon-dp.md) stockait sa valeur sur 181
niveaux de stockage; cinq réservoirs d'une même vallée, gérés conjointement sur
la même grille, demanderaient $181^5 \approx 2 \times 10^{11}$ états de grille,
chacun avec sa propre optimisation à une étape à chaque stade. Une grille de $N$
nœuds par axe en $D$ dimensions compte $N^D$ nœuds. C'est la malédiction de la
dimension que Bellman a nommée, et elle tient à la représentation plutôt qu'au
problème : la fonction de valeur d'un grand système a souvent une structure,
comme la régularité ou la dépendance à quelques grandeurs agrégées, qu'une
grille ne peut pas exploiter et qu'une représentation plus souple peut
exploiter.

La deuxième est que les successeurs tombent entre les états stockés. Même en
une dimension, le
[chapitre sur l'horizon fini](finite-horizon-dp.md#handling-continuous-spaces-with-interpolation)
a dû décider quelle valeur employer pour un successeur qui tombe entre deux
états de la grille. Sa réponse, une famille de fonctions cardinales, a
transformé la table en une fonction définie partout, et son analyse de la
propagation de l'erreur de reconstruction à travers la récurrence était un
premier exemple de ce que cette partie du livre étudie en général.

La troisième est que l'apprentissage à partir de données ne visite que certains
états. Dans les chapitres à venir, le modèle est remplacé par un simulateur ou
par des transitions enregistrées $(s, a, r, s')$. Un lot de transitions couvre
une fraction infime d'un grand espace d'états, et chaque cible de Bellman
$r + \gamma v(s')$ exige une valeur en un successeur $s'$ qui n'a peut-être
jamais été visité. Une table n'a rien à dire à cet endroit. Une fonction
ajustée aux états visités, si : elle généralise à partir d'eux, et la qualité
de cette généralisation décide de la qualité de la méthode.

Le remplaçant de la table est donc une fonction $v(s; \theta)$ dotée d'un
vecteur de paramètres $\theta$ de taille fixe. Elle doit être peu coûteuse à
évaluer en tout état. Il doit être possible de l'ajuster à un ensemble fini de
paires (état, valeur cible), ce qui est un problème d'apprentissage supervisé.
Et elle doit extrapoler de façon sensée entre ces états et au-delà.
L'interpolation sur une grille est peu coûteuse et facile à ajuster, mais elle
n'extrapole qu'aux cellules de sa grille et sa taille croît comme $N^D$. Un
polynôme ou une autre base fixe est linéaire en $\theta$, ce qui donne des
ajustements en forme close et une théorie complète, mais ses caractéristiques
doivent être choisies d'avance. Les ensembles d'arbres et les réseaux de
neurones choisissent leurs caractéristiques à partir des données. Les méthodes
d'itération Q ajustée employées en pratique, de la FQI à base d'arbres de
{cite:t}`ernst2005tree` à NFQ {cite}`riedmiller2005neural` et NFQCA
{cite}`Hafner2011`, s'en servent.

Remplacer $v$ par $v(\cdot;\theta)$ soulève deux questions. L'équation de
Bellman $v = \Bellman v$ impose une condition par état, une infinité lorsque
l'espace d'états est continu, alors que $\theta$ a un nombre fini de
composantes. Quel ensemble fini de conditions doit déterminer $\theta$? Ce
chapitre y répond avec le gabarit des résidus pondérés de l'analyse numérique
{cite}`Chakraverty2019,AtkinsonPotra1987`, en employant des bases linéaires,
pour lesquelles les conditions peuvent être écrites explicitement et résolues
en forme close. Une fois les conditions fixées, quel algorithme trouve
$\theta$, et la contraction qui faisait converger l'itération sur les valeurs
survit-elle à l'approximation? Le
[chapitre suivant](approximate-bellman-equations.md) répond à la seconde
question pour une règle d'ajustement quelconque, réseaux de neurones compris,
et se termine par les étapes qui mènent de là à FQI et à NFQCA.

(sec-wr-fishery)=
## Un exemple motivant : l'exploitation d'une pêcherie proche d'un point de bascule

Cet exemple montre ce qui tourne mal lorsqu'une fonction de valeur sur un
continuum est stockée dans quelques nombres, et comment le choix de la
représentation modifie les décisions que produit la fonction stockée. Un
organisme de réglementation fixe le quota annuel d'un stock de poissons. Avant
la saison, un relevé donne le stock $x$, mesuré comme une fraction de la
capacité de charge de l'habitat. Le régulateur choisit l'*échappement*
$s \in [0, x]$, la part du stock laissée dans l'eau pour frayer. Le reste,
$x - s$, est pêché et vendu à un prix unitaire. Au cours de l'année suivante, le
stock échappé croît jusqu'à

$$
x' = z\, f(s), \qquad f(s) = s \exp\!\Big(r \Big(1 - \frac{s}{K}\Big) \frac{s - C}{K}\Big),
$$

où $z$ est un choc log-normal de moyenne un qui représente l'environnement : le
nombre de jeunes poissons qui survivent varie beaucoup d'une année à l'autre.
Cette loi de croissance, que {cite:t}`Boettiger2015` emploient pour étudier les
points de bascule dans les pêcheries, présente un effet Allee. Sous le seuil
$C$, les poissons sont trop clairsemés pour se renouveler, et le stock diminue
même lorsque personne ne le pêche (panneau (a) de
{numref}`fig-fishery-problem`). Nous prenons $r = 2$, $K = 1$, $C = 0.3$, un
choc dont le logarithme a un écart type de $0.15$, et $\gamma = 0.95$. Ce sont
des valeurs pédagogiques, et non l'évaluation d'un stock réel. L'équation
d'optimalité de Bellman est

$$
v^\star(x) = \max_{0 \le s \le x} \Big\{ x - s + \gamma\, \mathbb{E}_z\big[ v^\star\big(z f(s)\big) \big] \Big\}, \qquad x \in [0, 1.6].
$$

Trois choses la distinguent des processus de décision markoviens finis des
chapitres précédents. L'état est un nombre réel, de sorte que $v^\star$ est une
fonction sur un intervalle et qu'aucune table ne la contient. L'action est
aussi un nombre réel, de sorte que la maximisation porte sur un intervalle. Et
l'espérance est une intégrale sur $z$, que nous calculons par une quadrature
de Gauss-Hermite à 21 points. Le stock successeur $z f(s)$ ne tombe presque
jamais sur un point où une valeur a été stockée, ce qui est la difficulté que
l'[interpolation](finite-horizon-dp.md#handling-continuous-spaces-with-interpolation)
traitait dans le chapitre sur l'horizon fini.

Ce problème n'a qu'une variable d'état, de sorte que nous pouvons encore nous
offrir une solution de référence : l'itération sur les valeurs sur une grille
de 4 001 points, avec interpolation linéaire entre eux et des échappements
restreints à la grille. Comme la valeur de la récolte $x - s$ est linéaire en
$s$, la maximisation sur $s \le x$ se réduit à un maximum courant le long de la
grille, et une mise à jour ne coûte qu'un seul passage. Diviser par deux le
nombre de points de la grille modifie la solution de moins de $0.001$.

:::{figure} _static/fishery_tipping_point/problem.svg
:label: fig-fishery-problem

La pêcherie. (a) Stock espéré l'année suivante en fonction de l'échappement,
avec la bande qui contient 80 % des années. Sous le point de bascule $C$, la
courbe se situe sous la diagonale : le stock diminue même sans pêche. (b) La
valeur optimale $v^\star$, calculée sur une grille fine. (c) L'échappement
optimal. L'ombrage marque les trois régions de la politique.
:::

La politique optimale a trois régions. Au-dessus de $S^\star = 0.712$, le
régulateur pêche jusqu'à l'échappement constant $S^\star$, la politique dont
{cite:t}`Reed1979` a démontré l'optimalité pour cette classe de modèles. Sa
condition $\gamma f'(S^\star) = 1$ donne $0.710$. Le petit écart vient des
mauvaises années, qui laissent le stock sous $S^\star$, là où un poisson laissé
pour frayer vaut plus que son prix. Entre $\hat x = 0.23$ et $S^\star$, la
pêcherie est fermée et le stock se reconstitue. Sous $\hat x$, la politique
épuise le stock : un stock aussi loin sous le point de bascule déclinera fort
probablement de toute façon, et sa prise d'aujourd'hui vaut plus que sa faible
chance de rétablissement. {cite:t}`Clark1973` a montré que maximiser la valeur
de la récolte peut ainsi rendre l'extinction optimale. Le droit des pêches
ajoute pour cette raison des limites de conservation, et ce modèle les omet.
Pour les méthodes de ce chapitre, ce qui compte est la forme de $v^\star$. Elle
est linéaire sur $[0, \hat x]$, présente un coude en $\hat x$, est multipliée
par plus de dix à la traversée du point de bascule, et est régulière ailleurs.

Supposons maintenant que nous ne puissions stocker que $n$ nombres. Deux choix
sont naturels.

- Un polynôme de Tchebychev, $\hat v(x; \theta) = \sum_{j=0}^{n-1} \theta_j T_j(2x/1.6 - 1)$.
  La base est régulière et globale : chaque coefficient modifie la valeur pour
  chaque stock.
- L'interpolation linéaire par morceaux sur $n$ nœuds uniformes, les fonctions
  chapeau du chapitre sur l'horizon fini. Chaque coefficient est la valeur en un
  nœud, et il ne modifie la valeur qu'entre les deux nœuds voisins.

Aucun des deux ne peut satisfaire l'équation de Bellman pour chaque stock. Le
résidu

$$
R(x; \theta) = (\Bellman \hat v)(x) - \hat v(x; \theta)
$$

mesure de combien elle échoue en $x$. La collocation choisit $n$ points
$x_1, \ldots, x_n$ et exige que le résidu s'y annule :

$$
R(x_i; \theta) = 0, \qquad i = 1, \ldots, n.
$$

Nous employons les nœuds de Tchebychev pour le polynôme et les nœuds
d'interpolation pour les fonctions chapeau, et nous résolvons les $n$ équations
non linéaires par itération fonctionnelle. Chaque itération calcule la mise à
jour de Bellman aux nœuds avec le $\hat v$ courant, puis choisit le $\theta$
qui reproduit ces cibles aux nœuds. La {numref}`fig-fishery-collocation`
montre les deux approximations avec $n = 12$.

:::{figure} _static/fishery_tipping_point/collocation.svg
:label: fig-fishery-collocation

Collocation avec $n = 12$ coefficients. (a) Les deux approximations comparées
à la valeur exacte, avec leurs nœuds de collocation. (b) Le résidu de Bellman
$R(x;\theta)$, qui s'annule aux nœuds. (c) L'échappement choisi par la
politique gloutonne de chaque approximation, comparé à l'échappement optimal.
:::

Les résidus du panneau (b) s'annulent aux nœuds, comme l'exige la collocation,
et tous deux culminent à environ $0.2$ entre ceux-ci. Les approximations
elles-mêmes diffèrent beaucoup plus. Un polynôme de degré 11 ne peut pas se
courber assez brusquement pour suivre la falaise, de sorte qu'il oscille
autour. Près de l'extrémité gauche, sa valeur pour un stock vide est $1.6$ au
lieu de zéro. La politique gloutonne hérite de ces oscillations (panneau (c)).
Elle épuise tout stock inférieur à $0.31$, au-delà du point de bascule, là où
la politique optimale ferme la pêcherie et laisse le stock se reconstituer. Un
stock de $0.31$ vaut $1.49$ sous la politique optimale et $0.31$, sa prise
immédiate, sous celle-ci. Entre $0.53$ et $0.67$, la politique pêche aussi des
stocks qu'il faudrait laisser croître. L'approximation linéaire par morceaux
est grossière entre ses nœuds, mais elle suit la falaise, et sa politique
gloutonne perd au plus $0.064$ où que ce soit. Ce pire cas se situe près de
$x = 0.21$, où la politique ferme la pêcherie un peu trop tôt.

L'exemple laisse trois questions ouvertes, et le reste du chapitre les traite
dans l'ordre. La collocation a exigé un résidu nul en douze points; où
d'autre pourrait-on rendre le résidu petit, et que décide ce choix? Les douze
coefficients ont été trouvés en répétant une mise à jour et un ajustement;
quand cette répétition converge-t-elle, et quoi d'autre pourrait les trouver?
Et le polynôme avait le plus petit résidu des deux approximations et une perte
de politique dix-huit fois plus grande; quelle grandeur doit servir à juger une
approximation?

- {download}`Modèle de la pêcherie, solveur de référence et collocation <code/fishery_tipping_point.py>`
- {download}`Générateur de l'expérience sur la pêcherie <scripts/build_fishery_tipping_point_artifacts.py>`
- {download}`Résultats sur la pêcherie pour chaque taille de base <artifacts/fishery_tipping_point/metrics.json>`

(sec-wr-framework)=
## D'une équation par état à $n$ équations

L'équation d'optimalité de Bellman $v = \Bellman v$ est vérifiée en chaque
état. Pour la pêcherie, cela fait une équation pour chaque stock de
$[0, 1.6]$, et l'inconnue est la fonction $v^\star$ tout entière. Remplacer $v$
par $v(\cdot;\theta)$ laisse $n$ inconnues, de sorte qu'on ne peut imposer que
$n$ conditions, et le choix de ces conditions constitue la méthode.

Les chapitres précédents ont fait la même démarche pour les trajectoires. Une
trajectoire en temps continu ne peut pas être choisie un instant à la fois;
la [collocation directe](continuous-time-collocation.md) représentait donc
l'état et la commande par des morceaux polynomiaux à nombre fini de
coefficients, imposait l'équation différentielle en des points choisis de
chaque intervalle, et remplaçait les intégrales du coût par une quadrature. La
fonction de valeur est traitée de la même manière : une base pour la fonction
inconnue, un nombre fini de conditions à la place de l'équation, et une
quadrature pour les espérances. La méthode des résidus pondérés
{cite}`Chakraverty2019,AtkinsonPotra1987` organise ces choix en cinq étapes.
Chaque étape ci-dessous est énoncée pour une équation de Bellman générale et
nomme les solutions de rechange à cet embranchement. La pêcherie a suivi un
chemin à travers les étapes, et un encadré après chaque étape reprend l'exemple
là où il a franchi cette étape; les étapes elles-mêmes n'ont besoin de rien
dans les encadrés.

### Étape 1 : choisir une fonction de valeur paramétrique

La fonction de valeur inconnue est remplacée par une famille $v(s;\theta)$
indexée par un vecteur de paramètres $\theta \in \mathbb{R}^n$. Rien dans les
cinq étapes n'exige que $\theta$ intervienne linéairement : un réseau de
neurones $v(s;\theta)$ à $n$ poids est une telle famille, et chaque condition
ci-dessous s'y applique. La linéarité compte pour la résolution. Lorsque la
famille est une combinaison linéaire de $n$ fonctions de base,

$$
v(s;\theta) = \sum_{j=1}^n \theta_j \varphi_j(s),
$$

chaque condition se réduit à de l'algèbre linéaire et peut être écrite
explicitement; ce chapitre emploie donc des familles linéaires pour rendre les
étapes explicites. Le
[chapitre suivant](approximate-bellman-equations.md) montre que le solveur qui
se transpose aux réseaux de neurones n'a besoin de rien de la représentation
au-delà de la possibilité de l'ajuster à des données.

Deux types de bases sont d'usage courant. Une base globale, comme les
polynômes de Tchebychev ou les modes de Fourier, donne à chaque coefficient une
influence sur tout l'espace d'états, et son erreur décroît exponentiellement en
$n$ lorsque la fonction est régulière. Une base locale, comme les fonctions
chapeau de l'interpolation linéaire par morceaux, les splines à petit support
ou les fonctions de base radiales, fait agir chaque coefficient près d'un nœud,
et son erreur décroît à un taux fixé par l'espacement des nœuds, quelle que
soit la régularité de la fonction.

::::{admonition} La pêcherie à l'étape 1
:class: note

La pêcherie a employé une base de chaque type, avec $n = 12$ : les polynômes de
Tchebychev $\varphi_j(x) = T_{j-1}(2x/1.6 - 1)$, et les fonctions chapeau de
douze nœuds uniformes, pour lesquelles $\theta_j$ est la valeur au nœud $j$. Sa
fonction de valeur présente un coude en $\hat x$ et une falaise au point de
bascule; c'est pourquoi les deux bases se sont comportées si différemment avec
le même $n$ : le polynôme a étalé la falaise sur tout l'intervalle, et les
fonctions chapeau l'ont confinée aux deux nœuds qui l'encadrent.
::::

### Étape 2 : définir le résidu de Bellman

Substituer l'approximation dans l'équation de Bellman laisse une fonction de
l'état,

$$
R(s;\theta) = (\Bellman v_\theta)(s) - v(s;\theta),
$$ (eq:bellman-residual)

le **résidu de Bellman**, où $v_\theta$ abrège $v(\cdot;\theta)$. Il mesure de
combien l'approximation ne satisfait pas l'équation en $s$, et il est la
fonction nulle exactement lorsque $v_\theta$ est un point fixe de $\Bellman$.
Une évaluation du résidu coûte une évaluation de l'opérateur de Bellman, soit
une maximisation sur les actions et une espérance sur le successeur, avec les
coefficients courants à l'intérieur des deux. Les coefficients interviennent
deux fois dans $R$, directement par $v(s;\theta)$ et de nouveau par l'espérance
à l'intérieur du maximum, et le maximum rend $R$ non linéaire en $\theta$ même
si $v_\theta$ est linéaire en $\theta$.

::::{admonition} La pêcherie à l'étape 2
:class: note

Avec le stock comme état et l'échappement comme action, le résidu est

$$
R(x;\theta) = \max_{0 \le s \le x}\Big\{x - s + \gamma\,\mathbb{E}_z\big[v(z f(s);\theta)\big]\Big\} - v(x;\theta),
$$

et une évaluation coûte une maximisation sur l'échappement et une quadrature à
21 nœuds sur le choc. Le panneau (b) de la {numref}`fig-fishery-collocation`
trace cette fonction pour les deux ajustements avec $n = 12$.
::::

(sec-wr-conditions)=
### Étape 3 : imposer $n$ conditions au résidu

Le résidu est une fonction et $\theta$ a $n$ composantes; la condition
$R(\cdot;\theta) = 0$ doit donc être remplacée par $n$ conditions scalaires.
Les choix possibles viennent de la manière dont on peut vérifier qu'une
fonction est nulle. Un vecteur $\mathbf{r} \in \mathbb{R}^m$ est nul si et
seulement si $\langle \mathbf{r}, \mathbf{y} \rangle = 0$ pour tout
$\mathbf{y}$, puisque $\mathbf{y} = \mathbf{r}$ donne $\|\mathbf{r}\|^2$. Il en
va de même pour les fonctions avec le produit scalaire pondéré

$$
\langle R, p \rangle_w = \int_{\mathcal{S}} R(s)\, p(s)\, w(s)\, ds,
$$

où $w > 0$ est un poids sur les états : $R$ est la fonction nulle si et
seulement si $\langle R, p \rangle_w = 0$ pour toute **fonction test** $p$.
Tester contre toutes les fonctions $p$ est aussi impossible que tester en
chaque état, mais tester contre $n$ fonctions test choisies $p_1, \ldots, p_n$
donne exactement $n$ équations,

$$
\langle R(\cdot;\theta), p_i \rangle_w = 0, \qquad i = 1, \ldots, n.
$$ (eq:weighted-residual-conditions)

Une méthode est définie par ses fonctions test, et trois choix de fonctions
test couvrent presque toutes les méthodes en usage.

| **Méthode** | **Fonctions test** $p_i$ | **Conditions** ($n$ équations) |
|:-----------|:--------------------------|:-------------------------------|
| Collocation | Masses ponctuelles en des états choisis $s_1, \ldots, s_n$ | $R(s_i; \theta) = 0$ |
| Galerkin | Les fonctions de base $\varphi_i$ | $\langle R(\cdot;\theta), \varphi_i \rangle_w = 0$ |
| Moindres carrés | Les dérivées $\partial R / \partial \theta_i$ du résidu | $\langle R(\cdot;\theta), \partial R/\partial\theta_i \rangle_w = 0$ |

#### Collocation : un résidu nul en des états choisis

La collocation est la méthode du
[chapitre sur la collocation en temps continu](continuous-time-collocation.md#sec-collocation-nodal-slopes),
appliquée à une autre équation. Là, chaque morceau d'une trajectoire était
stocké par ses valeurs aux nœuds, dans une base de Lagrange dont les fonctions
cardinales valent un à leur propre nœud et zéro aux autres, et l'équation
différentielle n'était imposée qu'aux points de collocation : en chacun
d'eux, la pente du polynôme d'interpolation devait égaler la pente $\mathbf f$
prescrite par la dynamique. Entre les points, rien n'était imposé. Ici,
l'inconnue est la fonction de valeur plutôt qu'une trajectoire, et l'équation
est l'équation de Bellman plutôt que $\dot{\mathbf x} = \mathbf f$. La
collocation choisit $n$ états $s_1, \ldots, s_n$ et exige que le résidu s'y
annule,

$$
R(s_i;\theta) = 0, \qquad i = 1, \ldots, n .
$$

Avec une base nodale, polynômes de Lagrange passant par les points de
collocation ou fonctions chapeau de l'interpolation linéaire par morceaux, le
coefficient $\theta_i$ est la valeur en $s_i$, et la condition s'écrit
$\theta_i = (\Bellman v_\theta)(s_i)$. La valeur stockée à chaque nœud doit
égaler la mise à jour de Bellman calculée à partir de la fonction interpolée,
comme la pente stockée à chaque nœud devait égaler la pente donnée par la
dynamique. La récurrence à rebours interpolée {eq}`eq:interpolated-bellman` du
chapitre sur l'horizon fini imposait la même condition, un stade à la fois.

Dans le langage des fonctions test, une masse ponctuelle en $s_i$ extrait la
valeur $R(s_i)$; la collocation est donc {eq}`eq:weighted-residual-conditions`
avec des masses ponctuelles comme fonctions test, comme le précise la
formulation par appariement dual de {cite:t}`LegrandJunca2025`. Aucune
intégrale sur les états n'apparaît, de sorte que chaque condition coûte une
évaluation de l'opérateur de Bellman. Les conditions annulent le résidu aux
nœuds et le laissent libre entre eux, comme le résidu de l'équation
différentielle est libre entre les points de collocation d'une trajectoire. Le nom
le dit : collocation vient du latin *collocare*, placer ensemble, et
l'approximation et l'équation exacte sont placées ensemble en un ensemble fini
de points, des instants pour une trajectoire et des états pour une fonction
de valeur.

Où placer les points de collocation est un choix à part entière. Pour un
polynôme de degré $n - 1$, le choix habituel est l'ensemble des zéros du
$n$-ième polynôme de Tchebychev, les **nœuds de Tchebychev**, et la
combinaison s'appelle **collocation orthogonale**. Ces mêmes nœuds servent de
nœuds à la quadrature de Gauss-Tchebychev, la matrice d'interpolation
$\Phi_{ij} = \varphi_j(s_i)$ construite sur eux est bien conditionnée, et le
polynôme qui passe par eux est à un petit facteur près de la meilleure
approximation polynomiale d'une fonction régulière, de sorte qu'un résidu qui
s'y annule reste petit entre eux. Des nœuds équidistants n'ont aucune de ces
propriétés dès que le degré est élevé. Pour une base nodale, les points de
collocation sont les nœuds de la base elle-même, où une fonction cardinale vaut
un et les autres s'annulent, de sorte que la matrice d'interpolation est
l'identité et que les coefficients sont les valeurs aux nœuds.

Notons $\Proj$ l'interpolation par les points de collocation, l'application
qui envoie une fonction $g$ sur l'élément de l'espace engendré tel que
$(\Proj g)(s_i) = g(s_i)$ en chaque nœud. Les conditions de collocation disent
que $v_\theta$ et $\Bellman v_\theta$ coïncident aux nœuds, c'est-à-dire

$$
v_\theta = \Proj\, \Bellman v_\theta .
$$ (eq:projected-bellman)

La solution approchée est un point fixe de l'opérateur de Bellman suivi d'une
projection, et c'est cette forme que l'étape 4 exploite.

::::{admonition} La pêcherie à l'étape 3
:class: note

La pêcherie a employé la collocation, avec les deux choix habituels de nœuds :
les nœuds de Tchebychev pour le polynôme, et les nœuds d'interpolation pour les
fonctions chapeau. Le panneau (b) de la {numref}`fig-fishery-collocation`
montre les résidus des deux ajustements : nuls aux douze nœuds de chaque base,
et d'environ $0.2$ à leurs pics entre les nœuds, pour les deux bases.
::::

#### Galerkin : un résidu orthogonal à la base

La méthode de Galerkin prend les fonctions de base elles-mêmes comme fonctions
test, $\langle R(\cdot;\theta), \varphi_i \rangle_w = 0$ pour
$i = 1, \ldots, n$. Par linéarité, le résidu est alors orthogonal à toute
fonction de l'espace d'approximation
$\mathcal{V} = \text{Vect}\{\varphi_1, \ldots, \varphi_n\}$,

$$
\Big\langle R, \sum_{i=1}^n c_i \varphi_i \Big\rangle_w = \sum_{i=1}^n c_i \langle R, \varphi_i \rangle_w = 0 \quad \text{pour tout } c \in \mathbb{R}^n,
$$

ce qui est la propriété qui définit une projection orthogonale : la composante
de $\Bellman v_\theta$ qui se trouve dans $\mathcal{V}$ est $v_\theta$
lui-même. Pour approcher un vecteur de $\mathbb{R}^3$ par un vecteur du plan
$xy$, on garde $(v_1, v_2, 0)$, et l'erreur $(0, 0, v_3)$ est orthogonale au
plan. Avec $\Proj$ désignant maintenant la projection $w$-orthogonale sur
$\mathcal{V}$, les conditions de Galerkin s'écrivent de nouveau
$v_\theta = \Proj\,\Bellman v_\theta$, l'équation de Bellman projetée
{eq}`eq:projected-bellman` avec un autre $\Proj$.

Les conditions sont des intégrales sur l'espace d'états; la méthode de
Galerkin a donc besoin d'une règle de quadrature pour elles, en plus de la
quadrature à l'intérieur de $\Bellman$. Avec une base de polynômes orthogonaux
et le poids qui lui correspond, les produits scalaires entre fonctions de base
sont connus en forme close et les systèmes sont bien conditionnés. Lorsque les
états sont plutôt échantillonnés selon $w$, les intégrales deviennent des
moyennes empiriques et les conditions de Galerkin deviennent les équations
normales d'une régression par moindres carrés : $v_\theta$ est l'ajustement par
moindres carrés de la base aux mises à jour $(\Bellman v_\theta)(s_m)$ aux
états échantillonnés. Les cibles de cette régression dépendent de sa propre
solution, et l'étape 4 brise cette circularité.

#### Moindres carrés : le plus petit résidu

Le choix le plus direct minimise la norme pondérée du résidu,

$$
\min_\theta \int_{\mathcal{S}} R(s; \theta)^2\, w(s)\, ds,
$$

dont les conditions du premier ordre sont {eq}`eq:weighted-residual-conditions`
avec les fonctions test $p_i = \partial R / \partial \theta_i$. Pour le résidu
de Bellman, il s'agit de la **minimisation du résidu de Bellman**. Ses
fonctions test,

$$
\frac{\partial R(s;\theta)}{\partial \theta_i} = \frac{\partial (\Bellman v_\theta)(s)}{\partial \theta_i} - \varphi_i(s),
$$

contiennent la dérivée de l'opérateur de Bellman par rapport aux
coefficients, dont la méthode de Galerkin et la collocation n'ont jamais
besoin. La calculer exige de dériver à travers le maximum, ce que l'étape 4
fait avec le théorème de l'enveloppe. La solution diffère aussi de la solution
de Galerkin, parce que le résidu est rendu orthogonal à un autre ensemble de
fonctions, et elle n'est pas une équation de Bellman projetée. Le
[chapitre suivant](approximate-bellman-equations.md) revient sur la différence
entre ajuster une mise à jour et minimiser le résidu, qui sépare les
algorithmes employés en pratique de celui-ci.

```{admonition} Pourquoi tester contre des fonctions suffit
:class: dropdown

L'argument constructif ci-dessus, qui choisit $p = R$, exige que $R$ soit une
fonction de norme finie. L'énoncé général, selon lequel un élément non nul d'un
espace normé est détecté par une forme linéaire continue, est une conséquence
du **théorème de Hahn-Banach**, souvent formulé ainsi : « le dual sépare les
points ». C'est aussi ce qui fait de la méthode de Galerkin une minimisation à
part entière. La norme duale de $R$ par rapport à $\mathcal{V}$,

$$
\lVert R \rVert_{\mathcal{V}^*} = \sup_{g \in \mathcal{V},\, \lVert g \rVert_w = 1} \lvert \langle R, g \rangle_w \rvert,
$$

mesure le résidu par son plus grand produit scalaire avec l'espace
d'approximation, et les conditions de Galerkin l'annulent. Les masses
ponctuelles ne sont pas des fonctions mais des distributions, et la formulation
par appariement dual de {cite:t}`LegrandJunca2025` traite la collocation et la
méthode de Galerkin de façon uniforme : les objets test agissent sur le résidu
par un appariement, et la norme induite mesure un résidu par son pire cas sur
les objets test. Le même point de vue motive des formulations adverses
{cite}`Zang2020` dans lesquelles l'approximation et les fonctions test sont
toutes deux apprises.
```

(sec-wr-solve)=
### Étape 4 : résoudre les $n$ équations

Les conditions de l'étape 3 forment $n$ équations à $n$ inconnues,
$G(\theta) = 0$ avec $G_i(\theta) = \langle R(\cdot;\theta), p_i \rangle_w$, ou
$G_i(\theta) = R(s_i;\theta)$ pour la collocation. Elles sont non linéaires,
parce que le maximum dans l'opérateur de Bellman l'est. Le
[chapitre sur l'horizon infini](infinite-horizon-mdps.md) a résolu l'équation
de Bellman exacte avec deux méthodes générales pour les équations
d'opérateurs : les approximations successives, qui ont donné l'itération sur
les valeurs, et la méthode de Newton-Kantorovitch, qui a donné l'itération sur
les politiques. Chacune a un pendant pour les $n$ équations.

(sec-wr-function-iteration)=
#### Itération fonctionnelle

Pour la collocation et la méthode de Galerkin, les conditions ont la forme de
point fixe $v_\theta = \Proj\,\Bellman v_\theta$ de
{eq}`eq:projected-bellman`, et le solveur naturel répète l'application,

$$
v_{k+1} = \Proj\, \Bellman v_k,
$$ (eq:projected-vi)

soit l'itération sur les valeurs avec une projection après chaque application
de l'opérateur. En coefficients, une itération de la version par collocation
compte deux étapes. La première évalue l'opérateur aux nœuds avec la fonction
courante, $t_i = (\Bellman v_k)(s_i)$ pour $i = 1, \ldots, n$. La seconde
choisit les coefficients qui reproduisent ces $n$ cibles, ce qui est le système
linéaire $\boldsymbol{\Phi}\theta_{k+1} = t$ avec la matrice d'interpolation
$\Phi_{ij} = \varphi_j(s_i)$. Avec des fonctions chapeau collocalisées à leurs
propres nœuds, la résolution est triviale, parce que $\boldsymbol\Phi$ est
l'identité. La version de Galerkin remplace la seconde étape par la projection
de la fonction cible, le système linéaire $M\theta_{k+1} = b_k$ avec la matrice
de masse $M_{ij} = \langle \varphi_i, \varphi_j \rangle_w$ et
$b_i = \langle \Bellman v_k, \varphi_i \rangle_w$. Le système est linéaire
quel que soit l'opérateur, parce que la cible $\Bellman v_k$ est maintenue fixe
pendant qu'on choisit $\theta_{k+1}$. C'est l'ajustement par moindres carrés
pondérés de la base à la cible, et avec des états échantillonnés, c'est une
régression ordinaire des mises à jour sur la base. La circularité de l'étape 3
disparaît : les cibles viennent de l'itéré précédent. Ajuster une fonction
paramétrique à des mises à jour de Bellman, puis recalculer les mises à jour,
est l'algorithme que le [chapitre suivant](approximate-bellman-equations.md)
appelle itération sur les valeurs ajustée et généralise à tout régresseur.

La convergence de l'itération est une question distincte. L'opérateur de
Bellman est une $\gamma$-contraction pour la norme sup, de sorte que
l'itération sur les valeurs converge à partir de tout point de départ
({prf:ref}`value-iteration-convergence`). Si $\Proj$ n'augmente pas les
distances en norme sup, la composition ne les augmente pas non plus, et
{eq}`eq:projected-vi` converge vers l'unique solution de l'équation de Bellman
projetée, l'erreur diminuant d'un facteur $\gamma$ par itération. Que $\Proj$
étire ou non les distances en norme sup dépend du signe de ses poids. Un
opérateur d'interpolation à fonctions cardinales non négatives renvoie une
combinaison convexe des cibles, qui ne peut pas les dépasser; c'est la
condition $\Lambda = 1$ de {prf:ref}`prop-interp-error-accumulation`, et un tel
$\Proj$ ne peut pas étirer les distances. L'interpolation polynomiale par les
nœuds de Tchebychev a des fonctions cardinales qui passent sous zéro entre les
nœuds; elle peut donc amplifier une différence entre deux vecteurs de cibles,
et l'itération peut alors diverger même si chaque itération reproduit
exactement ses cibles. Le [chapitre suivant](approximate-bellman-equations.md)
démontre que le premier type d'ajustement préserve toujours la contraction et
montre ce qui se passe avec le second. Même lorsqu'elle converge, l'itération
fonctionnelle est lente lorsque $\gamma$ est proche de un : l'erreur diminue
d'un facteur $\gamma$ par itération, de sorte que la réduire d'un facteur
$10^{-10}$ avec $\gamma = 0.95$ demande environ 450 itérations.

::::{admonition} La pêcherie à l'étape 4
:class: note

La {numref}`fig-fishery-sweep` répète l'expérience de collocation pour $n$
allant de 4 à 64, avec l'itération fonctionnelle comme solveur.

:::{figure} _static/fishery_tipping_point/sweep.svg
:label: fig-fishery-sweep

Collocation par itération fonctionnelle pour $n$ allant de 4 à 64. (a) Plus
grande erreur de la valeur approchée. (b) Plus grande perte de sa politique
gloutonne, évaluée exactement. Les croix marquent les $n$ pour lesquels
l'itération avec le polynôme a divergé.
:::

Avec les fonctions chapeau, l'itération a convergé pour tout $n$, en environ
410 itérations. Avec le polynôme, elle a divergé pour $n = 4$ et pour
$n = 9, 10, 11$ : les coefficients ont crû sans borne, même si chaque
itération reproduisait exactement ses cibles. La mise à jour réduisait encore
la distance entre deux fonctions de valeur d'un facteur $\gamma$; l'étape
d'interpolation l'étirait, assez pour annuler la contraction.

Le balayage montre aussi comment les deux bases dépensent leur précision.
L'interpolation linéaire par morceaux s'améliore régulièrement à mesure qu'on
ajoute des nœuds, parce que chaque nœud résout la valeur localement. Le
polynôme est erratique : ajouter un coefficient peut multiplier son erreur par
cinq, et la falaise au point de bascule est étalée sur tout l'intervalle
jusqu'à ce que le degré soit assez élevé pour la résoudre. À partir d'environ
40 coefficients, il dépasse les fonctions chapeau, et son erreur diminue
ensuite beaucoup plus vite avec $n$. La meilleure base dépend donc de $n$ et
de la forme de la solution.
::::

(sec-wr-newton)=
#### Méthode de Newton

La solution de rechange traite $G(\theta) = 0$ comme un problème de recherche
de zéros et le linéarise,

$$
\theta_{k+1} = \theta_k - J_G(\theta_k)^{-1} G(\theta_k),
$$

où $J_G$ est la jacobienne de $G$. Pour la collocation,
$G_i(\theta) = v(s_i;\theta) - (\Bellman v_\theta)(s_i)$, et les entrées de la
jacobienne sont

$$
\frac{\partial G_i}{\partial \theta_j} = \varphi_j(s_i) - \frac{\partial (\Bellman v_\theta)(s_i)}{\partial \theta_j}.
$$

Le premier terme est une fonction de base en un nœud. Le second est la dérivée
de l'opérateur de Bellman par rapport à un coefficient, la grandeur dont les
conditions des moindres carrés de l'étape 3 avaient aussi besoin, et il
contient un maximum. Le maximum n'est pas dérivable là où deux actions sont à
égalité, mais partout où le maximiseur est unique, sa dérivée est donnée par le
théorème de l'enveloppe.

*Avant de poursuivre, dérivez $h(\theta) = \max_a g(a, \theta)$ par la règle de
dérivation des fonctions composées, en notant $a^*(\theta)$ le maximiseur. Quel
terme fait intervenir $\partial a^*/\partial\theta$, et pourquoi s'annule-t-il?*

```{prf:theorem} Théorème de l'enveloppe
:label: thm-envelope

Soit $g(a, \theta)$ continûment dérivable par rapport à ses deux arguments, et
supposons que $h(\theta) = \max_a g(a, \theta)$ soit atteint en un unique
maximiseur intérieur $a^*(\theta)$ qui dépend de $\theta$ de façon dérivable.
Alors

$$
\nabla_\theta h(\theta) = \nabla_\theta g(a, \theta)\big|_{a = a^*(\theta)} .
$$

La dérivée du maximum est la dérivée partielle de l'objectif au maximiseur, le
maximiseur étant maintenu fixe.
```

```{prf:proof}
Par la règle de dérivation des fonctions composées,
$\nabla_\theta h = \nabla_\theta g(a^*(\theta), \theta) + \big(\nabla_a g(a^*(\theta), \theta)\big)^{\!\top} \partial a^*/\partial\theta$,
et le second terme s'annule parce que $\nabla_a g = 0$ en un maximiseur intérieur.
```

Pour l'opérateur de Bellman,
$(\Bellman v_\theta)(s) = \max_a \{ r(s,a) + \gamma\, \mathbb{E}[v(s';\theta) \mid s, a] \}$,
l'objectif est linéaire en $\theta$ avec les coefficients
$\gamma\,\mathbb{E}[\varphi_j(s') \mid s, a]$, de sorte que

$$
\frac{\partial (\Bellman v_\theta)(s)}{\partial \theta_j} = \gamma\, \mathbb{E}\big[\varphi_j(s') \mid s, a^*(s;\theta)\big],
$$ (eq:envelope-bellman)

où $a^*(s;\theta)$ est l'action gloutonne en $s$ pour les coefficients
courants. La dérivée est l'espérance de la fonction de base au successeur sous
l'action gloutonne, et la calculer coûte une quadrature de plus par nœud et par
fonction de base. La même formule vaut dans le cas des actions discrètes, et
sur un ensemble fini d'actions, le maximiseur est unique sauf en cas
d'égalité.

Substituer {eq}`eq:envelope-bellman` dans le pas de Newton lui donne une forme
familière. Soit $\pi_k$ la politique gloutonne pour $v_{\theta_k}$, soit
$\mathbf{r}_{\pi_k}$ le vecteur de ses récompenses aux nœuds, et soit
$\mathbf{P}_{\pi_k}\boldsymbol{\Phi}$ la matrice d'entrées
$\mathbb{E}[\varphi_j(s') \mid s_i, \pi_k(s_i)]$. Alors
$J_G = \boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi_k}\boldsymbol{\Phi}$ et
$G(\theta_k) = \boldsymbol\Phi\theta_k - \mathbf{r}_{\pi_k} - \gamma \mathbf{P}_{\pi_k}\boldsymbol\Phi\theta_k$,
et la mise à jour de Newton $J_G \theta_{k+1} = J_G\theta_k - G(\theta_k)$
se simplifie en

$$
(\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi_k}\boldsymbol{\Phi})\, \theta_{k+1} = \mathbf{r}_{\pi_k}.
$$ (eq:collocation-newton)

C'est l'équation d'évaluation de la politique $\pi_k$, imposée aux nœuds de
collocation et résolue dans l'espace engendré par la base. La méthode de
Newton appliquée aux équations de collocation alterne une amélioration
gloutonne et une évaluation approchée, exactement comme la méthode de Newton
appliquée à l'équation de Bellman exacte est l'itération sur les politiques
({ref}`sec-newton-kantorovich-pi`). Avec l'opérateur de Bellman lisse du
[chapitre sur la régularisation](regularized-dp.md), le maximum est remplacé
par un log-sum-exp, l'opérateur est dérivable partout, et le même pas vaut avec
la politique softmax à la place de la politique gloutonne
({ref}`sec-smooth-pi-newton`).

La méthode de Newton converge quadratiquement près de la solution, en quelques
itérations au lieu des centaines de l'itération fonctionnelle, et elle n'exige
pas que $\Proj\,\Bellman$ soit une contraction, puisqu'elle n'itère jamais
cette application. Elle le paie de deux façons. Chaque itération forme et
résout le système jacobien, dont les entrées exigent les espérances de chaque
fonction de base en chaque nœud. Et elle dépend du point de départ : lancée
loin de la solution, elle peut diverger, ou converger vers un point fixe que
les $n$ équations possèdent et que l'équation de Bellman ne possède pas.

| **Méthode** | **Convergence** | **Coût par itération** | **Exige** |
|:-----------|:----------------|:-----------------------|:----------|
| Itération fonctionnelle | Linéaire, facteur $\gamma$ | $n$ mises à jour et une résolution linéaire avec une matrice fixe | Que $\Proj\,\Bellman$ soit une contraction |
| Méthode de Newton | Quadratique près de la solution | $n$ mises à jour, $n^2$ espérances et une nouvelle résolution linéaire | Un bon point de départ |

Les deux sont complémentaires. Lorsque l'ajustement est monotone et que
$\gamma$ est modéré, l'itération fonctionnelle est simple et robuste. Lorsque
$\gamma$ est proche de un ou que la base est un polynôme de degré élevé, une
approche hybride fonctionne bien : quelques itérations fonctionnelles avec une
approximation grossière et monotone pour atteindre le bassin d'attraction, puis
la méthode de Newton avec la base précise. C'est ainsi que Rust a résolu son
modèle de moteurs d'autobus ({ref}`sec-rust-inverse`), et le
[chapitre suivant](approximate-bellman-equations.md) développe le volet Newton
sous la forme de l'itération sur les politiques approchée.

### Étape 5 : vérifier la solution

Les conditions de l'étape 3 ne contrôlent le résidu que là où elles ont été
imposées, aux nœuds de collocation ou contre les fonctions test choisies. La
première vérification consiste donc à évaluer le résidu là où il n'a pas été
contrôlé : en de nombreux états entre les nœuds, avec une quadrature plus fine
à l'intérieur de l'opérateur que celle employée pour l'ajustement, et contre
des fonctions de base d'ordre supérieur dans le cas de Galerkin.

Un petit résidu partout borne bel et bien l'erreur de valeur. Puisque
$v_\theta - v^\star = (v_\theta - \Bellman v_\theta) + (\Bellman v_\theta - \Bellman v^\star)$
et que $\Bellman$ est une $\gamma$-contraction,

$$
\|v_\theta - v^\star\|_\infty \le \frac{\|R(\cdot;\theta)\|_\infty}{1 - \gamma},
$$

où la norme de $R$ est prise sur tous les états, et non sur les nœuds. La borne
est lâche en général, et ni le résidu ni l'erreur de valeur ne sont ce qu'on
utilise. C'est la politique gloutonne qu'on utilise, et sa perte est la
différence entre la valeur optimale et la valeur obtenue en agissant selon
l'approximation. Les trois grandeurs répondent à des questions différentes. Le
résidu est ce que la méthode contrôle. L'erreur de valeur est ce que le résidu
borne, par l'entremise de $1/(1-\gamma)$. La perte de la politique est ce qui
compte, et elle ne dépend de l'erreur de valeur que par les différences entre
actions; une approximation peut se tromper sur ces différences alors que ses
valeurs sont précises, et les obtenir justes alors que ses valeurs ne le sont
pas. Le
[chapitre suivant](approximate-bellman-equations.md) borne la perte de la
politique par l'erreur de valeur et met les deux bornes à l'épreuve sur un
problème qu'un organisme gouvernemental résout chaque année.

::::{admonition} La pêcherie à l'étape 5
:class: note

Le panneau (b) de la {numref}`fig-fishery-collocation` constitue la
vérification du résidu, et il trouve des résidus d'environ $0.2$ entre les
nœuds pour les deux bases. La borne sur l'erreur de valeur est alors
$0.2 / 0.05 = 4$, à comparer à l'étendue complète de $v^\star$, qui est de
$4.5$. Les erreurs de valeur réelles sont de $1.66$ pour le polynôme et de
$0.52$ pour les fonctions chapeau; la borne est donc lâche, et elle classe les
deux approximations dans le mauvais ordre : le polynôme a le plus petit
résidu.

:::{include} artifacts/fishery_tipping_point/results.md
:::

Le tableau ajoute la perte de la politique. Pour $n = 12$, le polynôme a un
résidu de $0.18$ contre $0.21$ pour les fonctions chapeau, et une perte de
politique dix-huit fois plus grande, parce que ses oscillations déplacent la
frontière de décision au-delà du point de bascule (panneau (c) de la
{numref}`fig-fishery-collocation`).
::::

## Exercices

Les exercices emploient la notation des cinq étapes. Les premiers sont de
courtes vérifications des définitions; la plupart des autres peuvent se faire
à la main, et {ref}`ex-wr-fishery-newton` exige le module de la pêcherie qui a
produit les figures.

:::{exercise} Condition d'orthogonalité
:label: ex-projection-check-1

Dans une méthode de Galerkin à fonctions de base $\varphi_i$, à quelles fonctions le résidu de Bellman doit-il être orthogonal, et qu'est-ce que cela dit de $\Proj\,\Bellman v_\theta$?
:::

:::{solution} ex-projection-check-1
:class: dropdown

Aux fonctions de base elles-mêmes : $\langle \Bellman v_\theta - v_\theta, \varphi_i\rangle_w = 0$ pour tout $i$. Le résidu est alors orthogonal à tout l'espace engendré, de sorte que la projection $w$-orthogonale de $\Bellman v_\theta$ sur cet espace est $v_\theta$ lui-même.
:::

:::{exercise} Collocation ou moindres carrés
:label: ex-projection-check-2

Quelle est la principale distinction entre la collocation et l'ajustement du résidu par moindres carrés, et laquelle des deux a besoin du théorème de l'enveloppe?
:::

:::{solution} ex-projection-check-2
:class: dropdown

La collocation force le résidu à s'annuler en des états choisis. Les moindres carrés minimisent le carré pondéré du résidu sur l'espace d'états. Les moindres carrés ont besoin du théorème de l'enveloppe, parce que leurs conditions du premier ordre contiennent la dérivée de l'opérateur de Bellman par rapport aux coefficients; la collocation n'en a besoin que si les équations de collocation sont résolues par la méthode de Newton.
:::

:::{exercise} Trois mesures de qualité
:label: ex-projection-check-6

Dans la pêcherie avec $n = 12$, le polynôme a le plus petit résidu et la plus grande perte de politique. Parmi le résidu, l'erreur de valeur et la perte de la politique, laquelle la collocation contrôle-t-elle, laquelle borne laquelle, et pourquoi le classement peut-il différer?
:::

:::{solution} ex-projection-check-6
:class: dropdown

La collocation ne contrôle le résidu qu'aux nœuds. La norme sup du résidu sur tous les états borne l'erreur de valeur par l'entremise de $1/(1-\gamma)$, et l'erreur de valeur borne la perte de la politique (chapitre suivant). Les deux bornes sont lâches, et la politique dépend des différences entre les valeurs des actions plutôt que des valeurs elles-mêmes; une approximation au résidu plus petit peut donc placer une frontière de décision dans la mauvaise région, comme le fait le polynôme de part et d'autre du point de bascule.
:::

:::{exercise} Trois méthodes sur une même équation
:label: ex-wr-three-methods

Considérons l'évaluation de politique sur $\mathcal{S} = [0, 1]$ avec la
récompense $r(x) = x$, la transition déterministe $x' = x/2$ et
$\gamma = 1/2$, de sorte que l'équation de Bellman est
$v(x) = x + \tfrac12 v(x/2)$.

1. Vérifiez que $v(x) = \tfrac43 x$ en est solution.
2. Approchez $v$ par $\hat v(x;\theta) = \theta_1 + \theta_2 x^2$, une base qui
   ne peut pas représenter la solution. Écrivez le résidu $R(x;\theta)$.
3. Faites la collocation en $x = 0$ et $x = 1$. Trouvez $\theta$.
4. Imposez les conditions de Galerkin avec le poids uniforme sur $[0, 1]$.
   Trouvez $\theta$.
5. Comparez les deux approximations par leur plus grand résidu et leur plus
   grande erreur de valeur sur $[0, 1]$. Laquelle est la meilleure selon
   chaque mesure?
:::

:::{solution} ex-wr-three-methods
:class: dropdown

1. $\tfrac43 x = x + \tfrac12 \cdot \tfrac43 \cdot \tfrac x2 = x + \tfrac13 x$.
2. $R(x;\theta) = x + \tfrac12(\theta_1 + \theta_2 x^2/4) - \theta_1 - \theta_2 x^2 = x - \tfrac12\theta_1 - \tfrac78\theta_2 x^2$.
3. $R(0) = -\tfrac12\theta_1 = 0$ donne $\theta_1 = 0$, et $R(1) = 1 - \tfrac78\theta_2 = 0$ donne $\theta_2 = \tfrac87$.
4. Les conditions $\int_0^1 R\,dx = \tfrac12 - \tfrac12\theta_1 - \tfrac{7}{24}\theta_2 = 0$ et $\int_0^1 R\,x^2\,dx = \tfrac14 - \tfrac16\theta_1 - \tfrac{7}{40}\theta_2 = 0$ donnent $\theta_1 = \tfrac38$ et $\theta_2 = \tfrac{15}{14}$.
5. Collocation : plus grand résidu $0.25$, plus grande erreur de valeur $0.389$ (près de $x = 0.58$). Galerkin : plus grand résidu $0.1875$, plus grande erreur de valeur $0.375$ (en $x = 0$). La méthode de Galerkin est ici légèrement meilleure selon les deux mesures. Les deux sont médiocres, parce qu'une parabole ne peut pas suivre une droite passant par l'origine et les extrémités : aucun choix de conditions ne répare une base qui ne peut pas représenter la solution. Les moindres carrés donnent par hasard la réponse de Galerkin dans ce problème, parce que le résidu est linéaire en $\theta$ et que ses dérivées $\partial R/\partial\theta_i$ valent $-\tfrac12$ et $-\tfrac78 x^2$, qui engendrent le même espace que la base.
:::

:::{exercise} Quelles itérations de collocation sont contractantes
:label: ex-wr-cardinal

Prenez la base de {ref}`ex-wr-three-methods` et les points de collocation $0$
et $1$.

1. Écrivez les fonctions cardinales $\ell_0, \ell_1$ de l'interpolation dans
   $\text{Vect}\{1, x^2\}$ par ces points, de sorte que
   $(\Proj g)(x) = \ell_0(x) g(0) + \ell_1(x) g(1)$.
2. Montrez qu'elles sont non négatives et de somme un sur $[0, 1]$. Qu'est-ce
   que cela implique pour l'itération fonctionnelle $v_{k+1} = \Proj\Bellman v_k$
   sur ce problème, et pour tout opérateur de Bellman qui est une contraction
   en norme sup?
3. Interpolez maintenant par le polynôme quadratique passant par $0$,
   $\tfrac12$ et $1$. Calculez la constante de Lebesgue
   $\Lambda = \max_x \sum_j |\ell_j(x)|$ et expliquez pourquoi l'argument de la
   partie 2 ne s'applique plus. C'est la constante de
   {prf:ref}`prop-interp-error-accumulation` dans le chapitre sur l'horizon
   fini.
:::

:::{solution} ex-wr-cardinal
:class: dropdown

1. $\ell_0(x) = 1 - x^2$ et $\ell_1(x) = x^2$.
2. Les deux sont dans $[0, 1]$ sur $[0, 1]$ et ont pour somme un; $\Proj g$ est donc une combinaison convexe de $g(0)$ et de $g(1)$ en tout $x$, et $\|\Proj f - \Proj g\|_\infty \le \|f - g\|_\infty$. La composition $\Proj\Bellman$ est alors une $\gamma$-contraction pour toute $\gamma$-contraction $\Bellman$, et l'itération fonctionnelle converge à partir de tout point de départ. Sur ce problème, elle converge vers la solution de collocation de {ref}`ex-wr-three-methods`.
3. Les fonctions cardinales sont $2(x - \tfrac12)(x-1)$, $-4x(x-1)$ et $2x(x-\tfrac12)$. La première et la dernière sont négatives sur certaines parties de $[0, 1]$, de sorte que $\Lambda = 1.25 > 1$. Une différence entre deux vecteurs de cibles peut être amplifiée d'un facteur allant jusqu'à $1.25$, ce qui l'emporte sur un facteur de contraction $\gamma > 0.8$; la non-expansivité, l'étape dont l'argument a besoin, n'est plus vérifiée.
:::

:::{exercise} La méthode de Newton en cas d'égalité
:label: ex-wr-envelope-tie

Soit $h(\theta) = \max\{\theta, 1 - \theta\}$, le maximum sur deux actions
dont les valeurs dépendent d'un seul coefficient.

1. Où $h$ est-elle dérivable, et que donne le théorème de l'enveloppe à ces
   endroits?
2. En $\theta = \tfrac12$, les deux actions sont à égalité. Quelle hypothèse
   de {prf:ref}`thm-envelope` n'est pas satisfaite? Toute pente comprise entre
   $-1$ et $1$ est une linéarisation valide à cet endroit; montrez que chaque
   choix correspond à l'évaluation de l'une des deux actions, ou d'une
   randomisation entre elles.
3. Expliquez pourquoi la méthode de Newton appliquée aux équations de
   collocation, qui est l'itération sur les politiques dans l'espace des
   coefficients ({eq}`eq:collocation-newton`), fonctionne encore lorsque la
   politique gloutonne présente des égalités en certains nœuds, et ce que
   change l'opérateur de Bellman lisse du chapitre sur la régularisation.
:::

:::{solution} ex-wr-envelope-tie
:class: dropdown

1. Partout sauf en $\theta = \tfrac12$. Le théorème de l'enveloppe donne la dérivée de l'action active : $1$ pour $\theta > \tfrac12$ et $-1$ pour $\theta < \tfrac12$.
2. Le maximiseur n'est pas unique. Les dérivées unilatérales valent $-1$ et $1$, et toute pente $\lambda \cdot 1 + (1-\lambda)(-1)$ avec $\lambda \in [0, 1]$ est un sous-gradient : c'est la dérivée de la valeur du choix randomisé qui prend la première action avec probabilité $\lambda$.
3. Une égalité signifie que deux actions ont la même valeur en ce nœud; évaluer l'une ou l'autre donne donc un pas de Newton valide : le pas résout l'équation d'évaluation d'une certaine politique gloutonne, et l'itération sur les politiques converge quelle que soit l'action gloutonne qu'elle choisit. L'opérateur de Bellman lisse remplace le maximum par un log-sum-exp, qui est dérivable partout; sa jacobienne emploie la politique softmax, qui répartit la probabilité entre les actions à égalité, et aucune règle de départage n'est nécessaire.
:::

:::{exercise} La méthode de Newton sur la pêcherie
:label: ex-wr-fishery-newton

L'itération fonctionnelle avec les douze fonctions chapeau demande environ 410
itérations sur la pêcherie. Implémentez le pas de Newton
{eq}`eq:collocation-newton` pour la même base, en employant `Bellman`,
`HatBasis` et `growth` de `code/fishery_tipping_point.py`, et comptez les
itérations.

1. Pour les coefficients courants, trouvez l'échappement glouton en chaque
   nœud. `Bellman.greedy` le renvoie sur la grille fine; lisez la valeur au
   point de la grille où tombe chaque nœud.
2. Construisez la matrice $\mathbf{P}_{\pi}\boldsymbol{\Phi}$ d'entrées
   $\mathbb{E}_z[\varphi_j(z f(s_i))]$, en employant les nœuds de quadrature
   `bellman.z` et les poids `bellman.w`. Pour les fonctions chapeau,
   $\boldsymbol\Phi$ est l'identité.
3. Résolvez $(\mathbf{I} - \gamma \mathbf{P}_{\pi}\boldsymbol{\Phi})\theta = \mathbf{r}_\pi$
   avec $\mathbf{r}_\pi$ les récoltes $x_i - s_i$ aux nœuds, et répétez.
4. Vérifiez que le résultat concorde avec le point fixe de l'itération
   fonctionnelle. Pourquoi le doit-il?
:::

:::{solution} ex-wr-fishery-newton
:class: dropdown

```python
import numpy as np
from fishery_tipping_point import Bellman, HatBasis, growth, X_MAX

bellman, basis = Bellman(), HatBasis(12)
nodes, gamma = basis.nodes, bellman.gamma

def hats(points):                                   # phi_j(points), one column per node
    return np.stack([np.interp(points, nodes, np.eye(basis.n)[j]) for j in range(basis.n)], axis=-1)

theta = np.zeros(basis.n)
for k in range(30):
    escapement = bellman.greedy(basis.function(theta))
    index = np.clip(np.searchsorted(bellman.x, nodes, side="right") - 1, 0, bellman.x.size - 1)
    s = escapement[index]                           # greedy escapement at each node
    successors = np.clip(np.outer(growth(s), bellman.z), 0, X_MAX)
    P_Phi = np.einsum("q,iqj->ij", bellman.w, hats(successors))
    new = np.linalg.solve(np.eye(basis.n) - gamma * P_Phi, nodes - s)
    step, theta = np.abs(new - theta).max(), new
    if step < 1e-10:
        break
print(k + 1, "Newton iterations")
```

Les pas valent $1.6$, $2.9$, $0.86$, $3 \times 10^{-7}$, puis zéro : cinq
itérations, contre environ 410 pour l'itération fonctionnelle, et les
coefficients concordent avec le point fixe de l'itération fonctionnelle à
$10^{-14}$ près. Ils doivent concorder, parce que les deux méthodes résolvent
les mêmes équations de collocation $\theta = \Proj\Bellman v_\theta$, qui ont
ici une solution unique puisque $\Proj\Bellman$ est une contraction; la
méthode de Newton la trouve seulement plus vite.
:::

:::{exercise} Choisir les nœuds du polynôme
:label: ex-wr-nodes

Avec 10 coefficients de Tchebychev, l'itération fonctionnelle sur la pêcherie
diverge. Explorez ce qu'on pourrait changer tout en gardant un polynôme de
degré 9 : plus de points de collocation que de coefficients avec un ajustement
par moindres carrés, un placement différent des nœuds, ou la méthode de Newton
lancée à partir de la solution linéaire par morceaux convergée. Pour chaque
option, indiquez si l'itération converge et quelle est la perte de politique du
résultat, et reliez ce que vous observez au signe des fonctions cardinales et à
la stratégie hybride de {ref}`sec-wr-solve`.
:::

:::{solution} ex-wr-nodes
:class: dropdown

Cet exercice est ouvert. Un ajustement par moindres carrés à plus de nœuds que
de coefficients n'est plus un moyenneur; il n'offre donc pas non plus de
garantie, même si le suréchantillonnage réduit la taille des poids. La méthode
de Newton n'itère pas $\Proj\Bellman$ et n'a besoin d'aucune contraction;
lancée à partir de la solution linéaire par morceaux, qui est proche, elle
constitue la stratégie hybride du chapitre. Quoi qui converge, jugez-le par la
perte de la politique, comme à l'étape 5 : un polynôme qui converge peut encore
placer la frontière de décision du mauvais côté du point de bascule.
:::

## Synthèse et perspectives

Remplacer la fonction de valeur par $v(\cdot;\theta)$ transforme l'équation de
Bellman en un résidu $R(\cdot;\theta) = \Bellman v_\theta - v_\theta$
qu'aucun choix de $\theta$ n'annule partout. La méthode des résidus pondérés
lui impose plutôt $n$ conditions : la collocation l'annule en des états
choisis, la méthode de Galerkin le rend orthogonal à la base, et les moindres
carrés minimisent sa norme. Les deux premières sont des équations de Bellman
projetées $v_\theta = \Proj\,\Bellman v_\theta$, et leurs solveurs sont les
deux solveurs de l'équation exacte auxquels on insère une projection :
l'itération fonctionnelle, qui est l'itération sur les valeurs avec un
ajustement après chaque balayage, et la méthode de Newton, qui est l'itération
sur les politiques où chaque évaluation est faite dans l'espace engendré par la
base. La pêcherie a montré que l'ajustement décide à la fois de la précision de
la solution et de la convergence de l'itération fonctionnelle, et qu'un résidu
plus petit ne signifie pas une meilleure politique.

Rien dans les cinq étapes n'est propre à l'équation de Bellman. Toute équation
d'opérateur $\Residual(f) = 0$ pour une fonction inconnue $f$ se traite de la
même manière, avec une base pour $f$, un résidu $\Residual(\hat f)$, des
fonctions test et l'un des deux solveurs, et le chapitre sur la collocation a
fait exactement cela pour les équations différentielles, avec des morceaux
polynomiaux pour la trajectoire et l'équation du mouvement imposée aux points
de collocation. Ce que l'équation de Bellman ajoute, c'est le maximum, qui rend
les équations non linéaires et fait de la jacobienne une affaire de théorème
de l'enveloppe, et la contraction, qui est ce qui rend l'itération
fonctionnelle envisageable.

L'étape 4 a laissé deux points en suspens. L'itération fonctionnelle a convergé
pour une base et pas pour l'autre, et l'explication, à savoir que l'ajustement
ne doit pas étirer les distances en norme sup, a été affirmée sans être
démontrée. La méthode de Newton s'est révélée être l'itération sur les
politiques dans l'espace des coefficients pour la collocation, sans qu'on dise
avec quelle précision chaque évaluation doit être faite pour que les politiques
s'améliorent. Le [chapitre suivant](approximate-bellman-equations.md) aborde
ces deux points pour une règle d'ajustement quelconque. L'itération
fonctionnelle devient l'itération sur les valeurs ajustée, une suite de
problèmes de régression à cibles de Bellman que tout régresseur peut résoudre,
et le schéma sous-jacent à l'itération Q ajustée, à NFQ et à NFQCA. Le chapitre
borne son erreur, montre quand l'ajustement la fait diverger et quels
ajustements ne le peuvent pas, et transforme la méthode de Newton en itération
sur les politiques approchée.
