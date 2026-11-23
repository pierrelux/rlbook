---
description: Formuler des problèmes de décision séquentielle en précisant leur état, leur canal d'action, leur incertitude, leurs observations, leurs contraintes et l'interface de modèle disponible.
kernelspec:
  name: python3
  display_name: Python 3
---
# Dynamique stochastique et observation partielle

Une dynamique déterministe associe un seul état suivant à chaque couple
état-action. Le bruit de procédé, les apports incertains et les interactions
non modélisées produisent plutôt une loi sur les états suivants possibles.
Cette distinction change le problème de décision : un correcteur peut devoir
comparer un coût espéré, une probabilité de défaillance ou un risque sur
l'ensemble de ces issues possibles, au lieu d'optimiser une seule trajectoire
nominale.

Comment définir la prédiction et la sélection d'action lorsque l'état suivant
est une loi de probabilité et que le correcteur n'observe peut-être pas
directement l'état réalisé?

Une première représentation rend explicite la source de l'aléa :

$$
x_{t+1}=f_t(x_t,u_t,\xi_t),
\qquad
\xi_t\mid x_t,u_t\sim p_t(\cdot\mid x_t,u_t).
$$

Pour une perturbation réalisée $\xi_t$, la fonction $f_t$ renvoie toujours un
seul état suivant. La loi conditionnelle de $\xi_t$ détermine la vraisemblance
des différentes réalisations et induit donc une loi sur $x_{t+1}$. Le modèle
gaussien additif habituel suppose une suite de bruits indépendants de
covariance fixe :

$$
x_{t+1}=Ax_t+Bu_t+\xi_t,
\qquad
\xi_t\sim\mathcal N(0,Q).
$$

La dynamique induite peut aussi être représentée directement par un noyau de
transition,

$$
P_t(A\mid x,u)
=\Pr(x_{t+1}\in A\mid x_t=x,u_t=u).
$$

Pour tout ensemble $A$ d'états suivants possibles, $P_t(A\mid x,u)$ est la
probabilité que l'état suivant appartienne à $A$ après l'application de
l'action $u$ dans l'état $x$. Cet objet s'appelle un **noyau de transition**.
La représentation par fonction et bruit montre comment l'aléa intervient et
peut permettre la dérivation trajectorielle, qui consiste à dériver une
transition échantillonnée en gardant fixe son tirage aléatoire. Le noyau
n'exige que la loi conditionnelle de l'état suivant; il couvre donc aussi les
simulateurs dont les variables aléatoires internes sont cachées.

En temps continu, le bruit doit être mis à l'échelle de façon cohérente avec la
durée d'un intervalle de temps. Une équation différentielle stochastique
sépare le taux de variation déterministe, appelé dérive, des accroissements
aléatoires à fluctuation rapide, appelés diffusion :

$$
dX_t=f(X_t,U_t)\,dt+\sigma(X_t,U_t)\,dW_t.
$$

Ici, $f$ est la dérive, $W_t$ est un mouvement brownien standard, et $\sigma$
transporte ses accroissements dans l'état. Sur un intervalle de longueur
$\Delta t$, le mouvement brownien vérifie

$$
W_{t+\Delta t}-W_t\sim\mathcal N(0,\Delta t I).
$$

La variance de l'accroissement est proportionnelle au temps écoulé.
L'analogue stochastique d'un pas d'Euler explicite, appelé schéma
d'Euler-Maruyama, prend donc la forme

$$
X_{k+1}\approx X_k+f(X_k,U_k)\Delta t
+\sigma(X_k,U_k)\sqrt{\Delta t}\,\varepsilon_k,
\qquad \varepsilon_k\sim\mathcal N(0,I).
$$

L'écart type de l'accroissement aléatoire est donc proportionnel à
$\sqrt{\Delta t}$, et sa covariance à $\Delta t$. Diviser par deux l'intervalle
d'échantillonnage ne revient pas à ajouter la même covariance deux fois plus
souvent. Un traitement complet des équations différentielles stochastiques
dépasse le cadre de ce livre; ces relations consignent la mise à l'échelle
nécessaire pour construire un modèle échantillonné cohérent. Le
[chapitre sur la commande stochastique par intégrales de chemin](path-integral-stochastic-control.md#sec-path-integral-control)
utilise plus loin ces moments browniens pour étendre la programmation
dynamique au temps continu et exprimer la rétroaction optimale par des
espérances sur les trajectoires.

### Stock de vélos et demande stochastique

La gestion d'un stock fournit un modèle de transition concret où des
événements aléatoires et des contraintes strictes interagissent. Une location
n'est possible que si un vélo est présent, et un retour n'est possible que si
un point d'ancrage est libre. Les décisions de redistribution modifient ces
conditions avant que les prochaines arrivées aléatoires ne se réalisent.

Entre 7 h et 10 h le 4 juillet 2024, les stations BIXI Berri--Cherrier et
Prince-Arthur--St-Urbain ont enregistré beaucoup plus de locations terminées
que de retours. La station de Maisonneuve--Aylmer a enregistré le flux
inverse. Les trois stations sont à moins de 1.6 km les unes des autres, si
bien qu'un petit camion de redistribution peut déplacer des vélos entre elles
en un seul intervalle de 15 minutes. Cette frontière volontairement locale
permet d'étudier la rétroaction sur les stocks sans avoir aussi à choisir des
itinéraires dans tout le réseau montréalais.

:::{figure} _static/bixi/bixi-model-interface.svg
:label: fig-bixi-model-interface
:alt: Schéma, fidèle aux coordonnées, de trois stations BIXI voisines et de leurs nombres de locations et de retours terminés pendant la matinée enregistrée.

La frontière des trois stations contient 99 points d'ancrage. Berri a
enregistré 79 locations terminées et 36 retours, Prince-Arthur 48 et 20, et de
Maisonneuve 12 et 84. Les événements proviennent de l'archive des
déplacements BIXI de 2024; les capacités proviennent d'un relevé de
l'information sur les stations obtenu le 2 septembre 2026.
:::

La simulation utilise pour chaque essai les capacités de stations consignées
dans un relevé de l'information sur les stations daté du 2 septembre 2026. Ces
valeurs rendent l'exemple reproductible, mais elles ne sont pas forcément
égales aux capacités en service le 4 juillet 2024.

| station | identifiant | capacité |
|---|---:|---:|
| Berri / Cherrier | 173 | 39 |
| Prince-Arthur / St-Urbain | 404 | 23 |
| de Maisonneuve / Aylmer (ouest) | 68 | 37 |

Soit $s_{i,k}$ le nombre de vélos à la station $i$, $c_i$ sa capacité et
$b_k$ le stock du camion. Un transfert $\rho_{i,k}$ est positif lorsque le
camion décharge des vélos à sa station courante et négatif lorsqu'il en
charge. Le stock après transfert au début d'un intervalle est
$q_{i,0}=s_{i,k}+\rho_{i,k}$.

Les tentatives de location et de retour arrivent ensuite sous la forme d'une
suite chronologique d'événements. Pour l'événement $e$, notons $R_{i,e}$
l'indicatrice d'une location servie et $A_{i,e}$ celle d'un retour accepté.
Leurs valeurs dépendent du stock juste avant l'événement :

$$
\begin{aligned}
R_{i,e}
&=\mathbf 1\{\text{l'événement $e$ est une location en $i$}\}
  \mathbf 1\{q_{i,e-1}\geq 1\},\\
A_{i,e}
&=\mathbf 1\{\text{l'événement $e$ est un retour en $i$}\}
  \mathbf 1\{q_{i,e-1}<c_i\},\\
q_{i,e}&=q_{i,e-1}-R_{i,e}+A_{i,e}.
\end{aligned}
$$

Si l'intervalle $k$ contient $N_k$ événements, son bilan de stock est

$$
s_{i,k+1}=q_{i,N_k},
\qquad
b_{k+1}=b_k-\sum_i\rho_{i,k}.
$$

La redistribution conserve la somme des stocks des stations et du camion. Les
déplacements des usagers peuvent néanmoins changer le total à l'intérieur de
cette frontière de trois stations, car leur autre extrémité peut se trouver
ailleurs dans le réseau.

Le procédé pédagogique part des stocks $(30,18,8)$ aux stations et d'un
camion vide de 16 places à de Maisonneuve. Les décisions sont prises toutes les
15 minutes. Un arrêt peut transférer au plus huit vélos, et le trajet entre
deux des trois stations prend un intervalle de décision. À une même station et
à un même instant, le simulateur traite les retours avant les locations. Il
consigne les locations perdues et les retours refusés au lieu de tronquer
silencieusement les tentatives.

Trois correcteurs reçoivent la même trace d'événements. Le premier ne déplace
jamais de vélo. Le deuxième fige un programme calculé avant 7 h à partir des
nombres moyens de déplacements terminés. Le troisième réévalue toutes les 15
minutes une règle déterministe écrite à la main à partir des stocks observés.
Il ne résout pas de problème d'optimisation à chaque instant de décision. Ici,
la *rétroaction* signifie que l'observation courante entre dans la règle; elle
n'implique ni optimisation ni apprentissage.

Les deux correcteurs actifs utilisent la cible sur l'horizon restant

$$
g_{i,n}=
\operatorname{clip}\left(
\frac{c_i}{2}
+\sum_{h=n}^{T-1}(\hat d_{i,h}-\hat a_{i,h}),
0.2c_i,0.8c_i
\right).
$$

La fonction $\operatorname{clip}(q,l,r)$ ramène $q$ dans l'intervalle $[l,r]$.
La cible part donc de la moitié de la capacité, monte lorsque les départs
futurs devraient dépasser les arrivées et reste entre 20 et 80 % de la
capacité de la station. Ici, $\hat d_{i,h}$ et $\hat a_{i,h}$ sont les nombres
moyens de départs et d'arrivées terminés à la station $i$ pendant l'intervalle
de 15 minutes $h$, calculés sur les 43 dates d'étalonnage. On définit le
déficit et l'excédent courants par rapport à la cible par

$$
\delta^-_{i,n}=[g_{i,n}-s_{i,n}]_+,
\qquad
\delta^+_{i,n}=[s_{i,n}-g_{i,n}]_+.
$$

L'opérateur de partie positive $[q]_+=\max(q,0)$ rend le déficit et l'excédent
non négatifs. Au plus l'un des deux est positif pour une station donnée. La
règle décide d'abord du transfert à la station courante du camion. Si le
camion transporte des vélos et que le déficit local est d'au moins deux, il
décharge. Si l'excédent local est d'au moins deux et que les autres stations
ont un déficit combiné d'au moins deux, il charge. La taille du transfert est
limitée par la limite de huit vélos par arrêt, la capacité du camion, les
vélos disponibles, les points d'ancrage libres et l'excédent ou le déficit en
cause. Si le camion transporte encore des vélos après le transfert, il se
dirige vers l'autre station dont le ratio $\delta^-_{i,n}/c_i$ est le plus
grand. S'il est vide, et qu'il reste à la fois un déficit et un excédent d'au
moins deux, il se dirige vers l'autre station dont le ratio
$\delta^+_{i,n}/c_i$ est le plus grand. La distance départage les égalités.
Enfin, lorsque le camion transporte des vélos restants mais que le déficit
total modélisé est inférieur à deux, il en remet le plus possible à sa station
courante plutôt que de les garder hors de portée des usagers.

Le programme figé est produit avant 7 h en appliquant cette même heuristique à
une trajectoire déterministe de flux moyens. Il garde ensuite ces actions
inchangées. Ce calcul en fait un plan en boucle ouverte, et non un second
optimiseur en ligne. Dans le programme obtenu, le camion attend à de
Maisonneuve, charge trois vélos à 7 h 45, se rend à Prince-Arthur et les
décharge à 8 h. Le correcteur par rétroaction recalcule plutôt la règle à
partir de chaque stock nouvellement observé.

Pour la comparaison stochastique, notons $C^z_{i,k,d}$ le nombre d'événements
terminés de type $z\in\{\mathrm{location},\mathrm{retour}\}$ à la station $i$
pendant l'intervalle $k$ à la date d'étalonnage $d$. Le taux empirique et le
nombre simulé sont

$$
\widehat\lambda^z_{i,k}
=\frac{1}{43}\sum_{d=1}^{43}C^z_{i,k,d},
\qquad
N^z_{i,k}\sim
\operatorname{Poisson}\!\left(\widehat\lambda^z_{i,k}\right).
$$

La loi de Poisson est un modèle de dénombrement dont la moyenne est ici
$\widehat\lambda^z_{i,k}$. Les tirages sont indépendants entre stations,
intervalles et types d'événements. Conditionnellement à $N^z_{i,k}=m$, le
simulateur place les $m$ instants d'événements de façon indépendante et
uniforme dans l'intervalle $k$. Il traite ces événements comme des tentatives
potentielles et accepte ou refuse chacune d'elles à l'aide des équations de
stock ci-dessus. Le modèle transforme ainsi des taux estimés à partir de
déplacements *terminés* en taux de tentatives. C'est une hypothèse de
modélisation déclarée : l'archive publique ne révèle pas la demande censurée
par une station vide ou pleine.

Pour chaque germe de 0 à 511, les trois correcteurs reçoivent la même trace
d'événements échantillonnée. Ce plan à nombres aléatoires communs compare leurs
décisions sous la même réalisation de la demande. Une condition distincte
d'impulsions appariées ajoute à chaque trace huit tentatives de location à
Prince-Arthur à 7 h 47 et huit tentatives de retour à de Maisonneuve à 7 h 57.
Ces ajouts sont fixes, et non des tirages de Poisson supplémentaires, et
chacun ne réussit que si le stock ou les points d'ancrage le permettent. La
règle de rétroaction utilise leurs conséquences sur les stocks dès la décision
de 8 h. Le correcteur figé reçoit la nouvelle observation, mais l'ignore et
exécute l'action choisie avant 7 h.

La comparaison enregistrée ci-dessous utilise la même trace d'événements pour
chaque correcteur. La carte, le remplissage des stations, le mouvement du
camion et les défaillances de service proviennent de trajectoires Python
versionnées. Le choix du correcteur et la tête de lecture ne font que parcourir
ces enregistrements; ils ne recalculent pas le procédé dans le navigateur.

```{code-cell} python
:tags: [remove-input]

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from bixi_replay import render_bixi_replay

display(HTML(render_bixi_replay(
    Path("artifacts/bixi/textbook_results.json"),
    replay_id="bixi-modeling-replay",
    fallback_id="fig-bixi-control-replay-fallback",
)))
```

:::{figure} _static/bixi/bixi-feedback-evidence.svg
:label: fig-bixi-control-replay-fallback
:class: pdf-fallback
:alt: Trajectoires des stocks des stations BIXI et intervalles appariés de défaillances de service sans redistribution, avec un programme figé et avec rétroaction sur les stocks.

Trajectoires statiques et résultats stochastiques appariés issus de la même
expérience Python. Le livre en ligne ajoute le choix du correcteur, la
lecture, l'avance pas à pas et le défilement.
:::

```{include} artifacts/bixi/results.md
```

La rétroaction réduit les défaillances de service dans ce modèle, mais elle
demande plus de travail de redistribution. Ce sont des résultats du simulateur
décrit, et non des mesures de défaillances de service ni des estimations d'un
effet causal dans l'exploitation de BIXI. L'archive publique omet les
tentatives infructueuses, les stocks historiques et les déplacements des
camions de l'exploitant. La matinée retenue l'a été après examen parce que son
déséquilibre est facile à voir; ce n'est pas un échantillon d'évaluation sans
biais {cite}`bixiOpenData2024,montrealBixiTrips,hulot2018bixi`.

:::{dropdown} Examiner la transition BIXI et la règle de rétroaction
```{literalinclude} code/bixi_control.py
:language: python
:start-at: class InventoryFeedbackController
:end-before: def _sha256
:linenos:
```

```{literalinclude} code/bixi_control.py
:language: python
:start-at: def simulate(
:end-before: def simulate_fluid
:linenos:
```

{download}`Télécharger l'expérience BIXI complète <code/bixi_control.py>`
:::

Le simulateur BIXI fournit directement au correcteur par rétroaction les stocks
des stations et du camion. Les capteurs physiques ne fournissent souvent que
des mesures indirectes et bruitées de l'état.

## Observabilité partielle

Un noyau de transition décrit l'incertitude sur l'état lui-même. Qu'est-ce qui
change lorsque le correcteur ne reçoit qu'une fonction bruitée ou ambiguë de
cet état?

Le correcteur BIXI reçoit directement les stocks modélisés. Beaucoup de
correcteurs reçoivent plutôt des lectures de capteurs qui ne contiennent
qu'une information partielle ou bruitée sur l'état prédictif. Un modèle
d'observation relie l'état caché à la mesure disponible :

$$
y_t=h(x_t,u_t,\nu_t),
\qquad
\nu_t\sim p_\nu.
$$

Ici, $p_\nu$ est la loi attribuée au bruit de mesure. Pour un capteur linéaire
gaussien, $y_t=Cx_t+\nu_t$ avec $\nu_t\sim\mathcal N(0,R)$. Un correcteur qui
ne s'appuie que sur $y_t$ peut perdre de l'information nécessaire à la
prédiction des transitions futures. Sous un modèle connu, la croyance a
posteriori

$$
b_t(A)=\Pr(x_t\in A\mid y_{0:t},u_{0:t-1})
$$

attribue une probabilité à chaque ensemble $A$ d'états cachés possibles. Cette
croyance est un **état d'information** : elle conserve le contenu prédictif de
l'historique des observations sous une forme qu'on peut mettre à jour après
chaque action et chaque mesure. Une estimation ponctuelle coûte souvent moins
cher à utiliser, mais elle peut écarter une incertitude qui influe sur les
décisions futures.

### Stabilisation d'une caméra sous observation partielle

Imaginons une caméra montée sur une nacelle à un axe, fixée à un véhicule en
mouvement. Le moteur cherche à garder l'horizon de niveau. Un petit choc peut
faire tourner la caméra, et une accélération latérale du véhicule peut
perturber ses capteurs même lorsque la caméra elle-même ne tourne pas. Le
correcteur doit déduire ce qui s'est passé à partir d'une centrale inertielle
plutôt que de l'angle réel.

L'état caché simulé est

$$
x=(\theta,\omega,b)
$$

où $\theta$ est l'angle de la caméra en radians, $\omega$ sa vitesse angulaire
en radians par seconde, et $b$ un décalage dans la lecture du gyroscope,
lui aussi en radians par seconde. La caméra est équilibrée autour de l'axe de
la nacelle, si bien que la gravité ne crée pas de couple de rotation dans ce
modèle. Son bilan de couples selon un axe est

$$
\dot\theta=\omega,
\qquad
J\dot\omega=u-c\omega+\tau_{\mathrm{ext}}(t),
\qquad |u|\leq u_{\max}.
$$

Ici, $J$ est le moment d'inertie de la caméra autour de l'axe de la nacelle,
mesuré en $\mathrm{kg\,m^2}$. Un $J$ plus grand signifie qu'un même couple
produit une accélération angulaire plus faible. Le couple moteur $u$ et le
couple externe $\tau_{\mathrm{ext}}$ sont mesurés en $\mathrm{N\,m}$, et $c$
est un coefficient d'amortissement rotatif visqueux mesuré en
$\mathrm{N\,m\,s/rad}$. La borne $u_{\max}$ représente la limite de couple du
moteur.

Le biais $b$ appartient au capteur, et non au procédé mécanique. Si la caméra
est immobile mais que $b=0.8$ degré par seconde, la lecture du gyroscope reste
centrée près de $0.8$ degré par seconde au lieu de zéro. Ce décalage
persistant diffère des petites fluctuations rapides qui varient
indépendamment d'une mesure à l'autre.

Les erreurs réelles des capteurs dépendent de la température, de l'étalonnage
et du matériel. Pour cette simulation de dix secondes, une marche aléatoire
fournit un modèle simple d'un décalage qui persiste mais peut dériver
{cite}`elSheimy2008allan` :

$$
b_{k+1}=b_k+\sigma_b\sqrt{\Delta t}\,\epsilon_k,
\qquad \epsilon_k\sim\mathcal N(0,1).
$$

Conditionnellement à $b_k$, le biais suivant a pour moyenne $b_k$, et sa
variation a pour écart type $\sigma_b\sqrt{\Delta t}$. Rien dans le modèle ne
ramène le biais vers zéro. Ici, $b_0=0.8$ degré par seconde et
$\sigma_b=0.02\ (^\circ/\mathrm{s})/\sqrt{\mathrm{s}}$, si bien que l'écart type
d'une variation du biais n'est que de $0.002$ degré par seconde à la période
d'échantillonnage du capteur, $\Delta t=0.01$ seconde. La marche aléatoire
choisie varie donc lentement par rapport au bruit de mesure d'un échantillon à
l'autre. C'est un modèle pédagogique reproductible, et non l'affirmation que
tout gyroscope dérive exactement de cette façon.

Le gyroscope mesure la vitesse angulaire, entachée à la fois du biais
persistant et du bruit de mesure qui varie d'un échantillon à l'autre :

$$
y_k^\omega=\omega_k+b_k+\nu_k^\omega.
$$

Intégrer ces lectures transforme une estimation de vitesse en estimation
d'angle. Sur $T$ secondes, un biais à peu près constant $b$ contribue environ
$Tb$ à cet angle estimé. Pour une caméra réellement immobile, intégrer un biais
non corrigé de $0.8$ degré par seconde produit donc une rotation estimée
d'environ huit degrés après dix secondes. Un bruit de mesure de moyenne nulle
tend à fluctuer dans les deux sens; le décalage persistant s'accumule dans un
seul sens.

L'accéléromètre crée une ambiguïté différente. Il mesure une *force
spécifique*, c'est-à-dire la force par unité de masse que perçoit sa masse
d'épreuve interne, plutôt que de rapporter directement un angle de caméra.
Lorsque le véhicule est à l'arrêt ou se déplace à vitesse constante, la force
de soutien qui s'oppose à la gravité fournit une référence verticale. Une
accélération latérale du véhicule modifie la même lecture du capteur et peut
donc ressembler à une inclinaison de la caméra
{cite}`androidMotionSensors2026`.

La simulation ne conserve que les deux axes du plan de rotation. Son modèle
d'accéléromètre est

$$
\begin{bmatrix}y_k^x\\y_k^y\end{bmatrix}
=R(-\theta_k)
\begin{bmatrix}a_{x,k}^{w}\\g\end{bmatrix}+\nu_k^a.
$$

Le vecteur $[a_{x,k}^{w},g]^\top$ contient l'accélération latérale du véhicule
et la référence de gravité dans le repère du monde. La rotation $R(-\theta_k)$
exprime ce vecteur selon les axes du capteur fixé à la caméra, et $\nu_k^a$
ajoute un bruit de mesure instantané. Si l'accélération latérale et le bruit
de mesure sont tous deux nuls, la direction de ce vecteur détermine
$\theta_k$. Avec une accélération latérale, l'angle déduit des deux mêmes
composantes devient

$$
\operatorname{atan2}(y_k^x,y_k^y)
=\theta_k+\operatorname{atan2}(a_{x,k}^{w},g).
$$

Par exemple, $a_x^w=3\ \mathrm{m/s^2}$ ajoute une inclinaison apparente
d'environ $17$ degrés. Une caméra de niveau sur un véhicule qui accélère
latéralement peut alors produire la même direction d'accéléromètre qu'une
caméra inclinée sur une base immobile. C'est le problème de l'observation
partielle : la lecture du capteur ne permet pas à elle seule de savoir quelle
situation cachée s'est produite.

Les deux capteurs échouent à des échelles de temps différentes.
L'accéléromètre fournit une référence stable à long terme lorsque la
translation du véhicule est modérée, mais une accélération latérale la fausse
immédiatement. Le gyroscope suit les rotations rapides sans les confondre avec
une translation, mais l'intégration d'un biais inconnu produit une dérive à
long terme. Un **observateur complémentaire** est un estimateur d'état
récursif qui combine ces deux échelles de temps. Il utilise le gyroscope pour
les variations rapides, l'accéléromètre pour la correction lente de l'angle,
et le désaccord persistant entre les deux pour estimer le biais du gyroscope
{cite}`mahony2008complementary`.

La comparaison ci-dessous ne change que l'estimateur. Un correcteur utilise la
direction de l'accéléromètre comme estimation de l'angle, un autre intègre le
gyroscope brut, et le troisième utilise l'observateur complémentaire. Chaque
essai reçoit le même procédé mécanique, la même réalisation du bruit, les
mêmes perturbations, la même limite de couple et la même loi de rétroaction :

$$
u_k=\operatorname{clip}
\left(-0.9\hat\theta_k-0.12\hat\omega_k,
-0.18,0.18\right).
$$

La fonction $\operatorname{clip}$ limite le couple moteur demandé à
l'intervalle $[-0.18,0.18]\ \mathrm{N\,m}$. Le procédé simulé utilise
$J=0.018\ \mathrm{kg\,m^2}$, $c=0.025\ \mathrm{N\,m\,s/rad}$, une intégration
mécanique à la milliseconde, et une mesure et une commande toutes les dix
millisecondes. La trace de capteurs de dix secondes est générée une seule fois
avec le germe 11, puis fournie telle quelle à chaque estimateur.

Les trois essais commencent avec la caméra inclinée de huit degrés. Un choc de
$0.14\ \mathrm{N\,m}$ entre 1.50 et 1.62 seconde fait tourner physiquement la
caméra; les deux capteurs inertiels devraient donc aider le correcteur à
réagir. De 4.0 à 4.6 secondes, la base accélère plutôt latéralement jusqu'à
$3\ \mathrm{m/s^2}$. Cet événement modifie la lecture de l'accéléromètre sans
faire tourner directement la caméra équilibrée. L'intervalle calme final met
en évidence l'erreur d'angle produite par l'intégration du gyroscope biaisé.

La reprise n'affiche les courbes que jusqu'à la tête de lecture courante. Ses
mesures utilisent l'angle simulé caché, accessible à l'expérience mais pas aux
correcteurs.

```{code-cell} python
:tags: [remove-input]

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from gimbal_replay import render_gimbal_replay

display(HTML(render_gimbal_replay(
    Path("artifacts/gimbal/textbook_results.json"),
    replay_id="gimbal-observation-replay",
    fallback_id="fig-gimbal-observation-fallback",
)))
```

:::{figure} _static/gimbal/partial-observability.svg
:label: fig-gimbal-observation-fallback
:class: pdf-fallback
:alt: Trois nacelles de caméra et leurs trajectoires d'angle réel sous estimation d'état par accéléromètre seul, par gyroscope intégré et par observateur complémentaire.

Instantanés statiques et trajectoires complètes issus de la même expérience
Python. Le livre en ligne ajoute la lecture, l'avance pas à pas et le
défilement.
:::

```{include} artifacts/gimbal/results.md
```

Les résultats suivent les deux mécanismes de défaillance. L'estimation par
accéléromètre seul réagit fortement à l'impulsion de translation.
L'estimation par gyroscope intégré ignore en grande partie cette impulsion,
mais dérive pendant l'intervalle calme. L'observateur complémentaire réduit
les deux erreurs en utilisant l'accéléromètre comme correction lente de
l'angle tiré du gyroscope. Une translation inconnue et soutenue peut encore
être prise pour une inclinaison. Ce sont des résultats d'une simulation
pédagogique transparente, et non des performances mesurées d'une nacelle
commerciale.

La même distinction apparaît en optique adaptative : les pentes locales du
front d'onde sont des observations de capteur, tandis que la phase
reconstruite est l'état utilisé pour la commande.

:::{dropdown} Examiner le procédé échantillonné, les capteurs, l'estimateur et le correcteur
```{literalinclude} code/gimbal_control.py
:language: python
:start-at: def sample_observation
:end-before: def compute_metrics
:linenos:
```

{download}`Télécharger l'expérience complète de la nacelle <code/gimbal_control.py>`
:::

Les équations d'état, d'action, de perturbation et d'observation décrivent la
sémantique du modèle. Un algorithme a encore besoin d'un moyen de calcul pour
utiliser ce modèle. La distinction suivante porte sur les opérations que le
modèle met à disposition : formules et dérivées, simulation par
réinitialisation et pas, transitions historiques fixes, ou nouvelle
interaction en ligne.

## Synthèse et perspectives

Les noyaux de transition séparent l'action choisie par le décideur de la
perturbation qui choisit un successeur réalisé. Les modèles d'observation
ajoutent une seconde séparation : l'état sert à la prédiction, tandis que la
mesure ne fournit que l'information disponible pour choisir l'action
suivante.

Ces équations précisent ce que signifie un modèle, mais pas comment un
algorithme peut s'en servir. L'algorithme peut-il dériver la transition,
réinitialiser un simulateur, rééchantillonner un état, ou seulement examiner
des transitions historiques? Le chapitre [Interfaces de modèle et composantes
apprises](model-interfaces.md) rend ces opérations explicites.

## Exercices

:::{exercise} Conservation et rétroaction dans la redistribution des vélos
:label: ex-dynamics-bixi-conservation

Avant que surviennent les événements des usagers, vérifiez que la
redistribution conserve la somme des stocks des stations et du camion.
Comparez ensuite le programme figé et la rétroaction sur les stocks après
l'impulsion de demande déclarée. Quelle nouvelle observation peut changer
l'action de rétroaction, et pourquoi l'action figée ne peut-elle pas changer?
:::

:::{solution} ex-dynamics-bixi-conservation
:class: dropdown

Puisque $q_{i,0}=s_{i,k}+\rho_{i,k}$ et $b_{k+1}=b_k-\sum_i\rho_{i,k}$,

$$
\sum_i q_{i,0}+b_{k+1}=\sum_i s_{i,k}+b_k.
$$

Les déplacements des usagers peuvent ensuite franchir la frontière des trois
stations et modifier ce total. À 8 h, la rétroaction observe les stocks
produits par les locations imprévues à Prince-Arthur et les retours à de
Maisonneuve; elle peut donc modifier son transfert ou son itinéraire. Le
programme figé a été fixé à partir de l'information disponible avant 7 h et ne
comporte aucune règle pour utiliser la nouvelle observation.
:::

:::{exercise} Inclinaison ou translation?
:label: ex-dynamics-gimbal-ambiguity

Construisez deux couples $(\theta,a_x^w)$ qui produisent la même inclinaison
déduite de l'accéléromètre sans bruit. Quelle information supplémentaire le
gyroscope fournit-il, et quelle ambiguïté demeure?
:::

:::{solution} ex-dynamics-gimbal-ambiguity
:class: dropdown

Les deux couples

$$
(\theta,a_x^w)=(10^\circ,0)
\quad\text{et}\quad
(\theta,a_x^w)=\left(0,g\tan 10^\circ\right)
$$

produisent une inclinaison apparente de $10^\circ$. Le gyroscope mesure la
vitesse angulaire plus un biais, ce qui aide à distinguer une rotation rapide
d'une impulsion de translation. Un biais inconnu du gyroscope s'accumule
toutefois en erreur d'angle, et une translation inconnue et soutenue reste
confondue avec la gravité dans l'accéléromètre.
:::

:::{exercise} Fonction et bruit, noyaux de transition
:label: ex-dynamics-reservoir-kernel

Supposons qu'un réservoir sans déversement ni évaporation obéisse à
$s_{t+1}=s_t-u_t+w_t$, où $w_t=\bar w_t+\epsilon_t$. Écrivez un modèle par
fonction et bruit, ainsi que le noyau de transition induit.
:::

:::{solution} ex-dynamics-reservoir-kernel
:class: dropdown

La forme à bruit explicite est

$$
s_{t+1}=s_t-u_t+\bar w_t+\epsilon_t.
$$

Si $\epsilon_t$ a pour loi conditionnelle $p_t$, alors le noyau induit est

$$
P_t(A\mid s,u)
=\Pr\!\left(s-u+\bar w_t+\epsilon_t\in A\right).
$$

Ce noyau est la mesure image de la loi du bruit par l'équation de bilan.
Ajouter des bornes ou un déversement changerait l'application et pourrait
créer une masse de probabilité à une frontière.
:::

:::{exercise} Une observation non markovienne
:label: ex-dynamics-nonmarkov-observation

Un correcteur d'inférence observe les longueurs de files et l'occupation
totale du cache, mais pas la longueur de sortie restante de chaque requête
active. Pourquoi ces mesures ne forment-elles pas en général un état
markovien? Quel état ou quel historique supplémentaire améliorerait la
prédiction?
:::

:::{solution} ex-dynamics-nonmarkov-observation
:class: dropdown

Deux ensembles de requêtes peuvent avoir les mêmes longueurs de files et la
même occupation du cache, mais des longueurs de sortie restantes différentes.
Leurs instants d'achèvement et leurs libérations de cache ont donc des lois
conditionnelles différentes. Un état complet peut conserver le travail
restant de chaque requête. Lorsque ce travail est caché, l'historique complet
des observations suffit en principe, et une loi a posteriori sur l'état au
niveau des requêtes est un état d'information compact sous un modèle connu.
:::

:::{exercise} Répartition avec des durées de service cachées
:label: ex-dynamics-modeling-dispatch

Un service informatique révise son fonctionnement une fois par minute. Quatre
unités de calcul peuvent traiter des tâches de l'une ou l'autre de deux files.
À chaque révision, le répartiteur voit le nombre de tâches en attente dans
chaque file, le nombre d'unités occupées, les modes courants des unités et une
mesure de puissance. Il choisit combien d'unités seront actives et quelle file
aura la priorité pendant la minute suivante. De nouvelles tâches arrivent à
des instants aléatoires. Chaque tâche en cours a un temps de traitement restant
non observé, et cette quantité détermine si la tâche se termine pendant la
minute suivante. L'objectif pénalise le temps d'attente et la consommation
d'énergie. Une limite stricte de puissance doit être respectée à chaque pas.

(a) Tracez la frontière du système et classez l'état, l'observation,
l'action, la perturbation, l'objectif, la contrainte, la représentation du
temps et l'horizon. Utilisez un horizon fini de 60 décisions.

(b) Expliquez pourquoi les longueurs de files observées et le nombre d'unités
occupées ne forment pas en général un état markovien. Donnez une description
d'état complet du système et un état d'information dont dispose le
répartiteur.

(c) Choisissez entre une dynamique déterministe et un noyau de transition
stochastique. Indiquez quelle phrase de l'énoncé détermine votre choix.

(d) Distinguez un programme d'affectation fixe sur 60 minutes d'une politique
de rétroaction. Quelles observations peuvent influencer l'action de
rétroaction après la première minute?

(e) Supposons que les seules données disponibles soient des transitions
consignées sous un programme d'affectation fixe. Quelle interface de modèle
cela fournit-il, et à quoi ne peut-on pas répondre directement au sujet d'une
nouvelle politique de rétroaction?
:::

:::{solution} ex-dynamics-modeling-dispatch
:class: dropdown

(a) La frontière contient les deux files, les quatre unités et leur
fonctionnement en ce qui touche la puissance. Un état complet contient les
tâches en file, l'unité affectée et le temps de traitement restant de chaque
tâche active, les modes des unités et toute variable nécessaire pour prédire
la puissance. L'observation donnée contient les deux longueurs de files, le
nombre d'unités occupées, les modes des unités et la puissance mesurée.
L'action est le nombre d'unités actives et la priorité entre files. Les
arrivées et les besoins non observés des tâches sont des perturbations. Le
coût d'étape combine l'attente et l'énergie, et le plafond de puissance est
une contrainte stricte. Le temps est discret, par pas d'une minute, sur 60
décisions.

(b) Des comptes égaux peuvent masquer des charges de travail restantes
différentes, et donc des lois d'achèvement différentes. L'état complet décrit
ci-dessus est markovien sous le modèle de tâches déclaré. L'historique complet
des observations et des actions est accessible au répartiteur; sous un modèle
stochastique connu, sa loi a posteriori sur les états cachés des tâches est un
état d'information plus compact.

(c) Un noyau de transition stochastique est la représentation naturelle, car
les tâches arrivent à des instants aléatoires et des besoins de traitement
cachés rendent les achèvements incertains. Une fonction déterministe n'est
possible qu'une fois les réalisations pertinentes du bruit fournies comme
entrées supplémentaires.

(d) Le programme fixe choisit les 60 actions à partir de l'information
initiale. Une politique de rétroaction peut modifier le nombre d'unités
actives ou la priorité après avoir observé de nouvelles longueurs de files,
l'occupation des unités, leurs modes ou la puissance.

(e) Ces données fournissent une interface de transitions historiques sous un
seul programme de comportement. Elles ne permettent pas de dire directement
comment des actions absentes de ce journal modifieraient les files ou la
puissance; évaluer une nouvelle politique de rétroaction exige donc des
hypothèses de couverture et d'identification, ou un modèle de transition
distinct.
:::

:::{exercise} Exercices sous maîtrise latente
:label: ex-dynamics-modeling-practice

Un système d'exercices en ligne présente un problème à chacune de 20 étapes.
Avant l'étape $t$, la maîtrise non observée $M_t$ d'un apprenant appartient à
l'une de trois catégories ordonnées. Le système choisit un problème facile,
moyen ou difficile $A_t$, puis observe l'exactitude de la réponse et le temps
de réponse $Y_t$. La loi de l'observation ne dépend que de $M_t$ et de $A_t$.
La maîtrise change ensuite aléatoirement selon une loi qui ne dépend que de
$M_t$ et de $A_t$, ce qui produit $M_{t+1}$. L'objectif est de maximiser la
probabilité que $M_{20}$, après la vingtième transition, soit la catégorie la
plus élevée, sous la contrainte de présenter au plus cinq problèmes
difficiles. Un simulateur réinitialisable peut échantillonner une transition et
une observation à partir de tout niveau de maîtrise et de toute difficulté
fournis, mais il n'expose ni équations ni dérivées.

(a) Classez l'état latent, l'observation, l'action, la dynamique stochastique,
l'objectif, la contrainte, l'horizon et l'interface de modèle. Incluez toute
variable comptable nécessaire pour faire respecter la limite de cinq problèmes
difficiles.

(b) Comparez trois entrées possibles de la règle de décision : la réponse la
plus récente, l'historique complet des observations et la loi a posteriori sur
les niveaux de maîtrise. Laquelle est un état d'information sous les relations
conditionnelles énoncées? Expliquez pourquoi.

(c) Écrivez la forme structurelle de la transition d'état, du modèle
d'observation et de la politique de rétroaction sans inventer de probabilités
numériques.

(d) Décrivez une suite de problèmes en boucle ouverte et une suite de
problèmes par rétroaction. Quelle partie de l'information disponible peut
modifier la seconde?

(e) Nommez un calcul que le simulateur réinitialisable permet directement, et
une opération qui exigerait une interface supplémentaire.
:::

:::{solution} ex-dynamics-modeling-practice
:class: dropdown

(a) L'état latent augmenté est $(M_t,c_t,t)$, où $c_t$ compte les problèmes
difficiles déjà utilisés. L'observation est l'exactitude et le temps de
réponse, et l'action est la difficulté. Le modèle comporte une transition
stochastique de la maîtrise et une fonction d'observation stochastique.
L'objectif terminal est $\Pr(M_{20}=\text{plus élevée})$, avec $c_t\leq5$
comme contrainte stricte. L'horizon contient 20 actions. Le modèle disponible
est un simulateur génératif par réinitialisation et échantillonnage.

(b) La dernière réponse seule perd en général de l'information sur les étapes
précédentes. L'historique complet suffit, mais il croît avec le temps. Sous
les relations conditionnelles énoncées, la loi a posteriori
$b_t(m)=\Pr(M_t=m\mid Y_{0:t-1},A_{0:t-1})$, augmentée de $c_t$ et de $t$, est
un état d'information markovien.

(c) Sans attribuer de valeurs numériques, la structure est

$$
Y_t\sim O(\,\cdot\mid M_t,A_t),\qquad
M_{t+1}\sim P(\,\cdot\mid M_t,A_t),\qquad
c_{t+1}=c_t+\mathbf 1\{A_t=\mathrm{difficile}\},
$$

avec une politique de rétroaction réalisable $A_t=\pi_t(b_t,c_t)$ qui exclut
l'action difficile dès que $c_t=5$.

(d) Une suite en boucle ouverte fixe les 20 difficultés avant de voir la
moindre réponse. Une suite par rétroaction choisit la difficulté suivante
après avoir mis à jour la loi a posteriori de la maîtrise à partir de
l'exactitude et du temps de réponse, tout en tenant compte du budget restant
de problèmes difficiles.

(e) Le simulateur permet directement des échantillons arbitraires à un pas et
des déroulements de Monte-Carlo à partir d'états latents et d'actions choisis.
Les probabilités exactes, les dérivées analytiques ou les équations locales
symboliques exigent une interface supplémentaire.
:::
