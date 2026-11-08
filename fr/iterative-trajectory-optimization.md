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

# iLQR et programmation dynamique différentielle

Un bateau qui approche d'un quai doit arriver au bon endroit, s'orienter le
long du poste d'amarrage et perdre l'élan qui lui reste. Une suite de
commandes de poussée détermine ces trois résultats. La
[formulation par tir](numerical-trajectory-optimization.md) transforme cette
tâche en une optimisation sur ces commandes : simuler une suite candidate,
mesurer son coût et l'améliorer. Comment un optimiseur peut-il exploiter la
structure temporelle de la simulation pour calculer cette amélioration?

Linéariser la dynamique le long de la trajectoire courante et approcher ses
coûts par des fonctions quadratiques donne un problème dont on peut éliminer
les commandes à rebours, un pas de temps à la fois. La correction obtenue est
ensuite mise à l'épreuve dans le simulateur non linéaire. Répéter ces
opérations donne la **commande linéaire quadratique itérative**, ou **iLQR**.
Conserver en plus la courbure de la dynamique donne la **programmation
dynamique différentielle**, ou **DDP**. Les deux méthodes calculent des
corrections autour du plan courant. Le problème local est la
[commande linéaire quadratique](#sec-lqr-dp) résolue par programmation
dynamique au chapitre précédent, et sa récurrence arrière peut aussi se lire
comme une suite de substitutions et de minimisations quadratiques.

Ce chapitre fait le pont entre la programmation dynamique exacte du chapitre
précédent et la [programmation dynamique stochastique](stochastic-dp.md) du
suivant. L'iLQR réutilise la récurrence arrière de la commande linéaire
quadratique, mais seulement localement : ses modèles quadratiques et ses gains
de rétroaction décrivent un voisinage d'une trajectoire, et non l'espace
d'états tout entier.

## Objectifs d'apprentissage

Après avoir suivi les dérivations et les exemples, vous devriez pouvoir :

- Construire un problème local de contrôle optimal quadratique autour d'un déroulement réalisable.
- Éliminer ses commandes à rebours et obtenir une correction affine à chaque pas.
- Implémenter des déroulements d'essai non linéaires avec recherche linéaire et régularisation.
- Repérer les termes de courbure de la dynamique qui distinguent la DDP de l'iLQR.
- Évaluer un plan d'accostage à partir de son coût, de sa vitesse d'arrivée et du dégagement de sa coque.
- Expliquer comment la charge thermique et les pertes de puissance non linéaires façonnent un calendrier de refroidissement borné.
- Relier la forme d'un calendrier de radiothérapie à la réparation des organes et à la courbure que conserve la DDP.

## Préalables

Le chapitre utilise la [formulation du contrôle optimal à horizon fini](discrete-time-optimal-control.md),
le [tir simple](numerical-trajectory-optimization.md), la
[commande linéaire quadratique](#sec-lqr-dp) issue de la programmation
dynamique à horizon fini, ainsi que les dérivées premières et secondes. La
[section sur la PQS de l'annexe de programmation non linéaire](#sec-sqp-approach)
présente les sous-problèmes généraux de PQS, et l'[annexe sur les problèmes à valeur initiale](appendix_ivps.md)
revoit l'intégration numérique.

(sec-boat-docking)=
## Un problème d'accostage

L'accostage exige que le bateau atteigne le poste d'amarrage, s'aligne sur le
quai et se débarrasse de son élan. Si le bateau atteint la bonne position avec
une vitesse de translation ou une vitesse angulaire appréciable, son élan
l'emportera de nouveau.

Considérons un bateau de deux mètres qui se déplace dans le plan horizontal.
Son état et sa commande sont

$$
\mathbf x=(p_x,p_y,\psi,v_x,v_y,\omega),\qquad
\mathbf u=(u_L,u_R)\in[-1,1]^2.
$$

La position et la vitesse sont mesurées dans des coordonnées fixes du monde,
$\psi$ est le cap et $\omega$ la vitesse angulaire. Chaque commande est une
fraction de la poussée maximale. Les commandes positives poussent vers l'avant
et les commandes négatives vers l'arrière. Des poussées inégales font tourner
le bateau. Aucune force latérale n'est commandée directement, si bien qu'une
approche avec un élan latéral exige une manœuvre.

Soit $\mathbf e(\psi)=(\cos\psi,\sin\psi)^\top$ le vecteur orienté le long du
bateau et $\mathbf n(\psi)=(-\sin\psi,\cos\psi)^\top$ le vecteur qui le
traverse. Le modèle inertiel simplifié est

$$
\begin{aligned}
\dot{\mathbf p}&=\mathbf v,&\qquad \dot\psi&=\omega,\\
m\dot{\mathbf v}
&=F_{\max}(u_L+u_R)\mathbf e
  -d_\parallel(\mathbf e^\top\mathbf v)\mathbf e
  -d_\perp(\mathbf n^\top\mathbf v)\mathbf n,\\
I\dot\omega&=bF_{\max}(u_R-u_L)-d_\omega\omega.
\end{aligned}
$$

La traînée s'oppose à la vitesse le long de chaque axe du bateau, avec une
résistance plus forte au mouvement latéral. La poussée modifie
l'accélération; relâcher les commandes n'arrête donc pas le bateau
immédiatement. Ces équations forment un modèle pédagogique synthétique pour
une eau calme, sans forces de contact ni vagues.

| Grandeur | Valeur |
| :--- | ---: |
| Masse $m$ | $50\ \mathrm{kg}$ |
| Inertie angulaire $I$ | $20\ \mathrm{kg\,m^2}$ |
| Bras de levier des propulseurs $b$ | $0.4\ \mathrm m$ |
| Force maximale par propulseur $F_{\max}$ | $25\ \mathrm N$ |
| Traînée longitudinale $d_\parallel$ | $8\ \mathrm{N\,s/m}$ |
| Traînée latérale $d_\perp$ | $30\ \mathrm{N\,s/m}$ |
| Traînée angulaire $d_\omega$ | $10\ \mathrm{N\,m\,s}$ |
| Longueur et largeur de la coque rectangulaire | $2\ \mathrm m$, $0.8\ \mathrm m$ |

Le quai occupe la région $p_y\leq0$, et l'état d'arrivée souhaité est
$\mathbf x_\star=(0,1,0,0,0,0)$. Le bateau doit se trouver parallèle au quai,
son centre à un mètre du bord. L'approche oblique part de
$(-8,5,-0.5,0.2,-0.1,0)$. Le cas de la dérive latérale ne change que $v_y$,
qui passe à $-0.5\ \mathrm{m/s}$. Les angles sont en radians. Les deux
expériences utilisent $T-1=200$ commandes constantes par morceaux, chacune
maintenue pendant $h=0.1\ \mathrm s$, pour un total de 20 secondes. Ainsi,
$\mathbf x_1$ est l'état initial et $\mathbf x_T$ l'état d'arrivée, comme dans
les chapitres précédents. Un pas de Runge--Kutta d'ordre quatre donne
$\mathbf x_{t+1}=\mathbf f_t(\mathbf x_t,\mathbf u_t)$ pour
$t=1,\ldots,T-1$.

Pour dissuader la coque de s'approcher du quai, soit $y_j(\mathbf x)$ la
hauteur, en coordonnées du monde, du coin $j$ du rectangle. Avec une marge
souhaitée $a=0.2\ \mathrm m$ et une longueur de lissage $s=0.08\ \mathrm m$,
on définit

$$
\rho(z)=s\log(1+\exp(z/s)),\qquad
\mathcal P(\mathbf x)=200\sum_{j=1}^4\rho(a-y_j(\mathbf x))^2.
$$

Il s'agit d'une pénalité lisse pour la violation de la marge. Son poids fini
permet des violations; les trajectoires finales feront donc aussi l'objet
d'une vérification géométrique distincte du dégagement. Cette vérification
utilise le rectangle complet, y compris la zone située hors de l'icône pointue
du bateau dans les figures.

Pour $\mathbf e_p=\mathbf p-(0,1)^\top$, les coûts courant et terminal sont

$$
\begin{aligned}
c_t(\mathbf x,\mathbf u)
&=h\big[0.03\|\mathbf e_p\|^2+0.1(1-\cos\psi)
       +0.1\|\mathbf v\|^2+0.1\omega^2
       +0.2\|\mathbf u\|^2+\mathcal P(\mathbf x)\big],\\
c_T(\mathbf x)
&=300\|\mathbf e_p\|^2+100(1-\cos\psi)
       +200\|\mathbf v\|^2+100\omega^2+\mathcal P(\mathbf x).
\end{aligned}
$$

Ces poids numériques s'appliquent aux coordonnées SI indiquées et aux
commandes de poussée adimensionnelles. La pénalité périodique sur le cap
attribue le même coût à des angles qui diffèrent d'un tour complet. Le
simulateur conserve un angle non ramené dans un intervalle; il n'introduit
pas de discontinuité en $\pm\pi$ dans la transition que l'on dérive.

La suite de commandes initiale est nulle : le bateau glisse donc sur son
erre et ralentit sous l'effet de la traînée. Il s'arrête loin du poste
d'amarrage. Des mises à jour successives des commandes peuvent d'abord l'en
rapprocher, puis ajuster le cap et la vitesse d'arrivée.

:::{figure} _static/boat_docking/iterations.svg
:label: fig-boat-iterations
:width: 100%

Itérés acceptés de l'iLQR pour l'approche oblique. Chaque courbe est un
déroulement non linéaire depuis le même état initial. Les silhouettes du
bateau sont espacées de quatre secondes; leur chevauchement signale un
mouvement lent. La silhouette en pointillé marque le poste d'amarrage
souhaité. La première mise à jour réduit fortement l'erreur de position,
tandis que les suivantes affinent le virage et la manœuvre d'arrêt.
:::

(sec-local-quadratic-trajectory)=
## Modèles de Taylor de la dynamique et du coût

Le modèle local décrit la variation du coût causée par une petite variation de
la suite de poussées. Partons du problème à horizon fini

$$
\begin{aligned}
\min_{\mathbf U}\quad &J(\mathbf U)
 :=c_T(\mathbf x_T)+\sum_{t=1}^{T-1}c_t(\mathbf x_t,\mathbf u_t),\\
\text{s.c.}\quad &\mathbf x_{t+1}=\mathbf f_t(\mathbf x_t,\mathbf u_t),
\quad t=1,\ldots,T-1,\\
&\mathbf x_1=\mathbf x_{\mathrm{init}}.
\end{aligned}
$$

Ici, $\mathbf U=(\mathbf u_1,\ldots,\mathbf u_{T-1})$ est la suite complète des
commandes. Son déroulement détermine tous les états ultérieurs. Il s'agit d'un
problème en boucle ouverte : la variable de décision est une suite de
commandes fixée d'avance, et non une règle qui réagit à l'état. L'iLQR renvoie
une telle suite. Les gains de rétroaction $K_t$ qui apparaissent plus bas sont
un sous-produit du calcul. Près du plan renvoyé, ils indiquent comment ajuster
chaque commande lorsque l'état s'écarte de l'état planifié, au moyen de
$\mathbf u=\bar{\mathbf u}+K(\mathbf x-\bar{\mathbf x})$, et ils ne disent
rien des états qui en sont éloignés.

La {numref}`fig-boat-tube` montre ce que signifie « près du plan ». Le bateau
suit le plan d'accostage final et reçoit une petite poussée extérieure après
deux secondes. Rejouer les poussées planifiées ignore ce choc. Les bateaux
dérivent calmement, mais aucun n'atteint le poste d'amarrage : ils le manquent
de 0.9 m en moyenne et jusqu'à 2.2 m. Appliquer la rétroaction locale ramène
les douze déroulements perturbés sur le plan, et ils arrivent à moins d'un
millimètre de celui-ci. Ces déroulements restent à l'intérieur d'un tube mince
autour de la trajectoire planifiée.

Le tube a une largeur limitée. Avec des chocs deux fois plus forts, montrés
dans l'animation ci-dessous, plusieurs déroulements avec rétroaction en
sortent et deux n'atteignent jamais le poste d'amarrage. Leurs corrections
demandent plus de poussée que les propulseurs ne peuvent en fournir, et le
bateau s'éloigne assez du plan pour que le modèle linéaire cesse d'être
exact. Les gains décrivent un voisinage d'une trajectoire, et tout ce chapitre
travaille à l'intérieur de tels voisinages.

:::{figure} _static/boat_docking/tube.svg
:label: fig-boat-tube
:width: 100%

Douze chocs aléatoires à 2 s (variations de vitesse allant jusqu'à 0.3 m/s
dans chaque direction, avec de petites variations de cap et de rotation)
appliqués au plan iLQR final de l'approche oblique. À gauche : les poussées
planifiées sont rejouées sans changement. À droite : les poussées suivent la
loi de rétroaction locale, écrêtée aux limites des propulseurs. Le tube
ombré, tracé dans les deux panneaux, est l'enveloppe des déroulements avec
rétroaction. Les cercles marquent les positions à 20 s.
:::

:::{iframe} ../interactive/boat-tube.html
:width: 100%
:title: Bateaux perturbés avec et sans rétroaction locale
:class: boat-tube-replay

Lancez la simulation ou parcourez-la dans le temps, et passez aux chocs plus
forts pour voir des bateaux sortir du tube. Les compteurs indiquent combien de
bateaux se trouvent dans le tube et au poste d'amarrage à l'instant choisi.
:::

On ne peut pas résoudre directement ce problème non linéaire. Le chapitre
précédent a montré qu'un problème de commande linéaire quadratique se résout
exactement par une récurrence arrière. L'idée de ce chapitre est de
construire un problème de commande linéaire quadratique qui ressemble au
nôtre près d'un plan courant, de le résoudre, d'utiliser sa solution pour
améliorer le plan, puis de recommencer.

Laissons de côté les bornes sur les commandes pendant la dérivation de ce
problème local. Le **plan courant** est une suite de commandes
$\bar{\mathbf U}$ accompagnée des états obtenus en la déroulant,
$\bar{\mathbf x}_{t+1}=\mathbf f_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)$.
Pour le bateau, le premier plan est la glissade sans poussée. Le plan est
notre estimation courante, et non une référence que le bateau doit suivre.
Pour se demander ce qui arrive si on le modifie, considérons des variations
$\delta\mathbf x_t=\mathbf x_t-\bar{\mathbf x}_t$ et
$\delta\mathbf u_t=\mathbf u_t-\bar{\mathbf u}_t$, empilées sous la forme
$\delta\mathbf z_t=(\delta\mathbf x_t,\delta\mathbf u_t)$. Ce sont des
changements hypothétiques que l'on choisit librement, et non des erreurs
mesurées.

Le modèle linéaire de la transition suit les étapes habituelles du
développement de Taylor. Une fonction scalaire d'une variable vérifie
$f(a+h)\approx f(a)+f'(a)h$. Avec deux entrées, les dérivées partielles
s'additionnent :
$f(\bar x+\delta x,\bar u+\delta u)\approx f(\bar x,\bar u)
+\partial_x f\,\delta x+\partial_u f\,\delta u$. La transition $\mathbf f_t$
possède un tel développement pour chacune de ses $n_x$ composantes. Chaque
composante donne une ligne, et empiler les lignes donne les matrices
jacobiennes $A_t$ et $B_t$ :

$$
\begin{aligned}
\mathbf f_t(\bar{\mathbf x}_t+\delta\mathbf x_t,
             \bar{\mathbf u}_t+\delta\mathbf u_t)
&=\mathbf f_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)
  +A_t\delta\mathbf x_t+B_t\delta\mathbf u_t
  +O(\|\delta\mathbf z_t\|^2),\\
A_t&=\left.\frac{\partial\mathbf f_t}{\partial\mathbf x}
      \right|_{(\bar{\mathbf x}_t,\bar{\mathbf u}_t)},\\
B_t&=\left.\frac{\partial\mathbf f_t}{\partial\mathbf u}
      \right|_{(\bar{\mathbf x}_t,\bar{\mathbf u}_t)}.
\end{aligned}
$$

Le membre de gauche est la transition évaluée à l'état et à la commande
modifiés, c'est-à-dire le nouvel état suivant $\mathbf x_{t+1}$. Comme le plan
courant est un déroulement, le premier terme du membre de droite est
$\mathbf f_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)=\bar{\mathbf x}_{t+1}$.
Soustraire $\bar{\mathbf x}_{t+1}$ des deux membres transforme le membre de
gauche en $\delta\mathbf x_{t+1}$ et annule ce terme à droite. Négliger le
reste quadratique donne la dynamique linéarisée
$\delta\mathbf x_{t+1}=A_t\delta\mathbf x_t+B_t\delta\mathbf u_t$.
Les deux déroulements partent du même état, donc $\delta\mathbf x_1=\mathbf0$.
Il n'y a pas de défaut constant, car le plan courant satisfait la dynamique
discrète. Des estimations d'états choisies indépendamment laisseraient en
général un tel défaut. On note $F_t=[A_t\ B_t]$ la jacobienne combinée.

Le coût courant se développe selon les mêmes étapes, un ordre plus loin. En
une variable, $f(a+h)\approx f(a)+f'(a)h+\tfrac12f''(a)h^2$. Appliquée au
vecteur empilé $\delta\mathbf z_t$, la dérivée devient un gradient et la
dérivée seconde une matrice hessienne. Séparer de nouveau $\delta\mathbf z_t$
en ses parties état et commande découpe le gradient en $c_{x,t}$ et
$c_{u,t}$, et la hessienne en quatre blocs : $c_{xx,t}$ pour la courbure en
l'état, $c_{uu,t}$ pour la courbure en la commande, et les blocs mixtes
$c_{xu,t}$ et $c_{ux,t}=c_{xu,t}^\top$. Le terme quadratique s'écrit donc
$\tfrac12\delta\mathbf x_t^\top c_{xx,t}\delta\mathbf x_t
+\delta\mathbf u_t^\top c_{ux,t}\delta\mathbf x_t
+\tfrac12\delta\mathbf u_t^\top c_{uu,t}\delta\mathbf u_t$, où les deux blocs
mixtes se sont combinés en un seul terme. Sous forme de matrice par blocs,

$$
c_t(\bar{\mathbf x}_t+\delta\mathbf x_t,
    \bar{\mathbf u}_t+\delta\mathbf u_t)
 = \bar c_t+
\begin{bmatrix}c_{x,t}\\c_{u,t}\end{bmatrix}^{\!\top}\delta\mathbf z_t
+\frac12\delta\mathbf z_t^\top
\begin{bmatrix}c_{xx,t}&c_{xu,t}\\c_{ux,t}&c_{uu,t}\end{bmatrix}
\delta\mathbf z_t+o(\|\delta\mathbf z_t\|^2).
$$

Ici, $\bar c_t=c_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)$ et
$c_{x,t}=\nabla_{\mathbf x}c_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)$;
les autres blocs du gradient et de la hessienne sont évalués au même couple du
plan courant.
On note $c_{z,t}$ le gradient empilé et $c_{zz,t}$ la hessienne par blocs
affichée. Les gradients sont des vecteurs colonnes, conformément à la
convention des chapitres précédents. On conserve les termes linéaires : les
gradients de chaque étape n'ont pas à s'annuler, même sur une trajectoire
optimale, car la dynamique couple les étapes; et pour un plan non optimal, ce
sont aussi eux qui pilotent sa correction.
Le bloc $c_{xu,t}$ mesure comment le coût marginal d'une commande varie avec
l'état.
Même si le coût courant ne comporte aucun terme croisé de ce genre,
l'élimination des commandes ultérieures peut en créer un dans la queue
quadratique.

Le coût terminal admet le même développement, sans coordonnée de commande :

$$
\begin{aligned}
c_T(\bar{\mathbf x}_T+\delta\mathbf x_T)
&=c_T(\bar{\mathbf x}_T)+p_T^\top\delta\mathbf x_T\\
&\quad+\frac12\delta\mathbf x_T^\top P_T\delta\mathbf x_T
 +o(\|\delta\mathbf x_T\|^2),\\
p_T&=\nabla_{\mathbf x}c_T(\bar{\mathbf x}_T),\qquad
P_T=\nabla^2_{\mathbf x\mathbf x}c_T(\bar{\mathbf x}_T).
\end{aligned}
$$

Négliger les restes de ces trois développements de Taylor donne le problème
local. À la constante $\sum_t\bar c_t+c_T(\bar{\mathbf x}_T)$ près, qui ne
dépend pas des variations, il s'écrit

$$
\begin{aligned}
\min_{\delta\mathbf u_1,\ldots,\delta\mathbf u_{T-1}}\quad
&\sum_{t=1}^{T-1}\Big(c_{z,t}^\top\delta\mathbf z_t
  +\frac12\delta\mathbf z_t^\top c_{zz,t}\,\delta\mathbf z_t\Big)
 +p_T^\top\delta\mathbf x_T
 +\frac12\delta\mathbf x_T^\top P_T\,\delta\mathbf x_T,\\
\text{s.c.}\quad
&\delta\mathbf x_{t+1}=A_t\delta\mathbf x_t+B_t\delta\mathbf u_t,
 \quad t=1,\ldots,T-1,\\
&\delta\mathbf x_1=\mathbf0.
\end{aligned}
$$ (eq-ilqr-local-problem)

C'est un problème de commande linéaire quadratique du type résolu au chapitre
précédent : une dynamique linéaire en les variations, et un coût quadratique
en celles-ci. Par rapport au problème linéaire quadratique standard, il
comporte des termes de coût linéaires et un terme croisé état--commande, que
la section suivante traite. Le résoudre donne une modification proposée du
plan.

Remarquons ce qui manque à ce problème local. Seuls les coûts ont été
développés au second ordre. La dynamique a été développée au premier ordre,
si bien que sa courbure n'apparaît nulle part dans
{eq}`eq-ilqr-local-problem`. Une [méthode de PQS](#sec-sqp-approach) générale
forme aussi un sous-problème quadratique local, mais pas tout à fait
celui-ci. La PQS prend un modèle quadratique du lagrangien et linéarise les
contraintes. Ici, les contraintes sont la dynamique, de sorte que la hessienne
du lagrangien est la hessienne du coût plus la somme sur $i$ du
multiplicateur $\lambda_i$ multiplié par la hessienne de la $i$e composante de
la transition. La formulation par tir n'est sans contraintes que par rapport
aux commandes, parce que le simulateur impose la dynamique, mais le problème
local ci-dessus garde la dynamique linéarisée comme contraintes sur les
variations. L'iLQR est donc une PQS dont on a retiré les hessiennes de la
dynamique pondérées par les multiplicateurs. Ces termes reviennent dans la
[section sur la DDP](#sec-ddp-curvature), où le coefficient $p_{t+1}$ de la
queue éliminée joue le rôle du multiplicateur, l'état adjoint.

## Élimination arrière

Le problème local {eq}`eq-ilqr-local-problem` est un problème déterministe à
horizon fini, à dynamique linéaire et à coûts quadratiques. C'est un problème
de commande linéaire quadratique, que l'induction arrière du chapitre
précédent résout donc. Cette section rappelle cet algorithme, le spécialise
au cas linéaire quadratique, puis applique la même spécialisation au problème
local. Chaque étape arrière minimise par rapport à une commande et la retire
du problème; c'est pourquoi la passe arrière de l'iLQR est souvent appelée
élimination arrière.

### Induction arrière

Pour un problème déterministe de dynamique $\mathbf x_{t+1}=\mathbf f_t(\mathbf x_t,\mathbf u_t)$,
de coûts d'étape $c_t$ et de coût terminal $c_T$,
{prf:ref}`backward-recursion-deterministic` calcule le coût restant optimal
$J_t^\star$ à rebours dans le temps :

$$
\begin{aligned}
J_T^\star(\mathbf x)&=c_T(\mathbf x),\\
J_t^\star(\mathbf x)&=\min_{\mathbf u}\Big[c_t(\mathbf x,\mathbf u)
  +J_{t+1}^\star\big(\mathbf f_t(\mathbf x,\mathbf u)\big)\Big],
  \qquad t=T-1,\ldots,1.
\end{aligned}
$$

Le minimiseur en chaque état définit la politique $\pi_t^\star(\mathbf x)$.
Une passe avant depuis l'état initial applique ensuite
$\mathbf u_t=\pi_t^\star(\mathbf x_t)$ et retrouve la trajectoire optimale. En
général, $J_t^\star$ est une fonction définie sur tout l'espace d'états, que le
chapitre précédent stockait sur une grille.

### Le cas linéaire quadratique : la récurrence de Riccati

Avec une dynamique linéaire $\mathbf x_{t+1}=A_t\mathbf x_t+B_t\mathbf u_t$, un
coût d'étape $\tfrac12\mathbf x^\top Q_t\mathbf x+\tfrac12\mathbf u^\top R_t\mathbf u$
et un coût terminal $\tfrac12\mathbf x^\top Q_T\mathbf x$, chaque coût restant
est quadratique, $J_t^\star(\mathbf x)=\tfrac12\mathbf x^\top P_t\mathbf x$, de
sorte que l'étape arrière n'a qu'une matrice à mettre à jour. Substituer la
dynamique dans $J_{t+1}^\star$ fait du crochet une fonction quadratique de
l'état et de la commande empilés,

$$
\frac12
\begin{bmatrix}\mathbf x\\\mathbf u\end{bmatrix}^{\!\top}
\begin{bmatrix}
Q_t+A_t^\top P_{t+1}A_t & A_t^\top P_{t+1}B_t\\
B_t^\top P_{t+1}A_t & R_t+B_t^\top P_{t+1}B_t
\end{bmatrix}
\begin{bmatrix}\mathbf x\\\mathbf u\end{bmatrix}.
$$

Annuler son gradient par rapport à la commande donne la rétroaction linéaire
$\mathbf u_t^\star=-K_t\mathbf x$, et la substituer en retour donne la
**récurrence de Riccati** de {prf:ref}`backward-recursion-lqr` :

$$
\begin{aligned}
K_t&=(R_t+B_t^\top P_{t+1}B_t)^{-1}B_t^\top P_{t+1}A_t,\\
P_t&=Q_t+A_t^\top P_{t+1}A_t
 -A_t^\top P_{t+1}B_t(R_t+B_t^\top P_{t+1}B_t)^{-1}B_t^\top P_{t+1}A_t,
\end{aligned}
$$

à partir de $P_T=Q_T$.

### Riccati pour le problème local

Le problème local a la même structure, avec les variations
$\delta\mathbf x_t,\delta\mathbf u_t$ comme état et commande. Son coût restant
terminal est le modèle de Taylor terminal, et son étape arrière est

$$
S_t(\delta\mathbf x)=\min_{\delta\mathbf u}\Big[
c_{z,t}^\top\delta\mathbf z+\frac12\delta\mathbf z^\top c_{zz,t}\,\delta\mathbf z
+S_{t+1}(A_t\delta\mathbf x+B_t\delta\mathbf u)\Big],
\qquad \delta\mathbf z=(\delta\mathbf x,\delta\mathbf u).
$$

Deux caractéristiques diffèrent de la commande linéaire quadratique des
manuels. Le plan courant n'est généralement pas optimal, si bien que les coûts
locaux comportent des termes linéaires, les gradients $c_{z,t}$ et $p_T$. Le
coût d'étape peut aussi contenir un terme croisé état--commande $c_{ux,t}$. À
cause des termes linéaires, le coût restant a lui aussi besoin d'une partie
linéaire :

$$
S_{t+1}(\delta\mathbf x)=s_{t+1}+p_{t+1}^\top\delta\mathbf x
+\frac12\delta\mathbf x^\top P_{t+1}\delta\mathbf x,
$$

où $p_T$ et $P_T$ sont le gradient et la hessienne de $c_T$ à la fin du plan.
Substituer la dynamique linéaire dans $S_{t+1}$ fait du crochet une fonction
quadratique de $\delta\mathbf z$, de coefficient linéaire $q_t$ et de
hessienne $M_t$. Écrits par blocs, à côté de la matrice linéaire quadratique
ci-dessus,

$$
\begin{aligned}
q_{x,t}&=c_{x,t}+A_t^\top p_{t+1}, &
M_{xx,t}&=c_{xx,t}+A_t^\top P_{t+1}A_t,\\
q_{u,t}&=c_{u,t}+B_t^\top p_{t+1}, &
M_{ux,t}&=c_{ux,t}+B_t^\top P_{t+1}A_t,\\
& & M_{uu,t}&=c_{uu,t}+B_t^\top P_{t+1}B_t.
\end{aligned}
$$

Les blocs de la hessienne sont les blocs du cas linéaire quadratique, avec
$Q_t$ remplacé par $c_{xx,t}$, $R_t$ par $c_{uu,t}$, et le terme croisé
$c_{ux,t}$ ajouté. Le vecteur $q_t$ est nouveau. Sous forme empilée, avec
$F_t=[A_t\ B_t]$, on a
$q_t=c_{z,t}+F_t^\top p_{t+1}$ et $M_t=c_{zz,t}+F_t^\top P_{t+1}F_t$. On
emploie $M_t$ pour cette hessienne parce que $H_t$ désignait le hamiltonien
dans le chapitre sur Pontryaguine.

Annuler le gradient par rapport à la commande, $M_{uu,t}\delta\mathbf u+M_{ux,t}\delta\mathbf x+q_{u,t}=0$,
donne une loi affine plutôt que linéaire :

$$
\delta\mathbf u_t=k_t+K_t\delta\mathbf x_t,\qquad
k_t=-M_{uu,t}^{-1}q_{u,t},\qquad K_t=-M_{uu,t}^{-1}M_{ux,t}.
$$

Le gain de rétroaction $K_t$ joue le rôle du gain de la commande linéaire
quadratique, écrit ici signe compris. Le terme d'anticipation $k_t$ provient
des termes linéaires : il modifie le plan même lorsque la variation d'état est
nulle. Substituer la commande en retour donne les coefficients du pas
précédent,

$$
\begin{aligned}
p_t&=q_{x,t}-M_{ux,t}^\top M_{uu,t}^{-1}q_{u,t},\\
P_t&=M_{xx,t}-M_{ux,t}^\top M_{uu,t}^{-1}M_{ux,t}.
\end{aligned}
$$

La mise à jour de $P_t$ est la récurrence de Riccati. Développée, elle s'écrit
$P_t=c_{xx,t}+A_t^\top P_{t+1}A_t-(c_{ux,t}+B_t^\top P_{t+1}A_t)^\top
(c_{uu,t}+B_t^\top P_{t+1}B_t)^{-1}(c_{ux,t}+B_t^\top P_{t+1}A_t)$, ce qui est
la formule de la commande linéaire quadratique lorsque $c_{ux,t}=0$. La mise à
jour de $p_t$ est la partie nouvelle. Elle transporte les gradients du coût à
rebours et, avec des gradients nuls, elle reste nulle et $k_t$ s'annule. Dans
le code, ces formules sont des résolutions de systèmes linéaires en
$M_{uu,t}$; il n'est pas nécessaire de calculer explicitement l'inverse d'une
matrice.

````{prf:algorithm} Passe arrière pour le problème local
:label: alg-ilqr-backward-pass

**Entrée :** les jacobiennes $A_t,B_t$ et les dérivées du coût $c_{z,t},c_{zz,t}$ le long du plan courant; le gradient terminal $p_T$ et la hessienne terminale $P_T$.

**Sortie :** les termes d'anticipation $k_t$ et les gains $K_t$ pour $t=1,\ldots,T-1$.

1. Pour $t=T-1,\ldots,1$ :
   1. Former $q_t=c_{z,t}+F_t^\top p_{t+1}$ et $M_t=c_{zz,t}+F_t^\top P_{t+1}F_t$.
   2. Résoudre $k_t=-M_{uu,t}^{-1}q_{u,t}$ et $K_t=-M_{uu,t}^{-1}M_{ux,t}$.
   3. Mettre à jour $p_t=q_{x,t}-M_{ux,t}^\top M_{uu,t}^{-1}q_{u,t}$ et $P_t=M_{xx,t}-M_{ux,t}^\top M_{uu,t}^{-1}M_{ux,t}$.
2. Renvoyer $k_t$ et $K_t$. Une passe avant depuis $\delta\mathbf x_1=\mathbf0$ avec $\delta\mathbf u_t=k_t+K_t\delta\mathbf x_t$ et $\delta\mathbf x_{t+1}=A_t\delta\mathbf x_t+B_t\delta\mathbf u_t$ résout le problème local.
````

La queue $S_t$ est le coût restant du problème local uniquement. Elle est
exacte près du plan courant, et ce n'est pas une fonction de valeur du système
non linéaire sur tout son espace d'états. Contrairement à la grille du
chapitre précédent, elle est stockée sous la forme d'un vecteur et d'une
matrice par pas, si bien que la passe arrière demande une quantité de travail
fixe par pas et croît linéairement avec l'horizon. Pour une dynamique
linéaire et des coûts quadratiques, elle résout exactement le problème
original dès que chaque $M_{uu,t}$ est définie positive. Le long d'une
trajectoire non linéaire, $A_t$, $B_t$ et les dérivées du coût ne décrivent
qu'une approximation locale. Il faut les recalculer après chaque modification
du plan. L'{ref}`ex-ilqr-two-step` effectue une passe arrière à la main sur un
problème scalaire.

(sec-ilqr-rollout)=
## Déroulements non linéaires et iLQR

La passe arrière renvoie, à chaque pas, une loi affine
$\delta\mathbf u_t=k_t+K_t\delta\mathbf x_t$ pour le problème local. L'iLQR la
transforme en un nouveau plan et recommence :

1. **Dérouler.** Simuler les commandes courantes $\bar{\mathbf U}$ dans la
   vraie dynamique. On obtient le plan courant $\bar{\mathbf x}_t$,
   $\bar{\mathbf u}_t$ et son coût.
2. **Passe arrière.** Linéariser la dynamique et développer les coûts autour de
   ce plan, puis résoudre le problème linéaire quadratique local avec
   {prf:ref}`alg-ilqr-backward-pass`. On obtient $k_t$ et $K_t$.
3. **Passe avant.** Appliquer la loi affine en boucle fermée dans le vrai
   simulateur, en partant du même état initial :

   $$
   \begin{aligned}
   \mathbf x_1^+&=\mathbf x_{\mathrm{init}},\\
   \mathbf u_t^+&=\bar{\mathbf u}_t+k_t+K_t(\mathbf x_t^+-\bar{\mathbf x}_t),\\
   \mathbf x_{t+1}^+&=\mathbf f_t(\mathbf x_t^+,\mathbf u_t^+).
   \end{aligned}
   $$

   Les nouveaux états et les nouvelles commandes forment le plan suivant.
4. **Recommencer** jusqu'à ce que le plan cesse de changer.

Cette boucle est la **commande linéaire quadratique itérative**, iLQR
{cite:p}`li2004iterative` : un problème de commande linéaire quadratique est
résolu à chaque itération, autour d'une trajectoire différente. La loi de
commande de l'étape 3 est la solution locale
$\mathbf u=\bar{\mathbf u}+\delta\mathbf u$, où la variation
$\delta\mathbf x_t$ est mesurée comme l'état simulé moins l'état planifié. Le
terme de rétroaction compte, car le vrai simulateur ne suit pas exactement le
modèle linéaire. Lorsque les nouveaux états s'écartent de la prédiction
linéaire, $K_t$ ajuste les commandes de la manière dont, selon le modèle
local, la meilleure commande varie avec l'état.

La boucle linéarise toujours autour d'une trajectoire produite par la vraie
dynamique. Les gains de l'ancienne linéarisation engendrent la nouvelle
trajectoire, et cette trajectoire donne la linéarisation suivante. Tout ce que
contiennent les étapes 2 et 3 découle de la dérivation ci-dessus : $q_t$,
$M_t$, $k_t$, $K_t$, $p_t$ et $P_t$ sortent tous de l'algèbre linéaire
quadratique. Deux garde-fous sont ajoutés à la main, un pas et une
régularisation, et la suite de cette section explique pourquoi chacun est
nécessaire.

### Pas et recherche linéaire

Le modèle local n'est exact que près du plan courant. Loin de l'optimum, le
pas complet de la passe avant peut emmener le bateau bien au-delà de ce
voisinage, et le vrai coût peut alors augmenter au lieu de diminuer. Rien dans
la dérivation de Riccati ne l'empêche.

Le remède habituel vient de l'optimisation numérique. On traite la solution
linéaire quadratique comme une direction de recherche, comme on traiterait un
pas de Newton, et on décide séparément de la distance à parcourir le long de
celle-ci. C'est une méthode de Newton amortie. Le pas $\alpha$ ne fait pas
partie de la dérivation de Riccati; il est ajouté après coup et ne multiplie
que le terme d'anticipation :

$$
\mathbf u_t^+=\bar{\mathbf u}_t+\alpha k_t+K_t(\mathbf x_t^+-\bar{\mathbf x}_t).
$$

Placer $\alpha$ sur $k_t$ seulement est un choix, justifié par le fait qu'il
donne un chemin entre deux trajectoires connues. Pour $\alpha=0$, chaque
variation d'état reste nulle, par récurrence le long de la passe avant, et
l'ancien plan revient. Pour $\alpha=1$, la passe avant fait le pas linéaire
quadratique complet, qui est la solution exacte lorsque le problème est
linéaire quadratique. Entre les deux, les gains maintiennent chaque essai
cohérent avec la vraie dynamique.

Une **recherche linéaire** choisit $\alpha$ à chaque itération. On essaie le
pas complet $\alpha=1$, on simule, et on garde le résultat si le vrai coût a
suffisamment diminué. Sinon, on divise $\alpha$ par deux et on simule de
nouveau. Chaque essai est un passage dans le simulateur non linéaire. Le pas
n'est donc pas un hyperparamètre fixe; il s'adapte à chaque itération. Quand
le modèle linéaire est exact, comme près de la convergence, le pas complet
passe habituellement du premier coup. Les vrais réglages sont le calendrier
des divisions par deux et le sens de « suffisamment ».

L'implémentation essaie $1,1/2,\ldots,2^{-11}$. Pour « suffisamment », la
passe arrière prédit la diminution du modèle local. Avec

$$
d_1=\sum_{t=1}^{T-1}q_{u,t}^\top k_t,\qquad
d_2=\frac12\sum_{t=1}^{T-1}k_t^\top M_{uu,t}k_t,
$$

la diminution prédite est
$\widehat\Delta J(\alpha)=-\alpha d_1-\alpha^2d_2$.
Un essai est accepté lorsque son vrai coût diminue strictement et d'au moins
une petite fraction de cette prédiction,
$J(\bar{\mathbf U})-J(\mathbf U^+)\geq10^{-4}\max(\widehat\Delta J(\alpha),0)$.
La prédiction guide l'acceptation; elle ne remplace pas l'évaluation de
l'objectif original.

### Régularisation

Le second problème apparaît dans la passe arrière elle-même. Loin de
l'optimum, la courbure en la commande $M_{uu,t}$ peut être indéfinie ou
presque singulière; la résolution en la commande échoue alors, ou propose une
modification énorme. On remplace la matrice utilisée dans cette résolution
par $M_{uu}+\mu I$, où $\mu>0$. Augmenter $\mu$ rend la correction plus
prudente et peut rendre la matrice définie positive. Le code vérifie une
factorisation de Cholesky, augmente $\mu$ après une passe arrière ou une
recherche linéaire ratée, et le diminue après un pas accepté. Les valeurs
numériques par défaut partent de $10^{-4}$, multiplient par dix après un échec
et divisent par trois après une acceptation, avec un plancher de $10^{-9}$.

Lorsque la régularisation modifie $k$ et $K$, les formules de $p_t$ et de $P_t$
données plus haut ne décrivent plus la substitution dans le modèle quadratique
original. L'implémentation utilise donc les expressions complètes

$$
\begin{aligned}
p_t&=q_x+M_{ux}^\top k+K^\top q_u+K^\top M_{uu}k,\\
P_t&=M_{xx}+M_{ux}^\top K+K^\top M_{ux}+K^\top M_{uu}K.
\end{aligned}
$$

Ici, $M_{uu}$ est le bloc de courbure original; la régularisation n'a servi
qu'à choisir la correction. Ces expressions restent aussi applicables lorsque
certaines commandes sont fixées à leurs bornes.

Avec ces deux garde-fous en place, la méthode complète s'écrit comme suit.

````{prf:algorithm} iLQR
:label: alg-trajectory-ilqr

**Entrée :** une transition discrète, des coûts courant et terminal, un état initial et une suite de commandes initiale réalisable.

1. Dérouler les commandes courantes et évaluer leur coût non linéaire.
2. Dériver la transition et les coûts le long de ce déroulement.
3. Exécuter la passe arrière ({prf:ref}`alg-ilqr-backward-pass`) pour obtenir $k_t$ et $K_t$. Augmenter la régularisation $\mu$ et réessayer si une résolution en la commande échoue.
4. Exécuter des passes avant avec $\alpha=1,\tfrac12,\tfrac14,\ldots$ dans le simulateur non linéaire.
5. Accepter le premier essai dont le vrai coût diminue suffisamment, et diminuer $\mu$. Si aucun ne convient, augmenter $\mu$ et revenir à l'étape 3.
6. Recommencer à partir de l'étape 2 jusqu'à ce que le résidu du gradient par rapport aux commandes soit petit, ou qu'une limite déclarée sur les itérations ou sur la régularisation soit atteinte.

**Sortie :** la dernière trajectoire acceptée, ses commandes et ses gains locaux, un historique des itérations et un statut de terminaison explicite.
````

L'itération utilise les dérivées secondes des coûts, mais seulement les
dérivées premières de la dynamique. On la décrit souvent comme une
approximation de Gauss--Newton de la DDP. Cette terminologie ne doit pas
masquer ce qui est approché : les termes omis proviennent de la courbure de la
dynamique, comme le montre la section suivante. Si un coût non linéaire est
lui-même une somme de résidus au carré, remplacer sa hessienne par un produit
de jacobiennes des résidus serait une approximation supplémentaire; le code
d'accostage utilise la hessienne exacte du coût.

Les gains renvoyés décrivent un voisinage du plan final.
L'animation d'accostage ci-dessous exécute un plan stocké. Les gains utilisés
à l'intérieur de l'optimiseur n'impliquent pas que l'animation replanifie à
mesure que le bateau avance.

(sec-ddp-curvature)=
## Courbure de la dynamique et DDP

L'iLQR néglige le reste quadratique du développement de Taylor de la
transition. Pour un bateau qui tourne, un changement de cap modifie la
direction de la poussée, si bien que des produits de variations d'état et de
commande peuvent influer sur l'état suivant. La DDP conserve ces termes du
second ordre dans la passe arrière. Tout le reste est inchangé.

### La dérivation des fonctions composées en une variable

Chaque étape arrière minimise un coût d'étape plus le coût restant évalué à
l'état suivant. En une variable, notons $g$ le coût restant et $f$ la
dynamique. La dérivée seconde de la composée comporte deux termes :

$$
(g\circ f)''=g''\,(f')^2+g'\,f''.
$$

Le premier terme est la courbure du coût restant multipliée par le carré de la
pente de la dynamique. Le second est la pente du coût restant multipliée par
la courbure de la dynamique. L'iLQR ne conserve que le premier terme. La DDP
conserve les deux.

### Une étape arrière développée au second ordre

Écrivons la quantité minimisée lors d'une étape arrière comme une fonction de
la variation empilée
$\delta\mathbf z_t=(\delta\mathbf x_t,\delta\mathbf u_t)$ :

$$
c_t(\bar{\mathbf z}_t+\delta\mathbf z_t)
+S_{t+1}\big(\mathbf f_t(\bar{\mathbf z}_t+\delta\mathbf z_t)-\bar{\mathbf x}_{t+1}\big).
$$

Développons-la au second ordre en $\delta\mathbf z_t$. La variation de l'état
suivant comporte une partie linéaire $F_t\delta\mathbf z_t$ et, pour la
composante $i$, une partie quadratique
$\tfrac12\delta\mathbf z_t^\top f^i_{zz,t}\delta\mathbf z_t$, où
$f^i_{zz,t}$ est la hessienne de la $i$e composante de $\mathbf f_t$ le long du
plan courant. Insérée dans la queue, la partie quadratique de $S_{t+1}$ ne voit
que la partie linéaire; le reste est d'ordre trois ou plus. La partie
linéaire $p_{t+1}^\top\delta\mathbf x_{t+1}$ de la queue voit en revanche les
deux. Le coefficient linéaire est le même $q_t=c_{z,t}+F_t^\top p_{t+1}$
qu'auparavant, et le coefficient quadratique est

$$
M_t=c_{zz,t}
+\underbrace{F_t^\top P_{t+1}F_t}_{g''(f')^2}
+\underbrace{\sum_{i=1}^{n_x}p_{t+1,i}\,f^i_{zz,t}}_{g'f''}.
$$

L'iLQR omet le dernier terme :

$$
M_t^{\mathrm{iLQR}}=c_{zz,t}+F_t^\top P_{t+1}F_t
+\cancel{\textstyle\sum_i p_{t+1,i}\,f^i_{zz,t}}.
$$

Ce sont les hessiennes de la dynamique pondérées par les multiplicateurs que le
[problème local](#sec-local-quadratic-trajectory) laissait de côté. Le
gradient de la queue $p_{t+1}$ joue le rôle du multiplicateur de la PQS :
c'est l'état adjoint, la sensibilité du coût restant à l'état suivant.

### La DDP conserve le terme

La DDP {cite:p}`tassa2014control` conserve le terme de courbure de la
dynamique. Par blocs,

$$
\begin{aligned}
M_{xx,t}^{\mathrm{DDP}}&=c_{xx,t}+A_t^\top P_{t+1}A_t
                    +\sum_i p_{t+1,i} f^i_{xx,t},\\
M_{ux,t}^{\mathrm{DDP}}&=c_{ux,t}+B_t^\top P_{t+1}A_t
                    +\sum_i p_{t+1,i} f^i_{ux,t},\\
M_{uu,t}^{\mathrm{DDP}}&=c_{uu,t}+B_t^\top P_{t+1}B_t
                    +\sum_i p_{t+1,i} f^i_{uu,t}.
\end{aligned}
$$

Tout ce qui suit applique la même recette avec le nouveau $M_t$ : les formules
de $k_t$, $K_t$, $p_t$ et $P_t$, la passe avant, la recherche linéaire et la
régularisation. Dans {prf:ref}`alg-trajectory-ilqr`, seule l'étape 3 change.

La DDP ne construit pas un nouveau problème local à dynamique quadratique pour
le résoudre ensuite par Riccati. Avec une dynamique quadratique, le problème
local ne serait plus linéaire quadratique. La DDP développe plutôt chaque
étape arrière directement au second ordre, et le terme supplémentaire
aboutit dans $M_t$.

````{prf:algorithm} Passe arrière pour la DDP
:label: alg-ddp-backward-pass

**Entrée :** les jacobiennes $A_t,B_t$, les hessiennes de la transition $f^i_{zz,t}$ et les dérivées du coût $c_{z,t},c_{zz,t}$ le long du plan courant; le gradient terminal $p_T$ et la hessienne terminale $P_T$.

**Sortie :** les termes d'anticipation $k_t$ et les gains $K_t$ pour $t=1,\ldots,T-1$.

1. Pour $t=T-1,\ldots,1$ :
   1. Former $q_t=c_{z,t}+F_t^\top p_{t+1}$ et $M_t=c_{zz,t}+F_t^\top P_{t+1}F_t+\sum_{i=1}^{n_x}p_{t+1,i}\,f^i_{zz,t}$.
   2. Résoudre $k_t=-M_{uu,t}^{-1}q_{u,t}$ et $K_t=-M_{uu,t}^{-1}M_{ux,t}$.
   3. Mettre à jour $p_t=q_{x,t}-M_{ux,t}^\top M_{uu,t}^{-1}q_{u,t}$ et $P_t=M_{xx,t}-M_{ux,t}^\top M_{uu,t}^{-1}M_{ux,t}$.
2. Renvoyer $k_t$ et $K_t$.
````

La seule ligne qui diffère de {prf:ref}`alg-ilqr-backward-pass` est la
formule de $M_t$. Le choix de la courbure peut néanmoins modifier à la fois la
direction d'un pas et la quantité de régularisation nécessaire pour
l'accepter.

Pour le bateau, même l'accélération en temps continu contient une dérivée
croisée état--commande :
$\partial^2\dot v_x/(\partial\psi\,\partial u_L)
=-(F_{\max}/m)\sin\psi$.
Les dérivées discrètes utilisées par le solveur tiennent aussi compte de la
façon dont le cap et la vitesse varient à l'intérieur du pas de RK4. La
différentiation automatique est appliquée à travers ce pas complet. Une
équation différentielle affine en ses commandes ne donne pas nécessairement
une transition numérique dont les dérivées secondes par rapport aux commandes
s'annulent.

La courbure supplémentaire coûte du calcul et peut être indéfinie. La DDP a
donc toujours besoin de régularisation et de tests d'acceptation non
linéaires. Aucune des deux méthodes ne garantit la meilleure manœuvre
d'accostage au sens global. Pour une dynamique affine, les hessiennes de la
transition sont nulles; les deux constructions arrière coïncident donc
lorsque tous les autres choix concordent.

## Limites des propulseurs et critères d'arrêt

Les commandes du bateau doivent rester dans $[-1,1]^2$, même lorsque la
correction sans contraintes demande davantage de poussée. Pour une variation
d'état nulle, le pas d'anticipation est la solution du problème quadratique
sous contraintes de boîte

$$
\min_k\ q_u^\top k+\frac12k^\top(M_{uu}+\mu I)k,
\qquad -\mathbf1-\bar{\mathbf u}_t\leq k\leq\mathbf1-\bar{\mathbf u}_t.
$$

Avec deux commandes, chaque coordonnée est soit libre, soit fixée à sa borne
inférieure, soit fixée à sa borne supérieure. L'implémentation pédagogique
examine les neuf combinaisons, minimise sur chaque face et garde le meilleur
candidat admissible. Une courbure définie positive en fait un programme
quadratique strictement convexe. Les systèmes plus grands peuvent recourir à
un solveur à ensemble actif plutôt que d'énumérer toutes les combinaisons
{cite:p}`tassa2014control`.

Les lignes de rétroaction des commandes saturées sont nulles tant que leur
ensemble actif reste fixe. Pour les coordonnées libres $\mathcal F$, on résout
$(M_{uu}+\mu I)_{\mathcal F\mathcal F}K_{\mathcal F}
=-(M_{ux})_{\mathcal F}$.
On utilise les formules de substitution complètes pour mettre à jour la queue
quadratique. Avec des bornes, cette queue décrit un voisinage où les ensembles
actifs choisis sont fixés; d'un ensemble actif à l'autre, l'expression
minimisée est en général quadratique par morceaux. Pendant l'essai non
linéaire, on projette la commande obtenue sur sa boîte avant la simulation.
Cela impose les bornes physiques lorsqu'un essai quitte le voisinage
représenté par la passe arrière. La vérification du coût non linéaire décide
ensuite s'il faut retenir cet essai.

Le test d'arrêt est distinct du succès de l'accostage. Soit
$g=\nabla_{\mathbf U}J$ le gradient de l'objectif de tir original, calculé par
la récurrence adjointe le long du déroulement non linéaire courant. Avec des
commandes normalisées, le solveur s'arrête lorsque

$$
\left\|\mathbf U-\operatorname{clip}(\mathbf U-g,-1,1)\right\|_\infty
\leq10^{-5}.
$$

En un optimum intérieur, ce critère se réduit à un petit gradient par rapport
aux commandes. À une borne, il admet aussi un gradient qui pointe vers une
amélioration interdite. L'expérience autorise au plus 120 mises à jour
acceptées. Si la régularisation dépasse $10^{12}$ sans essai acceptable, le
solveur conserve sa dernière trajectoire acceptée et signale qu'il ne
parvient plus à progresser. Un petit résidu certifie une stationnarité
approximative du premier ordre, et non un chemin globalement optimal ni une
arrivée réussie.

(sec-docking-results)=
## Trajectoires d'accostage et futurs enregistrés

Les deux choix de courbure produisent-ils la même manœuvre à partir du même
plan initial? Les deux méthodes reçoivent l'initialisation en glissade, des
coûts et des limites de commande identiques, et les mêmes réglages d'arrêt.
Les tableaux suivants présentent leurs coûts non linéaires finaux et leurs
vérifications à l'arrivée. Chaque suite de commandes finale est aussi rejouée
avec quatre sous-pas de RK4 par intervalle de commande, ce qui donne une
grille de validation de $0.025\ \mathrm s$.

:::{include} artifacts/boat_docking/results.md
:::

Les quatre exécutions satisfont le test d'arrêt et les vérifications
d'accostage : erreur de position inférieure à $0.2\ \mathrm m$, erreur de cap
inférieure à $3^\circ$, vitesse inférieure à $0.05\ \mathrm{m/s}$, vitesse
angulaire inférieure à $1^\circ/\mathrm s$, poussée bornée et dégagement de
la coque positif. L'écart de position maximal entre la trajectoire de
planification et le rejeu plus fin est inférieur à $10^{-6}\ \mathrm m$ dans
ces exécutions. Le dégagement est vérifié sur la grille plus fine; cela reste
une validation échantillonnée, et non une preuve d'évitement des collisions
en temps continu.

:::{figure} _static/boat_docking/paths.svg
:label: fig-boat-paths
:width: 100%

Trajectoires finales avec la même initialisation et le même objectif. Le
trait bleu plein correspond à l'iLQR et le trait orange tireté à la DDP; le
chemin gris pointillé est la glissade initiale. Les silhouettes montrent
l'orientation toutes les quatre secondes. La DDP effectue une manœuvre de
virage différente dans l'approche oblique. Les deux chemins finaux coïncident
presque dans le cas de la dérive latérale.
:::

Dans l'approche oblique, l'iLQR atteint un coût de 8.0515 en 18 mises à jour
acceptées, tandis que la DDP atteint une autre trajectoire stationnaire, de
coût 11.9879, en 34 mises à jour. La trajectoire de la DDP fait deux tours
complets près du poste d'amarrage. Son cap final, non ramené dans un
intervalle, vaut environ $720^\circ$, ce que le coût périodique sur le cap
traite comme la même orientation que zéro. Atteindre cette trajectoire
stationnaire n'efface pas le mouvement supplémentaire accumulé en chemin.
Dans le cas de la dérive latérale, les deux méthodes atteignent un coût de
7.6674, avec 13 mises à jour acceptées pour l'iLQR et 45 pour la DDP. Ce sont
les résultats de deux résolutions locales particulières, et non un classement
général des algorithmes. L'écart observé dans le premier cas montre aussi
pourquoi comparer uniquement le nombre d'itérations peut masquer une
différence entre les solutions obtenues.

:::{figure} _static/boat_docking/convergence.svg
:label: fig-boat-convergence
:width: 100%

Coût non linéaire effectif à chaque itération acceptée, y compris le
déroulement initial à l'itération zéro. L'axe vertical est logarithmique. Les
pas d'essai rejetés sont exclus du décompte horizontal; les nombres
d'évaluations avant et de tentatives arrière figurent dans les diagnostics
téléchargeables.
:::

À l'intérieur d'un plan choisi, le temps de simulation n'a pas le même sens
que l'itération de l'optimiseur. À deux secondes, par exemple, le bateau
occupe l'état à deux secondes de ce déroulement. Ses états restants
prédisent où il ira si le reste des commandes stockées est appliqué. Des
silhouettes également espacées dans le temps s'écartent les unes des autres
quand le bateau va vite et se regroupent quand il s'immobilise. La poussée
inverse aide à dissiper l'élan avant l'arrivée.

:::{figure} _static/boat_docking/future-and-motion.svg
:label: fig-boat-future
:width: 100%

Le plan iLQR final pour l'approche oblique. Le bateau plein marque le temps de
simulation choisi; des silhouettes translucides montrent les poses futures
toutes les deux secondes. Le panneau de droite donne la vitesse et les deux
poussées physiques. Une poussée négative pendant l'approche freine le bateau.
Le repère vertical indique le même instant que le bateau plein.
:::

:::{iframe} ../interactive/boat-docking.html
:width: 100%
:title: Accostage d'un bateau avec l'iLQR et la DDP
:class: boat-docking-replay

Choisissez un scénario et une itération de l'optimiseur, puis parcourez ou
lancez le temps de simulation à l'intérieur de ce plan fixé. Le chemin à
venir et les silhouettes du bateau affichent ses états prédits restants. Les
erreurs d'arrivée et les graphiques se rapportent à l'itération choisie, y
compris les plans intermédiaires qui n'accostent pas avec succès.
:::

Le rejeu lit des simulations enregistrées; l'explorer ne relance donc aucun
optimiseur. Recalculer un plan à partir d'états nouvellement observés est
l'étape supplémentaire introduite avec la
[commande à horizon glissant](receding-horizon-control.md).

(sec-thermoacoustic-pulldown)=
## Mise en froid d'un réfrigérateur thermoacoustique

Un réfrigérateur thermoacoustique à ondes stationnaires est un tube scellé,
rempli de gaz, muni d'un haut-parleur à une extrémité et contenant un
empilement de minces plaques parallèles
({numref}`fig-thermoacoustic-geometry`a). Le haut-parleur excite une onde
sonore stationnaire dans le tube. Chaque particule de gaz proche d'une plaque
est poussée d'avant en arrière le long de celle-ci; elle se comprime et
s'échauffe en se déplaçant vers l'extrémité du haut-parleur, puis se détend
et se refroidit en s'en éloignant. À l'extrémité de sa course située du côté
du haut-parleur, la particule est plus chaude que la plaque et lui cède de la
chaleur; à l'autre extrémité, elle est plus froide que la plaque et lui en
prend. Chaque cycle déplace donc un peu de chaleur d'un cran le long de la
plaque, et les particules forment ensemble une chaîne qui se passe la chaleur
de main en main, de l'extrémité éloignée de l'empilement vers l'extrémité du
haut-parleur. Le son sert à déplacer de la chaleur d'un endroit froid vers un
endroit plus chaud, à l'encontre du sens dans lequel elle s'écoulerait
d'elle-même. Une thermopompe domestique fait la même chose avec un
compresseur lorsqu'elle extrait la chaleur de l'air extérieur froid pour la
livrer à l'intérieur; un réfrigérateur est la même machine, utilisée pour son
côté froid. Un échangeur de chaleur froid, à l'extrémité éloignée de
l'empilement, est relié à la charge utile, le terme de réfrigération qui
désigne l'objet ou le compartiment à refroidir, et un échangeur de chaleur
chaud, à l'extrémité du haut-parleur, rejette la chaleur dans l'air ambiant.
Il n'y a pas de pièces mobiles autres que le haut-parleur, et le seul bouton
de réglage est l'intensité avec laquelle on l'excite. Le problème de commande
est ce que les ingénieurs en réfrigération appellent une mise en froid :
amener la charge utile de la température ambiante à sa consigne après la mise
en marche. Ici, cela revient à choisir un calendrier d'amplitude d'excitation
qui amène la charge utile à une température cible avant une échéance fixe,
tout en dépensant peu d'énergie acoustique.

:::{iframe} ../interactive/thermoacoustic-refrigerator.html
:label: fig-thermoacoustic-geometry
:width: 100%
:title: Réfrigérateur thermoacoustique : appareil, modèle à paramètres localisés et plans de mise en froid enregistrés
:class: thermoacoustic-replay
:placeholder: _static/thermoacoustic_pulldown/geometry.svg

Géométrie du réfrigérateur thermoacoustique et son abstraction à deux états.
(a) L'appareil : un piston de haut-parleur à l'extrémité gauche d'un
résonateur scellé, un échangeur de chaleur chaud, un empilement de minces
plaques et un échangeur de chaleur froid. L'enveloppe tiretée est l'amplitude
de pression de l'onde stationnaire. Les particules de gaz qui oscillent le
long des plaques pompent la chaleur de l'échangeur froid vers l'échangeur
chaud. (b) Le modèle à paramètres localisés que voit l'optimiseur. L'état est
le couple des températures des échangeurs, l'action est l'amplitude
d'excitation, le coût d'étape facture le travail acoustique, et le coût
terminal pénalise l'erreur finale de température du côté froid. Dans le livre
en ligne, la figure rejoue les plans iLQR et DDP enregistrés : la course du
piston, l'onde et le mouvement des particules suivent l'action, les teintes
des échangeurs et des nœuds suivent l'état, et les panneaux du bas comparent
le plan à un calendrier d'amplitude constante. La version statique montre les
deux mêmes panneaux.
:::

Dans le livre en ligne, le rejeu lit les plans enregistrés et n'intègre que
la comparaison à amplitude constante; il ne relance pas l'optimiseur.

Plutôt que de simuler l'acoustique, on regroupe tout l'appareil en deux masses
thermiques ({numref}`fig-thermoacoustic-geometry`b) : l'échangeur froid avec
sa charge utile, et l'échangeur chaud. L'état est formé de leurs températures,
$\mathbf x_t=(T_{c,t},T_{h,t})$, et l'action est l'amplitude d'excitation
normalisée $u_t\in[0,1]$; il s'agit donc d'un problème à état continu de
dimension deux et à action continue de dimension un. Soit
$\Delta T=T_h-T_c$ l'écart de température aux bornes de l'empilement et
$\eta(\Delta T)=1-\Delta T/\Delta T_0$ une efficacité de pompage qui décroît
linéairement avec cet écart. La dynamique en temps continu est

$$
\begin{aligned}
\dot Q_c &= k_q u^2\eta(\Delta T),&
\dot W &= k_w u^2\eta(\Delta T)+k_vu^2+k_3u^3,\\
C_c\dot T_c &= -\dot Q_c+Q_{\mathrm{load}},&
C_h\dot T_h &= \dot Q_c+\dot W-UA(T_h-T_{\mathrm{amb}}).
\end{aligned}
$$

Ici, $\dot Q_c$ est la chaleur pompée hors du côté froid, $\dot W$ le travail
acoustique injecté par l'excitateur, $Q_{\mathrm{load}}$ une fuite de chaleur
parasite vers la charge utile, et $UA$ la conductance de l'échangeur chaud
vers l'air ambiant. Les deux flux de chaleur varient comme $u^2$, car la
puissance acoustique est quadratique en l'amplitude de l'onde, et tous deux
faiblissent quand l'écart augmente; c'est l'approximation d'empilement court
de la thermoacoustique
{cite:p}`swift1988thermoacoustic,swift2017thermoacoustics`. Deux termes de
pertes complètent le travail : un terme visqueux en $u^2$ et un terme cubique
$k_3u^3$ qui rend une excitation forte disproportionnellement coûteuse. Ce
couplage compte pour l'optimiseur. Exciter fort pompe la chaleur plus vite,
mais déverse aussi cette chaleur, plus le travail, dans le côté chaud, ce qui
élève $T_h$, élargit l'écart et diminue l'efficacité de tous les pas
suivants. Les ingénieurs en réfrigération mesurent cette efficacité par le
coefficient de performance (COP), le rapport $\dot Q_c/\dot W$ entre la
chaleur pompée et le travail dépensé pendant que l'excitateur fonctionne.
Contrairement à un rendement, il peut dépasser un, car le travail ne fait que
déplacer de la chaleur au lieu de produire du froid. Dans ce modèle, le COP
diminue quand l'écart augmente et, à cause du terme cubique, quand
l'amplitude augmente; un des scénarios ci-dessous retire le terme cubique
pour que le COP ne dépende plus de l'amplitude. Tous les coefficients sont
des valeurs pédagogiques synthétiques, et non des mesures d'un appareil
particulier.

| Grandeur | Valeur |
| :--- | ---: |
| Capacités thermiques froide et chaude $C_c,C_h$ | $20,50\ \mathrm{J/K}$ |
| Conductance du côté chaud $UA$ | $0.5\ \mathrm{W/K}$ |
| Température ambiante $T_{\mathrm{amb}}$ | $20\ ^\circ\mathrm C$ |
| Charge parasite du côté froid $Q_{\mathrm{load}}$ | $1\ \mathrm W$ |
| Coefficient de pompage $k_q$ | $5\ \mathrm W$ |
| Coefficients de travail et de pertes visqueuses $k_w,k_v$ | $4,1\ \mathrm W$ |
| Coefficient de pertes cubiques $k_3$ | $2\ \mathrm W$ |
| Écart de température critique $\Delta T_0$ | $40\ \mathrm K$ |
| Température cible $T^\star$ | $5\ ^\circ\mathrm C$ |

Les deux températures partent de la température ambiante, $20\ ^\circ\mathrm C$.
Un pas de Runge--Kutta d'ordre quatre de longueur $h=1\ \mathrm s$ définit la
transition déterministe $\mathbf x_{t+1}=\mathbf f_t(\mathbf x_t,u_t)$ sur un
horizon de $T-1=300$ actions; l'épisode dure donc exactement cinq minutes. Le
coût a la même forme, étapes plus terme terminal, que le problème
d'accostage :

$$
c_t(\mathbf x_t,u_t)=h\,w_E\dot W(\mathbf x_t,u_t),\qquad
c_T(\mathbf x_T)=w_T(T_{c,T}-T^\star)^2,
\qquad w_E=0.05,\quad w_T=10.
$$

Le coût d'étape ne facture que l'énergie, et la cible n'intervient que par le
coût terminal; l'objectif est donc creux au sens de l'apprentissage par
renforcement : rien n'indique à l'optimiseur quel degré de froid atteindre aux
instants intermédiaires. La pénalité terminale est souple; un plan peut donc
échanger un peu d'erreur à l'arrivée contre une énergie plus faible. Les deux
méthodes partent de la suite constante en boucle ouverte $u_t=0.5$ et
utilisent les mêmes bornes sur l'action et les mêmes réglages d'arrêt.

:::{include} artifacts/thermoacoustic_pulldown/results.md
:::

:::{figure} _static/thermoacoustic_pulldown/pulldown.svg
:label: fig-thermoacoustic-pulldown
:width: 100%

Températures froide et chaude avec le plan de référence, et un déroulement à
pleine amplitude constante pour comparaison. Le panneau inférieur montre
leurs amplitudes d'excitation. La ligne cible et les courbes de température
montrent le compromis entre la consommation d'énergie et la température
froide finale.
:::

Le plan de référence excite assez fort au début, relâche, puis remonte
jusqu'à la pleine amplitude près de l'échéance. L'excitation précoce profite
de l'écart de température initialement faible, mais elle réchauffe aussi le
côté chaud et affaiblit le pompage ultérieur. La charge fixe ajoute la même
chaleur totale sur tout plan de 300 secondes. Le calendrier des commandes
modifie tout de même l'écart de température, et donc la température froide
finale. Fonctionner à pleine amplitude d'un bout à l'autre consomme environ
$1426\ \mathrm J$ et atteint
$2.20\ ^\circ\mathrm C$; son coût non linéaire est de $149.886$, contre $51.224$
pour le plan local de référence. Le coût terminal souple n'a pas à favoriser
l'état final le plus froid possible.

:::{figure} _static/thermoacoustic_pulldown/variants.svg
:label: fig-thermoacoustic-variants
:width: 100%

Calendriers d'amplitude finaux calculés pour le cas de référence, le cas sans
charge, le cas à côté chaud lent et le cas sans pertes cubiques. Les trois
premiers sont des plans iLQR; le plan sans pertes cubiques provient de la
DDP. Retirer la charge du côté froid aplatit le calendrier, tandis que
modifier la dynamique du côté chaud produit une longue rampe.
:::

Sans la charge parasite, une amplitude presque constante est économique. Pour
voir la tendance, fixons temporairement l'écart de température et posons
$v=u^2$. La chaleur pompée est alors proportionnelle à $v$, tandis que les
pertes cubiques sont proportionnelles à $v^{3/2}$, fonction convexe pour
$v\geq0$. Pour une quantité totale de pompage fixée, l'inégalité de Jensen
favorise l'étalement de $v$ dans le temps. Le calendrier effectif n'est pas
exactement constant, parce que l'écart et la température du côté chaud
réagissent toujours aux commandes. Avec $C_h=200\ \mathrm{J/K}$ et
$UA=0.2\ \mathrm{W/K}$, l'amplitude croît pendant la majeure partie de
l'horizon, puis s'annule pour les deux derniers intervalles. Le stockage
thermique et le rejet de chaleur ont tous deux changé dans cette expérience;
elle n'isole donc l'effet d'aucun des deux paramètres.

Le [terme de courbure de la DDP](#sec-ddp-curvature) a aussi ici une
interprétation physique directe. Pour l'équation du côté froid en temps
continu,

$$
\frac{\partial^2\dot T_c}{\partial u\,\partial\Delta T}
=\frac{2k_qu}{C_c\Delta T_0}.
$$

Augmenter l'écart de température affaiblit le pompage, et cette dérivée mixte
mesure comment cet effet varie avec l'amplitude. Le solveur dérive la
transition RK4 complète, dont la hessienne contient aussi des effets
provenant des étages d'intégration intermédiaires.

:::{figure} _static/thermoacoustic_pulldown/convergence.svg
:label: fig-thermoacoustic-convergence
:width: 100%

Coût non linéaire effectif après chaque mise à jour acceptée, pour l'iLQR et
la DDP, dans le cas de référence et dans le cas sans pertes cubiques. L'axe
vertical est logarithmique. Les courbes comparent ces résolutions locales
issues de la même suite de commandes initiale, et non un classement général
des méthodes.
:::

Lorsque $k_3=0$, le pompage et le travail sont tous deux quadratiques en
l'amplitude pour un écart de température fixé. Leur rapport instantané ne
dépend alors plus de l'amplitude, ce qui laisse des préférences plus faibles
entre certains calendriers. Dans l'exécution enregistrée, l'iLQR atteint sa
limite d'itérations tandis que la DDP satisfait le test d'arrêt; les deux
atteignent presque le même coût final. Pour le côté chaud lent, la DDP
atteint un plan local différent, de coût légèrement inférieur à celui de
l'iLQR. Abaisser la cible à $0\ ^\circ\mathrm C$ produit un plan saturé à
$u_t=1$ sur tout l'horizon fixé, ce qui met à l'épreuve l'étape arrière sous
contraintes de boîte. Ces résultats dépendent de la suite initiale. Partir,
pour le cas de référence, d'amplitudes constantes $0.1$ et $0.9$ mène à
d'autres plans locaux; aucun résultat ne certifie donc un minimum global.

(sec-radiotherapy-fractionation)=
## Planifier les fractions d'une radiothérapie

La radiothérapie délivre rarement sa dose en une seule séance. Un traitement
est découpé en fractions, typiquement une par jour de semaine pendant
plusieurs semaines, parce que les tissus sains réparent une partie des
dommages causés par le rayonnement entre les séances, alors que beaucoup de
tumeurs en réparent moins. Chaque faisceau qui atteint la tumeur traverse
aussi des organes sains à l'entrée et à la sortie; un plan de traitement doit
donc décider à la fois de l'intensité de l'irradiation chaque jour et des
directions d'où elle provient. Avec un vecteur de commande par séance et un
état qui résume les dommages accumulés jusque-là, il s'agit encore d'un
problème de contrôle optimal à horizon fini, auquel l'iLQR s'applique
directement.

L'exemple ci-dessous est une coupe synthétique du thorax, avec une tumeur au
centre et trois organes autour : la moelle épinière, le cœur et un poumon.
Trois faisceaux visent la tumeur, et chacun traverse exactement un organe
avant de l'atteindre ({numref}`fig-radiotherapy-replay`). La commande $\mathbf u_t\in\mathbb
R^3$ contient les trois intensités des faisceaux de la séance $t$, chacune
comprise entre $0$ et $3\ \mathrm{Gy}$. Les faisceaux sont mis à l'échelle de
sorte que l'intensité $u_{t,j}$ ajoute $u_{t,j}$ gray à la dose moyenne de la
tumeur. La dose moyenne reçue par la structure $k$ est
$d_{t,k}=\mathbf a_k^\top\mathbf u_t$, où la matrice de dose de lignes
$\mathbf a_k$ est calculée une fois pour toutes à partir de la géométrie des
faisceaux. La ligne de la tumeur est $(1,1,1)$, et chaque ligne d'organe
comporte une seule grande entrée, comprise entre $1.28$ et $1.50$, pour le
faisceau qui le traverse. Ce faisceau entre par l'organe avant d'avoir perdu
de son intensité dans les tissus; l'organe reçoit donc davantage que la
tumeur.

:::{iframe} ../interactive/radiotherapy-fractionation.html
:label: fig-radiotherapy-replay
:width: 100%
:title: Fractionnement d'une radiothérapie : plans iLQR et DDP enregistrés
:class: radiotherapy-replay
:placeholder: _static/radiotherapy_fractionation/plan.svg

Le problème de fractionnement et son plan optimisé. À gauche : la coupe
synthétique, avec la dose cumulée du traitement selon le plan DDP final. Des
flèches marquent l'entrée des trois faisceaux. En haut à droite : la dose que
chaque séance délivre à la tumeur, répartie par faisceau, comparée au plan
uniforme de 3 Gy (tireté). En bas à droite : la santé de la tumeur et des
trois organes avec le plan optimisé (trait plein) et le plan uniforme
(pointillé). Dans le livre en ligne, la figure rejoue les plans enregistrés
séance par séance et permet au lecteur de parcourir les itérés acceptés de
l'iLQR et de la DDP.
:::

L'état est un score de santé pour chaque structure, selon la convention des
travaux de planification clinique dont cet exemple est adapté
{cite:p}`fu2019convex`. La tumeur part de $h_{1,\text{tumeur}}=1$ et est
maîtrisée lorsque sa santé atteint $0$. Chaque organe part de $0$, et sa
santé devient négative à mesure que les dommages s'accumulent; $-1$ est sa
tolérance. Une séance endommage la structure $k$ selon le modèle linéaire
quadratique de la radiobiologie {cite:p}`fowler1989linear`, puis les jours
qui précèdent la séance suivante permettent aux organes de se réparer et à la
tumeur de repousser :

$$
h_{t+1,k}=\Big(h_{t,k}-\frac{\alpha_kd_{t,k}+\beta_kd_{t,k}^2}{E_k}\Big)\rho_k(t),
\qquad
\rho_k(t)=\begin{cases}e^{\lambda g_t},&k=\text{tumeur},\\ e^{-r_kg_t},&k\ \text{un organe}.\end{cases}
$$

Ici, $g_t$ est le nombre de jours avant la séance suivante : un du lundi au
jeudi et trois le vendredi. L'échelle $E_k$ est l'effet cumulé qui déplace
une structure d'une unité de santé, de non traitée à maîtrisée pour la
tumeur, et d'intacte à la tolérance pour un organe. Le terme quadratique
$\beta_kd^2$ rend une grande fraction plus dommageable que deux fractions de
moitié moindre totalisant la même dose. Les tissus sains à réaction tardive
ont un $\beta$ plus grand relativement à $\alpha$ que la plupart des tumeurs,
et c'est cette différence qui rend le fractionnement du traitement
avantageux. Les coefficients ci-dessous sont des valeurs pédagogiques
synthétiques aux rapports plausibles, et non des paramètres cliniques.

| Structure | $\alpha_k$ (1/Gy) | $\beta_k$ (1/Gy²) | Échelle d'effet $E_k$ | Réparation quotidienne $r_k$ ou repousse $\lambda$ |
| :--- | ---: | ---: | ---: | ---: |
| Tumeur | 0.30 | 0.030 | 22 | repousse 0.005 |
| Moelle épinière | 0.10 | 0.050 | 2 | 0.01 |
| Cœur | 0.10 | 0.030 | 2 | 0.04 |
| Poumon | 0.10 | 0.020 | 2 | 0.10 |

Le coût facture un petit montant pour l'intensité des faisceaux, pénalise à
chaque séance une santé des organes inférieure à la tolérance, exige que la
tumeur atteigne zéro à la fin du traitement et récompense la santé des
organes qui subsiste à la fin :

$$
\begin{aligned}
c_t(\mathbf h,\mathbf u)&=\tfrac{0.01}{2}\|\mathbf u\|^2+50\sum_{k\ne\text{tumeur}}s(-h_k-1),\\
c_T(\mathbf h)&=200\,s(h_{\text{tumeur}})^2+50\sum_{k\ne\text{tumeur}}s(-h_k-1)-\sum_{k\ne\text{tumeur}}h_k,
\end{aligned}
$$

où $s(z)=\log(1+e^{25z})/25$ est une version lisse de $\max(z,0)$, comme dans
les coûts d'accostage. Le traitement comporte $T-1=20$ séances sur quatre
semaines. Les deux solveurs partent de faisceaux égaux de $0.7\ \mathrm{Gy}$,
soit un plan uniforme de $2.1\ \mathrm{Gy}$ par séance qui ne maîtrise pas la
tumeur, et utilisent la même passe arrière sous contraintes de boîte que pour
le problème d'accostage.

:::{include} artifacts/radiotherapy_fractionation/results.md
:::

L'option curative la plus simple donne des faisceaux égaux à chaque séance,
augmentés jusqu'à ce que la tumeur atteigne tout juste zéro. Il faut pour
cela $3\ \mathrm{Gy}$ par séance et $60\ \mathrm{Gy}$ au total, ce qui amène
la moelle épinière à une santé de $-2.26$, plus de deux fois sa tolérance; le
cœur termine aussi juste au-delà de sa limite. Le plan optimisé maîtrise la
tumeur avec $56\ \mathrm{Gy}$ et garde chaque organe dans sa tolérance. Trois
caractéristiques du calendrier de la {numref}`fig-radiotherapy-replay`
découlent du modèle. Le faisceau qui traverse le poumon porte plus de la
moitié de la dose, et celui qui traverse la moelle le moins, parce que le
poumon se répare le plus vite et a le plus petit $\beta$, tandis que la
moelle se répare le plus lentement et a le plus grand. Les doses sont
concentrées en début de traitement : un dommage précoce à un organe dispose
de plus de jours pour être réparé avant la fin du traitement, et une
destruction précoce de la tumeur laisse moins de tumeur susceptible de
repousser. Les vendredis reçoivent une dose supplémentaire, car trois jours de
réparation les suivent. Les courbes de santé des organes se stabilisent après
la première semaine, lorsque les nouveaux dommages de chaque séance sont à peu
près compensés par la réparation. Les traitements cliniques ne concentrent
pas ainsi les doses au début; ce profil reflète le modèle de réparation du
premier ordre et l'absence de limites de toxicité aiguë, qu'un modèle de
planification clinique inclurait.

:::{figure} _static/radiotherapy_fractionation/iterations.svg
:label: fig-radiotherapy-iterations
:width: 100%

À gauche : coût non linéaire effectif après chaque mise à jour acceptée, pour
l'iLQR et la DDP, à partir du même plan uniforme de 2.1 Gy, sur un axe
logarithmique. À droite : la dose que des itérés choisis de la DDP délivrent
à la tumeur à chaque séance. La première mise à jour pousse la dose vers les
dernières séances, où elle est le plus directement liée au coût terminal de
la tumeur; les mises à jour suivantes la déplacent vers le début du
traitement et vers les vendredis.
:::

Les deux méthodes atteignent le même plan, mais la DDP a besoin de 11 mises à
jour acceptées et l'iLQR de 53. La transition explique cet écart. Elle est
linéaire en l'état de santé et quadratique en les faisceaux; le seul terme de
transition du second ordre que la DDP ajoute à la
[récurrence arrière](#sec-ddp-curvature) est donc

$$
\sum_kp_{t+1,k}\,\frac{\partial^2h_{t+1,k}}{\partial\mathbf u_t^2}
=-2\sum_kp_{t+1,k}\,\frac{\beta_k\rho_k(t)}{E_k}\,\mathbf a_k\mathbf a_k^\top,
$$

où $p_{t+1,k}$ est la dérivée du coût restant par rapport à $h_{t+1,k}$. Le
long du plan final, cette dérivée est positive pour la tumeur, dont la santé
restante coûte, et négative pour chaque organe, dont la santé est
récompensée. Les termes des organes ajoutent donc une courbure positive : ils
indiquent à la passe arrière qu'intensifier un faisceau endommage son organe
plus que proportionnellement. Le terme de la tumeur retranche de la courbure,
car une fraction plus grande détruit aussi la tumeur plus que
proportionnellement. L'iLQR omet ces deux termes et traite l'effet de chaque
gray supplémentaire comme fixe autour du plan courant; il lui faut donc plus
de mises à jour pour découvrir l'avantage d'étaler la dose entre les séances
et les faisceaux.

## Exercices

:::{exercise} Induction arrière à la main
:label: ex-ilqr-two-step

Considérons un problème scalaire avec $T=3$, un état initial $x_1=0$, la
dynamique $x_{t+1}=x_t+u_t$ et le coût

$$
J=\frac12u_1^2+\frac12u_2^2+\frac\beta2(x_3-r)^2,\qquad \beta>0.
$$

1. Considérez $x_2$ comme donné, substituez $x_3=x_2+u_2$ et minimisez par
   rapport à $u_2$. Écrivez le $u_2$ minimisant comme une fonction de $x_2$ et
   calculez le coût restant.
2. Utilisez $x_2=u_1$ pour minimiser par rapport à $u_1$. Trouvez les deux
   commandes et l'état terminal.
3. Pour $\beta=r=1$, vérifiez que $u_1=u_2=1/3$, $x_3=2/3$ et $J=1/6$. Pourquoi
   la position finale n'atteint-elle pas $r$?
4. Pourquoi est-il utile que l'étape 1 renvoie $u_2$ comme une fonction de
   $x_2$ plutôt que comme un nombre? Que se passe-t-il lorsque $\beta\to\infty$?
:::

:::{solution} ex-ilqr-two-step
:class: dropdown

La dernière commande est $u_2=\beta(r-x_2)/(1+\beta)$, et sa queue minimisée
vaut $\beta(x_2-r)^2/[2(1+\beta)]$. Avec $x_1=0$, minimiser l'expression
restante donne $u_1=\beta r/(1+2\beta)$, puis la même valeur pour $u_2$.
Ainsi, $x_3=2\beta r/(1+2\beta)$. Pour $\beta=r=1$, on obtient
$u_1=u_2=1/3$ et $x_3=2/3$, donc $J=\tfrac1{18}+\tfrac1{18}+\tfrac1{18}=1/6$.
La position finale reste en deçà parce que l'objectif fait un compromis entre
l'erreur terminale et l'effort; l'arrivée est pénalisée, et non imposée comme
une égalité.

La fonction $u_2(x_2)$ est une loi de rétroaction : si l'état qui atteint la
dernière étape change, la dernière commande minimisante change avec lui. Dans
le cas vectoriel, cette dépendance devient le gain $K_t$. À mesure que le
poids terminal augmente, les deux commandes tendent vers $r/2$ et l'état
terminal tend vers $r$.
:::

:::{exercise} Le cas particulier linéaire quadratique
:label: ex-ilqr-linear-case

Supposons que la dynamique soit affine et que chaque coût soit quadratique,
avec un objectif réduit strictement convexe et sans bornes sur les commandes.
Expliquez pourquoi une seule passe arrière non régularisée suivie d'un
déroulement non linéaire complet résout le problème à partir de n'importe
quelle trajectoire initiale réalisable. Qu'est-ce qui change avec la DDP?
:::

:::{solution} ex-ilqr-linear-case
:class: dropdown

Les développements locaux sont exacts. L'élimination arrière minimise donc le
problème quadratique original en coordonnées de perturbation, et $\alpha=1$
retrouve sa solution. Le simulateur affine concorde exactement avec la
dynamique linéarisée des perturbations. Toutes les hessiennes de la
transition s'annulent; la DDP produit donc la même correction. C'est aussi un
test d'implémentation utile : comparer les commandes renvoyées avec une
résolution dense de l'objectif quadratique réduit.
:::

:::{exercise} Un terme de courbure dans le modèle du bateau
:label: ex-ilqr-boat-curvature

Discrétisez temporairement le bateau par la méthode d'Euler explicite.
Dérivez la dérivée mixte
$\partial^2 f^{v_x}/(\partial\psi\,\partial u_L)$. Indiquez où elle intervient
dans la passe arrière de la DDP et pourquoi l'iLQR l'omet.
:::

:::{solution} ex-ilqr-boat-curvature
:class: dropdown

La mise à jour d'Euler de la vitesse contient
$h(F_{\max}/m)(u_L+u_R)\cos\psi$. La dérivée mixte vaut donc
$-h(F_{\max}/m)\sin\psi$. La DDP la multiplie par la composante $v_x$ de
$p_{t+1}$ et l'ajoute à l'entrée de courbure état--commande correspondante.
L'iLQR utilise la transition au premier ordre, dont les dérivées secondes
sont nulles dans le modèle local. L'expérience réelle dérive plutôt RK4; sa
dérivée mixte inclut donc les étages d'intégration intermédiaires.
:::

:::{exercise} Retirer la pénalité d'arrêt
:label: ex-ilqr-terminal-speed

Mettez à zéro le poids de la vitesse de translation terminale et relancez
l'approche oblique. Comparez l'erreur de position, la vitesse terminale et les
commandes avec l'exécution par défaut. Recommencez avec un horizon de huit
secondes. Arriver près du poste d'amarrage implique-t-il que l'accostage a
réussi?
:::

:::{solution} ex-ilqr-terminal-speed
:class: dropdown

Utilisez `make_problem(velocity_weight=0.)` et gardez tous les autres choix
fixes pour la première comparaison. Pour l'expérience plus courte, posez
`steps=80` dans `BoatParameters` et régénérez les commandes initiales. Retirer
le terme de vitesse terminale supprime une incitation explicite à s'arrêter au
temps final. Les coûts courants de position et de vitesse peuvent tout de
même mener à une arrivée lente, surtout sur l'horizon le plus long; l'échec
n'est donc pas inévitable. Mesurez la vitesse finale indépendamment de
l'erreur de position et conservez les mêmes tolérances d'accostage pour les
deux objectifs. Un passage peu coûteux par la cible n'est pas une arrivée
stationnaire réussie.
:::

:::{exercise} Saturation et pas rejeté
:label: ex-ilqr-saturation

Pour une matrice $2\times2$ définie positive $M$ dont les entrées hors
diagonale sont non nulles, comparez la solution exacte du programme
quadratique sous contraintes de boîte avec l'écrêtage composante par
composante de $-M^{-1}q$. Expliquez pourquoi elles peuvent différer. Que doit
conserver l'optimiseur de trajectoires si tous les essais non linéaires sont
rejetés?
:::

:::{solution} ex-ilqr-saturation
:class: dropdown

Fixer une coordonnée à une borne modifie l'équation de stationnarité de
l'autre coordonnée par l'intermédiaire de l'entrée hors diagonale de $M$.
L'écrêtage composante par composante ne résout pas cette nouvelle équation.
La minimisation sur la face, elle, la résout. Si aucun essai non linéaire
n'est accepté, le solveur conserve ses commandes et ses états précédents et
signale qu'il n'a pas pu progresser; il ne doit pas les remplacer par un
essai non accepté ni déclarer la convergence sur la seule foi d'un rejet.
:::

:::{exercise} Charge thermique et calendrier d'amplitude
:label: ex-ilqr-pulldown-load

Mettez à zéro la charge parasite du côté froid et relancez le problème de
référence du réfrigérateur. Expliquez, à l'aide de la convexité des pertes
cubiques, pourquoi l'amplitude optimisée devient presque constante. Rétablissez
ensuite la charge et doublez l'horizon pour le porter à $600\ \mathrm s$.
Prédisez comment le calendrier d'amplitude et la consommation d'énergie
changent.
:::

:::{solution} ex-ilqr-pulldown-load
:class: dropdown

Sans charge, le refroidissement obtenu tôt n'est pas directement annulé par
une fuite de chaleur constante. Si l'écart de température était fixé, une
quantité donnée de refroidissement fixerait la somme des $v_t=u_t^2$. Les
pertes cubiques sont proportionnelles à $v_t^{3/2}$; l'inégalité de Jensen
favorise donc des $v_t$ égaux dans le temps. L'écart de température évolue
tout de même, ce qui explique la variation modeste du calendrier calculé.
Avec la charge rétablie, l'horizon plus long permet une amplitude plus faible
sur davantage d'intervalles, mais la fuite ajoute de la chaleur pendant tout
ce temps supplémentaire. Dans ce modèle, la solution locale à
$600\ \mathrm s$ commence près de $0.59$, atteint $0.89$ vers
$550\ \mathrm s$ et utilise la pleine amplitude dans les derniers
intervalles. Ses quelque $1392\ \mathrm J$ d'énergie dépassent la
consommation du cas de référence plus court.
:::

## Sources informatiques

L'implémentation sépare la récurrence arrière générique des modèles physiques
et des générateurs d'artefacts :

- {download}`Solveur iLQR et DDP partagé <code/trajectory_optimization.py>`
- {download}`Dynamique, coûts et validation du bateau <code/boat_docking.py>`
- {download}`Générateur des expériences et des figures <scripts/build_boat_docking_artifacts.py>`
- {download}`Générateur de la figure et de l'animation du tube <scripts/build_boat_tube_figure.py>`
- {download}`Diagnostics numériques et empreintes des sources <artifacts/boat_docking/metrics.json>`
- {download}`Modèle et coûts thermoacoustiques <code/thermoacoustic_pulldown.py>`
- {download}`Figure de la géométrie thermoacoustique <code/thermoacoustic_geometry.py>`
- {download}`Générateur de l'expérience thermoacoustique <scripts/build_thermoacoustic_pulldown_artifacts.py>`
- {download}`Diagnostics thermoacoustiques <artifacts/thermoacoustic_pulldown/metrics.json>`
- {download}`Données du rejeu thermoacoustique <interactive/thermoacoustic-refrigerator-data.json>`
- {download}`Fantôme, dynamique et coûts de radiothérapie <code/radiotherapy_fractionation.py>`
- {download}`Générateur de l'expérience de radiothérapie <scripts/build_radiotherapy_fractionation_artifacts.py>`
- {download}`Diagnostics de radiothérapie <artifacts/radiotherapy_fractionation/metrics.json>`

Exécutez `uv run python scripts/build_boat_docking_artifacts.py` depuis la
racine du dépôt pour reproduire les résolutions, les figures, le tableau des
résultats et les données du navigateur. Les compilations normales du livre
lisent ces artefacts. Des tests vérifient indépendamment les dérivées, la
courbure composée de la DDP, le calcul scalaire et la concordance entre
l'élimination arrière et une résolution quadratique dense.

Exécutez `uv run python scripts/build_thermoacoustic_pulldown_artifacts.py`
pour régénérer le tableau des résultats du réfrigérateur, les figures, les
diagnostics et les données du rejeu dans le navigateur. Exécutez
`uv run python scripts/build_radiotherapy_fractionation_artifacts.py` pour les
résolutions, les figures, le tableau et les données de rejeu de la
radiothérapie. Cet exemple est adapté du dépôt
[ddp-fractionation](https://github.com/pierrelux/ddp-fractionation), qui
applique l'iLQR sous contraintes de boîte à un fantôme clinique comportant un
millier de petits faisceaux.

## Synthèse et perspectives

Un déroulement non linéaire réalisable fournit le point autour duquel la
dynamique et les coûts sont approchés. L'élimination arrière d'un problème
quadratique local fournit ensuite à la fois une modification d'anticipation
et une correction dépendant de l'état. L'iLQR répète ce calcul avec une
dynamique linéarisée; la DDP inclut les termes du second ordre que la
dérivation des fonctions composées tire de la transition. Les déroulements
non linéaires, la régularisation et les tests d'acceptation relient ces
calculs locaux à une suite décroissante de coûts de trajectoire effectifs.
Le réfrigérateur applique le même calcul à une amplitude d'excitation
bornée, l'objectif mettant en concurrence la consommation d'énergie et la
température terminale. Le traitement de radiothérapie l'applique à trois
faisceaux bornés sur vingt séances, où la réparation des organes entre les
séances façonne le calendrier et où la réponse quadratique à la dose donne à
la DDP une courbure que l'iLQR ignore.

La queue quadratique produite par l'élimination est un modèle local du coût
restant le long d'une trajectoire. Pour les problèmes linéaires quadratiques,
c'est le [coût restant optimal](#sec-lqr-dp) lui-même. Lorsque le modèle
prescrit une distribution sur les successeurs au lieu d'un unique état
suivant, la [programmation dynamique stochastique](stochastic-dp.md) remplace
le coût de continuation par une espérance conditionnelle.
