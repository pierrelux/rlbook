# Commande stochastique par intégrales de chemin

La programmation dynamique stochastique choisit une action de rétroaction en
minimisant le coût immédiat augmenté du coût de continuation espéré. En temps
continu, cette récurrence devient une équation aux dérivées partielles pour la
fonction de valeur. Des simulations vers l'avant peuvent-elles calculer
l'action optimale sans résoudre cette équation sur une grille d'états?

Pour une classe de systèmes dont l'effort de commande est quadratique et dont
le bruit physique lui est assorti, une transformation logarithmique rend
l'équation de la valeur linéaire. Sa solution est une espérance sur des
trajectoires, que l'on peut estimer par échantillonnage. On obtient ainsi des
conditions sous lesquelles les mises à jour pondérées introduites dans le
[chapitre sur MPPI](model-predictive-path-integral-control.md) calculent une
action de rétroaction courante optimale. Une particule qui traverse deux
passages concurrents montre ensuite comment l'incertitude physique peut
modifier l'itinéraire préféré.

Le coût optimal obtenu de cette façon est un minimum lisse,
$V=-\lambda\log\mathbb E[e^{-S/\lambda}]$, où $S$ est le coût d'une trajectoire
et où l'espérance suit la dynamique non commandée. Il a la forme du
log-sum-exp de la [programmation dynamique régularisée par l'entropie](regularized-dp.md),
pris sur des trajectoires entières plutôt que sur des actions, et l'effort de
commande quadratique joue le rôle d'une pénalité d'entropie relative par
rapport à la dynamique non commandée. Le temps est ici continu et l'horizon
fini, alors que ce chapitre-là traite de problèmes discrets actualisés.

## Objectifs d'apprentissage

Après avoir suivi la dérivation et les expériences, vous devriez pouvoir :

- Dériver l'équation stochastique de Hamilton-Jacobi-Bellman à partir de la
  récurrence de Bellman sur un intervalle de temps court.
- Énoncer la condition d'assortiment entre le bruit et le coût de commande, et
  l'utiliser pour obtenir une représentation de la fonction de valeur par
  intégrale de chemin.
- Estimer une action de rétroaction courante à partir d'incréments de bruit
  pondérés, y compris la correction pour un processus d'échantillonnage
  commandé.
- Distinguer une action de rétroaction optimale de l'ajustement d'une
  gaussienne à une distribution sur des suites de commandes en boucle ouverte.
- Comparer des correcteurs fondés sur la dynamique stochastique et sur la
  dynamique moyenne au moyen de perturbations appariées, de solutions de
  référence numériques et de vérifications de la discrétisation.

## Préalables

Ce chapitre utilise la récurrence de Bellman et les politiques dépendant de
l'état de [la programmation dynamique à horizon fini](finite-horizon-dp.md) et
de [la programmation dynamique stochastique](stochastic-dp.md). Les incréments
gaussiens sont revus dans [Dynamique stochastique et observation partielle](stochastic-dynamics-observation.md).
La comparaison avec MPPI utilise l'ajustement de la moyenne gaussienne et la
correction d'échantillonnage préférentiel de @sec-mppi-distribution. La
comparaison avec la régularisation par l'entropie renvoie à [la programmation dynamique régularisée par l'entropie](regularized-dp.md),
mais aucune dérivation de ce chapitre n'en dépend. L'équation de la valeur en
temps continu et sa transformation sont dérivées ci-dessous.

(sec-path-integral-control)=
## Commande stochastique par rétroaction

Comment la récurrence de Bellman change-t-elle lorsque les commandes et les
perturbations agissent de façon continue dans le temps? Un modèle scalaire à
incréments gaussiens fournit la mise à l'échelle du bruit nécessaire au
passage à la limite en temps court. La fonction de valeur qui en résulte tient
compte du fait que les actions futures répondront aux observations futures.

### Un modèle avec bruit physique

Considérons une position scalaire dont la commande $u_k$ fixe la vitesse
moyenne sur un intervalle de longueur $h$. Supposons que le mouvement subisse
aussi, pendant chaque intervalle, un déplacement gaussien indépendant :

$$
x_{k+1}=x_k+hu_k+\sqrt{\nu h}\,\xi_k,
\qquad \xi_k\sim\mathcal N(0,1),\qquad \nu>0.
$$

Ici, $\nu$ est la variance du déplacement physique par unité de temps. Sur $N$
intervalles, la variance accumulée du bruit est $N\nu h$. Subdiviser une durée
fixe en pas plus petits préserve donc la variance totale. C'est le facteur
$\sqrt h$ qui donne cette propriété : utiliser $h$ à la place ferait tendre la
variance totale du bruit vers zéro à mesure que les pas rapetissent.

Le mouvement brownien est le processus en temps continu qui fournit ces
incréments gaussiens indépendants. Si $W_s$ désigne le mouvement brownien
standard, alors $W_{s+h}-W_s\sim\mathcal N(0,h)$. Le modèle en temps continu
s'écrit

$$
dX_s=u_s\,ds+\sqrt\nu\,dW_s.
$$

Pour une commande maintenue constante sur un intervalle, l'intégration de
cette équation donne exactement la transition gaussienne ci-dessus. La
notation $dW_s$ désigne un incrément du processus aléatoire. Ce n'est pas une
vitesse ordinaire que l'on pourrait évaluer puis multiplier par un pas de
temps.

Ce caractère aléatoire physique subsiste même lorsque la commande est fixée.
Un simulateur tire des incréments indépendants pour représenter les avenirs
possibles à partir de l'état observé; ces tirages ne prédisent pas la
perturbation particulière que subira le système. Pour utiliser la notation des
déroulements de @sec-mppi-distribution, on écrit la même transition sous la
forme $x_{k+1}=x_k+h(u_k+\epsilon_k)$, où
$\epsilon_k=\sqrt{\nu/h}\,\xi_k$. Sa variance est $\nu/h$, parce qu'elle est
multipliée par $h$ pour produire le déplacement physique. Dans ce modèle,
c'est le bruit physique qui détermine la covariance des perturbations utilisée
dans la dérivation par intégrales de chemin.

### Le problème de commande stochastique

Soit $\mathbf X_s\in\mathbb R^n$ qui suit la diffusion affine en la commande

$$
d\mathbf X_s=
[\mathbf a(s,\mathbf X_s)+D(s,\mathbf X_s)\mathbf u_s]\,ds
+B(s,\mathbf X_s)\,d\mathbf W_s.
$$

La dérive $\mathbf a$ décrit le mouvement non commandé, $D$ transforme la
commande de dimension $m$ en mouvement de l'état, et $B$ transforme le bruit
brownien en fluctuations de l'état. Pour la particule scalaire, ces objets
sont $\mathbf a=0$, $D=1$ et $B=\sqrt\nu$. Les commandes peuvent dépendre de
l'état et du temps courants, mais pas du bruit futur.

Pour un instant final fixé $t_f$, on définit

$$
V(t,\mathbf x)=\inf_{\mathbf u}
\mathbb E_{t,\mathbf x}^{\mathbf u}\left[
c_f(\mathbf X_{t_f})+
\int_t^{t_f}\left(c(s,\mathbf X_s)
+\frac12\mathbf u_s^\top R\mathbf u_s\right)ds\right],
\qquad R\succ0.
$$

L'espérance est conditionnelle à $\mathbf X_t=\mathbf x$ et suit le
correcteur par rétroaction choisi. L'infimum porte sur les correcteurs dont
l'action courante n'utilise que l'information disponible à l'instant
courant. La valeur $V$ est le plus petit coût restant espéré à partir de cet
état.

Supposons les coefficients suffisamment réguliers pour que la diffusion existe
sans explosion, les coûts bornés inférieurement et d'espérances finies, et la
fonction de valeur une fois continûment dérivable en temps et deux fois en
état. Ces hypothèses autorisent le calcul différentiel classique qui suit.
L'ensemble des commandes est $\mathbb R^m$, et le bruit et l'effort doivent
satisfaire

$$
B B^\top=\lambda D R^{-1}D^\top,\qquad \lambda>0.
$$

L'égalité vaut en chaque instant et en chaque état. Elle assortit à la fois
les directions et les amplitudes de la commande et du bruit. Pour la
particule brownienne scalaire, $D=1$, $B=\sqrt\nu$ et $R=r$, de sorte que
$\lambda=r\nu$. Des choix indépendants de ces trois quantités violeraient en
général l'égalité. Les coordonnées d'état non actionnées sont permises, mais
un bruit indépendant supplémentaire en dehors des directions de commande ne
satisfait pas ce modèle.

### L'équation stochastique de Hamilton-Jacobi-Bellman

La récurrence de Bellman stochastique compare le coût immédiat au coût de
continuation optimal espéré. Sur un court intervalle $\delta$, sa forme en
temps continu est :

$$
V(t,\mathbf x)=\inf_{\mathbf u}
\mathbb E\left[
\left(c(t,\mathbf x)+\frac12\mathbf u^\top R\mathbf u\right)\delta
+V(t+\delta,\mathbf X_{t+\delta})\right]+o(\delta).
$$

Pour ce développement, la commande est fixée sur le court intervalle.
L'incrément d'état a pour moyenne $(\mathbf a+D\mathbf u)\delta+o(\delta)$ et
pour covariance $BB^\top\delta+o(\delta)$. Sa contribution d'ordre deux
survit donc à la division par $\delta$. Le développement de la dernière
espérance donne

$$
\mathbb E[V(t+\delta,\mathbf X_{t+\delta})]
=V+\delta\left[
V_t+(\mathbf a+D\mathbf u)^\top\nabla V
+\frac12\operatorname{tr}(BB^\top\nabla^2V)\right]+o(\delta).
$$

En soustrayant $V$, en divisant par $\delta$ et en passant à la limite, on
obtient l'équation stochastique de Hamilton-Jacobi-Bellman (HJB) :

$$
0=V_t+c+\mathbf a^\top\nabla V
+\frac12\operatorname{tr}(BB^\top\nabla^2V)
+\min_{\mathbf u}\left\{
\frac12\mathbf u^\top R\mathbf u+\mathbf u^\top D^\top\nabla V
\right\}.
$$

Pour la particule, la contribution de la diffusion est $\nu V_{xx}/2$. Les
déplacements browniens sont d'ordre $\sqrt\delta$; leurs carrés contribuent
donc à l'ordre $\delta$, le même ordre que la dérive et le coût courant.
Omettre la dérivée seconde remplacerait cette récurrence stochastique par une
récurrence déterministe.

Seuls les termes entre accolades dépendent de l'action courante. Comme $R$ est
définie positive, la dérivation de cette forme quadratique strictement
convexe donne son unique minimiseur :

$$
R\mathbf u+D^\top\nabla V=\mathbf0,
\qquad \mathbf u^\star=-R^{-1}D^\top\nabla V.
$$

La substitution élimine la minimisation et laisse

$$
0=V_t+c+\mathbf a^\top\nabla V
-\frac12\nabla V^\top D R^{-1}D^\top\nabla V
+\frac12\operatorname{tr}(BB^\top\nabla^2V),
\qquad V(t_f,\mathbf x)=c_f(\mathbf x).
$$

Le terme quadratique en $\nabla V$ rend cette équation non linéaire, même si
la minimisation sur la commande était elle-même explicite.

## Représentation par intégrale de chemin

Peut-on transformer l'équation non linéaire de la valeur en une espérance que
l'on peut estimer par des simulations vers l'avant? La condition d'assortiment
du bruit permet à un changement de variable logarithmique d'annuler le terme
quadratique en le gradient de la valeur. L'équation linéaire qui en résulte a
une solution probabiliste.

### La transformation logarithmique

L'équation HJB minimisée contient à la fois un terme quadratique en $\nabla V$
et un terme de diffusion faisant intervenir $\nabla^2V$. Un logarithme peut
relier ces termes, car sa dérivée seconde contient le carré d'une dérivée
première. On définit $\Psi=e^{-V/\lambda}$, ou de façon équivalente
$V=-\lambda\log\Psi$. La fonction positive $\Psi$ s'appelle la désirabilité :
les états de plus faible coût optimal ont un $\Psi$ plus grand. Ses dérivées
sont

$$
V_t=-\lambda\frac{\Psi_t}{\Psi},\qquad
\nabla V=-\lambda\frac{\nabla\Psi}{\Psi},\qquad
\nabla^2V=-\lambda\frac{\nabla^2\Psi}{\Psi}
+\lambda\frac{\nabla\Psi\nabla\Psi^\top}{\Psi^2}.
$$

Après substitution dans l'équation HJB, les termes quadratiques se combinent
en

$$
\frac{\lambda}{2\Psi^2}
\nabla\Psi^\top
\left(BB^\top-\lambda DR^{-1}D^\top\right)\nabla\Psi.
$$

L'assortiment du bruit annule cette expression. En multipliant l'équation
restante par $-\Psi/\lambda$, on obtient l'équation rétrograde linéaire

$$
\Psi_t+\mathbf a^\top\nabla\Psi
+\frac12\operatorname{tr}(BB^\top\nabla^2\Psi)
-\frac{c}{\lambda}\Psi=0,
\qquad
\Psi(t_f,\mathbf x)=e^{-c_f(\mathbf x)/\lambda}.
$$

Cette annulation est la restriction qui sous-tend la solution par intégrale de
chemin {cite:p}`Kappen2005PathIntegrals`. Une pénalité de commande arbitraire
ou une diffusion supplémentaire arbitraire laisse une équation non linéaire
après la transformation.

### Une espérance vers l'avant

L'équation linéaire a une solution probabiliste. Soit $P_0$ la loi des
trajectoires engendrées par le processus non commandé
$d\mathbf X_s=\mathbf a\,ds+B\,d\mathbf W_s$. Le long d'une telle trajectoire,
considérons

$$
M_s=\exp\left[-\frac1\lambda\int_t^s c(r,\mathbf X_r)\,dr\right]
\Psi(s,\mathbf X_s).
$$

La règle de dérivation en chaîne stochastique, ou formule d'Itô, développe les
variations de $\Psi$ au moyen des mêmes dérivées premières et secondes que le
calcul en temps court. La dérive de $M_s$ est son facteur exponentiel
multiplié par
$\Psi_t+\mathbf a^\top\nabla\Psi+
\tfrac12\operatorname{tr}(BB^\top\nabla^2\Psi)-c\Psi/\lambda$,
qui est nul. En supposant que l'intégrale stochastique restante est
intégrable et de moyenne nulle, on a $\mathbb E[M_{t_f}\mid\mathbf X_t=\mathbf x]=M_t$.
La condition terminale donne donc

$$
\Psi(t,\mathbf x)=
\mathbb E_{P_0}\left[
e^{-S/\lambda}\mid\mathbf X_t=\mathbf x\right],
\qquad
S=c_f(\mathbf X_{t_f})+\int_t^{t_f}c(s,\mathbf X_s)\,ds.
$$

C'est la représentation de Feynman-Kac. L'espérance intègre sur des
trajectoires complètes de la diffusion non commandée, d'où le nom
d'*intégrale de chemin*. Elle remplace une équation aux dérivées partielles
rétrograde par des simulations vers l'avant qui partent de l'état particulier
qui nous intéresse. Le coût quadratique de commande est absent de $S$, car son
effet est déjà pris en compte par l'assortiment du bruit et le logarithme.

### Coût quadratique et rétroaction optimale

Pour la particule scalaire avec le coût terminal
$c_f(x)=\tfrac\kappa2(x-a)^2$, où $\kappa>0$, et sans coût courant sur l'état,
l'extrémité non commandée est $X_{t_f}\sim\mathcal N(x,\nu\tau)$, où
$\tau=t_f-t>0$. L'espérance est une intégrale gaussienne ordinaire :

$$
\Psi(t,x)=\frac1{\sqrt{2\pi\nu\tau}}
\int_{\mathbb R}\exp\left[-\frac12\left(
\frac{(y-x)^2}{\nu\tau}+\frac\kappa\lambda(y-a)^2
\right)\right]dy.
$$

Les deux termes de l'exposant sont quadratiques en l'extrémité $y$. En
complétant le carré avec
$\bar y=(\lambda x+\kappa\nu\tau a)/(\lambda+\kappa\nu\tau)$, on obtient

$$
\frac{(y-x)^2}{\nu\tau}+\frac\kappa\lambda(y-a)^2
=\left(\frac1{\nu\tau}+\frac\kappa\lambda\right)(y-\bar y)^2
+\frac{\kappa(x-a)^2}{\lambda+\kappa\nu\tau}.
$$

Le terme constant sort de l'intégrale, et l'intégration de la gaussienne
restante fournit son facteur de normalisation. Avec $\lambda=r\nu$, on obtient

$$
\Psi(t,x)=
\left(1+\frac{\kappa\tau}{r}\right)^{-1/2}
\exp\left[-\frac{\kappa(x-a)^2}
{2\lambda(1+\kappa\tau/r)}\right].
$$

En prenant son logarithme et en dérivant, on obtient une valeur et une loi de
rétroaction que l'on peut vérifier sans Monte-Carlo :

$$
V(t,x)=\frac{\kappa r(x-a)^2}{2(r+\kappa\tau)}
+\frac\lambda2\log\left(1+\frac{\kappa\tau}{r}\right),
\qquad
u^\star(t,x)=\frac{\kappa(a-x)}{r+\kappa\tau}.
$$

Le bruit augmente le coût optimal espéré par l'intermédiaire du terme
logarithmique. Dans cet exemple quadratique, il ne modifie pas la loi de
rétroaction. C'est l'**équivalence de certitude**. Il faut des coûts d'état non
quadratiques ou des barrières pour observer des décisions qui dépendent de
l'intensité de la diffusion.

En $x=0$, $a=1$, $r=1$ et $\tau=1$, l'action courante vaut
$\kappa/(1+\kappa)$, la moyenne trouvée dans la vérification quadratique
déterministe de @sec-mppi-distribution. Ici, le correcteur répond aussi aux
observations ultérieures en réévaluant $u^\star(t,x)$ au nouvel état et au
nouvel instant. La concordance des actions initiales ne rend pas les règles
d'exécution identiques. L'exercice @ex-mppi-certainty-equivalence vérifie
directement l'équation de la valeur.

## Échantillonner l'action optimale

Comment des trajectoires échantillonnées peuvent-elles estimer l'action de
rétroaction sans dériver numériquement la fonction de valeur? Le premier
incrément de bruit modifie à la fois l'état suivant et le poids espéré du coût
futur. Leur corrélation fournit la dérivée nécessaire à l'action courante.

### Estimer la commande à partir des incréments de bruit

La représentation de la valeur semble encore exiger une dérivée par rapport à
l'état pour obtenir la commande. Pour la particule, toutefois, un premier
incrément de bruit positif modifie le poids espéré restant d'environ
$\Psi_x\sqrt\nu\,\Delta W$. Multiplier par $\Delta W$ et prendre la moyenne
conserve ce terme, car $\mathbb E[(\Delta W)^2]=\delta$; les termes constants
par rapport à l'incrément ont une moyenne nulle. La corrélation du premier
incrément avec le poids futur estime donc une dérivée par rapport à l'état.

Pour exprimer ce calcul avec plusieurs directions de commande, on suppose $D$
de rang colonne plein et l'on écrit $B=DL$, avec $L$ constante et inversible
telle que $LL^\top=\lambda R^{-1}$. Conditionnellement à un court incrément
initial, le poids espéré restant est $\Psi(t+\delta,\mathbf X_{t+\delta})$
multiplié par le facteur de coût courant initial. Son développement d'ordre un
et $\mathbb E[\Delta\mathbf W\Delta\mathbf W^\top]=\delta I$ donnent

$$
\mathbb E_{P_0}[e^{-S/\lambda}L\Delta\mathbf W]
=\delta\,LL^\top D(t,\mathbf x)^\top
\nabla\Psi(t,\mathbf x)+o(\delta).
$$

Les termes déterministes ont une corrélation nulle avec le bruit de moyenne
nulle. En divisant par $\delta\Psi$, on retrouve la commande optimale :

$$
\mathbf u^\star(t,\mathbf x)=
\lim_{\delta\downarrow0}
\frac{\mathbb E_{P_0}
[e^{-S/\lambda}L(\mathbf W_{t+\delta}-\mathbf W_t)]}
{\delta\,\mathbb E_{P_0}[e^{-S/\lambda}]}.
$$

Les trajectoires futures de faible coût reçoivent un poids plus grand, et
leurs déplacements aléatoires initiaux déterminent la direction de commande à
appliquer maintenant. La formule est conditionnelle à l'état courant
spécifié; les estimations aux instants futurs exigent le conditionnement sur
les états correspondants.

### Échantillonner autour d'une trajectoire commandée

Les simulations non commandées n'atteignent parfois que rarement la cible. On
simule plutôt sous une commande d'échantillonnage adaptée $\bar{\mathbf u}$
avec la même diffusion, et l'on note $Q_{\bar{\mathbf u}}$ la loi des
trajectoires qui en résulte. La formule de changement de mesure de Girsanov
donne

$$
\frac{dP_0}{dQ_{\bar{\mathbf u}}}
=\exp\left[-\frac1\lambda\left(
\frac12\int_t^{t_f}\bar{\mathbf u}_s^\top R\bar{\mathbf u}_s\,ds
+\int_t^{t_f}\bar{\mathbf u}_s^\top RL\,d\mathbf W_s^{\bar{\mathbf u}}
\right)\right].
$$

Ici, $\mathbf W^{\bar{\mathbf u}}$ est un mouvement brownien sous la loi
d'échantillonnage. Parmi les conditions suffisantes figure la condition
habituelle d'intégrabilité exponentielle
$\mathbb E_{Q_{\bar{\mathbf u}}}
\exp[\tfrac12\int_t^{t_f}\|L^{-1}\bar{\mathbf u}_s\|^2ds]<\infty$,
qui garantit que ce rapport de vraisemblance définit une mesure de
probabilité. Le rapport est l'équivalent en temps continu du développement de
la densité gaussienne de @sec-mppi-distribution : le carré accumulé du
décalage de la moyenne fournit la première intégrale, et les termes croisés
entre la moyenne et le bruit fournissent la seconde.

On absorbe le rapport dans un coût de trajectoire corrigé,

$$
S^{\bar{\mathbf u}}=S+
\frac12\int_t^{t_f}\bar{\mathbf u}_s^\top R\bar{\mathbf u}_s\,ds
+\int_t^{t_f}\bar{\mathbf u}_s^\top RL\,d\mathbf W_s^{\bar{\mathbf u}}.
$$

Son intégrale stochastique est de moyenne nulle, mais doit rester à
l'intérieur du poids exponentiel. L'estimateur de la commande devient alors

$$
\mathbf u^\star(t,\mathbf x)=\bar{\mathbf u}(t,\mathbf x)+
\lim_{\delta\downarrow0}
\frac{\mathbb E_{Q_{\bar{\mathbf u}}}
[e^{-S^{\bar{\mathbf u}}/\lambda}L\Delta\mathbf W^{\bar{\mathbf u}}]}
{\delta\,\mathbb E_{Q_{\bar{\mathbf u}}}
[e^{-S^{\bar{\mathbf u}}/\lambda}]}.
$$

La perturbation pondérée corrige la commande d'échantillonnage courante. Le
résultat exact vaut pour différentes commandes d'échantillonnage admissibles,
bien que leurs variances de Monte-Carlo puissent différer sensiblement
{cite:p}`Thijssen2015Feedback`.

### Discrétisation et mise à jour de MPPI

Pour une simulation de pas $h$, le schéma d'Euler-Maruyama donne

$$
\mathbf x_{k+1}=\mathbf x_k+h\mathbf a_k
+hD_k(\mathbf u_k+\boldsymbol\epsilon_k),
\qquad
\boldsymbol\epsilon_k=\frac{L\boldsymbol\xi_k}{\sqrt h},
\qquad \boldsymbol\xi_k\sim\mathcal N(\mathbf0,I).
$$

L'incrément aléatoire de l'état est $\sqrt h\,D_kL\boldsymbol\xi_k$; sa
covariance est donc $hD_kLL^\top D_k^\top=hB_kB_k^\top$, comme l'exige la
diffusion. La perturbation de l'entrée a elle-même pour covariance
$\Sigma_h=\lambda R^{-1}/h$. Diviser $h$ par deux multiplie donc son écart
type par $\sqrt2$, tout en divisant par $\sqrt2$ l'écart type de l'incrément
physique de l'état. Les deux mises à l'échelle portent sur des quantités
différentes. Notons $S_h$ le coût terminal sur l'état augmenté d'une
approximation par quadrature du coût courant sur l'état. Avec
$\lambda\Sigma_h^{-1}=hR$, l'approximation des intégrales de
$S^{\bar{\mathbf u}}$ donne

$$
S_h+\sum_{k=0}^{N-1}h\left(
\frac12\mathbf u_k^\top R\mathbf u_k
+\mathbf u_k^\top R\boldsymbol\epsilon_k\right).
$$

Pour une suite nominale déterministe, le premier terme de la somme est commun
à tous les déroulements et s'annule dans les poids normalisés. Le terme croisé
restant est exactement la correction de vraisemblance discrète
$\lambda\sum_k\mathbf u_k^\top\Sigma_h^{-1}\boldsymbol\epsilon_k$.
Pour un correcteur d'échantillonnage dépendant de l'état, le terme quadratique
varie en général d'une trajectoire à l'autre et doit être conservé.

La perturbation pondérée a la même forme que la mise à jour de
@sec-mppi-distribution. Cette dérivation-là ajuste une gaussienne à la loi de
Gibbs $P^*$, de densité proportionnelle à $e^{-S/\lambda}p_0$, où $p_0$ est
une densité de référence sur les suites d'entrées. Son résultat et celui de la
commande stochastique décrivent des objets différents :

| Calcul | Résultat avant l'approximation numérique | Approximation utilisée dans le calcul |
| --- | --- | --- |
| Pondération de Gibbs et ajustement gaussien | Moyenne de $P^*$, un ajustement exact dans la famille gaussienne à covariance fixée | Un nombre fini d'échantillons préférentiels estime cette moyenne; l'ajustement écarte les autres caractéristiques de $P^*$ |
| Commande stochastique assortie | Action de rétroaction courante optimale, obtenue par l'espérance limite sur le premier incrément de bruit | La discrétisation temporelle et un nombre fini d'échantillons approchent l'espérance conditionnelle |

La moyenne de la distribution décrit une suite complète en boucle ouverte. La
formule de la commande stochastique donne une action de rétroaction courante
conditionnelle à un état spécifié. Ses actions de rétroaction futures exigent
aussi les états futurs. Pour des modèles généraux non linéaires et sous
contraintes, ni la moyenne gaussienne ni l'échantillonnage fini ne garantissent
l'optimalité globale ou l'admissibilité.

L'espérance dans l'objectif de commande original reste un coût espéré
ordinaire. La pondération exponentielle sur la diffusion non commandée résout
cet objectif grâce à la transformation de l'équation HJB et à l'assortiment du
bruit. Prendre l'exponentielle des coûts de scénarios de perturbation
arbitraires issus d'un modèle sans rapport n'établit pas le même résultat. En
particulier, l'incertitude physique sur les perturbations et l'exploration
numérique supplémentaire doivent être spécifiées séparément chaque fois que
l'interprétation par une diffusion assortie ne s'applique pas.

(sec-brownian-passages)=
## Mouvement brownien à travers des passages concurrents

Quand l'incertitude change-t-elle la direction de la force optimale? L'exemple
quadratique ne permet pas de répondre à cette question, car son action
optimale possède l'équivalence de certitude. On conserve maintenant la même
particule entièrement observée et le même effort quadratique, et l'on change la
géométrie du coût sur l'état.

Imaginons une particule qui part de $x=-0.3$. Entre les instants $1$ et $2$,
elle doit rester dans l'une de deux ouvertures : un intervalle étroit
$[-0.18,0.18]$ près de sa position de départ, ou un intervalle plus large
$[-1.7,-0.7]$ plus loin à gauche. Une trajectoire possible se dirige vers la
droite dans l'ouverture étroite, y reste tant que le passage est actif, puis
s'approche de la cible terminale $x=0$ à l'instant $3$. Une autre se dirige
vers la gauche dans l'ouverture large et revient ensuite vers la cible. Avec de
faibles fluctuations, l'itinéraire le plus court demande moins d'effort. Des
fluctuations plus fortes rendent plus difficile le maintien dans l'ouverture
étroite.

Dans les figures, la coordonnée horizontale est le temps et la coordonnée
verticale est la position scalaire de la particule. Les ouvertures décrivent un
coût qui dépend à la fois du temps et de la position; ce ne sont pas un second
état dynamique. Cette construction suit les exemples de la double fente et des
passages inégaux de {cite:t}`Kappen2005PathIntegrals`. Les pénalités finies et
les paramètres numériques ci-dessous sont des adaptations pour un exemple
pédagogique exécutable.

:::{figure} _static/brownian_mppi/brownian-passages.svg
:label: fig-brownian-passages
:width: 100%

Réalisations appariées à deux niveaux de bruit physique. Les régions grises
entraînent une pénalité finie. Les trajectoires bleues utilisent la
rétroaction par intégrales de chemin échantillonnées; les trajectoires orange
en tirets utilisent la rétroaction obtenue en planifiant sous la dynamique
moyenne. Les mêmes incréments browniens agissent sur les deux correcteurs dans
chaque essai. Ici, $r=1$ et $\lambda=\nu r$.
:::

### Une double fente de largeur finie avec solution analytique

Le passage complet exige une suite de positions réussies. Pour isoler la
décision du moment où choisir un itinéraire, on remplace d'abord cette durée
par un unique instant de franchissement $t_s<T$. La particule paie une pénalité
forfaitaire $F$ si $X_{t_s}$ se trouve hors d'une union d'intervalles
$\mathcal A=\bigcup_j[a_j,b_j]$, ainsi qu'un coût terminal $\kappa X_T^2/2$.
La diffusion et l'effort restent

$$
dX_s=u_s\,ds+\sqrt\nu\,dW_s,
\qquad \frac r2u_s^2,
\qquad \lambda=\nu r.
$$

La désirabilité $\Psi=e^{-V/\lambda}$ est le poids exponentiel espéré sous le
mouvement brownien non commandé, comme on l'a dérivé à @sec-path-integral-control.
Le conditionnement sur $Y=X_{t_s}$ sépare le franchissement du transport
terminal :

$$
\Psi(t,x)=\int_{\mathbb R}
\mathcal N(y;x,\nu(t_s-t))
 e^{-F\mathbf1_{\{y\notin\mathcal A\}}/\lambda}
 \Psi_{\mathrm{free}}(t_s,y)\,dy.
$$

Le calcul quadratique ci-dessus donne

$$
\Psi_{\mathrm{free}}(t,x)=
\sqrt{\frac r{r+\kappa(T-t)}}
\exp\!\left[-\frac{\kappa x^2}{2\nu(r+\kappa(T-t))}\right].
$$

En multipliant les deux facteurs gaussiens et en complétant le carré, il
reste une gaussienne normalisée en $y$. Notons $\delta=t_s-t$ le temps restant
avant le franchissement et $A=r/\kappa+T-t_s$ le temps restant après le
franchissement augmenté du rapport entre l'effort et le coût terminal. Le
produit gaussien a pour moyenne et variance

$$
\beta=\frac A{A+\delta},\qquad m=\beta x,\qquad
s^2=\nu\delta\beta.
$$

Le coût terminal attire sa moyenne vers zéro : $0<\beta<1$. Sous cette
gaussienne pondérée par le coût, la probabilité de se trouver dans une
ouverture est

$$
P_{\mathcal A}(x)=\sum_j\left[
\Phi\!\left(\frac{b_j-m}{s}\right)
-\Phi\!\left(\frac{a_j-m}{s}\right)\right],
$$

où $\Phi$ est la fonction de répartition de la loi normale centrée réduite.
À l'intérieur d'une ouverture, le poids de franchissement vaut un; à
l'extérieur, il vaut $e^{-F/\lambda}$. La moyenne de ces deux valeurs donne

$$
\Psi(t,x)=\Psi_{\mathrm{free}}(t,x)
\underbrace{\left[e^{-F/\lambda}
 +(1-e^{-F/\lambda})P_{\mathcal A}(x)\right]}_{G(t,x)},
\qquad
u^*(t,x)=-\frac{\kappa x}{r+\kappa(T-t)}
+\nu\frac{\partial_xG(t,x)}{G(t,x)}.
$$

C'est une référence analytique pour des largeurs de passage finies et une
pénalité finie. Faire tendre $F\to\infty$ interdit le franchissement hors
d'une ouverture. Pour des ouvertures symétriques qui se réduisent à des points
en $\pm a$, leurs coûts terminaux égaux se simplifient dans le rapport qui
donne la commande, ce qui donne

$$
u^*(t,x)=\frac{a\tanh(ax/(\nu\delta))-x}{\delta}.
$$

En $x=0$, la symétrie impose $u^*=0$. La réponse à un petit déplacement est

$$
\partial_xu^*(t,0)=\frac1\delta
\left(\frac{a^2}{\nu\delta}-1\right).
$$

Si $\nu\delta>a^2$, la force ramène un petit déplacement vers le centre : on a
le temps de différer l'engagement. Si $\nu\delta<a^2$, la force amplifie le
déplacement vers l'une des ouvertures. Une fluctuation réalisée peut alors
déterminer l'itinéraire que suit la rétroaction. Au point de symétrie exact,
une action nulle exprime la symétrie; elle ne garantit pas qu'une trajectoire
réalisée passera par le centre de l'une ou l'autre ouverture.

:::{figure} _static/brownian_mppi/brownian-feedback.svg
:label: fig-brownian-feedback
:width: 100%

À gauche : dans la limite de fentes étroites symétriques, avec $a=\nu=1$, la
réponse près de $x=0$ change à l'approche du franchissement. À droite :
l'échantillonnage préférentiel vers l'avant est comparé à la formule à largeur
finie en $x=0.2$, avec les ouvertures $[-1.15,-0.85]$ et $[0.85,1.15]$,
$t_s=2$, $T=3$, $r=\kappa=\nu=1$ et $F=20$. Chaque estimation fait la moyenne
de 32 lots indépendants de 4096 échantillons; les barres indiquent les
intervalles de confiance à 95 % de cette moyenne.
:::

### Rester à l'intérieur de passages inégaux

Un passage d'épaisseur finie exige de rester dans une ouverture pendant une
certaine durée. Pour la géométrie de la @fig-brownian-passages, on définit

$$
q(t,x)=40\,\mathbf1_{\{1<t\leq2\}}
\mathbf1_{\{x\notin[-0.18,0.18]\cup[-1.7,-0.7]\}},
\qquad
J(u)=\mathbb E\!\left[
\frac12X_3^2+\int_0^3\left(q(s,X_s)+\frac12u_s^2\right)ds
\right].
$$

On fixe $r=\kappa=1$ et l'on compare $\nu=0.015$ à $\nu=0.3$. La condition
d'assortiment exige de faire passer en même temps $\lambda$ de $0.015$ à
$0.3$. Conserver une température fixe distincte changerait la relation avec ce
problème de commande stochastique.

La simulation utilise $h=0.025$ et évalue la pénalité de passage à
l'extrémité droite de chaque transition. À chaque décision, le correcteur par
intégrales de chemin échantillonne 4096 avenirs jusqu'à la fin du passage. La
queue quadratique terminale, de l'instant $2$ à l'instant $3$, est intégrée
analytiquement. Trois lois de proposition gaussiennes sur les trajectoires
engendrent les avenirs : une guide la particule vers chaque ouverture, et une
conserve une dérive nulle. Leur mélange améliore la couverture lorsque les
trajectoires de passage réussies sont rares sous un échantillonnage passif.

La probabilité des trajectoires sous ce mélange doit apparaître dans la
correction. Si $q_j$ est la densité de la suite complète d'incréments sous
l'itinéraire $j$, le dénominateur est $q_{\mathrm{mix}}=\frac13\sum_{j=1}^3q_j$.
Le même itinéraire est choisi pour tout un avenir simulé; un produit de
densités de mélange par pas décrirait donc une autre procédure
d'échantillonnage. Soit $p_0$ la densité jointe des incréments non commandés,
chacun de loi $\mathcal N(0,\nu h)$. Pour des suites d'incréments
échantillonnées $\Delta x^{(i)}$, on normalise les poids suivants pour que leur
somme vaille un, et l'on utilise leur premier incrément pondéré pour estimer
l'action :

$$
w_i\propto e^{-S_i/\lambda}\frac{p_0(\Delta x^{(i)})}
 {q_{\mathrm{mix}}(\Delta x^{(i)})},
\qquad
u(t,x)\approx\frac1h\sum_iw_i\Delta x^{(i)}_0.
$$

Les itinéraires de guidage choisissent où échantillonner. Le premier incrément
pondéré choisit la force réellement appliquée. L'incrément brownien physique
appliqué après ce choix provient d'un flux de nombres aléatoires distinct.

:::{figure} _static/brownian_mppi/brownian-samples.svg
:label: fig-brownian-samples
:width: 100%

Avenirs issus d'une mise à jour à bruit élevé en $t=0$. Les courbes bleues
font partie des échantillons de plus grand poids; les courbes grises ont des
poids plus faibles. Le panneau de droite montre combien d'échantillons portent
l'essentiel du poids normalisé. La taille d'échantillon effective,
$1/\sum_iw_i^2$, diagnostique la concentration des poids; elle ne mesure pas le
nombre d'avenirs physiquement possibles.
:::

Une référence indépendante évalue l'espérance de Feynman-Kac discrétisée par
convolution gaussienne rétrograde sur une grille de positions :

$$
\Psi_k(x)=\int\mathcal N(y;x,\nu h)
 e^{-h q(t_{k+1},y)/\lambda}\Psi_{k+1}(y)\,dy,
\qquad \Psi_N(x)=e^{-x^2/(2\lambda)}.
$$

La dérivation du noyau gaussien donne le même estimateur par le premier
incrément, sans échantillonnage de Monte-Carlo. C'est une référence pour la
formule par intégrales de chemin discrétisée en temps. Appliquer sa force
comme une rétroaction constante par morceaux introduit tout de même une
approximation liée au pas de temps par rapport au contrôle optimal en temps
continu. Nous n'identifions pas son coût réalisé à pas fini à la valeur exacte
en temps continu.

Le correcteur de comparaison résout un programme dynamique déterministe avec
le même coût sur l'état et le même effort de commande, en utilisant
$x_{k+1}=x_k+hu_k$. Avec cette référence de comparaison, on se demande ce que
l'on perd en remplaçant la diffusion future par sa moyenne. Il applique sa
première action à l'état bruité observé et recalcule l'action après chaque
pas. Les deux méthodes utilisent donc une rétroaction; elles diffèrent par la
prise en compte ou non de la diffusion future dans la planification. La grille
déterministe des positions a un pas de $0.0025$, ce qui correspond à un pas
d'action de $0.1$, avec une borne numérique sur l'action de $24$. La
convolution de référence utilise un pas de $0.005$ sur $[-6,6]$.

### Choix du passage, coût et échecs

La prise en compte de la diffusion change-t-elle le passage utilisé, et ce
choix réduit-il le coût réalisé? La comparaison utilise 128 suites
indépendantes de perturbations physiques, appariées entre les correcteurs.
L'appariement soumet les méthodes aux mêmes fluctuations; une amélioration
apparente ne peut donc pas s'expliquer par le fait qu'une méthode aurait reçu
un ensemble de perturbations plus facile.

| Bruit physique | Coût, intégrales de chemin échantillonnées | Coût, dynamique moyenne | Coût, référence numérique |
| --- | ---: | ---: | ---: |
| $\nu=0.015$ | $0.221\pm0.085$ | $5.310\pm0.914$ | $0.197\pm0.078$ |
| $\nu=0.3$ | $1.808\pm0.289$ | $11.303\pm1.184$ | $1.767\pm0.279$ |

Les entrées sont des moyennes sur les essais, avec des intervalles de
confiance à 95 % fondés sur l'approximation normale. Le correcteur
échantillonné et la référence numérique indépendante donnent des coûts moyens
semblables. La planification sous la dynamique moyenne a un coût plus élevé
aux deux niveaux de bruit, même si elle replanifie elle aussi à partir de
l'état observé.

À faible bruit, la particule ne se trouve dans le passage large à l'instant
$1.5$ sous aucune des deux méthodes de planification. À bruit élevé, leurs
choix se séparent :

| Correcteur, $\nu=0.3$ | Essais dans le passage large | Intervalle de Wilson à 95 % pour la proportion |
| --- | ---: | ---: |
| Rétroaction par intégrales de chemin échantillonnées | $126/128$ | $[0.945,0.996]$ |
| Planification sous la dynamique moyenne | $32/128$ | $[0.183,0.332]$ |

L'occupation tient compte de tous les essais, y compris ceux qui entraînent
une pénalité. La prise en compte de la diffusion plus forte favorise
l'itinéraire large malgré son détour. C'est l'effet que la cible quadratique
seule ne pouvait pas montrer.

Une pénalité finie permet encore des passages manqués. Un essai échoue si
l'une des positions vérifiées se trouve hors des deux ouvertures pendant
$1<t\leq2$ :

| Bruit physique | Échecs, intégrales de chemin échantillonnées | Échecs, dynamique moyenne |
| --- | ---: | ---: |
| $\nu=0.015$ | $17/128$ | $94/128$ |
| $\nu=0.3$ | $28/128$ | $122/128$ |

À bruit élevé, l'effort de commande moyen est de $1.142$ pour la rétroaction
échantillonnée et de $1.531$ pour la planification sous la dynamique moyenne.
L'écart de coût comprend à la fois l'effort et la pénalité accumulée pour les
positions hors des passages.

:::{figure} _static/brownian_mppi/brownian-outcomes.svg
:label: fig-brownian-outcomes
:width: 100%

Coûts et issues des passages sous des perturbations physiques appariées. Les
barres de coût sont des intervalles à 95 % pour les moyennes; les barres de
proportion sont des intervalles de Wilson à 95 %. La référence numérique et le
correcteur échantillonné ont des issues semblables. La pénalité finie
n'impose aucune garantie sur la probabilité d'échec.
:::

### Erreur d'échantillonnage et erreur de discrétisation

La solution de référence permet de faire varier la précision numérique
indépendamment du bruit physique. À l'état initial à bruit élevé, faire passer
le nombre de candidats de 256 à 16 384 réduit l'erreur quadratique moyenne de
la première action par rapport à la référence par convolution de $0.910$ à
$0.103$, sur 32 lots indépendants. Raffiner le pas de temps de $0.05$ à
$0.00625$ fait passer la première action de référence de $-0.681$ à $-0.749$.
Près du régime où l'itinéraire bascule, cette sensibilité est plus grande :
pour $\nu=0.15$, le même raffinement fait passer l'action de $-0.173$ à
$-0.412$. La démonstration sur grille finie doit donc être interprétée à la
résolution indiquée. Les franchissements entre les instants vérifiés ne sont
pas comptés.

Raffiner la grille déterministe de $0.0025$ à $0.00125$ fait passer son coût
moyen apparié de $5.310$ à $4.915$ à faible bruit et de $11.303$ à $11.446$ à
bruit élevé. Aucune commande exécutée n'atteint la borne numérique sur
l'action. L'écart de coût persiste sous ce raffinement, bien que la référence
déterministe elle-même ne soit pas entièrement convergée. Élargir le domaine
de convolution de $[-6,6]$ à $[-8,8]$ laisse son action initiale à bruit élevé
inchangée à la précision affichée. Tous les réglages, les tableaux d'essais et
les balayages de convergence sont conservés dans les {download}`résultats de l'expérience <artifacts/brownian_mppi/results.json>`.

:::{iframe} ../interactive/brownian-mppi.html
:width: 100%
:title: Rejeu des passages browniens
:class: mppi-replay

Rejouez un essai apparié, modifiez le niveau de bruit physique et examinez
l'effort de commande accumulé et les sorties hors des passages. Le rejeu
utilise des trajectoires enregistrées.
:::

Le correcteur fondé sur la dynamique moyenne attend souvent près du bord d'une
ouverture. Quel terme de son objectif prédit attribue un coût au fait de rester
près de ce bord, et quels événements futurs cette prédiction omet-elle? Le
correcteur stochastique tient compte du coût de correction des fluctuations
futures, ce qui peut favoriser un passage plus large même lorsque l'atteindre
exige un plus long trajet.

## Synthèse et perspectives

L'équation HJB stochastique est l'équation de Bellman en temps continu de la
commande par rétroaction sous bruit physique. Avec un effort quadratique et
des directions de commande et de bruit assorties, la transformation
logarithmique rend cette équation non linéaire linéaire. La valeur transformée
est une espérance exponentielle sur des trajectoires non commandées, et les
corrélations avec le premier incrément de bruit donnent l'action courante
optimale.

L'échantillonnage préférentiel permet de tirer ces trajectoires autour d'une
commande utile. La discrétisation temporelle produit ensuite la forme par
perturbations pondérées qu'utilise MPPI. Le résultat en temps continu porte
sur une action conditionnelle à l'état courant; un nombre fini d'échantillons
et un pas de temps fini introduisent des erreurs, et une suite enregistrée
exige encore une replanification pour fournir une rétroaction.

L'exemple quadratique possède l'équivalence de certitude, alors que des
passages concurrents font que les fluctuations futures influent sur le choix
de l'itinéraire. Les essais appariés et les calculs de référence numériques
séparent cet effet de modélisation des erreurs d'échantillonnage et de
discrétisation. L'hypothèse d'assortiment limite aussi la portée de la
théorie : l'optimiseur de l'avion de @sec-aircraft-mppi utilise un ajustement
distributionnel général à des coûts espérés estimés, ce qui, à lui seul,
n'établit pas une rétroaction optimale.

## Sources de calcul

Les expériences sur les passages lisent des résultats précalculés lors de la
construction du livre. L'échantillonneur de trajectoires et le calcul de
référence ont des implémentations distinctes :

- {download}`Dynamique brownienne, échantillonneur de trajectoires et solutions de référence <code/brownian_mppi.py>`
- {download}`Expérience brownienne et générateur de figures <scripts/build_brownian_mppi_artifacts.py>`
- {download}`Paramètres et résultats de l'expérience <artifacts/brownian_mppi/results.json>`
- {download}`Tableaux des essais appariés <artifacts/brownian_mppi/trials.npz>`

Le [rejeu brownien](interactive/brownian-mppi.html) affiche les simulations
enregistrées sans noyau Python. Les commandes de régénération se trouvent dans
le README du dépôt.

## Exercices

:::{exercise} Assortiment du bruit et discrétisation temporelle
:label: ex-mppi-noise-scaling

Considérons $dX_s=u_s\,ds+\sigma\,dW_s$ avec l'effort courant
$\tfrac12ru_s^2$, où $r>0$ et $\sigma>0$.

**(a)** Trouver $\lambda$ et la variance de la perturbation de commande
nécessaires pour que $x_{k+1}=x_k+h(u_k+\epsilon_k)$ corresponde à la
diffusion.

**(b)** Si l'on divise $h$ par deux, comment l'écart type de $\epsilon_k$
doit-il changer? Qu'advient-il de la variance physique accumulée sur un
horizon de temps fixé si l'on maintient plutôt fixe la variance de la
perturbation?

**(c)** Un double intégrateur stochastique obéit à
$dX_s=V_s\,ds$ et $dV_s=u_s\,ds+\sigma\,dW_s$.
Sa coordonnée de position non actionnée viole-t-elle l'assortiment du bruit?
L'ajout d'un incrément brownien indépendant directement à $dX_s$ le
préserverait-il?
:::

:::{solution} ex-mppi-noise-scaling
:class: dropdown

**(a)** L'assortiment exige $\sigma^2=\lambda/r$, d'où
$\lambda=r\sigma^2$. L'incrément aléatoire de position doit avoir pour
variance $\sigma^2h$. Comme il est égal à $h\epsilon_k$, la variance de la
perturbation est $\operatorname{Var}(\epsilon_k)=\sigma^2/h=\lambda/(rh)$.

**(b)** La variance double et l'écart type est multiplié par $\sqrt2$. Avec
une variance de perturbation fixe $s^2$ et une durée $T$, la variance
accumulée de la position est $(T/h)h^2s^2=Ths^2$, qui tend vers zéro lorsque
$h\to0$. Cette discrétisation tend vers une dynamique déterministe plutôt que
vers la diffusion prescrite.

**(c)** Dans l'ordre d'état $(X,V)$, on prend $D=(0,1)^\top$ et
$B=(0,\sigma)^\top$. Alors $BB^\top=\lambda D r^{-1}D^\top$ avec la valeur de
$\lambda$ ci-dessus. Le bruit et la commande agissent tous deux sur la
vitesse; la position non actionnée ne pose donc aucun problème. Un bruit de
position indépendant ajouterait à la variance de la position un terme positif
que le membre de droite ne peut pas représenter.
:::

:::{exercise} Équivalence de certitude et coûts non linéaires
:label: ex-mppi-certainty-equivalence

Pour $dX=u\,ds+\sqrt\nu\,dW$, minimiser le coût terminal espéré
$\tfrac\kappa2(X_{t_f}-a)^2$ augmenté de l'effort courant $\tfrac r2u^2$.
On suppose $r,\kappa>0$, on pose $\lambda=r\nu$ et l'on note $\tau=t_f-t$.

**(a)** Vérifier que la fonction de valeur

$$
V(t,x)=\frac{\kappa r(x-a)^2}{2(r+\kappa\tau)}
+\frac{r\nu}{2}\log(1+\kappa\tau/r)
$$

satisfait l'équation HJB scalaire et la condition terminale.

**(b)** Dériver la rétroaction optimale et expliquer quelle partie de la
valeur varie avec $\nu$ lorsque $r$ est maintenu fixe.

**(c)** Cet exemple montre-t-il que le bruit ne modifie jamais une décision
optimale? Proposer une modification pour laquelle cette conclusion n'est pas
justifiée.
:::

:::{solution} ex-mppi-certainty-equivalence
:class: dropdown

**(a)** Posons $P(t)=\kappa r/(r+\kappa\tau)$ et
$b(t)=(r\nu/2)\log(1+\kappa\tau/r)$, de sorte que
$V=P(x-a)^2/2+b$. Leurs dérivées sont
$\dot P=P^2/r$ et $\dot b=-\nu P/2$.
En substituant $V_x=P(x-a)$ et $V_{xx}=P$ dans
$V_t-V_x^2/(2r)+\nu V_{xx}/2=0$, les termes quadratiques et les termes
constants s'annulent. En $t=t_f$, $P=\kappa$ et $b=0$, ce qui donne le coût
terminal.

**(b)** La rétroaction est
$u^\star=-V_x/r=\kappa(a-x)/(r+\kappa\tau)$.
Seul le terme indépendant de l'état $b(t)$ dépend de $\nu$. Il augmente le
coût espéré, mais disparaît lorsqu'on prend la dérivée par rapport à l'état.

**(c)** L'annulation utilise la fonction de valeur quadratique et un bruit
additif indépendant de la commande. Une pénalité sur l'état avec un obstacle,
deux régions cibles séparées ou une dérive non linéaire ne préservent pas
nécessairement cette structure. Par exemple, deux passages de largeurs
différentes peuvent exiger des efforts de commande espérés différents pour
contrer la diffusion, même lorsque leurs longueurs de trajet déterministes
sont comparables.
:::

:::{exercise} Comparer des correcteurs de passage au moyen d'essais appariés
:label: ex-mppi-paired-passages

Utiliser les {download}`essais de passage <artifacts/brownian_mppi/trials.npz>` enregistrés.
Les tableaux nommés `<noise>_<method>_cost` contiennent un coût par essai,
avec `noise` égal à `low_noise` ou `high_noise` et `method` égal à
`path_integral` ou `mean_dynamics`. Des indices identiques utilisent les mêmes
incréments browniens physiques.

**(a)** Pour chaque niveau de bruit, calculer les différences par essai
$d_j=C_j^{\mathrm{PI}}-C_j^{\mathrm{mean}}$. Donner leur moyenne empirique et
l'intervalle à 95 % fondé sur l'approximation normale
$\bar d\pm1.96s_d/\sqrt n$, où $s_d$ est l'écart type empirique.

**(b)** Expliquer pourquoi cet intervalle répond à une autre question qu'un
intervalle pour le coût d'une trajectoire future de la particule. Pourquoi
faut-il former la différence avant de calculer son erreur type?

**(c)** Concevoir une expérience complémentaire qui distingue une erreur
d'échantillonnage d'une erreur de discrétisation temporelle. Préciser ce qui
est maintenu fixe dans chaque comparaison et quelle solution de référence est
utilisée. Indiquer ce que les échecs enregistrés vous empêchent d'affirmer au
sujet du correcteur.
:::

:::{solution} ex-mppi-paired-passages
:class: dropdown

**(a)** Le calcul utilise la différence à l'intérieur de chaque essai :

```python
import numpy as np

with np.load("artifacts/brownian_mppi/trials.npz") as trials:
    for noise in ("low_noise", "high_noise"):
        d = (trials[f"{noise}_path_integral_cost"]
             - trials[f"{noise}_mean_dynamics_cost"])
        half_width = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
        print(noise, d.mean(), half_width)
```

Les différences moyennes valent environ $-5.090\pm0.883$ à faible bruit et
$-9.496\pm1.070$ à bruit élevé. Les valeurs négatives favorisent la
rétroaction par intégrales de chemin.

**(b)** Cet intervalle décrit l'incertitude sur la différence de coût moyen
estimée sur des essais indépendants. Un coût réalisé individuel peut être loin
de cette moyenne. L'appariement apporte aussi un terme de covariance :
$\operatorname{Var}(C^{\mathrm{PI}}-C^{\mathrm{mean}})
=\operatorname{Var}(C^{\mathrm{PI}})+\operatorname{Var}(C^{\mathrm{mean}})
-2\operatorname{Cov}(C^{\mathrm{PI}},C^{\mathrm{mean}})$.
Calculer les différences préserve la covariance produite par les
perturbations physiques partagées; traiter les deux tableaux comme
indépendants l'écarte.

**(c)** Maintenir fixes le modèle, l'état, le pas de temps et la grille de
référence tout en faisant varier le nombre de candidats sur des germes
d'échantillonnage indépendants. Comparer la première action à l'action obtenue
par convolution rétrograde pour ce pas de temps. Séparément, faire varier le
pas de temps dans le calcul de référence en maintenant fixes les coefficients
physiques, y compris $\lambda=r\nu$ et la mise à l'échelle correspondante des
perturbations. Ces comparaisons isolent l'erreur de Monte-Carlo et la
résolution temporelle. Les échecs observés et les franchissements non vérifiés
entre les pas de temps excluent toute affirmation d'admissibilité garantie du
passage.
:::

## Autoévaluation

:::{exercise} Bruit physique et échantillonnage
:label: ex-mppi-check-noise

Si l'optimiseur cesse de générer des trajectoires candidates, la particule
brownienne cesse-t-elle de fluctuer? Quelle source d'aléa a été supprimée?
:::

:::{solution} ex-mppi-check-noise
:class: dropdown

Les incréments browniens physiques demeurent. Arrêter l'échantillonneur
supprime les avenirs simulés qui servent à estimer une mise à jour; cela ne
change pas le bruit de processus de la particule.
:::
