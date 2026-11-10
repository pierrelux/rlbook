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
# Commande à horizon glissant

Une trajectoire en boucle ouverte précise les commandes depuis un état initial
jusqu'à un instant final. Pendant l'exécution, la commande $\mathbf u_t$ ne
dépend que de l'horloge : on lit des valeurs précalculées ou on interpole entre
elles. Les
[exemples MPPI](model-predictive-path-integral-control.md#sec-mppi) employaient
déjà une autre règle d'exécution : appliquer une action, observer le nouvel
état et mettre à jour le reste du plan. La commande à horizon glissant applique
cette règle à tout optimiseur à horizon fini adéquat, y compris les méthodes de
transcription développées dans les chapitres précédents.

Les systèmes réels subissent des erreurs de modélisation, des perturbations externes et du bruit de mesure qui s'accumulent au fil du temps. Une trajectoire précalculée perd de sa pertinence à mesure que ces perturbations éloignent l'état réel du système du chemin prédit. La solution consiste à introduire une rétroaction, c'est-à-dire à prendre des décisions de commande qui réagissent à l'état courant au lieu de suivre aveuglément un calendrier prédéterminé. La programmation dynamique fournit le cadre théorique pour dériver des politiques de rétroaction au moyen des fonctions de valeur et des équations de Bellman, mais il existe une approche plus directe qui tire parti des méthodes d'optimisation de trajectoires déjà développées.

Le planificateur à horizon fini peut-il lui-même devenir un correcteur à
rétroaction en résolvant de nouveau le problème chaque fois qu'un nouvel état
est mesuré?

## Objectifs d'apprentissage

Après avoir étudié les dérivations et l'exemple d'équilibrage, vous devriez
pouvoir :

- Transformer un problème de contrôle optimal à horizon fini en loi de
  rétroaction d'état par replanification, et énoncer précisément cette loi de
  rétroaction et son domaine.
- Expliquer pourquoi un horizon court sans éléments terminaux peut laisser un
  système instable sans commande effective, l'immobiliser près d'une
  contrainte ou perdre la faisabilité.
- Démontrer la faisabilité récursive et la convergence en boucle fermée à
  partir d'un coût terminal, d'un ensemble terminal et d'un correcteur local,
  au moyen de la suite candidate décalée.
- Construire un coût terminal quadratique et un ensemble terminal ellipsoïdal
  pour un système linéaire soumis à des contraintes de boîte, et vérifier les
  deux numériquement.
- Déterminer quand la valeur à horizon fini est une borne inférieure, une
  borne supérieure ou ni l'une ni l'autre de la valeur à horizon infini.

## Préalables

Le chapitre utilise la [formulation du contrôle optimal à horizon fini](discrete-time-optimal-control.md),
les [méthodes de transcription](numerical-trajectory-optimization.md) qui la
transforment en programme non linéaire, et l'exécution à horizon glissant
employée par [MPPI](model-predictive-path-integral-control.md#sec-mppi).
L'exemple résolu traite de petits programmes quadratiques convexes; il suffit
de savoir que de tels programmes ont une solution unique et se résolvent de
façon fiable. La programmation dynamique et la commande linéaire quadratique ne
sont pas requises. Lorsque l'exemple utilise un gain de la commande linéaire
quadratique, la propriété sur laquelle il s'appuie est dérivée ici.

(closing-the-loop-by-replanning)=
## Fermer la boucle par replanification

Quelle action de chaque solution à horizon fini faut-il appliquer avant de
mesurer l'état et de décaler l'horizon vers l'avant?

La commande prédictive crée un correcteur à rétroaction en résolvant à répétition des problèmes d'optimisation de trajectoires. Plutôt que de calculer une seule trajectoire pour toute la durée de la tâche, la commande prédictive résout à chaque pas de temps un problème à horizon fini qui part de l'état mesuré courant. Le correcteur n'applique ensuite que la première action de commande de cette solution avant de recommencer tout le processus. Cette stratégie transforme n'importe quelle méthode d'optimisation de trajectoires en correcteur à rétroaction.

L'exemple de la batterie dans [Interfaces de modèle](model-interfaces.md#fast-charging-when-resistance-drifts)
utilisait un limiteur de courant transparent : il vérifiait une enveloppe
locale de tension et de température après la mise à jour d'un paramètre de
résistance. La commande prédictive conserve ce modèle structuré, mais optimise
une suite complète de courants futurs et répète l'optimisation à mesure que les
mesures arrivent.

(the-receding-horizon-principle)=
### Le principe de l'horizon glissant

Ce qui définit la commande prédictive, c'est sa stratégie d'horizon glissant. À chaque pas de temps, le correcteur résout un problème d'optimisation qui regarde une durée fixe dans l'avenir, mais cette fenêtre de prédiction avance constamment dans le temps. L'horizon « glisse » parce qu'il part toujours de l'instant courant et s'étend vers l'avant de la même durée.

Considérons le problème de contrôle optimal en temps discret sous forme de Bolza :

$$
\begin{aligned}
\text{minimiser} \quad & c_N(\mathbf{x}_N) + \sum_{k=0}^{N-1} c_k(\mathbf{x}_k, \mathbf{u}_k) \\
\text{sous les contraintes} \quad & \mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k) \\
& \mathbf{g}_k(\mathbf{x}_k, \mathbf{u}_k) \leq \mathbf{0} \\
& \mathbf{x}_{\text{min}} \leq \mathbf{x}_k \leq \mathbf{x}_{\text{max}} \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}_k \leq \mathbf{u}_{\text{max}} \\
\text{étant donné} \quad & \mathbf{x}_0 = \mathbf{x}_{\text{courant}}
\end{aligned}
$$

Il s'agit du problème de Bolza des [problèmes de contrôle optimal à horizon fini](discrete-time-optimal-control.md),
dont l'indice est compté à partir de l'instant courant plutôt qu'à partir du
début de la tâche. Si l'état est mesuré à l'instant $t$, alors
$\mathbf x_0=\mathbf x_{\text{courant}}$ est cette mesure, $\mathbf x_k$ est
l'état prédit pour l'instant $t+k$, et les $N$ commandes
$\mathbf u_0,\ldots,\mathbf u_{N-1}$ couvrent l'horizon de prédiction, comme
dans l'algorithme MPPI. Pour une tâche variant dans le temps, $c_k$,
$\mathbf f_k$ et $\mathbf g_k$ sont le coût d'étape, la dynamique et les
contraintes de l'instant $t+k$. Le coût terminal $c_N$ est imputé à la fin de
l'horizon de prédiction, et non à la fin de la tâche.

Au pas de temps $t$, ce problème optimise sur l'intervalle $[t, t+N]$. Au pas suivant $t+1$, l'horizon se décale vers $[t+1, t+N+1]$. Ce qui fait fonctionner ce mécanisme, c'est que seule la première commande $\mathbf{u}_0^\star$ de chaque optimisation est appliquée. Les commandes restantes $\mathbf{u}_1^\star, \ldots, \mathbf{u}_{N-1}^\star$ sont écartées, bien qu'elles puissent initialiser l'optimisation suivante par démarrage à chaud.

Ce principe de l'horizon glissant procure une rétroaction sans calculer de politique explicite. En mettant constamment à jour ses prédictions à partir des mesures courantes, la commande prédictive corrige naturellement les perturbations et les erreurs de modèle. Le gaspillage apparent, qui consiste à calculer la plus grande partie de la trajectoire sans l'utiliser, est en réalité le mécanisme qui procure la robustesse.

### Un exemple d'équilibrage

Tenir un bâton debout en équilibre sur le bout du doigt montre ce que la
replanification apporte. La position verticale est un équilibre, mais une
petite inclinaison croît d'elle-même, de sorte qu'une suite de commandes
précalculée légèrement erronée ne reste pas légèrement erronée. Linéarisé
autour de la position verticale, l'angle $\theta$ obéit à

$$
\ddot\theta = \frac{g}{\ell}\,\theta + u ,
$$

où $\ell$ est la longueur du bâton et $u$ est l'accélération angulaire
commandée par le bout du doigt. Avec $g=9.81$ m/s$^2$ et $\ell=1$ m, une
inclinaison croît d'un facteur $e$ environ toutes les
$\sqrt{\ell/g}\approx 0.32$ s. On échantillonne le modèle exactement sous un
bloqueur d'ordre zéro de période $\Delta t=0.05$ s, ce qui donne un modèle
linéaire $\mathbf x_{k+1}=A\mathbf x_k+B u_k$ avec
$\mathbf x=(\theta,\dot\theta)$. La commande est limitée à $|u_k|\le 5$
rad/s$^2$, et l'angle doit rester dans $|\theta_k|\le 0.35$ rad. Le coût
d'étape est quadratique,

$$
c(\mathbf x,u)=\mathbf x^\top Q\mathbf x+R u^2,\qquad Q=\operatorname{diag}(10,1),\quad R=0.05 .
$$

Chaque sous-problème de commande prédictive est donc un programme quadratique
convexe en les $N$ commandes et les $N+1$ états prédits. Le coût terminal
$\mathbf x^\top P\mathbf x$ et l'ensemble terminal
$\{\mathbf x:\mathbf x^\top P\mathbf x\le\alpha\}$ utilisés dans cette section
sont construits dans
[Construction des éléments terminaux](#constructing-terminal-ingredients-for-linear-systems).
Pour l'instant, considérez-les comme donnés.

L'expérience part de $\mathbf x_0=(0.2,\,0.6)$ : le bâton est incliné et
continue de tomber. Un plan d'horizon $N=40$ (deux secondes) est calculé une
seule fois à partir du modèle nominal. On compare ensuite deux exécutions de ce
plan sur un procédé dont le bâton est 15 % plus court que ne le suppose le
modèle, et qui reçoit une chiquenaude de $0.5$ rad/s sur sa vitesse angulaire à
$t=1$ s. La première exécution applique les 40 commandes planifiées en boucle
ouverte. La seconde résout de nouveau le même problème à partir de chaque état
mesuré et n'applique que la première commande.

Avant de regarder le résultat, prédisez quel effet nuit en premier à
l'exécution en boucle ouverte : le bâton plus court ou la chiquenaude.

:::{figure} _static/receding_horizon/replanning.svg
:label: fig-rhc-replanning
:width: 100%

Exécution d'un plan en boucle ouverte comparée à une replanification à chaque
pas, sur un procédé dont le bâton est 15 % plus court que celui du modèle. La
courbe en tirets est le plan appliqué au modèle nominal. Les lignes pointillées
marquent les limites d'angle et de commande. L'exécution en boucle ouverte
franchit la limite d'angle à $t=0.85$ s, avant la chiquenaude. La
replanification sature la commande pendant les $0.4$ premières secondes,
absorbe la chiquenaude et maintient $|\theta|\le 0.28$ rad.
:::

L'erreur de modèle suffit à elle seule à mettre en échec le plan en boucle
ouverte. Le bâton plus court tombe plus vite que prévu, les commandes
planifiées sont trop faibles pour le rattraper, et l'angle franchit sa limite
à $t=0.85$ s, avant l'arrivée de la chiquenaude. L'exécution en boucle ouverte
ne mesure jamais l'état et ne peut donc pas détecter l'erreur. Le correcteur
replanifié résout la même optimisation avec le même modèle erroné, mais chaque
nouvelle résolution part de l'état mesuré, si bien que l'erreur de prédiction
accumulée est corrigée un pas à la fois. Chacun de ses 60 sous-problèmes est
réalisable, et la chiquenaude n'apparaît que comme une seconde salve de
commande. La replanification ne répare pas le modèle. Elle limite la durée
pendant laquelle une erreur de modèle peut agir avant d'être mesurée.

### Le correcteur défini par la replanification

Pour un problème invariant dans le temps, comme la régulation du bâton, le
sous-problème ne dépend de l'instant $t$ qu'à travers son état initial. Notons
$\mathbf u_0^\star(\mathbf x)$ la première commande optimale lorsque le
sous-problème part de $\mathbf x$. La replanification exécute alors la loi de
rétroaction d'état

$$
\kappa_N(\mathbf x)=\mathbf u_0^\star(\mathbf x),\qquad
\mathbf x_{t+1}=\mathbf f\big(\mathbf x_t,\kappa_N(\mathbf x_t)\big).
$$

La loi n'est définie que là où le sous-problème a une solution. On appelle cet
ensemble l'**ensemble réalisable** $\mathcal X_N$ : les états à partir
desquels une suite de commandes admissibles de longueur $N$ satisfait toutes
les contraintes et se termine dans l'ensemble terminal. La loi $\kappa_N$ n'est
jamais écrite explicitement. Elle est évaluée un état à la fois par un
optimiseur. Le chapitre sur les [correcteurs paramétriques](parametric-controllers.md)
étudie l'application $\mathbf x\mapsto\kappa_N(\mathbf x)$ comme un objet à
part entière. Pour une dynamique linéaire, des coûts quadratiques et des
contraintes polyédriques, c'est une fonction continue et affine par morceaux
de l'état {cite}`Bemporad2002`.

:::{figure} _static/receding_horizon/feasible-sets.svg
:label: fig-rhc-feasible-sets
:width: 100%

À gauche : ensembles réalisables du correcteur du bâton avec coût terminal et
ensemble terminal, pour $N=5$ (le plus foncé), $10$ et $20$ (le plus clair),
sur une grille couvrant les limites d'angle. L'ellipse est l'ensemble terminal
$\mathcal X_f$ et l'étoile est l'état initial de la
{numref}`fig-rhc-replanning`. À droite : la première action $\kappa_{20}$ le
long de la droite $\dot\theta=0.8$ rad/s, avec la loi linéaire quadratique
saturée en gris. La courbe de la commande prédictive s'arrête à
$\theta\approx 0.205$ rad, là où le sous-problème devient irréalisable.
:::

La {numref}`fig-rhc-feasible-sets` montre ces deux objets pour le bâton. Les
ensembles réalisables sont emboîtés,
$\mathcal X_5\subset\mathcal X_{10}\subset\mathcal X_{20}$ : un horizon plus
long laisse plus de pas pour atteindre l'ellipse terminale, de sorte que plus
d'états initiaux conviennent. Sur la grille, ils couvrent 35 %, 43 % et 49 %
du rectangle représenté. L'état initial $(0.2,0.6)$ se trouve hors de
l'ellipse terminale, et il faut au moins neuf pas pour l'atteindre. Hors des
bandes ombrées, le bâton tombe trop vite dans le sens de son inclinaison pour
être rattrapé par des commandes d'au plus $5$ rad/s$^2$ sans violer la limite
d'angle.

Le panneau de droite contient une leçon propre à ce petit exemple. Partout où
le sous-problème est réalisable, $\kappa_{20}$ coïncide avec la loi linéaire
saturée $\operatorname{sat}(K\mathbf x)$ à $4\times10^{-5}$ rad/s$^2$ près, où
$K$ est le gain de la commande linéaire quadratique à partir duquel le coût
terminal est construit. Pour un état de dimension deux avec une seule entrée,
la contrainte active le long des chemins optimaux est habituellement la borne
sur l'entrée au premier pas, et l'optimiseur redécouvre la saturation.
L'optimiseur ajoute deux choses dont la loi saturée est dépourvue. D'abord, il
signale qu'un état hors de $\mathcal X_{20}$ est irréalisable au lieu de
renvoyer une commande vouée à l'échec. Ensuite, l'argument de stabilité
ci-dessous l'accompagne. Avec plus d'états, plus d'entrées ou des contraintes
d'état qui deviennent actives en cours de route, la loi de l'optimiseur
s'écarte de toute loi linéaire saturée.

Le démarrage à chaud s'insère naturellement ici. Des états consécutifs sont
proches, de sorte que la solution précédente décalée
$(\mathbf u_1^\star,\ldots,\mathbf u_{N-1}^\star,\mathbf u_{N-1}^\star)$ est
une bonne estimation initiale pour la résolution suivante. La preuve de
stabilité ci-dessous utilise presque la même suite décalée, avec un dernier
élément choisi plus soigneusement.

### Choix de l'horizon et formulation du problème

Le choix de l'horizon de prédiction dépend de l'objectif de commande. Nous distinguons trois cas, chacun exigeant une formulation mathématique différente.

#### Régulation à horizon infini

Pour les problèmes de stabilisation où le système doit fonctionner indéfiniment autour d'un équilibre, la dynamique, les contraintes et le coût d'étape sont généralement invariants dans le temps, si bien qu'on omet leur indice : $\mathbf f_k=\mathbf f$, $\mathbf g_k=\mathbf g$ et $c_k=c$. Le véritable objectif est :

$$
J_\infty = \sum_{k=0}^{\infty} c(\mathbf{x}_k, \mathbf{u}_k)
$$

Comme ce problème ne peut pas être résolu directement, la commande prédictive l'approche par :

$$
\begin{aligned}
\text{minimiser} \quad & c_N(\mathbf{x}_N) + \sum_{k=0}^{N-1} c(\mathbf{x}_k, \mathbf{u}_k) \\
\text{sous les contraintes} \quad & \mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) \\
& \mathbf{x}_N \in \mathcal{X}_f \\
& \text{autres contraintes}
\end{aligned}
$$

Le coût terminal $c_N(\mathbf{x}_N)$ approche $\sum_{k=N}^{\infty} c(\mathbf{x}_k, \mathbf{u}_k)$, le coût restant au-delà de l'horizon. La contrainte terminale $\mathbf{x}_N \in \mathcal{X}_f$ garantit que l'état atteint une région où l'on dispose d'un correcteur stabilisant connu. Sans ces éléments terminaux, l'approximation à horizon fini peut produire un comportement instable, car le correcteur ignore les conséquences au-delà de l'horizon.

#### Tâches de durée finie

Pour une tâche qui se termine à l'instant $T$, le véritable objectif à partir de l'instant courant $t$ est la partie restante de l'objectif de Bolza, écrite en temps absolu :

$$
J_{[t, T]} = c_T(\mathbf{x}_T) + \sum_{s=t}^{T-1} c_s(\mathbf{x}_s, \mathbf{u}_s)
$$

La formulation de la commande prédictive doit s'adapter au fil du temps. Tant que la fin de la tâche se trouve au-delà de l'horizon de prédiction, le sous-problème conserve le coût terminal de substitution $c_N$. Dès que $t+N\geq T$, l'horizon se réduit aux $T-t$ pas restants :

$$
\text{minimiser} \quad
\begin{cases}
c_N(\mathbf{x}_N) + \displaystyle\sum_{k=0}^{N-1} c_k(\mathbf{x}_k, \mathbf{u}_k), & t+N < T, \\[2ex]
c_T(\mathbf{x}_{T-t}) + \displaystyle\sum_{k=0}^{T-t-1} c_k(\mathbf{x}_k, \mathbf{u}_k), & t+N \geq T.
\end{cases}
$$

Dans le second cas, $\mathbf{x}_{T-t}$ est la prédiction de l'état final $\mathbf{x}_T$, de sorte que le coût terminal propre à la tâche s'applique. À l'approche de la fin de la tâche, l'horizon se réduit et le coût terminal passe de l'approximation $c_N$ au véritable coût final $c_T$. Cela empêche le correcteur d'optimiser au-delà de la fin de la tâche, ce qui produirait des actions de commande dénuées de sens ou agressives.

#### Tâches périodiques

Certains systèmes fonctionnent selon des cycles répétitifs où le comportement optimal dépend de l'heure de la journée, du jour de la semaine ou de la saison. Considérons un immeuble commercial où le chauffage coûte plus cher la nuit, où le prix de l'électricité varie d'heure en heure et où les profils d'occupation se répètent chaque jour. Le correcteur prédictif doit tenir compte de ces motifs périodiques tout en planifiant sur un horizon fini.

Pour des tâches de période $T_p$, comme l'exploitation quotidienne d'un immeuble, la formulation tient compte des transitions d'une période à la suivante :

$$
\begin{aligned}
\text{minimiser} \quad & \sum_{k=0}^{N-1} c_k(\mathbf{x}_k, \mathbf{u}_k) \\
\text{où} \quad & \varphi_k = (t + k) \mod T_p \\
& c_k = \begin{cases}
c_{\text{jour}} & \text{si } \varphi_k \in [6\text{ h}, 18\text{ h}] \\
c_{\text{nuit}} & \text{sinon}
\end{cases}
\end{aligned}
$$

Le coût d'étape $c_k$ change avec la phase $\varphi_k$ de l'instant prédit $t+k$ au sein de la période. Les contraintes peuvent elles aussi dépendre de la phase, pour refléter des exigences d'exploitation différentes selon le moment.

### L'algorithme de commande prédictive

La procédure complète de commande prédictive met en œuvre le principe de l'horizon glissant par optimisation répétée :

````{prf:algorithm} Commande prédictive avec gestion de l'horizon
:label: alg-mpc-complete

**Entrées :**
- Horizon de prédiction nominal $N$
- Période d'échantillonnage $\Delta t$
- Type de tâche : {infinie, finie se terminant à l'instant $T$, périodique de période $T_p$}
- Fonctions de coût et dynamique
- Contraintes

**Procédure :**

1. Initialiser le temps $t \leftarrow 0$
2. Mesurer l'état initial $\mathbf{x}_{\text{courant}}$

3. **Tant que** la tâche se poursuit :

   4. **Déterminer l'horizon effectif et les coûts :**
      - Si la tâche est infinie : 
        - $N_t \leftarrow N$
        - Utiliser le coût terminal $c_N$ et la contrainte $\mathcal{X}_f$
      - Si la tâche est finie :
        - $N_t \leftarrow \min(N, T - t)$
        - Si $t + N_t = T$ : utiliser le coût final $c_T$
        - Sinon : utiliser l'approximation $c_N$
      - Si la tâche est périodique :
        - $N_t \leftarrow N$
        - Ajuster les coûts et les contraintes selon la phase

   5. **Résoudre l'optimisation :**
      Minimiser sur $\mathbf{u}_{0:N_t-1}$ sous les contraintes de dynamique, les autres contraintes et $\mathbf{x}_0 = \mathbf{x}_{\text{courant}}$

   6. **Appliquer la commande à horizon glissant :**
      - Extraire $\mathbf{u}^\star_0$ de la solution
      - L'appliquer au système pendant la durée $\Delta t$
      - Mesurer le nouvel état $\mathbf{x}_{\text{courant}}$
      - Avancer le temps : $t \leftarrow t + 1$

7. **Fin Tant que**
````

### Linéarisation successive et approximations quadratiques

Tout solveur qui améliore un plan à horizon fini par approximations
successives peut servir à l'intérieur de la commande prédictive : résoudre à
partir du nouvel état observé, exécuter la première action, puis recommencer.
Le [chapitre sur iLQR et DDP](iterative-trajectory-optimization.md), qui suit
la programmation dynamique, développe un tel solveur. Les gains de rétroaction
de sa passe arrière et la décision externe de replanifier sont des opérations
distinctes.

Pour de nombreux problèmes de régulation et de suivi, la dynamique et les coûts non linéaires que l'on rencontre peuvent être approchés localement par des fonctions linéaires et quadratiques. L'idée de base est de linéariser le système autour du point de fonctionnement courant et d'approcher le coût par une forme quadratique. Chaque sous-problème de commande prédictive se ramène alors à un **programme quadratique**, qu'on peut résoudre de façon fiable et très rapide avec des solveurs standards.

Supposons que la vraie dynamique soit non linéaire,

$$
\mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k).
$$

Autour d'une trajectoire nominale $(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k)$, on effectue un développement au premier ordre :

$$
\mathbf{x}_{k+1} \approx \mathbf{f}_k(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k) 
+ A_k(\mathbf{x}_k - \bar{\mathbf{x}}_k) 
+ B_k(\mathbf{u}_k - \bar{\mathbf{u}}_k),
$$

avec les jacobiennes

$$
A_k = \left.\frac{\partial \mathbf{f}_k}{\partial \mathbf{x}}\right|_{(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k)}, 
\qquad
B_k = \left.\frac{\partial \mathbf{f}_k}{\partial \mathbf{u}}\right|_{(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k)}.
$$

De même, si le coût d'étape est non linéaire,

$$
c_k(\mathbf{x}_k,\mathbf{u}_k),
$$

on l'approche par une forme quadratique au voisinage du point nominal :

$$
c_k(\mathbf{x}_k,\mathbf{u}_k) \;\approx\; 
\|\mathbf{x}_k - \mathbf{x}_k^{\mathrm{ref}}\|_{Q_k}^2 
+ \|\mathbf{u}_k - \mathbf{u}_k^{\mathrm{ref}}\|_{R_k}^2,
$$

avec des matrices de pondération semi-définies positives $Q_k$ et $R_k$, où $\|\mathbf{v}\|_{Q}^2=\mathbf{v}^\top Q\,\mathbf{v}$.

Le sous-problème de commande prédictive qui en résulte a la forme

$$
\begin{aligned}
\min_{\mathbf{x}_{0:N},\mathbf{u}_{0:N-1}} \quad &
\|\mathbf{x}_N - \mathbf{x}_N^{\mathrm{ref}}\|_{P}^2
+ \sum_{k=0}^{N-1} 
\left(
\|\mathbf{x}_k - \mathbf{x}_k^{\mathrm{ref}}\|_{Q_k}^2
+ \|\mathbf{u}_k - \mathbf{u}_k^{\mathrm{ref}}\|_{R_k}^2
\right) \\
\text{s.c.} \quad &
\mathbf{x}_{k+1} = A_k \mathbf{x}_k + B_k \mathbf{u}_k + \mathbf{d}_k, \\
& \mathbf{u}_{\min} \leq \mathbf{u}_k \leq \mathbf{u}_{\max}, \\
& \mathbf{x}_{\min} \leq \mathbf{x}_k \leq \mathbf{x}_{\max}, \\
& \mathbf{x}_0 = \mathbf{x}_{\text{courant}} ,
\end{aligned}
$$

où $\mathbf{d}_k = \mathbf{f}_k(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k) - A_k \bar{\mathbf{x}}_k - B_k \bar{\mathbf{u}}_k$ rend compte du décalage affine local. La pondération terminale $P$ joue le rôle du coût terminal $c_N$.

Comme la dynamique est désormais linéaire et le coût quadratique, ce problème d'optimisation est un programme quadratique convexe. Les programmes quadratiques sont intéressants en pratique : on peut les résoudre à des fréquences de l'ordre du kilohertz avec des méthodes numériques éprouvées, ce qui en fait la pierre angulaire de nombreuses implémentations temps réel de la commande prédictive.

À chaque pas de la commande prédictive, le correcteur met à jour sa linéarisation autour du nouveau point de fonctionnement, construit le programme quadratique local et le résout. Le processus se répète, le modèle linéaire et le coût quadratique étant rafraîchis à chaque réoptimisation. Malgré l'approximation, on obtient ainsi un correcteur en boucle fermée qui hérite de la rapidité de calcul des programmes quadratiques tout en conservant la capacité de suivre des trajectoires du système non linéaire sous-jacent.

## Garanties théoriques

L'optimisation répétée crée une rétroaction, mais quels éléments terminaux
font persister la faisabilité et converger l'état en boucle fermée?

L'approximation à horizon fini de la commande prédictive soulève une nouvelle difficulté : le correcteur ne voit pas les conséquences au-delà de l'horizon. Sans une conception adéquate, cette myopie peut déstabiliser même des systèmes simples. La solution consiste à encoder soigneusement de l'information sur le problème à horizon infini dans l'optimisation à horizon fini, au moyen de ses conditions terminales.

Avant d'entrer dans les mathématiques, il faut d'abord établir ce que signifie « stabilité » et à quelles tâches s'adressent ces garanties théoriques, car la notion de stabilité varie beaucoup d'un objectif de commande à l'autre.

### Notions de stabilité selon les tâches de commande

Les conditions terminales procurent différents types de garanties selon l'objectif de commande. Pour les problèmes de régulation, où la tâche consiste à amener l'état vers un équilibre fixe $(\mathbf{x}_\mathrm{eq}, \mathbf{u}_\mathrm{eq})$ (souvent ramené à l'origine), la garantie de stabilité est la **stabilité asymptotique** : à partir d'un état suffisamment proche de l'équilibre, l'état en boucle fermée satisfait $\mathbf{x}_t \to \mathbf{x}_\mathrm{eq}$ lorsque $t\to\infty$, tandis que les contraintes restent satisfaites tout au long de la trajectoire (**faisabilité récursive**). Cela exige que le coût d'étape $c(\mathbf{x},\mathbf{u})$ soit défini positif en l'écart à l'équilibre.

Lorsqu'on suit une consigne constante, la tâche consiste à suivre une référence constante $(\mathbf{x}^\mathrm{ref},\mathbf{u}^\mathrm{ref})$ qui résout les équations du régime permanent. On traite ce problème en travaillant en **coordonnées d'erreur** $\tilde{\mathbf{x}}=\mathbf{x}-\mathbf{x}^\mathrm{ref}$ et $\tilde{\mathbf{u}}=\mathbf{u}-\mathbf{u}^\mathrm{ref}$, ce qui transforme le problème de suivi en problème de régulation pour le système d'erreur. La garantie de stabilité devient le **suivi** asymptotique, c'est-à-dire $\tilde{\mathbf{x}}_t \to \mathbf{0}$, là encore avec faisabilité récursive.

Les conditions terminales présentées ci-dessous concernent surtout la régulation et le suivi d'une référence constante. Le suivi variant dans le temps et la commande prédictive économique exigent des techniques supplémentaires, comme la commande prédictive à tubes et la théorie de la dissipativité.

### Ce qui échoue sans éléments terminaux

Retirons le coût terminal et l'ensemble terminal du correcteur du bâton en
conservant tout le reste. La {numref}`fig-rhc-horizon` montre la boucle fermée
à partir du même état initial pour plusieurs horizons, sur le modèle exact et
sans perturbations.

:::{figure} _static/receding_horizon/horizon.svg
:label: fig-rhc-horizon
:width: 100%

Angle en boucle fermée sans éléments terminaux pour $N=1,3,5,10$ (nuances
d'orange), et avec coût et ensemble terminaux pour $N=10$ (bleu), sur le
modèle exact. Une croix marque le premier sous-problème irréalisable. Avec
$N=5$, le bâton se stabilise à $\theta\approx0.32$ rad et y reste.
:::

Les échecs diffèrent par leur nature, et chacun a une cause claire.

- Avec $N=1$, l'objectif est $c(\mathbf x_0,u_0)$, et l'état suivant n'y
  intervient pas. Comme $\mathbf x_0$ est fixé, le minimiseur est la commande
  la moins coûteuse, $u_0=0$, à moins qu'une contrainte sur $\mathbf x_1$
  n'impose davantage. Le correcteur ne fait rien jusqu'à ce que la limite
  d'angle soit sur le point d'être franchie, et à ce moment aucune commande
  admissible ne peut l'empêcher : le sous-problème au pas 3 est irréalisable.
- Avec $N=3$, l'horizon couvre $0.15$ s, moins de la moitié du temps
  nécessaire pour que l'inclinaison croisse d'un facteur $e$. Dans cette
  fenêtre, repousser le bâton coûte de l'effort de commande et rapporte peu en
  coût d'angle, si bien que le plan ne fait que le nécessaire pour maintenir
  l'angle prédit sous sa limite. Le sous-problème devient de nouveau
  irréalisable au pas 3.
- Avec $N=5$, chaque sous-problème est réalisable, mais la boucle fermée
  converge vers $\theta\approx0.32$ rad, et non vers la position verticale. À
  cet endroit, chaque plan retient le bâton avec une commande constante et
  maintient l'angle prédit juste à l'intérieur de la limite à la fin de
  l'horizon. La loi de replanification a acquis un second équilibre. La
  faisabilité n'implique donc pas la convergence vers la cible.
- Avec $N=10$, l'horizon est assez long, et le bâton revient à la verticale.
  Son coût en boucle fermée est de $20.83$, contre $20.15$ avec les éléments
  terminaux. L'écart restant est le prix d'un horizon qui ne voit qu'une
  partie du rétablissement.

Un horizon assez long finit par stabiliser le système sans aucun élément
terminal, et une théorie quantifie ce « assez long » {cite}`GrunePannek2017`.
La longueur requise dépend du système et de l'état initial, et rien dans le
sous-problème n'indique quand elle est trop courte. Les éléments terminaux
proposent un autre compromis. Ils fournissent un certificat valable pour tout
horizon, au prix d'un ensemble réalisable plus petit : avec l'ensemble
terminal, le même état initial est irréalisable pour tout $N\le 8$.

### Commande prédictive avec garanties de stabilité

Pour obtenir des garanties théoriques, on enrichit le problème de commande prédictive à horizon fini de trois composantes interdépendantes. Le **coût terminal** $c_N(\mathbf{x})$ approche le coût restant au-delà de l'horizon et sert de substitut à la queue à horizon infini, qu'on ne peut pas optimiser explicitement. L'**ensemble de contrainte terminale** $\mathcal{X}_f$ définit une région où l'on sait localement stabiliser le système. Enfin, le **correcteur terminal** $\pi_f(\mathbf{x})$ est une politique de rétroaction locale qui stabilise le système à l'intérieur de $\mathcal{X}_f$.

Ces composantes doivent satisfaire des conditions de compatibilité précises
pour procurer des garanties théoriques. L'énoncé ci-dessous porte sur la
régulation vers l'origine avec une dynamique $\mathbf f$, un coût d'étape $c$
et des contraintes $\mathbf g(\mathbf x,\mathbf u)\le\mathbf 0$ invariants dans
le temps, ces contraintes incluant les bornes sur l'état et sur l'entrée. Il
suit {cite}`Mayne2000`.

````{prf:theorem} Faisabilité récursive et stabilité asymptotique
:label: thm-mpc-stability

Supposons que $\mathbf f(\mathbf 0,\mathbf 0)=\mathbf 0$, que le coût d'étape
satisfasse $c(\mathbf x,\mathbf u)\ge\gamma\|\mathbf x\|^2$ pour un certain
$\gamma>0$, et que $c_N\ge 0$. Supposons que l'ensemble terminal
$\mathcal X_f$ contienne l'origine et que le correcteur local $\pi_f$
satisfasse deux conditions.

**Invariance sous la commande** : pour tout $\mathbf{x} \in \mathcal{X}_f$, on a $\mathbf{f}(\mathbf{x}, \pi_f(\mathbf{x})) \in \mathcal{X}_f$ (l'ensemble est invariant) et $\mathbf{g}(\mathbf{x}, \pi_f(\mathbf{x})) \leq \mathbf{0}$ (les contraintes restent satisfaites).

**Décroissance de Lyapunov** : pour tout $\mathbf{x} \in \mathcal{X}_f$,

   $$c_N(\mathbf{f}(\mathbf{x}, \pi_f(\mathbf{x}))) - c_N(\mathbf{x}) \leq -c(\mathbf{x}, \pi_f(\mathbf{x}))$$

   où $c$ est le coût d'étape.

Supposons que le procédé suive exactement le modèle de prédiction. Alors, pour
tout état initial $\mathbf x_0\in\mathcal X_N$, la boucle fermée
$\mathbf x_{t+1}=\mathbf f(\mathbf x_t,\kappa_N(\mathbf x_t))$ possède les
propriétés suivantes :

1. **Faisabilité récursive** : $\mathbf x_t\in\mathcal X_N$ pour tout $t$, de
   sorte que chaque sous-problème a une solution.
2. **Décroissance de la valeur** : $V_N(\mathbf x_{t+1})\le V_N(\mathbf x_t)-c(\mathbf x_t,\kappa_N(\mathbf x_t))$.
3. **Borne sur le coût et convergence** :
   $\sum_{t=0}^{\infty}c(\mathbf x_t,\kappa_N(\mathbf x_t))\le V_N(\mathbf x_0)$,
   et $\mathbf x_t\to\mathbf 0$.
````

La stabilité asymptotique, au sens d'une fonction de Lyapunov, exige en outre
que $V_N$ soit majorée par un multiple de $\|\mathbf x\|^2$ au voisinage de
l'origine. C'est le cas, par exemple, lorsque $c_N$ est quadratique et que
l'origine se trouve à l'intérieur de $\mathcal X_f$
{cite}`RawlingsMayneDiehl2017`. Les trois propriétés énumérées sont celles sur
lesquelles s'appuie le praticien, et elles découlent d'une seule construction.

```{prf:proof}
Fixons $\mathbf x\in\mathcal X_N$ et soit
$(\mathbf u_0^\star,\ldots,\mathbf u_{N-1}^\star)$ une solution optimale du
sous-problème à partir de $\mathbf x$, avec les états prédits
$\mathbf x_0^\star=\mathbf x,\mathbf x_1^\star,\ldots,\mathbf x_N^\star$ et
$\mathbf x_N^\star\in\mathcal X_f$. Le correcteur applique
$\mathbf u_0^\star$ et, comme le modèle est exact, l'état suivant est
$\mathbf x^+=\mathbf x_1^\star$.

*Un candidat réalisable à partir de $\mathbf x^+$.* Considérons la suite décalée

$$
\tilde{\mathbf u}=\big(\mathbf u_1^\star,\ldots,\mathbf u_{N-1}^\star,\ \pi_f(\mathbf x_N^\star)\big).
$$

À partir de $\mathbf x^+$, ses $N-1$ premières commandes reproduisent les états
prédits $\mathbf x_2^\star,\ldots,\mathbf x_N^\star$, et ceux-ci satisfont les
contraintes puisqu'ils les satisfaisaient dans la solution d'origine. La
dernière commande satisfait $\mathbf g(\mathbf x_N^\star,\pi_f(\mathbf x_N^\star))\le\mathbf 0$,
et le nouvel état terminal
$\mathbf f(\mathbf x_N^\star,\pi_f(\mathbf x_N^\star))$ appartient à
$\mathcal X_f$, ces deux faits découlant de l'invariance sous la commande. Par
conséquent, $\tilde{\mathbf u}$ est réalisable, donc
$\mathbf x^+\in\mathcal X_N$. Une récurrence sur $t$ démontre la propriété 1.

*Son coût.* Le candidat retire la première étape de la solution d'origine et
ajoute une étape sous $\pi_f$ à la fin. Son coût est

$$
\tilde V=V_N(\mathbf x)-c(\mathbf x,\mathbf u_0^\star)
+\Big[c(\mathbf x_N^\star,\pi_f(\mathbf x_N^\star))
+c_N\big(\mathbf f(\mathbf x_N^\star,\pi_f(\mathbf x_N^\star))\big)
-c_N(\mathbf x_N^\star)\Big].
$$

Le crochet est non positif d'après la condition de décroissance de Lyapunov.
Comme $V_N(\mathbf x^+)$ est le coût optimal à partir de $\mathbf x^+$, il ne
dépasse le coût d'aucun candidat réalisable :

$$
V_N(\mathbf x^+)\le\tilde V\le V_N(\mathbf x)-c(\mathbf x,\kappa_N(\mathbf x)).
$$

C'est la propriété 2.

*Sommation.* En additionnant la propriété 2 sur $t=0,\ldots,T-1$, on obtient la somme télescopique

$$
\sum_{t=0}^{T-1}c(\mathbf x_t,\kappa_N(\mathbf x_t))\le V_N(\mathbf x_0)-V_N(\mathbf x_T)\le V_N(\mathbf x_0),
$$

en utilisant $V_N\ge0$. En faisant tendre $T\to\infty$, on obtient la borne
sur le coût. Les termes d'une série convergente à termes non négatifs tendent
vers zéro, donc
$\gamma\|\mathbf x_t\|^2\le c(\mathbf x_t,\kappa_N(\mathbf x_t))\to0$.
```

La preuve n'utilise jamais l'optimalité du plan au-delà du premier pas. Elle
compare le nouvel optimum à un seul candidat réalisable. Deux conséquences
pratiques en découlent. D'abord, les garanties survivent à un solveur inexact
pourvu que chaque plan renvoyé soit réalisable et ne coûte pas plus que le
candidat décalé. De nombreuses implémentations temps réel le vérifient
explicitement et appliquent le candidat lorsque le solveur ne parvient pas à
l'améliorer. Ensuite, l'hypothèse de modèle exact n'intervient qu'à un seul
endroit, $\mathbf x^+=\mathbf x_1^\star$. En présence d'erreur de modèle ou de
perturbations, l'état mesuré diffère de l'état prédit, et le candidat n'est pas
nécessairement réalisable à partir de celui-ci. Dans la
{numref}`fig-rhc-replanning`, les valeurs enregistrées $V_N(\mathbf x_t)$
décroissent à chaque pas jusqu'à la chiquenaude, puis bondissent de $0.61$ à
$3.23$. Le rétablissement des garanties en présence de telles erreurs est
l'objet de la commande prédictive robuste, traitée au
[chapitre suivant](mpc-variants-reliability.md#robust-mpc).

La borne de la propriété 3 fournit aussi un énoncé quantitatif que
l'expérience du bâton permet de vérifier. Avec les éléments terminaux et
$N=10$, le coût en boucle fermée à partir de $\mathbf x_0=(0.2,0.6)$ vaut
$20.15$, soit $V_{10}(\mathbf x_0)$ à la précision du solveur près. Les deux
sous-sections suivantes expliquent pourquoi la borne est atteinte ici.

(constructing-terminal-ingredients-for-linear-systems)=
### Construction des éléments terminaux pour les systèmes linéaires

Les conditions du {prf:ref}`thm-mpc-stability` semblent abstraites, mais pour
un système linéaire $\mathbf x_{k+1}=A\mathbf x_k+B\mathbf u_k$ avec un coût
d'étape quadratique $\mathbf x^\top Q\mathbf x+\mathbf u^\top R\mathbf u$, on
peut les satisfaire avec deux calculs matriciels et un scalaire.

*Correcteur et coût terminaux.* Choisissons un correcteur terminal linéaire
$\pi_f(\mathbf x)=K\mathbf x$ dont la matrice en boucle fermée $A_K=A+BK$ a
toutes ses valeurs propres à l'intérieur du cercle unité, et un coût terminal
quadratique $c_N(\mathbf x)=\mathbf x^\top P\mathbf x$. En substituant dans la
condition de décroissance de Lyapunov, on obtient

$$
\mathbf x^\top\big(A_K^\top PA_K-P+Q+K^\top RK\big)\mathbf x\le0\qquad\text{pour tout }\mathbf x\in\mathcal X_f .
$$

La condition est satisfaite avec égalité pour tout $\mathbf x$ lorsque $P$
résout l'équation de Lyapunov discrète

$$
A_K^\top PA_K-P+Q+K^\top RK=0 .
$$

Comme $A_K$ est stable, la solution est unique et égale à la série
$P=\sum_{k=0}^{\infty}(A_K^\top)^k(Q+K^\top RK)A_K^k$. Par conséquent,
$\mathbf x^\top P\mathbf x$ est *exactement* le coût obtenu en suivant
$\mathbf u=K\mathbf x$ indéfiniment à partir de $\mathbf x$, pourvu qu'aucune
contrainte ne soit violée en cours de route. Avec ce choix, le coût terminal
n'est plus une estimation de la queue. C'est le coût d'une queue admissible.

Tout $K$ stabilisant convient. Le choix naturel est le gain qui rend cette
queue aussi peu coûteuse que possible, celui de la commande linéaire
quadratique à horizon infini. Sa matrice $P$ résout l'équation algébrique de
Riccati discrète et son gain est $K=-(R+B^\top PB)^{-1}B^\top PA$. Ces
formules sont dérivées par programmation dynamique dans
[Programmation dynamique à horizon fini](finite-horizon-dp.md). L'argument
présenté ici n'a besoin que de l'équation de Lyapunov, que satisfait le couple
de la commande linéaire quadratique.

*Ensemble terminal.* Le coût de la queue n'a de sens que là où $K\mathbf x$
respecte les contraintes à tous les pas futurs. Prenons un ellipsoïde
$\mathcal E_\alpha=\{\mathbf x:\mathbf x^\top P\mathbf x\le\alpha\}$. Il est
invariant sous $A_K$ pour tout $\alpha$ : d'après l'équation de Lyapunov,

$$
(A_K\mathbf x)^\top P(A_K\mathbf x)=\mathbf x^\top P\mathbf x-\mathbf x^\top(Q+K^\top RK)\mathbf x\le\mathbf x^\top P\mathbf x\le\alpha .
$$

Il reste à choisir $\alpha$ de sorte que toutes les contraintes soient
respectées dans $\mathcal E_\alpha$. Écrivons chaque contrainte, après
substitution de $\mathbf u=K\mathbf x$, comme une paire de demi-espaces
$\pm\mathbf a^\top\mathbf x\le h$. Pour le bâton, les lignes sont
$\mathbf a=K^\top$ avec $h=u_{\max}$, et $\mathbf a=(1,0)$ avec
$h=\theta_{\max}$. La plus grande valeur de $\mathbf a^\top\mathbf x$ sur
$\mathcal E_\alpha$ est $\sqrt{\alpha\,\mathbf a^\top P^{-1}\mathbf a}$,
résultat d'un court calcul par multiplicateurs de Lagrange que demandent les
exercices. Le plus grand niveau admissible est donc

$$
\alpha=\min_{\text{lignes}}\ \frac{h^2}{\mathbf a^\top P^{-1}\mathbf a}.
$$

Avec $\mathcal X_f=\mathcal E_\alpha$, les deux conditions du théorème sont
satisfaites. L'invariance et le respect des contraintes sont assurés par
construction, et la décroissance est satisfaite avec égalité. C'est la
construction de {cite}`ChenAllgower1998`, particularisée aux modèles
linéaires. Pour un modèle non linéaire, on applique la même recette à la
linéarisation autour de l'équilibre, en réduisant $\alpha$ jusqu'à ce que les
termes non linéaires négligés soient dominés.

Pour le bâton, le gain de la commande linéaire quadratique est
$K=(-23.69,\,-7.55)$, et la borne sur l'entrée est la ligne active, ce qui
donne $\alpha=4.02$. L'état initial des expériences vérifie
$\mathbf x_0^\top P\mathbf x_0=15.98$, bien en dehors de $\mathcal X_f$, ce qui
explique qu'il faille au moins neuf pas pour l'atteindre. Deux vérifications
numériques du script de construction confirment la construction. L'itération
de la récurrence de Riccati à horizon fini à partir de $P=Q$ converge vers la
même matrice $P$ à $3\times10^{-12}$ près, et sur 2 000 points aléatoires de
$\mathcal X_f$, la plus grande valeur de
$c_N(A_K\mathbf x)-c_N(\mathbf x)+c(\mathbf x,K\mathbf x)$ est de l'ordre de
$10^{-15}$, c'est-à-dire nulle aux erreurs d'arrondi près.

### Bornes de sous-optimalité

Comment la valeur à horizon fini $V_N(\mathbf x)$ se compare-t-elle à la
valeur à horizon infini $V_\infty(\mathbf x)$ du problème contraint? La
réponse dépend des éléments terminaux, et les trois choix utilisés pour le
bâton donnent trois réponses différentes. La {numref}`fig-rhc-values` les
représente.

:::{figure} _static/receding_horizon/values.svg
:label: fig-rhc-values
:width: 100%

Valeur optimale à horizon fini à partir de $\mathbf x_0=(0.2,0.6)$ en fonction
de l'horizon, pour trois choix terminaux. Sans éléments terminaux, la valeur
croît vers $V_\infty$ par en dessous. Avec le seul coût terminal linéaire
quadratique, elle peut aussi se situer sous $V_\infty$. Avec coût terminal et
ensemble terminal, le problème est irréalisable pour $N\le8$ (zone ombrée),
puis égal à $V_\infty$ à la précision du solveur près à partir de $N=9$.
:::

*Sans éléments terminaux, $V_N$ est une borne inférieure.* Prenons la suite
infinie optimale et conservons ses $N$ premières commandes. Elles sont
réalisables pour le problème à $N$ pas, et leur coût n'omet que des termes de
queue non négatifs. Par conséquent, $V_N(\mathbf x)\le V_\infty(\mathbf x)$.
Le même argument montre que $V_N$ est non décroissante en $N$. Pour le bâton,
$V_1=0.76$ et $V_{10}=13.57$, contre $V_\infty=20.15$.

*Avec un coût terminal seul, $V_N$ n'est ni l'une ni l'autre.* Le coût
terminal linéaire quadratique $\mathbf x^\top P\mathbf x$ est le coût de la
queue non contrainte $\mathbf u=K\mathbf x$. Hors de $\mathcal X_f$, cette
queue viole une contrainte, et son coût peut être inférieur à celui de toute
queue admissible. Avec $N=1$, le bâton donne $V_1=17.38<V_\infty$. On ne peut
pas obtenir de borne supérieure sans l'ensemble terminal.

*Avec coût terminal et ensemble terminal, $V_N$ est une borne supérieure.*
Tout plan réalisable à $N$ pas, prolongé par $\pi_f$ à partir de son état
terminal, est une suite infinie admissible. D'après l'équation de Lyapunov,
son coût est égal à l'objectif à $N$ pas, car $c_N$ est exactement le coût du
prolongement. Le problème à horizon infini peut choisir parmi davantage de
suites, donc $V_\infty(\mathbf x)\le V_N(\mathbf x)$ sur $\mathcal X_N$. La
borne exige que $c_N$ soit au moins égal au coût obtenu en suivant $\pi_f$ à
partir de chaque point de $\mathcal X_f$. Un coût terminal qui sous-estime
cette queue perd la borne supérieure.

#### Dérivation de l'erreur d'approximation

La question intéressante est de borner l'erreur d'approximation
$\varepsilon_N=V_N(\mathbf x)-V_\infty(\mathbf x)\ge0$ dans le troisième cas.
Elle représente le coût d'être contraint d'utiliser $\pi_f$ au-delà de
l'horizon au lieu de continuer à optimiser.

Notons $(\mathbf{u}_0^\star, \mathbf{u}_1^\star, \ldots)$ la suite de commandes optimale à horizon infini et $(\mathbf{x}_0^\star, \mathbf{x}_1^\star, \ldots)$ la trajectoire d'états correspondante, où $\mathbf{x}_0^\star = \mathbf{x}$. Le coût à horizon infini est :

$$V_\infty(\mathbf{x}) = \sum_{k=0}^{\infty} c(\mathbf{x}_k^\star, \mathbf{u}_k^\star)$$

Supposons l'horizon assez long pour que cette trajectoire optimale soit entrée
dans l'ensemble terminal au pas $N$, de sorte que
$\mathbf x_N^\star\in\mathcal X_f$. Ses $N$ premières commandes sont alors
réalisables pour le problème à $N$ pas, avec le coût

$$\tilde{V}_N(\mathbf{x}) = \sum_{k=0}^{N-1} c(\mathbf{x}_k^\star, \mathbf{u}_k^\star) + c_N(\mathbf{x}_N^\star)$$

où $c_N(\mathbf{x}_N^\star)$ approche le coût de queue $\sum_{k=N}^{\infty} c(\mathbf{x}_k^\star, \mathbf{u}_k^\star)$.

Comme $V_N(\mathbf{x})$ est le coût optimal à horizon $N$ (qui peut faire mieux que cette suite tronquée particulière), on a $V_N(\mathbf{x}) \leq \tilde{V}_N(\mathbf{x})$. L'erreur d'approximation satisfait donc :

$$\varepsilon_N \leq \tilde{V}_N(\mathbf{x}) - V_\infty(\mathbf{x}) = c_N(\mathbf{x}_N^\star) - \sum_{k=N}^{\infty} c(\mathbf{x}_k^\star, \mathbf{u}_k^\star)$$

Cette borne montre que l'erreur d'approximation dépend de la qualité avec
laquelle le coût terminal approche le coût de queue optimal le long de la
trajectoire optimale à horizon infini, une fois que cette trajectoire est
entrée dans $\mathcal X_f$. Si $\mathbf x_N^\star\notin\mathcal X_f$, la suite
tronquée est irréalisable pour le problème à $N$ pas et la borne ne dit rien.

Le bâton illustre le cas extrême. À l'intérieur de $\mathcal X_f$, la loi
linéaire quadratique satisfait toutes les contraintes; elle est donc optimale
aussi pour le problème contraint, et $c_N=V_\infty$ dans cette région. La
borne donne alors $\varepsilon_N=0$ dès que $\mathbf x_N^\star\in\mathcal X_f$,
ce que montre la {numref}`fig-rhc-values` à partir de $N=9$. Combinée avec la
propriété 3 du théorème, elle donne

$$
V_\infty(\mathbf x_0)\le\sum_{t}c(\mathbf x_t,\kappa_N(\mathbf x_t))\le V_N(\mathbf x_0)=V_\infty(\mathbf x_0),
$$

de sorte que le coût en boucle fermée est égal à $V_\infty$, comme on l'a
mesuré. Pour des systèmes dont la queue optimale n'est pas linéaire,
$\varepsilon_N$ est positive et décroît avec $N$ à une vitesse fixée par la
rapidité avec laquelle les trajectoires optimales s'approchent de
$\mathcal X_f$.

## Synthèse et perspectives

La commande à horizon glissant transforme un optimiseur à horizon fini en
rétroaction en le réinitialisant à partir de chaque état mesuré et en
n'appliquant que la première action planifiée. Il en résulte une loi de
rétroaction d'état $\kappa_N$, définie sur l'ensemble réalisable
$\mathcal X_N$ et évaluée par optimisation. Pour le bâton en équilibre, la
replanification avec un modèle erroné maintient le bâton debout, alors que le
même plan exécuté en boucle ouverte le laisse tomber en moins d'une seconde.

Un horizon court ne garantit rien à lui seul. Il peut laisser un système
instable sans commande effective, perdre la faisabilité ou créer un équilibre
parasite près d'une contrainte. Un coût terminal, un ensemble terminal et un
correcteur local invariant relient le problème tronqué au problème à horizon
infini. La suite candidate décalée démontre alors la faisabilité récursive, la
décroissance de la valeur et un coût en boucle fermée au plus égal à
$V_N(\mathbf x_0)$. Pour les systèmes linéaires, le coût terminal résout une
équation de Lyapunov, et l'ensemble terminal est un ellipsoïde dont la taille
découle des contraintes sous forme explicite.

La boucle de base laisse plusieurs choix de conception en suspens. Comment le
même mécanisme de replanification doit-il représenter le suivi, les objectifs
économiques, l'incertitude, les décisions hybrides, les échecs du solveur et
les échéances temps réel strictes? Le chapitre [Variantes de la commande
prédictive et exploitation fiable](mpc-variants-reliability.md) organise ces
choix. Les garanties présentées ici supposaient aussi un modèle exact, et les
formulations robustes et stochastiques de ce chapitre traitent de ce qui les
remplace.

## Exercices

:::{exercise} Un procédé scalaire instable
:label: ex-rhc-scalar

Considérons $x_{k+1}=2x_k+u_k$ avec $|u_k|\le1$ et le coût d'étape $x^2+u^2$.

1. Montrez que la commande prédictive à un pas sans éléments terminaux
   applique $u=0$ à partir de tout état, et concluez qu'elle ne stabilise pas
   l'origine.
2. Prenez le correcteur terminal $u=Kx$ avec $K=-3/2$. Résolvez l'équation de
   Lyapunov scalaire pour obtenir $P$, et vérifiez la condition de
   décroissance.
3. Trouvez le plus grand $\alpha$ tel que $\{x:Px^2\le\alpha\}$ respecte la
   borne sur l'entrée sous $u=Kx$. Exprimez l'ensemble terminal comme un
   intervalle.
:::

:::{solution} ex-rhc-scalar
:class: dropdown

1. Avec $N=1$, l'objectif est $x_0^2+u_0^2$, et $x_0$ est fixé, donc le
   minimiseur est $u_0=0$. La boucle fermée est $x_{t+1}=2x_t$, qui diverge à
   partir de tout état non nul.
2. $A_K=2-\tfrac32=\tfrac12$ et $Q+K^\top RK=1+\tfrac94=\tfrac{13}4$.
   L'équation de Lyapunov $\tfrac14P-P+\tfrac{13}4=0$ donne $P=\tfrac{13}3$.
   La décroissance est satisfaite avec égalité :
   $P(\tfrac12x)^2-Px^2+\tfrac{13}4x^2=(\tfrac{13}{12}-\tfrac{13}3+\tfrac{13}4)x^2=0$.
3. $|Kx|\le1$ signifie $|x|\le\tfrac23$. La formule donne
   $\alpha=h^2/(a^2/P)=1/(\tfrac94\cdot\tfrac3{13})=\tfrac{52}{27}$, et
   $\{x:\tfrac{13}3x^2\le\tfrac{52}{27}\}=\{|x|\le\tfrac23\}$.
:::

:::{exercise} Un pas avec le coût terminal linéaire quadratique
:label: ex-rhc-one-step-lqr

Soit $P$ la solution de l'équation algébrique de Riccati discrète pour
$(A,B,Q,R)$, et soit $K=-(R+B^\top PB)^{-1}B^\top PA$. Montrez que la
commande prédictive à un pas avec le coût terminal $\mathbf x^\top P\mathbf x$
et sans contrainte active applique exactement $\mathbf u=K\mathbf x$. Quelle
courbe de la {numref}`fig-rhc-values` cela explique-t-il pour $N=1$, et
pourquoi cette valeur se situe-t-elle sous $V_\infty$?
:::

:::{solution} ex-rhc-one-step-lqr
:class: dropdown

L'objectif est
$\mathbf x^\top Q\mathbf x+\mathbf u^\top R\mathbf u+(A\mathbf x+B\mathbf u)^\top P(A\mathbf x+B\mathbf u)$.
En annulant son gradient par rapport à $\mathbf u$, on obtient
$(R+B^\top PB)\mathbf u=-B^\top PA\mathbf x$, c'est-à-dire $\mathbf u=K\mathbf x$.
Sans contraintes, sa valeur est $\mathbf x^\top P\mathbf x$ d'après l'équation
de Riccati, soit le coût à horizon infini non contraint. C'est la courbe verte
en $N=1$. À partir de $\mathbf x_0=(0.2,0.6)$, la queue linéaire quadratique
non contrainte dépasse la borne sur la commande; elle est donc moins coûteuse
que toute queue admissible, et la valeur $17.38$ se situe sous la valeur
contrainte $V_\infty=20.15$.
:::

:::{exercise} Fonction d'appui d'un ellipsoïde
:label: ex-rhc-ellipsoid

Soit $P$ symétrique définie positive et $\alpha>0$. Montrez que
$\max\{\mathbf a^\top\mathbf x:\mathbf x^\top P\mathbf x\le\alpha\}=\sqrt{\alpha\,\mathbf a^\top P^{-1}\mathbf a}$,
et déduisez-en la formule du niveau terminal $\alpha$.
:::

:::{solution} ex-rhc-ellipsoid
:class: dropdown

Le maximum d'une fonction linéaire sur l'ellipsoïde est atteint sur sa
frontière. La stationnarité de
$\mathbf a^\top\mathbf x-\lambda(\mathbf x^\top P\mathbf x-\alpha)$ donne
$\mathbf x=P^{-1}\mathbf a/(2\lambda)$. En substituant dans
$\mathbf x^\top P\mathbf x=\alpha$, on obtient
$2\lambda=\sqrt{\mathbf a^\top P^{-1}\mathbf a/\alpha}$, et la valeur est
$\mathbf a^\top P^{-1}\mathbf a/(2\lambda)=\sqrt{\alpha\,\mathbf a^\top P^{-1}\mathbf a}$.
Le demi-espace $\mathbf a^\top\mathbf x\le h$ contient l'ellipsoïde exactement
lorsque cette valeur est au plus $h$, c'est-à-dire lorsque
$\alpha\le h^2/(\mathbf a^\top P^{-1}\mathbf a)$. Par symétrie, la même
condition couvre $-\mathbf a^\top\mathbf x\le h$. Le minimum sur toutes les
lignes donne $\alpha$.
:::

:::{exercise} L'équilibre parasite
:label: ex-rhc-parking

À l'aide de `code/receding_horizon.py`, exécutez le correcteur sans éléments
terminaux avec $N=5$ à partir de $(0.2,0.6)$ pendant 200 pas. Examinez l'état
final, la commande appliquée et la trajectoire prédite du dernier
sous-problème. Expliquez pourquoi l'état est un point fixe de la boucle
fermée, et pourquoi il n'est pas un point fixe de la dynamique en boucle
ouverte avec $u=0$. Trouvez le plus petit horizon pour lequel le bâton revient
à la verticale à partir de cet état initial.
:::

:::{solution} ex-rhc-parking
:class: dropdown

La boucle fermée se stabilise en $\dot\theta=0$, $\theta\approx0.32$ rad, avec
une commande constante proche de $-(g/\ell)\,\theta\approx-3.1$ rad/s$^2$, qui
compense l'accélération gravitationnelle. Le dernier plan maintient le bâton
presque immobile, et son angle prédit s'approche de la limite à la fin de
l'horizon. Sur cinq pas, repousser le bâton coûte plus d'effort de commande
que cela n'économise de coût d'angle, si bien que le plan qui maintient l'état
est optimal, et l'application de sa première commande reproduit le même état.
Avec $u=0$, ce même état s'effondre. Avec $N=6$, le bâton s'immobilise encore,
près de $\theta\approx0.29$ rad. $N=7$ est le plus petit horizon qui le ramène
à la verticale en moins de 200 pas, avec un coût en boucle fermée de $34.1$,
et $N=8$ abaisse ce coût à $24.1$.
:::

:::{exercise} Là où la preuve échoue sous erreur de modèle
:label: ex-rhc-mismatch

Répétez l'expérience de replanification de la {numref}`fig-rhc-replanning`, et
enregistrez $V_N(\mathbf x_t)$ le long de la boucle fermée. Repérez chaque pas
où $V_N(\mathbf x_{t+1})>V_N(\mathbf x_t)-c(\mathbf x_t,\mathbf u_t)$. Quelle
ligne de la preuve du {prf:ref}`thm-mpc-stability` échoue à ces pas?
Raccourcissez ensuite davantage le bâton du procédé et trouvez la longueur à
laquelle le correcteur replanifié rencontre pour la première fois un
sous-problème irréalisable.
:::

:::{solution} ex-rhc-mismatch
:class: dropdown

La preuve suppose $\mathbf x^+=\mathbf x_1^\star$. La chiquenaude viole cette
hypothèse de $0.5$ rad/s en un seul pas, et $V_N$ bondit de $0.61$ à $3.23$.
Le bâton plus court la viole d'une petite quantité à chaque pas, de sorte que
l'inégalité de décroissance peut échouer d'une certaine marge même là où $V_N$
continue de décroître. Le candidat décalé est construit à partir de l'état
prédit; il n'est donc pas nécessairement réalisable à partir de l'état mesuré.
La faisabilité récursive n'est plus garantie et, pour un bâton suffisamment
court, elle échoue. Le seuil dépend de la chiquenaude et de l'état initial,
ce qui explique pourquoi la commande prédictive robuste resserre les
contraintes à l'aide de bornes sur l'erreur de prédiction.
:::

:::{exercise} Une limite de vitesse dans l'ensemble terminal
:label: ex-rhc-rate

Ajoutez la contrainte $|\dot\theta_k|\le1$ rad/s au bâton au moyen de
`StickParameters(rate_max=1.0)`. Recalculez $\alpha$ et indiquez quelle ligne
est désormais active. Recalculez les ensembles réalisables de la
{numref}`fig-rhc-feasible-sets` et vérifiez si $\kappa_{20}$ coïncide toujours
avec $\operatorname{sat}(K\mathbf x)$.
:::

:::{solution} ex-rhc-rate
:class: dropdown

Le niveau terminal devient le minimum sur trois lignes. Pour la ligne de
vitesse, $\mathbf a=(0,1)$ et $h=1$ donnent
$h^2/(\mathbf a^\top P^{-1}\mathbf a)\approx4.39$. Pour la ligne d'angle, la
valeur est $8.99$, et pour la ligne d'entrée, $4.02$. La borne sur l'entrée
reste active, de sorte que $\alpha$ et $\mathcal X_f$ ne changent pas, bien
que la borne de vitesse s'en approche. La fonction `terminal_level` effectue
le même calcul. Les ensembles réalisables sont coupés par les droites
horizontales $\dot\theta=\pm1$. Là où la borne de vitesse devient active le
long du chemin prédit optimal, la première commande peut différer de la loi
linéaire saturée; il faut donc mesurer l'écart plutôt que le supposer.
:::

## Sources numériques

Le modèle, le sous-problème de commande prédictive et les vérifications se
trouvent dans un seul module, et les figures et les nombres sont produits par
un seul script :

- {download}`Modèle du bâton en équilibre et commande prédictive linéaire <code/receding_horizon.py>`
- {download}`Script des expériences et des figures <scripts/build_receding_horizon_artifacts.py>`
- {download}`Nombres cités dans ce chapitre <artifacts/receding_horizon/metrics.json>`

Exécutez `uv run python scripts/build_receding_horizon_artifacts.py` à partir
de la racine du dépôt pour régénérer les quatre figures et les mesures, en
environ une minute. Les compilations habituelles du livre lisent les figures
enregistrées. Les tests de `tests/test_receding_horizon.py` vérifient
l'équation de Lyapunov, le niveau terminal, l'inégalité du candidat décalé, la
faisabilité récursive et les bornes sur $V_N$ sur de courtes exécutions.

## Autoévaluation

:::{exercise} Horizon glissant
:label: ex-mpc-check-1

Un solveur de commande prédictive renvoie une suite $(\mathbf{u}_0^\star,\ldots,\mathbf{u}_{N-1}^\star)$. Quelles commandes applique-t-on normalement avant de résoudre de nouveau le problème?
:::

:::{solution} ex-mpc-check-1
:class: dropdown

Seule la première commande (ou un premier bloc court de commandes) est appliquée. L'état est mesuré de nouveau et l'horizon est décalé avant la réoptimisation.
:::

:::{exercise} Éléments terminaux
:label: ex-mpc-check-2

Quels rôles jouent un coût terminal et une contrainte terminale dans la commande prédictive à horizon fini?
:::

:::{solution} ex-mpc-check-2
:class: dropdown

Le coût terminal approche la valeur au-delà de l'horizon; la contrainte terminale peut maintenir le point final dans une région à partir de laquelle un correcteur connu demeure réalisable et stable.
:::

:::{exercise} Réponse aux perturbations
:label: ex-mpc-check-3

Pourquoi le fait de résoudre de nouveau la même optimisation à horizon fini après chaque mesure procure-t-il une rétroaction, même si le modèle de prédiction est déterministe?
:::

:::{solution} ex-mpc-check-3
:class: dropdown

Le nouvel état mesuré contient l'effet accumulé des perturbations et de l'erreur de modèle. Réinitialiser l'optimisation à partir de cet état modifie les commandes planifiées en conséquence.
:::
