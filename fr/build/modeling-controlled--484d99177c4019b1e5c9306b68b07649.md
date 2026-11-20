---
description: Formuler des problèmes de décision séquentielle en précisant leur état, leur canal d'action, leur incertitude, leurs observations, leurs contraintes et l'interface de modèle disponible.
kernelspec:
  name: python3
  display_name: Python 3
---
# Systèmes commandés : frontières, états et actions

L'introduction a présenté l'état, l'action, l'évolution, l'objectif et les
contraintes comme les ingrédients d'un problème de décision séquentielle.
Avant qu'un algorithme puisse comparer des actions, comment relier ces
ingrédients à un système physique précis et à un décideur précis?

On peut rendre une balançoire simulée facile à commander en plaçant un moteur
au pivot de suspension. Une personne sur une vraie balançoire ne dispose
d'aucun moteur de ce genre. Elle peut changer la forme de son corps, et la
chaîne peut tirer mais pas pousser. Un optimiseur peut produire une trajectoire
réussie pour le premier modèle alors que ni son action ni sa suspension
n'existent dans le système visé.

Cet échec survient avant le choix de l'algorithme de commande ou
d'apprentissage. Un **modèle de décision** doit dire ce qui évolue, ce qu'on
peut observer, quelles interventions sont possibles, comment l'incertitude
intervient et quelles contraintes doivent être respectées. Sa **structure
d'information** précise ce que sait le décideur au moment où chaque action est
choisie. Ces choix mènent des systèmes physiques aux équations d'état, puis aux
simulateurs, aux transitions enregistrées et aux environnements interactifs.
Chaque étude de cas met à l'épreuve une partie du modèle qui en résulte.

L'optimisation de trajectoires, la commande prédictive, la programmation
dynamique et l'apprentissage par renforcement raisonnent tous sur les
conséquences des actions. Ils ne peuvent comparer que les conséquences que le
modèle rend possibles, y compris le résultat prédit d'une action qui n'a pas
encore été essayée. Une telle solution de rechange non observée s'appelle un
**contrefactuel**. Un actionneur fictif engendre des plans irréalisables, un
état incomplet détruit de l'information prédictive, et un jeu de données
historiques ne peut pas répondre à des questions contrefactuelles sur des
actions qu'il ne couvre pas. La modélisation détermine donc quels calculs et
quelles conclusions auront un sens par la suite.

Les choix de modélisation suivent un ordre de dépendance utile. La frontière
du système sépare d'abord les variables internes des influences externes.
Cette frontière détermine l'état, l'observation, l'action et la perturbation.
La dynamique décrit ensuite comment ces grandeurs interagissent, tandis que
l'objectif et les contraintes définissent le problème de décision. Enfin,
l'interface de modèle disponible détermine si un algorithme peut dériver des
équations, lancer de nouvelles simulations ou seulement analyser un journal
fixe. Le chapitre suit cet ordre.

::::{admonition} Objectifs d'apprentissage
:class: note

Après la lecture de ce chapitre, vous devriez pouvoir :

- formuler un problème de décision séquentielle en précisant sa frontière, son
  état, son action, sa perturbation, son observation, son objectif et ses
  contraintes;
- repérer le canal physique par lequel une action modifie le système;
- écrire des modèles d'état déterministes et stochastiques en temps continu et
  en temps discret;
- distinguer l'état de l'observation, et une suite d'actions en boucle ouverte
  d'une politique de rétroaction;
- déterminer quelles opérations sont fournies par des équations, des
  simulateurs, des transitions enregistrées et des environnements interactifs;
- séparer la structure connue des composantes apprises et concevoir une
  expérience contrôlée capable de révéler une erreur de modélisation.
::::

::::{admonition} Préalables
:class: tip

Le chapitre suppose connues l'algèbre linéaire, le calcul différentiel à
plusieurs variables, les probabilités élémentaires et la signification d'une
équation différentielle ordinaire. L'intégration numérique est revue dans
[Résoudre des problèmes à valeur initiale](appendix_ivps.md).
::::

## Frontières du système et canaux d'action

Quelles variables physiques appartiennent au système commandé, et quelles
interventions le décideur peut-il réellement appliquer à travers cette
frontière?

Une balançoire, un pont roulant et un houlomoteur oscillent tous. Leur
ressemblance visuelle n'en fait pas le même problème de commande. La personne
sur la balançoire change la forme de son corps, le pont roulant accélère le
point de suspension, et le houlomoteur modifie une charge dissipative pendant
que la mer fournit le forçage.

La frontière indique quels composants physiques appartiennent au système
modélisé et quelles influences viennent de l'extérieur. Le canal d'action
indique ensuite l'intervention dont dispose le décideur à l'intérieur de cette
frontière. Ces deux choix doivent précéder les équations, car ils déterminent
quels termes d'entrée les équations ont le droit de contenir.

:::{exercise} Repérer le canal d'action
:label: ex-dynamics-opening-action-channel

Avant de lire le tableau, prédisez quel système doit créer l'oscillation, lequel
doit la supprimer et lequel doit conserver le mouvement pour en extraire de
l'énergie. Nommez ensuite la grandeur physique que chaque correcteur peut
modifier.
:::

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from modeling_interfaces import make_overview_figure
```

```{code-cell} python
:tags: [remove-input]
:label: fig-modeling-action-channels
:caption: Les systèmes partagent une silhouette oscillante, mais offrent des interfaces de commande différentes. La personne sur la balançoire change sa forme interne, le pont roulant accélère son pivot et le houlomoteur modifie une charge dissipative non négative. Leurs mouvements souhaités et leurs actions réalisables diffèrent donc.

overview_figure = make_overview_figure()
display(overview_figure)
plt.close(overview_figure)
```

| système | l'état doit décrire | l'action modifie | influence externe | objectif | contrainte stricte |
|---|---|---|---|---|---|
| balançoire | angle de suspension et posture de la personne | forme interne du corps | gravité et traînée | ajouter de l'énergie mécanique | une chaîne peut tirer mais pas pousser |
| pont roulant | position du chariot et angle de la charge | accélération du chariot | gravité et amortissement | déplacer la charge et supprimer le balancement | l'accélération est bornée |
| houlomoteur | angle et vitesse du volet | amortissement de la prise de force | vagues incidentes | extraire de l'énergie dans les limites de mouvement | l'amortissement passif est non négatif |

:::{solution} ex-dynamics-opening-action-channel
:class: dropdown

La personne sur la balançoire crée l'oscillation en changeant la forme de son
corps. Le pont roulant supprime le balancement de la charge en modifiant
l'accélération du chariot. Le houlomoteur conserve le mouvement tout en
extrayant de l'énergie en choisissant un amortissement de prise de force non
négatif. Les systèmes se ressemblent, mais aucun n'offre une force arbitraire
appliquée à la coordonnée qui oscille.
:::

Les trois systèmes exigent des entrées différentes, mais leurs équations
mécaniques s'organisent selon un même gabarit comptable. Soit $q$ le vecteur
des coordonnées indépendantes nécessaires pour décrire la configuration, comme
un angle ou la position d'un chariot. Ces coordonnées s'appellent des
**positions généralisées**. Leurs vitesses sont $v=\dot q$; $z$ regroupe des
états supplémentaires de l'actionneur ou de l'environnement; et $u$ est
l'entrée commandée. Un modèle de dimension finie assez général s'écrit

$$
\dot q=v,
\qquad
M(q,z)\dot v+c(q,v,z,\dot z)+\nabla_q V(q,z)+r(q,v,z)
=B(q,v,z)u+G(q,v,z)\xi_v(t),
\qquad
\dot z=g(q,v,z,u,\xi_z(t)).
$$

La matrice de masse $M$ associe aux accélérations des coordonnées les forces
ou couples correspondants, appelés collectivement **forces généralisées**.
Elle est définie positive, ce qui garantit que toute vitesse non nulle a une
énergie cinétique positive. Le terme $c$ regroupe les forces d'inertie qui
dépendent de la vitesse et de la configuration, y compris les effets de
Coriolis et centrifuges. Le gradient $\nabla_q V$ figure du côté gauche, si
bien que la force conservative correspondante est $-\nabla_q V$. Le terme $r$
représente la résistance dissipative. Les colonnes de $B$ et de $G$ précisent
les directions physiques par lesquelles entrent les actions $u$ et les entrées
externes $\xi=(\xi_v,\xi_z)$. La dissipation exige $v^\top r(q,v,z)\geq0$, de
sorte que la résistance ne peut pas ajouter d'énergie mécanique. Une action
peut agir directement par $Bu$ ou indirectement en modifiant $z$. Une
coordonnée mobile dont on modélise l'inertie appartient à $q$; un paramètre
fixe ou un signal imposé n'a pas besoin de sa propre équation d'état.

On ne peut pas déduire le terme $B(q,v,z)u$ du seul mot « action ». Une entrée
vectorielle sans contrainte omettrait la géométrie d'actionnement interne de
la balançoire, la limite d'accélération du pont roulant et la contrainte de
passivité du houlomoteur.

:::{sidebar} Lien facultatif : modèles hamiltoniens à ports
Le gabarit mécanique ci-dessus est un cas particulier, exprimé en coordonnées,
d'une description énergétique plus large. Un modèle hamiltonien à ports
s'écrit

$$
\dot x=[J(x)-R(x)]\nabla H(x)+G(x)u,
\qquad y=G(x)^\top\nabla H(x),
$$

où $H$ emmagasine l'énergie, la matrice antisymétrique $J$ l'achemine, $R$ la
dissipe, et $(u,y)$ forment un port de puissance externe. La même structure
peut relier des composants mécaniques, électriques, fluidiques et thermiques.
Lorsque $H$ ne dépend pas explicitement du temps, son bilan d'énergie est

$$
\dot H=-\nabla H(x)^\top R(x)\nabla H(x)+y^\top u.
$$

Le premier terme dissipe l'énergie emmagasinée, tandis que $y^\top u$ est la
puissance fournie par le port {cite}`vanDerSchaftJeltsema2014`. Ce chapitre
utilise la forme en coordonnées, plus concrète.
:::

La frontière du système détermine aussi quelles variables sont externes. En
mer, le couple des vagues incidentes est une perturbation non commandée pour
le houlomoteur. Dans un bassin à houle de laboratoire, une personne peut
commander un batteur motorisé qui génère les vagues. La consigne du batteur
devient alors une action, tandis que le couple hydrodynamique qui en résulte
reste une force sur le volet. Le houlomoteur n'a pas changé; ce sont la
frontière modélisée, le décideur et l'actionneur physique qui ont changé.

Une fois la frontière et le canal d'action fixés, les autres variables
internes doivent être résumées sous une forme qui prédit la suite. Ce résumé
prédictif est l'état.

## Modèles d'état

Une fois la frontière et le canal d'action fixés, quelles variables résument
assez bien le passé pour prédire l'effet de la prochaine action?

Un état résume l'information passée nécessaire pour prédire l'évolution future
sous une action choisie. Le mot « nécessaire » dépend du modèle. Par exemple,
la température de l'air intérieur peut à elle seule ne pas prédire la
température suivante si l'on omet la chaleur emmagasinée dans les murs. Deux
pièces dont l'air est à la même température, mais dont les murs ne le sont
pas, réagissent différemment une fois le chauffage éteint. Ajouter la
température des murs à l'état lève cette ambiguïté dans un modèle thermique
simple.

Une fois l'état choisi, il devrait rendre l'historique antérieur inutile pour
la prédiction à un pas. Pour un modèle stochastique en temps discret, cette
exigence est la propriété de Markov :

$$
\Pr(x_{t+1}\in A\mid x_{0:t},u_{0:t})
=\Pr(x_{t+1}\in A\mid x_t,u_t),
$$

pour tout ensemble $A$ d'états suivants possibles. Le membre de gauche est
conditionné sur tout l'historique des états et des actions, tandis que celui
de droite ne retient que l'état et l'action courants. Leur égalité signifie
que l'historique plus ancien n'apporte aucune information supplémentaire sur
l'état suivant. La définition est relative au modèle déclaré et à l'échelle de
temps. Un simulateur peut stocker un grand état interne tout en ne présentant
au correcteur qu'une observation plus petite, et cette observation n'a pas
besoin d'être elle-même markovienne.

Un modèle d'état peut conserver les perturbations et les erreurs de capteur
comme entrées explicites. En temps discret, un tel modèle prend la forme

$$
x_{t+1}=f_t(x_t,u_t,\xi_t),
\qquad
y_t=h_t(x_t,u_t,\nu_t).
$$

Ici, $f_t$ fait avancer l'état, tandis que $h_t$ associe à l'état et à
l'action la sortie de capteur $y_t$. La variable $\xi_t$ est une perturbation
du procédé qui modifie l'évolution physique, alors que $\nu_t$ est un bruit de
mesure qui ne modifie que l'observation rapportée. Pour des valeurs fixées de
ces entrées, les deux équations sont des applications déterministes. La
section stochastique leur attribuera plus loin des lois de probabilité.

En temps continu, les équations correspondantes sont

$$
\dot x(t)=f(x(t),u(t),\xi(t)),
\qquad
y(t)=h(x(t),u(t),\nu(t)).
$$

La première équation précise un taux de variation plutôt qu'une mise à jour à
un pas. Beaucoup de lois physiques sont plus compactes en temps continu, car
la mécanique, la théorie des circuits, la dynamique des fluides, les transferts
thermiques et la cinétique chimique décrivent des taux, des flux et des bilans
de conservation. Ces disciplines fournissent une structure réutilisable : la
géométrie, les lois de conservation, les unités et les flux d'énergie
admissibles n'ont pas à être réappris à partir de trajectoires. Les paramètres
inconnus ou les relations empiriques, comme les lois de traînée ou de
transfert thermique, peuvent plutôt être estimés à l'intérieur de cette
structure.

Les équations continues exigent encore une convention d'échantillonnage avant
qu'un correcteur numérique puisse les utiliser. Les capteurs et les interfaces
de commande fonctionnent à des instants discrets, même si le procédé physique
continue d'évoluer entre deux mises à jour. Étant donné une règle de maintien,
qui précise comment une action est conservée entre deux mises à jour, et une
période d'échantillonnage $\Delta t$, l'intégration de l'équation
différentielle définit une transition discrète $x_{k+1}=F_{\Delta t}(x_k,u_k)$.
L'application $F_{\Delta t}$ s'appelle le flot exact sur un intervalle
d'échantillonnage. Un intégrateur numérique approche cette application.
Changer $\Delta t$ ou la règle de maintien change la transition échantillonnée,
et non la loi physique sous-jacente. Cela peut aussi changer le modèle de
perturbation, car un événement bref peut être visible sur une grille
d'échantillonnage et absent d'une autre.

:::{figure} _static/sampling-and-integration.svg
:label: fig-sampling-and-integration
:alt: Un modèle scalaire continu de vitesse est échantillonné à des périodes fine et grossière. La représentation fine de la perturbation par bloqueur d'ordre zéro capte une impulsion brève, tandis que la représentation grossière par l'extrémité gauche la manque et prédit le mauvais point d'arrivée.

Une période d'échantillonnage et un intégrateur sur l'intervalle définissent
ensemble la transition discrète. L'action est maintenue à
$0.6\ \mathrm{m/s^2}$, et la vraie perturbation n'agit que de $0.4$ à $0.6$
seconde. Avec $\Delta t=0.1$ seconde, ses deux intervalles échantillonnés
reproduisent la contribution de $0.182\ \mathrm{m/s}$ au point d'arrivée. Avec
$\Delta t=1$ seconde, l'échantillon de perturbation à l'extrémité gauche est
nul, si bien que le modèle grossier maintenu prédit $0.379\ \mathrm{m/s}$ au
lieu de $0.562\ \mathrm{m/s}$. L'équation différentielle et l'intégrateur sur
l'intervalle sont inchangés; la période d'échantillonnage modifie à la fois
$F_{\Delta t}$ et ce que la représentation échantillonnée de la perturbation
peut exprimer.
:::

Un échantillonnage grossier des sorties n'efface pas à lui seul un événement
connu. Un intégrateur qui reçoit le véritable instant de l'événement et
$\xi(t)$ peut encore résoudre l'impulsion par des sous-pas internes. L'erreur
de la {numref}`fig-sampling-and-integration` survient parce que le modèle
grossier remplace $\xi(t)$ par des échantillons pris à l'extrémité gauche et
maintenus par un bloqueur d'ordre zéro. Le maintien d'ordre zéro signifie que
chaque valeur échantillonnée reste constante jusqu'à l'échantillon suivant.

Les applications non linéaires générales ci-dessus peuvent être difficiles à
analyser ou à optimiser. Les modèles d'état linéaires offrent un cas
particulier, local ou exact, qui se prête au calcul :

$$
\dot x=Ax+Bu+E\xi,
\qquad
y=Cx+Du.
$$

La matrice $A$ décrit l'évolution autonome de l'état, $B$ décrit comment
l'action modifie cette évolution, et $E$ décrit comment entrent les
perturbations. La matrice $C$ sélectionne ou combine des composantes de l'état
pour former les mesures, tandis que $D$ rend compte de tout effet immédiat de
l'action sur la mesure. Ainsi, $B$ et $C$ répondent à des questions
différentes. Un actionneur peut influencer toutes les directions de l'état sans
que les capteurs mesurent toutes ses composantes. Inversement, mesurer l'état
complet n'implique pas que les actionneurs disponibles puissent le déplacer
dans toutes les directions.

La linéarité est un choix de modélisation, et non l'affirmation que le monde
est linéaire à toutes les échelles. Un système non linéaire peut être bien
approché par un modèle linéaire au voisinage d'un équilibre. La programmation
dynamique elle-même ne se limite ni aux modèles linéaires ni à la
stabilisation : sa récurrence de Bellman s'applique aussi aux transitions non
linéaires et stochastiques. Dans l'exemple du pendule inversé présenté plus
loin, une approximation linéaire locale fournit un correcteur stabilisant,
tandis que [l'optimisation de trajectoires](discrete-time-optimal-control.md)
utilise les équations non linéaires pour amener le pendule dans la région
locale de ce correcteur.

La dynamique répond à la question contrefactuelle « que se passe-t-il après
cette action? ». Un problème de décision doit en outre préciser quelles
conséquences sont préférées, lesquelles sont interdites et quelle information
peut servir au moment de choisir chaque action.

## Problèmes de décision et classes de politiques

Un modèle d'état prédit l'évolution sous une entrée, mais quels objectif,
contraintes et structure d'information transforment cette prédiction en
problème de décision?

En apprentissage automatique, un modèle désigne souvent un prédicteur
paramétré. Un **modèle de décision** contient davantage : un état, des actions
disponibles, une dynamique, des observations, un objectif, des contraintes et
une structure d'information. La prédiction n'en est qu'une composante, mais
les décisions exigent des trajectoires contrefactuelles sous des actions qui
n'ont peut-être pas encore été observées.

Pour un horizon fini de $T$ actions, notons $P_t(\cdot\mid x,u)$ la loi de
l'état suivant après l'application de l'action $u$ dans l'état $x$.
L'ensemble $\mathcal U_t(x)$ contient les actions réalisables, et
$\mathcal X_t$ contient les états admissibles. Un coût d'étape $\ell_t(x,u)$
évalue une décision, tandis que $\ell_T(x)$ évalue l'état terminal. Une
politique de rétroaction $\pi_t$ associe une action à l'information
disponible $I_t$. Ces objets définissent le problème

$$
\begin{aligned}
\underset{\pi_0,\ldots,\pi_{T-1}}{\operatorname{minimiser}}\quad
& \mathbb{E}^{\pi}\!\left[\sum_{t=0}^{T-1}
  \ell_t(x_t,u_t)+\ell_T(x_T)\right] \\
\text{sous les contraintes}\quad
& x_{t+1}\sim P_t(\cdot\mid x_t,u_t), \\
& u_t=\pi_t(I_t)\in\mathcal U_t(x_t), \\
& x_t\in\mathcal X_t \quad \text{presque sûrement}.
\end{aligned}
$$

L'espérance fait la moyenne du coût cumulé sur les trajectoires engendrées par
la politique et la dynamique stochastique. La contrainte d'état est respectée
**presque sûrement**, c'est-à-dire avec probabilité un sous cette loi des
trajectoires. La remplacer par une contrainte sur la valeur moyenne, ou par une
**contrainte en probabilité** qui tolère une probabilité de violation donnée,
définirait un autre ensemble réalisable. La formulation précise donc ce qui
est optimisé, comment l'incertitude se propage, quelle information une
politique peut utiliser et quelles trajectoires sont admissibles. Omettre l'un
de ces éléments change le problème, même si l'on utilise ensuite la même classe
d'environnements et le même réseau de neurones.

Un **plan en boucle ouverte** choisit toute la suite d'actions à partir de
l'information disponible au départ :

$$
(u_0,\ldots,u_{T-1})=\mu(I_0).
$$

La suite reste fixe une fois le plan établi. Une **politique de rétroaction**
choisit plutôt chaque action à partir de l'information disponible à cet
instant :

$$
u_t=\pi_t(I_t).
$$

:::{figure} _static/open-loop-vs-feedback.svg
:label: fig-open-loop-vs-feedback
:alt: Deux correcteurs subissent la même perturbation imprévue. Le correcteur en boucle ouverte conserve ses actions précalculées, tandis que le correcteur par rétroaction modifie ses actions suivantes après avoir observé l'état déplacé.

Un plan en boucle ouverte s'engage sur les actions futures avant le début de la
trajectoire. Une politique de rétroaction réévalue sa règle de décision après
chaque nouvelle observation. Cette règle peut être un optimiseur, une politique
apprise ou une heuristique écrite à la main; le mot *rétroaction* décrit
l'information qui parvient à l'action, et non la manière dont la règle a été
obtenue. Dans cette illustration, $x_{k+1}=0.8x_k+u_k+\xi_k$ : le programme en
boucle ouverte ne contient que des zéros, tandis que la règle de rétroaction
$u_k=-0.6x_k$ réagit une fois la perturbation visible en $x_2$. L'exemple
montre la circulation de l'information; il ne garantit pas que toute règle de
rétroaction donne de bons résultats.
:::

Une trajectoire planifiée peut donner de bons résultats sous la condition
initiale et la prévision de perturbation supposées, tout en réagissant mal à
une petite poussée, à un actionneur en retard ou à une arrivée imprévue. Les
chapitres suivants construisent des plans en boucle ouverte, les transforment
en rétroaction par replanification et calculent des fonctions de valeur
conditionnelles à l'état.

La classe de politiques et le modèle sont des choix distincts. Plusieurs axes
indépendants décrivent le modèle lui-même.

| axe | possibilités courantes | conséquence |
|---|---|---|
| temps | continu ou discret | équations différentielles ou pas de transition |
| évolution | déterministe ou stochastique | un seul état suivant ou une loi sur les états suivants |
| état | continu, discret ou mixte | géométrie de l'espace d'états et méthodes de résolution applicables |
| action | continue, discrète ou mixte | capacité de commande et méthode d'optimisation |
| observation | état complet ou mesure partielle et bruitée | rétroaction d'état ou estimation à partir d'historiques d'information |
| horizon | fini, infini ou avec arrêt | conditions terminales et définition de l'objectif |

Des expressions comme « commande continue » ne précisent qu'une ligne de ce
tableau. Elles ne déterminent ni la représentation du temps, ni l'incertitude,
ni le modèle d'observation, ni l'information dont dispose le correcteur.

L'exemple de la balançoire combine maintenant ces choix de modélisation. Son
action est continue, son procédé est simulé en temps continu, sa règle de
rétroaction utilise des observations à des instants discrets, et sa contrainte
de suspension change les équations lorsque la chaîne se détend. La comparaison
isole ce qui tourne mal lorsque le canal d'action ou la contrainte est mal
modélisé.

(internal-actuation-in-swingrl)=
## Actionnement interne et contraintes unilatérales dans SwingRL

Comment l'audit d'un modèle concret peut-il révéler qu'un correcteur
apparemment efficace s'appuie sur un actionneur ou une force de contact que le
système réel ne peut pas produire?

Une personne qui veut se balancer plus haut ne peut pas commander un couple au
pivot de suspension. Les interventions disponibles sont des changements de
forme du corps, et la chaîne peut transmettre une tension mais pas une
compression. Un tour complet met ces deux hypothèses à rude épreuve : le
correcteur doit injecter assez d'énergie par des mouvements internes tout en
restant réalisable pour une suspension unilatérale.

L'environnement [`swing-rl`](https://github.com/pierrelux/swing-rl) transforme
ces exigences en problème de commande. Son modèle articulé debout offre deux
actions normalisées dans $[-1,1]^2$ : une consigne d'accroupissement et une
consigne d'inclinaison du torse. L'action modifie la géométrie du corps au lieu
d'appliquer un couple caché au pivot de suspension.

:::{figure} _static/swing-reduced-coordinates.svg
:label: fig-swing-reduced-coordinates
:alt: Géométrie du modèle réduit de la balançoire, montrant l'angle de suspension thêta, la distance rho entre le pivot et le centre de masse, le décalage alpha du centre de masse, l'angle absolu psi du centre de masse et l'angle du corps bêta.

Les coordonnées réduites décrivent la personne de façon globale plutôt que
d'en suivre chaque articulation. L'angle de suspension est $\theta$. Le centre
de masse se trouve à une distance $\rho$ du pivot et est décalé de $\alpha$
par rapport à la suspension, si bien que son angle absolu est
$\psi=\theta+\alpha$. L'angle $\beta$ décrit l'orientation du corps par
rapport à la suspension. S'accroupir modifie surtout $\rho$; s'incliner
modifie $\alpha$ et $\beta$.
:::

Le modèle réduit rend le canal d'action interne visible dans une seule
équation. Soit $m$ la masse de la personne, $g$ l'accélération
gravitationnelle, $\theta$ l'angle de suspension, $\rho$ la distance entre le
pivot et le centre de masse, $\alpha$ le décalage de ce dernier par rapport à
la ligne de suspension, $\psi=\theta+\alpha$, $\beta$ l'orientation du corps
par rapport à la suspension, et $J$ l'inertie de la personne autour de son
centre de masse. En notant $\tau_{\mathrm{amort}}$ le couple d'amortissement
modélisé, le bilan selon la coordonnée de suspension est

$$
(m\rho^2+J)\ddot\theta
=-m\rho^2\ddot\alpha
-2m\rho\dot\rho\dot\psi
-J\ddot\beta
-\dot J(\dot\theta+\dot\beta)
-mg\rho\sin\psi
+\tau_{\mathrm{amort}}.
$$

Aucun couple commandé au pivot ne figure du côté droit. L'inclinaison du torse
contribue plutôt par les termes d'accélération en $\ddot\alpha$ et
$\ddot\beta$, tandis que l'accroupissement modifie $\rho$ et contribue par
$-2m\rho\dot\rho\dot\psi$. En multipliant cette force généralisée par la
vitesse angulaire $\dot\theta$, on obtient sa puissance mécanique. Près d'un
passage au point bas, $\dot\psi\approx\dot\theta$, si bien que la contribution
de l'accroupissement vaut approximativement

$$
\left(-2m\rho\dot\rho\dot\psi\right)\dot\theta
\approx -2m\rho\dot\rho\dot\theta^2.
$$

Se raccourcir signifie $\dot\rho<0$. Puisque $m>0$, $\rho>0$ et
$\dot\theta^2\geq0$, la contribution est positive dans les deux sens de
déplacement. Le modèle explique donc comment une personne peut ajouter de
l'énergie sans actionneur fictif au pivot.

Le canal d'action n'est que la moitié de l'audit; la suspension doit aussi
respecter la bonne contrainte de force. La comparaison appariée applique une
seule **loi de rétroaction de phase**, une règle qui cadence les consignes
d'accroupissement et d'inclinaison selon la phase observée de l'oscillation,
au corps debout sous deux modèles de suspension fournis par SwingRL. Une tige
rigide peut transmettre une tension ou une compression. Une chaîne peut tirer
mais pas pousser. Tout le reste est fixé : les paramètres du corps, l'état
initial, les limites d'accroupissement et d'inclinaison, la règle de
rétroaction, l'horizon de 32 secondes et l'intervalle d'échantillonnage de
0.02 seconde. Pendant les 0.25 premières secondes, les deux simulations
multiplient leurs consignes d'accroupissement et d'inclinaison par

$$
r(t)=\min(t/0.25,1)
$$

de sorte que les deux commandes croissent linéairement de zéro jusqu'à leur
pleine valeur au lieu de sauter dès le premier pas. Cette rampe de démarrage
commune empêche qu'une discontinuité initiale de la commande soit comptée
comme un relâchement de la chaîne.

Le succès exige que l'angle de suspension déroulé varie d'au moins $2\pi$,
soit un tour complet. Au moment de ce franchissement, la distance entre le
pivot et le siège doit aussi être d'au moins $0.9L$, où $L$ est la longueur de
la suspension entièrement tendue. Ce second test empêche de compter une
rotation pendant laquelle le siège reste ramassé près du pivot.

Les deux simulations utilisent la même fonction qui associe à l'observation
courante des consignes d'accroupissement et d'inclinaison. Une fois la chaîne
relâchée, toutefois, la tige et la chaîne suivent des trajectoires différentes
et produisent des observations différentes. Appliquer la même fonction de
rétroaction à ces observations différentes peut produire des commandes
ultérieures différentes. La comparaison fixe donc la règle de rétroaction, et
non toute la suite de commandes en boucle ouverte.

La comparaison enregistrée ci-dessous applique cette règle de rétroaction aux
deux modèles de suspension. Le procédé de gauche permet une force axiale dans
les deux sens, tandis que celui de droite impose une chaîne qui ne peut que
tirer. Chaque graphique n'affiche la trajectoire que jusqu'à l'instant de
lecture choisi. L'angle déroulé continue de s'accumuler au-delà de $\pm\pi$, de
sorte qu'un tour complet reste visible. La courbe de compression demandée
montre la force de poussée qu'exigerait la trajectoire de la tige, force que la
chaîne ne peut pas fournir. La courbe d'extension montre la distance entre le
pivot et le siège rapportée à $L$. Les boutons d'événements mènent à des
instants déjà repérés dans la trajectoire Python : le relâchement de la chaîne,
l'impact au moment où elle se retend (le claquement) et le premier tour complet
de la simulation avec tige.

```{code-cell} python
:tags: [remove-input]

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from swing_control import model_audit_player_html

display(HTML(model_audit_player_html(
    Path("_static/swing_modeling"),
    Path("artifacts/swing_modeling/events.json"),
    fallback_id="fig-swing-model-audit-fallback",
)))
```

:::{figure} _static/swing_modeling/model_audit.svg
:label: fig-swing-model-audit-fallback
:class: pdf-fallback
:alt: Images clés appariées de SwingRL avec tige et avec chaîne, accompagnées des courbes complètes de l'angle, de la compression demandée à la tige et de l'extension de la chaîne.

Images clés appariées et diagnostics complets tirés des mêmes trajectoires
Python. Le livre en ligne ajoute l'animation enregistrée au format 16:9 et la
navigation par événements.
:::

```{include} artifacts/swing_modeling/results.md
```

La tige accomplit une rotation, mais une partie de ce mouvement exige que la
suspension pousse le siège loin du pivot. Une vraie chaîne ne peut pas fournir
cette poussée. Elle se détend lorsque la force de traction deviendrait
autrement négative, et le siège et la personne se déplacent alors librement
sous l'effet de la gravité jusqu'à ce que la chaîne soit de nouveau
entièrement tendue. À cet instant, la chaîne se retend d'un coup sec, la
vitesse change brusquement et une partie de l'énergie mécanique est perdue. Dans
le langage de la modélisation, il s'agit d'un **système hybride** : il passe
des équations du régime tendu à celles du régime détendu et comporte une
réinitialisation au moment où la chaîne se retend. Modifier un coefficient de
l'équation de la tige toujours tendue ne peut pas créer le mode de vol libre
manquant ni le claquement.

Ce correcteur accomplit la tâche sur le modèle à tige bidirectionnelle et
échoue sur le procédé à chaîne unilatérale. La comparaison ne classe pas la
commande par rapport à l'apprentissage par renforcement, et elle n'établit pas
qu'il est impossible de faire tourner une balançoire à chaînes. Elle isole une
hypothèse physique : un succès sur le modèle à tige ne prouve en rien que la
même trajectoire reste réalisable lorsque la suspension ne peut pas pousser.

Une approximation apprise, ajustée uniquement sur des trajectoires de la tige,
hériterait du même angle mort. Les données restent utiles pour estimer les
paramètres de la personne, l'amortissement et les pertes au contact une fois
la structure de modes unilatérale représentée. Le chapitre
[Gradients de politique](policy-gradients.md) évalue plus loin PPO et le
correcteur structuré avec la même sémantique d'action et le même critère de
succès de SwingRL.

L'audit de la balançoire a modifié une contrainte structurelle en gardant la
règle de décision fixe. La question de modélisation suivante concerne des
résultats qui varient même lorsque l'état et l'action sont fixés, comme les
arrivées de demandes ou les perturbations de l'environnement.

:::{dropdown} Examiner le scénario SwingRL commun et l'audit du modèle
```{literalinclude} code/swing_control.py
:language: python
:start-at: class SwingScenario
:end-before: class SwingModelAuditAnimation
:linenos:
```

{download}`Télécharger l'exemple complet de commande SwingRL <code/swing_control.py>`

{download}`Télécharger l'audit enregistré du modèle <_static/swing_modeling/model_audit.mp4>`
:::

## Synthèse et perspectives

Un modèle de système commandé fixe la frontière du système, les variables qui
portent l'état, le canal physique d'action et la classe de règles de décision
autorisées à utiliser cet état. L'audit de SwingRL montre pourquoi ces choix
précèdent l'optimisation : un correcteur ne peut pas réparer un canal d'action
ou un mode de contact que le modèle a omis.

Les équations d'état déterministes associent encore un seul successeur à
chaque état et à chaque action. Qu'est-ce qui change lorsque des perturbations
rendent ce successeur aléatoire, ou lorsque le correcteur n'observe qu'une
mesure ambiguë? Le chapitre [Dynamique stochastique et observation
partielle](stochastic-dynamics-observation.md) répond à cette question.

## Exercices

:::{exercise} Un modèle de décision complet
:label: ex-dynamics-model-checklist

Choisissez un système séquentiel familier et précisez sa frontière, son état,
son observation, son action, sa perturbation, son objectif, sa structure
d'information et une contrainte stricte. Chaque élément doit se rapporter au
même décideur et à la même échelle de temps.
:::

:::{solution} ex-dynamics-model-checklist
:class: dropdown

Plusieurs réponses sont possibles. Pour un thermostat de pièce, un modèle
cohérent est le suivant : la frontière contient la pièce, les murs et
l'appareil de chauffage; l'état contient les températures de l'air intérieur
et des murs; l'observation est une lecture bruitée de la température de l'air;
l'action est la puissance de chauffage; la température extérieure et
l'occupation sont des perturbations; l'objectif arbitre entre l'écart de
température et l'énergie; l'information au temps $t$ est l'historique des
observations et des actions; et $0\leq u_t\leq P_{\max}$ est une contrainte
stricte. Une autre réponse est valable si les huit objets utilisent une même
frontière et une même période d'échantillonnage.
:::

:::{exercise} Les canaux d'action sont physiques
:label: ex-dynamics-action-channels

Dans la {numref}`fig-modeling-action-channels`, expliquez pourquoi remplacer
chaque entrée par une force généralisée sans contrainte change les trois
problèmes de décision.
:::

:::{solution} ex-dynamics-action-channels
:class: dropdown

Ce remplacement donne à la balançoire un actionneur fictif au pivot au lieu
d'une commande par la forme du corps, et peut contourner l'incapacité de la
chaîne à pousser. Pour le pont roulant, il remplace l'accélération bornée du
chariot par un forçage direct de la charge. Il permet aussi au houlomoteur
d'injecter une force signée au lieu de choisir un amortissement de prise de
force non négatif. Chaque substitution change les trajectoires réalisables, et
pas seulement les symboles qui servent à les décrire.
:::

:::{exercise} L'échantillonnage change la transition
:label: ex-dynamics-euler-sampling

Pour $\dot x=-ax+bu$ avec une entrée maintenue par un bloqueur d'ordre zéro,
dérivez les mises à jour d'Euler explicite pour des pas $\Delta t$ et
$\Delta t/2$. Quels coefficients discrets changent avec la période
d'échantillonnage?
:::

:::{solution} ex-dynamics-euler-sampling
:class: dropdown

Avec un pas $\Delta t$,

$$
x_{k+1}=(1-a\Delta t)x_k+b\Delta t\,u_k.
$$

Avec un pas $\Delta t/2$,

$$
x_{k+1}=\left(1-\frac{a\Delta t}{2}\right)x_k
{}+\frac{b\Delta t}{2}u_k.
$$

Le coefficient de l'état et celui de l'entrée changent tous les deux. Deux
demi-pas composent aussi une approximation à pas fini différente d'un seul pas
d'Euler complet, même si les deux convergent vers le même flot continu lorsque
$\Delta t\to0$.
:::

:::{exercise} Quand se raccourcir ajoute de l'énergie
:label: ex-dynamics-swing-power

Multipliez $-2m\rho\dot\rho\dot\psi$ par $\dot\theta$ et utilisez
$\dot\psi\approx\dot\theta$ près du point bas de la balançoire. Pourquoi
raccourcir le rayon de la personne peut-il ajouter de l'énergie dans les deux
sens de déplacement?
:::

:::{solution} ex-dynamics-swing-power
:class: dropdown

La puissance approximative apportée à la coordonnée de suspension est

$$
\left(-2m\rho\dot\rho\dot\psi\right)\dot\theta
\approx-2m\rho\dot\rho\dot\theta^2.
$$

Comme $m,\rho>0$ et que se raccourcir donne $\dot\rho<0$, ce terme est
positif. Le carré élimine le signe de $\dot\theta$, si bien que la conclusion
vaut pour les passages dans le sens horaire comme dans le sens antihoraire.
:::

:::{exercise} Un modèle hybride tendu-détendu
:label: ex-dynamics-hybrid-swing

Décrivez les phases tendue et détendue de la balançoire comme un modèle
hybride. Précisez l'état continu, le mode discret, la condition de
relâchement, la condition de remise en tension et la réinitialisation à
l'impact; une équation complète du mouvement n'est pas demandée.
:::

:::{solution} ex-dynamics-hybrid-swing
:class: dropdown

On utilise un état mécanique continu qui contient les positions, les vitesses
et la configuration de la personne, ainsi qu'un mode
$m\in\{\text{tendu},\text{détendu}\}$. Le mode tendu impose la contrainte de
longueur de la chaîne et exige une tension non négative. Le relâchement se
produit lorsque la tension nécessaire au maintien de cette contrainte atteint
zéro et deviendrait négative. Le mode détendu suit une dynamique balistique
sans contrainte. La remise en tension se produit lorsque la distance entre le
pivot et le siège atteint la longueur de la chaîne avec une vitesse radiale
dirigée vers l'extérieur. La réinitialisation à l'impact supprime la vitesse
radiale inadmissible et dissipe l'énergie qui lui est associée.
:::

:::{exercise} Ce que des données de tige ne peuvent pas identifier
:label: ex-dynamics-rod-data

Le correcteur structuré accomplit un tour sur le modèle à tige et échoue sur le
modèle à chaîne. Pourquoi un modèle de transition neuronal plus grand, ajusté
uniquement sur des trajectoires de la tige, peut-il échouer à résoudre cet
écart?
:::

:::{solution} ex-dynamics-rod-data
:class: dropdown

Les trajectoires de la tige ne contiennent ni relâchement, ni vol libre, ni
remise en tension, ni impact. Une plus grande capacité d'approximation ne peut
pas identifier un mode absent des données. Des interventions avec la chaîne ou
une structure hybride unilatérale explicite doivent d'abord fournir ce
comportement manquant; les données peuvent ensuite estimer des paramètres à
l'intérieur de cette structure.
:::

:::{exercise} Un audit de SwingRL à un seul facteur
:label: ex-dynamics-swing-intervention

Prolongez la comparaison SwingRL avec le correcteur assis, ou désactivez soit
l'accroupissement, soit l'inclinaison dans le correcteur debout. Avant de
lancer la simulation, prédisez un diagnostic qui devrait changer. Expliquez
ensuite quelle hypothèse de modélisation l'intervention met à l'épreuve.
:::

:::{solution} ex-dynamics-swing-intervention
:class: dropdown

Un audit valable ne change qu'un facteur, consigne d'abord une prédiction et
compare le résultat à la référence. Le correcteur assis change la morphologie
de la personne et l'actionnement disponible. Désactiver l'accroupissement
supprime le canal de puissance $-2m\rho\dot\rho\dot\theta^2$; désactiver
l'inclinaison supprime les termes angulaires entraînés. L'angle déroulé
maximal, la réussite du tour, le travail du canal d'action et l'instant du
relâchement ou de la remise en tension sont des diagnostics appropriés. Un
résultat n'appuie que l'hypothèse de modélisation modifiée, et non un
classement général des correcteurs.
:::
