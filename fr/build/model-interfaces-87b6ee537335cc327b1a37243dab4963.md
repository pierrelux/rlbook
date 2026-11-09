---
description: Formuler des problèmes de décision séquentielle en précisant leur état, leur canal d'action, leur incertitude, leurs observations, leurs contraintes et l'interface de modèle disponible.
kernelspec:
  name: python3
  display_name: Python 3
---
# Interfaces de modèle et composantes apprises

La dynamique stochastique et les modèles d'observation précisent la loi de
l'état suivant et l'information renvoyée au correcteur. Lesquels de ces objets
mathématiques un algorithme peut-il réellement évaluer, dériver,
rééchantillonner ou apprendre à partir de données enregistrées?

Deux modèles peuvent produire la même trajectoire nominale tout en offrant des
opérations différentes à un algorithme. L'accès aux équations, aux dérivées,
aux réinitialisations et à de nouvelles interactions détermine ce qu'on peut
calculer. L'interface compte donc indépendamment de la fidélité physique du
modèle : savoir qu'une transition existe n'est pas la même chose que pouvoir
la dériver, la rééchantillonner ou l'interroger sous une nouvelle action.

| interface disponible | opérations qu'elle permet directement | méthodes développées plus loin |
|---|---|---|
| équations et dérivées | évaluer la dynamique locale, linéariser, dériver les contraintes | transcription directe, commande linéaire quadratique, commande prédictive par gradient |
| fonction de transition à un pas | réinitialiser et avancer à partir de couples état-action choisis | méthodes de tir, commande prédictive par simulation, programmation dynamique fondée sur un modèle |
| simulateur génératif | échantillonner des trajectoires sans examiner les équations internes | estimation de Monte-Carlo, recherche sans dérivées, apprentissage de politiques |
| transitions historiques | ajuster ou évaluer des modèles sur la loi enregistrée | identification de système, méthodes de valeur et Q ajustées, évaluation hors ligne |
| environnement interactif et objectif | recueillir de nouvelles transitions sous des actions choisies | apprentissage par renforcement en ligne |

Les lignes ne s'excluent pas mutuellement. Un simulateur peut être
différentiable, un modèle explicite peut contenir des paramètres inconnus, et
des données historiques peuvent servir à ajuster un nouveau modèle à un pas. Le
tableau consigne l'information fournie à l'algorithme, et non l'origine ou la
fidélité du modèle.

Un programme est un modèle lorsque son exécution définit des transitions
d'état. MuJoCo, par exemple, combine les équations des corps rigides, la
détection des contacts, les forces de contrainte, les capteurs et le rendu
derrière une interface de réinitialisation et de pas. On peut interroger cette
interface sans disposer d'une expression utilisable de sa fonction de
transition locale complète. Un simulateur à événements discrets définit lui
aussi un modèle de transition, même si son horloge avance d'un événement
asynchrone au suivant.

Qualifier une méthode de **sans modèle** précise quelle information de
transition ses mises à jour peuvent utiliser. Dans ce livre, une mise à jour
sans modèle reçoit de l'expérience échantillonnée plutôt qu'un accès direct à
la fonction de transition, aux probabilités de transition ou à leurs dérivées.
Une méthode en ligne peut choisir une action et observer la transition qui en
résulte. Une méthode hors ligne est limitée aux transitions de son journal.
Aucune des deux ne peut demander directement la loi exacte des issues d'une
action non essayée dans le même état. Toutes deux exigent néanmoins une
observation, un ensemble d'actions, une récompense, un processus
d'échantillonnage et des hypothèses déclarés qui relient les échantillons
disponibles au déploiement.

(inference-serving-as-a-controlled-system)=
### Le service d'inférence comme système commandé

Le service d'inférence réunit plusieurs choix de modélisation à l'intérieur
d'une même frontière. Il combine des bilans conservatifs de travail et de cache
avec des cartes de performance mesurées, possède à la fois des états au niveau
des requêtes et des états agrégés, et peut être abordé par des équations, par
simulation, par des traces historiques ou par un processus en direct.
L'objectif de commande est d'ordonnancer le travail et de choisir les réglages
matériels tout en respectant les limites de latence, de mémoire, de puissance
et de température.

Un serveur d'inférence reçoit des requêtes à des instants irréguliers et
produit une réponse pour chacune. Un modèle de langue à décodeur seul traite
une requête en deux phases. La phase de **préremplissage** traite l'invite en
parallèle et produit le premier jeton de sortie. La phase de **décodage**
produit les jetons restants une itération à la fois. Les longs
préremplissages peuvent retarder les courtes itérations de décodage des
requêtes déjà en cours. L'ordonnancement au niveau des itérations et le
préremplissage par blocs sont des approches de production pour maîtriser cette
interaction {cite}`yu2022orca,agrawal2024sarathi`. Le simulateur pédagogique
ci-dessous utilise une règle d'entrelacement réduite plutôt que de reproduire
l'un ou l'autre de ces systèmes.

#### Frontière, état et action

La frontière du système entoure un GPU et son processus de service. Le modèle
de langue et le moteur d'inférence se trouvent à l'intérieur de la frontière.
Les arrivées de requêtes et les conditions thermiques ambiantes viennent de
l'extérieur.

:::{figure} _static/inference-serving-boundary.svg
:label: fig-inference-serving-boundary
:alt: Diagramme de frontière du système contenant un processus de service et un GPU. Les requêtes et les conditions ambiantes viennent de l'extérieur; les réponses et les observations en sortent; les actions d'horloge et d'ordonnancement viennent d'un correcteur.

La frontière contient la file de requêtes, l'ordonnanceur, l'exécution du
modèle de langue, le cache clé-valeur et l'état matériel du GPU. Le correcteur
observe l'état de la file et du cache, le travail terminé, la puissance, la
température et la fréquence d'horloge réalisée; il choisit une fréquence
demandée et une règle d'ordonnancement. Les arrivées sont des perturbations, et
la longueur finale de la sortie d'une requête reste cachée jusqu'à ce que
cette requête soit terminée. Les autres GPU, le routage réseau et les
applications en aval restent hors de ce modèle.
:::

Cette frontière de service fixe est accessible par quatre des interfaces
ci-dessus. Les bilans de jetons, de cache et de chaleur fournissent des
équations. Le simulateur à événements fondé sur les profils peut être
réinitialisé et exécuté sous des actions d'horloge et d'ordonnancement
choisies. Les traces historiques et les relevés de profils ne contiennent que
les états et les actions enregistrés; ils ne peuvent donc pas répondre
directement à des questions contrefactuelles hors de cette couverture. Un
processus Qwen/vLLM en direct accepte de nouvelles requêtes et des réglages
d'horloge choisis, et produit de nouveaux échantillons de transitions.
Plusieurs interfaces peuvent coexister; leurs opérations disponibles diffèrent
même si le processus de service est le même.

L'état complet du simulateur au niveau des requêtes consigne, pour chaque
requête, sa phase, son âge, le travail d'invite restant, les jetons produits,
la longueur finale de la sortie et l'allocation de cache. La longueur finale de
la sortie fait partie de l'état caché du simulateur, mais elle n'est pas
révélée au correcteur. Avec un intervalle de commande d'une seconde, le modèle
de commande réduit utilise l'état agrégé

$$
x_t=(p_t,d_t,m_t,T_t,f_t),
$$

où $p_t$ est le travail de préremplissage en file et $d_t$ le travail de
décodage inachevé. La grandeur $m_t$ est l'occupation du cache clé-valeur; ce
cache stocke des représentations intermédiaires de l'attention pour éviter de
recalculer, à chaque pas de décodage, les jetons déjà traités. Les autres
variables sont la température du GPU $T_t$ et la fréquence d'horloge graphique
réalisée $f_t$, la fréquence matérielle qui détermine la vitesse d'exécution
du travail sur le GPU. Ce vecteur est un état du modèle réduit, mais il n'est
pas une description markovienne de toutes les trajectoires au niveau des
requêtes. La commande

$$
u_t=(f_t^{\mathrm{dem}},\sigma_t)
$$

combine une fréquence d'horloge demandée et une règle d'ordonnancement
$\sigma_t$. L'ordonnanceur décide quelle phase reçoit chaque pas de service et
quelle quantité de travail de préremplissage peut être traitée avant de revenir
aux décodages actifs. La fréquence demandée modifie le rythme et le coût
énergétique de ce travail. Le matériel peut réaliser une fréquence plus basse
sous une limite de puissance ou de température; $f_t^{\mathrm{dem}}$ et $f_t$
sont donc des variables distinctes {cite}`nvidiaSmi2026`.

Les arrivées de requêtes forment la perturbation. Une arrivée fournit son
instant et la longueur de son invite, mais la longueur finale de sa sortie
reste inconnue du correcteur jusqu'à l'arrivée du jeton de fin de séquence.
L'observation contient l'âge des requêtes en file, les jetons d'invite et de
sortie terminés, l'utilisation du cache, la puissance, la température et la
fréquence réalisée. Cette structure d'information empêche un correcteur
d'ordonnancer à l'aide de la longueur future consignée dans une trace
d'évaluation.

#### Conservation et cartes étalonnées

Les bilans agrégés exposent une structure qu'il n'est pas nécessaire
d'apprendre. Si $A_t^p$ jetons d'invite arrivent, que $C_t^p$ jetons d'invite
sont traités, que $N_t^p$ requêtes terminent leur préremplissage et que
$C_t^d$ jetons de décodage sont produits, alors

$$
\begin{aligned}
p_{t+1} &= p_t+A_t^p-C_t^p,\\
d_{t+1} &= d_t+W_t^d(N_t^p)-C_t^d.
\end{aligned}
$$

Le terme $W_t^d(N_t^p)$ désigne le travail de sortie, encore inconnu, associé
aux requêtes nouvellement admises au décodage. Le simulateur au niveau des
requêtes ne rend ce travail visible qu'à mesure que les jetons sont produits.
Il suit l'occupation continue des jetons d'invite et des jetons produits, y
compris les préremplissages partiels, ainsi qu'une réserve fixe par requête.
Cette occupation est libérée lorsqu'une requête se termine. Cette comptabilité
agrégée préserve le bilan de cache modélisé, mais elle ne reproduit pas
l'allocateur de blocs d'un moteur de service. Un bilan thermique agrégé prend
la forme

$$
T_{t+1}=T_t+\frac{\Delta t}{C_\theta}
\left(P(x_t,u_t)-\frac{T_t-T_{\mathrm{amb}}}{R_\theta}\right).
$$

Ici, $T_{\mathrm{amb}}$ est la température ambiante, $C_\theta$ la capacité
thermique effective et $R_\theta$ la résistance thermique vers
l'environnement. La carte de puissance $P(x_t,u_t)$ ajoute de la chaleur,
tandis que $(T_t-T_{\mathrm{amb}})/R_\theta$ en retire lorsque le GPU est plus
chaud que son environnement.

La conservation des jetons et du cache détermine la forme de la transition. Des
mesures restent nécessaires pour la carte des débits de service, la carte de
puissance $P$ et les paramètres thermiques. Le protocole de profilage vise
Qwen2.5-7B-Instruct servi par vLLM sur un NVIDIA L4
{cite}`qwen25report,kwon2023vllm`. Le profil matériel prévu couvre cinq
niveaux de fréquence demandée, plusieurs longueurs d'invite et trois niveaux de
concurrence, c'est-à-dire trois nombres différents de requêtes servies en même
temps. La construction du livre lit un profil versionné et ne lance jamais de
serveur de modèle. Son manifeste indique si les cartes proviennent de mesures
L4 terminées ou d'un substitut d'ingénierie antérieur aux mesures, et chaque
résultat affiché indique cette provenance.

#### La taille du lot change la carte de service

Le profil L4 enregistré à 64.8 W montre pourquoi une carte de service doit
conserver la concurrence. À une fréquence demandée de 2 040 MHz avec une
invite de 1 024 jetons, le débit agrégé de décodage passe de 17.5 jetons
produits par seconde pour une requête à 105.6 pour huit requêtes
simultanées. Le débit agrégé de préremplissage varie beaucoup moins d'une
taille de lot à l'autre. Regrouper les mesures en un seul débit par fréquence
efface cette différence et peut donner à un correcteur une mauvaise prédiction
de la vidange des files.

:::{figure} _static/inference_serving/batch-calibration.svg
:label: fig-inference-batch-calibration
:alt: Débits de préremplissage et de décodage à cinq fréquences GPU demandées, avec des courbes distinctes pour une, quatre et huit requêtes simultanées. Le débit de décodage augmente fortement avec la concurrence; le débit de préremplissage varie peu.

La taille du lot change la capacité de décodage dans le profil L4 enregistré
le 2 septembre 2026 avec un plafond de 64.8 W. Chaque point est la médiane de
cinq lots simultanés; les moustaches montrent leur minimum et leur maximum. La
longueur d'invite est fixe dans chaque panneau. Les cinq fréquences demandées
sont conservées, y compris le plateau de débit au-dessus de 1 125 MHz.
:::

Le plateau justifie aussi une référence à fréquence fixe modérée. Un
correcteur par rétroaction peut réduire l'énergie par rapport à la fréquence
maximale sans faire mieux qu'une fréquence constante bien choisie.
L'expérience en direct choisit donc une fréquence constante sur des requêtes
de développement avant de la comparer au gestionnaire de fréquence et à la
commande prédictive sur de nouvelles requêtes. L'expérience en direct actuelle
utilise un plafond de 40 W et exige de nouvelles mesures de puissance de
service et de repos à ce plafond. Les courbes à 64.8 W restent des éléments
historiques; la reprise ci-dessous conserve le modèle pédagogique réduit
d'origine.

Le correcteur en direct distingue les requêtes qui attendent leur première
sortie de celles qui sont déjà en cours de diffusion. Il estime le travail de
sortie restant à partir des longueurs des requêtes de développement et des
jetons déjà reçus, puis ajuste son modèle de transition sur des mesures de
service mixte. Ces observations conservent des différences utiles dans la
charge de travail sans révéler les longueurs finales des requêtes inachevées
ni supposer que l'état obtenu est entièrement markovien.

{download}`Télécharger la table d'étalonnage par phase, contexte et concurrence (CSV) <artifacts/inference_serving/live_control_development/batch_calibration.csv>`

Une étude mesurée distincte compare maintenant des fréquences fixes, les
fréquences graphiques gérées par le GPU et une heuristique causale fondée sur
la demande et la phase, sous le réglage configuré de 40 W. Sur quatre blocs de
requêtes Azure réservés à l'évaluation, chacun répété deux fois, la fréquence
fixe de 660 MHz a consommé 10.31 % moins d'énergie GPU, d'après la télémétrie,
que les fréquences gérées par le GPU; l'heuristique adaptative a consommé
encore 0.77 % de moins que la fréquence fixe de 660 MHz. Toutes les
vérifications appariées de latence relative ont réussi, mais le candidat a
échoué à la qualification de développement et les trois méthodes ont manqué la
cible absolue de délai avant le premier jeton sur le bloc de code le plus
court. Ce sont des observations exploratoires, et non des économies
adaptatives établies qui respectent les exigences de latence. L'heuristique
est distincte du gestionnaire de fréquence et de la commande prédictive, dont
l'étalonnage reste incomplet. L'[étude mesurée](interactive/gpu-adaptive-study.html)
comprend le criblage des cinq fréquences, les sondes sur la forme des
requêtes, les fréquences observées et les échecs conservés.

{download}`Lire l'étude complète et les résultats appariés (PDF) <artifacts/inference_serving/live_control_preparation/adaptive-study-v3/final-delivery-v1/article/article.pdf>`

#### Charge de travail et règle d'ordonnancement

Les équations de conservation précisent comment le travail circule dans le
serveur, mais une expérience a aussi besoin d'un processus d'arrivée
reproductible et d'un ordonnanceur déclaré. Les réglages suivants définissent
la simulation enregistrée. L'expérience en direct révisée mesure plutôt la
capacité en service mixte et fige ses cibles de latence à partir de requêtes de
développement avant la confirmation.

La charge simulée est un extrait de cinq minutes de la trace Azure 2023 de
génération de code, qui consigne les instants des requêtes et leurs nombres de
jetons d'entrée et de sortie {cite}`azureLLMTrace2023,patel2024splitwise`. Les
instants d'arrivée sont dilatés une seule fois pour amener l'utilisation à la
fréquence maximale près de 80 %. Cette dilatation du temps conserve la taille
et l'ordre des requêtes tout en étalant les arrivées sur un intervalle plus
long. Si $\rho_{\max}$ est le temps de service isolé de toutes les requêtes à
la fréquence profilée la plus élevée, divisé par la longueur de la fenêtre
d'origine, la dilatation et les arrivées normalisées sont

$$
\kappa=\max\!\left(1,\frac{\rho_{\max}}{0.8}\right),
\qquad
t_i'=\kappa(t_i-t_0).
$$

Le facteur $\kappa$ vaut au moins un; il peut donc laisser la trace inchangée
ou ralentir ses arrivées, mais ne les comprime jamais dans un intervalle plus
court.

Tous les correcteurs simulés reçoivent les mêmes requêtes immuables après
cette transformation. La limite simulée du délai avant le premier jeton vaut
deux fois la valeur médiane de référence pour une invite de 1 024 jetons à une
concurrence de un et à la fréquence la plus élevée. La limite du temps par
jeton de sortie vaut 1.5 fois la médiane correspondante. Ce sont des niveaux de
référence pour une comparaison contrôlée, et non des objectifs de niveau de
service revendiqués pour des systèmes en production.

| grandeur | réglage de la simulation enregistrée |
|---|---|
| modèle | Qwen2.5-7B-Instruct, révision `acbd96531cda22292a3ceaa67e984955d3965282` {cite}`qwen25modelcard` |
| moteur d'inférence | serveur OpenAI de vLLM, version 0.28.0 {cite}`vllmDocker028` |
| accélérateur visé | un NVIDIA L4 |
| profil de fréquences | cinq fréquences graphiques demandées; fréquence réalisée, puissance, utilisation et température conservées |
| trace de requêtes | cinq premières minutes pour l'évaluation; 20 premières requêtes pour l'animation |
| échantillonnage du correcteur | une seconde |
| simulation du service | pas de 0.1 seconde; priorité au décodage ou un bloc de préremplissage d'abord; la capacité de préremplissage inutilisée peut retourner au décodage |
| longueur de sortie | cachée au correcteur jusqu'à la fin de la requête |

Dans chaque période d'une seconde de commande de l'horloge, l'ordonnanceur
attribue chaque pas de simulation de 0.1 seconde au décodage ou le commence par
un bloc de préremplissage. Si ce bloc se termine avant l'épuisement du budget
de service du pas, la capacité restante retourne au décodage. Lorsque les deux
phases ont du travail, le décodage actif reçoit la priorité un pas sur deux, et
une priorité absolue lorsque le cache est très sollicité. La règle de
**préremplissage par blocs entrelacés de 512 jetons** précise le canal d'action
du modèle pédagogique; elle ne reproduit ni l'ordonnanceur à lots mixtes de
vLLM ni Sarathi-Serve.

#### Reprise enregistrée et portée

Les 20 premières requêtes forment la reprise enregistrée ci-dessous. Elles
passent par l'arrivée, des blocs de préremplissage entrelacés de 512 jetons, le
décodage autorégressif et l'achèvement. Les graphiques ne montrent que le début
de trajectoire atteint par la tête de lecture, et la longueur de sortie reste
cachée au correcteur jusqu'à l'achèvement de chaque requête. Un badge de
provenance distingue un profil mesuré vérifié d'un substitut d'ingénierie
antérieur aux mesures.

```{code-cell} python
:tags: [remove-input]

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="modeling",
)))
```

:::{figure} _static/inference_serving/modeling.svg
:label: fig-inference-serving-model-fallback
:class: pdf-fallback
:alt: Courbes du nombre de requêtes inachevées et de la fréquence GPU réalisée pour la reprise de modélisation à 20 requêtes.

Résumé statique des requêtes inachevées et de la fréquence réalisée, issu de la
même simulation au niveau des requêtes. Le livre en ligne offre la lecture,
l'avance pas à pas, le défilement, le détail des phases et le choix du
correcteur.
:::

Des vérifications de conservation rendent compte de chaque requête, de chaque
jeton traité et de chaque allocation de cache. À la fin de la reprise
versionnée, les 20 requêtes sont terminées, les files de préremplissage et de
décodage sont vides, et toute l'occupation de cache modélisée a été libérée.
Cela établit la fermeture des flux modélisés. Cela ne montre ni que les
variables agrégées sont markoviennes pour le procédé au niveau des requêtes,
ni que les cartes de débit de service, de puissance ou de température sont
exactes. Un profil provisoire permet de tester le logiciel de simulation et de
commande, mais ne fournit aucune donnée empirique de latence ou d'énergie. Dans
les deux cas, le modèle ne couvre pas tous les moteurs d'inférence ni tous les
GPU. Le transfert réseau, la segmentation en jetons côté hôte, la
communication entre plusieurs GPU, les effets sur la qualité du modèle et les
pannes hors du processus de service restent hors de la frontière.

:::{dropdown} Examiner la transition au niveau des requêtes
```{literalinclude} code/inference_serving.py
:language: python
:start-at: def simulate
:end-before: def compute_metrics
:linenos:
```

{download}`Télécharger le modèle complet du service d'inférence <code/inference_serving.py>`
:::

### Censure dans les journaux de déplacements terminés

L'exemple d'inférence peut produire de nouvelles trajectoires à partir d'un
simulateur ou d'un processus en direct. L'archive BIXI fournit une interface
plus restrictive : un journal fixe de déplacements terminés. La distinction
compte, car un journal d'événements terminés consigne ce que le système a
servi, et non toutes les demandes que les usagers ont tentées. Une location
terminée prouve qu'un vélo était disponible, mais une station vide ne laisse
aucune trace de déplacement pour l'usager qui n'a pas pu partir. De même, un
retour terminé prouve qu'un point d'ancrage était libre, sans compter les
usagers qui ont trouvé la station pleine.

:::{figure} _static/bixi/bixi-completed-trip-censoring.svg
:label: fig-bixi-completed-trip-censoring
:alt: Deux historiques de demande tentée produisent le même journal de déplacements BIXI terminés, parce que des tentatives supplémentaires surviennent pendant qu'une station est vide.

Deux mondes peuvent produire le même journal de déplacements terminés. Dans le
second, des usagers supplémentaires tentent de louer un vélo pendant que la
station est vide. Le journal ne permet pas d'identifier ces tentatives
censurées.
:::

Le flux public d'état des stations peut indiquer les stocks courants pendant
qu'il fonctionne, mais il ne fournit pas de canal d'action pour redistribuer
les vélos. Les déplacements terminés historiques omettent la demande non
satisfaite, les stocks passés des stations et les déplacements de
l'exploitant. Un modèle d'arrivées ajusté peut être utile, mais évaluer une
nouvelle politique de redistribution à partir de ces journaux exige des
hypothèses sur les tentatives manquantes et sur la manière dont l'exploitation
a produit les données observées. L'exemple BIXI exécutable garde ces
hypothèses visibles en déclarant un modèle de demande tentée et en consignant
chaque événement refusé.

:::{dropdown} Examiner le contre-exemple sur les données historiques
```{literalinclude} code/bixi_control.py
:language: python
:start-at: def make_censoring_counterexample
:linenos:
```
:::


### La génération de texte comme système séquentiel

La distinction entre un modèle de transition et une politique peut aussi
s'estomper lorsqu'un même objet logiciel semble produire une trajectoire
entière. La génération de texte autorégressive sépare les deux. Pour une
invite fixe $p$, posons

$$
s_t=(p,y_1,\ldots,y_{t-1}),
\qquad
a_t=y_t.
$$

La transition de base ajoute le jeton choisi :

$$
s_{t+1}=\operatorname{concat}(s_t,a_t).
$$

Cette transition est déterministe une fois le jeton choisi; l'incertitude
vient de la politique sur les jetons $\pi(a_t\mid s_t)$. Un modèle de langue
fournit donc une politique, tandis que la mise à jour du préfixe fournit la
transition. Un modèle de récompense ou de préférence est un objet
supplémentaire, et non une propriété inhérente à la génération de texte.

Cette frontière simplifiée change lorsque le modèle appelle des outils, reçoit
de nouveaux messages de l'utilisateur ou interagit avec une application
externe. Les sorties des outils et les réponses de l'utilisateur deviennent
alors des observations produites par un environnement extérieur à la règle de
concaténation du préfixe. La troncature du contexte exige aussi de préciser
explicitement le contexte conservé ou l'état de mémoire.

L'interface détermine quelles données et quelles questions contrefactuelles
sont disponibles. La décision de modélisation qui reste porte sur la structure
de transition à fixer et sur les composantes à estimer à partir de ces
données.

## Structure connue et composantes apprises

Lorsque les données révèlent un écart, quelles composantes du modèle faut-il
réétalonner, et quelles lois de conservation, quelle géométrie et quelles
contraintes doivent rester explicites?

L'incertitude sur une composante n'exige pas de remplacer tout le modèle. On
peut conserver la transition connue et n'apprendre que l'écart entre sa
prédiction et l'état suivant observé :

$$
x_{t+1}=f_{\mathrm{connu}}(x_t,u_t)
+r_\theta(x_t,u_t,\eta_t)+\xi_t.
$$

La correction $r_\theta$ s'appelle un **résidu appris**, parce qu'elle rend
compte de ce qui reste une fois que le modèle connu a fait sa prédiction. Le
vecteur de contexte $\eta_t$ peut contenir la météo, des informations sur la
charge utile ou d'autres conditions mesurées. Ce résidu a une tâche plus
étroite qu'une transition complète de type boîte noire.

| système | structure conservée | composante apprise plausible | expérience d'audit du modèle |
|---|---|---|---|
| balançoire | actionnement interne et tension unilatérale | paramètres de la personne, d'amortissement et de pertes au contact | mesurer le premier relâchement |
| corridor de vélos | conservation des stocks et du camion, capacités des points d'ancrage et du camion | loi de la demande tentée et temps de trajet | introduire un choc de demande déclaré |
| nacelle de caméra | dynamique du corps rigide, limite de couple et géométrie des capteurs | évolution du biais du gyroscope ou modèle de perturbation | séparer la rotation mécanique de la translation de la base |
| service d'inférence | conservation des requêtes, des jetons et du cache | résidus de débit de service, de puissance et de température | réserver à l'évaluation des longueurs d'invite, des fréquences et des niveaux de concurrence |
| cellule de batterie | bilan de charge, circuit équivalent et conservation thermique | variation de résistance | appliquer une impulsion de courant de diagnostic, puis répéter la recharge |

(fast-charging-when-resistance-drifts)=
### Recharge rapide quand la résistance dérive

Le cas de la batterie sépare une structure de modèle connue d'un paramètre qui
a changé. Le bilan de charge, la topologie du circuit et la conservation
thermique restent fixes, tandis qu'une brève impulsion de courant estime un
facteur d'échelle de la résistance qui influe sur la sécurité de la recharge.
Cette séparation permet de se demander si un étalonnage ciblé peut corriger une
erreur de prédiction précise sans réapprendre toute la transition.

Le bilan de charge détermine comment le courant modifie l'état de charge. Avec
la convention de signe du livre, où la recharge est positive,

$$
\dot z=\frac{I}{3600Q},
$$

où $z$ est l'état de charge, $I$ le courant de recharge en ampères et $Q$ la
capacité de la cellule en ampères-heures. Une cellule de 5 Ah rechargée à
10 A passerait de 20 à 80 % en

$$
\frac{0.6(5\ \mathrm{Ah})}{10\ \mathrm A}
=0.3\ \mathrm h
=18\ \mathrm{min},
$$

si c'était la seule équation pertinente. La tension aux bornes et la
température peuvent limiter le courant avant que cette cible de charge soit
atteinte.

On ne peut pas prédire la tension aux bornes à partir du seul bilan de charge,
car le courant produit aussi une hausse résistive immédiate et une réponse de
polarisation plus lente. Le procédé de référence représente ces deux réponses
par le [modèle de Thévenin à une branche RC](https://docs.pybamm.org/en/v26.6.2.0/source/api/models/equivalent_circuit/thevenin.html)
documenté dans PyBaMM {cite}`sulzer2021pybamm,barletta2022thevenin`. Il
utilise aussi deux états thermiques globaux, en traitant la cellule et son
support comme des objets de température uniforme. Son état contient
$x=(z,v_p,T_c,T_j)$ : l'état de charge, la tension de polarisation, la
température de la cellule et celle du gabarit. Le gabarit est le support qui
entoure la cellule et avec lequel elle échange de la chaleur. Le modèle
électrique utilise une tension à vide $U_{\mathrm{oc}}(z)$, soit la tension de
la cellule lorsqu'aucun courant ne circule, une résistance série $R_0$ et une
branche $R_1$--$C_1$ formée d'une résistance et d'un condensateur. Sa tension
$v_p$ varie à l'échelle de temps $R_1C_1$; elle représente donc le
comportement de polarisation plus lent. En notant $V$ la tension aux bornes,
les équations avec recharge positive sont

$$
\dot v_p=-\frac{v_p}{R_1C_1}+\frac{I}{C_1},
\qquad
V=U_{\mathrm{oc}}(z)+v_p+IR_0.
$$

PyBaMM traite un courant positif comme une décharge; l'implémentation change
donc le signe du courant de recharge positif utilisé dans ces équations.

L'étalonnage commence par une expérience contrôlée plutôt que par une recharge
complète. La cellule repose à $I=0$ pendant 20 secondes, reçoit un courant de
recharge commandé de 5 A pendant 10 secondes, puis repose encore 40 secondes.
Les échantillons sont pris toutes les 0.5 seconde. L'enclenchement du courant
produit un saut de tension immédiat à travers $R_0$, suivi de la réponse de
tension plus lente $v_p$ de la branche RC. Son interruption révèle la
décroissance de cette réponse plus lente. L'impulsion est donc une entrée
choisie délibérément pour rendre la partie résistive du modèle visible dans le
relevé de tension.

Le relevé de diagnostic versionné contient le courant commandé, l'état de
charge simulé et la tension aux bornes. Un bruit de capteur gaussien
indépendant, d'écart type 1 mV et de germe 11, est ajouté à la tension. Cet
ajustement n'utilise pas les mesures de température. Les températures de la
cellule et du gabarit sont surveillées pendant les recharges complètes qui
suivent, tandis que leurs paramètres de modèle restent fixes.

Le circuit nominal utilise $R_0=15$ m$\Omega$, $R_1=10$ m$\Omega$ et
$C_1=2400$ F. L'étalonnage ne permet que la modification à un paramètre
suivante :

$$
R_0(\alpha)=\alpha R_0,
\qquad
R_1(\alpha)=\alpha R_1,
\qquad
C_1(\alpha)=\frac{C_1}{\alpha}.
$$

Le multiplicateur sans dimension $\alpha$ mesure la résistance par rapport au
circuit nominal. Ainsi, $\alpha=1$ correspond à la cellule nominale, et
$\alpha=1.8$ rend les deux résistances 80 % plus grandes. La variation inverse
de $C_1$ maintient la constante de temps RC, $R_1C_1$, à 24 secondes. C'est un
moyen contrôlé d'isoler l'amplitude de la réponse résistive, et non
l'affirmation que tout vieillissement modifie ces trois composants dans cette
proportion.

Sous cette restriction, la tension de l'impulsion est linéaire en $\alpha$.
Notons $\bar v_{p,k}$ la tension de polarisation prédite à l'échantillon $k$
par la branche RC nominale, pilotée par le courant connu $I_k$. On définit la
tension mesurée au-dessus de la tension à vide et la prédiction de cet excès
par le modèle nominal par

$$
y_k=V_k^{\mathrm{mes}}-U_{\mathrm{oc}}(z_k),
\qquad
\phi_k=I_kR_0+\bar v_{p,k}.
$$

Ainsi, $y_k$ est l'excès de tension observé à l'échantillon $k$, tandis que
$\phi_k$ est l'excès prédit lorsque le facteur d'échelle de la résistance vaut
un.

Comme la branche mise à l'échelle a la même constante de temps et part du
repos, $v_{p,k}=\alpha\bar v_{p,k}$. En notant $\epsilon_k$ le bruit de mesure
de la tension, les échantillons mesurés vérifient donc

$$
y_k=\alpha\phi_k+\epsilon_k.
$$

L'estimation est la valeur de $\alpha$ qui rend la somme des carrés des
erreurs de prédiction de tension aussi petite que possible, la valeur étant
restreinte à un intervalle plausible :

$$
\hat\alpha
=\underset{0.7\leq a\leq2.5}{\arg\min}
\sum_{k:\,|\phi_k|>10^{-8}\,\mathrm V}
\left(y_k-a\phi_k\right)^2.
$$

Avant l'impulsion, $I_k$ et $\bar v_{p,k}$ sont tous deux nuls; ces
échantillons au repos ne contiennent donc aucune information sur $\alpha$.
L'échelon de courant rend $\phi_k$ non nul, et la relaxation qui suit fournit
des échantillons supplémentaires. Cette expérience peut identifier le
multiplicateur commun parce qu'il est la seule inconnue de l'ajustement : le
rapport $R_0/R_1$ et la constante de temps sont fixés. Elle ne peut pas estimer
séparément $R_0$, $R_1$ et $C_1$.

La valeur ajustée entre dans la règle de recharge comme estimation de la
résistance par le correcteur, $\hat\alpha$. Un **limiteur de courant** est une
règle de sécurité locale qui associe à l'état estimé présent une demande de
courant admissible. À chaque mise à jour de commande, toutes les secondes, le
limiteur reçoit les valeurs simulées de $z$, $v_p$ et $T_c$ et demande le plus
grand courant compatible avec trois limites. Soit $[q]_+=\max(q,0)$, et soit
$h_{cj}=0.55$ W/K le coefficient de transfert thermique modélisé entre la
cellule et le gabarit. La règle est

$$
I=\min\left\{
10,
\left[\frac{4.17-U_{\mathrm{oc}}(z)-v_p}{\hat\alpha R_0}\right]_+,
\sqrt{\left[
\frac{h_{cj}(34.5-T_c)}{\hat\alpha(R_0+R_1)}
\right]_+}
\right\}.
$$

Le premier terme est la limite matérielle de 10 A. Le deuxième empêche la
prédiction instantanée de la tension aux bornes par le modèle de dépasser sa
garde de 4.17 V. Le troisième part de la marge de température restante,
$34.5-T_c$. Multiplier cette marge par $h_{cj}$ donne une allocation simple
pour l'échauffement résistif, modélisé ici par $I^2\hat\alpha(R_0+R_1)$.
Résoudre cette inégalité en $I$ donne le plafond en racine carrée de la règle.
Ce plafond est prudent; il ne prévoit pas la trajectoire future de la
température. Le procédé simulé lui-même est vérifié par rapport aux bornes de
4.20 V et de 35 degrés Celsius. Une résistance ajustée plus grande abaisse les
deux plafonds de courant, celui fondé sur la tension comme celui fondé sur la
température.

Ce limiteur est une règle locale fixe, et non une optimisation sur une suite
future de courants. La recharge rapide sous contraintes motive aussi des
méthodes de commande prédictive plus riches
{cite}`gonzalezSaenz2024fastCharging`.

Trois essais appariés servent à vérifier si le limiteur atteint la cible de
charge en respectant ses bornes sur le procédé de référence, si le modèle
périmé préserve la marge de tension après le changement de résistance, et si
la mise à jour d'un seul paramètre rétablit cette marge.

| essai | facteur du procédé $\alpha$ | facteur du limiteur $\hat\alpha$ |
|---|---:|---:|
| cellule neuve, nominale | 1.0 | 1.0 |
| résistance élevée, modèle périmé | 1.8 | 1.0 |
| résistance élevée, modèle étalonné | 1.8 | ajusté à partir de l'impulsion |

L'audit enregistré de la batterie compare les trois essais appariés selon la
charge, la tension aux bornes, la température de la cellule et le courant
demandé. Chaque courbe n'affiche que le début atteint par la tête de lecture.
Les commandes d'événements mènent à la première réduction du courant, à la
première violation d'une borne du procédé et à la cible de 80 % lorsque ces
événements existent.

```{code-cell} python
:tags: [remove-input]

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from battery_replay import render_battery_replay

display(HTML(render_battery_replay(
    Path("artifacts/battery/textbook_results.json"),
    replay_id="battery-fast-charging-replay",
    fallback_id="fig-battery-fast-charging-fallback",
)))
```

:::{figure} _static/battery/fast-charging.svg
:label: fig-battery-fast-charging-fallback
:class: pdf-fallback
:alt: État de charge, tension, température et courant pour une cellule neuve, une cellule à résistance élevée commandée avec un modèle périmé, et la même cellule après étalonnage de la résistance.

Le modèle périmé perd sa marge de tension après le changement de résistance.
La mise à jour d'un seul facteur d'échelle de résistance ajusté rétablit la
marge testée, au prix d'une recharge plus longue. Le livre en ligne ajoute une
lecture synchronisée et la navigation par événements.
:::

```{include} artifacts/battery/results.md
```

Dans cette simulation déclarée, la comparaison appuie une conclusion étroite :
lorsque le procédé diffère du modèle du correcteur exactement par le facteur
d'échelle commun de la résistance décrit ci-dessus, l'impulsion estime ce
facteur assez bien pour que le même limiteur reste à l'intérieur des bornes de
tension et de température testées. L'ajustement modifie la chute de tension
prédite et la chaleur résistive pour un courant candidat; il ne remplace ni la
conservation de la charge, ni la topologie du circuit, ni les états
thermiques.

La comparaison n'établit pas une règle de recharge sûre pour un produit
physique. Chaque essai donne au limiteur l'état simulé exact; elle ne teste
donc pas l'estimation d'état. Un seul facteur ajusté ne peut pas non plus
représenter la perte de capacité, le dépôt de lithium métallique, une courbe
de tension à vide erronée, un biais de capteur, des gradients spatiaux de
température ou d'autres mécanismes de dégradation électrochimique.

Le chapitre [Commande prédictive](receding-horizon-control.md) optimisera une
suite complète d'entrées futures sous des contraintes d'état et d'entrée. Le
limiteur évalue ici une règle de courant locale et fixe, pour que la
conséquence de la modification d'un seul paramètre du modèle reste visible.

:::{dropdown} Examiner le limiteur local et l'ajustement à un paramètre
```{literalinclude} code/battery_control.py
:language: python
:start-at: def predictive_current_governor
:end-before: def _extract_state
:linenos:
```

```{literalinclude} code/battery_control.py
:language: python
:start-at: def fit_resistance_scale
:end-before: def run_battery_audit
:linenos:
```

{download}`Télécharger l'audit complet du modèle de batterie <code/battery_control.py>`

{download}`Télécharger le moteur de rendu de la reprise enregistrée <code/battery_replay.py>`
:::

Un code de simulateur ou de correcteur généré peut s'exécuter tout en
représentant la mauvaise action, en omettant un mode physique ou en laissant
une exigence de sécurité hors de l'objectif et des contraintes. L'exécution
teste le comportement du logiciel. Les affirmations sur le système visé
exigent d'examiner le modèle et d'obtenir des preuves par des interventions qui
mettent en évidence l'hypothèse contestée.

Un résidu ne peut pas réparer un canal d'action ou un mode physique manquant
si les données d'entraînement n'en contiennent jamais la trace. La
comparaison SwingRL a révélé l'échec en changeant le modèle de suspension tout
en gardant le correcteur fixe. Des interventions comparables sont nécessaires
pour décider quelle structure doit rester et quelle composante doit être
apprise.


## Exercices

:::{exercise} État de cache agrégé
:label: ex-dynamics-cache-balance

Pour le service d'inférence, écrivez un bilan de cache agrégé qui distingue
l'occupation ajoutée pendant le traitement des jetons d'invite et de sortie, la
réserve ajoutée lorsqu'une requête entre en décodage, et l'occupation libérée à
l'achèvement d'une requête. Donnez une raison pour laquelle le total agrégé
peut être insuffisant pour la prédiction.
:::

:::{solution} ex-dynamics-cache-balance
:class: dropdown

Un bilan compatible avec l'abstraction pédagogique est

$$
m_{t+1}=m_t+C_t^p+C_t^d+rN_t^p
-\sum_{j\in\mathcal C_t}L_{j,t},
$$

où $C_t^p$ et $C_t^d$ sont les jetons d'invite et de décodage traités, $rN_t^p$
est la réserve fixe des requêtes qui entrent en décodage, $\mathcal C_t$ est
l'ensemble des requêtes qui se terminent, et $L_{j,t}$ est toute l'occupation
libérée par la requête terminée $j$. Des totaux égaux peuvent masquer des
longueurs de sortie restantes différentes d'une requête à l'autre; les
instants futurs d'achèvement et de libération du cache peuvent donc différer.
:::

:::{exercise} Capacités de trois interfaces de modèle
:label: ex-dynamics-interface-capabilities

Pour un solveur d'EDO différentiable, un simulateur robotique à
réinitialisation et pas, et une table fixe de transitions historiques,
énumérez les opérations que chacun offre. Nommez une méthode du livre que
chaque interface permet directement.
:::

:::{solution} ex-dynamics-interface-capabilities
:class: dropdown

Le solveur d'EDO différentiable permet des déroulements sous des entrées
choisies et les dérivées de ces déroulements, ce qui peut servir aux méthodes
de tir ou à la commande prédictive par gradient. Le simulateur à
réinitialisation et pas permet des déroulements échantillonnés contrefactuels
à partir d'états et d'actions choisis, ce qui peut servir à la commande
prédictive par simulation, à l'évaluation de Monte-Carlo ou à l'apprentissage
de politiques. Les transitions historiques permettent l'ajustement et
l'évaluation sur leur loi enregistrée, y compris l'identification de système
ou l'itération Q ajustée, mais ne répondent pas à elles seules à des questions
contrefactuelles arbitraires.
:::

:::{exercise} Génération de texte : transition ou politique?
:label: ex-dynamics-language-transition

Pour la génération de texte autorégressive, distinguez la transition de
l'environnement de la politique sur les jetons. Comment le modèle change-t-il
lorsqu'un appel d'outil renvoie de l'information tirée d'une base de données
externe?
:::

:::{solution} ex-dynamics-language-transition
:class: dropdown

Pour un préfixe de texte fermé, la transition est la concaténation
déterministe, $s_{t+1}=\operatorname{concat}(s_t,a_t)$, tandis que le modèle de
langue fournit la loi des jetons $\pi(a_t\mid s_t)$. Un appel d'outil franchit
cette frontière. La valeur renvoyée est une observation externe, et l'état
doit conserver le résultat d'outil ou l'historique d'interaction nécessaire
pour prédire les transitions suivantes.
:::

:::{exercise} Censure à une station vide
:label: ex-dynamics-bixi-censoring

À partir du contre-exemple des déplacements terminés, ajoutez une tentative de
location pendant que la station est vide. Pourquoi la demande latente
change-t-elle alors que le journal des déplacements terminés reste inchangé?
:::

:::{solution} ex-dynamics-bixi-censoring
:class: dropdown

L'usager supplémentaire ne peut pas partir, faute de vélo disponible. La
tentative augmente donc la demande de un, mais ne produit aucun relevé de
location terminée. Les deux mondes produisent le même journal de déplacements
observé, même si leurs historiques de demande tentée diffèrent.
:::

:::{exercise} Sensibilité à la charge offerte
:label: ex-dynamics-inference-dilation

Utilisez `data/inference_serving/azure_code_evaluation.csv` et
`normalize_offered_load` de `code/inference_serving.py`. Recalculez la
dilatation de la charge offerte après avoir retiré les dix requêtes dont les
invites sont les plus longues. Quelles grandeurs du modèle changent, et
quelles équations de conservation restent inchangées?
:::

:::{solution} ex-dynamics-inference-dilation
:class: dropdown

Les dix invites retirées contiennent chacune 7 436 jetons. Pour le profil
versionné, la dilatation passe de

$$
3.5538284803\quad\text{à}\quad3.4524186516.
$$

Le total du service isolé, les instants d'arrivée normalisés et la suite
exogène de requêtes changent. La forme algébrique des bilans de requêtes, de
jetons, de cache et de chaleur ne change pas.
:::

:::{exercise} Étalonnage de la batterie
:label: ex-dynamics-battery-calibration

En ignorant la tension et la température, calculez le temps idéal nécessaire
pour faire passer une cellule de $5\ \mathrm{Ah}$ de 20 à 80 % d'état de
charge à $10\ \mathrm A$. Utilisez ensuite l'équation de sortie de Thévenin
pour expliquer pourquoi un limiteur dont l'estimation de résistance est
périmée peut violer la borne de tension alors que la conservation de la charge
est correcte. Quelle structure doit rester fixe, et quelle grandeur peut être
estimée à partir de l'impulsion de courant connue et de la tension mesurée?
:::

:::{solution} ex-dynamics-battery-calibration
:class: dropdown

La charge requise est $0.6(5\ \mathrm{Ah})=3\ \mathrm{Ah}$, donc

$$
\Delta t=\frac{3\ \mathrm{Ah}}{10\ \mathrm A}
=0.3\ \mathrm h=18\ \mathrm{min}.
$$

Comme $V=U_{\mathrm{oc}}+v_p+IR_0$, une résistance sous-estimée amène le
limiteur à sous-estimer la tension aux bornes et à demander trop de courant.
La conservation de la charge, la topologie du circuit et les équations
thermiques restent fixes. Pendant l'impulsion connue de 5 A, le saut de
tension immédiat et la relaxation plus lente rendent visible la réponse
résistive. Ajuster cette réponse estime la seule inconnue permise, $\alpha$,
où $R_0$ et $R_1$ valent $\alpha$ fois leurs valeurs nominales et $C_1$ est
divisé par $\alpha$. L'expérience met à jour ce seul paramètre du correcteur;
elle ne réapprend pas les équations de bilan et n'identifie pas séparément les
trois éléments du circuit.
:::

## Synthèse et perspectives

Les équations, les simulateurs à réinitialisation et pas, les transitions
historiques et les environnements interactifs offrent des opérations
différentes même lorsqu'ils décrivent la même évolution nominale. Les bilans
et les contraintes connus peuvent rester explicites pendant que les données
estiment des paramètres incertains, une dynamique résiduelle, des valeurs ou
des politiques. L'exemple de la batterie rend ce partage concret : les bilans
électrique et thermique restent fixes, et un seul facteur de résistance est
réétalonné.

Une interface de modèle peut désormais produire ou contraindre des
trajectoires candidates. Quelle suite d'actions admissible donne les meilleurs
résultats à partir d'une condition initiale donnée? [Le problème de contrôle
optimal à horizon fini](discrete-time-optimal-control.md) en est la première
réponse, et il reste en boucle ouverte jusqu'à ce qu'on permette aux mesures
ultérieures de modifier la décision.

## Sources numériques

La dépendance SwingRL est figée au commit
[`d579663`](https://github.com/pierrelux/swing-rl/commit/d579663fc81c044729f4d3ab60bf63bcdbd27b9a).
Les trois chapitres de modélisation exécutent du code propre à chaque domaine
et lisent des résultats d'expériences versionnés :

- {download}`Correcteur SwingRL et audit du modèle <code/swing_control.py>`
- {download}`Modèle de stock BIXI et correcteurs <code/bixi_control.py>`
- {download}`Moteur de rendu de la reprise BIXI <code/bixi_replay.py>`
- {download}`Procédé, estimateurs et correcteur de la nacelle de caméra <code/gimbal_control.py>`
- {download}`Moteur de rendu de la reprise de la nacelle <code/gimbal_replay.py>`
- {download}`Audit du modèle de batterie et ajustement des paramètres <code/battery_control.py>`
- {download}`Moteur de rendu de la reprise de la batterie <code/battery_replay.py>`
- {download}`Impulsion de diagnostic de la batterie <artifacts/battery/diagnostic_pulse.csv>`
- {download}`Trajectoires enregistrées de la batterie <artifacts/battery/trajectories.npz>` et {download}`données de reprise pour le navigateur <artifacts/battery/textbook_results.json>`
- {download}`Manifeste des résultats de la batterie <artifacts/battery/manifest.json>`
- {download}`Générateur des figures conceptuelles <code/modeling_interfaces.py>`
- {download}`SVG d'échantillonnage et d'intégration <_static/sampling-and-integration.svg>`
- {download}`SVG boucle ouverte et rétroaction <_static/open-loop-vs-feedback.svg>`
- {download}`SVG des coordonnées de la balançoire <_static/swing-reduced-coordinates.svg>`
- {download}`SVG de la frontière du service d'inférence <_static/inference-serving-boundary.svg>`
- {download}`Modèle du service d'inférence <code/inference_serving.py>`
- {download}`Moteur de rendu de la reprise d'inférence <code/inference_replay.py>`

Les commandes d'animation sont intégrées au site statique et ne nécessitent
aucun noyau Python en direct.
