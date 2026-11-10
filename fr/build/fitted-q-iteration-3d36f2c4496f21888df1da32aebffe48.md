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

# Apprendre à partir de données historiques et en ligne

Les algorithmes des chapitres précédents choisissaient les états où les
cibles étaient calculées et tiraient autant de successeurs qu'ils le
voulaient. En pratique, les données sont imposées. Un journal de transitions
$(s, a, r, s')$ a été enregistré par quiconque exploitait le système, avec un
successeur par transition, et on ne peut tirer aucun nouveau successeur; ou
bien un tampon se remplit à mesure que l'agent agit, de sorte que les données
changent pendant l'ajustement de la fonction. Ce chapitre porte sur
l'itération Q ajustée sur de telles données. Les algorithmes par lots viennent
en premier : l'itération Q ajustée à base d'arbres d'Ernst et l'itération Q
ajustée neuronale, appliquées à deux journaux, l'un provenant de
l'ordonnanceur de service d'inférence du chapitre sur l'horizon infini et
l'autre constitué des alertes de collision que l'ESA a reçues pour ses
satellites. Les deux montrent que l'erreur d'une politique ajustée a deux
sources, l'approximation là où deux actions sont presque à égalité et la
couverture là où le journal n'a jamais essayé une action, et le second montre
quoi faire lorsque quelques événements rares dominent la régression. Le
chapitre ouvre ensuite l'étape d'ajustement. Écrire explicitement les pas de
gradient d'un ajustement neuronal et les fusionner avec la boucle extérieure
transforme l'itération Q ajustée en DQN, avec son réseau cible et son réseau
en ligne; laisser le tampon évoluer pendant l'apprentissage donne DQN
proprement dit, Double DQN et, à la limite d'une transition par mise à jour,
le Q-learning. Une section de synthèse expose l'espace de conception que
couvrent ces choix, et le chapitre se termine par des pertes qui modélisent
le bruit des cibles de Bellman mieux que l'erreur quadratique.

Comment le tampon, la cible de Bellman, l'approximateur de fonctions et le
budget d'optimisation se combinent-ils lorsqu'un seul successeur observé est
disponible pour chaque transition stockée?

::::{admonition} Objectifs d'apprentissage
:class: note

Après l'étude de ce chapitre, vous devriez pouvoir :
- Écrire l'itération Q ajustée sur un ensemble fixe de transitions, et dire où la loi empirique des données remplace le modèle
- Attribuer les erreurs d'une politique ajustée à l'approximation, là où des actions sont presque à égalité, ou à la couverture, là où les données n'ont jamais essayé une action
- Mettre à l'échelle les cibles d'une régression afin que quelques événements rares et coûteux ne dominent pas l'ajustement
- Aplatir les boucles imbriquées de l'itération Q ajustée neuronale pour obtenir DQN, et expliquer les rôles de son réseau en ligne et de son réseau cible
- Situer un algorithme dans l'espace de conception de l'itération Q ajustée selon son tampon, sa fonction cible, son ajustement, son initialisation et sa perte

**Préalables :** l'itération sur les valeurs ajustée et son analyse d'erreur ([méthodes de valeur ajustées](approximate-bellman-equations.md)), les fonctions de valeur d'action, le paradigme de l'échantillon unique et le biais de maximisation ([estimation de Bellman par Monte-Carlo](monte-carlo-bellman-estimation.md)).
::::

(sec-fqi-batch)=
## Apprendre à partir d'un ensemble fixe de transitions

L'opérateur de Bellman sur les valeurs d'action contient une espérance sous
la loi de transition, que nous notons $P$, sur les transitions
$\tau = (s, a, r, s')$ :

$$
(\Bellman q)(s,a) = r(s,a) + \gamma \int \max_{a'} q(s',a')\, P(ds' \mid s,a).
$$

Un algorithme par lots ne voit jamais $P$. Il reçoit un jeu de données
$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$ et travaille avec la **loi
empirique**

$$
\hat{P}_{\mathcal{D}} = \frac{1}{N} \sum_{\tau \in \mathcal{D}} \delta_\tau,
$$

où $\delta_\tau$ est une masse ponctuelle en la transition $\tau$.
Échantillonner selon $\hat{P}_{\mathcal{D}}$ signifie choisir uniformément au
hasard une transition enregistrée, et l'espérance sous $P$ de toute fonction
d'une transition est estimée par sa moyenne sur le jeu de données :
c'est l'estimateur par moyenne empirique du
[chapitre sur les mises à jour de Bellman échantillonnées](monte-carlo-bellman-estimation.md),
appliqué aux transitions. L'itération Q ajustée sur un lot est l'itération sur
les valeurs sous cette loi empirique. La cible d'une transition est
$y_i = r_i + \gamma \max_{a'} q(s'_i, a')$, avec l'unique successeur
enregistré, et l'ajustement fait la moyenne de l'erreur des cibles sur les
transitions du jeu de données. Dans le langage de la statistique, c'est un
estimateur par substitution : là où l'algorithme a besoin du $P$ inconnu, il
lui substitue $\hat{P}_{\mathcal{D}}$.

```{prf:remark} La loi empirique comme mesure de probabilité
:class: dropdown

La loi empirique $\hat{P}_{\mathcal{D}} = \frac{1}{N}\sum_{\tau \in \mathcal{D}} \delta_{\tau}$ est une mesure de probabilité discrète sur $N$ points, que les espaces d'états et d'actions soient continus ou discrets. Pour tout ensemble mesurable $A \subseteq \mathcal{S} \times \mathcal{A} \times \mathbb{R} \times \mathcal{S}$,

$$
\hat{P}_{\mathcal{D}}(A) = \frac{1}{N}\sum_{\tau \in \mathcal{D}} \mathbb{1}[\tau \in A] = \frac{|\{\tau \in \mathcal{D} : \tau \in A\}|}{N},
$$

de sorte qu'elle attribue la probabilité $1/N$ à chaque transition enregistrée et zéro ailleurs. Une transition qui ne s'est jamais produite n'a aucun poids, ce qui est l'énoncé formel du problème de couverture que mettent en évidence les deux expériences ci-dessous.
```

```{prf:algorithm} Itération Q ajustée sur un lot
:label: fitted-q-iteration-batch

**Entrée :** jeu de données $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$, approximateur de fonctions $q(s,a; \boldsymbol{\theta})$, facteur d'actualisation $\gamma$, nombre maximal d'itérations $n_{\max}$

**Sortie :** paramètres appris de la fonction Q $\boldsymbol{\theta}$

1. Initialiser $\boldsymbol{\theta}_0$
2. $n \leftarrow 0$
3. **répéter**  $\quad$ // **Boucle extérieure : itération sur les valeurs**
4. $\quad$ **// Construire le jeu de données de régression avec les cibles de Bellman**
5. $\quad$ $X^{(n)} \leftarrow \{(s_i, a_i) : (s_i, a_i, r_i, s'_i) \in \mathcal{D}\}$
6. $\quad$ **pour** chaque $(s_i, a_i, r_i, s'_i) \in \mathcal{D}$ **faire**
7. $\quad\quad$ $y_i^{(n)} \leftarrow r_i + \gamma \max_{a' \in \mathcal{A}} q(s'_i, a'; \boldsymbol{\theta}_n)$
8. $\quad$ **fin pour**
9. $\quad$ **// Boucle intérieure : ajuster la fonction Q aux cibles**
10. $\quad$ $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(X^{(n)}, y^{(n)}, \boldsymbol{\theta}_{\text{init}}, K)$
11. $\quad$ $n \leftarrow n+1$
12. **jusqu'à** convergence ou $n \geq n_{\max}$
13. **retourner** $\boldsymbol{\theta}_n$
```

Les couples état-action $X^{(n)}$ sont les mêmes à chaque itération; seules
les cibles changent, parce qu'elles sont calculées avec la fonction Q
courante. L'opération `fit` de la ligne 10 minimise
$\sum_i \ell(q(s_i, a_i; \boldsymbol{\theta}), y_i^{(n)})$ pour une perte
$\ell$, par la procédure qu'exige la classe de fonctions, à partir d'une
initialisation $\boldsymbol{\theta}_{\text{init}}$ et avec un budget
d'optimisation $K$. L'apprentissage supervisé initialiserait aléatoirement et
irait jusqu'à convergence. Les deux algorithmes par lots d'origine diffèrent
précisément par ces choix.

{cite:t}`ernst2005tree` ont instancié ce schéma avec des arbres extrêmement
aléatoires, un ensemble dont la prédiction est la moyenne des cibles d'une
feuille. Chaque ajustement fait croître les arbres à partir de rien, un
démarrage à froid, et va jusqu'au bout, puisque la construction d'un arbre
n'a pas de forme incrémentale. Un arbre à partition fixe est un moyenneur, de
sorte que le théorème de convergence du
[chapitre sur les méthodes de valeur ajustées](approximate-bellman-equations.md)
s'applique dès que les partitions cessent de changer, et les arbres acceptent
des entrées de grande dimension sans conception de caractéristiques.
{cite:t}`riedmiller2005neural` a remplacé l'ensemble par un réseau de neurones
$q(s,a;\boldsymbol\theta)$, qui interpole de façon lisse sur l'espace
état-action, et l'a ajusté par descente de gradient (RProp, choisi pour sa
faible sensibilité aux hyperparamètres) jusqu'à ce que la perte cesse de
diminuer. Le réseau est démarré à chaud : chaque ajustement part des
paramètres de l'itération précédente, ce qui est naturel pour une méthode de
gradient et constitue le premier des choix que la seconde moitié de ce
chapitre transforme en DQN.

```{prf:remark} États buts dans l'itération Q ajustée neuronale
:class: dropdown

Pour les tâches épisodiques comportant des états buts $S^+$ et des états interdits $S^-$, Riedmiller a modifié les cibles :

$$
y_i^{(n)} = \begin{cases}
c(s_i, a_i, s'_i) & \text{si } s'_i \in S^+ \text{ (but atteint)} \\
C^- & \text{si } s'_i \in S^- \text{ (état interdit, typiquement } C^- = 1\text{)} \\
c(s_i, a_i, s'_i) + \gamma \max_{a'} q(s'_i, a'; \boldsymbol{\theta}_n) & \text{sinon,}
\end{cases}
$$

où $c(s, a, s')$ est le coût immédiat. Un état but n'a aucun coût futur ni terme d'autoamorçage, un état interdit porte une pénalité fixe, et tout autre état utilise la mise à jour de Bellman. L'heuristique d'indication du but (*hint-to-goal*) ajoute des transitions synthétiques issues d'états buts avec une cible nulle, ce qui fixe la fonction Q à zéro dans la région du but et encode explicitement la condition aux limites.
```

### Expérience : ordonnancement à partir de données de transition fixes

Le processus de décision markovien d'ordonnancement d'inférence réduit des
[processus de décision markoviens à horizon infini](infinite-horizon-mdps.md#exact-scheduling-mdp-for-inference-serving)
admet une solution exacte par itération sur les valeurs, parce que sa matrice
de transition est connue. Supposons plutôt que seule une table fixe de
transitions soit conservée. Chaque ligne contient un état réduit $(p,d)$, une
action d'ordonnancement, le coût $c$ et l'état réduit suivant. La forme
générique ci-dessus, qui maximise la récompense, devient une forme qui
minimise le coût après un changement de signe. L'itération Q ajustée peut
réutiliser ces lignes, mais elle ne reçoit aucune donnée directe sur une
région état-action que la politique de comportement ne visite jamais. Toute
prédiction dans cette région provient du biais inductif du régresseur.

Chaque ligne de transition est échantillonnée à partir du processus de
décision markovien fini, et non enregistrée directement dans vLLM. Son modèle
d'arrivées provient de la trace Azure normalisée, tandis que ses taux et
puissances par phase proviennent d'une campagne de mesure sur un NVIDIA L4 de
Qwen/Qwen2.5-7B-Instruct servi par vLLM 0.28.0. Les jeux de données héritent
donc d'un étalonnage matériel mesuré et de toutes les hypothèses d'agrégation
du modèle réduit.

Deux procédures d'échantillonnage produisent des jeux de données appariés de
50 000 transitions. Le jeu de données **large** tire un état réduit
uniformément, puis tire uniformément parmi ses actions admissibles. Le jeu de
données **étroit** suit des trajectoires à partir du système vide sous une
heuristique de priorité au décodage, et choisit une action admissible
aléatoire avec probabilité 0.05. Les deux utilisent la graine 0 et le même
processus de décision markovien. Leurs lois d'échantillonnage état-action
différentes constituent le changement contrôlé.

Chaque balayage de l'itération Q ajustée construit les cibles

$$
y_i^{(n)}=c_i+0.99\min_{u'\in\mathcal A(s_i')}
q_n(s_i',u')
$$

et ajuste un régresseur Extra-Trees conjoint état-action. L'ensemble contient
200 arbres de profondeur maximale 12 et de taille minimale de feuille 2. La
graine aléatoire est 0; les ajustements successifs des arbres utilisent les
graines déterministes 0 à 49 sur les 50 balayages de l'itération Q ajustée.
Les trois actions d'ordonnancement entrent sous forme d'un codage un parmi n,
conformément à la représentation conjointe état-action de l'algorithme par
lots générique ci-dessus. Avant chaque ajustement, les lignes ayant le même
état et la même action sont fusionnées en un point dont la cible est leur
moyenne et dont le poids est leur nombre. La taille minimale de feuille de 1
compte donc des couples état-action distincts, et non des lignes : une valeur
plus grande forcerait chaque feuille à faire la moyenne des cibles d'au moins
deux couples différents, ce qui brouille les petites différences entre les
actions d'un même état.

Ne changer que la loi d'échantillonnage permet de tester l'effet de la
couverture sur une procédure d'itération Q ajustée par ailleurs identique. La
politique exacte de programmation dynamique et les deux politiques ajustées
sont évaluées sur les mêmes 2 000 épisodes à graine fixée, chacun de 300 pas.
Le rendement et le désaccord entre politiques mesurent la concordance avec
l'objectif du processus de décision markovien réduit et avec sa solution
exacte. Le temps d'attente moyen, les arrivées perdues et la couverture
état-action observée situent les différences opérationnelles. Les arrêts du
décodage comptent les périodes pendant lesquelles des tâches de décodage
étaient actives mais où la politique a fait autre chose. Dans ce modèle, ce
ne sont pas des erreurs : la politique exacte interrompt volontairement le
décodage pendant qu'elle effectue un préremplissage pour agrandir le lot.

```{code-cell} python
:tags: [remove-input]
:label: fig-inference-scheduling-fqi
:caption: Itération Q ajustée par Extra-Trees sur deux jeux de données fixes de 50 000 transitions échantillonnés à partir du même processus de décision markovien d'ordonnancement étalonné par des mesures sur L4. Ce sont des transitions du modèle, et non des journaux de vLLM. Le jeu de données large échantillonne uniformément les états et les actions admissibles; le jeu de données étroit suit la priorité au décodage avec 5 pour cent d'actions aléatoires. Chaque table de politique couvre les 49 états et indique si l'action affichée figure dans le tampon d'entraînement de cette politique.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="fqi",
)))
```

:::{figure} _static/inference_serving/fqi.svg
:label: fig-inference-scheduling-fqi-fallback
:class: pdf-fallback
:alt: Comparaison statique de l'itération sur les valeurs exacte et des politiques Q ajustées apprises à partir de données de transition larges et étroites.

Tables de politique sur les 49 états. En chaque état, un cercle blanc indique
que le tampon contient des visites de l'action choisie par la politique
affichée. La version interactive ajoute la lecture, l'avance pas à pas et le
déplacement manuel dans une trajectoire de file d'attente échantillonnée à
partir du modèle de transition réduit plutôt qu'observée directement dans
vLLM.
:::

La programmation dynamique exacte et les deux politiques Q ajustées utilisent
les mêmes 2 000 épisodes à graine fixée de 300 pas. La couverture est
calculée à partir des couples état-action admissibles présents dans chaque
tampon de transitions fixe.

```{code-cell} python
:tags: [remove-input]

import pandas as pd

fqi_metrics = pd.read_csv(
    "artifacts/inference_serving/metrics_fqi.csv"
).set_index("controller")
fqi_metrics[
    [
        "mean_discounted_return",
        "mean_queue_length",
        "mean_waiting_time_s",
        "mean_decode_stalls",
        "mean_dropped_arrivals",
        "coverage_fraction",
        "policy_disagreement_fraction",
    ]
].T.rename_axis("metric").round(3)
```

{download}`Télécharger toutes les métriques de l'itération Q ajustée (CSV) <artifacts/inference_serving/metrics_fqi.csv>`

Le tampon large couvre tous les couples état-action admissibles, et sa
politique ajustée est en désaccord avec la politique exacte dans 6 des 49
états (12.24 pour cent). Les six se trouvent près de la courbe de
commutation, là où la politique exacte effectue un préremplissage et où la
politique ajustée décode. Chacun de ces couples état-action apparaît environ
340 fois dans le tampon; les erreurs ne viennent donc pas de données
manquantes. Dans ces états, les valeurs d'action exactes du préremplissage et
du décodage diffèrent d'au plus 1.24, alors que les valeurs elles-mêmes vont
de 254 à 891. Un régresseur précis à une fraction de pour cent près peut
encore classer dans le mauvais ordre deux actions aussi proches. Le rendement
actualisé moyen de la politique large est de $-242.25$, contre $-230.04$ pour
la politique exacte, et son temps d'attente moyen est de 5.76 secondes plutôt
que 5.24.

Le tampon étroit couvre 41.73 pour cent des couples admissibles et est en
désaccord avec la politique exacte dans 19 états (38.78 pour cent). Dans 15 de
ces états, le tampon ne contient aucune visite de l'action que choisit la
politique ajustée. Le journal suit la priorité au décodage, qui n'effectue
jamais de préremplissage pendant qu'une requête est en décodage; le
régresseur n'a donc aucune donnée sur les décisions de mise en lot que prend
la politique exacte. Ses choix dans ces états sont des extrapolations : il
reste inactif dans plusieurs états où six requêtes sont en décodage, et il
poursuit le préremplissage au-delà du point où la politique exacte décode.
Son rendement actualisé moyen est de $-286.08$, son temps d'attente moyen de
11.20 secondes, et il perd 1.40 arrivée par épisode contre 0.20 pour la
politique exacte. Le rendement est l'opposé du coût actualisé; des valeurs
plus élevées sont donc meilleures.

Les deux jeux de données ont la même taille; la différence entre leurs
politiques ne peut donc pas être attribuée au nombre de lignes de transition.
Les erreurs du tampon large viennent de l'approximation, là où l'écart entre
actions est petit; les erreurs du tampon étroit viennent de la couverture, là
où la politique de comportement n'a jamais pris l'action en question. Ce sont
la première et la troisième difficulté nommées après la borne d'erreur du
[chapitre sur les méthodes de valeur ajustées](approximate-bellman-equations.md#sec-abe-error) :
un ajustement est précis en moyenne, pas aux états où deux actions sont
presque à égalité, et il n'est maîtrisé que là où il y a des données.

L'étalonnage matériel est mesuré, alors que l'entraînement et l'évaluation
restent à l'intérieur du processus de décision markovien à 49 états énoncé.
Ces résultats portent sur la couverture hors ligne sous cette abstraction. Ce
ne sont pas un banc d'essai de l'ordonnancement de vLLM ni du matériel NVIDIA
L4 d'un déploiement à l'autre.

:::{dropdown} Examiner l'expérience d'itération Q ajustée
```{literalinclude} code/inference_control.py
:language: python
:start-at: def fit_fqi
:end-before: def _reduced_policy_trajectory
:linenos:
```
{download}`Télécharger l'implémentation complète de la commande de l'inférence <code/inference_control.py>`
:::

(sec-fqi-conjunctions)=
### Exemple : évitement de collision appris à partir d'alertes enregistrées

Cet exemple n'a pas de modèle, seulement un journal. Lorsqu'on prévoit qu'un satellite et un autre objet, le plus souvent un débris, passeront près l'un de l'autre,
l'opérateur reçoit environ trois messages de données de conjonction par jour
pendant la semaine qui précède l'instant du rapprochement maximal. Chaque
message donne une probabilité de collision estimée à partir du plus récent
suivi. L'opérateur peut planifier une manœuvre d'évitement ou attendre le
message suivant. Attendre procure de l'information, parce que la plupart des
alarmes disparaissent à mesure que le suivi s'améliore. Cela rend aussi la
manœuvre plus coûteuse : plus près de la rencontre, il faut une plus grande
variation de vitesse pour la même distance de passage, et il reste moins de
temps pour planifier. L'ESA planifie ses manœuvres au moins deux jours avant
le rapprochement maximal, et ses missions en orbite basse réagissent à des
probabilités de collision comprises entre $10^{-5}$ et $10^{-4}$
{cite}`Uriot2022`. L'ESA a publié les messages de 13 154 conjonctions de ses
satellites pour un concours d'apprentissage automatique {cite}`Uriot2022`, et
nous les utilisons comme journal.

Nous posons un problème d'arrêt optimal sur ces suites. Une décision est prise
à chaque message reçu au moins deux jours avant le rapprochement maximal, et
une fois de plus à l'échéance de deux jours avec le dernier message. Une
manœuvre $t$ jours avant le rapprochement maximal coûte $c(t) = (1 + 2/t)/2$,
en unités d'une manœuvre à deux jours. Un événement sans manœuvre coûte
$C = 10^5$ fois la probabilité de collision de son dernier message, la
meilleure estimation disponible, de sorte qu'une manœuvre à l'échéance
s'équilibre à une probabilité de $10^{-5}$. L'état est ce que sait l'opérateur
au moment d'une décision : le temps $t$ jusqu'au rapprochement maximal, la
dernière log-probabilité $\rho$ et l'âge de ce message. Les coûts sont des
choix pédagogiques, et les analystes de l'ESA examinent bien plus qu'un seul
nombre.

Deux caractéristiques du problème permettent à l'itération Q ajustée de
fonctionner à partir du seul journal. Une manœuvre met fin au problème à un
coût connu, de sorte que $q(s, \text{manœuvre}) = c(t)$ n'a pas besoin d'être
appris. Et attendre ne change pas les messages qui arrivent : les suites
publiées ne contiennent aucune manœuvre, et chacune montre donc ce qui suit
« attendre ». L'itération Q ajustée n'a qu'à apprendre
$W(s) = q(s, \text{attendre})$, par régression sur les transitions
enregistrées $(s_i, s_i')$ avec les cibles

$$
y_i = \begin{cases} C\, p_{\text{final}} & \text{si } s_i \text{ est l'échéance,} \\
\min\{c(t_i'),\, W_k(s_i')\} & \text{sinon,} \end{cases}
\qquad W_{k+1} = \operatorname{fit}\big(\{(s_i, y_i)\}\big).
$$

Le minimum sur les deux actions est le $\max$ de l'opérateur de Bellman pour
des coûts. Il n'y a aucune espérance à calculer : chaque successeur enregistré
est un tirage de $s'$, le paradigme de l'échantillon unique du
[chapitre sur les mises à jour de Bellman échantillonnées](monte-carlo-bellman-estimation.md).
Nous ajustons des arbres extrêmement aléatoires, le moyenneur de l'itération
Q ajustée à base d'arbres {cite}`ernst2005tree`, sur les époques de décision
enregistrées, et nous jugeons chaque politique hors échantillon : chaque
événement est décidé par une politique ajustée sans lui.

:::{figure} _static/conjunction_wait_or_burn/wait_or_burn.svg
:label: fig-conjunction-wait-or-burn

Évitement de collision à partir des messages de conjonction de l'ESA. (a) La
probabilité de collision des 742 événements dont l'estimation a franchi la
probabilité d'équilibre, environ $10^{-5}$ (tirets), avant l'échéance. Les 59
en couleur ont fini au-dessus. Les messages sous $10^{-8}$, dont la moitié indiquent une probabilité de $10^{-30}$, sortent du panneau. (b) Coût total de chaque politique sur l'ensemble des événements, réparti entre le
coût des manœuvres et celui des collisions espérées. Ne jamais manœuvrer coûte 2 668.
(c) La politique apprise par itération Q ajustée sur tous les événements, pour un message qui
vient d'arriver. Au-dessus de la frontière bleue, elle manœuvre. La ligne en
tirets est la probabilité d'équilibre du message pris tel quel.
:::

:::{include} artifacts/conjunction_wait_or_burn/results.md
:::

Manœuvrer à la première alarme est sûr mais coûteux : cette règle effectue
742 manœuvres pour 62 événements qui en exigeaient une, parce que 92 alarmes
sur 100 se résorbent ({numref}`fig-conjunction-wait-or-burn`, panneau (a)).
Attendre l'échéance puis agir selon le dernier message réduit le coût de
moitié. C'est la règle qu'un opérateur écrirait à la main. On n'a dit à
l'itération Q ajustée ni qu'attendre est payant ni à quelle probabilité agir,
et elle arrive à environ 7 % de la règle manuelle, 313 contre 291. Sa règle
apprise (panneau (c)) laisse passer une nouvelle alarme à plus de trois jours
de l'échéance, à moins que la probabilité ne vaille de vingt à quarante-cinq
fois la valeur d'équilibre, auquel cas elle manœuvre tôt, pendant qu'une
manœuvre coûte encore peu. Le seuil s'abaisse à l'approche de l'échéance,
mais à l'échéance il exige encore environ trois fois la probabilité
d'équilibre, un pari que les messages ultérieurs baisseront, et ce pari
manque davantage d'événements qui exigeaient une manœuvre, 20 contre 6 pour
la règle manuelle. Deux choix décident si l'itération Q ajustée fonctionne
seulement ici, et tous deux découlent de l'analyse d'erreur des chapitres
précédents.

1. L'ajustement doit ajuster la bonne quantité. La cible brute
   $C\, p_{\text{final}}$ est à queue lourde. Un événement du journal se
   termine à une probabilité de 2 % et coûte environ 2 070 si l'on ne fait
   rien, plus de trois fois le coût de tous les autres événements réunis.
   Les arbres extrêmement aléatoires prédisent la moyenne des cibles d'une
   feuille; ce seul coût est donc partagé par tous les états des feuilles où
   tombent ses époques, et ces états paraissent dangereux. Ajustée sur tous
   les événements, la politique à cibles brutes effectue 342 manœuvres;
   ajustée sans ce seul événement, 141. Hors échantillon, elle effectue 374
   manœuvres et coûte 441, moitié plus que la règle manuelle. La seconde
   version ajuste le rapport $y_i / (C\, 10^{\rho_i})$, le coût relatif à ce
   que prédit le message courant à lui seul, puis remultiplie l'ajustement.
   Le régresseur apprend alors un facteur de correction de la propre
   estimation de l'opérateur. Pour l'événement dangereux, ce facteur reste
   entre 0.7 et 7, une valeur ordinaire, et la politique effectue 166
   manœuvres avec l'événement dans les données et 175 sans lui.
2. L'ajustement ne vaut que ce que valent les données là où les décisions se
   prennent. Cinq sur six des 123 882 époques de décision ont des
   probabilités inférieures à $10^{-7}$, où attendre est manifestement le bon
   choix, et nous les excluons de l'ajustement. Seuls 62 événements exigeaient
   une manœuvre. C'est le problème de couverture derrière les coefficients de
   concentrabilité du
   [chapitre sur les méthodes de valeur ajustées](approximate-bellman-equations.md#sec-abe-error) :
   l'erreur de régression qui compte est celle des quelques états où les deux
   actions sont proches, et aucune quantité de données ailleurs ne la réduit.
   La politique a posteriori, à 147, n'est pas un objectif : elle connaît
   l'estimation des messages qui arrivent après l'échéance, ce qu'aucune
   politique ne peut utiliser.

- {download}`Modèle de conjonction et itération Q ajustée <code/conjunction_wait_or_burn.py>`
- {download}`Générateur de l'expérience de conjonction <scripts/build_conjunction_wait_or_burn_artifacts.py>`
- {download}`Résultats de conjonction <artifacts/conjunction_wait_or_burn/metrics.json>`
- {download}`Messages de conjonction utilisés (colonnes du jeu d'entraînement de l'ESA, CC BY 4.0) <data/conjunctions/kelvins_cdm_extract.csv.gz>`

(sec-fqi-flattening)=
## Des boucles imbriquées aux réseaux cibles

L'itération Q ajustée a une structure imbriquée : la boucle extérieure calcule
les cibles de Bellman, et la boucle intérieure y ajuste l'approximateur de
fonctions. Pour un ensemble d'arbres, la boucle intérieure est un seul appel
qui construit les arbres, et l'imbrication est la seule présentation
possible. Pour un réseau de neurones, la boucle intérieure est une suite de
pas de gradient,

$$
\boldsymbol{\theta}_n^{(k+1)} = \boldsymbol{\theta}_n^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}\big(\boldsymbol{\theta}_n^{(k)}; \mathcal{D}_n^{\text{fit}}\big), \qquad k = 0, \ldots, K-1,
$$

qui part de $\boldsymbol{\theta}_n^{(0)} = \boldsymbol{\theta}_n$ et se termine
par $\boldsymbol{\theta}_{n+1} = \boldsymbol{\theta}_n^{(K)}$, où
$\mathcal{D}_n^{\text{fit}}$ est le jeu de données de régression de la
$n$-ième itération extérieure. Deux boucles à nombre de pas fixé peuvent
s'écrire comme une seule, et cette réécriture montre d'où vient le réseau
cible de DQN.

### Rendre la structure imbriquée explicite

En partant de {prf:ref}`fitted-q-iteration-batch` et en remplaçant l'appel à
`fit` par $K$ pas de gradient explicites, on obtient ce qui suit.

```{prf:algorithm} Itération Q ajustée neuronale avec boucle intérieure explicite
:label: nfqi-explicit-inner

**Entrée** : processus de décision markovien $(S, A, P, R, \gamma)$, jeu de données de transitions hors ligne $\mathcal{D}$, réseau de neurones $q(s,a; \boldsymbol{\theta})$, taux d'apprentissage $\alpha$, nombre de pas d'optimisation intérieurs $K$, initialisation $\boldsymbol{\theta}_0$

**Sortie** : paramètres $\boldsymbol{\theta}$ de l'approximation de la fonction Q

1. Initialiser $\boldsymbol{\theta}_0$ aléatoirement
2. $n \leftarrow 0$  $\quad$ // **Compteur de la boucle extérieure (itération sur les valeurs)**
3. **répéter**
    1. **// Calculer les cibles de Bellman avec la fonction Q courante**
    2. $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
    3. **pour** chaque $(s,a,r,s') \in \mathcal{D}$ **faire**
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    4. **// Boucle d'optimisation intérieure : ajustement aux cibles par descente de gradient**
    5. $\boldsymbol{\theta}_n^{(0)} \leftarrow \boldsymbol{\theta}_n$ $\quad$ // Démarrage à chaud à partir de l'itération extérieure précédente
    6. $k \leftarrow 0$ $\quad$ // **Compteur de la boucle intérieure (régression)**
    7. **répéter**
        1. $\boldsymbol{\theta}_n^{(k+1)} \leftarrow \boldsymbol{\theta}_n^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n^{(k)}; \mathcal{D}_n^{\text{fit}})$
        2. $k \leftarrow k + 1$
    8. **jusqu'à** $k = K$ $\quad$ // Optimisation partielle : exactement K pas de gradient
    9. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n^{(K)}$
    10. $n \leftarrow n + 1$
4. **jusqu'à** la fin de l'entraînement
5. **retourner** $\boldsymbol{\theta}_n$
```

Les cibles de l'itération extérieure $n$ sont calculées avec
$\boldsymbol{\theta}_n$, les paramètres à la fin de l'itération extérieure
précédente, et restent fixes pendant les $K$ pas intérieurs. Les itérés
intérieurs $\boldsymbol{\theta}_n^{(k)}$ pour $k > 0$ n'entrent jamais dans
une cible. L'ajustement est démarré à chaud à partir de
$\boldsymbol{\theta}_n$, les paramètres mêmes qui ont calculé les cibles, et
il s'arrête après $K$ pas plutôt qu'à convergence : ce sont les deux écarts
par rapport à l'apprentissage supervisé qu'a introduits l'itération Q ajustée
neuronale, et tout ce qui suit les conserve.

### Aplatir en une seule boucle

Remplaçons la paire de compteurs $(n, k)$ par un seul compteur de pas
$t = nK + k$. Les paramètres qui calculent les cibles, $\boldsymbol{\theta}_n$,
sont conservés dans une copie séparée qui ne change que lorsque $t$ franchit
un multiple de $K$, la frontière entre deux itérations extérieures. Cette
copie est le **réseau cible** $\boldsymbol{\theta}_{\text{target}}$. Les
paramètres que mettent à jour les pas de gradient, $\boldsymbol{\theta}_t$,
correspondent aux itérés intérieurs $\boldsymbol{\theta}_n^{(k)}$, et on les
appelle le **réseau en ligne**, en ligne au sens où il est continuellement mis
à jour, et non au sens de la façon dont les données ont été recueillies. Rien
d'autre ne change.

```{prf:algorithm} Itération Q ajustée neuronale aplatie
:label: nfqi-flattened

**Entrée** : processus de décision markovien $(S, A, P, R, \gamma)$, jeu de données de transitions hors ligne $\mathcal{D}$, réseau de neurones $q(s,a; \boldsymbol{\theta})$, taux d'apprentissage $\alpha$, période de mise à jour de la cible $K$, initialisation $\boldsymbol{\theta}_0$

**Sortie** : paramètres $\boldsymbol{\theta}$ de l'approximation de la fonction Q

1. Initialiser $\boldsymbol{\theta}_0$ aléatoirement
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. $t \leftarrow 0$ $\quad$ // **Compteur unique de la boucle aplatie**
4. **tant que** l'entraînement se poursuit **faire**
    1. **// Calculer les cibles avec le réseau cible fixe**
    2. $\mathcal{D}_t^{\text{fit}} \leftarrow \emptyset$
    3. **pour** chaque $(s,a,r,s') \in \mathcal{D}$ **faire**
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{\text{target}})$
        2. $\mathcal{D}_t^{\text{fit}} \leftarrow \mathcal{D}_t^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    4. **// Pas de gradient sur le réseau en ligne**
    5. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t; \mathcal{D}_t^{\text{fit}})$
    6. **// Mise à jour périodique du réseau cible (marque la frontière d'une itération extérieure)**
    7. **si** $t \bmod K = 0$ **alors**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    8. $t \leftarrow t + 1$
5. **retourner** $\boldsymbol{\theta}_t$
```

Au pas $t$, l'itération extérieure est $n = \lfloor t/K \rfloor$ et le pas
intérieur est $k = t \bmod K$. Le réseau cible vaut $\boldsymbol{\theta}_n$
pour $t$ allant de $nK$ à $(n+1)K - 1$ et devient $\boldsymbol{\theta}_{n+1}$
en $t = (n+1)K$, et $\boldsymbol{\theta}_t$ est $\boldsymbol{\theta}_n^{(k)}$.
Avec $K = 10$, le pas intérieur $k = 7$ de l'itération extérieure $n = 3$ est
le pas global $t = 37$. Le jeu de données est toujours le $\mathcal{D}$ fixe;
seule l'indexation a changé. Dans cette lecture, un réseau cible est la
frontière d'une itération extérieure de l'itération Q ajustée, gardée en
mémoire. Lorsque DQN est présenté avec un réseau cible mis à jour tous les
$K$ pas, c'est cette itération Q ajustée aplatie munie d'un tampon évolutif,
que la section suivante ajoute.

### Mises à jour lisses de la cible par moyenne mobile exponentielle

La copie périodique peut être remplacée par un réseau cible qui suit un peu
le réseau en ligne à chaque pas,

$$
\boldsymbol{\theta}_{\text{target}} \leftarrow \tau\,\boldsymbol{\theta}_{t+1} + (1-\tau)\,\boldsymbol{\theta}_{\text{target}},
$$

une **moyenne mobile exponentielle** (aussi appelée moyenne de Polyak) de
taux $\tau \in (0, 1)$. Pour un petit $\tau$, typiquement entre $0.001$ et
$0.01$, la cible est en retard d'environ $1/\tau$ pas sur le réseau en ligne,
et les cibles dérivent continûment au lieu de sauter tous les $K$ pas. Ce
schéma est devenu courant avec DDPG {cite}`lillicrap2015continuous` et il est
utilisé par TD3 {cite}`fujimoto2018addressing` et SAC
{cite}`haarnoja2018soft`.

```{prf:algorithm} Itération Q ajustée neuronale aplatie avec mises à jour de la cible par moyenne mobile exponentielle
:label: nfqi-flattened-ema

**Entrée** : processus de décision markovien $(S, A, P, R, \gamma)$, jeu de données de transitions hors ligne $\mathcal{D}$, réseau de neurones $q(s,a; \boldsymbol{\theta})$, taux d'apprentissage $\alpha$, taux de moyenne mobile exponentielle $\tau \in (0, 1)$, initialisation $\boldsymbol{\theta}_0$

**Sortie** : paramètres $\boldsymbol{\theta}$ de l'approximation de la fonction Q

1. Initialiser $\boldsymbol{\theta}_0$ aléatoirement
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. $t \leftarrow 0$ $\quad$ // **Compteur unique de la boucle aplatie**
4. **tant que** l'entraînement se poursuit **faire**
    1. **// Calculer les cibles avec le réseau cible fixe**
    2. $\mathcal{D}_t^{\text{fit}} \leftarrow \emptyset$
    3. **pour** chaque $(s,a,r,s') \in \mathcal{D}$ **faire**
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{\text{target}})$
        2. $\mathcal{D}_t^{\text{fit}} \leftarrow \mathcal{D}_t^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    4. **// Pas de gradient sur le réseau en ligne**
    5. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t; \mathcal{D}_t^{\text{fit}})$
    6. **// MODIFIÉ : mise à jour lisse par moyenne mobile exponentielle à chaque pas**
    7. $\boldsymbol{\theta}_{\text{target}} \leftarrow \tau\boldsymbol{\theta}_{t+1} + (1-\tau)\boldsymbol{\theta}_{\text{target}}$
    8. $t \leftarrow t + 1$
5. **retourner** $\boldsymbol{\theta}_t$
```

(sec-fqi-online)=
## DQN et données en ligne

L'algorithme aplati échantillonnait dans un jeu de données fixe. Laissons
plutôt le jeu de données changer : un **tampon de rejeu** $\mathcal{B}_t$ de
capacité fixe $B$ reçoit une nouvelle transition à chaque pas d'interaction
et abandonne la plus ancienne lorsqu'il est plein, de sorte que
$\mathcal{B}_t = \{\tau_{t-B+1}, \ldots, \tau_t\}$ et que sa loi empirique
$\hat{P}_{\mathcal{B}_t}$ glisse vers l'avant dans le temps. Chaque pas de
gradient utilise un mini-lot de $b$ transitions tirées uniformément dans le
tampon courant. Le réseau en ligne acquiert un second rôle : en plus d'être
les paramètres en cours d'entraînement, il est la politique qui recueille les
nouvelles transitions, par une sélection d'action $\varepsilon$-gloutonne.
« En ligne » prend maintenant les deux sens à la fois : des paramètres mis à
jour continuellement et des données recueillies pendant l'apprentissage.

Le réseau Q profond (DQN, *Deep Q-Network*) {cite}`mnih2013atari` est cet
algorithme avec un réseau convolutif et des réglages modérés : un tampon
d'environ $10^6$ transitions, des mini-lots d'environ 32 et un réseau cible
copié tous les $K \approx 10^4$ pas. C'est l'itération Q ajustée aplatie avec
un tampon évolutif, et chacune de ses composantes a sa place dans la vue
imbriquée : le réseau cible marque les itérations extérieures, le pas de
gradient sur mini-lot est la boucle intérieure, et le tampon est le jeu de
données, qui change maintenant lentement sous la politique que définit le
réseau en ligne.

```{prf:algorithm} Réseau Q profond (DQN)
:label: dqn

**Entrée** : processus de décision markovien $(S, A, P, R, \gamma)$, réseau de neurones $q(s,a; \boldsymbol{\theta})$, taux d'apprentissage $\alpha$, période de mise à jour de la cible $K$, capacité du tampon de rejeu $B$, taille de mini-lot $b$, taux d'exploration $\varepsilon$

**Sortie** : paramètres $\boldsymbol{\theta}$ de l'approximation de la fonction Q

1. Initialiser $\boldsymbol{\theta}_0$ aléatoirement
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. Initialiser le tampon de rejeu $\mathcal{B}$ avec la capacité $B$
4. $t \leftarrow 0$
5. **tant que** l'entraînement se poursuit **faire**
    1. Observer l'état courant $s$
    2. **// Utiliser le réseau en ligne pour recueillir des données**
    3. Choisir l'action : $a \leftarrow \begin{cases} \arg\max_{a'} q(s,a';\boldsymbol{\theta}_t) & \text{avec probabilité } 1-\varepsilon \\ \text{action aléatoire} & \text{avec probabilité } \varepsilon \end{cases}$
    4. Exécuter $a$, observer la récompense $r$ et l'état suivant $s'$
    5. Stocker $(s,a,r,s')$ dans $\mathcal{B}$, en remplaçant la plus ancienne transition s'il est plein
    6. Tirer de $\mathcal{B}$ un mini-lot de $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$
    7. Pour chaque transition tirée $(s_i,a_i,r_i,s_i')$ :
        1. $y_i \leftarrow r_i + \gamma \color{blue}{\max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{\text{target}})}$ (sélection ET évaluation avec le réseau cible)
    8. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b (q(s_i,a_i;\boldsymbol{\theta}_t) - y_i)^2$
    9. **si** $t \bmod K = 0$ **alors**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    10. $t \leftarrow t + 1$
6. **retourner** $\boldsymbol{\theta}_t$
```

La triade mortelle du [chapitre sur les méthodes de valeur ajustées](approximate-bellman-equations.md)
est toute présente : un réseau de neurones, des cibles par autoamorçage et un
tampon dont les transitions ont été recueillies par des politiques
antérieures. Ce sont les choix de conception qui la tiennent en échec. Les
cibles sont maintenues fixes pendant $K$ pas par le réseau cible, chaque
ajustement est démarré à chaud, et le grand tampon fait la moyenne des
données sur de nombreuses politiques, de sorte que la loi des données
qu'utilise chaque ajustement change lentement même si la politique change à
chaque pas.

(sec-fqi-double-dqn)=
### Double DQN

La cible de DQN, $\max_{a'} q(s', a'; \boldsymbol{\theta}_{\text{target}})$,
utilise deux fois le réseau cible, pour choisir la meilleure action suivante
et pour l'évaluer, et le [chapitre sur les mises à jour de Bellman échantillonnées](monte-carlo-bellman-estimation.md#sec-mc-overestimation)
a montré qu'un tel maximum est biaisé vers le haut par la réutilisation du
bruit gagnant. Double DQN {cite}`van2016deep` répartit les deux rôles entre
les deux réseaux que DQN maintient déjà : le réseau en ligne sélectionne,
$a^*_i = \arg\max_{a'} q(s'_i, a'; \boldsymbol{\theta}_t)$, et le réseau cible
évalue, $y_i = r_i + \gamma\, q(s'_i, a^*_i; \boldsymbol{\theta}_{\text{target}})$.
Le tampon et l'ajustement sont inchangés; seule la fonction cible diffère.

```{prf:algorithm} Double réseau Q profond (Double DQN)
:label: double-dqn

**Entrée** : processus de décision markovien $(S, A, P, R, \gamma)$, réseau de neurones $q(s,a; \boldsymbol{\theta})$, taux d'apprentissage $\alpha$, période de mise à jour de la cible $K$, capacité du tampon de rejeu $B$, taille de mini-lot $b$, taux d'exploration $\varepsilon$

**Sortie** : paramètres $\boldsymbol{\theta}$ de l'approximation de la fonction Q

1. Initialiser $\boldsymbol{\theta}_0$ aléatoirement
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. Initialiser le tampon de rejeu $\mathcal{B}$ avec la capacité $B$
4. $t \leftarrow 0$
5. **tant que** l'entraînement se poursuit **faire**
    1. Observer l'état courant $s$
    2. **// Utiliser le réseau en ligne pour recueillir des données**
    3. Choisir l'action : $a \leftarrow \begin{cases} \arg\max_{a'} q(s,a';\boldsymbol{\theta}_t) & \text{avec probabilité } 1-\varepsilon \\ \text{action aléatoire} & \text{avec probabilité } \varepsilon \end{cases}$
    4. Exécuter $a$, observer la récompense $r$ et l'état suivant $s'$
    5. Stocker $(s,a,r,s')$ dans $\mathcal{B}$, en remplaçant la plus ancienne transition s'il est plein
    6. Tirer de $\mathcal{B}$ un mini-lot de $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$
    7. Pour chaque transition tirée $(s_i,a_i,r_i,s_i')$ :
        1. $a^*_i \leftarrow \color{red}{\arg\max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_t)}$ (sélection de l'action avec le réseau en ligne)
        2. $y_i \leftarrow r_i + \gamma \color{green}{q(s_i',a^*_i; \boldsymbol{\theta}_{\text{target}})}$ (évaluation de l'action avec le réseau cible)
    8. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b (q(s_i,a_i;\boldsymbol{\theta}_t) - y_i)^2$
    9. **si** $t \bmod K = 0$ **alors**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    10. $t \leftarrow t + 1$
6. **retourner** $\boldsymbol{\theta}_t$
```

Les deux réseaux ne sont pas des estimations indépendantes au sens de
l'argument du chapitre sur Monte-Carlo, puisque le réseau cible est une copie
retardée du réseau en ligne. Leurs erreurs sont néanmoins moins corrélées que
les erreurs d'un seul réseau utilisé deux fois, parce qu'ils ont été ajustés à
des cibles différentes à des moments différents, et la réduction de la
surestimation qu'on obtient ainsi est substantielle en pratique.

### Le Q-learning : le cas limite

DQN utilise un tampon d'environ $10^6$ transitions, un réseau cible mis à jour
tous les $10^4$ pas et des mini-lots de 32. À l'extrême opposé, avec un tampon
d'une transition, un réseau cible mis à jour à chaque pas et un mini-lot de
un, l'algorithme est le Q-learning {cite}`watkins1989learning`. Chaque
transition est utilisée une fois puis rejetée, $\hat{P}_{\mathcal{B}_t}$ est
une masse ponctuelle en la transition courante, et les paramètres qui
calculent la cible sont ceux qui viennent d'être mis à jour.

Le Q-learning est une méthode d'**approximation stochastique**. La procédure
de Robbins–Monro résout une équation $F(\boldsymbol{\theta}) = 0$ dans
laquelle $F(\boldsymbol{\theta}) = \mathbb{E}[f(\boldsymbol{\theta}; Z)]$ est
une espérance qui n'est jamais calculée, au moyen de la mise à jour
$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t f(\boldsymbol{\theta}_t; Z_t)$
sur un tirage bruité $Z_t$ à la fois; avec des pas tels que $\alpha_t \to 0$
et $\sum_t \alpha_t = \infty$, les itérés convergent vers une solution. Pour
le Q-learning, l'équation est l'équation de Bellman sous forme de résidu,
$\Bellman q - q = 0$, et le tirage est une transition. Le résidu en une
transition $(s,a,r,s')$ est l'**erreur de différence temporelle**

$$
R(s,a,r,s'; \boldsymbol{\theta}) = r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}) - q(s,a; \boldsymbol{\theta}),
$$

dont l'espérance sur les transitions est le résidu de Bellman en $(s,a)$,
moyenné selon la loi de la politique de comportement. Chaque transition en
est un échantillon bruité, et la mise à jour fait un pas de gradient sur le
carré de l'erreur de différence temporelle avec la cible maintenue fixe :
c'est le semi-gradient du chapitre sur les équations de Bellman approchées.

```{prf:algorithm} Q-learning
:label: q-learning

**Entrée** : processus de décision markovien $(S, A, P, R, \gamma)$, approximateur de fonctions $q(s,a; \boldsymbol{\theta})$, taux d'apprentissage $\alpha$, taux d'exploration $\varepsilon$

**Sortie** : paramètres $\boldsymbol{\theta}$ de l'approximation de la fonction Q

1. Initialiser $\boldsymbol{\theta}_0$
2. $t \leftarrow 0$
3. **tant que** l'entraînement se poursuit **faire**
    1. Observer l'état courant $s$
    2. Choisir l'action : $a \leftarrow \begin{cases} \arg\max_{a' \in A} q(s,a';\boldsymbol{\theta}_t) & \text{avec probabilité } 1-\varepsilon \\ \text{action aléatoire} & \text{avec probabilité } \varepsilon \end{cases}$
    3. Exécuter $a$, observer la récompense $r$ et l'état suivant $s'$
    4. Calculer la cible de différence temporelle : $y \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_t)$
    5. Mettre à jour : $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} (q(s,a;\boldsymbol{\theta}_t) - y)^2$
    6. $t \leftarrow t + 1$
4. **retourner** $\boldsymbol{\theta}_t$
```

Le gradient de la ligne 5 dépend de la classe de fonctions. Avec des
caractéristiques indicatrices
$\boldsymbol{\phi}(s,a) = \boldsymbol{e}_{(s,a)}$, la mise à jour est la règle
tabulaire $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max_{a'} Q(s',a') - Q(s,a))$,
qui converge sous les conditions de l'approximation stochastique. Avec des
caractéristiques linéaires
$q(s,a;\boldsymbol\theta) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(s,a)$,
le gradient est $\boldsymbol{\phi}(s,a)$, et la convergence n'est pas
garantie : les moindres carrés ne sont pas un moyenneur, et l'opérateur max
empêche d'appliquer l'argument valable sur la politique suivie. Avec un
réseau de neurones, le gradient provient de la rétropropagation et il n'y a
plus aucune garantie.

### Analyse de convergence par la méthode de l'EDO

Le Q-learning tabulaire converge avec des pas décroissants et une exploration
suffisante {cite}`watkins1992qlearning,jaakkola1994convergence,tsitsiklis1994asynchronous`.
La technique de preuve usuelle est la **méthode de l'équation différentielle
ordinaire (EDO)** {cite}`kushner2003stochastic,borkar2000ode`, qui étudie une
approximation stochastique au moyen d'un système dynamique déterministe
associé. L'argument comporte deux étapes. À mesure que les pas diminuent, la
récurrence stochastique
$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha_t h(\boldsymbol{\theta}_t, Z_t)$
suit l'EDO $\dot{\boldsymbol{\theta}} = \bar{h}(\boldsymbol{\theta})$, où
$\bar{h}(\boldsymbol{\theta}) = \mathbb{E}[h(\boldsymbol{\theta}, Z)]$ est la
direction de mise à jour moyenne. On montre ensuite que l'EDO converge, au
moyen d'une fonction de Lyapounov, et l'algorithme stochastique converge vers
le même point.

Pour le Q-learning tabulaire, la mise à jour au couple visité $(s,a)$ est
$Q_{t+1}(s,a) = Q_t(s,a) + \alpha_t [r + \gamma \max_{a'} Q_t(s',a') - Q_t(s,a)]$,
et son espérance sur la transition, sachant que $(s,a)$ est visité, est le
résidu de Bellman en $(s,a)$. L'EDO est

$$
\frac{dQ(s,a)}{dt} = \sum_{s'} p(s'|s,a) \big[r(s,a,s') + \gamma \max_{a'} Q(s',a') \big] - Q(s,a),
$$

dont les équilibres sont les solutions de l'équation d'optimalité de Bellman,
c'est-à-dire $Q^*$ seule. La distance $\|Q - Q^*\|_\infty$ décroît le long de
ses trajectoires parce que l'opérateur de Bellman est une contraction : c'est
la fonction de Lyapounov. L'exploration intervient par l'exigence que chaque
couple soit visité infiniment souvent; sous ergodicité, la fréquence de mise à
jour de chaque $(s,a)$ converge vers la loi stationnaire de la politique de
comportement, que l'EDO absorbe sous forme d'une échelle de temps propre à
chaque coordonnée. Avec des caractéristiques linéaires, la dérive de l'EDO
n'est plus le résidu de Bellman en chaque coordonnée, la projection
intervient, et le [chapitre sur les équations de Bellman
approchées](approximate-bellman-equations.md) a montré ce qui peut alors mal
tourner.

(sec-fqi-design-space)=
## L'espace de conception de l'itération Q ajustée

Tous les algorithmes ci-dessus ont la même structure à deux niveaux, avec
trois ingrédients à tout instant $t$ :

1. Un **tampon** $\mathcal{B}_t$ de transitions, avec sa loi empirique
   $\hat{P}_{\mathcal{B}_t}$. C'est un jeu de données fixe $\mathcal{D}$ dans
   les algorithmes par lots, un tampon circulaire de capacité $B$ dans DQN, et
   l'unique transition courante dans le Q-learning.
2. Une **fonction cible** $g(s,a,r,s'; \boldsymbol{\theta})$ qui calcule une
   cible de régression à partir d'une transition et des paramètres de la
   fonction Q courante : $r + \gamma \max_{a'} q(s',a';\boldsymbol\theta)$
   pour le maximum strict, le log-sum-exp du
   [chapitre sur la régularisation](regularized-dp.md) pour le maximum lissé,
   et la sélection et l'évaluation découplées de Double DQN. Contrairement à
   l'opérateur de Bellman, qui agit sur des fonctions, $g$ agit sur des
   transitions.
3. Un **ajustement** : une perte $\ell$, une initialisation
   $\boldsymbol{\theta}_{\text{init}}$ et un budget d'optimisation $K$.

Faire passer le tampon par la fonction cible transforme la loi sur les
transitions en une loi sur des paires de régression,
$\hat{P}_t^{\text{fit}} = (\mathrm{id}, g(\cdot;\boldsymbol\theta_t))_\#\hat{P}_{\mathcal{B}_t}$,
et l'ajustement minimise le risque empirique
$\mathbb{E}_{((s,a),y) \sim \hat{P}_t^{\text{fit}}}[\ell(q(s,a;\boldsymbol\theta), y)]$.
La loi du tampon ne change que lorsque des données sont recueillies; la loi
de régression change chaque fois que les cibles sont recalculées. Dans
l'abstrait, toute la famille s'écrit

$$
\begin{aligned}
&\textbf{répéter } n = 0, 1, 2, \ldots \\
&\quad \text{Tirer des transitions selon } \hat{P}_{\mathcal{B}_n} \text{ et construire les cibles par } g(\cdot; \boldsymbol{\theta}_n) \\
&\quad \boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\hat{P}_n^{\text{fit}}, \boldsymbol{\theta}_{\text{init}}, K) \\
&\textbf{jusqu'à } \text{convergence.}
\end{aligned}
$$

La fréquence de réutilisation des données se mesure par le **taux de
rejeu**, le nombre de pas d'optimisation par transition recueillie :

$$
\text{Taux de rejeu} = \begin{cases}
K \cdot N_{\text{époques}} \cdot N / b & \text{par lots (taille de lot } b \text{ parmi } N \text{ transitions)} \\
b & \text{DQN (un pas par transition, taille de lot } b \text{)} \\
1 & \text{Q-learning.}
\end{cases}
$$

Un grand taux de rejeu réutilise de nombreuses fois la même loi empirique, ce
qui est l'approximation par moyenne empirique; un taux de rejeu de un utilise
chaque transition une seule fois, ce qui est l'approximation stochastique.
Davantage de réutilisation réduit la variance des estimations sous
$\hat{P}_{\mathcal{B}_t}$ et accroît le risque d'ajuster les particularités
d'un tampon qui reflète d'anciennes politiques ou une couverture étroite.

```{prf:remark} Deux notions d'autoamorçage
:class: dropdown

Deux « autoamorçages » différents se rencontrent dans l'itération Q ajustée. Échantillonner avec remise selon $\hat{P}_{\mathcal{B}_t}$ approche des espérances sous le $P$ inconnu par des espérances sous la loi empirique : c'est le bootstrap des statisticiens; l'entraînement par mini-lots est ce rééchantillonnage, et le taux de rejeu est le nombre d'échantillons bootstrap par transition. Calculer $y = r + \gamma \max_{a'} q(s',a';\boldsymbol{\theta})$ remplace la valeur de continuation inconnue par l'estimation courante : c'est l'autoamorçage de l'apprentissage par différence temporelle et la source du biais de maximisation du chapitre précédent. Le tampon encode le premier, la fonction cible le second, et les corrections du biais (double estimation, corrections apprises, pertes de Gumbel) n'agissent que sur le second.
```

Le tableau situe dans cet espace les algorithmes de ce chapitre et du
suivant.

| **Algorithme** | **Approximateur** | **Cible** | **Budget d'ajustement** | **Initialisation** | **Données** | **Correction du biais** |
|:--------------|:-----------------|:-----------|:---------------|:-------------------|:---------|:-------------|
| FQI {cite}`ernst2005tree` | Arbres extrêmement aléatoires | Maximum strict | Complet | À froid | Lot fixe | Aucune |
| NFQI {cite}`riedmiller2005neural` | Réseau de neurones | Maximum strict | Complet | À chaud | Lot fixe | Aucune |
| Q-learning {cite}`watkins1989learning` | Quelconque | Maximum strict | $K=1$ | À chaud | Une transition | Aucune |
| DQN {cite}`mnih2013atari` | Réseau profond | Maximum strict | $K=1$ | À chaud | Tampon de rejeu | Aucune |
| Double DQN {cite}`van2016deep` | Réseau profond | Maximum découplé | $K=1$ | À chaud | Tampon de rejeu | Double estimation |
| Soft Q {cite}`haarnoja2017reinforcement` | Réseau de neurones | Log-sum-exp | $K$ pas | À chaud | Tampon de rejeu | Aucune |

Les méthodes à actions continues NFQCA, DDPG et SAC ajoutent un axe de plus,
l'acteur qui remplace le maximum dans la cible, et relèvent du
[chapitre suivant](amortized-action-optimization.md).

## Pertes de régression et modèles de bruit

À un instant $t$ fixé, l'ajustement est un problème d'estimation statistique :
étant donné des échantillons tirés selon $\hat{P}_t^{\text{fit}}$, choisir
$\boldsymbol{\theta}$ de façon à minimiser
$\mathbb{E}_{((s,a),y)\sim \hat{P}_t^{\text{fit}}}[\ell(q(s,a;\boldsymbol{\theta}), y)]$.
La perte $\ell$ est un modèle de bruit pour les cibles. L'erreur quadratique
est la vraisemblance d'un bruit gaussien, l'erreur absolue celle d'un bruit
de Laplace, la régression de Gumbel celle d'un bruit de valeurs extrêmes, et
une perte de classification sur des intervalles de valeurs ne fait aucune
hypothèse paramétrique. Les cibles de Bellman sont bruitées, biaisées par le
maximum et obtenues par autoamorçage; le choix n'est donc pas neutre, et deux
solutions de rechange à l'erreur quadratique ont donné de bons résultats en
pratique.

### Régression de Gumbel

La théorie des valeurs extrêmes dit que le maximum de plusieurs erreurs
gaussiennes a des queues distribuées selon une loi de Gumbel.
{cite:t}`garg2023extreme` ont pris cette loi au sérieux dans Extreme
Q-Learning (XQL) : au lieu de modéliser l'erreur d'une cible comme gaussienne,

$$
y_i = q^*(s_i, a_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2),
$$

ils la modélisent comme une variable de Gumbel,

$$
y_i = q^*(s_i, a_i) + \varepsilon_i, \quad \varepsilon_i \sim \text{Gumbel}(0, \beta),
$$

une loi asymétrique vers la droite, parce qu'une cible construite à partir
d'un maximum tombe parfois bien au-dessus de la valeur qu'elle estime. Avec
$u_i = q(s_i, a_i; \boldsymbol\theta) - y_i$, l'opposé de la
log-vraisemblance est, aux constantes près, la perte de régression de Gumbel

$$
\mathcal{L}_{\text{Gumbel}}(\boldsymbol{\theta}) = \sum_i \left[\exp\left(\frac{u_i}{\beta}\right) - \frac{u_i}{\beta} - 1\right],
$$

avec une température $\beta$ qui fixe la lourdeur de la queue. La perte est
nulle en $u_i = 0$ et convexe, et son gradient par rapport à $q$ est

$$
\frac{\partial \mathcal{L}_{\text{Gumbel}}}{\partial q} = \frac{1}{\beta}\left[\exp\left(\frac{q - y}{\beta}\right) - 1\right].
$$

Lorsque $q < y$, l'exponentielle est petite et le gradient tend vers
$-1/\beta$, de sorte qu'une cible bien au-dessus de la prédiction ne la tire
vers le haut que linéairement; lorsque $q > y$, le gradient croît
exponentiellement avec l'erreur. La perte pénalise la surestimation bien plus
que la sous-estimation, ce qui pousse l'ajustement vers des valeurs prudentes
lorsque les cibles sont biaisées vers le haut par le maximum. L'erreur
quadratique est la bonne vraisemblance pour des moyennes, par le théorème
central limite; la régression de Gumbel est la bonne vraisemblance pour des
maxima d'un bruit à queues légères, par la théorie des valeurs extrêmes.

```{code-cell} python
:tags: [hide-input]
:label: fig-gumbel-loss
:caption: La régression de Gumbel comparée à l'erreur quadratique, en fonction de l'erreur $q - y$ d'une prédiction. (a) La perte $\exp(u/\beta) - u/\beta - 1$ croît exponentiellement pour les surestimations ($q > y$) et linéairement pour les sous-estimations; une température $\beta$ plus petite accentue l'asymétrie. (b) Sa dérivée par rapport à $q$, qui fixe la force avec laquelle l'ajustement réagit : forte pente pour les surestimations, plateau près de $-1/\beta$ pour les sous-estimations, alors que la dérivée de l'erreur quadratique (tirets) est symétrique.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))
from book_figure_theme import INK, MUTED, PALETTE, figure_style, show

error = np.linspace(-3, 3, 500)                       # q - y
betas = {0.5: PALETTE["vermilion"], 1.0: PALETTE["orange"], 2.0: PALETTE["blue"]}

with figure_style():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for beta, color in betas.items():
        ax.plot(error, np.exp(error / beta) - error / beta - 1, color=color)
        bx.plot(error, (np.exp(error / beta) - 1) / beta, color=color)
    ax.plot(error, error ** 2, color=MUTED, lw=1, ls="--")
    bx.plot(error, 2 * error, color=MUTED, lw=1, ls="--")
    for axis in (ax, bx):
        axis.axvline(0, color=MUTED, lw=.6)
        axis.set_xlabel("error $q - y$")
        axis.set_xlim(-3, 3)
    ax.set_ylim(0, 10)
    bx.set_ylim(-5, 10)
    ax.text(1.0, 9.2, r"$\beta = 0.5$", color=betas[0.5], fontsize=8.5, ha="right")
    ax.text(1.75, 8.0, r"$\beta = 1$", color=betas[1.0], fontsize=8.5, ha="right")
    ax.text(2.95, 3.2, r"$\beta = 2$", color=betas[2.0], fontsize=8.5, ha="right")
    ax.text(-2.0, 7.4, "squared\nerror", color=MUTED, fontsize=8.5)
    bx.text(-2.9, -3.8, "squared\nerror", color=MUTED, fontsize=8.5)
    ax.set_ylabel("loss")
    bx.set_ylabel(r"derivative with respect to $q$")
    ax.set_title("(a) Gumbel loss")
    bx.set_title("(b) Its gradient")
    fig.tight_layout(w_pad=2)
show(fig)
```

{numref}`fig-gumbel-loss` trace la perte et son gradient pour trois
températures. La perte gère d'elle-même l'asymétrie du bruit, sans variance à
estimer ni poids à calculer, et XQL a amélioré à la fois des méthodes fondées
sur la valeur et des méthodes acteur-critique, en particulier en
apprentissage par renforcement hors ligne, où le biais du maximum s'accumule
au fil des itérations sans nouvelles données pour le corriger.

```{prf:remark} XQL et soft Q-learning : fonction cible contre perte
:class: dropdown

XQL et le soft Q-learning font tous deux intervenir la loi de Gumbel, à des endroits différents de l'espace de conception. Le soft Q-learning modifie la fonction cible en utilisant l'opérateur de Bellman lissé du [chapitre sur la régularisation](regularized-dp.md),

$$
g^{\text{soft}}(s,a,r,s'; \boldsymbol{\theta}) = r + \gamma \frac{1}{\beta}\log\sum_{a'} \exp(\beta q(s',a'; \boldsymbol{\theta})),
$$

avec la perte quadratique inchangée; la loi de Gumbel y intervient par la construction Gumbel-max du log-sum-exp, qui sert à rendre la politique optimale stochastique et non à modéliser le bruit des cibles. XQL conserve le maximum strict dans la cible et remplace la perte par $\ell_{\text{Gumbel}}(q,y) = \exp((q-y)/\beta) - (q-y)/\beta - 1$, où la loi de Gumbel modélise le bruit des erreurs de différence temporelle. Le log-sum-exp réduit le biais de maximisation en faisant la moyenne sur les actions, une propriété de la cible; la perte de Gumbel pénalise la surestimation, une propriété de la perte. Les deux peuvent être combinés.
```

### Q-learning par classification

Rien ne change en amont : les transitions sont tirées selon
$\hat{P}_{\mathcal{B}_t}$ et passent par la même fonction cible. Ce qui
change, c'est la représentation de la valeur et la perte. Choisissons une
grille $z_1 < z_2 < \cdots < z_K$ qui couvre les rendements plausibles. Le
réseau produit des logits $\ell_{\boldsymbol{\theta}}(s,a) \in \mathbb{R}^K$,
convertis en probabilités et en une valeur scalaire par

$$
p_{\boldsymbol{\theta}}(k \mid s,a) = \frac{\exp(\ell_{\boldsymbol{\theta}}(s,a)_k)}{\sum_{j=1}^K \exp(\ell_{\boldsymbol{\theta}}(s,a)_j)}, \quad q(s,a; \boldsymbol{\theta}) = \sum_{k=1}^K z_k \, p_{\boldsymbol{\theta}}(k \mid s,a).
$$

Chaque cible scalaire $y_i$ devient une loi sur les intervalles par le
**codage à deux intervalles** (*two-hot*). Si $y_i$ tombe entre $z_j$ et
$z_{j+1}$,

$$
q_j(y_i) = \frac{z_{j+1} - y_i}{z_{j+1} - z_j}, \quad q_{j+1}(y_i) = \frac{y_i - z_j}{z_{j+1} - z_j}, \quad q_k(y_i) = 0 \text{ pour } k \notin \{j, j+1\}.
$$

Ce sont des coordonnées barycentriques, et $\sum_k z_k q_k(y_i) = y_i$
retrouve exactement le scalaire : le codage est l'interpolation linéaire du
[chapitre sur l'horizon fini](finite-horizon-dp.md)
({prf:ref}`backward-recursion-interp`) appliquée à l'axe des valeurs.

```{code-cell} python
:tags: [hide-input]
:label: fig-two-hot
:caption: Codage à deux intervalles sur 11 intervalles couvrant $[-5, 5]$. (a) La cible $y = 0.7$ attribue le poids $0.7$ à l'intervalle situé en $1$ et $0.3$ à l'intervalle situé en $0$, les coordonnées barycentriques de $0.7$ entre eux, de sorte que la moyenne pondérée des intervalles redonne $0.7$. (b) Trois cibles, chacune répartie sur les deux intervalles qui l'entourent.

def two_hot(y, bins):
    """Weights q_k(y): linear interpolation of y between the two bins around it."""
    q = np.zeros(len(bins))
    y = np.clip(y, bins[0], bins[-1])
    j = min(np.searchsorted(bins, y, side="right") - 1, len(bins) - 2)
    t = (y - bins[j]) / (bins[j + 1] - bins[j])
    q[j], q[j + 1] = 1 - t, t
    return q

bins = np.linspace(-5, 5, 11)
with figure_style():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    q = two_hot(0.7, bins)
    ax.bar(bins, q, width=.55, color=PALETTE["blue"])
    ax.axvline(0.7, ymax=.88, color=PALETTE["vermilion"], lw=1.2, ls="--")
    ax.text(0.7, 1.0, "$y = 0.7$", color=PALETTE["vermilion"], fontsize=8.5, ha="center")
    for k in np.flatnonzero(q):
        side = 1 if bins[k] > 0.7 else -1
        ax.text(bins[k] + .35 * side, q[k] - .03, f"{q[k]:.1f}", ha="left" if side > 0 else "right",
                va="top", fontsize=8.5, color=INK)
    ax.text(-4.8, .55, rf"$\Sigma_k\, z_k\, q_k(y) = {bins @ q:.1f}$", fontsize=8.5, color=INK)
    ax.set_title("(a) One target")
    for y, color, shift in ((-2.3, PALETTE["green"], -.18), (0.7, PALETTE["blue"], 0), (3.8, PALETTE["purple"], .18)):
        bx.bar(bins + shift, two_hot(y, bins), width=.18, color=color)
        bx.text(y, 1.0, f"$y = {y}$", color=color, fontsize=8.5, ha="center")
    bx.set_title("(b) Three targets")
    for axis in (ax, bx):
        axis.set_xticks(bins, [f"${b:.0f}$" for b in bins])
        axis.set_xlabel("bin value $z_k$")
    ax.set_ylabel("weight $q_k(y)$")
    ax.set_ylim(0, 1.12)
    fig.tight_layout(w_pad=2)
show(fig)
```

Dans {numref}`fig-two-hot`, chaque cible place sa masse sur les deux
intervalles qui l'entourent, avec un poids qui décroît avec la distance à
chaque intervalle. La perte est l'entropie croisée

$$
\mathcal{L}_{\text{CE}}(\boldsymbol{\theta}) = -\mathbb{E}_{((s,a),y) \sim \hat{P}_t^{\text{fit}}}\left[ \sum_{k=1}^K q_k(y) \log p_{\boldsymbol{\theta}}(k \mid s, a) \right],
$$

qui projette la loi cible sur la loi prédite dans la géométrie du simplexe
plutôt que dans celle de la droite réelle. Son gradient par rapport aux logits
est $p_{\boldsymbol\theta} - q$, borné quelle que soit la taille de la cible.
C'est de là que vient sa robustesse. Chaque échantillon contribue un gradient
borné par intervalle, là où l'erreur quadratique contribue un gradient
proportionnel à l'erreur; la grille finie écrête les cibles extrêmes à ses
intervalles extrêmes, de sorte qu'une valeur aberrante ne peut pas fixer
l'échelle de la régression; et le codage à deux intervalles répartit chaque
cible sur deux intervalles, un lissage des étiquettes qui fait la moyenne des
cibles bruitées en un même $(s,a)$. Le codage est un moyenneur au sens de
{prf:ref}`gordon-averager`, une combinaison convexe qui préserve l'ordre et
les bornes, associé à un réseau qui n'en est pas un, de sorte que le
Q-learning par classification est un hybride entre une structure de cibles
monotone et un ajustement flexible. {cite:t}`farebrother2024stop` ont
constaté que DQN et CQL avec l'erreur quadratique se dégradent lorsque le
réseau Q est agrandi jusqu'à un grand ResNet, alors que la perte de
classification (HL-Gauss, qui lisse la cible avec une gaussienne au lieu du
codage à deux intervalles) conserve ses performances.

## Exercices

Les premiers exercices sont de courtes vérifications. Les autres suivent le
chapitre : d'abord les données par lots, puis les boucles et les cibles de
DQN, puis l'espace de conception. {ref}`ex-fqi-chain` est un court programme.

:::{exercise} Deux boucles
:label: ex-fqi-check-1

Que se passe-t-il dans la boucle extérieure et dans la boucle intérieure de l'itération Q ajustée?
:::

:::{solution} ex-fqi-check-1
:class: dropdown

La boucle extérieure fige la fonction Q courante et construit les cibles de Bellman. La boucle intérieure ajuste un nouvel approximateur de fonctions à ces cibles.
:::

:::{exercise} Support hors ligne
:label: ex-fqi-check-2

Pourquoi un ajustement de régression très précis reste-t-il dangereux pour les actions mal représentées dans un jeu de données hors ligne?
:::

:::{solution} ex-fqi-check-2
:class: dropdown

L'ajustement n'est pas contraint hors du support des données. La maximisation peut retenir des valeurs d'action extrapolées qui sont numériquement grandes mais qu'aucune transition du jeu de données n'appuie.
:::

:::{exercise} Cibles figées
:label: ex-fqi-check-3

Quelle instabilité un réseau cible réduit-il?
:::

:::{solution} ex-fqi-check-3
:class: dropdown

Il ralentit le mouvement des cibles de régression. Sans lui, chaque mise à jour des paramètres modifie simultanément les prédictions et les cibles ajustées, ce qui crée un objectif qui se déplace rapidement.
:::

:::{exercise} Comparaison à couverture appariée
:label: ex-fqi-check-4

Les deux tampons d'ordonnancement d'inférence contiennent le même nombre de
transitions. Pourquoi le tampon qui a le moins de couples état-action
distincts peut-il quand même atteindre une erreur d'entraînement plus faible
tout en produisant une politique moins bonne?
:::

:::{solution} ex-fqi-check-4
:class: dropdown

Les transitions répétées facilitent le problème de régression empirique dans
la région visitée, mais elles ne fournissent aucune cible de Bellman pour les
actions et les états omis. La sélection gloutonne de l'action peut donc
dépendre d'une extrapolation sans appui, même lorsque l'ajustement est précis
à l'intérieur du tampon.
:::

:::{exercise} Mettre les cibles à l'échelle
:label: ex-fqi-check-5

Dans l'exemple d'évitement de collision, pourquoi un seul événement de probabilité de collision de 2 % a-t-il modifié les manœuvres de la politique ajustée en des centaines d'autres états, et qu'a changé le fait d'ajuster un rapport plutôt que le coût brut?
:::

:::{solution} ex-fqi-check-5
:class: dropdown

Les arbres extrêmement aléatoires prédisent la moyenne des cibles d'une feuille; un seul coût trois fois plus grand que tous les autres réunis relève donc la prédiction de chaque état qui partage une feuille avec les époques de cet événement. Ajuster le coût relativement à la propre estimation du message courant transforme la cible en un facteur de correction de taille ordinaire, si bien que l'événement ne domine plus les feuilles qui le contiennent.
:::

:::{exercise} Indices d'une boucle aplatie
:label: ex-fqi-indices

DQN copie son réseau cible tous les $K = 500$ pas de gradient.

1. Au pas global $t = 12\,345$, de quelle itération extérieure $n$ et de quel
   pas intérieur $k$ de l'algorithme imbriqué s'agit-il?
2. Quels paramètres ont calculé les cibles utilisées à ce pas, dans la
   notation de {prf:ref}`nfqi-explicit-inner`?
3. Avec des copies franches tous les $K$ pas, les paramètres qui calculent une
   cible ont en moyenne $(K-1)/2$ pas d'âge. Avec une moyenne mobile
   exponentielle de taux $\tau$, ils ont en moyenne $(1-\tau)/\tau$ pas d'âge.
   Quel $\tau$ correspond à $K = 10^4$?
:::

:::{solution} ex-fqi-indices
:class: dropdown

1. $n = \lfloor 12\,345/500 \rfloor = 24$ et $k = 12\,345 - 24 \times 500 = 345$.
2. $\boldsymbol{\theta}_{24}$, les paramètres à la fin de l'itération extérieure 23, conservés dans le réseau cible depuis le pas $12\,000$.
3. $(1-\tau)/\tau = (K-1)/2$ donne $\tau = 2/(K+1) \approx 2 \times 10^{-4}$. Les valeurs typiques $\tau \in [10^{-3}, 10^{-2}]$ de DDPG et de TD3 correspondent à des copies tous les 200 à 2 000 pas.
:::

:::{exercise} Une cible de Double DQN
:label: ex-fqi-double-target

Une transition a la récompense $r = 0.5$ et le successeur $s'$, et
$\gamma = 0.9$. En $s'$, le réseau en ligne prédit $(1.0, 1.3, 0.9)$ pour
trois actions, et le réseau cible prédit $(1.2, 1.0, 1.1)$.

1. Calculer la cible de DQN et la cible de Double DQN.
2. Laquelle des deux utilise la même estimation pour sélectionner et pour
   évaluer?
3. Supposons que les vraies valeurs en $s'$ soient $(1.0, 1.0, 1.0)$. Quelle
   cible est la plus proche de la vérité, et quel biais de
   {ref}`sec-mc-overestimation` l'autre présente-t-elle?
:::

:::{solution} ex-fqi-double-target
:class: dropdown

1. DQN : $0.5 + 0.9 \times \max(1.2, 1.0, 1.1) = 1.58$. Double DQN : le réseau en ligne sélectionne la deuxième action, que le réseau cible évalue à $1.0$, d'où $0.5 + 0.9 \times 1.0 = 1.40$.
2. DQN : le réseau cible sélectionne la première action et l'évalue aussi.
3. La vraie cible est $0.5 + 0.9 = 1.40$. Double DQN l'atteint, et DQN surestime de $0.18$ parce qu'il évalue l'action que le bruit de sa propre estimation a fait paraître la meilleure : c'est le biais d'évaluation.
:::

:::{exercise} La triade mortelle, algorithme par algorithme
:label: ex-fqi-triad

Pour chaque algorithme, dire lesquels des trois éléments de la triade
mortelle il présente (une approximation de fonctions qui n'est pas un
moyenneur, des cibles par autoamorçage, des données issues d'une politique
autre que celle qu'on évalue), et ce qui, le cas échéant, garantit sa
convergence.

1. L'itération Q ajustée à base d'arbres sur un lot fixe, avec des partitions figées après quelques itérations.
2. L'itération Q ajustée neuronale sur un lot fixe.
3. DQN.
4. Le Q-learning tabulaire avec une exploration $\varepsilon$-gloutonne.
5. L'évaluation de politique ajustée avec des caractéristiques linéaires sur des données recueillies par la politique évaluée.
:::

:::{solution} ex-fqi-triad
:class: dropdown

1. L'autoamorçage et des données hors politique, mais l'ajustement est un moyenneur une fois les partitions figées, de sorte que {prf:ref}`santos-vigo-aguiar-stability` s'applique à l'opérateur de Bellman empirique.
2. Les trois. Aucune garantie; les démarrages à chaud et les ajustements menés jusqu'à convergence sur des cibles fixes limitent les dégâts.
3. Les trois, avec un tampon qui se déplace. Aucune garantie; le réseau cible, le démarrage à chaud et le grand tampon sont les mesures d'atténuation.
4. L'autoamorçage et des données hors politique, mais une table est un moyenneur au sens trivial. La convergence découle de la méthode de l'EDO avec des pas décroissants et des visites de chaque couple.
5. Des caractéristiques linéaires et l'autoamorçage, mais des données issues de la politique évaluée, de sorte que l'opérateur projeté est contractant pour la $\xi$-norme ({ref}`sec-abe-convergence`).
:::

:::{exercise} La couverture sur une chaîne
:label: ex-fqi-chain

Construire une chaîne de dix états $0, \ldots, 9$ avec deux actions. L'action
« droite » déplace d'un état vers la droite avec probabilité $0.9$ et reste sur
place avec probabilité $0.1$; « gauche » déplace d'un état vers la gauche de la
même façon. La récompense vaut $1$ lorsqu'on entre dans l'état 9 ou qu'on y
reste, et $\gamma = 0.9$.

1. Enregistrer 5 000 transitions à partir d'états uniformément aléatoires
   avec des actions uniformément aléatoires, et exécuter l'itération Q ajustée
   avec une table comme ajustement : chaque cible est moyennée sur les
   transitions de son couple état-action, et les couples sans données gardent
   la valeur $0$. Évaluer exactement la politique gloutonne à partir de
   l'état $0$.
2. Recommencer avec un journal dans lequel la politique de comportement va
   toujours à gauche dans les états 5 à 8.
3. Expliquer pourquoi la seconde politique échoue dans les états 0 à 4, où le
   journal contient des données pour les deux actions.
:::

:::{solution} ex-fqi-chain
:class: dropdown

Avec le journal large, la politique gloutonne va à droite partout et sa valeur
à partir de l'état 0 est $3.90$, la valeur optimale. Avec le journal étroit,
« droite » n'a aucune donnée dans les états 5 à 8; sa valeur y reste donc à
$0$, ce qui fait égalité avec « gauche » et perd le départage. La politique va
à gauche dans les états 5 à 8, et la valeur qu'elle atteint à partir de
l'état 0 est $0$. Dans les états 0 à 4, les deux actions ont des données, mais
les cibles de « droite » sont construites à partir des valeurs des états 1 à
5, qui sont fausses : l'erreur dans les états non couverts se propage vers
l'arrière à travers les cibles par autoamorçage. Un ajustement qui généralise,
comme des arbres sur l'indice de l'état, pourrait extrapoler « droite » à
partir de l'état 4 ou 9, et cette extrapolation déciderait du résultat :
c'est la situation du tampon étroit d'ordonnancement d'inférence.
:::

:::{exercise} Les pertes comme modèles de bruit
:label: ex-fqi-losses

1. Montrer que minimiser l'erreur quadratique revient au maximum de
   vraisemblance lorsque $y = q + \varepsilon$ avec $\varepsilon$ gaussien, et
   que minimiser la perte de Gumbel revient au maximum de vraisemblance
   lorsque $\varepsilon \sim \text{Gumbel}(0, \beta)$, dont la densité est
   $\tfrac1\beta\exp\big(-\varepsilon/\beta - e^{-\varepsilon/\beta}\big)$.
2. Le codage à deux intervalles écrit une cible comme une combinaison convexe
   de deux valeurs d'intervalle. De quel résultat du chapitre sur les méthodes
   de valeur ajustées en fait-il une instance, et quelle interpolation du
   chapitre sur l'horizon fini réutilise-t-il?
3. L'exemple des satellites mettait ses cibles à l'échelle par l'estimation
   du message courant. Expliquer pourquoi il s'agit d'un changement de modèle
   de bruit, au même sens qu'en 1, et ce qu'une perte de classification avec
   une grille fixe d'intervalles aurait fait de l'événement dont le coût
   valait 2 070.
:::

:::{solution} ex-fqi-losses
:class: dropdown

1. Avec $\varepsilon$ gaussien, l'opposé de la log-vraisemblance de $y$ est $(q - y)^2/2\sigma^2$ plus une constante. Avec $\varepsilon = y - q$ de loi de Gumbel, c'est $\varepsilon/\beta + e^{-\varepsilon/\beta}$ plus une constante; en posant $u = q - y = -\varepsilon$, on obtient $e^{u/\beta} - u/\beta$, la perte de Gumbel à la constante $-1$ près.
2. C'est un moyenneur ({prf:ref}`gordon-averager`) appliqué à l'axe des valeurs, et il réutilise l'interpolation linéaire à poids positifs ou nuls de {prf:ref}`backward-recursion-interp`.
3. Ajuster le rapport $y/(C\,10^{\rho})$ suppose que les erreurs sont proportionnelles au coût prédit plutôt qu'additives, un modèle de bruit multiplicatif; un seul grand coût ne domine donc plus l'erreur quadratique de sa feuille. Une perte de classification aurait écrêté la cible de 2 070 au dernier intervalle, ce qui borne aussi son influence, mais perd l'information sur l'ampleur du coût.
:::

:::{exercise} Un tampon de votre conception
:label: ex-fqi-open

Prendre le processus de décision markovien d'ordonnancement d'inférence du
chapitre. Concevoir une politique de comportement qui recueille un jeu de
données de 50 000 transitions de couverture aussi restreinte que possible
tout en maintenant la politique ajustée à moins de 2 % du rendement exact, ou
montrer qu'une telle politique n'existe pas. Rapporter la couverture, le
désaccord avec la politique exacte et le rendement, et relier votre
conception à l'écart entre actions aux états où la politique exacte passe du
préremplissage au décodage.
:::

:::{solution} ex-fqi-open
:class: dropdown

Cet exercice est ouvert. Les deux tampons du chapitre encadrent la réponse :
le tampon large couvre tout et perd pourtant 5 % du rendement près de la
courbe de commutation, où les valeurs d'action diffèrent d'au plus 1.24. Une
bonne conception concentre les données sur les états voisins de cette courbe,
où l'écart entre actions est petit et où les erreurs de classement sont
probables, et dépense peu sur les états qui en sont éloignés, où tout
ajustement raisonnable classe correctement les actions.
:::

## Synthèse

L'itération Q ajustée sur des données est l'itération sur les valeurs
ajustée dans laquelle le modèle est remplacé par la loi empirique d'un
tampon : des cibles issues de transitions enregistrées, un successeur
chacune, et un ajustement qui fait la moyenne de son erreur sur ces
transitions. Les deux expériences ont localisé les erreurs d'une politique
ajustée. Dans le journal d'ordonnancement à couverture complète, la politique
ne s'est trompée que là où deux actions étaient presque à égalité, une erreur
d'approximation que davantage de données n'éliminerait pas; dans le journal
qui suivait une heuristique, elle s'est trompée là où l'heuristique n'avait
jamais essayé l'action de la politique exacte, une erreur de couverture
qu'aucun régresseur ne peut réparer. Sur les alertes de collision de l'ESA,
l'itération Q ajustée a appris à partir du seul journal une règle à environ
7 % de celle qu'un opérateur écrirait à la main, une fois ses cibles mises à
l'échelle afin qu'un événement rare ne puisse pas dominer l'ajustement.

Écrire explicitement les pas de gradient d'un ajustement neuronal et les
fusionner avec la boucle extérieure produit les réseaux en ligne et cible de
DQN : le réseau cible est constitué des paramètres de l'itération extérieure
gardés en mémoire, et le réseau en ligne est l'itéré intérieur, qui recueille
aussi les données dès qu'on laisse le tampon évoluer. Double DQN ne modifie
que la fonction cible, afin de découpler la sélection de l'évaluation, et le
Q-learning est la limite d'une transition par mise à jour, une approximation
stochastique dont la méthode de l'EDO établit la convergence pour les tables.
Tous occupent un même espace de conception, dont les coordonnées sont un
tampon, une fonction cible et un ajustement, et la perte de l'ajustement est
un modèle de bruit pour les cibles, que la régression de Gumbel et les pertes
de classification représentent mieux que l'erreur quadratique.

L'itération Q ajustée peut maintenant apprendre des valeurs d'action à partir
de tampons fixes ou évolutifs, mais chaque cible exige encore
$\max_a q(s,a)$. Que se passe-t-il lorsque cette recherche d'action est
elle-même un programme non linéaire? [L'optimisation amortie des
actions](amortized-action-optimization.md) apprend un acteur qui remplace la
recherche répétée tout en conservant le critique Q ajusté.
