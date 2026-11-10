# Des modèles à l'apprentissage

Sur le tokamak TCV, une politique apprise a commandé un plasma en envoyant des
consignes de tension à 19 bobines magnétiques, à une fréquence de 10 kHz
{cite:p}`Degrave2022`. Cette politique n'a ni formé le plasma, ni choisi les
capteurs, ni délimité une région de fonctionnement sûre, ni décidé de ce que
signifiait une consigne de tension. Une commande classique formait et
stabilisait d'abord le plasma, et un simulateur précisait les conditions
physiques et opérationnelles dans lesquelles la politique était entraînée.

L'algorithme d'apprentissage opérait donc à l'intérieur d'un problème de
décision déjà formulé. La frontière du système, l'état et les observations, les
actions disponibles, l'objectif et les conditions d'arrêt étaient fixés avant
l'entraînement. Ces choix déterminent ce qu'une politique apprise peut
observer, provoquer et optimiser.

:::{admonition} Définition de travail
:class: course-definition

L'**apprentissage par renforcement** étudie comment des données issues de
l'interaction ou d'expériences enregistrées peuvent servir à évaluer et à
améliorer des décisions dont les conséquences se déploient à travers un
système dynamique. Les actions présentes modifient les états futurs, et donc
les possibilités et les résultats futurs.

Il s'agit d'une classe de problèmes plutôt que d'un algorithme particulier.
Les méthodes par différence temporelle, les méthodes fondées sur la valeur et
les méthodes par gradient de politique sont des familles de solutions. L'état,
l'ensemble des actions, l'objectif, les contraintes et l'interface de
transition définissent le problème qu'on leur demande de résoudre.
:::

La prise de décision séquentielle est le sujet plus large. L'optimisation de
trajectoires, la commande par rétroaction et la programmation dynamique exacte
peuvent résoudre certains problèmes séquentiels sans apprendre à partir de
données. L'apprentissage par renforcement intervient lorsqu'un modèle, une
fonction de valeur ou une politique doit être estimé à partir d'échantillons ou
d'interactions. Le livre développe d'abord la structure mathématique commune,
puis introduit les situations où les modèles, les valeurs ou les politiques
doivent être estimés à partir de données.

Une application n'arrive pas avec cette interface toute faite. Sa formulation
doit préciser où s'arrête le système, ce que l'on sait au moment de choisir une
action, quelles interventions sont physiquement possibles, comment les
résultats sont évalués et quelles exigences ne peuvent pas être échangées
contre de la récompense. Un optimiseur ne peut utiliser que les variables, les
objectifs et les contraintes présents dans cette formulation. Une exigence
omise ne peut pas influencer sa solution.

## L'interface entre l'agent et l'environnement

La boucle habituelle entre l'agent et l'environnement commence une fois qu'un
signal d'état, des actions et des récompenses ont été choisis.
{cite:t}`SuttonBarto2018` traitent le signal d'état comme la sortie d'un
système de prétraitement et laissent de côté une grande partie de sa
construction afin d'étudier la prise de décision. Leur abstraction isole le
problème d'apprentissage; elle ne fait pas disparaître les choix de
formulation qui la précèdent.

La même boucle apparaît dans d'autres disciplines. La théorie de la commande,
la recherche opérationnelle et l'économétrie donnent d'autres noms à des
échanges causaux apparentés et leur posent des questions différentes.

<div id="fig-interaction-loop-vocabularies" class="interaction-lenses-figure" role="figure" aria-label="Trois vocabulaires pour une même interaction causale.">
<div class="interaction-lenses" role="group" aria-label="La même interaction séquentielle décrite dans les vocabulaires de l'apprentissage par renforcement, de la théorie de la commande, et de la recherche opérationnelle ou de l'économétrie.">
  <div class="interaction-lens">
    <div class="interaction-lens-title">Apprentissage par renforcement</div>
    <div class="interaction-loop" role="img" aria-label="Un agent envoie une action à un environnement et reçoit l'état suivant et une récompense.">
      <div class="interaction-node">
        <span>Agent</span>
        <span class="interaction-node-subtitle">politique <i>π</i></span>
      </div>
      <div class="interaction-arrows" aria-hidden="true">
        <span>action <i>A</i><sub>t</sub></span>
        <div class="interaction-arrow-glyph"><span>→</span><span>←</span></div>
        <span>état <i>S</i><sub>t+1</sub><br />récompense <i>R</i><sub>t+1</sub></span>
      </div>
      <div class="interaction-node interaction-node-world">
        <span>Environnement</span>
        <span class="interaction-node-subtitle">transition <i>p</i></span>
      </div>
    </div>
    <p>Comment des modèles ou de l'expérience peuvent-ils évaluer et améliorer une politique?</p>
    <div class="interaction-context">objectif encodé par la récompense</div>
  </div>

  <div class="interaction-lens">
    <div class="interaction-lens-title">Théorie de la commande</div>
    <div class="interaction-loop" role="img" aria-label="Un correcteur envoie une entrée à un procédé et reçoit une mesure en rétroaction.">
      <div class="interaction-node">
        <span>Correcteur</span>
        <span class="interaction-node-subtitle">loi de rétroaction <i>κ</i></span>
      </div>
      <div class="interaction-arrows" aria-hidden="true">
        <span>entrée <i>u</i><sub>t</sub></span>
        <div class="interaction-arrow-glyph"><span>→</span><span>←</span></div>
        <span>mesure <i>y</i><sub>t</sub></span>
      </div>
      <div class="interaction-node interaction-node-world">
        <span>Procédé</span>
        <span class="interaction-node-subtitle">dynamique <i>f</i></span>
      </div>
    </div>
    <p>Quelle loi de rétroaction amène le procédé à respecter son cahier des charges?</p>
    <div class="interaction-context">consigne · perturbations · coût · contraintes</div>
  </div>

  <div class="interaction-lens">
    <div class="interaction-lens-title">Recherche opérationnelle et économétrie</div>
    <div class="interaction-loop" role="img" aria-label="Une règle de décision applique une intervention à un système ou à une population et reçoit l'état suivant et un résultat.">
      <div class="interaction-node">
        <span>Règle de décision</span>
        <span class="interaction-node-subtitle">objectif + contraintes</span>
      </div>
      <div class="interaction-arrows" aria-hidden="true">
        <span>décision <i>d</i><sub>t</sub></span>
        <div class="interaction-arrow-glyph"><span>→</span><span>←</span></div>
        <span>covariables et<br />résultat</span>
      </div>
      <div class="interaction-node interaction-node-world">
        <span>Système ou population</span>
        <span class="interaction-node-subtitle">processus générateur des données</span>
      </div>
    </div>
    <p>Que provoquerait chaque intervention réalisable, et laquelle faut-il retenir?</p>
    <div class="interaction-context">chocs · faisabilité · hypothèses causales · bien-être</div>
  </div>
</div>

<div class="interaction-lenses-caption"><strong>Trois vocabulaires pour une même interaction causale.</strong>
L'apprentissage par renforcement décrit un agent qui agit dans un
environnement; la théorie de la commande décrit un correcteur qui agit sur un
procédé; la recherche opérationnelle et l'économétrie décrivent des décisions
ou des interventions appliquées à un système qui évolue. La correspondance est
structurelle et non exacte : une mesure ne révèle pas forcément l'état, une
récompense ou un coût doit être spécifié, et les contrefactuels de
l'économétrie exigent des hypothèses d'identification. Le panneau de
l'apprentissage par renforcement suit la convention d'interaction de Sutton et
Barto (2018).</div>
</div>

Les trois panneaux décrivent des échanges causaux apparentés, mais leurs termes
ne sont pas interchangeables. Une mesure de capteur peut ne pas révéler l'état
du système, une récompense ou un coût doit être choisi, et les conclusions
causales tirées de données d'observation exigent des hypothèses
d'identification. Passer d'un vocabulaire à l'autre n'est utile que si ces
différences restent explicites.

## La formulation du TCV

Le simulateur du TCV combinait la dynamique des circuits conducteurs, un
modèle de plasma à frontière libre et une dynamique agrégée du courant de
plasma. Des modèles des capteurs et des alimentations, identifiés à partir de
données, représentaient le filtrage, les retards, le bruit de mesure et les
décalages de tension. Des plages de paramètres fondées sur l'expérience
représentaient l'incertitude, tandis que l'objectif et les règles d'arrêt
représentaient les exigences d'exploitation. Après la passation, la politique
apprise jouait le rôle de correcteur par rétroaction dans ce système ainsi
spécifié.

L'apprentissage par renforcement a utilisé le modèle physique et le correcteur
classique de formation du plasma au lieu de les remplacer. Un simulateur
physique détaillé n'est pas nécessaire dans toutes les applications, mais la
structure disponible reste utile lorsqu'elle détermine les actions
réalisables, les quantités conservées ou les contraintes de sécurité. On
retrouve le même partage en commande sûre des robots : la dynamique nominale,
les coûts et les contraintes encodent les connaissances préalables, et les
composantes incertaines sont apprises à partir de données {cite:p}`Brunke2022`.

Les systèmes réels imposent aussi des échantillons limités, des retards, des
contraintes de sécurité, une observabilité partielle, de la stochasticité, de
la non-stationnarité, des objectifs multiples, une sélection d'action en temps
réel et des changements de distribution {cite:p}`DulacArnold2021`. Ces
conditions touchent la formulation elle-même. Elles déterminent quel état est
prédictif, quelle action est réalisable, quelles données permettent
l'évaluation et quelle conclusion une expérience réussie peut justifier.

## Lien avec la théorie de l'apprentissage par renforcement

Les équations de Bellman, la programmation dynamique, les méthodes de
Monte-Carlo et par différence temporelle, l'approximation de la fonction de
valeur et les gradients de politique partent tous d'une interface de décision
séquentielle déjà spécifiée. Ces fondements occupent la seconde moitié du
livre. Les premiers chapitres étudient les étapes de modélisation et de
commande qui construisent cette interface et la mettent à l'épreuve.

Lorsque la dynamique et les contraintes sont connues, on peut optimiser une
trajectoire à partir d'une condition initiale donnée. En répétant
l'optimisation, on transforme ces plans en rétroaction. La programmation
dynamique calcule des décisions conditionnelles à l'état pour l'ensemble des
états futurs possibles, et l'approximation devient nécessaire lorsque le
modèle, la fonction de valeur ou la politique requis sont inconnus ou trop
coûteux à représenter exactement.

L'apprentissage par renforcement sans modèle et l'apprentissage par
renforcement fondé sur un modèle partent tous deux d'une interface spécifiée.
Si seules des transitions et des récompenses sont disponibles, des méthodes
fondées sur l'échantillonnage peuvent opérer sur cette interface. Si l'on
dispose aussi d'équations, de la géométrie, de lois de conservation ou de
contraintes strictes, celles-ci peuvent rester explicites. Avec un modèle
partiel, les données peuvent servir à estimer des paramètres incertains, une
dynamique résiduelle, des fonctions de valeur ou des politiques sans remplacer
la structure connue.

## Formuler le problème de décision

Un problème de décision séquentielle doit répondre à cinq questions avant
qu'un algorithme puisse le résoudre :

1. Quelles variables résument l'information nécessaire aux décisions futures?
2. Quelles actions le décideur peut-il réellement appliquer?
3. Comment les actions, les perturbations et l'incertitude modifient-elles le système?
4. Quels résultats l'objectif doit-il récompenser ou pénaliser?
5. Quelles contraintes physiques, opérationnelles ou de sécurité doivent toujours être respectées?

Les réponses déterminent quels calculs et quelles conclusions sont possibles.
Une équation différentielle explicite donne accès aux dérivées et aux lois de
conservation. Un simulateur peut n'offrir que des opérations de
réinitialisation et de transition. Des données historiques ne peuvent pas
répondre à des questions contrefactuelles arbitraires sans hypothèses
supplémentaires. L'interaction en direct fournit de nouvelles données, mais
leur collecte prend du temps et peut endommager le système.

## Structure du livre

Le développement technique suit l'information dont on dispose sur le problème
de décision. Chaque méthode présentée par la suite repose sur une description
de la façon dont les actions mènent aux observations futures, que cette
description provienne d'équations, d'un simulateur, de transitions
enregistrées ou d'une interaction en direct.

| Partie | Question | Résultat |
|---|---|---|
| Modélisation des décisions séquentielles | Qu'est-ce qui évolue, que peut-on observer et où l'action intervient-elle? | Un état, une action, une perturbation, un objectif, un ensemble de contraintes et une interface de modèle |
| Optimisation de trajectoires | Quelle suite d'actions donne de bons résultats à partir d'une condition initiale? | Un plan à horizon fini, ainsi que les conditions et les méthodes numériques qui servent à le calculer |
| Commande prédictive | Comment le plan peut-il réagir lorsque les mesures révèlent un état différent? | Un comportement en boucle fermée obtenu par replanification répétée |
| Programmation dynamique | Peut-on calculer des décisions conditionnelles à l'état avant d'observer l'état? | Des fonctions de valeur et des politiques définies sur de nombreux états possibles |
| Programmation dynamique approchée (au sein de la programmation dynamique) | Comment résoudre les équations de Bellman lorsque leurs fonctions ou leurs espérances ne peuvent pas être représentées exactement? | Des fonctions de valeur projetées, échantillonnées et ajustées |
| Optimisation des politiques | Comment optimiser une classe restreinte de politiques lorsque la sélection d'action ou l'évaluation du rendement est intraitable? | Des sélecteurs d'action amortis, des méthodes acteur-critique et des gradients de politique directs |

Cet ordre ne suppose pas un modèle mécaniste complet. Il garde l'information
disponible explicite : les lois de conservation, la géométrie et les
contraintes connues restent dans la formulation, tandis que les paramètres
incertains, la dynamique résiduelle, les fonctions de valeur ou les politiques
deviennent des cibles d'apprentissage.

La commande prédictive est la première construction en boucle fermée du livre :
un nouveau plan à horizon fini est calculé à partir de chaque état mesuré. La
programmation dynamique construit la rétroaction autrement, en résolvant pour
les décisions sur toute une famille d'états. L'approximation de fonctions et
l'optimisation stochastique rendent ensuite ces objets en boucle fermée
calculables lorsque les fonctions de valeur, les espérances ou les recherches
d'action exactes ne sont pas disponibles.

Ces structures reviennent en apprentissage par renforcement, en théorie de la
commande et en recherche opérationnelle, ce qui permet aux méthodes de passer
d'une discipline à l'autre pourvu que les hypothèses qui les accompagnent
soient préservées.

## Comment utiliser ce livre

Les chapitres sont cumulatifs, mais chacun commence par ses préalables et
renvoie au matériel de révision pertinent. Les lecteurs qui viennent de
l'apprentissage par renforcement trouveront peut-être les chapitres de
modélisation et d'optimisation de trajectoires moins familiers. Ceux qui
viennent de la commande auront peut-être besoin de plus de temps pour les
opérateurs de Bellman, l'échantillonnage et l'approximation de fonctions. Les
annexes offrent de brèves révisions des problèmes à valeur initiale et de la
programmation non linéaire, au moment où ces outils deviennent nécessaires.

Les dérivations font partie de l'argument; elles ne sont pas une vérification
décorative. Arrêtez-vous avant une étape non triviale, tentez-la sur papier,
puis comparez votre résultat avec le texte. Faites les courtes
autoévaluations pendant que la notation est encore fraîche. Les exercices plus
longs sont prévus pour une seconde lecture et relient souvent plusieurs
chapitres.

Les exemples numériques répondent à des questions précises de modélisation ou
d'algorithmique. Leur code source, leurs paramètres et leurs résultats
enregistrés sont fournis pour qu'on puisse examiner et modifier le résultat.
Exécuter chaque exemple est facultatif, mais un résultat numérique doit être lu
avec les hypothèses et le diagnostic qui l'appuient.

## Autoévaluation

:::{exercise} Le problème avant l'algorithme
:label: ex-intro-check-1

Un correcteur de production doit respecter une limite de stockage stricte. Le choix d'un algorithme d'apprentissage suffit-il à définir le problème d'apprentissage par renforcement? Nommez deux éléments de structure qu'il faut d'abord spécifier.
:::

:::{solution} ex-intro-check-1
:class: dropdown

Non. Il faut au minimum définir l'état et les décisions disponibles, ainsi que l'objectif et la contrainte stricte de stockage. La structure de transition ou de rétroaction doit aussi préciser comment les décisions influencent les états futurs.
:::

:::{exercise} Une structure commune
:label: ex-intro-check-2

Quelle caractéristique fait de l'optimisation de trajectoires, de la commande prédictive et de l'apprentissage par renforcement des membres d'une même famille de méthodes de décision?
:::

:::{solution} ex-intro-check-2
:class: dropdown

Toutes trois choisissent des actions dont les conséquences se déploient dans le temps à travers un système dynamique; les décisions présentes doivent donc tenir compte des objectifs et des contraintes futurs.
:::

:::{exercise} L'information disponible
:label: ex-intro-check-3

Supposons qu'un environnement fournisse déjà un état markovien, un ensemble
d'actions, une récompense et un échantillonneur de transitions. Quel travail de
formulation a déjà été fait? Quels calculs supplémentaires deviennent possibles
si l'on dispose aussi d'une dynamique différentiable et de contraintes
strictes?
:::

:::{solution} ex-intro-check-3
:class: dropdown

Le concepteur de l'environnement a choisi une frontière, une représentation de
l'état, des actions, une interface de transition et une récompense, même si ces
choix doivent encore être vérifiés au regard de l'application visée. Les
méthodes d'apprentissage par renforcement fondées sur l'échantillonnage peuvent
opérer sur cette interface. Une dynamique différentiable et des contraintes
explicites permettent en plus l'optimisation directe de trajectoires, la
commande prédictive, la conception de rétroactions locales et la programmation
dynamique fondée sur un modèle, lorsque leurs autres hypothèses sont
satisfaites.
:::
