# Commande prédictive par intégrale de chemin

Les chapitres précédents ont formulé l'optimisation de trajectoires
déterministe comme un programme non linéaire. Le tir et la collocation
transforment la dynamique, les coûts et les contraintes en un problème de
dimension finie qu'on peut résoudre avec un solveur de PNL. Les méthodes à base
de gradient sont efficaces lorsque des dérivées précises sont disponibles.
Comment améliorer une trajectoire lorsqu'on peut simuler le système et évaluer
son coût, mais que la dérivation à travers ce calcul est difficile ou
coûteuse?

La commande prédictive par intégrale de chemin (MPPI) construit une mise à
jour par échantillonnage. Elle tire des suites de commandes candidates, simule
leurs trajectoires et forme une moyenne pondérée qui favorise les issues de
moindre coût. Cela ne requiert aucune dérivée de la dynamique ni du coût, et
les déroulements candidats peuvent s'exécuter en parallèle. La dérivation part
d'une loi sur les suites de commandes : les coûts déterminent quelles suites
reçoivent davantage de probabilité, et l'ajustement d'une gaussienne à cette
loi donne la mise à jour par moyenne pondérée. Cette construction s'applique
même à un simulateur déterministe. Un exemple d'aéronef étend le calcul à des
vents incertains en estimant le coût espéré de chaque vol candidat avant de
lui attribuer son poids. Un exemple de réacteur applique la même mise à jour à
des champs de température et de composition régis par une équation aux
dérivées partielles, d'abord en planifiant avec l'équation elle-même, puis
avec un modèle de substitution appris.
L'exécution de la première action suivie d'une nouvelle planification à
partir de l'observation suivante relie ces calculs au chapitre suivant, sur la
commande à horizon glissant.

## Objectifs d'apprentissage

Après avoir étudié les exemples et les exercices, vous devriez pouvoir :

- Calculer une mise à jour MPPI, y compris la correction nécessaire lorsqu'on
  échantillonne autour d'une suite de commandes nominale non nulle.
- Dériver la pondération exponentielle à partir d'un objectif régularisé par
  la divergence KL et obtenir la mise à jour de la commande en ajustant la
  moyenne d'une gaussienne.
- Distinguer les échantillons qui servent à explorer les commandes de ceux qui
  servent à estimer le coût espéré d'un candidat sous des perturbations
  physiques.
- Évaluer un plan échantillonné à l'aide de vérifications de faisabilité, de
  la concentration des poids et de comparaisons sous des perturbations
  communes.
- Exécuter la première action d'un plan échantillonné et mettre à jour le
  reste du plan à partir de nouvelles observations.
- Réduire un problème de commande de champ à quelques paramètres
  d'échantillonnage bornés, et valider chaque mise à jour échantillonnée dans
  la simulation complète avant qu'elle remplace un plan réalisable.

## Préalables

L'algorithme utilise le [déroulement par tir simple](numerical-trajectory-optimization.md)
et la [formulation du contrôle optimal à horizon fini](discrete-time-optimal-control.md).
La dérivation par échantillonnage requiert les lois gaussiennes et les
espérances; la divergence KL et l'échantillonnage préférentiel sont développés
ici. L'exemple de l'aéronef utilise des modèles de perturbation conditionnels
tirés de
[Dynamique stochastique et observation partielle](stochastic-dynamics-observation.md).
L'exemple du réacteur énonce le modèle en volumes finis qu'il utilise et ne
suppose aucune connaissance préalable des équations aux dérivées partielles.

(sec-mppi)=
## Améliorer une suite de commandes par échantillonnage

Nous savons déjà évaluer une suite de commandes : simuler le système et
calculer le coût qui en résulte. Comment améliorer cette suite? Au lieu de
calculer un gradient, on essaie de nombreuses variations aléatoires, on évalue
chacune d'elles et on forme une moyenne pondérée qui donne plus d'influence aux
candidats de moindre coût. Cette moyenne devient la suite de commandes
suivante.

Si le système possède $d_u$ composantes de commande, alors
$\mathbf u_k\in\mathbb R^{d_u}$ contient les entrées appliquées ensemble au
pas de temps $k$. Le plan courant
$\mathbf U=(\mathbf u_0,\ldots,\mathbf u_{N-1})$ regroupe ces vecteurs sur
les $N$ pas. Ainsi, $\mathbf u_k$ est le vecteur de commande d'un pas, tandis
que $\mathbf U$ est la suite complète, qui compte $Nd_u$ composantes une fois
empilée en un seul vecteur.

Tirons $K$ suites de perturbations gaussiennes $\boldsymbol\epsilon^{(i)}$ et
simulons chaque candidat $\mathbf V^{(i)}=\mathbf U+\boldsymbol\epsilon^{(i)}$.
Après avoir attribué des poids normalisés $w_i$ aux trajectoires obtenues, on
forme la nouvelle suite de commandes

$$
\mathbf U^+=\sum_{i=1}^K w_i\mathbf V^{(i)}
=\mathbf U+\sum_{i=1}^K w_i\boldsymbol\epsilon^{(i)},
\qquad w_i\geq0,\qquad \sum_{i=1}^K w_i=1.
$$

Le même poids de trajectoire multiplie chaque action de sa suite. Comment ces
poids doivent-ils dépendre des coûts? Donner le même poids à chaque candidat,
$w_i=1/K$, ignorerait les résultats des déroulements. Comme les perturbations
sont de moyenne nulle, cette mise à jour renverrait $\mathbf U$ en moyenne,
quels que soient les candidats qui ont bien performé. À l'autre extrême,
attribuer un poids de un au candidat de moindre coût et zéro aux autres
sélectionnerait une seule suite. Un candidat de coût presque identique
n'aurait aucune influence, et un léger changement dans le classement du
premier candidat pourrait faire basculer toute la mise à jour.

MPPI permet à plusieurs candidats de faible coût de contribuer, leur influence
diminuant progressivement à mesure que le coût augmente. Considérons d'abord la
suite initiale $\mathbf U=\mathbf0$. Les suites candidates sont alors les
perturbations gaussiennes de moyenne nulle elles-mêmes. Cette gaussienne est la
**loi de référence** : elle reste fixe même lorsque des mises à jour
ultérieures éloignent de zéro la moyenne d'échantillonnage. Soit $S_i$ le coût
d'état accumulé le long du déroulement $i$, y compris son coût terminal. MPPI
lui attribue le score positif $\exp(-S_i/\lambda)$ et divise par la somme de
tous les scores :

$$
w_i=\frac{\exp(-S_i/\lambda)}{\sum_{j=1}^K\exp(-S_j/\lambda)},
\qquad \lambda>0.
$$

Il s'agit de la fonction softmax des coûts négatifs, de température
$\lambda$. Le signe moins attribue aux coûts plus faibles des scores plus
élevés, et le dénominateur fait en sorte que la somme des poids vaille un. Le
paramètre $\lambda$ a les mêmes unités que le coût, de sorte que $S_i/\lambda$
est sans dimension. Pour voir comment il règle la force de la préférence,
comparons deux candidats. Leur dénominateur commun se simplifie :

$$
\frac{w_i}{w_j}
=\frac{\exp(-S_i/\lambda)}{\exp(-S_j/\lambda)}
=\exp\!\left(-\frac{S_i-S_j}{\lambda}\right).
$$

Si le déroulement $i$ coûte $\lambda\log2$ de plus que le déroulement $j$, il
reçoit deux fois moins de poids. Ce sont donc les différences de coût, et non
le zéro arbitraire de l'échelle des coûts, qui déterminent l'influence
relative. Pour un lot fixé, un petit $\lambda$ fait en sorte que même des
différences de coût modestes influent fortement sur les poids; un grand
$\lambda$ les rend moins déterminantes. Lorsque $\lambda\to0^+$, les poids se
concentrent sur le candidat de moindre coût, ou se répartissent également
entre des minima ex æquo. Lorsque $\lambda\to\infty$, ils tendent vers $1/K$.
On retrouve les deux options ci-dessus, désormais reliées par une règle de
pondération continue.

Les courbes de la @fig-mppi-exponential-weights montrent cette dépendance au
coût et à la température. Chaque courbe compare le poids d'un déroulement au
poids du meilleur candidat du même lot.

:::{figure} _static/brownian_mppi/exponential-weights.svg
:label: fig-mppi-exponential-weights
:alt: Trois courbes exponentielles représentent le poids d'un déroulement relativement au meilleur déroulement en fonction de son excédent de coût. La courbe de température un demi décroît le plus vite, celle de température un décroît à une vitesse intermédiaire et celle de température deux décroît le plus lentement. Une ligne pointillée horizontale indique la limite des poids égaux. À la température un, un excédent de coût de log deux donne la moitié du poids du meilleur déroulement.

Des températures plus faibles réduisent plus fortement l'influence des
trajectoires plus coûteuses. Avec $S_{\min}=\min_i S_i$, l'axe horizontal
représente l'excédent de coût $\Delta S=S_i-S_{\min}$, et l'axe vertical le
rapport $w_i/w_{\mathrm{best}}=\exp(-\Delta S/\lambda)$. Les coûts et les
températures utilisent la même unité de coût arbitraire. Le point marqué
montre que $\Delta S=\log2$ réduit de moitié le poids relatif lorsque
$\lambda=1$. La ligne horizontale pointillée est la limite des poids égaux
lorsque $\lambda\to\infty$. Un rapport de un en $\Delta S=0$ ne signifie pas
que le meilleur déroulement reçoit tout le poids normalisé : la somme des
véritables $w_i$ sur le lot vaut toujours un.
:::

La référence gaussienne favorise déjà les petites commandes. Sa contribution à
l'effort de commande apparaîtra explicitement dans l'objectif sur les lois
présenté plus bas; $S_i$ ne comprend que les coûts d'état et le coût terminal.
Les poids affichés s'appliquent lorsqu'on échantillonne selon cette référence,
avec $\mathbf U=\mathbf0$. Échantillonner autour d'un plan non nul modifie la
fréquence à laquelle chaque candidat est tiré et exige une correction
correspondante de son poids.

L'attrait calculatoire tient à ce que chaque déroulement et chaque évaluation
de coût peuvent s'exécuter indépendamment, ce qui rend les grands lots propices
à l'exécution sur processeur graphique (GPU)
{cite:p}`Williams2017InformationTheoretic`. Ce parallélisme ne garantit pas
qu'un nombre modeste d'échantillons trouvera des suites de commandes de faible
coût : de longs horizons, de nombreuses variables de commande et des régions
réalisables étroites peuvent rendre la recherche difficile. Nous vérifierons à
la fois la qualité de l'échantillonnage et la faisabilité en utilisant la mise
à jour dans les exemples de l'aéronef et du réacteur.

Jusqu'ici, la pondération exponentielle est une règle pour combiner un lot
d'essais. Pour la justifier, il faut un objectif dont la solution donne ces
poids, et une raison d'utiliser leur moyenne comme plan suivant. Les deux
découlent du fait de traiter les suites candidates comme des échantillons d'une
loi et d'optimiser cette loi avant d'y ajuster une nouvelle suite de
commandes.

(sec-mppi-distribution)=
## Des poids de trajectoires à MPPI

Quel problème d'optimisation donne les poids exponentiels et leur mise à jour
par moyenne pondérée? Une loi sur les suites de commandes permet d'arbitrer
entre un faible coût de trajectoire et une préférence pour de petites entrées.
La loi qui en résulte n'a pas nécessairement une forme simple, mais lui
ajuster une gaussienne produit une moyenne qui peut servir de plan suivant.
L'échantillonnage estime ensuite cette moyenne. Tout au long de cette
dérivation, l'aléa provient de la recherche sur les commandes; le simulateur
lui-même peut être déterministe.

### Lois de référence et d'échantillonnage

Pour un déroulement général, écrivons
$\mathbf{x}_{k+1}=\mathbf{f}_k(\mathbf{x}_k,\mathbf{v}_k)$, où
$\mathbf v_k\in\mathbb R^{d_u}$ est le vecteur de commande candidat au pas
$k$. Regroupons ces vecteurs dans la suite candidate complète
$\mathbf V=(\mathbf v_0,\ldots,\mathbf v_{N-1})$, de même structure que la
suite nominale $\mathbf U$. L'état initial est fixé, de sorte que chaque suite
d'entrées détermine une suite d'états simulée. Définissons le coût d'état et
le coût terminal d'un déroulement par

$$
S(\mathbf V)=c_N(\mathbf{x}_N)
+\sum_{k=0}^{N-1}c_k(\mathbf{x}_k).
$$

Chaque $c_k$ est le coût d'un intervalle; s'il provient d'un coût courant
continu, sa discrétisation inclut la longueur de l'intervalle.

Dans la formulation du contrôle optimal vue plus tôt, un coût d'étape
$c_k(\mathbf x_k,\mathbf u_k)$ pouvait pénaliser à la fois l'état et l'effort
de commande. Ici, $c_k(\mathbf x_k)$ ne contient que le coût d'état; les
entrées influent encore sur $S(\mathbf V)$ par l'intermédiaire des états
simulés. L'effort de commande sera pris en compte séparément par une pénalité
de divergence KL entre la loi des entrées et une loi de référence.

Choisissons une loi de référence gaussienne $P_0$ centrée en zéro pour
exprimer une préférence pour de petites entrées. Comme on le dérive plus bas,
ce choix fait de la pénalité KL un coût quadratique sur l'amplitude des
commandes nominales; la pénalité s'annule lorsque la suite nominale est nulle.
Utilisons une loi d'échantillonnage gaussienne distincte $Q_{\mathbf U}$ sur
les suites d'entrées complètes, centrée sur le plan courant $\mathbf U$. À
chaque pas $k$, la commande candidate $\mathbf v_k$ a pour moyenne
$\mathbf0$ sous $P_0$ et $\mathbf u_k$ sous $Q_{\mathbf U}$. Les deux lois
utilisent la même covariance définie positive
$\Sigma\in\mathbb R^{d_u\times d_u}$ pour les composantes de commande d'un
même pas. L'hypothèse d'indépendance dans le temps donne les densités jointes
sur la suite complète :

$$
p_0(\mathbf V)=\prod_{k=0}^{N-1}\mathcal N(\mathbf v_k;\mathbf0,\Sigma),
\qquad
q_{\mathbf U}(\mathbf V)=
\prod_{k=0}^{N-1}\mathcal N(\mathbf v_k;\mathbf u_k,\Sigma).
$$

Centrer la référence en zéro est un choix de modélisation. Une référence
gaussienne pourrait plutôt être centrée sur une suite de base prescrite; la
pénalité KL imputerait alors un coût aux écarts quadratiques des commandes
nominales par rapport à cette suite de base. Dans les deux cas, les coûts
d'état et le coût terminal peuvent favoriser des commandes éloignées de la
moyenne de référence lorsque leurs avantages justifient l'effort.

La référence reste fixe pendant une mise à jour de planification. La moyenne
d'échantillonnage est la suite nominale entière $\mathbf U$. Cette moyenne est
prise sur des tirages répétés de candidats, avec un vecteur de commande moyen
distinct $\mathbf u_k$ pour chaque pas de temps. Modifier $\mathbf u_k$
déplace le centre des commandes candidates au pas $k$. Initialiser
$\mathbf U=\mathbf0$ revient à poser $\mathbf u_k=\mathbf0$ pour tout $k$, de
sorte que les lois d'échantillonnage et de référence coïncident au départ. Les
mises à jour ultérieures peuvent modifier les commandes nominales à chaque pas
et déplacer les échantillons vers un nouveau plan, tandis que la référence
reste centrée en zéro. L'échantillonneur peut donc changer l'endroit où il
cherche sans changer la référence qui définit l'objectif.

### Pondération exponentielle de la référence

L'échantillonnage selon la référence détermine la fréquence à laquelle
différentes suites d'entrées apparaissent dans un lot. L'attribution d'un
poids de coût à chaque tirage modifie ensuite la contribution de cette
occurrence à la moyenne pondérée. Le poids total attribué à une suite dépend à
la fois de sa fréquence d'échantillonnage et de son poids par occurrence, et
c'est pourquoi ces deux facteurs se multiplient.

Par exemple, considérons d'abord une référence qui n'admet que deux suites
d'entrées possibles, $a$ et $b$, choisies avec les probabilités $3/4$ et
$1/4$. Supposons que leurs coûts soient $S(a)=\lambda\log3$ et $S(b)=0$, de
sorte que chaque tirage de $a$ reçoit le score $1/3$ et chaque tirage de $b$
le score $1$. Leurs probabilités de référence multipliées par ces scores sont

$$
\frac34\times\frac13=\frac14
\quad\text{pour }a,
\qquad
\frac14\times1=\frac14
\quad\text{pour }b.
$$

En divisant par la somme, $1/2$, on obtient la probabilité $1/2$ pour chaque
suite. La référence tire $a$ trois fois plus souvent, mais chaque occurrence
reçoit le tiers du poids, si bien que les deux effets se compensent.

Pour des entrées continues, la même opération multiplie la densité de
référence $p_0(\mathbf V)$ par le score $e^{-S(\mathbf V)/\lambda}$ en chaque
suite $\mathbf V$, puis divise par l'intégrale de ce produit :

$$
\begin{aligned}
p^\star(\mathbf V)
&=\frac{e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)}{Z},
\qquad \lambda>0,\\
Z&=\int e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)\,d\mathbf V
=\mathbb E_{P_0}[e^{-S(\mathbf V)/\lambda}].
\end{aligned}
$$

Le score $e^{-S(\mathbf V)/\lambda}$ n'est pas lui-même une densité de
probabilité, et le produit ne définit pas une loi jointe de deux variables
aléatoires. Les deux facteurs portent sur la même suite d'entrées : $p_0$
précise sa densité avant toute prise en compte des coûts, et le score ajuste
cette densité selon son coût. La constante de normalisation $Z$ fait en sorte
que la densité obtenue s'intègre à un.

Supposons que $S$ soit mesurable et minorée, et que $0<Z<\infty$. Le résultat
normalisé est une **loi de Gibbs** : deux trajectoires de même densité de
référence dont les coûts diffèrent de $\Delta S$ ont un rapport de densités
$e^{-\Delta S/\lambda}$. Le paramètre $\lambda$ a les unités du coût et règle
l'intensité avec laquelle ces coûts influent sur le rapport.

Les poids du lot initial ne contiennent que les scores, car le tirage selon
$P_0$ tient déjà compte des probabilités de référence. La multiplication par
$p_0$ apparaît lorsqu'on décrit la loi que représentent ces échantillons
pondérés. Cela explique comment la pondération modifie une loi; le choix
particulier d'un score exponentiel est justifié par l'objectif, présenté
ci-dessous, qui combine le coût espéré et la divergence KL.

### L'objectif défini par les poids

Les poids exponentiels résolvent un problème d'optimisation sur les lois des
suites d'entrées. On cherche une loi $Q$ qui réduit le coût d'état et le coût
terminal espérés tout en payant une pénalité pour s'écarter de $P_0$. On
mesure cet écart par la **divergence de Kullback-Leibler**. Pour des densités
$q,p$, elle vaut

$$
D_{\mathrm{KL}}(Q\|P)
=\mathbb E_{\mathbf V\sim Q}\left[\log\frac{q(\mathbf V)}{p(\mathbf V)}\right].
$$

Cette divergence est non négative et ne s'annule que lorsque les deux lois
coïncident. Pour qu'elle soit finie, $Q$ ne doit attribuer aucune probabilité
là où $P$ a une densité nulle, condition appelée continuité absolue. Avec
$\lambda>0$ qui fixe le coût d'un écart à la référence, le premier problème
d'optimisation est

```{math}
:label: eq-mppi-distribution-objective
\min_Q\;\mathcal J(Q),
\qquad
\mathcal J(Q)
=\mathbb E_{\mathbf V\sim Q}[S(\mathbf V)]
+\lambda D_{\mathrm{KL}}(Q\|P_0).
```

Ici, $Q$ parcourt les lois sur les suites d'entrées complètes pour lesquelles
les deux termes sont finis; elle n'est pas restreinte à une famille gaussienne.

Rappelons que $P^\star$ est la loi obtenue en multipliant la densité de
référence par le score de coût exponentiel, puis en normalisant. Sa densité
est $p^\star(\mathbf V)=e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)/Z$. Pour
vérifier que cette loi résout le problème d'optimisation, on peut exprimer
l'écart entre $\mathcal J(Q)$ et son minimum comme une divergence non
négative. En substituant cette densité dans le logarithme, on obtient

$$
\log\frac{q(\mathbf V)}{p^\star(\mathbf V)}
=\log\frac{q(\mathbf V)}{p_0(\mathbf V)}
+\frac{S(\mathbf V)}{\lambda}+\log Z.
$$

En prenant l'espérance sous $Q$, on retrouve les deux termes de
$\mathcal J(Q)$ :

$$
\begin{aligned}
D_{\mathrm{KL}}(Q\|P^\star)
&=D_{\mathrm{KL}}(Q\|P_0)
+\frac{1}{\lambda}\mathbb E_Q[S(\mathbf V)]+\log Z\\
&=\frac{\mathcal J(Q)}{\lambda}+\log Z.
\end{aligned}
$$

En réarrangeant, on isole l'objectif :

$$
\mathcal J(Q)=-\lambda\log Z
+\lambda D_{\mathrm{KL}}(Q\|P^\star)
\geq-\lambda\log Z.
$$

La quantité $Z$ est fixée par $S$, $P_0$ et $\lambda$, de sorte que le terme
$-\lambda\log Z$ ne dépend pas de $Q$. Choisir $Q=P^\star$ annule le terme
restant, $\lambda D_{\mathrm{KL}}(Q\|P^\star)$. Toute autre loi donne une
divergence positive et donc une valeur de l'objectif plus grande. Cela
démontre que $P^\star=\operatorname*{arg\,min}_Q\mathcal J(Q)$, de valeur
minimale $\mathcal J(P^\star)=-\lambda\log Z$.

Le poids de coût exponentiel est contenu dans la définition de $P^\star$; il
n'apparaît plus explicitement dans l'identité KL parce que nous avons pris des
logarithmes. L'écriture explicite de la densité de la loi minimisante donne

$$
p^\star(\mathbf V)
=\frac{1}{Z}\,
\underbrace{e^{-S(\mathbf V)/\lambda}}_{\text{poids de coût exponentiel}}
\,p_0(\mathbf V).
$$

Multiplier la densité de référence $p_0(\mathbf V)$ par le facteur indiqué et
normaliser par $Z$, c'est la pondération exponentielle présentée plus haut. La
borne sur $\mathcal J(Q)$ démontre que la loi ainsi construite résout
@eq-mppi-distribution-objective.

Pour une gaussienne $Q_{\mathbf U}$ de même covariance que la référence, la
pénalité KL est une fonction quadratique de ses commandes moyennes :

$$
\lambda D_{\mathrm{KL}}(Q_{\mathbf U}\|P_0)
=\frac\lambda2\sum_{k=0}^{N-1}
\mathbf u_k^\top\Sigma^{-1}\mathbf u_k.
$$

Par exemple, choisissons $\lambda\Sigma^{-1}=hR$, où $h$ est la longueur de
l'intervalle et $R\succ0$ précise le coût de chaque direction d'entrée. Cette
pénalité vaut alors $\tfrac12 h\sum_k\mathbf u_k^\top R\mathbf u_k$, soit la
discrétisation de l'intégrale d'effort de commande
$\tfrac12\int\mathbf u(t)^\top R\mathbf u(t)\,dt$ pour des commandes
maintenues constantes sur chaque intervalle. Ainsi, au sein de cette famille
gaussienne, s'écarter de la référence de moyenne nulle entraîne un coût
quadratique sur les commandes nominales. On tient compte ainsi de l'effort de
commande sans l'ajouter à $S$. Modifier $\lambda$ à $\Sigma$ fixée change
cette pénalité ainsi que la concentration des poids.

La solution $P^\star$ attribue des probabilités à des suites d'entrées
complètes. Un correcteur a néanmoins besoin d'une suite nominale à stocker et
à exécuter. On obtient cette suite en approchant $P^\star$ par une gaussienne
dont la moyenne peut être stockée comme plan.

### Ajustement de la moyenne d'une gaussienne

La densité $p^\star(\mathbf V)=e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)/Z$
n'est en général ni gaussienne ni indépendante dans le temps. Pour approcher
cette loi, considérons la même famille gaussienne que notre échantillonneur,
avec une suite de moyennes candidate
$\mathbf M=(\mathbf m_0,\ldots,\mathbf m_{N-1})$ :

$$
q_{\mathbf M}(\mathbf V)
=\prod_{k=0}^{N-1}\mathcal N(\mathbf v_k;\mathbf m_k,\Sigma).
$$

Seule la suite des moyennes est ajustable; la covariance $\Sigma$ à chaque pas
reste fixe. Le second problème d'optimisation choisit la gaussienne qui
minimise sa divergence par rapport à $P^\star$ dans le sens suivant :

```{math}
:label: eq-mppi-gaussian-objective
\mathbf U^+
=\operatorname*{arg\,min}_{\mathbf M}
D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M}).
```

Ici, $P^\star$ est fixée, et la variable est une suite de moyennes de
dimension finie $\mathbf M$. Le développement de cet objectif sépare les deux
log-densités :

$$
D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M})
=\mathbb E_{P^\star}[\log p^\star(\mathbf V)]
-\mathbb E_{P^\star}[\log q_{\mathbf M}(\mathbf V)].
$$

La première espérance ne dépend pas de $\mathbf M$. Dans la seconde, la
normalisation gaussienne reste elle aussi fixe, car $\Sigma$ est fixée. La
substitution de la log-densité gaussienne donne donc

$$
D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M})
=C+\frac12\mathbb E_{P^\star}
\left[\sum_{k=0}^{N-1}
(\mathbf v_k-\mathbf m_k)^\top\Sigma^{-1}
(\mathbf v_k-\mathbf m_k)\right].
$$

La constante $C$ regroupe les termes de la divergence qui ne dépendent pas de
$\mathbf M$. Minimiser la divergence équivaut donc à minimiser la distance
quadratique espérée entre la moyenne candidate et une suite d'entrées tirée de
$P^\star$, les distances étant mesurées par $\Sigma^{-1}$. En supposant des
moments d'ordre deux finis, la dérivation donne

$$
\nabla_{\mathbf m_k}D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M})
=\Sigma^{-1}\left(\mathbf m_k-\mathbb E_{P^\star}[\mathbf v_k]\right).
$$

L'annulation de ce gradient donne l'unique minimiseur, puisque $\Sigma^{-1}$
est définie positive :

$$
\mathbf u_k^+=\mathbb E_{P^\star}[\mathbf v_k],
\qquad k=0,\ldots,N-1.
$$

Avec une solution exacte de @eq-mppi-gaussian-objective, la gaussienne mise à
jour aurait la même moyenne que $P^\star$. L'équation ci-dessus désigne cette
moyenne comme la quantité à calculer; elle ne fournit pas encore de moyen de
l'évaluer. Nous connaissons la densité de $P^\star$ à sa constante de
normalisation près : pour toute suite candidate $\mathbf V$, un déroulement
donne $S(\mathbf V)$, ce qui permet d'évaluer
$e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)$. En général, toutefois, le calcul de
$Z$ ou de $\mathbb E_{P^\star}[\mathbf v_k]$ exige une intégration en grande
dimension, et nous ne disposons pas d'un échantillonneur direct de $P^\star$.

L'échantillonnage préférentiel, à la @sec-mppi-importance, estimera cette
moyenne à partir de candidats tirés de la gaussienne $Q_{\mathbf U}$, selon
laquelle nous savons échantillonner. Les poids normalisés tiendront compte de
la différence entre les deux lois sans exiger la valeur de $Z$. La mise à jour
calculée n'approchera donc que la moyenne de $P^\star$; l'égalité ci-dessus
décrit la cible exacte du calcul. L'exemple quadratique ci-dessous fournit
d'abord un cas où cette cible peut être calculée analytiquement.

Même si sa moyenne était calculée exactement, la gaussienne conserverait la
covariance $\Sigma$ à chaque pas. Elle peut donc perdre des caractéristiques de
$P^\star$ : si les suites de faible coût forment des groupes distincts, leur
moyenne peut se situer entre ces groupes.

Le premier problème, @eq-mppi-distribution-objective, choisit une loi entière
en minimisant $\mathbb E_Q[S(\mathbf V)]+\lambda
D_{\mathrm{KL}}(Q\|P_0)$. Le second, @eq-mppi-gaussian-objective, garde fixe
sa solution $P^\star$ et choisit la moyenne d'une gaussienne en minimisant
$D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M})$. Si l'on restreignait plutôt le
premier problème aux gaussiennes, l'identité ci-dessus donnerait

$$
\operatorname*{arg\,min}_{\mathbf M}
\left\{\mathbb E_{Q_{\mathbf M}}[S(\mathbf V)]
+\lambda D_{\mathrm{KL}}(Q_{\mathbf M}\|P_0)\right\}
=\operatorname*{arg\,min}_{\mathbf M}
D_{\mathrm{KL}}(Q_{\mathbf M}\|P^\star).
$$

L'ordre des lois dans cette divergence est inversé par rapport à
@eq-mppi-gaussian-objective. La divergence KL n'est pas symétrique, si bien que
ces deux choix de moyenne gaussienne ne coïncident pas nécessairement
{cite:p}`Williams2017InformationTheoretic`.

### Une vérification quadratique à un pas

Pour un système scalaire $x_1=x_0+v$ avec $x_0=0$ et la cible $1$, prenons
$S(v)=\tfrac\kappa2(v-1)^2$ avec $\kappa>0$, $P_0=\mathcal N(0,1)$ et
$\lambda=1$. La dynamique est déterministe : des points d'arrivée différents
résultent de l'essai d'entrées différentes. En multipliant la densité de
référence par $e^{-S(v)}$ et en complétant le carré, on obtient

$$
p^\star(v)\propto
\exp\!\left[-\frac12\bigl(v^2+\kappa(v-1)^2\bigr)\right]
\propto\exp\!\left[-\frac{1+\kappa}{2}
\left(v-\frac\kappa{1+\kappa}\right)^2\right].
$$

Le premier problème d'optimisation a donc pour solution
$P^\star=\mathcal N(\kappa/(1+\kappa),1/(1+\kappa))$. La pondération a
déplacé la référence vers la cible et réduit sa variance. Le second problème
maintient la variance à la valeur choisie de un et n'ajuste que la moyenne de
la gaussienne :

$$
u^+=\operatorname*{arg\,min}_{m}
D_{\mathrm{KL}}\!\left(
\mathcal N\!\left(\frac{\kappa}{1+\kappa},\frac{1}{1+\kappa}\right)
\,\middle\|\,\mathcal N(m,1)\right)
=\frac{\kappa}{1+\kappa}.
$$

L'approximation qui en résulte est $Q_{u^+}=\mathcal N(\kappa/(1+\kappa),1)$.
Elle a la bonne moyenne, mais elle est plus étalée que $P^\star$. La
@fig-mppi-gaussian-mean montre les deux étapes d'optimisation pour
$\kappa=2\log2$.

:::{figure} _static/brownian_mppi/gaussian-mean.svg
:label: fig-mppi-gaussian-mean
:alt: Deux graphiques de densité partagent les mêmes échelles d'entrée et de densité de probabilité. À gauche, la multiplication d'une référence gaussienne de moyenne nulle par le poids de coût exponentiel produit une gaussienne plus étroite centrée vers 0.581. À droite, la gaussienne d'approximation partage cette moyenne mais conserve la variance de référence égale à un, si bien qu'elle est plus étalée que la loi pondérée par le coût.

Les deux problèmes d'optimisation de MPPI modifient des propriétés différentes
de la loi des entrées. Pour le déroulement scalaire $x_1=v$, la cible $1$, le
coût $S(v)=\tfrac\kappa2(v-1)^2$, $\kappa=2\log2$ et $\lambda=1$, la référence
est $P_0=\mathcal N(0,1)$. À gauche : minimiser le coût espéré plus
$D_{\mathrm{KL}}(Q\|P_0)$ donne la densité bleue en trait plein
$p^\star(v)\propto e^{-S(v)}p_0(v)$, de moyenne $u^+\approx0.581$ et de
variance $1/(1+\kappa)\approx0.419$. La courbe grise en tirets est la
référence. À droite : minimiser $D_{\mathrm{KL}}(P^\star\|Q_m)$ sur
$Q_m=\mathcal N(m,1)$ donne la densité orange en tirets-points $q_{u^+}$, de
même moyenne que $p^\star$ et de variance un. Les lignes verticales
pointillées marquent les moyennes. Les deux panneaux montrent des densités
analytiques, sans erreur d'échantillonnage.
:::

Dans cet exemple quadratique, $u^+$ minimise aussi l'objectif déterministe
$\tfrac\kappa2(u-1)^2+\tfrac12u^2$ : l'annulation de sa dérivée
$\kappa(u-1)+u$ donne $u=\kappa/(1+\kappa)$. Cette concordance vérifie la mise
à jour dans l'exemple; pour un déroulement général, prendre la moyenne de
$P^\star$ ne minimise pas nécessairement le coût déterministe.

Pour vérifier le calcul à échantillon fini, posons $\kappa=2\log2$ et partons
de $u=0$. Supposons que trois tirages selon la référence, arrondis pour ce
calcul, soient $-1,0,1$. L'évaluation de leurs coûts donne

| Entrée candidate $v$ | Point d'arrivée $x_1$ | Coût $S(v)$ | Poids non normalisé | Poids normalisé |
| --- | --- | --- | --- | --- |
| $-1$ | $-1$ | $4\log 2$ | $1/16$ | $1/25$ |
| $0$ | $0$ | $\log 2$ | $1/2$ | $8/25$ |
| $1$ | $1$ | $0$ | $1$ | $16/25$ |

La moyenne pondérée vaut

$$
u^+=\frac1{25}(-1)+\frac8{25}(0)+\frac{16}{25}(1)=\frac35.
$$

Cette estimation à trois échantillons, $0.6$, est proche de la moyenne exacte
$2\log2/(1+2\log2)\approx0.581$. Les trois candidats contribuent; la mise à
jour ne se contente pas de sélectionner la meilleure entrée échantillonnée,
$1$. Échantillonner autour du plan mis à jour produira un mélange différent de
candidats, si bien que conserver la même moyenne cible exige d'ajuster leurs
poids.

(sec-mppi-importance)=
### Corriger selon l'endroit où les échantillons ont été tirés

La moyenne $\mathbf u_k^+=\mathbb E_{P^\star}[\mathbf v_k]$ est la solution
exacte du problème d'approximation gaussienne. Pour l'estimer dans le cas d'un
déroulement général, nous disposons de deux opérations : tirer une suite selon
$Q_{\mathbf U}$, et évaluer son coût $S(\mathbf V)$. Ces opérations peuvent
s'utiliser sans échantillonneur direct de $P^\star$ ni valeur analytique de sa
constante de normalisation $Z$.

En substituant la définition de $p^\star$ dans l'espérance recherchée et en
écrivant $Z$ sous la forme de l'intégrale qui le définit, on obtient

$$
\mathbf u_k^+
=\frac{\int \mathbf v_k e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)\,d\mathbf V}
{\int e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)\,d\mathbf V}.
$$

Les deux intégrales font intervenir la densité non normalisée que nous savons
évaluer. En multipliant et en divisant leurs intégrandes par
$q_{\mathbf U}(\mathbf V)$, on les exprime comme des espérances sous la loi
selon laquelle nous savons échantillonner :

$$
\mathbf u_k^+
=\frac{\mathbb E_{Q_{\mathbf U}}
\left[e^{-S(\mathbf V)/\lambda}
\frac{p_0(\mathbf V)}{q_{\mathbf U}(\mathbf V)}\mathbf v_k\right]}
{\mathbb E_{Q_{\mathbf U}}
\left[e^{-S(\mathbf V)/\lambda}
\frac{p_0(\mathbf V)}{q_{\mathbf U}(\mathbf V)}\right]}.
$$

Ce changement de loi d'échantillonnage est l'**échantillonnage
préférentiel**. Un lot tiré selon $Q_{\mathbf U}$ peut estimer les deux
espérances, en utilisant le score
$e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)/q_{\mathbf U}(\mathbf V)$ pour
chaque candidat. La division par la somme de ces scores normalise les poids
des échantillons, de sorte qu'il n'est pas nécessaire de connaître $Z$.

Par exemple, déplacer une loi de proposition scalaire vers une moyenne
positive rend les entrées positives plus fréquentes, même si la loi cible n'a
pas changé. Une pondération par le seul coût d'état confondrait cette
fréquence d'échantillonnage supplémentaire avec une probabilité cible
supplémentaire. Le facteur $p_0/q_{\mathbf U}$ la supprime. Comme les deux
gaussiennes ont un support complet, toute trajectoire de densité cible
positive reste accessible. En écrivant
$\mathbf v_k=\mathbf u_k+\boldsymbol\epsilon_k$ et en développant leurs
exposants quadratiques, on obtient

$$
\begin{aligned}
\log\frac{p_0(\mathbf U+\boldsymbol\epsilon)}
{q_{\mathbf U}(\mathbf U+\boldsymbol\epsilon)}
&=-\frac12\sum_k\left[
(\mathbf u_k+\boldsymbol\epsilon_k)^\top\Sigma^{-1}
(\mathbf u_k+\boldsymbol\epsilon_k)
-\boldsymbol\epsilon_k^\top\Sigma^{-1}\boldsymbol\epsilon_k
\right]\\
&=-\sum_k\left[
\mathbf u_k^\top\Sigma^{-1}\boldsymbol\epsilon_k
+\frac12\mathbf u_k^\top\Sigma^{-1}\mathbf u_k\right].
\end{aligned}
$$

Le dernier terme quadratique est identique pour tous les échantillons et
disparaît des poids normalisés. Par conséquent, chaque suite échantillonnée $i$
reçoit le coût corrigé

$$
\widetilde S_i=S(\mathbf U+\boldsymbol\epsilon^{(i)})
+\lambda\sum_{k=0}^{N-1}
\mathbf u_k^\top\Sigma^{-1}\boldsymbol\epsilon_k^{(i)}.
$$

### Une mise à jour stable à échantillon fini

Le rapport d'espérances ci-dessus est exact. Pour calculer une mise à jour
approchée, on remplace les deux espérances par des moyennes sur $K$
échantillons indépendants tirés selon $Q_{\mathbf U}$. Pour la stabilité
numérique, on soustrait $\rho=\min_i\widetilde S_i$ avant l'exponentiation.
Cette soustraction commune laisse les poids normalisés inchangés et évite les
dépassements de capacité :

$$
w_i=\frac{e^{-(\widetilde S_i-\rho)/\lambda}}
{\sum_{j=1}^K e^{-(\widetilde S_j-\rho)/\lambda}},
\qquad
\mathbf u_k^+=\mathbf u_k+
\sum_{i=1}^K w_i\boldsymbol\epsilon_k^{(i)}.
$$

Le même poids de trajectoire complète met à jour chaque pas de temps. Ce
rapport de moyennes empiriques est un estimateur par échantillonnage
préférentiel autonormalisé. Il est en général biaisé pour $K$ fini et converge
vers la moyenne cible lorsque les moments pondérés pertinents sont
intégrables. Les itérations suivantes déplacent la loi de proposition vers des
trajectoires utiles; avec des espérances exactes, la moyenne cible serait déjà
indépendante de la loi de proposition.

Un déroulement quadratique rend explicite l'effet de cette correction. Prenons
$x_{k+1}=x_k+hv_k$, $x_0=0$, $N=10$, $h=0.1$ et le coût terminal
$2(x_N-1)^2$. Avec $\lambda=0.5$ et $\Sigma=5$, la moyenne cible exacte vaut
$0.8$ à chaque pas de temps. Échantillonner autour de $u_k=0.4$ en omettant la
correction donne plutôt l'estimation $0.88$, même avec un nombre arbitraire
d'échantillons. L'exercice @ex-mppi-likelihood-correction dérive ces deux
limites. Déplacer la loi de proposition devrait modifier la précision de
l'estimation, et non la moyenne estimée.

### Exécution à horizon glissant

La mise à jour donne une suite d'actions pour un état initial particulier.
L'erreur de modèle ou des perturbations physiques peuvent faire en sorte que
l'état observé suivant diffère de la prédiction du déroulement. Poursuivre avec
la suite stockée ignorerait cet écart. On applique plutôt seulement la première
action et on utilise l'état nouvellement observé comme condition initiale d'un
autre problème de planification. MPPI répète le calcul de trajectoire à partir
de cette condition, comme dans les correcteurs par échantillonnage de
{cite:t}`williams2017mppi`.

```{prf:algorithm} Commande prédictive par intégrale de chemin
:label: mppi

**Entrées :** état mesuré $\mathbf x$, horizon $N$, applications de
déroulement $\mathbf f_k$, coûts d'état $c_k$, coût terminal $c_N$, covariance
$\Sigma\succ0$, paramètre $\lambda>0$, nombre d'échantillons $K$, et un
nombre fini $I$ d'itérations d'échantillonnage par observation.

**Initialisation :** fixer la suite nominale $\mathbf U$ à zéro ou à une
estimation initiale disponible.

1. À chaque observation, répéter $I$ fois les étapes suivantes :
   1. Sauvegarder la suite nominale courante $\mathbf U$.
   2. Indépendamment pour $i=1,\ldots,K$ :
      1. Tirer $\boldsymbol\epsilon_k^{(i)}\sim\mathcal N(\mathbf0,\Sigma)$
         pour $k=0,\ldots,N-1$.
      2. Poser $\mathbf x_0^{(i)}=\mathbf x$ et $S_i=0$.
      3. Pour $k=0,\ldots,N-1$, accumuler
         $S_i\leftarrow S_i+c_k(\mathbf x_k^{(i)})$ et simuler
         $\mathbf x_{k+1}^{(i)}=
         \mathbf f_k(\mathbf x_k^{(i)},
         \mathbf u_k+\boldsymbol\epsilon_k^{(i)})$.
      4. Ajouter le coût terminal et la correction de vraisemblance :
         $\widetilde S_i=S_i+c_N(\mathbf x_N^{(i)})
         +\lambda\sum_k\mathbf u_k^\top\Sigma^{-1}
         \boldsymbol\epsilon_k^{(i)}$.
   3. Calculer $\rho=\min_i\widetilde S_i$ et normaliser
      $w_i\propto\exp[-(\widetilde S_i-\rho)/\lambda]$.
   4. Mettre à jour toutes les commandes simultanément :
      $\mathbf u_k\leftarrow\mathbf u_k+
      \sum_iw_i\boldsymbol\epsilon_k^{(i)}$.
2. Appliquer $\mathbf u_0$ pendant un intervalle de commande et obtenir
   l'observation de l'état suivant.
3. Décaler la suite restante d'une case vers l'avant, répéter sa dernière
   commande pour remplir la dernière case, et revenir à l'étape 1.
```

Cette version suppose des entrées gaussiennes non restreintes et des coûts de
déroulement finis. Les contraintes d'état non convexes exigent un traitement
supplémentaire : même une moyenne d'entrées issues de trajectoires réalisables
peut engendrer une trajectoire irréalisable. L'écrêtage des échantillons
modifie aussi leur densité, de sorte que le rapport de vraisemblance gaussien
affiché ne s'applique pas tel quel aux variables écrêtées. Une covariance
d'exploration choisie différente de la covariance de référence exige le
rapport de densités correspondant complet.

Les déroulements peuvent être simulés en parallèle. Leur efficacité dépend
néanmoins du recouvrement entre la loi de proposition et la loi cible. La
quantité $K_{\mathrm{eff}}=1/\sum_i w_i^2$ résume la concentration des poids :
elle vaut $K$ pour des poids égaux et tend vers un lorsqu'un seul déroulement
domine. Elle diagnostique la concentration de l'échantillonnage plutôt qu'elle
ne certifie l'optimalité ou le respect des contraintes. Une mise à jour à
échantillon fini peut augmenter le coût du déroulement, en particulier
lorsque quelques échantillons seulement portent l'essentiel du poids. Des poids
concentrés suggèrent d'augmenter le nombre d'échantillons ou d'améliorer le
recouvrement entre la loi de proposition et la loi cible. Augmenter $\lambda$
avec une référence fixe affaiblit la pondération par le coût d'état, mais
modifie aussi l'objectif d'optimisation. Modifier seulement la covariance de
proposition exige la correction de vraisemblance correspondante. L'exécution à
horizon glissant procure la rétroaction par des observations répétées; la
suite stockée elle-même est en boucle ouverte.

### Lien avec la planification par diffusion

La loi sur les plans de faible coût relie aussi ce calcul à l'optimisation de
trajectoires moderne par diffusion. Dans la diffusion fondée sur un modèle,
l'objectif définit une densité cible au moyen d'un poids de coût exponentiel.
Des perturbations gaussiennes et des moyennes pondérées par le coût
fournissent ensuite des estimations qui servent à raffiner les trajectoires
au fil d'une suite de niveaux de bruit
{cite:p}`Pan2024ModelBasedDiffusion`. La moyenne pondérée de MPPI constitue un
point de départ utile pour comprendre cette construction. La mise à jour MPPI
présentée ici utilise une covariance d'échantillonnage fixe; elle n'entraîne
pas de réseau de débruitage et n'exécute pas de calendrier de bruit de
diffusion. Dans les deux algorithmes, les perturbations aléatoires peuvent
servir uniquement à explorer les commandes. Les perturbations physiques
introduisent une source de variation distincte : un même plan candidat peut
produire des coûts différents sous différentes réalisations de la
perturbation. L'exemple de l'aéronef utilise plusieurs vents simulés pour
estimer le coût attribué à chaque candidat.

(sec-aircraft-mppi)=
## Planification d'un vol sous des vents incertains

Considérons un Airbus A320 qui relie Montréal–Trudeau (CYUL) à
Toronto–Pearson (CYYZ). La tâche consiste à choisir la montée, la croisière,
la descente et la durée du vol de manière à consommer peu de carburant tout en
arrivant à moins de 1 km horizontalement et de 30 m verticalement de la
destination. Un vent favorable à une certaine altitude peut justifier une
montée, mais son évolution ultérieure est incertaine. Comment la mise à jour
par commandes échantillonnées peut-elle tenir compte de ces différentes issues
possibles lorsqu'elle compare des vols candidats?

La méthode de la @sec-mppi-distribution peut optimiser un coût espéré estimé à
partir d'un simulateur. Elle exige des échantillons distincts pour les
commandes candidates et pour les vents futurs : plusieurs vents doivent être
évalués avant qu'un candidat reçoive son poids de coût. Le modèle suivant rend
explicites ces deux quantités aléatoires.

### Le modèle en vol et son état de vent

Dans le modèle de point matériel de l'A320 d'OpenAP {cite:p}`Sun2022`, l'état
est $(p_E,p_N,h,m,t)$ : la position est et nord dans une carte locale,
l'altitude en mètres, la masse en kilogrammes et le temps écoulé en secondes.
Ici, $h$ désigne l'altitude; les pas de temps numériques seront donnés en
secondes. Les commandes sont le nombre de Mach $M$, la vitesse verticale $v_z$
en mètres par seconde et le cap $\chi$ en radians, compté dans le sens horaire
à partir du nord. Si $v=\mathrm{TAS}(M,h)$ est la vitesse vraie et
$\gamma=\arctan(v_z/v)$, le modèle s'écrit

$$
\begin{aligned}
\dot p_E &= v\cos\gamma\sin\chi+\bar w_E(p_E,p_N,h)+g_E,\\
\dot p_N &= v\cos\gamma\cos\chi+\bar w_N(p_E,p_N,h)+g_N,\\
\dot h &= v_z,\qquad
\dot m=-f_{\mathrm{fuel}}(m,v,h,v_z),\qquad \dot t=1.
\end{aligned}
$$

Les équations horizontales séparent le mouvement relatif à l'air du
déplacement dû au vent : $\bar w$ est le champ de vent moyen fixe et $g$ est
une rafale. Le débit de carburant $f_{\mathrm{fuel}}$ réduit la masse à mesure
que l'aéronef vole. La convention de cap et l'approximation de l'angle de
trajectoire reproduisent les équations de point matériel de `OpenAP.top`. Les
conversions en nœuds, en pieds et en pieds par minute n'interviennent qu'à
l'appel des interfaces d'OpenAP qui les exigent. Les dérivées numériques sont
testées contre le modèle symbolique original de `OpenAP.top` en des états de
montée, de croisière et de descente. L'optimisation elle-même utilise la
simulation vers l'avant et l'échantillonnage gaussien.

Le vol de référence couvre l'intégralité de la phase en vol, avec les
conditions initiales et nominales suivantes :

| Grandeur | Valeur |
| --- | --- |
| Masse initiale | $85\%$ de la masse maximale au décollage |
| Altitude de départ et d'arrivée | $100$ pi |
| Altitude de croisière nominale | $7000$ m |
| Durée nominale | $3600$ s |

Le roulage au sol, les opérations sur piste et la circulation aérienne sont
hors du modèle. Le nombre de Mach, la vitesse verticale, le cap, l'altitude, la
masse et le couloir de vol géographique ont des bornes explicites. Des
inégalités sur la poussée, la portance et l'excédent d'énergie reproduisent
les vérifications approchées de l'enveloppe de `OpenAP.top`. Les variations
entre commandes adjacentes sont aussi vérifiées par rapport à des limites de
$0.2$ Mach, de $500$ pi/min en vitesse verticale et de $15$ degrés en cap. Ce
sont des contraintes du modèle pédagogique.

Le champ moyen $\bar w$ est l'instantané ERA5 aux niveaux de pression, versé
dans le dépôt, du 1er juin 2023 à 12 h 00 UTC
{cite:p}`ERA5PressureLevels2018`. Le prétraitement convertit une seule fois les
valeurs GRIB d'origine en un tableau numérique. Le simulateur interpole en
pression, en latitude et en longitude, en utilisant la pression de
l'atmosphère type internationale (ISA) à l'altitude de l'aéronef. Hors de la
plage de pression de l'instantané, de $200$ à $925$ hPa, il maintient la
valeur à la frontière; en particulier, les vents près du sol utilisent le plus
bas niveau de pression disponible. L'instantané reste fixe dans le temps
pendant un vol.

Une rafale observée maintenant devrait influencer la prévision à court terme,
mais elle ne devrait pas déterminer le vent pour tout le vol. Un modèle
d'Ornstein-Uhlenbeck représente cette persistance et cette perte graduelle de
prévisibilité pour $g=(g_E,g_N)$. Sa transition exacte sur un incrément de
temps $\Delta$ est

$$
g_{t+\Delta}=e^{-\Delta/\tau}g_t+
\sqrt{1-e^{-2\Delta/\tau}}\,
\begin{bmatrix}\sigma_E&0\\0&\sigma_N\end{bmatrix}\xi,
\qquad \xi\sim\mathcal N(0,I),
$$

avec $\tau=300$ s, $\sigma_E=5$ m/s et $\sigma_N=4$ m/s. Ces paramètres sont
des hypothèses illustratives, et non des estimations tirées de l'unique
instantané ERA5. La rafale est spatialement uniforme dans cet exemple. L'état
courant de l'aéronef et la rafale courante sont observés à chaque instant de
replanification. Conditionnellement à cette observation, la rafale future a
pour moyenne $e^{-\Delta/\tau}g_t$ et une variance conditionnelle croissante.
L'ajout de $g_t$ à l'état de l'aéronef en fait un modèle markovien
entièrement observé. L'exemple ne comporte aucun filtre d'observation.

### Coût espéré sous plusieurs vents futurs

Un vol candidat peut bien se comporter sous un vent et mal sous un autre.
Supposons que deux vents équiprobables donnent les coûts suivants :

| Candidat | Premier vent | Second vent | Coût espéré |
| --- | ---: | ---: | ---: |
| A | $0$ | $20$ | $10$ |
| B | $8$ | $8$ | $8$ |

Le coût espéré favorise B. Avec $\lambda=1$, toutefois, la moyenne des poids
exponentiels donne à A le score $(1+e^{-20})/2\approx0.5$, alors que B reçoit
$e^{-8}\approx0.00034$. Un seul vent favorable domine le poids de A. Faire la
moyenne de tels poids optimiserait donc un autre critère. L'exercice
@ex-mppi-aircraft-expected-cost approfondit cette distinction.

Pour l'objectif de coût espéré visé, on fait d'abord la moyenne des coûts sur
les vents futurs. Si $z^{(i)}$ précise un vol candidat et $W^{(\ell)}$ est
l'un de $L$ vents futurs conditionnels à la rafale observée, on utilise

$$
\widehat C(z^{(i)})=\frac1L\sum_{\ell=1}^L
 C\bigl(z^{(i)},W^{(\ell)}\bigr).
$$

Chaque candidat reçoit le même ensemble de vents futurs. L'appariement des
vents fait en sorte que chaque différence de coût compare deux plans dans les
mêmes conditions, ce qui peut réduire la variance de leur différence estimée.
L'ensemble est indépendant du vent futur qui se produira réellement.

Le coût $C$ est le carburant brûlé, auquel s'ajoutent des pénalités
quadratiques pour l'erreur d'arrivée horizontale et verticale et pour les
violations de l'enveloppe. L'erreur d'arrivée est mise à l'échelle par des
tolérances de $1000$ m horizontalement et de $30$ m verticalement. Une unité
au carré de l'une ou l'autre erreur d'arrivée mise à l'échelle ajoute un coût
de $80$ kg-équivalent; les violations de l'enveloppe reçoivent une pénalité
plus forte. Le code enregistre les erreurs physiques séparément de cet
objectif combiné.

### Paramètres de vol lisses et leurs poids

L'aéronef possède trois canaux de commande sur tout un vol. Perturber
indépendamment chaque court intervalle produirait une suite de commandes
grande et irrégulière. Un candidat est plutôt un vecteur
$z\in\mathbb R^{13}$ : un paramètre de durée et quatre nœuds intérieurs pour
chacun des trois canaux de commande. Les nœuds définissent de petites
perturbations linéaires par morceaux autour d'un vol de référence géométrique.
La référence nominale tient compte de l'état observé et du vent moyen
conditionnel; c'est sa simulation vers l'avant qui détermine le point
d'arrivée réel. La durée est multipliée par $e^{0.025z_0}$ et limitée à
$[0.90,1.12]$ fois la durée restante nominale. Cette famille restreinte garde
le premier exemple petit du point de vue du calcul. Elle n'engendre pas toutes
les trajectoires réalisables de l'A320.

La perturbation de vitesse verticale a une intégrale temporelle nulle, si bien
qu'elle préserve la variation d'altitude de la référence. C'est une
restriction de la paramétrisation des commandes. Le simulateur ne projette
jamais la position finale de l'aéronef sur la destination. Les perturbations
de Mach et de cap peuvent modifier la position d'arrivée, et chaque
proposition doit être vérifiée après simulation.

Utilisons une référence $p_0=\mathcal N(0,0.7^2I)$ et une loi de proposition
$q_\mu=\mathcal N(\mu,0.7^2I)$ sur les paramètres de nœuds et de durée $z$.
Les poids et la moyenne proposée sont

$$
w_i\propto
\exp\!\left[-\frac{\widehat C(z^{(i)})}{\lambda}\right]
\frac{p_0(z^{(i)})}{q_\mu(z^{(i)})},
\qquad \mu^+=\sum_iw_i z^{(i)},\qquad \lambda=25\ \text{kg-équivalent}.
$$

Le rapport de vraisemblance est calculé avant l'application déterministe qui
associe aux paramètres une durée bornée et des profils de commande.
L'écrêtage de la durée ne fait pas de sa loi physique une gaussienne; la
densité gaussienne demeure une densité sur $z$. La correction de la loi de
proposition apparaît une fois par candidat, après la moyenne sur les
perturbations.

Il y a désormais deux nombres d'échantillons aux rôles différents : $K$
vecteurs de paramètres candidats approchent la moyenne pondérée, tandis que
$L$ vents futurs approchent le coût espéré de chaque candidat. La référence
sur les paramètres régularise les modifications du vol. Elle ne décrit pas les
rafales physiques, qui agissent sur le mouvement par rapport au sol et ont
leur propre dépendance temporelle.

La mise à jour utilise la construction sur les lois de la
@sec-mppi-distribution, avec $\widehat C$ comme score. Même si les coûts
espérés étaient évalués exactement, l'ajustement de la moyenne d'une
gaussienne ne résoudrait pas en général l'optimisation déterministe sur les
paramètres de vol. Le budget d'échantillonnage fini introduit une autre
approximation, de sorte que le plan proposé doit être vérifié avant son
exécution.

### Accepter et exécuter un plan

La prédiction utilise des intervalles de commande de 60 s. Avant d'accepter
une moyenne pondérée, le code construit son profil de commande et le rejoue
sur une grille de 10 s sous le vent moyen conditionnel. Il vérifie l'enveloppe
et les deux tolérances d'arrivée. Le plan proposé doit aussi améliorer le coût
estimé sur l'ensemble de vents commun. Le meilleur plan accepté jusqu'ici est
le plan en place. Une mise à jour invalide ou qui n'apporte pas d'amélioration
laisse ce plan validé en place. La référence géométrique initiale doit passer
la même validation. Lors d'une décision ultérieure, le reste du plan précédent
est aussi rejoué à partir de l'état observé et peut être conservé comme plan
en place. La validation comprend la variation entre la dernière commande
exécutée et la nouvelle première commande. Si aucune des deux références n'est
réalisable, la simulation signale un échec de planification.

Ces vérifications ne certifient la faisabilité que sous la moyenne
conditionnelle et à la résolution vérifiée. Les évaluations de scénarios
révèlent les violations possibles d'arrivée et de contraintes sous d'autres
vents. Ni une moyenne pondérée ni une collection finie de scénarios réussis ne
garantit la faisabilité pour toutes les perturbations gaussiennes.

### Comparer des vols figés et replanifiés

L'expérience pose deux questions distinctes : réagir au vent observé
améliore-t-il l'arrivée, et l'échantillonnage du vent futur incertain
fait-il mieux que sa moyenne conditionnelle? Trois stratégies d'exécution
utilisent des réalisations physiques de vent appariées :

1. **Plan figé :** calculer un plan pour tout le trajet à l'aide du vent moyen
   au départ et l'exécuter pendant le reste du vol.
2. **Replanification sur la moyenne conditionnelle :** observer l'état et la
   rafale toutes les 60 s, optimiser le reste du trajet à l'aide de la rafale
   moyenne conditionnelle et exécuter la minute de commandes suivante.
3. **Replanification stochastique :** utiliser les mêmes observations et le
   même calendrier d'exécution, mais faire la moyenne du coût de chaque
   candidat sur un ensemble conditionnel de vents futurs avant de mettre à
   jour ses paramètres.

Le vol enregistré sous vent moyen est la référence déterministe validée qui
sert à construire le plan figé. Les deux variantes de replanification
construisent une référence géométrique à partir de l'état mesuré, la
comparent au reste du plan précédent, puis effectuent deux mises à jour de la
moyenne gaussienne. Elles réutilisent le vol précédent comme candidat au rôle
de plan en place, tout en réinitialisant les coordonnées de perturbation autour
de la nouvelle référence géométrique. Au sein d'une même décision, chaque
moyenne acceptée initialise la mise à jour d'échantillonnage suivante. Étendre
le démarrage à chaud par suite décalée de l'algorithme @mppi pour réutiliser
les coordonnées de nœuds précédentes constitue une autre expérience de calcul.

La trajectoire de rafale réalisée possède son propre flux de nombres
aléatoires. L'optimiseur ne reçoit que la rafale observée courante et génère
des prévisions conditionnelles indépendantes. Tous les candidats d'une même
décision partagent l'ensemble de prévisions, ce qui réduit le bruit dans leurs
différences de coût. Les correcteurs rencontrent la même fonction de rafale
réalisée sous-jacente même lorsque leurs durées diffèrent. Les rafales
réalisées sont maintenues sur chaque intervalle d'exécution de 10 s; un rejeu
à 5 s vérifie l'intégration numérique tout en préservant ces entrées
réalisées.

:::{figure} _static/aircraft_mppi/flight_comparison.svg
:label: fig-aircraft-mppi-profiles
:width: 100%

Profils de vol complets pour la première réalisation de rafales appariée. Le
plan figé exécute les mêmes commandes que la référence sous vent moyen, si bien
que les deux brûlent la même quantité de carburant dans ce modèle, mais leurs
positions au sol diffèrent. La replanification modifie les commandes en
réponse à la position et à la rafale observées. La ligne horizontale du
panneau d'approche finale marque la tolérance d'arrivée de 1 km.
:::

### Arrivée, carburant et sensibilité numérique

Dans les trois réalisations de vent enregistrées, le plan figé manque chaque
fois la tolérance d'arrivée. Les deux méthodes de replanification arrivent
dans la tolérance lors des trois essais. Leur différence de carburant est plus
petite que la distinction entre atteindre et manquer la destination; l'arrivée
et le carburant sont donc présentés séparément :

:::{include} artifacts/aircraft_mppi/results.md
:::

:::{figure} _static/aircraft_mppi/metrics.svg
:label: fig-aircraft-mppi-metrics
:width: 100%

Résultats individuels sur trois réalisations de rafales indépendantes, avec
les barres de la moyenne et de l'écart type empirique. Le petit nombre
d'essais permet d'examiner ces exemples; il n'établit ni un avantage fiable en
carburant ni une probabilité d'échec. De minuscules résidus proches de la
précision machine proviennent de l'intégration en virgule flottante à la
frontière d'altitude terminale.
:::

La replanification compense le déplacement accumulé sous le plan figé. Que
l'échantillonnage des rafales futures améliore aussi la consommation de
carburant dépend du modèle de vent, de la famille de trajectoires et du budget
d'échantillonnage. La replanification sur la moyenne conditionnelle constitue
donc une référence de comparaison sérieuse. Une seule exécution favorable ne
suffit pas à prouver que le planificateur stochastique est meilleur.

:::{figure} _static/aircraft_mppi/wind_uncertainty.svg
:label: fig-aircraft-mppi-uncertainty
:width: 100%

À gauche : l'intervalle prédit entre les 10e et 90e centiles de l'erreur
terminale, tiré de l'ensemble de planification courant. Le planificateur sur
la moyenne conditionnelle n'a qu'un seul futur déterministe, si bien que sa
bande est de largeur nulle. À droite : points d'arrivée des scénarios
conditionnels à un instant de replanification, avec la région d'arrivée de
1 km. Ces intervalles tirés de petits ensembles sont des diagnostics de
prévision, et non des régions de confiance étalonnées ni des contraintes.
:::

Augmenter le nombre d'échantillons candidats et augmenter le nombre de
scénarios de vent agissent sur des approximations différentes. Des nombres de
candidats de $16,32,64,128$ sont comparés avec $L=6$; des nombres de scénarios
de $1,3,6,12,24$ sont comparés avec $K=32$. Quatre germes de planification
indépendants sont utilisés à la décision initiale pour le trajet complet.
Chaque plan obtenu est évalué sur 128 vents futurs nouveaux et communs
qu'aucun optimiseur n'a vus. Cela sépare la variation due à la recherche des
candidats de la variation du coût espéré estimé. Deux mises à jour projetées
ne s'améliorent pas nécessairement de façon monotone avec l'un ou l'autre
nombre d'échantillons, et cette vérification à la première décision n'établit
pas la convergence de la politique de rétroaction entière.

Les comparaisons aux extrémités du balayage enregistré sont :

| Budget varié | Budget fixé | Coût moyen hors échantillon au plus petit budget | Coût moyen hors échantillon au plus grand budget |
| --- | --- | ---: | ---: |
| $K:16\to128$ | $L=6$ | $57\,396$ | $50\,110$ |
| $L:1\to24$ | $K=32$ | $58\,353$ | $51\,319$ |

Les coûts sont exprimés en kg-équivalent et comprennent de fortes pénalités
pour de rares vents futurs dans lesquels le vol échoue. Ce ne sont pas des totaux
de carburant. Dans la comparaison du nombre de candidats, les écarts types
empiriques entre germes de planification sont de $10\,008$ et $7\,338$. Les
budgets intermédiaires donnent des résultats non monotones; ces extrémités
montrent donc la sensibilité aux budgets d'échantillonnage sans établir la
convergence.

Les {download}`diagnostics de vol <artifacts/aircraft_mppi/metrics.json>`
comprennent le carburant, les erreurs d'arrivée, les résidus sur grille fine,
les tailles d'échantillon effectives, les mises à jour acceptées et rejetées,
les temps d'exécution et l'étude d'échantillonnage. Le rejeu local affiche les
trajectoires et les observations correspondantes. Pour prolonger l'expérience,
augmentez le nombre de vents réalisés indépendants avant de tirer des
conclusions sur la fiabilité, et comparez le gain d'une paramétrisation des
commandes plus riche avec le gain d'un plus grand nombre d'échantillons.

:::{iframe} ../interactive/aircraft-mppi.html
:width: 100%
:title: Rejeu du vol stochastique Montréal–Toronto
:class: mppi-replay

Choisissez un correcteur et rejouez le vol enregistré. La prévision affichée
correspond à la décision de planification la plus récente, tandis que
l'affichage des rafales suit le vent réalisé observé.
:::

(sec-reactor-mppi)=
## Commande des éléments chauffants d'un réacteur à écoulement continu

L'état de l'aéronef comprenait quelques grandeurs physiques et une rafale. De
nombreux systèmes de procédés sont plutôt décrits par des champs : la
température et la composition en chaque point d'une enceinte. Dans un
calcinateur flash, des particules d'argile transportées par un gaz chaud sont
chauffées rapidement pendant qu'une réaction endothermique les transforme en
un matériau réactif qui peut remplacer une partie du clinker dans le ciment
{cite:p}`Cantisani2024FlashCalciner`. Les éléments chauffants agissent sur de
larges régions d'un tel réacteur, alors que les grandeurs qui importent, comme
la conversion du matériau qui en sort et la température locale la plus
élevée, dépendent de la répartition de la chaleur le long de l'écoulement.

MPPI n'a besoin que de pouvoir simuler une suite de commandes candidate et
évaluer son coût, et une équation aux dérivées partielles (EDP) discrétisée
fournit les deux. Le nombre de valeurs d'état n'intervient pas directement
dans la mise à jour : les poids dépendent des coûts des candidats, et la mise
à jour fait la moyenne de paramètres de commande. La dimension de l'état
détermine le coût de chaque déroulement et la part de l'état que quelques
actionneurs peuvent influencer. Cette section commande un réacteur à 4 096
valeurs d'état au moyen de 24 paramètres échantillonnés, d'abord en planifiant
avec l'EDP elle-même, puis avec un modèle de substitution appris.

### Un modèle de réacteur à deux champs

Le matériau s'écoule de gauche à droite dans un domaine rectangulaire de
longueur $8$ m et de largeur $2$ m, $x$ étant mesuré vers l'aval et $y$ dans
la largeur. Deux champs décrivent son état : la température $T(x,y,t)$ en
kelvins et la fraction de réactif restante $c(x,y,t)\in[0,1]$. La conversion
est $a=1-c$. Les deux champs sont transportés par une vitesse vers l'aval
prescrite $v_x(y)$, étalés par diffusion et couplés par une réaction qui
dépend de la température :

$$
\begin{aligned}
\partial_t c+v_x(y)\,\partial_x c&=D\nabla^2c-k(T)\,c,\\
\partial_t T+v_x(y)\,\partial_x T&=D\nabla^2T-\beta(T-T_{\mathrm{wall}})
-\Delta T_{\mathrm{rxn}}\,k(T)\,c
+q_{\max}\sum_{j=1}^4u_j\,\mathbb 1_{\Omega_j}(x,y).
\end{aligned}
$$

La vitesse de réaction est de la forme d'Arrhenius
$k(T)=0.1\exp[12000(1/950-1/T)]$ s$^{-1}$. La conversion d'une unité de
réactif refroidit le matériau de $\Delta T_{\mathrm{rxn}}$, et la chaleur est
perdue vers la température de paroi au taux $\beta$. Chaque élément chauffant
couvre un quadrant $\Omega_j$ du domaine, et son entrée $u_j\in[0,1]$ est une
fraction du taux de chauffage maximal $q_{\max}$. L'élément H1 couvre la
moitié amont du bas ($y<1$ m), H2 la moitié aval du bas, H3 la moitié amont du
haut et H4 la moitié aval du haut. Le matériau entre avec une fraction de
réactif égale à un.

| Grandeur | Valeur |
| --- | --- |
| Domaine et maillage | $8$ m $\times$ $2$ m, $64\times32$ cellules |
| Diffusivité $D$ des deux champs | $5\times10^{-4}$ m$^2$/s |
| Taux de perte de chaleur $\beta$ et température de paroi | $0.01$ s$^{-1}$, $700$ K |
| Refroidissement réactionnel $\Delta T_{\mathrm{rxn}}$ | $60$ K par unité de réactif |
| Taux de chauffage maximal $q_{\max}$ | $20$ K/s |
| Température d'entrée | $800+10\cos(\pi y/2)$ K |
| Limites d'exploitation | $\max T\leq1070$ K, conversion en sortie $\geq95\%$ |

Ce sont des hypothèses pédagogiques inspirées de la calcination flash, et non
des paramètres étalonnés de calcinateur. Le modèle omet la séparation des
phases solide et gazeuse, la chimie détaillée, les distributions de temps de
séjour et la géométrie industrielle.

Le facteur d'Arrhenius rend la conversion sensible à la température. À
$950$ K, la vitesse est de $0.1$ s$^{-1}$, soit une constante de temps de
$10$ s. À $900$ K, elle tombe à environ $0.050$ s$^{-1}$, et à $1000$ K elle
monte à environ $0.19$ s$^{-1}$. Le matériau passe environ $40$ s dans le
réacteur, si bien qu'une parcelle qui reste $50$ K plus froide que son
voisinage pendant une partie de ce temps sort avec nettement plus de matériau
n'ayant pas réagi. Chauffer davantage augmente la vitesse, mais augmente aussi
la température maximale, qui doit rester sous $1070$ K partout. Comme la
vitesse d'écoulement peut varier dans la largeur, la conversion en sortie est
pondérée par le flux,

$$
a_{\mathrm{out}}=1-\frac{\sum_i v_x(y_i)\,c(8,y_i)}{\sum_i v_x(y_i)},
$$

où les sommes portent sur les cellules de sortie. Elle doit rester d'au moins
$95\%$.

Une discrétisation en volumes finis à $64\times32$ cellules transforme l'EDP
en $2\times64\times32=4\,096$ équations différentielles ordinaires.
L'advection utilise des flux décentrés amont du premier ordre; la diffusion
utilise des flux centrés, sans flux diffusif à travers les frontières. Une
méthode de Runge-Kutta explicite à deux étages qui préserve la stabilité forte
fait avancer le temps par sous-pas d'au plus $0.1$ s. Les entrées des éléments
chauffants sont maintenues pendant $\Delta t=2$ s, et l'intégration sur un
intervalle de commande définit la transition

$$
X_{k+1}=F(X_k,u_k),\qquad X_k\in\mathbb R^{4096},\quad u_k\in[0,1]^4,
$$

qui joue le rôle de l'application de déroulement $\mathbf f_k$ de
l'algorithme @mppi. Le maillage partitionne l'espace physique; ce n'est pas
une grille sur l'espace d'état de dimension 4 096. Le raffiner rend chaque
déroulement plus coûteux sans changer le nombre d'entrées des éléments
chauffants.

Deux profils d'écoulement apparaissent ci-dessous, tous deux de vitesse
moyenne $0.2$ m/s. Le problème de rétablissement du champ utilise un
écoulement cisaillé, plus rapide au centre du canal,
$v_x(y)=0.16+0.24\,\eta(1-\eta)$ m/s avec $\eta=y/2$. Le matériau met alors
environ $36$ s à traverser le réacteur au centre et près de $50$ s près des
parois, si bien qu'une perturbation s'incline et s'étire en se déplaçant.
L'expérience avec modèle de substitution utilise un écoulement uniforme à
$0.2$ m/s.

### Rétablir le champ d'exploitation

Maintenir les quatre éléments chauffants à $0.46$ pendant $400$ s sous
l'écoulement cisaillé produit un état d'exploitation stationnaire. Sa
température monte d'environ $800$ K à l'entrée jusqu'à un maximum de
$1063.54$ K, et sa conversion en sortie est d'environ $95.8\%$. Notons ses
champs $T_\star$ et $a_\star$. À partir de cet état, une poche froide entre
par la partie inférieure de l'entrée :

$$
\Delta T_{\mathrm{in}}(y,t)=-100\exp\!\left[-\frac{(y-0.55)^2}{2(0.22)^2}\right]
\sin^2\!\left(\frac{\pi(t-6)}{8}\right)\ \mathrm K,
\qquad 6\leq t\leq14\ \mathrm s,
$$

et $\Delta T_{\mathrm{in}}=0$ sinon. La commande commence à $14$ s, lorsque la
poche entière est entrée dans le réacteur et traverse la moitié inférieure, où
agissent H1 et H2. Si les éléments chauffants restent à $0.46$, le matériau
plus froid réagit trop lentement : la conversion en sortie tombe à $94.75\%$ et
reste sous la limite de $95\%$ pendant $7.5$ s.

La tâche consiste à ramener les deux champs à l'état d'exploitation tout en
respectant les limites. On sait que l'entrée est revenue à son profil normal,
les deux champs sont mesurés exactement à chaque décision, et l'EDP est
déterministe. Comme dans [la dérivation de la mise à jour](#sec-mppi-distribution),
l'aléa n'intervient que par les commandes candidates.

L'écart à l'état d'exploitation est mesuré par l'erreur de champ normalisée

$$
E(X)=\frac1{2048}\sum_{\text{cellules}}\left[
\left(\frac{T-T_\star}{20\ \mathrm K}\right)^2+
\left(\frac{a-a_\star}{0.1}\right)^2\right].
$$

Une cellule trop froide de $20$ K contribue autant qu'une cellule dont la
conversion est trop faible de $0.1$. L'erreur quadratique pénalise le
réchauffement autant que le refroidissement, si bien que la chaleur apportée
au matériau situé hors de la poche froide est comptée contre un plan. Chaque
élément chauffant agit sur un quadrant de $512$ cellules, et quatre entrées de
ce genre ne peuvent pas fixer indépendamment les $4\,096$ valeurs d'état. Une
partie de l'erreur doit subsister jusqu'à ce que le matériau froid quitte le
réacteur.

MPPI planifie sur $N=20$ intervalles, soit un horizon de $40$ s égal au temps
de séjour moyen. En notant $u_{-1}$ la commande appliquée pendant l'intervalle
précédent, l'objectif de planification est

$$
J=\frac1{20}\sum_{k=0}^{19}\left[E(X_{k+1})
+\frac{0.01}{4}\sum_{j=1}^4\left(\frac{u_{k,j}-0.46}{0.05}\right)^2
+\frac{0.005}{4}\sum_{j=1}^4\left(\frac{u_{k,j}-u_{k-1,j}}{0.05}\right)^2\right]
+E(X_{20}).
$$

Le deuxième terme décourage les écarts au chauffage nominal, et le troisième
décourage les variations brusques d'un intervalle à l'autre. Plus tôt dans le
chapitre, l'effort de commande n'intervenait que par la pénalité KL et $S$ ne
contenait que des coûts d'état. Ici, $J$ joue le rôle de $S$ et comprend des
termes de commande exprimés dans les unités physiques des éléments chauffants.
La dérivation de $P^\star$ utilisait seulement le fait que $S$ attribue un
coût à chaque suite candidate complète; ces termes modifient donc la loi cible
sans modifier la mise à jour.

Les limites interviennent de deux façons. Pendant l'échantillonnage, un
candidat dont les états aux instants de décision espacés de $2$ s dépassent
$1070$ K en un point quelconque, ou font passer la conversion en sortie sous
$95\%$, reçoit un coût infini et donc un poids nul. Dans l'objectif sur les
lois, cela restreint la référence aux suites qui passent la vérification. Cet
ensemble n'est pas nécessairement convexe, de sorte que la moyenne de la loi
de Gibbs qui en résulte ne respecte pas nécessairement les limites, comme dans
l'exercice @ex-mppi-execution. Chaque mise à jour proposée exige donc sa propre
vérification avant de pouvoir remplacer le plan stocké.

### Échantillonner dans un espace de commandes borné

Un plan contient $20\times4=80$ valeurs de chauffage. Les échantillonner
indépendamment produirait des calendriers irréguliers et gaspillerait des
échantillons sur des fluctuations rapides que pénalise le troisième terme de
$J$. Comme dans l'exemple de l'aéronef, un candidat est décrit par quelques
paramètres lisses : six nœuds par élément chauffant, interpolés linéairement
aux $20$ instants de décision. On obtient ainsi $24$ paramètres échantillonnés,
regroupés dans un tableau $z$ de taille $6\times4$.

Les entrées des éléments chauffants doivent rester dans $[0,1]$, alors que les
échantillons gaussiens ne sont pas bornés. L'écrêtage modifierait la loi des
valeurs écrêtées, comme on l'a noté après l'algorithme @mppi. Une application
logistique envoie plutôt des paramètres latents sur les valeurs physiques des
nœuds,

$$
u_{\mathrm{knots}}(z)=\operatorname{sigmoid}\bigl(\operatorname{logit}(u_{\mathrm{seed}})+0.12\,z\bigr),
$$

appliquée composante par composante. Le tableau $u_{\mathrm{seed}}$ contient le
plan stocké évalué aux six instants des nœuds, de sorte que $z=0$ reproduit le
plan stocké à l'erreur d'interpolation près. La dérivée de la sigmoïde est
$u(1-u)$, qui vaut environ $0.25$ au voisinage de $u=0.46$. Une variation
latente de $0.7$, soit un écart type de la loi d'échantillonnage ci-dessous,
déplace donc un nœud d'environ $0.12\times0.7\times0.25\approx0.021$, soit
deux points de pourcentage de chauffage.

Les densités gaussiennes sont définies sur $z$. La référence est
$p_0=\mathcal N(0,0.49I)$ et la loi de proposition est
$q_\mu=\mathcal N(\mu,0.49I)$. À chaque décision, le décodeur et la référence
restent fixes, tandis que $\mu$ part de zéro. Chacune des quatre mises à jour
tire $K=256$ candidats $z^{(i)}\sim q_\mu$, les décode, les simule dans l'EDP
et calcule

$$
w_i\propto\exp\!\left[-\frac{J(z^{(i)})}{\lambda}\right]
\frac{p_0(z^{(i)})}{q_\mu(z^{(i)})},
\qquad
\mu_{\mathrm{weighted}}=\sum_{i=1}^Kw_i\,z^{(i)},
\qquad \lambda=0.01.
$$

Il s'agit de la [mise à jour corrigée par échantillonnage préférentiel](#sec-mppi-importance)
en coordonnées latentes, avec $\Sigma=0.49I$ et la moyenne $\mu$ à la place
des commandes nominales. La température de pondération $\lambda$ est sans
dimension, comme $J$, et n'a aucun rapport avec la température en kelvins.
Comme le zéro latent se décode en le plan stocké, la pénalité KL impute un
coût à la modification de ce plan, et non au chauffage en tant que tel. C'est
l'option, [signalée avec la loi de référence](#sec-mppi-distribution), qui
consiste à centrer la référence sur une suite de base, le plan précédent
jouant ici ce rôle. La moyenne est prise en coordonnées latentes, de sorte que
le calendrier de chauffage décodé diffère en général de la moyenne pondérée
des calendriers physiques des candidats.

La température fixe les différences de coût qui comptent. À la première
décision, le plan stocké a $J=0.0488$. Avant correction, un candidat dont le
coût dépasse celui d'un autre de $\lambda\log2\approx0.0069$, soit environ
$14\%$ de cette valeur, reçoit la moitié du poids.

La correction de vraisemblance modifie sensiblement les poids dès que $\mu$
s'est déplacée. À la quatrième mise à jour de la première décision, après
trois mises à jour acceptées, le candidat 12 avait un coût de $0.05649$ et le
candidat 180 un coût de $0.05840$. La pondération exponentielle par le coût à
elle seule donnerait au candidat 12 environ $e^{0.191}\approx1.21$ fois le
poids du candidat 180. Son poids enregistré, $0.0682$, était au contraire
inférieur à la moitié de celui du candidat 180, $0.1565$. Ce rapport
correspond à des coûts corrigés qui diffèrent de
$-\lambda\log(0.0682/0.1565)\approx0.0083$ en faveur du candidat 180, si bien
que le terme de correction contribue pour environ $0.0102$. Le candidat 12 se
trouve plus loin dans la direction vers laquelle la loi de proposition s'était
déjà déplacée, là où elle échantillonne plus souvent que la référence.
L'exercice @ex-mppi-reactor-latent détaille ce calcul.

### Valider et exécuter une mise à jour échantillonnée

Chaque mise à jour se termine par la vérification de la moyenne proposée. Le
correcteur décode $\mu+\alpha(\mu_{\mathrm{weighted}}-\mu)$ pour les fractions
de pas $\alpha\in\{1,\tfrac12,\tfrac14,\tfrac18\}$ successivement, et accepte
la première qui fait passer $J$ sous la valeur du plan stocké et qui réussit
une vérification plus fine. Cette vérification enregistre l'état de l'EDP
toutes les $0.25$ s sur le plan de $40$ s, suivi de $40$ s supplémentaires au
chauffage nominal, et teste les deux limites à chaque enregistrement. Si aucune
fraction de pas ne réussit, $\mu$ et le plan stocké restent tous deux
inchangés.

Après quatre mises à jour, le correcteur applique à l'EDP les $2$ premières
secondes du plan stocké, décale le plan d'un intervalle et ajoute le chauffage
nominal à la fin. Le plan stocké est la suite exacte de chauffage qui a réussi
la validation; son approximation à six nœuds $u_{\mathrm{seed}}$ ne sert qu'à
centrer le cycle d'échantillonnage suivant. Comme le procédé est la même EDP
déterministe avec une entrée connue, l'exécution de la première action
reproduit l'état prédit. Le plan décalé suivi du chauffage nominal répète
alors une trajectoire qui a déjà réussi la vérification fine, à l'exception de
ses $2$ dernières secondes. C'est la queue de validation au chauffage nominal
qui rend cette continuation disponible : si toutes les mises à jour
ultérieures sont rejetées, le correcteur détient encore un plan vérifié. La
[commande à horizon glissant](receding-horizon-control.md) développe cette
propriété, la faisabilité récursive, avec des conditions terminales à la place
d'une queue simulée.

La procédure a besoin d'un plan réalisable à la première décision. Le
chauffage inchangé ne convient pas, puisqu'il viole la limite de conversion.
L'expérience fournit un calendrier initial qui maintient H1 à $0.50$ pendant
$8$ s et le ramène à $0.46$ au cours des $8$ s suivantes, les autres éléments
chauffants restant au nominal. Ce calendrier est fourni au correcteur, et non
trouvé par lui; MPPI part de ce calendrier et le révise à chaque décision.

### Rétablissement enregistré

Les trois calendriers partent du même état perturbé à $14$ s et sont
enregistrés jusqu'à $80$ s sur une grille de sortie de $0.25$ s. L'erreur de
champ intégrée est $\int_{14}^{80}E(X(t))\,dt$, et l'énergie de chauffage est
l'intégrale de la fraction de chauffage moyenne sur le même intervalle. Les
deux se mesurent en secondes; l'énergie de chauffage compte des secondes de
chauffage moyen à pleine puissance, puisque le modèle n'a pas de capacité
thermique permettant de la convertir en joules.

| Calendrier | Erreur de champ intégrée | Énergie de chauffage | Température maximale | Conversion en sortie la plus basse |
| --- | ---: | ---: | ---: | ---: |
| Chauffage inchangé | $3.709$ s | $30.360$ s | $1063.54$ K | $94.747\%$ |
| Calendrier initial fourni, exécuté tel quel | $2.098$ s | $30.484$ s | $1066.94$ K | $95.271\%$ |
| MPPI | $1.962$ s | $30.439$ s | $1066.16$ K | $95.329\%$ |

Le calendrier fourni élimine à lui seul $43\%$ de l'erreur intégrée du
chauffage inchangé et rétablit le respect de la limite de conversion. MPPI en
élimine $47.1\%$, soit $6.5\%$ d'erreur de moins que le calendrier fourni,
tout en utilisant un peu moins de chaleur que ce calendrier et en restant
$3.8$ K sous la limite de température.

:::{figure} _static/thermal_reactor/recovery-storyboard.svg
:label: fig-reactor-recovery-fields
:width: 100%
:alt: Quatre rangées de cartes du réacteur à 22, 34 et 46 secondes montrent l'erreur de température et de conversion par rapport à l'état d'exploitation. Une région froide bleue dans la moitié inférieure se déplace d'environ 2.5 mètres jusqu'à la sortie. Sous MPPI, la région bleue est plus pâle et entourée d'un léger réchauffement rouge à 22 secondes.

Écarts à l'état d'exploitation à $22$, $34$ et $46$ s. Les deux premières
rangées montrent l'erreur de température sous chauffage inchangé et sous
MPPI; les deux dernières rangées montrent l'erreur de conversion. Le bleu
indique un matériau plus froid ou une conversion plus faible. Les lignes
pointillées délimitent les quadrants des éléments chauffants. Le déficit est
transporté vers l'aval et incliné par l'écoulement cisaillé. MPPI le réduit,
tandis que chaque élément chauffant réchauffe aussi le matériau autour de la
poche.
:::

Les commandes de la @fig-reactor-recovery-control suivent la poche. MPPI
augmente H1 d'environ cinq points de pourcentage pendant que la poche traverse
le quadrant inférieur amont, puis le ramène au nominal vers $32$ s. Entre
environ $36$ et $62$ s, il abaisse H2, jusqu'à $0.8$ point de pourcentage vers
$48$ s, pendant que le matériau réchauffé par H1 traverse le quadrant inférieur
aval; l'erreur quadratique pénalise cet excès de chaleur. Les éléments
chauffants du haut varient de moins d'un point de pourcentage. La conversion
en sortie fléchit encore au départ de la poche, mais sa valeur la plus basse
est de $95.33\%$ au lieu de $94.75\%$.

:::{figure} _static/thermal_reactor/recovery-control.svg
:label: fig-reactor-recovery-control
:width: 100%
:alt: Quatre panneaux couvrant de 14 à 80 secondes. L'erreur de champ normalisée diminue le plus vite pour MPPI et pour le calendrier fourni. La conversion en sortie ne passe sous 0.95 que sous chauffage inchangé. La température maximale reste sous la limite de 1070 kelvins pour tous les calendriers. Le panneau des éléments chauffants montre H1 environ 0.05 au-dessus du nominal au début, H2 légèrement sous le nominal entre environ 36 et 62 secondes, et H3 et H4 près du nominal.

Rétablissement à partir de l'état perturbé à $14$ s. En haut à gauche :
erreur de champ normalisée $E(X(t))$. En haut à droite : conversion en sortie,
avec la limite de $95\%$ en tirets. En bas à gauche : température maximale,
avec la limite de $1070$ K en tirets. En bas à droite : entrées des éléments
chauffants exécutées par MPPI, relativement au chauffage nominal $0.46$,
chacune maintenue pendant $2$ s.
:::

Les statistiques d'échantillonnage montrent quelle part de ce résultat provient
des mises à jour acceptées. Sur $33$ décisions, $24$ des $132$ mises à jour ont
été acceptées. Seules quatre des mises à jour acceptées ont utilisé la moyenne
pondérée complète; les $20$ autres ont exigé une fraction de pas de $\tfrac12$
ou moins. Neuf acceptations ont eu lieu lors des trois premières décisions et
onze autres entre $32$ et $42$ s, à l'approche de la poche vers la sortie. À
partir de $44$ s, seules $3$ des $72$ mises à jour ont été acceptées. À ce
moment, le plan stocké revenait déjà au chauffage nominal, et les
perturbations aléatoires tendaient à ajouter de l'erreur à un champ qui se
rétablissait de lui-même. La taille d'échantillon effective a varié de $19.8$
à $64.1$ sur $256$ candidats, et entre $108$ et $217$ candidats par lot ont
réussi la vérification des contraintes. Avec un poids concentré sur quelques
dizaines de candidats, la moyenne pondérée est une proposition bruitée, et les
fractions de pas échangent du progrès contre de la fiabilité.

La décision initiale a aussi été répétée avec trois germes d'optimiseur pour
chacun des nombres de candidats $128$, $256$ et $512$. Les neuf exécutions ont
toutes abaissé l'objectif du calendrier fourni, de $5.3\%$ à $6.7\%$. La
taille d'échantillon effective a crû avec le nombre de candidats, de
$4.3$–$18.9$ à $128$ jusqu'à $32.5$–$63.0$ à $512$. Les objectifs finaux ne se
sont pas améliorés avec ce nombre : leur dispersion entre germes à nombre fixé
dépassait les différences entre les moyennes obtenues à des nombres
différents. Ces répétitions mesurent la variabilité de l'optimiseur en un seul
état physique. Elles n'établissent pas que le plan enregistré est
globalement optimal.

La planification a pris en moyenne $13.9$ s par décision, y compris les
vérifications fines et la compilation à la première utilisation. C'est plus
long que l'intervalle de commande de $2$ s : l'enregistrement est un calcul
hors ligne, et non une démonstration de commande en temps réel. Le modèle
numérique a été vérifié séparément en répétant les commandes MPPI exécutées
avec des pas d'intégration réduits de moitié, ce qui modifie le champ de
température de $0.0014$ K en moyenne quadratique, et sur un maillage de
$128\times64$, ce qui le modifie de $0.78$ K. Le maillage plus fin atteint
$1068.42$ K et une conversion en sortie minimale de $95.36\%$, toujours dans
les deux limites. L'advection décentrée amont du premier ordre ajoute une
diffusion numérique d'environ $v_x\Delta x/2\approx0.013$ m$^2$/s, bien
supérieure à $D$, si bien qu'une partie de l'étalement visible de la poche est
une propriété de la discrétisation.

### Planifier avec un modèle de substitution appris

Chaque décision de rétablissement a simulé $256$ candidats sur $20$
intervalles lors de chacune des quatre mises à jour, soit $20\,480$
déroulements d'intervalle avant validation. Pour cette EDP grossière, ce budget
est abordable. Un maillage plus fin, des phases gazeuse et solide séparées ou
un modèle de mécanique des fluides numérique d'un calcinateur industriel le
rendraient coûteux. Une réponse courante consiste à apprendre à partir de
simulations un modèle de transition approché rapide, un modèle de
substitution, et à l'utiliser pour les déroulements des candidats. MPPI
n'exige que des déroulements et des coûts, de sorte que cette substitution ne
modifie aucune partie de la mise à jour. Elle modifie le modèle qui classe les
candidats : les poids favorisent désormais les candidats que le modèle de
substitution prédit comme bons.

L'expérience avec modèle de substitution utilise la version à écoulement
uniforme du réacteur, avec une tâche différente. Le chauffage nominal est de
$0.48$ dans les quatre quadrants, et la température d'entrée
$800+10\cos(\pi y/2)+g_t$ K fluctue. L'écart $g_t$ suit le modèle
d'Ornstein-Uhlenbeck de l'exemple de l'aéronef, avec un temps de corrélation de
$20$ s et un écart type stationnaire de $15$ K; il est maintenu constant sur
chaque intervalle de $2$ s et observé à chaque décision. L'objectif est
économique : utiliser le moins de chaleur possible tout en maintenant la
conversion en sortie à au moins $95\%$ et la température à au plus $1070$ K.
Le coût de planification fait la moyenne, sur $20$ pas, de

$$
\ell_k=\frac14\sum_{j=1}^4u_{k,j}
+5\left[\frac{(0.95-a_{\mathrm{out},k+1})_+}{0.05}\right]^2
+10\left[\frac{(\max T_{k+1}-1070\ \mathrm K)_+}{20\ \mathrm K}\right]^2
+\frac{0.05}{4}\sum_{j=1}^4\left(\frac{u_{k,j}-u_{k-1,j}}{0.1}\right)^2,
$$

où $(\cdot)_+=\max(\cdot,0)$, la conversion et la température étant évaluées à
la fin de l'intervalle $k$.

Le modèle de substitution est un réseau convolutif résiduel qui prédit la
variation des deux champs sur un intervalle de $2$ s. Ses six canaux d'entrée
sont les champs normalisés de température et de réactif, la carte des
fractions de chauffage, la température d'entrée normalisée et deux canaux de
coordonnées. Quatre convolutions $3\times3$ à $16$ canaux cachés et à
dilatations $1,2,4,1$ permettent à chaque cellule de sortie de dépendre
d'entrées situées jusqu'à huit cellules de distance, plus que les $3.2$
cellules que parcourt le matériau en un intervalle. Après chaque pas prédit, la
fraction de réactif est projetée sur $[0,1]$; la température n'est pas
modifiée. L'entraînement a utilisé $256$ épisodes de l'EDP de $120$ s avec un
chauffage varié, des démarrages à chaud et à froid, des transitoires spatiaux
lisses et des fluctuations d'entrée, répartis par épisode entier en $179$
épisodes d'entraînement, $38$ de validation et $39$ de test. La perte
combinait l'erreur à un pas, l'erreur de déroulement à cinq pas et l'erreur en
sortie. Le point de contrôle utilisé ici a été sélectionné après trois époques
d'Adam, dans les limites d'un budget de calcul local sur processeur central.

| Déroulements de $40$ s hors échantillon, $39$ épisodes de test | Résultat |
| --- | ---: |
| Erreur quadratique moyenne de la température | $4.59$ K |
| Erreur absolue moyenne de la conversion en sortie | $0.0039$ |
| 95e centile de la sous-prédiction de la température maximale | $8.31$ K |
| Plus grand dépassement de borne de la fraction de réactif avant projection | $0.068$ |

Ces moyennes doivent être lues au regard des marges qui décident de la
faisabilité. Sous chauffage constant, la conversion en sortie la plus basse
dans les essais ci-dessous est de $95.09\%$, soit une marge de $0.09$ point de
pourcentage, alors que l'erreur moyenne de conversion du modèle de
substitution est de $0.39$ point de pourcentage. Un plan accepté sur la seule
base des prédictions du modèle de substitution pourrait violer la limite de
conversion dans l'EDP, et une sous-prédiction de $8$ K de la température
maximale pourrait masquer une violation de la limite de température.

Le correcteur conserve donc deux rôles à l'EDP. Chaque commande appliquée fait
avancer l'EDP sous l'entrée réalisée, et chaque plan proposé est simulé dans
l'EDP avant d'être accepté. Le planificateur échantillonne $K=128$ candidats en
coordonnées latentes avec
$u(z)=\operatorname{sigmoid}(\operatorname{logit}(0.48)+0.35z)$ et la
référence $p_0=\mathcal N(0,0.49I)$, qui se décode maintenant en un chauffage
nominal constant. Sa pénalité KL impute un coût aux écarts au chauffage
nominal, tandis que la moyenne de la loi de proposition est initialisée à
partir du plan précédent décalé. Les coûts des candidats proviennent de
déroulements du modèle de substitution, et les poids utilisent
$\lambda=0.025$. Un candidat qui utilise en moyenne un point de pourcentage de
chauffage de moins, sans pénalité active, reçoit environ
$e^{0.01/0.025}\approx1.5$ fois le poids. Après chacune des deux mises à jour
par décision, la moyenne pondérée n'est acceptée que si elle abaisse le coût
selon le modèle de substitution et si un déroulement de l'EDP sous l'entrée
moyenne conditionnelle maintient la conversion à $95\%$ ou plus et la
température à $1070$ K ou moins à chaque sortie de $2$ s. Sinon, le plan
précédent est conservé. Si le plan précédent décalé échoue à la vérification
de l'EDP à la décision suivante, on vérifie plutôt le chauffage nominal
constant; si les deux échouent, l'épisode s'arrête et enregistre un échec de
planification.

Deux variantes de prévision séparent la réaction à l'entrée observée de la
planification sous son incertitude, comme dans l'exemple de l'aéronef. Le
correcteur à prévision moyenne évalue chaque candidat sous la moyenne
conditionnelle de l'entrée future. Le correcteur stochastique l'évalue sous
quatre futurs conditionnels, communs à tous les candidats, et fait la moyenne
des quatre coûts avant la pondération.

### Planification avec modèle de substitution et avec l'EDP comparées

Huit essais appariés partent de l'état chaud et durent $160$ s. Au sein d'un
essai, le chauffage constant et les deux correcteurs MPPI subissent la même
suite d'entrées réalisée.

| Correcteur | Énergie de chauffage moyenne | Conversion en sortie la plus basse | Température la plus élevée | Mises à jour acceptées | Temps de planification par décision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chauffage constant | $76.800$ s | $95.09\%$ | $1059.5$ K | aucune | sans objet |
| MPPI avec modèle de substitution, prévision moyenne | $75.807$ s | $95.30\%$ | $1051.9$ K | $127$ sur $1280$ | $1.32$ s |
| MPPI avec modèle de substitution, prévision stochastique | $75.759$ s | $95.16\%$ | $1052.4$ K | $143$ sur $1280$ | $4.45$ s |

Les valeurs les plus basses et les plus élevées sont prises sur tous les
essais aux sorties de $2$ s. Aucun épisode ne s'est arrêté, et chaque exécution
est restée dans les deux limites sur la grille de $2$ s ainsi que lors d'un
audit séparé à $0.1$ s des commandes exécutées. Les différences appariées
d'énergie de chauffage, avec des intervalles de Student à $95\%$ sur les huit
essais, sont de $-0.99\pm0.30$ s pour MPPI à prévision moyenne moins le
chauffage constant, de $-1.04\pm0.34$ s pour MPPI stochastique moins le
chauffage constant, et de $-0.048\pm0.099$ s pour MPPI stochastique moins MPPI
à prévision moyenne. Les deux variantes de MPPI utilisent environ $1.3\%$ de
chaleur de moins que le chauffage constant. Les essais n'établissent pas de
différence entre les deux variantes de prévision. Environ une mise à jour
proposée sur dix a été acceptée.

:::{figure} _static/thermal_reactor/paired-trials.svg
:label: fig-reactor-surrogate-trials
:width: 100%
:alt: Trois diagrammes à points comparent le chauffage constant, MPPI à prévision moyenne et MPPI stochastique sur huit essais. L'énergie de chauffage est de 76.8 secondes pour le chauffage constant et se situe entre environ 75.1 et 76.4 secondes pour les deux variantes de MPPI. La conversion en sortie la plus basse reste au-dessus de la ligne 0.95 pour tous les essais. La température la plus élevée reste sous la ligne de 1070 kelvins, plus basse pour MPPI que pour le chauffage constant.

Huit essais appariés de $160$ s de MPPI avec modèle de substitution dans le
réacteur à écoulement uniforme. À gauche : énergie de chauffage. Au centre :
conversion en sortie la plus basse, avec la limite de $95\%$ en tirets. À
droite : température la plus élevée, avec la limite de $1070$ K en tirets.
Chaque point est un essai, et les trois correcteurs d'un même essai partagent
une même suite d'entrées réalisée.
:::

Pour isoler l'effet du modèle de déroulement, le même planificateur a été
exécuté à partir de l'état chaud avec un écart d'entrée nul, une fois avec des
déroulements du modèle de substitution et une fois avec des déroulements de
l'EDP, en utilisant le même germe aléatoire et les mêmes réglages. Chaque plan
de $40$ s obtenu a ensuite été évalué dans l'EDP sur $32$ nouveaux futurs
d'entrée. Les premières commandes des éléments chauffants étaient presque
identiques, $(0.484,0.474,0.486,0.480)$ selon le modèle de substitution et
$(0.484,0.472,0.486,0.478)$ selon l'EDP, et les coûts hors échantillon dans
l'EDP étaient de $0.47331$ et $0.47250$. La planification a pris $14.57$ s
avec le modèle de substitution et $11.25$ s avec l'EDP, y compris dans les
deux cas la validation dans l'EDP et l'enregistrement d'instantanés des
candidats. Pour cette EDP grossière, le modèle de substitution n'a apporté
aucun gain de vitesse. Un modèle de substitution n'est rentable que lorsque le
modèle qu'il remplace est nettement plus coûteux à dérouler.

Les deux plans coïncident presque en raison de la structure de $\ell_k$. Pour
un candidat qui ne déclenche aucune des pénalités, le coût se compose du
chauffage moyen et du terme de variation, qui ne dépendent que des commandes.
Le modèle de substitution et l'EDP attribuent à un tel candidat le même coût.
Pour les deux plans retenus, le coût sur les prévisions de planification
concordait avec le coût hors échantillon dans l'EDP à $10^{-6}$ près. Une
telle concordance entre des modèles et des futurs d'entrée différents signifie
que les pénalités n'ont essentiellement rien apporté dans l'une ou l'autre
évaluation. Le modèle de déroulement n'influe sur les poids que par les
pénalités, c'est-à-dire en décidant quels candidats à faible chauffage violent
la limite de conversion et quels candidats à fort chauffage dépassent la
limite de température. La précision du modèle de substitution importe près de
la frontière de l'ensemble réalisable, qui est aussi l'endroit vers lequel un
objectif économique pousse le plan. Les erreurs moyennes sur des trajectoires
hors échantillon ne mesurent pas la précision à cet endroit. La vérification
d'acceptation dans l'EDP teste plutôt le seul plan qui sera exécuté, sous
l'entrée moyenne conditionnelle.

Les deux expériences partagent la même structure. L'échantillonnage explore une
petite paramétrisation bornée du calendrier de chauffage; des poids
exponentiels avec correction de vraisemblance proposent une mise à jour; et le
modèle complet décide si la mise à jour remplace un plan dont on sait déjà
qu'il est réalisable. Dans l'expérience de rétablissement, le modèle complet
produit aussi les coûts des candidats. Dans l'expérience avec modèle de
substitution, un modèle appris les produit, et ses erreurs importent là où les
contraintes sont presque actives.

:::{iframe} ../interactive/thermal-reactor-mppi.html#recovery
:width: 100%
:title: Rétablissement du réacteur thermique et rejeu du modèle de substitution
:class: thermal-reactor-replay

Le rejeu s'ouvre sur l'expérience de rétablissement du champ. Lancez
l'enregistrement pour suivre la poche froide sous chauffage inchangé et sous
MPPI, affichez l'erreur par rapport à l'état d'exploitation, ou ouvrez la mise
à jour de commande échantillonnée pour voir six candidats et leurs poids sur le
lot complet à $14$, $22$ et $34$ s. L'onglet « Surrogate vs PDE » compare le
modèle appris avec l'EDP sous des commandes et des historiques d'entrée
identiques dans l'expérience à écoulement uniforme. Chaque image de champ est
un déroulement numérique enregistré; la page n'effectue aucune simulation.
:::

## Synthèse et perspectives

MPPI part du déroulement par tir simple et modifie la façon dont ses entrées
candidates sont combinées. Une loi de référence favorise les petites entrées;
des poids exponentiels sur le coût d'état favorisent les trajectoires utiles;
un ajustement gaussien transforme la loi qui en résulte en une suite de
commandes stockée. La correction par échantillonnage préférentiel préserve la
cible lorsque les échantillons sont tirés autour d'une suite nominale non
nulle.

L'application à l'aéronef sépare l'exploration des commandes de l'incertitude
sur le vent et fait la moyenne des coûts sur les vents avant d'attribuer son
poids à un candidat. Augmenter le nombre de commandes candidates et augmenter
le nombre de vents futurs agissent sur des erreurs d'estimation différentes.
Ni l'un ni l'autre ne garantit qu'une moyenne pondérée est réalisable ou
améliore le coût; les mises à jour candidates sont donc validées avant de
remplacer le plan stocké.

L'application au réacteur utilise la même mise à jour sur une EDP discrétisée
à 4 096 valeurs d'état. Six nœuds bornés par élément chauffant réduisent la
recherche à 24 paramètres latents, dans lesquels sont calculées les densités
gaussiennes et la correction de vraisemblance, et les candidats qui violent une
limite reçoivent un poids nul. Comme la moyenne de candidats réalisables n'est
pas nécessairement réalisable, chaque mise à jour proposée est vérifiée dans
l'EDP, amortie au besoin, et rejetée sinon; une fois le champ en voie de
rétablissement, la plupart des mises à jour ont été rejetées. Remplacer l'EDP
par un modèle de substitution appris dans la boucle d'échantillonnage change le
modèle qui classe les candidats, tandis que l'EDP décide toujours du plan
exécuté. Avec un objectif économique, les erreurs du modèle de substitution
importent surtout près des limites d'exploitation, c'est-à-dire là où le plan
opère.

Le correcteur a encore besoin d'une règle d'exécution. Appliquer une action,
observer le résultat et résoudre de nouveau procure une rétroaction, même si
chaque suite stockée est en boucle ouverte. La
[commande à horizon glissant](receding-horizon-control.md) approfondit cette
règle, y compris le choix de l'horizon, les conditions terminales et les
garde-fous de faisabilité. Ces questions se posent que le planificateur
utilise des dérivées ou des trajectoires échantillonnées.

Le nom de la méthode reflète aussi un lien avec des espérances sur des chemins
stochastiques entiers. Le chapitre sur la
[commande stochastique par intégrale de chemin](path-integral-stochastic-control.md),
qui suit la programmation dynamique régularisée par l'entropie, développe ce
lien et énonce les hypothèses supplémentaires sous lesquelles la pondération
des chemins donne une action de rétroaction courante optimale.

## Sources numériques

Les expériences de l'aéronef et du réacteur utilisent de petits modules
numériques et lisent des résultats précalculés lors de la compilation du
livre. L'évaluateur de l'aéronef fait la moyenne des coûts de chaque candidat
sur un ensemble commun de vents futurs possibles.

- {download}`Poids d'échantillonnage préférentiel gaussiens et mises à jour de la moyenne <code/mppi_control.py>`
- {download}`Modèle de l'aéronef, scénarios de vent et optimiseur de trajectoires <code/aircraft_mppi.py>`
- {download}`Expérience de l'aéronef et générateur de figures <scripts/build_aircraft_mppi_artifacts.py>`
- {download}`Instantané numérique ERA5 <data/aircraft/era5_wind.npz>` et
  {download}`provenance et sommes de contrôle <data/aircraft/era5_wind.json>`
- {download}`EDP du réacteur, éléments chauffants et perturbations d'entrée <code/thermal_reactor.py>`
  et {download}`écoulement cisaillé et impulsion froide à l'entrée <code/thermal_plume.py>`
- {download}`Rétablissement du champ directement avec l'EDP <code/thermal_recovery.py>`
- {download}`Modèle de substitution convolutif <code/thermal_surrogate.py>` et
  {download}`MPPI avec modèle de substitution et vérifications d'acceptation dans l'EDP <code/thermal_control.py>`
- {download}`Étapes de l'expérience du réacteur <scripts/build_thermal_reactor.py>`
- {download}`Décisions de rétablissement <artifacts/thermal_recovery/decisions.json>`
  et {download}`essais avec modèle de substitution <artifacts/thermal_reactor/evaluation.json>`

Les scripts d'expérience enregistrent les paramètres physiques, les réglages
du planificateur, les germes et les diagnostics de validation avec leurs
résultats. Le [rejeu de l'aéronef](interactive/aircraft-mppi.html) et le
[rejeu du réacteur](interactive/thermal-reactor-mppi.html) affichent des
simulations enregistrées sans noyau Python. Les commandes de régénération se
trouvent dans le fichier README du dépôt.

## Exercices

:::{exercise} Poids et mises à jour de la commande
:label: ex-mppi-weights

Une itération de MPPI a la commande nominale scalaire $u=0$, les
perturbations $(-1,0,1)$, les coûts de trajectoire corrigés
$(\log4,\log2,0)$ et $\lambda=1$.

**(a)** Calculez les poids normalisés et la commande mise à jour.

**(b)** Ajoutez $1000$ à chaque coût. Expliquez pourquoi la mise à jour
mathématique est inchangée et comment la calculer sans soupassement
numérique.

**(c)** Gardez les coûts corrigés fixes et faites tendre $\lambda$ vers zéro.
Quelle perturbation détermine la mise à jour limite?
:::

:::{solution} ex-mppi-weights
:class: dropdown

**(a)** Les poids non normalisés sont $(1/4,1/2,1)$, de somme $7/4$. Les
poids normalisés sont $(1/7,2/7,4/7)$, ce qui donne
$u^+=-(1/7)+(4/7)=3/7$.

**(b)** Le facteur commun $e^{-1000}$ se simplifie entre le numérateur et le
dénominateur. Une évaluation directe provoquerait un soupassement en
arithmétique à virgule flottante ordinaire. Soustraire le coût corrigé
minimal avant l'exponentiation redonne les trois poids d'origine.

**(c)** Le coût minimal unique appartient à la perturbation $1$. Son poids
normalisé tend vers un, donc $u^+\to1$. La précision selon laquelle les coûts
corrigés sont fixes importe : une correction de vraisemblance peut elle-même
dépendre de $\lambda$.
:::

:::{exercise} La correction de vraisemblance
:label: ex-mppi-likelihood-correction

Considérez le déroulement déterministe $x_{k+1}=x_k+hv_k$ avec $x_0=0$, la
cible $a=1$, $N=10$ intervalles de longueur $h=0.1$ et le coût terminal
$\tfrac\kappa2(x_N-a)^2$ avec $\kappa=4$. Posez $\lambda=0.5$ et choisissez
une covariance de référence gaussienne $\Sigma_h=\lambda/(rh)$ avec $r=1$, de
sorte que sa pénalité KL donne l'effort quadratique sur la commande moyenne.
La moyenne de la loi de proposition est $u_k=0.4$ à chaque pas.

**(a)** Complétez le carré dans la densité de Gibbs pour trouver sa moyenne.
Vous pouvez utiliser le fait que la moyenne d'une gaussienne est égale au
minimiseur de la forme quadratique de sa log-densité changée de signe.

**(b)** Trouvez la moyenne estimée lorsque les poids incluent le coût d'état
mais omettent le rapport des densités de la référence et de la loi de
proposition.

**(c)** Implémentez les deux estimateurs et comparez leur commande moyenne,
calculée sur les dix pas de temps, à mesure que le nombre d'échantillons
augmente. Augmenter le nombre d'échantillons corrige-t-il l'écart de la
partie (b)?
:::

:::{solution} ex-mppi-likelihood-correction
:class: dropdown

**(a)** À une constante près, l'opposé de la log-densité de Gibbs est

$$
\frac{rh}{2\lambda}\sum_kv_k^2
+\frac\kappa{2\lambda}\left(h\sum_kv_k-a\right)^2.
$$

Sa condition du premier ordre est
$r v_k+\kappa(h\sum_jv_j-a)=0$. Toutes les coordonnées de son minimiseur sont
égales, ce qui donne

$$
\mathbb E_{P^\star}v_k=\frac{\kappa a}{r+\kappa Nh}=0.8.
$$

Le résultat ne dépend pas de la moyenne de la loi de proposition, car le
rapport de vraisemblance annule la densité de proposition dans l'espérance.

**(b)** Omettre ce rapport remplace la forme quadratique de référence
$\sum_kv_k^2$ par $\sum_k(v_k-0.4)^2$. La condition du premier ordre
correspondante est $r(v_k-0.4)+\kappa(h\sum_jv_j-a)=0$, de sorte que la
moyenne vaut

$$
\frac{0.4r+\kappa a}{r+\kappa Nh}=0.88.
$$

Les deux procédures convergent donc vers des espérances différentes.

**(c)** Une implémentation minimale utilise les mêmes perturbations pour les
deux estimations afin que leur différence soit plus facile à observer :

```python
import numpy as np

rng = np.random.default_rng(728)
N, h, r, kappa, lam = 10, 0.1, 1.0, 4.0, 0.5
U = np.full(N, 0.4)
for K in (1_000, 10_000, 100_000):
    eps = rng.normal(size=(K, N)) * np.sqrt(lam / (r * h))
    inputs = U + eps
    cost = 0.5 * kappa * (h * inputs.sum(axis=1) - 1.0) ** 2
    corrected = cost + h * r * (eps @ U)
    estimates = []
    for scores in (corrected, cost):
        weights = np.exp(-(scores - scores.min()) / lam)
        weights /= weights.sum()
        estimates.append((weights @ inputs).mean())
    print(K, *estimates)
```

Les estimations tendent vers $0.8$ et $0.88$. Davantage d'échantillons
réduisent l'erreur de Monte-Carlo; ils ne réparent pas un rapport de
vraisemblance omis. Une exécution particulière ne tend pas nécessairement de
façon monotone vers sa limite.
:::

:::{exercise} Déroulements stochastiques de l'aéronef et coût espéré
:label: ex-mppi-aircraft-expected-cost

Un simulateur d'aéronef reçoit une suite de commandes $\mathbf U$ et un
historique de rafales $W$, et renvoie le coût $C(\mathbf U,W)$. L'objectif est
$\mathbb E_W C(\mathbf U,W)$.

**(a)** Un optimiseur par échantillonnage génère un historique de rafales par
suite de commandes candidate et attribue un poids proportionnel à
$e^{-C(\mathbf U,W)/\lambda}$. Expliquez pourquoi recueillir davantage
d'échantillons ainsi pondérés ne permet pas, en général, de retrouver
l'objectif de coût espéré visé.

**(b)** Considérez un candidat sûr de coût constant $1$ et un candidat risqué
de coût $0$ avec probabilité $0.9$ et $20$ avec probabilité $0.1$. Pour
$\lambda=1$, comparez leurs coûts espérés et leurs scores exponentiels
$-\log\mathbb E[e^{-C}]$.

**(c)** Décrivez un plan d'échantillonnage qui classe les commandes candidates
selon une estimation du coût espéré. Ce plan garantit-il que la moyenne
pondérée est optimale ou réalisable?
:::

:::{solution} ex-mppi-aircraft-expected-cost
:class: dropdown

**(a)** Pour un candidat fixé, la moyenne des poids estime
$\mathbb E_W[e^{-C/\lambda}]$, alors que le score souhaité dépend de
$\mathbb E_W C$. L'exponentielle est non linéaire. Les historiques de rafales
favorables reçoivent un poids disproportionné, si bien que cette procédure
peut favoriser un candidat dont le coût espéré est plus élevé. La correction
référence-proposition de l'échantillonnage des commandes n'élimine pas cette
différence dans l'objectif relatif aux perturbations.

**(b)** Le candidat sûr a un coût espéré de $1$ et un score exponentiel de
$1$. Le candidat risqué a un coût espéré de $2$, mais un score exponentiel de
$-\log(0.9+0.1e^{-20})\approx0.105$. Le coût espéré favorise le candidat sûr;
le score exponentiel favorise le candidat risqué.

**(c)** Pour chaque candidat, simulez plusieurs historiques de rafales et
formez $\widehat C(\mathbf U)=L^{-1}\sum_{\ell=1}^L C(\mathbf U,W^{(\ell)})$.
Utilisez ensuite $\widehat C$ comme score du candidat dans l'optimiseur par
échantillonnage. Des historiques de rafales communs à tous les candidats
peuvent rendre les comparaisons appariées moins variables. À mesure que le
nombre d'échantillons de perturbation augmente, ce score estime le coût
espéré visé. L'ajustement subséquent de la moyenne gaussienne ne minimise pas
nécessairement ce coût sur les plans déterministes. L'échantillonnage fini
introduit une erreur supplémentaire, et une moyenne de plans réalisables peut
violer des contraintes non convexes. Le plan proposé exige encore des
vérifications de coût et de faisabilité.
:::

:::{exercise} Décaler un plan et vérifier sa moyenne
:label: ex-mppi-execution

Un correcteur stocke $(u_0,u_1,u_2)=(0.6,0.4,0.2)$ pour trois intervalles.
Après le premier intervalle, il observe un nouvel état.

**(a)** Quelle commande a été exécutée, et quelle suite initialise le problème
de planification suivant si la dernière commande est répétée? Quelle autre
entrée du déroulement change lors de cette observation?

**(b)** Considérez le système distinct à un pas $x_1=u$ avec la contrainte
terminale $x_1\in[-1.1,-0.9]\cup[0.9,1.1]$. Deux candidats réalisables sont
$u=-1$ et $u=1$. Leur moyenne à poids égaux est-elle réalisable? Que devrait
faire une implémentation dotée d'un plan en place validé si cette moyenne est
proposée?
:::

:::{solution} ex-mppi-execution
:class: dropdown

**(a)** Le correcteur a exécuté $0.6$. Son estimation initiale décalée est
$(0.4,0.2,0.2)$, et l'état mesuré devient l'état initial de chaque nouveau
déroulement. L'estimation est ensuite mise à jour par échantillonnage; elle
n'engage pas à exécuter ses trois composantes.

**(b)** La moyenne est $u=0$, qui produit $x_1=0$, hors des deux intervalles
permis. Des échantillons réalisables ne garantissent pas une moyenne
réalisable sous des contraintes non convexes. L'implémentation devrait valider
la proposition et conserver le plan en place réalisable lorsque cette
validation échoue, comme dans l'exemple de l'aéronef.
:::

:::{exercise} Le budget de calcul d'une décision du réacteur
:label: ex-mppi-reactor-budget

Le correcteur de rétablissement du champ de l'[exemple du réacteur](#sec-reactor-mppi)
tire $K=256$ tableaux latents lors de chacune des quatre mises à jour par
décision. Chaque candidat est simulé pendant $20$ intervalles de $2$ s, avec
des sous-pas de l'EDP d'au plus $0.1$ s.

**(a)** Combien de valeurs d'état un déroulement candidat fait-il avancer à
chaque sous-pas, et sur combien de nombres porte la moyenne pondérée?

**(b)** En ignorant la validation, combien de simulations d'intervalles de
$2$ s et au moins combien de sous-pas une décision exige-t-elle?

**(c)** Le maillage est raffiné de $64\times32$ à $128\times64$ cellules. À
l'aide de la condition d'advection $v_x\Delta t\leq\Delta x$ avec
$v_x\leq0.22$ m/s, vérifiez que le plafond de $0.1$ s sur les sous-pas
détermine toujours leur taille. Par quel facteur la dimension de l'état, le
nombre de paramètres échantillonnés et le travail par décision changent-ils?

**(d)** La paramétrisation à six nœuds ne peut pas représenter toutes les
suites de $80$ valeurs de chauffage. Donnez un exemple de calendrier qu'elle
exclut, et expliquez ce que cette restriction implique pour l'affirmation
selon laquelle le calendrier MPPI enregistré est optimal.
:::

:::{solution} ex-mppi-reactor-budget
:class: dropdown

**(a)** Chacune des $2\,048$ cellules porte une température et une fraction de
réactif, si bien qu'un déroulement fait avancer $4\,096$ valeurs d'état. La
moyenne pondérée porte sur les $6\times4=24$ paramètres latents.

**(b)** Une décision simule $256\times4\times20=20\,480$ intervalles. Chaque
intervalle exige au moins $2/0.1=20$ sous-pas, de sorte qu'au moins
$409\,600$ sous-pas sont nécessaires, chacun évaluant deux fois le membre de
droite dans la méthode de Runge-Kutta à deux étages.

**(c)** Le maillage raffiné a $\Delta x=8/128=0.0625$ m, de sorte que la
condition d'advection permet $\Delta t\leq0.0625/0.22\approx0.28$ s, et le
plafond de $0.1$ s reste déterminant. La dimension de l'état est multipliée
par quatre, jusqu'à $16\,384$. Le nombre de paramètres échantillonnés reste
$24$. Chaque sous-pas met à jour quatre fois plus de cellules, si bien que le
travail par décision est multiplié par environ quatre. Les poids et la mise à
jour conservent la même dimension.

**(d)** Une seule augmentation de $2$ s d'un élément chauffant au milieu de
l'horizon, toutes les valeurs voisines restant au nominal, n'est pas linéaire
par morceaux entre les six instants des nœuds et ne peut pas être
représentée. MPPI ne cherche qu'au sein de la famille des nœuds, de sorte que
le calendrier enregistré est au mieux un bon membre de cette famille.
L'échantillonnage fini, l'ajustement de la moyenne gaussienne, la référence
régularisante et les fractions de pas limitent même cette affirmation; le
résultat appuie donc une amélioration par rapport au calendrier fourni, et non
l'optimalité.
:::

:::{exercise} Échantillonner des entrées de chauffage bornées
:label: ex-mppi-reactor-latent

Un nœud de chauffage est décodé par
$u(z)=\operatorname{sigmoid}(\operatorname{logit}(u_{\mathrm{seed}})+sz)$
avec $s>0$. La référence est $\mathcal N(0,\sigma^2I)$ et la loi de
proposition est $\mathcal N(\mu,\sigma^2I)$, toutes deux sur le tableau latent
$z$.

**(a)** Prenez un nœud scalaire avec $u_{\mathrm{seed}}=0.9$ et $s=0.35$, et
deux candidats $z=2$ et $z=-2$ de poids égaux. Comparez la moyenne pondérée
décodée $u(\bar z)$ avec la moyenne des deux entrées décodées. Laquelle le
correcteur du réacteur propose-t-il, et pourquoi aucune ne sort-elle de
$[0,1]$?

**(b)** Montrez que les poids
$w_i\propto e^{-J_i/\lambda}p_0(z^{(i)})/q_\mu(z^{(i)})$ sont égaux à
$w_i\propto e^{-\widetilde J_i/\lambda}$ avec le coût corrigé
$\widetilde J_i=J_i+\lambda\mu^\top z^{(i)}/\sigma^2$, à un facteur commun à
tous les candidats près. Comparez avec le
[coût corrigé](#sec-mppi-importance) dérivé plus tôt dans le chapitre.

**(c)** Lors de la première décision enregistrée, $\lambda=0.01$ et
$\sigma^2=0.49$. Dans le même lot, le candidat 12 avait un coût de $0.05649$ et
un poids de $0.0682$, tandis que le candidat 180 avait un coût de $0.05840$ et
un poids de $0.1565$. Utilisez la partie (b) pour calculer
$\mu^\top(z^{(12)}-z^{(180)})$ et interprétez son signe.

**(d)** Une autre implémentation échantillonne les valeurs physiques des
nœuds selon une gaussienne et les écrête à $[0,1]$ avant la simulation.
Comment devrait-elle calculer la correction de vraisemblance, et quel est un
inconvénient par rapport au décodeur logistique?
:::

:::{solution} ex-mppi-reactor-latent
:class: dropdown

**(a)** Ici, $\operatorname{logit}(0.9)=\log9\approx2.197$ et $sz=\pm0.7$.
Les entrées décodées sont $\operatorname{sigmoid}(2.897)\approx0.948$ et
$\operatorname{sigmoid}(1.497)\approx0.817$, dont la moyenne vaut environ
$0.882$. La moyenne latente est $\bar z=0$, qui se décode en $0.9$. Le
correcteur fait la moyenne des paramètres latents et propose $u(\bar z)=0.9$,
sous réserve de validation. Les deux nombres se trouvent dans $(0,1)$ : la
sigmoïde envoie tout nombre réel dans $(0,1)$, et une moyenne de nombres de
$(0,1)$ y reste. Ils diffèrent parce que la sigmoïde est courbe : près de la
borne supérieure, des pas latents égaux produisent une augmentation plus
petite que la diminution.

**(b)** Pour des gaussiennes de covariance commune,

$$
\log\frac{p_0(z)}{q_\mu(z)}
=-\frac{\|z\|^2}{2\sigma^2}+\frac{\|z-\mu\|^2}{2\sigma^2}
=-\frac{\mu^\top z}{\sigma^2}+\frac{\|\mu\|^2}{2\sigma^2}.
$$

Le dernier terme est le même pour tous les candidats et disparaît après
normalisation. Ainsi,
$e^{-J_i/\lambda}p_0/q_\mu\propto\exp[-(J_i+\lambda\mu^\top z^{(i)}/\sigma^2)/\lambda]$.
Écrire $z^{(i)}=\mu+\boldsymbol\epsilon^{(i)}$ ne modifie la correction que
d'une autre constante commune, ce qui donne la forme
$\lambda\mu^\top\Sigma^{-1}\boldsymbol\epsilon^{(i)}$ du coût corrigé
précédent, avec $\Sigma=\sigma^2I$.

**(c)** Le rapport des poids donne la différence des coûts corrigés,

$$
\widetilde J_{12}-\widetilde J_{180}
=-\lambda\log\frac{0.0682}{0.1565}\approx0.0083.
$$

La différence de coûts est $J_{12}-J_{180}=-0.0019$, de sorte que la
correction contribue pour $0.0102=\lambda\mu^\top(z^{(12)}-z^{(180)})/\sigma^2$.
Par conséquent,
$\mu^\top(z^{(12)}-z^{(180)})\approx0.0102\times0.49/0.01\approx0.50$. Le signe
positif signifie que le candidat 12 se trouve plus loin dans la direction vers
laquelle la moyenne de la loi de proposition s'est déjà déplacée. La loi de
proposition tire de tels candidats plus souvent que ne le ferait la
référence, et la correction réduit leur poids pour compenser.

**(d)** La densité gaussienne est celle de l'échantillon non écrêté; le
rapport doit donc être calculé pour cette variable, l'écrêtage étant traité
comme une partie de l'application déterministe qui associe des commandes aux
échantillons, comme pour la durée du vol de l'aéronef. Tout échantillon
au-delà d'une borne produit alors la même commande et le même coût. Ces
échantillons sont indiscernables pour les poids, et la moyenne pondérée des
valeurs non écrêtées peut elle-même se trouver hors de $[0,1]$. Le décodeur
logistique garde distincts des échantillons latents distincts et décode
toujours la moyenne à l'intérieur des bornes.
:::

:::{exercise} Quand le modèle de déroulement importe
:label: ex-mppi-reactor-surrogate

Dans l'expérience avec modèle de substitution, chaque pas du coût de
planification d'un candidat est $\ell_k$, qui combine le chauffage moyen, une
pénalité de conversion, une pénalité de température et une pénalité de
variation.

**(a)** Supposez que, pour un certain candidat, aucune des deux pénalités
n'est active sous le modèle de substitution ni sous l'EDP. Montrez que les deux
modèles lui attribuent le même coût de planification. Sur quels candidats les
erreurs du modèle de substitution peuvent-elles influer?

**(b)** Le 95e centile de la sous-prédiction de la température maximale par
le modèle de substitution, sur des déroulements hors échantillon, est de
$8.31$ K. Un plan proposé a une température maximale prédite de $1064$ K.
Cela établit-il qu'il respecte la limite de $1070$ K dans l'EDP? Quelle étape
du correcteur protège le procédé, et sous quelle suite d'entrées cette étape
vérifie-t-elle le plan?

**(c)** Une variante resserre les limites utilisées dans les déroulements du
modèle de substitution à $1070-\delta$ K et $95\%+\varepsilon$. Décrivez le
compromis dans le choix de $\delta$ et $\varepsilon$ lorsque l'objectif est de
minimiser le chauffage.

**(d)** En un état donné, la planification avec le modèle de substitution a
pris $14.57$ s et la planification avec l'EDP $11.25$ s. Quand vous
attendriez-vous à ce qu'un modèle de substitution réduise le temps de
planification? Quelles preuves exigeriez-vous avant de supprimer la
vérification d'acceptation dans l'EDP?
:::

:::{solution} ex-mppi-reactor-surrogate
:class: dropdown

**(a)** Sans pénalité active, $\ell_k$ se réduit au chauffage moyen plus le
terme de variation, deux fonctions des seules commandes. Tous les
déroulements qui restent valides donnent donc le même coût sous l'un ou
l'autre modèle. Les erreurs du modèle de substitution ne peuvent modifier que
le coût des candidats pour lesquels au moins un modèle prédit une pénalité
active, c'est-à-dire les candidats proches de la limite de conversion ou de
température. Minimiser le chauffage pousse le plan vers la limite de
conversion; ce sont donc ces candidats qui comptent.

**(b)** Non. Le chiffre de $8.31$ K est un centile sur des fenêtres hors
échantillon, et non une borne : dans une fenêtre sur vingt, le modèle de
substitution a sous-prédit davantage, et un plan optimisé n'est pas tiré de la
loi des données hors échantillon. Ajouter $8.31$ K à $1064$ K dépasse déjà
$1070$ K. La vérification d'acceptation dans l'EDP simule le plan proposé avant
qu'il remplace le plan stocké. Elle utilise la moyenne conditionnelle de
l'entrée future, de sorte qu'elle ne garantit pas la faisabilité sous l'entrée
réellement réalisée. L'audit sur une grille de $0.1$ s ne vérifie les
commandes exécutées qu'après coup.

**(c)** Des marges plus grandes font rejeter davantage de candidats proches
des limites par les déroulements du modèle de substitution. Moins de
propositions échouent alors à la vérification dans l'EDP, et les violations
sous l'entrée réalisée deviennent moins probables, mais le correcteur renonce
à des réductions de chauffage qui étaient réalisables. Des marges plus petites
permettent des plans plus proches des limites, au prix de davantage de
propositions rejetées et d'une moindre protection contre l'erreur du modèle de
substitution et l'incertitude sur l'entrée. Des marges étalonnées à partir des
erreurs du modèle de substitution près des limites, plutôt qu'à partir des
erreurs moyennes, rendraient ce choix moins arbitraire.

**(d)** Un modèle de substitution réduit le temps de planification lorsque les
déroulements du modèle complet dominent le calcul et que le modèle de
substitution est beaucoup moins coûteux à évaluer en grands lots, par exemple
avec un maillage plus fin, une physique multiphasique ou un modèle de mécanique
des fluides numérique. Supprimer la vérification dans l'EDP exigerait des
preuves sur les erreurs près de la frontière des contraintes pour les types de
plans que propose l'optimiseur, et non seulement des erreurs moyennes sur des
trajectoires semblables à celles de l'entraînement, ainsi que des essais en
boucle fermée dont les commandes exécutées sont auditées dans le modèle
complet.
:::

:::{exercise} Un modèle de substitution pour le rétablissement du champ
:label: ex-mppi-reactor-project

L'expérience de rétablissement du champ planifie avec l'EDP, alors que le
modèle de substitution disponible a été entraîné pour le réacteur à
écoulement uniforme avec des fluctuations d'entrée aléatoires. Ce projet comble
cet écart à l'aide des modules énumérés dans les sources numériques.

**(a)** Générez des épisodes d'entraînement de l'EDP sous l'écoulement
cisaillé qui comprennent des impulsions froides à l'entrée de position, de
largeur, de profondeur et de durée variées, ainsi que des calendriers de
chauffage variés. Entraînez le réseau résiduel et rapportez les erreurs hors
échantillon pour l'erreur de champ $E$, la température maximale et la
conversion en sortie.

**(b)** Remplacez les déroulements de l'EDP par des déroulements du modèle de
substitution pour évaluer les candidats dans le correcteur de rétablissement,
en conservant inchangée la validation dans l'EDP. Comparez l'erreur de champ
intégrée, les mises à jour acceptées et le temps de planification avec
l'enregistrement obtenu directement avec l'EDP.

**(c)** Testez le correcteur sur une impulsion dont la position se situe hors
de la plage d'entraînement. Quels diagnostics enregistrés montreraient que le
modèle de substitution n'est pas fiable avant qu'un plan soit exécuté?
:::

:::{solution} ex-mppi-reactor-project
:class: dropdown

Les réponses dépendent des données d'entraînement et du budget. Les
comparaisons utiles gardent fixes l'état initial, le calendrier fourni, les
réglages du planificateur et les germes de l'optimiseur, afin que les
différences proviennent du modèle de déroulement. Comme l'objectif de suivi
dépend du champ entier, les erreurs du modèle de substitution influent sur le
coût de chaque candidat, contrairement à ce qui se passe avec l'objectif
économique de l'exercice @ex-mppi-reactor-surrogate. Lorsque le modèle de
substitution classe les candidats autrement que l'EDP, une mise à jour
acceptée peut abaisser l'objectif du modèle de substitution tout en augmentant
celui de l'EDP; il faut donc enregistrer les deux pour chaque proposition.
Pour la partie (c), la fraction des propositions rejetées par la vérification
dans l'EDP, l'écart entre l'objectif prédit par le modèle de substitution et
l'objectif dans l'EDP de chaque proposition, et les valeurs brutes de la
fraction de réactif hors de $[0,1]$ avant projection signalent toutes que le
modèle de substitution opère hors de sa loi d'entraînement.
:::

## Autoévaluation

:::{exercise} Échantillonnage et rétroaction
:label: ex-mppi-check-feedback

Qu'est-ce qui fait du correcteur exécuté un correcteur à rétroaction : tirer
de nombreuses trajectoires, ou conditionner un nouveau plan sur l'état observé
suivant?
:::

:::{solution} ex-mppi-check-feedback
:class: dropdown

Conditionner un nouveau plan sur l'état observé procure la rétroaction. Une
suite de commandes calculée à partir de nombreux échantillons puis figée
s'exécute toujours en boucle ouverte.
:::

:::{exercise} L'aléa dans le rétablissement du réacteur
:label: ex-mppi-check-reactor

Dans l'expérience de rétablissement du champ, quelles grandeurs sont
aléatoires : le procédé, l'entrée ou les commandes candidates? Qu'est-ce qui
change lorsque le germe de l'optimiseur change?
:::

:::{solution} ex-mppi-check-reactor
:class: dropdown

Seules les commandes candidates sont aléatoires. L'EDP est déterministe et
l'entrée est connue après l'impulsion, si bien que les nombres aléatoires ne
servent qu'à explorer les calendriers de chauffage. Un germe différent modifie
les candidats échantillonnés et donc le plan. Sur les neuf répétitions de la
décision initiale, l'objectif final a varié d'environ $1.5\%$.
:::
