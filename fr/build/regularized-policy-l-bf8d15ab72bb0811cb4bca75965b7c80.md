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
# Apprentissage de politiques régularisé et fondé sur les résidus

DDPG et TD3 traitent les actions continues en apprenant une politique déterministe qui amortit l'opération $\arg\max_a q(s,a)$. Une politique déterministe n'explore pas d'elle-même. Elle a besoin d'un bruit ajouté à ses actions (des perturbations gaussiennes dans TD3) et, sans une couverture suffisante de l'espace état-action, elle peut se fixer sur un comportement déterministe sous-optimal.

Les estimateurs de gradient rendent les acteurs stochastiques entraînables, tandis que la programmation dynamique
régularisée précise la politique stochastique qu'ils doivent approcher. Peut-on combiner les deux pour éviter à la fois l'intégrale sur les actions continues
dans l'équation de Bellman adoucie et l'optimisation répétée d'un maximum
strict?

[La programmation dynamique régularisée par l'entropie](regularized-dp.md) présente une solution de rechange : des processus de décision markoviens où l'agent maximise le rendement espéré augmenté d'un bonus pour le caractère aléatoire de la politique. On obtient ainsi des politiques stochastiques dont l'exploration est intégrée à l'objectif lui-même. L'opérateur de Bellman lissé remplace le maximum strict par un maximum adouci :

$$
v^*(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}} \exp\left(\beta \cdot q^*(s,a)\right)
$$

où $\beta = 1/\alpha$ est l'inverse de la température et $\alpha$ le poids de la régularisation entropique. Pour des espaces d'actions finis, ce log-somme-exp se calcule facilement. Mais pour des actions continues $\mathcal{A} \subset \mathbb{R}^m$, la somme devient une intégrale :

$$
v^*(s) = \frac{1}{\beta} \log \int_{\mathcal{A}} \exp\left(\beta \cdot q^*(s,a)\right) da
$$

Cette intégrale sur l'espace d'actions n'a en général pas de forme close. Le lissage qui donne des politiques stochastiques crée donc son propre problème de calcul, analogue à l'$\arg\max$ sur des actions continues dans l'itération Q ajustée standard.

## D'une intégrale intraitable à une espérance calculable

L'intégrale qui définit le logarithme de la fonction de partition sur des actions continues
peut se réécrire comme une espérance sous la politique en cours d'apprentissage.

L'acteur-critique adouci (SAC) {cite:p}`haarnoja2018soft,haarnoja2018sacapplications` exploite une équivalence entre l'intégrale intraitable et une espérance. La politique optimale sous régularisation entropique est la distribution de Boltzmann $\pi^*(a|s) \propto \exp(\beta \cdot q^*(s,a))$. Sous cette politique, la fonction de valeur adoucie devient :

$$
v^*(s) = \mathbb{E}_{a \sim \pi^*(\cdot|s)}\left[q^*(s,a) - \alpha \log \pi^*(a|s)\right]
$$

On convertit ainsi l'intégrale intraitable en une espérance que l'on peut estimer par échantillonnage. SAC apprend une politique paramétrique $\pi_{\boldsymbol{\phi}}$ qui approche la distribution de Boltzmann, ce qui permet de sélectionner rapidement une action par une seule passe avant. Pour les cibles d'autoamorçage, SAC tire $\tilde{a}' \sim \pi_{\boldsymbol{\phi}}(\cdot|s')$ et calcule :

$$
y = r + \gamma \left[\min_{j=1,2} q^j_{\boldsymbol{\theta}_{\text{target}}}(s', \tilde{a}') - \alpha \log \pi_{\boldsymbol{\phi}}(\tilde{a}'|s')\right]
$$

Le minimum sur les réseaux Q jumeaux applique l'astuce du double Q écrêté de TD3. L'exploration provient du caractère stochastique de la politique plutôt que d'un bruit externe.

## Apprendre la politique : approcher la distribution de Boltzmann

La mise à jour du réseau Q suppose une politique $\pi_{\boldsymbol{\phi}}$ qui approche la distribution de Boltzmann $\pi^*(a|s) \propto \exp(\beta \cdot q^*(s,a))$. L'entraînement d'une telle politique pose un problème : la distribution de Boltzmann exige la fonction de partition $Z(s) = \int_{\mathcal{A}} \exp(\beta \cdot q(s,a))da$, l'intégrale même que l'on cherche à éviter. SAC contourne cette difficulté en minimisant la divergence de Kullback-Leibler (KL) de la politique à la distribution de Boltzmann (non normalisée) :

$$
\min_{\boldsymbol{\phi}} \mathbb{E}_{s \sim \mathcal{D}}\left[D_{KL}\left(\pi_{\boldsymbol{\phi}}(\cdot|s) \| \frac{\exp(\beta \cdot q_{\boldsymbol{\theta}}(s,\cdot))}{Z_{\boldsymbol{\theta}}(s)}\right)\right]
$$

Puisque $\log Z_{\boldsymbol{\theta}}(s)$ ne dépend pas de $\boldsymbol{\phi}$, ce problème se réduit à :

$$
\min_{\boldsymbol{\phi}} \mathbb{E}_{s \sim \mathcal{D}}\mathbb{E}_{a \sim \pi_{\boldsymbol{\phi}}(\cdot|s)}\left[\alpha \log \pi_{\boldsymbol{\phi}}(a|s) - q_{\boldsymbol{\theta}}(s,a)\right]
$$

Cet objectif déplace la probabilité vers les actions de grande valeur Q, tandis que le terme $\log \pi_{\boldsymbol{\phi}}$ pénalise la concentration de la masse de probabilité et maintient l'entropie. Le bonus d'entropie provient de la structure de la divergence KL plutôt que d'un terme de régularisation explicite.

Pour estimer les gradients de cet objectif, on se heurte à un problème technique : les paramètres de la politique $\boldsymbol{\phi}$ apparaissent dans la distribution d'échantillonnage $\pi_{\boldsymbol{\phi}}$, ce qui rend $\nabla_{\boldsymbol{\phi}} \mathbb{E}_{a \sim \pi_{\boldsymbol{\phi}}}[\cdot]$ difficile à calculer. SAC utilise une politique gaussienne $\pi_{\boldsymbol{\phi}}(a|s) = \mathcal{N}(\mu_{\boldsymbol{\phi}}(s), \sigma_{\boldsymbol{\phi}}(s)^2)$ avec l'astuce de reparamétrisation. On exprime les échantillons comme une fonction déterministe des paramètres et d'un bruit indépendant :

$$
a = f_{\boldsymbol{\phi}}(s, \epsilon) = \mu_{\boldsymbol{\phi}}(s) + \sigma_{\boldsymbol{\phi}}(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Cela fait sortir $\boldsymbol{\phi}$ de la distribution d'échantillonnage pour le placer dans l'intégrande :

$$
\min_{\boldsymbol{\phi}} \mathbb{E}_{s \sim \mathcal{D}}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\left[\alpha \log \pi_{\boldsymbol{\phi}}(f_{\boldsymbol{\phi}}(s,\epsilon)|s) - q_{\boldsymbol{\theta}}(s,f_{\boldsymbol{\phi}}(s,\epsilon))\right]
$$

On peut maintenant dériver à travers $f_{\boldsymbol{\phi}}$ et le réseau Q, comme DDPG dérive à travers une politique déterministe. SAC prolonge cette idée en tirant un bruit $\epsilon$ à chaque pas de gradient plutôt que de renvoyer une seule action déterministe.


```{prf:algorithm} Acteur-critique adouci (SAC)
:label: sac

**Entrée :** processus de décision markovien $(S, \mathcal{A}, P, R, \gamma)$, réseaux Q jumeaux $q^1(s,a; \boldsymbol{\theta}^1), q^2(s,a; \boldsymbol{\theta}^2)$, politique $\pi_{\boldsymbol{\phi}}$, taux d'apprentissage $\alpha_q, \alpha_\pi$, capacité du tampon de rejeu $B$, taille des mini-lots $b$, taux de la moyenne mobile exponentielle $\tau$, poids de l'entropie $\alpha$

**Sortie :** paramètres des fonctions Q $\boldsymbol{\theta}^1, \boldsymbol{\theta}^2$, paramètres de la politique $\boldsymbol{\phi}$

1. Initialiser $\boldsymbol{\theta}^1_0, \boldsymbol{\theta}^2_0, \boldsymbol{\phi}_0$ aléatoirement
2. $\boldsymbol{\theta}^1_{\text{target}} \leftarrow \boldsymbol{\theta}^1_0$, $\boldsymbol{\theta}^2_{\text{target}} \leftarrow \boldsymbol{\theta}^2_0$
3. Initialiser le tampon de rejeu $\mathcal{B}$ de capacité $B$
4. $t \leftarrow 0$
5. **tant que** l'entraînement se poursuit **faire**
    1. Observer l'état $s$
    2. Tirer une action : $a \sim \pi_{\boldsymbol{\phi}_t}(\cdot|s)$ $\quad$ // La politique stochastique assure l'exploration
    3. Exécuter $a$, observer la récompense $r$ et l'état suivant $s'$
    4. Stocker $(s,a,r,s')$ dans $\mathcal{B}$, en remplaçant la plus ancienne transition si le tampon est plein
    5. Tirer de $\mathcal{B}$ un mini-lot $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$
    6. **// Mettre à jour les réseaux Q : autoamorçage avec une estimation à un échantillon de la valeur adoucie**
    7. **pour** chaque $s'_i$ **faire** tirer $\tilde{a}'_i \sim \pi_{\boldsymbol{\phi}_t}(\cdot|s'_i)$
    8. $y_i \leftarrow r_i + \gamma \left[\min(q^1(s'_i, \tilde{a}'_i; \boldsymbol{\theta}^1_{\text{target}}), q^2(s'_i, \tilde{a}'_i; \boldsymbol{\theta}^2_{\text{target}})) - \alpha \log \pi_{\boldsymbol{\phi}_t}(\tilde{a}'_i|s'_i)\right]$
    9. $\boldsymbol{\theta}^1_{t+1} \leftarrow \boldsymbol{\theta}^1_t - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b(q^1(s_i,a_i;\boldsymbol{\theta}^1_t) - y_i)^2$
    10. $\boldsymbol{\theta}^2_{t+1} \leftarrow \boldsymbol{\theta}^2_t - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b(q^2(s_i,a_i;\boldsymbol{\theta}^2_t) - y_i)^2$
    11. **// Mettre à jour la politique : minimiser la divergence KL à la distribution de Boltzmann**
    12. **pour** chaque $s_i$ **faire** tirer $\epsilon_i \sim \mathcal{N}(0,I)$ et calculer $\hat{a}_i = f_{\boldsymbol{\phi}_t}(s_i, \epsilon_i)$ $\quad$ // Reparamétrisation
    13. $\boldsymbol{\phi}_{t+1} \leftarrow \boldsymbol{\phi}_t - \alpha_\pi \nabla_{\boldsymbol{\phi}} \frac{1}{b}\sum_{i=1}^b \left[\alpha \log \pi_{\boldsymbol{\phi}_t}(\hat{a}_i|s_i) - \min_{j=1,2} q^j(s_i, \hat{a}_i; \boldsymbol{\theta}^j_{t+1})\right]$
    14. **// Mise à jour par moyenne mobile exponentielle des cibles des réseaux Q**
    15. $\boldsymbol{\theta}^1_{\text{target}} \leftarrow \tau \boldsymbol{\theta}^1_{t+1} + (1-\tau)\boldsymbol{\theta}^1_{\text{target}}$
    16. $\boldsymbol{\theta}^2_{\text{target}} \leftarrow \tau \boldsymbol{\theta}^2_{t+1} + (1-\tau)\boldsymbol{\theta}^2_{\text{target}}$
    17. $t \leftarrow t + 1$
6. **retourner** $\boldsymbol{\theta}^1_t, \boldsymbol{\theta}^2_t, \boldsymbol{\phi}_t$
```

L'algorithme entrelace trois mises à jour. Les réseaux Q (lignes 7-10) suivent l'itération Q ajustée avec la cible de Bellman adoucie : tirer une action suivante $\tilde{a}'_i$ de la politique courante, calculer la cible corrigée par l'entropie $y_i = r_i + \gamma[\min_j q^j_{\text{target}}(s'_i, \tilde{a}'_i) - \alpha \log \pi_{\boldsymbol{\phi}}(\tilde{a}'_i|s'_i)]$, puis minimiser l'erreur quadratique. Le minimum sur les réseaux Q jumeaux atténue la surestimation, comme dans TD3. La politique (lignes 12-13) est mise à jour pour approcher la distribution de Boltzmann induite par la fonction Q courante, à l'aide de l'astuce de reparamétrisation pour estimer le gradient. Les réseaux cibles sont mis à jour par moyenne mobile exponentielle (lignes 15-16) pour stabiliser l'entraînement.

La politique stochastique joue le même rôle d'amortissement que dans DDPG et TD3 : elle remplace l'opération intraitable $\arg\max$ par une passe avant rapide dans un réseau. La régularisation entropique de SAC produit l'exploration par le caractère stochastique propre à la politique plutôt que par un bruit externe. SAC est par conséquent moins sensible aux hyperparamètres, et il n'y a aucun calendrier d'exploration à régler.

## Apprentissage par cohérence des chemins (PCL)

DDPG, TD3 et SAC suivent tous le même gabarit de résolution, issu de l'itération Q ajustée : calculer les cibles de Bellman avec la fonction Q courante, ajuster la fonction Q à ces cibles, recommencer. C'est l'**approximation successive**, l'itération fonctionnelle $v_{k+1} = \Proj \Bellman v_k$ du [chapitre sur les méthodes de valeur ajustée](approximate-bellman-equations.md).

L'apprentissage par cohérence des chemins (PCL) {cite:p}`Nachum2017` résout l'équation de Bellman autrement. Au lieu d'itérer l'opérateur, il minimise directement un **résidu**. C'est l'approche par moindres carrés des méthodes des résidus pondérés : résoudre $\Residual(v) = 0$ en minimisant $\|\Residual(v)\|^2$. Elle repose sur une structure que les méthodes à un pas n'utilisent pas : l'opérateur de Bellman lissé sous une dynamique déterministe.

### La propriété de cohérence des chemins

Considérons l'équation de Bellman de la fonction Q issue de [la programmation dynamique régularisée par l'entropie](regularized-dp.md). Sous une dynamique stochastique générale, elle fait intervenir une espérance sur les états suivants :

$$
q^*(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}[v^*(s')]
$$

Supposons la dynamique déterministe : $s' = f(s,a)$. L'état suivant est déterminé de façon unique, et l'espérance disparaît :

$$
q^*(s,a) = r(s,a) + \gamma v^*(f(s,a))
$$

La fonction de valeur est reliée aux fonctions Q par le maximum adouci :

$$
v^*(s) = \alpha \log \int_{\mathcal{A}} \exp(q^*(s,a)/\alpha) da
$$

Opposons deux cas : les politiques générales et la politique de Boltzmann optimale.

**Pour des politiques générales**, la valeur est égale à une espérance :

$$
v^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}[q^\pi(s,a) - \alpha \log \pi(a|s)]
$$ (eq:v-general-policy)

Il s'agit d'une moyenne. Pour une seule action observée $a$, on a :

$$
q^\pi(s,a) - \alpha\log\pi(a|s) = v^\pi(s) + \varepsilon(s,a)
$$

où $\varepsilon(s,a)$ est une erreur d'échantillonnage telle que $\mathbb{E}_{a \sim \pi}[\varepsilon(s,a)] = 0$. Les actions individuelles donnent des estimations bruitées qui fluctuent autour de la moyenne.

**Pour la politique optimale sous régularisation entropique**, la forme de Boltzmann de la politique donne une identité qui vaut exactement pour chaque action. La politique optimale est :

$$
\pi^*(a|s) = \frac{\exp(q^*(s,a)/\alpha)}{\exp(v^*(s)/\alpha)}
$$

En prenant le logarithme et en réarrangeant :

$$
v^*(s) = q^*(s,a) - \alpha \log \pi^*(a|s) \quad \text{pour tout } a
$$ (eq:v-q-exact-boltzmann)

Cette relation vaut exactement pour chaque action $a$, et pas seulement en espérance. Il n'y a aucune erreur d'échantillonnage. L'avantage $q^*(s,a) - v^*(s)$ est encodé dans la log-probabilité : les actions sous-optimales ont un faible $q^*(s,a)$, mais aussi un grand $-\alpha\log\pi^*(a|s)$ (une faible probabilité correspond à une log-probabilité très négative), et ces termes s'équilibrent exactement pour donner $v^*(s)$.

Prenons maintenant un segment de trajectoire $(s_0, a_0, s_1, a_1, \ldots, s_d)$ dont chaque transition suit la dynamique déterministe $s_{t+1} = f(s_t, a_t)$. Partons de $q^*(s_0, a_0) = r_0 + \gamma v^*(s_1)$ et utilisons l'équation {eq}`eq:v-q-exact-boltzmann` pour substituer exactement $v^*(s_1) = q^*(s_1,a_1) - \alpha\log\pi^*(a_1|s_1)$ :

$$
q^*(s_0, a_0) = r_0 + \gamma[q^*(s_1,a_1) - \alpha\log\pi^*(a_1|s_1)]
$$

Substituons $q^*(s_1,a_1) = r_1 + \gamma v^*(s_2)$ :

$$
q^*(s_0, a_0) = r_0 + \gamma r_1 - \gamma\alpha\log\pi^*(a_1|s_1) + \gamma^2 v^*(s_2)
$$

Poursuivons ce télescopage sur $d$ pas. Chaque substitution est exacte :

$$
q^*(s_0, a_0) = \sum_{t=0}^{d-1} \gamma^t r_t - \alpha \sum_{t=1}^{d-1} \gamma^t \log\pi^*(a_t|s_t) + \gamma^d v^*(s_d)
$$

Appliquons une fois de plus l'équation {eq}`eq:v-q-exact-boltzmann` pour obtenir $v^*(s_0) = q^*(s_0,a_0) - \alpha\log\pi^*(a_0|s_0)$ :

$$
v^*(s_0) = \sum_{t=0}^{d-1} \gamma^t [r_t - \alpha\log\pi^*(a_t|s_t)] + \gamma^d v^*(s_d)
$$ (eq:path-consistency-exact)

En réarrangeant, on obtient le **résidu de cohérence des chemins** :

$$
R(s_0:s_d; \pi^*, v^*) = v^*(s_0) - \gamma^d v^*(s_d) - \sum_{t=0}^{d-1} \gamma^t[r_t - \alpha\log\pi^*(a_t|s_t)] = 0
$$ (eq:path-residual)

L'équation télescopée est une identité exacte : $R = 0$ pour toute suite d'actions, et pas seulement en espérance. La politique de comportement n'apparaît jamais, car la contrainte vaut comme une identité déterministe pour toute suite observée $(s_0, a_0, \ldots, s_d)$. Cela permet l'apprentissage hors politique sans échantillonnage préférentiel.

```{prf:remark} Politiques générales et politiques de Boltzmann optimales
:class: dropdown

La distinction entre les équations {eq}`eq:v-general-policy` et {eq}`eq:v-q-exact-boltzmann` est subtile, mais cruciale.

**Pour des politiques générales** (équation {eq}`eq:v-general-policy`), la valeur est une moyenne sur des actions tirées de la politique. Les actions individuelles donnent des estimations bruitées : si l'on tire $a \sim \pi(\cdot|s)$, alors $q^\pi(s,a) - \alpha\log\pi(a|s) = v^\pi(s) + \varepsilon$, où $\varepsilon$ est une variable aléatoire de moyenne nulle. Il faut faire la moyenne de nombreux échantillons pour estimer $v^\pi(s)$ avec précision. Un télescopage à plusieurs pas accumulerait ces erreurs d'échantillonnage $\varepsilon_0, \varepsilon_1, \ldots, \varepsilon_{d-1}$ et produirait des résidus bruités même à la vraie solution. L'apprentissage hors politique exigerait des poids d'importance pour corriger l'utilisation d'actions issues d'une autre politique de comportement.

**Pour la politique optimale régularisée par l'entropie** (équation {eq}`eq:v-q-exact-boltzmann`), la structure de Boltzmann réduit l'espérance à une identité ponctuelle. La relation $v^*(s) = q^*(s,a) - \alpha\log\pi^*(a|s)$ vaut exactement pour chaque action $a$, optimale ou non. Une action sous-optimale a un faible $q^*(s,a)$ (un faible rendement espéré) et un faible $\pi^*(a|s)$ (une faible probabilité), ce qui rend $-\alpha\log\pi^*(a|s)$ grand. Ces termes s'équilibrent précisément pour donner $v^*(s)$. Il n'existe aucune erreur d'échantillonnage. Le télescopage est exact et produit un résidu nul pour toute suite d'actions, et pas seulement en espérance. L'apprentissage hors politique fonctionne parce que la contrainte vaut comme une identité déterministe pour tout chemin observé.

Cette propriété est propre aux opérateurs de maximum adouci. Pour le maximum strict, $v^*(s) = \max_a q^*(s,a)$ ne vaut que lorsque $a$ est optimale. Les actions sous-optimales satisfont $v^*(s) > q^*(s,a)$, une inégalité qui ne permet pas de construire un résidu.
```

### Exigences structurelles : dynamique déterministe et régularisation entropique

PCL a deux exigences, une dynamique déterministe et une régularisation entropique, et chacune lève un obstacle précis.

#### Dynamique déterministe : éviter le problème du double échantillonnage

Sous une dynamique stochastique, l'équation de Bellman de la fonction Q comporte une espérance sur les états suivants :

$$
q^*(s,a) = r(s,a) + \gamma \mathbb{E}_{s' \sim p(\cdot|s,a)}[v^*(s')]
$$

La relation exacte {eq}`eq:v-q-exact-boltzmann` vaut toujours, de sorte qu'on peut écrire la contrainte de cohérence des chemins. Mais considérons maintenant ce que PCL minimise : le résidu **au carré** $\mathbb{E}[R^2]$, où

$$
R = v_{\boldsymbol{\phi}}(s_0) - \gamma^d v_{\boldsymbol{\phi}}(s_d) - \sum_{t=0}^{d-1} \gamma^t[r_t - \alpha\log\pi_{\boldsymbol{\theta}}(a_t|s_t)]
$$

Au véritable optimum $(v^*, \pi^*)$, la contrainte est $\mathbb{E}[R] = 0$, ce qui entraîne $(\mathbb{E}[R])^2 = 0$. Mais PCL minimise $\mathbb{E}[R^2]$ et, par l'inégalité de Jensen :

$$
\mathbb{E}[R^2] \geq (\mathbb{E}[R])^2
$$

avec égalité seulement lorsque $R$ est de variance nulle. Sous une dynamique stochastique, même à l'optimum, les résidus des trajectoires individuelles sont des variables aléatoires de moyenne nulle mais de variance positive (à cause du bruit de transition). Ramener $\mathbb{E}[R^2]$ à zéro exigerait de faire tendre $\text{Var}(R) \to 0$, ce qui est impossible et éloigne la solution du véritable optimum.

C'est le **problème du double échantillonnage** de Baird {cite:p}`Baird1995`. Pour obtenir un gradient sans biais de $(\mathbb{E}[R])^2$, il faut :

$$
\nabla (\mathbb{E}[R])^2 = 2\mathbb{E}[R] \cdot \nabla \mathbb{E}[R] = 2\mathbb{E}[R] \cdot \mathbb{E}[\nabla R]
$$

Cela exige deux échantillons indépendants de l'état suivant à partir de la même paire $(s,a)$ : l'un pour estimer $\mathbb{E}[R]$ et l'autre pour $\mathbb{E}[\nabla R]$. Avec un simulateur, c'est possible. Avec des trajectoires réelles, ce ne l'est pas.

Sous une dynamique déterministe, $R$ est déterministe (aucun bruit de transition), donc $\mathbb{E}[R^2] = (\mathbb{E}[R])^2$ et l'inégalité de Jensen devient une égalité. Minimiser le résidu au carré équivaut à résoudre $\mathbb{E}[R] = 0$.

#### Régularisation entropique : permettre la cohérence pour toutes les actions

Tentons la même dérivation de la cohérence des chemins avec l'opérateur de Bellman à maximum strict. Sous une dynamique déterministe, la fonction Q satisfait :

$$
q^*(s,a) = r(s,a) + \gamma v^*(f(s,a))
$$

où $v^*(s) = \max_{a'} q^*(s,a')$ et où la politique optimale est $\pi^*(s) = \arg\max_a q^*(s,a)$ (déterministe).

Essayons maintenant de relier $v^*(s)$ à une action observée arbitraire $a$. Pour l'action optimale $a^* \in \arg\max_{a'} q^*(s,a')$, on a :

$$
v^*(s) = q^*(s,a^*)
$$

Mais pour une action sous-optimale $a \ne a^*$ :

$$
v^*(s) = \max_{a'} q^*(s,a') > q^*(s,a)
$$

C'est une inégalité, et non une équation. Aucune formule n'exprime $v^*(s)$ en fonction de $q^*(s,a)$ pour des actions sous-optimales.

Tentons le télescopage à plusieurs pas. Partons de $q^*(s_0, a_0) = r_0 + \gamma v^*(s_1)$. Pour continuer, il faut exprimer $v^*(s_1)$ à l'aide de l'action observée $a_1$. Mais on dispose seulement de :

$$
v^*(s_1) \geq q^*(s_1, a_1)
$$

avec égalité seulement si $a_1$ se trouve être optimale en $s_1$. On ne peut pas substituer cette relation dans l'équation de la fonction Q pour obtenir un télescopage exact. La dérivation échoue dès la première étape.

Comparons avec le cas du maximum adouci. La structure de Boltzmann donne l'équation {eq}`eq:v-q-exact-boltzmann` : $v^*(s) = q^*(s,a) - \alpha\log\pi^*(a|s)$ pour toutes les actions $a$. Le terme de log-probabilité compense exactement la sous-optimalité : les actions de faible probabilité ont un grand $-\alpha\log\pi^*(a|s)$, qui s'ajoute au faible $q^*(s,a)$ pour redonner $v^*(s)$. Cela permet une substitution exacte à chaque étape :

$$
v^*(s_1) = q^*(s_1, a_1) - \alpha\log\pi^*(a_1|s_1) \quad \text{(exact pour tout } a_1\text{)}
$$

Le télescopage se poursuit sans inégalités ni restrictions sur les actions choisies. Le Q-learning à plusieurs pas avec maximum strict n'a pas de justification théorique pour des données hors politique, car lorsqu'on observe une trajectoire comportant des actions sous-optimales, on ne peut pas écrire de contrainte exacte de cohérence des chemins.

Les deux exigences sont structurelles :

| **Exigence** | **Obstacle levé** |
|:----------------|:--------------|
| Dynamique déterministe | Biais du double échantillonnage : garantit $\mathbb{E}[R^2] = (\mathbb{E}[R])^2$ |
| Régularisation entropique | Cohérence pour toutes les actions (équation {eq}`eq:v-q-exact-boltzmann`) |

Sans dynamique déterministe, la minimisation du résidu est biaisée. Sans régularisation entropique, la contrainte ne vaut que pour les actions optimales.

### L'objectif d'apprentissage

L'équation {eq}`eq:path-residual` fournit une contrainte que le couple optimal $(v^*, \pi^*)$ doit satisfaire : le résidu est nul pour tout chemin observé. Pour des approximations paramétriques $(v_{\boldsymbol{\phi}}, \pi_{\boldsymbol{\theta}})$ qui ne sont pas encore optimales, le résidu est non nul :

$$
R(s_0:s_d; \boldsymbol{\theta}, \boldsymbol{\phi}) = v_{\boldsymbol{\phi}}(s_0) - \gamma^d v_{\boldsymbol{\phi}}(s_d) - \sum_{t=0}^{d-1} \gamma^t[r_t - \alpha \log \pi_{\boldsymbol{\theta}}(a_t|s_t)]
$$

PCL minimise le résidu au carré sur les segments de chemin observés :

$$
\min_{\boldsymbol{\theta}, \boldsymbol{\phi}} \sum_{\text{segments}} \frac{1}{2} R(s_i:s_{i+d}; \boldsymbol{\theta}, \boldsymbol{\phi})^2
$$

Il s'agit de la minimisation du résidu de Bellman, la condition des moindres carrés du [chapitre sur les fonctions de valeur paramétriques](weighted-residual-methods.md). SAC calcule des cibles $y_i$ et s'y ajuste (approximation successive). PCL minimise directement le résidu, sans calculer de cibles ni effectuer d'étape d'ajustement distincte.

La descente de gradient donne :

$$
\begin{align}
\boldsymbol{\theta}_{k+1} &= \boldsymbol{\theta}_k + \eta_\pi \sum_i R_i \cdot \alpha \sum_{t=0}^{d-1} \gamma^t \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}_k}(a_{i+t}|s_{i+t}) \\
\boldsymbol{\phi}_{k+1} &= \boldsymbol{\phi}_k - \eta_v \sum_i R_i \left[\nabla_{\boldsymbol{\phi}} v_{\boldsymbol{\phi}_k}(s_i) - \gamma^d \nabla_{\boldsymbol{\phi}} v_{\boldsymbol{\phi}_k}(s_{i+d})\right]
\end{align}
$$

où $R_i = R(s_i:s_{i+d}; \boldsymbol{\theta}_k, \boldsymbol{\phi}_k)$. Les grands résidus entraînent de plus grandes mises à jour.

```{prf:algorithm} Apprentissage par cohérence des chemins (PCL)
:label: pcl

**Entrée :** processus de décision markovien à dynamique déterministe $s_{t+1} = f(s_t, a_t)$, politique $\pi_{\boldsymbol{\theta}}$, fonction de valeur $v_{\boldsymbol{\phi}}$, poids de l'entropie $\alpha$, longueur des chemins $d$, taux d'apprentissage $\eta_\pi, \eta_v$, capacité du tampon de rejeu $B$

**Sortie :** paramètres de la politique $\boldsymbol{\theta}$, paramètres de la valeur $\boldsymbol{\phi}$

1. Initialiser $\boldsymbol{\theta}_0$, $\boldsymbol{\phi}_0$
2. Initialiser le tampon de rejeu $\mathcal{R}$ de capacité $B$
3. $k \leftarrow 0$
4. **tant que** l'entraînement se poursuit **faire**
    1. Tirer une trajectoire $\tau = (s_0, a_0, r_0, \ldots, s_T)$ de $\pi_{\boldsymbol{\theta}_k}$ et la stocker dans $\mathcal{R}$
    2. Tirer une trajectoire $\tau'$ de $\mathcal{R}$
    3. **pour** chaque segment de $d$ pas de $\tau'$ **faire**
        1. Calculer le résidu : $R_i \leftarrow v_{\boldsymbol{\phi}_k}(s_i) - \gamma^d v_{\boldsymbol{\phi}_k}(s_{i+d}) - \sum_{t=0}^{d-1} \gamma^t[r_{i+t} - \alpha \log \pi_{\boldsymbol{\theta}_k}(a_{i+t}|s_{i+t})]$
        2. Mettre à jour la politique : $\boldsymbol{\theta}_{k+1} \leftarrow \boldsymbol{\theta}_k + \eta_\pi \alpha R_i \sum_{t=0}^{d-1} \gamma^t \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}_k}(a_{i+t}|s_{i+t})$
        3. Mettre à jour la valeur : $\boldsymbol{\phi}_{k+1} \leftarrow \boldsymbol{\phi}_k - \eta_v R_i\left[\nabla_{\boldsymbol{\phi}} v_{\boldsymbol{\phi}_k}(s_i) - \gamma^d \nabla_{\boldsymbol{\phi}} v_{\boldsymbol{\phi}_k}(s_{i+d})\right]$
    4. Retirer les trajectoires les plus anciennes si $|\mathcal{R}| > B$
    5. $k \leftarrow k + 1$
5. **retourner** $\boldsymbol{\theta}_k$, $\boldsymbol{\phi}_k$
```

L'algorithme recueille des trajectoires de la politique courante et les stocke dans un tampon de rejeu. À chaque itération, il tire une trajectoire (éventuellement ancienne) et effectue une descente de gradient sur le résidu de chemin pour tous les segments de $d$ pas. Le tampon de rejeu permet l'apprentissage hors politique : les trajectoires d'anciennes politiques, les démonstrations d'experts ou les comportements exploratoires fournissent tous des signaux d'entraînement valides.

### Paramétrisation unifiée : un seul réseau Q

{prf:ref}`pcl` utilise des réseaux distincts pour la politique et la valeur. On peut cependant utiliser un seul réseau Q $q_{\boldsymbol{\theta}}(s,a)$ et en dériver les deux :

$$
v_{\boldsymbol{\theta}}(s) = \alpha \log \sum_{a} \exp(q_{\boldsymbol{\theta}}(s,a)/\alpha), \qquad \pi_{\boldsymbol{\theta}}(a|s) = \frac{\exp(q_{\boldsymbol{\theta}}(s,a)/\alpha)}{\sum_{a'} \exp(q_{\boldsymbol{\theta}}(s,a')/\alpha)}
$$

Le résidu de chemin devient :

$$
R(s_i:s_{i+d}; \boldsymbol{\theta}) = v_{\boldsymbol{\theta}}(s_i) - \gamma^d v_{\boldsymbol{\theta}}(s_{i+d}) - \sum_{t=0}^{d-1} \gamma^t[r_{i+t} - \alpha \log \pi_{\boldsymbol{\theta}}(a_{i+t}|s_{i+t})]
$$

et le gradient combine les contributions de la valeur et de la politique au moyen des mêmes paramètres. Avec cette paramétrisation, il n'y a plus d'acteur ni de critique distincts : un seul réseau Q joue les deux rôles.

### Liens avec les méthodes existantes

**Cas à un pas ($d=1$)** : le résidu de chemin devient $R(s:s'; \boldsymbol{\theta}, \boldsymbol{\phi}) = v_{\boldsymbol{\phi}}(s) - \gamma v_{\boldsymbol{\phi}}(s') - r + \alpha\log\pi_{\boldsymbol{\theta}}(a|s)$. Pour la paramétrisation unifiée, où $v_{\boldsymbol{\theta}}(s) = q_{\boldsymbol{\theta}}(s,a) - \alpha\log\pi_{\boldsymbol{\theta}}(a|s)$ exactement, on obtient $R = q_{\boldsymbol{\theta}}(s,a) - r - \gamma v_{\boldsymbol{\theta}}(s')$, le résidu de Bellman adouci. Minimiser $\sum_i R_i^2$ équivaut au Q-learning adouci, bien que SAC résolve ce problème par approximation successive (calcul des cibles, ajustement) plutôt que par minimisation directe du résidu.

**Sans entropie ($\alpha \to 0$)** : le résidu devient $R = v(s_i) - \gamma^d v(s_{i+d}) - \sum_t \gamma^t r_t$, l'opposé de l'avantage à $d$ pas. Mais contrairement à A2C/A3C, où $v$ suit la valeur de la politique courante, la valeur de PCL converge vers $v^*$, car le résidu couple la politique et la valeur au moyen de la condition d'optimalité.

**Plusieurs pas avec maximum strict** : il n'existe aucun analogue. L'opérateur de Bellman à maximum strict $\max_a q(s,a)$ ne possède pas de relation ponctuelle exacte comme l'équation {eq}`eq:v-q-exact-boltzmann`. Un télescopage à plusieurs pas accumulerait des erreurs dues à l'opérateur max, et la contrainte ne serait valide qu'en espérance sous la politique optimale. La structure du maximum adouci permet une cohérence des chemins exacte et hors politique.

### PCL et SAC : minimisation du résidu ou approximation successive

Les deux méthodes résolvent des processus de décision markoviens régularisés par l'entropie, avec des stratégies de résolution différentes :

| **Aspect** | **SAC** | **PCL** |
|:-----------|:--------|:--------|
| **Méthode de résolution** | Approximation successive : calculer des cibles $y_i$, ajuster $q$ aux cibles | Minimisation du résidu : minimiser directement $\sum_i R_i^2$ |
| **Structure des mises à jour** | Calcul des cibles + étape de régression | Un seul pas de gradient sur le résidu au carré |
| **Réseaux cibles** | Nécessaires (marquent les frontières des itérations externes) | Aucun (contrainte sur le résidu, pas d'ajustement à des cibles) |
| **Horizon temporel** | Différence temporelle à un pas : $y = r + \gamma V(s')$ | Chemins à plusieurs pas : cumul sur $d$ pas |
| **Traitement hors politique** | Tampon de rejeu avec biais d'un seul échantillon | Aucun échantillonnage préférentiel (fonctionne pour toute trajectoire) |
| **Exigence sur la dynamique** | Transitions stochastiques générales | Transitions **déterministes** $s' = f(s,a)$ |
| **Architecture** | Réseaux Q jumeaux + réseau de politique | Un seul réseau Q (paramétrisation unifiée) |

PCL exige une dynamique déterministe. Elle y gagne le télescopage à plusieurs pas et l'apprentissage hors politique sans poids d'importance, mais seulement pour des systèmes déterministes (manipulation robotique, nombreuses tâches de commande). SAC fonctionne pour des processus de décision markoviens stochastiques généraux.

### PCL comme amortissement

PCL amortit à un autre niveau que DDPG, TD3 et SAC. Ces méthodes amortissent la maximisation sur les actions : elles apprennent un réseau de politique qui renvoie directement $\arg\max_a q(s,a)$. PCL amortit la résolution de l'équation de Bellman elle-même. Au lieu d'appliquer à répétition l'opérateur de Bellman (qui exige $\int_{\mathcal{A}} \exp(q/\alpha) da$ à chaque itération), PCL tire des segments de chemin et minimise leur résidu. Le coût de calcul de la vérification de l'optimalité sur tous les états et toutes les longueurs de chemin est réparti sur l'entraînement au moyen de mises à jour de gradient échantillonnées.

## Commande prédictive par intégrale de chemin

Le [développement de MPPI](model-predictive-path-integral-control.md#sec-mppi) construit
une distribution sur des suites finies de commandes en inclinant exponentiellement une
distribution de référence vers les trajectoires de faible coût. Une proposition
gaussienne décalée rend ces trajectoires plus faciles à échantillonner. Des poids
d'importance corrigent ensuite le changement de proposition, et leurs perturbations
pondérées mettent à jour la suite nominale. Le correcteur applique sa première action
et décale le reste de la suite pour initialiser l'optimisation suivante.

Ce calcul partage une forme exponentielle avec les politiques issues de la
régularisation entropique, mais les distributions et les problèmes d'optimisation
exigent des définitions distinctes. SAC régularise une politique d'action
conditionnelle à l'état. La construction de MPPI fondée sur la théorie de
l'information régularise une distribution sur des trajectoires relativement à une
loi de référence précisée {cite:p}`Williams2017InformationTheoretic`. Dans la
[dérivation par intégrale de chemin en temps continu](path-integral-stochastic-control.md#sec-path-integral-control),
qui est distincte, la loi de référence est engendrée par une diffusion
non commandée, et le coût quadratique de la commande doit être accordé à la
covariance de son bruit. Aucune de ces dérivations n'identifie une mise en œuvre
arbitraire de MPPI à SAC ou à la solution de Bellman adoucie de PCL.

La mise à jour de la moyenne gaussienne ajuste la distribution de trajectoires
inclinée en minimisant la divergence KL de cette distribution à la famille
gaussienne. Elle ne minimise pas nécessairement l'objectif original de coût espéré
sur les politiques gaussiennes. Sa mise en œuvre échantillonnée comporte aussi une
erreur de discrétisation et une erreur d'échantillonnage fini.
L'[exemple de l'avion](model-predictive-path-integral-control.md#sec-aircraft-mppi)
estime aussi le coût espéré de chaque candidat sur les vents futurs possibles.
Pour le modèle de commande stochastique correspondant, les
[exemples de passages browniens](path-integral-stochastic-control.md#sec-brownian-passages)
comparent la rétroaction échantillonnée à des solutions de référence analytiques et
numériques.

### Optimisation en ligne et amortissement

Un réseau acteur répartit le calcul sur les exemples d'entraînement et renvoie
une action par une passe avant. MPPI consacre des évaluations du modèle à l'état
courant, avec un démarrage à chaud à partir du plan précédent. Une nouvelle
fonction de coût peut être utilisée dès l'optimisation suivante, même si un calcul
limité et une mauvaise proposition peuvent encore empêcher une mise à jour utile.

La comparaison des calculs sépare l'endroit où l'effort est dépensé des erreurs
qui subsistent :

| Aspect | Acteur appris | MPPI |
|:--|:--|:--|
| Sélection de l'action | Évaluation du réseau de politique | Lots répétés de déroulements du modèle |
| Réutilisation d'une décision à l'autre | Paramètres entraînés | Suite nominale de commandes décalée |
| Usage du modèle au déploiement | Souvent inutile | Un modèle de déroulement est nécessaire |
| Sources d'approximation | Approximation de la politique et de la valeur, estimation, optimisation | Erreur de modèle, discrétisation, proposition restreinte, échantillonnage fini |
| Objectif modifié | Exige généralement un entraînement supplémentaire | Entre dans les coûts des déroulements suivants |

Pour un modèle de déroulement déterministe, une itération de MPPI avec $K$ suites et
$H$ pas de temps utilise $KH$ évaluations de transition. Estimer le coût espéré de
chaque suite sur $L$ scénarios de perturbation porte ce nombre à $KHL$. Ces
évaluations peuvent être regroupées en lots sur les suites et les scénarios, tandis
que chaque trajectoire avance toujours dans le temps. Le choix pratique dépend du
coût du modèle, du calcul parallèle disponible et de la précision exigée à chaque
décision.

## Une autre voie : les méthodes fondées sur l'équation d'Euler

Les méthodes développées dans ce chapitre paramètrent toutes des politiques, mais elles restent ancrées dans l'équation de Bellman. NFQCA, DDPG, TD3 et SAC apprennent des fonctions Q par approximation successive, puis en dérivent des politiques en maximisant ces fonctions Q. PCL minimise un résidu de chemin dérivé de l'équation de Bellman adoucie. La politique sert d'optimiseur amorti pour un objectif fondé sur la valeur.

Une approche différente, développée en économie computationnelle {cite:p}`Judd1992,Rust1996,ndp`, paramètre elle aussi les politiques, mais résout une tout autre équation fonctionnelle. Considérons un problème de commande à états et actions continus, à dynamique déterministe $s' = f(s,a)$ et à récompense dérivable $r(s,a)$. L'action optimale $\pi^*(s)$ satisfait la condition du premier ordre :

$$
\frac{\partial r(s,a)}{\partial a}\Big|_{a=\pi^*(s)}
+
\gamma\, \frac{\partial v^*(s')}{\partial s'}\Big|_{s'=f(s,\pi^*(s))}\,
\frac{\partial f(s,a)}{\partial a}\Big|_{a=\pi^*(s)}
=
0.
$$

Cette équation d'Euler exprime l'optimalité au moyen de dérivées plutôt qu'au moyen de l'opérateur max. Pour des problèmes à structure particulière (la classe d'Euler, où la dynamique est affine en l'état commandé), les théorèmes de l'enveloppe éliminent entièrement $v^*$ et donnent une équation fonctionnelle fermée $\mathcal{E}(\pi)(s) = 0$ portant sur la seule politique.

Avec une politique paramétrée $\pi_{\boldsymbol{\theta}}(s)$, on peut discrétiser par collocation ou par projection de Galerkin :

$$
G(\boldsymbol{\theta}) := \begin{bmatrix}
\mathcal{E}(\pi_{\boldsymbol{\theta}})(s_1) \\
\vdots \\
\mathcal{E}(\pi_{\boldsymbol{\theta}})(s_N)
\end{bmatrix} = 0.
$$

Il s'agit d'une recherche de racines, et non d'une itération de point fixe. Des méthodes de type Newton remplacent l'approximation successive de l'itération Q ajustée. L'opérateur d'Euler n'est pas une contraction, de sorte que les garanties de convergence dépendent du problème.

Qu'est-ce que cela signifie pour l'apprentissage par renforcement? L'approche d'Euler partage l'idée d'amortissement : apprendre un réseau de politique qui renvoie directement des actions. Mais l'objectif d'entraînement provient des conditions d'optimalité du premier ordre plutôt que des résidus de Bellman ou de la maximisation d'une fonction Q. Les objectifs de type Euler fournissent-ils des signaux d'entraînement utiles aux méthodes acteur-critique, et les conditions du premier ordre aident-elles lorsque la dynamique est connue ou apprise? Ces questions restent ouvertes; les deux traditions ont eu peu de contacts.

## Synthèse

SAC convertit l'intégrale de la valeur adoucie en une espérance sous une politique
stochastique apprise. PCL minimise plutôt un résidu de cohérence à plusieurs pas, et
MPPI met à jour une distribution de trajectoires à horizon fini à chaque décision. L'approche d'Euler offre une autre solution de rechange : résoudre une
équation de politique du premier ordre plutôt qu'un point fixe de Bellman.

MPPI évite une approximation persistante par un acteur en effectuant l'échantillonnage
et l'optimisation en ligne. Ses poids exponentiels ressemblent à ceux des politiques
régularisées par l'entropie, mais sa distribution de référence, son horizon et ses
hypothèses d'échantillonnage définissent un problème d'optimisation différent. Un
calcul limité et l'erreur de modèle subsistent dans l'une comme dans l'autre approche.

Tous ces objectifs conservent une équation d'optimalité dérivée d'une relation de
valeur ou de cohérence des chemins. Peut-on dériver directement le rendement espéré
sans d'abord prendre une équation de Bellman comme objectif d'entraînement? [Les
gradients de politique et les méthodes acteur-critique](policy-gradients.md) utilisent
le score de la trajectoire et réintroduisent les fonctions de valeur comme critiques
réducteurs de variance.

## Autoévaluation

:::{exercise} Compromis de calcul
:label: ex-amortization-check-2

Comparez MPPI et un réseau acteur sur le plan du calcul en ligne et de l'erreur d'approximation.
:::

:::{solution} ex-amortization-check-2
:class: dropdown

MPPI consacre de nombreux déroulements du modèle à chaque décision et évite une
approximation persistante par un acteur, mais conserve les erreurs de modèle, de
discrétisation, de proposition et d'échantillonnage fini. Un acteur est peu coûteux
en ligne, mais peut introduire une erreur, car ses paramètres ne font qu'approcher
la règle de décision souhaitée. Une suite MPPI démarrée à chaud et un acteur entraîné
réutilisent le calcul de manières différentes.
:::
