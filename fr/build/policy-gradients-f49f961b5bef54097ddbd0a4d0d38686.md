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
# Gradients de politique et méthodes acteur-critique

Les chapitres précédents optimisaient les politiques au moyen de fonctions Q, de
relations de Bellman adoucies ou d'équations de cohérence des chemins. Le
rendement espéré lui-même peut-il fournir l'objectif de la politique sans qu'on
dérive la dynamique de l'environnement? L'estimateur par fonction de score
s'applique directement aux trajectoires, car les termes de politique sont les
seuls facteurs de leur loi de probabilité qui dépendent des paramètres de la
politique.

L'estimateur exige seulement de pouvoir évaluer et dériver $\log
\pi_{\boldsymbol{w}}(a|s)$; il fonctionne donc aussi avec des actions discrètes,
pour lesquelles la reparamétrisation n'est pas disponible.

Soit $G(\tau) \equiv \sum_{t=0}^T r(s_t, a_t)$ la somme des récompenses non actualisées d'une trajectoire $\tau$. Le problème d'optimisation stochastique auquel nous faisons face consiste à maximiser :

$$
J(\boldsymbol{w}) = \mathbb{E}_{\tau \sim p(\tau;\boldsymbol{w})}[G(\tau)]
$$

où $\tau = (s_0,a_0,s_1,a_1,...)$ est une trajectoire et $G(\tau)$ le rendement total.
En appliquant l'estimateur par fonction de score, on obtient :

$$
\begin{align*}
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) &= \nabla_{\boldsymbol{w}}\mathbb{E}_{\tau}[G(\tau)] \\
&= \mathbb{E}_{\tau}\left[G(\tau)\nabla_{\boldsymbol{w}}\log p(\tau;\boldsymbol{w})\right] \\
&= \mathbb{E}_{\tau}\left[G(\tau)\nabla_{\boldsymbol{w}}\sum_{t=0}^T\log \pi_{\boldsymbol{w}}(a_t|s_t)\right] \\
&= \mathbb{E}_{\tau}\left[G(\tau)\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\right]
\end{align*}
$$

Cet estimateur n'exige plus de connaître les probabilités de transition, car la probabilité d'une trajectoire se factorise ainsi :

$$
p(\tau;\boldsymbol{w}) = p(s_0)\prod_{t=0}^T \pi_{\boldsymbol{w}}(a_t|s_t)p(s_{t+1}|s_t,a_t)
$$

Seule la politique dépend donc de $\boldsymbol{w}$. En prenant le logarithme de ce produit, on obtient une somme dans laquelle tous les termes indépendants de $\boldsymbol{w}$ disparaissent. L'estimateur final tire des trajectoires selon la distribution $p(\tau; \boldsymbol{w})$ et calcule :

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) \approx \frac{1}{N}\sum_{i=1}^N\left[G(\tau^{(i)})\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)})\right]
$$

C'est une application directe de l'estimateur par fonction de score. En pratique, on utilise toutefois rarement cette forme; on y apporte plutôt plusieurs améliorations pour réduire davantage la variance.

### Tirer parti de l'indépendance conditionnelle

Étant donné la propriété de Markov du processus de décision markovien, les récompenses $r_k$ pour $k < t$ sont conditionnellement indépendantes de l'action $a_t$ étant donné l'historique $h_t = (s_0,a_0,...,s_{t-1},a_{t-1},s_t)$. Il suffit donc de considérer les récompenses futures pour calculer les gradients de politique.

$$
\begin{align*}
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) &= \mathbb{E}_{\tau}\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\sum_{k=0}^T r_k\right] \\
&= \mathbb{E}_{\tau}\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\left(\sum_{k=0}^{t-1} r_k + \sum_{k=t}^T r_k\right)\right] \\
&= \mathbb{E}_{\tau}\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\sum_{k=t}^T r_k\right]
\end{align*}
$$

L'hypothèse d'indépendance conditionnelle signifie que le terme $\mathbb{E}_{\tau}\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\sum_{k=0}^{t-1} r_k \right]$ s'annule. Pour le voir, factorisons la distribution des trajectoires ainsi :

$$
p(\tau) = p(s_0,...,s_t,a_0,...,a_{t-1}) \pi_{\boldsymbol{w}}(a_t|s_t) p(s_{t+1},...,s_T,a_{t+1},...,a_T|s_t,a_t)
$$

On peut maintenant réécrire un seul terme de cette somme comme suit :

$$
\mathbb{E}_{\tau}\left[\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\sum_{k=0}^{t-1} r_k\right] = \mathbb{E}_{s_{0:t},a_{0:t-1}}\left[\sum_{k=0}^{t-1} r_k \, \mathbb{E}_{a_t}\left[\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\right]\right]
$$

L'espérance intérieure est nulle, car

$$
\begin{align*}
\mathbb{E}_{a_t}\left[\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\right] &= \int \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\pi_{\boldsymbol{w}}(a_t|s_t)da_t \\
&= \int \frac{\nabla_{\boldsymbol{w}}\pi_{\boldsymbol{w}}(a_t|s_t)}{\pi_{\boldsymbol{w}}(a_t|s_t)}\pi_{\boldsymbol{w}}(a_t|s_t)da_t \\
&= \int \nabla_{\boldsymbol{w}}\pi_{\boldsymbol{w}}(a_t|s_t)da_t \\
&= \nabla_{\boldsymbol{w}}\int \pi_{\boldsymbol{w}}(a_t|s_t)da_t \\
&= \nabla_{\boldsymbol{w}}1 = 0
\end{align*}
$$ (eq:score-function-zero-mean)

L'estimateur de Monte-Carlo devient :

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) \approx \frac{1}{N}\sum_{i=1}^N\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)})\sum_{k=t}^T r_k^{(i)}\right]
$$

On obtient ainsi l'algorithme REINFORCE :

```{prf:algorithm} REINFORCE
:label: reinforce

**Entrée :** paramétrisation de la politique $\pi_{\boldsymbol{w}}(a|s)$  
**Sortie :** paramètres de la politique mis à jour $\boldsymbol{w}$  
**Hyperparamètres :** taux d'apprentissage $\alpha$, nombre d'épisodes $N$, longueur des épisodes $T$

1. Initialiser les paramètres $\boldsymbol{w}$
2. Pour épisode = 1, ..., $N$, faire :
   1. Recueillir une trajectoire $\tau = (s_0, a_0, r_0, ..., s_T, a_T, r_T)$ avec la politique $\pi_{\boldsymbol{w}}(a|s)$
   2. Calculer les rendements : $G_t = \sum_{k=t}^T r_k$ pour $t = 0, ..., T$
   3. Calculer l'estimation du gradient : $\hat{g} = \sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t) G_t$
   4. Mettre à jour la politique : $\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha \hat{g}$
3. Retourner $\boldsymbol{w}$
```

L'avantage de cet estimateur par rapport à l'estimateur naïf (qui pondérerait chaque fonction de score par le rendement total de la trajectoire $G(\tau)$) est qu'il a généralement une variance plus faible. Cette réduction de variance découle de la structure d'indépendance conditionnelle que nous avons exploitée : les récompenses passées ne dépendent pas des actions futures. Plus formellement, cet estimateur est une instance d'une technique de réduction de variance appelée méthode de Monte-Carlo conditionnelle étendue.

#### Le point de vue de la perte de substitution

L'algorithme ci-dessus calcule explicitement une estimation du gradient $\hat{g}$. En pratique, les implémentations qui reposent sur des cadres de différentiation automatique procèdent autrement : elles définissent une **perte de substitution** dont le gradient coïncide avec l'estimateur REINFORCE. Pour une seule trajectoire, considérons :

$$
L_{\text{surrogate}}(\boldsymbol{w}) = -\sum_{t=0}^T \log \pi_{\boldsymbol{w}}(a_t|s_t) \, G_t
$$

où les rendements $G_t$ et les actions $a_t$ sont traités comme des constantes fixes (détachées du graphe de calcul). En prenant le gradient par rapport à $\boldsymbol{w}$ :

$$
\nabla_{\boldsymbol{w}} L_{\text{surrogate}} = -\sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t) \, G_t
$$

Minimiser cette perte de substitution par descente de gradient donne la même mise à jour que maximiser le rendement espéré par REINFORCE. Le signe négatif convertit notre problème de maximisation en un problème de minimisation adapté aux optimiseurs standard.

Cette perte de substitution n'est **pas** le rendement espéré $J(\boldsymbol{w})$ que l'on cherche à maximiser. C'est un artifice de calcul qui produit le bon gradient aux valeurs courantes des paramètres. Plusieurs propriétés la distinguent d'une véritable fonction de perte :

1. **Elle change à chaque itération.** Les rendements $G_t$ proviennent de trajectoires tirées sous la politique courante. Après la mise à jour de $\boldsymbol{w}$, il faut recueillir de nouvelles trajectoires et construire une nouvelle perte de substitution.

2. **Sa valeur n'a pas de signification.** Contrairement à l'apprentissage supervisé, où la perte mesure l'erreur de prédiction, la valeur numérique de $L_{\text{surrogate}}$ n'a pas d'interprétation directe. Seul son gradient compte.

3. **Elle n'est valide que localement.** La perte de substitution ne fournit le bon gradient qu'aux paramètres utilisés pour recueillir les données. S'éloigner beaucoup de ces paramètres invalide l'estimation du gradient.

Ce point de vue explique pourquoi le code des gradients de politique paraît souvent différent du pseudocode ci-dessus. Au lieu de calculer explicitement $\hat{g}$, les implémentations définissent la perte de substitution et appellent `loss.backward()` :

```python
# Surrogate loss implementation (single trajectory)
log_probs = [policy.log_prob(a_t, s_t) for s_t, a_t in trajectory]
returns = compute_returns(rewards)
surrogate_loss = -sum(lp * G for lp, G in zip(log_probs, returns))
surrogate_loss.backward()  # computes REINFORCE gradient
optimizer.step()
```

### Réduction de variance par variables de contrôle

Rappelons que l'estimateur de gradient REINFORCE, après exploitation de l'indépendance conditionnelle, prend la forme :

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) \approx \frac{1}{N}\sum_{i=1}^N\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)})\sum_{k=t}^T r_k^{(i)}\right]
$$

Il s'agit d'une somme sur les trajectoires et les pas de temps. La contribution au gradient au pas de temps $t$ de la trajectoire $i$ est :

$$
\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)})\sum_{k=t}^T r_k^{(i)}
$$

Bien que sans biais, cet estimateur souffre d'une forte variance, car le rendement $\sum_{k=t}^T r_k$ peut varier considérablement d'une trajectoire à l'autre, même pour une même paire état-action. La méthode des variables de contrôle offre une façon rigoureuse de réduire cette variance.

#### Théorie générale des variables de contrôle

Pour un estimateur général $Z$ d'une quantité $\mu = \mathbb{E}[Z]$ et une variable de contrôle $C$ d'espérance connue $\mathbb{E}[C]=0$, on peut construire :

$$
Z_{\text{cv}} = Z - \alpha C
$$

Cet estimateur reste sans biais, puisque $\mathbb{E}[Z_{\text{cv}}] = \mathbb{E}[Z] - \alpha\mathbb{E}[C] = \mathbb{E}[Z]$. Sa variance est :

$$
\text{Var}(Z_{\text{cv}}) = \text{Var}(Z) + \alpha^2\text{Var}(C) - 2\alpha\text{Cov}(Z,C)
$$

C'est le terme $-2\alpha\text{Cov}(Z,C)$ qui permet de réduire la variance. Si $Z$ et $C$ sont positivement corrélées, on peut choisir $\alpha > 0$ pour rendre ce terme négatif et grand en valeur absolue, ce qui réduit la variance globale. Toutefois, le terme $\alpha^2\text{Var}(C)$ croît quadratiquement avec $\alpha$ : si l'on prend $\alpha$ trop grand, ce terme quadratique finit par dominer et la variance augmente au lieu de diminuer. Vue comme une fonction de $\alpha$, la variance est une parabole tournée vers le haut, qui possède un minimum unique. En posant $\frac{d}{d\alpha}\text{Var}(Z_{\text{cv}}) = 0$, on obtient :

$$
\alpha^* = \frac{\text{Cov}(Z,C)}{\text{Var}(C)}
$$

C'est le coefficient de la régression par moindres carrés ordinaires : on prédit l'estimateur $Z$ en utilisant la variable de contrôle $C$ comme prédicteur. Puisque $\mathbb{E}[C] = 0$, le modèle linéaire est $Z \approx \mathbb{E}[Z] + \alpha^* C$, où $\alpha^*$ est la pente obtenue par moindres carrés ordinaires. L'estimateur avec variable de contrôle $Z_{\text{cv}} = Z - \alpha^* C$ calcule le résidu : la partie de $Z$ que $C$ ne peut pas expliquer.

En substituant $\alpha^*$ dans la formule de la variance, on obtient :

$$
\text{Var}(Z_{\text{cv}}) = \text{Var}(Z) - \frac{[\text{Cov}(Z,C)]^2}{\text{Var}(C)} = (1 - R^2) \text{Var}(Z)
$$

où $R^2 = \frac{[\text{Cov}(Z,C)]^2}{\text{Var}(Z)\text{Var}(C)}$ est le coefficient de détermination de la régression de $Z$ sur $C$. La réduction de variance vaut $R^2 \text{Var}(Z)$ : mieux $C$ prédit $Z$, plus on élimine de variance.

#### Application à REINFORCE

Dans le cadre de l'apprentissage par renforcement, notre estimateur de gradient REINFORCE est une somme sur les pas de temps : $\sum_{t=0}^T Z_t$, où chaque $Z_t$ représente la contribution au gradient au pas de temps $t$. On applique des variables de contrôle séparément à chaque terme. Puisque $\text{Var}(\sum_t Z_t) = \sum_t \text{Var}(Z_t) + \sum_{t \neq s} \text{Cov}(Z_t, Z_s)$, réduire la variance de chaque $Z_t$ réduit la variance totale, bien qu'on ne traite pas explicitement les termes de covariance entre pas de temps.

Pour une trajectoire donnée à l'état $s_t$, la contribution au gradient au temps $t$ est :

$$
Z_t = \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\sum_{k=t}^T r_k
$$

C'est le produit de la fonction de score $\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)$ et du rendement restant $\sum_{k=t}^T r_k$. On peut soustraire du rendement n'importe quelle fonction $b(s_t)$ de l'état sans introduire de biais, pourvu que $b(s_t)$ ne dépende pas de $a_t$. En effet :

$$
\mathbb{E}_{a_t \sim \pi_{\boldsymbol{w}}(\cdot|s_t)}\left[\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)b(s_t)\right] = b(s_t)\mathbb{E}_{a_t}\left[\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\right] = 0
$$

où la dernière égalité découle de l'identité de la fonction de score {eq}`eq:score-function-zero-mean`.

On peut maintenant définir notre variable de contrôle :

$$
C_t = \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t) \cdot b(s_t)
$$

où $b(s_t)$ est une **fonction de référence** qui ne dépend que de l'état. Elle satisfait $\mathbb{E}[C_t|s_t] = 0$. Notre estimateur avec variable de contrôle devient :

$$
Z_{t,\text{cv}} = Z_t - C_t = \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\left(\sum_{k=t}^T r_k - b(s_t)\right)
$$

La référence optimale $b^*(s_t)$ minimise la variance. Pour la trouver, considérons par simplicité le cas d'un paramètre scalaire. Posons $g(a_t) \equiv \nabla_w \log \pi_w(a_t|s_t)$ et $G_t = \sum_{k=t}^T r_k$. On veut résoudre :

$$
b^*(s_t) = \arg\min_{b} \text{Var}_{a_t \sim \pi_{\boldsymbol{w}}(\cdot|s_t)}\left[g(a_t)(G_t - b)\right]
$$

Comme la moyenne ne dépend pas de $b$, minimiser la variance équivaut à minimiser le moment d'ordre deux $\mathbb{E}[g(a_t)^2(G_t - b)^2|s_t]$. En développant et en dérivant par rapport à $b$, on obtient :

$$
b^*(s_t) = \frac{\mathbb{E}_{a_t|s_t}\left[g(a_t)^2 G_t\right]}{\mathbb{E}_{a_t|s_t}\left[g(a_t)^2\right]}
$$

Pour des paramètres vectoriels $\boldsymbol{w}$, on minimise un substitut scalaire, comme la trace de la matrice de covariance, ce qui donne la même formule avec $\|\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\|^2$ à la place de $g(a_t)^2$ :

$$
b^*(s_t) = \frac{\mathbb{E}_{a_t|s_t}\left[\|\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\|^2 G_t\right]}{\mathbb{E}_{a_t|s_t}\left[\|\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\|^2\right]}
$$

C'est la référence optimale exacte : une moyenne pondérée des rendements dont les poids sont les normes au carré de la fonction de score. En pratique, on traite cette norme au carré comme à peu près constante d'une action à l'autre en un état donné, ce qui mène au choix plus simple et largement répandu :

$$
b(s_t) \approx \mathbb{E}[G_t|s_t] = v^{\pi_{\boldsymbol{w}}}(s_t)
$$

Avec cette approximation, la contribution au gradient à variance réduite au pas de temps $t$ devient :

$$
Z_{\text{cv},t} = \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\left(\sum_{k=t}^T r_k - v^{\pi_{\boldsymbol{w}}}(s_t)\right)
$$

Le terme entre parenthèses est exactement la **fonction d'avantage** : $A^{\pi_{\boldsymbol{w}}}(s_t, a_t) = q^{\pi_{\boldsymbol{w}}}(s_t, a_t) - v^{\pi_{\boldsymbol{w}}}(s_t)$, où la fonction Q est approchée par le rendement de Monte-Carlo $\sum_{k=t}^T r_k$. L'estimation complète du gradient pour une trajectoire est alors la somme sur tous les pas de temps :

$$
\hat{g} = \sum_{t=0}^T Z_{\text{cv},t} = \sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\left(G_t - v^{\pi_{\boldsymbol{w}}}(s_t)\right)
$$


En pratique, on n'a pas accès à la vraie fonction de valeur et il faut l'apprendre. Contrairement aux méthodes du [chapitre sur l'amortissement](amortized-action-optimization.md), où l'on apprenait des fonctions de valeur pour approcher la fonction Q *optimale*, notre but est ici l'**évaluation de la politique** : estimer la valeur de la politique *courante* $\pi_{\boldsymbol{w}}$. Les mêmes techniques d'approximation de fonctions s'appliquent, mais on vise $v^{\pi_{\boldsymbol{w}}}$ plutôt que $v^*$. L'approche la plus simple consiste à régresser les états sur les rendements de Monte-Carlo, ce qui revient à apprendre ce que {cite:t}`williams1992reinforce` appelait une « valeur de référence » :

```{prf:algorithm} Gradient de politique avec valeur de référence simple
:label: policy-grad-baseline

**Entrée :** paramétrisation de la politique $\pi_{\boldsymbol{w}}(a|s)$, fonction de référence $b(s;\boldsymbol{\theta})$  
**Sortie :** paramètres de la politique mis à jour $\boldsymbol{w}$  
**Hyperparamètres :** taux d'apprentissage $\alpha_w$, $\alpha_\theta$, nombre d'épisodes $N$, longueur des épisodes $T$

1. Initialiser les paramètres $\boldsymbol{w}$, $\boldsymbol{\theta}$
2. Pour épisode = 1, ..., $N$, faire :
   1. Recueillir une trajectoire $\tau = (s_0, a_0, r_0, ..., s_T, a_T, r_T)$ avec la politique $\pi_{\boldsymbol{w}}(a|s)$
   2. Calculer les rendements : $G_t = \sum_{k=t}^T r_k$ pour $t = 0, ..., T$
   3. Mettre à jour la référence : $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha_\theta \nabla_{\boldsymbol{\theta}}\sum_{t=0}^T (G_t - b(s_t;\boldsymbol{\theta}))^2$
   4. Calculer l'estimation du gradient : $\hat{g} = \sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)(G_t - b(s_t;\boldsymbol{\theta}))$
   5. Mettre à jour la politique : $\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha_w \hat{g}$
3. Retourner $\boldsymbol{w}$
```

Aujourd'hui, pour mettre en œuvre cet algorithme, on utilise toujours des mini-lots afin de tirer pleinement parti des processeurs graphiques. Une variante plus représentative de cet algorithme serait donc :


```{prf:algorithm} Gradient de politique avec variable de contrôle optimale et mini-lots
:label: policy-grad-cv-batch

**Entrée :** paramétrisation de la politique $\pi_{\boldsymbol{w}}(a|s)$, fonction de valeur $v(s;\boldsymbol{\theta})$  
**Sortie :** paramètres de la politique mis à jour $\boldsymbol{w}$  
**Hyperparamètres :** taux d'apprentissage $\alpha_w$, $\alpha_\theta$, nombre d'itérations $N$, longueur des épisodes $T$, taille des lots $B$, taille des mini-lots $M$

1. Initialiser les paramètres $\boldsymbol{w}$, $\boldsymbol{\theta}$
2. Pour itération = 1, ..., N :
    1. Initialiser un tampon vide $\mathcal{D}$
    2. Pour b = 1, ..., B :
        1. Recueillir une trajectoire $\tau_b = (s_0, a_0, r_0, ..., s_T, a_T, r_T)$ avec la politique $\pi_{\boldsymbol{w}}(a|s)$
        2. Calculer les rendements : $G_t = \sum_{k=t}^T r_k$ pour tout t
        3. Stocker les triplets $(s_t, a_t, G_t)_{t=0}^T$ dans $\mathcal{D}$
    3. Pour époque_valeur = 1, ..., K :
        1. Tirer de $\mathcal{D}$ un mini-lot $\mathcal{B}_v$ de taille $M$
        2. Calculer la perte de valeur : $L_v = \frac{1}{M}\sum_{(s,a,G) \in \mathcal{B}_v} (v(s;\boldsymbol{\theta}) - G)^2$
        3. Mettre à jour la fonction de valeur : $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha_\theta \nabla_{\boldsymbol{\theta}}L_v$
    4. Calculer les avantages : $A(s,a) = G - v(s;\boldsymbol{\theta})$ pour tout $(s,a,G) \in \mathcal{D}$
    5. Normaliser les avantages : $A \leftarrow \frac{A - \mu_A}{\sigma_A}$
    6. Pour époque_politique = 1, ..., J :
        1. Tirer de $\mathcal{D}$ un mini-lot $\mathcal{B}_\pi$ de taille $M$
        2. Calculer la perte de politique : $L_\pi = -\frac{1}{M}\sum_{(s,a,A) \in \mathcal{B}_\pi} \log \pi_{\boldsymbol{w}}(a|s)A$
        3. Mettre à jour la politique : $\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha_w \nabla_{\boldsymbol{w}}L_\pi$
3. Retourner $\boldsymbol{w}$
```

La fonction de valeur est entraînée en régressant directement les états sur leurs rendements de Monte-Carlo échantillonnés $G$. La normalisation des avantages (étape 2.5) ne fait pas partie de la dérivation de la référence optimale, mais elle améliore l'optimisation en pratique et est courante dans les implémentations modernes.

### Estimation généralisée de l'avantage

La construction de la référence nous a donné un estimateur du gradient de la forme :

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)}) \left(G_t^{(i)} - v(s_t^{(i)})\right)
$$

où $G_t = \sum_{k=t}^T r_k$ est le rendement de Monte-Carlo à partir du temps $t$. Pour chaque paire état-action visitée $(s_t, a_t)$, le terme entre parenthèses

$$
\widehat{A}_t^{\text{MC}} = G_t - v(s_t)
$$

est une estimation de Monte-Carlo de l'avantage $A^{\pi}(s_t, a_t) = q^{\pi}(s_t, a_t) - v^{\pi}(s_t)$. Si la référence est égale à la vraie fonction de valeur, $v = v^{\pi}$, alors $\mathbb{E}[\widehat{A}_t^{\text{MC}} | s_t, a_t] = A^{\pi}(s_t, a_t)$, et cet estimateur est sans biais.

Cet estimateur présente cependant deux limites. D'abord, il a une forte variance, car $G_t$ dépend de toutes les récompenses futures. Ensuite, il n'utilise la fonction de valeur que comme référence, et non comme prédicteur des rendements à long terme. On rejette essentiellement l'information contenue dans $v(s_{t+1}), v(s_{t+2}), \ldots$

L'estimation généralisée de l'avantage (GAE) remédie à ces problèmes en construisant une famille d'estimateurs qui interpolent entre le Monte-Carlo pur et l'autoamorçage pur. Un paramètre $\lambda$ règle le compromis entre biais et variance.

#### Décomposer l'avantage de Monte-Carlo

Fixons une fonction de valeur $v(s)$ (pas nécessairement égale à $v^{\pi}$) et définissons le **résidu** à un pas :

$$
\delta_t = r_t + \gamma v(s_{t+1}) - v(s_t)
$$

Partons de l'avantage de Monte-Carlo, puis ajoutons et retranchons $\gamma v(s_{t+1})$ :

$$
\begin{align*}
G_t - v(s_t) &= r_t + \gamma G_{t+1} - v(s_t) \\
&= r_t + \gamma v(s_{t+1}) - v(s_t) + \gamma(G_{t+1} - v(s_{t+1})) \\
&= \delta_t + \gamma(G_{t+1} - v(s_{t+1}))
\end{align*}
$$

En appliquant récursivement cette décomposition, on obtient :

$$
G_t - v(s_t) = \sum_{l=0}^{T-t} \gamma^l \delta_{t+l}
$$

L'avantage de Monte-Carlo est exactement la somme actualisée des résidus futurs. C'est une identité algébrique, et non une approximation.

La suite $\{\delta_{t+l}\}_{l \geq 0}$ fournit des corrections incrémentales à la fonction de valeur à mesure qu'on avance dans le temps. Le terme $\delta_t$ ne dépend que de $(s_t, a_t, s_{t+1})$; $\delta_{t+1}$ dépend de $(s_{t+1}, a_{t+1}, s_{t+2})$, et ainsi de suite. À mesure que $l$ augmente, les corrections deviennent plus bruitées (elles dépendent de plus d'issues aléatoires) et plus sensibles aux erreurs de la fonction de valeur aux états ultérieurs. Bien que la somme complète soit sans biais lorsque $v = v^{\pi}$, elle peut avoir une forte variance et être fortement perturbée par l'erreur d'approximation de $v$.

#### La GAE comme estimateur à rétrécissement

La décomposition ci-dessus suggère une famille d'estimateurs qui réduisent le poids des résidus plus éloignés dans le futur. Soit $\lambda \in [0,1]$; définissons :

$$
A_t^{\lambda} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}
$$ (eq:gae-definition)

C'est l'**estimateur généralisé de l'avantage** $A_t^{\text{GAE}(\gamma,\lambda)}$.

Deux cas particuliers illustrent les extrêmes. Lorsque $\lambda = 1$, on retrouve l'avantage de Monte-Carlo :

$$
A_t^{\lambda=1} = \sum_{l=0}^{T-t} \gamma^l \delta_{t+l} = G_t - v(s_t)
$$

Lorsque $\lambda = 0$, on ne garde que le résidu immédiat :

$$
A_t^{\lambda=0} = \delta_t = r_t + \gamma v(s_{t+1}) - v(s_t)
$$

Les valeurs intermédiaires $0 < \lambda < 1$ interpolent entre ces extrêmes. L'influence de $\delta_{t+l}$ décroît géométriquement comme $(\gamma\lambda)^l$. Le paramètre $\lambda$ agit comme un paramètre de rétrécissement : un petit $\lambda$ rapproche l'estimateur du résidu à un pas; un grand $\lambda$ lui permet de se comporter davantage comme l'avantage de Monte-Carlo.

Si $v = v^{\pi}$ est la vraie fonction de valeur, alors $\mathbb{E}[\delta_t | s_t, a_t] = A^{\pi}(s_t, a_t)$ et $\mathbb{E}[\delta_{t+l} | s_t, a_t] = 0$ pour $l \geq 1$. Dans ce cas :

$$
\mathbb{E}[A_t^{\lambda} | s_t, a_t] = \sum_{l=0}^{T-t} (\gamma\lambda)^l \mathbb{E}[\delta_{t+l} | s_t, a_t] = A^{\pi}(s_t, a_t)
$$

pour tout $\lambda \in [0,1]$. Lorsque la fonction de valeur est exacte, la GAE est sans biais quel que soit $\lambda$; changer $\lambda$ n'affecte que la variance.

En pratique, on approche $v^{\pi}$ par un approximateur de fonctions, et les résidus $\delta_{t+l}$ héritent de l'erreur d'approximation. Les résidus éloignés font intervenir plusieurs applications de la fonction de valeur approchée et sont davantage contaminés par l'erreur de modélisation. Réduire leur poids (choisir $\lambda < 1$) introduit un biais, mais peut réduire la variance et limiter l'effet de ces erreurs.

#### Mélange d'estimateurs à plusieurs pas

Les rendements à plusieurs pas offrent un autre point de vue sur la GAE. Définissons le rendement à $k$ pas à partir du temps $t$ :

$$
G_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k v(s_{t+k})
$$

et l'estimateur correspondant de l'avantage à $k$ pas, $A_t^{(k)} = G_t^{(k)} - v(s_t)$. Chaque $A_t^{(k)}$ utilise $k$ récompenses avant l'autoamorçage; un $k$ plus grand signifie plus de variance, mais moins d'erreur d'autoamorçage.

L'estimateur GAE peut s'écrire comme un mélange géométrique :

$$
A_t^{\lambda} = (1-\lambda) \sum_{k=1}^{T-t} \lambda^{k-1} A_t^{(k)}
$$

La GAE est une moyenne pondérée des estimateurs de l'avantage à $k$ pas, où les horizons plus courts reçoivent plus de poids lorsque $\lambda$ est petit.

#### Utiliser la GAE dans le gradient de politique

Une fois $\lambda$ choisi, on substitue $A_t^{\lambda}$ à $G_t - v(s_t)$ dans l'estimateur du gradient de politique :

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)}) A_t^{\lambda,(i)}
$$

On utilise toujours une variable de contrôle pour réduire la variance (la référence $v$), mais on construit maintenant la cible d'avantage en lissant la suite des résidus $\{\delta_t\}$ au moyen d'un noyau à décroissance géométrique.

Pour la fonction de valeur, il est commode de définir le **$\lambda$-rendement** :

$$
G_t^{\lambda} = A_t^{\lambda} + v(s_t)
$$

Lorsque $\lambda = 1$, $G_t^{\lambda}$ se réduit au rendement de Monte-Carlo; lorsque $\lambda = 0$, il devient la cible autoamorcée à un pas $r_t + \gamma v(s_{t+1})$.

```{prf:algorithm} Gradient de politique avec GAE et mini-lots
:label: policy-grad-gae-batch

**Entrée :** paramétrisation de la politique $\pi_{\boldsymbol{w}}(a|s)$, fonction de valeur $v(s;\boldsymbol{\theta})$  
**Sortie :** paramètres de la politique mis à jour $\boldsymbol{w}$  
**Hyperparamètres :** taux d'apprentissage $\alpha_w$, $\alpha_\theta$, nombre d'itérations $N$, longueur des épisodes $T$, taille des lots $B$, taille des mini-lots $M$, facteur d'actualisation $\gamma$, paramètre de GAE $\lambda$

1. Initialiser les paramètres $\boldsymbol{w}$, $\boldsymbol{\theta}$
2. Pour itération = 1, ..., $N$ :
    1. Initialiser un tampon vide $\mathcal{D}$
    2. Pour $b = 1, ..., B$ :
        1. Recueillir une trajectoire $\tau_b = (s_0, a_0, r_0, ..., s_T, a_T, r_T)$ avec la politique $\pi_{\boldsymbol{w}}(a|s)$
        2. Calculer les prédictions de valeur $v_t = v(s_t;\boldsymbol{\theta})$ pour $t = 0, \ldots, T$ et poser $v_{T+1} = 0$
        3. Calculer les résidus : $\delta_t = r_t + \gamma v_{t+1} - v_t$ pour $t = 0, \ldots, T$
        4. Calculer les avantages GAE à rebours :
            1. Poser $A_T = \delta_T$
            2. Pour $t = T-1, ..., 0$ : $A_t = \delta_t + (\gamma\lambda) A_{t+1}$
        5. Calculer les $\lambda$-rendements : $G_t^{\lambda} = A_t + v_t$ pour $t = 0, \ldots, T$
        6. Stocker les quadruplets $(s_t, a_t, A_t, G_t^{\lambda})_{t=0}^T$ dans $\mathcal{D}$
    3. Pour époque_valeur = 1, ..., $K$ :
        1. Tirer de $\mathcal{D}$ un mini-lot $\mathcal{B}_v$ de taille $M$
        2. Calculer la perte de valeur : $L_v = \frac{1}{M}\sum_{(s,\cdot,\cdot,G^{\lambda}) \in \mathcal{B}_v} (v(s;\boldsymbol{\theta}) - G^{\lambda})^2$
        3. Mettre à jour la fonction de valeur : $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha_\theta \nabla_{\boldsymbol{\theta}}L_v$
    4. Normaliser les avantages : $A \leftarrow \frac{A - \mu_A}{\sigma_A}$
    5. Pour époque_politique = 1, ..., $J$ :
        1. Tirer de $\mathcal{D}$ un mini-lot $\mathcal{B}_\pi$ de taille $M$
        2. Calculer la perte de politique : $L_\pi = -\frac{1}{M}\sum_{(s,a,A,\cdot) \in \mathcal{B}_\pi} \log \pi_{\boldsymbol{w}}(a|s) A$
        3. Mettre à jour la politique : $\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha_w \nabla_{\boldsymbol{w}}L_\pi$
3. Retourner $\boldsymbol{w}$
```

Lorsque $\lambda = 1$, cet algorithme se réduit (à la normalisation des avantages près) à l'algorithme avec référence de Monte-Carlo présenté plus haut dans le chapitre. Lorsque $\lambda = 0$, les avantages deviennent les résidus à un pas $\delta_t$, et les $\lambda$-rendements se réduisent aux cibles autoamorcées standard à un pas.

#### L'acteur-critique comme limite $\lambda = 0$

Le cas $\lambda = 0$ est particulièrement simple. L'avantage devient :

$$
A_t^{\lambda=0} = \delta_t = r_t + \gamma v(s_{t+1}) - v(s_t)
$$

et la mise à jour de la politique se réduit à :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha_w \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t) \delta_t
$$

tandis que la mise à jour de la valeur devient une régression standard à un pas vers $r_t + \gamma v(s_{t+1})$. On obtient l'algorithme acteur-critique en ligne :

```{prf:algorithm} Acteur-critique avec résidus à un pas
:label: actor-critic-td0

**Entrée :** paramétrisation de la politique $\pi_{\boldsymbol{w}}(a|s)$, fonction de valeur $v(s;\boldsymbol{\theta})$  
**Sortie :** paramètres de la politique mis à jour $\boldsymbol{w}$  
**Hyperparamètres :** taux d'apprentissage $\alpha_w$, $\alpha_\theta$, nombre d'épisodes $N$, longueur des épisodes $T$, facteur d'actualisation $\gamma$

1. Initialiser les paramètres $\boldsymbol{w}$, $\boldsymbol{\theta}$
2. Pour épisode = 1, ..., $N$, faire :
   1. Initialiser l'état $s_0$
   2. Pour $t = 0, ..., T$, faire :
      1. Tirer une action : $a_t \sim \pi_{\boldsymbol{w}}(\cdot|s_t)$
      2. Exécuter $a_t$, observer $r_t$, $s_{t+1}$
      3. Calculer le résidu : $\delta_t = r_t + \gamma v(s_{t+1};\boldsymbol{\theta}) - v(s_t;\boldsymbol{\theta})$
      4. Mettre à jour la fonction de valeur : $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha_\theta \delta_t \nabla_{\boldsymbol{\theta}}v(s_t;\boldsymbol{\theta})$
      5. Mettre à jour la politique : $\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha_w \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t) \delta_t$
3. Retourner $\boldsymbol{w}$
```

Sutton a dérivé cet algorithme dans sa thèse de 1984, comme « heuristique adaptative » pour l'attribution temporelle du mérite. Dans le langage de ce chapitre, c'est le membre $\lambda = 0$ de la famille GAE : il utilise le résidu le plus local, $\delta_t$, à la fois comme cible de la fonction de valeur et comme estimation de l'avantage pour le gradient de politique.

## Méthodes du rapport de vraisemblance en apprentissage par renforcement

Comment les rapports d'importance et les objectifs de substitution contraints ou
écrêtés réutilisent-ils des trajectoires recueillies sous une politique antérieure?

L'estimateur par fonction de score de la section précédente est un cas particulier de la méthode du rapport de vraisemblance, dans lequel la loi de proposition est égale à la distribution cible. Considérons maintenant le cas général où elles diffèrent.

Rappelons l'estimateur de gradient par rapport de vraisemblance du début de ce chapitre. Pour un objectif $J(\theta) = \mathbb{E}_{x \sim p(x;\theta)}[f(x)]$ et une loi de proposition quelconque $q(x)$ :

$$
\nabla_\theta J(\theta) = \mathbb{E}_{x \sim q(x)}\left[f(x) \rho(x; q, \theta) \nabla_\theta \log p(x;\theta)\right]
$$

où $\rho(x; q, \theta) = \frac{p(x;\theta)}{q(x)}$ est le rapport de vraisemblance. La dérivée partielle $\frac{\partial \rho}{\partial \theta} = \rho \nabla_\theta \log p$ vaut parce que $x$ est traité comme fixe, puisqu'il a été tiré de $q$, qui ne dépend pas de $\theta$.

En apprentissage par renforcement, posons $x = \tau$ une trajectoire, $f(\tau) = G(\tau)$ le rendement, $p(\tau;\boldsymbol{w})$ la distribution des trajectoires sous la politique $\pi_{\boldsymbol{w}}$, et $q(\tau)$ la distribution des trajectoires sous une autre politique $\pi_q$. Le gradient devient :

$$
\nabla_{\boldsymbol{w}} J(\boldsymbol{w}) = \mathbb{E}_{\tau \sim \pi_q}\left[G(\tau) \rho(\tau) \sum_{t=0}^T \nabla_{\boldsymbol{w}} \log \pi_{\boldsymbol{w}}(a_t|s_t)\right]
$$

où le rapport de vraisemblance des trajectoires se simplifie, car les probabilités de transition s'annulent :

$$
\rho(\tau) = \frac{p(\tau;\boldsymbol{w})}{q(\tau)} = \prod_{t=0}^T \frac{\pi_{\boldsymbol{w}}(a_t|s_t)}{\pi_q(a_t|s_t)} = \prod_{t=0}^T \rho_t
$$

Ce produit de $T+1$ rapports peut devenir extrêmement grand ou petit lorsque $T$ croît, ce qui entraîne une forte variance. La structure temporelle offre un certain répit : puisque $\mathbb{E}_{a \sim \pi_q}[\rho] = 1$, les rapports futurs $\rho_{k}$ pour $k > t$, qui n'affectent pas la récompense $r_t$, peuvent être marginalisés. Les rapports passés $\rho_{0:t-1}$ restent toutefois nécessaires pour pondérer correctement la probabilité d'atteindre l'état $s_t$.

En pratique, des algorithmes comme PPO et TRPO font une approximation supplémentaire : ils n'utilisent que le rapport par pas $\rho_t$ plutôt que le produit cumulé $\rho_{0:t}$. On ignore ainsi l'écart entre les distributions d'états induites par les deux politiques. Combiné à une référence $b(s_t)$, l'estimateur approché est :

$$
\nabla_{\boldsymbol{w}} J(\boldsymbol{w}) \approx \mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T \rho_t \nabla_{\boldsymbol{w}} \log \pi_{\boldsymbol{w}}(a_t|s_t) (G_t - b(s_t))\right]
$$ (eq:lr-estimator)

Cette approximation revient à maximiser l'**objectif de substitution pondéré par importance** :

$$
L^{\text{IS}}(\boldsymbol{w}) = \mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T \rho_t A_t\right]
$$ (eq:importance-weighted-surrogate)

où $A_t = G_t - b(s_t)$. En prenant le gradient par rapport à $\boldsymbol{w}$, seul $\rho_t$ dépend de $\boldsymbol{w}$ (puisque les trajectoires sont tirées de $\pi_q$) :

$$
\nabla_{\boldsymbol{w}} L^{\text{IS}}(\boldsymbol{w}) = \mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T A_t \nabla_{\boldsymbol{w}} \rho_t\right]
$$

Le gradient du rapport est :

$$
\nabla_{\boldsymbol{w}} \rho_t = \nabla_{\boldsymbol{w}} \frac{\pi_{\boldsymbol{w}}(a_t|s_t)}{\pi_q(a_t|s_t)} = \frac{\nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(a_t|s_t)}{\pi_q(a_t|s_t)} = \rho_t \nabla_{\boldsymbol{w}} \log \pi_{\boldsymbol{w}}(a_t|s_t)
$$

En substituant :

$$
\nabla_{\boldsymbol{w}} L^{\text{IS}}(\boldsymbol{w}) = \mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T \rho_t \nabla_{\boldsymbol{w}} \log \pi_{\boldsymbol{w}}(a_t|s_t) A_t\right]
$$

On retrouve l'équation {eq}`eq:lr-estimator`. Lorsque $\pi_q = \pi_{\boldsymbol{w}}$, les rapports valent $\rho_t = 1$ et on retrouve l'estimateur par fonction de score. L'erreur d'approximation croît à mesure que les politiques s'écartent l'une de l'autre, ce qui motive les mécanismes de région de confiance et d'écrêtage présentés plus bas.

### Variance et condition de domination

Le rapport $\rho_t = \pi_{\boldsymbol{w}}(a_t|s_t)/\pi_q(a_t|s_t)$ ne se comporte bien que lorsque les deux politiques sont semblables. Si $\pi_{\boldsymbol{w}}$ attribue une forte probabilité à une action à laquelle $\pi_q$ attribue une faible probabilité, le rapport explose. Par exemple, si $\pi_q(a|s) = 0.01$ et $\pi_{\boldsymbol{w}}(a|s) = 0.5$, alors $\rho = 50$, ce qui amplifie tout bruit dans l'estimation de l'avantage.

L'échantillonnage préférentiel exige aussi la **condition de domination** : le support de $\pi_{\boldsymbol{w}}$ doit être contenu dans celui de $\pi_q$. Si $\pi_{\boldsymbol{w}}(a|s) > 0$ mais $\pi_q(a|s) = 0$, le rapport n'est pas défini. Les politiques stochastiques ont habituellement un support complet, mais le rapport peut tout de même devenir arbitrairement grand lorsque $\pi_q(a|s) \to 0$.

Un usage courant consiste à poser $\pi_q = \pi_{\boldsymbol{w}_{\text{old}}}$, une version antérieure de la politique. On peut ainsi réutiliser les données sur plusieurs pas de gradient : recueillir les trajectoires une fois, puis mettre à jour $\boldsymbol{w}$ plusieurs fois. Mais chaque mise à jour éloigne davantage $\boldsymbol{w}$ de $\boldsymbol{w}_{\text{old}}$, ce qui rend les rapports plus extrêmes. Le signal de gradient finit par être dominé par quelques échantillons de grand poids.

### Optimisation proximale de politique

Ces problèmes de variance suggèrent une solution naturelle : garder le rapport $\rho_t$ proche de 1 en s'assurant que la nouvelle politique reste proche de la politique de comportement. L'objectif de substitution pondéré par importance $L^{\text{IS}}(\boldsymbol{w})$ de {eq}`eq:importance-weighted-surrogate` reste alors bien conditionné.

L'optimisation de politique par région de confiance (TRPO) formalise cette idée en ajoutant une contrainte sur la divergence KL entre l'ancienne et la nouvelle politique :

$$
\max_{\boldsymbol{w}} L^{\text{IS}}(\boldsymbol{w}) \quad \text{sous la contrainte} \quad \mathbb{E}_s\left[D_{\text{KL}}(\pi_{\boldsymbol{w}_{\text{old}}}(\cdot|s) \| \pi_{\boldsymbol{w}}(\cdot|s))\right] \leq \delta
$$

La contrainte KL garantit que les deux distributions restent semblables, ce qui borne l'ampleur que peuvent prendre les poids d'importance. C'est un problème d'optimisation sous contraintes, et l'on pourrait en principe appliquer des méthodes standard comme la descente de gradient projeté ou les approches par lagrangien augmenté (présentées dans le [chapitre sur l'optimisation de trajectoires](discrete-time-optimal-control.md)). TRPO procède autrement : il utilise une approximation de Taylor du second ordre de la contrainte KL autour des paramètres courants et résout le sous-problème de région de confiance qui en résulte par la méthode du gradient conjugué. Cela exige de calculer la matrice d'information de Fisher (la hessienne de la divergence KL), ce qui alourdit le calcul.

L'optimisation proximale de politique (PPO) obtient un comportement semblable par un mécanisme plus simple : plutôt que de contraindre les distributions à rester semblables, elle écrête directement le rapport $\rho_t$ pour l'empêcher de trop s'éloigner de 1. Il s'agit d'une garantie inscrite dans la construction plutôt que d'une contrainte imposée à l'optimisation.

#### Des espérances sur les trajectoires aux moyennes sur les paires état-action

Avant de définir l'objectif de PPO, il faut préciser la relation entre l'objectif de substitution au niveau des trajectoires {eq}`eq:importance-weighted-surrogate` et l'objectif au niveau des paires état-action que PPO optimise réellement. L'objectif de substitution pondéré par importance est défini comme une espérance sur les trajectoires :

$$
L^{\text{IS}}(\boldsymbol{w}) = \mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T \rho_t A_t\right]
$$

On peut le réécrire comme une espérance sur des paires état-action en introduisant une distribution d'échantillonnage. Pour un horizon fini $T$, définissons la **distribution marginale temporelle moyennée** :

$$
\xi_{\pi_q}(s, a) = \frac{1}{T+1} \sum_{t=0}^{T} d_t^{\pi_q}(s) \pi_q(a|s)
$$ (eq:time-marginal)

où $d_t^{\pi_q}(s)$ est la probabilité de se trouver dans l'état $s$ au temps $t$ en suivant la politique $\pi_q$ depuis la distribution initiale. C'est le mélange uniforme des distributions état-action indexées par le temps : on choisit un pas de temps $t$ uniformément au hasard dans $\{0, 1, \ldots, T\}$, puis on tire $(s, a)$ de la distribution conjointe à ce pas de temps.

Avec cette définition, l'espérance sur les trajectoires devient :

$$
\mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T \rho_t A_t\right] = (T+1) \cdot \mathbb{E}_{(s,a) \sim \xi_{\pi_q}}\left[\rho(s, a) A(s, a)\right]
$$

Le facteur $(T+1)$ n'est qu'une constante qui n'affecte pas l'optimisation. Cette reformulation montre que l'objectif de substitution pondéré par importance équivaut à une espérance sur des paires état-action tirées de la distribution marginale temporelle moyennée. Ce n'est ni une distribution stationnaire ni une distribution de visite actualisée, mais le mélange empirique induit par la procédure de déroulement à horizon fini.

#### L'objectif de substitution écrêté

PPO remplace le terme linéaire pondéré par importance $\rho A$ par une version écrêtée. Pour une paire état-action $(s, a)$ d'avantage $A$ et de rapport d'importance $\rho(\boldsymbol{w}) = \pi_{\boldsymbol{w}}(a|s) / \pi_{\boldsymbol{w}_{\text{old}}}(a|s)$, définissons l'**objectif écrêté par échantillon** :

$$
\ell^{\text{CLIP}}(\boldsymbol{w}; s, a, A) = \min\left(\rho(\boldsymbol{w}) A, \, \text{clip}(\rho(\boldsymbol{w}), 1-\epsilon, 1+\epsilon) A\right)
$$ (eq:ppo-per-sample)

où $\epsilon$ est un hyperparamètre (typiquement 0.1 ou 0.2) et où $\text{clip}(x, a, b) = \max(a, \min(x, b))$ restreint $x$ à l'intervalle $[a, b]$.

L'**objectif de PPO au niveau de la population** est alors :

$$
L^{\text{CLIP}}(\boldsymbol{w}) = \mathbb{E}_{(s,a,A) \sim \xi_{\pi_{\boldsymbol{w}_{\text{old}}}}}\left[\ell^{\text{CLIP}}(\boldsymbol{w}; s, a, A)\right]
$$ (eq:ppo-clip-population)

où l'espérance porte sur la distribution marginale temporelle moyennée {eq}`eq:time-marginal` induite par $\pi_{\boldsymbol{w}_{\text{old}}}$.

En pratique, on ne calcule jamais cette espérance exactement. On recueille plutôt un lot de transitions $\mathcal{D} = \{(s_t^{(i)}, a_t^{(i)}, A_t^{(i)})\}$ en exécutant $\pi_{\boldsymbol{w}_{\text{old}}}$, et on approche l'espérance par une moyenne empirique :

$$
\hat{L}^{\text{CLIP}}(\boldsymbol{w}; \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{(s,a,A) \in \mathcal{D}} \ell^{\text{CLIP}}(\boldsymbol{w}; s, a, A)
$$ (eq:ppo-clip)

C'est la même approximation par substitution que dans l'[itération Q ajustée](fitted-q-iteration.md) : on remplace la distribution de population inconnue par la distribution empirique $\hat{P}_{\mathcal{D}}$ induite par le lot recueilli, puis on calcule la moyenne empirique. L'objectif de substitution empirique $\hat{L}^{\text{CLIP}}$ est simplement une espérance sous $\hat{P}_{\mathcal{D}}$. Aucune hypothèse de stationnarité ou de visite actualisée n'est nécessaire : on fait seulement la moyenne sur les transitions recueillies.

#### Intuition du mécanisme d'écrêtage

L'opérateur $\min$ de {eq}`eq:ppo-per-sample` retient l'estimation la plus pessimiste. Considérons les deux cas :

- **Avantage positif** ($A > 0$) : l'action est meilleure que la moyenne, et l'on veut donc augmenter $\pi_{\boldsymbol{w}}(a|s)$. Le terme non écrêté $\rho A$ croît avec $\rho$. Le terme écrêté cesse de croître dès que $\rho > 1 + \epsilon$. Prendre le minimum signifie que l'on ne profite d'une augmentation de $\rho$ que jusqu'à $1 + \epsilon$.

- **Avantage négatif** ($A < 0$) : l'action est pire que la moyenne, et l'on veut donc diminuer $\pi_{\boldsymbol{w}}(a|s)$. Le terme non écrêté $\rho A$ devient moins négatif (il s'améliore) lorsque $\rho$ diminue. Le terme écrêté cesse de s'améliorer dès que $\rho < 1 - \epsilon$. Prendre le minimum signifie que l'on ne profite d'une diminution de $\rho$ que jusqu'à $1 - \epsilon$.

Dans les deux cas, l'écrêtage supprime l'incitation à pousser le rapport de probabilités au-delà de l'intervalle $[1-\epsilon, 1+\epsilon]$. La nouvelle politique reste ainsi proche de l'ancienne, sans qu'on calcule ni contraigne explicitement la divergence KL.

```{prf:algorithm} Optimisation proximale de politique (PPO-Clip)
:label: ppo

**Entrée :** politique $\pi_{\boldsymbol{w}}(a|s)$, fonction de valeur $v(s;\boldsymbol{\theta})$  
**Sortie :** paramètres mis à jour $\boldsymbol{w}$  
**Hyperparamètres :** paramètre d'écrêtage $\epsilon$, taux d'apprentissage $\alpha_w$, $\alpha_\theta$, nombre d'itérations $N$, taille des lots $B$, taille des mini-lots $M$, nombre d'époques par itération $K$, paramètres de GAE $\gamma$, $\lambda$

1. Initialiser les paramètres $\boldsymbol{w}$, $\boldsymbol{\theta}$
2. Pour itération = 1, ..., $N$ :
    1. Recueillir un lot $\mathcal{D}$ de $B$ trajectoires avec la politique $\pi_{\boldsymbol{w}}$
    2. Calculer les avantages GAE $A_t$ et les $\lambda$-rendements $G_t^{\lambda}$ pour tous les pas de temps
    3. Stocker les anciennes log-probabilités : $\log \pi_{\text{old}}(a_t|s_t) = \log \pi_{\boldsymbol{w}}(a_t|s_t)$ pour tous les $(s_t, a_t)$
    4. Normaliser les avantages : $A \leftarrow \frac{A - \mu_A}{\sigma_A}$
    5. Pour époque = 1, ..., $K$ :
        1. Mélanger $\mathcal{D}$ et le partitionner en mini-lots de taille $M$
        2. Pour chaque mini-lot $\mathcal{B}$ :
            1. Calculer les rapports : $\rho = \exp(\log \pi_{\boldsymbol{w}}(a|s) - \log \pi_{\text{old}}(a|s))$ pour tous les $(s, a) \in \mathcal{B}$
            2. Calculer les objectifs par échantillon : $\ell^{\text{CLIP}} = \min(\rho A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon) A)$
            3. Calculer l'objectif de substitution empirique : $\hat{L}^{\text{CLIP}} = \frac{1}{|\mathcal{B}|}\sum_{(s,a,A) \in \mathcal{B}} \ell^{\text{CLIP}}$
            4. Calculer la perte de valeur : $L_v = \frac{1}{|\mathcal{B}|}\sum_{(s,G^{\lambda}) \in \mathcal{B}} (v(s;\boldsymbol{\theta}) - G^{\lambda})^2$
            5. Mettre à jour la politique : $\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha_w \nabla_{\boldsymbol{w}} \hat{L}^{\text{CLIP}}$
            6. Mettre à jour la fonction de valeur : $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha_\theta \nabla_{\boldsymbol{\theta}} L_v$
3. Retourner $\boldsymbol{w}$
```

L'algorithme recueille un lot de trajectoires, puis effectue $K$ époques de mises à jour par mini-lots sur les mêmes données. L'objectif de substitution empirique $\hat{L}^{\text{CLIP}}$ approche l'objectif de population {eq}`eq:ppo-clip-population` au moyen d'échantillons tirés de la distribution marginale temporelle moyennée. Le rapport $\rho$ est calculé dans l'espace logarithmique pour des raisons de stabilité numérique. L'écrêtage supprime l'incitation locale à pousser un rapport de probabilités échantillonné au-delà de l'intervalle choisi. Il ne garantit pas que les mises à jour échantillonnées trouveront une politique qui réussit, et il ne dit rien des contraintes physiques absentes de la récompense ou de l'environnement.

### Expérience : PPO sur le procédé SwingRL

Le modèle SwingRL du [chapitre sur la modélisation](modeling-controlled-systems.md#internal-actuation-in-swingrl)
fournit une comparaison appariée entre une structure fournie et l'optimisation de
politique par échantillonnage. Les deux correcteurs agissent sur le même cycliste
articulé debout, utilisent les deux mêmes commandes bornées, reçoivent les mêmes
observations et comptent un tour complet non enroulé comme un succès. Le correcteur
structuré reçoit une règle de pompage verrouillée en phase dont les phases ont été
choisies par un balayage du modèle. PPO reçoit des trajectoires et la récompense de
l'environnement.

L'expérience demande si la mise à jour de gradient de politique écrêtée découvre un
tour complet selon un protocole unique, fixe et reproductible. Le protocole a été
choisi avant l'examen des cinq exécutions finales.

| grandeur | valeur |
|---|---:|
| réseaux de politique et de valeur | réseaux tanh distincts $2 \times 64$ |
| interactions demandées | 1 000 000 par germe |
| lot de déroulements | 8 environnements $\times$ 256 pas |
| époques et mini-lot de PPO | 4 époques, 256 échantillons |
| optimiseur | Adam, taux d'apprentissage $3\times10^{-4}$ à décroissance linéaire |
| actualisation et GAE | $\gamma=0.995$, $\lambda=0.95$ |
| coefficients d'écrêtage et d'entropie | $0.2$ et $0.001$ |
| germes | 0, 1, 2, 3, 4 |
| évaluation | 100 états initiaux fixes toutes les 50 000 interactions demandées |

Chaque lot complet de déroulements contient 2 048 transitions, de sorte que la
dernière mise à jour est enregistrée à 1 001 472 interactions. Un point de contrôle
est la première mise à jour complète atteignant ou dépassant sa cible nominale de
50 000 interactions. Les angles d'évaluation sont uniformes entre $-5^\circ$ et
$5^\circ$, les vitesses angulaires sont uniformes entre $-0.1$ et $0.1$ rad/s, et la
politique déterministe utilise la tangente hyperbolique de la moyenne gaussienne.
L'environnement et le correcteur structuré de référence utilisent les mêmes 100 états.

```{code-cell} python
:tags: [remove-input]
:label: fig-swing-ppo-replay
:caption: Rejeu enregistré de l'entraînement pour le germe 0. Les points de contrôle surviennent toutes les 50 000 interactions demandées; les lots complets de déroulements produisent les nombres enregistrés qu'affiche le curseur. Chaque point de contrôle utilise la politique déterministe sauvegardée, à partir du même angle initial de $3^\circ$. La courbe montre les rendements bruts des épisodes terminés et une moyenne mobile sur les huit points précédents. Les 21 politiques sauvegardées sont condensées en 63 secondes; la surimpression indique la durée d'entraînement réelle écoulée.

import sys
from pathlib import Path
from IPython.display import HTML, display

sys.path.insert(0, "code")
from swing_ppo import replay_player_html

display(HTML(replay_player_html(Path("_static/swing_ppo"))))
```

Le rejeu est une illustration tirée d'une exécution précisée d'avance; ce n'est pas
la comparaison probante. Les observations futures restent cachées jusqu'à ce que
l'animation atteigne leur point de contrôle. Le résultat sur cinq germes, présenté
ci-dessous, porte la comparaison entre les exécutions.

{download}`Télécharger le rejeu enregistré du germe 0 <_static/swing_ppo/training_replay.mp4>`

```{figure} _static/swing_ppo/swing_ppo_learning.png
:label: fig-swing-ppo-learning
:alt: Deux graphiques comparent cinq germes de PPO à un correcteur structuré de SwingRL. PPO reste à un taux de succès nul, tandis que son rendement moyen s'améliore légèrement. Le correcteur structuré réussit à partir de chaque état d'évaluation et obtient un rendement beaucoup plus grand.
:width: 100%

Cinq germes de PPO précisés d'avance, évalués sur les mêmes 100 états initiaux fixes. Les tracés bleus fins montrent les germes individuels; le tracé plein et la bande montrent la moyenne et un intervalle $t$ bilatéral à 95 %, le germe servant d'unité statistique. PPO améliore son rendement au début, mais ne réussit jamais un tour complet. Le correcteur structuré réussit à partir de chaque état et obtient un rendement moyen de 191.8; cette valeur est annotée hors échelle dans le panneau du rendement afin que le changement plus faible de PPO reste visible.
```

Les politiques finales se fixent sur un comportement de faible amplitude. Sur les cinq
déroulements de démonstration, le plus grand angle se situe entre $5.45^\circ$ et
$5.62^\circ$. Le rendement s'améliore parce que la politique modifie la posture du
cycliste tout en gardant modestes les pénalités d'effort et de temps, mais le
comportement n'approche jamais la condition de succès de $360^\circ$.

| correcteur | transitions d'entraînement | succès sur états réservés | rendement moyen | pire tension de suspension |
|---|---:|---:|---:|---:|
| correcteur de phase structuré | 0 | 100 % | 191.79 | $-637$ N |
| PPO, moyenne des cinq politiques finales | 1 001 472 par germe | 0 % | $-5.11$ | $+387$ N |

La comparaison n'établit pas de classement général entre la commande et
l'apprentissage par renforcement. Le correcteur structuré reçoit la phase de
l'oscillation et une famille de correcteurs adaptée au mécanisme, tandis que PPO doit
inférer une coordination utile à partir de rendements échantillonnés. À l'inverse, le
succès nominal du correcteur structuré exige une tension de suspension négative
pendant 7.37 % de ses pas actifs. Une tige rigide peut fournir cette force vers
l'extérieur; une chaîne de balançoire de terrain de jeu ne le peut pas. PPO évite la
violation en bougeant à peine, ce qui est réalisable mais ne résout pas la tâche.

Les deux échecs répondent à des questions d'audit différentes. L'optimisation n'a pas
trouvé de politique échantillonnée qui réussit selon ce protocole. La politique
structurée qui réussit révèle un modèle à liaison rigide inadéquat. Davantage
d'interactions ou un objectif d'apprentissage révisé pourraient remédier au premier
échec. Un modèle de chaîne unilatérale ou une contrainte explicite de tension est
nécessaire pour le second.

:::{admonition} Audit du modèle
:class: tip
Supposons que le bonus de succès soit augmenté jusqu'à ce que PPO apprenne à faire
tourner la liaison rigide. Quelle mesure rapportée empêcherait encore le résultat
d'établir un correcteur de balançoire de terrain de jeu qui réussit? Identifiez le
mode physique manquant et décrivez une évaluation qui le révélerait.
:::

L'[implémentation Python complète](code/swing_ppo.py) comprend l'échantillonnage de
la politique, les log-probabilités corrigées pour la tangente hyperbolique, la GAE,
les mises à jour de PPO, l'évaluation de SwingRL par lots, la sérialisation des points
de contrôle et le rendu du rejeu. Le
[registre de l'expérience](artifacts/swing_ppo/README.md) précise l'organisation des
artefacts et la commande de reproduction. L'entraînement de la politique est
délibérément absent de la compilation MyST; la page lit les points de contrôle, les
tableaux, les figures et l'animation enregistrés.

## Le théorème du gradient de politique

Comment l'estimateur par score au niveau des trajectoires peut-il se réécrire comme
une espérance sur la visite actualisée des états et sur les valeurs des actions?

Les algorithmes développés jusqu'ici (REINFORCE, acteur-critique, GAE et PPO) estiment tous les gradients de politique à partir de trajectoires échantillonnées. Établissons maintenant le fondement théorique de ces estimateurs en dérivant le théorème du gradient de politique dans le cadre actualisé à horizon infini.

{cite:t}`sutton1999policy` ont donné la dérivation originale. Nous présentons ici une autre approche fondée sur le théorème des fonctions implicites, qui formule l'optimisation de la politique comme un problème à deux niveaux :

$$
\max_{\mathbf{w}} \alpha^\top \mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}
$$

sous la contrainte :

$$
(\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}}) \mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}} = \mathbf{r}_{\pi_{\boldsymbol{w}}}
$$

Le théorème des fonctions implicites affirme que, si le problème $F(\mathbf{v}, \mathbf{w}) = 0$ admet une solution, on peut « reparamétrer » notre problème sous la forme $F(\mathbf{v}(\mathbf{w}), \mathbf{w})$, où $\mathbf{v}(\mathbf{w})$ est une fonction implicite de $\mathbf{w}$. Si la jacobienne $\frac{\partial F}{\partial \mathbf{v}}$ est inversible, alors :

$$
\frac{d\mathbf{v}(\mathbf{w})}{d\mathbf{w}} = -\left(\frac{\partial F(\mathbf{v}(\mathbf{w}), \mathbf{w})}{\partial \mathbf{v}}\right)^{-1}\frac{\partial F(\mathbf{v}(\mathbf{w}), \mathbf{w})}{\partial \mathbf{w}}
$$

Notre notation rend ici explicite le fait que la dérivée doit être évaluée à la racine $(\mathbf{v}(\mathbf{w}), \mathbf{w})$ de $F$. Pour la suite de cette dérivation, nous omettrons cette dépendance afin d'alléger la notation.

En appliquant ce résultat à notre cas, avec $F(\mathbf{v}, \mathbf{w}) = (\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})\mathbf{v} - \mathbf{r}_{\pi_{\boldsymbol{w}}}$ :

$$
\frac{\partial \mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}} = (\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^{-1}\left(\frac{\partial \mathbf{r}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}} + \gamma \frac{\partial \mathbf{P}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}}\mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}\right)
$$

Alors :

$$
\begin{align*}
\nabla_{\mathbf{w}}J(\mathbf{w}) &= \alpha^\top \frac{\partial \mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}} \\
&= \mathbf{x}_\alpha^\top\left(\frac{\partial \mathbf{r}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}} + \gamma \frac{\partial \mathbf{P}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}}\mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}\right)
\end{align*}
$$

où nous avons défini la distribution de visite actualisée des états :

$$
\mathbf{x}_\alpha^\top \equiv \alpha^\top(\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^{-1}.
$$

Rappelons la notation vectorielle des processus de décision markoviens du [chapitre sur les processus de décision markoviens à horizon infini](infinite-horizon-mdps.md) :

$$
\begin{align*}
\mathbf{r}_\pi(s) &\equiv \sum_{a \in \mathcal{A}_s} \pi(a \mid s) \, r(s, a), \\
[\mathbf{P}_\pi]_{s,s'} &\equiv \sum_{a \in \mathcal{A}_s} \pi(a \mid s) \, p(s' \mid s, a).
\end{align*}
$$

En dérivant par rapport à $\mathbf{w}$, on obtient :

$$
\begin{align*}
\left[\frac{\partial \mathbf{r}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}}\right]_s &= \sum_{a \in \mathcal{A}_s} \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s) \, r(s,a), \\
\left[\frac{\partial \mathbf{P}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}}\mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}\right]_s &= \sum_{a \in \mathcal{A}_s} \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s)\sum_{s'} p(s' \mid s,a) \, v_\gamma^{\pi_{\boldsymbol{w}}}(s').
\end{align*}
$$

En substituant :

$$
\begin{align*}
\nabla_{\mathbf{w}}J(\mathbf{w}) &= \sum_s x_\alpha(s)\left(\sum_a \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s) \, r(s,a) + \gamma\sum_a \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s)\sum_{s'} p(s' \mid s,a) \, v_\gamma^{\pi_{\boldsymbol{w}}}(s')\right) \\
&= \sum_s x_\alpha(s)\sum_a \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s)\left(r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) \, v_\gamma^{\pi_{\boldsymbol{w}}}(s')\right)
\end{align*}
$$

C'est le théorème du gradient de politique, où $x_\alpha(s)$ est la distribution de visite actualisée des états et où le terme entre parenthèses est la fonction de valeur d'action $q^{\pi_{\boldsymbol{w}}}(s,a)$.


### Distribution normalisée de visite actualisée des états

La visite actualisée des états $x_\alpha(s)$ n'est pas normalisée. L'expression obtenue ci-dessus n'est donc pas une espérance. On peut toutefois la transformer en espérance en normalisant par $1 - \gamma$. Remarquons que, pour toute distribution initiale $\alpha$ :

$$
\sum_s x_\alpha(s) = \alpha^\top(\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^{-1}\mathbf{1} = \frac{\alpha^\top\mathbf{1}}{1-\gamma} = \frac{1}{1-\gamma}
$$

En définissant la distribution normalisée des états $\xi_\alpha(s) = (1-\gamma)x_\alpha(s)$, on peut donc écrire :

$$
\begin{align*}
\nabla_{\mathbf{w}}J(\mathbf{w}) &= \frac{1}{1-\gamma}\sum_s \xi_\alpha(s)\sum_a \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s)\left(r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) \, v_\gamma^{\pi_{\boldsymbol{w}}}(s')\right) \\
&= \frac{1}{1-\gamma}\mathbb{E}_{s\sim\xi_\alpha}\left[\sum_a \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s) \, q^{\pi_{\boldsymbol{w}}}(s,a)\right]
\end{align*}
$$

Nous avons maintenant exprimé le théorème du gradient de politique au moyen d'espérances sous la distribution normalisée de visite actualisée des états. Mais que signifie échantillonner selon $\xi_\alpha$? Rappelons que $\mathbf{x}_\alpha^\top = \alpha^\top(\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^{-1}$. Le développement en série de Neumann (valide lorsque $\|\gamma \mathbf{P}_{\pi_{\boldsymbol{w}}}\| < 1$, ce qui est le cas pour $\gamma < 1$ puisque $\mathbf{P}_{\pi_{\boldsymbol{w}}}$ est une matrice stochastique) donne :

$$
\boldsymbol{\xi}_\alpha^\top = (1-\gamma)\alpha^\top\sum_{k=0}^{\infty} (\gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^k
$$

On peut ensuite sortir le premier terme de cette somme pour obtenir :

$$
\begin{align*}
\boldsymbol{\xi}_\alpha^\top &= (1-\gamma)\alpha^\top\sum_{k=0}^{\infty} (\gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^k \\
&= (1-\gamma)\alpha^\top + (1-\gamma)\alpha^\top\sum_{k=1}^{\infty} (\gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^k \\
&= (1-\gamma)\alpha^\top + (1-\gamma)\alpha^\top\gamma\mathbf{P}_{\pi_{\boldsymbol{w}}}\sum_{k=0}^{\infty} (\gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^k \\
&= (1-\gamma)\alpha^\top + \gamma\boldsymbol{\xi}_\alpha^\top \mathbf{P}_{\pi_{\boldsymbol{w}}}
\end{align*}
$$


L'équation d'équilibre :

$$
\boldsymbol{\xi}_\alpha^\top = (1-\gamma)\alpha^\top + \gamma\boldsymbol{\xi}_\alpha^\top \mathbf{P}_{\pi_{\boldsymbol{w}}}
$$


montre que $\boldsymbol{\xi}_\alpha$ est une distribution de mélange : avec probabilité $1-\gamma$, on tire un état de la distribution initiale $\alpha$ (réinitialisation), et avec probabilité $\gamma$, on suit la dynamique de la politique $\mathbf{P}_{\pi_{\boldsymbol{w}}}$ depuis l'état courant (continuation). Cette interprétation se rattache directement au processus géométrique : à chaque pas, on s'arrête et on tire à nouveau selon $\alpha$ (avec probabilité $1-\gamma$), ou on continue à suivre la politique (avec probabilité $\gamma$).

```{code-cell} python
import numpy as np

def sample_from_discounted_visitation(
    alpha, 
    policy, 
    transition_model, 
    gamma, 
    n_samples=1000
):
    """Sample states from the discounted visitation distribution.
    
    Args:
        alpha: Initial state distribution (vector of probabilities)
        policy: Function (state -> action probabilities)
        transition_model: Function (state, action -> next state probabilities)
        gamma: Discount factor
        n_samples: Number of states to sample
    
    Returns:
        Array of sampled states
    """
    samples = []
    n_states = len(alpha)
    rng = np.random.default_rng(2026)
    
    # Initialize state from alpha
    current_state = rng.choice(n_states, p=alpha)
    
    for _ in range(n_samples):
        samples.append(current_state)
        
        # With probability (1-gamma): reset
        if rng.random() > gamma:
            current_state = rng.choice(n_states, p=alpha)
        # With probability gamma: continue
        else:
            # Sample action from policy
            action_probs = policy(current_state)
            action = rng.choice(len(action_probs), p=action_probs)
            
            # Sample next state from transition model
            next_state_probs = transition_model(current_state, action)
            current_state = rng.choice(n_states, p=next_state_probs)
    
    return np.array(samples)

# Example usage for a simple 2-state MDP
alpha = np.array([0.7, 0.3])  # Initial distribution
policy = lambda s: np.array([0.8, 0.2])  # Dummy policy
transition_model = lambda s, a: np.array([0.9, 0.1])  # Dummy transitions
gamma = 0.9

samples = sample_from_discounted_visitation(alpha, policy, transition_model, gamma)

# Check empirical distribution
print("Empirical state distribution:")
print(np.bincount(samples) / len(samples))
```

Bien que les mathématiques montrent qu'échantillonner selon la distribution de visite actualisée $\boldsymbol{\xi}_\alpha$ donnerait des estimations sans biais du gradient de politique, Thomas (2014) a montré que cette mise en œuvre peut nuire aux performances en pratique. Le problème vient de ce que l'arrêt précoce des trajectoires (avec probabilité $1-\gamma$) réduit la quantité effective de données recueillies dans chaque trajectoire. Cet arrêt précoce affaiblit le signal d'apprentissage, car de nombreuses trajectoires n'atteignent pas d'états terminaux ou de récompenses significatifs.

En pratique, on tire donc habituellement des trajectoires complètes du processus non actualisé (en exécutant la politique jusqu'à sa terminaison naturelle ou jusqu'à un horizon fixe), tout en utilisant $\gamma$ dans l'estimation de l'avantage. Cette approche préserve l'intégralité du signal d'apprentissage de chaque trajectoire et donne empiriquement de meilleures performances.

C'est l'un des nombreux cas, en apprentissage par renforcement, où la procédure théoriquement optimale diffère de la meilleure mise en œuvre pratique.

### L'architecture acteur-critique

Le théorème du gradient de politique montre que le gradient dépend de la fonction de valeur d'action $q^{\pi_{\boldsymbol{w}}}(s,a)$. En pratique, on n'a pas accès à la vraie fonction $q$ et il faut l'estimer. On aboutit ainsi à l'architecture **acteur-critique** : l'**acteur** maintient la politique $\pi_{\boldsymbol{w}}$, tandis que le **critique** maintient une estimation de la fonction de valeur.

Cette architecture remonte à la thèse de Sutton de 1984, où il proposait le critique heuristique adaptatif (*Adaptive Heuristic Critic*). L'acteur utilise les estimations de valeur du critique pour calculer les estimations de l'avantage nécessaires au gradient de politique, tandis que le critique apprend à partir des mêmes trajectoires, engendrées par l'acteur. Les algorithmes développés plus haut (REINFORCE avec référence, GAE et l'acteur-critique à un pas) sont tous des instances de cette architecture.

On apprend simultanément deux fonctions qui dépendent l'une de l'autre, ce qui pose un problème de stabilité. Le gradient de l'acteur utilise les estimations du critique, mais le critique est entraîné sur des données engendrées par la politique de l'acteur. Si les deux changent trop vite, le processus d'apprentissage peut devenir instable.

{cite:t}`konda2002thesis` a analysé ce problème d'apprentissage couplé et établi des garanties de convergence sous une condition à **deux échelles de temps** : le critique doit se mettre à jour plus vite que l'acteur. Intuitivement, le critique doit « suivre » la fonction de valeur de la politique courante avant que l'acteur n'utilise ces estimations pour se mettre à jour. Si l'acteur avance trop vite, il utilise des estimations de valeur périmées ou inexactes, ce qui donne de mauvaises estimations du gradient.

En pratique, on met cela en œuvre au moyen de taux d'apprentissage différents : un taux d'apprentissage plus grand $\alpha_\theta$ pour le critique et un taux plus petit $\alpha_w$ pour l'acteur, avec $\alpha_\theta > \alpha_w$. On peut aussi effectuer plusieurs mises à jour du critique pour chaque mise à jour de l'acteur. L'algorithme acteur-critique adouci présenté plus tôt dans le [chapitre sur l'amortissement](amortized-action-optimization.md) suit le même principe : il hérite de la structure acteur-critique tout en intégrant la régularisation entropique et en apprenant directement des fonctions Q.

L'architecture acteur-critique se rattache aussi au point de vue de l'optimisation à deux niveaux du théorème du gradient de politique : le problème externe optimise la politique, tandis que le problème interne résout la fonction de valeur pour cette politique. La condition à deux échelles de temps garantit que le problème interne est approximativement résolu avant qu'on fasse un pas sur le problème externe.

## Méthodes de reparamétrisation en apprentissage par renforcement

Lorsque les actions et la dynamique admettent des chemins d'échantillonnage
dérivables, comment la dérivation le long des chemins modifie-t-elle la variance et
les exigences de modélisation de l'optimisation des politiques?

Lorsque la dynamique est connue ou peut être apprise, la reparamétrisation offre une solution de rechange aux méthodes par fonction de score. En exprimant les actions et les transitions d'état comme des fonctions déterministes d'un bruit, on peut rétropropager à travers les trajectoires pour calculer des gradients de politique de variance plus faible que celle des estimateurs par fonction de score.

### Gradients de valeur stochastiques

L'astuce de reparamétrisation exige de pouvoir exprimer la variable aléatoire comme une fonction déterministe d'un bruit. En apprentissage par renforcement, cela s'applique naturellement lorsqu'on dispose d'un modèle appris de la dynamique. Considérons une politique stochastique $\pi_{\boldsymbol{w}}(a|s)$ que l'on peut reparamétrer sous la forme $a = \pi_{\boldsymbol{w}}(s,\epsilon)$, où $\epsilon \sim p(\epsilon)$, et un modèle de dynamique $s' = f(s,a,\xi)$, où $\xi \sim p(\xi)$ représente le caractère stochastique de l'environnement. Les deux transformations sont déterministes étant donné les variables de bruit.

Grâce à ces reparamétrisations, on peut écrire un rendement à $n$ pas comme une fonction dérivable du bruit :

$$
R_n(s_0,\{\epsilon_i\},\{\xi_i\}) = \sum_{i=0}^{n-1} \gamma^i r(s_i,a_i)
$$

où $a_i = \pi_{\boldsymbol{w}}(s_i,\epsilon_i)$ et $s_{i+1} = f(s_i,a_i,\xi_i)$ pour $i=0,...,n-1$. L'objectif devient :

$$
J(\boldsymbol{w}) = \mathbb{E}_{\{\epsilon_i\},\{\xi_i\}}[R_n(s_0,\{\epsilon_i\},\{\xi_i\})]
$$

On peut maintenant appliquer l'estimateur de gradient par reparamétrisation :

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) = \mathbb{E}_{\{\epsilon_i\},\{\xi_i\}}\left[\nabla_{\boldsymbol{w}}R_n(s_0,\{\epsilon_i\},\{\xi_i\})\right]
$$

Ce gradient peut se calculer par différentiation automatique à travers la suite des évaluations de la politique et du modèle. Le calcul exige de rétropropager à travers $n$ pas de déroulement du modèle, ce qui devient coûteux pour un grand $n$, mais évite la forte variance des estimateurs par fonction de score.

Le cadre des gradients de valeur stochastiques (SVG) {cite:p}`Heess2015` emploie cette approche tout en introduisant un objectif hybride qui combine des déroulements du modèle avec l'autoamorçage d'une fonction de valeur :

$$
J^{\text{SVG}(n)}(\boldsymbol{w}) = \mathbb{E}_{\{\epsilon_i\},\{\xi_i\}}\left[\sum_{i=0}^{n-1} \gamma^i r(s_i,a_i) + \gamma^n q(s_n,a_n;\theta)\right]
$$

La fonction de valeur terminale $q(s_n,a_n;\theta)$ approche la valeur au-delà de l'horizon $n$, ce qui permet des déroulements plus courts tout en tenant compte de la valeur à long terme. On obtient ainsi un éventail d'algorithmes paramétrés par $n$.

#### SVG(0) : reparamétrisation sans modèle

Lorsque $n=0$, l'objectif se réduit à :

$$
J^{\text{SVG}(0)}(\boldsymbol{w}) = \mathbb{E}_{s \sim \rho}\mathbb{E}_{\epsilon \sim p(\epsilon)}\left[q(s,\pi_{\boldsymbol{w}}(s,\epsilon);\theta)\right]
$$

Aucun modèle n'est nécessaire. On dérive simplement le critique par rapport aux actions tirées de la politique reparamétrée. C'est l'approche utilisée dans DDPG {cite:p}`lillicrap2015continuous` (avec une politique déterministe, où $\epsilon$ est absent) et dans SAC {cite:p}`haarnoja2018soft` (où $\epsilon$ produit la composante stochastique). Le gradient est :

$$
\nabla_{\boldsymbol{w}} J^{\text{SVG}(0)} = \mathbb{E}_{s,\epsilon}\left[\nabla_a q(s,a;\theta)\big|_{a=\pi_{\boldsymbol{w}}(s,\epsilon)} \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s,\epsilon)\right]
$$

Cela exige seulement que le critique $q$ soit dérivable par rapport aux actions, et non un modèle appris de la dynamique. Tout le biais provient des erreurs d'approximation de la fonction de valeur.

#### De SVG(1) à SVG($n$) : déroulements fondés sur un modèle

Pour $n \geq 1$, on déroule un modèle appris de la dynamique sur $n$ pas avant de faire l'autoamorçage avec le critique. Considérons SVG(1) :

$$
J^{\text{SVG}(1)}(\boldsymbol{w}) = \mathbb{E}_{s,\epsilon,\xi}\left[r(s,\pi_{\boldsymbol{w}}(s,\epsilon)) + \gamma q(f(s,\pi_{\boldsymbol{w}}(s,\epsilon),\xi), \pi_{\boldsymbol{w}}(s',\epsilon');\theta)\right]
$$

où $s' = f(s,\pi_{\boldsymbol{w}}(s,\epsilon),\xi)$ est l'état suivant prédit par le modèle. Le gradient passe maintenant à la fois par la récompense et par la transition du modèle. Augmenter $n$ propage plus directement l'information sur les récompenses à travers le déroulement du modèle, ce qui réduit la dépendance envers le critique. Les erreurs du modèle s'accumulent toutefois sur l'horizon. Si le modèle est inexact, des déroulements plus longs peuvent dégrader les performances.

#### SVG($\infty$) : optimisation entièrement fondée sur un modèle

Lorsque $n \to \infty$, on élimine entièrement le critique :

$$
J^{\text{SVG}(\infty)}(\boldsymbol{w}) = \mathbb{E}_{\{\epsilon_i\},\{\xi_i\}}\left[\sum_{i=0}^{T-1} \gamma^i r(s_i,\pi_{\boldsymbol{w}}(s_i,\epsilon_i))\right]
$$

C'est l'optimisation de politique entièrement fondée sur un modèle, qui dérive à travers la trajectoire entière. Des approches comme PILCO {cite:p}`deisenroth2011pilco` et Dreamer {cite:p}`hafner2019dreamer` opèrent dans ce régime. Avec un modèle exact, on obtient le signal de gradient le plus direct. Le compromis porte sur le calcul : rétropropager à travers des centaines de pas de temps est coûteux, et l'amplitude des gradients peut exploser ou s'évanouir sur de longs horizons.

Le choix de $n$ traduit un compromis fondamental entre biais et variance. Un petit $n$ s'appuie sur le critique pour estimer la valeur à long terme et hérite de ses erreurs d'approximation. Un grand $n$ s'appuie sur le modèle et accumule ses erreurs de prédiction. En pratique, des valeurs intermédiaires comme $n=5$ ou $n=10$ fonctionnent souvent bien lorsqu'elles sont combinées à un modèle appris raisonnablement exact.

#### Inférence du bruit pour l'apprentissage hors politique

Une difficulté subtile survient lorsqu'on combine la reparamétrisation avec le rejeu d'expérience. SVG se prête naturellement à l'apprentissage hors politique : les états $s$ peuvent être tirés d'un tampon de rejeu plutôt que de la politique courante. La reparamétrisation exige toutefois les variables de bruit $\epsilon$ qui ont engendré chaque action.

Pour des données sur politique, on peut simplement stocker $\epsilon$ avec chaque transition $(s, a, r, s')$. Pour des données hors politique, recueillies sous une autre politique, le bruit est inconnu. Pour appliquer des gradients de reparamétrisation à de telles données, il faut *inférer* le bruit qui aurait produit l'action observée sous la politique courante.

Pour des politiques inversibles, c'est simple. Si $a = \pi_{\boldsymbol{w}}(s, \epsilon)$ avec $\epsilon \sim \mathcal{N}(0, I)$, et si la politique prend la forme $a = \mu_{\boldsymbol{w}}(s) + \sigma_{\boldsymbol{w}}(s) \odot \epsilon$ (comme une politique gaussienne), on peut retrouver le bruit exactement :

$$
\epsilon = \frac{a - \mu_{\boldsymbol{w}}(s)}{\sigma_{\boldsymbol{w}}(s)}
$$

Ce $\epsilon$ retrouvé peut alors servir au calcul du gradient. Cela introduit toutefois une dépendance subtile : le $\epsilon$ inféré dépend des paramètres courants de la politique $\boldsymbol{w}$, et pas seulement des données. À mesure que la politique change pendant l'entraînement, une même action $a$ correspond à des valeurs de bruit différentes.

Pour le bruit de la dynamique $\xi$, la situation est plus complexe. Si l'on dispose d'un modèle probabiliste $s' = f(s, a, \xi)$ et que l'on observe l'état suivant réel $s'$, on pourrait en principe inférer $\xi$. En pratique, le caractère stochastique de l'environnement est souvent traité comme irréductible : on ne peut pas rejouer exactement la même réalisation du bruit. SVG traite cette difficulté de l'une des façons suivantes : (1) en utilisant des modèles déterministes et en ignorant le caractère stochastique de l'environnement, (2) en simulant à nouveau à partir du modèle plutôt qu'en utilisant les états suivants observés, ou (3) en utilisant une pondération d'importance pour corriger l'écart entre les distributions.

Le point de vue de l'inférence du bruit relie les gradients de reparamétrisation à la question plus large de l'attribution du mérite en apprentissage par renforcement. En suivant explicitement quelles réalisations du bruit ont mené à quelles issues, on peut attribuer la valeur plus précisément aux paramètres de la politique plutôt qu'à des échantillons chanceux ou malchanceux.

Lorsque la dynamique est déterministe ou peut être reparamétrée avec exactitude, les méthodes de type SVG offrent une solution de rechange efficace aux méthodes par fonction de score développées à la section précédente. Cependant, de nombreux problèmes d'apprentissage par renforcement font intervenir une dynamique inconnue ou difficile à modéliser avec exactitude. Dans ces cadres, les méthodes par fonction de score restent l'outil principal, puisqu'elles exigent seulement de pouvoir échantillonner des trajectoires sous la politique.

## Synthèse

Le score de la trajectoire fournit un gradient de politique sans modèle, car les
termes de transition ne dépendent pas des paramètres de la politique. Les rendements
conditionnels, les références dépendant de l'état, les estimations généralisées de
l'avantage et les critiques appris réduisent sa variance. Les rapports d'importance
comparent ensuite les données d'une politique plus ancienne à la politique courante,
tandis que PPO écrête ces rapports pour limiter la mise à jour.

L'expérience SwingRL a distingué une mise en œuvre correcte de PPO d'un résultat de
commande réussi. Sur cinq millions d'interactions d'entraînement au total, PPO a
amélioré le rendement façonné, mais n'a produit aucun tour complet. Le correcteur
structuré a résolu le problème nominal à liaison rigide sans entraînement de
politique, puis a échoué à un autre test, car la trajectoire obtenue exigeait qu'une
chaîne pousse. Les diagnostics de l'algorithme, le succès de la tâche et la validité
du modèle restent donc des parties distinctes de l'évaluation.

Le théorème du gradient de politique exprime la même dérivée au moyen de la visite
actualisée des états et des valeurs des actions. L'approximation de ces valeurs
produit l'architecture acteur-critique, dans laquelle une mise à jour plus rapide du
critique fournit le signal utilisé par l'acteur.

Lorsque des modèles de la dynamique sont disponibles, la reparamétrisation par
gradients de valeur stochastiques offre une solution de rechange de variance plus
faible. SVG(0) retrouve des méthodes acteur-critique comme DDPG et SAC, tandis que
SVG($\infty$) dérive à travers une trajectoire simulée complète. Le choix qui en
résulte est maintenant explicite : les méthodes par score échangent de la variance
contre des hypothèses minimales sur le modèle; les méthodes le long des chemins
échangent des hypothèses de dérivabilité plus fortes contre une attribution du mérite
de variance plus faible.

## Autoévaluation

:::{exercise} Ce qu'exige la fonction de score
:label: ex-pg-check-1

L'estimateur du gradient de politique par rapport de vraisemblance exige-t-il de dériver la dynamique de l'environnement? Pourquoi?
:::

:::{solution} ex-pg-check-1
:class: dropdown

Non. Il dérive le logarithme de la probabilité des actions échantillonnées sous la politique et pondère ce score par les rendements ou les avantages échantillonnés; l'environnement n'a qu'à engendrer des trajectoires.
:::

:::{exercise} Biais de la référence
:label: ex-pg-check-2

Pourquoi une référence dépendant de l'état peut-elle réduire la variance sans biaiser le gradient de politique par fonction de score?
:::

:::{solution} ex-pg-check-2
:class: dropdown

Conditionnellement à un état, le score espéré de la politique est nul : $\sum_a\pi(a|s)\nabla\log\pi(a|s)=0$. Multiplier ce score par une référence qui ne dépend que de l'état donne donc une espérance nulle.
:::
