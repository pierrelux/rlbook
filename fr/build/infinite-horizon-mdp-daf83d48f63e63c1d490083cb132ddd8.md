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
# Processus de décision markoviens à horizon infini

La programmation dynamique stochastique à horizon fini tire sa condition aux
limites de la récompense terminale. Quelle équation de valeur reste-t-il
lorsque le processus n'a pas de date terminale? L'actualisation rend bornée la
suite infinie des récompenses et transforme les équations de Bellman en
problèmes de point fixe.

La récompense totale espérée non actualisée de la politique $\boldsymbol{\pi} \in
\Pi^{\mathrm{HR}}$ est

$$
v^{\boldsymbol{\pi}}(s) = \mathbb{E}\left[\sum_{t=1}^{\infty} r(S_t, A_t)\right]
$$

Un inconvénient de ce modèle est qu'on peut facilement obtenir des valeurs égales à $+\infty$ ou à $-\infty$, même dans un cadre aussi simple qu'un processus de décision markovien à un seul état qui reboucle sur lui-même et dont la récompense accumulée est non nulle.

Il est donc souvent plus commode de travailler avec une autre formulation, qui garantit l'existence d'une limite : la récompense totale actualisée espérée de la politique $\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}$ est définie par :

$$
v_\gamma^{\boldsymbol{\pi}}(s) \equiv \lim_{N \rightarrow \infty} \mathbb{E}\left[\sum_{t=1}^N \gamma^{t-1} r(S_t, A_t)\right]
$$

pour $0 \leq \gamma < 1$ et lorsque $\max_{s \in \mathcal{S}} \max_{a \in \mathcal{A}_s}|r(s, a)| = R_{\max} < \infty$, auquel cas $|v_\gamma^{\boldsymbol{\pi}}(s)| \leq (1-\gamma)^{-1} R_{\max}$.


Enfin, une autre possibilité pour le cadre à horizon infini est ce qu'on appelle la récompense moyenne, ou gain, de la politique $\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}$, définie par :

$$
g^{\boldsymbol{\pi}}(s) \equiv \lim_{N \rightarrow \infty} \frac{1}{N} \mathbb{E}\left[\sum_{t=1}^N r(S_t, A_t)\right]
$$

Nous n'utiliserons pas cette formulation dans ce cours, en raison des difficultés pratiques et théoriques qui lui sont inhérentes. 

En prolongeant la notion d'optimalité déjà vue pour les modèles à horizon fini, on dit qu'une politique $\boldsymbol{\pi}^*$ est optimale au sens actualisé pour un $\gamma$ donné si : 

$$
v_\gamma^{\boldsymbol{\pi}^*}(s) \geq v_\gamma^{\boldsymbol{\pi}}(s) \quad \text { pour tout } s \in S \text { et toute } \boldsymbol{\pi} \in \Pi^{\mathrm{HR}}
$$

De plus, la valeur d'un processus de décision markovien actualisé, $v_\gamma^*(s)$, est définie par :

$$
v_\gamma^*(s) \equiv \max _{\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}} v_\gamma^{\boldsymbol{\pi}}(s)
$$

Le plus souvent, on désigne $v_\gamma$ simplement comme la fonction de valeur optimale. 

Comme dans le cadre à horizon fini, le modèle actualisé à horizon infini ne requiert pas de politiques dépendantes de l'historique, puisque pour toute $\boldsymbol{\pi} \in \Pi^{HR}$ il existe une $\boldsymbol{\pi}^{\prime} \in \Pi^{MR}$ qui donne la même récompense totale actualisée :
$$
v_\gamma^*(s) \equiv \max_{\boldsymbol{\pi} \in \Pi^{HR}} v_\gamma^{\boldsymbol{\pi}}(s)=\max_{\boldsymbol{\pi} \in \Pi^{MR}} v_\gamma^{\boldsymbol{\pi}}(s) .
$$

## Interprétation de l'actualisation comme horizon aléatoire

Une terminaison géométrique peut-elle donner au facteur d'actualisation un sens
probabiliste, plutôt que d'en faire un simple artifice algébrique?
On peut motiver l'actualisation à la fois du point de vue de la modélisation et comme moyen de garantir que la récompense totale reste bornée. Du point de vue de la modélisation, l'actualisation permet d'accorder plus ou moins de poids aux récompenses immédiates par rapport aux conséquences à long terme. Il existe aussi une autre interprétation, qui part d'un modèle à horizon fini dont la date de fin est incertaine. Plus précisément :

Notons $v_\nu^{\boldsymbol{\pi}}(s)$ la récompense totale espérée obtenue en suivant la politique $\boldsymbol{\pi}$ lorsque la longueur de l'horizon $\nu$ est aléatoire. On la définit par :

$$
v_\nu^{\boldsymbol{\pi}}(s) \equiv \mathbb{E}_s^{\boldsymbol{\pi}}\left[\mathbb{E}_\nu\left\{\sum_{t=1}^\nu r(S_t, A_t)\right\}\right]
$$


````{prf:theorem} Interprétation de l'actualisation comme horizon aléatoire
:label: prop-5-3-1
Supposons que l'horizon $\nu$ suive une loi géométrique de paramètre $\gamma$, $0 \leq \gamma < 1$, indépendante de la politique, de sorte que 
$P(\nu=n) = (1-\gamma) \gamma^{n-1}, \, n=1,2, \ldots$; alors $v_\nu^{\boldsymbol{\pi}}(s) = v_\gamma^{\boldsymbol{\pi}}(s)$ pour tout $s \in \mathcal{S}$.
````

````{prf:proof}
Voir la proposition 5.3.1 de {cite}`Puterman1994`.

Par définition de la fonction de valeur à horizon fini et par la loi de l'espérance totale :

$$
v_\nu^{\boldsymbol{\pi}}(s) = \sum_{n=1}^{\infty} P(\nu=n) \cdot v_n^{\boldsymbol{\pi}}(s) = \sum_{n=1}^{\infty} (1-\gamma) \gamma^{n-1} \cdot E_s^{\boldsymbol{\pi}} \left\{\sum_{t=1}^n r(S_t, A_t)\right\}.
$$

En regroupant l'espérance et la somme sur $n$ :

$$
v_\nu^{\boldsymbol{\pi}}(s) = E_s^{\boldsymbol{\pi}} \left\{\sum_{n=1}^{\infty} (1-\gamma) \gamma^{n-1} \sum_{t=1}^n r(S_t, A_t)\right\}.
$$

**Interversion des sommes :** sous l'hypothèse de récompenses bornées $|r(s,a)| \leq R_{\max}$ et avec $\gamma < 1$, on a

$$
E_s^{\boldsymbol{\pi}} \left\{\sum_{n=1}^{\infty} \sum_{t=1}^n |r(S_t, A_t)| \cdot (1-\gamma) \gamma^{n-1}\right\} \leq R_{\max} \sum_{n=1}^{\infty} n (1-\gamma) \gamma^{n-1} = \frac{R_{\max}}{1-\gamma} < \infty,
$$
ce qui justifie l'interversion de l'ordre de sommation par le théorème de Fubini.

Pour inverser l'ordre, on remarque que les couples $(n,t)$ avec $1 \leq t \leq n$ peuvent être réindexés en fixant d'abord $t$, puis en faisant varier $n$ de $t$ à $\infty$ :

$$
\sum_{n=1}^{\infty} \sum_{t=1}^n = \sum_{t=1}^{\infty} \sum_{n=t}^{\infty}.
$$

Par conséquent :
\begin{align*}
v_\nu^{\boldsymbol{\pi}}(s) &= E_s^{\boldsymbol{\pi}} \left\{\sum_{t=1}^{\infty} r(S_t, A_t) \sum_{n=t}^{\infty} (1-\gamma) \gamma^{n-1}\right\}.
\end{align*}

**Calcul de la somme intérieure :** avec le changement d'indice $m = n - t + 1$ (donc $n = m + t - 1$) :
\begin{align*}
\sum_{n=t}^{\infty} (1-\gamma) \gamma^{n-1} &= \sum_{m=1}^{\infty} (1-\gamma) \gamma^{m+t-2} \\
&= \gamma^{t-1} (1-\gamma) \sum_{m=1}^{\infty} \gamma^{m-1} \\
&= \gamma^{t-1} (1-\gamma) \cdot \frac{1}{1-\gamma} = \gamma^{t-1}.
\end{align*}

En substituant :

$$
v_\nu^{\boldsymbol{\pi}}(s) = E_s^{\boldsymbol{\pi}} \left\{\sum_{t=1}^{\infty} \gamma^{t-1} r(S_t, A_t)\right\} = v_\gamma^{\boldsymbol{\pi}}(s).
$$

````


Sous cet horizon géométrique, la longueur espérée vaut $\mathbb{E}[\nu] = 1/(1-\gamma)$; on l'appelle souvent l'**horizon effectif** du problème actualisé : $\gamma = 0.99$ correspond à planifier environ 100 pas à l'avance en moyenne. On peut simuler directement cette construction. Après avoir recueilli chaque récompense, on lance une pièce qui tombe sur pile avec probabilité $\gamma$ : on continue sur pile et on termine l'épisode sur face. Le rendement non actualisé d'un tel épisode est un estimateur sans biais de $v_\gamma^{\boldsymbol{\pi}}(s)$. En pratique, on simule rarement de cette façon. Tronquer les épisodes au hasard jette les récompenses qui suivent le moment où la pièce tombe sur face, ce qui augmente la variance de l'estimation et, en apprentissage, gaspille un signal qui aurait pu servir. L'équivalence reste utile à garder en tête, car elle permet à un algorithme d'échanger un facteur d'actualisation contre une probabilité d'arrêt, et inversement.

### Représenter un processus de décision markovien par un graphe, et l'état cimetière

On représente souvent les petits processus de décision markoviens par des graphes orientés, comme le fait Puterman pour ses exemples jouets. Chaque état est un nœud. Chaque action disponible dans un état fait partir de ce nœud des flèches, une pour chaque état suivant possible, étiquetées par la probabilité $p(j \mid s, a)$. Les ensembles d'actions peuvent varier d'un état à l'autre : dans la {numref}`fig-mdp-graph-absorbing`(a), l'état $s_1$ possède deux actions et l'état $s_2$ une seule, comme le permettent les ensembles $\mathcal{A}_s$ qui dépendent de l'état. Sans les actions, un tel dessin est le graphe de transition d'une chaîne de Markov. Fixer une règle de décision $\pi$ revient à choisir (ou à tirer au hasard) les flèches qui sortent de chaque nœud, et il ne reste que cette chaîne de Markov, dont la matrice de transition est la matrice $\mathbf{P}_\pi$ de la section suivante. Un système commandé est une famille de chaînes de Markov, une par règle de décision, et $\mathbf{P}_\pi$ intègre le choix de l'action dans les probabilités de transition.

:::{figure} _static/mdp-graph-absorbing.svg
:label: fig-mdp-graph-absorbing
:alt: À gauche, un processus de décision markovien à deux états représenté par un graphe dont les flèches sont étiquetées par les actions. À droite, le même graphe où chaque probabilité est multipliée par gamma, avec un état absorbant supplémentaire atteint avec probabilité un moins gamma depuis chaque état.

(a) Un processus de décision markovien à deux états représenté par un graphe. En $s_1$, l'action $a_1$ maintient le système en $s_1$ et l'action $a_2$ le fait passer en $s_2$; en $s_2$, seule $a_1$ est disponible. (b) Le même processus avec un état cimetière $\Delta$. Chaque probabilité de transition d'origine est multipliée par $\gamma$, et depuis chaque état le système passe en $\Delta$ avec probabilité $1-\gamma$, quelle que soit l'action. $\Delta$ est absorbant et ne rapporte aucune récompense.
:::

Le graphe donne une troisième lecture de l'actualisation. Ajoutons un nouvel état $\Delta$, parfois appelé état cimetière ou état puits. Depuis chaque état $s$ et pour chaque action $a$, on passe en $\Delta$ avec probabilité $1-\gamma$ et, sinon, on suit la dynamique d'origine, de sorte que les nouvelles probabilités de transition sont $\tilde p(j \mid s,a) = \gamma\, p(j \mid s,a)$ pour $j \in \mathcal{S}$ et $\tilde p(\Delta \mid s,a) = 1-\gamma$. Une fois en $\Delta$, le système y reste pour toujours et reçoit une récompense nulle. Dans ce processus auxiliaire, la récompense totale espérée non actualisée de toute politique est égale à sa valeur actualisée dans le processus d'origine, car la probabilité d'être encore en vie au pas $t$ vaut $\gamma^{t-1}$. C'est la construction du théorème {prf:ref}`prop-5-3-1`, écrite sous forme d'un état plutôt que d'une variable aléatoire, et c'est ainsi qu'on ramène les problèmes actualisés à des problèmes de récompense totale avec états absorbants, parfois appelés problèmes de plus court chemin stochastique.

Tous les processus n'ont pas d'état absorbant. Un système peut fonctionner indéfiniment et, pour que son comportement à long terme soit bien défini, il faut une certaine structure. L'hypothèse habituelle est l'ergodicité : la chaîne $\mathbf{P}_\pi$ se stabilise dans une loi stationnaire qui répartit sa masse sur de nombreux états, et les moyennes à long terme sont des moyennes sous cette loi. C'est le cadre du critère de récompense moyenne évoqué au début du chapitre, que nous ne développons pas.

## Représentation vectorielle des processus de décision markoviens

Comment les noyaux de transition, les récompenses et les politiques
deviennent-ils des matrices et des vecteurs qui font apparaître les équations
de Bellman comme des équations d'opérateurs?

Soit V l'ensemble des fonctions réelles bornées sur un espace d'états discret S. Autrement dit, toute fonction $ f \in V $ satisfait la condition :

$$
\|f\| = \max_{s \in S} |f(s)| < \infty.
$$
où la notation $ \|f\| $ désigne la norme sup (ou norme $ \ell_\infty $) de la fonction $ f $. 

Lorsqu'on travaille avec des espaces d'états discrets, on peut interpréter les éléments de V comme des vecteurs et les opérateurs linéaires sur V comme des matrices, ce qui permet de tirer parti des outils de l'algèbre linéaire. La norme sup (norme $\ell_\infty$) d'une matrice $\mathbf{H}$ est définie par :

$$
\|\mathbf{H}\| \equiv \max_{s \in S} \sum_{j \in S} |\mathbf{H}_{s,j}|
$$

où $\mathbf{H}_{s,j}$ désigne la composante $(s, j)$ de la matrice $\mathbf{H}$.

Pour une règle de décision markovienne $\pi \in \Pi^{MD}$, on définit :

\begin{align*}
\mathbf{r}_\pi(s) &\equiv r(s, \pi(s)), \quad \mathbf{r}_\pi \in \mathbb{R}^{|S|}, \\
[\mathbf{P}_\pi]_{s,j} &\equiv p(j \mid s, \pi(s)), \quad \mathbf{P}_\pi \in \mathbb{R}^{|S| \times |S|}.
\end{align*}

Pour une règle de décision aléatoire $\pi \in \Pi^{MR}$, ces définitions deviennent :

\begin{align*}
\mathbf{r}_\pi(s) &\equiv \sum_{a \in A_s} \pi(a \mid s) \, r(s, a), \\
[\mathbf{P}_\pi]_{s,j} &\equiv \sum_{a \in A_s} \pi(a \mid s) \, p(j \mid s, a).
\end{align*}

Dans les deux cas, $\mathbf{r}_\pi$ désigne un vecteur de récompenses dans $\mathbb{R}^{|S|}$, dont chaque composante $\mathbf{r}_\pi(s)$ représente la récompense associée à l'état $s$. De même, $\mathbf{P}_\pi$ est une matrice de probabilités de transition dans $\mathbb{R}^{|S| \times |S|}$, qui décrit les probabilités de transition sous la règle de décision $\pi$.

Pour une politique markovienne non stationnaire $\boldsymbol{\pi} = (\pi_1, \pi_2, \ldots) \in \Pi^{MR}$, la récompense totale actualisée espérée est donnée par :

$$
\mathbf{v}_\gamma^{\boldsymbol{\pi}}(s)=\mathbb{E}\left[\sum_{t=1}^{\infty} \gamma^{t-1} r\left(S_t, A_t\right) \,\middle|\, S_1 = s\right].
$$

En notation vectorielle, et en écrivant $\mathbf{P}_{\boldsymbol{\pi}}^{t-1} \equiv \mathbf{P}_{\pi_1} \mathbf{P}_{\pi_2} \cdots \mathbf{P}_{\pi_{t-1}}$ pour la matrice de transition à $t-1$ pas de la politique non stationnaire (avec $\mathbf{P}_{\boldsymbol{\pi}}^{0} \equiv \mathbf{I}$), elle s'écrit :

$$
\begin{aligned}
\mathbf{v}_\gamma^{\boldsymbol{\pi}} &= \sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_{\boldsymbol{\pi}}^{t-1} \mathbf{r}_{\pi_t} \\
&= \mathbf{r}_{\pi_1} + \gamma \mathbf{P}_{\pi_1} \mathbf{r}_{\pi_2} + \gamma^2 \mathbf{P}_{\pi_1} \mathbf{P}_{\pi_2} \mathbf{r}_{\pi_3} + \cdots \\
&= \mathbf{r}_{\pi_1} + \gamma \mathbf{P}_{\pi_1} \left( \mathbf{r}_{\pi_2} + \gamma \mathbf{P}_{\pi_2} \mathbf{r}_{\pi_3} + \gamma^2 \mathbf{P}_{\pi_2} \mathbf{P}_{\pi_3} \mathbf{r}_{\pi_4} + \cdots \right).
\end{aligned}
$$

Chaque terme se lit directement : $[\mathbf{P}_{\boldsymbol{\pi}}^{t-1}]_{s,j}$ est la probabilité d'être dans l'état $j$ au temps $t$ en partant de $s$, de sorte que $\gamma^{t-1}\mathbf{P}_{\boldsymbol{\pi}}^{t-1}\mathbf{r}_{\pi_t}$ est la récompense actualisée espérée recueillie $t-1$ pas plus tard. Les produits de matrices font la moyenne sur tous les chemins que le système peut suivre sous la politique, ce qu'une estimation de Monte-Carlo approcherait par simulation.

En factorisant $\gamma \mathbf{P}_{\pi_1}$ dans tous les termes après le premier, on obtient une relation récursive :

$$
\mathbf{v}_\gamma^{\boldsymbol{\pi}} = \mathbf{r}_{\pi_1} + \gamma \mathbf{P}_{\pi_1} \mathbf{v}_\gamma^{\boldsymbol{\pi}^{\prime}},
$$

où $\boldsymbol{\pi}^{\prime} = (\pi_2, \pi_3, \ldots)$ est la politique décalée d'un pas. En déroulant cette récurrence, on retrouve la série ci-dessus; la série est donc la solution de la récurrence.


Pour une politique stationnaire $\boldsymbol{\pi} = \mathrm{const}(\pi)$ de règle de décision constante $\pi$, la récompense totale espérée se simplifie en :

$$
\begin{align*}
\mathbf{v}_\gamma^{\pi} &= \mathbf{r}_\pi+ \gamma \mathbf{P}_\pi \mathbf{v}_\gamma^{\pi} \\
&=\sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_\pi^{t-1} \mathbf{r}_{\pi}
\end{align*}
$$

Cette dernière expression est appelée développement en série de Neumann; son existence est garantie sous les hypothèses de récompenses bornées et d'un facteur d'actualisation strictement inférieur à un. 

```{prf:theorem} Série de Neumann et inversibilité
:label: neumann-series

Le **rayon spectral** d'une matrice $\mathbf{H}$ est défini par :

$$
\rho(\mathbf{H}) \equiv \max_{i} |\lambda_i(\mathbf{H})|
$$

où les $\lambda_i(\mathbf{H})$ sont les valeurs propres de $\mathbf{H}$.

**Existence de la série de Neumann :** pour toute matrice $\mathbf{H}$, la série de Neumann

$$
\sum_{t=0}^{\infty} \mathbf{H}^t = \mathbf{I} + \mathbf{H} + \mathbf{H}^2 + \cdots
$$

converge si et seulement si $\rho(\mathbf{H}) < 1$. Lorsque cette condition est satisfaite, la matrice $(\mathbf{I} - \mathbf{H})$ est inversible et

$$
(\mathbf{I} - \mathbf{H})^{-1} = \sum_{t=0}^{\infty} \mathbf{H}^t.
$$

```
Notons que pour toute norme matricielle subordonnée $\|\cdot\|$ (c'est-à-dire une norme qui satisfait $\|\mathbf{H}\mathbf{v}\| \leq \|\mathbf{H}\| \cdot \|\mathbf{v}\|$ pour tout vecteur $\mathbf{v}$) et toute matrice $\mathbf{H}$, le rayon spectral est borné par :

$$
\rho(\mathbf{H}) \leq \|\mathbf{H}\|.
$$


Cette inégalité offre un moyen pratique de vérifier la condition de convergence $\rho(\mathbf{H}) < 1$ : il suffit de vérifier la condition plus simple $\|\mathbf{H}\| < 1$ au lieu de calculer directement les valeurs propres.

On peut maintenant vérifier que $(\mathbf{I} - \gamma \mathbf{P}_\pi)$ est inversible et que la série de Neumann converge.

1. **Norme de la matrice de transition :** puisque $\mathbf{P}_\pi$ est une matrice stochastique (chaque ligne somme à 1 et tous les coefficients sont positifs ou nuls), sa norme $\ell_\infty$ vaut :

   $$
   \|\mathbf{P}_\pi\| = \max_{s \in S} \sum_{j \in S} [\mathbf{P}_\pi]_{s,j} = \max_{s \in S} 1 = 1.
   $$

2. **Norme de la matrice mise à l'échelle :** par la propriété d'homogénéité des normes, on a :

   $$
   \|\gamma \mathbf{P}_\pi\| = |\gamma| \cdot \|\mathbf{P}_\pi\| = |\gamma| \cdot 1 = |\gamma|.
   $$

3. **Borne sur le rayon spectral :** puisque le rayon spectral est borné par la norme matricielle :

   $$
   \rho(\gamma \mathbf{P}_\pi) \leq \|\gamma \mathbf{P}_\pi\| = |\gamma|.
   $$

4. **Vérification de la convergence :** puisque $0 \leq \gamma < 1$ par hypothèse, on a :

   $$
   \rho(\gamma \mathbf{P}_\pi) \leq |\gamma| < 1.
   $$
   
   Cette inégalité stricte garantit que $(\mathbf{I} - \gamma \mathbf{P}_\pi)$ est inversible et que la série de Neumann converge.

Par conséquent, le développement en série de Neumann converge et donne :

$$
\mathbf{v}_\gamma^{\pi} = (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi = \sum_{t=0}^{\infty} (\gamma \mathbf{P}_\pi)^t \mathbf{r}_\pi = \sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_\pi^{t-1} \mathbf{r}_\pi.
$$

La somme du milieu est indexée à partir de $t=0$ pour correspondre à l'énoncé du théorème {prf:ref}`neumann-series`, tandis que la dernière conserve la convention $S_1 = s$ utilisée pour la fonction de valeur. Il s'agit de la même série, décalée d'un indice.

Par conséquent, pour une politique stationnaire, $\mathbf{v}_\gamma^{\pi}$ peut être obtenue comme solution de l'équation linéaire :

$$
\mathbf{v} = \mathbf{r}_\pi+ \gamma \mathbf{P}_\pi\mathbf{v},
$$

qu'on peut réécrire sous la forme :

$$
(\mathbf{I} - \gamma \mathbf{P}_\pi) \mathbf{v} = \mathbf{r}_\pi.
$$

On peut aussi caractériser $\mathbf{v}_\gamma^{\pi}$ comme la solution d'une équation d'opérateur. Plus précisément, on définit la transformation $\mathrm{L}_\pi$ par

$$
\mathrm{L}_\pi \mathbf{v} \equiv \mathbf{r}_\pi+\gamma \mathbf{P}_\pi\mathbf{v}
$$

pour tout $\mathbf{v} \in V$. Intuitivement, $\mathrm{L}_\pi$ prend en entrée une fonction de valeur $\mathbf{v}$ et renvoie une nouvelle fonction de valeur qui combine les récompenses immédiates ($\mathbf{r}_\pi$) et les valeurs futures actualisées ($\gamma \mathbf{P}_\pi\mathbf{v}$). 

```{note}
Bien qu'on parle souvent de $\mathrm{L}_\pi$ comme d'un « opérateur linéaire » dans la littérature en apprentissage par renforcement, il s'agit techniquement d'un **opérateur affine** (ou transformation affine), et non d'un opérateur linéaire au sens strict. Pour le voir, rappelons qu'un opérateur linéaire $\mathcal{T}$ doit satisfaire :

1. **Additivité :** $\mathcal{T}(\mathbf{v}_1 + \mathbf{v}_2) = \mathcal{T}(\mathbf{v}_1) + \mathcal{T}(\mathbf{v}_2)$
2. **Homogénéité :** $\mathcal{T}(\alpha \mathbf{v}) = \alpha \mathcal{T}(\mathbf{v})$ pour tout scalaire $\alpha$

Or, $\mathrm{L}_\pi$ ne passe pas le test d'additivité :

$$
\mathrm{L}_\pi(\mathbf{v}_1 + \mathbf{v}_2) = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi(\mathbf{v}_1 + \mathbf{v}_2) = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_1 + \gamma \mathbf{P}_\pi\mathbf{v}_2
$$

alors que

$$
\mathrm{L}_\pi(\mathbf{v}_1) + \mathrm{L}_\pi(\mathbf{v}_2) = (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_1) + (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_2) = 2\mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_1 + \gamma \mathbf{P}_\pi\mathbf{v}_2.
$$

La présence du terme constant $\mathbf{r}_\pi$ rend $\mathrm{L}_\pi$ affine plutôt que linéaire. Un opérateur affine a la forme $\mathcal{A}(\mathbf{v}) = \mathbf{b} + \mathcal{T}(\mathbf{v})$, où $\mathbf{b}$ est un vecteur constant et $\mathcal{T}$ un opérateur linéaire. Dans notre cas, $\mathbf{b} = \mathbf{r}_\pi$ et $\mathcal{T}(\mathbf{v}) = \gamma \mathbf{P}_\pi\mathbf{v}$.

Malgré cette distinction technique, l'expression « opérateur linéaire » est couramment employée dans la littérature en apprentissage par renforcement pour désigner $\mathrm{L}_\pi$, par un léger abus de langage.
```

On considère donc $\mathrm{L}_\pi$ comme un opérateur qui envoie les éléments de $V$ dans $V$, c'est-à-dire $\mathrm{L}_\pi: V \rightarrow V$. Le fait que la fonction de valeur d'une politique soit la solution d'une équation de point fixe s'exprime alors par l'énoncé : 

$$
\mathbf{v}_\gamma^{\pi}=\mathrm{L}_\pi \mathbf{v}_\gamma^{\pi}.
$$

C'est une **équation de point fixe** : la fonction de valeur $\mathbf{v}_\gamma^{\pi}$ est un point fixe de l'opérateur $\mathrm{L}_\pi$.

## Résolution des équations d'opérateurs

Quelles méthodes itératives, et lesquelles apparentées à celle de Newton,
permettent de résoudre un point fixe lorsque l'inversion directe d'une matrice
est impossible ou inappropriée?

L'équation d'opérateur rencontrée dans les processus de décision markoviens, $\mathbf{v}_\gamma^{\pi} = \mathrm{L}_\pi \mathbf{v}_\gamma^{\pi}$, est un cas particulier d'une classe plus générale de problèmes appelés équations d'opérateurs. Ces équations apparaissent dans de nombreux domaines des mathématiques et des sciences appliquées, des équations différentielles à l'analyse fonctionnelle.

Les équations d'opérateurs peuvent prendre plusieurs formes, chacune avec ses propres caractéristiques et méthodes de résolution :

1. **Forme de point fixe** : $x = \mathrm{T}(x)$, où $\mathrm{T}: X \rightarrow X$.
   Cette forme est courante dans les problèmes de point fixe, comme notre équation de processus de décision markovien : on cherche un point fixe $x^*$ tel que $x^* = \mathrm{T}(x^*)$.

2. **Équation d'opérateur générale** : $\mathrm{T}(x) = y$, où $\mathrm{T}: X \rightarrow Y$.
   Ici, $X$ et $Y$ peuvent être des espaces différents. On cherche un $x \in X$ qui satisfait l'équation pour un $y \in Y$ donné.

3. **Équation non linéaire** : $\mathrm{T}(x) = 0$, où $\mathrm{T}: X \rightarrow Y$.
   Cas particulier de l'équation d'opérateur générale où l'on cherche les racines, ou zéros, de l'opérateur.

4. **Inéquation variationnelle** : trouver $x^* \in K$ tel que $\langle \mathrm{T}(x^*), x - x^* \rangle \geq 0$ pour tout $x \in K$.
   Ici, $K$ est une partie convexe fermée de $X$, et $\mathrm{T}: K \rightarrow X^*$ (l'espace dual de $X$). Ces problèmes apparaissent souvent en optimisation, en théorie des jeux et dans les équations aux dérivées partielles.

:::{iframe} ../interactive/fixed-point-lab.html
:label: fig-fixed-point-lab
:width: 100%
:title: Laboratoire de point fixe
:class: fixed-point-lab-replay

La même itération sur quatre inconnues : un nombre, une trajectoire de pendule, la
valeur d'un processus de décision markovien à un état et la fonction de valeur
de l'eau du réservoir. Chaque méthode trace une droite passant par $(x, \mathrm{T}(x))$
avec une certaine pente $s$, puis saute au point où cette droite coupe $y = x$.
Une pente nulle donne les approximations successives et l'itération sur les
valeurs. La pente de $\mathrm{T}$ donne la méthode de Newton et l'itération sur les
politiques.
:::

### Méthode des approximations successives

Pour les équations sous forme de point fixe, l'outil de base est la méthode des approximations successives, aussi appelée itération de point fixe : partir d'une estimation quelconque et appliquer l'opérateur de façon répétée. Son succès dépend de l'opérateur, et la condition suffisante habituelle est que celui-ci soit une contraction.

````{prf:definition} Application contractante
:label: def-contraction-mapping

Soit $(X, \|\cdot\|)$ un espace normé. Un opérateur $\mathrm{T}: X \rightarrow X$ est une **contraction** pour $\|\cdot\|$ s'il existe une constante $q \in [0,1)$, le **facteur de contraction**, telle que pour tous $x, y \in X$ :

$$
\|\mathrm{T}(x) - \mathrm{T}(y)\| \leq q \, \|x - y\|.
$$
````

Autrement dit, une contraction est une application lipschitzienne dont la constante de Lipschitz est strictement inférieure à un. Cette propriété est liée à une norme : un opérateur peut être une contraction pour une norme et ne pas l'être pour une autre. Pour les processus de décision markoviens, la norme qui convient est la norme sup $\|\cdot\|_\infty$, et désormais $\|\cdot\|$ sans indice désigne la norme sup. Lorsque nous dirons plus loin que $\mathrm{L}_\pi$ ou $\Bellman$ est une contraction, l'énoncé précis sera qu'il s'agit d'une **contraction pour la norme sup**. Les résultats démontrés pour la norme sup ne se transposent pas gratuitement à d'autres normes; le fait qu'un opérateur de Bellman composé avec une projection ne soit pas nécessairement une contraction pour une norme $\ell_2$ pondérée est l'une des racines techniques de l'instabilité que Sutton appelle la triade mortelle.

````{prf:theorem} Théorème du point fixe de Banach
:label: banach-fixed-point

Soit $(X, \|\cdot\|)$ un espace de Banach (un espace normé complet) et soit $\mathrm{T}: X \rightarrow X$ une contraction de facteur $q \in [0,1)$. Alors :

1. $\mathrm{T}$ admet un point fixe **unique** $x^* = \mathrm{T}(x^*)$.
2. Pour tout $x_0 \in X$, la suite $x_{n+1} = \mathrm{T}(x_n)$ converge vers $x^*$.
3. L'erreur satisfait la borne a priori $\|x_n - x^*\| \leq \frac{q^n}{1-q}\|x_1 - x_0\|$ et la borne a posteriori $\|x_{n+1} - x^*\| \leq \frac{q}{1-q}\|x_{n+1} - x_n\|$.
````

Le théorème est un énoncé d'existence et d'unicité, et sa démonstration est constructive : la suite qui sert à montrer qu'un point fixe existe est l'algorithme qui le calcule. L'unicité compte, car rien dans une équation $x = \mathrm{T}(x)$ prise seule n'exclut de nombreuses solutions. Le théorème ne donne qu'une condition suffisante. Un opérateur qui n'est pas une contraction peut tout de même avoir un point fixe, et d'autres théorèmes de point fixe couvrent certains de ces cas.

:::{iframe} ../interactive/contraction-cloud.html
:label: fig-contraction-cloud
:width: 100%
:title: Nuage contracté
:class: contraction-cloud-replay

Ce que rétrécit une contraction. Chaque point est une fonction de valeur $v = (v(1), v(2))$ d'un
processus de décision markovien à deux états, et chaque pas applique l'opérateur
d'optimalité de Bellman (défini plus loin dans ce chapitre) à tous les points à la fois.
Le carré en pointillés, dont le côté est le diamètre du nuage pour la norme sup,
rétrécit d'un facteur $\gamma$ à chaque pas, et le nuage entier s'effondre sur
l'unique point fixe $v^*$.
:::

````{prf:algorithm} Approximations successives
:label: successive-approximation

**Entrée :** un opérateur $\mathrm{T}: X \rightarrow X$, une estimation initiale $x_0 \in X$ et une tolérance $\epsilon > 0$  
**Sortie :** un itéré $x_{n+1}$ dont le dernier pas satisfait $\|x_{n+1} - x_n\| < \epsilon$

1. Initialiser $n = 0$  
2. **répéter**  
    3. Calculer $x_{n+1} = \mathrm{T}(x_n)$  
    4. Si $\|x_{n+1} - x_n\| < \epsilon$, **renvoyer** $x_{n+1}$  
    5. Poser $n = n + 1$  
6. **jusqu'à** convergence ou atteinte du nombre maximal d'itérations  

````

Le critère d'arrêt ne mesure que la taille du dernier pas, une quantité calculée à partir des itérés eux-mêmes. À lui seul, il ne dit rien de la distance au point fixe. Pour en faire une telle garantie, il faut un travail supplémentaire, et c'est ce que fournit la borne a posteriori du théorème {prf:ref}`banach-fixed-point` pour une contraction : s'arrêter lorsque $\|x_{n+1} - x_n\| < \epsilon$ garantit $\|x_{n+1} - x^*\| < \frac{q}{1-q}\epsilon$. Lorsque $q$ est proche de un, un petit pas peut encore laisser l'itéré loin de $x^*$. Pour les processus de décision markoviens, nous choisirons la tolérance en conséquence, de sorte que l'itération sur les valeurs renvoie une politique $\varepsilon$-optimale ({prf:ref}`value-iteration-convergence`). Comme l'erreur diminue au moins d'un facteur $q$ à chaque pas, la vitesse de convergence est linéaire.

Cependant, la condition de contraction n'est pas la seule qui puisse mener à la convergence. Par exemple, si $\mathrm{T}$ est non expansif (c'est-à-dire lipschitzien de constante de Lipschitz 1) et si $X$ est un espace de Banach doté de certaines propriétés géométriques (par exemple uniformément convexe), alors, sous des conditions supplémentaires (par exemple que $\mathrm{T}$ ait au moins un point fixe), la méthode des approximations successives peut encore converger, quoique potentiellement plus lentement que dans le cas d'une contraction.

En pratique, pour des problèmes particuliers comme les processus de décision markoviens ou les équations différentielles, les propriétés de l'opérateur correspondent souvent naturellement à l'une de ces conditions de convergence. Par exemple, dans les processus de décision markoviens actualisés, l'opérateur de Bellman est une contraction pour la norme du supremum, ce qui garantit la convergence de l'itération sur les valeurs.

Pour l'évaluation d'une politique, les approximations successives remplacent la résolution linéaire par la boucle $\mathbf{v}_{k+1} = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}_k$. Une résolution directe de $(\mathbf{I} - \gamma \mathbf{P}_\pi)\mathbf{v} = \mathbf{r}_\pi$ coûte de l'ordre de $|S|^3$ opérations, alors que chaque itération coûte un produit matrice-vecteur, et la boucle n'a jamais besoin de former ni de factoriser une matrice. Ici, $\mathrm{L}_\pi$ est une contraction de facteur $\gamma$, par le même argument de norme que celui utilisé pour la série de Neumann.

Le second outil général pour les équations d'opérateurs est la méthode de Newton-Kantorovitch, qui étend la méthode de Newton aux espaces de fonctions. Nous la reportons à la section {ref}`sec-newton-kantorovich-pi` : appliquée à l'équation d'optimalité de Bellman, elle donne l'itération sur les politiques, tout comme les approximations successives appliquées à la même équation donnent l'itération sur les valeurs.

## Équations d'optimalité des processus de décision markoviens à horizon infini

Comment la maximisation action par action transforme-t-elle l'évaluation d'une
politique fixée en équation d'optimalité de Bellman?

Rappelons que, dans le cadre à horizon fini, les équations d'optimalité sont :

$$
v_n(s) = \max_{a \in A_s} \left\{r(s, a) + \gamma \sum_{j \in S} p(j | s, a) v_{n+1}(j)\right\}
$$

où $v_n(s)$ est la fonction de valeur au pas de temps $n$ pour l'état $s$, $A_s$ est l'ensemble des actions disponibles dans l'état $s$, $r(s, a)$ est la fonction de récompense, $\gamma$ est le facteur d'actualisation et $p(j | s, a)$ est la probabilité de transition de l'état $s$ vers l'état $j$ sous l'action $a$.

Intuitivement, on s'attend à ce qu'en faisant tendre $n$ vers l'infini, on obtienne les équations non linéaires :

$$
v(s) = \max_{a \in A_s} \left\{r(s, a) + \gamma \sum_{j \in S} p(j | s, a) v(j)\right\}
$$

qu'on appelle les équations d'optimalité, ou équations de Bellman, des processus de décision markoviens à horizon infini.

On peut adopter le point de vue de la théorie des opérateurs en définissant des opérateurs sur l'espace $V$ des fonctions réelles bornées sur l'espace d'états $S$. Pour une règle markovienne déterministe $\pi \in \Pi^{MD}$, on définit l'**opérateur d'évaluation de politique** :

$$
(\BellmanPi v)(s) = r(s,\pi(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,\pi(s)) v(j)
$$

La terminologie varie ici. Bertsekas, de même que Sutton et Barto, appellent $\BellmanPi$ l'opérateur de Bellman associé à $\pi$, et une partie de la littérature théorique en apprentissage par renforcement l'appelle l'opérateur de cohérence de Bellman. Dans ce livre, nous réservons le nom d'opérateur de Bellman à l'opérateur avec le maximum, défini ci-dessous, ce qui est plus proche de l'usage de Bellman lui-même, et nous appelons $\BellmanPi$ l'opérateur d'évaluation de politique, puisque c'est à évaluer une politique qu'il sert.

L'**opérateur d'optimalité de Bellman** s'écrit alors :

$$
\Bellman \mathbf{v} \equiv \max_{\pi \in \Pi^{MD}} \left\{\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}\right\}
$$

où $\Pi^{MD}$ est l'ensemble des règles de décision markoviennes déterministes, $\mathbf{r}_\pi$ est le vecteur de récompenses sous la règle de décision $\pi$ et $\mathbf{P}_\pi$ est la matrice de probabilités de transition sous la règle de décision $\pi$.

Notons que, même si l'on écrit $\max_{\pi \in \Pi^{MD}}$, on n'implémente pas l'opérateur ci-dessus en énumérant toutes les règles de décision. Comme on compare les politiques composante par composante à partir de leurs fonctions de valeur, la maximisation sur l'espace des règles markoviennes déterministes se ramène plutôt à la mise à jour suivante, écrite composante par composante :

$$
(\Bellman \mathbf{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}
$$

Les deux formes de $\Bellman$, un maximum sur les règles de décision et un maximum sur les actions dans chaque état, sont égales. La proposition suivante le précise.

```{prf:proposition}
L'opérateur $\Bellman$ défini comme une maximisation sur les règles de décision markoviennes déterministes :

$$(\Bellman \mathbf{v})(s) = \max_{\pi \in \Pi^{MD}} \left\{r(s,\pi(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,\pi(s)) v(j)\right\}$$

est équivalent à la maximisation sur les actions composante par composante :

$$(\Bellman \mathbf{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}$$
```

```{prf:proof}
Fixons $s$. Posons 

$$
Q_v(s,a) \triangleq r(s,a)+\gamma\sum_{j}p(j\mid s,a)\,v(j).
$$

Pour toute règle $\pi \in \Pi^{MD}$, on a $(\BellmanPi v)(s)=Q_v(s,\pi(s))\le \max_{a\in\mathcal{A}_s}Q_v(s,a)$.

En prenant le maximum sur $\pi$, on obtient

$$
\max_{\pi\in\Pi^{MD}}(\BellmanPi v)(s) \le \max_{a\in\mathcal{A}_s}Q_v(s,a).
$$

Réciproquement, choisissons un **sélecteur glouton** $\pi^v\in\Pi^{MD}$ tel que, pour chaque $s$,

$$\pi^v(s)\in\arg\max_{a\in\mathcal{A}_s}Q_v(s,a)$$

(c'est possible puisque $\mathcal{A}_s$ est fini; sinon, on utilise un sélecteur $\varepsilon$-glouton mesurable). Alors

$$
(\Bellman _{\pi^v}v)(s)=Q_v(s,\pi^v(s))=\max_{a\in\mathcal{A}_s}Q_v(s,a),
$$

donc $\max_{\pi}(\BellmanPi v)(s)\ge \max_{a}Q_v(s,a)$. La combinaison des deux inégalités donne l'égalité.
```

Par commodité, on définit le **sélecteur glouton** $\mathrm{Greedy}(v) \in \Pi^{MD}$, qui extrait une règle de décision d'une fonction de valeur :

$$
\mathrm{Greedy}(v)(s) \in \arg\max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}
$$

De façon équivalente, $\pi = \mathrm{Greedy}(v)$ est n'importe quelle règle de décision telle que $\BellmanPi v = \Bellman v$. Puterman appelle une telle règle **$v$-améliorante**. Une règle $v^*_\gamma$-améliorante, c'est-à-dire gloutonne par rapport à la fonction de valeur optimale, est dite **conservatrice**. Dans la littérature en apprentissage par renforcement, on parle simplement de politiques **gloutonnes** dans les deux cas : étant donné $v$, la politique peut agir de façon myope, car $v$ résume déjà l'avenir. Ce sélecteur permettra d'exprimer les algorithmes de façon concise :
- **Itération sur les valeurs :** itérer $v_{k+1} = \Bellman v_k$, puis extraire $\pi = \mathrm{Greedy}(v_k)$ à la terminaison
- **Itération sur les politiques :** $\pi_{k+1} = \mathrm{Greedy}(v^{\pi_k})$, où $v^{\pi_k}$ résout $v = \mathrm{L}_{\pi_k}v$

### Optimalité des politiques déterministes stationnaires

La définition de l'optimalité donnée au début du chapitre compare $\boldsymbol{\pi}^*$ à toutes les politiques de $\Pi^{\mathrm{HR}}$ : dépendantes de l'historique, non stationnaires et aléatoires. La comparaison se fait composante par composante, de sorte que $\boldsymbol{\pi}^*$ doit être au moins aussi bonne que toute autre politique dans chaque état de départ. Cette classe est beaucoup trop vaste pour être explorée directement. Une règle dépendante de l'historique $\pi_t(a \mid h_t)$, avec $h_t = (s_1, a_1, \ldots, a_{t-1}, s_t)$, dépend d'un historique qui croît avec $t$, et une politique non stationnaire exige une règle par étape, soit une infinité pour un horizon infini. Même en se limitant aux politiques markoviennes déterministes, énumérer toutes les politiques et les évaluer une à une est hors de portée. Le résultat suivant montre qu'on ne perd rien à explorer un ensemble beaucoup plus petit.

````{prf:theorem} Équation d'optimalité et politiques stationnaires optimales
:label: thm-stationary-optimal

Supposons que $0 \leq \gamma < 1$, que les récompenses soient bornées et que $S$ et chaque $\mathcal{A}_s$ soient finis. Alors :

1. $\Bellman$ est une contraction pour la norme sup de facteur $\gamma$; par le théorème {prf:ref}`banach-fixed-point`, l'équation d'optimalité $v = \Bellman v$ admet donc une solution unique, et cette solution est la fonction de valeur optimale $v^*_\gamma$.
2. Toute règle de décision conservatrice $\pi^* \in \mathrm{Greedy}(v^*_\gamma)$ définit une politique déterministe stationnaire $\mathrm{const}(\pi^*)$ qui est optimale sur tout $\Pi^{\mathrm{HR}}$ :

$$
v_\gamma^{\pi^*} = v^*_\gamma = \sup_{\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}} v_\gamma^{\boldsymbol{\pi}}.
$$
````

Voir {cite}`Puterman1994`, section 6.2, pour la démonstration. La propriété de contraction est établie dans la démonstration de la proposition {prf:ref}`value-iteration-convergence` ci-dessous. Le théorème a deux conséquences. D'abord, le supremum sur $\Pi^{\mathrm{HR}}$ est atteint, par une politique déterministe stationnaire; on peut donc restreindre la recherche à l'ensemble $\Pi^{MD}$ des règles de décision markoviennes déterministes appliquées à chaque étape. Ensuite, la fonction de valeur optimale est unique, mais la politique optimale ne l'est pas nécessairement : lorsque plusieurs actions atteignent le maximum dans un état, chaque choix donne une politique optimale. Le théorème est propre au cas actualisé à horizon infini. Dans un problème à horizon fini, la politique optimale dépend en général de l'étape.

### Fonctions Q

On peut aussi écrire les équations d'optimalité au moyen d'une fonction d'un état et d'une action. La **fonction Q optimale** (appelée *facteurs Q* dans la littérature en recherche opérationnelle) est l'unique solution de

$$
q(s,a) = r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j \mid s,a) \max_{a' \in \mathcal{A}_j} q(j,a').
$$

Comme pour $v$, le membre de droite définit un opérateur sur les fonctions de $(s,a)$, et les chapitres suivants l'écrivent avec le même symbole, $q^* = \Bellman q^*$. C'est une contraction pour la norme sup de facteur $\gamma$ lorsque $\gamma < 1$; $q^*$ existe donc et est unique. Les deux fonctions optimales se déterminent mutuellement :

$$
v^*_\gamma(s) = \max_{a \in \mathcal{A}_s} q^*(s,a), \qquad q^*(s,a) = r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j \mid s,a)\, v^*_\gamma(j),
$$

et une règle conservatrice se lit directement : $\pi^*(s) \in \arg\max_{a \in \mathcal{A}_s} q^*(s,a)$. Du point de vue probabiliste, $q^*$ conditionne à la fois sur l'état et sur la première action, alors que $v^*_\gamma$ fait la moyenne sur l'action. L'optimalité reste définie au moyen de $v$, car c'est ainsi que les politiques sont ordonnées.

Lorsque $\mathbf{P}$ et $r$ sont connus, comme dans les exemples de ce chapitre, il n'y a aucune raison de préférer $q$ : calculer $v$ coûte moins cher, et la règle gloutonne s'en déduit. La fonction Q prend son importance en apprentissage par renforcement sans modèle, où l'on observe des transitions et des récompenses échantillonnées, mais pas les probabilités de transition. Extraire une action gloutonne de $v$ exige l'espérance $\sum_j p(j \mid s,a)\, v(j)$, qui nécessite le modèle. L'extraire de $q$ exige seulement un maximum sur l'argument d'action, ce qui fonctionne pour une fonction $q$ apprise, un réseau de neurones par exemple, sans connaître $p$. C'est pourquoi la forme en fonction Q est le point de départ des [chapitres fondés sur la simulation](monte-carlo-bellman-estimation.md). Cette commodité a un prix. Une fonction Q possède une valeur par couple état-action au lieu d'une par état, et elle conserve le détail que $v$ efface en faisant la moyenne; apprendre $q^*$ demande donc en général plus de capacité, plus d'échantillons et une meilleure couverture des actions qu'apprendre $v^*_\gamma$.

## Algorithmes de résolution des équations d'optimalité

Quel compromis de calcul sépare l'itération sur les valeurs, l'évaluation de
politique et l'amélioration de politique?

Les équations d'optimalité sont des équations d'opérateurs. On peut donc leur appliquer des méthodes numériques générales. Appliquer la méthode des approximations successives à l'équation d'optimalité de Bellman donne une méthode appelée « itération sur les valeurs » en programmation dynamique. À la suite de Puterman et de l'ancienne littérature en recherche opérationnelle, nous réservons ce nom à l'itération avec le maximum. La même itération appliquée à l'opérateur affine $\mathrm{L}_\pi$ est l'évaluation itérative de politique, bien que les ouvrages d'apprentissage par renforcement appellent souvent itération sur les valeurs toute itération de cette forme. Une application directe du schéma des approximations successives donne l'algorithme suivant :

````{prf:algorithm} Itération sur les valeurs
:label: value-iteration

**Entrée** Un processus de décision markovien $(S, A, P, R, \gamma)$ et une tolérance $\varepsilon > 0$  

**Sortie** Une fonction de valeur $v$ et une politique $\pi$ $\varepsilon$-optimales  

1. Initialiser $v_0(s) = 0$ pour tout $s \in S$  
2. $n \leftarrow 0$  
3. **répéter**  

    1. Pour chaque $s \in S$ :  

        1. $v_{n+1}(s) \leftarrow (\Bellman v_n)(s) = \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v_n(j)\right\}$  

    2. $\delta \leftarrow \|v_{n+1} - v_n\|_\infty$  
    3. $n \leftarrow n + 1$  

4. **jusqu'à ce que** $\delta < \frac{\varepsilon(1-\gamma)}{2\gamma}$  
5. Extraire la politique gloutonne : $\pi \leftarrow \mathrm{Greedy}(v_n)$, où

    $$\mathrm{Greedy}(v)(s) \in \arg\max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v(j)\right\}$$

6. **renvoyer** $v_n, \pi$  
````

Le critère de terminaison de cet algorithme repose sur une borne précise, qui donne des garanties sur la qualité de la solution. C'est différent de l'apprentissage supervisé, où l'on utilise souvent des critères de terminaison arbitraires, fondés sur le budget de calcul ou sur l'arrêt précoce lorsque la courbe d'apprentissage s'aplatit. L'apprentissage supervisé dispose rarement de bornes de généralisation qu'on puisse calculer et utiliser comme critère d'arrêt.

En programmation dynamique, en revanche, on peut dériver diverses bornes utilisables en pratique. Ces bornes permettent d'arrêter la procédure avec une garantie sur la précision de la fonction de valeur et, par conséquent, sur l'optimalité de la politique obtenue.

L'itération sur les valeurs de {cite:t}`SuttonBarto2018` s'arrête lorsque la variation passe sous un petit seuil $\theta$, sans relier $\theta$ à la qualité de la politique renvoyée. Le seuil $\varepsilon(1-\gamma)/(2\gamma)$ ci-dessus est choisi pour que la réponse soit accompagnée d'un certificat : on fixe d'avance la précision $\varepsilon$, la boucle tourne jusqu'à ce que le test soit satisfait, et la politique gloutonne renvoyée est alors garantie $\varepsilon$-optimale (proposition {prf:ref}`value-iteration-convergence`). Deux mises en garde s'imposent. La garantie porte sur l'arithmétique exacte; les erreurs d'arrondi en virgule flottante ne font pas partie de l'analyse, et lorsque $\gamma$ est proche de un, un très petit $\varepsilon$ peut être en deçà de ce que float32, voire float64, peut résoudre. Par ailleurs, l'utilité d'un certificat dépend de l'application. Personne n'en a besoin pour résoudre un monde en grille, mais un économiste qui annonce la valeur monétaire d'une politique, ou un ingénieur qui doit documenter une marge de sécurité, en a besoin.

La fonction de valeur n'est qu'un instrument. Ce qu'on veut au bout du compte, c'est une politique, et, étant donné $v^*_\gamma$ (ou une approximation quelconque de celle-ci), on en obtient une par l'étape gloutonne $\mathrm{Greedy}(v)$ : pour chaque état, on calcule l'anticipation à un pas $r(s,a) + \gamma \sum_j p(j \mid s,a) v(j)$ pour chaque action et on garde la meilleure. Avec des tableaux de forme $(|\mathcal{S}|, |\mathcal{A}|)$, il s'agit d'un seul `argmax` selon l'axe des actions. On ne simule rien vers l'avant, car $v$ résume déjà l'avenir.

````{prf:proposition} Convergence de l'itération sur les valeurs 
:label: value-iteration-convergence
(Adaptée du théorème 6.3.1 de {cite:t}`Puterman1994`)

Soit $v_0$ une fonction de valeur initiale quelconque, $\varepsilon > 0$ une précision voulue, et soit $\{v_n\}$ la suite de fonctions de valeur engendrée par l'itération sur les valeurs, c'est-à-dire $v_{n+1} = \Bellman v_n$ pour $n \geq 0$, où $\Bellman$ est l'opérateur d'optimalité de Bellman. Alors :

1. $v_n$ converge vers la fonction de valeur optimale $v^*_\gamma$,
2. l'algorithme se termine en temps fini,
3. la politique obtenue $\pi_\varepsilon$ est $\varepsilon$-optimale, et
4. à la terminaison de l'algorithme, $v_{n+1}$ est à au plus $\varepsilon/2$ de $v^*_\gamma$.

````

````{prf:proof}
Les parties 1 et 2 découlent directement du fait que $\Bellman$ est une application contractante. Par le théorème du point fixe de Banach, il possède donc un point fixe unique (qui est $v^*_\gamma$), et l'application répétée de $\Bellman$ converge vers ce point fixe. De plus, cette convergence se fait à vitesse géométrique, ce qui garantit qu'on atteint la condition de terminaison en temps fini.

Pour montrer que l'opérateur d'optimalité de Bellman $\Bellman$ est une application contractante, il faut prouver que, pour toutes fonctions de valeur $v$ et $u$ :

$$\|\Bellman v - \Bellman u\|_\infty \leq \gamma \|v - u\|_\infty$$

où $\gamma \in [0,1)$ est le facteur d'actualisation et $\|\cdot\|_\infty$ la norme du supremum.

Commençons par écrire la définition de $\Bellman v$ et de $\Bellman u$ :

$$\begin{align*}
(\Bellman v)(s) &= \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v(j)\right\}\\
(\Bellman u)(s) &= \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)u(j)\right\}
\end{align*}$$

Fixons un état $s$. Soit $a_v$ une action qui atteint le maximum dans $(\Bellman v)(s)$, et $a_u$ une action qui atteint le maximum dans $(\Bellman u)(s)$. On borne la différence $(\Bellman v)(s) - (\Bellman u)(s)$ par le haut et par le bas, une direction à la fois.

**Borne supérieure.** Puisque $a_v$ maximise pour $v$, $(\Bellman v)(s)$ est *égal* à la valeur de $a_v$ :

$$(\Bellman v)(s) = r(s,a_v) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_v)v(j).$$

L'action $a_v$ ne maximise pas nécessairement pour $u$; l'insérer dans le maximum qui définit $(\Bellman u)(s)$ ne peut donc donner qu'une valeur inférieure ou égale :

$$(\Bellman u)(s) \geq r(s,a_v) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_v)u(j).$$

En soustrayant la deuxième ligne de la première, les récompenses $r(s,a_v)$ se simplifient, puisque les deux lignes utilisent la même action, et le résultat est une borne supérieure, puisqu'on soustrait une borne inférieure de $(\Bellman u)(s)$ :

$$(\Bellman v)(s) - (\Bellman u)(s) \leq \gamma \sum_{j \in \mathcal{S}} p(j|s,a_v)\bigl(v(j) - u(j)\bigr).$$

**Borne inférieure.** On échange les rôles de $v$ et de $u$, et on utilise $a_u$ à la place. Maintenant, $(\Bellman u)(s)$ est égal à la valeur de $a_u$ sous $u$, alors que $(\Bellman v)(s)$ est au moins égal à la valeur de $a_u$ sous $v$ :

$$\begin{align*}
(\Bellman u)(s) &= r(s,a_u) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_u)u(j),\\
(\Bellman v)(s) &\geq r(s,a_u) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_u)v(j).
\end{align*}$$

En soustrayant de la même façon :

$$(\Bellman u)(s) - (\Bellman v)(s) \leq \gamma \sum_{j \in \mathcal{S}} p(j|s,a_u)\bigl(u(j) - v(j)\bigr).$$

**Majoration de chaque somme par la norme max.** Chaque somme est une moyenne des différences $v(j) - u(j)$ (ou $u(j) - v(j)$), avec des poids $p(j|s,a) \geq 0$ dont la somme vaut $1$. Une moyenne n'est jamais plus grande que son plus grand terme, et chaque terme satisfait $v(j) - u(j) \leq |v(j) - u(j)| \leq \|v - u\|_\infty$. Par conséquent, pour l'une ou l'autre action $a \in \{a_v, a_u\}$ :

$$\sum_{j \in \mathcal{S}} p(j|s,a)\bigl(v(j) - u(j)\bigr) \leq \sum_{j \in \mathcal{S}} p(j|s,a)\,\|v - u\|_\infty = \|v - u\|_\infty,$$

et il en va de même avec $u(j) - v(j)$ au lieu de $v(j) - u(j)$. En insérant ce résultat dans les deux bornes ci-dessus, on obtient

$$(\Bellman v)(s) - (\Bellman u)(s) \leq \gamma \|v - u\|_\infty \quad\text{et}\quad (\Bellman u)(s) - (\Bellman v)(s) \leq \gamma \|v - u\|_\infty.$$

Un nombre $x$ tel que $x \leq c$ et $-x \leq c$ satisfait $|x| \leq c$, donc

$$|(\Bellman v)(s) - (\Bellman u)(s)| \leq \gamma \|v - u\|_\infty.$$

Comme cela vaut pour tout $s \in \mathcal{S}$, en prenant le supremum sur $s$, on obtient :

$$\|\Bellman v - \Bellman u\|_\infty \leq \gamma \|v - u\|_\infty$$

Ainsi, $\Bellman$ est une application contractante de facteur de contraction $\gamma$.

Démontrons maintenant les parties 3 et 4. Supposons que l'algorithme vienne de se terminer, c'est-à-dire que $\|v_{n+1} - v_n\|_\infty < \frac{\varepsilon(1-\gamma)}{2\gamma}$ pour un certain $n$. On veut montrer que la fonction de valeur courante $v_{n+1}$ et la politique $\pi_\varepsilon$ qui en est tirée sont proches de l'optimum.

Par l'inégalité triangulaire :

$$\|v^{\pi_\varepsilon}_\gamma - v^*_\gamma\|_\infty \leq \|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty + \|v_{n+1} - v^*_\gamma\|_\infty$$

Pour le premier terme, puisque $v^{\pi_\varepsilon}_\gamma$ est le point fixe de $\mathrm{L}_{\pi_\varepsilon}$ et que $\pi_\varepsilon$ est gloutonne par rapport à $v_{n+1}$ (c'est-à-dire $\mathrm{L}_{\pi_\varepsilon}v_{n+1} = \Bellman v_{n+1}$) :

$$
\begin{aligned}
\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty &= \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \\
&\leq \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - \mathrm{L}_{\pi_\varepsilon}v_{n+1}\|_\infty + \|\mathrm{L}_{\pi_\varepsilon}v_{n+1} - v_{n+1}\|_\infty \\
&= \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - \mathrm{L}_{\pi_\varepsilon}v_{n+1}\|_\infty + \|\Bellman v_{n+1} - v_{n+1}\|_\infty \\
&\leq \gamma\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty + \gamma\|v_{n+1} - v_n\|_\infty
\end{aligned}
$$

où l'on a utilisé le fait que $\Bellman$ et $\mathrm{L}_{\pi_\varepsilon}$ sont tous deux des contractions de facteur $\gamma$, et que $v_{n+1} = \Bellman v_n$.

En réarrangeant :

$$\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \leq \frac{\gamma}{1-\gamma}\|v_{n+1} - v_n\|_\infty$$

De même, puisque $v^*_\gamma$ est le point fixe de $\Bellman$ :

$$\|v_{n+1} - v^*_\gamma\|_\infty = \|\Bellman v_n - \Bellman v^*_\gamma\|_\infty \leq \gamma\|v_n - v^*_\gamma\|_\infty \leq \frac{\gamma}{1-\gamma}\|v_{n+1} - v_n\|_\infty$$

Puisque $\|v_{n+1} - v_n\|_\infty < \frac{\varepsilon(1-\gamma)}{2\gamma}$ :

$$\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \leq \frac{\gamma}{1-\gamma} \cdot \frac{\varepsilon(1-\gamma)}{2\gamma} = \frac{\varepsilon}{2}$$

$$\|v_{n+1} - v^*_\gamma\|_\infty \leq \frac{\gamma}{1-\gamma} \cdot \frac{\varepsilon(1-\gamma)}{2\gamma} = \frac{\varepsilon}{2}$$

En combinant ces résultats :

$$\|v^{\pi_\varepsilon}_\gamma - v^*_\gamma\|_\infty \leq \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon$$

Ceci achève la démonstration : $v_{n+1}$ est à au plus $\varepsilon/2$ de $v^*_\gamma$ (partie 4) et $\pi_\varepsilon$ est $\varepsilon$-optimale (partie 3).
````

### Un critère d'arrêt plus fin : la semi-norme d'étendue

Un bon critère d'arrêt s'arrête le plus tôt possible tout en fournissant la précision demandée. Le test fondé sur la norme sup n'est pas le meilleur dont on dispose. Un test plus fin mesure la variation entre itérés au moyen de la **semi-norme d'étendue**

$$
\operatorname{sp}(u) = \max_{s} u(s) - \min_{s} u(s).
$$

Il s'agit d'une semi-norme plutôt que d'une norme, car $\operatorname{sp}(c\mathbf{1}) = 0$ pour toute constante $c$ : l'étendue ignore un décalage uniforme de toutes les valeurs. Soit $d = v_{n+1} - v_n$ la dernière variation produite par l'itération sur les valeurs, de plus petite et de plus grande composantes $m = \min_s d(s)$ et $M = \max_s d(s)$. Deux propriétés de $\Bellman$ donnent des bornes des deux côtés. Il est monotone ($u \le w$ composante par composante entraîne $\Bellman u \le \Bellman w$), et il décale les constantes d'un facteur $\gamma$ : $\Bellman(u + c\mathbf{1}) = \Bellman u + \gamma c \mathbf{1}$. De $v_{n+1} \ge v_n + m\mathbf{1}$, on tire $\Bellman v_{n+1} \ge v_{n+1} + \gamma m \mathbf{1}$, et en répétant l'argument,

$$
v_{n+1} + \frac{\gamma}{1-\gamma}\, m\, \mathbf{1} \;\le\; v^*_\gamma \;\le\; v_{n+1} + \frac{\gamma}{1-\gamma}\, M\, \mathbf{1}.
$$

Le même argument avec $\mathrm{L}_\pi$ au lieu de $\Bellman$, pour $\pi = \mathrm{Greedy}(v_{n+1})$, montre que $v^{\pi}_\gamma \ge v_{n+1} + \frac{\gamma}{1-\gamma} m \mathbf{1}$. Par soustraction,

$$
\|v^*_\gamma - v^{\pi}_\gamma\|_\infty \le \frac{\gamma}{1-\gamma}\operatorname{sp}(v_{n+1} - v_n).
$$

Ainsi, s'arrêter lorsque $\operatorname{sp}(v_{n+1} - v_n) < \varepsilon(1-\gamma)/\gamma$ renvoie aussi une politique gloutonne $\varepsilon$-optimale ({cite:t}`Puterman1994`, section 6.6). Puisque $\operatorname{sp}(d) \le 2\|d\|_\infty$, ce test n'est jamais plus tardif que le test fondé sur la norme sup, et il est souvent beaucoup plus précoce. L'itération sur les valeurs a tendance à passer de nombreuses itérations à décaler toutes les valeurs d'à peu près la même quantité, ce que l'étendue ignore. L'avance obtenue dépend de la vitesse de mélange des chaînes $\mathbf{P}_\pi$ : lorsque chaque état atteint rapidement tous les autres, les variations $d$ deviennent presque constantes d'un état à l'autre après quelques itérations et leur étendue s'effondre, alors que leur norme sup ne diminue toujours que comme $\gamma^n$. Les bornes donnent aussi une meilleure estimation de la valeur que $v_{n+1}$ lui-même : le point milieu $v_{n+1} + \frac{\gamma}{1-\gamma}\frac{m+M}{2}\mathbf{1}$ est à au plus $\frac{\gamma}{2(1-\gamma)}\operatorname{sp}(d)$ de $v^*_\gamma$.

L'expérience ci-dessous vérifie ces deux affirmations. Elle tire des processus de décision markoviens aléatoires à 30 états et 4 actions, exécute l'itération sur les valeurs à partir de $v_0 = 0$ avec $\gamma = 0.95$ et $\varepsilon = 10^{-3}$ sous chacun des critères d'arrêt, puis évalue exactement la politique gloutonne renvoyée par une résolution linéaire. Pour faire varier la vitesse de mélange des chaînes, chaque matrice de transition est mélangée avec l'identité : avec probabilité `stay`, le système reste où il est. Chaque ligne fait la moyenne sur cinq processus aléatoires, et les deux dernières colonnes donnent le pire écart d'optimalité $\max_s (v^*_\gamma(s) - v^\pi_\gamma(s))$ sur ces cinq processus, que la théorie interdit de dépasser $\varepsilon$.

```{code-cell} python
import numpy as np


def random_mdp(seed, n_states=30, n_actions=4, stay=0.0):
    """Random transitions P[s, a, j] and rewards r[s, a]; `stay` adds self-loops."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(0.3 * np.ones(n_states), size=(n_states, n_actions))
    P = (1 - stay) * P + stay * np.eye(n_states)[:, None, :]
    r = rng.normal(size=(n_states, n_actions))
    return P, r


def value_iteration(P, r, gamma, eps, test):
    """Value iteration from v = 0 until the chosen stopping test passes."""
    v = np.zeros(P.shape[0])
    n = 0
    while True:
        n += 1
        v_next = (r + gamma * P @ v).max(axis=1)
        d = v_next - v
        if test == "sup" and np.abs(d).max() < eps * (1 - gamma) / (2 * gamma):
            break
        if test == "span" and d.max() - d.min() < eps * (1 - gamma) / gamma:
            break
        v = v_next
    greedy = (r + gamma * P @ v_next).argmax(axis=1)  # greedy w.r.t. the last iterate
    return n, greedy


def policy_value(P, r, gamma, policy):
    """Exact evaluation: solve (I - gamma P_pi) v = r_pi."""
    idx = np.arange(P.shape[0])
    return np.linalg.solve(np.eye(P.shape[0]) - gamma * P[idx, policy], r[idx, policy])


gamma, eps = 0.95, 1e-3
rows = [f"{'stay':>5} | {'sup-norm iters':>14} {'span iters':>10} | {'worst gap (sup)':>15} {'worst gap (span)':>16}"]
for stay in (0.0, 0.5, 0.9, 0.99):
    iters = {"sup": [], "span": []}
    gaps = {"sup": [], "span": []}
    for seed in range(5):
        P, r = random_mdp(seed, stay=stay)
        _, pi_star = value_iteration(P, r, gamma, 1e-10, "sup")  # reference optimal policy
        v_star = policy_value(P, r, gamma, pi_star)
        for test in ("sup", "span"):
            n, pi = value_iteration(P, r, gamma, eps, test)
            iters[test].append(n)
            gaps[test].append((v_star - policy_value(P, r, gamma, pi)).max())
    rows.append(f"{stay:>5} | {np.mean(iters['sup']):>14.0f} {np.mean(iters['span']):>10.0f} | "
                f"{max(gaps['sup']):>15.1e} {max(gaps['span']):>16.1e}")
print("\n".join(rows))
```

Avec des transitions aléatoires à mélange rapide (`stay = 0`), le test d'étendue s'arrête après environ 10 itérations, contre environ 200 pour le test fondé sur la norme sup. À mesure que les chaînes deviennent plus collantes, l'avantage diminue, et à `stay = 0.99` les deux tests s'arrêtent à environ 15 % l'un de l'autre. Dans tous les cas, les deux politiques renvoyées sont à au plus $\varepsilon$ de l'optimum.

### Laboratoire de contraction de Bellman

Utilisez les commandes ci-dessous pour modifier le facteur d'actualisation, la persistance des transitions, l'asymétrie des récompenses et la valeur de départ. Avant de déplacer $\gamma$, prédisez comment il modifiera la pente de l'enveloppe d'erreur. Le panneau du milieu compare l'erreur observée et le résidu de Bellman à la borne de contraction; le texte en dessous indique la politique gloutonne finale.

```{marimo-config}
:echo: false
:error: false
:pyproject:

requires-python = ">=3.12"
dependencies = ["matplotlib", "numpy"]
```

```{marimo} python
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np

discount = mo.ui.slider(start=0.10, stop=0.99, step=0.01, value=0.90, label="Discount γ")
persistence = mo.ui.slider(start=0.50, stop=0.99, step=0.01, value=0.85, label="Transition persistence")
reward_asymmetry = mo.ui.slider(start=-2.0, stop=2.0, step=0.1, value=0.8, label="Reward asymmetry")
initial_value = mo.ui.slider(start=-10.0, stop=10.0, step=0.5, value=0.0, label="Initial value scale")
mo.vstack([discount, persistence, reward_asymmetry, initial_value])
```

```{marimo} python
gamma_lab = discount.value
p_lab = persistence.value
reward_gap_lab = reward_asymmetry.value

transitions_lab = np.array([
    [[p_lab, 1 - p_lab], [1 - p_lab, p_lab]],
    [[1 - p_lab, p_lab], [p_lab, 1 - p_lab]],
])
rewards_lab = np.array([
    [1.0 + reward_gap_lab, 0.0],
    [0.0, 1.0 - reward_gap_lab],
])

def bellman_lab(value_lab):
    q_lab = rewards_lab + gamma_lab * np.einsum("asj,j->sa", transitions_lab, value_lab)
    return q_lab.max(axis=1), q_lab

v_star_lab = np.zeros(2)
for _ in range(1000):
    next_star_lab, _ = bellman_lab(v_star_lab)
    if np.max(np.abs(next_star_lab - v_star_lab)) < 1e-12:
        break
    v_star_lab = next_star_lab

value_lab = np.array([initial_value.value, -initial_value.value], dtype=float)
trace_lab = [value_lab.copy()]
residual_lab = []
error_lab = [np.max(np.abs(value_lab - v_star_lab))]
for _ in range(30):
    next_value_lab, _ = bellman_lab(value_lab)
    residual_lab.append(np.max(np.abs(next_value_lab - value_lab)))
    value_lab = next_value_lab
    trace_lab.append(value_lab.copy())
    error_lab.append(np.max(np.abs(value_lab - v_star_lab)))

trace_lab = np.asarray(trace_lab)
error_lab = np.asarray(error_lab)
residual_lab = np.asarray(residual_lab)
bound_lab = error_lab[0] * gamma_lab ** np.arange(error_lab.size)
_, final_q_lab = bellman_lab(value_lab)
policy_lab = final_q_lab.argmax(axis=1)
```

```{marimo} python
fig_lab, axes_lab = plt.subplots(1, 2, figsize=(10, 3.6))
axes_lab[0].plot(trace_lab[:, 0], label="state 0")
axes_lab[0].plot(trace_lab[:, 1], label="state 1")
axes_lab[0].axhline(v_star_lab[0], color="C0", linestyle=":", alpha=0.7)
axes_lab[0].axhline(v_star_lab[1], color="C1", linestyle=":", alpha=0.7)
axes_lab[0].set(xlabel="Iteration", ylabel="Value", title="Value-iteration trace")
axes_lab[0].legend()

axes_lab[1].semilogy(error_lab, label="actual error")
axes_lab[1].semilogy(bound_lab, linestyle="--", label="contraction bound")
axes_lab[1].semilogy(np.arange(1, residual_lab.size + 1), residual_lab, linestyle=":", label="Bellman residual")
axes_lab[1].set(xlabel="Iteration", ylabel="Sup norm", title="Error certificate")
axes_lab[1].legend()
fig_lab.tight_layout()

mo.vstack([
    fig_lab,
    mo.md(
        f"**Greedy policy:** state 0 → action {policy_lab[0]}, "
        f"state 1 → action {policy_lab[1]}.  "
        f"Final residual: **{residual_lab[-1]:.2e}**; "
        f"final error: **{error_lab[-1]:.2e}**."
    ),
])
```

:::{figure} _static/bellman-contraction-fallback.png
:label: fig-bellman-contraction-fallback
:class: pdf-fallback
:alt: Aperçu statique du laboratoire de contraction de Bellman

Aperçu statique de trajectoires de l'itération sur les valeurs et d'une borne de contraction géométrique. La version en ligne du livre offre des commandes pour $\gamma$, les transitions, les récompenses et la valeur initiale.
:::

(sec-policy-iteration)=
### Itération sur les politiques

L'itération sur les valeurs explore l'espace des fonctions de valeur et en tire une politique à la fin; elle atteint donc la politique de façon indirecte. L'itération sur les politiques explore directement l'espace des politiques. Bellman appelait cette idée l'approximation dans l'espace des politiques, et {cite:t}`Howard1960` en a fait l'algorithme ci-dessous. Chaque itération comporte deux étapes. La première évalue la politique courante, ce qui est le problème linéaire qu'on sait déjà résoudre; toute routine d'évaluation de politique peut donc être réutilisée comme sous-programme. La seconde améliore la politique en agissant de façon gloutonne par rapport à la valeur qu'on vient de calculer.

````{prf:algorithm} Itération sur les politiques
:label: policy-iteration-standard

**Entrée :** processus de décision markovien $(S, A, P, R, \gamma)$
**Sortie :** politique optimale $\pi^*$

1. Initialiser : $n = 0$, choisir une règle de décision arbitraire $\pi_0 \in \Pi^{MD}$
2. **répéter**
   3. (Évaluation de la politique) Obtenir $\mathbf{v}^n$ en résolvant :
   
      $$(\mathbf{I}-\gamma \mathbf{P}_{\pi_n}) \mathbf{v} = \mathbf{r}_{\pi_n}$$

   4. (Amélioration de la politique) Choisir $\pi_{n+1} = \mathrm{Greedy}(\mathbf{v}^n)$, où :

       $$\pi_{n+1} \in \arg\max_{\pi \in \Pi^{MD}}\left\{\mathbf{r}_\pi+\gamma \mathbf{P}_\pi \mathbf{v}^n\right\}$$
       
       ou, de façon équivalente, pour chaque $s$ :
       
       $$\pi_{n+1}(s) \in \arg\max_{a \in \mathcal{A}_s}\left\{r(s,a)+\gamma \sum_j p(j|s,a) \mathbf{v}^n(j)\right\}$$
       
       Poser $\pi_{n+1} = \pi_n$ si possible.

   5. Si $\pi_{n+1} = \pi_n$, **renvoyer** $\pi^* = \pi_n$

   6. $n = n + 1$
7. **jusqu'à** convergence
````

Rappelons que $\mathbf{P}_{\pi_n}$ et $\mathbf{r}_{\pi_n}$ sont construits en faisant la moyenne sur les actions que choisit la règle : $[\mathbf{P}_{\pi}]_{s,j} = \sum_a \pi(a \mid s)\, p(j \mid s,a)$ et $\mathbf{r}_\pi(s) = \sum_a \pi(a \mid s)\, r(s,a)$. La consigne de départage de l'étape 4 (« conserver $\pi_n$ s'il fait partie des maximiseurs ») compte pour le critère d'arrêt. Sans elle, l'algorithme pourrait tourner en rond entre des règles gloutonnes également bonnes et ne jamais voir deux fois la même règle.

````{prf:theorem} Terminaison finie de l'itération sur les politiques
:label: thm-pi-finite

Supposons que $S$ et chaque $\mathcal{A}_s$ soient finis et que $0 \le \gamma < 1$. Alors les valeurs produites par l'itération sur les politiques sont croissantes, $\mathbf{v}^{n+1} \ge \mathbf{v}^n$, et l'algorithme s'arrête après un nombre fini d'itérations avec une politique optimale.
````

````{prf:proof}
Puisque $\pi_{n+1}$ est gloutonne par rapport à $\mathbf{v}^n$, $\mathrm{L}_{\pi_{n+1}} \mathbf{v}^n = \Bellman \mathbf{v}^n \ge \mathrm{L}_{\pi_n}\mathbf{v}^n = \mathbf{v}^n$. Après réarrangement, $\mathbf{r}_{\pi_{n+1}} \ge (\mathbf{I} - \gamma \mathbf{P}_{\pi_{n+1}})\mathbf{v}^n$. L'inverse $(\mathbf{I} - \gamma \mathbf{P}_{\pi_{n+1}})^{-1} = \sum_t (\gamma\mathbf{P}_{\pi_{n+1}})^t$ a des coefficients positifs ou nuls; multiplier les deux membres par cet inverse préserve donc l'inégalité et donne $\mathbf{v}^{n+1} \ge \mathbf{v}^n$. L'amélioration est stricte dans au moins un état, sauf si $\Bellman \mathbf{v}^n = \mathbf{v}^n$, auquel cas $\mathbf{v}^n = v^*_\gamma$ et $\pi_n$ est optimale. Une suite strictement améliorante ne répète jamais une règle, et il existe un nombre fini de règles markoviennes déterministes; l'algorithme doit donc s'arrêter. Voir {cite:t}`Puterman1994`, section 6.4.
````

La démonstration entraîne deux remarques. D'abord, le critère d'arrêt est exact. Lorsque la même règle apparaît deux fois, $\mathbf{v}^n$ résout l'équation d'optimalité et la politique est optimale, et pas seulement $\varepsilon$-optimale. Cet argument dénombre les politiques; il est donc propre aux espaces d'états et d'actions finis. Pour des espaces continus, la convergence exige l'appareil de Newton-Kantorovitch de la section {ref}`sec-newton-kantorovich-pi`. Ensuite, la démonstration n'a jamais utilisé le fait que $\pi_{n+1}$ est gloutonne dans *chaque* état. Toute $\pi_{n+1}$ telle que $\mathrm{L}_{\pi_{n+1}}\mathbf{v}^n \ge \mathbf{v}^n$, avec inégalité stricte dans au moins un état, améliore la politique. Plus précisément, $\mathbf{v}^{n+1} \ge \mathbf{v}^n$ partout, et l'inégalité est stricte dans chaque état à partir duquel les états modifiés sont accessibles. Il suffit donc de changer l'action dans un seul état par itération, à condition que cet état soit *améliorable*, c'est-à-dire qu'une action $a$ vérifie $q_{\pi_n}(s,a) > q_{\pi_n}(s,\pi_n(s))$. L'argument de dénombrement reste valable tel quel : les valeurs ne diminuent jamais et augmentent strictement quelque part, de sorte qu'aucune politique ne se répète, et la boucle se termine après un nombre fini d'évaluations. Deux détails changent. Le critère d'arrêt doit devenir « aucun état n'est améliorable », car un pas qui ne touche qu'un état peut laisser la politique inchangée alors que d'autres états peuvent encore être améliorés. Et si l'état à mettre à jour est choisi sans vérifier qu'il est améliorable, par exemple au hasard ou selon un balayage fixe, la convergence exige une couverture : chaque état doit continuer d'être examiné, puisqu'un état qu'on n'examine jamais n'est jamais amélioré. Cela comptera plus loin, car de nombreux algorithmes d'apprentissage par renforcement contiennent une boucle d'itération sur les politiques dont l'étape d'amélioration n'est que partielle (une petite modification à chaque pas, peut-être seulement dans les états visités), et la question de savoir quels états sont visités devient alors celle de l'exploration. Un état que l'agent ne visite jamais garde l'action qu'il avait.

L'amélioration partielle converge, mais elle exige davantage d'évaluations. L'expérience ci-dessous exécute l'itération sur les politiques avec évaluation exacte sur des processus de décision markoviens aléatoires à 30 états et 4 actions, sous quatre règles d'amélioration : améliorer chaque état améliorable (itération sur les politiques standard), seulement l'état dont l'avantage $\max_a q_{\pi_n}(s,a) - q_{\pi_n}(s,\pi_n(s))$ est le plus grand, un état améliorable choisi au hasard, ou un état choisi au hasard, qu'il soit améliorable ou non. Chaque règle s'arrête sur une politique optimale.

```{code-cell} python
import numpy as np


def random_mdp(seed, n_states=30, n_actions=4):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(0.3 * np.ones(n_states), size=(n_states, n_actions))
    return P, rng.normal(size=(n_states, n_actions))


def evaluate(P, r, gamma, pi):
    idx = np.arange(P.shape[0])
    return np.linalg.solve(np.eye(P.shape[0]) - gamma * P[idx, pi], r[idx, pi])


def policy_iteration(P, r, gamma, rule, seed=0, max_iter=100_000):
    rng = np.random.default_rng(seed)
    S = P.shape[0]
    pi = np.zeros(S, dtype=int)
    for n in range(max_iter):
        v = evaluate(P, r, gamma, pi)
        q = r + gamma * P @ v
        advantage = q.max(axis=1) - q[np.arange(S), pi]
        improvable = np.flatnonzero(advantage > 1e-12)
        if improvable.size == 0:  # no state can be improved: pi is optimal
            return n, pi
        if rule == "all":  # standard policy iteration
            pi = np.where(advantage > 1e-12, q.argmax(axis=1), pi)
        elif rule == "best state":  # the state with the largest advantage
            s = improvable[advantage[improvable].argmax()]
            pi[s] = q[s].argmax()
        elif rule == "random improvable":  # one improvable state at random
            s = rng.choice(improvable)
            pi[s] = q[s].argmax()
        elif rule == "random state":  # one state at random, improvable or not
            s = rng.integers(S)
            if advantage[s] > 1e-12:
                pi[s] = q[s].argmax()
    return max_iter, pi


gamma = 0.95
rows = [f"{'improvement rule':>18} | {'evaluations (mean of 5 MDPs)':>28} | optimal"]
for rule in ("all", "best state", "random improvable", "random state"):
    counts, all_optimal = [], True
    for seed in range(5):
        P, r = random_mdp(seed)
        n, pi = policy_iteration(P, r, gamma, rule, seed)
        _, pi_ref = policy_iteration(P, r, gamma, "all")
        all_optimal &= np.allclose(evaluate(P, r, gamma, pi), evaluate(P, r, gamma, pi_ref))
        counts.append(n)
    rows.append(f"{rule:>18} | {np.mean(counts):>28.1f} | {all_optimal}")
print("\n".join(rows))
```

L'itération sur les politiques standard demande environ 3 évaluations, alors qu'un seul état améliorable par itération en demande de 20 à 30. Choisir l'état à l'aveugle en coûte environ 200, car la plupart des tirages tombent sur des états déjà gloutons. La garantie elle-même n'est qu'une terminaison finie : l'argument de dénombrement borne le nombre d'évaluations par le nombre de politiques, qui croît exponentiellement avec le nombre d'états, et il existe des processus de décision markoviens sur lesquels des règles à un seul état mal choisies exigent effectivement un nombre exponentiel de pas.

En pratique, l'itération sur les politiques converge en très peu d'itérations, souvent une poignée, même pour de grands problèmes. L'itération sur les politiques est la méthode de Newton appliquée à l'équation de Bellman, un lien établi par {cite:t}`PutermanBrumelle1979` et développé à la section {ref}`sec-newton-kantorovich-pi`. Chaque itération coûte toutefois plus cher qu'un pas d'itération sur les valeurs. Les deux algorithmes paient environ $|\mathcal{S}|^2|\mathcal{A}|$ opérations pour une étape gloutonne, mais l'itération sur les politiques y ajoute une évaluation directe de la politique, une résolution linéaire d'ordre $|\mathcal{S}|^3$.

### Itération sur les politiques modifiée

L'évaluation d'une politique peut aussi se faire par approximations successives, et rien n'oblige à mener cette boucle interne jusqu'à convergence. L'arrêter après quelques balayages donne l'**itération sur les politiques modifiée** {cite}`PutermanShin1978`. Une façon de voir ce que fait cette méthode passe par la série de Neumann. Écrivons l'évaluation exacte de la règle gloutonne $\pi = \mathrm{Greedy}(\mathbf{v})$ comme une correction de l'estimation courante :

$$
\mathbf{v}^{\pi}_\gamma = \mathbf{v} + (\mathbf{I} - \gamma\mathbf{P}_\pi)^{-1}(\Bellman \mathbf{v} - \mathbf{v}) = \mathbf{v} + \sum_{k=0}^{\infty} (\gamma \mathbf{P}_\pi)^k\, (\Bellman \mathbf{v} - \mathbf{v}).
$$

Le pas est le résidu de Bellman $\Bellman\mathbf{v} - \mathbf{v}$, propagé à travers l'inverse. L'itération sur les politiques modifiée tronque la série après $m+1$ termes :

$$
\mathbf{v}^{n+1} = \mathbf{v}^n + \sum_{k=0}^{m} (\gamma \mathbf{P}_{\pi_{n+1}})^k\, (\Bellman \mathbf{v}^n - \mathbf{v}^n) = (\mathrm{L}_{\pi_{n+1}})^{m+1}\, \mathbf{v}^n,
$$

où la seconde forme, $m+1$ balayages d'évaluation de politique à partir de $\mathbf{v}^n$, est celle qu'on implémente. L'entier $m \ge 0$ est un hyperparamètre qui fait la transition entre les deux algorithmes vus jusqu'ici. Avec $m = 0$, la mise à jour est $\mathbf{v}^{n+1} = \mathrm{L}_{\pi_{n+1}}\mathbf{v}^n = \Bellman\mathbf{v}^n$, c'est-à-dire l'itération sur les valeurs. Avec $m = \infty$, c'est l'évaluation exacte, c'est-à-dire l'itération sur les politiques. Entre les deux, chaque itération coûte $m+1$ produits matrice-vecteur et aucune résolution linéaire.

````{prf:algorithm} Itération sur les politiques modifiée
:label: alg-modified-policy-iteration

**Entrée :** processus de décision markovien $(S, A, P, R, \gamma)$, nombre de balayages d'évaluation $m \ge 0$, tolérance $\varepsilon > 0$  
**Sortie :** une politique $\varepsilon$-optimale

1. Initialiser $\mathbf{v}^0$ avec $\Bellman\mathbf{v}^0 \ge \mathbf{v}^0$, par exemple $\mathbf{v}^0 = \frac{\min_{s,a} r(s,a)}{1-\gamma}\mathbf{1}$; poser $n = 0$
2. **répéter**
    1. (Amélioration) $\pi_{n+1} \leftarrow \mathrm{Greedy}(\mathbf{v}^n)$, de sorte que $\mathrm{L}_{\pi_{n+1}}\mathbf{v}^n = \Bellman\mathbf{v}^n$
    2. Si $\|\Bellman\mathbf{v}^n - \mathbf{v}^n\|_\infty < \frac{\varepsilon(1-\gamma)}{2\gamma}$, **renvoyer** $\mathrm{Greedy}(\Bellman\mathbf{v}^n)$
    3. (Évaluation partielle) $\mathbf{u} \leftarrow \Bellman\mathbf{v}^n$; répéter $m$ fois : $\mathbf{u} \leftarrow \mathbf{r}_{\pi_{n+1}} + \gamma\mathbf{P}_{\pi_{n+1}}\mathbf{u}$
    4. $\mathbf{v}^{n+1} \leftarrow \mathbf{u}$; $n \leftarrow n+1$
````

Le critère d'arrêt est celui de l'itération sur les valeurs, et il donne la même garantie, à condition que la politique renvoyée soit gloutonne par rapport à $\Bellman\mathbf{v}^n$ (une étape gloutonne de plus), exactement comme l'itération sur les valeurs renvoie la politique gloutonne par rapport à son dernier itéré. À partir d'un $\mathbf{v}^0$ tel que $\Bellman\mathbf{v}^0 \ge \mathbf{v}^0$, les itérés croissent de façon monotone vers $v^*_\gamma$ et la politique renvoyée est $\varepsilon$-optimale ({cite:t}`Puterman1994`, section 6.5). Le nombre de balayages $m$ peut aussi varier d'une itération à l'autre.

Pourquoi une évaluation tronquée devrait-elle fonctionner? Dans la méthode de Newton, l'inverse de la hessienne (ou l'inverse de la jacobienne) remet à l'échelle le pas de gradient, comme un calendrier de pas adapté à la courbure locale. Ici, $(\mathbf{I} - \gamma\mathbf{P}_\pi)^{-1}$ joue ce rôle pour le résidu de Bellman, et l'itération sur les valeurs le remplace par l'identité. Tronquer la série de Neumann le remplace par une approximation, de la même façon que les méthodes de quasi-Newton et de Newton inexacte utilisent des hessiennes approchées ou des résolutions linéaires approchées. La section {ref}`sec-newton-kantorovich-pi` le précise.

(exact-scheduling-mdp-for-inference-serving)=
## Processus de décision markovien exact pour l'ordonnancement d'un service d'inférence

Comment ramener un système de grande taille, décrit à l'échelle des requêtes, à
un processus de décision markovien fini, et quelles distinctions prédictives
disparaissent dans cette réduction?

Jusqu'ici, les exemples d'inférence traitaient la règle d'ordonnancement comme
fixée et l'horloge du GPU comme l'action. Dans une autre interface de décision,
l'horloge est fixée et l'action est la phase qui reçoit la prochaine unité de
service. Le préremplissage traite une invite en attente et fait passer sa
requête dans le lot de décodage. Le décodage fait avancer chaque requête de ce
lot d'un pas de génération de jetons. Servir l'une ou l'autre phase retarde
l'autre.

Le compromis vient du traitement par lots. Un pas de décodage coûte à peu près
le même temps et la même puissance, qu'il fasse avancer une requête ou six, car
la génération d'un jeton est limitée par la lecture des poids du modèle en
mémoire, et non par l'arithmétique. Sur l'exécution mesurée sur NVIDIA L4
utilisée ci-dessous, un seul flux de décodage produit un jeton toutes les
57.0 ms; avec quatre flux concurrents, chaque flux produit encore un jeton
toutes les 59.5 ms, et avec huit, toutes les 60.8 ms. Six flux terminent donc
les requêtes presque six fois plus vite qu'un seul. Un ordonnanceur qui décode
dès qu'une requête est prête garde le lot petit et gaspille cette capacité. Un
ordonnanceur qui continue de préremplir pour faire grossir le lot laisse en
attente les requêtes qui s'y trouvent déjà. La résolution du processus de
décision markovien donne le point de bascule.

Un état markovien exact à l'échelle des requêtes contiendrait chaque longueur
d'invite, chaque nombre de jetons générés, chaque allocation de cache et chaque
temps d'attente. Pour le calcul, ces quantités sont agrégées dans l'état fini

$$
s=(p,d)\in\{0,\ldots,6\}^2,
$$

où $p$ compte les tâches de préremplissage en attente et $d$ les tâches de
décodage actives. Les actions sont

$$
\mathcal A=\{\text{préremplissage},\text{décodage},\text{inaction}\}.
$$

Une action est masquée lorsque sa phase est vide. Le préremplissage est aussi
masqué à $d=6$, ce qui représente la limite du cache dans cette abstraction. La
politique entière tient donc dans un seul tableau $7\times 7$, que la figure
ci-dessous représente en entier.

Le temps avance par périodes de décision de 0.1 seconde. Une arrivée de
Bernoulli de probabilité $\lambda$ par période est calibrée à partir de la
version normalisée en charge de la même trace Azure de cinq minutes que celle
du chapitre de modélisation. Les débits et les puissances des phases
proviennent d'une exécution mesurée sur NVIDIA L4 de Qwen/Qwen2.5-7B-Instruct
servi par vLLM 0.28.0, au niveau d'horloge demandé de 1 125 MHz; sa médiane
d'horloge graphique réalisée, équilibrée selon la taille des lots, était de
939.375 MHz.

- **Préremplissage.** Le débit de préremplissage mesuré et la longueur moyenne
  des invites de la trace donnent la probabilité $q_{\mathrm p}$ qu'une action
  de préremplissage termine une invite. Une invite terminée passe de $p$ à $d$.
- **Décodage.** Le temps par jeton de sortie d'un flux seul et la longueur
  moyenne des sorties de la trace donnent la probabilité $q_1$ qu'un flux de
  décodage seul termine sa réponse en une période. Avec $d$ flux actifs, chaque
  flux se termine indépendamment avec probabilité $1-(1-q_1)^{\eta_d}$, où
  $\eta_d=\mathrm{TPOT}(1)/\mathrm{TPOT}(d)$ est le ralentissement mesuré d'un
  flux dans un lot de taille $d$, interpolé entre les tailles de lot mesurées 1,
  4 et 8. Le nombre de terminaisons suit une loi binomiale; sa moyenne croît
  donc presque linéairement en $d$.
- **Arrivées et rejets.** Après le service, une arrivée rejoint la file de
  préremplissage. Une arrivée qui trouve $p=6$ est rejetée.

Ces choix définissent une matrice de transition complète $P_{ss'}^u$ sur 49 états.

Le coût à un pas compte les requêtes dans le système, les arrivées rejetées et
l'énergie :

$$
c(s,u)=p+d
+100\,\mathbb E[N_{\mathrm{drop}}\mid s,u]
+0.1\frac{E(u)}{E_{\max}}.
$$

Selon la loi de Little, la moyenne à long terme de $p+d$ divisée par le taux
d'arrivée est le temps moyen qu'une requête passe dans le système; le premier
terme est donc un coût de latence. Une requête rejetée coûte autant que garder
une requête pendant 100 périodes, soit 10 secondes. C'est environ deux fois le
temps moyen qu'une requête passe dans le système sous la politique optimale;
rejeter une requête n'est donc jamais une façon bon marché de raccourcir la
file. Pour le préremplissage, le décodage et l'inaction respectivement, le
profil mesuré donne $E(u)=(6.427,6.274,2.339)$ joules par période de décision.
Chaque valeur est le résumé de puissance mesuré pour la phase au niveau demandé
de 1 125 MHz, multiplié par 0.1 seconde, et non une mesure directe de l'énergie
à l'échelle des requêtes. Seul le rapport $E(u)/E_{\max}$ entre dans le coût
d'étape.

Avec $\gamma=0.99$, l'itération sur les valeurs, en minimisation de coût, applique

$$
\begin{aligned}
Q_n(s,u)&=c(s,u)+\gamma\sum_{s'}P_{ss'}^uV_n(s'),\\
V_{n+1}(s)&=\min_{u\in\mathcal A(s)}Q_n(s,u).
\end{aligned}
$$

L'itération s'arrête lorsque $\lVert V_{n+1}-V_n\rVert_\infty<10^{-10}$. Le
résidu de Bellman final est vérifié de façon indépendante et doit être
inférieur à $10^{-8}$.

La figure compare la politique de l'itération sur les valeurs à deux règles
transparentes. La **priorité au préremplissage** préremplit chaque fois qu'une
invite attend et que le lot a de la place. La **priorité au décodage** décode
chaque fois qu'une tâche de décodage est active. Chaque rejeu applique une
politique pendant 60 secondes à partir d'un système vide. Les trois rejeux
utilisent les mêmes nombres aléatoires pour les arrivées et le service; ils
voient donc les mêmes requêtes, et toute différence entre eux provient des
décisions.

```{code-cell} python
:tags: [remove-input]
:label: fig-inference-scheduling-dp
:caption: Itération sur les valeurs exacte sur le processus de décision markovien d'ordonnancement de l'inférence calibré sur les mesures du L4, comparée à la priorité au préremplissage et à la priorité au décodage. Chaque politique est représentée sur les 49 états. Le panneau voisin indique le coût actualisé exact de la politique à partir du système vide et son plus petit écart $V^\pi(s)-V^*(s)$ sur l'ensemble des états. Le rejeu échantillonne les transitions de file dans le modèle réduit, à l'horloge fixe du profil mesuré; ces trajectoires sont simulées et ne sont pas des observations directes de vLLM.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="scheduling",
)))
```

:::{figure} _static/inference_serving/scheduling.svg
:label: fig-inference-scheduling-dp-fallback
:class: pdf-fallback
:alt: Tableaux des politiques de l'itération sur les valeurs exacte, de la priorité au préremplissage et de la priorité au décodage dans le processus de décision markovien d'ordonnancement à 49 états.

La politique exacte, la priorité au préremplissage et la priorité au décodage
sur les 49 états, avec le coût actualisé exact de chaque politique à partir du
système vide. La version interactive ajoute des commandes pour lire, parcourir
pas à pas et faire défiler un épisode échantillonné dans le modèle de
transition réduit sous chaque politique. La trajectoire de la file est simulée
plutôt qu'observée directement dans vLLM.
:::

La politique optimale est une courbe de bascule. Lorsqu'une ou deux invites
attendent, elle préremplit jusqu'à ce que quatre requêtes soient en décodage,
puis elle décode. Lorsque trois invites ou plus attendent, elle remplit le lot
jusqu'à cinq avant de décoder. Elle décode chaque fois qu'aucune invite
n'attend et ne reste inactive que dans l'état vide. La priorité au décodage est
l'extrême opposé. Elle ne préremplit que lorsque le lot est vide, de sorte
qu'elle n'exécute jamais plus d'un flux, et un flux seul termine moins de
requêtes par période (0.059) qu'il n'en arrive (0.071). Sa file d'invites
se remplit donc, et des arrivées sont rejetées.

Comment le lecteur peut-il vérifier que l'itération sur les valeurs a renvoyé
la politique optimale plutôt qu'une politique plausible? Pour une politique
fixée $\pi$, l'équation d'évaluation de politique
$V^\pi=c_\pi+\gamma P_\pi V^\pi$ est un système linéaire à 49 inconnues, et sa
résolution donne le coût actualisé exact de $\pi$ dans chaque état. Une
politique optimale satisfait $V^*(s)\le V^\pi(s)$ dans chaque état pour toute
politique stationnaire $\pi$; chaque heuristique doit donc perdre partout, et
pas seulement en moyenne.

```{code-cell} python
:tags: [remove-input]

import json

import pandas as pd

scheduling = json.loads(
    Path("artifacts/inference_serving/textbook_results.json").read_text()
)["scheduling"]
names = {
    "exact_dp": "exact DP",
    "prefill_priority": "prefill priority",
    "decode_priority": "decode priority",
}
evaluations = {
    "exact_dp": scheduling["exact_policy_evaluation"],
    **scheduling["heuristic_evaluation"],
}
pd.DataFrame(
    {
        names[name]: {
            "cost from empty, exact": f"{check['cost_from_empty']:.2f}",
            "smallest gap V^π − V* over 49 states": f"{max(check['minimum_gap'], 0.0):.2f}",
            "largest gap V^π − V* over 49 states": f"{max(check['maximum_gap'], 0.0):.2f}",
            "states where the action differs": check["states_where_action_differs"],
            "mean time in system (s), simulated": f"{evaluations[name]['mean_waiting_time_s']:.2f}",
            "dropped arrivals per episode, simulated": f"{evaluations[name]['mean_dropped_arrivals']:.2f}",
        }
        for name, check in (
            (name, scheduling["policy_checks"][name]) for name in names
        )
    }
).rename_axis("quantity")
```

Les lignes exactes proviennent des résolutions linéaires. Les lignes simulées
font la moyenne de 2 000 épisodes à graine fixée de 300 périodes à partir du
système vide. L'itération sur les valeurs a demandé 2 435 balayages sur ce
problème à forte actualisation. Son résidu de Bellman recalculé de façon
indépendante vaut $9.851\times10^{-11}$, et l'évaluation de sa politique par
une résolution linéaire reproduit sa fonction de valeur à $10^{-8}$ près dans
chaque état. La priorité au préremplissage est moins bonne que la politique
optimale d'au moins 3.02 dans chaque état et s'en écarte dans 8 états, tous
situés sur la courbe de bascule, où la priorité au préremplissage continue de
remplir le lot jusqu'à six. La priorité au décodage est moins bonne d'au moins
131.82 dans chaque état. Dans les épisodes simulés, elle rejette 4.35 arrivées
par épisode et double à peu près le temps qu'une requête passe dans le système.
La priorité au préremplissage rejette un peu moins d'arrivées que la politique
optimale, 0.14 contre 0.20 par épisode. La politique optimale minimise le coût
actualisé total, et non chacun de ses termes, et accepte quelques rejets de
plus en échange d'un temps d'attente plus court dans le lot.

Le processus de décision markovien d'ordonnancement réduit a les paramètres de
transition et le certificat d'itération sur les valeurs suivants.

```{code-cell} python
:tags: [remove-input]

dp_row = pd.read_csv("artifacts/inference_serving/metrics_dp.csv").iloc[0]
pd.DataFrame(
    {
        "value": [
            f"{dp_row['bellman_residual']:.3e}",
            f"{int(dp_row['iterations']):,}",
            dp_row["profile_status"],
            f"{dp_row['arrival_probability']:.6f}",
            f"{dp_row['prefill_completion_probability']:.6f}",
            f"{dp_row['decode_completion_probability']:.6f}",
            f"{dp_row['decode_batch_efficiency_at_cap']:.3f}",
            f"{dp_row['prefill_energy_j']:.3f}",
            f"{dp_row['decode_energy_j']:.3f}",
            f"{dp_row['idle_energy_j']:.3f}",
        ]
    },
    index=[
        "Bellman residual",
        "value-iteration sweeps",
        "profile provenance",
        "arrival probability λ",
        "prefill completion probability q_p",
        "single-stream decode completion probability q_1",
        "per-stream decode speed at d = 6, relative to d = 1",
        "prefill energy (J)",
        "decode energy (J)",
        "idle energy (J)",
    ],
).rename_axis("quantity")
```

{download}`Télécharger le certificat de programmation dynamique (CSV) <artifacts/inference_serving/metrics_dp.csv>`

Le certificat porte sur la matrice de transition à 49 états fournie. Le
résultat est exact pour le processus de décision markovien fini énoncé, et non
pour vLLM. L'agrégation des âges et des longueurs des requêtes efface des
distinctions qui peuvent influer sur l'attente due au blocage en tête de file
et sur la libération du cache; toutes les invites et toutes les réponses sont
traitées comme ayant la longueur moyenne de la trace. Le noyau de transition
calibré est stationnaire, l'horloge est fixe, la température est absente de
l'état, le lot est plafonné à six, et l'ensemble d'actions exclut les lots
mixtes de préremplissage et de décodage. Les mesures tirées d'un seul
déploiement sur L4 calibrent les débits des phases, le ralentissement dû aux
lots et les puissances; elles n'établissent pas les performances
d'ordonnancement sur l'ensemble des systèmes L4. Le rejeu à l'échelle des
requêtes des chapitres précédents reste un outil d'audit du modèle et ne fait
pas partie de la preuve d'optimalité.

:::{dropdown} Examiner le processus de décision markovien d'ordonnancement, l'itération sur les valeurs et la vérification de la politique
```{literalinclude} code/inference_control.py
:language: python
:start-at: def make_scheduling_mdp
:end-before: def _scheduling_clock_calibration
:linenos:
```

```{literalinclude} code/inference_control.py
:language: python
:start-at: def solve_scheduling_mdp
:end-before: def _sample_next_state
:linenos:
```

{download}`Télécharger l'implémentation complète de la commande de l'inférence <code/inference_control.py>`
:::

(sec-newton-kantorovich-pi)=
## Newton-Kantorovitch et itération sur les politiques

L'itération sur les valeurs est la méthode des approximations successives appliquée à l'équation d'optimalité de Bellman. Cette section développe l'autre outil général pour les équations d'opérateurs, la méthode de Newton-Kantorovitch, et montre que l'appliquer à la même équation donne l'algorithme d'itération sur les politiques de la section {ref}`sec-policy-iteration`. Cela explique pourquoi l'itération sur les politiques demande si peu d'itérations, et pourquoi l'itération sur les politiques modifiée, qui ne résout le système de Newton qu'approximativement, fonctionne quand même.

### Méthode de Newton-Kantorovitch

La méthode de Newton-Kantorovitch généralise la méthode de Newton des espaces vectoriels de dimension finie aux espaces de fonctions de dimension infinie : au lieu d'itérer dans l'espace des vecteurs, on itère dans l'espace des fonctions. 

On écrit souvent la méthode de Newton sous la forme de la mise à jour familière :

$$
x_{k+1} = x_k - [DF(x_k)]^{-1} F(x_k),
$$
ce qui donne l'impression que la méthode consiste à « prendre une dérivée et l'inverser ». Ce que fait la méthode de Newton, en dimension finie comme en dimension infinie, c'est une **linéarisation**.

À chaque pas, l'idée est de remplacer l'opérateur non linéaire $F:X \to Y$ par un modèle de substitution local de la forme

$$
F(x+h) \approx F(x) + Lh,
$$
où $L$ est une application linéaire qui décrit comment de petites perturbations de l'entrée se propagent en variations de la sortie. Il s'agit d'un développement de type Taylor dans les espaces de Banach : c'est la dérivée qui définit un tel opérateur linéaire.

Pour trouver une racine de $F$, on impose que le modèle de substitution s'annule au prochain itéré :

$$
0 = F(x+h) \approx F(x) + Lh.
$$
La résolution de cette équation linéaire donne l'incrément $h$. En dimension finie, $L$ est la matrice jacobienne; dans les espaces de Banach, ce doit être la **dérivée de Fréchet**.

Mais qu'est-ce au juste qu'une dérivée de Fréchet en dimension infinie? Pour le comprendre, il faut généraliser la notion de dérivée du calcul différentiel en dimension finie. Dans les espaces de dimension infinie, il existe plusieurs notions de différentiabilité, chacune avec sa force et ses exigences propres :

**1. Dérivée de Gâteaux (directionnelle)**

On dit que la dérivée de Gâteaux de $F$ en $x$ dans une direction donnée $h$ est :

$$
F'(x; h) = \lim_{t \to 0} \frac{F(x + th) - F(x)}{t}
$$

Cette quantité mesure comment la fonction $F$ varie le long de la demi-droite $x + th$. Même si cette limite existe séparément pour chaque direction $h$, rien ne garantit que la dérivée soit linéaire en $h$. La dérivée de Gâteaux peut donc exister dans toutes les directions sans pour autant fournir une bonne approximation linéaire.

**2. Dérivée directionnelle d'Hadamard**

Au lieu de considérer une seule direction de perturbation, on considère maintenant un faisceau de perturbations autour de $h$. On se demande comment la fonction varie lorsqu'on approche la direction cible à partir de directions voisines. On dit que $F$ admet une dérivée directionnelle d'Hadamard si :

$$
F'(x; h) = \lim_{\substack{t \downarrow 0 \\ h' \to h}} \frac{F(x + t h') - F(x)}{t}
$$

C'est une condition plus forte que la différentiabilité au sens de Gâteaux, car elle exige que la limite soit uniforme sur les directions voisines. Elle ne garantit toutefois toujours pas la linéarité en $h$.

**3. Dérivée de Fréchet**

C'est la plus forte des trois notions : $F$ est différentiable au sens de Fréchet en $x$ s'il existe un opérateur linéaire borné $L$ tel que :

$$
\lim_{h \to 0} \frac{\|F(x + h) - F(x) - Lh\|}{\|h\|} = 0
$$

Contrairement aux dérivées de Gâteaux et d'Hadamard, la dérivée de Fréchet exige un opérateur linéaire $L$ qui approche $F$ uniformément. Ses propriétés :

- $L$ doit être **linéaire** en $h$ (contrairement aux dérivées directionnelles ci-dessus)
- L'erreur d'approximation est un $o(\|h\|)$, uniforme dans toutes les directions
- C'est la « vraie » dérivée : elle généralise la matrice jacobienne à la dimension infinie
- Notation : $L = F'(x)$ ou $DF(x)$

**Relation :**

$$
\text{différentiable au sens de Fréchet} \Rightarrow \text{dérivable directionnellement au sens d'Hadamard} \Rightarrow \text{dérivable directionnellement au sens de Gâteaux}
$$

Dans le contexte de la méthode de Newton-Kantorovitch, on travaille avec un opérateur $F: X \to Y$, où $X$ et $Y$ sont tous deux des espaces de Banach. La dérivée de Fréchet $F'(x)$ est la meilleure approximation linéaire de $F$ près de $x$, et c'est l'opérateur $L$ utilisé dans la linéarisation $F(x+h) \approx F(x) + F'(x)h$.

Ces subtilités techniques mises à part, la méthode de Newton-Kantorovitch a la même structure que la méthode de Newton en dimension finie :

1. **Linéariser l'opérateur** :
   Étant donné une approximation $ x_n $, on considère la dérivée de Fréchet de $ F $, notée $ F'(x_n) $. Cette dérivée est un opérateur linéaire qui fournit une approximation locale de $ F $ près de $ x_n $.

2. **Poser le pas de Newton** :
   La méthode résout ensuite l'équation linéarisée pour obtenir une correction $ h_n $ :

   $$
   F'(x_n) h_n = -F(x_n).
   $$
   Cette équation représente un système linéaire dans lequel $ h_n $ est choisi de sorte que l'opérateur linéarisé $ F(x_n) + F'(x_n)h_n $ soit nul.

3. **Mettre à jour la solution** :
   La nouvelle approximation $ x_{n+1} $ est alors donnée par :

   $$
   x_{n+1} = x_n + h_n.
   $$
   Ce pas de correction raffine $ x_n $ et le rapproche de la vraie solution.

4. **Répéter jusqu'à convergence** :
   On répète les étapes de linéarisation et de mise à jour jusqu'à ce que la solution $ x_n $ converge à la tolérance voulue, ce qu'on peut vérifier en contrôlant que $ \|F(x_n)\| $ est suffisamment petit, ou en surveillant la norme $ \|x_{n+1} - x_n\| $.

La convergence de la méthode de Newton-Kantorovitch ne repose pas sur le fait que $ F $ soit une contraction sur tout le domaine (contrairement à la méthode des approximations successives). Les propriétés de convergence de la méthode de Newton-Kantorovitch sont les suivantes :

1. **Convergence locale** : sous des conditions peu contraignantes (par exemple, $F$ différentiable au sens de Fréchet et $F'(x)$ inversible près de la solution), la méthode converge localement. Autrement dit, si l'estimation initiale est suffisamment proche de la vraie solution, la méthode converge.

2. **Convergence globale** : la convergence globale n'est pas garantie en général. Toutefois, sous des conditions plus fortes (par exemple, $F$ analytique et satisfaisant certaines bornes), la méthode peut converger globalement.

3. **Vitesse de convergence** : lorsque la méthode converge, elle présente en général une convergence quadratique. Autrement dit, l'erreur à chaque pas est proportionnelle au carré de l'erreur au pas précédent :

   $$
   \|x_{n+1} - x^*\| \leq C\|x_n - x^*\|^2
   $$

   où $x^*$ est la vraie solution et $C$ une constante. La convergence quadratique est beaucoup plus rapide que la convergence linéaire de la méthode des approximations successives.

### Newton-Kantorovitch appliquée à l'optimalité de Bellman

Appliquons maintenant le cadre de Newton-Kantorovitch à l'équation d'optimalité de Bellman. Soit

$$
(\Bellman v)(s) = \max_{a \in A(s)} \left\{ r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) v(s') \right\}.
$$

Le problème consiste à trouver $v$ tel que $\Bellman v = v$ ou, de façon équivalente, $\mathrm{B}(v) := \Bellman v - v = 0$. L'opérateur $\Bellman$ est affine par morceaux, donc non différentiable globalement, mais il est directionnellement dérivable partout au sens d'Hadamard, et différentiable au sens de Fréchet aux points où le maximiseur est unique.

On considère trois points de vue complémentaires pour comprendre et calculer sa dérivée.

#### Premier point de vue : maximum d'applications affines

Sous forme tabulaire, pour des espaces d'états et d'actions finis, l'opérateur de Bellman peut s'écrire comme un maximum point par point d'applications affines :

$$
(\Bellman v)(s) = \max_{a \in A(s)} \left\{ r(s,a) + \gamma (P_a v)(s) \right\},
$$
où $P_a \in \mathbb{R}^{|S| \times |S|}$ est la matrice de transition associée à l'action $a$. Chaque $Q_a v := r^a + \gamma P_a v$ est affine en $v$. L'opérateur $\Bellman$ calcule donc, dans chaque état, l'enveloppe supérieure d'un ensemble fini de fonctions affines.

En tout $v$, on appelle **ensemble actif** dans l'état $s$ l'ensemble

$$
\mathcal{A}^*(s; v) := \arg\max_{a \in A(s)} (Q_a v)(s).
$$

La dérivée directionnelle d'Hadamard existe alors et est donnée par

$$
(\Bellman '(v; h))(s) = \max_{a \in \mathcal{A}^*(s; v)} \gamma (P_a h)(s).
$$

Si l'ensemble actif est un singleton, cette expression devient linéaire en $h$, et $\Bellman$ est différentiable au sens de Fréchet en $v$, avec

$$
\Bellman'(v) = \gamma P_{\pi_v},
$$

où $\pi_v(s) := a^*(s)$ est la politique gloutonne en $v$. 
<!-- En présence d'égalités, la dérivée devient multivoque : le sous-différentiel de Clarke est formé des matrices stochastiques dont les lignes sont des combinaisons convexes des $\gamma P_a$ pour $a \in \mathcal{A}^*(s; v)$. -->

#### Deuxième point de vue : théorème de l'enveloppe

Considérons maintenant une fonction de valeur approchée par une combinaison linéaire de fonctions de base :

$$
v_c(s) = \sum_j c_j \phi_j(s).
$$

En un nœud $s_i$, on définit la maximisation paramétrique

$$
v_i(c) := (\Bellman v_c)(s_i) = \max_{a \in A(s_i)} \left\{ r(s_i,a) + \gamma \sum_j c_j \mathbb{E}_{s' \mid s_i, a}[\phi_j(s')] \right\}.
$$

On définit

$$
F_i(a, c) := r(s_i,a) + \gamma \sum_j c_j \mathbb{E}_{s' \mid s_i, a}[\phi_j(s')],
$$

de sorte que $v_i(c) = \max_a F_i(a, c)$. Puisque $F_i$ est linéaire en $c$, on peut appliquer le **théorème de l'enveloppe** (théorème de Danskin) : si l'optimiseur $a_i^*(c)$ est unique ou choisi de façon mesurable, alors

$$
\frac{\partial v_i}{\partial c_j}(c) = \gamma \mathbb{E}_{s' \mid s_i, a_i^*(c)}[\phi_j(s')].
$$

Il n'est pas nécessaire de dériver l'optimiseur $a_i^*(c)$ lui-même. Le résultat s'étend au cas du sous-différentiel lorsque des égalités se produisent; la jacobienne devient alors multivoque.

Ce résultat est utile pour résoudre l'équation de collocation $\Phi c = v(c)$. La méthode de Newton exige la jacobienne $v'(c)$, et cette expression permet de la calculer sans faire intervenir aucune dérivée de l'action optimale.

#### Troisième point de vue : le théorème des fonctions implicites

Le troisième point de vue applique le théorème des fonctions implicites pour comprendre quand l'opérateur de Bellman est différentiable malgré la présence d'un opérateur max. Le problème de maximisation définit une relation implicite entre la fonction de valeur et l'action optimale, et le théorème des fonctions implicites indique quand cette relation est suffisamment lisse pour qu'on puisse dériver à travers elle.

L'opérateur de Bellman est défini par

$$
(\Bellman v)(s) = \max_{a} \left\{ r(s,a) + \gamma \sum_j p(j \mid s,a) v(j) \right\}.
$$

La difficulté est que l'opérateur max encode une sélection discrète : celle de l'action qui atteint le maximum. Pour appliquer le théorème des fonctions implicites, on reformule ce problème comme suit. Pour chaque action $a$, on définit la **fonction de valeur d'action** :

$$
Q_a(v, s) := r(s,a) + \gamma \sum_j p(j \mid s,a) v(j).
$$

L'action optimale en $v$ satisfait la **condition d'optimalité** :

$$
Q_{a^*(s)}(v, s) \geq Q_a(v, s) \quad \text{pour tout } a.
$$

Supposons maintenant qu'en un $v$ particulier, l'action $a^*(s)$ soit un **maximiseur local strict**, au sens où il existe $\delta > 0$ tel que

$$
Q_{a^*(s)}(v, s) > Q_a(v, s) + \delta \quad \text{pour tout } a \neq a^*(s).
$$

Cette inégalité stricte est la condition de régularité nécessaire au théorème des fonctions implicites. Elle garantit que l'action optimale est unique en $v$ et le reste dans un voisinage de $v$.

Pour le voir, considérons une perturbation quelconque $v + h$ avec $\|h\|$ petit. Puisque $Q_a$ est linéaire en $v$, on a :

$$
Q_a(v+h, s) = Q_a(v, s) + \gamma \sum_j p(j \mid s,a) h(j).
$$

Le terme de perturbation est borné : $|\gamma \sum_j p(j \mid s,a) h(j)| \leq \gamma \|h\|$. Par conséquent, pour $\|h\| < \delta/\gamma$, l'écart strict garantit que

$$
Q_{a^*(s)}(v+h, s) > Q_a(v+h, s) \quad \text{pour tout } a \neq a^*(s).
$$

Ainsi, $a^*(s)$ reste l'unique maximiseur dans tout le voisinage $\{v + h : \|h\| < \delta/\gamma\}$.

Le théorème des fonctions implicites s'applique alors : dans ce voisinage, l'application $v \mapsto a^*(s; v)$ est **constante** (et donc lisse) et prend la valeur $a^*(s)$. On peut ainsi écrire

$$
(\Bellman v)(s) = Q_{a^*(s)}(v, s) = r(s,a^*(s)) + \gamma \sum_j p(j \mid s,a^*(s)) v(j)
$$
comme une formule explicite valable dans tout le voisinage. Puisque $Q_{a^*(s)}(\cdot, s)$ est une fonction affine (donc lisse) de $v$, on peut la dériver :

$$
\frac{d}{dv} (\Bellman v)(s) = \gamma P_{a^*(s)}.
$$

Plus précisément, pour toute perturbation $h$ :

$$
(\Bellman (v+h))(s) = (\Bellman v)(s) + \gamma \sum_j p(j \mid s,a^*(s)) h(j) + o(\|h\|).
$$

C'est la dérivée de Fréchet :

$$
\Bellman'(v) = \gamma P_{\pi_v},
$$

où $\pi_v(s) = a^*(s)$ est la politique gloutonne.

**Le rôle du théorème des fonctions implicites** : il garantit que, lorsque le maximiseur est unique avec un écart strict (la condition de régularité), la fonction argmax $v \mapsto a^*(s; v)$ est localement constante, ce qui supprime la non-différentiabilité de l'opérateur max. Sans cette condition de régularité (plus précisément, aux points où plusieurs actions sont à égalité pour l'optimalité), le théorème des fonctions implicites ne s'applique pas, et l'opérateur n'est pas différentiable au sens de Fréchet. Le point de vue de l'ensemble actif (premier point de vue) et le théorème de l'enveloppe (deuxième point de vue) fournissent les outils pour traiter ces points non lisses.

### Lien avec l'itération sur les politiques

Revenons au pas de Newton-Kantorovitch :

$$
(I - \Bellman'(v_n)) h_n = v_n - \Bellman v_n,
\quad
v_{n+1} = v_n - h_n.
$$

Supposons que $\Bellman'(v_n) = \gamma P_{\pi_{v_n}}$ pour la politique gloutonne $\pi_{v_n}$. Alors

$$
(I - \gamma P_{\pi_{v_n}}) v_{n+1} = r^{\pi_{v_n}},
$$

ce qui est exactement l'évaluation de la politique $\pi_{v_n}$. Recalculer la politique gloutonne à partir de $v_{n+1}$ donne l'itéré suivant.

Ainsi, **l'itération sur les politiques est la méthode de Newton-Kantorovitch** appliquée à l'équation d'optimalité de Bellman. Aux points de non-différentiabilité (lorsque des égalités se produisent), l'opérateur reste semi-lisse, et l'itération sur les politiques correspond à une méthode de Newton semi-lisse. C'est le théorème de l'enveloppe qui justifie la simplification de la jacobienne en $\gamma P_{\pi_v}$, ce qui évite d'avoir à dériver à travers l'optimiseur. L'équivalence est ainsi établie.

### Le point de vue de Newton semi-lisse

Les trois points de vue développés ci-dessus (l'ensemble actif, le théorème de l'enveloppe et le théorème des fonctions implicites) pointent tous vers un cadre plus profond pour comprendre les méthodes de type Newton appliquées à des opérateurs non lisses. Ce cadre, celui des méthodes de Newton semi-lisses, a été développé précisément pour traiter des opérateurs qui, comme l'opérateur de Bellman, sont lisses par morceaux sans être différentiables globalement. Le lien entre l'itération sur les politiques et les méthodes de Newton semi-lisses a été développé de façon rigoureuse dans des travaux récents {cite}`Gargiani2022`.

La méthode de Newton-Kantorovitch classique suppose que l'opérateur est différentiable au sens de Fréchet partout. La dérivée existe, est unique et varie continûment avec le point de base. Mais l'opérateur de Bellman $\Bellman$ viole cette hypothèse en toute fonction de valeur pour laquelle plusieurs actions sont à égalité pour l'optimalité dans un état. En de tels points, le théorème des fonctions implicites ne s'applique pas, et il n'existe pas de dérivée de Fréchet unique. 

Les méthodes de Newton semi-lisses contournent cette difficulté en remplaçant la notion de jacobienne unique par une dérivée généralisée qui décrit le comportement de l'opérateur près des points non lisses. La dérivée généralisée la plus utilisée est le sous-différentiel de Clarke, qu'on peut voir comme l'enveloppe convexe de toutes les « jacobiennes candidates » obtenues comme limites lorsqu'on approche le point non lisse depuis différentes directions.

Pour le résidu de Bellman $\mathrm{B}(v) = \Bellman v - v$, le sous-différentiel de Clarke en un point $v$ peut être caractérisé explicitement à l'aide de notre premier point de vue. Rappelons que, dans chaque état $s$, on a défini l'ensemble actif $\mathcal{A}^*(s; v) = \arg\max_a Q_a(v, s)$. Lorsque cet ensemble contient plusieurs actions, l'opérateur n'est pas différentiable au sens de Fréchet. Il reste toutefois directionnellement dérivable dans toutes les directions, et le sous-différentiel de Clarke est formé de toutes les matrices de la forme

$$
\partial \mathrm{B}(v) = \left\{ I - \gamma P_\pi : \pi(s) \in \mathcal{A}^*(s; v) \text{ pour tout } s \right\}.
$$

Autrement dit, la jacobienne généralisée est l'ensemble de toutes les matrices $I - \gamma P_\pi$ où $\pi$ est une politique quelconque qui choisit une action dans l'ensemble actif de chaque état. Lorsque le maximiseur est unique partout, cet ensemble se réduit à un singleton, et l'on retrouve la dérivée de Fréchet classique. Lorsque des égalités se produisent, l'ensemble contient plusieurs éléments : précisément les combinaisons convexes mentionnées dans le premier point de vue.

La méthode de Newton semi-lisse pour résoudre $\mathrm{B}(v) = 0$ procède en choisissant à chaque itération un élément $J_k \in \partial \mathrm{B}(v_k)$ et en résolvant

$$
J_k h_k = -\mathrm{B}(v_k), \quad v_{k+1} = v_k + h_k.
$$

Il en ressort que tout choix dans le sous-différentiel de Clarke donne une mise à jour valide de type Newton. Dans le contexte de l'équation de Bellman, choisir $J_k = I - \gamma P_{\pi_k}$, où $\pi_k$ est une politique gloutonne quelconque, correspond exactement à l'étape d'évaluation de politique de l'itération sur les politiques. La liberté de choisir l'action en cas d'égalité se traduit par la liberté de choisir l'élément du sous-différentiel à utiliser.

Sous des conditions de régularité appropriées (plus précisément, lorsque la fonction résidu est BD-régulière ou CD-régulière), la méthode de Newton semi-lisse converge localement à vitesse quadratique {cite}`Gargiani2022`. Autrement dit, près de la solution, l'erreur décroît quadratiquement :
$$
\|v_{k+1} - v^*\| \leq C \|v_k - v^*\|^2.
$$

Ce résultat théorique explique une observation empirique faite depuis longtemps en pratique : l'itération sur les politiques converge en général en très peu d'itérations, souvent une poignée seulement, même lorsque les espaces d'états et d'actions sont énormes et que l'espace des politiques possibles est de taille exponentielle. 

Le cadre de Newton semi-lisse suggère aussi tout un éventail de méthodes intermédiaires entre l'itération sur les valeurs et l'itération sur les politiques. L'itération sur les valeurs peut être interprétée comme une méthode de type Newton dans laquelle on choisit $J_k = I$ à chaque itération, en ignorant entièrement la dépendance de $\Bellman$ en $v$. Ce choix garantit la convergence globale grâce à la propriété de contraction, mais sacrifie la vitesse de convergence locale quadratique. L'itération sur les politiques, à l'autre extrême, utilise la jacobienne généralisée complète $J_k = I - \gamma P_{\pi_k}$ et atteint une convergence quadratique, au prix de la résolution d'un système linéaire à chaque itération.

Entre ces extrêmes se trouvent des méthodes qui utilisent des jacobiennes approchées ou qui ne résolvent le système de Newton qu'approximativement. L'itération sur les politiques modifiée en fait partie : elle conserve la jacobienne $I - \gamma P_{\pi_k}$, mais remplace son inverse par les $m+1$ premiers termes de la série de Neumann. Une autre variante naturelle consiste à choisir $J_k = \alpha I$ pour un scalaire $\alpha > 0$. On obtient alors la mise à jour

$$
v_{k+1} = \frac{\alpha - 1}{\alpha} v_k + \frac{1}{\alpha} \Bellman v_k.
$$

C'est l'itération sur les valeurs relaxée, de pas $1/\alpha$. Prendre $\alpha > 1$ amortit le pas : la mise à jour reste une contraction globale, mais de module plus grand, $1 - (1-\gamma)/\alpha$, et elle est donc plus lente que l'itération sur les valeurs standard. Prendre $\alpha < 1$ allonge le pas, ce qui donne une surrelaxation. L'argument de la norme sup ne donne alors une contraction que pour $\alpha > (1+\gamma)/2$, de module $(1+\gamma)/\alpha - 1$, qui dépasse $\gamma$. Près de la solution, cependant, la vitesse est fixée par les valeurs propres de $\gamma P_{\pi^*}$. Le mode constant se contracte maintenant à $1 - (1-\gamma)/\alpha < \gamma$; la surrelaxation est donc plus rapide chaque fois que $P_{\pi^*}$ n'a aucune valeur propre proche de $-1$. Pour tout $\alpha$, la mise à jour n'exige que des opérations point par point au lieu de la résolution d'un système linéaire. Chaque choix d'approximation de la jacobienne généralisée donne un algorithme : la jacobienne exacte donne l'itération sur les politiques, l'identité donne l'itération sur les valeurs, et l'identité mise à l'échelle donne l'itération sur les valeurs relaxée.

L'itération sur les politiques est donc une méthode de Newton semi-lisse, une classe de méthodes employée bien au-delà de la programmation dynamique. Les méthodes de Newton semi-lisses sont utilisées en optimisation (pour les problèmes de complémentarité et les inéquations variationnelles), en optimisation sous contraintes d'équations aux dérivées partielles (pour les problèmes avec contraintes sur la commande) et en économie (pour les problèmes d'équilibre). L'équation de Bellman est un cas particulier d'équation lisse par morceaux, et la théorie de la convergence développée pour ces équations s'y applique.

### L'itération sur les politiques comme méthode de Newton

En comparant le pas de Newton ci-dessus à l'algorithme {prf:ref}`policy-iteration-standard`, on constate que la linéarisation a lieu à l'étape 3 d'évaluation de la politique, où l'on résout $(\mathbf{I}-\gamma \mathbf{P}_{\pi_n}) \mathbf{v} = \mathbf{r}_{\pi_n}$, et que l'étape d'amélioration recalcule le point en lequel l'opérateur est linéarisé. Le système linéaire peut être résolu par des méthodes directes (élimination de Gauss), par approximations successives ou par des méthodes de Krylov comme GMRES. Le résoudre seulement de façon approchée, avec quelques balayages d'approximations successives, donne l'algorithme {prf:ref}`alg-modified-policy-iteration` : une méthode de Newton inexacte dont la direction provient d'une série de Neumann tronquée.

(sec-sailing-wind-shifts)=
## Exemple : remonter au vent vers une bouée par vent changeant

Comment un navigateur doit-il remonter vers une bouée placée au vent lorsque la
direction du vent ne cesse de changer, et les règles empiriques que les
régatiers apprennent sur l'eau découlent-elles de l'équation de Bellman?

Un voilier ne peut pas naviguer face au vent. Pour atteindre une bouée située
au vent, il navigue en formant un angle avec le vent sur un *bord*, le vent
arrivant d'un côté du bateau, puis il vire en passant dans le vent pour
naviguer sur l'autre bord. Le bateau est *tribord amures* lorsque le vent
arrive par son côté droit et *bâbord amures* lorsqu'il arrive par son côté
gauche. Chaque virement de bord coûte du temps. Lorsque le vent tourne, l'un
des bords pointe plus près de la bouée (le vent *adonne*) et l'autre pointe
plus loin (le vent *refuse*). On enseigne aux régatiers à virer dans les
refusantes et à éviter de rejoindre trop tôt les *lignes d'approche*, c'est-à-dire
les lignes à partir desquelles la bouée peut être atteinte en un seul bord.
{cite:t}`Vanderbei1996Sailing` a posé ce problème comme un programme dynamique
stochastique sur une grille. Il est ensuite devenu le banc d'essai « sailing »
de planification en ligne de {cite:t}`Peret2004`, de {cite:t}`Kocsis2006` et
de {cite:t}`Bonet2012`, où l'itération sur les valeurs fournit les valeurs
optimales exactes auxquelles on compare les planificateurs par échantillonnage.

**États.** Le plan d'eau est une grille de $21\times 21$ points de passage.
L'état est $s=(x,y,w,\tau)$ : le point de passage $(x,y)$ du bateau, avec $x$
orienté vers l'est et $y$ vers le nord; la direction
$w\in\{\mathrm{N},\mathrm{NE},\ldots,\mathrm{NW}\}$ d'où souffle le vent; et
l'amure courante $\tau\in\{\text{bâbord},\text{tribord}\}$. L'amure doit faire
partie de l'état, car le coût du prochain bord en dépend. Il y a
$21^2\times 8\times 2=7056$ états. Le départ est le point de passage $(10,2)$
et la bouée est en $(10,18)$, 16 rangées au nord. La bouée est un état
absorbant de coût nul.

**Actions.** Chaque bord mène à l'un des 8 points de passage voisins; l'action
est donc un cap $a\in\{\mathrm{N},\mathrm{NE},\ldots,\mathrm{NW}\}$. Soit
$\theta(w,a)\in\{0,45,\ldots,180\}$ l'angle en degrés entre la direction du
vent et le cap. Le cap $\theta=0$ pointe face au vent et n'est pas permis, pas
plus qu'un cap qui fait sortir du plan d'eau.

**Coûts.** Le coût d'un bord est sa durée en minutes. À la suite de
{cite:t}`Vanderbei1996Sailing`, un bord en ligne droite entre deux points de
passage voisins prend $t(45)=4$ minutes au près, $t(90)=3$ minutes au travers
et $t(135)=2$ minutes au grand largue. Le vent arrière prend $t(180)=1$
minute, la valeur par défaut du code du banc d'essai de {cite:t}`Bonet2012`.
Un bord en diagonale est $\sqrt 2$ fois plus long, et on lui fait prendre
$\sqrt 2$ fois plus de temps, ce qui reproduit la plage de coûts des bords, de
1 à 8.6 minutes, rapportée par {cite:t}`Kocsis2006`. Soit $\tau'(w,a,\tau)$
l'amure après le bord : tribord si le vent vient de la droite du cap, bâbord
s'il vient de la gauche, et inchangée au vent arrière. Un virement de bord fait
perdre $\delta=3$ minutes {cite:p}`Vanderbei1996Sailing`. Le coût est

$$
c(s,a)=\ell(a)\,t\big(\theta(w,a)\big)+\delta\,\mathbb 1\{\tau'(w,a,\tau)\neq\tau\},
$$

où $\ell(a)=1$ pour N, E, S, W et $\ell(a)=\sqrt2$ pour les diagonales.

**Transitions.** Le bateau se déplace de façon déterministe vers le point de
passage voisin $(x',y')$ dans la direction $a$. À la fin du bord, le vent garde
sa direction ou tourne de $45^\circ$ selon une chaîne de Markov $q(w'\mid w)$,
de sorte que

$$
p\big((x',y',w',\tau')\mid s,a\big)=q(w'\mid w).
$$

On compare trois chaînes. Dans la chaîne *stable*, le vent ne change jamais. La
chaîne *oscillante* utilise des valeurs pédagogiques pour une brise qui oscille
autour du nord : depuis le nord, le vent tourne à gauche ou à droite avec
probabilité 0.1 chacune; depuis toute autre direction, il revient vers le nord
avec probabilité 0.2, s'en éloigne davantage avec probabilité 0.1 et, sinon,
persiste. La chaîne *irrégulière* est la matrice par défaut du code du banc
d'essai de {cite:t}`Bonet2012`, tournée de sorte que le vent qu'elle favorise
souffle du nord. Elle ne conserve la direction du vent qu'avec une probabilité
de 0.2 à 0.4.

**Objectif.** Le but est le plus petit temps espéré jusqu'à la bouée,

$$
v^*(s)=\min_\pi \mathbb E^\pi_s\Big[\sum_{t=0}^{T-1}c(S_t,A_t)\Big],
$$

où $T$ est le premier bord qui se termine à la bouée. C'est un problème de
*plus court chemin stochastique* : il n'y a pas d'actualisation, et l'horizon
est aléatoire, car la course se termine lorsque la bouée est atteinte. Un
facteur d'actualisation rendrait une minute en fin de course moins coûteuse
qu'une minute au départ, ce qui n'est pas ce que veut un régatier; on garde
donc $\gamma=1$. L'opérateur de Bellman

$$
(\Bellman v)(s)=\min_{a\in\mathcal A_s}\Big\{c(s,a)+\sum_{w'}q(w'\mid w)\,v\big(x',y',w',\tau'(w,a,\tau)\big)\Big\}
$$

n'est alors pas une contraction pour la norme sup, et l'argument de la
proposition {prf:ref}`value-iteration-convergence` ne s'applique pas. Un autre
argument s'applique. Une certaine politique atteint la bouée avec probabilité
un (une politique *propre*), et chaque bord coûte au moins une minute, de sorte
qu'une politique qui n'atteint pas la bouée a un coût espéré infini. Sous ces
deux conditions, {cite:t}`Bertsekas1991` montrent que $v^*$ est l'unique point
fixe de $\Bellman$ et que l'itération sur les valeurs y converge à partir de
toute fonction initiale. On exécute l'algorithme {prf:ref}`value-iteration`
avec $\min$ au lieu de $\max$ à partir de $v_0=0$, et on s'arrête lorsque
$\|v_{n+1}-v_n\|_\infty<10^{-9}$ minute. On exécute aussi l'algorithme
{prf:ref}`policy-iteration-standard` à partir d'une politique de référence
décrite ci-dessous.

:::{iframe} ../interactive/sailing-wind-shifts.html
:label: fig-sailing-wind-shifts
:width: 100%
:title: Naviguer par vent changeant : la politique de l'itération sur les valeurs, des courses contre un navigateur des lignes d'approche, et les balayages de l'itération sur les valeurs
:class: sailing-wind-shifts-replay
:placeholder: _static/sailing_wind_shifts/sailing.svg

La politique optimale pour la brise oscillante. (a) Vent du nord, bateau
tribord amures. Chaque flèche est le cap optimal depuis ce point de passage;
les flèches rouges changent d'amure. Les courbes de niveau donnent le nombre
espéré de minutes jusqu'à la bouée (triangle), et les traits pointillés sont
les lignes d'approche du vent moyen. (b) Le vent a reculé au nord-ouest, ce qui
fait refuser tous les bateaux tribord amures; presque partout, la politique
vire bâbord amures, une amure qui pointe maintenant droit vers le nord. (c) Une
course, dans laquelle le bateau optimal (bleu) et le navigateur des lignes
d'approche (orange) voient le même vent à chaque bord; les cercles marquent les
virements. Cette course est une large victoire du bateau optimal, 90.9 contre
106.9 minutes, alors que l'écart espéré est de 2.5 minutes. (d) Erreur en norme
sup de l'itération sur les valeurs à partir de $v_0=0$, son résidu, et l'erreur
de l'itération sur les politiques à partir du navigateur des lignes
d'approche. Dans la version en ligne du livre, le rejeu fait courir les deux
bateaux bord par bord avec la flèche du vent et le champ de flèches pour le
vent courant, montre la distribution des temps d'arrivée sur 4 000 courses et
anime les balayages de l'itération sur les valeurs.
:::

La référence est un *navigateur des lignes d'approche*, qui ignore les
variations du vent pour décider où virer. Ce navigateur reste tribord amures
jusqu'à atteindre la ligne d'approche bâbord du vent moyen (nord), puis navigue
bâbord amures vers la bouée, et fait l'inverse du côté droit du parcours. À
chaque bord, il prend le cap permis sur cette amure qui se termine le plus près
de la bouée. Une seconde référence, le planificateur *à équivalent certain*,
replanifie à chaque bord comme si le vent qu'il voit maintenant ne devait
jamais changer. Le tableau donne le temps espéré depuis le départ avec le vent
du nord. Toutes les valeurs sont des évaluations exactes de politiques. Pour la
politique optimale et le navigateur des lignes d'approche, la moyenne de 4 000
courses simulées par chaîne concorde avec elles à quatre erreurs types près.

:::{include} artifacts/sailing_wind_shifts/results.md
:::

Par vent stable, les trois politiques coïncident. Le meilleur parcours ne
comporte qu'un virement : 16 bords en diagonale de $4\sqrt2$
minutes plus un virement de 3 minutes, soit 93.5 minutes, et le script de
construction vérifie que l'itération sur les valeurs renvoie exactement ce
nombre. Les variations du vent raccourcissent la course en moyenne pour toutes
les politiques, car un bateau adonné peut naviguer droit vers le nord à 4
minutes par rangée au lieu de $4\sqrt2\approx5.7$. La politique optimale en
tire le meilleur parti. Dans la brise oscillante, elle demande 84.1 minutes
contre 86.6 pour le navigateur des lignes d'approche; dans la brise
irrégulière, 72.4 contre 75.2. Le gain est d'environ 3 % de la course, et
c'est un gain en espérance. Sur 4 000 courses appariées dans la brise
oscillante, le bateau optimal arrive premier dans 62 % des cas et à égalité
dans 8 %; le navigateur des lignes d'approche gagne donc encore environ trois
courses sur dix. Le planificateur à équivalent certain se situe entre les
deux, à 85.6 et 74.3 minutes : réagir au vent observé maintenant capte une
partie du gain, et planifier en fonction du vent à venir capte le reste.

Deux éléments du savoir des régatiers se lisent dans la politique. Le premier
est *virer dans les refusantes*. Prenons l'eau libre sous la bouée, à au moins
trois colonnes à l'intérieur des deux lignes d'approche, et examinons chaque
état où le vent a tourné de $45^\circ$ par rapport au nord, sur l'une ou
l'autre amure. Dans la brise oscillante, la politique optimale termine le bord
suivant sur l'amure adonnante dans les 900 états de ce type; dans la brise
irrégulière, elle le fait dans 93 % d'entre eux. Une refusante de $45^\circ$
fait naviguer le bateau refusé en travers du parcours sans progresser vers la
bouée; payer 3 minutes pour virer en vaut donc la peine, même si le vent risque
de revenir bientôt. Le panneau (b) de la {numref}`fig-sailing-wind-shifts` le
montre sous la forme d'un champ de flèches rouges. Le second est *rester loin
des lignes d'approche*. Par vent moyen, un bateau tribord amures situé huit
rangées sous la bouée vire dès qu'il se trouve quatre colonnes à gauche de la
bouée, à mi-chemin de la ligne d'approche (panneau (a)). La raison est celle
que donnent les régatiers. Sur la ligne d'approche elle-même, toute variation
du vent nuit : une refusante impose un virement de plus, et une adonnante
signifie que le bateau a parcouru plus de chemin que nécessaire. Rester près
du milieu du parcours garde les deux options ouvertes. Pour un planificateur
qui suppose que le vent observé maintenant persistera, comme le planificateur à
équivalent certain, tous les points de virement à l'intérieur des lignes
d'approche se valent; rien ne l'en éloigne donc. Aucune de ces deux règles
n'est inscrite dans le modèle; toutes deux sortent de la minimisation dans
$\Bellman$.

Les deux algorithmes atteignent $v^*$ de façons différentes. L'itération sur
les valeurs à partir de $v_0=0$ calcule, au balayage $n$, le plus petit coût
espéré de $n$ bords, sans pénalité pour l'endroit où le bateau termine. Pour un
bateau loin de la bouée, les $n$ bords les moins chers sont des bords bon
marché au travers et au portant qui ne mènent pas à la bouée, de sorte que
$v_n$ reste nettement inférieur à $v^*$ à cet endroit. Les valeurs se
stabilisent d'abord près de la bouée, et la région à moins d'une minute de
$v^*$ s'étend vers l'extérieur à partir de la bouée jusqu'à couvrir le plan
d'eau au balayage 37. Le mode itération sur les valeurs du rejeu le montre.
L'erreur en norme sup reste au-dessus d'une minute pendant les 36 premiers
balayages, puis s'effondre. Dans la brise oscillante, le résidu atteint
exactement zéro après 42 balayages. Sous la politique optimale, tout parcours
atteint la bouée en au plus 33 bords, quoi que fasse le vent; l'itération sur
les valeurs se termine donc comme une induction arrière avec un horizon de
quelques dizaines de bords, au lieu de converger géométriquement. L'itération
sur les politiques à partir du navigateur des lignes d'approche demande 6
étapes d'amélioration. Chaque étape évalue une politique entière, le système
linéaire de l'algorithme {prf:ref}`policy-iteration-standard` (résolu ici en
itérant l'opérateur de Bellman propre à cette politique), et l'erreur passe de
14.4 minutes à zéro.

Le modèle est une version pédagogique du vrai problème. Les vraies variations
du vent sont en général beaucoup plus petites que $45^\circ$, et elles se
produisent dans le temps, et non aux points de passage; la vitesse du bateau
dépend aussi de la force du vent, qui est fixe ici. Les durées des bords et le
délai de virement sont les valeurs publiées du banc d'essai, et non des
mesures faites sur un bateau, et la chaîne oscillante, le plan d'eau et les
deux références sont nos propres choix. Aucune donnée de navigation mesurée
n'est utilisée. Ce qui se transpose, c'est la structure : un coût à venir qui
dépend du vent autant que de la position, une fonction de valeur qui est
l'unique point fixe de l'opérateur de Bellman sans aucune actualisation, et des
règles empiriques de navigation retrouvées comme propriétés de la politique
optimale.

- {download}`Modèle de navigation et solveur <code/sailing_wind_shifts.py>`
- {download}`Script de construction de l'expérience de navigation <scripts/build_sailing_wind_shifts_artifacts.py>`
- {download}`Résultats de navigation pour les trois chaînes de vent <artifacts/sailing_wind_shifts/metrics.json>`

(sec-bridge-deck-maintenance)=
## Exemple : entretien d'un tablier de pont

Quand un organisme responsable des routes doit-il réparer un tablier de pont
en béton qui n'a pas encore cédé, et combien une règle fondée sur l'état
fait-elle économiser par rapport à un calendrier fixe?

Chaque pont routier des États-Unis est inspecté, habituellement tous les deux
ans, et l'inspecteur attribue au tablier une cote d'état de 9 (excellent) à 0
(défaillant). Les cotes sont transmises chaque année au National Bridge
Inventory (NBI), où la cote du tablier est l'élément 58. Les tabliers se
dégradent lentement et de façon aléatoire, par fissuration, corrosion des
armatures et écaillage. Un organisme peut ne rien faire, sceller et colmater
le tablier, le resurfacer ou le réhabiliter, ou le remplacer. Les travaux
coûtent de l'argent maintenant; un tablier en mauvais état coûte de l'argent
plus tard, à cause du risque d'une défaillance qui ferme le pont. Ce compromis
se répète chaque année sans fin naturelle; on le modélise donc comme un
processus de décision markovien actualisé à horizon infini, qu'on résout avec
les algorithmes de ce chapitre.

**États.** L'état est la cote du tablier, observée une fois par année. Les
cotes 9 à 5 restent distinctes, et les cotes 4 et moins sont regroupées en un
seul état *mauvais*, de sorte que $\mathcal S=\{9,8,7,6,5,{\le}4\}$. Le
regroupement des cotes les plus basses suit le modèle de tablier de
{cite:t}`Saifullah2024DDMAC`, dont nous utilisons les coûts plus bas.

**Actions.** Dans chaque état, $\mathcal A_s=\{\text{rien},\text{mineure},\text{majeure},\text{remplacer}\}$ :
une réparation mineure (scellement et colmatage), une réparation majeure
(réhabilitation ou resurfaçage) ou un nouveau tablier.

**Transitions.** Le noyau de l'inaction $p(j\mid s,\text{rien})$ est estimé
par dénombrement à partir des données du NBI. On utilise les historiques de
tabliers rassemblés par {cite:t}`Liu2023BridgeCNN`, qui contiennent la cote de
l'élément 58 de 88 596 tabliers pour chaque année de 1993 à 2019 et sont
reproduits dans leur dépôt de code public. Aucun historique de cet échantillon
ne remonte jamais; celui-ci ne contient donc que des tabliers qui n'ont reçu
aucun travail faisant monter la cote. Avec $n_{sj}$ le nombre d'années au
cours desquelles un tablier est passé de la cote $s$ à la cote $j$,
l'estimateur du maximum de vraisemblance est

$$
\hat p(j\mid s,\text{rien})=\frac{n_{sj}}{\sum_{k} n_{sk}} ,
$$

calculé à partir de 2 303 496 paires d'années consécutives. La diagonale est
grande : un tablier reste à la cote 9 avec probabilité 0.85 par année, à 8
avec 0.92, à 7 avec 0.97, à 6 avec 0.97 et à 5 avec 0.98. Ces valeurs
correspondent à des séjours moyens de 6.5, 12.9, 30.9, 36.4 et 55.7 ans. Aucun
tablier ne s'améliore sans travaux; l'état mauvais est donc absorbant sous
« rien ».

Les effets des travaux sont des hypothèses pédagogiques, dans l'esprit du
modèle de composant de {cite:t}`Andriotis2020DRLInspection`. Soit $g(s,a)$ la
cote juste après les travaux : une réparation mineure fait monter la cote d'un
cran, mais ne peut pas réparer un tablier en mauvais état, une réparation
majeure la fait monter de deux crans, jusqu'à 8 au plus, et le remplacement
donne un tablier neuf, $g(s,\text{remplacer})=9$. Le tablier se dégrade ensuite
pendant le reste de l'année comme si rien n'avait été fait :

$$
p(j\mid s,a)=\hat p\big(j\mid g(s,a),\text{rien}\big).
$$

**Coûts.** Les coûts sont exprimés en dollars américains par mètre carré de
tablier. Les coûts des travaux sont $k(\text{rien})=0$, $k(\text{mineure})=400$,
$k(\text{majeure})=1\,200$ et $k(\text{remplacer})=2\,650$, d'après le tableau 7
de {cite:t}`Saifullah2024DDMAC`. Leur tableau 6 donne une probabilité annuelle
de défaillance $q(s)$ de 0.001 pour les cotes 9 et 8, de 0.005 pour 7, 6 et 5,
et de 0.01 pour l'état mauvais, et ils imputent à une défaillance dix fois le
coût de reconstruction. On prend $F=10\times2\,650=26\,500$ et on laisse de
côté le coût supplémentaire qu'ils comptabilisent pendant qu'un pont défaillant
est hors service. Le coût d'une période est le coût des travaux plus le coût
espéré de défaillance du tablier réparé,

$$
c(s,a)=k(a)+q\big(g(s,a)\big)\,F .
$$

Un tablier laissé tel quel coûte donc 26.5 par année aux cotes 9 et 8, 132.5
aux cotes 7 à 5, et 265 lorsqu'il est en mauvais état.

**Actualisation.** On utilise $\gamma=0.97$, soit un taux annuel de 3 %, comme
{cite:t}`Saifullah2024DDMAC`.

La valeur optimale $v^\star(s)$ est le plus petit coût actualisé espéré à
partir de la cote $s$, et elle résout l'équation de Bellman avec un minimum au
lieu du maximum,

$$
v^\star(s)=\min_{a\in\mathcal A_s}\Big\{c(s,a)+\gamma\sum_{j\in\mathcal S}p(j\mid s,a)\,v^\star(j)\Big\}.
$$

On la résout deux fois. L'itération sur les valeurs (algorithme
{prf:ref}`value-iteration`, avec un minimum) part de $v_0=0$ et s'arrête
lorsque $\lVert v_{n+1}-v_n\rVert_\infty<\varepsilon(1-\gamma)/(2\gamma)$, avec
$\varepsilon=10^{-6}$ dollar par mètre carré. Avec $\gamma=0.97$, l'erreur
diminue d'environ 3 % par balayage, et le critère est satisfait après 728
balayages. L'itération sur les politiques (algorithme
{prf:ref}`policy-iteration-standard`) part de « ne rien faire nulle part »,
résout le système linéaire $6\times6$
$(\mathbf I-\gamma\mathbf P_\pi)\mathbf v=\mathbf c_\pi$ à chaque pas, et
s'arrête après trois évaluations. Sa deuxième politique répare déjà chaque
tablier coté 7 ou moins; la troisième change l'action à deux cotes et est
optimale. C'est le comportement de Newton décrit plus haut : une poignée de
résolutions linéaires au lieu de centaines de contractions. Les deux méthodes
renvoient la même politique, et les tests la vérifient par rapport aux
$4^6=4\,096$ politiques déterministes stationnaires.

On compare la politique optimale à deux règles qui n'exigent aucun modèle. La
règle *réactive* ne fait rien jusqu'à ce que le tablier soit en mauvais état,
puis le remplace. Le *calendrier* applique une réparation mineure tous les $k$
ans, quelle que soit la cote, et remplace le tablier lorsqu'il est en mauvais
état; on rapporte le meilleur $k$ entre 1 et 40. Comme le calendrier dépend de
l'année, on l'évalue exactement sur la chaîne de Markov du couple (année modulo
$k$, cote).

:::{iframe} ../interactive/bridge-deck-maintenance.html
:label: fig-bridge-deck-maintenance
:width: 100%
:title: Entretien d'un tablier de pont : politique optimale contre règles réactive et calendaire
:class: bridge-deck-maintenance-replay
:placeholder: _static/bridge_deck_maintenance/policy.svg

(a) Probabilités de transition sans travaux entre les cotes du tablier,
dénombrées à partir des données du NBI sur 88 596 tabliers, de 1993 à 2019;
les entrées inférieures à 0.005 sont laissées vides. (b) Action optimale à
chaque cote pour des coûts de défaillance de 2 à 40 fois le coût de
remplacement; la colonne encadrée est le cas de base. (c) Coût actualisé espéré
à partir de chaque cote dans le cas de base pour la politique optimale, le
meilleur calendrier (une réparation mineure tous les 9 ans) et la règle
réactive. Dans la version en ligne du livre, la figure est un rejeu : deux
groupes de 12 tabliers partent des mêmes cotes et utilisent les mêmes nombres
aléatoires pendant 50 ans, l'un sous la politique optimale et l'autre sous une
règle de référence, avec des équipes qui signalent chaque réparation et des
compteurs du coût actualisé cumulé.
:::

Dans le cas de base, la politique optimale ne fait rien aux cotes 9 et 8,
applique une réparation mineure aux cotes 7, 6 et 5, et une réparation majeure
lorsque le tablier est en mauvais état. Elle ne remplace jamais un tablier. La
politique est à seuil : les travaux commencent à la première cote où le risque
de défaillance fait un bond, de 26.5 à 132.5 dollars par mètre carré par
année, et les travaux s'alourdissent à mesure que le tablier se dégrade. Une
réparation mineure à la cote 7 coûte 400 et ramène le tablier à 8, ce qui
économise environ 106 par année tant qu'il y reste, soit 12.9 ans en moyenne.
Pour un tablier neuf, le coût actualisé espéré est de 1 864 dollars par mètre
carré, contre 2 951 pour le meilleur calendrier et 3 200 pour la règle
réactive.

:::{include} artifacts/bridge_deck_maintenance/results.md
:::

Le tableau montre que la réponse est dictée par le prix de la défaillance.
Lorsqu'une défaillance ne coûte que deux fois le coût de remplacement, la
meilleure politique est de ne jamais toucher au tablier. L'état mauvais coûte
alors 53 par année, et aucune réparation n'est rentable à un taux
d'actualisation de 3 %. La structure à seuil n'est pas garantie non plus. Avec
un coût de défaillance de cinq fois le remplacement, la politique optimale
répare les tabliers cotés 7 et 6, mais laisse tels quels les tabliers cotés 5.
Une réparation mineure fait passer un tablier de 5 à 6, où la probabilité de
défaillance est la même; elle ne fait donc que retarder l'arrivée dans l'état
mauvais. Avec un coût de défaillance de 40 fois le remplacement, un tablier
coté 6 reçoit une réparation majeure et un tablier coté 5 une réparation
mineure, pour la même raison : une réparation majeure fait passer 6 à 8, hors
de la bande à risque élevé, alors qu'à partir de 5 elle n'atteindrait que 7.
Les politiques optimales monotones découlent de conditions sur les coûts et
les transitions {cite:p}`Puterman1994`, et le tableau des défaillances en
escalier les viole. Dans chaque ligne, la politique optimale bat les deux
règles, et l'écart croît avec le coût de défaillance, jusqu'à 3 720 dollars
par mètre carré par rapport au meilleur calendrier et 10 344 par rapport à la
règle réactive à 40 fois le remplacement.

Le rejeu montre la même comparaison sur des trajectoires échantillonnées. Dans
le cas de base, après 50 ans, les 12 tabliers sous la politique optimale ont
coûté en moyenne 1 863 dollars par mètre carré, en valeur actualisée, contre
3 196 sous la règle réactive et 2 880 sous le meilleur calendrier. Le
calendrier gaspille de l'argent sur des tabliers encore cotés 9 ou 8, tandis
que la règle réactive laisse pendant des décennies des tabliers aux cotes 7 à
5, où le risque annuel est cinq fois plus élevé.

Plusieurs parties de ce modèle sont réelles, et plusieurs ne le sont pas. Le
noyau de l'inaction provient de véritables relevés d'inspection, mais il est
mis en commun pour tous les tabliers, sans égard à l'âge, à la circulation, au
climat ni au matériau, et il est estimé à partir d'un échantillon qui exclut
tout tablier dont la cote a remonté. Les tabliers qui se sont dégradés vite
sont ceux qui ont le plus probablement été réparés; les séjours moyens sont
donc probablement trop longs. Les coûts des travaux et les probabilités de
défaillance sont des valeurs publiées que leurs auteurs qualifient
d'indicatives, les effets des réparations sont des hypothèses, et le coût ne
comporte aucun terme pour les retards de circulation ou l'imposition d'une
limite de charge. Les inspections sont traitées comme exactes, alors que les
vraies cotes sont bruitées, ce qui transforme le problème en celui, partiellement
observé, qu'étudient {cite:t}`Andriotis2020DRLInspection`. Enfin, un organisme
gère des milliers de tabliers avec un seul budget, ce qui couple les problèmes
par tablier que nous avons résolus séparément ici.

- {download}`Modèle de tablier de pont et solveur <code/bridge_deck_maintenance.py>`
- {download}`Script de construction de l'expérience sur le tablier de pont <scripts/build_bridge_deck_maintenance_artifacts.py>`
- {download}`Dénombrements des transitions de cotes de tablier du NBI <artifacts/bridge_deck_maintenance/deck_transition_counts.csv>`
- {download}`Résultats sur le tablier de pont pour chaque coût de défaillance <artifacts/bridge_deck_maintenance/metrics.json>`

(sec-tamarisk-river)=
## Exemple : lutter contre une plante envahissante dans un réseau fluvial

Quel tronçon d'une rivière faut-il traiter chaque année pour contenir une
plante envahissante, et jusqu'à quelle taille la rivière peut-elle croître
avant qu'une réponse exacte ne devienne hors de portée?

Le tamaris, ou cèdre salé, est un arbuste qui a envahi les berges de nombreuses
rivières de l'ouest des États-Unis. Il évince des arbres indigènes comme les
saules et les peupliers, et ses graines voyagent avec l'eau, de sorte qu'une
infestation en amont réensemence tout ce qui se trouve en aval. Les
gestionnaires du territoire peuvent tuer les plants d'un tronçon de rivière, ou
les tuer et planter des espèces indigènes à leur place, mais chaque traitement
coûte de l'argent et seule une partie du réseau peut être traitée chaque année.
{cite:t}`Hall2012` ont formulé ce problème comme un processus de décision
markovien sur un réseau fluvial, et {cite:t}`Taleghan2015` l'ont utilisé comme
banc d'essai pour la planification avec un simulateur. Nous construisons une
petite version de leur modèle, la résolvons exactement par itération sur les
valeurs et par itération sur les politiques, puis regardons les méthodes
exactes manquer de place à mesure que la rivière grandit.

**États.** La rivière est un arbre binaire de $n$ tronçons, les portions de
rivière comprises entre deux confluences. L'eau s'écoule des feuilles vers le
tronçon 1, l'exutoire. Chaque tronçon possède un emplacement pour un arbre,
comme dans les plus petites instances de {cite:t}`Taleghan2015`, et cet
emplacement est vide ($E$), occupé par un arbre indigène ($N$) ou occupé par un
tamaris ($T$). L'état est $s=(s_1,\ldots,s_n)\in\{E,N,T\}^n$, de sorte que
$|\mathcal S|=3^n$.

**Actions.** Pour chaque tronçon $i$, le gestionnaire choisit
$a_i\in\{0,1,2\}$ : ne rien faire, éradiquer ou restaurer. L'éradication tue un
plant de tamaris avec probabilité $\chi=0.85$. La restauration éradique
d'abord, puis plante un arbre indigène dans l'emplacement vide, qui prend
racine avec probabilité $\beta=0.65$. Sans budget, l'action conjointe
$a=(a_1,\ldots,a_n)$ parcourt $3^n$ choix. Avec le budget de
{cite:t}`Taleghan2015`, au plus un tronçon est traité par année,

$$
\mathcal A=\Big\{a\in\{0,1,2\}^n:\ \textstyle\sum_{i}\mathbf 1\{a_i\neq 0\}\le 1\Big\},
\qquad |\mathcal A| = 2n+1 .
$$

**Transitions.** Une année comporte trois étapes. D'abord, le traitement agit
indépendamment sur chaque tronçon, avec un noyau $3\times 3$
$\alpha_{a_i}(y_i\mid s_i)$. Ensuite, chaque plant meurt avec probabilité
$0.2$, avec le noyau $\delta(z_i\mid y_i)$. Enfin, les survivants $z$
réensemencent les emplacements vides. Chaque plant survivant libère 100
graines, et une graine issue du tronçon $k$ tombe dans le tronçon $i$ avec
probabilité

$$
K_{ki}\ \propto\ 0.5^{\,d(k,i)}\;0.1^{\,u(k,i)},
$$

où $d(k,i)$ et $u(k,i)$ comptent les pas vers l'aval et vers l'amont sur le
chemin de $k$ à $i$, et où le poids restant s'écoule par l'exutoire. Les
valeurs 0.5 et 0.1 sont les paramètres de dispersion vers l'aval et vers
l'amont de {cite:t}`Taleghan2015`; la forme géométrique est notre
simplification du modèle qu'ils citent. Chaque tronçon vide reçoit aussi en
moyenne 1 graine de tamaris et 4 graines indigènes provenant de l'extérieur du
réseau. On traite les graines qui atteignent $i$ comme des variables de Poisson
de moyennes

$$
\lambda^T_i(z)=1+100\sum_{k:\,z_k=T}K_{ki},\qquad
\lambda^N_i(z)=4+100\sum_{k:\,z_k=N}K_{ki}.
$$

Un emplacement vide reste vide si aucune graine n'arrive. Sinon, une graine
l'occupe, et le tamaris l'emporte avec probabilité
$1.5\lambda^T_i/(1.5\lambda^T_i+\lambda^N_i)$, où 1.5 est l'une des valeurs de
l'avantage compétitif du tamaris examinées par {cite:t}`Hall2012`. Un
emplacement occupé garde son plant. Notons $g_i(j_i\mid z)$ la loi qui en
résulte pour le tronçon $i$ l'année suivante. Étant donné les survivants, les
tronçons se remplissent indépendamment, de sorte que

$$
p(j\mid s,a)=\sum_{y,z}\ \prod_{i=1}^n\alpha_{a_i}(y_i\mid s_i)\
\prod_{i=1}^n\delta(z_i\mid y_i)\ \prod_{i=1}^n g_i(j_i\mid z),
\qquad
\mathbf P_a=\mathbf A_a\,\mathbf D\,\mathbf G ,
$$

où $\mathbf A_a=\alpha_{a_1}\otimes\cdots\otimes\alpha_{a_n}$ et
$\mathbf D=\delta\otimes\cdots\otimes\delta$ sont des produits de Kronecker de
matrices $3\times 3$, et $\mathbf G$ une matrice dense $3^n\times 3^n$. Le
couplage entre tronçons n'intervient que par $\mathbf G$.

**Coûts et actualisation.** À la suite de {cite:t}`Taleghan2015`, chaque
tronçon envahi coûte 1.0 pour les dommages plus 0.1 pour l'emplacement de
tamaris qu'il contient, et les traitements coûtent $\kappa=(0,\,0.5,\,0.9)$
pour ne rien faire, éradiquer et restaurer :

$$
c(s,a)=1.1\sum_{i=1}^n\mathbf 1\{s_i=T\}+\sum_{i=1}^n\kappa(a_i),
\qquad \gamma=0.9 .
$$

Les coûts sont en unités abstraites. Le but est de minimiser le coût actualisé
espéré $\mathbb E\big[\sum_{t\ge 0}\gamma^t c(S_t,A_t)\big]$.

**Solution.** L'itération sur les valeurs (algorithme
{prf:ref}`value-iteration`, avec un minimum au lieu du maximum) calcule

$$
v_{k+1}(s)=\min_{a\in\mathcal A}\Big\{c(s,a)+\gamma\sum_{j}p(j\mid s,a)\,v_k(j)\Big\},
$$

et s'arrête lorsque $\|v_{k+1}-v_k\|_\infty<\varepsilon(1-\gamma)/(2\gamma)$,
avec $\varepsilon=10^{-6}$. Grâce à la forme factorisée de $\mathbf P_a$,
aucune matrice de transition n'est jamais stockée. Chaque balayage calcule une
fois $\mathbf w=\mathbf D\mathbf G v_k$, puis applique à $\mathbf w$ les petits
facteurs de $\mathbf A_a$ pour chaque action. L'itération sur les politiques
(algorithme {prf:ref}`policy-iteration-standard`) part de « ne rien faire »,
résout exactement $(\mathbf I-\gamma\mathbf P_\pi)v=\mathbf c_\pi$ à chaque pas
et améliore de façon gloutonne. Sur la rivière à 7 tronçons, avec
$3^7=2\,187$ états et 15 actions, l'itération sur les valeurs s'arrête après
148 balayages et l'itération sur les politiques après 4 améliorations. Les
deux fonctions de valeur concordent à $5\times10^{-7}$ près.

:::{iframe} ../interactive/tamarisk-river.html
:label: fig-tamarisk-river
:width: 100%
:title: Lutter contre le tamaris dans un réseau fluvial : la politique optimale contre une heuristique
:class: tamarisk-river-replay
:placeholder: _static/tamarisk_river/river.svg

Un épisode simulé sur une rivière à 7 tronçons entièrement envahie au départ.
Chaque segment est un tronçon, coloré selon ce qui y pousse; l'eau s'écoule
vers l'exutoire. Un anneau marque le tronçon traité cette année-là, R pour
restaurer et E pour éradiquer. En haut : la politique optimale calculée par
itération sur les valeurs. En bas : l'heuristique qui restaure le tronçon
envahi le plus en amont. Les deux rangées utilisent les mêmes nombres
aléatoires, et les titres donnent le coût actualisé accumulé avant cette
année-là. Dans la version en ligne du livre, le rejeu avance année par année,
et le lecteur peut changer la rivière de départ, la politique de comparaison
et la graine aléatoire.
:::

Chaque politique du tableau ci-dessous est évaluée exactement, par la
résolution de ses équations d'évaluation de politique; les nombres sont donc
des espérances et non des moyennes d'épisodes simulés.

:::{include} artifacts/tamarisk_river/results.md
:::

La politique optimale traite un tronçon envahi dans chacun des 2 059 états qui
contiennent du tamaris, et pourtant elle ne concorde avec la règle « le plus en
amont d'abord » que dans 678 d'entre eux, et avec la règle « le plus en aval
d'abord » que dans 672. À partir d'une rivière entièrement envahie, elle
restaure le tronçon 2, situé entre deux tronçons de tête et l'exutoire, plutôt
qu'un tronçon de tête ou l'exutoire lui-même. Lorsqu'un seul plant de tamaris
se trouve parmi des indigènes, elle éradique au lieu de restaurer, car la pluie
de graines indigènes regarnira généralement l'emplacement et le traitement le
moins cher suffit. Ces choix abaissent le coût espéré à partir d'une rivière
entièrement envahie de 38.33 sous la meilleure heuristique à 36.89, et de 14.53
à 13.30 en moyenne sur tous les états de départ. Le budget compte davantage que
le choix du tronçon. Restaurer chaque année chaque tronçon envahi, ce que le
budget interdit, coûte 19.83 à partir d'une rivière entièrement envahie, et la
politique optimale sans budget coûte 18.98.

Les mêmes solveurs s'exécutent sur des rivières de 1 à 8 tronçons
({numref}`fig-tamarisk-river-scaling`). Le nombre de balayages de l'itération
sur les valeurs change à peine, de 140 à 150, car il dépend de $\gamma$ et de
$\varepsilon$ plutôt que de la taille du modèle. L'itération sur les politiques
demande 3 ou 4 améliorations quelle que soit la taille. Ce qui croît, c'est le
travail par balayage. Le nombre d'états croît comme $3^n$, et un tableau dense
de $p(j\mid s,a)$ contiendrait $9^n(2n+1)$ nombres avec le budget et $27^n$
sans lui. Pour 7 tronçons sans budget, cela fait environ $10^{10}$ nombres, soit
84 Go. La mise à jour factorisée évite ce tableau, mais la matrice de couplage
$\mathbf G$ à elle seule compte $9^n$ entrées : 43 millions pour 8 tronçons, et
environ $3.5\times10^9$, soit 28 Go, pour 10. Sur la machine qui a produit les
figures, le modèle à 8 tronçons a pris environ 10 s à construire et quelques
secondes à résoudre, et le modèle à 7 tronçons sans budget a pris plusieurs
fois plus de temps à résoudre qu'avec budget. Les temps varient d'une exécution
à l'autre; la croissance, non. La même explosion touche les actions. Le budget
maintient l'ensemble des actions conjointes à $2n+1$ éléments, mais sans lui,
la minimisation de chaque mise à jour parcourt $3^n$ actions conjointes.

:::{figure} _static/tamarisk_river/scaling.svg
:label: fig-tamarisk-river-scaling

À gauche : le nombre d'états, le nombre d'actions conjointes avec le budget
d'un traitement, et le nombre d'entrées d'un tableau dense de $p(j\mid s,a)$
sans budget, en fonction du nombre de tronçons $n$. À droite : le temps en
secondes pour construire et résoudre le modèle sur une machine. Les temps
inférieurs à 0.1 s sont dominés par les frais fixes.
:::

Le modèle est une version pédagogique du banc d'essai. Les valeurs des
paramètres de mortalité, de production de graines, de dispersion, de succès
des traitements, de coûts et du facteur d'actualisation proviennent de
{cite:t}`Taleghan2015`, et l'avantage compétitif, de {cite:t}`Hall2012`, mais
la rivière, les tronçons à un seul emplacement, le noyau de dispersion
géométrique, la pluie de graines de Poisson et l'ordre des événements au cours
d'une année sont nos propres choix, et aucune donnée de terrain n'entre dans le
modèle. Les vraies rivières comptent des centaines de tronçons et de nombreux
emplacements par tronçon. {cite:t}`Taleghan2015` décrivent la dynamique
complète comme un réseau bayésien dynamique trop complexe pour l'inférence
exacte, et planifient plutôt à partir d'un simulateur. L'exemple illustre tout
de même le point sur lequel s'appuie le reste du livre. L'itération sur les
valeurs et l'itération sur les politiques convergent en un nombre de pas qui
dépend à peine de la taille de la rivière, mais chaque pas touche tous les
états, et le nombre d'états est exponentiel en le nombre de tronçons. Gérer une
vraie rivière exige des fonctions de valeur représentées de façon compacte et
des mises à jour estimées à partir d'échantillons, ce qui est l'objet des
[méthodes de valeur ajustée](approximate-bellman-equations.md) et de
[l'apprentissage à partir de données historiques et en ligne](fitted-q-iteration.md).

- {download}`Modèle de rivière à tamaris et solveurs <code/tamarisk_river.py>`
- {download}`Script de construction de l'expérience sur la rivière à tamaris <scripts/build_tamarisk_river_artifacts.py>`
- {download}`Résultats et temps de calcul pour la rivière à tamaris <artifacts/tamarisk_river/metrics.json>`

## Synthèse et perspectives

L'actualisation transforme une suite infinie de récompenses en une valeur
bornée et fait des opérateurs de Bellman des contractions pour la norme sup.
L'itération sur les valeurs applique la méthode des approximations
successives; l'itération sur les politiques alterne une résolution linéaire à
politique fixée et une amélioration gloutonne, et l'itération sur les
politiques modifiée fait la transition entre les deux en tronquant cette
résolution. Le critère d'arrêt de l'itération sur les valeurs s'accompagne d'un
certificat d'$\varepsilon$-optimalité, et la semi-norme d'étendue en donne un
plus fin. L'exemple de l'ordonnancement de l'inférence a utilisé cette
résolution linéaire comme vérification : aucune politique heuristique n'a battu
la politique de l'itération sur les valeurs dans aucun des 49 états. Il a aussi
montré le prix de la réduction de l'état : un processus de décision markovien
plus petit fait la moyenne sur des distinctions qui peuvent encore influer sur
les terminaisons futures.

Le maximum strict de l'opérateur de Bellman choisit une seule action et n'est
pas différentiable en cas d'égalité. La règle de décision peut-elle rester
stochastique pendant que l'opérateur devient lisse? La
[programmation dynamique régularisée par l'entropie](regularized-dp.md)
répond à cette question en plaçant un régulariseur convexe sur la loi des
actions.

## Autoévaluation

:::{exercise} Facteur de contraction
:label: ex-dp-check-1

Si deux fonctions de valeur diffèrent d'au plus $\varepsilon$ en norme sup, de combien au plus leurs mises à jour de Bellman actualisées peuvent-elles différer?
:::

:::{solution} ex-dp-check-1
:class: dropdown

D'au plus $\gamma\varepsilon$. L'opérateur de Bellman actualisé est une $\gamma$-contraction pour la norme sup.
:::

:::{exercise} Certificat par le résidu
:label: ex-dp-check-2

L'itération sur les valeurs produit un résidu de Bellman $\|Tv-v\|_\infty=0.02$ avec $\gamma=0.9$. Donnez la borne supérieure habituelle de $\|v-v^*\|_\infty$.
:::

:::{solution} ex-dp-check-2
:class: dropdown

$\|v-v^*\|_\infty\leq \|Tv-v\|_\infty/(1-\gamma)=0.02/0.1=0.2$.
:::

:::{exercise} Évaluation ou amélioration
:label: ex-dp-check-3

Quelle étape de l'itération sur les politiques exige la résolution d'un problème linéaire à politique fixée, et quelle étape prend un maximum action par action?
:::

:::{solution} ex-dp-check-3
:class: dropdown

L'évaluation de la politique résout $(I-\gamma P_\pi)v=r_\pi$. L'amélioration de la politique calcule les valeurs d'action à partir de ce $v$ et choisit une action gloutonne dans chaque état.
:::

:::{exercise} État d'ordonnancement réduit
:label: ex-dp-check-4

Donnez une paire d'états de service à l'échelle des requêtes qui correspondent
au même état réduit $(p,d)$, mais qui peuvent avoir des lois différentes des
terminaisons futures.
:::

:::{solution} ex-dp-check-4
:class: dropdown

Deux états peuvent avoir les mêmes nombres de requêtes en attente et en
décodage alors que leurs requêtes actives ont des longueurs de sortie restantes
différentes. L'état réduit écarte ces longueurs; son noyau de transition fait
donc la moyenne sur celles-ci.
:::
