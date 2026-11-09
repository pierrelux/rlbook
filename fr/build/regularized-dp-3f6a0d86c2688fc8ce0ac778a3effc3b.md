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

# Programmation dynamique régularisée par l'entropie

La programmation dynamique à horizon infini choisit les actions au moyen d'un
maximum strict. Cet opérateur n'est pas différentiable en cas d'égalité et
produit un choix glouton déterministe. Un objectif de commande modifié peut-il
produire à la fois un opérateur de Bellman lisse et une politique optimale
stochastique? La régularisation convexe fournit le lien.

Alors que les équations d'optimalité de Bellman usuelles utilisent l'opérateur max pour déterminer la meilleure action, une autre formulation, connue sous le nom d'équations d'optimalité de Bellman lisses (ou « douces »), le remplace par une approximation lisse du maximum. L'histoire de ces équations traverse deux communautés qui se lisent rarement l'une l'autre. Elles apparaissent dans les travaux de l'économètre structurel John Rust sur le remplacement des moteurs d'autobus {cite}`rust1987optimal`, accompagnées d'un problème inverse et d'un algorithme pour le résoudre. {cite:t}`NgRussell2000`, à qui l'on attribue habituellement l'introduction de l'apprentissage par renforcement inverse, citent Rust, mais le lien s'est ensuite largement perdu. {cite:t}`ziebart2008maximum` a redécouvert les mêmes équations sous le nom d'apprentissage par renforcement inverse à entropie maximale, d'après le principe d'entropie maximale. {cite:t}`haarnoja2017reinforcement` se sont appuyés sur Ziebart pour obtenir le *soft Q-learning*, et l'algorithme *soft actor-critic* {cite}`haarnoja2018soft`, l'un des algorithmes d'apprentissage par renforcement profond les plus utilisés pour la commande continue, a suivi un an plus tard.

Dans le cadre à horizon infini, les équations d'optimalité de Bellman lisses prennent la forme :

$$ v^\star_\beta(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp\left(\beta\left(r(s, a) + \gamma \sum_{j \in \mathcal{S}} p(j | s, a) v^\star_\beta(j)\right)\right) $$

On note $v^\star_\beta$ cette fonction de valeur lisse, pour la distinguer de la valeur optimale $v^*_\gamma$ de l'équation de Bellman stricte.

En adoptant le point de vue de la théorie des opérateurs, on peut définir un opérateur non linéaire $\mathrm{L}_\beta$ tel que la fonction de valeur lisse d'un processus de décision markovien (PDM) soit la solution de l'équation de point fixe suivante :

$$ (\mathrm{L}_\beta \mathbf{v})(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right) $$

Lorsque $\beta \to \infty$, $\mathrm{L}_\beta$ converge vers l'opérateur de Bellman usuel $\Bellman$. On peut de plus montrer que l'opérateur de Bellman lisse est une contraction pour la norme du supremum et qu'il possède donc un unique point fixe. Toutefois, contrairement au cadre « strict » habituel, le point fixe de $\mathrm{L}_\beta$ est associé à la fonction de valeur d'une politique stochastique optimale définie par la distribution softmax :

   $$ \pi(a|s) = \frac{\exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v^\star_\beta(j)\right)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a') + \gamma \sum_{j \in \mathcal{S}} p(j|s,a') v^\star_\beta(j)\right)\right)} $$

La terminologie prête à confusion : la politique « softmax » ci-dessus est l'équivalent lisse de l'argmax de l'équation d'optimalité originale, et non du max; c'est un soft-argmax. L'équivalent lisse du max est le log-sum-exp. On appelle aussi cette politique une politique de Boltzmann ou de Gibbs. Lorsque la température $1/\beta$ tend vers zéro, la softmax se concentre sur les actions qui réalisent le maximum, et l'on retrouve la politique gloutonne et l'équation de Bellman stricte.

### Politiques stochastiques : exploration et fonction de partition

Passer de l'argmax à la softmax transforme les politiques déterministes en politiques stochastiques, ce qui est à la fois utile et coûteux. C'est utile parce que le bruit sur les actions est une façon d'explorer. En apprentissage par renforcement, l'agent recueille habituellement ses propres données, et s'il prenait toujours l'action qu'il croit la meilleure, il n'apprendrait jamais rien des autres. Le problème n'est pas propre à l'apprentissage par renforcement. Un ingénieur qui identifie un modèle d'un procédé doit lui aussi choisir des entrées qui révèlent la dynamique; en identification de système, on appelle cela *exciter* le système : une entrée persistante d'excitation est une entrée assez riche pour que les paramètres soient identifiables à partir des données.

C'est coûteux lorsque l'ensemble des actions est continu. La politique softmax est alors une densité $\pi(a \mid s) \propto \exp(\beta q(s,a))$ dont la constante de normalisation, la fonction de partition $\int \exp(\beta q(s,a))\, da$, est une intégrale qui n'a en général pas de forme analytique. La valeur log-sum-exp pose le même problème, puisqu'elle est le logarithme de cette intégrale. C'est la fonction de partition intraitable bien connue des modèles à base d'énergie et des modèles graphiques probabilistes. La quadrature numérique est envisageable en faible dimension, mais on l'utilise rarement. {cite:t}`haarnoja2017reinforcement` échantillonnaient la politique softmax sur des actions continues au moyen d'une descente de gradient variationnelle de Stein amortie, ce qui s'est révélé coûteux; *soft actor-critic* {cite}`haarnoja2018soft` ajuste plutôt une politique paramétrique traitable à la softmax, une idée sur laquelle nous reviendrons dans les chapitres sur les méthodes de valeur ajustées et l'apprentissage des politiques. La notation pour les actions continues, et ce qui change lorsque les sommes deviennent des intégrales, se trouvent à la section {ref}`sec-continuous-actions`.



La formulation lisse présente deux avantages. D'abord, il est plus facile d'optimiser à travers un opérateur lisse. Contrairement à $\gamma$, on considère $\beta$ comme un hyperparamètre de l'algorithme, que l'on peut régler pour atteindre le niveau de précision voulu.

Ensuite, nous avons présenté les équations de façon pragmatique, en remplaçant le max par le log-sum-exp (un maximum lisse) et l'argmax par la softmax (un argmax lisse). Les mêmes équations découlent de deux autres points de départ, et ce chapitre suit les trois voies :

1. **Lisser l'opérateur.** Remplacer chaque max par un log-sum-exp, comme ci-dessus. C'est la voie de la sous-section suivante, et sa justification est computationnelle : un opérateur lisse a des gradients partout.
2. **Du bruit sur les récompenses.** Conserver l'équation de Bellman stricte, mais supposer que la récompense de chaque action est perturbée par un bruit de Gumbel indépendant que le décideur observe et que le modélisateur n'observe pas. L'intégration du bruit donne le log-sum-exp ({ref}`sec-gumbel-rewards`). C'est ainsi que {cite:t}`rust1987optimal` a motivé les équations, dans le sillage des modèles de choix discrets de l'économétrie.
3. **La régularisation par l'entropie.** Conserver les récompenses, mais ajouter dans chaque état un bonus proportionnel à l'entropie de la politique. L'équation de Bellman régularisée est de nouveau l'équation log-sum-exp ({ref}`sec-regularized-mdps`). C'est la lecture qui sous-tend l'apprentissage par renforcement à entropie maximale et *soft actor-critic*.

Chaque voie apporte ses propres outils : le lissage donne la différentiabilité, le modèle de bruit donne une vraisemblance pour le comportement observé, et le point de vue de la régularisation donne la dualité convexe. Une quatrième lecture, proche de la troisième, part des modèles à base d'énergie de l'apprentissage supervisé, où une softmax transforme des scores non normalisés en une distribution; c'est essentiellement ce qu'a fait {cite:t}`ziebart2008maximum`. Elle se rattache aux modèles graphiques probabilistes, où trouver une politique lisse optimale devient un problème d'inférence. C'est l'idée de la commande comme inférence, dont la réciproque, l'inférence comme commande, sert aujourd'hui à dériver des échantillonneurs et des méthodes d'inférence amortie au moyen de l'apprentissage par renforcement {cite}`levine2018reinforcement`.

### Algorithme d'itération sur les valeurs lisse

Avant d'itérer $\mathrm{L}_\beta$, il faut vérifier qu'il est légitime de l'itérer, c'est-à-dire qu'il s'agit d'une contraction. C'en est une, pour la norme sup et avec le même facteur $\gamma$ que l'opérateur de Bellman strict. La preuve utilise les trois mêmes faits que pour $\Bellman$. Le log-sum-exp est monotone, il commute avec l'ajout d'une constante à tous ses arguments, et $\mathrm{L}_\beta$ envoie une translation uniforme $c\mathbf{1}$ de $v$ sur une translation $\gamma c\mathbf{1}$ de $\mathrm{L}_\beta v$. De $v \le u + \|v-u\|_\infty\mathbf{1}$, on tire $\mathrm{L}_\beta v \le \mathrm{L}_\beta u + \gamma\|v-u\|_\infty\mathbf{1}$, et l'échange de $u$ et de $v$ donne la contraction. Une mise en garde au sujet de la réciproque : ne pas réussir à montrer qu'un opérateur est une contraction ne signifie pas que son point fixe n'existe pas ni qu'aucun algorithme ne le trouve. La contraction est une condition suffisante commode, et d'autres théorèmes de point fixe couvrent d'autres cas.

Les approximations successives appliquées à $\mathrm{L}_\beta$ constituent l'**itération sur les valeurs lisse**. Elle a la structure de l'itération sur les valeurs, avec un log-sum-exp partout où l'itération sur les valeurs avait un max, et une politique softmax à la fin là où l'itération sur les valeurs renvoyait une politique gloutonne.

```{prf:algorithm} Itération sur les valeurs lisse
:label: smooth-value-iteration

**Entrée :** PDM $(\mathcal{S}, \mathcal{A}, r, p, \gamma)$, inverse de la température $\beta > 0$, tolérance $\varepsilon > 0$

**Sortie :** Fonction de valeur $v$ à moins de $\varepsilon/2$ de $v^\star_\beta$ et politique stochastique $\pi$

1. Initialiser $v(s) \leftarrow 0$ pour tout $s \in \mathcal{S}$
2. **répéter**
3. $\quad$ **pour** chaque état $s \in \mathcal{S}$ et chaque action $a \in \mathcal{A}_s$ **faire**
4. $\quad\quad q(s,a) \leftarrow r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)$
5. $\quad$ **fin pour**
6. $\quad v_{\text{nouv}}(s) \leftarrow \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp(\beta \cdot q(s,a))$ pour tout $s \in \mathcal{S}$
7. $\quad \Delta \leftarrow \max_s |v_{\text{nouv}}(s) - v(s)|$
8. $\quad v \leftarrow v_{\text{nouv}}$
9. **jusqu'à ce que** $\Delta < \frac{\varepsilon(1-\gamma)}{2\gamma}$
10. Extraire la politique : calculer $q(s,a)$ à partir de $v$ comme à la ligne 4, et poser $\pi(a|s) \leftarrow \frac{\exp(\beta \cdot q(s,a))}{\sum_{a' \in \mathcal{A}_s} \exp(\beta \cdot q(s,a'))}$
11. **renvoyer** $v, \pi$
```

**Différences avec l'itération sur les valeurs usuelle :**
- La ligne 6 utilise $\frac{1}{\beta} \log \sum_a \exp(\beta \cdot q(s,a))$ au lieu de $\max_a q(s,a)$
- La ligne 10 extrait une politique stochastique au moyen de la softmax au lieu d'une politique argmax déterministe
- Lorsque $\beta \to \infty$, l'algorithme converge vers l'itération sur les valeurs usuelle
- Des valeurs de $\beta$ plus faibles produisent des politiques plus stochastiques, d'entropie plus élevée

**Ce que garantit la règle d'arrêt.** Comme $\mathrm{L}_\beta$ est une $\gamma$-contraction, la borne a posteriori du théorème du point fixe de Banach s'applique sans changement, et le test d'arrêt de la ligne 9 garantit $\|v - v^\star_\beta\|_\infty \le \varepsilon/2$. La preuve selon laquelle l'itération sur les valeurs renvoie une politique $\varepsilon$-optimale se transpose aussi, avec un changement de sens. La politique softmax satisfait $\mathrm{L}_{\pi,\Omega} v = \mathrm{L}_\beta v$ pour l'opérateur d'évaluation régularisé par l'entropie de la section {ref}`sec-regularized-operators`, avec le régularisateur $\Omega(\pi(\cdot|s)) = \frac{1}{\beta}\sum_a \pi(a|s)\log\pi(a|s)$, tout comme une politique gloutonne satisfait $\mathrm{L}_\pi v = \Bellman v$. En reprenant la preuve de {prf:ref}`value-iteration-convergence` avec ces opérateurs, on montre que la politique renvoyée est $\varepsilon$-optimale *pour l'objectif régularisé par l'entropie*. À quelle distance de l'optimum pour les récompenses originales se trouve-t-elle? Puisque $\max_a x_a \le \frac{1}{\beta}\log\sum_a e^{\beta x_a} \le \max_a x_a + \frac{\log|\mathcal{A}|}{\beta}$, où $|\mathcal{A}|$ est le plus grand nombre d'actions dans un état, les deux points fixes satisfont

$$
v^*_\gamma \;\le\; v^\star_\beta \;\le\; v^*_\gamma + \frac{\log |\mathcal{A}|}{\beta(1-\gamma)}\mathbf{1},
$$

et la politique softmax renvoyée par l'algorithme perd au plus $\varepsilon + \frac{\log|\mathcal{A}|}{\beta(1-\gamma)}$ par rapport à la valeur optimale du PDM non régularisé. La température ajoute donc un biais qui ne diminue pas avec davantage d'itérations; c'est le prix du lissage et de l'exploration.

(sec-gumbel-rewards)=
## Bruit de Gumbel sur les récompenses

Pourquoi l'ajout de perturbations de Gumbel indépendantes aux récompenses des
actions produit-il une valeur log-sum-exp avant que la perturbation soit
observée?

On peut obtenir l'équation de Bellman lisse en considérant un cadre dans lequel un bruit de Gumbel s'ajoute à la fonction de récompense. Cette dérivation apporte un éclairage théorique et se rattache aussi à des situations de modélisation concrètes où les récompenses subissent des perturbations aléatoires.

### Étape 1 : définir le PDM augmenté par un bruit de Gumbel

À chaque période et dans chaque état $s$, on tire un **vecteur de chocs indexé par les actions** :

$$\boldsymbol{\epsilon}_t(s) = \big(\epsilon_t(s,a)\big)_{a \in \mathcal{A}_s}, \quad \text{où } \epsilon_t(s,a) \text{ i.i.d.} \sim \mathrm{Gumbel}(\mu_\epsilon, 1/\beta)$$

Ces chocs sont indépendants d'une période, d'un état et d'une action à l'autre, et indépendants de la dynamique de transition du PDM $p(\cdot | s, a)$.

La **distribution de Gumbel** de paramètre de position $\mu$ et de paramètre d'échelle $1/\beta$ a pour densité de probabilité :

$$ f(x; \mu, 1/\beta) = \beta\exp\left(-\beta(x-\mu)-\exp(-\beta(x-\mu))\right) $$

Pour générer une variable aléatoire de loi de Gumbel, on peut utiliser la méthode d'inversion : $X = \mu - \frac{1}{\beta} \log(-\log(U))$, où $U$ est uniforme sur $(0,1)$.

```{admonition} Chocs de moyenne nulle
:class: tip
Pour que les chocs soient de moyenne nulle, on pose $\mu_\epsilon = -\gamma_E/\beta$, où $\gamma_E \approx 0.5772$ est la constante d'Euler-Mascheroni. Ce choix élimine une constante additive qui apparaîtrait sinon dans l'équation de Bellman lisse. Par souci de simplicité, nous adopterons cette convention partout.
```

On définit maintenant un **PDM augmenté** avec :
- **État augmenté** : $\tilde{s} = (s, \boldsymbol{\epsilon})$, où $s \in \mathcal{S}$ et $\boldsymbol{\epsilon} \in \mathbb{R}^{|\mathcal{A}_s|}$
- **Récompense augmentée** : $\tilde{r}(\tilde{s}, a) = r(s,a) + \epsilon(a)$
- **Transition augmentée** : $\tilde{p}(\tilde{s}' | \tilde{s}, a) = p(s' | s, a) \cdot p(\boldsymbol{\epsilon}')$

La transition se factorise parce que le vecteur de chocs suivant $\boldsymbol{\epsilon}'$ est tiré indépendamment de l'état et de l'action courants (indépendance conditionnelle).

```{admonition} L'espace d'états augmenté n'est pas dénombrable
:class: warning
Même si l'espace d'états original $\mathcal{S}$ et l'espace d'actions $\mathcal{A}$ sont finis, l'espace d'états augmenté $\tilde{\mathcal{S}} = \mathcal{S} \times \mathbb{R}^{|\mathcal{A}|}$ est **infini non dénombrable**, car chaque vecteur de chocs $\boldsymbol{\epsilon}$ est une variable aléatoire continue. L'espace est de dimension finie, avec $|\mathcal{A}|$ coordonnées pour les chocs, mais une fonction de valeur définie sur cet espace est un élément d'un espace fonctionnel de dimension infinie. Les chocs ne sont pas non plus bornés; les récompenses $r(s,a)+\epsilon(a)$ ne sont donc pas bornées, et la théorie en norme sup du chapitre précédent ne s'applique pas telle quelle. Rien ne se brise, car les chocs ont une moyenne finie et interviennent de façon additive, mais c'est une raison de plus de ne pas travailler directement sur cet espace. Par conséquent :
- On ne peut pas énumérer les états augmentés
- Les méthodes de programmation dynamique tabulaires ne s'appliquent pas directement
- La fonction de valeur augmentée $\tilde{v}(s, \boldsymbol{\epsilon})$ envoie un espace continu dans $\mathbb{R}$

**C'est ce qui justifie de marginaliser immédiatement sur les chocs** pour obtenir une représentation de dimension finie.
```

### Étape 2 : l'équation de Bellman stricte sur l'espace d'états augmenté

L'équation d'optimalité de Bellman du PDM augmenté est :

$$ \tilde{v}(s, \boldsymbol{\epsilon}) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \epsilon(a) + \gamma \mathbb{E}_{s', \boldsymbol{\epsilon}'}\left[\tilde{v}(s', \boldsymbol{\epsilon}') \mid s, a\right] \right\} $$

L'espérance porte ici sur le **prochain état augmenté** $(s', \boldsymbol{\epsilon}')$, qui comprend à la fois l'état suivant $s' \sim p(\cdot | s, a)$ et le vecteur de chocs suivant $\boldsymbol{\epsilon}' \sim p(\cdot)$.

C'est une équation de Bellman parfaitement bien définie, et il existe une politique stationnaire optimale :

$$\pi(s, \boldsymbol{\epsilon}) \in \operatorname{argmax}_{a \in \mathcal{A}_s} \left\{ r(s,a) + \epsilon(a) + \gamma \mathbb{E}_{s', \boldsymbol{\epsilon}'}\left[\tilde{v}(s', \boldsymbol{\epsilon}') \mid s, a\right] \right\}$$

Cette équation est toutefois **intraitable numériquement** parce que :
- L'espace d'états est continu (non dénombrable); $\tilde v$ est donc une fonction définie sur un continuum
- Les chocs sont renouvelés à chaque période
- Il faudrait résoudre pour $\tilde{v}$ sur un domaine non dénombrable

**On ne résout jamais cette équation directement.** On s'en sert plutôt comme d'un outil mathématique pour dériver l'équation de Bellman lisse.

### Étape 3 : définir la fonction de valeur ex ante (inclusive)

L'idée est ici de considérer la **valeur espérée avant d'observer les chocs courants**. On définit ce que certains auteurs en économétrie appellent la **valeur inclusive** ou **valeur ex ante** :

$$ v(s) := \mathbb{E}_{\boldsymbol{\epsilon}}\big[\tilde{v}(s, \boldsymbol{\epsilon})\big] $$

C'est la valeur d'être dans l'état $s$ **avant** d'observer le vecteur de chocs $\boldsymbol{\epsilon}$ de la période courante.

```{admonition} Deux fonctions de valeur différentes
:class: note
Il est essentiel de distinguer :
- $\tilde{v}(s, \boldsymbol{\epsilon})$ : la valeur **après** l'observation des chocs (conditionnelle à $\boldsymbol{\epsilon}$), définie sur l'espace d'états augmenté
- $v(s)$ : la valeur **avant** l'observation des chocs (marginalisée sur $\boldsymbol{\epsilon}$), définie sur l'espace d'états original

La fonction $v(s)$ est celle que l'on calcule réellement et qui nous intéresse. La valeur augmentée $\tilde{v}$ n'existe que comme outil de preuve.
```

### Étape 4 : séparer les composantes déterministe et aléatoire

On prend maintenant l'espérance de l'équation de Bellman augmentée par rapport aux **seuls chocs courants** (tout ce qui ne dépend pas du $\boldsymbol{\epsilon}$ courant peut sortir de l'espérance).

D'abord, par la loi des espérances itérées et l'indépendance des chocs dans le temps :

$$ \mathbb{E}_{\boldsymbol{\epsilon}'}\big[\tilde{v}(s', \boldsymbol{\epsilon}')\big] = v(s') $$

Cela découle de la définition de $v$ et du fait que le choc suivant est indépendant de tout le reste.

Définissons maintenant la **partie déterministe** du membre de droite :

$$ x_a(s) := r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) $$

C'est le rendement espéré lorsqu'on prend l'action $a$ dans l'état $s$, **sans le choc**. Avec cette notation, l'équation de Bellman augmentée devient :

$$ \tilde{v}(s, \boldsymbol{\epsilon}) = \max_{a \in \mathcal{A}_s} \left\{ x_a(s) + \epsilon(a) \right\} $$

En prenant l'espérance sur $\boldsymbol{\epsilon}$ des deux côtés :

$$ v(s) = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\max_{a \in \mathcal{A}_s} \left\{ x_a(s) + \epsilon(a) \right\}\right] $$

```{admonition} Espérance d'un max, et non max d'une espérance
:class: important
Remarquez bien : on a $\mathbb{E}[\max(\cdot)]$, **et non** $\max \mathbb{E}[\cdot]$. On **n'échange pas** le max et l'espérance.

L'expression $\mathbb{E}_{\boldsymbol{\epsilon}}[\max_a \{x_a + \epsilon(a)\}]$ est l'espérance du maximum d'utilités perturbées par un bruit de Gumbel. L'identité d'utilité aléatoire de Gumbel évalue cette quantité sous forme analytique.
```

### Étape 5 : appliquer l'identité d'utilité aléatoire de Gumbel

On invoque maintenant un résultat de la théorie des valeurs extrêmes :

```{prf:lemma} Identité d'utilité aléatoire de Gumbel
:label: gumbel-random-utility

Soit $\epsilon_1, \ldots, \epsilon_m$ des variables aléatoires i.i.d. de loi $\mathrm{Gumbel}(\mu_\epsilon, 1/\beta)$. Pour toutes valeurs déterministes $x_1, \ldots, x_m \in \mathbb{R}$ :

$$ \max_{i=1,\ldots,m} \{x_i + \epsilon_i\} \overset{d}{=} \frac{1}{\beta} \log \sum_{i=1}^m \exp(\beta x_i) + \zeta $$

où $\zeta \sim \mathrm{Gumbel}(\mu_\epsilon, 1/\beta)$ (même distribution que les chocs originaux).

En prenant l'espérance :

$$ \mathbb{E}\left[\max_{i=1,\ldots,m} \{x_i + \epsilon_i\}\right] = \frac{1}{\beta} \log \sum_{i=1}^m \exp(\beta x_i) + \mu_\epsilon + \frac{\gamma_E}{\beta} $$

où $\gamma_E \approx 0.5772$ est la constante d'Euler-Mascheroni.

**Avec des chocs de moyenne nulle** ($\mu_\epsilon = -\gamma_E/\beta$), le terme constant disparaît :

$$ \mathbb{E}\left[\max_{i=1,\ldots,m} \{x_i + \epsilon_i\}\right] = \frac{1}{\beta} \log \sum_{i=1}^m \exp(\beta x_i) $$
```

En appliquant cette identité à notre problème (avec des chocs de moyenne nulle) :

$$ v(s) = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\max_{a \in \mathcal{A}_s} \{x_a(s) + \epsilon(a)\}\right] = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp(\beta x_a(s)) $$

En substituant la définition de $x_a(s)$ :

$$ v(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right) $$

Nous sommes arrivés à l'**équation de Bellman lisse**.

### Étape 6 : récapitulatif de la dérivation

Pour récapituler l'enchaînement logique :

1. On a construit un PDM augmenté d'état $(s, \boldsymbol{\epsilon})$, dans lequel des chocs perturbent les récompenses
2. On a écrit l'équation de Bellman usuelle de ce PDM augmenté (max strict, mais sur un espace d'états non dénombrable)
3. On a défini la valeur ex ante $v(s) = \mathbb{E}_{\boldsymbol{\epsilon}}[\tilde{v}(s, \boldsymbol{\epsilon})]$ pour éliminer la composante continue des chocs
4. On a séparé les termes déterministes et aléatoires : $\tilde{v}(s, \boldsymbol{\epsilon}) = \max_a \{x_a(s) + \epsilon(a)\}$
5. On a appliqué l'identité de Gumbel pour évaluer $\mathbb{E}_{\boldsymbol{\epsilon}}[\max_a \{\cdots\}]$ sous forme analytique, comme un log-sum-exp

Le PDM augmenté par des chocs n'existe **que comme outil mathématique**. On n'approche jamais $\tilde{v}$, on ne discrétise jamais $\boldsymbol{\epsilon}$ et on n'énumère jamais l'espace d'états augmenté. Le seul objet de calcul que l'on manipule est $v(s)$ sur l'espace d'états original (fini), qui satisfait l'équation de Bellman lisse.

### Dérivation de la politique lisse optimale

Maintenant que nous avons dérivé la fonction de valeur lisse, nous pouvons aussi obtenir la politique optimale correspondante. La question est la suivante : **quelle politique suivre dans le PDM original (sans conditionner explicitement sur les chocs)?**

Dans le PDM augmenté, la politique optimale est déterministe, mais dépend de la réalisation des chocs :

$$\pi(s, \boldsymbol{\epsilon}) \in \operatorname{argmax}_{a \in \mathcal{A}_s} \left\{ r(s,a) + \epsilon(a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) \right\}$$

On veut toutefois une politique pour l'espace d'états **original** $s$ (et non pour l'état augmenté). On l'obtient en **marginalisant sur les chocs courants**, ce qui revient à se demander : « quelle est la probabilité que l'action $a$ soit optimale lorsqu'on fait la moyenne sur toutes les réalisations possibles des chocs? »

Définissons une fonction indicatrice :

$$ I_a(\boldsymbol{\epsilon}) = \begin{cases} 
   1 & \text{si } a \in \operatorname{argmax}_{a' \in \mathcal{A}_s} \left\{ x_{a'}(s) + \epsilon(a') \right\} \\
   0 & \text{sinon}
   \end{cases} $$

où $x_a(s) = r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)$ comme précédemment.

La probabilité ex ante que l'action $a$ soit optimale dans l'état $s$ est :

$$ \pi(a|s) = \mathbb{E}_{\boldsymbol{\epsilon}}[I_a(\boldsymbol{\epsilon})] = \mathbb{P}_{\boldsymbol{\epsilon}}\left(a \in \operatorname{argmax}_{a'} \left\{ x_{a'}(s) + \epsilon(a') \right\}\right) $$

C'est la probabilité que l'action $a$ réalise le maximum lorsque les utilités sont perturbées par un bruit de Gumbel.

```{prf:lemma} Probabilité Gumbel-max (softmax)
:label: gumbel-softmax

Soit $\epsilon_1, \ldots, \epsilon_m$ des variables aléatoires i.i.d. de loi $\mathrm{Gumbel}(\mu_\epsilon, 1/\beta)$. Pour toutes valeurs déterministes $x_1, \ldots, x_m \in \mathbb{R}$, la probabilité que l'indice $i$ réalise le maximum est :

$$ \mathbb{P}\left(i \in \operatorname{argmax}_j \{x_j + \epsilon_j\}\right) = \frac{\exp(\beta x_i)}{\sum_{j=1}^m \exp(\beta x_j)} $$

Ce résultat vaut quel que soit le paramètre de position $\mu_\epsilon$.
```

En appliquant ce résultat à notre problème :

$$ \pi(a|s) = \frac{\exp\left(\beta x_a(s)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta x_{a'}(s)\right)} = \frac{\exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v(j)\right)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a') + \gamma \sum_{j \in \mathcal{S}} p(j|s,a')v(j)\right)\right)} $$

C'est la **politique softmax**, ou **politique de Gibbs/Boltzmann**, d'inverse de la température $\beta$.

**Propriétés :**
- Lorsque $\beta \to \infty$ : la politique devient déterministe et se concentre sur la ou les actions de plus grand $x_a(s)$ (on retrouve la politique gloutonne usuelle)
- Lorsque $\beta \to 0$ : la politique devient uniforme sur toutes les actions (entropie maximale)
- Pour $\beta > 0$ fini : la politique est stochastique, et la masse de probabilité est proportionnelle à l'exponentielle des valeurs Q

Cela complète la dérivation : l'équation de Bellman lisse fournit une fonction de valeur $v(s)$, et la politique optimale correspondante est la softmax des valeurs Q.

(sec-gumbel-side-by-side)=
### Le PDM original et le PDM augmenté côte à côte

La dérivation passe d'un problème de décision à l'autre. Le premier est le PDM original résolu avec l'équation de Bellman lisse, ou de façon équivalente avec un bonus d'entropie ({ref}`sec-regularized-mdps`). Le second est le PDM augmenté, dans lequel le décideur résout une équation de Bellman stricte, mais observe un nouveau choc sur la récompense de chaque action. Le tableau les place l'un à côté de l'autre. On note $q(s,a) = r(s,a) + \gamma\sum_{j} p(j\mid s,a)\, v^\star_\beta(j)$, qui est le $x_a(s)$ de la dérivation.

| | PDM original, équation de Bellman lisse | PDM augmenté, chocs de Gumbel |
|---|---|---|
| État | $s \in \mathcal{S}$ | $(s, \boldsymbol{\epsilon}) \in \mathcal{S} \times \mathbb{R}^{\lvert\mathcal{A}_s\rvert}$ |
| Bruit | seulement les transitions $p(j \mid s,a)$ | les transitions, plus un nouveau vecteur de Gumbel i.i.d. $\boldsymbol{\epsilon}_t$ à chaque période |
| Récompense de l'action $a$ | $r(s,a)$, plus le bonus $\beta^{-1} H(\pi(\cdot \mid s))$ | $r(s,a) + \epsilon(a)$ |
| Transition | $p(j \mid s,a)$ | $p(j \mid s,a)\, g(\boldsymbol{\epsilon}')$, où $g$ est la densité de Gumbel |
| Équation de Bellman | $v = \mathrm{L}_\beta v$, un log-sum-exp | $\tilde v(s,\boldsymbol{\epsilon}) = \max_a \{q(s,a) + \epsilon(a)\}$, un max strict |
| Politique optimale | stochastique, $\pi_\beta(a \mid s) \propto \exp(\beta q(s,a))$ | déterministe, $\sigma(s,\boldsymbol{\epsilon}) = \operatorname{argmax}_a \{q(s,a) + \epsilon(a)\}$ |
| Valeur | $v^\star_\beta(s)$ | $\tilde v(s,\boldsymbol{\epsilon})$, avec $\mathbb{E}_{\boldsymbol{\epsilon}}[\tilde v(s,\boldsymbol{\epsilon})] = v^\star_\beta(s)$ |
| Qui observe quoi | l'agent et le modélisateur observent tous deux $s$ | l'agent observe $s$ et $\boldsymbol{\epsilon}$; le modélisateur n'observe que $s$ |

La {numref}`fig-gumbel-augmented-pgm` représente les deux problèmes sous forme de modèles graphiques, une tranche de temps après l'autre.

:::{figure} _static/gumbel-augmented-pgm.svg
:label: fig-gumbel-augmented-pgm
:alt: Deux modèles graphiques côte à côte. À gauche, des états s_t et des actions a_t, avec une flèche de chaque état vers son action et des flèches de l'état et de l'action vers l'état suivant. À droite, le même graphe avec un nœud supplémentaire en pointillés epsilon_t au-dessus de chaque état, une flèche de epsilon_t vers a_t, et a_t dessiné comme un double cercle. Des cadres pointillés regroupent s_t et epsilon_t.

(a) Le PDM original sous la politique softmax. L'action $a_t$ est un enfant aléatoire de $s_t$, de distribution conditionnelle $\pi_\beta(\cdot \mid s_t)$. (b) Le PDM augmenté. Un vecteur de chocs $\boldsymbol{\epsilon}_t$ (en tirets) est tiré indépendamment à chaque période, et l'état augmenté est $(s_t, \boldsymbol{\epsilon}_t)$ (cadres pointillés). L'action est une fonction déterministe de l'état augmenté (double cercle). Le choc n'a aucune flèche vers $s_{t+1}$ ni vers $\boldsymbol{\epsilon}_{t+1}$. Intégrer $\boldsymbol{\epsilon}_t$ dans chaque tranche de (b) donne (a).
:::

La figure montre pourquoi la construction fonctionne. Dans le panneau (b), $\boldsymbol{\epsilon}_t$ a un seul enfant, l'action $a_t$, et l'action est une fonction déterministe de $(s_t, \boldsymbol{\epsilon}_t)$. Intégrer le choc dans la tranche $t$ ne modifie donc que la distribution conditionnelle de $a_t$ :

$$
\mathbb{P}(a_t = a \mid s_t) = \int \mathbb{1}\{a = \sigma(s_t, \boldsymbol{\epsilon})\}\, g(\boldsymbol{\epsilon})\, d\boldsymbol{\epsilon} = \pi_\beta(a \mid s_t),
$$

par le {prf:ref}`gumbel-softmax`. En faisant de même dans chaque tranche, on transforme le graphe (b) en graphe (a) : les deux modèles donnent la même distribution sur les trajectoires état-action $(s_0, a_0, s_1, a_1, \ldots)$. Les flèches absentes comptent. Si $\boldsymbol{\epsilon}_t$ influençait aussi $s_{t+1}$, ou si les chocs étaient corrélés dans le temps, leur intégration laisserait une dépendance au passé, et le processus marginal ne serait plus un PDM en $s$. {cite:t}`rust1987optimal` énonce ce point comme une hypothèse d'*indépendance conditionnelle*, et son estimateur en dépend.

Les récompenses ne sont pas égales trajectoire par trajectoire. L'agent augmenté recueille $r(s_t,a_t) + \epsilon_t(a_t)$, alors que le PDM original verse $r(s_t,a_t)$ et le bonus d'entropie. Elles sont égales en moyenne. Conditionnellement à l'action gagnante, la valeur gagnante $\max_a\{q(s,a)+\epsilon(a)\}$ a la même distribution quelle que soit l'action qui l'emporte, une propriété de la distribution de Gumbel. Sa moyenne est $\frac{1}{\beta}\log\sum_{a'} \exp(\beta q(s,a'))$ par le {prf:ref}`gumbel-random-utility`, de sorte que

$$
\mathbb{E}\big[\epsilon(a) \,\big|\, \sigma(s,\boldsymbol{\epsilon}) = a\big] = \frac{1}{\beta}\log\sum_{a'} e^{\beta q(s,a')} - q(s,a) = -\frac{1}{\beta}\log \pi_\beta(a \mid s).
$$

Le choc que l'agent recueille sur l'action qu'il choisit vaut, en moyenne, $-\frac{1}{\beta}\log\pi_\beta(a\mid s)$. La moyenne sur $a \sim \pi_\beta(\cdot\mid s)$ donne $\frac{1}{\beta}H(\pi_\beta(\cdot \mid s))$, le bonus d'entropie. La pénalité par action $-\alpha \log \pi(a \mid s)$ qui apparaît plus loin dans la formulation régularisée est ce choc espéré, avec $\alpha = 1/\beta$. Une action rarement choisie ne l'est que lorsque son choc est grand, et le bonus d'entropie est la moyenne de ces grands chocs.

### Vérification numérique de l'équivalence

Le PDM augmenté est un outil théorique : il explique d'où vient le log-sum-exp et fournit une vraisemblance pour les choix observés, mais personne ne le résout pour calculer une politique. Sur un petit problème, on peut tout de même le résoudre par force brute et vérifier qu'il donne la même réponse que le PDM original. La cellule ci-dessous tire un PDM aléatoire à 5 états et 3 actions, fixe $\gamma = 0.9$ et $\beta = 2$, et calcule la solution de trois façons.

1. *PDM original.* Itération sur les valeurs lisse avec le log-sum-exp, puis la politique softmax.
2. *PDM original, lecture régularisée.* La valeur de cette politique softmax avec son bonus d'entropie, obtenue par le système linéaire $(\mathbf{I}-\gamma\mathbf{P}_\pi)\mathbf{v} = \mathbf{r}_\pi + \beta^{-1}\mathbf{H}_\pi$.
3. *PDM augmenté.* Itération sur les valeurs avec le max strict sur la valeur ex ante, $v(s) \leftarrow \mathbb{E}_{\boldsymbol{\epsilon}}[\max_a\{r(s,a)+\gamma\sum_j p(j\mid s,a)v(j)+\epsilon(a)\}]$, où l'espérance est remplacée par une moyenne sur 50 000 tirages de Gumbel, sans aucun log-sum-exp. On simule ensuite 20 000 épisodes dans lesquels l'agent observe de nouveaux chocs à chaque pas, prend l'action déterministe $\operatorname{argmax}_a\{q(s,a)+\epsilon(a)\}$ et recueille $r + \epsilon(a)$.

```{code-cell} python
import numpy as np

rng = np.random.default_rng(0)
n_states, n_actions, gamma, beta = 5, 3, 0.9, 2.0
P = rng.dirichlet(0.5 * np.ones(n_states), size=(n_states, n_actions))  # P[s, a, j]
r = rng.normal(size=(n_states, n_actions))
loc, scale = -np.euler_gamma / beta, 1 / beta  # mean-zero Gumbel shocks


def smooth_max(q):
    """(1/beta) log sum_a exp(beta q), computed stably along the last axis."""
    m = q.max(axis=-1)
    return m + np.log(np.exp(beta * (q - m[..., None])).sum(axis=-1)) / beta


# Original MDP: smooth value iteration, then the softmax policy.
v = np.zeros(n_states)
for _ in range(500):
    v = smooth_max(r + gamma * P @ v)
q = r + gamma * P @ v
pi = np.exp(beta * (q - v[:, None]))
entropy = -(pi * np.log(pi)).sum(axis=1)

# Original MDP, regularized reading: evaluate pi with its entropy bonus,
# (I - gamma P_pi) v = r_pi + H_pi / beta.
P_pi = np.einsum("sa,saj->sj", pi, P)
v_reg = np.linalg.solve(np.eye(n_states) - gamma * P_pi, (pi * r).sum(axis=1) + entropy / beta)

# Augmented MDP: hard Bellman equation, ex-ante value. The expectation over the
# current shocks is an average over n_draws fixed Gumbel draws; no log-sum-exp.
n_draws = 50_000
shocks = rng.gumbel(loc, scale, size=(n_draws, n_states, n_actions))
v_aug = np.zeros(n_states)
for _ in range(150):
    v_aug = (r + gamma * P @ v_aug + shocks).max(axis=2).mean(axis=0)
q_aug = r + gamma * P @ v_aug

# Simulate the augmented MDP: the agent sees fresh shocks every step and takes
# the deterministic action argmax_a q_aug(s, a) + eps(a).
n_episodes, horizon = 20_000, 150
s = np.zeros(n_episodes, dtype=int)
returns = np.zeros(n_episodes)
visits = np.zeros((n_states, n_actions))
shock_total = np.zeros((n_states, n_actions))
cdf = P.cumsum(axis=2)
for t in range(horizon):
    eps = rng.gumbel(loc, scale, size=(n_episodes, n_actions))
    a = (q_aug[s] + eps).argmax(axis=1)
    eps_a = eps[np.arange(n_episodes), a]
    returns += gamma**t * (r[s, a] + eps_a)
    np.add.at(visits, (s, a), 1)
    np.add.at(shock_total, (s, a), eps_a)
    s = (rng.random(n_episodes)[:, None] > cdf[s, a]).sum(axis=1)
freq = visits / visits.sum(axis=1, keepdims=True)
mean_shock = shock_total / visits

fmt = lambda x: " ".join(f"{y:6.3f}" for y in x)
lines = [
    "Values v(s)",
    f"  original, smooth value iteration        {fmt(v)}",
    f"  original, entropy-regularized eval      {fmt(v_reg)}",
    f"  augmented, hard max over Gumbel draws   {fmt(v_aug)}",
    f"  augmented, simulated return from s=0    {returns.mean():6.3f} +/- {2 * returns.std() / np.sqrt(n_episodes):.3f}",
    "",
    "Policy: softmax pi(a|s) vs. action frequencies of the augmented agent",
    "  s |  softmax pi(.|s)      |  frequency in augmented MDP",
]
lines += [f"  {i} | {fmt(pi[i])} | {fmt(freq[i])}" for i in range(n_states)]
lines += [
    "",
    "Shocks: -log pi(a|s)/beta vs. mean shock eps(a) when a is chosen",
    "  s |  -log pi(.|s)/beta    |  mean chosen shock",
]
lines += [f"  {i} | {fmt(-np.log(pi[i]) / beta)} | {fmt(mean_shock[i])}" for i in range(n_states)]
lines += [
    "",
    f"  entropy bonus H/beta per state     {fmt(entropy / beta)}",
    f"  mean chosen shock per state        {fmt(shock_total.sum(1) / visits.sum(1))}",
]
print("\n".join(lines))
```

Les trois fonctions de valeur concordent à l'erreur de Monte-Carlo près, qui reste inférieure à $0.01$ pour des valeurs voisines de $10$, et le rendement simulé de l'agent augmenté à partir de l'état $0$ correspond à $v^\star_\beta(0)$ à l'intérieur de son intervalle de confiance. L'agent augmenté ne randomise jamais, et pourtant les fréquences de ses actions reproduisent les probabilités softmax à la troisième décimale près. Les deux derniers blocs vérifient l'identité sur les chocs : le choc moyen sur une action choisie correspond à $-\beta^{-1}\log\pi_\beta(a\mid s)$, et sa moyenne par état correspond au bonus d'entropie. Le coût explique pourquoi la voie de Gumbel reste théorique. Un seul balayage par force brute évalue $50\,000 \times 5 \times 3$ maxima, là où le log-sum-exp demande $5 \times 3$ exponentielles, et la réponse obtenue par force brute est encore inexacte au troisième chiffre.

<!-- ## Point de vue de la commande comme inférence

Les équations d'optimalité de Bellman lisses peuvent aussi être dérivées du point de vue de l'inférence probabiliste. Pour le voir, revenons à l'idée de la section précédente, dans laquelle nous avons introduit une fonction indicatrice $I_a(\epsilon)$ pour indiquer si une action $a$ est optimale pour une réalisation particulière du bruit $\epsilon$ :

$$ I_a(\epsilon) = \begin{cases} 
   1 & \text{si } a \in \operatorname{argmax}_{a' \in \mathcal{A}_s} \left\{ r(s,a') + \epsilon(a') + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, a'\right] \right\} \\
   0 & \text{sinon}
   \end{cases} $$

En prenant l'espérance sur le bruit $\epsilon$, nous avons obtenu une version douce de cette indicatrice :

$$ \begin{align*}
\mathbb{E}_\epsilon[I_a(\epsilon)] &= \mathbb{P}\left(a \in \operatorname{argmax}_{a' \in \mathcal{A}_s} \left\{ r(s,a') + \epsilon(a') + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, \epsilon, a'\right] \right\}\right) \\
&= \frac{\exp\left(\beta\left(r(s,a) + \gamma \mathbb{E}_{s'}\left[v_\gamma^\star(s')\mid s, a\right]\right)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a') + \gamma \mathbb{E}_{s'}\left[v_\gamma^\star(s')\mid s, a'\right]\right)\right)}
\end{align*} $$

Avec cette fonction indicatrice, on peut « inférer » l'action optimale dans n'importe quel état. C'est l'intuition et le point de départ du point de vue de la commande comme inférence, dans lequel on définit directement une variable d'« optimalité » à valeurs continues $O_t$ à chaque pas de temps $t$. On définit la probabilité d'optimalité d'une paire état-action par :

$$ p(O_t = 1 | s_t, a_t) = \exp(\beta r(s_t, a_t)) $$

À partir de cette notion d'optimalité douce, on peut formuler le PDM comme un modèle graphique probabiliste. On définit les probabilités suivantes :

1. Probabilité de transition d'état : $p(s_{t+1} | s_t, a_t)$ (donnée par la dynamique du PDM)
2. Politique a priori : $p(a_t | s_t)$ (que l'on supposera uniforme par souci de simplicité)
3. Probabilité d'optimalité : $p(O_t = 1 | s_t, a_t) = \exp(\beta r(s_t, a_t))$

Cette formulation traduit l'idée que les paires état-action plus récompensées ont plus de chances d'être « optimales », ce qui correspond directement à l'attribution douce de l'optimalité obtenue en prenant l'espérance sur le bruit de Gumbel.

Le problème de commande peut maintenant être posé comme un problème d'inférence : on cherche la distribution a posteriori des actions sachant que tous les pas de temps sont optimaux :

$$ p(a_t | s_t, O_{1:T} = 1) $$

où $O_{1:T} = 1$ signifie $O_t = 1$ pour tout $t$ de 1 à T.

### Passage de messages

Pour résoudre ce problème d'inférence, on peut utiliser une technique des modèles graphiques probabilistes appelée passage de messages, plus précisément l'algorithme de propagation de croyances. Le passage de messages permet de calculer efficacement des distributions marginales dans un modèle graphique en échangeant des messages locaux entre les nœuds. Les messages circulent entre les nœuds dans les deux sens, vers l'avant et vers l'arrière. Chaque message représente une croyance sur la distribution d'une variable, fondée sur l'information dont dispose le nœud émetteur. Une fois les messages échangés, chaque nœud met à jour sa croyance sur la variable qui lui est associée en combinant tous les messages reçus.

Dans notre cas, ce sont surtout les messages rétrogrades qui nous intéressent; ils propagent vers le passé l'information sur l'optimalité future. Définissons le message rétrograde $\beta_t(s_t)$ par :

$$ \beta_t(s_t) = p(O_{t:T} = 1 | s_t) $$

Il représente la probabilité d'optimalité de tous les pas de temps futurs, étant donné l'état courant. On peut le calculer récursivement :

$$ \beta_t(s_t) = \sum_{a_t} p(a_t | s_t) p(O_t = 1 | s_t, a_t) \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \beta_{t+1}(s_{t+1}) $$


En prenant le logarithme et en supposant une loi a priori uniforme sur les actions, on obtient :

$$ \log \beta_t(s_t) = \log \sum_{a_t} \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \exp(\beta (r(s_t, a_t) + \gamma v(_{t+1}) + \frac{1}{\beta} \log \beta_{t+1}(s_{t+1}))) $$

Si l'on définit la fonction de valeur douce par $V_t(s_t) = \frac{1}{\beta} \log \beta_t(s_t)$, on peut réécrire l'équation ci-dessus sous la forme :

$$ V_t(s_t) = \frac{1}{\beta} \log \sum_{a_t} \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \exp(\beta (r(s_t, a_t) + \gamma V_{t+1}(s_{t+1}))) $$

C'est exactement l'équation de Bellman lisse dérivée plus haut, mais interprétée maintenant comme le résultat d'une inférence probabiliste dans un modèle graphique.

### Dérivation de la politique optimale

La récurrence des messages rétrogrades dérivée plus haut suppose une politique a priori uniforme $p(a_t | s_t)$. Notre but est toutefois de trouver une politique optimale. On peut extraire efficacement cette politique optimale en calculant la distribution a posteriori des actions à partir des messages rétrogrades.

En partant de la définition de la probabilité conditionnelle et en appliquant la règle de Bayes, on peut écrire :

$$ \begin{align}
p(a_t | s_t, O_{1:T} = 1) &= \frac{p(O_{1:T} = 1 | s_t, a_t) p(a_t | s_t)}{p(O_{1:T} = 1 | s_t)} \\
&\propto p(a_t | s_t) p(O_t = 1 | s_t, a_t) p(O_{t+1:T} = 1 | s_t, a_t) \\
&= p(a_t | s_t) p(O_t = 1 | s_t, a_t) \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \beta_{t+1}(s_{t+1})
\end{align} $$

Ici, $\beta_{t+1}(s_{t+1}) = p(O_{t+1:T} = 1 | s_{t+1})$ est notre message rétrograde.

Substituons maintenant nos définitions de la probabilité d'optimalité et de la fonction de valeur douce :

$$ \begin{align}
p(a_t | s_t, O_{1:T} = 1) &\propto p(a_t | s_t) \exp(\beta r(s_t, a_t)) \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \exp(\beta \gamma V_{t+1}(s_{t+1})) \\
&= p(a_t | s_t) \exp(\beta (r(s_t, a_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) V_{t+1}(s_{t+1})))
\end{align} $$

Après normalisation, et en supposant une loi a priori uniforme $p(a_t | s_t)$, on obtient la règle de décision randomisée :

$$ d(a_t | s_t) = \frac{\exp(\beta (r(s_t, a_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) V_{t+1}(s_{t+1})))}{\sum_{a'_t} \exp(\beta (r(s_t, a'_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a'_t) V_{t+1}(s_{t+1})))} $$ -->

(sec-regularized-mdps)=
## Processus de décision markoviens régularisés

Quelle pénalité convexe sur les distributions d'actions engendre, par dualité
conjuguée, un maximum lisse souhaité?

Les PDM régularisés {cite}`geist2019` offrent un autre point de vue sur l'origine des équations de Bellman lisses. Ce cadre propose une approche plus générale, dans laquelle on cherche des politiques optimales selon le critère à horizon infini tout en tenant compte d'un régularisateur qui oriente le type de politiques que l'on cherche à obtenir.

Mettons en place la notation nécessaire. Rappelons d'abord que l'opérateur d'évaluation d'une politique stationnaire de règle de décision $\pi$ est défini par :

$$ \BellmanPi \mathbf{v} = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v} $$

où $\mathbf{r}_\pi$ est le vecteur des récompenses espérées sous la politique $\pi$, $\gamma$ est le facteur d'actualisation et $\mathbf{P}_\pi$ est la matrice des probabilités de transition d'état sous $\pi$. Un objet complémentaire à la fonction de valeur est la représentation par la fonction q (ou facteur Q) :

$$ \begin{align*}
q_\gamma^{\pi}(s, a) &= r(s, a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v_\gamma^{\pi}(j) \\
v_\gamma^{\pi}(s) &= \sum_{a \in \mathcal{A}_s} \pi(a | s) q_\gamma^{\pi}(s, a) 
\end{align*} $$

L'opérateur d'évaluation de la politique peut alors s'écrire en fonction de la fonction q :

$$ [\BellmanPi v](s) = \langle \pi(\cdot | s), q(s, \cdot) \rangle $$

### Transformée de Legendre-Fenchel

L'outil central de la théorie des PDM régularisés est la transformée de Legendre-Fenchel, aussi appelée conjuguée convexe. Pour une fonction fortement convexe $\Omega: \Delta(\mathcal{A}_s) \rightarrow \mathbb{R}$, sa transformée de Legendre-Fenchel $\Omega^*: \mathbb{R}^{\mathcal{A}_s} \rightarrow \mathbb{R}$ est définie par :

$$ \Omega^*(q(s, \cdot)) = \max_{\pi(\cdot|s) \in \Delta(\mathcal{A}_s)} \langle \pi(\cdot | s), q(s, \cdot) \rangle - \Omega(\pi(\cdot | s)) $$

Une propriété importante de cette transformée est qu'elle possède un unique argument maximisant, donné par le gradient de $\Omega^*$ :

$$ \nabla \Omega^*(q(s, \cdot)) = \arg\max_\pi \langle \pi(\cdot | s), q(s, \cdot) \rangle - \Omega(\pi(\cdot | s)) $$

Le gradient est aussi lipschitzien : si $\Omega$ est $\kappa$-fortement convexe, une variation de $q$ déplace la politique maximisante d'au plus $1/\kappa$ fois cette variation. Un régularisateur plus fortement convexe rend donc la politique moins sensible aux erreurs sur $q$. Un exemple important de régularisateur est l'entropie négative, qui donne naissance aux équations de Bellman lisses, comme nous allons le voir. Pondérée par la température $1/\beta$, elle est $(1/\beta)$-fortement convexe; la politique softmax se déplace donc d'au plus $\beta$ fois la variation de $q$ (en mesurant $q$ en norme sup et les politiques en norme $\ell_1$).

(sec-regularized-operators)=
## Opérateurs de Bellman régularisés

Comment le régularisateur et sa conjuguée convexe modifient-ils l'évaluation
de Bellman, l'amélioration de la politique et leurs propriétés de contraction?

Ces notions en place, on peut maintenant définir les opérateurs de Bellman régularisés :

1. **Opérateur d'évaluation de politique régularisé** $(\mathrm{L}_{\pi,\Omega})$ :

   $$ [\mathrm{L}_{\pi,\Omega} v](s) = \langle q(s,\cdot), \pi(\cdot | s) \rangle - \Omega(\pi(\cdot | s)) $$

2. **Opérateur d'optimalité de Bellman régularisé** $(\mathrm{L}_\Omega)$ :
           
   $$ [\mathrm{L}_\Omega v](s) = [\max_\pi \mathrm{L}_{\pi,\Omega} v ](s) = \Omega^*(q(s, \cdot)) $$

On peut montrer que l'ajout d'un régularisateur dans ces opérateurs régularisés préserve les propriétés de contraction, et donc l'existence d'une solution aux équations d'optimalité et la convergence des approximations successives.

La fonction de valeur régularisée d'une politique stationnaire de règle de décision $\pi$, notée $v_{\pi,\Omega}$, est l'unique point fixe de l'équation d'opérateur :

$$\text{trouver $v$ tel que } \enspace v = \mathrm{L}_{\pi,\Omega} v$$

Sous les hypothèses habituelles sur le facteur d'actualisation et le caractère borné de la récompense, la valeur d'une politique peut aussi s'obtenir sous forme analytique en résolvant en $\mathbf{v}$ le système d'équations linéaires :

$$ (\mathbf{I} - \gamma \mathbf{P}_\pi) \mathbf{v} =  \mathbf{r}_\pi - \boldsymbol{\Omega}_\pi $$

où $[\boldsymbol{\Omega}_\pi](s) = \Omega(\pi(\cdot|s))$ est le vecteur des termes de régularisation dans chaque état.

La fonction de valeur d'action associée $q_{\pi,\Omega}$ est donnée par :

$$\begin{align*}
q_{\pi,\Omega}(s, a) &= r(s, a) + \sum_{j \in \mathcal{S}} \gamma p(j|s,a) v_{\pi,\Omega}(j) \\
v_{\pi,\Omega}(s) &= \sum_{a \in \mathcal{A}_s} \pi(a | s) q_{\pi,\Omega}(s, a) - \Omega(\pi(\cdot | s))
\end{align*} $$

La fonction de valeur optimale régularisée $v^*_\Omega$ est alors l'unique point fixe de $\mathrm{L}_\Omega$ dans l'équation de point fixe :

$$\text{trouver $v$ tel que } v = \mathrm{L}_\Omega v$$

La fonction de valeur d'action associée $q^*_\Omega$ est donnée par :

$$ \begin{align*}
q^*_\Omega(s, a) &= r(s, a) + \sum_{j \in \mathcal{S}} \gamma p(j|s,a) v^*_\Omega(j) \\
v^*_\Omega(s) &= \Omega^*(q^*_\Omega(s, \cdot))\end{align*} $$

Un résultat important de la théorie des PDM régularisés est qu'il existe une unique politique régularisée optimale. Plus précisément, si $\pi^*_\Omega$ est une règle de décision conservatrice (c'est-à-dire $\pi^*_\Omega = \arg\max_\pi \mathrm{L}_{\pi,\Omega} v^*_\Omega$), alors la politique stationnaire randomisée $\boldsymbol{\pi} = \mathrm{const}(\pi^*_\Omega)$ est l'unique politique régularisée optimale.

En pratique, une fois $v^*_\Omega$ trouvée, on peut dériver la règle de décision optimale en prenant le gradient de la conjuguée convexe évalué en la fonction de valeur d'action optimale :

$$ \pi^*(\cdot | s) = \nabla \Omega^*(q^*_\Omega(s, \cdot)) $$

### Retrouver les équations de Bellman lisses

Dans ce cadre, on retrouve les équations de Bellman lisses en choisissant pour $\Omega$ l'entropie négative pondérée par la température $1/\beta$, et l'on obtient la politique softmax comme gradient de la conjuguée convexe. Montrons-le explicitement :

1. Avec le régularisateur d'entropie négative pondérée :

   $$ \Omega(\pi(\cdot|s)) = \frac{1}{\beta}\sum_{a \in \mathcal{A}_s} \pi(a|s) \log \pi(a|s) $$

2. La conjuguée convexe :

   $$ \Omega^*(q(s, \cdot)) = \frac{1}{\beta}\log \sum_{a \in \mathcal{A}_s} \exp \big(\beta q(s,a)\big) $$

3. Écrivons maintenant l'équation d'optimalité de Bellman régularisée :

   $$ v^*_\Omega(s) = \Omega^*(q^*_\Omega(s, \cdot)) $$

4. En substituant les expressions de $\Omega^*$ et de $q^*_\Omega$ :

   $$ v^*_\Omega(s) = \frac{1}{\beta}\log \sum_{a \in \mathcal{A}_s} \exp \left(\beta\Big(r(s, a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v^*_\Omega(j)\Big)\right) $$

On retrouve la forme de l'équation de Bellman lisse dérivée plus haut, où l'opération log-sum-exp remplace l'opération max de l'équation de Bellman usuelle. Avec $\beta = 1$, l'entropie négative non pondérée donne la même équation à température unité.

De plus, la politique optimale est donnée par le gradient de $\Omega^*$ :

$$ \pi^*(a|s) = \nabla \Omega^*(q^*_\Omega(s, \cdot))_a = \frac{\exp(\beta q^*_\Omega(s,a))}{\sum_{a' \in \mathcal{A}_s} \exp(\beta q^*_\Omega(s,a'))} $$

C'est la politique softmax familière rencontrée dans le cadre des PDM lisses.

### Algorithme d'itération sur les politiques lisse

Une identité relie le log-sum-exp à l'opérateur d'évaluation régularisé. Soit $\pi_q(\cdot \mid s)$ la softmax de $\beta q(s,\cdot)$. Alors

$$
\frac{1}{\beta}\log\sum_{a} \exp\big(\beta q(s,a)\big) = \sum_a \pi_q(a \mid s)\, q(s,a) + \frac{1}{\beta} H\big(\pi_q(\cdot \mid s)\big),
$$

ce que l'on vérifie en substituant $\log \pi_q(a \mid s) = \beta q(s,a) - \log\sum_{a'} \exp(\beta q(s,a'))$ dans l'entropie. La ligne 6 de l'{prf:ref}`smooth-value-iteration` peut donc s'écrire « former la politique softmax, puis ajouter son $q$ espéré et son bonus d'entropie ». Les algorithmes énoncés de cette façon sont parfois appelés itération sur les valeurs régularisée par l'entropie, mais il s'agit de l'itération sur les valeurs lisse avec le log-sum-exp développé. L'itération sur les politiques est différente : elle évalue complètement chaque politique avant de l'améliorer.

```{prf:algorithm} Itération sur les politiques lisse
:label: policy-iteration-smooth

**Entrée :** PDM $(\mathcal{S}, \mathcal{A}, r, p, \gamma)$, inverse de la température $\beta > 0$, tolérance $\varepsilon > 0$

**Sortie :** Politique stochastique $\pi$, $\varepsilon$-optimale pour l'objectif régularisé par l'entropie, et sa valeur $v$

1. Initialiser $\pi(a|s) \leftarrow 1/|\mathcal{A}_s|$ pour tout $s \in \mathcal{S}$, $a \in \mathcal{A}_s$ (politique uniforme)
2. **répéter**
3. $\quad$ **Évaluation :** résoudre $(\mathbf{I} - \gamma \mathbf{P}_\pi)\mathbf{v} = \mathbf{r}_\pi + \frac{1}{\beta}\mathbf{H}_\pi$, où $[\mathbf{r}_\pi](s) = \sum_a \pi(a|s) r(s,a)$ et $[\mathbf{H}_\pi](s) = -\sum_a \pi(a|s)\log\pi(a|s)$
4. $\quad q(s,a) \leftarrow r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)$ pour tout $s \in \mathcal{S}$, $a \in \mathcal{A}_s$
5. $\quad \Delta \leftarrow \max_{s \in \mathcal{S}} \big(\frac{1}{\beta}\log\sum_{a \in \mathcal{A}_s}\exp(\beta q(s,a)) - v(s)\big)$, le résidu de Bellman $\|\mathrm{L}_\beta v - v\|_\infty$
6. $\quad$ **si** $\Delta < \varepsilon(1-\gamma)$ **alors renvoyer** $v, \pi$
7. $\quad$ **Amélioration :** $\pi(a|s) \leftarrow \frac{\exp(\beta q(s,a))}{\sum_{a' \in \mathcal{A}_s} \exp(\beta q(s,a'))}$ pour tout $s \in \mathcal{S}$, $a \in \mathcal{A}_s$
```

**Propriétés de l'itération sur les politiques lisse :**

1. **Évaluation régularisée par l'entropie.** La ligne 3 est le système linéaire qui donne la valeur régularisée $v_{\pi,\Omega}$ de la section {ref}`sec-regularized-operators`, avec le vecteur d'entropie $\mathbf{H}_\pi$ au membre de droite. Remplacer la résolution exacte par quelques balayages de $v \leftarrow \mathbf{r}_\pi + \frac{1}{\beta}\mathbf{H}_\pi + \gamma\mathbf{P}_\pi v$ donne la version lisse de l'{prf:ref}`alg-modified-policy-iteration`.
2. **Amélioration stochastique.** La ligne 7 utilise la softmax au lieu de l'argmax. Lorsque $\beta \to \infty$, la méthode redonne l'itération sur les politiques usuelle, et un $\beta$ plus faible donne des politiques d'entropie plus élevée. Comme dans le cas strict, il n'est pas nécessaire d'appliquer l'amélioration dans chaque état pour que la méthode progresse. Les boucles d'itération sur les politiques cachées dans les méthodes acteur-critique en tirent parti.
3. **Ce que garantit la règle d'arrêt.** Le résidu de la ligne 5 est positif ou nul, car $v$ est la valeur régularisée de $\pi$ et la softmax maximise l'objectif régularisé à un pas; on a donc $\mathrm{L}_\beta v \ge \mathrm{L}_{\pi,\Omega} v = v$. Pour tout $v$, l'inégalité triangulaire et la propriété de contraction donnent $\|v - v^\star_\beta\|_\infty \le \|v - \mathrm{L}_\beta v\|_\infty + \gamma\|v - v^\star_\beta\|_\infty$, de sorte que

   $$
   \|v - v^\star_\beta\|_\infty \le \frac{\|\mathrm{L}_\beta v - v\|_\infty}{1-\gamma}.
   $$

   C'est la borne a posteriori du {prf:ref}`banach-fixed-point` écrite pour un pas de $\mathrm{L}_\beta$ à partir de $v$. Puisque $v$ est la valeur régularisée exacte de la politique $\pi$ courante, le test de la ligne 6 garantit que $\pi$ elle-même est à moins de $\varepsilon$ de la valeur régularisée optimale dans chaque état. Un test sur la variation de la politique, comme $\|\pi - \pi_{\text{anc}}\| < \varepsilon$, ne donnerait aucune garantie de ce genre.
4. **Convergence.** L'algorithme converge vers l'unique fonction de valeur et l'unique politique régularisées optimales. Contrairement à l'itération sur les politiques usuelle, il ne se termine pas après un nombre fini d'itérations : les politiques softmax forment un continuum, et les itérés s'approchent de l'optimum sans l'atteindre; c'est pourquoi la boucle s'arrête sur une tolérance. La convergence est rapide, puisque chaque itération est un pas de Newton ({ref}`sec-smooth-pi-newton`).

(sec-smooth-pi-newton)=
### L'itération sur les politiques lisse est la méthode de Newton

Le lien entre l'itération sur les politiques et la méthode de Newton-Kantorovitch ({ref}`sec-newton-kantorovich-pi`) est plus facile à voir ici que dans le cas strict, car $\mathrm{L}_\beta$ est différentiable partout et il n'y a pas d'égalités à craindre. La dérivée du log-sum-exp par rapport à ses arguments est la softmax; la dérivée de Fréchet de $\mathrm{L}_\beta$ en $v$ est donc

$$
\mathrm{L}_\beta'(v) = \gamma \mathbf{P}_{\pi_v}, \qquad \pi_v(a \mid s) = \frac{\exp(\beta q_v(s,a))}{\sum_{a'}\exp(\beta q_v(s,a'))}, \quad q_v(s,a) = r(s,a) + \gamma\sum_j p(j\mid s,a) v(j),
$$

la matrice de transition de la politique softmax. Un pas de Newton sur $v - \mathrm{L}_\beta v = 0$ résout

$$
(\mathbf{I} - \gamma\mathbf{P}_{\pi_v})\, v_{\text{nouv}} = \mathrm{L}_\beta v - \gamma\mathbf{P}_{\pi_v} v.
$$

À la politique softmax, le log-sum-exp est égal au $q$ espéré plus le bonus d'entropie, $\mathrm{L}_\beta v = \mathbf{r}_{\pi_v} + \gamma\mathbf{P}_{\pi_v} v + \frac{1}{\beta}\mathbf{H}_{\pi_v}$. Le membre de droite vaut donc $\mathbf{r}_{\pi_v} + \frac{1}{\beta}\mathbf{H}_{\pi_v}$, et le pas de Newton est exactement l'évaluation régularisée par l'entropie de la politique softmax. L'itération sur les politiques lisse est la méthode de Newton-Kantorovitch appliquée à l'équation de Bellman lisse, sans l'appareil semi-lisse qu'exige le max strict.

{cite:t}`rust1987optimal` a résolu son équation de Bellman lisse de cette façon. Il ne l'appelait pas itération sur les politiques. Il y a reconnu une équation d'opérateur, un problème de recherche de racine dans un espace fonctionnel, et lui a appliqué la méthode de Newton-Kantorovitch, après quelques rondes d'approximations successives pour s'approcher suffisamment de la solution pour que la méthode de Newton converge rapidement.

### Équivalence entre les équations de Bellman lisses et les PDM régularisés par l'entropie

Nous avons maintenant vu trois façons d'arriver à l'équation de Bellman lisse : comme un maximum lissé, à partir d'un bruit de Gumbel sur les récompenses, et comme l'équation d'optimalité d'un PDM régularisé. La troisième peut s'écrire comme un problème d'optimisation sur les politiques, la forme qu'utilise l'apprentissage par renforcement à entropie maximale. Avec un poids d'entropie $\alpha = 1/\beta$, le PDM régularisé par l'entropie s'écrit

$$
\max_\pi\; \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t \Big( r(s_t, a_t) + \alpha H\big(\pi(\cdot|s_t)\big) \Big) \right],
\qquad H(\pi(\cdot|s)) = -\sum_a \pi(a|s) \log \pi(a|s).
$$

Puisque $H(\pi(\cdot|s)) = \mathbb{E}_{a \sim \pi(\cdot|s)}[-\log \pi(a|s)]$, le bonus d'entropie dans l'état $s$ est la moyenne d'un terme par action, et l'objectif est égal à

$$
\mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t \big( r(s_t, a_t) - \alpha \log \pi(a_t|s_t) \big) \right].
$$

La régularisation par l'entropie revient donc à payer $-\alpha\log\pi(a|s)$ pour prendre l'action $a$, un coût élevé pour les actions que la politique prend rarement. C'est la forme qu'estiment les méthodes fondées sur des échantillons comme *soft actor-critic*, puisqu'elle ne demande que la probabilité de l'action effectivement prise. Dans le PDM augmenté de la section {ref}`sec-gumbel-side-by-side`, la même quantité est le choc espéré sur l'action choisie.

L'équation d'optimalité de Bellman de ce problème maximise, dans chaque état, sur la distribution $\pi(\cdot|s)$ :

$$
v(s) = \max_{\pi(\cdot|s) \in \Delta(\mathcal{A}_s)} \sum_a \pi(a|s) \big( q(s,a) - \alpha \log \pi(a|s) \big),
\qquad q(s,a) = r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)\, v(j).
$$

Le problème intérieur est la conjuguée convexe de l'entropie négative pondérée, que nous avons calculée dans « Retrouver les équations de Bellman lisses ». Une vérification directe est aussi rapide. En notant $\pi_q$ la softmax de $\beta q(s,\cdot)$, pour toute distribution $\pi$,

$$
\sum_a \pi(a|s) \big( q(s,a) - \alpha \log \pi(a|s) \big) = \alpha \log\sum_a \exp\big(q(s,a)/\alpha\big) - \alpha\, \mathrm{KL}\big(\pi(\cdot|s) \,\big\|\, \pi_q(\cdot|s)\big),
$$

et la divergence KL est positive ou nulle, et nulle seulement en $\pi = \pi_q$. Le maximum est le log-sum-exp avec $\beta = 1/\alpha$, atteint par la politique softmax. La fonction de valeur qui résout l'équation de Bellman lisse est donc la valeur optimale du PDM régularisé par l'entropie, et la politique softmax en est la politique optimale. Les deux descriptions ne diffèrent que par la façon de présenter le problème : comme un lissage de l'opérateur de Bellman, ou comme un compromis entre la récompense et l'entropie de la politique. Les algorithmes d'apprentissage par renforcement à entropie maximale comme *soft actor-critic* utilisent les deux : l'opérateur lisse pour sa différentiabilité, et l'objectif régularisé pour expliquer pourquoi la politique doit rester stochastique.

(sec-continuous-actions)=
## Espaces d'actions et d'états continus

Qu'est-ce qui change dans l'équation de Bellman lisse lorsque les sommes sur les actions deviennent des intégrales?

Jusqu'ici, les ensembles d'actions étaient finis. Dans les problèmes de commande, l'action est habituellement un vecteur $a \in \mathcal{A}_s \subseteq \mathbb{R}^m$ (un couple, une dose, un débit), et l'état est souvent continu lui aussi. L'équation de Bellman lisse se transpose dès que l'on fixe une *mesure de référence* $\mu$ sur les actions. Pour un ensemble fini, $\mu$ est la mesure de comptage, et les intégrales par rapport à celle-ci sont des sommes. Pour un pavé ou une autre partie bornée de $\mathbb{R}^m$, c'est habituellement la mesure de Lebesgue $da$. Ce peut aussi être une mesure de probabilité, une politique a priori $\pi_0(da \mid s)$. Pour des états continus, il suffit d'écrire le noyau de transition comme une mesure, $p(ds' \mid s,a)$. Le tableau transpose la notation de ce chapitre.

| Objet | $\mathcal{S}$ et $\mathcal{A}_s$ finis | $\mathcal{S}$ continu et $\mathcal{A}_s \subseteq \mathbb{R}^m$ |
|---|---|---|
| Valeur suivante espérée | $\sum_{j \in \mathcal{S}} p(j \mid s,a)\, v(j)$ | $\int_{\mathcal{S}} v(s')\, p(ds' \mid s,a)$ |
| Règle de décision | vecteur de probabilités $\pi(\cdot \mid s) \in \Delta(\mathcal{A}_s)$ | densité $\pi(\cdot \mid s)$ par rapport à $\mu$, $\int_{\mathcal{A}_s} \pi(a \mid s)\, \mu(da) = 1$ |
| Moyenne sous la politique | $\sum_a \pi(a \mid s)\, q(s,a)$ | $\int_{\mathcal{A}_s} \pi(a \mid s)\, q(s,a)\, \mu(da)$ |
| Entropie | $H(\pi) = -\sum_a \pi(a)\log\pi(a) \ge 0$ | $h_\mu(\pi) = -\int \pi(a)\log\pi(a)\, \mu(da)$, de signe quelconque |
| Maximum lisse | $\frac{1}{\beta}\log\sum_a e^{\beta q(s,a)}$ | $\frac{1}{\beta}\log\int_{\mathcal{A}_s} e^{\beta q(s,a)}\, \mu(da)$ |
| Règle de décision optimale | softmax sur $\mathcal{A}_s$ | densité de Gibbs $\pi_\beta(a \mid s) = \exp\big(\beta(q(s,a) - v(s))\big)$ |
| Biais par rapport au max strict | entre $0$ et $\frac{\log\lvert\mathcal{A}\rvert}{\beta}$ par pas | au plus $\frac{\log \mu(\mathcal{A}_s)}{\beta}$, mais possiblement négatif; voir plus bas |

Avec cette notation, l'équation de Bellman lisse s'écrit

$$
v^\star_\beta(s) = \frac{1}{\beta}\log\int_{\mathcal{A}_s} \exp\big(\beta\, q(s,a)\big)\, \mu(da), \qquad
q(s,a) = r(s,a) + \gamma \int_{\mathcal{S}} v^\star_\beta(s')\, p(ds' \mid s,a),
$$

et la règle de décision optimale est la densité de Gibbs $\pi_\beta(a \mid s) = \exp\big(\beta(q(s,a) - v^\star_\beta(s))\big)$, dont l'intégrale par rapport à $\mu$ vaut un parce que $v^\star_\beta(s)$ est le logarithme de la constante de normalisation. L'argument de contraction de la section sur l'itération sur les valeurs lisse n'utilise que la monotonie et la translation par une constante, et les deux restent vraies pour l'intégrale. L'opérateur est donc une $\gamma$-contraction en norme sup, pourvu que l'intégrale soit finie. Avec des récompenses bornées, elle est finie lorsque $\mu(\mathcal{A}_s) < \infty$, par exemple pour un pavé d'actions muni de la mesure de Lebesgue ou pour n'importe quelle politique a priori.

**La forme variationnelle.** La lecture régularisée se transpose aussi. Pour toute densité $\nu$ par rapport à $\mu$, notons $Z(s) = \int e^{\beta q(s,a)}\mu(da)$. Alors

$$
\int \nu(a)\, q(s,a)\, \mu(da) - \frac{1}{\beta}\int \nu(a) \log \nu(a)\, \mu(da)
= \frac{1}{\beta}\log Z(s) - \frac{1}{\beta}\, \mathrm{KL}\big(\nu \,\big\|\, \pi_\beta(\cdot \mid s)\big),
$$

comme on le vérifie en substituant $\log \pi_\beta = \beta q - \log Z$. Le membre de gauche est l'objectif régularisé de la section {ref}`sec-regularized-operators` avec $\Omega(\nu) = \frac{1}{\beta}\int \nu\log\nu\, d\mu$. Comme la divergence KL est positive ou nulle et ne s'annule qu'en $\nu = \pi_\beta$, le maximum sur $\nu$ est le logarithme de l'intégrale, et le maximiseur est la densité de Gibbs. C'est la version continue de la paire de Legendre-Fenchel (entropie négative, log-sum-exp); en physique statistique, on l'appelle le principe variationnel de Gibbs.

**La mesure de référence compte.** Avec un ensemble fini d'actions, renommer les actions ne change rien. Avec des actions continues, l'entropie dépend de $\mu$, et cette dépendance se manifeste de trois façons.

- *Les politiques a priori donnent une régularisation KL.* Si $\mu = \pi_0(\cdot \mid s)$ est une mesure de probabilité, alors $\int \nu \log \nu\, d\mu = \mathrm{KL}(\nu \,\|\, \pi_0)$, et le problème devient un apprentissage par renforcement régularisé par la divergence KL : l'agent est pénalisé lorsqu'il s'éloigne de $\pi_0$. Dans le cas fini, la loi a priori uniforme donne $\mathrm{KL}(\nu\,\|\,\text{uniforme}) = \log\lvert\mathcal{A}_s\rvert - H(\nu)$; la régularisation par l'entropie et la régularisation KL diffèrent donc d'une constante par état et ont la même politique optimale.
- *L'entropie différentielle n'a pas de plancher.* Sous la mesure de Lebesgue, $h(\pi) = -\int \pi \log \pi\, da$ est l'entropie différentielle, négative pour des densités concentrées : une gaussienne de variance $\sigma^2$ en dimension un a $h = \frac{1}{2}\log(2\pi e \sigma^2)$, négative dès que $\sigma^2 < 1/(2\pi e)$. Elle dépend aussi des unités de l'action. Le changement d'échelle $a \mapsto c\,a$ dans $\mathbb{R}^m$ ajoute $m \log\lvert c\rvert$ à $h$. Cela ne fait que décaler toutes les valeurs d'une constante et laisse la politique inchangée, mais un changement de variable non linéaire, par exemple d'un couple à son logarithme, modifie la politique optimale.
- *Les ensembles d'actions non bornés demandent des précautions.* Si $\mathcal{A}_s = \mathbb{R}^m$ est muni de la mesure de Lebesgue et que les récompenses sont bornées, l'intégrale diverge et l'équation de Bellman lisse n'a pas de solution. Les remèdes sont un ensemble d'actions borné (*soft actor-critic* comprime ses échantillons gaussiens dans un pavé au moyen d'une $\tanh$), une politique a priori comme mesure de référence, ou des récompenses qui décroissent assez vite en $\lVert a\rVert$, comme un coût quadratique sur l'action.

**À quelle distance du maximum strict.** De la borne du cas fini $\max_a q \le \frac{1}{\beta}\log\sum_a e^{\beta q} \le \max_a q + \frac{\log\lvert\mathcal{A}\rvert}{\beta}$, seule la moitié supérieure subsiste : $\frac{1}{\beta}\log\int e^{\beta q}\,d\mu \le \sup_a q(s,a) + \frac{\log \mu(\mathcal{A}_s)}{\beta}$, ce qui donne $v^\star_\beta \le v^*_\gamma + \frac{\log \mu(\mathcal{A})}{\beta(1-\gamma)}$ lorsque $\mu(\mathcal{A}_s) \le \mu(\mathcal{A})$ pour tout $s$. La borne inférieure tombe, car un ensemble d'actions de faible volume contribue peu à l'intégrale, même s'il contient le maximiseur. Le bon outil est la méthode de Laplace. Si $q(s,\cdot)$ est deux fois dérivable et possède un unique maximiseur $a^\star(s)$ à l'intérieur de $\mathcal{A}_s$, et si $\mathbf{H}(s) = -\nabla_a^2 q(s, a^\star(s))$ est définie positive, alors, sous la mesure de Lebesgue,

$$
\frac{1}{\beta}\log\int_{\mathcal{A}_s} e^{\beta q(s,a)}\, da
= q(s,a^\star(s)) + \frac{m}{2\beta}\log\frac{2\pi}{\beta} - \frac{1}{2\beta}\log\det \mathbf{H}(s) + o(1/\beta).
$$

L'écart avec le maximum strict diminue comme $\frac{m\log\beta}{2\beta}$, un peu plus lentement que le taux $1/\beta$ du cas fini, et la densité de Gibbs se concentre autour de $a^\star(s)$ comme une gaussienne de covariance $(\beta\mathbf{H}(s))^{-1}$.

**$q$ quadratique et commande linéaire quadratique.** Si $q(s,a) = q_0(s) - \frac{1}{2}(a - \bar a(s))^\top \mathbf{H}(s)(a - \bar a(s))$ est exactement quadratique en $a$ avec $\mathbf{H}(s) \succ 0$, la formule de Laplace est exacte. La politique de Gibbs est la gaussienne $\mathcal{N}\big(\bar a(s), (\beta\mathbf{H}(s))^{-1}\big)$, et $v^\star_\beta(s) = q_0(s) + \frac{1}{2\beta}\log\det\big(2\pi(\beta\mathbf{H}(s))^{-1}\big)$. Les problèmes linéaires quadratiques sont de ce type. Avec la dynamique $x' = Ax + Bu + w$ et la récompense $-\frac{1}{2}x^\top Q x - \frac{1}{2}u^\top R u$, une valeur quadratique $-\frac{1}{2}x^\top P x + c$ rend $q$ quadratique en $u$ avec $\mathbf{H} = R + \gamma B^\top P B$, qui ne dépend pas de $x$. Le terme en log-déterminant est alors une constante; $P$ résout donc la même équation de Riccati actualisée que sans régularisation. La politique de commande linéaire quadratique régularisée par l'entropie est gaussienne, de moyenne égale à la rétroaction linéaire habituelle $-\gamma(R + \gamma B^\top P B)^{-1} B^\top P A\, x$ et de covariance $\big(\beta(R + \gamma B^\top P B)\big)^{-1}$. La régularisation ajoute un bruit d'exploration autour du correcteur linéaire quadratique, davantage dans les directions où le coût est plat, et ne modifie pas le gain.

**Chocs de Gumbel sur un continuum.** La construction du PDM augmenté exige un choc par action. Sur un continuum d'actions, on ne peut pas attacher des chocs indépendants à chaque point, puisque le supremum d'une infinité non dénombrable de variables de Gumbel indépendantes est infini. Le remplaçant est un *processus de Gumbel* {cite}`maddison2014astar` : une fonction aléatoire $G$ sur $\mathcal{A}_s$ telle que, pour tout ensemble mesurable $B$, le maximum $\max_{a \in B} G(a)$ suit une loi de Gumbel de position $\log\mu(B)$ et d'échelle unité, indépendamment sur des ensembles disjoints. Pour un tel processus, $\max_a\{\beta q(s,a) + G(a)\}$ suit une loi de Gumbel de position $\log\int e^{\beta q(s,a)}\mu(da)$, et son argmax a la densité de Gibbs. Avec $\epsilon_t = G_t/\beta$, la dérivation de la section {ref}`sec-gumbel-rewards` fonctionne avec une fonction aléatoire à la place d'un vecteur aléatoire, et la {numref}`fig-gumbel-augmented-pgm` décrit toujours le modèle. C'est encore moins un objet de calcul que dans le cas fini. {cite:t}`maddison2014astar` s'en servent pour construire un échantillonneur exact, l'échantillonnage A*, mais les algorithmes de commande continue, dont *soft actor-critic*, travaillent directement avec la densité de Gibbs et l'approchent au moyen d'une politique paramétrique.

(sec-rust-inverse)=
## Les moteurs d'autobus de Rust et l'apprentissage par renforcement inverse

Que peut-on apprendre des objectifs d'un décideur à partir des traces de ce qu'il a fait?

L'article de Rust de 1987 vaut d'être lu pour trois raisons : il contient l'équation de Bellman lisse, il pose ce qu'on appelle aujourd'hui l'apprentissage par renforcement inverse, et il résout ce problème par une méthode à deux niveaux qui repose sur la différentiation implicite, un outil devenu populaire en apprentissage automatique trois décennies plus tard pour l'optimisation des hyperparamètres.

**Le problème.** Harold Zurcher était le surintendant de l'entretien de la société d'autobus Madison Metropolitan. Rust disposait de ses registres d'entretien : pour chaque autobus, la lecture de l'odomètre chaque mois, et les mois où Zurcher avait remplacé le moteur. Remplacer un moteur coûte cher, alors que garder un vieux moteur en marche coûte de plus en plus à mesure que le kilométrage augmente. La question était de savoir comment Zurcher décidait du moment du remplacement. Rust est un économètre structurel. L'économétrie structurelle ne s'arrête pas à la prédiction du comportement; elle postule un mécanisme qui l'engendre, ici un décideur qui résout un PDM, et ajuste les paramètres de ce mécanisme aux données. Le mécanisme ajusté peut alors répondre à des questions auxquelles les données seules ne peuvent pas répondre, par exemple comment le comportement changerait si le prix d'un moteur changeait.

**Le modèle.** L'état $s$ est le kilométrage depuis le dernier remplacement, discrétisé en classes. Chaque mois, il y a deux actions : garder ($a=0$) ou remplacer ($a=1$). Garder coûte $c(s;\theta)$, croissant en $s$; remplacer coûte un montant fixe $RC$ et remet le kilométrage à zéro. Avec des paramètres $\theta$ qui regroupent $RC$ et ceux de $c$, la récompense est $r_\theta(s,0) = -c(s;\theta)$ et $r_\theta(s,1) = -RC - c(0;\theta)$, et les transitions du kilométrage sont estimées directement à partir des données d'odomètre. Les données contredisent toute politique déterministe : deux autobus au même kilométrage sont parfois traités différemment. La réponse de Rust a été le modèle de Gumbel de la section {ref}`sec-gumbel-rewards`. Zurcher observe, en plus de $s$, un choc $\epsilon(a)$ pour chaque action, que l'économètre n'observe pas, et agit de façon optimale compte tenu des deux. L'intégration des chocs donne l'équation de Bellman lisse et les probabilités de choix softmax

$$
v_\theta = \mathrm{L}_{\beta,\theta}\, v_\theta, \qquad \pi_\theta(a \mid s) = \frac{\exp(\beta q_\theta(s,a))}{\sum_{a'}\exp(\beta q_\theta(s,a'))}.
$$

C'est la seconde raison, plus profonde, pour laquelle Rust a utilisé ces équations. En plus d'être lisses, elles découlent des modèles d'utilité aléatoire de {cite:t}`McFadden1974`, le fondement statistique standard des choix discrets en économétrie, de la même façon que les moindres carrés découlent de l'hypothèse d'un bruit gaussien. En termes d'apprentissage automatique, le logit conditionnel de McFadden est une régression logistique multiclasse dotée d'une interprétation structurelle.

**L'estimateur.** Étant donné des paires observées $(s_t, a_t)$, on maximise la log-vraisemblance $\ell(\theta) = \sum_t \log \pi_\theta(a_t \mid s_t)$. Chaque évaluation de $\ell$ exige de résoudre l'équation de Bellman lisse pour le $\theta$ courant; le problème est donc à deux niveaux : une maximisation extérieure sur $\theta$ autour d'un problème intérieur de point fixe. Rust a appelé son algorithme l'algorithme du point fixe imbriqué (NFXP, pour *nested fixed point*). Le gradient de $\ell$ demande $\partial v_\theta / \partial\theta$, que fournit le théorème des fonctions implicites sans qu'il faille dériver à travers les itérations du solveur intérieur. En dérivant $v_\theta = \mathrm{L}_{\beta,\theta} v_\theta$ et en utilisant la dérivée de $\mathrm{L}_\beta$ de la section {ref}`sec-smooth-pi-newton`,

$$
\frac{\partial v_\theta}{\partial \theta} = (\mathbf{I} - \gamma\mathbf{P}_{\pi_\theta})^{-1}\, \frac{\partial \mathbf{r}_{\pi_\theta}}{\partial\theta}, \qquad \frac{\partial \mathbf{r}_{\pi_\theta}}{\partial\theta}(s) = \sum_a \pi_\theta(a\mid s)\, \frac{\partial r_\theta(s,a)}{\partial \theta}.
$$

La matrice à inverser est celle de l'évaluation de la politique, déjà factorisée par les pas de Newton du solveur intérieur. Pour des récompenses linéaires en les paramètres, $r_\theta(s,a) = \theta^\top\phi(s,a)$, cette dérivée est le vecteur des comptes de caractéristiques actualisés espérés sous la politique softmax, la quantité que {cite:t}`ziebart2008maximum` fait correspondre aux comptes de caractéristiques empiriques. Le score d'une observation est alors

$$
\nabla_\theta \log\pi_\theta(a\mid s) = \beta\Big(\nabla_\theta q_\theta(s,a) - \sum_{a'}\pi_\theta(a'\mid s)\,\nabla_\theta q_\theta(s,a')\Big), \qquad \nabla_\theta q_\theta(s,a) = \nabla_\theta r_\theta(s,a) + \gamma\sum_j p(j\mid s,a)\,\frac{\partial v_\theta}{\partial \theta}(j).
$$

````{prf:algorithm} Maximum de vraisemblance par point fixe imbriqué (NFXP)
:label: nfxp

**Entrée :** Paires état-action observées $\{(s_t, a_t)\}$, modèle de transition $p$, facteur d'actualisation $\gamma$, échelle $\beta$, paramètres initiaux $\theta_0$  
**Sortie :** Paramètres de récompense estimés $\hat\theta$

1. **répéter** pour $k = 0, 1, \ldots$
    1. (Problème intérieur) Résoudre $v = \mathrm{L}_{\beta,\theta_k} v$ : quelques balayages de l'{prf:ref}`smooth-value-iteration`, puis des pas de Newton ({ref}`sec-smooth-pi-newton`), avec démarrage à chaud à partir du $v$ précédent
    2. Calculer $\pi_{\theta_k}$, $q_{\theta_k}$ et $\partial v_{\theta_k}/\partial\theta$ par la formule des fonctions implicites ci-dessus
    3. Calculer $\nabla \ell(\theta_k) = \sum_t \nabla_\theta\log\pi_{\theta_k}(a_t\mid s_t)$
    4. (Pas extérieur) $\theta_{k+1} \leftarrow \theta_k + H_k^{-1}\nabla\ell(\theta_k)$, où $H_k$ est une estimation définie positive de la courbure (Rust utilisait le produit extérieur des scores, la méthode BHHH)
2. **jusqu'à ce que** $\|\nabla\ell(\theta_k)\|$ soit petit
3. **renvoyer** $\hat\theta = \theta_k$
````

Ce cadre repose sur des hypothèses fortes, et les nommer explique pourquoi l'apprentissage par renforcement inverse est difficile en général.

- *Rationalité.* On suppose que le décideur résout exactement le PDM. Les gens ne le font pas; la littérature sur la rationalité limitée étudie en quoi ils s'en écartent, et Rust lui-même a exprimé son scepticisme quant au fait que les gens suivent le principe d'optimalité de Bellman. L'hypothèse reste un point de départ utile, comme l'est un modèle linéaire.
- *Observabilité.* Rust observe l'état et l'action. Dans bien des applications, l'un ou l'autre manque, voire les deux. Avec des historiques de prix, par exemple, un prix inchangé peut être une décision de le maintenir ou simplement l'absence de décision, et l'action elle-même doit être inférée, ce qui rend le problème partiellement observable.
- *Identifiabilité.* De nombreuses fonctions de récompense expliquent tout aussi bien le même comportement. Avec des maxima stricts, une récompense constante rend toute politique optimale, et l'ajout d'un terme de façonnage $\gamma\Phi(s') - \Phi(s)$ à n'importe quelle récompense laisse les politiques optimales inchangées {cite}`NgHaradaRussell1999`. L'application qui associe un comportement aux récompenses n'est pas injective; les données peuvent donc au mieux isoler un ensemble de récompenses candidates. L'économétrie structurelle rétablit l'identification en restreignant la récompense à une famille paramétrique, en fixant le facteur d'actualisation et l'échelle $\beta$, et en normalisant la récompense d'une action. L'acceptabilité de ces restrictions dépend du but : prédire ce que ferait Zurcher face à de nouveaux prix de moteurs est une exigence plus faible que retrouver ce qu'il voulait « vraiment ».

## Synthèse et perspectives

La régularisation par l'entropie remplace la sélection stricte des actions par
un opérateur conjugué lisse et produit une politique optimale stochastique. Les
perturbations de Gumbel, les mises à jour de Bellman en log-sum-exp et
l'itération sur les politiques régularisée décrivent la même construction des
points de vue probabiliste, variationnel et algorithmique. Représentés comme
des modèles graphiques, le PDM original et le PDM augmenté par des chocs de
Gumbel diffèrent d'un nœud de choc par pas de temps, et l'intégration de ce
nœud redonne la politique softmax; le choc espéré sur l'action choisie est le
bonus d'entropie. Les mêmes équations valent pour des actions continues dès
qu'une mesure de référence est fixée, avec des intégrales à la place des sommes
et une densité de Gibbs à la place de la softmax, et le choix de cette mesure
fait alors partie du modèle.
L'itération sur les politiques lisse est la méthode de Newton appliquée à
l'équation de Bellman lisse, et l'estimateur à point fixe imbriqué de Rust
utilise la même dérivée pour retrouver les paramètres de récompense à partir du
comportement observé.

La même forme log-sum-exp apparaît sur des trajectoires entières en temps
continu. [La commande stochastique par intégrales de chemin](path-integral-stochastic-control.md)
étudie des systèmes soumis à des perturbations gaussiennes dont l'équation de
Hamilton-Jacobi-Bellman devient linéaire sous une transformation logarithmique,
de sorte que la rétroaction optimale peut être estimée à partir de trajectoires
pondérées.

Le lissage modifie l'opérateur, mais ne dispense pas de représenter sa fonction
de valeur inconnue. Comment réduire une équation fonctionnelle de dimension
infinie à un nombre fini de coefficients et de conditions de test? [Les
fonctions de valeur paramétriques](weighted-residual-methods.md) développent
l'appareil général d'approximation.

## Autoévaluation

:::{exercise} Limite de température
:label: ex-smoothing-check-1

Qu'advient-il d'une politique softmax lorsque la température d'entropie $\alpha$ tend vers zéro, en supposant une unique action maximisante?
:::

:::{solution} ex-smoothing-check-1
:class: dropdown

Sa probabilité se concentre sur l'action maximisante; la politique douce tend donc vers la politique gloutonne déterministe ordinaire.
:::

:::{exercise} Maximum lisse
:label: ex-smoothing-check-2

Pourquoi est-il souvent plus facile d'optimiser à travers un log-sum-exp qu'à travers un maximum strict?
:::

:::{solution} ex-smoothing-check-2
:class: dropdown

Le log-sum-exp est différentiable et combine l'information de toutes les actions. Un maximum strict n'est pas différentiable en cas d'égalité et ne transmet l'information locale de gradient que par les maximiseurs retenus.
:::

:::{exercise} L'entropie comme choc espéré
:label: ex-smoothing-check-3

Dans le PDM augmenté, l'agent ne randomise jamais. D'où vient, dans ses récompenses, le bonus d'entropie du PDM original?
:::

:::{solution} ex-smoothing-check-3
:class: dropdown

Des chocs sur les actions qu'il choisit. Sachant que l'action $a$ l'emporte, son choc a pour moyenne $-\frac{1}{\beta}\log\pi_\beta(a\mid s)$, qui est grande pour les actions qui l'emportent rarement. La moyenne sur l'action gagnante donne $\frac{1}{\beta}H(\pi_\beta(\cdot\mid s))$, le bonus d'entropie.
:::

:::{exercise} Unités d'une action continue
:label: ex-smoothing-check-4

Une action unidimensionnelle est mesurée en newtons-mètres, et vous passez aux kilonewtons-mètres. Qu'advient-il de la valeur optimale et de la politique optimale régularisées par l'entropie sous la mesure de référence de Lebesgue? Et si vous passez au logarithme du couple?
:::

:::{solution} ex-smoothing-check-4
:class: dropdown

Diviser l'action par $1000$ retranche $\log 1000$ à l'entropie différentielle de chaque densité; chaque valeur est donc décalée de la constante $-\frac{\log 1000}{\beta(1-\gamma)}$, et la politique, ramenée en newtons-mètres, est inchangée. Un changement de variable logarithmique est non linéaire : la mesure de Lebesgue dans la nouvelle variable est une mesure différente sur les couples; le régularisateur et la politique optimale changent donc tous les deux.
:::
