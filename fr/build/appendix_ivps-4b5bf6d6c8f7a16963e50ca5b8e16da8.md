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

# Résolution des problèmes à valeur initiale

Une équation différentielle ordinaire définit une trajectoire par son taux de
variation. Un **problème à valeur initiale** fournit l'état d'où part cette
trajectoire :

$$
\dot{\mathbf{x}}(t)=\mathbf{f}(t,\mathbf{x}(t)),
\qquad
\mathbf{x}(t_0)=\mathbf{x}_0.
$$

Sauf pour des champs de vecteurs particuliers, l'état à un instant ultérieur
n'a pas d'expression analytique. Un intégrateur numérique fait avancer une
approximation d'un point du maillage au suivant. Étant donné $t_n$, un état
approché $\mathbf{x}_n$ et un pas $h_n=t_{n+1}-t_n$, un pas définit une
application

$$
\mathbf{x}_{n+1}
=\Phi_{h_n}(t_n,\mathbf{x}_n)
\approx \mathbf{x}(t_{n+1}).
$$

Cette application à un pas est la partie d'un solveur d'EDO qu'utilisent les
méthodes de tir. Les sections qui suivent dérivent cinq applications à pas
fixe : Euler explicite, Euler implicite, la règle des trapèzes implicite, la
méthode prédicteur-correcteur de Heun et la méthode de Runge-Kutta classique
d'ordre quatre.

## L'identité intégrale derrière un pas

En intégrant l'EDO sur un intervalle, on obtient la relation exacte

$$
\mathbf{x}(t_{n+1})
=\mathbf{x}(t_n)
+\int_{t_n}^{t_{n+1}}
\mathbf{f}(s,\mathbf{x}(s))\,ds.
$$

Chaque méthode de cette annexe remplace l'intégrale inconnue par un nombre
fini d'évaluations de $\mathbf{f}$. Leurs positions, leurs poids et les
équations des étages déterminent ensemble la précision de la méthode, son coût
de calcul et sa stabilité.

Deux échelles d'erreur sont utiles. L'**erreur de troncature locale** mesure
l'erreur commise par un pas qui part de l'état exact. L'**erreur globale**
mesure l'erreur accumulée après avoir parcouru un intervalle de temps fixé.
Une méthode d'ordre $p$ a une erreur globale en $O(h^p)$ lorsqu'un pas uniforme
$h$ tend vers zéro, pourvu que la solution et le champ de vecteurs soient
suffisamment lisses.

## Euler progressif (explicite)

La méthode d'Euler progressive, aussi appelée **Euler explicite**, remplace le
champ de vecteurs sur tout l'intervalle par sa valeur à l'extrémité gauche :

$$
\mathbf{x}_{n+1}
=\mathbf{x}_n+h_n\mathbf{f}(t_n,\mathbf{x}_n).
$$

La mise à jour est explicite, car toutes les quantités du membre de droite
sont connues. Elle demande une évaluation du champ de vecteurs par pas et a
une précision globale d'ordre un. La même formule découle du développement de
Taylor d'ordre un de $\mathbf{x}(t_n+h_n)$.

Pour le système balistique

$$
\dot y=v,
\qquad
\dot v=-g,
$$

la méthode d'Euler explicite donne

$$
y_{n+1}=y_n+h_nv_n,
\qquad
v_{n+1}=v_n-h_ng.
$$

La mise à jour de la vitesse est exacte, car sa dérivée est constante. Celle
de la position ne l'est pas, car la vitesse change pendant l'intervalle.

## Euler rétrograde (implicite)

La méthode d'Euler rétrograde, aussi appelée **Euler implicite**, évalue le
champ de vecteurs à l'extrémité droite, inconnue :

$$
\mathbf{x}_{n+1}
=\mathbf{x}_n+h_n\mathbf{f}(t_{n+1},\mathbf{x}_{n+1}).
$$

Cette équation est implicite. Un pas exige de résoudre le résidu non linéaire

$$
\mathbf{r}(\mathbf{z})
:=\mathbf{z}-\mathbf{x}_n
-h_n\mathbf{f}(t_{n+1},\mathbf{z})
=\mathbf{0}
$$

en $\mathbf{z}=\mathbf{x}_{n+1}$. Une itération de point fixe ou un algorithme
de recherche de racines comme la méthode de Newton peut effectuer cette
résolution interne. Une résolution interne qui échoue ou converge mal modifie
l'application à un pas effective; sa tolérance sur le résidu fait donc partie
de la précision de l'intégration.

La méthode d'Euler implicite a une précision globale d'ordre un. Son principal
avantage est la stabilité : les modes fortement amortis restent stables pour
des tailles de pas qui feraient diverger la méthode d'Euler explicite.

## Règle des trapèzes implicite

La règle des trapèzes approche l'intégrale du champ de vecteurs par la moyenne
de ses valeurs aux deux extrémités :

$$
\mathbf{x}_{n+1}
=\mathbf{x}_n
+\frac{h_n}{2}
\left[
\mathbf{f}(t_n,\mathbf{x}_n)
+\mathbf{f}(t_{n+1},\mathbf{x}_{n+1})
\right].
$$

L'état à l'extrémité droite est inconnu; cette règle est donc implicite elle
aussi. Son résidu non linéaire est

$$
\mathbf{r}(\mathbf{z})
:=\mathbf{z}-\mathbf{x}_n
-\frac{h_n}{2}
\left[
\mathbf{f}(t_n,\mathbf{x}_n)
+\mathbf{f}(t_{n+1},\mathbf{z})
\right].
$$

Résoudre $\mathbf{r}(\mathbf{z})=\mathbf{0}$ avec une précision suffisante
produit une méthode d'ordre deux. La règle des trapèzes implicite est
symétrique en temps et stable sur tout le demi-plan complexe gauche.
Contrairement à la méthode d'Euler implicite, elle n'amortit toutefois pas
fortement les modes extrêmement rapides; un grand pas peut faire alterner le
signe de ces modes alors qu'ils décroissent très lentement.

## Itération prédicteur-correcteur et méthode de Heun

Une prédiction d'Euler fournit une estimation initiale commode pour l'équation
des trapèzes implicite :

$$
\mathbf{x}_{n+1}^{(0)}
=\mathbf{x}_n+h_n\mathbf{f}(t_n,\mathbf{x}_n).
$$

En substituant cette prédiction dans l'évaluation à l'extrémité droite, on
obtient

$$
\mathbf{x}_{n+1}^{(1)}
=\mathbf{x}_n
+\frac{h_n}{2}
\left[
\mathbf{f}(t_n,\mathbf{x}_n)
+\mathbf{f}(t_{n+1},\mathbf{x}_{n+1}^{(0)})
\right].
$$

S'arrêter après cette correction donne la **méthode de Heun**, aussi appelée
méthode des trapèzes explicite. C'est une méthode de Runge-Kutta explicite
d'ordre deux : la pente à l'extrémité est évaluée à l'état prédit par Euler,
et non à l'état final inconnu.

Répéter la correction,

$$
\mathbf{x}_{n+1}^{(m+1)}
=\mathbf{x}_n
+\frac{h_n}{2}
\left[
\mathbf{f}(t_n,\mathbf{x}_n)
+\mathbf{f}(t_{n+1},\mathbf{x}_{n+1}^{(m)})
\right],
$$

constitue une itération de point fixe, ou itération de Picard, pour l'équation
des trapèzes implicite. Si l'itération converge, sa limite est le pas des
trapèzes implicite. Ce n'est pas une itération de Newton. La méthode de Newton
utilise plutôt la jacobienne du résidu et effectue la mise à jour

$$
\mathbf{z}^{(m+1)}
=\mathbf{z}^{(m)}
-\left[
\mathbf{I}-\frac{h_n}{2}
\nabla_{\mathbf{x}}\mathbf{f}(t_{n+1},\mathbf{z}^{(m)})
\right]^{-1}
\mathbf{r}(\mathbf{z}^{(m)}).
$$

La distinction compte lorsque le pas est grand ou que la dynamique est raide.
L'itération de Picard exige que son application de point fixe soit
contractante, alors qu'une résolution de Newton bien globalisée peut converger
dans des cas où l'itération de Picard échoue.

## Méthode de Runge-Kutta classique d'ordre quatre

La méthode RK4 classique évalue quatre pentes. À partir de
$(t_n,\mathbf{x}_n)$, on définit

$$
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(t_n,\mathbf{x}_n),\\
\mathbf{k}_2 &= \mathbf{f}\!\left(t_n+\frac{h_n}{2},
  \mathbf{x}_n+\frac{h_n}{2}\mathbf{k}_1\right),\\
\mathbf{k}_3 &= \mathbf{f}\!\left(t_n+\frac{h_n}{2},
  \mathbf{x}_n+\frac{h_n}{2}\mathbf{k}_2\right),\\
\mathbf{k}_4 &= \mathbf{f}\!\left(t_n+h_n,
  \mathbf{x}_n+h_n\mathbf{k}_3\right).
\end{aligned}
$$

Le pas est

$$
\mathbf{x}_{n+1}
=\mathbf{x}_n
+\frac{h_n}{6}
\left(\mathbf{k}_1+2\mathbf{k}_2+2\mathbf{k}_3+\mathbf{k}_4\right).
$$

RK4 est explicite et a une précision globale d'ordre quatre pour les problèmes
lisses. Les deux évaluations au point milieu ont lieu au même instant, mais en
des états prédits différents. Ce sont des étages du calcul de Runge-Kutta, et
non les valeurs d'un même polynôme d'interpolation cubique en quatre nœuds de
collocation distincts. La méthode RK4 classique n'est donc pas une méthode de
collocation. Certaines méthodes de Runge-Kutta *implicites* s'obtiennent par
collocation de Gauss, de Radau ou de Lobatto, mais cette correspondance
n'inclut pas la méthode RK4 classique.

## Une implémentation compacte à pas fixe

Les implémentations ci-dessous exposent directement les applications à un pas.
Les méthodes implicites utilisent un solveur générique de racines non
linéaires, initialisé par une prédiction d'Euler explicite.

```{code-cell} python
import numpy as np
from scipy.optimize import root


def forward_euler_step(fun, t, x, h):
    x = np.asarray(x, dtype=float)
    return x + h * fun(t, x)


def backward_euler_step(fun, t, x, h):
    x = np.asarray(x, dtype=float)
    guess = forward_euler_step(fun, t, x, h)
    residual = lambda z: z - x - h * fun(t + h, z)
    solution = root(residual, guess)
    residual_norm = np.linalg.norm(residual(solution.x), ord=np.inf)
    tolerance = 1e-10 * (1 + np.linalg.norm(x, ord=np.inf))
    if residual_norm > tolerance:
        raise RuntimeError(
            f"backward Euler residual {residual_norm:.3e}: {solution.message}"
        )
    return solution.x


def implicit_trapezoid_step(fun, t, x, h):
    x = np.asarray(x, dtype=float)
    slope_left = fun(t, x)
    guess = x + h * slope_left
    residual = lambda z: z - x - 0.5 * h * (slope_left + fun(t + h, z))
    solution = root(residual, guess)
    residual_norm = np.linalg.norm(residual(solution.x), ord=np.inf)
    tolerance = 1e-10 * (1 + np.linalg.norm(x, ord=np.inf))
    if residual_norm > tolerance:
        raise RuntimeError(
            f"trapezoidal residual {residual_norm:.3e}: {solution.message}"
        )
    return solution.x


def heun_step(fun, t, x, h):
    x = np.asarray(x, dtype=float)
    slope_left = fun(t, x)
    prediction = x + h * slope_left
    return x + 0.5 * h * (slope_left + fun(t + h, prediction))


def rk4_step(fun, t, x, h):
    x = np.asarray(x, dtype=float)
    k1 = fun(t, x)
    k2 = fun(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = fun(t + 0.5 * h, x + 0.5 * h * k2)
    k4 = fun(t + h, x + h * k3)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_fixed_step(step, fun, x0, t0, tf, h):
    n_steps = int(np.ceil((tf - t0) / h))
    times = np.linspace(t0, tf, n_steps + 1)
    states = np.empty((n_steps + 1, np.size(x0)), dtype=float)
    states[0] = np.asarray(x0, dtype=float)
    for n in range(n_steps):
        step_size = times[n + 1] - times[n]
        states[n + 1] = step(fun, times[n], states[n], step_size)
    return times, states


decay = lambda t, x: -2.0 * x
exact_terminal = np.exp(-6.0)
methods = {
    "forward Euler": forward_euler_step,
    "backward Euler": backward_euler_step,
    "implicit trapezoid": implicit_trapezoid_step,
    "Heun": heun_step,
    "classical RK4": rk4_step,
}

for name, step in methods.items():
    _, states = integrate_fixed_step(step, decay, [1.0], 0.0, 3.0, 0.25)
    error = abs(states[-1, 0] - exact_terminal)
    print(f"{name:20s} terminal error: {error:.3e}")
```

## Stabilité et raideur

La précision décrit la façon dont l'erreur diminue lorsque $h$ tend vers zéro.
La stabilité demande si des pas numériques répétés préservent la décroissance
qualitative de l'EDO pour la taille de pas choisie. Une méthode d'ordre élevé
peut tout de même être instable si son pas est trop grand.

L'équation test scalaire

$$
\dot x=\lambda x,
\qquad \operatorname{Re}(\lambda)<0,
$$

a une solution exacte décroissante. L'application d'une méthode à un pas donne
$x_{n+1}=R(z)x_n$ avec $z=h\lambda$. La solution numérique ne croît pas là où
$|R(z)|\leq 1$ et décroît strictement là où $|R(z)|<1$. Les cinq méthodes ont
les fonctions de stabilité suivantes :

| méthode | ordre global | explicite? | fonction de stabilité $R(z)$ |
|---|---:|:---:|---|
| Euler explicite | 1 | oui | $1+z$ |
| Euler implicite | 1 | non | $1/(1-z)$ |
| trapèzes implicite | 2 | non | $(1+z/2)/(1-z/2)$ |
| Heun | 2 | oui | $1+z+z^2/2$ |
| RK4 classique | 4 | oui | $1+z+z^2/2+z^3/6+z^4/24$ |

Pour $\lambda<0$ réel, la méthode d'Euler explicite exige
$0<h<2/|\lambda|$. La méthode d'Euler implicite et la règle des trapèzes
implicite sont **A-stables** : leurs régions de stabilité contiennent tout le
demi-plan gauche. La méthode d'Euler implicite est aussi **L-stable**, puisque
$R(z)\to 0$ lorsque $z\to-\infty$. La règle des trapèzes a plutôt
$R(z)\to-1$; elle peut donc conserver une faible oscillation d'un pas à
l'autre dans des modes que la solution exacte amortit presque immédiatement.
Les méthodes de Heun et RK4 peuvent offrir une meilleure précision par pas que
la méthode d'Euler explicite sur des problèmes lisses non raides, mais elles
restent toutes deux des méthodes explicites aux régions de stabilité bornées.

Un problème à valeur initiale est **raide** lorsque des modes à décroissance
rapide obligent une méthode explicite à prendre des pas beaucoup plus courts
que l'échelle de temps sur laquelle la solution recherchée varie. Les méthodes
implicites coûtent plus cher par pas, car elles résolvent des équations non
linéaires, mais leur stabilité peut permettre des pas beaucoup plus grands. La
stabilité ne dispense pas de vérifier la précision : une solution grossière
stable peut tout de même être inexacte.

## L'application à un pas dans les méthodes de tir

Un système commandé ajoute une entrée prescrite sur chaque intervalle
d'intégration :

$$
\dot{\mathbf{x}}(t)
=\mathbf{f}(t,\mathbf{x}(t),\mathbf{u}(t)).
$$

Par exemple, avec un bloqueur d'ordre zéro, chaque étage utilisé pour avancer
sur $[t_n,t_{n+1}]$ reçoit la même valeur $\mathbf{u}_n$. L'application de
n'importe lequel des intégrateurs ci-dessus définit alors

$$
\mathbf{x}_{n+1}
=\Phi_{h_n}(t_n,\mathbf{x}_n,\mathbf{u}_n).
$$

Dans le **tir simple**, l'optimiseur choisit les paramètres de commande, et
chaque état s'obtient en appliquant $\Phi$ de façon répétée. Les états sont
des valeurs intermédiaires du calcul plutôt que des variables d'optimisation
indépendantes.

Dans le **tir multiple**, l'état initial de chaque segment est une variable
d'optimisation. L'intégration produit un état terminal prédit
$\widehat{\mathbf{s}}_{k+1}$ pour le segment $k$, et le programme non linéaire
impose le défaut de continuité

$$
\mathbf{s}_{k+1}-\widehat{\mathbf{s}}_{k+1}=\mathbf{0}.
$$

L'intégrateur fait donc partie du modèle que voit l'optimiseur. Sa taille de
pas, l'interpolation de la commande, la tolérance de la résolution non
linéaire et la différentiabilité influent toutes sur l'objectif et les
contraintes qui en résultent. Après l'optimisation, il convient de rejouer la
trajectoire de commande fixée avec une tolérance d'intégration plus serrée ou
un pas plus fin. Un grand écart entre le déroulement utilisé pendant
l'optimisation et ce déroulement de validation indique que le maillage
d'intégration était inadéquat.

## Du tir à la collocation

Le tir fait avancer un état de façon séquentielle au moyen d'une application à
un pas ou par segment. [L'optimisation de trajectoires en temps continu](continuous-time-collocation.md)
développe l'approche simultanée : les valeurs de l'état sur un intervalle
deviennent des variables d'optimisation, un polynôme interpole ces valeurs, et
l'EDO est imposée en des nœuds choisis. Ce chapitre traite des représentations
polynomiales, des matrices de dérivation, de la quadrature et de la
collocation directe. Garder ces idées là-bas évite une seconde dérivation,
concurrente, dans cette annexe sur les problèmes à valeur initiale.

## Résumé

Les méthodes d'Euler explicite, de Heun et RK4 classique sont des méthodes
explicites à un pas d'ordres un, deux et quatre. La méthode d'Euler implicite
et la règle des trapèzes implicite exigent une résolution non linéaire
interne; elles sont d'ordres un et deux et stables sur tout le demi-plan
gauche. Une prédiction d'Euler suivie d'une correction des trapèzes constitue
la méthode explicite de Heun. Des corrections répétées forment l'itération de
Picard pour l'équation des trapèzes implicite, tandis que la méthode de Newton
utilise la jacobienne du résidu.

Pour le tir, n'importe laquelle de ces méthodes devient une application de
transition finie $\Phi_h$. Le tir simple enchaîne l'application à partir d'un
seul état initial; le tir multiple enchaîne des intégrations plus courtes et
contraint leurs extrémités à concorder. La collocation directe remplace cette
construction séquentielle par des variables nodales simultanées et des
contraintes algébriques issues de l'EDO.
