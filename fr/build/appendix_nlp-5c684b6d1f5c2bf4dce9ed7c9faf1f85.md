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

# Programmation non linéaire

À moins de faire des hypothèses particulières sur la dynamique et la structure des coûts, un PCOTD est, dans sa forme la plus générale, un programme mathématique non linéaire (couramment appelé PNL, à ne pas confondre avec le traitement automatique du langage naturel). Un PNL peut se formuler ainsi :

$$
\begin{aligned}
\text{minimiser } & f(\mathbf{x}) \\
\text{sous les contraintes } & \mathbf{g}(\mathbf{x}) \leq \mathbf{0} \\
& \mathbf{h}(\mathbf{x}) = \mathbf{0}
\end{aligned}
$$

Ici, $\mathbf x$ désigne un vecteur de décision générique. Lorsqu'on applique
cette annexe à un problème de trajectoire, c'est le vecteur empilé $\mathbf z$,
et non un seul état physique. L'objectif $f$ et les applications de contrainte
$\mathbf g,\mathbf h$ correspondent alors à $F,G,H$ dans les chapitres sur la
commande.

où :
- $f: \mathbb{R}^n \to \mathbb{R}$ est la fonction objectif
- $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$ représente les contraintes d'inégalité
- $\mathbf{h}: \mathbb{R}^n \to \mathbb{R}^\ell$ représente les contraintes d'égalité

Contrairement à l'optimisation sans contraintes couramment utilisée en apprentissage profond, l'optimalité d'une solution en optimisation sous contraintes doit tenir compte à la fois de la valeur de l'objectif et de l'admissibilité vis-à-vis des contraintes. Pour l'illustrer, considérons le problème suivant, qui comporte à la fois des contraintes d'égalité et d'inégalité :

$$
\begin{align*}
\text{Minimiser} \quad & f(x_1, x_2) = (x_1 - 1)^2 + (x_2 - 2.5)^2 \\
\text{sous les contraintes} \quad & g(x_1, x_2) = (x_1 - 1)^2 + (x_2 - 1)^2 - 1.5 \leq 0, \\
& h(x_1, x_2) = x_2 - \left(0.5 \sin(2 \pi x_1) + 1.5\right) = 0.
\end{align*}
$$

Dans cet exemple, la fonction objectif $f(x_1, x_2)$ est quadratique, la contrainte d'inégalité $g(x_1, x_2)$ définit une région réalisable circulaire centrée en $(1, 1)$ de rayon $\sqrt{1.5}$, et la contrainte d'égalité $h(x_1, x_2)$ exige que $x_2$ se trouve sur une sinusoïde. Le code suivant montre la différence entre les solutions sans contraintes et sous contraintes de ce problème.

```{code-cell} python
:tags: [hide-input]

#  label: appendix_nlp-cell-01
#  caption: Rendered output from the preceding code cell.

%config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults
from scipy.optimize import minimize

# Define the objective function
def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

# SciPy SLSQP expects a nonnegative inequality function: return -g(x).
def constraint(x):
    return -(x[0] - 1)**2 - (x[1] - 1)**2 + 1.5

# Define the gradient of the objective function
def objective_gradient(x):
    return np.array([2*(x[0] - 1), 2*(x[1] - 2.5)])

# Define the gradient of the inequality constraint function
def constraint_gradient(x):
    return np.array([-2*(x[0] - 1), -2*(x[1] - 1)])

# Define the sine wave equality constraint function
def sine_wave_equality_constraint(x):
    return x[1] - (0.5 * np.sin(2 * np.pi * x[0]) + 1.5)

# Define the gradient of the sine wave equality constraint function
def sine_wave_equality_constraint_gradient(x):
    return np.array([-np.pi * np.cos(2 * np.pi * x[0]), 1])

# Define the constraints including the sine wave equality constraint
sine_wave_constraints = [{'type': 'ineq', 'fun': constraint, 'jac': constraint_gradient},  # Inequality constraint
                         {'type': 'eq', 'fun': sine_wave_equality_constraint, 'jac': sine_wave_equality_constraint_gradient}]  # Sine wave equality constraint

# Define only the inequality constraint
inequality_constraints = [{'type': 'ineq', 'fun': constraint, 'jac': constraint_gradient}]

# Initial guess
x0 = [1.25, 1.5]

# Solve the optimization problem with the sine wave equality constraint
res_sine_wave_constraint = minimize(objective, x0, method='SLSQP', jac=objective_gradient, 
                                    constraints=sine_wave_constraints, options={'disp': False})

x_opt_sine_wave_constraint = res_sine_wave_constraint.x

# Solve the optimization problem with only the inequality constraint
res_inequality_only = minimize(objective, x0, method='SLSQP', jac=objective_gradient, 
                               constraints=inequality_constraints, options={'disp': False})

x_opt_inequality_only = res_inequality_only.x

# Solve the unconstrained optimization problem for reference
res_unconstrained = minimize(objective, x0, method='SLSQP', jac=objective_gradient, options={'disp': False})
x_opt_unconstrained = res_unconstrained.x

# Generate data for visualization
x = np.linspace(-1, 4, 400)
y = np.linspace(-1, 4, 400)
X, Y = np.meshgrid(x, y)
Z = (X - 1)**2 + (Y - 2.5)**2  # Objective function values
constraint_values = (X - 1)**2 + (Y - 1)**2

# Data for sine wave constraint
x_sine = np.linspace(-1, 4, 400)
y_sine = 0.5 * np.sin(2 * np.pi * x_sine) + 1.5

# Visualization with Improved Color Scheme
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=100, cmap='viridis', alpha=0.6)  # Heatmap for the objective function

# Plot all the optimal points
plt.plot(x_opt_inequality_only[0], x_opt_inequality_only[1], 'ro', label='Optimal Solution (Inequality Only)', markersize=8, markeredgecolor='black')
plt.plot(x_opt_sine_wave_constraint[0], x_opt_sine_wave_constraint[1], 'mo', label='Optimal Solution (Sine Wave Equality & Inequality)', markersize=8, markeredgecolor='black')
plt.plot(x_opt_unconstrained[0], x_opt_unconstrained[1], 'co', label='Unconstrained Minimum', markersize=8, markeredgecolor='black')

# Adjust constraint boundary colors
plt.contour(X, Y, constraint_values, levels=[1.5], colors='navy', linewidths=2, linestyles='dashed')
plt.contourf(X, Y, constraint_values, levels=[0, 1.5], colors='skyblue', alpha=0.3)

# Plot the sine wave equality constraint with a high contrast color
plt.plot(x_sine, y_sine, 'lime', linestyle='--', linewidth=2, label='Sine Wave Equality Constraint')

plt.xlim([-1, 4])
plt.ylim([-1, 4])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Example NLP')
plt.legend(loc='upper left', fontsize='small', edgecolor='black', fancybox=True)
plt.grid(True)
# Set the aspect ratio to be equal so the circle appears correctly
plt.gca().set_aspect('equal', adjustable='box')
```


(the-lagrangian-duality-and-optimality-conditions)=
## Le lagrangien, la dualité et les conditions d'optimalité

Un optimiseur sous contraintes doit faire diminuer l'objectif tout en
respectant les contraintes. Le lagrangien réunit ces deux exigences en une
seule fonction :

$$
\mathcal L(\mathbf x,\boldsymbol\lambda,\boldsymbol\mu)
=f(\mathbf x)+\sum_{i=1}^m\mu_i g_i(\mathbf x)
+\sum_{j=1}^\ell\lambda_j h_j(\mathbf x),
\qquad \boldsymbol\mu\ge\mathbf0,
\quad \boldsymbol\lambda\in\mathbb R^\ell.
$$

Les **multiplicateurs de Lagrange** attribuent des poids aux résidus des
contraintes. Les multiplicateurs d'inégalité $\mu_i$ sont positifs ou nuls,
car un résidu positif $g_i(\mathbf x)>0$ est une violation qui doit faire
augmenter la quantité minimisée. Les multiplicateurs d'égalité $\lambda_j$
peuvent être de signe quelconque, car $h_j(\mathbf x)$ peut violer sa
contrainte dans un sens comme dans l'autre. Ces signes suivent la convention
$\mathbf g(\mathbf x)\le\mathbf0$ utilisée ici.

### Le problème primal comme problème min–max

Supposons qu'un joueur choisisse $\mathbf x$ pour minimiser le lagrangien et
qu'un second joueur choisisse les multiplicateurs pour le maximiser. Si le
joueur des multiplicateurs peut répondre après avoir vu $\mathbf x$, tout choix
non réalisable peut être écarté. Pour une inégalité violée $g_i(\mathbf x)>0$,
faire tendre $\mu_i$ vers l'infini rend le lagrangien arbitrairement grand.
Pour une égalité violée, choisir $\lambda_j$ du même signe que
$h_j(\mathbf x)$ et faire croître sa valeur absolue a le même effet.

Pour un $\mathbf x$ réalisable, tous les termes d'égalité s'annulent et chaque
terme d'inégalité est négatif ou nul. Le joueur des multiplicateurs peut
atteindre $f(\mathbf x)$ en posant $\boldsymbol\mu=\mathbf0$, et ne peut pas
obtenir une valeur plus grande. Ainsi,

$$
\sup_{\boldsymbol\mu\ge\mathbf0,\,\boldsymbol\lambda\in\mathbb R^\ell}
\mathcal L(\mathbf x,\boldsymbol\lambda,\boldsymbol\mu)
=
\begin{cases}
f(\mathbf x),&\mathbf x\text{ est réalisable},\\
+\infty,&\mathbf x\text{ n'est pas réalisable}.
\end{cases}
$$

Minimiser cette fonction redonne exactement le problème sous contraintes
original :

$$
p^\star
:=\inf_{\mathbf x:\,\mathbf g(\mathbf x)\le\mathbf0,\,\mathbf h(\mathbf x)=\mathbf0}
f(\mathbf x)
=\inf_{\mathbf x}\sup_{\boldsymbol\mu\ge\mathbf0,\,\boldsymbol\lambda\in\mathbb R^\ell}
\mathcal L(\mathbf x,\boldsymbol\lambda,\boldsymbol\mu).
$$

Cette représentation n'exige ni convexité ni différentiabilité. Les notations
$\inf$ et $\sup$ admettent des valeurs approchées sans être atteintes; en
particulier, aucun multiplicateur fini n'atteint $+\infty$ en un point non
réalisable. La représentation utilise aussi des ensembles de multiplicateurs
non bornés. Plafonner les multiplicateurs donnerait plutôt une pénalité finie
pour les violations des contraintes.

### Le problème dual et les bornes inférieures

Inverser l'ordre modifie ce que le joueur des multiplicateurs peut imposer. Il
doit maintenant choisir un seul jeu de poids avant que le joueur qui minimise
choisisse $\mathbf x$. Pour des multiplicateurs fixés, on définit la
**fonction duale**

$$
q(\boldsymbol\lambda,\boldsymbol\mu)
=\inf_{\mathbf x}\mathcal L(\mathbf x,\boldsymbol\lambda,\boldsymbol\mu).
$$

L'infimum intérieur porte sur tous les $\mathbf x$, y compris les choix non
réalisables. Pour tout $\bar{\mathbf x}$ réalisable et tout
$\boldsymbol\mu\ge\mathbf0$,

$$
q(\boldsymbol\lambda,\boldsymbol\mu)
\le\mathcal L(\bar{\mathbf x},\boldsymbol\lambda,\boldsymbol\mu)
\le f(\bar{\mathbf x}).
$$

Par conséquent, chaque choix de multiplicateurs fournit une borne inférieure
de la valeur optimale sous contraintes. Le **problème dual** cherche la plus
grande de ces bornes :

$$
d^\star
:=\sup_{\boldsymbol\mu\ge\mathbf0,\,\boldsymbol\lambda\in\mathbb R^\ell}
q(\boldsymbol\lambda,\boldsymbol\mu)
\le p^\star.
$$

Cette inégalité est la **dualité faible**. Elle vaut aussi pour les problèmes
non convexes. Lorsque $d^\star=p^\star$, on a **dualité forte** : choisir les
multiplicateurs en premier donne la même valeur que de les laisser répondre à
$\mathbf x$. L'égalité de ces valeurs n'indique pas à elle seule que l'un ou
l'autre optimum est atteint.

### Points de selle et multiplicateurs fixés

Un **point de selle du lagrangien** est un point réalisable $\mathbf x^\star$
et une paire de multiplicateurs $(\boldsymbol\mu^\star,\boldsymbol\lambda^\star)$,
avec $\boldsymbol\mu^\star\ge\mathbf0$, qui satisfont

$$
\mathcal L(\mathbf x^\star,\boldsymbol\lambda,\boldsymbol\mu)
\le
\mathcal L(\mathbf x^\star,\boldsymbol\lambda^\star,\boldsymbol\mu^\star)
\le
\mathcal L(\mathbf x,\boldsymbol\lambda^\star,\boldsymbol\mu^\star)
$$

pour tout $\mathbf x$, tout $\boldsymbol\mu\ge\mathbf0$ et tout
$\boldsymbol\lambda\in\mathbb R^\ell$. L'inégalité de gauche dit que le joueur
des multiplicateurs ne peut pas augmenter la valeur lorsque $\mathbf x^\star$
est fixé. L'inégalité de droite dit que le joueur qui minimise ne peut pas la
diminuer lorsque ces multiplicateurs sont fixés. C'est un équilibre du jeu à
deux joueurs.

L'inégalité de droite est plus forte que la résolution du problème min–max
primal. Elle exige que $\mathbf x^\star$ minimise le lagrangien sur tous les
$\mathbf x$ face à une paire de multiplicateurs fixée, même lorsque le
$\mathbf x$ concurrent n'est pas réalisable. Un point de selle certifie donc
l'optimalité globale primale et duale, avec des valeurs égales.
Réciproquement, si les optimums primal et dual sont tous deux atteints et que
leurs valeurs finies concordent, leurs optimiseurs forment un point de selle.
La convexité est une façon d'obtenir ces propriétés, mais elle ne fait pas
partie de la définition du point de selle.

### Conditions de Karush-Kuhn-Tucker

Pour un problème lisse, un minimum global du lagrangien en $\mathbf x$ doit
avoir un gradient nul. Le côté des multiplicateurs dans les inégalités de
selle exige aussi l'admissibilité et la complémentarité. Ces exigences donnent
les **conditions de Karush-Kuhn-Tucker (KKT)** :

````{prf:definition} Conditions KKT
:label: kkt-conditions

Un point $(\mathbf x^\star,\boldsymbol\lambda^\star,\boldsymbol\mu^\star)$
satisfait les conditions KKT lorsque

$$
\begin{aligned}
\nabla_{\mathbf x}\mathcal L(\mathbf x^\star,\boldsymbol\lambda^\star,\boldsymbol\mu^\star)
&=\mathbf0 &&\text{(stationnarité)},\\
\mathbf g(\mathbf x^\star)\le\mathbf0,\qquad
\mathbf h(\mathbf x^\star)&=\mathbf0 &&\text{(admissibilité primale)},\\
\boldsymbol\mu^\star&\ge\mathbf0 &&\text{(admissibilité duale)},\\
\mu_i^\star g_i(\mathbf x^\star)&=0\quad\text{pour tout }i
&&\text{(complémentarité)}.
\end{aligned}
$$
````

La complémentarité décrit la réponse du joueur des multiplicateurs à un point
réalisable. Si $g_i(\mathbf x^\star)<0$, un multiplicateur positif ferait
diminuer le lagrangien; la maximisation exige donc $\mu_i^\star=0$. Si
$g_i(\mathbf x^\star)=0$, tout multiplicateur positif ou nul donne la même
contribution en ce point. Une contrainte saturée peut donc porter un poids
positif, bien que la saturation n'exige pas que ce poids soit positif.

Les conditions KKT apparaissent aussi en des minimums locaux sous contraintes,
même lorsqu'aucun point de selle n'existe. Si l'objectif et les contraintes
sont continûment dérivables et qu'une **qualification des contraintes** est
satisfaite en un minimiseur local, alors il existe des multiplicateurs qui
satisfont les conditions KKT. Une qualification suffisante est la **LICQ**,
l'indépendance linéaire des gradients des contraintes d'égalité et des
gradients des inégalités actives en ce point. Elle garantit que les
contraintes linéarisées sont assez régulières pour que le théorème des
multiplicateurs s'applique.

Ici, **du premier ordre** signifie que les conditions utilisent les valeurs
des fonctions et leurs dérivées premières au point candidat. La stationnarité
exige

$$
\nabla f(\mathbf x^\star)
+\sum_i\mu_i^\star\nabla g_i(\mathbf x^\star)
+\sum_j\lambda_j^\star\nabla h_j(\mathbf x^\star)=\mathbf0.
$$

Elle équilibre le gradient de l'objectif et une somme pondérée des gradients
des contraintes. Elle ne compare pas les valeurs du lagrangien en d'autres
points et ne détermine pas sa courbure. Les conditions KKT seules n'impliquent
donc pas nécessairement que $\mathbf x^\star$ minimise le lagrangien, même
localement. Il ne faut pas confondre la description des conditions KKT comme
des équations et des inégalités en un point avec les comparaisons globales des
inégalités de selle.

Considérons par exemple

$$
\min_{x\in\mathbb R}-x^2\qquad\text{sous la contrainte }x=0,
\qquad \mathcal L(x,\lambda)=-x^2+\lambda x.
$$

L'unique point réalisable est $x^\star=0$; donc $p^\star=0$. La
représentation primale exacte fonctionne encore : en $x=0$, le supremum
intérieur est nul, et en tout $x\ne0$, il vaut $+\infty$. Le gradient de
l'égalité vaut $1$; la LICQ est donc satisfaite, et la stationnarité donne
$-2x^\star+\lambda^\star=0$. Ainsi, $(0,0)$ satisfait les conditions KKT.
Pourtant, $\mathcal L(x,0)=-x^2$ a un maximum strict en zéro. Pour tout
$\lambda$ fixé, il n'est pas non plus borné inférieurement lorsque $|x|$
croît. Par conséquent, $q(\lambda)=-\infty$ pour tout $\lambda$,
$d^\star=-\infty$, et aucun point de selle n'existe. Même un optimiseur global
sous contraintes qui satisfait les conditions KKT ne minimise pas
nécessairement le lagrangien.

### Convexité et conditions suffisantes

Supposons que $f$ et chaque $g_i$ soient des fonctions convexes dérivables sur
$\mathbb R^n$, et que les fonctions d'égalité $h_j$ soient affines. Pour tout
$\boldsymbol\mu\ge\mathbf0$, le lagrangien est alors convexe en $\mathbf x$.
La stationnarité devient suffisante pour un minimum global, car l'inégalité de
convexité donne

$$
\begin{aligned}
\mathcal L(\mathbf x,\boldsymbol\lambda^\star,\boldsymbol\mu^\star)
&\ge \mathcal L(\mathbf x^\star,\boldsymbol\lambda^\star,\boldsymbol\mu^\star)\\
&\quad+\nabla_{\mathbf x}\mathcal L(\mathbf x^\star,\boldsymbol\lambda^\star,\boldsymbol\mu^\star)^\top
(\mathbf x-\mathbf x^\star)\\
&=\mathcal L(\mathbf x^\star,\boldsymbol\lambda^\star,\boldsymbol\mu^\star).
\end{aligned}
$$

On obtient ainsi l'inégalité de selle de droite. L'admissibilité primale et
des multiplicateurs positifs ou nuls donnent
$\mathcal L(\mathbf x^\star,\boldsymbol\lambda,\boldsymbol\mu)\le f(\mathbf x^\star)$,
tandis que la complémentarité donne
$\mathcal L(\mathbf x^\star,\boldsymbol\lambda^\star,\boldsymbol\mu^\star)=f(\mathbf x^\star)$.
Ensemble, elles fournissent l'inégalité de selle de gauche. Ainsi, tout point
KKT de ce problème convexe est un point de selle global; aucune qualification
des contraintes supplémentaire n'est nécessaire pour ce sens de la
suffisance.

Une qualification des contraintes est nécessaire pour garantir l'existence de
multiplicateurs KKT en un optimiseur. Pour le problème convexe qui vient d'être
spécifié, la **condition de Slater** exige un point qui satisfait toutes les
égalités et chaque inégalité au sens strict. Si la condition de Slater est
satisfaite et que la valeur optimale primale est finie, la dualité forte est
vérifiée et l'optimum dual est atteint. Si l'optimum primal est lui aussi
atteint, un optimiseur et des multiplicateurs convenables satisfont les
conditions KKT et forment un point de selle. Ces résultats de dualité convexe
sont développés au [chapitre 5 de *Convex Optimization* de Boyd et Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

Pour les problèmes de trajectoire non convexes, les conditions KKT fournissent
des conditions nécessaires sous des hypothèses de régularité, et les méthodes
numériques cherchent des points qui satisfont ces conditions. Les inégalités
de selle globales exigent un argument distinct. La représentation min–max
primale exacte reste valide dans les deux cas.

### Multiplicateurs dans l'exemple sous contraintes

Résolvons maintenant notre problème d'exemple ci-dessus, cette fois avec [Ipopt](https://coin-or.github.io/Ipopt/) au moyen de l'interface [Pyomo](http://www.pyomo.org/), afin d'accéder aux multiplicateurs de Lagrange trouvés par le solveur.

```{code-cell} python

#  label: appendix_nlp-cell-02
#  caption: Rendered output from the preceding code cell.

from pyomo.environ import *
from pyomo.opt import SolverFactory
import math

# Define the Pyomo model
model = ConcreteModel()

# Define the variables
model.x1 = Var(initialize=1.25)
model.x2 = Var(initialize=1.5)

# Define the objective function
def objective_rule(model):
    return (model.x1 - 1)**2 + (model.x2 - 2.5)**2
model.obj = Objective(rule=objective_rule, sense=minimize)

# Define the inequality constraint (circle)
def inequality_constraint_rule(model):
    return (model.x1 - 1)**2 + (model.x2 - 1)**2 <= 1.5
model.ineq_constraint = Constraint(rule=inequality_constraint_rule)

# Define the equality constraint (sine wave) using Pyomo's math functions
def equality_constraint_rule(model):
    return model.x2 == 0.5 * sin(2 * math.pi * model.x1) + 1.5
model.eq_constraint = Constraint(rule=equality_constraint_rule)

# Create a suffix component to capture dual values
model.dual = Suffix(direction=Suffix.IMPORT)

# Create a solver
solver=SolverFactory('ipopt')

# Solve the problem
results = solver.solve(model, tee=False)

# Check if the solver found an optimal solution
if (results.solver.status == SolverStatus.ok and 
    results.solver.termination_condition == TerminationCondition.optimal):
    
    # Print the results
    print(f"x1: {value(model.x1)}")
    print(f"x2: {value(model.x2)}")
    
    # Print the objective value
    print(f"Objective value: {value(model.obj)}")

    # Print the Lagrange multipliers (dual values)
    print("\nLagrange multipliers:")
    ineq_lambda = None
    eq_lambda = None
    for c in model.component_objects(Constraint, active=True):
        for index in c:
            dual_val = model.dual[c[index]]
            print(f"{c.name}[{index}]: {dual_val}")
            if c.name == "ineq_constraint":
                ineq_lambda = dual_val
            elif c.name == "eq_constraint":
                eq_lambda = dual_val
else:
    print("Solver did not find an optimal solution.")
    print(f"Solver Status: {results.solver.status}")
    print(f"Termination Condition: {results.solver.termination_condition}")
```


Le point calculé se trouve strictement à l'intérieur du cercle; la
complémentarité exige donc que son multiplicateur d'inégalité soit nul, à la
tolérance numérique près. L'inférence réciproque serait invalide : un
multiplicateur nul ne montre pas à lui seul qu'une contrainte est inactive.
Chaque égalité doit être satisfaite en un point réalisable, quelle que soit la
valeur de son multiplicateur. Sous des hypothèses de sensibilité convenables,
les multiplicateurs décrivent aussi comment la valeur optimale varie lorsque
les membres de droite des contraintes sont perturbés; le signe dépend de la
convention utilisée pour écrire ces perturbations.

Pour une application de contrainte vectorielle, on note $J_h(\mathbf x)$ sa
jacobienne, dont la $j$-ième ligne est $\nabla h_j(\mathbf x)^\top$; on
définit $J_g$ de la même façon. Les gradients des fonctions scalaires sont des
vecteurs colonnes.

#### Théorème des multiplicateurs de Lagrange

Avec des contraintes d'égalité seulement, les conditions nécessaires KKT se
réduisent au théorème des multiplicateurs de Lagrange. Elles identifient des
candidats stationnaires; d'autres conditions sont nécessaires pour établir
qu'un candidat est un minimum local.

````{prf:theorem} Théorème des multiplicateurs de Lagrange
Considérons le problème d'optimisation sous contraintes :

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{sous les contraintes} \quad & h_i(\mathbf{x}) = 0, \quad i = 1, \ldots, \ell
\end{aligned}
$$

où $\mathbf{x} \in \mathbb{R}^n$, $f: \mathbb{R}^n \to \mathbb{R}$ et $h_i: \mathbb{R}^n \to \mathbb{R}$ pour $i = 1, \ldots, \ell$.

Supposons que :
1. $f$ et les $h_i$ sont des fonctions continûment dérivables.
2. Les gradients $\nabla h_i(\mathbf{x}^*)$ sont linéairement indépendants en un minimiseur local $\mathbf{x}^*$.

Il existe alors des multiplicateurs de Lagrange uniques $\lambda_i^* \in \mathbb{R}$, $i = 1, \ldots, \ell$, tels que les conditions d'optimalité du premier ordre suivantes soient satisfaites :

1. Stationnarité : $\nabla f(\mathbf{x}^*) + \sum_{i=1}^\ell \lambda_i^* \nabla h_i(\mathbf{x}^*) = \mathbf{0}$
2. Admissibilité primale : $h_i(\mathbf{x}^*) = 0$, pour $i = 1, \ldots, \ell$
````

Remarquons que les énoncés de stationnarité et d'admissibilité primale disent simplement que la dérivée du lagrangien par rapport aux variables primales ou duales doit être nulle en une solution optimale sous contraintes. Autrement dit :

$$
\nabla_{\mathbf{x}, \boldsymbol{\lambda}} \mathcal L(\mathbf{x}^*, \boldsymbol{\lambda}^*) = \mathbf{0}
$$

Notons $\mathbf F(\mathbf x,\boldsymbol\lambda)$ ce gradient combiné. Sous les
hypothèses du théorème, un minimiseur local et ses multiplicateurs donnent un
zéro de $\mathbf F$. La méthode de Newton peut chercher de tels zéros, mais en
atteindre un ne suffit pas à certifier un minimum ou un point de selle du
lagrangien.

#### Méthode de Newton

Pour un résidu vectoriel $\mathbf F$, on note $J_F$ sa jacobienne, dont les lignes sont les gradients des composantes. Ce résidu vectoriel est distinct de l'objectif scalaire $F$ utilisé en optimisation de trajectoires.

La méthode de Newton est une procédure numérique de résolution des problèmes de recherche de racines. Il s'agit de systèmes d'équations non linéaires de la forme :

Trouver $\mathbf{z}^* \in \mathbb{R}^n$ tel que $\mathbf{F}(\mathbf{z}^*) = \mathbf{0}$

où $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$ est une fonction continûment dérivable. La méthode de Newton consiste alors à appliquer la suite d'itérés suivante :

$$
\mathbf{z}^{k+1} = \mathbf{z}^k - [J_F(\mathbf{z}^k)]^{-1} \mathbf{F}(\mathbf{z}^k)
$$

où $\mathbf{z}^k$ est le $k$-ième itéré et $J_F(\mathbf{z}^k)$ est la matrice jacobienne de $\mathbf{F}$ évaluée en $\mathbf{z}^k$.

La méthode de Newton présente une convergence quadratique locale : si l'estimation initiale $\mathbf{z}^0$ est suffisamment proche de la vraie solution $\mathbf{z}^*$ et que $J_F(\mathbf{z}^*)$ est inversible, la méthode converge quadratiquement vers $\mathbf{z}^*$ {cite}`ortega_rheinboldt_1970`. La méthode est toutefois sensible à l'estimation initiale; si celle-ci est trop éloignée de la solution recherchée, la méthode de Newton peut ne pas converger ou converger vers une autre racine. Pour atténuer ce problème, on a développé un ensemble de techniques appelées méthodes de continuation numérique {cite}`allgower_georg_1990`. Ces méthodes élargissent en pratique le bassin d'attraction de la méthode de Newton en résolvant une suite de problèmes apparentés, d'un problème facile jusqu'au problème cible. Cette approche rappelle plusieurs notions de l'apprentissage automatique et de l'inférence statistique : l'apprentissage par curriculum en apprentissage automatique, où les modèles sont entraînés sur des données de plus en plus complexes; le recuit (*tempering*) des échantillonneurs de Monte-Carlo par chaînes de Markov (MCMC), qui ajuste graduellement la distribution cible pour améliorer le mélange; et les modèles de diffusion modernes, qui reposent sur une idée semblable de transformation graduelle du bruit en données structurées.

##### Implémentation efficace de la méthode de Newton

Remarquons que chaque pas de la méthode de Newton fait intervenir l'inverse d'une matrice jacobienne. Or, une règle cardinale de l'algèbre linéaire numérique est d'éviter de calculer explicitement des inverses de matrices : il ne devrait que rarement, voire jamais, y avoir de `np.linalg.inv` dans votre code. L'approche numériquement stable et efficace consiste plutôt à résoudre un système d'équations linéaires à chaque pas.
Étant donné l'itéré de la méthode de Newton :

$$
\mathbf{z}^{k+1} = \mathbf{z}^k - [J_F(\mathbf{z}^k)]^{-1} \mathbf{F}(\mathbf{z}^k)
$$

on peut le reformuler comme une procédure en deux étapes :

1. Résoudre le système linéaire : $\underbrace{[J_F(\mathbf{z}^k)]}_{\mathbf{A}} \Delta \mathbf{z}^k = -\mathbf{F}(\mathbf{z}^k)$
2. Mettre à jour : $\mathbf{z}^{k+1} = \mathbf{z}^k + \Delta \mathbf{z}^k$

La structure du système linéaire de l'étape 1 se prête souvent à des méthodes de résolution spécialisées. Dans le contexte de la différentiation automatique, les solveurs linéaires sans matrice sont particulièrement utiles. Ces solveurs peuvent trouver une solution sans former explicitement la matrice A; il leur suffit de pouvoir évaluer des produits matrice-vecteur ou vecteur-matrice. Parmi les exemples typiques de telles méthodes figurent les méthodes classiques de décomposition de matrice (par exemple l'itération de Richardson) ou les méthodes du gradient conjugué, par exemple au moyen de [`sparse.linalg.cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html). Une autre méthode utile est la méthode du résidu minimal généralisé (GMRES), implémentée dans SciPy au moyen de [`sparse.linalg.gmres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html), qui est utile face à des systèmes non symétriques et indéfinis.

En examinant la structure de la matrice $\mathbf{A}$ dans l'application particulière où la fonction $\mathbf{F}$ est la dérivée du lagrangien, nous découvrirons aussi une structure importante appelée la matrice KKT. Cette structure nous permettra ensuite de dériver un sous-problème de programmation quadratique comme partie d'une procédure itérative plus large de résolution des problèmes sous contraintes d'égalité et d'inégalité par programmation quadratique successive (PQS).

#### Résolution des programmes sous contraintes d'égalité par la méthode de Newton

Pour chercher des candidats stationnaires en optimisation sous contraintes
d'égalité, la méthode de Newton cherche un zéro de la fonction $\mathbf{F}(\mathbf{z}) = \nabla_{\mathbf{x}, \boldsymbol{\lambda}} \mathcal L(\mathbf{x}, \boldsymbol{\lambda})$. Ici, $\mathbf{F}$ représente la dérivée du lagrangien, et $\mathbf{z} = (\mathbf{x}, \boldsymbol{\lambda})$ réunit les variables primales $\mathbf{x}$ et les variables duales (multiplicateurs de Lagrange) $\boldsymbol{\lambda}$. Explicitement :

$$
\mathbf{F}(\mathbf{z}) = \begin{bmatrix} \nabla_{\mathbf{x}} \mathcal L(\mathbf{x}, \boldsymbol{\lambda}) \\ \mathbf{h}(\mathbf{x}) \end{bmatrix} = \begin{bmatrix} \nabla f(\mathbf{x}) + \sum_{i=1}^\ell \lambda_i \nabla h_i(\mathbf{x}) \\ \mathbf{h}(\mathbf{x}) \end{bmatrix}.
$$

La méthode de Newton consiste à linéariser $\mathbf{F}(\mathbf{z})$ autour de l'itéré courant $\mathbf{z}^k = (\mathbf{x}^k, \boldsymbol{\lambda}^k)$, puis à résoudre le système linéaire qui en résulte. À chaque itération $k$, la méthode de Newton met à jour l'estimation courante en résolvant le système linéaire :

$$
\mathbf{z}^{k+1} = \mathbf{z}^k - [J_F(\mathbf{z}^k)]^{-1} \mathbf{F}(\mathbf{z}^k).
$$

Au lieu d'inverser explicitement la matrice jacobienne $J_F(\mathbf{z}^k)$, on résout toutefois le système linéaire :

$$
\underbrace{J_F(\mathbf{z}^k)}_{\mathbf{A}} \Delta \mathbf{z}^k = -\mathbf{F}(\mathbf{z}^k),
$$

où $\Delta \mathbf{z}^k = (\Delta \mathbf{x}^k, \Delta \boldsymbol{\lambda}^k)$ représente le pas de Newton pour les variables primales et duales. En substituant l'expression de $\mathbf{F}(\mathbf{z})$ et de sa jacobienne, le système devient :

$$
\begin{bmatrix}
\nabla^2_{\mathbf{x}\mathbf{x}} \mathcal L(\mathbf{x}^k, \boldsymbol{\lambda}^k) & J_h(\mathbf{x}^k)^T \\
J_h(\mathbf{x}^k) & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{x}^k \\
\Delta \boldsymbol{\lambda}^k
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla f(\mathbf{x}^k) + J_h(\mathbf{x}^k)^T \boldsymbol{\lambda}^k \\
\mathbf{h}(\mathbf{x}^k)
\end{bmatrix}.
$$

La matrice du membre de gauche est appelée la matrice KKT, puisqu'elle découle des conditions de Karush-Kuhn-Tucker de ce problème d'optimisation.
La solution de ce système fournit les mises à jour $\Delta \mathbf{x}^k$ et $\Delta \boldsymbol{\lambda}^k$, qui servent ensuite à mettre à jour les variables primales et duales :

$$
\mathbf{x}^{k+1} = \mathbf{x}^k + \Delta \mathbf{x}^k, \quad \boldsymbol{\lambda}^{k+1} = \boldsymbol{\lambda}^k + \Delta \boldsymbol{\lambda}^k.
$$

##### Démonstration
Le code suivant montre comment mettre en œuvre cette idée avec Jax. Dans cette démonstration, on minimise une fonction objectif quadratique sous une seule contrainte d'égalité, un problème qui s'énonce formellement ainsi :

$$
\begin{aligned}
\min_{x \in \mathbb{R}^2} \quad & f(x) = (x_1 - 2)^2 + (x_2 - 1)^2 \\
\text{sous la contrainte} \quad & h(x) = x_1^2 + x_2^2 - 1 = 0
\end{aligned}
$$

Géométriquement, la contrainte $h(x)$ décrit un cercle unité centré à l'origine. Pour résoudre ce problème par la méthode des multiplicateurs de Lagrange, on forme le lagrangien :

$$
\mathcal L(x, \lambda) = f(x) + \lambda h(x) = (x_1 - 2)^2 + (x_2 - 1)^2 + \lambda(x_1^2 + x_2^2 - 1)
$$

Pour ce problème particulier, il se trouve qu'on peut aussi trouver une solution analytique sans même recourir à la méthode de Newton. Les conditions d'optimalité du premier ordre donnent le système d'équations suivant :
\begin{align*}
   2(x_1 - 2) + 2\lambda x_1 &= 0 \\
   2(x_2 - 1) + 2\lambda x_2 &= 0 \\
   x_1^2 + x_2^2 - 1 &= 0\\
\end{align*}

Des deux premières équations, on tire :
 
   $$x_1 = \frac{2}{1 + \lambda}, \quad x_2 = \frac{1}{1 + \lambda}$$

que l'on peut substituer dans la troisième équation, celle de la contrainte, pour obtenir :
  
  $$(\frac{2}{1 + \lambda})^2 + (\frac{1}{1 + \lambda})^2 = 1 \Leftrightarrow \lambda = \sqrt{5} - 1$$

Cette valeur du multiplicateur de Lagrange peut ensuite être substituée dans les équations ci-dessus pour obtenir $x_1 = \frac{2}{\sqrt{5}}$ et $x_2 =  \frac{1}{\sqrt{5}}$.
On peut vérifier numériquement (et visuellement sur le graphique suivant) que le point $(2/\sqrt{5}, 1/\sqrt{5})$ est bien le point du cercle unité le plus proche de $(2, 1)$.

```{code-cell} python
:tags: [hide-input]

#  label: appendix_nlp-cell-03
#  caption: Rendered output from the preceding code cell.

%config InlineBackend.figure_format = 'retina'
import jax
import jax.numpy as jnp
from jax import grad, jit, jacfwd
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults

# Define the objective function and constraint
def f(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

def h(x):
    return x[0]**2 + x[1]**2 - 1

# Lagrangian
def L(x, lambda_):
    return f(x) + lambda_ * h(x)

# Gradient and Hessian of Lagrangian
grad_L_x = jit(grad(L, argnums=0))
grad_L_lambda = jit(grad(L, argnums=1))
hess_L_xx = jit(jacfwd(grad_L_x, argnums=0))
hess_L_xlambda = jit(jacfwd(grad_L_x, argnums=1))

# Newton's method
@jit
def newton_step(x, lambda_):
    grad_x = grad_L_x(x, lambda_)
    grad_lambda = grad_L_lambda(x, lambda_)
    hess_xx = hess_L_xx(x, lambda_)
    hess_xlambda = hess_L_xlambda(x, lambda_).reshape(-1)
    
    # Construct the full KKT matrix
    kkt_matrix = jnp.block([
        [hess_xx, hess_xlambda.reshape(-1, 1)],
        [hess_xlambda, jnp.array([[0.0]])]
    ])
    
    # Construct the right-hand side
    rhs = jnp.concatenate([-grad_x, -jnp.array([grad_lambda])])
    
    # Solve the KKT system
    delta = jnp.linalg.solve(kkt_matrix, rhs)
    
    return x + delta[:2], lambda_ + delta[2]

def solve_constrained_optimization(x0, lambda0, max_iter=100, tol=1e-6):
    x, lambda_ = x0, lambda0
    
    for i in range(max_iter):
        x_new, lambda_new = newton_step(x, lambda_)
        if jnp.linalg.norm(jnp.concatenate([x_new - x, jnp.array([lambda_new - lambda_])])) < tol:
            break
        x, lambda_ = x_new, lambda_new
    
    return x, lambda_, i+1

# Analytical solution
def analytical_solution():
    x1 = 2 / jnp.sqrt(5)
    x2 = 1 / jnp.sqrt(5)
    lambda_opt = jnp.sqrt(5) - 1
    return jnp.array([x1, x2]), lambda_opt

# Solve the problem numerically
x0 = jnp.array([0.5, 0.5])
lambda0 = 0.0
x_opt_num, lambda_opt_num, iterations = solve_constrained_optimization(x0, lambda0)

# Compute analytical solution
x_opt_ana, lambda_opt_ana = analytical_solution()

# Verify the result
print("\nNumerical Solution:")
print(f"Constraint violation: {h(x_opt_num):.6f}")
print(f"Objective function value: {f(x_opt_num):.6f}")

print("\nAnalytical Solution:")
print(f"Constraint violation: {h(x_opt_ana):.6f}")
print(f"Objective function value: {f(x_opt_ana):.6f}")

print("\nComparison:")
x_diff = jnp.linalg.norm(x_opt_num - x_opt_ana)
lambda_diff = jnp.abs(lambda_opt_num - lambda_opt_ana)
print(f"Difference in x: {x_diff}")
print(f"Difference in lambda: {lambda_diff}")

# Precision test
rtol = 1e-5  # relative tolerance
atol = 1e-8  # absolute tolerance

x_close = jnp.allclose(x_opt_num, x_opt_ana, rtol=rtol, atol=atol)
lambda_close = jnp.isclose(lambda_opt_num, lambda_opt_ana, rtol=rtol, atol=atol)

print("\nPrecision Test:")
print(f"x values are close: {x_close}")
print(f"lambda values are close: {lambda_close}")

if x_close and lambda_close:
    print("The numerical solution matches the analytical solution within the specified tolerance.")
else:
    print("The numerical solution differs from the analytical solution more than the specified tolerance.")

# Visualize the result
plt.figure(figsize=(12, 10))

# Create a mesh for the contour plot
x1_range = jnp.linspace(-1.5, 2.5, 100)
x2_range = jnp.linspace(-1.5, 2.5, 100)
X1, X2 = jnp.meshgrid(x1_range, x2_range)
Z = jnp.array([[f(jnp.array([x1, x2])) for x1 in x1_range] for x2 in x2_range])

# Plot filled contours
contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.7, extent=[-1.5, 2.5, -1.5, 2.5])
plt.colorbar(contour, label='Objective Function Value')

# Plot the constraint
theta = jnp.linspace(0, 2*jnp.pi, 100)
x1 = jnp.cos(theta)
x2 = jnp.sin(theta)
plt.plot(x1, x2, color='red', linewidth=2, label='Constraint')

# Plot the optimal points (numerical and analytical) and initial point
plt.scatter(x_opt_num[0], x_opt_num[1], color='red', s=100, edgecolor='white', linewidth=2, label='Numerical Optimal Point')
plt.scatter(x_opt_ana[0], x_opt_ana[1], color='blue', s=100, edgecolor='white', linewidth=2, label='Analytical Optimal Point')
plt.scatter(x0[0], x0[1], color='green', s=100, edgecolor='white', linewidth=2, label='Initial Point')

# Add labels and title
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.title('Constrained Optimization: Numerical vs Analytical Solution', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Set the axis limits explicitly
plt.xlim(-1.5, 2.5)
plt.ylim(-1.5, 2.5)

plt.tight_layout()
```


(sec-sqp-approach)=
### L'approche PQS : développement de Taylor et approximation quadratique

La programmation quadratique successive (PQS) aborde la résolution des programmes sous contraintes en résolvant itérativement une suite de sous-problèmes plus simples. Plus précisément, ces sous-problèmes sont des programmes quadratiques qui approchent le problème original autour de l'itéré courant au moyen d'un modèle quadratique de la fonction objectif et d'un modèle linéaire des contraintes. Supposons que l'on ait le problème d'optimisation suivant, avec des contraintes d'égalité :

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{sous les contraintes} \quad & \mathbf{h}(\mathbf{x}) = \mathbf{0}.
\end{aligned}
$$

À l'itération $k$, on cherche un déplacement
$\Delta\mathbf x=\mathbf x-\mathbf x^k$. Un développement de Taylor d'ordre
deux de l'objectif donne

\begin{align*}
f(\mathbf{x}) \approx f(\mathbf{x}^k) + \nabla f(\mathbf{x}^k)^T (\mathbf{x} - \mathbf{x}^k) + \frac{1}{2} (\mathbf{x} - \mathbf{x}^k)^T \nabla^2 f(\mathbf{x}^k) (\mathbf{x} - \mathbf{x}^k).
\end{align*}

La hessienne détermine comment le gradient du modèle varie à mesure que le
déplacement croît. Pour un problème sous contraintes, les équations de Newton
font aussi intervenir la courbure des contraintes. La PQS intègre cette
courbure au moyen de la hessienne du lagrangien,

$$
B_k=\nabla^2_{\mathbf x\mathbf x}
\mathcal L(\mathbf x^k,\boldsymbol\lambda^k)
=\nabla^2 f(\mathbf x^k)+
\sum_i\lambda_i^k\nabla^2 h_i(\mathbf x^k),
$$

et définit l'objectif du pas

$$
m_k(\Delta\mathbf x)=
\nabla f(\mathbf x^k)^\top\Delta\mathbf x
+\frac12\Delta\mathbf x^\top B_k\Delta\mathbf x.
$$

C'est un modèle qui sert à choisir un pas sous contraintes, plutôt que le
développement de Taylor de $f$ seule. On omet la constante $f(\mathbf x^k)$,
car elle n'influe pas sur le déplacement qui minimise.

Chaque égalité est remplacée par son résidu courant augmenté de sa variation
prédite au premier ordre :

$$
\mathbf{h}(\mathbf{x}) \approx \mathbf{h}(\mathbf{x}^k) + J_h(\mathbf{x}^k) (\mathbf{x} - \mathbf{x}^k).
$$

Annuler cette expression affine revient à demander au pas d'annuler le résidu
courant des contraintes. En un point réalisable, la condition devient
$J_h(\mathbf x^k)\Delta\mathbf x=\mathbf0$; le pas se trouve donc dans
l'espace tangent. Le programme quadratique qui en résulte est

$$
\begin{aligned}
\underset{\Delta\mathbf x}{\operatorname{minimiser}}\quad&
\nabla f(\mathbf{x}^k)^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T B_k \Delta \mathbf{x} \\
\text{sous les contraintes} \quad & J_h(\mathbf{x}^k) \Delta \mathbf{x} + \mathbf{h}(\mathbf{x}^k) = \mathbf{0},
\end{aligned}
$$

La résolution du programme quadratique donne à la fois un déplacement
$\Delta\mathbf x^k$ et un vecteur de multiplicateurs
$\boldsymbol\lambda_{\mathrm{QP}}$. Un pas complet de PQS met à jour les
estimations primale et duale par

\begin{align*}
\mathbf{x}^{k+1} = \mathbf{x}^k + \Delta \mathbf{x}^k,
\qquad
\boldsymbol\lambda^{k+1}=\boldsymbol\lambda_{\mathrm{QP}}.
\end{align*}

Les égalités linéarisées peuvent être satisfaites même lorsque
$\mathbf h(\mathbf x^k+\Delta\mathbf x^k)\ne\mathbf0$. Par exemple, un pas
tangent non nul à partir d'un point d'un cercle quitte le cercle. Les
algorithmes pratiques utilisent donc une recherche linéaire ou une région de
confiance pour évaluer les progrès sur le problème original, et peuvent
modifier la hessienne ou relâcher des contraintes linéarisées incompatibles.
Le pseudocode suivant décrit l'itération locale à pas complet, en supposant
que ses programmes quadratiques ont des solutions.

````{prf:algorithm} Programmation quadratique successive (PQS)
:label: alg-sqp

**Entrée :** Estimation initiale $\mathbf{x}^0$, multiplicateurs de Lagrange initiaux $\boldsymbol{\lambda}^0$, tolérance $\epsilon > 0$.

**Sortie :** Solution $\mathbf{x}^*$, multiplicateurs de Lagrange $\boldsymbol{\lambda}^*$.

**Procédure :**

1. **Calculer la solution du programme quadratique :** Résoudre le sous-problème quadratique pour obtenir $\Delta \mathbf{x}^k$. Le solveur de programmes quadratiques fournit aussi les multiplicateurs de Lagrange mis à jour $\boldsymbol{\lambda}^{k+1}$ associés aux contraintes.
2. **Mettre à jour les estimations :** Mettre à jour les variables primales :

   $$
   \mathbf{x}^{k+1} = \mathbf{x}^k + \Delta \mathbf{x}^k.
   $$

   Fixer les variables duales aux valeurs mises à jour $\boldsymbol{\lambda}^{k+1}$ tirées de la solution du programme quadratique.
3. **Répéter jusqu'à convergence :** Poursuivre les itérations jusqu'à ce que $\|\Delta \mathbf{x}^k\| < \epsilon$ et que les conditions KKT soient satisfaites.
````

#### Lien avec la méthode de Newton dans le cas des contraintes d'égalité

Comme l'objectif du programme quadratique est quadratique et ses contraintes
affines, ses conditions de stationnarité et d'admissibilité forment un système
linéaire :

$$
\begin{bmatrix}
B_k & J_h(\mathbf x^k)^\top\\
J_h(\mathbf x^k) & \mathbf0
\end{bmatrix}
\begin{bmatrix}
\Delta\mathbf x^k\\
\boldsymbol\lambda_{\mathrm{QP}}
\end{bmatrix}
=-
\begin{bmatrix}
\nabla f(\mathbf x^k)\\
\mathbf h(\mathbf x^k)
\end{bmatrix}.
$$

Le multiplicateur du programme quadratique est une nouvelle estimation, alors
que la méthode de Newton calcule une variation de cette estimation. Posons
$\Delta\boldsymbol\lambda^k=
\boldsymbol\lambda_{\mathrm{QP}}-\boldsymbol\lambda^k$.
La substitution donne

\begin{align*}
\begin{bmatrix}
B_k & J_h(\mathbf{x}^k)^T \\
J_h(\mathbf{x}^k) & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{x}^k \\
\Delta \boldsymbol{\lambda}^k
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla f(\mathbf{x}^k) + J_h(\mathbf{x}^k)^T \boldsymbol{\lambda}^k \\
\mathbf{h}(\mathbf{x}^k)
\end{bmatrix}
\end{align*}

Avec la hessienne exacte du lagrangien $B_k$, c'est la méthode de Newton
appliquée aux équations KKT originales. Lorsque $J_h$ est de rang ligne plein
et que $B_k$ est définie positive sur les directions non nulles du noyau de
$J_h$, le système linéaire détermine l'unique minimiseur du programme
quadratique et son multiplicateur. Sans courbure convenable, une solution des
équations de stationnarité ne minimise pas nécessairement le programme
quadratique. Remplacer $B_k$ par une approximation définie positive donne une
méthode de Newton modifiée dont les sous-problèmes quadratiques sont convexes.

La structure quadratique et affine fait que toutes les matrices de dérivées
restent fixes pendant la résolution d'un programme quadratique. Les programmes
quadratiques sous contraintes d'égalité, sous les conditions ci-dessus,
n'exigent qu'une résolution linéaire; ceux qui comportent des inégalités
exigent un travail supplémentaire pour imposer l'admissibilité et la
complémentarité. Les factorisations creuses et la réutilisation entre systèmes
apparentés rendent ce travail efficace, comme on l'explique dans la
[section sur la PQS du chapitre sur le contrôle optimal](#sec-sqp-newton).

### PQS pour l'optimisation sous contraintes d'inégalité

Jusqu'ici, nous avons appliqué les idées de la programmation quadratique successive (PQS) à des problèmes ne comportant que des contraintes d'égalité. Étendons maintenant ce cadre aux problèmes d'optimisation qui comportent aussi des contraintes d'inégalité.
Considérons un problème d'optimisation non linéaire général qui comporte à la fois des contraintes d'égalité et d'inégalité :


\begin{align*}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{sous les contraintes} \quad & \mathbf{g}(\mathbf{x}) \leq \mathbf{0}, \\
& \mathbf{h}(\mathbf{x}) = \mathbf{0}.
\end{align*}

Comme précédemment, on approche ce problème en construisant une approximation quadratique de l'objectif et une linéarisation des contraintes. Le sous-problème quadratique à chaque itération se formule alors ainsi :

\begin{align*}
\text{Minimiser} \quad & \nabla f(\mathbf{x}^k)^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T \nabla^2_{\mathbf{x}\mathbf{x}} \mathcal L(\mathbf{x}^k, \boldsymbol{\lambda}^k, \boldsymbol{\mu}^k) \Delta \mathbf{x} \\
\text{sous les contraintes} \quad & J_g(\mathbf{x}^k) \Delta \mathbf{x} + \mathbf{g}(\mathbf{x}^k) \leq \mathbf{0}, \\
& J_h(\mathbf{x}^k) \Delta \mathbf{x} + \mathbf{h}(\mathbf{x}^k) = \mathbf{0},
\end{align*}

où $\Delta \mathbf{x} = \mathbf{x} - \mathbf{x}^k$ représente la direction du
pas pour les variables primales. Comme dans le cas des égalités, le pseudocode
suivant décrit l'itération locale à pas complet et suppose des sous-problèmes
quadratiques résolubles; une implémentation générale exige des règles de choix
du pas et des modifications des sous-problèmes.

````{prf:algorithm} Programmation quadratique successive (PQS) avec contraintes d'inégalité
:label: alg-sqp-ineq

**Entrée :** Estimation initiale $\mathbf{x}^0$, multiplicateurs initiaux $\boldsymbol{\lambda}^0, \boldsymbol{\mu}^0$, tolérance $\epsilon > 0$.

**Sortie :** Solution $\mathbf{x}^*$, multiplicateurs de Lagrange $\boldsymbol{\lambda}^*, \boldsymbol{\mu}^*$.

**Procédure :**

1. **Initialisation :**
   Poser $k = 0$.

2. **Répéter :**

   a. **Construire le sous-problème quadratique :**
   Formuler le sous-problème quadratique à partir de l'itéré courant $\mathbf{x}^k$, $\boldsymbol{\lambda}^k$ et $\boldsymbol{\mu}^k$ :

   $$
   \begin{aligned}
   \text{Minimiser} \quad & \nabla f(\mathbf{x}^k)^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T \nabla^2_{\mathbf{x}\mathbf{x}} \mathcal L(\mathbf{x}^k, \boldsymbol{\lambda}^k, \boldsymbol{\mu}^k) \Delta \mathbf{x} \\
   \text{sous les contraintes} \quad & J_g(\mathbf{x}^k) \Delta \mathbf{x} + \mathbf{g}(\mathbf{x}^k) \leq \mathbf{0}, \\
   & J_h(\mathbf{x}^k) \Delta \mathbf{x} + \mathbf{h}(\mathbf{x}^k) = \mathbf{0}.
   \end{aligned}
   $$

   b. **Résoudre le sous-problème quadratique :**
   Calculer $\Delta \mathbf{x}^k$ et obtenir les multiplicateurs de Lagrange mis à jour $\boldsymbol{\lambda}^{k+1}$ et $\boldsymbol{\mu}^{k+1}$.

   c. **Mettre à jour les estimations :**
   Mettre à jour les variables primales et les multiplicateurs :

   $$
   \mathbf{x}^{k+1} = \mathbf{x}^k + \Delta \mathbf{x}^k.
   $$

   d. **Vérifier la convergence :**
   Si $\|\Delta \mathbf{x}^k\| < \epsilon$ et que les conditions KKT sont satisfaites, arrêter. Sinon, poser $k = k + 1$ et recommencer.

3. **Renvoyer :**
   $\mathbf{x}^* = \mathbf{x}^{k+1}, \boldsymbol{\lambda}^* = \boldsymbol{\lambda}^{k+1}, \boldsymbol{\mu}^* = \boldsymbol{\mu}^{k+1}$.
````

#### Démonstration avec JAX et CVXPy

Considérons le problème suivant, sous contraintes d'égalité et d'inégalité :

\begin{align*}
\min_{x \in \mathbb{R}^2} \quad & f(x) = (x_1 - 2)^2 + (x_2 - 1)^2 \\
\text{sous les contraintes} \quad & g(x) = x_1^2 - x_2 \leq 0  \\
& h(x) = x_1^2 + x_2^2 - 1 = 0
\end{align*}

Cet exemple reprend le précédent, mais ajoute une contrainte d'inégalité en forme de parabole. On exige que la solution se trouve non seulement sur le cercle qui définit la contrainte d'égalité, mais aussi sur la parabole ou au-dessus. Pour résoudre le sous-problème quadratique, on utilise la bibliothèque [CVXPY](https://www.cvxpy.org/). Bien que le lagrangien et ses dérivées puissent facilement se calculer à la main, on utilise [JAX](https://jax.readthedocs.io/) par souci de généralité :

```{code-cell} python
:tags: [hide-input]

#  label: appendix_nlp-cell-04
#  caption: Rendered output from the preceding code cell.

%config InlineBackend.figure_format = 'retina'
import jax
import jax.numpy as jnp
from jax import grad, jit, jacfwd, hessian
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults

# Define the objective function and constraints
@jit
def f(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

@jit
def h(x):
    return jnp.array([x[0]**2 + x[1]**2 - 1])

@jit
def g(x):
    return jnp.array([x[0]**2 - x[1]])  # Corrected inequality constraint: x[1] >= x[0]^2

# Compute gradients and Jacobians using JAX
grad_f = jit(grad(f))
hess_f = jit(hessian(f))
jac_h = jit(jacfwd(h))
jac_g = jit(jacfwd(g))

@jit
def lagrangian(x, lambda_, mu):
    return f(x) + jnp.dot(lambda_, h(x)) + jnp.dot(mu, g(x))

hess_L = jit(hessian(lagrangian, argnums=0))

def solve_qp_subproblem(x, lambda_, mu):
    n = len(x)
    delta_x = cp.Variable(n)
    
    # Convert JAX arrays to numpy for cvxpy
    grad_f_np = np.array(grad_f(x))
    hess_L_np = np.array(hess_L(x, lambda_, mu))
    jac_h_np = np.array(jac_h(x))
    jac_g_np = np.array(jac_g(x))
    h_np = np.array(h(x))
    g_np = np.array(g(x))
    
    obj = cp.Minimize(grad_f_np.T @ delta_x + 0.5 * cp.quad_form(delta_x, hess_L_np))
    
    constraints = [
        jac_h_np @ delta_x + h_np == 0,
        jac_g_np @ delta_x + g_np <= 0
    ]
    
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    return delta_x.value, prob.constraints[0].dual_value, prob.constraints[1].dual_value

def sqp(x0, max_iter=100, tol=1e-6):
    x = x0
    lambda_ = jnp.zeros(1)
    mu = jnp.zeros(1)
    
    for i in range(max_iter):
        delta_x, new_lambda, new_mu = solve_qp_subproblem(x, lambda_, mu)
        
        if jnp.linalg.norm(delta_x) < tol:
            break
        
        x = x + delta_x
        lambda_ = new_lambda
        mu = new_mu
        
    return x, lambda_, mu, i+1

# Initial point
x0 = jnp.array([0.5, 0.5])

# Solve using SQP
x_opt, lambda_opt, mu_opt, iterations = sqp(x0)

print(f"Optimal x: {x_opt}")
print(f"Optimal lambda: {lambda_opt}")
print(f"Optimal mu: {mu_opt}")
print(f"Iterations: {iterations}")

# Visualize the result
plt.figure(figsize=(12, 10))

# Create a mesh for the contour plot
x1_range = jnp.linspace(-1.5, 2.5, 100)
x2_range = jnp.linspace(-1.5, 2.5, 100)
X1, X2 = jnp.meshgrid(x1_range, x2_range)
Z = jnp.array([[f(jnp.array([x1, x2])) for x1 in x1_range] for x2 in x2_range])

# Plot filled contours
contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.7)
plt.colorbar(contour, label='Objective Function Value')

# Plot the equality constraint
theta = jnp.linspace(0, 2*jnp.pi, 100)
x1_eq = jnp.cos(theta)
x2_eq = jnp.sin(theta)
plt.plot(x1_eq, x2_eq, color='red', linewidth=2, label='Equality Constraint')

# Plot the inequality constraint and shade the feasible region
x1_ineq = jnp.linspace(-1.5, 2.5, 100)
x2_ineq = x1_ineq**2
plt.plot(x1_ineq, x2_ineq, color='orange', linewidth=2, label='Inequality Constraint')

# Shade the feasible region for the inequality constraint
x2_lower = jnp.minimum(x2_ineq, 2.5)
plt.fill_between(x1_ineq, x2_lower, 2.5, color='gray', alpha=0.2, hatch='\\/...', label='Inequality-feasible region')

# Plot the optimal and initial points
plt.scatter(x_opt[0], x_opt[1], color='red', s=100, edgecolor='white', linewidth=2, label='Optimal Point')
plt.scatter(x0[0], x0[1], color='green', s=100, edgecolor='white', linewidth=2, label='Initial Point')

# Add labels and title
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.title('SQP for Inequality Constraints with CVXPY and JAX', fontsize=14)
plt.legend(fontsize=10, loc='upper center')
plt.grid(True, linestyle='--', alpha=0.7)

# Set the axis limits explicitly
plt.xlim(-1.5, 2.5)
plt.ylim(-1.5, 2.5)

plt.tight_layout()

# Verify the result
print(f"\nEquality constraint violation: {h(x_opt)[0]:.6f}")
print(f"Inequality constraint violation: {g(x_opt)[0]:.6f}")
print(f"Objective function value: {f(x_opt):.6f}")
```


### L'algorithme d'Arrow-Hurwicz-Uzawa

La représentation min–max primale motive des mises à jour qui descendent dans
les variables primales et montent dans les multiplicateurs. La méthode
d'Arrow-Hurwicz-Uzawa {cite}`arrow1958studies` utilise ces dérivées premières
au lieu des sous-problèmes quadratiques qu'utilise la PQS. Avec la notation de
la discussion précédente sur la dualité,
$\mathcal L(\mathbf x,\boldsymbol\lambda,\boldsymbol\mu)
=f(\mathbf x)+\boldsymbol\mu^\top\mathbf g(\mathbf x)
+\boldsymbol\lambda^\top\mathbf h(\mathbf x)$, où les multiplicateurs
d'égalité ne sont soumis à aucune restriction et où les multiplicateurs
d'inégalité sont positifs ou nuls.

Ces mises à jour ne calculent pas le supremum intérieur de la formulation
primale. Chaque itération fait un pas fini dans les variables des deux
joueurs. La méthode qui en résulte s'appelle la descente-montée de gradient. Sa
convergence exige une analyse distincte, même lorsque le problème possède un
point de selle. Pour des contraintes d'égalité, les mises à jour alternées
prennent la forme suivante :

````{prf:algorithm} Arrow-Hurwicz-Uzawa pour des contraintes d'égalité seulement
:label: ahuz-eq

**Entrée :** Estimation initiale $\mathbf{x}^0$, $\boldsymbol{\lambda}^0$, tailles de pas $\alpha$, $\beta$
**Sortie :** Itérés finaux primal et des multiplicateurs $\mathbf{x}^k$, $\boldsymbol{\lambda}^k$

1: **pour** $k = 0, 1, 2, \ldots$ jusqu'à convergence **faire**

2:     $\mathbf{x}^{k+1} = \mathbf{x}^k - \alpha \nabla_{\mathbf{x}} \mathcal L(\mathbf{x}^k, \boldsymbol{\lambda}^k)$  **(Mise à jour primale)**

3:     $\boldsymbol{\lambda}^{k+1} = \boldsymbol{\lambda}^k + \beta \nabla_{\boldsymbol{\lambda}} \mathcal L(\mathbf{x}^{k+1}, \boldsymbol{\lambda}^k)$  **(Mise à jour duale)**

4: **fin pour**

5: **renvoyer** $\mathbf{x}^k$, $\boldsymbol{\lambda}^k$
````

Pour tenir compte du fait que le multiplicateur de Lagrange doit être positif ou nul pour les contraintes d'inégalité, on peut reprendre l'idée de la descente de gradient projetée pour les contraintes de bornes et envisager une étape de projection, ou d'écrêtage, qui garantit que cette condition reste satisfaite tout au long des itérations. L'algorithme prend alors la forme suivante :

````{prf:algorithm} Arrow-Hurwicz-Uzawa pour des contraintes d'égalité et d'inégalité
:label: ahuz-full

**Entrée :** Estimation initiale $\mathbf{x}^0$, $\boldsymbol{\lambda}^0$, $\boldsymbol{\mu}^0 \geq 0$, tailles de pas $\alpha$, $\beta$, $\gamma$
**Sortie :** Itérés finaux $\mathbf{x}^k$, $\boldsymbol{\lambda}^k$, $\boldsymbol{\mu}^k$

1: **pour** $k = 0, 1, 2, \ldots$ jusqu'à convergence **faire**

2:     $\mathbf{x}^{k+1} = \mathbf{x}^k - \alpha \nabla_{\mathbf{x}} \mathcal L(\mathbf{x}^k, \boldsymbol{\lambda}^k, \boldsymbol{\mu}^k)$  **(Mise à jour primale)**

3:     $\boldsymbol{\lambda}^{k+1} = \boldsymbol{\lambda}^k + \beta \, \nabla_{\boldsymbol{\lambda}} \mathcal L(\mathbf{x}^{k+1}, \boldsymbol{\lambda}^k, \boldsymbol{\mu}^k)$  **(Mise à jour duale pour les contraintes d'égalité)**

4:     $\boldsymbol{\mu}^{k+1} = [\boldsymbol{\mu}^k + \gamma \nabla_{\boldsymbol{\mu}} \mathcal L(\mathbf{x}^{k+1}, \boldsymbol{\lambda}^k, \boldsymbol{\mu}^k)]_+$  **(Mise à jour duale avec écrêtage pour les contraintes d'inégalité)**

5: **fin pour**

6: **renvoyer** $\mathbf{x}^k$, $\boldsymbol{\lambda}^k$, $\boldsymbol{\mu}^k$
````

Ici, $[\cdot]_+$ désigne la projection sur l'orthant positif, ce qui garantit que $\boldsymbol{\mu}$ reste positif ou nul.

Un point de selle n'attire pas nécessairement ces itérés. Pour le lagrangien
bilinéaire $\mathcal L(x,\lambda)=\lambda x$, les mises à jour pour les
égalités seulement donnent $x^{k+1}=x^k-\alpha\lambda^k$ et
$\lambda^{k+1}=\lambda^k+\beta x^{k+1}$. Leur matrice de mise à jour a un
déterminant égal à un; elle ne peut donc pas être contractante dans toutes les
directions. La convexité et l'existence d'un point de selle ne suffisent donc
pas à elles seules à garantir la convergence de cette itération. Des
hypothèses ou des modifications supplémentaires, comme des termes augmentés ou
des pas d'extragradient, sont nécessaires pour obtenir une garantie de
convergence applicable.

```{code-cell} python
:tags: [hide-input]

#  label: appendix_nlp-cell-05
#  caption: Rendered output from the preceding code cell.

%config InlineBackend.figure_format = 'retina'
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults

# Define the objective function and constraints
@jit
def f(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

@jit
def h(x):
    return jnp.array([x[0]**2 + x[1]**2 - 1])

@jit
def g(x):
    return jnp.array([x[0]**2 - x[1]])  # Inequality constraint: x[1] >= x[0]^2

# Define the Lagrangian
@jit
def lagrangian(x, lambda_, mu):
    return f(x) + jnp.dot(lambda_, h(x)) + jnp.dot(mu, g(x))

# Compute gradients of the Lagrangian
grad_L_x = jit(grad(lagrangian, argnums=0))
grad_L_lambda = jit(grad(lagrangian, argnums=1))
grad_L_mu = jit(grad(lagrangian, argnums=2))

# Define the Arrow-Hurwicz-Uzawa update step
@jit
def update(carry, t):
    x, lambda_, mu, opt_state_x, opt_state_lambda, opt_state_mu = carry
    
    # Compute gradients
    grad_x = grad_L_x(x, lambda_, mu)
    grad_lambda = grad_L_lambda(x, lambda_, mu)
    grad_mu = grad_L_mu(x, lambda_, mu)
    
    # Update primal variables (minimization)
    updates_x, opt_state_x = optimizer_x.update(grad_x, opt_state_x)
    x = optax.apply_updates(x, updates_x)
    
    # Update dual variables (maximization)
    updates_lambda, opt_state_lambda = optimizer_lambda.update(grad_lambda, opt_state_lambda)
    lambda_ = optax.apply_updates(lambda_, -updates_lambda)  # Positive update for maximization
    
    updates_mu, opt_state_mu = optimizer_mu.update(grad_mu, opt_state_mu)
    mu = optax.apply_updates(mu, -updates_mu)  # Positive update for maximization
    
    # Project mu onto the non-negative orthant
    mu = jnp.maximum(mu, 0)
    
    return (x, lambda_, mu, opt_state_x, opt_state_lambda, opt_state_mu), x

def arrow_hurwicz_uzawa(x0, lambda0, mu0, max_iter=1000):
    # Initialize optimizers
    global optimizer_x, optimizer_lambda, optimizer_mu
    optimizer_x = optax.adam(learning_rate=0.01)
    optimizer_lambda = optax.adam(learning_rate=0.01)
    optimizer_mu = optax.adam(learning_rate=0.01)
    
    opt_state_x = optimizer_x.init(x0)
    opt_state_lambda = optimizer_lambda.init(lambda0)
    opt_state_mu = optimizer_mu.init(mu0)
    
    init_carry = (x0, lambda0, mu0, opt_state_x, opt_state_lambda, opt_state_mu)
    
    # Use jax.lax.scan for the optimization loop
    (x, lambda_, mu, _, _, _), trajectory = jax.lax.scan(update, init_carry, jnp.arange(max_iter))
    
    return x, lambda_, mu, trajectory

# Initial point
x0 = jnp.array([0.5, 0.5])
lambda0 = jnp.zeros(1)
mu0 = jnp.zeros(1)

# Solve using Arrow-Hurwicz-Uzawa
x_opt, lambda_opt, mu_opt, trajectory = arrow_hurwicz_uzawa(x0, lambda0, mu0, max_iter=1000)

print(f"Final x: {x_opt}")
print(f"Final lambda: {lambda_opt}")
print(f"Final mu: {mu_opt}")

# Visualize the result
plt.figure(figsize=(12, 10))

# Create a mesh for the contour plot
x1_range = jnp.linspace(-1.5, 2.5, 100)
x2_range = jnp.linspace(-1.5, 2.5, 100)
X1, X2 = jnp.meshgrid(x1_range, x2_range)
Z = jnp.array([[f(jnp.array([x1, x2])) for x1 in x1_range] for x2 in x2_range])

# Plot filled contours
contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.7)
plt.colorbar(contour, label='Objective Function Value')

# Plot the equality constraint
theta = jnp.linspace(0, 2*jnp.pi, 100)
x1_eq = jnp.cos(theta)
x2_eq = jnp.sin(theta)
plt.plot(x1_eq, x2_eq, color='red', linewidth=2, label='Equality Constraint')

# Plot the inequality constraint and shade the feasible region
x1_ineq = jnp.linspace(-1.5, 2.5, 100)
x2_ineq = x1_ineq**2
plt.plot(x1_ineq, x2_ineq, color='orange', linewidth=2, label='Inequality Constraint')

# Shade the feasible region for the inequality constraint
x2_lower = jnp.minimum(x2_ineq, 2.5)
plt.fill_between(x1_ineq, x2_lower, 2.5, color='gray', alpha=0.2, hatch='\\/...', label='Inequality-feasible region')

# Plot the optimal and initial points
plt.scatter(x_opt[0], x_opt[1], color='red', s=100, edgecolor='white', linewidth=2, label='Final Point')
plt.scatter(x0[0], x0[1], color='green', s=100, edgecolor='white', linewidth=2, label='Initial Point')

# Plot the optimization trajectory using scatter plot
scatter = plt.scatter(trajectory[:, 0], trajectory[:, 1], c=jnp.arange(len(trajectory)), 
                      cmap='cool', s=10, alpha=0.5)
plt.colorbar(scatter, label='Iteration')

# Add labels and title
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.title('Arrow-Hurwicz-Uzawa Algorithm with JAX and Adam (Corrected Min/Max)', fontsize=14)
plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.grid(True, linestyle='--', alpha=0.7)

# Set the axis limits explicitly
plt.xlim(-1.5, 2.5)
plt.ylim(-1.5, 2.5)

plt.tight_layout()


# Verify the result
print(f"\nEquality constraint violation: {h(x_opt)[0]:.6f}")
print(f"Inequality constraint violation: {g(x_opt)[0]:.6f}")
print(f"Objective function value: {f(x_opt):.6f}")
```


### Descente de gradient projetée

L'algorithme d'Arrow-Hurwicz-Uzawa offrait une façon de traiter les contraintes au moyen de variables duales et d'un schéma de mise à jour primal-dual. Une autre approche couramment utilisée en optimisation sous contraintes est la **descente de gradient projetée**. L'idée est simple : faire un pas de descente de gradient comme si le problème était sans contraintes, puis projeter le résultat sur l'ensemble réalisable. Formellement :

$$
\mathbf{x}_{k+1} = \mathcal{P}_C\big(\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)\big),
$$

où $\mathcal{P}_C$ est la projection sur l'ensemble réalisable $C$, $\alpha$ est la taille du pas et $f(\mathbf{x})$ est la fonction objectif.

La descente de gradient projetée est particulièrement efficace lorsque la projection est peu coûteuse à calculer. Un exemple courant est celui des **contraintes de pavé** (ou contraintes de bornes), où l'ensemble réalisable est un hyperrectangle :

$$
C = \{ \mathbf{x} \mid \mathbf{x}_{\mathrm{lb}} \leq \mathbf{x} \leq \mathbf{x}_{\mathrm{ub}} \}.
$$

Dans ce cas, la projection se réduit à une opération d'écrêtage composante par composante :

$$
[\mathcal{P}_C(\mathbf{x})]_i = \max\big(\min([\mathbf{x}]_i, [\mathbf{x}_{\mathrm{ub}}]_i), [\mathbf{x}_{\mathrm{lb}}]_i\big).
$$

Pour les problèmes sous contraintes de bornes, la descente de gradient projetée est presque aussi facile à implémenter que la descente de gradient usuelle, car l'étape de projection n'est qu'une opération d'écrêtage. Pour des contraintes plus générales, en revanche, la projection peut exiger la résolution d'un problème d'optimisation distinct, qui peut être aussi difficile que la tâche originale. Voici l'algorithme pour un problème de la forme :

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{sous les contraintes} \quad & \mathbf{x}_{\mathrm{lb}} \leq \mathbf{x} \leq \mathbf{x}_{\mathrm{ub}}.
\end{aligned}
$$

```{prf:algorithm} Descente de gradient projetée pour des contraintes de bornes
:label: proj-grad-descent-bound-constraints

**Entrée :** Point initial $\mathbf{x}_0$, taille de pas $\alpha$, bornes $\mathbf{x}_{\mathrm{lb}}, \mathbf{x}_{\mathrm{ub}}$, 
           nombre maximal d'itérations $\max_\text{iter}$, tolérance $\varepsilon$

1. Initialiser $k = 0$
2. Tant que $k < \max_\text{iter}$ et que la convergence n'est pas atteinte :
    1. Calculer le gradient : $\mathbf{g}_k = \nabla f(\mathbf{x}_k)$
    2. Mettre à jour : $\mathbf{x}_{k+1} = \text{clip}(\mathbf{x}_k - \alpha \mathbf{g}_k,\; \mathbf{x}_{\mathrm{lb}}, \mathbf{x}_{\mathrm{ub}})$
    3. Vérifier la convergence : si $\|\mathbf{x}_{k+1} - \mathbf{x}_k\| < \varepsilon$, arrêter
    4. $k = k + 1$
3. Renvoyer $\mathbf{x}_k$
```

La fonction d'écrêtage est définie par :

$$
\text{clip}(x, x_{\mathrm{lb}}, x_{\mathrm{ub}}) = \max\big(\min(x, x_{\mathrm{ub}}), x_{\mathrm{lb}}\big).
$$

Sous des conditions peu restrictives, comme la continuité lipschitzienne du gradient, la descente de gradient projetée converge vers un point stationnaire du problème sous contraintes. Sa simplicité et son faible coût en font un choix courant chaque fois que la projection peut être calculée efficacement.
