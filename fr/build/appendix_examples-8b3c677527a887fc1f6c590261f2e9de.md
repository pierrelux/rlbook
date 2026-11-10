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

# Exemples de problèmes de contrôle optimal en temps continu
### Échangeur de chaleur

![Échangeur de chaleur](_static/heat_exchanger.svg)


On considère un système dans lequel un fluide s'écoule dans un tube; le but est de régler la température du fluide en ajustant au fil du temps la température de la paroi du tube. La température de la paroi, notée $ T_w(t) $, peut varier en fonction du temps, mais elle reste la même sur toute la longueur du tube. En revanche, la température du fluide dans le tube, $ T(z, t) $, dépend à la fois de sa position $ z $ le long du tube et du temps $ t $. Elle évolue selon l'équation aux dérivées partielles suivante :

$$
\frac{\partial T}{\partial t} = -v \frac{\partial T}{\partial z} + \frac{h}{\rho C_p} (T_w(t) - T)
$$

où :
- $ v $ : la vitesse moyenne du fluide qui circule dans le tube,
- $ h $ : la facilité avec laquelle la chaleur passe de la paroi au fluide,
- $ \rho $ et $ C_p $ : la masse volumique et la capacité thermique du fluide.

Cette équation décrit comment la température du fluide change à mesure qu'il avance dans le tube et interagit avec la température de la paroi. Le fluide entre dans le tube avec une température initiale $ T_0 $ à l'entrée (où $ z = 0 $). Notre objectif est d'ajuster la température de la paroi $ T_w(t) $ de sorte qu'à un instant final donné $ t_f $, la température du fluide atteigne une distribution souhaitée $ T_s(z) $ le long du tube. En régime permanent (c'est-à-dire lorsqu'on ne tient plus compte des variations dans le temps), $ T_s(z) $ obéit à la relation :

$$
\frac{d T_s}{d z} = \frac{h}{v \rho C_p}[\theta - T_s]
$$

où $ \theta $ est une température constante que l'on souhaite maintenir à la paroi. L'objectif est de régler la température de la paroi $ T_w(t) $ de sorte qu'à la fin de l'intervalle de temps $ t_f $, la température du fluide $ T(z, t_f) $ soit aussi proche que possible de la distribution souhaitée $ T_s(z) $. On formalise ce but en minimisant la quantité suivante :

$$
I = \int_0^L \left[T(z, t_f) - T_s(z)\right]^2 dz
$$

où $ L $ est la longueur du tube. On exige de plus que la température de la paroi ne dépasse pas une valeur maximale admissible $ T_{\max} $ :

$$
T_w(t) \leq T_{\max}
$$


### Réacteur nucléaire

![Schéma d'un réacteur nucléaire](_static/nuclear_reactor.svg)

Dans un réacteur nucléaire, les neutrons interagissent avec des noyaux fissiles et provoquent la fission nucléaire. Ce processus produit d'autres neutrons ainsi que des noyaux fissiles plus petits appelés précurseurs. Les précurseurs absorbent ensuite d'autres neutrons et engendrent des neutrons « retardés ». L'énergie cinétique de ces produits est convertie en énergie thermique par collisions avec les atomes voisins. La puissance du réacteur est déterminée par la concentration de neutrons disponibles pour la fission nucléaire.

La cinétique de la réaction peut être modélisée par un système d'équations différentielles ordinaires :

\begin{align*}
\dot{x}(t) &= \frac{r(t)x(t) - \alpha x^2(t) - \beta x(t)}{\tau} + \mu y(t), & x(0) &= x_0 \\
\dot{y}(t) &= \frac{\beta x(t)}{\tau} - \mu y(t), & y(0) &= y_0
\end{align*}

où :
- $x(t)$ : concentration de neutrons au temps $t$
- $y(t)$ : concentration de précurseurs au temps $t$
- $t$ : temps
- $r(t) = r[u(t)]$ : degré de variation de la multiplication des neutrons au temps $t$, en fonction du déplacement $u(t)$ de la barre de commande
- $\alpha$ : coefficient de réactivité
- $\beta$ : fraction de neutrons retardés
- $\mu$ : constante de désintégration des précurseurs
- $\tau$ : temps moyen nécessaire à un neutron pour produire un neutron ou un précurseur

On peut ajuster la puissance selon la demande en insérant ou en retirant une barre de commande qui absorbe les neutrons. L'insertion de la barre absorbe des neutrons, ce qui réduit le flux de chaleur et la puissance, tandis que son retrait a l'effet inverse.

L'objectif est de faire passer la concentration de neutrons $x(t)$ d'une valeur initiale $x_0$ à une valeur stable $x_\mathrm{f}$ au temps $t_\mathrm{f}$ tout en minimisant le déplacement de la barre de commande. On peut formuler ce but comme un problème de contrôle optimal, où l'on cherche la fonction de commande $u(t)$ qui minimise la fonctionnelle objectif :

\begin{equation*}
I = \int_0^{t_\mathrm{f}} u^2(t) \, \mathrm{d}t
\end{equation*}

sous les conditions finales :

\begin{align*}
x(t_\mathrm{f}) &= x_\mathrm{f} \\
\dot{x}(t_\mathrm{f}) &= 0
\end{align*}

et la contrainte $|u(t)| \leq u_\mathrm{max}$

### Chimiothérapie

La chimiothérapie utilise des médicaments pour tuer les cellules cancéreuses. Ces médicaments peuvent toutefois avoir aussi des effets toxiques sur les cellules saines de l'organisme. Pour optimiser l'efficacité de la chimiothérapie tout en réduisant ses effets secondaires, on peut formuler un problème de contrôle optimal.

La concentration du médicament $y_1(t)$ et le nombre de cellules immunitaires $y_2(t)$, de cellules saines $y_3(t)$ et de cellules cancéreuses $y_4(t)$ dans un organe, à tout instant $t$ de la chimiothérapie, peuvent être modélisés par un système d'équations différentielles ordinaires :

\begin{align*}
\dot{y}_1(t) &= u(t) - \gamma_6 y_1(t) \\
\dot{y}_2(t) &= \dot{y}_{2,\text{entrée}} + r_2 \frac{y_2(t) y_4(t)}{\beta_2 + y_4(t)} - \gamma_3 y_2(t) y_4(t) - \gamma_4 y_2(t) - \alpha_2 y_2(t) \left(1 - e^{-y_1(t) \lambda_2}\right) \\
\dot{y}_3(t) &= r_3 y_3(t) \left(1 - \beta_3 y_3(t)\right) - \gamma_5 y_3(t) y_4(t) - \alpha_3 y_3(t) \left(1 - e^{-y_1(t) \lambda_3}\right) \\
\dot{y}_4(t) &= r_1 y_4(t) \left(1 - \beta_1 y_4(t)\right) - \gamma_1 y_3(t) y_4(t) - \gamma_2 y_2(t) y_4(t) - \alpha_1 y_4(t) \left(1 - e^{-y_1(t) \lambda_1}\right)
\end{align*}

où :
- $y_1(t)$ : concentration du médicament dans l'organe au temps $t$
- $y_2(t)$ : nombre de cellules immunitaires dans l'organe au temps $t$
- $y_3(t)$ : nombre de cellules saines dans l'organe au temps $t$
- $y_4(t)$ : nombre de cellules cancéreuses dans l'organe au temps $t$
- $\dot{y}_{2,\text{entrée}}$ : taux constant de cellules immunitaires qui entrent dans l'organe pour combattre les cellules cancéreuses
- $u(t)$ : taux d'injection du médicament dans l'organe au temps $t$
- $r_i, \beta_i$ : constantes des termes de croissance
- $\alpha_i, \lambda_i$ : constantes des termes de décroissance dus à l'action du médicament
- $\gamma_i$ : constantes des autres termes de décroissance

L'objectif est de minimiser le nombre de cellules cancéreuses $y_4(t)$ à un instant donné $t_\mathrm{f}$ tout en utilisant le moins de médicament possible afin de réduire ses effets toxiques. On peut formuler ce but comme un problème de contrôle optimal, où l'on cherche la fonction de commande $u(t)$ qui minimise la fonctionnelle objectif :

\begin{equation*}
I = y_4(t_\mathrm{f}) + \int_0^{t_\mathrm{f}} u(t) \, \mathrm{d}t
\end{equation*}

sous les contraintes de la dynamique du système, des conditions initiales et de $u(t) \geq 0$.

D'autres contraintes peuvent s'ajouter :
- Maintenir un nombre minimal de cellules saines pendant le traitement :
  \begin{equation*}
  y_3(t) \geq y_{3,\min}
  \end{equation*}
- Imposer une limite supérieure à la dose de médicament :
  \begin{equation*}
  u(t) \leq u_{\max}
  \end{equation*}

### Corruption gouvernementale

Dans ce modèle de Feichtinger et Wirl (1994), on cherche à comprendre ce qui incite les politiciens à se livrer à la corruption ou à la combattre. Le modèle traite la popularité d'un politicien comme un processus dynamique influencé par la mémoire que garde le public de la corruption récente et passée. L'objectif est de trouver les conditions dans lesquelles des politiciens intéressés choisiraient d'être honnêtes ou malhonnêtes.

Le modèle introduit la notation suivante :

- $C(t)$ : conscience (connaissance) accumulée de la corruption passée au temps $t$
- $u(t)$ : ampleur de la corruption (variable de commande du politicien) au temps $t$
- $\delta$ : taux d'oubli de la corruption passée
- $P(t)$ : popularité du politicien au temps $t$
- $g(P)$ : fonction de croissance de la popularité; $g''(P) < 0$
- $f(C)$ : fonction mesurant la perte de popularité causée par $C$; $f'(C) > 0$, $f''(C) \geq 0$
- $U_1(P)$ : avantages associés à la popularité; $U_1'(P) > 0$, $U_1''(P) \leq 0$
- $U_2(u)$ : avantages tirés des pots-de-vin et de la fraude; $U_2'(u) > 0$, $U_2''(u) < 0$
- $r$ : taux d'actualisation

La dynamique de la mémoire que garde le public de la corruption récente et passée, $C(t)$, est modélisée par :

\begin{align*}
\dot{C}(t) &= u(t) - \delta C(t), \quad C(0) = C_0
\end{align*}

L'évolution de la popularité du politicien $P(t)$ est régie par :

\begin{align*}
\dot{P}(t) &= g(P(t)) - f(C(t)), \quad P(0) = P_0
\end{align*}

Le politicien cherche à maximiser l'objectif suivant :

\begin{equation*}
\int_0^{\infty} e^{-rt} [U_1(P(t)) + U_2(u(t))] \, \mathrm{d}t
\end{equation*}

sous les contraintes de la dynamique de la conscience de la corruption et de la popularité.

Le problème de contrôle optimal se formule ainsi :

\begin{align*}
\max_{u(\cdot)} \quad & \int_0^{\infty} e^{-rt} [U_1(P(t)) + U_2(u(t))] \, \mathrm{d}t \\
\text{s.c.} \quad & \dot{C}(t) = u(t) - \delta C(t), \quad C(0) = C_0 \\
& \dot{P}(t) = g(P(t)) - f(C(t)), \quad P(0) = P_0
\end{align*}

Les variables d'état sont la conscience accumulée de la corruption passée $C(t)$ et la popularité du politicien $P(t)$. La variable de commande est l'ampleur de la corruption $u(t)$. La fonctionnelle objectif représente le flux actualisé des avantages tirés de l'honnêteté (la popularité) et de la malhonnêteté (la corruption).
