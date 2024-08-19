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

# Optimal Control Problems

```{epigraph}
The sciences do not try to explain, they hardly even try to interpret, they mainly make models. By a model is meant a mathematical construct which, with the addition of certain verbal interpretations, describes observed phenomena. The justification of such a mathematical construct is solely and precisely that it is expected to work.

-- John von Neumann
```

This course considers two broad categories of environments, each with its own specialized solution methods: deterministic and stochastic environments. Stochastic problems are mathematically more general than their deterministic counterparts. However, despite this generality, it's important to note that algorithms for stochastic problems are not necessarily more powerful than those designed for deterministic ones when used in practice. We should in keep in mind that stochasticity and determinism are assumed properties of the world, which we model—perhaps imperfectly—into our algorithms. In this course, we adopt a pragmatic perspective on that matter, and will make assumptions only to the extent that those assumptions help us design algorithms which are ultimately useful in practice: a lesson which we have certainly learned from the sucess of deep learning methods over the last years. 

With this pragmatic stance, it makes sense to start our journey with perhaps the simplest kind of model for optimal control: one where we model the world deterministically and where we assume that time evolves in a discrete manner. A particular instance of this family of problems is known as a resource allocation problems and can be found across various fields of operations research. To make things more concrete, let's consider the problem of irrigation management as posed by  {cite:t}`Hall1968`, in which a decision maker is tasked with finding the optimal amount of water to allocate throughout the various growth stages of a plant in order to maximize the yield. Clearly, allocating all the water -- thereby flooding the crop -- in the first stage would be a bad idea. Similarly, letting the crop dry out and only watering at the end is also suboptimal. Our solution -- that is a prescription for the amount of water to use at any point in time -- should balance those two extremes. In order to achieve this goal, our system should ideally be informed by the expected growth dynamics of the given crops as a function of the water provided: a process which depends at the very least on physical properties known to influence the soil moisture. The basic model considered in this paper describes the evolution of the moisture constent through the equation: 

\begin{align*}
w_{t+1} = w_t+\texttip{\eta}{efficiency} u_t-e_t + \phi_t
\end{align*}

where $\eta$ is a fixed "efficiency" constant determining the soil moisture response to irrigation, and $e_t$ is a known quantity summarizing the effects of water loss due to evaporation and finally $\phi_t$ represents the expected added moisture due to precipitation. Furthermore, we should ensure that the total amount of water used throughout the season should not exceed the maximum allowed amount.To avoid situations where our crop receives too little or too much water, we further impose the condition that $w_p \leq w_{t+1} \leq w_f$ for all stages for some given values of the so-called permanent wilting percentage $w_p$ and field capacity $w_f$. Depending on the moisture content of the soil, the yield will vary up to a maximum value $Y_max$ depending on the water deficiencies incurred throughout the season. The authors make the assumptions that such deficiencies interact multiplicatively across stages such that the total yield is given by
$\left[\prod_{t=1}^N d_t(w_t)\right] Y_{\max}$. Due to the operating cost of watering operations (fog example, energy consumption of the pumps, human labor, etc), are more meaningful objective is to maximize $\prod_{t=1}^N d_t\left(w_t\right) Y_{\max } - \sum_{t=1}^N c_t\left(u_t\right)$. 
 The problem specification laid out above can be turned into the following mathematical program:

\begin{alignat*}{2}
\text{minimize} \quad & \sum_{t=1}^N c_t(u_t) - a_N Y_{\max} & \\
\text{such that} \quad 
& w_{t+1} = w_t + \eta u_t - e_t + \phi_t, & \quad & t = 1, \dots, N-1, \\
& q_{t+1} = q_t - u_t, & \quad & t = 1, \dots, N-1, \\
& a_{t+1} = d_t(w_t) a_t, & \quad & t = 1, \dots, N-1, \\
& w_p \leq w_{t} \leq w_f, & \quad & t = 1, \dots, N, \\
& 0 \leq u_t \leq q_t, & \quad & t = 1, \dots, N, \\
& 0 \leq q_t, & \quad & t = 1, \dots, N, \\
& a_0 = 1, & & \\
\text{given} \quad & w_1, q_N. &
\end{alignat*}

The multiplicative form of the objective function coming from the yield term has been eliminated through by adding a new variable, $a_t$, representing the product accumulation of water deficiencies since the begining of the season. The three quantities $(w_t, q_t, a_t)$ summarize the entirety of what the decision maker needs to know in order to make good decisions. We say that it is a state variable whose time evolution is specified via so-called dynamics function, which we commonly denote by $f_t$. In the context of optimal control theory, we refer more generally to $u_t$ as a "control" variable while in other communities it is called an "action". Whether the problem is posed as a minimization problem or maximization problem is also a matter of communities, with control theory typically posing problems in terms of cost minimization while OR and RL commmunities usual adopt a reward maximization perspective. In this course, we will alternate between the two equivalent formulations while ensuring that context is sufficiently clear to understand which one is used. 

# Different Flavors of Optimal Control Problems 

The problem stated above, is known as a Discrete-Time Optimal Control Problem (DOCP), which we write more generically as: 

````{prf:definition} Discrete-Time Optimal Control Problem of Bolza Type
:label: bolza-docp
\begin{alignat*}{2}
\text{minimize} \quad & c_T(\mathbf{x}_T) + \sum_{t=1}^T c_t(\mathbf{x}_t, \mathbf{u}_t) & \\
\text{such that} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), & \quad & t = 1, \dots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, & \quad & t = 1, \dots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, & \quad & t = 1, \dots, T, \\
\text{given} \quad & \mathbf{x}_1. &
\end{alignat*}
````
The objective function in a Bolza problem comprises of two terms: the sum of immediate cost or rewards per stage, and a terminal cost function (sometimes called "scrap value"). If the terminal cost function is omited from the objective function, then the resulting DOCP is said to be of Lagrange type. 

````{prf:definition} Discrete-Time Optimal Control Problem of Lagrange Type
:label: lagrange-docp
\begin{alignat*}{2}
\text{minimize} \quad & \sum_{t=1}^T c_t(\mathbf{x}_t, \mathbf{u}_t) & \\
\text{such that} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), & \quad & t = 1, \dots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, & \quad & t = 1, \dots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, & \quad & t = 1, \dots, T, \\
\text{given} \quad & \mathbf{x}_1. &
\end{alignat*}
````
Finally, a Mayer problem is one in which the objective function only contains a terminal cost function without explicitely accounting for immediate costs encountered across stages: 
````{prf:definition} Discrete-Time Optimal Control Problem of Mayer Type
:label: mayer-docp
\begin{alignat*}{2}
\text{minimize} \quad & c_T(\mathbf{x}_T) & \\
\text{such that} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), & \quad & t = 1, \dots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, & \quad & t = 1, \dots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, & \quad & t = 1, \dots, T, \\
\text{given} \quad & \mathbf{x}_1. &
\end{alignat*}
````

# Reduction to Mayer Problems

While it might appear at first glance that Bolza problems are somehow more expressive or powerful, we can show that both Lagrange and Bolza problems can be reduced to a Mayer problem through the idea of "state augmentation". 
The overall idea is that the explicit sum of costs can be eliminated by maininting a running sum of costs as an additional state variable $y_t$. The augmented system equation then becomes: 

\begin{align*}
\tilde{f}_t\left(\tilde{x}_t, u_t\right) \triangleq \left(\begin{array}{c}
f_t\left(x_t, u_t\right) \\
y_t+c_t\left(x_t, u_t\right)
\end{array}\right) 
\end{align*}
where $\mathbf{\tilde{x}}_t \triangleq (\mathbf{x}_t, y_t)$ is the concatenation of the running cost so far and the underlying state of the original system. The terminal cost function is finally computed with 
\begin{align*}
\tilde{c}_T(\mathbf{\tilde{x}_T})  \triangleq c_T\left(x_T\right)+y_T \equiv \tilde{c}_T\left(\tilde{x}_T\right)
\end{align*}

This transformation is often helpful to simplify mathematical derivations (as are about to see shortly) but could also be used to streamline algorithmic implementation (by maintaining one version of the code rather than three with many if/else statements). That being said, there could also be computational advantanges to working with the specific problem types rather than the one size-fits-for-all Mayer reduction. This is true for example with the collocation-type methods where we want to avoid treating $y$ as any state variable.

```{bibliography}
```