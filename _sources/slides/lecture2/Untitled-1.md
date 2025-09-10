---
marp: true
size: 16:9
math: mathjax
---

# Understanding the mcph Codebase

The `mcph` codebase is a collection of C++ functions and header files for hydrological management within the Rio Tinto Alcan hydroelectric system in Saguenay-Lac-Saint-Jean.

It focuses on:
- Converting between water levels and volumes.
- Calculating power output.
- Determining maximum water discharge.

Complementary PDF documents provide context on the network, operations, constraints, and hydrology.

---

## Codebase Structure

- **`niv2vol.h`/`niv2vol.cpp`**: Water level-volume conversions.
- **`prod.h`/`prod.cpp`**: Power output calculations for specific plants.
- **`qmax.h`/`qmax.cpp`**: Maximum discharge calculations for specific plants.
- **`mcph.h`**: Declarations for external (FORTRAN) hydrological functions.
- **`main.cpp`**: Example usage of the implemented functions.

---

## Core Functionality

- **Reservoir Management**: Track water storage by converting levels and volumes.
  - `niv2vol(int no_inst, double niv)`
  - `vol2niv(int no_inst, double vol)`
  - `meter2ft(double niv)`
- **Power Production**: Calculate power based on water levels, flow, and available units.
  - `CalculPuissance(...)`
- **Discharge Determination**: Calculate maximum safe water release.
  - `DebitMaximum(...)`

---

## How it Works

The C++ code acts as an interface to external routines (likely FORTRAN, declared in `mcph.h`) that perform complex hydrological and hydraulic modeling.

The system uses hydrological data and plant operational status to:
1. Manage reservoir levels and volumes.
2. Compute power generation potential.
3. Calculate maximum discharge rates.

The workflow in `main.cpp` demonstrates setting parameters and calling these functions to simulate operations.

---

## Mathematical Concepts

The core mathematical concepts are implemented in external routines and described in the documentation:

- **Volume-Level Relationships:** Describe the relationship between reservoir water level and stored volume, specific to each reservoir's geometry.
  $$ \text{Volume} = f(\text{Level}) $$
  (Implemented in external calls like `evlm_`, `evpd_`, etc.)

- **Power Generation:** Power output is proportional to water flow and hydraulic head.
  $$ P \propto Q \times H $$
  Where $P$ is power, $Q$ is flow rate, and $H$ is hydraulic head.
  More complex piecewise linear models are also mentioned[cite: 4]:
  $$p_{n,t} \le s_{n,t+1}\theta_{v,n,1,t}+u_{n,t}\theta_{v,n,2,t}+\theta_{v,n,0,t} \quad \forall v=1,2,...,V_{n,t}$$
  [cite: 4]

---

## Mathematical Concepts (cont.)

- **Hydrological Balance:** Reservoir volume change is governed by inflows, outflows, and other factors.
  $$ S_{t+1} = S_t - U_t + Q_t \quad \forall t=0,1,2,...,T $$
  [cite: 2]
  (More detailed mass balance equation[cite: 17]: $s_{t+1}=s_{t}+\Delta_{t}(q_{t}+Au_{t})$) [cite: 17]
  Where $S_t$ is storage, $U_t$ is total release, and $Q_t$ is natural inflow.

- **Optimization:** Methods used to maximize production while respecting constraints.
  Example objective function for minimization[cite: 13]:
  $$min~z=E[\sum_{t=1}^{T}(i_{t}-e_{t})+\lambda v_{t}^{sup}+\gamma v_{t}^{inf}]$$
  [cite: 13]
  Where $i_t, e_t$ are imports/exports, $\lambda, \gamma$ are penalty coefficients, and $v_t^{sup}, v_t^{inf}$ are storage deviation variables.

---

## Context from Documentation

The PDF documents provide vital context:
- Description of the Rio Tinto Alcan hydroelectric network.
- Hydrological characteristics of river basins.
- Operational constraints (min/max levels and flows) for each power plant and reservoir.
- Procedures for managing normal and exceptional hydrological events[cite: 36].
- Details on the water management plan and system[cite: 441, 500].

---

## In Summary

The `mcph` codebase provides a C++ interface for essential hydrological calculations within the Rio Tinto Alcan system, relying on specialized external routines. The system's operation and the underlying mathematical principles, including volume-level relationships, power generation mechanics, hydrological balance, and optimization, are further detailed in the accompanying documentation, which outlines the constraints and management strategies for safe and efficient hydroelectric operations.