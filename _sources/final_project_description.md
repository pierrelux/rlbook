# Final Project: Real-World Applications of RL and Optimal Control

## Overview

This final project is your opportunity to explore what reinforcement learning and optimal control can achieve beyond Atari and Mujoco. While the Mila community knows these benchmarks well, far fewer can imagine the concrete problems where these methods make a tangible difference.

**Your mission**: Find, formulate, and solve a real-world control problem, then communicate it during a poster exhibition at Mila's Agora.

## Project Objective

Identify and implement a **real-world application** of RL and/or optimal control that is:

1. **Grounded in reality**: Based on an actual system from systems & control literature
2. **Technically sound**: Properly formulated with clear objectives, constraints, and dynamics
3. **Well-implemented**: Demonstrated through clean, documented, verified code
4. **Effectively communicated**: Presented via poster and demonstration

## What Counts as a Real-World Application?

### DO: Draw from Systems & Control Literature

- **CDC** (IEEE Conference on Decision and Control)
- **L4DC** (Learning for Dynamics and Control)
- **IFAC** conferences (World Congress, domain-specific symposia)
- **arXiv cs.SY** (Systems and Control category)

### DON'T: Use Standard ML Benchmarks

- Atari games, Mujoco, PyBullet, OpenAI Gym classic control
- Typical NeurIPS/ICML/ICLR benchmarks

**Why?** The goal is to discover applications most of Mila has never seen.

## Application Domains: Examples

**Energy**: Building HVAC, battery management, data center cooling (DeepMind: 40% reduction), wind farms, smart grids

**Healthcare**: Artificial pancreas, anesthesia control, personalized treatment, drug dosing

**Transportation**: Traffic signals, eco-driving, warehouse robotics, fleet management, ship navigation

**Manufacturing**: Chemical processes, production scheduling, quality control

**Agriculture**: Irrigation, greenhouse climate, water reservoirs, precision farming

**Other**: Adaptive optics, water distribution networks, bioprocesses

See the [Resources](#resources) section concrete GitHub examples across these domains.

## Verification: Critical!

You're working in an unfamiliar domain, possibly using LLM-generated code. **How do you know your results are correct?**

**Required verification approaches** (use multiple):

1. **Sanity checks**: Do signs, magnitudes, and units make sense? (e.g., not 10,000 insulin units or negative energy)
2. **Limiting cases**: Does turning off control make things worse? Does a simplified version match analytical solutions?
3. **Literature comparison**: Do your results align with published benchmarks (quantitatively and qualitatively)?
4. **Physical constraints**: Energy conservation? Mass balance? Valid state bounds?
5. **Sensitivity analysis**: Do parameters change results smoothly?

**What we expect**:
- Document your verification approach in code and poster
- Compare with published results
- Include sanity check tests in your code
- Be honest about limitations and uncertainty

**Red flags**: No literature comparison, results too good to be true (99% improvement), blind trust in code

**Green flags**: Multiple independent checks, thoughtful discussion of uncertainty, validated against benchmarks

**LLMs**: Use them! But you're responsible for correctness. Think of LLMs as junior collaborators requiring careful review.

## Deliverables

1. **Code repository**: Well-documented, reproducible, with verification tests
2. **Poster**: Clear communication of problem, approach, results, insights
3. **5-minute presentation**: During poster session at the Mila Agora on December 15th, from 12h30 to 14h30. Coffee will be served. In person presence is required. 

## Evaluation Criteria

- **Real-world grounding** (25%): Based on actual system? Practical constraints?
- **Problem formulation** (25%): Clear objectives, constraints, dynamics?
- **Technical quality** (25%): Sound implementation, **verification**, methodology choice, literature comparison
- **Creativity** (15%): Underexplored applications, novel combinations, insights
- **Communication** (10%): Poster clarity and presentation effectiveness

## Step Outside Your Comfort Zone

**Important**: If you already work in a specific field (robotics, healthcare, energy, etc.), **choose a different domain**. 

Why? (1) It's unfair to recycle existing research, and (2) it's a missed opportunity to discover something new.

If you work on buildings, try healthcare. If you're in robotics, explore agriculture. **Consider teaming up** with someone from a completely different field. Cross-disciplinary teams produce the most interesting projects.


## Resources

### Example GitHub Projects

**Energy & Sustainability:**
- [Battery MPC](https://github.com/Midren/MPC_for_battery_operation), [Wind Farm Koopman MPC](https://github.com/TUHH-ICS/2023-code-IFAC-Koopman-Model-Predictive-Control-for-Wind-Farm) (IFAC 2023), [Solar Power Tracking](https://github.com/mh-mansouri/Model_Predictive_Controller_for_Maximum_Power_Tracking_Photovoltaic), [Building HVAC (LBL)](https://github.com/lbl-srg/MPCPy)

**Healthcare & Biomedical:**
- [Artificial Pancreas](https://github.com/JuliusFrontzek/ad_meal_prep_control), [Biomechanics](https://github.com/pyomeca/bioptim), [Bioreactors](https://github.com/biosustain)

**Transportation:**
- [Ship Navigation](https://github.com/ppotoc/MPC-Autonomous-Ship-Navigation), [Adaptive Cruise Control](https://github.com/ahmedjjameel/ACC-using-MPC-Simulink), [LEAP-C Benchmarks](https://github.com/leap-c/leap-c)

**Water & Infrastructure:**
- [Water Network Pressure Control](https://github.com/GiacomoGaluppini/Multi-Node-Real-Time-Control-of-Pressure-in-Water-Distribution-Networks-via-Model-Predictive-Control)

**Hybrid & Data-Driven:**
- [TransformerMPC](https://github.com/Vrushabh27/transformermpc), [RL vs MPC Comparison](https://github.com/rozenk30/Quantitative-Comparison-of-RL-and-MPC), [Data-Driven Models](https://github.com/tincandanedo/DATA-DRIVEN_CONTROL-ORIENTED_MODELS), [COFLEX (TU Delft)](https://github.com/TUDelft-DataDrivenControl/COFLEX)

**Tools & Other:**
- [Acados solver + projects](https://docs.acados.org/list_of_projects/index.html), [MPC.jl](https://github.com/NMikusova/mpc.jl), [Adaptive Optics](https://github.com/jinsungkim96/MPC-SensorlessAO), [Nonlinear MPC](https://github.com/AnkushChak89/ApproximateNonlinearMPC)

**Note**: Use these for inspiration to understand problem formulation and verification. Don't just fork and modify.

### Course Examples
Eco-cruise, compressor surge, bioreactor optimization, anesthesia control

### Literature
- Conference proceedings: CDC (IEEE Xplore), L4DC (PMLR), IFAC
- arXiv cs.SY category

## Project Format & Timeline

- **Individual or teams of 2-3 students** (larger scope expected for teams)

**Timeline**:
- **Nov 8 (Fri)**: Topic proposal due. Email blurb, sent to me and Michel (application, literature sources, planned approach)
- **Nov 22 (Fri)**: Progress check-in (brief presentation of preliminary results, verification plan)
- **Dec 15 (Sun)**: Poster session at Mila Agora, final submission (code repository + poster PDF)