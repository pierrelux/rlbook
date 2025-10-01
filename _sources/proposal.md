# Physics-Informed Reinforcement Learning for Closed-Loop Anesthesia Control

Reinforcement learning (RL) provides a principled framework for sequential decision making under uncertainty (Sutton and Barto 2018). Despite its success in simulated environments, RL faces important limitations that prevent deployment in safety-critical applications: training is data-hungry, policies can violate safety constraints, and adaptation to heterogeneous systems is slow. A growing body of work suggests that embedding physics-based models into RL algorithms can mitigate these challenges by narrowing the search space and improving interpretability (Willard et al. 2020).

My future supervisor's group has demonstrated this approach in building climate control, where physics-based models of HVAC systems are integrated into RL algorithms to achieve efficient, safe control (Taboga et al. 2024). The same principle can be extended to clinical settings, where pharmacokinetic and pharmacodynamic (PK/PD) models describe how drugs are absorbed, distributed, and exert their effects. The proposed project will focus on anesthesia as a motivating case, with the broader goal of advancing RL methodology for safety-critical control.

## Objectives and Hypothesis

The research centers on five methodological questions in RL: (1) How to construct hybrid models that integrate physics-based dynamics with learned residual functions that capture model misspecification and patient-specific variation? (2) How to incorporate explicit safety constraints into RL algorithms? (3) How to improve data efficiency by exploiting physics priors and simulators? (4) How to enable rapid adaptation to new environments, framed here as new patients? (5) How to design policies that are interpretable enough to support human oversight?

Our primary hypothesis is that embedding well-established PK/PD models into RL algorithms will create hybrid representations where physics-based equations capture the bulk of drug dynamics, while learned residual functions capture model misspecification and patient-specific variation, leading to more data-efficient, safe, and interpretable control policies.

## Experimental Approach and Methods

Anesthesia provides a testbed where these RL questions can be studied concretely. Current practice involves either manual dosing by anesthesiologists or target-controlled infusion (Shafer 1993), where infusion rates are chosen to achieve a target drug concentration predicted by population-average PK/PD models. While clinically useful, these approaches are open-loop and do not adapt to measured patient responses during surgery. Classical controllers (PID, adaptive control) have been trialed in this setting (Absalom and Kenny 2003), but they are limited in flexibility.

The project will proceed in four phases. **Phase 1** will establish standard propofol PK/PD baselines (Schnider/Eleveld models) and build a diverse library of simulated patients using open simulators (PAS, AReS) that encode multi-drug physiology and provide BIS/MAP proxies. **Phase 2** will develop hybrid models by augmenting compartment PK/PD with effect-site dynamics using small residual functions (MLPs constrained by regularization and monotonicity) that map effect-site concentration and covariates to depth proxies.

**Phase 3** represents the main research angle: a model-based RL controller that plans over the hybrid dynamics, with a safety layer encoding drug limits, maximum rate-of-change, and physiological bounds. We will compare to pure model-free baselines and classical PID/adaptive controllers, re-implementing prior deep RL approaches (Schamberg et al. 2020) for apples-to-apples evaluation. **Phase 4** will implement an offline-replay pipeline to compare policy dosing against recorded intraoperative logs, with fallback to strict in-silico evaluation and meta-learning across patient libraries if clinical data access is limited.

## Significance

As a machine learning researcher with experience in clinical applications (Daccache et al. 2025; Harutyunyan et al. 2024), I have witnessed firsthand the gap between AI's theoretical potential and its real-world deployment in safety-critical domains. While RL has achieved remarkable success in simulated environments, its translation to clinical practice remains limited by fundamental challenges: poor data efficiency, difficulty transferring across heterogeneous patients, and lack of interpretability for human oversight.

This research addresses these core limitations by demonstrating how physics priors can fundamentally transform RL's applicability to safety-critical control. The work will produce the first framework for physics-informed RL in anesthesia, showing how mechanistic models can dramatically improve data efficiency and enable rapid adaptation across patient populations. Beyond anesthesia, this methodology will advance the state-of-the-art in RL for safety-critical applications more broadly, from autonomous vehicles to industrial control systems.

The societal impact is profound: automated anesthesia systems could improve patient outcomes, reduce healthcare costs, and address the global shortage of anesthesiologists. More fundamentally, this research represents an significant step toward trustworthy AI systems that can operate safely in the real world, where the stakes are high and mistakes are unacceptable. By bridging the gap between AI theory and clinical practice, this work will demonstrate how machine learning can truly make a difference in people's lives.

---

# Bibliography and Citations

Absalom, A. R., & Kenny, G. N. C. (2003). Closed-loop control of anesthesia. *British Journal of Anaesthesia*, 90(1), 115–120.

Daccache, N., Wu, Y., Jeffries, S. D., Zako, J., Harutyunyan, R., Pelletier, E. D., Laferrière-Langlois, P., & Hemmerling, T. M. (2025). Safety and recovery profile of patients after inhalational anaesthesia versus target-controlled or manual total intravenous anaesthesia: a systematic review and meta-analysis of randomised controlled trials. *British Journal of Anaesthesia*.

Harutyunyan, R., Gilardino, M. S., Wu, Y., & Hemmerling, T. M. (2024). Description of a novel web-based liposuction system to estimate fat volume and distribution. *Aesthetic Surgery Journal*.

Aubouin-Pairault, B., et al. (2023). PAS: a Python Anesthesia Simulator for drug control. *Journal of Open Source Software*, 8(88), 5480.

Merlo, M., et al. (2024). AReS: A patient simulator to facilitate testing of automated anesthesia. *PubMed*.

Taboga, V., Gehring, C., Le Cam, M., Dagdougui, H., & Bacon, P.-L. (2024). Neural differential equations for temperature control in buildings under demand response programs. *Applied Energy*, 368.

Liu, Y., Wu, X., & Luo, H. (2022). Reinforcement learning for closed-loop propofol anesthesia control. *IEEE Transactions on Biomedical Engineering*, 69(11), 3302–3313.

Schamberg, G., Badgeley, M., & Brown, E. N. (2020). Controlling the level of unconsciousness by titrating propofol with deep reinforcement learning. *arXiv preprint arXiv:2008.12333*.

Schnider, T. W., Minto, C. F., Shafer, S. L., et al. (1998). The influence of age on propofol pharmacodynamics. *Anesthesiology*, 88(1), 26–42.

Shafer, S. L. (1993). Advances in propofol pharmacokinetics and pharmacodynamics. *Journal of Clinical Anesthesia*, 5(6 Suppl), 14S–21S.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

Willard, J., Jia, X., Xu, S., Steinbach, M., & Kumar, V. (2020). Integrating physics-based modeling with machine learning: A survey. *Computing in Science & Engineering*, 22(6), 39–53.