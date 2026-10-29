| Final drug | Resistant genotype | No steering | Fixed, 1 drug | Feedback, 1 drug | Fixed, 2 drugs | Feedback, 2 drugs | Fixed, 3 drugs | Feedback, 3 drugs |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ampicillin (AMP) | `1111` | 0.62 | 0.25 | **0.11** | 0.00 | 0.00 | 0.00 | 0.00 |
| Amoxicillin (AM) | `1101` | 0.75 | 0.62 | **0.53** | 0.51 | **0.44** | 0.51 | **0.33** |
| Cefaclor (CEC) | `0011` | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cefotaxime (CTX) | `1111` | 0.17 | 0.04 | **0.01** | 0.00 | 0.00 | 0.00 | 0.00 |
| Ceftizoxime (ZOX) | `0111` | 0.86 | 0.81 | **0.73** | 0.75 | **0.65** | 0.75 | **0.57** |
| Cefuroxime (CXM) | `0111` | 0.56 | 0.33 | **0.29** | 0.19 | **0.15** | 0.11 | **0.08** |
| Ceftriaxone (CRO) | `1111` | 0.28 | 0.20 | **0.09** | 0.05 | **0.01** | 0.00 | 0.00 |
| Amoxicillin + clavulanate (AMC) | `1101` | 0.67 | 0.39 | **0.33** | 0.23 | **0.16** | 0.13 | **0.09** |
| Ceftazidime (CAZ) | `0110` | 0.39 | 0.08 | **0.03** | 0.00 | 0.00 | 0.00 | 0.00 |
| Cefotetan (CTT) | `0111` | 0.18 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Ampicillin + sulbactam (SAM) | `1111` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Cefprozil (CPR) | `0101` | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cefpodoxime (CPD) | `1111` | 0.81 | 0.74 | **0.66** | 0.57 | **0.44** | 0.50 | **0.31** |
| Piperacillin + tazobactam (TZP) | `0101` | 0.80 | 0.72 | **0.66** | 0.72 | **0.57** | 0.68 | **0.49** |
| Cefepime (FEP) | `1111` | 0.48 | 0.39 | **0.31** | 0.36 | **0.21** | 0.33 | **0.14** |

Probability that the population ends at the most resistant genotype of the final drug, when every starting genotype is equally likely. *Fixed* is the best drug sequence found by exhaustive search; *feedback* is the backward-induction policy, which observes the genotype before each drug. Bold marks a feedback value lower than the best fixed sequence.
