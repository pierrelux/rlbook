| Wind | Optimal (value iteration) | Certainty-equivalent | Layline sailor | Value-iteration sweeps | Policy-iteration improvements |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Steady wind from the north | 93.5 | 93.5 | 93.5 | 39 | 10 |
| Oscillating breeze (teaching values) | 84.1 | 85.6 | 86.6 | 42 | 6 |
| Shifty breeze (mdp-engine matrix) | 72.4 | 74.3 | 75.2 | 45 | 5 |

Expected time in minutes from the start to the windward mark, 16 rows upwind on a 21 by 21 lake, with the wind from the north at the start. *Certainty-equivalent* plans each leg as if the wind seen now will never change; the *layline sailor* tacks only on the laylines of the mean wind. Value iteration starts from zero and stops when the sup-norm residual is below 1e-09 minutes; policy iteration starts from the layline sailor.
