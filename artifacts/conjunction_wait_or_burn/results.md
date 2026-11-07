| Policy | Maneuvers | Maneuver cost | Expected collision cost | Total | Missed |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Never maneuver | 0 | 0 | 2668 | 2668 | 62 of 62 |
| Maneuver at the first alarm | 742 | 513 | 89 | 603 | 3 of 62 |
| Decide at the deadline | 157 | 157 | 134 | 291 | 6 of 62 |
| Fitted Q-iteration, relative target | **165** | **138** | **175** | **313** | **20 of 62** |
| Fitted Q-iteration, raw target | 374 | 356 | 85 | 441 | 8 of 62 |
| Hindsight (knows the final estimate) | 88 | 60 | 87 | 147 | 0 of 62 |

Totals over the 11,942 conjunctions with a message at least 2 days before closest approach, in units of a maneuver two days out. The expected collision cost of an event without a maneuver is 1e+05 times the collision probability of its last message. *Missed* counts the events whose last message exceeds the break-even probability of $10^{-5}$ and that received no maneuver. The fitted Q-iteration rows are out of sample: each event is decided by a policy fitted without it (5-fold cross-validation by event), averaged over three random splits.
