# Bridge-deck maintenance as a discounted MDP

These files support the bridge-deck example in `infinite-horizon-mdps.md`. Regenerate them from the repository root with:

```bash
uv run python scripts/build_bridge_deck_maintenance_artifacts.py
```

## Where the transition matrix comes from

The shipped do-nothing transition counts (`deck_transition_counts.csv`) were computed from **NBI records mirrored on GitHub**, not from the FHWA download site, which was unreachable when this example was built. The mirror is `Deck/Samples/Feature_Matrix_fold_1.mat` in https://github.com/HengLiu2021/MUSTEM-Model-using-NBI, the data of Liu, Nehme and Lu (2023, *Structure and Infrastructure Engineering* 19(6):731–744). It holds the NBI deck condition rating (item 58) of 88,596 decks for each year from 1993 to 2019. Every history in it is non-increasing, so the sample already excludes decks whose rating went up after work, and it contains no rating below 3. Counting the 2,303,496 consecutive-year pairs (26 per deck) gives the 7 × 7 matrix over ratings 9 to 3. `initial_rating_counts.csv` holds the 1993 ratings of the same decks.

The builder chooses its source in this order:

1. **Raw NBI files.** If `data/nbi/` exists, every `.txt` or `.csv` file under it is read as an NBI comma-delimited file with a header row and `'` as the quote character, as distributed at https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm (for example `data/nbi/NE19.txt` or `data/nbi/2019/2019HwyBridgesDelimitedAllStates.txt`). The year comes from a four-digit year in the file or folder name, or else from the two digits before the extension. Required columns: `STATE_CODE_001`, `STRUCTURE_NUMBER_008`, `DECK_COND_058`. Optional: `YEAR_RECONSTRUCTED_106`. A bridge is identified by its state code and structure number. A pair of consecutive years is counted only if both ratings are numeric (not `N`), the rating did not go up, and the reconstruction year did not change. Ratings 0 to 3 are pooled. The counts and initial ratings are rewritten from these files.
2. **The GitHub mirror.** `--mustem-mat PATH/Deck/Samples/Feature_Matrix_fold_1.mat` re-estimates the counts from the mirror (needs `h5py`; the repository is 2.7 GB).
3. **The shipped counts**, otherwise.

`metrics.json` records which source was used under `transition_source`.

## Model

States are deck ratings 9, 8, 7, 6, 5 and "4 or below" (poor, absorbing without work). Actions are do nothing, minor repair, major repair and deck replacement, with costs of 0, 400, 1,200 and 2,650 USD per m² (Saifullah et al., arXiv:2401.12455, Table 7). The yearly risk cost is the failure probability of the deck's rating after the work (their Table 6: 0.001, 0.001, 0.005, 0.005, 0.005, 0.01) times a failure cost of 10 times the replacement cost, with γ = 0.97, both from the same paper. The effects of the repairs are teaching assumptions: a minor repair raises the rating by one but cannot fix a poor deck, a major repair raises it by two up to at most 8, and replacement gives a new deck. The deck then deteriorates for the rest of the year under the do-nothing matrix.

The builder solves the MDP by value iteration and by policy iteration for failure costs of 2, 5, 10, 20 and 40 times the replacement cost. It compares the optimal policy with two baselines, evaluated exactly: reactive replacement (do nothing until poor, then replace) and a minor repair every k years plus replacement when poor, with the best k from 1 to 40. It exits with an error if the two algorithms disagree, if the Bellman residual exceeds 1e-6, or if a baseline beats the optimal policy in any state. It writes `metrics.json`, `results.md`, the figure in `../../_static/bridge_deck_maintenance/`, and the replay data `../../interactive/bridge-deck-maintenance-data.json` (12 decks, 50 years, common random numbers for all policies).
