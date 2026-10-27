# Experiment log

All values below were reported from the project notebook or subsequent Colab runs. The **Kaggle public score** is the leaderboard score reported after submission, not the score on hidden final test labels. Results from different preprocessing or CV schemes are **not controlled head-to-head comparisons**.

## Initial six-model benchmark (five-fold OOF log RMSE)

| Model | OOF RMSE | Fold standard deviation |
|---|---:|---:|
| CatBoost | **0.123416** | 0.017742 |
| XGBoost | 0.127807 | 0.017740 |
| LightGBM | 0.130961 | 0.020070 |
| RandomForest | 0.143398 | 0.018869 |
| ElasticNet | 0.147986 | 0.042264 |
| Ridge | 0.151313 | 0.039970 |

## Ensemble assessment

- Top-four equal-weight blend: **0.126442** on saved OOF predictions.
- Cross-fitted weight blend: **0.124029**, compared with best original single-model score **0.123416**. Caveat: base OOF models may have training overlap with the meta-validation rows; this is not fully nested stacking validation.
- Fitted blend weights: CatBoost **79.39%**, XGBoost **19.47%**, LightGBM **0.19%**, RandomForest **0.95%**.
- Best original CatBoost retained.

## CatBoost hyperparameter search

- Optuna 12 trials, three-fold tuning: best **0.128151** (not comparable to five-fold baseline).
- Best parameters: iterations 900, depth 4, learning rate 0.0696465652, L2 regularization 1.480894512.
- Re-evaluated with five-fold validation (different preprocessing from initial benchmark): **0.126849**.

## Outlier diagnostics

- IDs **524 and 1299** were large, quality-10 properties in Edwards with unexpectedly low sale prices.
- These two observations accounted for **14.24%** of the initial CatBoost OOF squared error, based on a diagnostic comparison.
- Excluding those two **evaluation** rows yields a *diagnostic-only* RMSE of **0.114373**. This is **not** a valid competition or full-CV score.
- Both observations fell in validation fold 3 under seed 42.
- Controlled native-categorical CatBoost test: full training **0.126997** vs exclusion from training folds only **0.125928**. Full validation retained all 1,460 houses. Selection of these IDs was informed by validation error analysis; treat as exploratory.

## Feature engineering comparison

- 79 raw features vs 86 features (+7 engineered features).
- Same controlled native-categorical configuration: 79-feature baseline **0.126997** vs 86-feature model **0.127932**.
- Adding the features as one group did not improve the controlled score. This does not individually rule out each feature.

## Kaggle public leaderboard

- Team: **bhoomshanker**, with Saketh Donepudi and Amrutha Kakumani.
- **Public score: 0.12416**.
- **Displayed rank: #1,188** at the time it was shared (rank changes over time).
- Kaggle leaderboard performance is distinct from training-set OOF performance and may vary on the private split.

## Selection and limitations

The retained submission corresponds to the saved original CatBoost predictions. Notebook execution, package versions, and validation decisions may produce small variations when rerun. No independent nested-CV confirmation was run for exploratory optimization claims.
