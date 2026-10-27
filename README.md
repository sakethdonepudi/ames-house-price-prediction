# 🏠 Ames House Price Prediction | Kaggle Regression Lab

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices-20BEFF)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

**Benchmarking six regression algorithms on Ames housing data, investigating outliers and ensemble methods, and producing a reproducible Kaggle submission.**

**Team:** `bhoomshanker` — Saketh Donepudi & Amrutha Kakumani  
**Competition:** [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)  
**Best recorded five-fold OOF log RMSE:** **0.123416** (CatBoost)  
**Public Kaggle score:** **0.12416**  
**Displayed leaderboard rank:** **#1,188** at submission time (dynamic)

> **Result interpretation:** Lower error is better. The OOF result is local cross-validation on training records; the public leaderboard score evaluates a subset of Kaggle test labels. The rank is a historical snapshot, not a live claim.

## Overview

The goal is to predict the **`SalePrice`** of homes in Ames, Iowa from structural, quality, size, neighborhood, and other property characteristics. The public competition supplies **1,460 labeled training rows**, **1,459 test rows**, and **79 raw predictor features**. Training uses **`log1p(SalePrice)`**; predictions are mapped back with **`expm1`**.

The Kaggle metric is **root mean squared logarithmic error**, computed on log-transformed prices (in this competition, `log(predicted)` and `log(actual)`). Our local OOF metric is **RMSE on `log1p` prices**, a closely related but not mathematically identical quantity. This detail matters when comparing small score differences.

## Results

### Original five-fold model benchmark

| Rank | Model | OOF log RMSE ↓ | Fold std. dev. |
|---:|---|---:|---:|
| 1 | **CatBoost** | **0.123416** | 0.017742 |
| 2 | XGBoost | 0.127807 | 0.017740 |
| 3 | LightGBM | 0.130961 | 0.020070 |
| 4 | Random Forest | 0.143398 | 0.018869 |
| 5 | ElasticNet | 0.147986 | 0.042264 |
| 6 | Ridge | 0.151313 | 0.039970 |

The best saved submission uses CatBoost. An equal-weight blend and an exploratory weighted blend did **not** outperform the best saved single model.

### Public Kaggle result

| Item | Value |
|---|---|
| Team | **bhoomshanker** |
| Public leaderboard score | **0.12416** |
| Historical displayed rank | **1,188** |
| Test prediction rows | **1,459** |
| Submission file | [`submissions/submission.csv`](submissions/submission.csv) |

These are **observed**, not promised or independently reverified leaderboard statistics. See the [experiment log](docs/EXPERIMENT_LOG.md) for all follow-up trials and limitations.

## Methodology

1. **Data loading:** Access the public Kaggle competition dataset using KaggleHub after accepting its rules.
2. **Exploration:** Inspect missingness and target distribution; identify numerical and categorical features.
3. **Preprocessing:** Fit median numerical imputation and missing-value indicators, standardize numerical fields, impute categorical missing values, and one-hot encode categoricals **inside each training fold**.
4. **Target:** Fit against `np.log1p(SalePrice)` to manage skew and assess proportional errors.
5. **Cross-validation:** `KFold(n_splits=5, shuffle=True, random_state=42)`. Each validation prediction comes from a model that did not fit on that row.
6. **Model comparison:** Ridge, ElasticNet, Random Forest, LightGBM, XGBoost, and CatBoost.
7. **Prediction:** Average log predictions from the five fold-trained CatBoost models, invert the transform and clip prices positive.
8. **Submission:** Export two columns: `Id` and `SalePrice`.

### Original CatBoost configuration

```python
CatBoostRegressor(
    iterations=1300,
    depth=5,
    learning_rate=0.03,
    l2_leaf_reg=5,
    loss_function="RMSE",
    random_seed=42,
    thread_count=2,
    verbose=False,
    allow_writing_files=False,
)
```

The **reproducible notebook** reconstructs the original fold-safe preprocessing and CatBoost setup. Since library versions and runtime environments can affect fits, rerun it to obtain your own measured score instead of assuming identical decimals.

### Derived property features in the retained pipeline

- `TotalFinishedSF`: Above-ground living area + total basement area.
- `TotalBathrooms`: Full baths + weighted half baths, including basement bathrooms.
- `TotalPorchSF`: Combined open/enclosed/seasonal porch, screen porch, and deck areas.
- `HouseAgeAtSale`: Sale year minus build year, clipped at zero.
- `YearsSinceRemodel`: Sale year minus remodel year, clipped at zero.
- `MSSubClass` and `MoSold` are treated as categorical values in the reproduction pipeline.

## Investigations and negative findings

**Ensembling.** The best single model scored **0.123416**, compared with **0.126442** for the top-four equal blend and **0.124029** for the exploratory cross-fit weighted blend. The weighting procedure does not provide fully nested validation of base models.

**Optuna.** A 12-trial three-fold search selected CatBoost with `iterations=900`, `depth=4`, `learning_rate≈0.06965`, and `l2_leaf_reg≈1.4809`. Five-fold evaluation in a different native-categorical setup produced **0.126849**, which did not beat the recorded original model.

**Outliers.** Two properties (IDs **524** and **1299**) contributed **14.24%** of total squared OOF error in the original diagnosis. Excluding them from **training only** improved a controlled alternative CatBoost setup from **0.126997** to **0.125928**, but remained behind the initial model. Their selection was guided by observed validation errors; no independent confirmation of the gain is claimed.

**Additional features.** Seven further engineered features increased a native-categorical experiment from 79 to 86 predictors but worsened its controlled score from **0.126997** to **0.127932**. This does not imply every added feature is unhelpful.

For reproducibility details, fold-level interpretation, and important caveats, consult **[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)**.

## Repository structure

```text
ames-house-price-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── house_prices_reproducible.ipynb
│   └── experiment_history.ipynb
├── submissions/
│   └── submission.csv
└── docs/
    └── EXPERIMENT_LOG.md
```

`experiment_history.ipynb` contains the original exploratory cell sequence with outputs cleared. The notebooks are not designed to be executed sequentially together: use **`house_prices_reproducible.ipynb`** for the clean baseline and the history notebook as an experiment record.

## Reproduce in Google Colab

1. Visit [the competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) and accept its rules.
2. Open [Google Colab](https://colab.research.google.com/) and upload [`notebooks/house_prices_reproducible.ipynb`](notebooks/house_prices_reproducible.ipynb).
3. Use **Runtime → Run all**. If KaggleHub requires login, the notebook will prompt for `kagglehub.login()`.
4. Confirm the five fold scores and aggregate OOF metric printed by the notebook.
5. Download the generated `submission.csv` and submit it on Kaggle.

For local execution, install dependencies with `pip install -r requirements.txt`, make the Kaggle data available to the notebook, and run it with Jupyter. Exact dependency versions are not locked, so results can vary slightly by environment.

### Submission validation

The committed `submissions/submission.csv` has been checked for:
- **1,459 rows**, with the **`Id,SalePrice`** header.
- Unique IDs, no missing or infinite price values, and positive prices.

## Ethics, data and reproducibility notes

- No private Kaggle credentials, `kaggle.json`, or API tokens belong in the repository.
- **Do not commit downloaded competition train/test CSVs.** Obtain them from the official competition under its applicable terms.
- The competition public score is not a substitute for independent evaluation; private test performance may differ.
- Post-hoc outlier selection and tuning reuse validation information. Treat the follow-up experiments as *exploratory*, not independently confirmed gains.
- GitHub code is provided under the [MIT License](LICENSE); this does **not** change the license/terms of Kaggle's dataset.

## Potential next steps

- Repeated or nested cross-validation to test whether improvements persist.
- Fold-by-fold residual analysis across price ranges and neighborhoods.
- Controlled feature-ablation studies on the original winning pipeline.
- Calibrated and independently validated model combinations if they demonstrate improved generalization.

## Contributors

- **Saketh Donepudi** — [GitHub](https://github.com/sakethdonepudi)
- **Amrutha Kakumani** — Kaggle team member

**Kaggle team:** `bhoomshanker`  
**Competition:** [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
