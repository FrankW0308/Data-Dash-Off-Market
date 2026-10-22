# Part B — Off-Market Prediction

Histogram gradient boosting achieved 9.08% MdAPE and 11.60% MAPE over 10,069 held-out sales. This is 49.2% lower MdAPE than the rolling community/type baseline (17.89%).

**Model.** Predict log(sold_price), then exponentiate to dollars. Inputs describe area, rooms, parking, type/style, synthetic location, lot dimensions and listing-date calendar features. Area ranges retain midpoint/width; lot missingness is preserved. Fixed remarks flags describe condition, amenities and layout.

Predictors exclude list price, sale-time fields, AVM estimates and IDs. Price supplies labels; sale date defines folds. No external data.

Log-price fitting emphasizes relative error. Categorical processing and any numeric imputation are fitted using only each fold's training rows; feature engineering is deterministic and row-local.

**Selection and evaluation.** Compare baselines, random forest, histogram gradient boosting and CatBoost. Tune only on October–November 2025 MdAPE; MAPE breaks ties. Freeze settings before December 2025–May 2026 validation. Each month, train strictly before month-start and predict that month's sales. This official sale-month protocol is not a strict per-home listing-time simulation.

| Model | Homes | MdAPE | MAPE |
|---|---:|---:|---:|
| Histogram gradient boosting | 10,069 | 9.08% | 11.60% |
| Community/type 12-month median | 10,069 | 17.89% | 22.25% |
| 10 nearest earlier sales | 10,069 | 20.84% | 27.43% |
| Global median | 10,069 | 38.46% | 46.68% |

Rows share the same untouched six-month validation homes. MdAPE is median absolute percentage error relative to sale price; MAPE is its mean. Lower is better.

**Frozen settings:** `max_iter=350; learning_rate=0.07; max_leaf_nodes=31; l2_regularization=10.0`; random seed 42. Full settings/tuning results: `model_comparison.md`.

**Interpretation.** 83.8% of held-out predictions are within 20% of sale price. Refit the selected model on all 150,000 training sales to generate the off-market submission. The CSV contains all 3,974 supplied off-market IDs once.

**Limits.** Synthetic data cannot establish real-world appraisal accuracy; unusual homes and market shifts remain risks. June–September 2026 test prices are hidden: these validation results do not measure hidden-test accuracy.

Source: official handout/starter, https://github.com/erwinzhang7/SSU-Data-Dash, commit `6c5831e12f523d617eccd1418d46a5cf2a0bb8b4`.
