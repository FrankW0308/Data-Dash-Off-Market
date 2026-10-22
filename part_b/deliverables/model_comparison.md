# Measured model-selection results

October–November 2025 tuning only. These scores select the model; they are not fresh final validation scores.

| Model | Homes | MdAPE | MAPE |
|---|---:|---:|---:|
| Histogram gradient boosting | 4,117 | 9.31% | 11.71% |
| CatBoost depth 7 / RMSE | 4,117 | 9.54% | 12.07% |
| CatBoost depth 7 / MAE | 4,117 | 9.58% | 12.03% |
| CatBoost depth 6 / RMSE | 4,117 | 9.94% | 12.64% |
| CatBoost depth 6 / no remarks | 4,117 | 10.51% | 13.35% |
| Random forest | 4,117 | 13.28% | 16.99% |
| Community/type 12-month median | 4,117 | 19.02% | 23.58% |
| 10 nearest earlier sales | 4,117 | 21.59% | 28.88% |
| Global median | 4,117 | 35.48% | 43.24% |

Selection rule: Lowest pooled Oct-Nov 2025 MdAPE; MAPE breaks ties. Selection frozen before Dec-May validation.

## Frozen configuration

```json
{
  "kind": "hgb",
  "max_iter": 350,
  "learning_rate": 0.07,
  "max_leaf_nodes": 31,
  "l2_regularization": 10.0
}
```

## Untouched final walk-forward backtest

| Model | Homes | MdAPE | MAPE |
|---|---:|---:|---:|
| Histogram gradient boosting | 10,069 | 9.08% | 11.60% |
| Community/type 12-month median | 10,069 | 17.89% | 22.25% |
| 10 nearest earlier sales | 10,069 | 20.84% | 27.43% |
| Global median | 10,069 | 38.46% | 46.68% |
