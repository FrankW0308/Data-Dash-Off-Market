# Part B data audit

- Training: **150,000 homes**, 22 columns; sale dates **2021-01-01 to 2026-05-31**.
- Off-market test: **3,974 homes**, 16 columns. No target is available; hidden-test accuracy cannot be measured locally.
- IDs are unique in each file, with no train/test ID overlap. Keep IDs solely for output alignment.
- Target median **$856,650**; 1st–99th percentile **$356,700–$4,344,403**; maximum **$30,000,000**. No nonpositive or nonfinite targets. A log-price target and robust percentage-error reporting are appropriate.
- Lot frontage/depth are missing for **58.2%** of training homes and **54.1%** of test homes. Other permitted inputs have no missing values. Preserve missingness indicators; impute only within each training fold if required.
- All house-area strings parse as a number or numeric range; use numeric midpoint, range width/indicator, and log area. Do not discard ranged areas.
- Six property types, 16 styles, five municipalities, and 145 communities occur in training. All four categorical fields have no unseen test values. Still provide unknown-category fallbacks.

## Validation and leakage controls

- Follow the official starter: six expanding-window folds, December 2025 through May 2026. Train on sales strictly before each calendar month; validate on sales within that month. Fit every learned transformation and target statistic only on that training fold.
- Never expose `sold_price`, `list_price`, `date_sold`, `days_on_market`, `external_avm_estimate`, or `external_avm_estimate_date` to the off-market predictor. The target is used only for fitting and evaluation; `date_sold` is used only to split the data.
- The AVM date is after the sale for **70.2%** of training rows. Even apparently earlier AVMs are absent at test time, so exclude both AVM columns entirely.
- The official sale-month test is not a strict as-of-listing simulation: some validation listings predate the model cutoff, as shown below. Do not infer transaction dates or days-on-market for test homes.
- Synthetic coordinates and neighbourhoods do not align with real maps; use the supplied location information without external geocoding.

| Month | Train homes | Validation homes | Listed before cutoff |
|---|---:|---:|---:|
| 2025-12 | 139,931 | 1,238 | 73.4% |
| 2026-01 | 141,169 | 1,058 | 72.6% |
| 2026-02 | 142,227 | 1,442 | 69.8% |
| 2026-03 | 143,669 | 1,816 | 61.0% |
| 2026-04 | 145,485 | 2,224 | 59.8% |
| 2026-05 | 147,709 | 2,291 | 58.7% |

## Feature recommendations

- Structured: type/style, area midpoint and range width, bedrooms, bathrooms, parking, garage, lot dimensions/area/missingness, latitude/longitude, municipality and community. Derived interactions may include bathrooms per bedroom, rooms per square foot, and property type × community.
- Time: listing year/month and months since a fixed origin; annual sine/cosine terms. Do not use test sale dates, which are unavailable. Tree models have limited extrapolation, so compare recent-history training windows or age weights using past-only tuning folds.
- Text: remarks contain strong condition information such as “fully redone”, “updated kitchen”, “original finishes”, “dated”, and “as-is”. Consider deterministic phrase features or a training-only vectorizer. Beware negation (e.g. “not renovated”) and phrases such as “move-in ready” alongside original finishes.
- No currency symbols or direct sale/asking-price phrases were found by the documented regex scan; this is a heuristic screen, not proof that free text has no leakage. Supplied remarks are allowed by the handout.
- If using neighbourhood medians or price-per-square-foot encodings, generate training representations out-of-fold or with earlier-sales-only rolling windows; a full-data target encoding leaks.
- Keep a fresh temporal holdout distinct from tuning, or disclose that the six monthly validation folds were used for model selection. Report only actual executed validation results.

The machine-readable companion includes complete schemas, missingness counts, text scan counts, and date coverage.
