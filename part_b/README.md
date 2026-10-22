# Part B: Off-Market Prediction

This is Member B's complete reproducible modeling contribution for SSU Data Dash
2026. It values homes without their list price. It includes model comparison,
hyperparameter selection, independent chronological validation, a fitted model,
the off-market submission, report text, figures, and presentation materials.

## Start here

- `deliverables/part_b_report.pdf` and `.docx`: the standalone Part B explanation.
- `deliverables/report_section.md`: concise text to integrate into the team's report.
- `deliverables/speaker_script_75_seconds.md`: approximately 75 seconds of English narration.
- `deliverables/chinese_handoff.md`: explanation for preparation in Chinese.
- `deliverables/part_b_slides.pptx`: two editable presentation slides.
- `outputs/metrics_summary.json`: measured scores, never hidden-test estimates.
- `../submissions/submission_off_market.csv`: exactly one prediction per off-market test home.
- `outputs/models/off_market_model.joblib`: fitted Python model. Only load trusted model files.
- `off_market_walkthrough.ipynb`: notebook walkthrough of the delivered results and code.

The entire team's submission still needs Member C's list-aware predictions,
member names, a combined PDF of at most four pages, and links to the team's GitHub
repository and YouTube video (at most five minutes). This package does not
publish to GitHub, upload a video, or submit to the organisers.

## Reproduce

Run from the repository root (`Data-Dash-Off-Market`). Python 3.12 was used.
The existing checkout is already isolated; a Git worktree is unnecessary.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r part_b/requirements.txt
.venv/bin/python -m pip install -r part_b/report_requirements.txt -r part_b/test_requirements.txt
```

Put the supplied `listings.parquet` and `test_off_market.parquet` in `data/`.
CSV equivalents are supported if Parquet files are absent. Training data is not
included in the portable deliverables ZIP; use the files supplied with the task.
`analysis/input_manifest.json` records the exact Parquet hashes used.

```bash
.venv/bin/python -m pytest part_b/tests -q
.venv/bin/python -m part_b all
.venv/bin/python -m part_b.make_deliverables
.venv/bin/python -m part_b check
```

`all` performs tuning, walk-forward validation, final refitting and prediction.
It needs several minutes on four CPU cores. It preserves successful per-model,
per-month prediction caches, keyed by data, model settings, code and dependency
versions. Changes invalidate the caches. Full retraining and cache reuse are
both explicit in `outputs/training.log` when output is redirected there.

Individual stages are `tune`, `validate`, `fit`, `predict`, and `check`.
`predict` reloads the fitted model without retraining and checks its provenance;
`fit` refits the already selected configuration and rewrites test predictions.
Use `--data-dir PATH`, `--output-dir PATH`, and `--threads 4` after the command
to override the defaults. The final CSV always goes under `submissions/`.

## Method and honest evaluation

The target is `sold_price`. All learned regressors model natural log price and
exponentiate their predictions. This limits domination by high-dollar sales;
it does not directly minimise the competition's median percentage error.

The feature builder uses only information present in `test_off_market`: property
type/style, size, rooms, parking/garage, lot geometry, synthetic location,
listing date, and fixed description keywords. Area ranges become their midpoint
plus range width. Missing lot measurements remain unknown, with missingness
flags. Twenty-two fixed text flags represent condition and amenities. There are
56 features with remarks and 33 without them. No outside dataset is used.

Excluded from predictors: `sold_price`, `list_price`, `date_sold`,
`days_on_market`, `external_avm_estimate`, `external_avm_estimate_date`, and the
identifier `ml_num`. Sale dates are used only to define earlier training sets
and retrospective validation months. All fitted imputation and category
encoding are confined to the training fold. Deterministic, row-local feature
creation can safely be computed once before splitting.

Three baselines are included: past-only global median; same-community/type
rolling-12-month median with documented fallbacks; and the starter's ten nearest
earlier sales. The accelerated nearest-sales implementation matched the official
starter exactly on a 40-home actual-data sample, including tied condo locations.
The rolling median follows the published description; unspecified fallback and
boundary conventions are ours, so it is not a reproduction of the hidden scorer.

The six learned candidates are random forest, histogram gradient boosting,
three CatBoost depth/loss settings, and a no-remarks CatBoost ablation. See
`run.py:CONFIGS` and `outputs/selection.json` for exact settings. Forest/HGB
impute numeric values and encode categories inside each fold. Their ordinal
codes impose an artificial order on some categories; CatBoost handles string
categories natively and numeric missingness without imputation.

Hyperparameters and model family are selected by pooled MdAPE on **October and
November 2025 only**, with MAPE breaking ties. Selection is then frozen. The
selected model and baselines receive the official expanding walk-forward check:
each sale month from **December 2025 through May 2026** uses only sales before
that month. Comparisons among all learned candidates are tuning results;
the separate six-month table reports untouched-period performance for the
selected candidate and baselines. The 10,069 validation homes are pooled once
per model; aggregate metrics are not averages of monthly medians.

This is the official **sale-month** protocol. Some validation homes were listed
before the fold cutoff, so it is not a strictly point-in-time valuation at every
home's listing date. Comparable-sales baselines additionally enforce sales
strictly before each subject's listing date. Hidden June–September test prices
are unavailable: the package makes no claim about achieved hidden-test error.

MdAPE is the median of `100 * abs(prediction - sold_price) / sold_price`;
MAPE is its mean. Lower is better. Home-level bootstrap intervals are descriptive
and understate uncertainty from common market movements and related properties.
Feature importance measures predictive contribution, not causation.

## Team integration

The adapter matches the official starter's model interface:

```python
from part_b.integration import make_off_market_model
model = make_off_market_model(threads=4).fit(past_sales)
predictions = model.predict(off_market_homes)
```

In the team's `models.make_model`, return that adapter for `off_market`; retain
Member C's model for `list_aware`. Keep `part_b/outputs/selection.json` with the
code so the selected configuration is explicit. Refit within each validation
fold; do not reuse the all-data fitted model to score past months.

Use the combined official checker once Part C is ready. The Part B file can be
checked independently with our `check` command or official
`check_submission.check_file(path, test_ids)`.

## Sources

Official case, rubric, starter and submission schema:
<https://github.com/erwinzhang7/SSU-Data-Dash>, inspected at commit
`6c5831e12f523d617eccd1418d46a5cf2a0bb8b4`. Relevant sources are `handout.md`,
`rubric.md`, `data.py`, `models.py`, `run.py`, and `check_submission.py`.
The supplied dataset is synthetic, competition-use-only, and contains no real
listing/address/price records. `THIRD_PARTY_LICENSE.txt` preserves the starter's
MIT attribution. No hidden prices were sought or recovered.
