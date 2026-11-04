# Combined model review — November 4, 2026

The combined implementation passes the official starter's synthetic integration
workflow and the model regression checks below. It is **not yet a validated,
complete competition submission**, and no first-place prediction is supported.

## Official reference

Reviewed [SSU-Data-Dash](https://github.com/erwinzhang7/SSU-Data-Dash/tree/6c5831e12f523d617eccd1418d46a5cf2a0bb8b4)
at commit `6c5831e12f523d617eccd1418d46a5cf2a0bb8b4`, including
[the handout](https://github.com/erwinzhang7/SSU-Data-Dash/blob/6c5831e12f523d617eccd1418d46a5cf2a0bb8b4/handout.md),
[rubric](https://github.com/erwinzhang7/SSU-Data-Dash/blob/6c5831e12f523d617eccd1418d46a5cf2a0bb8b4/rubric.md),
runner, feature functions, submission checker, requirements, and tests.
The uploaded `models (1).py` matched the previously combined root `models.py`.

## Findings and corrections

1. **List-aware refits retained old models.** A type-specific model trained on
   an earlier call survived when the new training set had fewer than 100 homes
   of that type. In a reproducible example, refitting a $600,000 listing from
   a 1.5 sold/list ratio to a 0.5 ratio still predicted $900,000 instead of
   $300,000. Each fit now rebuilds all models from the supplied training set.
   The official runner already creates fresh instances, so this defect affected
   custom instance reuse rather than its default monthly workflow.
2. **The published repository lacked list-aware dependencies and the combined
   runner.** Added the unmodified official `features.py`, `data.py`, `run.py`,
   `check_submission.py`, dependency pins, starter tests, pytest configuration,
   and MIT license. Both tracks can now use the official pipeline when data is
   available. The historical `python -m part_b` pipeline remains separate.
3. **List-aware input/output checks were missing.** Invalid/nonfinite/nonpositive
   list or sold prices now fail clearly; prediction requires fitting; empty
   input returns an empty vector; invalid output prices are rejected. Thread
   limits now apply to both default models. The normal-data ratio model and its
   >=100-home type threshold are preserved.
4. **The 9.08% score does not measure this file.** Historical `part_b` used HGB
   with 350 iterations, learning rate 0.07, 31 leaves, and L2 10, with different
   features/preprocessing. Root `OffMarketModel` uses 500 iterations, rate 0.05,
   63 leaves, minimum leaf size 30, and L2 20. Removed its unsupported claim
   that these settings had documented October–November tuning. Preserved the
   model settings rather than claiming an untested accuracy improvement.

## Code-level requirements

- `make_model('off_market')` and `make_model('list_aware')` return fresh models
  implementing `fit(train)` and `predict(homes)`.
- Off-market predictors exclude list price, target, sale date, days on market,
  third-party AVM fields, and IDs. Extra columns cannot change predictions.
- List-aware features use allowed property/listing inputs and list price; they
  ignore post-sale fields and IDs. Learning sold/list ratios from training
  targets is permitted; hidden test targets are never used.
- Encoders and regressors fit only the provided training rows. **The caller must
  supply earlier sales.** The official runner trains on sales before each
  validation month; the model itself does not impose chronological filtering.
- No outside dataset, real listing/sales record, hidden-target search, remote API,
  or pretrained external price model is used by this implementation.

These are checks against the public rules, not organiser certification.

## Validation executed

Installed the official pinned dependencies in a separate Python 3.12 environment:
NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, PyArrow 22.0.0,
matplotlib 3.10.9, huggingface_hub 2.1.1, and pytest 8.4.2.
`pip check` passed.

- Uploaded file in an isolated copy of the official starter: **18 tests passed**.
- Corrected repository: **49 tests passed** — 18 starter tests, 22 new model
  regression cases, and 9 historical feature/baseline tests.
- The starter end-to-end test executes `validate`, `report`, `predict`, and
  `check` for **both tracks on synthetic fixtures**, including six monthly
  folds and both temporary prediction files. It does not create final entries.
- Model tests cover Joey prediction parity using real official feature
  functions, refit safety, forbidden-field invariance, arbitrary row ordering,
  empty input, fitted-state checks, invalid prices, serialization, and more
  than 255/unseen off-market categories.

Executed from the repository root:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/ssu-audit-matplotlib \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/workspace/.cloud-onboarding/Data-Dash-Off-Market/audit/official-venv/bin/python \
  -m pytest tests part_b/tests -q
```

**No actual-data accuracy, hidden-test accuracy, or rank was measured for this
combined implementation.** Raw datasets are absent. The current environment
network policy does not allow Hugging Face; the earlier source request returned
a proxy 403. A saved draft allowlist addition alone does not apply that access.

## Still required before submission

1. Supply official `listings` (150,000 rows), `test_off_market` (3,974), and
   `test_list_aware` (4,064), as Parquet or CSV. Use originals, not a cleaned file
   that changes the meaning of inputs. All three are needed for the full workflow.
2. Freeze the candidate and run the official expanding December 2025–May 2026
   validation on both tracks. Record this exact source version and dependency
   versions with the results. Do not reuse historical `part_b` scores for it.
   Tune on earlier months; if the official validation months have already
   influenced choices, disclose that and do not call them untouched holdout data.
3. Compare against the relevant baselines. Hidden-test accuracy points require
   MdAPE below the organiser's same-type/neighbourhood 12-month median baseline
   on the hidden homes. Off-market is worth 20 points and list-aware 10; MAPE
   breaks ties. The published validation baseline is not the hidden threshold.
4. Run `predict` and `check`. Both CSVs must have exactly `ml_num,predicted_price`,
   exact test IDs for their own track, no duplicates, and positive finite prices.
   The existing off-market CSV belongs to the historical model; no final combined
   CSVs have been generated, and the list-aware submission is still absent.
5. Complete the combined **maximum four-page PDF** using the official report
   template, every member's full name, both measured validation MdAPEs, failure
   analysis, limitations, and source URLs. Include the accessible GitHub link
   and an unlisted/public **maximum five-minute YouTube video**. Video numbers
   must come from the team's own validation. The Part B documents/ZIP are not
   the complete team submission.
6. Submit through the organiser's form by **November 8, 2026, 11:59 p.m. EST**
   (Toronto time). Uploading code to GitHub does not submit the competition entry.

For competitiveness, prioritise the off-market track, compare a small number of
candidates on earlier chronological folds, and inspect errors by month, property
type, price band, and community. Quantify uncertainty honestly. Accuracy is
30/100 first-round points; method, insight, honesty, communication, and
reproducibility account for 70. Finalists are judged under a separate rubric.
No change should be described as better until real-data validation supports it.
