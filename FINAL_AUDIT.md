# Final readiness audit — November 4, 2026

**Current update: 65 code tests pass.** Original Parquet files have now been
supplied and verified. The original-data workflow completed validation,
refitting, prediction, and both official CSV checks. Its validation MdAPEs are
8.53% off-market and 4.50% list-aware. A subsequent nullable-property-type fix
was verified to preserve the feature values used by those predictions.
See [the current summary](feature_validation/summary.json).

The sections below retain the earlier audit and its cleaned-data finding for
traceability. Its original-data availability blocker has been resolved. The
user deferred the report/video work; these code checks do not certify every
submission requirement, 100% price accuracy, or unknown organiser tests.

## Earlier audit

The published source audited was `e4ce094`, checked against the official
[SSU Data Dash handout](https://github.com/erwinzhang7/SSU-Data-Dash/blob/6c5831e12f523d617eccd1418d46a5cf2a0bb8b4/handout.md)
and [rubric](https://github.com/erwinzhang7/SSU-Data-Dash/blob/6c5831e12f523d617eccd1418d46a5cf2a0bb8b4/rubric.md).

## Passed

- Reran all 63 existing starter, model, feature, and historical baseline cases
  with the official dependency pins: **63 passed, zero failures**. `pip check`
  also passed. Tests include the official two-track synthetic end-to-end workflow.
- `features.py` equals `featuresfinal.py`, and `models.py` equals `modelsfinal.py`.
  Both model entry points use the shared feature module. Retain the original
  module filenames or use the complete repository: `modelsfinal.py` imports
  `features.py`, rather than `featuresfinal.py`.
- Factory/fit/predict interfaces, fold-fitted category handling, forbidden-field
  boundaries, row ordering, missingness, model serialization, and refit safety
  passed their tests. The official runner uses sales before each validation month.
- Both user-uploaded submission CSVs pass the structural checker against their
  supplied test IDs: 3,974 off-market and 4,064 list-aware homes, no duplicate or
  missing IDs, and positive finite prices. This is not a price-accuracy test.

## Blocking data finding

Compared the cleaned training CSV with the repository's retained labelled
validation artifact from its documented original-data run. All 10,069 IDs and
monthly fold counts match, but **255 sold-price labels changed**. Each affected
original price, ranging from $3,088,200 to $17,035,700, became **$843,000** in the
cleaned file. Across all 150,000 cleaned training rows, $843,000 appears 3,736 times.
The original Parquet files recorded in the old audit are no longer available in
this workspace, so this comparison uses retained original-run records, not a
fresh authoritative download.

This invalidates presenting the cleaned-label 9.03%/4.63% MdAPEs as results
against unchanged original competition targets. Clipping a training target can
be an intentional modeling choice, but evaluation must use the actual held-out
sale prices and the transformation must be disclosed.

Diagnostic rescoring of the existing predictions against the retained original
validation labels gives:

| Model | MdAPE | MAPE |
|---|---:|---:|
| Combined off-market | 9.1586% | 12.9077% |
| List-aware with Joey's uploaded features | 4.6679% | 7.6899% |

These models were still trained on altered inputs. The table is a diagnostic
rescore, not a new original-data training run or a hidden-test result. Test-input
cleaning also needs review; the list-aware test contains a concentrated group of
median-valued list prices. No prediction files were replaced by this audit.

## Remaining requirements

1. Obtain the original competition `listings.parquet`, `test_off_market.parquet`,
   and `test_list_aware.parquet`, and the cleaning code if cleaned inputs will
   remain part of the method. Fit preprocessing on earlier training data only;
   preserve original validation sale prices.
2. Rerun the official December 2025–May 2026 chronological evaluation, then
   generate both final CSVs from the frozen, documented model and check them.
   Existing predictions trained on the cleaned labels need reassessment.
3. Update the GitHub submission files: it currently contains the historical
   Part B off-market CSV and lacks the list-aware CSV. The user-uploaded pair
   passes format checks, but the list-aware file was not reproduced from the
   earlier supplied code/input combination.
4. Complete the combined report using the official template: at most four pages,
   all team members' full names, both correctly measured validation MdAPEs,
   method, error analysis, limitations, sources, and GitHub/YouTube links. The
   existing two-page Part B PDF is a historical handoff, not the final report.
5. Verify the YouTube video is accessible, unlisted/public, and at most five
   minutes. Its numbers must come from the team's own valid evaluation. Verify
   organiser access to the GitHub repository.
6. Submit through the organiser's form by **November 8, 2026, 11:59 p.m. EST**
   (Toronto time). Submission through that form has not been verified.

The environment currently blocks the documented Hugging Face source under its
active network policy. Uploading the original files or applying the required
network settings would unblock the data rerun. No hidden-test sale prices are
needed or sought; 100% hidden-test accuracy or first place cannot be certified.
