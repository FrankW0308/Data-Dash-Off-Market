# Part B: questions to prepare for

**Who uses this model?** A homeowner or analyst needs a first estimate before
choosing an asking price. The estimate is a starting point for review rather
than a replacement for an inspection or professional appraisal.

**Why exclude list price?** This is specifically the off-market track. Its test
file does not contain list price, and competition rules prohibit using it.

**Why exclude the third-party AVM?** It is absent from the test file and about
70.19% of its training timestamps occur after the sale. It cannot be relied on
as information available to a pre-sale prediction.

**Why log price?** It reduces the influence of exceptionally expensive homes and
focuses learning on proportional differences. The regressors still optimise
their stated log-space loss, not MdAPE directly. Model selection uses MdAPE.

**Why no random train/test split?** A random split would mix later sales into
earlier predictions. Expanding monthly folds imitate forecasting from past
sales to a future sale month.

**Did you tune on the reported six months?** No. The finite candidate set is
ranked on October and November 2025. The selected configuration is frozen
before the December 2025–May 2026 walk-forward check. All candidate comparisons
are explicitly marked as tuning results.

**Is this strictly as of the listing date?** The official evaluation is by sale
month. Some held-out homes were listed before the training cutoff, so the
learned-model backtest is not a per-home listing-time simulation. Comparable
sales are restricted further to sales strictly before each subject was listed.

**Why use remarks?** Descriptions contain observable condition and amenity cues.
We use a small fixed set of interpretable patterns and compare a no-remarks
candidate. Keyword matches can miss negation, inconsistent wording or misleading
descriptions, so they require caution.

**What do the metrics mean?** MdAPE is the median percentage miss, reflecting a
typical home. MAPE is the mean percentage miss, exposing large errors. Neither
is an accuracy percentage or a guarantee for an individual property.

**Can you claim a competition score?** No. June–September 2026 sale prices are
hidden. Our scores describe earlier held-out months only. The submission checker
verifies format and completeness, not predictive accuracy.

**Does a synthetic coordinate correspond to a Toronto street?** No. Neighbourhoods
and coordinates are synthetic. We use internal relative location only and do
not join real-world map, listing or sales data.

**What remains for the team?** Combine Part B's short report section and slides
with A, C and D, add member names and project/video links, respect the four-page
and five-minute limits, and check both final prediction files together.
