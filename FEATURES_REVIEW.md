# Combined features and competition requirements

Reviewed against the [official SSU Data Dash repository](https://github.com/erwinzhang7/SSU-Data-Dash/tree/6c5831e12f523d617eccd1418d46a5cf2a0bb8b4),
including its handout, rubric, runner, and tests. The upstream reference was
refreshed and remains commit `6c5831e12f523d617eccd1418d46a5cf2a0bb8b4`.

| Function | Purpose | Output |
|---|---|---|
| `make_off_market_features` | Default `OffMarketModel` in both model files | 55 columns: 49 numeric and 6 categorical |
| `off_market_features` | Numeric starter `FeatureModel` compatibility | 55 numeric columns, including 6 fixed property-type flags |
| `list_aware_features` | Joey's sold-to-list model | His original 17 columns, including all four list-price signals |

The list-aware branch uses its own original base features. Adding richer
off-market features therefore does not silently add inputs to Joey's model.
Both `models.py` and `modelsfinal.py` import the shared off-market builder;
the feature file never imports either model file.

## Feature coverage

Off-market features cover area midpoints, missing-area flags, rooms, parking,
garage, valid lot dimensions and ratios, property type/style, municipality and
community interactions, synthetic coordinates, listing-date trend/seasonality,
and fixed description cues for condition and amenities. Structurally unknown
condo lot measurements remain unknown even when a cleaned CSV filled them.

Joey's list-aware features retain his square-footage, rooms, parking, location,
listing-time and property-type inputs, plus list price, list price per square
foot, Detached/list-price interaction, and Semi-Detached/list-price interaction.
CSV date strings are accepted in addition to already parsed dates.

## Rules enforced by the implementation

- Off-market never reads `list_price`, `sold_price`, `date_sold`,
  `days_on_market`, AVM values/dates, or `ml_num` as predictors.
- List-aware may use list price; it never reads sale targets, sale dates,
  days on market, AVM fields, or identifiers as predictors.
- Transformations use only the individual row. There are no batch-derived
  categories, target averages, whole-dataset imputers, or fitted vocabularies.
- Category encoders are fitted by `OffMarketModel` inside the supplied training
  fold. Unknown categories and numeric missingness are handled by that pipeline.
- The official runner supplies only sales before each validation month. Feature
  functions do not replace that chronological splitting requirement.
- No external dataset, real map/listing/sale records, network service, or hidden
  target recovery is used. Property locations remain synthetic.
- Both feature APIs preserve row index/order and have fixed column schemas;
  the numeric API can be passed to the starter's `FeatureModel`.

Recent comparable sales, target-encoded community prices, or market aggregates
are optional modeling ideas in the handout, not mandatory features. Adding them
would require state fitted only to earlier training sales; they are deliberately
absent from these stateless feature functions.

## Verification and performance limits

Exact frame equality was checked against the pre-refactor rich off-market
builder on all 150,000 supplied training rows and 3,974 off-market test rows.
Joey's list-aware feature values, column order, and dtypes were checked against
his uploaded `features joey.py` on all training rows and 4,064 list-aware test
rows. Additional tests exercise leakage boundaries, missingness, fixed schemas,
row-local behavior, parsing, model wiring, and end-to-end runner compatibility.

Joey's three extra price-derived columns change the list-aware feature set from
the previous official baseline. Its earlier 4.69% validation MdAPE must not be
reused for this new version. Fresh list-aware validation used the official
December 2025–May 2026 monthly folds on 10,069 homes: **4.63% MdAPE, 6.30% MAPE**.
The off-market feature tables and model settings are unchanged; its preceding
matching-model validation measured **9.03% MdAPE, 12.39% MAPE** on those months.
[The validation summary](feature_validation/summary.json) records the source
fingerprints, equivalence checks, and scope of each result. Detailed new
list-aware outputs are under `/workspace/deliverables/features_review`.
Hidden-test accuracy remains unknown.
The supplied cleaned CSVs were used as received; their original cleaning process
has not been independently audited.

This completes the feature code portion of the requirements. The submission
still needs predictions generated from the final frozen code, both CSVs checked
against their test IDs, measured results and error analysis, the combined
maximum-four-page report with all team names, a maximum-five-minute video, and
the organiser's submission form. Existing prediction files were preserved during
this feature task.
