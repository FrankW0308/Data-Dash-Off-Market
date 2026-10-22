# Data Dash 2026 — Part B: Off-Market Prediction

Reproducible off-market home-price modeling: code, model comparison, chronological
validation, fitted model, test predictions, reports, slides and a presentation script.

**[Download the complete Part B package](https://github.com/FrankW0308/Data-Dash-Off-Market/raw/refs/heads/main/Part_B_Off_Market_Complete.zip)**

## Measured results

Histogram gradient boosting was selected using October–November 2025 only.
These separate December 2025–May 2026 expanding monthly validation results
cover 10,069 held-out homes:

| Model | MdAPE | MAPE |
|---|---:|---:|
| Selected histogram gradient boosting | **9.08%** | **11.60%** |
| Community/type rolling 12-month median | 17.89% | 22.25% |
| Ten nearest earlier sales | 20.84% | 27.43% |
| Global median | 38.46% | 46.68% |

Hidden-test accuracy is unknown. The official sale-month protocol is not a strict
per-home listing-time simulation. No list price, post-sale field, external AVM
or identifier enters the learned features. All property/location data is synthetic.

## Deliverables

- [PDF report](part_b/deliverables/part_b_report.pdf) and [editable Word report](part_b/deliverables/part_b_report.docx)
- [Short report insert](part_b/deliverables/report_section.md) for the team's four-page report
- [PowerPoint slides](part_b/deliverables/part_b_slides.pptx) and [75-second script](part_b/deliverables/speaker_script_75_seconds.md)
- [Chinese explanation](part_b/deliverables/chinese_handoff.md) and [Q&A notes](part_b/qa_notes.md)
- [Executed notebook](part_b/off_market_walkthrough.ipynb)
- [3,974 checked off-market predictions](submissions/submission_off_market.csv)
- [Fitted model](part_b/outputs/models/off_market_model.joblib) and [metadata](part_b/outputs/models/model_metadata.json)
- [Tuning and validation comparisons](part_b/deliverables/model_comparison.md)

## Reproduce

Use Python 3.12. From the repository root:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r part_b/requirements.txt -r part_b/report_requirements.txt -r part_b/test_requirements.txt
.venv/bin/python -m pytest part_b/tests -q
```

Place the supplied `listings.parquet` and `test_off_market.parquet` in `data/`.
Raw datasets, the Python environment and intermediate caches are excluded from
Git and the ZIP. See [data instructions](data/README.md) and
[exact input hashes](part_b/analysis/input_manifest.json).

```bash
.venv/bin/python -m part_b all
.venv/bin/python -m part_b.make_deliverables
.venv/bin/python -m part_b check
```

Nine feature/baseline tests passed, and all seven notebook code cells were executed.
Reloading the model reproduced all 3,974 predictions byte-for-byte.
Read the [detailed workflow and integration guide](part_b/README.md) or
[start here](PART_B_START_HERE.md). The team still needs Parts A/C/D, member names,
a combined report, and repository/video links; this repository is not a competition submission.

Official [rules and starter](https://github.com/erwinzhang7/SSU-Data-Dash) were
inspected at commit `6c5831e12f523d617eccd1418d46a5cf2a0bb8b4`.
No outside data is used. The synthetic dataset is for competition use only;
[third-party attribution](part_b/THIRD_PARTY_LICENSE.txt) is retained.
