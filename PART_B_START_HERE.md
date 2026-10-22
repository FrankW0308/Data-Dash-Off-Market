# Your completed Part B package

Start with [the PDF report](part_b/deliverables/part_b_report.pdf) or [the Chinese handoff explanation](part_b/deliverables/chinese_handoff.md).

Selected model: histogram gradient boosting, fitted to log sale price. The earlier October–November 2025 tuning period selected its configuration. The separate December 2025–May 2026 walk-forward backtest produced **9.08% MdAPE and 11.60% MAPE** on **10,069** held-out homes. Community/type rolling-median baseline MdAPE: **17.89%**. These are validation results; hidden-test accuracy is unknown.

## What to hand to your teammates

- `part_b/deliverables/report_section.md`: short Part B insert for the four-page team report.
- `part_b/deliverables/part_b_report.docx`: editable standalone explanation.
- `part_b/deliverables/part_b_slides.pptx`: two slides.
- `part_b/deliverables/speaker_script_75_seconds.md`: rehearse and record your own presentation segment.
- `submissions/submission_off_market.csv`: all 3,974 predictions, verified by the official checker.
- `part_b/deliverables/model_comparison.md`: measured tuning and independent validation tables.
- `part_b/outputs/models/off_market_model.joblib`: fitted model.
- `part_b/off_market_walkthrough.ipynb`: code and results walkthrough.
- `part_b/qa_notes.md`: questions and explanations to prepare for.

## Reproduce or integrate

Read [part_b/README.md](part_b/README.md). It documents dependencies, source data, exact commands, validation design, and the adapter for the team's starter code. Raw data and the Python environment are excluded from this ZIP; use your supplied Parquet files.

The team still needs to combine Parts A/C/D, include member names and repository/video links, respect the four-page/five-minute limits, and submit both tracks. Publishing this repository does not submit the work to the competition or upload a presentation video.
