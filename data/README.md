# Input data (not bundled in the portable Part B ZIP)

Place the user-supplied `listings.parquet`, `test_off_market.parquet`, and
`test_list_aware.parquet` here for the combined two-track workflow. Raw files are
not included in this repository. CSV versions can be used instead if the
corresponding Parquet file is absent. Set `DATA_DIR` to this folder when using
the root runner; without it, that runner downloads the official files.

The supplied dataset README identifies SSU Data Dash 2026's synthetic Toronto
home-price dataset: 150,000 training sales (January 2021–May 2026) and 3,974
off-market test homes (hidden June–September 2026 sale prices). Competition use
only; no real listing, address or price appears. Do not seek hidden targets.

Dataset source: https://huggingface.co/datasets/Baliyang/SSU_data_dash

Competition rules and code: https://github.com/erwinzhang7/SSU-Data-Dash

Exact input file sizes and SHA-256 checksums used for this analysis are recorded
in `../part_b/analysis/input_manifest.json`. These large inputs are Git-ignored
and excluded from the deliverables ZIP; retain your uploaded originals.
