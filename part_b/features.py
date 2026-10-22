"""Deterministic features available when an off-market home is valued.

This module never estimates anything from other rows. Training targets, sale
dates, list prices, AVM values and identifiers are deliberately not read, even
if a caller passes a complete training table. Fit all learned transformations
inside the training fold; this module itself has no fitting step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS = [
    "house_type_name", "house_style", "municipality", "community", "community_type"
]

NUMERIC_INPUTS = [
    "bedrooms", "washrooms", "parking_total", "garage", "lot_front", "lot_depth",
    "latitude", "longitude",
]
REQUIRED_COLUMNS = [
    "date_listed", "house_type_name", "house_style", "house_area", "municipality",
    "community", *NUMERIC_INPUTS,
]

# Fixed, human-readable descriptions; no vocabulary is learned on validation or
# test rows. These encode property/condition cues, not amounts appearing in text.
TEXT_PATTERNS = {
    "renovated": r"\b(?:renovat\w*|remodel\w*|fully redone|fully updated|newly updated)\b",
    "move_in_ready": r"\b(?:turnkey|move[- ]in[- ]ready|nothing to do|unpack|well[- ]maintained)\b",
    "updated_finishes": r"\b(?:fresh paint|newer flooring|new flooring|new bathroom|updated kitchen|modern kitchen)\b",
    "needs_work": r"\b(?:needs? work|needs? (?:some )?updat\w*|needs? (?:some )?renovat\w*|fixer|handyman|tlc|in need of)\b",
    "original_finishes": r"\b(?:original (?:finish\w*|condition|kitchen)|dated|broadloom|good bones)\b",
    "as_is": r"\b(?:as[- ]is|estate sale|sold as|selling as)\b",
    "luxury": r"\b(?:luxur\w*|prestig\w*|executive|high[- ]end|designer|custom[- ]built)\b",
    "premium_finishes": r"\b(?:marble|granite|quartz|hardwood|stainless|gourmet|spa[- ]like)\b",
    "finished_basement": r"\b(?:finished basement|basement (?:suite|apartment)|lower[- ]level (?:suite|apartment))\b",
    "separate_entrance": r"\b(?:separate entrance|private entrance|side entrance|in[- ]law|nanny suite)\b",
    "outdoor_space": r"\b(?:backyard|back yard|garden|patio|terrace|balcony|deck)\b",
    "views": r"\b(?:views?|overlook\w*|panoramic|unobstructed|skyline)\b",
    "waterfront": r"\b(?:waterfront|lakefront|lake views?|water views?|lake ontario)\b",
    "natural_light": r"\b(?:sun[- ]filled|sunny|bright|natural light|floor[- ]to[- ]ceiling)\b",
    "amenities": r"\b(?:concierge|gym|fitness|pool|sauna|party room|guest suites?)\b",
    "transit": r"\b(?:transit|subway|ttc|go station|go train|lrt)\b",
    "parks_schools": r"\b(?:parks?|trails?|ravine|schools?|library|libraries)\b",
    "storage": r"\b(?:locker|storage|walk[- ]in closet|pantry)\b",
    "parking": r"\b(?:parking|garage|driveway|carport)\b",
    "rental_investment": r"\b(?:invest\w*|rental|rent\w* income|income potential|income property|tenant)\b",
    "open_layout": r"\b(?:open[- ]concept|open[- ]plan|spacious|oversized|generous)\b",
    "quiet_privacy": r"\b(?:quiet|private|privacy|cul[- ]de[- ]sac|peaceful)\b",
}


def parse_house_area(values: pd.Series) -> pd.DataFrame:
    """Return midpoint and range width in square feet; invalid values stay NaN.

    Examples: ``622`` -> (622, 0), ``900-1099`` -> (999.5, 199).
    A range is an interval measurement rather than a precise observed area.
    """
    text = values.astype("string").str.replace(",", "", regex=False).str.strip()
    parts = text.str.extract(r"^(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?$")
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    valid = (low > 0) & (high >= low)
    return pd.DataFrame({
        "square_feet": ((low + high) / 2).where(valid),
        "area_range_width": (high - low).where(valid),
    }, index=values.index)


def make_features(homes: pd.DataFrame, include_text: bool = True) -> pd.DataFrame:
    """Build a fixed-schema feature frame from off-market prediction inputs.

    Categories are strings (including a missing-category sentinel) for CatBoost.
    Other columns are float32; genuine unknown numeric values remain NaN for
    training-fold imputation or a learner with native missing-value handling.
    Additional input columns are ignored. Missing required columns fail loudly.
    """
    required = [*REQUIRED_COLUMNS, *( ["remarks"] if include_text else [])]
    missing = sorted(set(required) - set(homes.columns))
    if missing:
        raise ValueError("Missing required off-market input columns: " + ", ".join(missing))

    X = pd.DataFrame(index=homes.index)
    for column in CATEGORICAL_COLUMNS[:-1]:
        X[column] = homes[column].astype("string").fillna("__MISSING__").astype(str)
    X["community_type"] = X["community"] + " | " + X["house_type_name"]
    for column in NUMERIC_INPUTS:
        X[column] = pd.to_numeric(homes[column], errors="coerce")

    area = parse_house_area(homes["house_area"])
    X["square_feet"] = area["square_feet"]
    X["area_range_width"] = area["area_range_width"]
    X["area_is_range"] = (area["area_range_width"] > 0).astype(float)
    X["area_missing"] = area["square_feet"].isna().astype(float)
    X["log_square_feet"] = np.log1p(area["square_feet"])
    X["lot_front_missing"] = X["lot_front"].isna().astype(float)
    X["lot_depth_missing"] = X["lot_depth"].isna().astype(float)
    X["lot_area"] = X["lot_front"] * X["lot_depth"]
    X["log_lot_area"] = np.log1p(X["lot_area"].where(X["lot_area"] >= 0))
    X["lot_to_house_area"] = X["lot_area"] / X["square_feet"]
    for column in ["bedrooms", "washrooms", "parking_total"]:
        X[column + "_per_1000sqft"] = 1000 * X[column] / X["square_feet"]
    X["bath_per_bedroom"] = X["washrooms"] / X["bedrooms"].where(X["bedrooms"] > 0)
    X["latitude_longitude"] = X["latitude"] * X["longitude"]

    listed = pd.to_datetime(homes["date_listed"], errors="coerce")
    X["listing_months_since_2021"] = (listed - pd.Timestamp("2021-01-01")).dt.total_seconds() / (86400 * 30.4375)
    X["listing_year"] = listed.dt.year
    X["listing_month"] = listed.dt.month
    X["listing_month_sin"] = np.sin(2 * np.pi * listed.dt.month / 12)
    X["listing_month_cos"] = np.cos(2 * np.pi * listed.dt.month / 12)

    if include_text:
        remarks = homes["remarks"].astype("string").fillna("")
        for name, pattern in TEXT_PATTERNS.items():
            X["text_" + name] = remarks.str.contains(pattern, case=False, regex=True, na=False).astype(float)
        X["text_word_count"] = remarks.str.count(r"\b\w+\b").astype(float)

    numeric = [column for column in X if column not in CATEGORICAL_COLUMNS]
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan).astype("float32")
    return X
