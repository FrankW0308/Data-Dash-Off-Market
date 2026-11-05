"""Combined features for the two SSU Data Dash competition tracks.

make_off_market_features returns the rich mixed numeric/categorical schema used
by OffMarketModel. The model fits category encoders on its training fold only.
off_market_features is a numeric compatibility view for the starter FeatureModel.
list_aware_features preserves Joey's supplied property/list-price interactions
and its original base independently of the richer off-market features.

All transformations are row-local. No labels, IDs, sale dates, list prices (for
off-market), AVM fields, learned target aggregates, outside data, or future rows
are read. Unknown numeric measurements remain NaN for the training-fold learner.
Property/location data is synthetic: no real map or sales data is joined.

Official interface and rules: https://github.com/erwinzhang7/SSU-Data-Dash
Starter-derived code: Copyright (c) 2026 Erzheng Zhang; MIT license in LICENSE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]


CATEGORICAL_COLUMNS = [
    "house_type_name", "house_style", "municipality", "community", "community_type",
    "municipality_community",
]

# In the supplied raw competition data these types have structurally unmeasured
# lot dimensions. The uploaded cleaned CSV instead fills those with medians.
# Masking them consistently prevents invented condo land from changing inference.
NO_LOT_TYPES = {"condo apt", "co-op apt", "condo townhouse"}

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

    text = values.astype("string").str.normalize("NFKC").str.strip()
    text = text.str.replace(r"[\u2010-\u2015\u2212]", "-", regex=True)
    text = text.str.replace(",", "", regex=False).str.replace(
        r"(?i)\s*(?:sq\.?\s*ft\.?|sqft|square\s+feet)\s*$", "", regex=True).str.strip()
    parts = text.str.extract(r"^(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?$")
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    valid = (low > 0) & (high >= low) & np.isfinite(low) & np.isfinite(high)
    return pd.DataFrame({
        "square_feet": (low / 2 + high / 2).where(valid),
        "area_range_width": (high - low).where(valid),
    }, index=values.index)


def make_off_market_features(homes: pd.DataFrame, include_text: bool = True) -> pd.DataFrame:
    """Build a fixed-schema feature frame from off-market prediction inputs.

    Categories are strings. Numeric columns are float64 to avoid overflow when
    deriving areas; genuinely unknown/invalid measurements remain NaN. All
    transformations are row-local and do not read labels, IDs or sale dates.
    Additional input columns are ignored. Missing required columns fail loudly.
    """

    if not isinstance(homes, pd.DataFrame):
        raise TypeError('homes must be a pandas DataFrame')
    if homes.columns.duplicated().any():
        raise ValueError('Duplicate input column names are not supported')
    required = [*REQUIRED_COLUMNS, *( ["remarks"] if include_text else [])]
    missing = sorted(set(required) - set(homes.columns))
    if missing:
        raise ValueError("Missing required off-market input columns: " + ", ".join(missing))

    X = pd.DataFrame(index=homes.index)
    for column in ['house_type_name', 'house_style', 'municipality', 'community']:
        X[column] = (homes[column].astype("string").str.normalize("NFKC")
                     .str.strip().str.replace(r"\s+", " ", regex=True)
                     .replace("", pd.NA).fillna("__MISSING__").astype(str))
    # Length-prefixed components avoid accidental category collisions when a
    # name contains a separator; include municipality in neighborhood identity.
    def combine(left, right):
        return left.str.len().astype(str) + ':' + left + right
    X['municipality_community'] = combine(X['municipality'], X['community'])
    X["community_type"] = combine(X['municipality_community'], X['house_type_name'])
    for column in NUMERIC_INPUTS:
        X[column] = pd.to_numeric(homes[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        if column in ['latitude', 'longitude']:
            bound = 90 if column == 'latitude' else 180
            # CSV and Parquet can differ by one floating-point ULP. Near a
            # tree split even 1e-14 degrees can change the selected branch.
            # Eight decimal degrees retain roughly millimeter precision while
            # giving equivalent source formats identical numeric coordinates.
            X[column] = X[column].where(X[column].between(-bound, bound)).round(8)
        else:
            X[column] = X[column].where(X[column] >= 0)
    no_lot = X['house_type_name'].str.casefold().isin(NO_LOT_TYPES)
    for column in ['lot_front', 'lot_depth']:
        X[column] = X[column].where((X[column] > 0) & ~no_lot)

    area = parse_house_area(homes["house_area"])
    X["square_feet"] = area["square_feet"]
    # Do not use interval-width flags: the cleaned CSV retains only midpoints.
    # This gives the same features for original ranges and cleaned numeric areas.
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

    listed = pd.to_datetime(homes["date_listed"], errors="raise", format='mixed', utc=True).dt.tz_localize(None)
    if listed.isna().any():
        raise ValueError('date_listed must not be missing')
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
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan).astype("float64")
    return X


def off_market_features(homes: pd.DataFrame, include_text: bool = True) -> pd.DataFrame:
    """Numeric features for the starter's FeatureModel interface.

    Fixed property-type flags keep the schema independent of the prediction
    batch. The default OffMarketModel uses make_off_market_features directly to
    retain categorical style/municipality/community information inside its
    fold-fitted encoder.
    """
    rich = make_off_market_features(homes, include_text=include_text)
    X = rich.drop(columns=CATEGORICAL_COLUMNS).copy()
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = rich["house_type_name"].eq(kind).astype(float)
    return X


def _validate_list_aware_inputs(homes):
    if not isinstance(homes, pd.DataFrame):
        raise TypeError("homes must be a pandas DataFrame")
    if homes.columns.duplicated().any():
        raise ValueError("Duplicate input column names are not supported")
    required = {"house_area", "bedrooms", "washrooms", "parking_total", "latitude",
                "longitude", "date_listed", "house_type_name"}
    missing = sorted(required - set(homes.columns))
    if missing:
        raise ValueError("Missing required list-aware input columns: " + ", ".join(missing))


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def _list_aware_base_features(homes):
    """Preserve Joey's original base so richer off-market features do not alter it."""
    _validate_list_aware_inputs(homes)
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    # Months since January 2021, so the model can follow prices over time.
    listed = pd.to_datetime(homes["date_listed"], format="mixed", errors="raise", utc=True).dt.tz_localize(None)
    if listed.isna().any():
        raise ValueError("date_listed must not be missing")
    X["months"] = (listed - pd.Timestamp("2021-01-01")).dt.days / 30.44
    # One 0/1 column per property type.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = homes["house_type_name"].eq(kind).fillna(False).astype(int)
    return X


def list_aware_features(homes):
    """Features for homes with a list price."""
    if not isinstance(homes, pd.DataFrame):
        raise TypeError("homes must be a pandas DataFrame")
    if "list_price" not in homes:
        raise ValueError("Missing required list-aware input columns: list_price")
    X = _list_aware_base_features(homes)

    X["list_price"] = homes["list_price"]

    # Listing price per square foot
    X["list_price_per_sqft"] = (
        homes["list_price"] / X["square_feet"].clip(lower=100)
    )

    # Separate price signals for different house types
    X["detached_list_price"] = (
        homes["list_price"]
        * homes["house_type_name"].eq("Detached").fillna(False).astype(int)
    )

    X["semi_list_price"] = (
        homes["list_price"]
        * homes["house_type_name"].eq("Semi-Detached").fillna(False).astype(int)
    )

    return X
