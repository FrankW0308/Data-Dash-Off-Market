"""Combined competition models with fit(train) / predict(homes) interfaces.

make_model("off_market") uses the off-market model from the uploaded models.py.
make_model("list_aware") uses Joey's sold-to-list ratio model.

The accompanying features.py contains the official list-aware features. It is
imported only when fitting or predicting with ListRatioModel, so off-market use
does not require that module. Use the official chronological runner for scoring;
the historical part_b scores do not measure this combined implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


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

def make_model(track, *, threads=4):
    """Return a fresh model. Keep the official one-argument call compatible."""

    if track == 'off_market':
        return OffMarketModel(threads=threads)
    if track == 'list_aware':
        return ListRatioModel(threads=threads)
    raise ValueError("track must be 'off_market' or 'list_aware'")

class OffMarketModel(BaseEstimator):
    """Fold-fitted histogram gradient boosting on log(sold_price).

    These settings differ from the historical part_b model. No matching
    real-data validation scores are recorded for this implementation yet.
    Select settings using earlier chronological tuning folds, never test homes.
    No fitted artifact, dataset, optional boosting package or network access is
    required. The caller must supply only earlier sales in each training fold.
    """

    def __init__(self, threads=4):
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise ValueError('threads must be a positive integer')
        self.threads = threads

    def fit(self, train):
        if not isinstance(train, pd.DataFrame):
            raise TypeError('train must be a pandas DataFrame')
        if train.columns.duplicated().any():
            raise ValueError('Duplicate input column names are not supported')
        if 'sold_price' not in train or train.empty:
            raise ValueError('Training requires nonempty sales with sold_price')
        if 'ml_num' in train and (train.ml_num.isna().any() or not train.ml_num.is_unique):
            raise ValueError('Training ml_num values must be unique and nonmissing')
        y = pd.to_numeric(train['sold_price'], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(y).all() or (y <= 0).any():
            raise ValueError('sold_price must contain positive finite prices')
        X = make_off_market_features(train)
        categorical = CATEGORICAL_COLUMNS
        numeric = [c for c in X if c not in categorical]
        # Native numeric missing-value routing avoids artificial median land/size
        # values. Capping category counts keeps every category truly categorical;
        # lexical ordinal codes must never become continuous location features.
        preprocess = ColumnTransformer([
            ('numeric', 'passthrough', numeric),
            ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value',
                unknown_value=-1, encoded_missing_value=-1,
                max_categories=255), categorical),
        ])
        mask = [False] * len(numeric) + [True] * len(categorical)
        regressor = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=30, l2_regularization=20.0,
            # Disable the estimator's random internal validation split: all
            # model selection is instead evaluated on earlier chronological folds.
            early_stopping=False,
            categorical_features=mask, random_state=42,
        )
        pipeline = Pipeline([('preprocess', preprocess), ('regressor', regressor)])
        with threadpool_limits(limits=self.threads):
            pipeline.fit(X, np.log(y))
        self.pipeline_ = pipeline
        self.feature_names_ = list(X.columns)
        self.training_rows_ = len(train)
        self.model_version_ = 'off-market-v2'
        return self

    def predict(self, homes):
        check_is_fitted(self, 'pipeline_')
        X = make_off_market_features(homes)[self.feature_names_]
        if X.empty:
            return np.empty(0, dtype=float)
        with threadpool_limits(limits=self.threads):
            predictions = np.exp(self.pipeline_.predict(X))
        if not np.isfinite(predictions).all() or (predictions <= 0).any():
            raise ValueError('Model generated invalid prices')
        return predictions


class ListPriceBaseline:
    """Starter list-aware baseline; Member C can replace this independently."""

    def fit(self, train):
        return self

    def predict(self, homes):
        return homes['list_price'].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Starter comparison: k nearest same-type sales strictly before listing.

    Based on the MIT-licensed SSU-Data-Dash starter by Erzheng Zhang (2026).
    Equal distances prefer newer sales, then the lexicographically smaller ID.
    """

    def __init__(self, k=10):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError('k must be a positive integer')
        self.k = k

    def fit(self, train):
        columns = ['ml_num', 'house_type_name', 'date_sold', 'latitude',
                   'longitude', 'sold_price']
        self.sales = train[columns].copy()
        self.sales['date_sold'] = pd.to_datetime(self.sales['date_sold'], errors='raise')
        if self.sales.empty or self.sales['date_sold'].isna().any():
            raise ValueError('Baseline requires nonempty sales with valid dates')
        self.sales = self.sales.sort_values(['date_sold', 'ml_num'], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby('house_type_name')))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        cutoff = pd.Timestamp(home['date_listed'])
        sales = self.sales_by_type.get(home['house_type_name'], self.sales)
        earlier = sales[sales['date_sold'] < cutoff]
        if earlier.empty:
            earlier = self.sales[self.sales['date_sold'] < cutoff]
        if earlier.empty:
            raise ValueError('No training sale predates the listing date')
        dx = (earlier['longitude'].to_numpy() - home['longitude']) * 0.72
        dy = earlier['latitude'].to_numpy() - home['latitude']
        nearest = np.argsort(np.sqrt(dx * dx + dy * dy), kind='stable')[:self.k]
        return float(np.median(earlier['sold_price'].to_numpy()[nearest]))


class FeatureModel:
    """Compatibility with starter FeatureModel(features.off_market_features).

    The provided callable must return a numeric feature matrix. The default
    OffMarketModel also handles categoricals and missing values inside each fold.
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train['sold_price']))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))


class ListRatioModel(BaseEstimator):
    """Joey's list-aware sold-to-list ratio model, split by house type.

    Requires features.list_aware_features from the original starter project.
    Types with fewer than 100 training rows use the general fallback model.
    """

    def __init__(self, threads=4):
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise ValueError('threads must be a positive integer')
        self.threads = threads
        self.models = {}
        self.fallback_model = HistGradientBoostingRegressor(
            random_state=0
        )

    @staticmethod
    def _list_prices(homes):
        if not isinstance(homes, pd.DataFrame):
            raise TypeError('homes must be a pandas DataFrame')
        if homes.columns.duplicated().any():
            raise ValueError('Duplicate input column names are not supported')
        if 'list_price' not in homes:
            raise ValueError('List-aware inputs require list_price')
        prices = pd.to_numeric(homes['list_price'], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError('list_price must contain positive finite prices')
        return prices

    def fit(self, train):
        from features import list_aware_features

        list_prices = self._list_prices(train)
        if train.empty or 'sold_price' not in train:
            raise ValueError('Training requires nonempty sales with sold_price')
        sold_prices = pd.to_numeric(train['sold_price'], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(sold_prices).all() or (sold_prices <= 0).any():
            raise ValueError('sold_price must contain positive finite prices')
        x = list_aware_features(train)
        y = np.log(sold_prices / list_prices)
        if not np.isfinite(y).all():
            raise ValueError('Sold-to-list ratios must produce finite log targets')

        # Rebuild every fitted model on each call: a smaller new training fold
        # must never retain a type-specific model from a previous fit.
        fallback_model = HistGradientBoostingRegressor(random_state=0)
        fitted_models = {}
        with threadpool_limits(limits=self.threads):
            fallback_model.fit(x, y)

            # Train separate models for each house type.
            for house_type in train["house_type_name"].dropna().unique():
                mask = train["house_type_name"].eq(house_type).fillna(False).to_numpy(dtype=bool)

                if mask.sum() < 100:
                    continue

                model = HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    random_state=0
                )

                model.fit(x.loc[mask], y[mask])
                fitted_models[house_type] = model

        self.fallback_model = fallback_model
        self.models = fitted_models
        self.is_fitted_ = True
        return self

    def predict(self, homes):
        check_is_fitted(self, 'is_fitted_')
        list_prices = self._list_prices(homes)
        if homes.empty:
            return np.empty(0, dtype=float)

        from features import list_aware_features
        x = list_aware_features(homes)

        with threadpool_limits(limits=self.threads):
            # General model prediction first.
            predicted_log_ratio = self.fallback_model.predict(x)

            # Replace with house-type-specific predictions.
            for house_type, model in self.models.items():
                mask = homes["house_type_name"].eq(house_type).fillna(False).to_numpy(dtype=bool)

                if mask.any():
                    predicted_log_ratio[mask] = model.predict(x.loc[mask])

        predictions = list_prices * np.exp(predicted_log_ratio)
        if not np.isfinite(predictions).all() or (predictions <= 0).any():
            raise ValueError('Model generated invalid prices')
        return predictions
