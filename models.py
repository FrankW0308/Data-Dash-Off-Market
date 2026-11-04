"""Combined competition models with fit(train) / predict(homes) interfaces.

make_model("off_market") uses the off-market model from the uploaded models.py.
make_model("list_aware") uses Joey's sold-to-list ratio model.

The accompanying features.py contains the rich off-market builder and Joey's
list-aware features. Both tracks use that shared module, with separate feature
boundaries. Use the official chronological runner for scoring; historical
part_b scores do not measure this combined implementation.
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


from features import (
    CATEGORICAL_COLUMNS, NO_LOT_TYPES, NUMERIC_INPUTS, REQUIRED_COLUMNS,
    TEXT_PATTERNS, make_off_market_features, parse_house_area,
)


def make_model(track, *, threads=4):
    """Return a fresh model. Keep the official one-argument call compatible."""

    if track == 'off_market':
        return OffMarketModel(threads=threads)
    if track == 'list_aware':
        return ListRatioModel(threads=threads)
    raise ValueError("track must be 'off_market' or 'list_aware'")

class OffMarketModel(BaseEstimator):
    """Fold-fitted histogram gradient boosting on log(sold_price).

    These settings differ from the historical part_b model. Evaluate with the
    official expanding monthly validation and keep each result tied to its exact
    feature/model source version. Never select settings using hidden test homes.
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
