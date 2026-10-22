"""Past-sales baselines for the off-market track.

Only sold prices of the supplied training rows are used. Every prediction further
requires ``date_sold < date_listed``. Thus a monthly training cutoff never permits
a comparable that had not sold when the subject home was listed.

The handout describes the scoring baseline as the median of the same property
type in the same community over the previous 12 months, but does not publish its
fallback or boundary conventions. ``RollingCommunityBaseline`` reproduces that
description with a calendar-year window [listing date - 12 months, listing date),
then falls back to same type/municipality, same type, and all recent sales; if
there are no recent sales, it uses all strictly earlier sales. These fallbacks
are our documented implementation, not a claim about the hidden scorer.

The nearest-sales definition follows the MIT-licensed official starter:
https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/models.py
Copyright (c) 2026 Erzheng Zhang. An independently implemented spatial index
accelerates candidate selection; final distances and tie ordering retain the
starter's definition (longitude scaled by 0.72, newest sale then ml_num).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


_MISSING = "__MISSING_BASELINE_CATEGORY__"


def _category(value):
    return _MISSING if pd.isna(value) else str(value)


def _training_frame(train, extra_columns=()):
    columns = list(dict.fromkeys(["sold_price", "date_sold", *extra_columns]))
    missing = set(columns).difference(train.columns)
    if missing:
        raise ValueError(f"Missing baseline training columns: {sorted(missing)}")
    rows = train.loc[:, columns].copy()
    if len(rows) == 0:
        raise ValueError("A baseline requires at least one training sale")
    rows["date_sold"] = pd.to_datetime(rows["date_sold"], errors="raise")
    rows["sold_price"] = pd.to_numeric(rows["sold_price"], errors="raise")
    if rows["date_sold"].isna().any():
        raise ValueError("Training sale dates must not be missing")
    if not np.isfinite(rows["sold_price"]).all() or (rows["sold_price"] <= 0).any():
        raise ValueError("Training sold prices must be positive and finite")
    return rows


def _listing_dates(homes):
    dates = pd.to_datetime(homes["date_listed"], errors="raise")
    if dates.isna().any():
        raise ValueError("Prediction listing dates must not be missing")
    return dates


def _sale_arrays(rows):
    ordered = rows.sort_values("date_sold", kind="stable")
    return (ordered["date_sold"].to_numpy(dtype="datetime64[ns]"),
            ordered["sold_price"].to_numpy(dtype=float))


def _window_median(arrays, cutoff, lower=None):
    dates, prices = arrays
    right = np.searchsorted(dates, np.datetime64(cutoff, "ns"), side="left")
    left = 0 if lower is None else np.searchsorted(
        dates, np.datetime64(lower, "ns"), side="left")
    return float(np.median(prices[left:right])) if right > left else None


class GlobalMedianBaseline:
    """Median of all available training sales strictly before each listing date."""

    def fit(self, train):
        self.sales_ = _sale_arrays(_training_frame(train))
        return self

    def predict(self, homes):
        dates = _listing_dates(homes)
        cache = {}
        result = []
        for date in dates:
            if date not in cache:
                value = _window_median(self.sales_, date)
                if value is None:
                    raise ValueError(f"No training sale predates listing date {date}")
                cache[date] = value
            result.append(cache[date])
        return np.asarray(result, dtype=float)


class RollingCommunityBaseline:
    """Rolling community/type median with documented, strictly past-only fallbacks.

    ``municipality`` is optional for compatibility; when absent, its fallback is
    skipped. No minimum group size is imposed by the handout, so a single eligible
    sale is enough. Missing categorical values form their own group.
    """

    def fit(self, train):
        columns = ["house_type_name", "community"]
        self.has_municipality_ = "municipality" in train.columns
        if self.has_municipality_:
            columns.append("municipality")
        rows = _training_frame(train, columns)
        for column in columns:
            rows[column] = rows[column].map(_category)
        self.global_ = _sale_arrays(rows)
        self.community_ = {
            key: _sale_arrays(group)
            for key, group in rows.groupby(["house_type_name", "community"], sort=False)
        }
        self.type_ = {
            key: _sale_arrays(group)
            for key, group in rows.groupby("house_type_name", sort=False)
        }
        self.municipality_ = {
            key: _sale_arrays(group)
            for key, group in rows.groupby(["house_type_name", "municipality"], sort=False)
        } if self.has_municipality_ else {}
        return self

    def predict(self, homes):
        dates = _listing_dates(homes)
        result, cache = [], {}
        for position, (_, home) in enumerate(homes.iterrows()):
            cutoff = dates.iloc[position]
            kind = _category(home["house_type_name"])
            community = _category(home["community"])
            municipality = _category(home.get("municipality", np.nan))
            key = (cutoff, kind, community, municipality)
            if key not in cache:
                lower = cutoff - pd.DateOffset(months=12)
                candidates = [self.community_.get((kind, community))]
                if self.has_municipality_:
                    candidates.append(self.municipality_.get((kind, municipality)))
                candidates.extend([self.type_.get(kind), self.global_])
                value = None
                for arrays in candidates:
                    if arrays is not None:
                        value = _window_median(arrays, cutoff, lower)
                        if value is not None:
                            break
                if value is None:
                    value = _window_median(self.global_, cutoff)
                if value is None:
                    raise ValueError(f"No training sale predates listing date {cutoff}")
                cache[key] = value
            result.append(cache[key])
        return np.asarray(result, dtype=float)


class NearestSalesBaseline:
    """Median of k nearest strictly earlier same-type training sales.

    If no earlier sale has the requested type, use all earlier training sales.
    Equal-distance ties prefer newer sales, then lexicographically smaller IDs,
    as in the official starter. A subject with no earlier training sale raises
    a useful error rather than returning an invalid NaN prediction.
    """

    def __init__(self, k=10):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k

    @staticmethod
    def _index(rows):
        longitude = rows["longitude"].to_numpy(dtype=float)
        latitude = rows["latitude"].to_numpy(dtype=float)
        if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
            raise ValueError("Nearest-sales baseline requires finite coordinates")
        return {
            "tree": BallTree(np.column_stack([longitude * 0.72, latitude])),
            "longitude": longitude,
            "latitude": latitude,
            "dates": rows["date_sold"].to_numpy(dtype="datetime64[ns]"),
            "prices": rows["sold_price"].to_numpy(dtype=float),
        }

    def fit(self, train):
        rows = _training_frame(
            train, ["ml_num", "house_type_name", "latitude", "longitude"])
        rows["ml_num"] = rows["ml_num"].astype(str)
        rows["house_type_name"] = rows["house_type_name"].map(_category)
        rows = rows.sort_values(["date_sold", "ml_num"], ascending=[False, True], kind="stable")
        self.all_ = self._index(rows)
        self.by_type_ = {
            key: self._index(group)
            for key, group in rows.groupby("house_type_name", sort=False)
        }
        return self

    def _predict_from_index(self, index, longitude, latitude, cutoff):
        n = len(index["prices"])
        point = np.array([[longitude * 0.72, latitude]])
        count = min(n, max(32, self.k * 4))
        while True:
            _, indices = index["tree"].query(point, k=count)
            eligible = indices[0][index["dates"][indices[0]] < cutoff]
            if len(eligible) >= self.k or count == n:
                break
            count = min(n, count * 2)
        if len(eligible) == 0:
            return None

        # The tree uses scaled coordinates; recompute the starter's exact order
        # of arithmetic and include a small radius tolerance for floating error.
        dx = (index["longitude"][eligible] - longitude) * 0.72
        dy = index["latitude"][eligible] - latitude
        distances = np.sqrt(dx * dx + dy * dy)
        kth = min(self.k, len(eligible)) - 1
        radius = float(np.partition(distances, kth)[kth]) + 1e-10
        candidate = index["tree"].query_radius(point, r=radius)[0]
        candidate = np.sort(candidate[index["dates"][candidate] < cutoff])
        dx = (index["longitude"][candidate] - longitude) * 0.72
        dy = index["latitude"][candidate] - latitude
        exact = np.sqrt(dx * dx + dy * dy)
        nearest = candidate[np.argsort(exact, kind="stable")[:self.k]]
        return float(np.median(index["prices"][nearest]))

    def predict(self, homes):
        dates = _listing_dates(homes).to_numpy(dtype="datetime64[ns]")
        result, cache = [], {}
        for position, (_, home) in enumerate(homes.iterrows()):
            longitude, latitude = float(home["longitude"]), float(home["latitude"])
            if not np.isfinite(longitude) or not np.isfinite(latitude):
                raise ValueError("Nearest-sales baseline requires finite coordinates")
            kind = _category(home["house_type_name"])
            cutoff = dates[position]
            key = (kind, longitude, latitude, cutoff)
            if key not in cache:
                index = self.by_type_.get(kind, self.all_)
                value = self._predict_from_index(index, longitude, latitude, cutoff)
                if value is None and index is not self.all_:
                    value = self._predict_from_index(self.all_, longitude, latitude, cutoff)
                if value is None:
                    raise ValueError(f"No training sale predates listing date {cutoff}")
                cache[key] = value
            result.append(cache[key])
        return np.asarray(result, dtype=float)
