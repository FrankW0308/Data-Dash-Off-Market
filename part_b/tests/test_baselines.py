"""Behavioral checks for strict date cutoffs and exact nearest-sales ranking."""

import numpy as np
import pandas as pd
import pytest

from part_b.baselines import (
    GlobalMedianBaseline,
    NearestSalesBaseline,
    RollingCommunityBaseline,
)


def _sales():
    return pd.DataFrame({
        "ml_num": ["old", "in_window", "just_before", "same_day", "future"],
        "sold_price": [10., 100., 300., 1e6, 1e7],
        "date_sold": pd.to_datetime([
            "2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-01-02"]),
        "house_type_name": "Detached", "community": "A", "municipality": "Toronto",
        "latitude": 43.7, "longitude": -79.4,
    })


def _home():
    return pd.DataFrame({
        "date_listed": [pd.Timestamp("2025-01-01")],
        "house_type_name": ["Detached"], "community": ["A"],
        "municipality": ["Toronto"], "latitude": [43.7], "longitude": [-79.4],
    })


def test_global_and_rolling_use_strict_past_and_calendar_window():
    train, homes = _sales(), _home()
    assert GlobalMedianBaseline().fit(train).predict(homes)[0] == 100.
    assert RollingCommunityBaseline().fit(train).predict(homes)[0] == 200.
    # Unknown community falls back to the same type/municipality, still past-only.
    homes["community"] = "unknown"
    assert RollingCommunityBaseline().fit(train).predict(homes)[0] == 200.


def test_nearest_ties_prefer_newest_eligible_sale():
    assert NearestSalesBaseline(k=1).fit(_sales()).predict(_home())[0] == 300.


def test_no_earlier_sales_is_explicit_failure():
    homes = _home().assign(date_listed=pd.Timestamp("1900-01-01"))
    for baseline in [GlobalMedianBaseline(), RollingCommunityBaseline(), NearestSalesBaseline()]:
        with pytest.raises(ValueError, match="No training sale predates"):
            baseline.fit(_sales()).predict(homes)


def test_spatial_acceleration_matches_brute_force_with_ties_and_future_sales():
    rng = np.random.default_rng(9)
    n = 800
    train = pd.DataFrame({
        "ml_num": [f"SYN-{i:05d}" for i in range(n)],
        "sold_price": rng.uniform(100_000, 3_000_000, n),
        "date_sold": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 450, n), unit="D"),
        "house_type_name": rng.choice(["Detached", "Condo Apt"], n),
        "latitude": rng.choice([43.7, 43.71, 43.75], n),
        "longitude": rng.choice([-79.4, -79.41, -79.45], n),
    })
    homes = train.sample(30, random_state=2).drop(columns=["sold_price", "date_sold"])
    homes["date_listed"] = pd.Timestamp("2024-02-01") + pd.to_timedelta(rng.integers(0, 450, len(homes)), unit="D")
    homes.iloc[0, homes.columns.get_loc("house_type_name")] = "unknown"
    expected = []
    ordered = train.sort_values(["date_sold", "ml_num"], ascending=[False, True])
    for _, home in homes.iterrows():
        earlier = ordered[(ordered["house_type_name"] == home["house_type_name"]) &
                          (ordered["date_sold"] < home["date_listed"])]
        if len(earlier) == 0:
            earlier = ordered[ordered["date_sold"] < home["date_listed"]]
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        nearest = np.argsort(np.sqrt(dx ** 2 + dy ** 2), kind="stable")[:10]
        expected.append(np.median(earlier["sold_price"].to_numpy()[nearest]))
    actual = NearestSalesBaseline(k=10).fit(train).predict(homes)
    np.testing.assert_array_equal(actual, expected)
