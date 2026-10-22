"""Protect the off-market feature boundary and interval-area interpretation."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from part_b.features import CATEGORICAL_COLUMNS, make_features, parse_house_area


def example_homes():
    return pd.DataFrame({
        "ml_num": ["one", "two"], "date_listed": ["2026-01-02", "2026-02-03"],
        "house_type_name": ["Detached", "Condo Apt"], "house_style": ["2-Storey", "Apartment"],
        "house_area": ["900-1099", "622"], "municipality": ["Toronto", "Toronto"],
        "community": ["Example", "Example"], "bedrooms": [2, 1], "washrooms": [2, 1],
        "parking_total": [1, 0], "garage": [1, 0], "lot_front": [20, np.nan],
        "lot_depth": [100, np.nan], "latitude": [43.7, 43.8], "longitude": [-79.4, -79.3],
        "remarks": ["Fully renovated, finished basement with a separate entrance.", None],
    })


class FeatureTests(unittest.TestCase):
    def test_point_range_and_unknown_area(self):
        result = parse_house_area(pd.Series(["622", "900-1099", None, "bad", "0", "200-100"]))
        self.assertEqual(result.square_feet.iloc[0], 622)
        self.assertEqual(result.area_range_width.iloc[0], 0)
        self.assertEqual(result.square_feet.iloc[1], 999.5)
        self.assertEqual(result.area_range_width.iloc[1], 199)
        self.assertTrue(result.iloc[2:].isna().all().all())

    def test_forbidden_columns_and_ids_cannot_change_features(self):
        homes = example_homes()
        before = make_features(homes)
        homes["ml_num"] = ["changed", "another"]
        for column in ["sold_price", "list_price", "date_sold", "days_on_market",
                       "external_avm_estimate", "external_avm_estimate_date"]:
            homes[column] = [123456, 999999999]
        assert_frame_equal(before, make_features(homes))

    def test_missing_lot_and_text_features(self):
        X = make_features(example_homes())
        self.assertTrue(np.isnan(X.loc[1, "lot_area"]))
        self.assertEqual(X.loc[1, "lot_front_missing"], 1)
        self.assertEqual(X.loc[0, "text_renovated"], 1)
        self.assertEqual(X.loc[0, "text_finished_basement"], 1)
        self.assertEqual(X.loc[0, "text_separate_entrance"], 1)
        self.assertEqual(X.loc[1, "text_word_count"], 0)
        self.assertTrue(all(X[c].dtype == object for c in CATEGORICAL_COLUMNS))

    def test_missing_required_column_fails(self):
        with self.assertRaisesRegex(ValueError, "house_area"):
            make_features(example_homes().drop(columns="house_area"))

    def test_text_can_be_omitted_for_ablation(self):
        X = make_features(example_homes().drop(columns="remarks"), include_text=False)
        self.assertFalse(any(c.startswith("text_") for c in X))


if __name__ == "__main__":
    unittest.main()
