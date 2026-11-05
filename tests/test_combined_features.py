"""Feature-boundary and model-integration checks on synthetic homes."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from threadpoolctl import threadpool_limits

import features
import models
import modelsfinal
from test_starter import make_listings


@pytest.fixture
def homes():
    frame = make_listings(6, seed=7)
    frame['house_type_name'] = features.PROPERTY_TYPES
    frame['house_area'] = ['600-699', '1200', '1400-1600', '800', '1000', '650']
    frame['lot_front'] = [20.0, 30.0, 25.0, 20.0, 20.0, 20.0]
    frame['lot_depth'] = 100.0
    frame['remarks'] = ['Renovated with a finished basement', None, '', '', '', '']
    frame.index = [31, 8, 101, 2, 19, 0]
    return frame


@pytest.mark.parametrize('builder', [
    features.make_off_market_features, features.off_market_features,
    features.list_aware_features,
])
def test_forbidden_fields_and_ids_cannot_change_features(builder, homes):
    before = builder(homes)
    altered = homes.copy()
    altered['ml_num'] = 'not-a-feature'
    for column in ('sold_price', 'date_sold', 'days_on_market',
                   'external_avm_estimate', 'external_avm_estimate_date'):
        altered[column] = 9e18
    if builder is not features.list_aware_features:
        altered['list_price'] = 9e18
    assert_frame_equal(builder(altered), before)


@pytest.mark.parametrize('builder,columns', [
    (features.make_off_market_features, 55),
    (features.off_market_features, 55),
    (features.list_aware_features, 17),
])
def test_features_are_row_local_with_fixed_schema_and_order(builder, columns, homes):
    full = builder(homes)
    assert full.shape == (len(homes), columns)
    assert_frame_equal(builder(homes.iloc[[4, 0, 3]]), full.iloc[[4, 0, 3]])
    batches = pd.concat([builder(homes.iloc[:2]), builder(homes.iloc[2:])])
    assert_frame_equal(batches, full)
    assert_frame_equal(builder(homes.iloc[:0]), full.iloc[:0])
    numeric = full.select_dtypes(include='number').to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
    altered = homes.copy()
    altered['house_type_name'] = 'Previously unseen type'
    altered['community'] = 'Previously unseen community'
    assert list(builder(altered).columns) == list(full.columns)


def test_off_market_preserves_unknown_condo_land_and_text_signals(homes):
    frame = homes.copy()
    frame.iloc[1, frame.columns.get_loc('house_area')] = '1,000–1,400 sq ft'
    rich = features.make_off_market_features(frame)
    condo = rich.loc[frame.house_type_name.isin(['Condo Apt', 'Condo Townhouse', 'Co-Op Apt'])]
    assert condo[['lot_front', 'lot_depth', 'lot_area']].isna().all().all()
    assert (condo[['lot_front_missing', 'lot_depth_missing']] == 1).all().all()
    assert rich.loc[8, 'square_feet'] == 1200.0
    assert rich.loc[8, 'lot_area'] == 3000.0
    assert rich.loc[31, 'text_renovated'] == 1.0
    assert rich.loc[31, 'text_finished_basement'] == 1.0
    assert rich.loc[8, 'text_word_count'] == 0.0


def test_off_market_parser_keeps_invalid_area_unknown():
    result = features.parse_house_area(pd.Series(
        ['622', '900-1099', '1,200 sq ft', '1000–1400', None, 'bad', '0', '200-100']))
    np.testing.assert_array_equal(result.square_feet.iloc[:4], [622, 999.5, 1200, 1200])
    assert result.iloc[4:].isna().all().all()


def test_numeric_view_runs_through_starter_feature_model():
    train = make_listings(120, seed=11)
    numeric = features.off_market_features(train)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in numeric.dtypes)
    assert set(features.CATEGORICAL_COLUMNS).isdisjoint(numeric.columns)
    with threadpool_limits(limits=2):
        model = models.FeatureModel(features.off_market_features).fit(train)
        predicted = model.predict(train.iloc[:7])
    assert predicted.shape == (7,)
    assert np.isfinite(predicted).all() and (predicted > 0).all()


def test_both_model_entrypoints_use_shared_rich_builder():
    assert models.make_off_market_features is features.make_off_market_features
    assert modelsfinal.make_off_market_features is features.make_off_market_features
    assert models.CATEGORICAL_COLUMNS is features.CATEGORICAL_COLUMNS
    assert modelsfinal.CATEGORICAL_COLUMNS is features.CATEGORICAL_COLUMNS


def test_joey_list_aware_schema_and_price_interactions_are_preserved(homes):
    actual = features.list_aware_features(homes)
    columns = ['square_feet', 'bedrooms', 'washrooms', 'parking', 'latitude',
               'longitude', 'months'] + ['is_' + kind for kind in features.PROPERTY_TYPES]
    columns += ['list_price', 'list_price_per_sqft', 'detached_list_price', 'semi_list_price']
    assert list(actual.columns) == columns
    expected_area = pd.Series([649.5, 1200, 1500, 800, 1000, 650], index=homes.index)
    np.testing.assert_array_equal(actual.square_feet, expected_area)
    np.testing.assert_array_equal(actual.list_price, homes.list_price)
    np.testing.assert_array_equal(actual.list_price_per_sqft,
                                  homes.list_price / expected_area.clip(lower=100))
    np.testing.assert_array_equal(actual.detached_list_price,
                                  homes.list_price * homes.house_type_name.eq('Detached'))
    np.testing.assert_array_equal(actual.semi_list_price,
                                  homes.list_price * homes.house_type_name.eq('Semi-Detached'))
    # Enhanced off-market-only inputs must not become new requirements for Joey.
    reduced = homes.drop(columns=['house_style', 'municipality', 'community',
                                  'garage', 'lot_front', 'lot_depth', 'remarks'])
    assert_frame_equal(features.list_aware_features(reduced), actual)


def test_missing_nullable_property_type_matches_object_missing_type(homes):
    nullable = homes.copy()
    nullable['house_type_name'] = nullable.house_type_name.astype('string')
    nullable.loc[nullable.index[0], 'house_type_name'] = pd.NA
    ordinary = nullable.copy()
    ordinary['house_type_name'] = ordinary.house_type_name.astype(object)
    assert_frame_equal(features.list_aware_features(nullable),
                       features.list_aware_features(ordinary))
    result = features.list_aware_features(nullable)
    flags = ['is_' + kind for kind in features.PROPERTY_TYPES]
    flags += ['detached_list_price', 'semi_list_price']
    assert (result.loc[nullable.index[0], flags] == 0).all()


def test_off_market_text_ablation_needs_no_remarks(homes):
    full = features.make_off_market_features(homes)
    without = features.make_off_market_features(homes.drop(columns='remarks'), include_text=False)
    assert not any(column.startswith('text_') for column in without)
    assert_frame_equal(without, full.loc[:, without.columns])


@pytest.mark.parametrize('builder', [features.make_off_market_features, features.list_aware_features])
def test_missing_and_duplicated_input_columns_fail(builder, homes):
    with pytest.raises(ValueError, match='house_area'):
        builder(homes.drop(columns='house_area'))
    duplicated = pd.concat([homes, homes[['house_area']]], axis=1)
    with pytest.raises(ValueError, match='Duplicate'):
        builder(duplicated)
