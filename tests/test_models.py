"""Behavioral checks for the combined models; all data here is synthetic.

These tests exercise integration and leakage boundaries, not competition accuracy.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

import data
import models
from test_starter import make_listings


@pytest.fixture
def training_sales():
    sales = make_listings(240, seed=43)
    sales['house_type_name'] = ['Detached'] * 160 + ['Condo Apt'] * 80
    sales['sold_price'] = sales.list_price * np.exp(
        0.05 * sales.bedrooms - 0.03 * sales.washrooms)
    return sales


@pytest.fixture
def query_homes(training_sales):
    homes = training_sales.iloc[[0, 17, 88, 160, 200, 239]].copy()
    # Deliberately non-monotonic indexes and an unseen property type exercise
    # positional output order and both specialized/general model paths.
    homes.index = [91, 4, 88, 12, 1001, 5]
    homes.iloc[-1, homes.columns.get_loc('house_type_name')] = 'Unseen Type'
    return homes


@pytest.fixture(scope='module')
def fitted_models():
    sales = make_listings(240, seed=43)
    sales['house_type_name'] = ['Detached'] * 160 + ['Condo Apt'] * 80
    sales['sold_price'] = sales.list_price * np.exp(
        0.05 * sales.bedrooms - 0.03 * sales.washrooms)
    with threadpool_limits(limits=2):
        return {track: models.make_model(track, threads=2).fit(sales)
                for track in ('off_market', 'list_aware')}


def test_factory_selects_tracks_and_rejects_unknown_track():
    assert isinstance(models.make_model('off_market'), models.OffMarketModel)
    assert isinstance(models.make_model('list_aware'), models.ListRatioModel)
    with pytest.raises(ValueError):
        models.make_model('typo')


def test_list_aware_preserves_original_joey_predictions(fitted_models, query_homes):
    # Recorded by executing Joey's uploaded implementation with the unmodified
    # uploaded features joey.py, starter make_listings fixture above, and sklearn 1.7.2.
    # Covers >=100-row Detached, <100-row Condo Apt, and unseen-type fallback.
    expected = np.array([
        1452521.1664183757, 2039148.929356776, 1610505.1803209053,
        746512.9731172494, 2085590.665975117, 649441.0824052467,
    ])
    homes = data.hide_answers(query_homes, 'list_aware')
    actual = fitted_models['list_aware'].predict(homes)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-7)


def test_list_aware_refit_does_not_reuse_previous_type_models():
    first = make_listings(160).assign(
        house_type_name='Detached', list_price=600_000.0, sold_price=900_000.0)
    second = first.iloc[:20].assign(sold_price=300_000.0)
    homes = data.hide_answers(second.iloc[:2], 'list_aware')
    with threadpool_limits(limits=2):
        reused = models.make_model('list_aware', threads=2).fit(first).fit(second)
        fresh = models.make_model('list_aware', threads=2).fit(second)
        np.testing.assert_array_equal(reused.predict(homes), fresh.predict(homes))
        np.testing.assert_allclose(reused.predict(homes), 300_000.0)


def test_list_aware_missing_property_type_is_independent_of_pandas_dtype(training_sales, query_homes):
    nullable_train = training_sales.copy()
    nullable_train['house_type_name'] = nullable_train.house_type_name.astype('string')
    nullable_train.loc[nullable_train.index[::30], 'house_type_name'] = pd.NA
    nullable_homes = data.hide_answers(query_homes, 'list_aware').copy()
    nullable_homes['house_type_name'] = nullable_homes.house_type_name.astype('string')
    nullable_homes.loc[nullable_homes.index[0], 'house_type_name'] = pd.NA
    ordinary_train = nullable_train.copy()
    ordinary_train['house_type_name'] = ordinary_train.house_type_name.astype(object)
    ordinary_homes = nullable_homes.copy()
    ordinary_homes['house_type_name'] = ordinary_homes.house_type_name.astype(object)
    nullable_model = models.make_model('list_aware', threads=2).fit(nullable_train)
    ordinary_model = models.make_model('list_aware', threads=2).fit(ordinary_train)
    actual = nullable_model.predict(nullable_homes)
    np.testing.assert_array_equal(actual, ordinary_model.predict(ordinary_homes))
    assert actual.shape == (len(nullable_homes),)
    assert np.isfinite(actual).all() and (actual > 0).all()


@pytest.mark.parametrize('track', ['off_market', 'list_aware'])
def test_forbidden_prediction_fields_and_ids_are_ignored(track, fitted_models, query_homes):
    homes = data.hide_answers(query_homes, track)
    altered = homes.copy()
    altered['ml_num'] = ['changed'] * len(altered)
    for column in data.AFTER_SALE_COLUMNS:
        altered[column] = 9e15
    if track == 'off_market':
        altered['list_price'] = 9e15
    expected = fitted_models[track].predict(homes)
    np.testing.assert_array_equal(fitted_models[track].predict(altered), expected)
    assert expected.shape == (len(homes),)
    assert np.isfinite(expected).all() and (expected > 0).all()


@pytest.mark.parametrize('track', ['off_market', 'list_aware'])
def test_output_order_empty_predictions_and_serialization(track, fitted_models, query_homes, tmp_path):
    model = fitted_models[track]
    homes = data.hide_answers(query_homes, track)
    expected = model.predict(homes)
    order = [5, 2, 0, 4, 1, 3]
    np.testing.assert_array_equal(model.predict(homes.iloc[order]), expected[order])
    empty = model.predict(homes.iloc[:0])
    assert empty.shape == (0,) and np.issubdtype(empty.dtype, np.floating)
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(homes), expected)


@pytest.mark.parametrize('column', ['sold_price', 'list_price'])
@pytest.mark.parametrize('invalid', [0.0, -1.0, np.nan, np.inf])
def test_list_aware_rejects_invalid_training_prices(training_sales, column, invalid):
    sales = training_sales.iloc[:20].copy()
    sales.iloc[0, sales.columns.get_loc(column)] = invalid
    with pytest.raises(ValueError):
        models.make_model('list_aware').fit(sales)


@pytest.mark.parametrize('invalid', [0.0, -1.0, np.nan, np.inf])
def test_list_aware_rejects_invalid_prediction_list_prices(fitted_models, query_homes, invalid):
    homes = data.hide_answers(query_homes, 'list_aware')
    homes.iloc[0, homes.columns.get_loc('list_price')] = invalid
    with pytest.raises(ValueError):
        fitted_models['list_aware'].predict(homes)


@pytest.mark.parametrize('track', ['off_market', 'list_aware'])
def test_predictions_require_fitting(track, query_homes):
    with pytest.raises(NotFittedError):
        models.make_model(track).predict(data.hide_answers(query_homes, track))


def test_off_market_handles_large_and_unseen_category_sets():
    train = make_listings(300)
    train['community'] = [f'Community {i}' for i in range(len(train))]
    homes = data.hide_answers(train.iloc[:3], 'off_market').copy()
    homes.iloc[0, homes.columns.get_loc('community')] = 'Unseen community'
    model = models.make_model('off_market', threads=2).fit(train)
    predicted = model.predict(homes)
    assert predicted.shape == (len(homes),)
    assert np.isfinite(predicted).all() and (predicted > 0).all()
