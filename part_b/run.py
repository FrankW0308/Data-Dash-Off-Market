"""Part B pipeline. From the repository root: python -m part_b all.

Tune ONLY on Oct/Nov 2025; freeze selection before the official Dec25-May26
walk-forward backtest. Test sale prices are never available to this program.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from .features import make_features
from .baselines import GlobalMedianBaseline, RollingCommunityBaseline, NearestSalesBaseline

ROOT = Path(__file__).resolve().parents[1]
TUNING_MONTHS = ['2025-10', '2025-11']
VALIDATION_MONTHS = [str(m) for m in pd.period_range('2025-12', '2026-05', freq='M')]
BASELINES = ['global_median', 'rolling_community_12m', 'nearest_sales_10']
CONFIGS = {
    'random_forest': dict(kind='rf', n_estimators=120, max_depth=24,
                          min_samples_leaf=3, max_features=0.8, max_samples=0.7),
    'hist_gradient_boosting': dict(kind='hgb', max_iter=350, learning_rate=0.07,
                                   max_leaf_nodes=31, l2_regularization=10.0),
    'catboost_d6_rmse': dict(kind='cat', depth=6, iterations=650,
                            learning_rate=0.06, loss_function='RMSE', l2_leaf_reg=5),
    'catboost_d7_rmse': dict(kind='cat', depth=7, iterations=950,
                            learning_rate=0.05, loss_function='RMSE', l2_leaf_reg=5),
    'catboost_d7_mae': dict(kind='cat', depth=7, iterations=950,
                           learning_rate=0.05, loss_function='MAE', l2_leaf_reg=5),
    'catboost_no_remarks': dict(kind='cat', depth=6, iterations=650,
                               learning_rate=0.06, loss_function='RMSE', l2_leaf_reg=5,
                               include_text=False),
}


def log(message):
    print(time.strftime('%H:%M:%S') + ' ' + message, flush=True)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if len(actual) == 0 or not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError('Metrics require nonempty, positive finite predictions.')
    signed = 100 * (predicted - actual) / actual
    ape = np.abs(signed)
    return dict(n=len(actual), mdape=float(np.median(ape)), mape=float(np.mean(ape)),
                median_signed_error=float(np.median(signed)),
                within_10_pct=float(100 * np.mean(ape <= 10)),
                within_20_pct=float(100 * np.mean(ape <= 20)))


class LearnedModel:
    """Fold-fitted preprocessing and log-price regression; no target-derived features."""
    def __init__(self, name, threads=4):
        self.name, self.threads = name, threads

    def fit(self, homes, y=None, prepared=None):
        config = dict(CONFIGS[self.name])
        kind = config.pop('kind')
        self.include_text = config.pop('include_text', True)
        X = make_features(homes, include_text=self.include_text) if prepared is None else prepared
        if not X.index.equals(homes.index):
            raise ValueError('Prepared feature rows do not match training rows.')
        self.columns = list(X.columns)
        target = np.log(homes.sold_price.to_numpy(float)) if y is None else np.log(y)
        self.cat_columns = list(X.select_dtypes(include=['object', 'category', 'string']).columns)
        if kind == 'cat':
            self.estimator = CatBoostRegressor(**config, random_seed=42,
                thread_count=self.threads, verbose=False, allow_writing_files=False)
            self.estimator.fit(X, target, cat_features=self.cat_columns)
        else:
            num = [c for c in self.columns if c not in self.cat_columns]
            # Fit categorical levels and imputation ONLY on this fold's earlier sales.
            pre = ColumnTransformer([
                ('num', SimpleImputer(strategy='median', add_indicator=True), num),
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                       encoded_missing_value=-1), self.cat_columns)])
            if kind == 'rf':
                reg = RandomForestRegressor(**config, random_state=42, n_jobs=self.threads)
            else:
                # Ordinal codes are categorical where the estimator's cardinality limit permits.
                missing_num = sum(X[c].isna().any() for c in num)
                cat_mask = [False] * (len(num) + missing_num)
                cat_mask += [X[c].nunique() <= 250 for c in self.cat_columns]
                reg = HistGradientBoostingRegressor(**config, random_state=42,
                    early_stopping=False, categorical_features=cat_mask)
            self.estimator = Pipeline([('preprocess', pre), ('regressor', reg)])
            with threadpool_limits(limits=self.threads):
                self.estimator.fit(X, target)
        return self

    def predict(self, homes, prepared=None):
        X = (make_features(homes, include_text=self.include_text) if prepared is None else prepared)[self.columns]
        if not X.index.equals(homes.index):
            raise ValueError('Prepared feature rows do not match prediction rows.')
        with threadpool_limits(limits=self.threads):
            pred = np.exp(self.estimator.predict(X))
        if not np.isfinite(pred).all() or (pred <= 0).any():
            raise ValueError('Model generated invalid prices.')
        return pred


def make_model(name, threads):
    baseline_classes = dict(global_median=GlobalMedianBaseline,
                            rolling_community_12m=RollingCommunityBaseline,
                            nearest_sales_10=NearestSalesBaseline)
    return baseline_classes[name]() if name in BASELINES else LearnedModel(name, threads)


def load_data(folder):
    def read(name):
        p = folder / f'{name}.parquet'
        if p.exists():
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(folder / f'{name}.csv')
        for c in ['date_sold', 'date_listed']:
            if c in df:
                df[c] = pd.to_datetime(df[c], errors='raise')
        if not df.ml_num.is_unique:
            raise ValueError(f'{name}: duplicate IDs')
        return df.reset_index(drop=True)
    train, test = read('listings'), read('test_off_market')
    if set(train.ml_num) & set(test.ml_num):
        raise ValueError('Training/test IDs overlap.')
    if not np.isfinite(train.sold_price).all() or (train.sold_price <= 0).any():
        raise ValueError('Invalid training targets.')
    return train, test


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + '\n', encoding='utf-8')


def fingerprint(train):
    h = hashlib.sha256(b'part-b-pipeline-v1')
    h.update(pd.util.hash_pandas_object(train, index=True).values.tobytes())
    for name in ['features.py', 'baselines.py', 'run.py']:
        h.update((Path(__file__).parent / name).read_bytes())
    h.update(json.dumps(CONFIGS, sort_keys=True).encode())
    for package in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'pyarrow', 'scipy']:
        h.update((package + '=' + importlib.metadata.version(package)).encode())
    return h.hexdigest()


def fold_predictions(train, name, month, args, data_hash):
    path = args.output_dir / 'cache' / f'{name}_{month}.csv'
    meta = path.with_suffix('.json')
    period = pd.Period(month, freq='M')
    past = train[train.date_sold < period.start_time]
    homes = train[train.date_sold.dt.to_period('M') == period]
    if len(past) == 0 or len(homes) == 0:
        raise ValueError(f'Empty training/validation month: {month}')
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get('fingerprint') == data_hash:
        cached = pd.read_csv(path)
        if cached.ml_num.tolist() == homes.ml_num.tolist():
            log(f'Reusing verified cache: {name} {month}')
            return cached
    log(f'Fitting {name}, {month}: {len(past):,} earlier sales / {len(homes):,} held-out homes')
    started = time.monotonic()
    model = make_model(name, args.threads)
    feature_frame = args._features.get(CONFIGS.get(name, {}).get('include_text', True))
    if isinstance(model, LearnedModel):
        model.fit(past, prepared=feature_frame.loc[past.index])
    else:
        model.fit(past)
    # Keep forbidden fields out of prediction, even for models that ignore them.
    allowed = homes.drop(columns=['sold_price', 'list_price', 'date_sold', 'days_on_market',
        'external_avm_estimate', 'external_avm_estimate_date'], errors='ignore')
    prediction = np.round(model.predict(allowed, prepared=feature_frame.loc[homes.index])
                          if isinstance(model, LearnedModel) else model.predict(allowed))
    result = homes[['ml_num', 'house_type_name', 'community', 'sold_price']].copy()
    result['model'], result['month'], result['predicted_price'] = name, month, prediction
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    score = metrics(result.sold_price, prediction)
    score.update(fingerprint=data_hash, train_n=len(past), seconds=time.monotonic() - started)
    save_json(meta, score)
    log(f'{name} {month}: MdAPE {score["mdape"]:.2f}% / MAPE {score["mape"]:.2f}% ({score["seconds"]:.1f}s)')
    return result


def summarize(frame):
    return sorted([dict(model=name, **metrics(g.sold_price, g.predicted_price))
                   for name, g in frame.groupby('model')], key=lambda s: (s['mdape'], s['mape']))


def tune(train, args, data_hash):
    frames = [fold_predictions(train, name, month, args, data_hash)
              for name in BASELINES + list(CONFIGS) for month in TUNING_MONTHS]
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(args.output_dir / 'tuning_predictions.csv', index=False)
    summary = summarize(result)
    learned = [s for s in summary if s['model'] in CONFIGS]
    selected = learned[0]['model']
    selection = dict(selected_model=selected, selected_config=CONFIGS[selected],
        tuning_months=TUNING_MONTHS, validation_months=VALIDATION_MONTHS,
        tuning_metrics=summary, fingerprint=data_hash,
        selection_rule='Lowest pooled Oct-Nov 2025 MdAPE; MAPE breaks ties. Selection frozen before Dec-May validation.')
    save_json(args.output_dir / 'selection.json', selection)
    log(f'Frozen selection: {selected}; tuning MdAPE {learned[0]["mdape"]:.2f}%')
    return selection


def read_selection(args, data_hash):
    selection = json.loads((args.output_dir / 'selection.json').read_text())
    if selection['fingerprint'] != data_hash:
        raise ValueError('Data/features changed since tuning; rerun tune before validation.')
    return selection


def validate(train, args, data_hash, selection):
    names = BASELINES + [selection['selected_model']]
    result = pd.concat([fold_predictions(train, n, m, args, data_hash)
                       for n in names for m in VALIDATION_MONTHS], ignore_index=True)
    result.to_csv(args.output_dir / 'validation_predictions.csv', index=False)
    summary = summarize(result)
    monthly = [dict(model=n, month=m, **metrics(g.sold_price, g.predicted_price))
               for (n, m), g in result.groupby(['model', 'month'])]
    pd.DataFrame(summary).to_csv(args.output_dir / 'validation_metrics.csv', index=False)
    pd.DataFrame(monthly).to_csv(args.output_dir / 'monthly_metrics.csv', index=False)
    selected = result[result.model == selection['selected_model']].copy()
    groups = []
    selected['price_band'] = pd.cut(selected.sold_price, [0, 500000, 1000000, 2000000, np.inf],
        labels=['Under $500k', '$500k-$1m', '$1m-$2m', '$2m+'])
    for col in ['house_type_name', 'price_band', 'community']:
        for value, g in selected.groupby(col, observed=True):
            groups.append(dict(group_by=col, group=str(value), **metrics(g.sold_price, g.predicted_price)))
    pd.DataFrame(groups).to_csv(args.output_dir / 'error_groups.csv', index=False)
    # Descriptive home-level intervals, not market-regime uncertainty.
    rng = np.random.default_rng(42)
    ape = 100 * np.abs(selected.predicted_price.to_numpy() - selected.sold_price.to_numpy()) / selected.sold_price.to_numpy()
    boots = np.array([[np.median(s), np.mean(s)] for s in
                      (rng.choice(ape, len(ape), replace=True) for _ in range(500))])
    intervals = dict(mdape=np.quantile(boots[:, 0], [.025, .975]).tolist(),
                     mape=np.quantile(boots[:, 1], [.025, .975]).tolist(),
                     note='500 iid home resamples; descriptive only, understates shared temporal/location uncertainty.')
    report = dict(selected_model=selection['selected_model'], tuning_months=TUNING_MONTHS,
        validation_months=VALIDATION_MONTHS, tuning_metrics=selection['tuning_metrics'],
        validation_metrics=summary, monthly_metrics=monthly, selected_95ci=intervals,
        fingerprint=data_hash, training_rows=len(train),
        validation_protocol='Official sale-month expanding walk-forward; date_sold < month start. Not a strict per-home listing-date simulation.')
    save_json(args.output_dir / 'metrics_summary.json', report)
    return report


def fit_final(train, args, selection):
    log('Refitting frozen selected model on all 150,000 training sales')
    model = make_model(selection['selected_model'], args.threads)
    if isinstance(model, LearnedModel):
        model.fit(train, prepared=args._features[CONFIGS[selection['selected_model']].get('include_text', True)])
    else:
        model.fit(train)
    folder = args.output_dir / 'models'
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / 'off_market_model.joblib', compress=3)
    if isinstance(model, LearnedModel) and isinstance(model.estimator, CatBoostRegressor):
        model.estimator.save_model(str(folder / 'off_market_model.cbm'))
        pd.DataFrame({'feature': model.columns,
            'importance': model.estimator.get_feature_importance()}).sort_values('importance', ascending=False).to_csv(
                args.output_dir / 'feature_importance.csv', index=False)
    save_json(folder / 'model_metadata.json', dict(model=selection['selected_model'],
        config=CONFIGS[selection['selected_model']], train_n=len(train),
        last_training_sale=str(train.date_sold.max().date()), fingerprint=selection['fingerprint']))
    return model


def predict(test, model, args):
    path = ROOT / 'submissions' / 'submission_off_market.csv'
    path.parent.mkdir(exist_ok=True)
    result = pd.DataFrame({'ml_num': test.ml_num, 'predicted_price': np.round(model.predict(test)).astype('int64')})
    result.to_csv(path, index=False)
    check_submission(path, test.ml_num)
    log(f'Wrote and checked {len(result):,} off-market predictions: {path}')
    return path


def check_submission(path, test_ids):
    df = pd.read_csv(path, dtype={'ml_num': str})
    assert list(df.columns) == ['ml_num', 'predicted_price'], 'Wrong columns'
    assert len(df) == len(test_ids) and df.ml_num.is_unique, 'Wrong rows/duplicate IDs'
    assert set(df.ml_num) == set(test_ids), 'Missing or extra test IDs'
    assert np.isfinite(df.predicted_price).all() and (df.predicted_price > 0).all(), 'Invalid price'
    log('Submission check PASS: exact schema, complete unique IDs, positive finite prices')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['all', 'tune', 'validate', 'fit', 'predict', 'check'])
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'part_b' / 'outputs')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test = load_data(args.data_dir)
    if args.command == 'check':
        check_submission(ROOT / 'submissions' / 'submission_off_market.csv', test.ml_num)
        return
    if args.command == 'predict':
        metadata = json.loads((args.output_dir / 'models' / 'model_metadata.json').read_text())
        if metadata['fingerprint'] != fingerprint(train):
            raise ValueError('Saved model does not match current data/code/dependencies; rerun all or fit as appropriate.')
        model = joblib.load(args.output_dir / 'models' / 'off_market_model.joblib')
        predict(test, model, args)
        return
    data_hash = fingerprint(train)
    log('Preparing deterministic row-local features once; all learned preprocessing remains fold-local')
    args._features = {True: make_features(train), False: make_features(train, include_text=False)}
    selection = tune(train, args, data_hash) if args.command in ['all', 'tune'] else read_selection(args, data_hash)
    if args.command in ['all', 'validate']:
        validate(train, args, data_hash, selection)
    if args.command in ['all', 'fit']:
        model = fit_final(train, args, selection)
        predict(test, model, args)
    log('Completed requested stage.')


if __name__ == '__main__':
    # Use the canonical module so saved model classes can load from notebooks too.
    from part_b.run import main as canonical_main
    canonical_main()
