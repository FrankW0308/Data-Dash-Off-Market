"""Adapter for the official starter's fit(train) / predict(homes) interface."""
import json
from pathlib import Path

from .run import CONFIGS, make_model


def make_off_market_model(threads=4):
    """Construct the frozen selected model, to refit inside any earlier-sales fold."""
    path = Path(__file__).parent / 'outputs' / 'selection.json'
    if not path.exists():
        raise FileNotFoundError('Run python -m part_b tune, or restore the delivered selection.json first.')
    selection = json.loads(path.read_text())
    name = selection['selected_model']
    if CONFIGS[name] != selection['selected_config']:
        raise ValueError('Saved selected hyperparameters differ from the source code.')
    return make_model(name, threads)
