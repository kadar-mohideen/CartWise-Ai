"""Read the raw Amazon files (ratings CSV, metadata JSON lines) and save/load tables."""
import gzip
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RATING_COLUMNS = ['item_id', 'user_id', 'rating', 'timestamp']   # order used by the ratings-only CSV
META_FIELDS = [
    'asin', 'title', 'brand', 'feature', 'description', 'category', 'categories',
    'main_cat', 'price', 'imageURL', 'imageURLHighRes', 'image',
]


def load_ratings_csv(path):
    """Load the ratings-only CSV. Handles a header row and swapped column order."""
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline().strip().split(',')
    if len(first) != 4:
        raise ValueError(f'Expected 4 columns in {path.name}, found {len(first)}: {first}')
    try:
        float(first[2])
        header = None
    except ValueError:
        header = 0
        log.info('Header row detected in %s', path.name)
    df = pd.read_csv(path, header=header, names=RATING_COLUMNS,
                     dtype={'item_id': str, 'user_id': str})
    # Product IDs (ASINs) are 10 characters; reviewer IDs are longer. Fix a swapped order.
    if len(df) and df['item_id'].astype(str).str.len().median() > df['user_id'].astype(str).str.len().median():
        log.warning('Columns look swapped (item/user). Swapping them back.')
        df = df.rename(columns={'item_id': 'user_id', 'user_id': 'item_id'})
    return df


def load_metadata(path, keep=META_FIELDS):
    """Stream a (gzipped) JSON-lines metadata file and keep only the useful fields."""
    path = str(path)
    opener = gzip.open if path.endswith('.gz') else open
    rows, bad = [], 0
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            rows.append({k: rec[k] for k in keep if k in rec})
    if bad:
        log.warning('Skipped %d malformed metadata lines', bad)
    return pd.DataFrame(rows)


def save_table(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Run the preprocessing step first.')
    return pd.read_parquet(path)
