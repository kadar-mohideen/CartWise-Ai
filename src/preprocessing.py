"""Cleaning functions. Each function returns (clean_data, report) so every decision is measurable."""
import html
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loader import load_metadata, load_ratings_csv, save_table

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r'<[^>]+>')
_WS_RE = re.compile(r'\s+')
_PRICE_RE = re.compile(r'\d[\d,]*\.?\d*')
IMAGE_COLUMNS = ['imageURLHighRes', 'imageURL', 'image']
# Some raw titles/descriptions contain scraped web-page code (JavaScript) instead of product text.
_CODE_RE = re.compile(
    r'function\s*\(|window\.\w+|\bvar\s+\w+\s*=|<script|\bue_\w+|\}\)\(|//\s*={3,}|\bdocument\.\w+',
    re.IGNORECASE)
_AMAZON_TITLE_RE = re.compile(r'Amazon\.com\s*:\s*([^:]{5,200}?)\s*:')
# Scraped lines that are not product descriptions (found in the real Amazon Fashion 'feature' field).
_JUNK_FEATURE_RE = re.compile(
    r'(package|product) dimensions|shipping weight|item model number|date first (listed|available)'
    r'|^asin\b|domestic shipping|international shipping|shipping information|best sellers rank'
    r'|average customer review', re.IGNORECASE)


def clean_text(value):
    """Turn strings / lists / missing values into one clean string (no HTML, no extra spaces)."""
    if value is None or value is pd.NA:
        return ''
    if isinstance(value, float) and np.isnan(value):
        return ''
    if isinstance(value, dict):
        return ''
    if isinstance(value, (list, tuple, np.ndarray)):
        value = ' '.join(clean_text(v) for v in value)
    text = html.unescape(str(value))
    text = _TAG_RE.sub(' ', text)
    return _WS_RE.sub(' ', text).strip()


def looks_like_code(value):
    """True if the raw value contains scraped JavaScript/page code."""
    return bool(_CODE_RE.search(str(value)))


def clean_title(value):
    """Normal titles pass through. A title polluted with page code is repaired from the
    'Amazon.com: <name>: ...' part when present, otherwise emptied (the product is then dropped)."""
    text = clean_text(value)
    if looks_like_code(value) or len(text) > 400:
        match = _AMAZON_TITLE_RE.search(text)
        return match.group(1).strip() if match else ''
    return text


def clean_plain(value, max_len=None):
    """Brand / description: empty if it is page code (or absurdly long for a brand)."""
    if looks_like_code(value):
        return ''
    text = clean_text(value)
    return '' if (max_len and len(text) > max_len) else text


def clean_features(value):
    """Keep real bullet points (material, fit, style); drop scraped shipping lines and page code."""
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return ''
    items = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
    kept = []
    for item in items:
        if looks_like_code(item):
            continue
        text = clean_text(item)
        if text and not _JUNK_FEATURE_RE.search(text):
            kept.append(text)
    return ' \u2022 '.join(kept)


def category_text(value):
    """Category lists become 'Clothing > Women > Shoes'."""
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = [clean_text(v) for v in value]
        return ' > '.join(p for p in parts if p)
    return clean_text(value)


def parse_price(value):
    """'$12.99' -> 12.99, '$10 - $20' -> 10.0 (first number), junk -> NaN. Never returns 0."""
    if value is None or value is pd.NA:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        v = float(value)
        return v if np.isfinite(v) and v > 0 else np.nan
    match = _PRICE_RE.search(str(value))
    if not match:
        return np.nan
    try:
        v = float(match.group().replace(',', ''))
    except ValueError:
        return np.nan
    return v if v > 0 else np.nan


def first_valid_url(value):
    """First well-formed http(s) URL in a string or list; http is upgraded to https. Else None."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float):
        return None
    candidates = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
    for c in candidates:
        if not isinstance(c, str):
            continue
        c = c.strip()
        if c.lower().startswith(('http://', 'https://')) and ' ' not in c:
            return 'https://' + c[len('http://'):] if c.lower().startswith('http://') else c
    return None


def clean_ratings(df):
    rep = {'rows_in': int(len(df))}
    df = df.copy()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    n = len(df)
    df = df.dropna(subset=['item_id', 'user_id', 'rating', 'timestamp'])
    rep['dropped_missing_values'] = int(n - len(df))

    for col in ('item_id', 'user_id'):
        df[col] = df[col].astype(str).str.strip()
    n = len(df)
    df = df[(df['item_id'] != '') & (df['user_id'] != '')]
    rep['dropped_empty_ids'] = int(n - len(df))

    n = len(df)
    df = df[df['rating'].between(1, 5)]
    rep['dropped_invalid_rating'] = int(n - len(df))

    df['timestamp'] = df['timestamp'].astype('int64')
    n = len(df)
    df = (df.sort_values('timestamp', kind='mergesort')
            .drop_duplicates(['user_id', 'item_id'], keep='last'))
    rep['dropped_duplicate_user_item'] = int(n - len(df))   # we keep the latest rating
    df = df.reset_index(drop=True)
    rep['rows_out'] = int(len(df))
    return df, rep


def _col(meta, name):
    if name in meta.columns:
        return meta[name]
    return pd.Series([None] * len(meta), index=meta.index, dtype=object)


def clean_metadata(meta):
    if 'asin' not in meta.columns:
        raise ValueError(f'No asin column in metadata. Columns found: {list(meta.columns)}')
    rep = {'rows_in': int(len(meta))}
    meta = meta[meta['asin'].notna()].copy()
    rep['dropped_missing_asin'] = rep['rows_in'] - int(len(meta))
    rep['titles_with_scraped_code'] = int(_col(meta, 'title').map(looks_like_code).sum())

    img = pd.Series([None] * len(meta), index=meta.index, dtype=object)
    for col in IMAGE_COLUMNS:
        if col in meta.columns:
            img = img.where(img.notna(), meta[col].map(first_valid_url))

    cat_source = 'category' if 'category' in meta.columns else 'categories'
    prod = pd.DataFrame({
        'item_id': meta['asin'].astype(str).str.strip(),
        'title': _col(meta, 'title').map(clean_title),
        'brand': _col(meta, 'brand').map(lambda v: clean_plain(v, 100)),
        'features': _col(meta, 'feature').map(clean_features),
        'description': _col(meta, 'description').map(clean_plain),
        'category': _col(meta, cat_source).map(category_text),
        'price': _col(meta, 'price').map(parse_price).astype(float),
        'image_url': img,
    })
    prod = prod[prod['item_id'] != '']

    # Duplicate product rows: keep the most complete one.
    prod['_score'] = ((prod['title'] != '').astype(int) + (prod['brand'] != '').astype(int)
                      + (prod['features'] != '').astype(int) + (prod['description'] != '').astype(int)
                      + prod['price'].notna().astype(int) + prod['image_url'].notna().astype(int))
    n = len(prod)
    prod = prod.sort_values('_score', ascending=False, kind='mergesort').drop_duplicates('item_id')
    rep['dropped_duplicate_asin'] = int(n - len(prod))
    prod = prod.drop(columns='_score')

    n = len(prod)
    prod = prod[prod['title'] != '']
    rep['dropped_no_title'] = int(n - len(prod))

    # Title is repeated so it carries more weight than the long description.
    text = (prod['title'] + ' ' + prod['title'] + ' ' + prod['brand'] + ' ' + prod['category']
            + ' ' + prod['features'] + ' ' + prod['description'])
    prod['text'] = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    prod['rich_text'] = (prod['features'].str.len() + prod['description'].str.len()) >= config.MIN_TEXT_CHARS

    prod = prod.sort_values('item_id').reset_index(drop=True)
    rep['rows_out'] = int(len(prod))
    if len(prod):
        rep['pct_with_price'] = round(float(prod['price'].notna().mean() * 100), 2)
        rep['pct_with_image'] = round(float(prod['image_url'].notna().mean() * 100), 2)
        rep['pct_with_brand'] = round(float((prod['brand'] != '').mean() * 100), 2)
        rep['pct_rich_text'] = round(float(prod['rich_text'].mean() * 100), 2)
    return prod, rep


def kcore_filter(df, u_min, i_min):
    """Repeatedly drop users/items with too few interactions until nothing changes."""
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        keep = (df['user_id'].map(user_counts) >= u_min) & (df['item_id'].map(item_counts) >= i_min)
        if keep.all():
            return df
        df = df[keep]
        if df.empty:
            return df


def join_and_filter(interactions, products, u_min, i_min, max_users=None, seed=config.RANDOM_SEED):
    rep = {'interactions_in': int(len(interactions))}
    inter = interactions[interactions['item_id'].isin(products['item_id'])]
    rep['dropped_no_metadata'] = rep['interactions_in'] - int(len(inter))

    inter = kcore_filter(inter, u_min, i_min)
    rep['after_activity_filter'] = int(len(inter))

    if max_users is not None and len(inter):
        users = sorted(inter['user_id'].unique())
        if len(users) > max_users:
            rng = np.random.default_rng(seed)
            chosen = {users[i] for i in rng.choice(len(users), size=max_users, replace=False)}
            inter = kcore_filter(inter[inter['user_id'].isin(chosen)], u_min, i_min)
            rep['after_user_sampling'] = int(len(inter))

    if inter.empty:
        raise ValueError('No interactions left after filtering. Lower u_min / i_min.')
    products = products[products['item_id'].isin(inter['item_id'])].reset_index(drop=True)
    inter = inter.sort_values(['timestamp', 'user_id', 'item_id'], kind='mergesort').reset_index(drop=True)
    rep.update({
        'final_interactions': int(len(inter)),
        'final_users': int(inter['user_id'].nunique()),
        'final_products': int(len(products)),
        'u_min': u_min, 'i_min': i_min, 'max_users': max_users, 'seed': seed,
    })
    return inter, products, rep


def run_pipeline(ratings_path, meta_path, out_dir, u_min=config.U_MIN, i_min=config.I_MIN,
                 max_users=None, seed=config.RANDOM_SEED):
    """Raw files -> data/processed/{interactions,products}.parquet + cleaning_report.json"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info('Loading ratings ...')
    ratings, r_rep = clean_ratings(load_ratings_csv(ratings_path))
    log.info('Loading metadata ...')
    products, m_rep = clean_metadata(load_metadata(meta_path))
    inter, products, j_rep = join_and_filter(ratings, products, u_min, i_min, max_users, seed)
    save_table(inter, out_dir / 'interactions.parquet')
    save_table(products, out_dir / 'products.parquet')
    report = {'ratings': r_rep, 'metadata': m_rep, 'join_and_filter': j_rep}
    with open(out_dir / 'cleaning_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    log.info('Saved processed data to %s', out_dir)
    return report