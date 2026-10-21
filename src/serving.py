"""Loads the saved artifacts once and answers website requests. No training, no scikit-learn."""
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .collaborative_filtering import SVDRecommender
from .content_based import ContentBasedRecommender
from .hybrid_recommender import HybridRecommender, describe
from .popularity import PopularityRecommender

log = logging.getLogger(__name__)

_NON_ALNUM = re.compile(r'[^a-z0-9]+')


def _norm_title(title):
    """Lower-case letters and digits only, so colour/size variants with the same title match."""
    return _NON_ALNUM.sub(' ', title.lower()).strip()

REQUIRED_FILES = ['catalog.csv.gz', 'history.csv.gz', 'content_meta.json', 'tfidf_matrix.npz',
                  'cf_factors.npz', 'popularity.json', 'metrics.json', 'config.json']


class Recommender:
    def __init__(self, models_dir):
        self.ready, self.error = False, None
        try:
            self._load(Path(models_dir))
            self.ready = True
        except Exception as exc:                      # the API must still start and report the problem
            self.error = f'{type(exc).__name__}: {exc}'
            log.error('Could not load model artifacts: %s', self.error)

    # ------------------------------------------------------------------ loading
    def _load(self, d):
        missing = [f for f in REQUIRED_FILES if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f'missing files in {d}: {missing}')
        with open(d / 'config.json', encoding='utf-8') as f:
            self.cfg = json.load(f)
        with open(d / 'metrics.json', encoding='utf-8') as f:
            self._metrics = json.load(f)
        self.content = ContentBasedRecommender.load(d)
        self.cf = SVDRecommender.load(d, self.content.item_ids)
        self.pop = PopularityRecommender.load(d)
        self.hybrid = HybridRecommender(self.content, self.cf, self.pop, self.cfg['hybrid_weight'])

        df = pd.read_csv(d / 'catalog.csv.gz', dtype={'item_id': str})
        for col in ['title', 'brand', 'category', 'features', 'description', 'image_url']:
            df[col] = df[col].fillna('').astype(str)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if df['item_id'].tolist() != self.content.item_ids:
            raise ValueError('catalog.csv.gz and the TF-IDF matrix are out of sync. Re-run training.')
        self._df = df
        self._records = df.to_dict('records')
        self._search_text = (df['title'] + ' ' + df['brand'] + ' ' + df['category']).str.lower()
        self._brand_lower = df['brand'].str.lower()
        self._pop_counts = np.array([self.pop.counts.get(i, 0) for i in df['item_id']], dtype=float)

        hist = pd.read_csv(d / 'history.csv.gz', dtype={'user_id': str, 'item_id': str})
        pos = self.cfg['positive_rating']
        self.liked = hist[hist['rating'] >= pos].groupby('user_id')['item_id'].apply(list).to_dict()
        self.seen = hist.groupby('user_id')['item_id'].apply(set).to_dict()
        self.customers = self._pick_customers()

    def _pick_customers(self):
        eligible = sorted(u for u in self.cf.user_index
                          if len(self.seen.get(u, ())) >= config.DEMO_CUSTOMER_MIN_INTERACTIONS)
        if not eligible:
            return []
        rng = np.random.default_rng(config.RANDOM_SEED)
        n = min(config.N_DEMO_CUSTOMERS, len(eligible))
        chosen = [eligible[i] for i in sorted(rng.choice(len(eligible), size=n, replace=False))]
        out = []
        for u in chosen:
            examples = [self._records[self.content.index[i]]['title'] for i in self.liked.get(u, [])[:3]
                        if i in self.content.index]
            out.append({'customer_id': u, 'n_interactions': len(self.seen.get(u, ())),
                        'liked_examples': examples})
        return out

    # ------------------------------------------------------------------ helpers
    def _product(self, i, detail=False):
        r = self._records[i]
        price = None if pd.isna(r['price']) else round(float(r['price']), 2)
        out = {'item_id': r['item_id'], 'title': r['title'], 'brand': r['brand'] or None,
               'category': r['category'] or None, 'price': price, 'has_price': price is not None,
               'image_url': r['image_url'] or None}
        if detail:
            out['description'] = r['description'] or None
            out['features'] = r['features'] or None
        return out

    @staticmethod
    def _price_filter(items, min_price, max_price, k):
        if min_price is None and max_price is None:
            return items[:k], 0
        kept, excluded = [], 0
        for it in items:
            p = it['price']
            if p is None:
                excluded += 1
            elif (min_price is None or p >= min_price) and (max_price is None or p <= max_price):
                kept.append(it)
        return kept[:k], excluded

    @staticmethod
    def _pool(k, min_price, max_price):
        return k * 10 if (min_price is not None or max_price is not None) else k * 5

    # ------------------------------------------------------------------ public API
    def health(self):
        info = {'status': 'ok' if self.ready else 'degraded', 'ready': self.ready, 'error': self.error}
        if self.ready:
            info.update({'products': len(self._records), 'demo_customers': len(self.customers)})
        return info

    def get_product(self, item_id):
        i = self.content.index.get(item_id)
        return None if i is None else self._product(i, detail=True)

    def search(self, query='', brand='', min_price=None, max_price=None, page=1, limit=24):
        mask = np.ones(len(self._df), dtype=bool)
        for token in query.lower().split():
            mask &= self._search_text.str.contains(token, regex=False).to_numpy()
        if brand.strip():
            mask &= (self._brand_lower == brand.strip().lower()).to_numpy()
        excluded = 0
        if min_price is not None or max_price is not None:
            has_price = self._df['price'].notna().to_numpy()
            excluded = int((mask & ~has_price).sum())
            mask &= has_price
            if min_price is not None:
                mask &= (self._df['price'] >= min_price).to_numpy()
            if max_price is not None:
                mask &= (self._df['price'] <= max_price).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-self._pop_counts[idx], kind='stable')]   # popular first
        start = (page - 1) * limit
        return {'total': int(len(order)), 'page': page, 'limit': limit,
                'excluded_missing_price': excluded,
                'items': [self._product(int(i)) for i in order[start:start + limit]]}

    def similar(self, item_id, k=10, min_price=None, max_price=None):
        i = self.content.index.get(item_id)
        if i is None:
            return None
        raw = self.content.similar(item_id, self._pool(k, min_price, max_price) + 20)
        seen_titles = {_norm_title(self._records[i]['title'])}
        items, hidden = [], 0
        for other, score in raw:
            j = self.content.index[other]
            title_key = _norm_title(self._records[j]['title'])
            if title_key in seen_titles:       # same product listed again (other colour or size)
                hidden += 1
                continue
            seen_titles.add(title_key)
            d = self._product(j)
            terms = self.content.explain(item_id, other)
            d['similarity'] = round(score, 4)
            d['shared_terms'] = terms
            d['reason'] = f'Text similarity {score:.0%}' + (f' (shared words: {", ".join(terms)})' if terms else '')
            items.append(d)
        kept, excluded = self._price_filter(items, min_price, max_price, k)
        message = None if kept else 'No similar products found (this product may have too little text, or the budget filter removed them).'
        return {'product': self._product(i, detail=True), 'items': kept,
                'excluded_missing_price': excluded, 'variants_hidden': hidden, 'message': message}

    def _wrap(self, customer_id, entries, k, min_price, max_price, message=None):
        items, seen_titles = [], set()
        for e in entries:
            d = self._product(self.content.index[e['item_id']])
            title_key = _norm_title(d['title'])
            if title_key in seen_titles:       # show each product once
                continue
            seen_titles.add(title_key)
            d.update({'score': round(e['score'], 4), 'source': e['source'],
                      'cf_score': None if e['cf_score'] is None else round(e['cf_score'], 4),
                      'content_score': None if e['content_score'] is None else round(e['content_score'], 4),
                      'reason': e['reason']})
            items.append(d)
        kept, excluded = self._price_filter(items, min_price, max_price, k)
        if not kept and message is None:
            message = 'No recommendations available (try a wider budget).'
        return {'customer_id': customer_id, 'items': kept, 'excluded_missing_price': excluded,
                'message': message}

    def collaborative(self, customer_id, k=10, min_price=None, max_price=None):
        pool = self._pool(k, min_price, max_price)
        if not self.cf.has_user(customer_id):
            entries = [{'item_id': i, 'score': float(self.pop.counts.get(i, 0)), 'cf_score': None,
                        'content_score': None, 'source': 'popularity_fallback',
                        'reason': 'Popular item. This customer has no history in the data.'}
                       for i in self.pop.recommend(customer_id, pool, set())]
            return self._wrap(customer_id, entries, k, min_price, max_price,
                              'Unknown customer: showing popular products instead.')
        entries = [{'item_id': i, 'score': s, 'cf_score': s, 'content_score': None,
                    'source': 'collaborative_only',
                    'reason': f'Customers with similar rating patterns liked this (model score {s:.2f}).'}
                   for i, s in self.cf.recommend_scored(customer_id, pool, self.seen.get(customer_id, set()))]
        return self._wrap(customer_id, entries, k, min_price, max_price)

    def hybrid_recs(self, customer_id, k=10, weight=None, min_price=None, max_price=None):
        pool = self._pool(k, min_price, max_price)
        raw = self.hybrid.recommend(customer_id, self.liked.get(customer_id, []), pool,
                                    self.seen.get(customer_id, set()), weight)
        entries = [dict(e, reason=describe(e)) for e in raw]
        message = None
        if entries and entries[0]['source'] == 'popularity_fallback':
            message = 'Not enough history for this customer: showing popular products instead.'
        return self._wrap(customer_id, entries, k, min_price, max_price, message)

    def metrics(self):
        return self._metrics