"""Content-based filtering: TF-IDF text vectors + cosine similarity.

scikit-learn is imported only inside fit(), so the deployed website does not need it.
"""
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import config


def top_indices(scores, k):
    """Indices of the k largest scores, best first."""
    k = min(k, scores.size)
    if k <= 0:
        return np.array([], dtype=int)
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top], kind='stable')]


class ContentBasedRecommender:
    def __init__(self, tfidf_params=None):
        self.tfidf_params = dict(config.TFIDF_PARAMS if tfidf_params is None else tfidf_params)

    def fit(self, products, text_col='text'):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(**self.tfidf_params)
        matrix = vectorizer.fit_transform(products[text_col].fillna('').tolist()).tocsr()
        self._set(products['item_id'].tolist(), matrix, vectorizer.get_feature_names_out())
        return self

    def _set(self, item_ids, matrix, terms):
        self.item_ids = list(item_ids)
        self.index = {it: i for i, it in enumerate(self.item_ids)}
        self.matrix = matrix.tocsr()           # rows are L2-normalised, so dot product = cosine
        self.terms = np.asarray(terms)

    def similar(self, item_id, k=10):
        """[(item_id, cosine_similarity)] most similar first; the item itself is excluded."""
        i = self.index.get(item_id)
        if i is None:
            raise KeyError(item_id)
        scores = np.asarray((self.matrix @ self.matrix[i].T).todense()).ravel()
        scores[i] = -np.inf
        return [(self.item_ids[j], float(scores[j])) for j in top_indices(scores, k) if scores[j] > 0]

    def explain(self, item_a, item_b, n=3):
        """Words that contribute most to the similarity of two products."""
        common = self.matrix[self.index[item_a]].multiply(self.matrix[self.index[item_b]]).tocoo()
        if common.nnz == 0:
            return []
        order = np.argsort(-common.data)[:max(n * 4, 8)]
        terms = [str(self.terms[common.col[j]]) for j in order]
        # drop a word when a longer phrase already contains it ('supreme' vs 'supreme tr')
        kept = [t for t in terms if not any(set(t.split()) < set(u.split()) for u in terms)]
        return kept[:n]

    def profile(self, history_items):
        """Average text vector of the items a customer liked. None if nothing is usable."""
        rows = [self.index[i] for i in history_items if i in self.index]
        if not rows:
            return None
        vec = np.asarray(self.matrix[rows].mean(axis=0)).ravel()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    def scores_for_profile(self, profile):
        return np.asarray(self.matrix @ profile).ravel()

    def recommend_for_history(self, history_items, k, exclude=None):
        profile = self.profile(history_items)
        if profile is None:
            return []
        scores = self.scores_for_profile(profile)
        idx = [self.index[i] for i in (exclude or ()) if i in self.index]
        if idx:
            scores[idx] = -np.inf
        return [(self.item_ids[j], float(scores[j])) for j in top_indices(scores, k) if scores[j] > 0]

    def save(self, directory):
        directory = Path(directory)
        sp.save_npz(directory / 'tfidf_matrix.npz', self.matrix)
        with open(directory / 'content_meta.json', 'w', encoding='utf-8') as f:
            json.dump({'item_ids': self.item_ids, 'terms': [str(t) for t in self.terms]}, f)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        matrix = sp.load_npz(directory / 'tfidf_matrix.npz')
        with open(directory / 'content_meta.json', encoding='utf-8') as f:
            meta = json.load(f)
        obj = cls()
        obj._set(meta['item_ids'], matrix, meta['terms'])
        return obj