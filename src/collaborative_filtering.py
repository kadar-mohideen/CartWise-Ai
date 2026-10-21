"""Collaborative filtering with Truncated SVD on a sparse user-item matrix.

Rows = users, columns = items, value = 1 if the user liked the item (rating >= 4).
Empty cells are zeros, so this model learns WHO LIKED WHAT, not exact star ratings.
Scores are computed one user at a time; the full reconstructed matrix is never built.
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import config
from .content_based import top_indices


class SVDRecommender:
    def __init__(self, n_components=20, positive_only=True, positive_rating=config.POSITIVE_RATING,
                 seed=config.RANDOM_SEED):
        self.n_components = n_components
        self.positive_only = positive_only
        self.positive_rating = positive_rating
        self.seed = seed

    def fit(self, train, item_ids):
        from sklearn.decomposition import TruncatedSVD
        self.item_ids = list(item_ids)
        self.item_index = {it: i for i, it in enumerate(self.item_ids)}
        df = train[train['item_id'].isin(self.item_index)]
        if self.positive_only:
            df = df[df['rating'] >= self.positive_rating]
            values = np.ones(len(df), dtype=np.float32)
        else:
            values = df['rating'].to_numpy(dtype=np.float32)
        users = sorted(df['user_id'].unique())
        self.user_index = {u: i for i, u in enumerate(users)}
        rows = df['user_id'].map(self.user_index).to_numpy()
        cols = df['item_id'].map(self.item_index).to_numpy()
        matrix = sp.csr_matrix((values, (rows, cols)), shape=(len(users), len(self.item_ids)),
                               dtype=np.float32)
        k = max(1, min(self.n_components, min(matrix.shape) - 1))
        svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self.user_factors = svd.fit_transform(matrix).astype(np.float32)
        self.item_factors = svd.components_.T.astype(np.float32)
        return self

    def has_user(self, user_id):
        return user_id in self.user_index

    def scores(self, user_id):
        """Score for every catalog item for one user (unknown user -> KeyError)."""
        return self.item_factors @ self.user_factors[self.user_index[user_id]]

    def recommend_scored(self, user_id, k, exclude=None):
        if not self.has_user(user_id):
            return []
        scores = self.scores(user_id).astype(np.float64)
        idx = [self.item_index[i] for i in (exclude or ()) if i in self.item_index]
        if idx:
            scores[idx] = -np.inf
        return [(self.item_ids[j], float(scores[j])) for j in top_indices(scores, k)
                if np.isfinite(scores[j])]

    def recommend(self, user_id, k, exclude=None):
        return [item for item, _ in self.recommend_scored(user_id, k, exclude)]

    def save(self, directory):
        np.savez_compressed(Path(directory) / 'cf_factors.npz',
                            user_factors=self.user_factors, item_factors=self.item_factors,
                            user_ids=np.array(list(self.user_index), dtype=str))

    @classmethod
    def load(cls, directory, item_ids):
        data = np.load(Path(directory) / 'cf_factors.npz', allow_pickle=False)
        obj = cls()
        obj.item_ids = list(item_ids)
        obj.item_index = {it: i for i, it in enumerate(obj.item_ids)}
        obj.user_factors = data['user_factors']
        obj.item_factors = data['item_factors']
        obj.user_index = {str(u): i for i, u in enumerate(data['user_ids'])}
        if obj.item_factors.shape[0] != len(obj.item_ids):
            raise ValueError('CF factors and catalog do not match. Re-run training.')
        return obj
