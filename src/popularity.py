"""Baseline: recommend what most people liked. Every other model must beat this to be useful."""
import json
from pathlib import Path


class PopularityRecommender:
    def fit(self, train, positive_rating):
        pos = train[train['rating'] >= positive_rating]
        counts = pos['item_id'].value_counts()
        if counts.empty:                       # nobody liked anything: fall back to all interactions
            counts = train['item_id'].value_counts()
        self.ranked = counts.index.tolist()
        self.counts = {k: int(v) for k, v in counts.items()}
        return self

    def recommend(self, user_id, k, exclude=None):
        exclude = exclude or set()
        out = []
        for item in self.ranked:
            if item in exclude:
                continue
            out.append(item)
            if len(out) >= k:
                break
        return out

    def save(self, directory):
        with open(Path(directory) / 'popularity.json', 'w', encoding='utf-8') as f:
            json.dump({'ranked': self.ranked, 'counts': self.counts}, f)

    @classmethod
    def load(cls, directory):
        with open(Path(directory) / 'popularity.json', encoding='utf-8') as f:
            data = json.load(f)
        obj = cls()
        obj.ranked, obj.counts = data['ranked'], data['counts']
        return obj
