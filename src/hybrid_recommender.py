"""Hybrid = weighted mix of the content score and the collaborative score."""
import numpy as np

from .content_based import top_indices


def minmax(scores, mask):
    """Scale scores to 0..1 using only eligible items (mask=True). Flat scores -> all zeros."""
    out = np.zeros(scores.shape, dtype=np.float64)
    if mask.any():
        vals = scores[mask]
        lo, hi = vals.min(), vals.max()
        if hi > lo:
            out[mask] = (vals - lo) / (hi - lo)
    return out


class HybridRecommender:
    """final = w * collaborative + (1 - w) * content   (both scaled to 0..1 per customer)."""

    def __init__(self, content, cf, popularity, weight=0.5):
        if content.item_ids != cf.item_ids:
            raise ValueError('Content and collaborative models must use the same item order.')
        self.content, self.cf, self.popularity = content, cf, popularity
        self.weight = weight

    def _popular(self, user_id, k, exclude):
        return [{'item_id': i, 'score': float(self.popularity.counts.get(i, 0)), 'cf_score': None,
                 'content_score': None, 'weight_used': None, 'source': 'popularity_fallback'}
                for i in self.popularity.recommend(user_id, k, exclude)]

    def recommend(self, user_id, history_items, k, exclude=None, weight=None):
        w = self.weight if weight is None else weight
        if not 0.0 <= w <= 1.0:
            raise ValueError('weight must be between 0 and 1')
        exclude = set(exclude or ())
        n = len(self.content.item_ids)
        mask = np.ones(n, dtype=bool)
        idx = [self.content.index[i] for i in exclude if i in self.content.index]
        if idx:
            mask[idx] = False

        profile = self.content.profile(history_items)
        has_cf = self.cf.has_user(user_id)
        if profile is None and not has_cf:                      # cold start
            return self._popular(user_id, k, exclude)

        content_n = minmax(self.content.scores_for_profile(profile), mask) if profile is not None else np.zeros(n)
        cf_n = minmax(self.cf.scores(user_id).astype(np.float64), mask) if has_cf else np.zeros(n)
        if profile is None:
            w_eff, source = 1.0, 'collaborative_only'
        elif not has_cf:
            w_eff, source = 0.0, 'content_only'
        else:
            w_eff, source = w, 'hybrid'

        final = w_eff * cf_n + (1.0 - w_eff) * content_n
        final[~mask] = -np.inf
        out = [{'item_id': self.content.item_ids[j], 'score': float(final[j]),
                'cf_score': float(cf_n[j]), 'content_score': float(content_n[j]),
                'weight_used': w_eff, 'source': source}
               for j in top_indices(final, k) if final[j] > 0]
        return out or self._popular(user_id, k, exclude)


def describe(entry):
    """Plain-English reason built only from the numbers the system calculated."""
    src = entry['source']
    if src == 'hybrid':
        return (f"Combined score {entry['score']:.2f}: similar-customer signal {entry['cf_score']:.2f}, "
                f"text-similarity signal {entry['content_score']:.2f} (collaborative weight {entry['weight_used']:.1f}).")
    if src == 'content_only':
        return f"Similar in text to items you liked (signal {entry['content_score']:.2f}). No purchase-pattern data for this customer."
    if src == 'collaborative_only':
        return f"Customers with similar tastes liked this (signal {entry['cf_score']:.2f})."
    return 'Popular item. There was not enough history for this customer.'
