"""Ranking metrics and the evaluation loop. Pure functions, easy to test by hand."""
import math

import numpy as np


def precision_at_k(recs, relevant, k):
    """Share of the top-k recommendations that are relevant."""
    if k <= 0:
        raise ValueError('k must be positive')
    return sum(1 for r in recs[:k] if r in relevant) / k


def recall_at_k(recs, relevant, k):
    """Share of all relevant items that appear in the top-k."""
    if not relevant:
        return 0.0
    return sum(1 for r in recs[:k] if r in relevant) / len(relevant)


def average_precision_at_k(recs, relevant, k):
    hits, total = 0, 0.0
    for i, r in enumerate(recs[:k], start=1):
        if r in relevant:
            hits += 1
            total += hits / i
    denom = min(len(relevant), k)
    return total / denom if denom else 0.0


def ndcg_at_k(recs, relevant, k):
    """Like precision but hits near the top count more."""
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(recs[:k], start=1) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def catalog_coverage(all_recs, catalog_size):
    """Share of the catalog that was recommended to at least one user."""
    if catalog_size <= 0:
        return 0.0
    return len(set(all_recs)) / catalog_size


def build_eval_cases(fit_df, eval_df, catalog_items, positive_rating):
    """user_id -> set of relevant items.

    Relevant = item the user liked (rating >= positive_rating) in the evaluation period.
    A user is evaluated only if they liked something in the training data (so they are not
    a cold-start user) and at least one relevant item is in the catalog and not already seen.
    """
    catalog = set(catalog_items)
    train_pos_users = set(fit_df.loc[fit_df['rating'] >= positive_rating, 'user_id'])
    ev = eval_df[(eval_df['rating'] >= positive_rating)
                 & eval_df['item_id'].isin(catalog)
                 & eval_df['user_id'].isin(train_pos_users)]
    seen = fit_df.groupby('user_id')['item_id'].apply(set).to_dict()
    cases = {}
    for user, items in ev.groupby('user_id')['item_id'].apply(set).items():
        rel = items - seen.get(user, set())
        if rel:
            cases[user] = rel
    return cases


def evaluate_recommender(recommend_fn, cases, seen_by_user, k, catalog_size, max_users=None, seed=42):
    """recommend_fn(user_id, k, exclude_set) -> list of item ids."""
    users = sorted(cases)
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = [users[i] for i in sorted(rng.choice(len(users), size=max_users, replace=False))]
    sums = {'precision': 0.0, 'recall': 0.0, 'map': 0.0, 'ndcg': 0.0}
    all_recs = []
    for u in users:
        recs = list(recommend_fn(u, k, seen_by_user.get(u, set())))[:k]
        rel = cases[u]
        sums['precision'] += precision_at_k(recs, rel, k)
        sums['recall'] += recall_at_k(recs, rel, k)
        sums['map'] += average_precision_at_k(recs, rel, k)
        sums['ndcg'] += ndcg_at_k(recs, rel, k)
        all_recs.extend(recs)
    n = len(users)
    out = {f'{name}@{k}': (v / n if n else 0.0) for name, v in sums.items()}
    out['coverage'] = catalog_coverage(all_recs, catalog_size)
    out['n_users'] = n
    return out
