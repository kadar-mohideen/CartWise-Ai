"""Train, tune on validation, evaluate on test, and save artifacts for the website.

Run from the project root:   python -m src.train

Protocol (no test-set peeking):
  1. Fit on TRAIN, choose SVD size and hybrid weight on VALIDATION.
  2. Refit on TRAIN+VALIDATION with those choices, report scores on TEST (used once).
  3. Refit on ALL data with the same choices, save that model for the website.
"""
import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from . import config
from .collaborative_filtering import SVDRecommender
from .content_based import ContentBasedRecommender
from .data_loader import load_table
from .evaluation import build_eval_cases, evaluate_recommender
from .hybrid_recommender import HybridRecommender
from .popularity import PopularityRecommender
from .splitting import time_split

log = logging.getLogger(__name__)


def _histories(df, positive_rating):
    liked = df[df['rating'] >= positive_rating].groupby('user_id')['item_id'].apply(list).to_dict()
    seen = df.groupby('user_id')['item_id'].apply(set).to_dict()
    return liked, seen


def _fit_content_pop(fit_df, products, positive_rating, tfidf_params):
    prod = products[products['item_id'].isin(set(fit_df['item_id']))].reset_index(drop=True)
    return (ContentBasedRecommender(tfidf_params).fit(prod),
            PopularityRecommender().fit(fit_df, positive_rating))


def _model_fns(content, cf, pop, hybrid, liked, weight=None):
    return {
        'popularity': lambda u, k, seen: pop.recommend(u, k, seen),
        'content': lambda u, k, seen: [i for i, _ in content.recommend_for_history(liked.get(u, []), k, seen)],
        'collaborative': lambda u, k, seen: cf.recommend(u, k, seen),
        'hybrid': lambda u, k, seen: [r['item_id'] for r in hybrid.recommend(u, liked.get(u, []), k, seen, weight)],
    }


def run_from_frames(inter, products, models_dir, k=config.K, svd_grid=None, weight_grid=None,
                    max_eval_users=config.MAX_EVAL_USERS, positive_rating=config.POSITIVE_RATING,
                    tfidf_params=None, seed=config.RANDOM_SEED):
    t0 = time.time()
    svd_grid = svd_grid or config.SVD_COMPONENTS_GRID
    weight_grid = weight_grid or config.HYBRID_WEIGHT_GRID
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = time_split(inter)
    log.info('Split rows: train=%d val=%d test=%d', len(train), len(val), len(test))

    # ---- 1. tune on validation ------------------------------------------------------
    content, pop = _fit_content_pop(train, products, positive_rating, tfidf_params)
    catalog = content.item_ids
    liked, _ = _histories(train, positive_rating)
    cases = build_eval_cases(train, val, catalog, positive_rating)
    _, seen = _histories(train, positive_rating)
    log.info('Validation: %d eligible users, catalog %d items', len(cases), len(catalog))

    def ev(fn):
        return evaluate_recommender(fn, cases, seen, k, len(catalog), max_eval_users, seed)

    ndcg_key = f'ndcg@{k}'
    cf_grid, best_cf, best_n, best_score = {}, None, None, -1.0
    for n in svd_grid:
        cf = SVDRecommender(n, positive_rating=positive_rating, seed=seed).fit(train, catalog)
        res = ev(lambda u, kk, s, cf=cf: cf.recommend(u, kk, s))
        cf_grid[str(n)] = res
        log.info('SVD components=%s -> %s=%.4f', n, ndcg_key, res[ndcg_key])
        if res[ndcg_key] > best_score:
            best_cf, best_n, best_score = cf, n, res[ndcg_key]

    hybrid = HybridRecommender(content, best_cf, pop)
    hybrid_grid, best_w, best_score = {}, None, -1.0
    for w in weight_grid:
        res = ev(lambda u, kk, s, w=w: [r['item_id'] for r in hybrid.recommend(u, liked.get(u, []), kk, s, w)])
        hybrid_grid[str(w)] = res
        log.info('Hybrid weight=%s -> %s=%.4f', w, ndcg_key, res[ndcg_key])
        if res[ndcg_key] > best_score:
            best_w, best_score = w, res[ndcg_key]

    fns = _model_fns(content, best_cf, pop, hybrid, liked, best_w)
    validation = {name: ev(fn) for name, fn in fns.items()
                  if name not in ('collaborative', 'hybrid')}
    validation['collaborative'] = cf_grid[str(best_n)]
    validation['hybrid'] = hybrid_grid[str(best_w)]

    # ---- 2. final test scores (train + validation fit) ------------------------------
    fit2 = pd.concat([train, val], ignore_index=True)
    content2, pop2 = _fit_content_pop(fit2, products, positive_rating, tfidf_params)
    cf2 = SVDRecommender(best_n, positive_rating=positive_rating, seed=seed).fit(fit2, content2.item_ids)
    hybrid2 = HybridRecommender(content2, cf2, pop2, best_w)
    liked2, seen2 = _histories(fit2, positive_rating)
    cases2 = build_eval_cases(fit2, test, content2.item_ids, positive_rating)
    log.info('Test: %d eligible users', len(cases2))
    test_res = {name: evaluate_recommender(fn, cases2, seen2, k, len(content2.item_ids), max_eval_users, seed)
                for name, fn in _model_fns(content2, cf2, pop2, hybrid2, liked2, best_w).items()}

    # ---- 3. final model for the website (all data) ----------------------------------
    content3, pop3 = _fit_content_pop(inter, products, positive_rating, tfidf_params)
    cf3 = SVDRecommender(best_n, positive_rating=positive_rating, seed=seed).fit(inter, content3.item_ids)
    content3.save(models_dir)
    cf3.save(models_dir)
    pop3.save(models_dir)

    cat = products.set_index('item_id').loc[content3.item_ids].reset_index()
    cat['description'] = cat['description'].str.slice(0, config.CATALOG_DESC_CHARS)
    cat['features'] = cat['features'].str.slice(0, config.CATALOG_FEATURE_CHARS)
    cat[['item_id', 'title', 'brand', 'category', 'features', 'description', 'price', 'image_url']] \
        .to_csv(models_dir / 'catalog.csv.gz', index=False)
    inter[['user_id', 'item_id', 'rating', 'timestamp']].to_csv(models_dir / 'history.csv.gz', index=False)

    metrics = {
        'k': k,
        'positive_rating': positive_rating,
        'rows': {'train': len(train), 'validation': len(val), 'test': len(test)},
        'catalog_items_validation': len(catalog),
        'catalog_items_test': len(content2.item_ids),
        'eligible_users': {'validation': len(cases), 'test': len(cases2)},
        'chosen': {'svd_components': best_n, 'hybrid_cf_weight': best_w},
        'validation': validation,
        'validation_svd_grid': cf_grid,
        'validation_hybrid_grid': hybrid_grid,
        'test': test_res,
        'notes': [
            'Test scores come from models fit on train+validation; the website model is refit on all data with the same settings.',
            'Only users with training history and at least one new liked item in the evaluation period are scored.',
        ],
    }
    with open(models_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    with open(models_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump({'hybrid_weight': best_w, 'positive_rating': positive_rating, 'k': k,
                   'svd_components': best_n}, f, indent=2)
    log.info('Done in %.1fs. Artifacts saved to %s', time.time() - t0, models_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train CartWise AI models')
    parser.add_argument('--processed', default=str(config.DATA_PROCESSED))
    parser.add_argument('--models', default=str(config.MODELS_DIR))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    processed = Path(args.processed)
    inter = load_table(processed / 'interactions.parquet')
    products = load_table(processed / 'products.parquet')
    metrics = run_from_frames(inter, products, args.models)
    print(json.dumps({'chosen': metrics['chosen'], 'test': metrics['test']}, indent=2))


if __name__ == '__main__':
    main()
