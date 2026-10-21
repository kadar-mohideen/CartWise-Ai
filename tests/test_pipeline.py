"""End-to-end on synthetic data: train -> save artifacts -> load for serving."""
import tempfile
from pathlib import Path

from src.serving import Recommender
from src.train import run_from_frames
from tests.helpers import make_synthetic

SMALL = dict(svd_grid=[4, 8], weight_grid=[0.0, 0.5, 1.0], max_eval_users=100,
             tfidf_params={'min_df': 1, 'ngram_range': (1, 1)})


def build(models_dir):
    inter, products = make_synthetic()
    return run_from_frames(inter, products, models_dir, **SMALL)


def test_train_saves_artifacts_and_real_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        metrics = build(tmp)
        for name in ['catalog.csv.gz', 'history.csv.gz', 'tfidf_matrix.npz', 'cf_factors.npz',
                     'content_meta.json', 'popularity.json', 'metrics.json', 'config.json']:
            assert (Path(tmp) / name).exists(), name
        for model in ['popularity', 'content', 'collaborative', 'hybrid']:
            assert set(['precision@10', 'recall@10', 'ndcg@10', 'map@10', 'coverage', 'n_users']) <= set(metrics['test'][model])
        assert metrics['eligible_users']['test'] > 0


def test_serving_layer():
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp)
        rec = Recommender(tmp)
        assert rec.ready and rec.health()['status'] == 'ok'

        res = rec.search('denim')
        assert res['total'] > 0 and all('denim' in (p['title'].lower()) for p in res['items'])
        assert rec.search('zzzz-no-match')['total'] == 0

        priced = rec.search('', min_price=0, max_price=1000)
        assert all(p['price'] is not None for p in priced['items']) and priced['excluded_missing_price'] > 0

        pid = res['items'][0]['item_id']
        sim = rec.similar(pid, 5)
        assert sim['items'] and pid not in [i['item_id'] for i in sim['items']]
        assert all('Text similarity' in i['reason'] for i in sim['items'])
        assert rec.similar('NOPE', 5) is None and rec.get_product('NOPE') is None

        cust = rec.customers[0]['customer_id']
        h = rec.hybrid_recs(cust, 5)
        assert h['items'] and not (set(i['item_id'] for i in h['items']) & rec.seen[cust])
        c = rec.collaborative(cust, 5)
        assert c['items'] and c['message'] is None

        unknown = rec.collaborative('ghost', 5)
        assert unknown['items'] and unknown['message'] and unknown['items'][0]['source'] == 'popularity_fallback'
        assert rec.hybrid_recs('ghost', 5)['message']


def test_missing_artifacts_do_not_crash():
    with tempfile.TemporaryDirectory() as tmp:
        rec = Recommender(tmp)
        assert not rec.ready and rec.health()['status'] == 'degraded' and 'missing files' in rec.error


def test_similar_hides_duplicate_titles():
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp)
        rec = Recommender(tmp)
        for item in rec.search('', limit=40)['items']:
            res = rec.similar(item['item_id'], 8)
            titles = [i['title'].lower() for i in res['items']]
            assert len(titles) == len(set(titles))
            assert item['title'].lower() not in titles