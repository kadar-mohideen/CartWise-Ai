import numpy as np

from src.collaborative_filtering import SVDRecommender
from src.content_based import ContentBasedRecommender
from src.hybrid_recommender import HybridRecommender, describe, minmax
from src.popularity import PopularityRecommender
from tests.helpers import make_synthetic


def _fit():
    inter, products = make_synthetic()
    content = ContentBasedRecommender({'min_df': 1, 'ngram_range': (1, 1)}).fit(products)
    cf = SVDRecommender(8).fit(inter, content.item_ids)
    pop = PopularityRecommender().fit(inter, 4)
    return inter, products, content, cf, pop


def test_popularity_excludes_seen():
    inter, _, _, _, pop = _fit()
    recs = pop.recommend('u', 5, {pop.ranked[0]})
    assert pop.ranked[0] not in recs and len(recs) == 5


def test_content_similar_excludes_self_and_is_sorted():
    _, _, content, _, _ = _fit()
    target = content.item_ids[0]
    sims = content.similar(target, 10)
    assert target not in [i for i, _ in sims]
    scores = [s for _, s in sims]
    assert scores == sorted(scores, reverse=True) and all(0 < s <= 1.0001 for s in scores)
    # items 0,4,8,... share the same cluster vocabulary, so the best match should be one of them
    assert content.item_ids.index(sims[0][0]) % 4 == 0
    assert content.explain(target, sims[0][0])           # shared words are reported


def test_content_handles_empty_text():
    inter, products = make_synthetic()
    products = products.copy()
    products.loc[0, 'text'] = ''
    content = ContentBasedRecommender({'min_df': 1}).fit(products)
    assert content.similar(content.item_ids[0], 5) == []  # no text -> no fake similarity


def test_cf_excludes_seen_and_handles_unknown_user():
    inter, _, content, cf, _ = _fit()
    user = inter.user_id.iloc[0]
    seen = set(inter.loc[inter.user_id == user, 'item_id'])
    recs = cf.recommend(user, 10, seen)
    assert len(recs) == 10 and not (set(recs) & seen)
    assert cf.recommend('nobody', 10, set()) == []


def test_minmax_scaling():
    scores = np.array([1.0, 3.0, 5.0, 100.0])
    mask = np.array([True, True, True, False])
    out = minmax(scores, mask)
    assert list(out[:3]) == [0.0, 0.5, 1.0] and out[3] == 0.0
    assert minmax(np.ones(3), np.ones(3, bool)).sum() == 0.0


def test_hybrid_weights_and_cold_start():
    inter, _, content, cf, pop = _fit()
    hyb = HybridRecommender(content, cf, pop, 0.5)
    user = inter.user_id.iloc[0]
    liked = inter[(inter.user_id == user) & (inter.rating >= 4)].item_id.tolist()
    seen = set(inter.loc[inter.user_id == user, 'item_id'])

    only_cf = [r['item_id'] for r in hyb.recommend(user, liked, 5, seen, weight=1.0)]
    only_content = [r['item_id'] for r in hyb.recommend(user, liked, 5, seen, weight=0.0)]
    assert only_cf == cf.recommend(user, 5, seen)
    assert only_content == [i for i, _ in content.recommend_for_history(liked, 5, seen)]

    mixed = hyb.recommend(user, liked, 5, seen, weight=0.5)
    assert mixed and all(r['item_id'] not in seen for r in mixed)
    assert all(0 <= r['cf_score'] <= 1 and 0 <= r['content_score'] <= 1 for r in mixed)
    assert abs(mixed[0]['score'] - (0.5 * mixed[0]['cf_score'] + 0.5 * mixed[0]['content_score'])) < 1e-9
    assert 'similar-customer' in describe(mixed[0])

    cold = hyb.recommend('new-customer', [], 5, set())
    assert cold and all(r['source'] == 'popularity_fallback' for r in cold)

    no_cf_user = hyb.recommend('new-customer', liked, 5, seen)          # history but unknown to CF
    assert no_cf_user[0]['source'] == 'content_only'


def test_hybrid_rejects_bad_weight():
    inter, _, content, cf, pop = _fit()
    try:
        HybridRecommender(content, cf, pop).recommend('u', [], 5, set(), weight=1.5)
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_explain_does_not_repeat_words_inside_phrases():
    import pandas as pd
    products = pd.DataFrame({'item_id': ['a', 'b', 'c'],
                             'text': ['nike flex supreme trainer shoe', 'nike flex supreme trainer shoe green',
                                      'silk scarf floral']})
    content = ContentBasedRecommender({'min_df': 1, 'ngram_range': (1, 2)}).fit(products)
    terms = content.explain('a', 'b', 3)
    assert terms
    for t in terms:
        for u in terms:
            assert not (set(t.split()) < set(u.split())), (t, u)