import math

from src.evaluation import (average_precision_at_k, build_eval_cases, catalog_coverage,
                            evaluate_recommender, ndcg_at_k, precision_at_k, recall_at_k)
import pandas as pd


def test_precision_example():
    assert precision_at_k(['a', 'b', 'c', 'd', 'e'], {'a', 'd', 'z'}, 5) == 0.4


def test_recall_example():
    assert abs(recall_at_k(['a', 'b', 'c', 'd', 'e'], {'a', 'd', 'z'}, 5) - 2 / 3) < 1e-9


def test_ndcg_example():
    # hits at ranks 1 and 4, two relevant items -> 1.431 / 1.631 = 0.877
    value = ndcg_at_k(['a', 'x', 'y', 'b', 'z'], {'a', 'b'}, 5)
    assert abs(value - (1 + 1 / math.log2(5)) / (1 + 1 / math.log2(3))) < 1e-9
    assert round(value, 3) == 0.877


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k(['a', 'b'], {'a', 'b'}, 5) == 1.0
    assert ndcg_at_k(['x', 'y'], {'a'}, 5) == 0.0
    assert recall_at_k(['a'], set(), 5) == 0.0


def test_average_precision():
    # hits at rank 1 and 3 -> (1/1 + 2/3) / 2
    assert abs(average_precision_at_k(['a', 'x', 'b'], {'a', 'b'}, 3) - (1 + 2 / 3) / 2) < 1e-9


def test_coverage():
    assert catalog_coverage(['a', 'b', 'a', 'c'], 10) == 0.3


def test_eval_cases_rules():
    fit = pd.DataFrame({'user_id': ['u1', 'u1', 'u2'], 'item_id': ['a', 'b', 'a'], 'rating': [5, 5, 2]})
    ev = pd.DataFrame({'user_id': ['u1', 'u1', 'u2', 'u3'], 'item_id': ['c', 'z', 'c', 'c'], 'rating': [5, 5, 5, 5]})
    cases = build_eval_cases(fit, ev, ['a', 'b', 'c'], 4)
    # u1 qualifies (item z is outside the catalog); u2 never liked anything in train; u3 has no history
    assert cases == {'u1': {'c'}}


def test_evaluate_recommender_runs():
    cases = {'u1': {'c'}}
    res = evaluate_recommender(lambda u, k, s: ['c', 'd'], cases, {'u1': {'a'}}, 2, 4)
    assert res['precision@2'] == 0.5 and res['recall@2'] == 1.0 and res['coverage'] == 0.5 and res['n_users'] == 1
