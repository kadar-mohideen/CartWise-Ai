from src.splitting import time_split
from tests.helpers import make_synthetic


def test_split_has_no_time_leakage():
    inter, _ = make_synthetic()
    train, val, test = time_split(inter)
    assert len(train) + len(val) + len(test) == len(inter)
    assert train.timestamp.max() <= val.timestamp.min()
    assert val.timestamp.max() <= test.timestamp.min()
    assert abs(len(train) / len(inter) - 0.70) < 0.03
