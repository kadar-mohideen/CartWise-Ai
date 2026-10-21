import numpy as np
import pandas as pd

from src.preprocessing import (clean_features, clean_metadata, clean_ratings, clean_text, first_valid_url,
                               join_and_filter, kcore_filter, parse_price)


def test_parse_price():
    assert parse_price('$12.99') == 12.99
    assert parse_price('$10.00 - $20.00') == 10.0
    assert parse_price('$1,299.50') == 1299.5
    assert np.isnan(parse_price('free')) and np.isnan(parse_price(None)) and np.isnan(parse_price(''))
    assert np.isnan(parse_price(0)) and np.isnan(parse_price(float('nan')))


def test_first_valid_url():
    assert first_valid_url(['bad', 'http://x.com/a.jpg']) == 'https://x.com/a.jpg'
    assert first_valid_url('https://x.com/a.jpg') == 'https://x.com/a.jpg'
    assert first_valid_url('not a url') is None and first_valid_url(None) is None and first_valid_url([]) is None


def test_clean_text():
    assert clean_text('<b>Nice</b>   shirt &amp; tie') == 'Nice shirt & tie'
    assert clean_text(['a', None, 'b']) == 'a b'
    assert clean_text(float('nan')) == ''


def test_clean_ratings_rules():
    df = pd.DataFrame({
        'item_id': ['i1', 'i1', 'i2', 'i3', None, 'i4'],
        'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u3'],
        'rating': [3, 5, 9, 4, 4, 'x'],
        'timestamp': [10, 20, 30, 40, 50, 60],
    })
    out, rep = clean_ratings(df)
    assert len(out) == 2                                    # u1-i1 (latest kept) and u2-i3
    assert out.loc[(out.user_id == 'u1') & (out.item_id == 'i1'), 'rating'].iloc[0] == 5
    assert rep['dropped_invalid_rating'] == 1 and rep['dropped_duplicate_user_item'] == 1
    assert not out.duplicated(['user_id', 'item_id']).any()


def test_clean_metadata_handles_messy_rows():
    meta = pd.DataFrame({
        'asin': ['A1', 'A1', 'A2', None, 'A3'],
        'title': ['Blue Jeans', 'Blue Jeans', '', 'Ghost', 'Red Scarf'],
        'brand': ['Levi', None, 'X', 'Y', 'Silk Co'],
        'feature': [['slim', 'fit'], None, None, None, ['soft']],
        'description': [None, ['<p>Classic</p>'], None, None, 'Warm scarf for winter days'],
        'price': ['$25.00', None, '$5', '$9', 'n/a'],
        'imageURL': [['http://img/1.jpg'], [], None, None, ['bad']],
    })
    prod, rep = clean_metadata(meta)
    assert sorted(prod.item_id) == ['A1', 'A3']             # no title (A2) and missing asin dropped
    a1 = prod[prod.item_id == 'A1'].iloc[0]
    assert a1.price == 25.0 and a1.image_url == 'https://img/1.jpg' and a1.brand == 'Levi'
    a3 = prod[prod.item_id == 'A3'].iloc[0]
    assert np.isnan(a3.price) and pd.isna(a3.image_url)    # missing stays missing, never invented
    assert rep['dropped_duplicate_asin'] == 1 and rep['dropped_no_title'] == 1


def test_kcore_and_join():
    inter = pd.DataFrame({'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u3'],
                          'item_id': ['a', 'b', 'c', 'a', 'b', 'a'],
                          'rating': [5] * 6, 'timestamp': range(6)})
    core = kcore_filter(inter, 2, 2)
    assert set(core.user_id) == {'u1', 'u2'} and set(core.item_id) == {'a', 'b'}
    products = pd.DataFrame({'item_id': ['a', 'b']})        # c has no metadata
    out, _, rep = join_and_filter(inter, products, 2, 2)
    assert set(out.item_id) == {'a', 'b'} and rep['dropped_no_metadata'] == 1


def test_clean_features_drops_scraped_shipping_lines():
    raw = ['Package Dimensions:\n  8 x 5.8 x 2.1 inches, Shipping Weight: 4 ounces', '100% cotton', 'Machine wash',
           'ASIN: B01G43V2GM', 'Date First Available: May 1, 2016']
    assert clean_features(raw) == '100% cotton \u2022 Machine wash'
    assert clean_features(None) == '' and clean_features(float('nan')) == '' and clean_features([]) == ''


def test_clean_title_repairs_scraped_page_code():
    from src.preprocessing import clean_plain, clean_title
    junk = ("var aPageStart = (new Date()).getTime(); function(d){var e=d.ue||{}} "
            "Amazon.com: NIKE Women's Flex Supreme TR 4 Cross Trainer: Nike: Shoes // ==============")
    assert clean_title(junk) == "NIKE Women's Flex Supreme TR 4 Cross Trainer"
    assert clean_title('var x = 1; function(){} nothing useful here') == ''
    assert clean_title('Plain Blue Jeans') == 'Plain Blue Jeans'
    assert clean_plain('window.ue_csm = 1; function(a){}') == '' and clean_plain('Soft cotton') == 'Soft cotton'
    assert clean_plain('x' * 150, 100) == ''


def test_clean_features_skips_code_and_uses_bullets():
    raw = ['100% cotton', 'var a = 1; function(x){}', 'Machine wash']
    assert clean_features(raw) == '100% cotton \u2022 Machine wash'