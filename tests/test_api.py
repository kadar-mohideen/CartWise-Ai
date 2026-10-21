"""API tests. Skipped automatically if FastAPI is not installed."""
import tempfile

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import create_app  # noqa: E402
from tests.test_pipeline import build  # noqa: E402


@pytest.fixture(scope='module')
def client():
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp)
        yield TestClient(create_app(tmp))


def test_health(client):
    r = client.get('/api/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'


def test_search_and_validation(client):
    assert client.get('/api/products', params={'q': 'denim'}).json()['total'] > 0
    assert client.get('/api/products', params={'limit': 0}).status_code == 422
    assert client.get('/api/products', params={'min_price': 50, 'max_price': 10}).status_code == 422


def test_product_404_and_similar(client):
    assert client.get('/api/products/NOPE').status_code == 404
    assert client.get('/api/products/NOPE/similar').status_code == 404
    pid = client.get('/api/products', params={'q': 'denim'}).json()['items'][0]['item_id']
    data = client.get(f'/api/products/{pid}/similar', params={'k': 5}).json()
    assert data['items'] and all(i['item_id'] != pid for i in data['items'])


def test_recommendations_and_unknown_customer(client):
    cust = client.get('/api/customers').json()['customers'][0]['customer_id']
    assert client.get(f'/api/hybrid/{cust}').json()['items']
    assert client.get(f'/api/hybrid/{cust}', params={'weight': 2}).status_code == 422
    assert client.get(f'/api/recommendations/{cust}').json()['items']
    ghost = client.get('/api/hybrid/ghost').json()
    assert ghost['message'] and ghost['items']


def test_metrics(client):
    data = client.get('/api/metrics').json()
    assert 'test' in data and 'hybrid' in data['test']


def test_api_reports_missing_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        c = TestClient(create_app(tmp))
        assert c.get('/api/health').json()['status'] == 'degraded'
        assert c.get('/api/products').status_code == 503
