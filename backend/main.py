"""FastAPI app: JSON API under /api, and the website (frontend/) served from the same app.

Run from the project root:   uvicorn backend.main:app --reload
or press the Run button on this file in VS Code.
"""
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from src import config
from src.serving import Recommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def create_app(models_dir=None):
    models_dir = Path(models_dir or os.getenv('MODELS_DIR', config.MODELS_DIR))
    rec = Recommender(models_dir)          # loaded once at startup, never per request
    app = FastAPI(title='CartWise AI API', version='1.0.0',
                  description='Hybrid product recommendations on Amazon Fashion reviews.')

    def ready():
        if not rec.ready:
            raise HTTPException(status_code=503, detail=f'Model artifacts unavailable: {rec.error}')

    def check_prices(min_price, max_price):
        if min_price is not None and max_price is not None and min_price > max_price:
            raise HTTPException(status_code=422, detail='min_price cannot be greater than max_price')

    @app.get('/api/health')
    def health():
        return rec.health()

    @app.get('/api/products')
    def products(q: str = Query('', max_length=100), brand: str = Query('', max_length=100),
                 min_price: Optional[float] = Query(None, ge=0), max_price: Optional[float] = Query(None, ge=0),
                 page: int = Query(1, ge=1, le=10000), limit: int = Query(24, ge=1, le=60)):
        ready()
        check_prices(min_price, max_price)
        return rec.search(q, brand, min_price, max_price, page, limit)

    @app.get('/api/products/{product_id}')
    def product(product_id: str):
        ready()
        p = rec.get_product(product_id)
        if p is None:
            raise HTTPException(status_code=404, detail=f'Product {product_id} not found')
        return p

    @app.get('/api/products/{product_id}/similar')
    def similar(product_id: str, k: int = Query(10, ge=1, le=30),
                min_price: Optional[float] = Query(None, ge=0), max_price: Optional[float] = Query(None, ge=0)):
        ready()
        check_prices(min_price, max_price)
        result = rec.similar(product_id, k, min_price, max_price)
        if result is None:
            raise HTTPException(status_code=404, detail=f'Product {product_id} not found')
        return result

    @app.get('/api/customers')
    def customers():
        ready()
        return {'customers': rec.customers}

    @app.get('/api/recommendations/{customer_id}')
    def recommendations(customer_id: str, k: int = Query(10, ge=1, le=30),
                        min_price: Optional[float] = Query(None, ge=0), max_price: Optional[float] = Query(None, ge=0)):
        ready()
        check_prices(min_price, max_price)
        return rec.collaborative(customer_id, k, min_price, max_price)

    @app.get('/api/hybrid/{customer_id}')
    def hybrid(customer_id: str, k: int = Query(10, ge=1, le=30),
               weight: Optional[float] = Query(None, ge=0, le=1),
               min_price: Optional[float] = Query(None, ge=0), max_price: Optional[float] = Query(None, ge=0)):
        ready()
        check_prices(min_price, max_price)
        return rec.hybrid_recs(customer_id, k, weight, min_price, max_price)

    @app.get('/api/metrics')
    def metrics():
        ready()
        return rec.metrics()

    frontend = config.ROOT / 'frontend'
    if frontend.exists():                  # mounted last so /api/... routes win
        app.mount('/', StaticFiles(directory=str(frontend), html=True), name='frontend')
    return app


app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.main:app', reload=True)
