# CartWise AI

Hybrid fashion product recommendation engine built on the UCSD Amazon Fashion (2018) review data.
It combines content-based filtering (TF-IDF + cosine similarity), collaborative filtering (Truncated SVD), a popularity baseline and a
weighted hybrid. A FastAPI backend serves the models, and a custom HTML/CSS/JavaScript shop front shows the results with explanations.

## Dataset and attribution
- Source: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html (Amazon Fashion: ratings-only CSV + product metadata JSON).
- Please cite: Jianmo Ni, Jiacheng Li, Julian McAuley. *Justifying recommendations using distantly-labeled reviews and fine-grained aspects.* EMNLP 2019.
- The page asks for citation and states no other license terms that I could find. Use it for study and research, and do not publish the raw files. They are excluded from Git.

## Data used
- Raw ratings: 883,636 rows, 749,233 users, 186,189 products. About 87% of users have only one rating.
- Raw metadata: 186,637 product rows (443 duplicate product IDs). Fill rates in the raw file: title 100%, brand 75%, image 70.7%, bullet points 66.4%, price 9.5%, description 8.5%. There is no category field.
- Cleaning (`src/preprocessing.py`): duplicates and invalid ratings removed, prices parsed (missing stays missing), image URLs checked, scraped shipping lines and page code removed from text, then only users and products with at least 2 interactions kept (`U_MIN = 2`, `I_MIN = 2`).
- Final data: 136,688 interactions (`<users>` users and `<products>` products: copy these two numbers from `data/processed/cleaning_report.json`).
- Time-based split: train 95,809, validation 20,473, test 20,406 interactions.
- Chosen on validation data only: 100 SVD factors, collaborative weight 0.5.

## Run order
Heavy data preparation runs once in Google Colab. Everything else runs on your laptop.

1. **Colab:** open `colab/colab_cells.txt`, paste the cells into a new Colab notebook, and upload the `src` folder to `MyDrive/CartWiseAI/src`. Run the cells (Cell 7 with `u_min=2, i_min=2`). Download `interactions.parquet`, `products.parquet` and `cleaning_report.json` from Drive into `data/processed/`.
2. **Laptop (VS Code terminal, project root, virtual environment active):**
   ```powershell
   pip install -r requirements.txt
   pytest -q                      # unit tests on small synthetic data
   python -m src.train            # tune, evaluate, save artifacts into models/
   uvicorn backend.main:app --reload
   ```
   Open http://127.0.0.1:8000 (website) and http://127.0.0.1:8000/docs (API).
3. Open `notebooks/01_eda.ipynb` (kernel: your `.venv`) and run all cells.

## How it works
| Part | File | Idea |
|---|---|---|
| Cleaning | `src/preprocessing.py` | Dedupe, validate ratings, parse prices (missing stays missing), clean image URLs, remove scraped junk, join, activity filter |
| Split | `src/splitting.py` | Oldest 70% train, next 15% validation, newest 15% test |
| Baseline | `src/popularity.py` | Most-liked items first |
| Content | `src/content_based.py` | TF-IDF vectors, cosine similarity, shared words as explanation |
| Collaborative | `src/collaborative_filtering.py` | Sparse user-item matrix, Truncated SVD, one user scored at a time |
| Hybrid | `src/hybrid_recommender.py` | `w * collaborative + (1 - w) * content`, per-customer 0-1 scaling, popularity fallback for cold start |
| Evaluation | `src/evaluation.py` | Precision@K, Recall@K, MAP@K, NDCG@K, catalog coverage |
| Training | `src/train.py` | Tune on validation, score once on test, refit on all data for the website |
| Serving | `src/serving.py`, `backend/main.py` | Artifacts loaded once at startup; one listing per title is shown |

"Liked" means rating >= 4 (`POSITIVE_RATING` in `src/config.py`). Only customers with training history and at least one new liked item in the evaluation period are scored.

## API
| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Status, and whether model artifacts loaded |
| `GET /api/products?q=&brand=&min_price=&max_price=&page=&limit=` | Search the catalog |
| `GET /api/products/{id}` | One product |
| `GET /api/products/{id}/similar?k=` | Content-based similar products with explanations |
| `GET /api/customers` | Demo customers from real data |
| `GET /api/recommendations/{customer_id}` | Collaborative recommendations |
| `GET /api/hybrid/{customer_id}?weight=&k=` | Hybrid recommendations with score breakdown |
| `GET /api/metrics` | Evaluation results (`models/metrics.json`) |

Example: `curl "http://127.0.0.1:8000/api/products?q=pants&limit=3"`

## Results (test set, K = 10, 2,822 users)
| Model | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | Coverage |
|---|---|---|---|---|---|
| Popularity baseline | 0.0023 | 0.0119 | 0.0077 | 0.0088 | 0.06% |
| Collaborative (SVD) | 0.0021 | 0.0160 | 0.0080 | 0.0102 | 2.9% |
| Content-based | 0.0056 | 0.0502 | 0.0257 | 0.0319 | 60.1% |
| Hybrid | 0.0060 | 0.0536 | 0.0274 | 0.0341 | 52.8% |

Exact values are in `models/metrics.json`.

- The hybrid scored highest on every accuracy metric, but its lead over content-only is small (about 11 extra hits among 2,822 users, NDCG about 7% higher). No significance test was run.
- Text similarity does most of the work. Collaborative filtering is roughly level with popularity because users average only about 2.3 interactions.
- Popularity recommends almost the same few items to everyone (coverage 0.06%).
- Absolute scores are low because the task is to find a few later-liked items among about 18,000 products. Random guessing would give a recall@10 of about 0.05%, so the hybrid is roughly 100 times better than random and about 4 times better than popularity on NDCG.

## Limitations
- Very sparse data: after filtering, users average about 2.3 interactions, so the collaborative model has little to learn from.
- Only about 10% of products have a price and about 8.5% have a description. There is no category field. The budget filter works only for products with a price, and the text model relies mostly on title, brand and bullet points. Missing values are shown as missing, never invented.
- Amazon lists colour and size variants as separate products. The website shows one listing per title, but the test score still counts variants, which probably helps the content model.
- Some product photos are size charts or missing, and some old image links may be dead.
- Reviews are not purchases. The activity filter used all dates, including the test period. No confidence intervals were computed.

## Pin versions, then deploy on Render
1. After everything works locally: `pip freeze`, then copy exact versions of pandas, numpy, scipy, fastapi, uvicorn into `requirements-deploy.txt` (use `pkg==version`).
2. Commit `models/` (the files must be small enough for normal GitHub; if not, host them elsewhere and download once at startup).
3. Render: New > Web Service > connect the repository. Build: `pip install -r requirements-deploy.txt`. Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`. Health check: `/api/health`. A `render.yaml` is included. Check Render's current docs for Python version and free-tier limits.
4. Open the live URL and test every feature before writing it here: **URL: (add after verified)**

## Tests
`pytest -q` runs metric, preprocessing, split, model, pipeline and API tests on small synthetic data (never the real dataset).