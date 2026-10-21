"""Central settings. Change numbers here, not inside the model code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / 'data' / 'processed'
MODELS_DIR = ROOT / 'models'

RANDOM_SEED = 42            # makes sampling and SVD repeatable

# Activity filter (users / items need at least this many interactions).
# Pick the final values AFTER you see the sparsity numbers from Colab.
U_MIN = 2
I_MIN = 2

POSITIVE_RATING = 4         # rating >= 4 counts as "liked" (our own choice)
TRAIN_FRAC = 0.70           # oldest 70% of interactions (by time) = train
VAL_FRAC = 0.15             # next 15% = validation, newest 15% = test
K = 10                      # length of recommendation lists in evaluation
MAX_EVAL_USERS = 3000       # evaluate on at most this many users (speed)

SVD_COMPONENTS_GRID = [20, 50, 100, 200]
HYBRID_WEIGHT_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]   # weight of the collaborative score

MIN_TEXT_CHARS = 20         # description+features shorter than this = "thin text"
TFIDF_PARAMS = {
    'stop_words': 'english',
    'min_df': 2,
    'max_features': 50000,
    'ngram_range': (1, 2),
    'sublinear_tf': True,
}

CATALOG_DESC_CHARS = 600    # shortened text saved for the website
CATALOG_FEATURE_CHARS = 400
DEMO_CUSTOMER_MIN_INTERACTIONS = 3
N_DEMO_CUSTOMERS = 40
