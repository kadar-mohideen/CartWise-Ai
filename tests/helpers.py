"""Small synthetic data used by the tests (never the real dataset)."""
import numpy as np
import pandas as pd

CLUSTER_WORDS = [
    ['denim', 'jeans', 'indigo', 'stretch', 'pocket', 'waist'],
    ['sneaker', 'running', 'sole', 'mesh', 'lace', 'cushion'],
    ['silk', 'scarf', 'floral', 'elegant', 'soft', 'wrap'],
    ['leather', 'wallet', 'belt', 'brown', 'buckle', 'stitched'],
]


def make_synthetic(n_users=160, n_items=80, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_items):
        c = i % len(CLUSTER_WORDS)
        words = list(rng.choice(CLUSTER_WORDS[c], size=4, replace=False))
        title = ' '.join(words) + ' fashion'
        items.append({
            'item_id': f'B{i:09d}', 'title': title, 'brand': f'Brand{c}',
            'category': f'Clothing > Group{c}', 'features': '', 'description': f'{title} made for everyday use',
            'price': float(10 + i) if i % 5 else np.nan,
            'image_url': f'https://example.com/{i}.jpg' if i % 7 else None,
            'text': f'{title} {title} Brand{c} Clothing Group{c} made for everyday use',
            'rich_text': True,
        })
    products = pd.DataFrame(items)
    rows = []
    for u in range(n_users):
        c = u % len(CLUSTER_WORDS)
        own = [i for i in range(n_items) if i % len(CLUSTER_WORDS) == c]
        chosen = list(rng.choice(own, size=9, replace=False)) + list(rng.choice(n_items, size=3, replace=False))
        seen = set()
        for i in chosen:
            if i in seen:
                continue
            seen.add(i)
            t = int(rng.integers(1_500_000_000, 1_530_000_000))   # spread over time, like real reviews
            rating = int(rng.choice([4, 5])) if i % len(CLUSTER_WORDS) == c else int(rng.choice([1, 2, 3]))
            rows.append({'user_id': f'A{u:013d}', 'item_id': f'B{i:09d}', 'rating': float(rating), 'timestamp': t})
    inter = pd.DataFrame(rows).sample(frac=1, random_state=seed)
    inter = inter.sort_values('timestamp', kind='mergesort').reset_index(drop=True)
    return inter, products
