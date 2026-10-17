"""Recommendation algorithms.

Every model scores all items from a user's rating vector `r` (length n_items,
0 = not rated). This lets the same code serve existing users and brand-new
users who rate a few items live in the app.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

LIKE = 4  # rating >= LIKE counts as "liked"


def _minmax(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


class Recommender:
    name = "base"

    def fit(self, products: pd.DataFrame, R: np.ndarray):
        self.products, self.R = products, R
        self.n_items = R.shape[1]
        return self

    def scores(self, r):
        raise NotImplementedError

    def recommend(self, r, k=10, category=None):
        s = self.scores(r).astype(float)
        s[r > 0] = -np.inf                                   # never recommend already rated items
        if category:
            s[(self.products["category"] != category).to_numpy()] = -np.inf
        top = np.argsort(-s)[:k]
        top = top[np.isfinite(s[top])]
        return top, s[top]


class Popularity(Recommender):
    """Baseline: most-liked items overall."""
    name = "Popularity"

    def fit(self, products, R):
        super().fit(products, R)
        self.pop = (R >= LIKE).sum(axis=0).astype(float)
        return self

    def scores(self, r):
        return self.pop


class ContentBased(Recommender):
    """TF-IDF over category/brand/tags; the user profile is a weighted mix of liked items."""
    name = "Content-based"

    def fit(self, products, R):
        super().fit(products, R)
        text = (products["category"] + " " + products["category"] + " " +
                products["brand"] + " " + products["tags"])
        X = TfidfVectorizer().fit_transform(text)
        self.X = normalize(X)
        self.sim = (self.X @ self.X.T).toarray()
        return self

    def scores(self, r):
        w = np.where(r > 0, r - 2.5, 0.0)                    # 4-5 pull towards, 1-2 push away
        if not w.any():
            return np.zeros(self.n_items)
        profile = self.X.T @ w
        return np.asarray(self.X @ profile).ravel()


class ItemCF(Recommender):
    """Item-based collaborative filtering (cosine similarity on rating columns)."""
    name = "Collaborative (item-based)"

    def fit(self, products, R):
        super().fit(products, R)
        A = np.where(R >= LIKE, R - 2.0, 0.0)               # liked ratings only, weighted
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1
        S = (A.T @ A) / np.outer(norms, norms)
        np.fill_diagonal(S, 0)
        self.sim = S
        return self

    def scores(self, r):
        a = np.where(r >= LIKE, r - 2.0, 0.0)
        return self.sim @ a


class Hybrid(Recommender):
    """Weighted blend of CF, content and popularity.
    Cold start (no history) falls back to popularity."""
    name = "Hybrid"

    def __init__(self, w_cf=0.5, w_cb=0.35, w_pop=0.15):
        self.w = (w_cf, w_cb, w_pop)

    def fit(self, products, R):
        super().fit(products, R)
        self.cf = ItemCF().fit(products, R)
        self.cb = ContentBased().fit(products, R)
        self.pop = Popularity().fit(products, R)
        return self

    def scores(self, r):
        if not (r > 0).any():
            return self.pop.scores(r)
        w_cf, w_cb, w_pop = self.w
        return (w_cf * _minmax(self.cf.scores(r)) +
                w_cb * _minmax(self.cb.scores(r)) +
                w_pop * _minmax(self.pop.scores(r)))

    def explain(self, r, item):
        """Return the liked item most similar to `item` (for 'Because you liked ...')."""
        liked = np.where(r >= LIKE)[0]
        if len(liked) == 0:
            return None
        sim = 0.5 * _minmax(self.cf.sim[item, liked]) + 0.5 * _minmax(self.cb.sim[item, liked])
        return int(liked[np.argmax(sim)])


def build_matrix(interactions: pd.DataFrame, n_users: int, n_items: int):
    R = np.zeros((n_users, n_items))
    R[interactions.user_id.to_numpy(), interactions.product_id.to_numpy()] = interactions.rating.to_numpy()
    return R


def all_models():
    return [Popularity(), ContentBased(), ItemCF(), Hybrid()]
