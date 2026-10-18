"""Recommendation algorithms.

Every model scores all items from a user's rating vector `r` (length n_items,
0 = not rated). This lets the same code serve existing users and brand-new
users who rate a few items live in the app.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from fuzzy_logic import infer, RULES, TERMS, OUTPUT_TERMS

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
        top = np.argsort(-s, kind="stable")[:k]
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
    return [Popularity(), ContentBased(), ItemCF(), Hybrid(), FuzzyRecommender()]


class FuzzyRecommender(Recommender):
    """Mamdani rule-based ranker; explicitly supplied context is request-local."""
    name = "Fuzzy (Mamdani)"

    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, products, R):
        super().fit(products, R)
        count = (R >= LIKE).sum(axis=0).astype(float)
        self.popularity = 100 * count / max(1.0, float(count.max()))
        return self

    def inputs(self, r, budget=None, preferred_categories=None):
        r = np.asarray(r, dtype=float)
        pref = np.full(self.n_items, 50.0)
        chosen = set(preferred_categories or [])
        if chosen:
            pref = np.where(self.products.category.isin(chosen), 100.0, 0.0)
        else:
            for category in self.products.category.unique():
                mask = (self.products.category == category).to_numpy()
                ratings = r[mask & (r > 0)]
                # Neutral prior equivalent to two observations, mapped 1..5 to 0..1.
                pref[mask] = 100 * (1 + ((ratings - 1) / 4).sum()) / (2 + len(ratings))
        origin = "explicit" if budget is not None else "unknown"
        if budget is None:
            positive = (r >= LIKE)
            if positive.any():
                budget = float(np.median(self.products.price.to_numpy()[positive]))
                origin = "history"
        if budget is not None:
            if not np.isfinite(budget) or budget <= 0:
                raise ValueError("Budget must be positive and finite")
            price_fit = 100 * np.clip(2 - self.products.price.to_numpy() / budget, 0, 1)
        else:
            price_fit = np.full(self.n_items, 50.0)
        return np.column_stack([pref, price_fit, self.popularity]), dict(
            budget=budget, budget_source=origin, explicit_categories=bool(chosen))

    def scores(self, r, budget=None, preferred_categories=None):
        values, _ = self.inputs(r, budget, preferred_categories)
        return infer(values, shift=self.shift)

    def recommend(self, r, k=10, category=None, *, budget=None,
                  preferred_categories=None, max_price=None):
        scores = self.scores(r, budget, preferred_categories).copy()
        scores[np.asarray(r) > 0] = -np.inf
        if category:
            scores[(self.products.category != category).to_numpy()] = -np.inf
        if max_price is not None:
            scores[(self.products.price > max_price).to_numpy()] = -np.inf
        # Deterministic item-ID tie breaking, also used by comparison models.
        top = np.argsort(-scores, kind="stable")[:k]
        top = top[np.isfinite(scores[top])]
        return top, scores[top]

    def explain(self, r, item, budget=None, preferred_categories=None):
        values, context = self.inputs(r, budget, preferred_categories)
        scores, trace = infer(values[[item]], shift=self.shift, trace=True)
        rule_index = int(np.argmax(trace["firing"][0]))
        a, b, c, out = RULES[rule_index]
        greek = ("χαμηλή", "μέτρια", "υψηλή")
        result = ("πολύ χαμηλή", "χαμηλή", "μέτρια", "υψηλή", "πολύ υψηλή")
        category = self.products.iloc[item].category
        prefix = "Δηλωμένη προτίμηση" if context["explicit_categories"] else "Ταίριασμα βάσει βαθμολογιών"
        if not np.any(np.asarray(r) > 0) and not context["explicit_categories"]:
            prefix = "Χωρίς προτιμήσεις: ουδέτερη αρχική εκτίμηση"
        price_note = ("δηλωμένο επιθυμητό ποσό" if context["budget_source"] == "explicit" else
                      "εκτίμηση τιμής από θετικές βαθμολογίες" if context["budget_source"] == "history" else
                      "χωρίς διαθέσιμο ποσό: ουδέτερη εκτίμηση")
        return (f"{prefix} για {category}: {greek[a]}. Συμβατότητα τιμής: {greek[b]} "
                f"({price_note}). Δημοφιλία: {greek[c]}. "
                f"Ο ισχυρότερος κανόνας δείχνει {result[out]} καταλληλότητα. "
                f"Συνολικό σκορ {scores[0]:.1f}/100 από όλους τους ενεργούς κανόνες.")
