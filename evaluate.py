"""Offline evaluation: hide 20% of each user's liked items, see if models rediscover them."""
import numpy as np
import pandas as pd
from pathlib import Path
from recommender import all_models, build_matrix, LIKE

DATA = Path(__file__).parent / "data"
K = 10


def split(R, seed=0, frac=0.2):
    rng = np.random.default_rng(seed)
    train, test = R.copy(), np.zeros_like(R, dtype=bool)
    for u in range(R.shape[0]):
        liked = np.where(R[u] >= LIKE)[0]
        if len(liked) < 5:
            continue
        hide = rng.choice(liked, size=max(1, int(len(liked) * frac)), replace=False)
        train[u, hide] = 0
        test[u, hide] = True
    return train, test


def ndcg(hits, n_relevant):
    """Binary NDCG: normalize by all held-out positives, not retrieved hits."""
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(hits), int(n_relevant))))
    return dcg / ideal if ideal else 0.0


def evaluate(products, R, k=K, seed=0, *, return_details=False, models=None):
    train, test = split(R, seed)
    users = np.where(test.any(axis=1))[0]
    out, details = [], []
    for model in (all_models() if models is None else models):
        model.fit(products, train)
        prec, rec, nd, seen = [], [], [], set()
        for u in users:
            top, _ = model.recommend(train[u], k)
            hits = [bool(test[u, i]) for i in top]
            prec.append(sum(hits) / k)
            rec.append(sum(hits) / test[u].sum())
            nd.append(ndcg(hits, test[u].sum()))
            details.append(dict(seed=seed, user_id=int(u), Model=model.name,
                                n_test=int(test[u].sum()),
                                n_train_likes=int((train[u] >= LIKE).sum()),
                                **{f"Precision@{k}": prec[-1], f"Recall@{k}": rec[-1],
                                   f"NDCG@{k}": nd[-1]}))
            seen.update(top.tolist())
        out.append(dict(Model=model.name, **{f"Precision@{k}": np.mean(prec), f"Recall@{k}": np.mean(rec),
                        f"NDCG@{k}": np.mean(nd), "Coverage": len(seen) / R.shape[1]}))
    summary = pd.DataFrame(out)
    return (summary, pd.DataFrame(details)) if return_details else summary


if __name__ == "__main__":
    products = pd.read_csv(DATA / "products.csv")
    inter = pd.read_csv(DATA / "interactions.csv")
    R = build_matrix(inter, inter.user_id.max() + 1, len(products))
    res = evaluate(products, R)
    res.to_csv(DATA / "evaluation_results.csv", index=False)
    print(res.round(3).to_string(index=False))
