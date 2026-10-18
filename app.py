"""E-shop with AI recommendations. Run:  streamlit run app.py"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from recommender import Hybrid, FuzzyRecommender, build_matrix, all_models

DATA = Path(__file__).parent / "data"
APP_VERSION = "fuzzy-mamdani-v1"
FEEDBACK = DATA / "feedback_fuzzy_v1.csv"
ICONS = {"Electronics": "🎧", "Fashion": "👟", "Home": "🏠", "Books": "📚", "Sports": "🚴", "Beauty": "💄"}

st.set_page_config(page_title="AI Shop", page_icon="🛍️", layout="wide")

# When the sidebar is hidden, show a full-height grey bar (with the expand arrow) instead of a lone icon.
st.markdown("""
<style>
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #4b5563; }
[data-testid="stExpandSidebarButton"] {
    position: fixed; top: 0; left: 0; width: 46px; height: 100vh; z-index: 999991;
    background: #e5e7eb; border-right: 1px solid #cfd3d9; border-radius: 0;
    display: flex; align-items: flex-start; justify-content: center; padding-top: 18px;
}
[data-testid="stExpandSidebarButton"]:hover { background: #d4d8de; }
.stApp:has([data-testid="stExpandSidebarButton"]) [data-testid="stMainBlockContainer"] { padding-left: 4.5rem; }
</style>
""", unsafe_allow_html=True)
IMG = Path(__file__).parent / "assets" / "images"


@st.cache_data
def load():
    products = pd.read_csv(DATA / "products.csv")
    inter = pd.read_csv(DATA / "interactions.csv")
    return products, inter


@st.cache_resource
def get_models():
    products, inter = load()
    R = build_matrix(inter, inter.user_id.max() + 1, len(products))
    models = {m.name: m.fit(products, R) for m in all_models()}
    return models, R


products, inter = load()
models, R = get_models()
n_items = len(products)

# ---------------- session state ----------------
if "my_ratings" not in st.session_state:
    st.session_state.my_ratings = np.zeros(n_items)
if "last_recs" not in st.session_state:
    st.session_state.last_recs = None


def product_card(pid, col, rate=True, why=None, key=""):
    p = products.iloc[pid]
    with col:
        with st.container(border=True):
            st.image(str(IMG / f"{pid}.png"), width="stretch")
            st.markdown(f"**{p['name']}**")
            st.caption(f"{p.category} · {', '.join(p.tags.split()[:2])}")
            st.markdown(f"**€{p.price:.2f}**")
            if why:
                st.caption(f"💡 {why}")
            if rate:
                cur = int(st.session_state.my_ratings[pid])
                val = st.feedback("stars", key=f"rate_{key}_{pid}")
                if val is not None and val + 1 != cur:
                    st.session_state.my_ratings[pid] = val + 1
                    st.rerun()
                elif cur:
                    st.caption(f"Your rating: {'⭐' * cur}")


def grid(ids, cols=4, whys=None, key=""):
    for i in range(0, len(ids), cols):
        row = st.columns(cols)
        for c, pid in zip(row, ids[i:i + cols]):
            product_card(int(pid), c, why=(whys or {}).get(int(pid)), key=key)


# ---------------- sidebar ----------------
st.sidebar.title("🛍️ AI Shop")
page = st.sidebar.radio("Go to", ["Shop", "For You", "How it works", "Evaluation", "Feedback"], key="nav")
mode = st.sidebar.radio("Shopper", ["New customer (you)", "Existing customer (demo)"])
if mode.startswith("Existing"):
    demo_user = st.sidebar.number_input("Customer ID", 0, int(R.shape[0] - 1), 7)
    r = R[int(demo_user)].copy()
    st.sidebar.caption(f"{int((r > 0).sum())} past ratings")
else:
    r = st.session_state.my_ratings
    st.sidebar.caption(f"You rated {int((r > 0).sum())} products")
    if st.sidebar.button("Reset my ratings"):
        st.session_state.my_ratings = np.zeros(n_items)
        st.rerun()
model_name = st.sidebar.selectbox("Algorithm", list(models), index=len(models) - 1)
model = models[model_name]

# ---------------- pages ----------------
if page == "Shop":
    st.title("Shop")
    st.write("Rate products with the stars to update your recommendations.")
    st.button("Δείτε τις προτάσεις σας →", on_click=lambda: st.session_state.update(nav="For You"))
    c1, c2, c3 = st.columns([2, 2, 1])
    cat = c1.selectbox("Category", ["All"] + sorted(products.category.unique()))
    q = c2.text_input("Search")
    sort = c3.selectbox("Sort", ["Default", "Price ↑", "Price ↓"])
    view = products
    if cat != "All":
        view = view[view.category == cat]
    if q:
        view = view[view["name"].str.contains(q, case=False) | view.tags.str.contains(q, case=False)]
    if sort != "Default":
        view = view.sort_values("price", ascending=sort == "Price ↑")
    pages = max(1, int(np.ceil(len(view) / 24)))
    current_page = int(st.number_input("Σελίδα καταλόγου", min_value=1, max_value=pages, value=1, step=1))
    start = (current_page - 1) * 24
    grid(view.product_id.iloc[start:start + 24].tolist(), key="shop")
    st.caption(f"Σελίδα {current_page}/{pages} · {len(view)} προϊόντα")

elif page == "For You":
    st.title("Recommended for you")
    if not (r > 0).any():
        st.info("Δεν έχετε βαθμολογήσει προϊόντα. Δηλώστε προτιμώμενες κατηγορίες και προαιρετικά "
                "επιθυμητό ποσό για τις ασαφείς συστάσεις, ή βαθμολογήστε προϊόντα στο Shop. "
                "Χωρίς προτιμήσεις, η ασαφής μηχανή χρησιμοποιεί ουδέτερες εκτιμήσεις και δημοφιλία.")
    elif int((r > 0).sum()) < 5:
        st.caption("Το ιστορικό είναι ακόμη μικρό. Περισσότερες βαθμολογίες βοηθούν στον σχηματισμό προτιμήσεων.")
    preferences = st.multiselect("Προτιμώμενες κατηγορίες (προαιρετικά, για την ασαφή μέθοδο)",
                                 sorted(products.category.unique()), key="preferred_categories")
    budget_enabled = st.checkbox("Θέλω να δηλώσω επιθυμητό ποσό", key="budget_enabled")
    budget = None
    max_price = None
    if budget_enabled:
        budget = float(st.number_input("Επιθυμητό ποσό (€)", min_value=1.0, value=100.0, step=10.0, key="budget"))
        strict = st.checkbox("Αυστηρό μέγιστο: απόκρυψη ακριβότερων προϊόντων", key="strict_budget")
        max_price = budget if strict else None
        st.caption("Η ασαφής μέθοδος χρησιμοποιεί το ποσό ως προτίμηση. Το αυστηρό μέγιστο ισχύει για κάθε μέθοδο.")
    cat = st.selectbox("Filter by category", ["All"] + sorted(products.category.unique()))
    k = st.slider("How many?", 4, 20, 8, step=4)
    category = None if cat == "All" else cat
    if isinstance(model, FuzzyRecommender):
        top, sc = model.recommend(r, k, category, budget=budget,
                                  preferred_categories=preferences, max_price=max_price)
        if budget is None:
            st.caption("Χωρίς δηλωμένο ποσό: χρησιμοποιείται η διάμεση τιμή των προϊόντων με 4–5 αστέρια, "
                       "εφόσον υπάρχουν. Είναι εκτίμηση από το ιστορικό, όχι δηλωμένος προϋπολογισμός.")
    else:
        # Request enough results before applying the common hard-price filter.
        top, sc = model.recommend(r, n_items if max_price is not None else k, category)
        if max_price is not None:
            eligible = products.iloc[top].price.to_numpy() <= max_price
            top, sc = top[eligible][:k], sc[eligible][:k]
    whys = {}
    if isinstance(model, FuzzyRecommender):
        whys = {int(pid): model.explain(r, int(pid), budget, preferences) for pid in top}
        st.caption("Η καταλληλότητα 0–100 είναι αποτέλεσμα ασαφών κανόνων, όχι πιθανότητα αγοράς. "
                   "Οι εξηγήσεις συνοψίζουν τον ισχυρότερο ενεργό κανόνα.")
    elif isinstance(model, Hybrid):
        for pid in top:
            src = model.explain(r, int(pid))
            whys[int(pid)] = f"Because you liked {products.iloc[src]['name']}" if src is not None else "Popular right now"
    else:
        whys = {int(p): f"Score {s:.2f}" for p, s in zip(top, sc)}
    st.session_state.last_recs = top
    if len(top) == 0:
        st.warning("Δεν υπάρχουν επιλέξιμα προϊόντα. Αλλάξτε κατηγορία ή αυξήστε το αυστηρό όριο τιμής.")
    grid(top.tolist(), whys=whys, key="rec")

elif page == "How it works":
    st.title("How the recommendations work")
    st.markdown("""
**Data** – ratings (1–5) of products. A rating ≥ 4 means the customer *liked* the product.

| Algorithm | Idea |
|---|---|
| **Popularity** | Recommend what most people liked (baseline, not personal). |
| **Content-based** | Describe each product by category, brand and tags (TF-IDF). Build a taste profile from the products you liked and find similar ones. |
| **Collaborative (item-based)** | Products that are liked by the *same people* are similar. Recommend items similar to those you liked. |
| **Hybrid** | Weighted blend: 50% collaborative + 35% content + 15% popularity. Falls back to popularity for new customers (*cold start*). |
| **Fuzzy (Mamdani)** | Preference match, price compatibility and popularity activate 27 IF–THEN rules. MIN for AND and implication, MAX for aggregation, centroid for a suitability score from 0 to 100. |

When you have liked products, hybrid recommendations include a **"Because you liked …"** similarity explanation.
This highlights a related product from your history; its usefulness needs to be evaluated with users.

**Νέα προεπιλογή: ασαφής λογική Mamdani.** Οι κατηγορίες προκύπτουν από τις βαθμολογίες σας
ή από τη ρητή επιλογή σας. Το επιθυμητό ποσό δηλώνεται προαιρετικά· διαφορετικά εκτιμάται από
τις θετικές βαθμολογίες. Άγνωστες προτιμήσεις παίρνουν ουδέτερη τιμή. Η δημοφιλία προέρχεται
από τις συνθετικές βαθμολογίες επίδειξης. Οι λεκτικοί κανόνες και οι συναρτήσεις συμμετοχής
είναι σχεδιαστικές επιλογές και δεν εκπαιδεύτηκαν από τα 11 ερωτηματολόγια.

Η νέα έκδοση δεν έχει ακόμη αξιολογηθεί από πραγματικούς χρήστες. Η προηγούμενη ανατροφοδότηση
χρησιμοποιήθηκε για τον σχεδιασμό της, όχι ως μέτρηση της νέας έκδοσης.
""")
    st.subheader("Dataset")
    a, b, c = st.columns(3)
    a.metric("Products", n_items)
    b.metric("Customers", R.shape[0])
    c.metric("Ratings", len(inter))
    st.bar_chart(inter.merge(products, on="product_id").groupby("category").size())
    st.bar_chart(inter.rating.value_counts().sort_index())

elif page == "Evaluation":
    st.title("Offline evaluation")
    st.write("For customers with at least five liked products, we hide 20% of their likes, rounded down "
             "(at least one), and check whether each algorithm retrieves them in its top-10. "
             "All models use the same training split; results describe synthetic ratings.")
    path = DATA / "evaluation_results.csv"
    if path.exists():
        res = pd.read_csv(path)
        st.dataframe(res.round(3), hide_index=True, width="stretch")
        st.bar_chart(res.set_index("Model")[["Precision@10", "Recall@10", "NDCG@10"]])
    if st.button("Re-run evaluation"):
        from evaluate import evaluate
        with st.spinner("Evaluating…"):
            res = evaluate(products, R)
            res.to_csv(path, index=False)
        st.rerun()
    st.caption("Precision@10 = share of the 10 recommendations that were hidden liked items · "
               "Recall@10 = share of hidden items recovered · NDCG rewards putting hits near the top · "
               "Coverage = share of the catalogue that is ever recommended.")

elif page == "Feedback":
    st.title("Tell us what you think")
    st.write("Try the **For You** page first (rate a few products in the Shop), then answer these quick questions.")
    st.caption("1 = strongly disagree · 5 = strongly agree. This is a custom questionnaire, not SUS.")
    with st.form("fb"):
        rel = st.slider("The recommendations matched my taste", 1, 5, 3)
        use = st.slider("The shop was easy to use", 1, 5, 3)
        explanation_seen = st.checkbox("I read a recommendation explanation")
        expl = st.slider("The recommendation explanations were helpful", 1, 5, 3)
        st.caption("If you did not see an explanation, its score will be recorded as not applicable.")
        sat = st.slider("Overall satisfaction", 1, 5, 3)
        comment = st.text_area("Comments (optional)")
        if st.form_submit_button("Send"):
            row = dict(time=datetime.now().isoformat(timespec="seconds"), n_ratings=int((r > 0).sum()),
                       algorithm=model_name, app_version=APP_VERSION, relevance=rel, ease_of_use=use,
                       explanations=expl if explanation_seen else np.nan,
                       satisfaction=sat, comment=comment)
            pd.DataFrame([row]).to_csv(FEEDBACK, mode="a", header=not FEEDBACK.exists(), index=False)
            st.success("Thank you!")
    if FEEDBACK.exists():
        fb = pd.read_csv(FEEDBACK)
        st.subheader(f"Real user responses ({len(fb)})")
        st.dataframe(fb[["relevance", "ease_of_use", "explanations", "satisfaction"]].mean().round(2).rename("mean (1–5)"))
