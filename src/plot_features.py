"""
Per-Review Sentiment Bridge Pipeline
======================================
Reformulation of the engagement vs quality analysis at the individual
review level, adding reviewer features and sentiment as a bridge target.

Unit of analysis: one row = one review (1.4M rows)

Three parallel LightGBM models:
  Model A → individual Rating         [star quality]
  Model B → sentiment_score (scaled)  [text quality]
  Model C → gap = Rating - sentiment  [rating inflation]

Feature classes:
  1. Recipe features    — nutrition, time, complexity, category
  2. Reviewer features  — generosity baseline, consistency, experience
  3. Review features    — position, days since pub, length, sentiment*

  * sentiment is a feature in Model A and Model C only,
    it IS the target in Model B.

SHAP variance decomposition:
  Groups SHAP magnitudes by feature class to answer:
  "How much does the recipe contribute vs the reviewer vs the context?"

Within-category analysis:
  Sentiment-star correlation per category → category reliability map

Outputs:
  plot_feature_review_level.html       ← interactive dashboard
  plot_feature_review_level.json       ← pre-computed for web app
  review_level_residuals.csv           ← per-review predictions & gaps
  shap_class_decomposition.csv         ← variance decomposition
  category_reliability.csv             ← per-category reliability map

Run from project root:
    python src/plot_feature_review_level.py
"""

import os
import re
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

import lightgbm as lgb
import shap
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
from plotly.colors import sample_colorscale

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DB_PATH        = "./data/tables/food_recipe.db"
OUTPUT_DIR     = "./plots/review_level_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_REVIEWER_REVIEWS = 3      # reviewer must have ≥ N reviews to be included
MIN_RECIPE_REVIEWS   = 5      # recipe must have ≥ N reviews
TOP_N_CATEGORIES     = 30
IMPORTANCE_TOP_PCT   = 0.50
RANDOM_STATE         = 42
SHAP_SAMPLE_SIZE     = 100_000  # rows to sample for SHAP (full 1.4M is slow)

LGB_PARAMS = dict(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    num_leaves        = 127,
    min_child_samples = 50,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    random_state      = RANDOM_STATE,
    verbose           = -1,
)

# ─────────────────────────────────────────────────────────────
# COLORS  (reuse existing palette)
# ─────────────────────────────────────────────────────────────
C = {
    "star"       : "#4C9BE8",   # Model A — star rating
    "sentiment"  : "#F28B30",   # Model B — sentiment
    "gap"        : "#9B59B6",   # Model C — inflation gap
    "recipe"     : "#2ECC71",   # recipe feature class
    "reviewer"   : "#E74C3C",   # reviewer feature class
    "review_ctx" : "#F39C12",   # review-context feature class
    "neutral"    : "#000000",
    "bg"         : "#FFFFFF",
    "grid"       : "rgba(255,255,255,0.07)",
    "text"       : "#000000",
    "zero"       : "rgba(255,80,80,0.55)",
}

NUTRITION_COLS = [
    "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
    "SodiumContent", "CarbohydrateContent", "FiberContent",
    "SugarContent", "ProteinContent",
]
TIME_COLS = ["CookTime", "PrepTime", "TotalTime"]

TECHNIQUE_KEYWORDS = [
    "fold", "temper", "deglaze", "proof", "emulsify", "sauté", "saute",
    "braise", "blanch", "julienne", "caramelize", "reduce", "flambé",
    "flambe", "knead", "macerate", "poach", "roast", "sear", "whisk",
    "marinate", "baste",
]


def hex_to_rgba(h: str, a: float = 1.0) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

C = {
    "star"       : "#4C9BE8",
    "sentiment"  : "#F28B30",
    "gap"        : "#9B59B6",
    "recipe"     : "#2ECC71",
    "reviewer"   : "#E74C3C",
    "review_ctx" : "#1612F3",
    "neutral"    : "#7F8C8D",
    "bg"         : "#FFFFFF",
    "text"       : "#000000",
    "grid"       : "rgba(255,255,255,0.07)",
    "zero"       : "rgba(255,80,80,0.55)",
}
 
# Blue-ish fill inside the triangle
TERNARY_BG   = "rgba(173, 216, 250, 0.28)"   # soft sky-blue wash
TERNARY_LINE = "#1A5A8A"                       # deep blue for axes
TERNARY_GRID = "rgba(26,  90, 138, 0.20)"     # same blue, faint
 
GROUP_COLOR_MAP = {
    "recipe"        : C["recipe"],
    "reviewer"      : C["reviewer"],
    "review_context": C["review_ctx"],
    "sentiment"     : C["sentiment"],
    "other"         : C["neutral"],
}
 
GROUP_LABELS = {
    "recipe"        : "Recipe",
    "reviewer"      : "Reviewer",
    "review_context": "Review Context",
    "sentiment"     : "Sentiment",
    "other"         : "Other",
}
 
 
# def pretty_feature_name(s):
#     """Convert raw feature name to display label."""
#     return (
#         str(s)
#         .replace("cat_",  "📂 ")
#         .replace("log_",  "⏱ ")
#         .replace("_", " ")
#         .title()
#     )


# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────
def load_data():
    print("📂  Loading data...")
    conn = sqlite3.connect(DB_PATH)
    recipes = pd.read_sql("SELECT * FROM recipes", conn)
    reviews = pd.read_sql("SELECT * FROM reviews", conn)
    conn.close()
    print(f"    Recipes : {len(recipes):,}")
    print(f"    Reviews : {len(reviews):,}")
    return recipes, reviews


# ─────────────────────────────────────────────────────────────
# 2. CLEAN RECIPES — same as before
# ─────────────────────────────────────────────────────────────
def _count_list_items(series):
    def _count(val):
        if pd.isna(val) or str(val).strip() == "":
            return 0
        val = re.sub(r"[\[\]\"']", "", str(val))
        return len([x for x in val.split(",") if x.strip()])
    return series.apply(_count)


def _count_steps(series):
    def _count(val):
        if pd.isna(val) or str(val).strip() == "":
            return 0
        val = re.sub(r"[\[\]\"']", "", str(val))
        parts = re.split(r"(?<=[.!?])\s+|\|", val)
        return max(1, len([p for p in parts if len(p.strip()) > 10]))
    return series.apply(_count)


def _technique_score(series):
    pattern = "|".join(TECHNIQUE_KEYWORDS)
    return series.fillna("").str.lower().str.count(pattern)


def clean_recipes(recipes: pd.DataFrame):
    print("🔧  Cleaning recipes...")
    df = recipes.copy()

    for col in NUTRITION_COLS + TIME_COLS + ["AggregatedRating", "ReviewCount", "RecipeServings"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ReviewCount"] = df["ReviewCount"].fillna(0)
    df = df[df["ReviewCount"] >= MIN_RECIPE_REVIEWS].copy()

    for col in NUTRITION_COLS + TIME_COLS:
        if col in df.columns:
            cap = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=cap)

    for col in NUTRITION_COLS:
        df[col] = df[col].fillna(df[col].median())

    for col in TIME_COLS:
        df[f"log_{col}"] = np.log1p(df[col].fillna(0))

    df["ingredient_count"]   = _count_list_items(df["RecipeIngredientParts"])
    df["keyword_count"]      = _count_list_items(df["Keywords"])
    df["instruction_steps"]  = _count_steps(df["RecipeInstructions"])
    df["instruction_length"] = df["RecipeInstructions"].fillna("").apply(
        lambda x: len(re.sub(r"[\[\]\"']", "", str(x)))
    )
    df["technique_score"]    = _technique_score(df["RecipeInstructions"])

    df["calorie_density"]       = df["Calories"] / df["RecipeServings"].clip(lower=1).fillna(1)
    df["protein_calorie_ratio"] = df["ProteinContent"] / df["Calories"].clip(lower=1)
    df["fat_sugar_combo"]       = df["FatContent"] * df["SugarContent"]
    df["fat_protein_combo"]     = df["FatContent"] * df["ProteinContent"]

    df["RecipeCategory"] = df["RecipeCategory"].fillna("Unknown").str.strip()
    top_cats = df["RecipeCategory"].value_counts().nlargest(50).index
    df["RecipeCategory_clean"] = df["RecipeCategory"].where(
        df["RecipeCategory"].isin(top_cats), other="Other"
    )
    cat_dummies = pd.get_dummies(df["RecipeCategory_clean"], prefix="cat", drop_first=True)
    df = pd.concat([df, cat_dummies], axis=1)

    print(f"    Recipes after filter: {len(df):,}")
    return df, list(cat_dummies.columns)


# ─────────────────────────────────────────────────────────────
# 3. CLEAN REVIEWS + SENTIMENT SCORING
# ─────────────────────────────────────────────────────────────
def score_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    print("💬  Scoring sentiment (VADER)...")
    analyzer = SentimentIntensityAnalyzer()

    # Batch process — faster than apply row-by-row
    texts = reviews["Review"].fillna("").astype(str).tolist()
    scores = [analyzer.polarity_scores(t)["compound"] for t in texts]

    # Scale compound [-1, +1] → [1, 5] to match star scale
    reviews = reviews.copy()
    reviews["sentiment_raw"]    = scores
    reviews["sentiment_scaled"] = (np.array(scores) + 1) / 2 * 4 + 1
    print(f"    Sentiment scored for {len(reviews):,} reviews.")
    return reviews


def clean_reviews(reviews: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    print("🔧  Cleaning reviews...")
    df = reviews.copy()

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating", "RecipeId", "AuthorId"])
    df = df[df["Rating"].between(1, 5)]

    # Keep only reviews for recipes that survived the recipe filter
    valid_recipe_ids = set(recipes["RecipeId"].astype(str))
    df["RecipeId"] = df["RecipeId"].astype(str)
    df = df[df["RecipeId"].isin(valid_recipe_ids)]

    # Parse dates
    for col in ["DateSubmitted", "DateModified"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"    Reviews after filter: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────
# 4. ENGINEER REVIEWER FEATURES (leave-one-out to avoid leakage)
# ─────────────────────────────────────────────────────────────
def engineer_reviewer_features(reviews: pd.DataFrame) -> pd.DataFrame:
    print("👤  Engineering reviewer features (leave-one-out)...")

    # Global per-reviewer stats
    rev_stats = (
        reviews.groupby("AuthorId")["Rating"]
        .agg(
            reviewer_total_reviews="count",
            reviewer_sum_rating="sum",
            reviewer_sum_sq_rating=lambda x: (x**2).sum(),
        )
        .reset_index()
    )

    # Merge onto reviews
    df = reviews.merge(rev_stats, on="AuthorId", how="left")

    # Leave-one-out mean rating: (sum - current) / (count - 1)
    df["reviewer_loo_mean"] = (
        (df["reviewer_sum_rating"] - df["Rating"])
        / (df["reviewer_total_reviews"] - 1).clip(lower=1)
    )

    # Leave-one-out std: approximate via running variance
    # std ≈ sqrt((sum_sq - current^2 - (sum-current)^2/(n-1)) / (n-2))
    n    = df["reviewer_total_reviews"]
    s    = df["reviewer_sum_rating"]
    sq   = df["reviewer_sum_sq_rating"]
    r    = df["Rating"]
    loo_n   = (n - 1).clip(lower=1)
    loo_s   = s - r
    loo_sq  = sq - r**2
    loo_var = (loo_sq - loo_s**2 / loo_n.clip(lower=1)) / (loo_n - 1).clip(lower=1)
    df["reviewer_loo_std"] = np.sqrt(loo_var.clip(lower=0))

    # Experience proxy: log total reviews
    df["reviewer_log_reviews"] = np.log1p(df["reviewer_total_reviews"])

    # Category familiarity: how many reviews this reviewer left in this category
    # (join back through recipe category)
    # We will add this after the full join — placeholder for now
    df["reviewer_loo_std"]  = df["reviewer_loo_std"].fillna(0)
    df["reviewer_loo_mean"] = df["reviewer_loo_mean"].fillna(df["Rating"].mean())

    # Filter: only keep reviewers with enough reviews
    df = df[df["reviewer_total_reviews"] >= MIN_REVIEWER_REVIEWS]

    print(f"    Reviews after reviewer filter: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────
# 5. BUILD JOINT DATAFRAME (review + recipe + reviewer)
# ─────────────────────────────────────────────────────────────
def build_joint(reviews: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    print("🔗  Joining reviews with recipe features...")

    # Recipe feature cols to join
    recipe_feature_cols = (
        NUTRITION_COLS
        + [f"log_{c}" for c in TIME_COLS]
        + [
            "ingredient_count", "keyword_count", "instruction_steps",
            "instruction_length", "technique_score", "calorie_density",
            "protein_calorie_ratio", "fat_sugar_combo", "fat_protein_combo",
            "RecipeServings", "RecipeCategory", "DatePublished",
        ]
        + [c for c in recipes.columns if c.startswith("cat_")]
    )
    recipe_feature_cols = [c for c in recipe_feature_cols if c in recipes.columns]

    recipes_slim = recipes[["RecipeId"] + recipe_feature_cols].copy()
    recipes_slim["RecipeId"] = recipes_slim["RecipeId"].astype(str)

    joint = reviews.merge(recipes_slim, on="RecipeId", how="inner")

    # ── Review-level features ─────────────────────────────────
    # Days since publication
    if "DatePublished" in joint.columns:
        joint["DatePublished"] = pd.to_datetime(joint["DatePublished"], errors="coerce")
        joint["days_since_pub"] = (
            joint["DateSubmitted"] - joint["DatePublished"]
        ).dt.days.clip(lower=0)
        joint["log_days_since_pub"] = np.log1p(joint["days_since_pub"].fillna(0))
    else:
        joint["log_days_since_pub"] = 0.0

    # Review sequence position within each recipe
    joint = joint.sort_values(["RecipeId", "DateSubmitted"])
    joint["review_position"] = joint.groupby("RecipeId").cumcount() + 1
    joint["log_review_position"] = np.log1p(joint["review_position"])

    # Review text length
    joint["review_length"] = joint["Review"].fillna("").str.len()
    joint["log_review_length"] = np.log1p(joint["review_length"])

    # Reviewer category familiarity
    cat_familiarity = (
        joint.groupby(["AuthorId", "RecipeCategory"])
        .cumcount()          # reviews in this category before current one
        .rename("reviewer_category_familiarity")
    )
    joint["reviewer_category_familiarity"] = cat_familiarity.values
    joint["log_cat_familiarity"] = np.log1p(joint["reviewer_category_familiarity"])

    # Gap = star rating − sentiment
    joint["gap"] = joint["Rating"] - joint["sentiment_scaled"]

    print(f"    Joint dataframe: {len(joint):,} rows")
    return joint


# ─────────────────────────────────────────────────────────────
# 6. BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────
def build_feature_matrix(joint: pd.DataFrame, cat_cols: list):
    print("🔧  Building feature matrix...")

    # ── Feature group definitions ─────────────────────────────
    recipe_features = (
        NUTRITION_COLS
        + [f"log_{c}" for c in TIME_COLS]
        + [
            "ingredient_count", "keyword_count", "instruction_steps",
            "instruction_length", "technique_score", "calorie_density",
            "protein_calorie_ratio", "fat_sugar_combo", "fat_protein_combo",
            "RecipeServings",
        ]
        + cat_cols
    )

    reviewer_features = [
        "reviewer_loo_mean",
        "reviewer_loo_std",
        "reviewer_log_reviews",
        "log_cat_familiarity",
    ]

    review_ctx_features = [
        "log_days_since_pub",
        "log_review_position",
        "log_review_length",
    ]

    # sentiment_scaled is a feature for star & gap models, target for sentiment model
    sentiment_feature = ["sentiment_scaled"]

    # ── Assemble full feature set ─────────────────────────────
    all_features = (
        [f for f in recipe_features   if f in joint.columns]
        + [f for f in reviewer_features  if f in joint.columns]
        + [f for f in review_ctx_features if f in joint.columns]
    )

    # Fill NaNs
    for col in all_features:
        if joint[col].dtype in [np.float64, np.int64, float, int]:
            joint[col] = joint[col].fillna(joint[col].median())

    X_base = joint[all_features].astype(float)
    X_base = X_base.loc[:, X_base.nunique() > 1]  # drop constants

    # For star model and gap model: add sentiment as feature
    X_with_sentiment = X_base.copy()
    if "sentiment_scaled" in joint.columns:
        X_with_sentiment["sentiment_scaled"] = joint["sentiment_scaled"].values

    y_star      = joint["Rating"].values.astype(float)
    y_sentiment = joint["sentiment_scaled"].values.astype(float)
    y_gap       = joint["gap"].values.astype(float)

    # Feature group membership for SHAP decomposition
    feat_groups = {}
    for f in X_with_sentiment.columns:
        if f in recipe_features or f.startswith("cat_"):
            feat_groups[f] = "recipe"
        elif f in reviewer_features:
            feat_groups[f] = "reviewer"
        elif f in review_ctx_features:
            feat_groups[f] = "review_context"
        elif f == "sentiment_scaled":
            feat_groups[f] = "sentiment"
        else:
            feat_groups[f] = "other"

    print(f"    Feature matrix (base)     : {X_base.shape[0]:,} × {X_base.shape[1]}")
    print(f"    Feature matrix (w/ senti) : {X_with_sentiment.shape[0]:,} × {X_with_sentiment.shape[1]}")
    return X_base, X_with_sentiment, y_star, y_sentiment, y_gap, feat_groups


# ─────────────────────────────────────────────────────────────
# 7. TRAIN LightGBM
# ─────────────────────────────────────────────────────────────
def train_lgb(X: pd.DataFrame, y: np.ndarray, label: str):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    print(f"🌲  Training LightGBM [{label}]...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    y_pred = model.predict(X)
    r2   = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"    [{label}] R²={r2:.4f}  RMSE={rmse:.4f}  best_iter={model.best_iteration_}")
    return model, r2, rmse, y_pred


# ─────────────────────────────────────────────────────────────
# 8. SHAP + VARIANCE DECOMPOSITION
# ─────────────────────────────────────────────────────────────
def compute_shap(model, X: pd.DataFrame, label: str,
                 sample_size: int = SHAP_SAMPLE_SIZE):
    print(f"🔍  Computing SHAP [{label}] on {min(len(X), sample_size):,} samples...")

    # Sample for speed — 1.4M rows is prohibitively slow for full SHAP
    if len(X) > sample_size:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    mean_abs = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X_sample.columns,
        name=f"shap_{label}",
    )
    print(f"    Top feature: {mean_abs.idxmax()} ({mean_abs.max():.5f})")
    return shap_vals, mean_abs, X_sample


def decompose_shap_by_group(mean_abs_shap: pd.Series,
                             feat_groups: dict) -> pd.DataFrame:
    """Sum mean |SHAP| by feature class → variance decomposition proxy."""
    rows = []
    total = mean_abs_shap.sum()
    group_sums = {}
    for feat, shap_val in mean_abs_shap.items():
        g = feat_groups.get(feat, "other")
        group_sums[g] = group_sums.get(g, 0) + shap_val

    for g, s in group_sums.items():
        rows.append({
            "feature_class"     : g,
            "total_shap"        : round(s, 6),
            "pct_of_total"      : round(s / total * 100, 2) if total > 0 else 0,
        })
    return pd.DataFrame(rows).sort_values("total_shap", ascending=False)

def build_shap_comparison_table(
    shap_star: pd.Series,
    shap_sent: pd.Series,
    shap_gap: pd.Series,
    feat_groups: dict,
    top_n: int = 80,
    rank_by: str = "mean"
) -> pd.DataFrame:

    # --- 1. Align features across models ---
    df = pd.concat(
        [
            shap_star.rename("star"),
            shap_sent.rename("sentiment"),
            shap_gap.rename("gap"),
        ],
        axis=1
    ).fillna(0.0)

    # --- 2. Raw sum (for optional ranking) ---
    df["raw_sum"] = df[["star", "sentiment", "gap"]].sum(axis=1)

    # --- 3. Normalize within each model ---
    norm = df[["star", "sentiment", "gap"]].copy()
    for c in ["star", "sentiment", "gap"]:
        s = norm[c].sum()
        norm[c] = norm[c] / s if s > 0 else 0.0

    df["star_norm"] = norm["star"]
    df["sentiment_norm"] = norm["sentiment"]
    df["gap_norm"] = norm["gap"]

    # --- 4. Ranking ---
    if rank_by == "mean":
        df["rank_score"] = df[["star_norm", "sentiment_norm", "gap_norm"]].mean(axis=1)
    elif rank_by == "max":
        df["rank_score"] = df[["star_norm", "sentiment_norm", "gap_norm"]].max(axis=1)
    else:
        df["rank_score"] = df["raw_sum"]

    # --- 5. Feature groups ---
    df["feature_group"] = [feat_groups.get(f, "other") for f in df.index]

    # --- 6. Ternary normalization (within feature) ---
    # Log-compress before ternary normalization to spread low-importance features
    # for c in ["star", "sentiment", "gap"]:
    #     df[c] = np.log1p(df[c] * 1e5)   # scale up first so log1p has range to work with
    tri_sum = df[["star", "sentiment", "gap"]].sum(axis=1).replace(0, np.nan)

    df["tri_star"] = df["star"] / tri_sum
    df["tri_sentiment"] = df["sentiment"] / tri_sum
    df["tri_gap"] = df["gap"] / tri_sum

    # --- 7. Select top features ---
    df = (
        df.sort_values("rank_score", ascending=False)
          .head(top_n)
          .reset_index()
          .rename(columns={"index": "feature"})
    )

    return df


# ─────────────────────────────────────────────────────────────
# 9. WITHIN-CATEGORY RELIABILITY MAP
# ─────────────────────────────────────────────────────────────
def category_reliability(joint: pd.DataFrame,
                          top_n: int = TOP_N_CATEGORIES) -> pd.DataFrame:
    print(f"\n📂  Computing category reliability map (top {top_n})...")

    top_cats = (
        joint["RecipeCategory"].value_counts()
        .nlargest(top_n).index.tolist()
    )

    results = []
    for cat in top_cats:
        sub = joint[joint["RecipeCategory"] == cat]
        if len(sub) < 50:
            continue

        r_pearson, p_pearson  = pearsonr(sub["Rating"], sub["sentiment_scaled"])
        r_spearman, p_spearman = spearmanr(sub["Rating"], sub["sentiment_scaled"])
        mean_gap  = sub["gap"].mean()
        std_gap   = sub["gap"].std()
        mean_star = sub["Rating"].mean()
        mean_sent = sub["sentiment_scaled"].mean()

        label = (
            "Reliable"   if r_pearson >= 0.5 else
            "Moderate"   if r_pearson >= 0.3 else
            "Unreliable"
        )

        results.append({
            "category"         : cat,
            "n_reviews"        : len(sub),
            "pearson_r"        : round(r_pearson, 4),
            "spearman_rho"     : round(r_spearman, 4),
            "mean_gap"         : round(mean_gap, 4),
            "std_gap"          : round(std_gap, 4),
            "mean_star"        : round(mean_star, 4),
            "mean_sentiment"   : round(mean_sent, 4),
            "reliability_label": label,
        })
        print(f"    {cat:<35} n={len(sub):>7,}  r={r_pearson:+.3f}  gap={mean_gap:+.3f}")

    return pd.DataFrame(results).sort_values("pearson_r")



# ─────────────────────────────────────────────────────────────
# 10. FIGURE HELPERS + SAVE JSON
# ─────────────────────────────────────────────────────────────
def pretty_feature_name(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace("cat_", "", regex=False)
         .str.replace("log_", "", regex=False)
         .str.replace("_", " ")
         .str.title()
    )


def apply_single_figure_layout(fig: go.Figure, title: str = "", *, height: int = 600,
                               showlegend: bool = False, barmode: str | None = None) -> go.Figure:
    fig.update_layout(
        title=None, #{"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 16, "color": C["text"]}},
        margin=dict(t=70, b=50, l=90, r=30),
        height=height,
        showlegend=showlegend,

        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",   # fully transparent
        plot_bgcolor="rgba(0,0,0,0)",    # fully transparent

        font={"color": "black", "family": "Inter, sans-serif"},
    )

    if barmode is not None:
        fig.update_layout(barmode=barmode)
    axis_style = dict(
        gridcolor=C["grid"],
        zerolinecolor=C["grid"],
        linecolor="rgba(255,255,255,0.15)",
        tickfont={"size": 10},
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def figure_to_payload(fig: go.Figure) -> dict:
    return json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))


def add_grouped_shap_bar(fig, comp_df: pd.DataFrame, row: int, col: int):
    d = comp_df.copy()
    d["feature_pretty"] = pretty_feature_name(d["feature"])
    d = d.sort_values("rank_score", ascending=True)

    fig.add_trace(go.Bar(
        x=d["star_norm"], y=d["feature_pretty"], orientation="h",
        name="Rating", marker_color=C["star"],
        hovertemplate="%{y}<br>Rating share: %{x:.3%}<extra></extra>",
        offsetgroup="rating", legendgroup="rating",
    ), row=row, col=col)
    fig.add_trace(go.Bar(
        x=d["sentiment_norm"], y=d["feature_pretty"], orientation="h",
        name="Sentiment", marker_color=C["sentiment"],
        hovertemplate="%{y}<br>Sentiment share: %{x:.3%}<extra></extra>",
        offsetgroup="sentiment", legendgroup="sentiment",
    ), row=row, col=col)
    fig.add_trace(go.Bar(
        x=d["gap_norm"], y=d["feature_pretty"], orientation="h",
        name="Gap", marker_color=C["gap"],
        hovertemplate="%{y}<br>Gap share: %{x:.3%}<extra></extra>",
        offsetgroup="gap", legendgroup="gap",
    ), row=row, col=col)
    fig.update_xaxes(title_text="Normalized mean |SHAP| within model", row=row, col=col)
    fig.update_yaxes(title_text="", row=row, col=col)


def add_shap_ternary(fig, comp_df: pd.DataFrame, row: int, col: int):
    """
    Render the ternary SHAP role plot into an existing subplot figure.
    One trace per feature-group so the legend shows coloured group labels.
    """
    d = comp_df.copy()
    d["feature_pretty"] = pretty_feature_name(d["feature"])
 
    # Marker size: proportional to total raw SHAP across all three models
    total_raw = d[["star", "sentiment", "gap"]].sum(axis=1)
    denom     = max(float(total_raw.max()), 1e-12)
    d["marker_size"] = 12 + 36 * (total_raw / denom)   # range [12, 48]
 
    # ── One trace per feature group (enables legend) ──────────────────────
    groups_present = d["feature_group"].unique()
 
    for grp in groups_present:
        subset = d[d["feature_group"] == grp]
        color  = GROUP_COLOR_MAP.get(grp, C["neutral"])
        label  = GROUP_LABELS.get(grp, grp.title())
 
        fig.add_trace(
            go.Scatterternary(
                a=subset["tri_star"],
                b=subset["tri_sentiment"],
                c=subset["tri_gap"],
                mode="markers",           # ← no text; all info in hover
                name=label,
                legendgroup=grp,
                showlegend=True,
                marker=dict(
                    size=subset["marker_size"].tolist(),
                    color=color,
                    opacity=1.0,
                    line=dict(
                        color="black",    # ← black border on every circle
                        width=1.8,        # ← bold border
                    ),
                    symbol="circle",
                ),
                customdata=np.column_stack([
                    subset["feature_pretty"],
                    subset["feature_group"],
                    subset["star"],
                    subset["sentiment"],
                    subset["gap"],
                    subset["tri_star"],
                    subset["tri_sentiment"],
                    subset["tri_gap"],
                ]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Group: %{customdata[1]}<br>"
                    "<br>"
                    "<b>Raw mean |SHAP|</b><br>"
                    "Rating   : %{customdata[2]:.5f}<br>"
                    "Sentiment: %{customdata[3]:.5f}<br>"
                    "Gap       : %{customdata[4]:.5f}<br>"
                    "<br>"
                    "<b>Within-feature split</b><br>"
                    "Rating   : %{customdata[5]:.1%}<br>"
                    "Sentiment: %{customdata[6]:.1%}<br>"
                    "Gap       : %{customdata[7]:.1%}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
 
    # ── Ternary axis styling ───────────────────────────────────────────────
    axis_common = dict(
        # title       = dict(font=dict(size=14, color=TERNARY_LINE, family="Inter, sans-serif")),
        tickfont    = dict(size=11, color=TERNARY_LINE),
        linecolor   = TERNARY_LINE,
        linewidth   = 3,              # ← bold axis lines
        gridcolor   = TERNARY_GRID,
        gridwidth   = 1.5,
        tickcolor   = TERNARY_LINE,
        showgrid    = True,
        ticks       = "outside",
        dtick       = 0.2,
        min         = 0,
    )
 
    fig.update_ternaries(
        bgcolor=TERNARY_BG,           # ← blue-ish fill inside triangle
        aaxis=dict(**axis_common, title=dict(text="<b>Rating</b>",
                    font=dict(size=14, color="black",       family="Inter, sans-serif"))),
        baxis=dict(**axis_common, title=dict(text="<b>Sentiment</b>",
                    font=dict(size=14, color="black",  family="Inter, sans-serif"))),
        caxis=dict(**axis_common, title=dict(text="<b>Gap</b>",
                    font=dict(size=14, color="black",        family="Inter, sans-serif"))),
        row=row,
        col=col,
    )


def add_shap_ridge_violin(
    fig,
    comp_df: pd.DataFrame,
    sv_star: np.ndarray,
    sv_sent: np.ndarray,
    sv_gap: np.ndarray,
    X_samp_star: pd.DataFrame,
    X_samp_sent: pd.DataFrame,
    X_samp_gap: pd.DataFrame,
    row: int,
    col: int,
):
    d = comp_df.copy()
    d["feature_pretty"] = pretty_feature_name(d["feature"])
    d = d.sort_values("rank_score", ascending=True)

    idx_star = {f: i for i, f in enumerate(X_samp_star.columns)}
    idx_sent = {f: i for i, f in enumerate(X_samp_sent.columns)}
    idx_gap = {f: i for i, f in enumerate(X_samp_gap.columns)}

    ridge_n = min(6000, len(X_samp_star), len(X_samp_sent), len(X_samp_gap))
    rng = np.random.RandomState(RANDOM_STATE)
    take_star = rng.choice(len(X_samp_star), ridge_n, replace=False) if len(X_samp_star) > ridge_n else np.arange(len(X_samp_star))
    take_sent = rng.choice(len(X_samp_sent), ridge_n, replace=False) if len(X_samp_sent) > ridge_n else np.arange(len(X_samp_sent))
    take_gap = rng.choice(len(X_samp_gap), ridge_n, replace=False) if len(X_samp_gap) > ridge_n else np.arange(len(X_samp_gap))

    for _, row_df in d.iterrows():
        feat = row_df["feature"]
        feat_label = row_df["feature_pretty"]
        if feat in idx_star:
            vals = sv_star[take_star, idx_star[feat]]
            fig.add_trace(go.Violin(
                x=vals, y=[feat_label] * len(vals), orientation="h", name="Rating",
                legendgroup="rating", scalegroup="rating", side="positive",
                line_color=C["star"], fillcolor=hex_to_rgba(C["star"], 0.35),
                opacity=0.55, width=0.9, points=False, meanline_visible=False,
                showlegend=False,
                hovertemplate=f"{feat_label}<br>Rating SHAP: %{{x:.5f}}<extra></extra>",
            ), row=row, col=col)
        if feat in idx_sent:
            vals = sv_sent[take_sent, idx_sent[feat]]
            fig.add_trace(go.Violin(
                x=vals, y=[feat_label] * len(vals), orientation="h", name="Sentiment",
                legendgroup="sentiment", scalegroup="sentiment", side="positive",
                line_color=C["sentiment"], fillcolor=hex_to_rgba(C["sentiment"], 0.30),
                opacity=0.50, width=0.7, points=False, meanline_visible=False,
                showlegend=False,
                hovertemplate=f"{feat_label}<br>Sentiment SHAP: %{{x:.5f}}<extra></extra>",
            ), row=row, col=col)
        if feat in idx_gap:
            vals = sv_gap[take_gap, idx_gap[feat]]
            fig.add_trace(go.Violin(
                x=vals, y=[feat_label] * len(vals), orientation="h", name="Gap",
                legendgroup="gap", scalegroup="gap", side="positive",
                line_color=C["gap"], fillcolor=hex_to_rgba(C["gap"], 0.28),
                opacity=0.45, width=0.5, points=False, meanline_visible=False,
                showlegend=False,
                hovertemplate=f"{feat_label}<br>Gap SHAP: %{{x:.5f}}<extra></extra>",
            ), row=row, col=col)

    fig.update_xaxes(title_text="Per-sample SHAP value distribution", row=row, col=col)
    fig.update_yaxes(title_text="", row=row, col=col)


def make_top_shap_bar_figure(shap_s: pd.Series, title: str, color: str, top_n: int = 20) -> go.Figure:
    top = shap_s.nlargest(top_n).sort_values()
    labels = pretty_feature_name(pd.Series(top.index)).tolist()
    fig = go.Figure(go.Bar(
        x=top.values, y=labels, orientation="h", marker_color=color, marker_opacity=0.85,
        hovertemplate="%{y}<br>Mean |SHAP|: %{x:.5f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_xaxes(title_text="Mean |SHAP|")
    fig.update_yaxes(title_text="")
    return apply_single_figure_layout(fig, title, height=700)


def make_decomposition_figure(decomp: pd.DataFrame, title: str) -> go.Figure:
    d = decomp.sort_values("pct_of_total", ascending=True)
    group_colors = {
        "recipe": C["recipe"],
        "reviewer": C["reviewer"],
        "review_context": C["review_ctx"],
        "sentiment": C["sentiment"],
        "other": C["neutral"],
    }
    fig = go.Figure(go.Bar(
        x=d["pct_of_total"],
        y=d["feature_class"].str.replace("_", " ").str.title(),
        orientation="h",
        marker_color=[group_colors.get(g, C["neutral"]) for g in d["feature_class"]],
        text=[f"{v:.1f}%" for v in d["pct_of_total"]],
        textposition="inside",
        textfont=dict(color="white", size=10),
        hovertemplate="%{y}<br>%{x:.2f}% of total SHAP<extra></extra>",
        showlegend=False,
    ))
    fig.update_xaxes(title_text="% of Total |SHAP|")
    fig.update_yaxes(title_text="")
    return apply_single_figure_layout(fig, title, height=500)


def make_combined_decomposition_figure(
    decomp_star: pd.DataFrame,
    decomp_sent: pd.DataFrame,
    decomp_gap: pd.DataFrame,
) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
        subplot_titles=(
            "SHAP Variance Decomposition — Rating",
            "SHAP Variance Decomposition — Sentiment",
            "SHAP Variance Decomposition — Gap",
        ),
        horizontal_spacing=0.08,
    )

    group_colors = {
        "recipe": C["recipe"],
        "reviewer": C["reviewer"],
        "review_context": C["review_ctx"],
        "sentiment": C["sentiment"],
        "other": C["neutral"],
    }

    for col_i, decomp in enumerate([decomp_star, decomp_sent, decomp_gap], start=1):
        d = decomp.sort_values("pct_of_total", ascending=True)
        fig.add_trace(go.Bar(
            x=d["pct_of_total"],
            y=d["feature_class"].str.replace("_", " ").str.title(),
            orientation="h",
            marker_color=[group_colors.get(g, C["neutral"]) for g in d["feature_class"]],
            text=[f"{v:.1f}%" for v in d["pct_of_total"]],
            textposition="inside",
            textfont=dict(color="white", size=10),
            hovertemplate="%{y}<br>%{x:.2f}% of total SHAP<extra></extra>",
            showlegend=False,
        ), row=1, col=col_i)
        fig.update_xaxes(title_text="% of Total |SHAP|", row=1, col=col_i)
        fig.update_yaxes(title_text="", row=1, col=col_i)

    return apply_single_figure_layout(fig, "Combined SHAP Variance Decomposition", height=560)


def make_model_r2_figure(meta: dict) -> go.Figure:
    models = ["Star Rating", "Sentiment", "Gap (Inflation)"]
    r2_vals = [meta["model_star"]["r2"], meta["model_sentiment"]["r2"], meta["model_gap"]["r2"]]
    fig = go.Figure(go.Bar(
        x=models, y=r2_vals, marker_color=[C["star"], C["sentiment"], C["gap"]],
        text=[f"R²={v:.4f}" for v in r2_vals], textposition="inside",
        textfont=dict(color="white", size=11), showlegend=False,
        hovertemplate="%{x}<br>R²=%{y:.4f}<extra></extra>",
    ))
    fig.update_yaxes(title_text="R²", range=[0, max(r2_vals) * 1.3 if len(r2_vals) else 1])
    return apply_single_figure_layout(fig, "Model R² Comparison", height=450)


def make_rating_histogram(joint: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=joint["Rating"].values, nbinsx=5, marker_color=C["star"], marker_opacity=0.8,
        hovertemplate="Rating: %{x}<br>Count: %{y:,}<extra></extra>", showlegend=False,
    ))
    fig.update_xaxes(title_text="Star Rating")
    fig.update_yaxes(title_text="Count")
    return apply_single_figure_layout(fig, "Star Rating Distribution", height=450)


def make_gap_histogram(joint: pd.DataFrame) -> go.Figure:
    gap_sample = joint["gap"].sample(min(100_000, len(joint)), random_state=RANDOM_STATE)
    fig = go.Figure(go.Histogram(
        x=gap_sample.values, nbinsx=40, marker_color=C["gap"], marker_opacity=0.8,
        hovertemplate="Gap: %{x:.2f}<br>Count: %{y:,}<extra></extra>", showlegend=False,
    ))
    fig.update_xaxes(title_text="Gap (Star − Sentiment)")
    fig.update_yaxes(title_text="Count")
    return apply_single_figure_layout(fig, "Sentiment-Star Gap Distribution", height=450)


def make_category_reliability_figure(cat_rel: pd.DataFrame) -> go.Figure:
    d = cat_rel.copy().sort_values("pearson_r", ascending=False).reset_index(drop=True)
    d["category_pretty"] = d["category"].astype(str).str.replace("_", " ", regex=False)

    # Blue-ish gradient keyed to reliability strength, kept light enough for dark text.
    r_min = float(d["pearson_r"].min())
    r_max = float(d["pearson_r"].max())
    denom = max(r_max - r_min, 1e-9)
    norm = ((d["pearson_r"] - r_min) / denom).clip(0, 1)
    bar_colors = sample_colorscale(
        [
            [0.00, "rgb(225,236,250)"],
            [0.35, "rgb(166,201,235)"],
            [0.65, "rgb(96,155,212)"],
            [1.00, "rgb(24,92,163)"],
        ],
        norm.tolist(),
    )

    text_pos = ["inside" if v >= 0.18 else "outside" for v in d["pearson_r"]]
    text_col = ["white" if v >= 0.42 else C["text"] for v in d["pearson_r"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["pearson_r"],
        y=d["category_pretty"],
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="black", width=1.4),
        ),
        text=d["category_pretty"],
        textposition=text_pos,
        textfont=dict(size=15, color=text_col),
        insidetextanchor="middle",
        cliponaxis=False,
        customdata=np.column_stack([
            d["n_reviews"],
            d["mean_star"],
            d["mean_sentiment"],
            d["mean_gap"],
            d["reliability_label"],
            d["spearman_rho"],
            d["std_gap"],
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Pearson r: %{x:.4f}<br>"
            "Spearman ρ: %{customdata[5]:.4f}<br>"
            "n reviews: %{customdata[0]:,}<br>"
            "Mean star: %{customdata[1]:.2f}<br>"
            "Mean sentiment: %{customdata[2]:.2f}<br>"
            "Mean gap: %{customdata[3]:+.2f}<br>"
            "Gap σ: %{customdata[6]:.2f}<br>"
            "Reliability: %{customdata[4]}<extra></extra>"
        ),
        showlegend=False,
    ))

    # # Subtle alternating bands for readability.
    # for i in range(len(d)):
    #     if i % 2 == 0:
    #         fig.add_shape(
    #             type="rect",
    #             xref="x domain",
    #             yref="y",
    #             x0=0, x1=1,
    #             y0=i - 0.5, y1=i + 0.5,
    #             fillcolor="rgba(110, 160, 220, 0.06)",
    #             line=dict(width=0),
    #             layer="below",
    #         )

    fig.add_vline(x=0, line_width=1.4, line_color="rgba(0,0,0,0.65)")

    fig.update_layout(
        margin=dict(t=30, b=30, l=55, r=55),
        height=max(520, 72 + 46 * len(d)),
        bargap=0.28,
        font=dict(size=16, color=C["text"], family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",   # transparent outer
        plot_bgcolor="rgba(0,0,0,0)",    # ← remove inner blue tint
    )

    fig.update_xaxes(
        title_text=f"Pearson 𝑟 (star rating vs sentiment)",
        side="bottom",
        ticks=None,
        ticklen=2,
        tickwidth=1.0,
        tickcolor="black",
        showline=True,
        linewidth=1.2,
        linecolor="black",
        mirror=False,
        range=[min(-0.01, r_min - 0.03), r_max + 0.1],
        showgrid=True,
        gridcolor="rgba(36, 94, 168, 0.16)",
        gridwidth=1.0,
        zeroline=False,
        title_font=dict(size=17, color=C["text"]),
        tickfont=dict(size=14, color=C["text"]),
    )
    fig.update_yaxes(
        title_text="",
        autorange="reversed",
        showline=False,   # ← remove axis line
        ticks="",         # ← remove ticks
        showticklabels=False,
        tickfont=dict(size=15, color=C["text"]),
        automargin=True,
    )

    return fig


def make_reliability_count_figure(cat_rel: pd.DataFrame) -> go.Figure:
    rel_colors_map = {"Reliable": C["recipe"], "Moderate": C["sentiment"], "Unreliable": C["reviewer"]}
    rel_counts = cat_rel["reliability_label"].value_counts()
    fig = go.Figure(go.Bar(
        x=rel_counts.index.tolist(), y=rel_counts.values.tolist(),
        marker_color=[rel_colors_map.get(k, C["neutral"]) for k in rel_counts.index],
        text=rel_counts.values.tolist(), textposition="outside", textfont=dict(color=C["text"]),
        showlegend=False, hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_yaxes(title_text="# Categories")
    return apply_single_figure_layout(fig, "Reliability Label Distribution", height=450)


def make_grouped_shap_figure(comp_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "bar"}]])
    add_grouped_shap_bar(fig, comp_df, row=1, col=1)
    return apply_single_figure_layout(fig, "Grouped Cross-Model SHAP Comparison", height=750, showlegend=True, barmode="group")


def make_ternary_figure(comp_df: pd.DataFrame) -> go.Figure:
    """
    Standalone ternary figure with full feature set and polished styling.
 
    Pass a comp_df built with top_n=9999 (all features) from
    build_shap_comparison_table so nothing is dropped.
    """
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "ternary"}]],
    )
 
    add_shap_ternary(fig, comp_df, row=1, col=1)
 
    # n_features = len(comp_df)
 
    fig.update_layout(
        # ── title ────────────────────────────────────────────────────────
        title=dict(
            # text=(
            #     "<b>SHAP Feature Role Map</b><br>"
            #     "<sup>Each circle = one feature. "
            #     "Position shows how its importance is split across the three models. "
            #     "Size = total SHAP magnitude.</sup>"
            # ),
            x=0.5, xanchor="center",
            font=dict(size=16, color=C["text"], family="Inter, sans-serif"),
        ),
 
        # ── legend ───────────────────────────────────────────────────────
        showlegend=True,
        legend=dict(
            # title=dict(
            #     text="<b>Feature Group</b>",
            #     font=dict(size=12, color=C["text"]),
            # ),
            bgcolor="rgba(255,255,255,0.0)",
            bordercolor="rgba(0,0,0,0.0)",
            borderwidth=1,
            font=dict(size=11, color=C["text"]),
            itemsizing="constant",
            x=0.65, y=0.95,
            xanchor="left",
        ),
 
        # ── size / margins ────────────────────────────────────────────────
        height=700,
        margin=dict(t=30, b=30, l=30, r=30),
 
        # ── background ───────────────────────────────────────────────────
        paper_bgcolor="rgba(0,0,0,0)",   # transparent — matches dashboard
        plot_bgcolor ="rgba(0,0,0,0)",
 
        font=dict(color=C["text"], family="Inter, sans-serif"),
 
        # ── annotation: feature count watermark ──────────────────────────
        # annotations=[
        #     dict(
        #         text=f"n = {n_features} features",
        #         xref="paper", yref="paper",
        #         x=0.01, y=0.01,
        #         showarrow=False,
        #         font=dict(size=10, color="rgba(0,0,0,0.35)"),
        #         xanchor="left",
        #     )
        # ],
    )
 
    return fig


def make_ridge_figure(comp_df: pd.DataFrame, sv_star: np.ndarray, sv_sent: np.ndarray, sv_gap: np.ndarray,
                      X_samp_star: pd.DataFrame, X_samp_sent: pd.DataFrame, X_samp_gap: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "violin"}]])
    add_shap_ridge_violin(fig, comp_df, sv_star, sv_sent, sv_gap, X_samp_star, X_samp_sent, X_samp_gap, row=1, col=1)
    return apply_single_figure_layout(fig, "Ridge-Style SHAP Distribution by Feature", height=850, showlegend=False)


def build_standalone_figures(payload: dict,
                             shap_star: pd.Series,
                             shap_sent: pd.Series,
                             shap_gap: pd.Series,
                             decomp_star: pd.DataFrame,
                             decomp_sent: pd.DataFrame,
                             decomp_gap: pd.DataFrame,
                             cat_rel: pd.DataFrame,
                             joint: pd.DataFrame,
                             feat_groups: dict,
                             sv_star: np.ndarray,
                             sv_sent: np.ndarray,
                             sv_gap: np.ndarray,
                             X_samp_star: pd.DataFrame,
                             X_samp_sent: pd.DataFrame,
                             X_samp_gap: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    meta = payload["meta"]
    comp_df = build_shap_comparison_table(
        shap_star=shap_star,
        shap_sent=shap_sent,
        shap_gap=shap_gap,
        feat_groups=feat_groups,
        top_n=15,
        rank_by="mean",
    )
    figures = {
        "top20_star": figure_to_payload(make_top_shap_bar_figure(shap_star, "Top 20 Features — Star Rating Model", C["star"])),
        "top20_sentiment": figure_to_payload(make_top_shap_bar_figure(shap_sent, "Top 20 Features — Sentiment Model", C["sentiment"])),
        "top20_gap": figure_to_payload(make_top_shap_bar_figure(shap_gap, "Top 20 Features — Gap (Inflation) Model", C["gap"])),
        "decomp_star": figure_to_payload(make_decomposition_figure(decomp_star, "SHAP Variance Decomposition — Star")),
        "decomp_sentiment": figure_to_payload(make_decomposition_figure(decomp_sent, "SHAP Variance Decomposition — Sentiment")),
        "decomp_gap": figure_to_payload(make_decomposition_figure(decomp_gap, "SHAP Variance Decomposition — Gap")),
        "decomp_combined": figure_to_payload(make_combined_decomposition_figure(decomp_star, decomp_sent, decomp_gap)),
        "model_r2": figure_to_payload(make_model_r2_figure(meta)),
        "rating_distribution": figure_to_payload(make_rating_histogram(joint)),
        "gap_distribution": figure_to_payload(make_gap_histogram(joint)),
        "category_reliability": figure_to_payload(make_category_reliability_figure(cat_rel)),
        "reliability_count": figure_to_payload(make_reliability_count_figure(cat_rel)),
        "grouped_cross_model_shap": figure_to_payload(make_grouped_shap_figure(comp_df)),
        "ternary_feature_role": figure_to_payload(make_ternary_figure(comp_df)),
        "ridge_shap_distribution": figure_to_payload(make_ridge_figure(comp_df, sv_star, sv_sent, sv_gap, X_samp_star, X_samp_sent, X_samp_gap)),
    }
    return figures, comp_df


def save_json(
    r2_star, rmse_star,
    r2_sent, rmse_sent,
    r2_gap, rmse_gap,
    shap_star: pd.Series,
    shap_sent: pd.Series,
    shap_gap: pd.Series,
    feat_groups: dict,
    decomp_star: pd.DataFrame,
    decomp_sent: pd.DataFrame,
    decomp_gap: pd.DataFrame,
    cat_rel: pd.DataFrame,
    joint: pd.DataFrame,
    sv_star: np.ndarray,
    sv_sent: np.ndarray,
    sv_gap: np.ndarray,
    X_samp_star: pd.DataFrame,
    X_samp_sent: pd.DataFrame,
    X_samp_gap: pd.DataFrame,
):
    print("\n💾  Saving JSON...")

    def top30(s):
        t = s.nlargest(30).reset_index()
        t.columns = ["feature", "mean_abs_shap"]
        return t.to_dict(orient="records")

    payload = {
        "meta": {
            "model_star": {"r2": round(r2_star, 4), "rmse": round(rmse_star, 4)},
            "model_sentiment": {"r2": round(r2_sent, 4), "rmse": round(rmse_sent, 4)},
            "model_gap": {"r2": round(r2_gap, 4), "rmse": round(rmse_gap, 4)},
            "n_reviews": int(len(joint)),
            "n_recipes": int(joint["RecipeId"].nunique()),
            "n_reviewers": int(joint["AuthorId"].nunique()),
            "mean_gap": round(float(joint["gap"].mean()), 4),
            "mean_star": round(float(joint["Rating"].mean()), 4),
            "mean_sentiment": round(float(joint["sentiment_scaled"].mean()), 4),
        },
        "top30_star": top30(shap_star),
        "top30_sentiment": top30(shap_sent),
        "top30_gap": top30(shap_gap),
        "decomp_star": decomp_star.to_dict(orient="records"),
        "decomp_sentiment": decomp_sent.to_dict(orient="records"),
        "decomp_gap": decomp_gap.to_dict(orient="records"),
        "category_reliability": cat_rel.to_dict(orient="records"),
    }

    standalone_figures, comp_df = build_standalone_figures(
        payload=payload,
        shap_star=shap_star,
        shap_sent=shap_sent,
        shap_gap=shap_gap,
        decomp_star=decomp_star,
        decomp_sent=decomp_sent,
        decomp_gap=decomp_gap,
        cat_rel=cat_rel,
        joint=joint,
        feat_groups=feat_groups,
        sv_star=sv_star,
        sv_sent=sv_sent,
        sv_gap=sv_gap,
        X_samp_star=X_samp_star,
        X_samp_sent=X_samp_sent,
        X_samp_gap=X_samp_gap,
    )

    payload["cross_model_shap_comparison"] = comp_df.to_dict(orient="records")
    payload["standalone_figures"] = standalone_figures
    payload["webapp_panels"] = {
        "feature_importance_main": "ternary_feature_role",
        "feature_importance_alt": [
            "ridge_shap_distribution",
            "grouped_cross_model_shap",
            "top20_star",
            "top20_sentiment",
            "top20_gap",
        ],
        "insight_figures": [
            "decomp_star",
            "decomp_sentiment",
            "decomp_gap",
            "decomp_combined",
            "model_r2",
            "rating_distribution",
            "gap_distribution",
            "category_reliability",
            "reliability_count",
        ],
    }

    out = os.path.join(OUTPUT_DIR, "plot_feature_review_level.json")
    with open(out, "w") as f:
        json.dump(payload, f, default=str, indent=2)
    print(f"    Saved → {out}")
    return payload, comp_df


# ─────────────────────────────────────────────────────────────
# 11. HTML DASHBOARD
# ─────────────────────────────────────────────────────────────
def build_html(
    payload: dict,
    shap_star: pd.Series,
    shap_sent: pd.Series,
    shap_gap: pd.Series,
    decomp_star: pd.DataFrame,
    decomp_sent: pd.DataFrame,
    decomp_gap: pd.DataFrame,
    cat_rel: pd.DataFrame,
    joint: pd.DataFrame,
    feat_groups: dict,
    sv_star: np.ndarray,
    sv_sent: np.ndarray,
    sv_gap: np.ndarray,
    X_samp_star: pd.DataFrame,
    X_samp_sent: pd.DataFrame,
    X_samp_gap: pd.DataFrame,
    comp_df: pd.DataFrame | None = None,
):
    print("🎨  Building HTML dashboard...")
    meta = payload["meta"]
    if comp_df is None:
        comp_df = build_shap_comparison_table(
            shap_star=shap_star,
            shap_sent=shap_sent,
            shap_gap=shap_gap,
            feat_groups=feat_groups,
            top_n=18,
            rank_by="mean",
        )

    fig = make_subplots(
        rows=6, cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "histogram"}, {"type": "histogram"}],
            [{"type": "bar", "colspan": 2}, None, {"type": "bar"}],
            [{"type": "bar"}, {"type": "ternary"}, {"type": "violin"}],
        ],
        subplot_titles=(
            "", "", "",
            "Top 20 Features — Star Rating Model",
            "Top 20 Features — Sentiment Model",
            "Top 20 Features — Gap (Inflation) Model",
            "SHAP Variance Decomposition — Star",
            "SHAP Variance Decomposition — Sentiment",
            "SHAP Variance Decomposition — Gap",
            "Model R² Comparison",
            "Star Rating Distribution",
            "Sentiment-Star Gap Distribution",
            "Category Reliability Map (Pearson r: star vs sentiment)",
            "",
            "Reliability Label Distribution",
            "Grouped Cross-Model SHAP Comparison",
            "Ternary Role of Features Across Models",
            "Ridge-Style SHAP Distribution by Feature",
        ),
        row_heights=[0.05, 0.22, 0.16, 0.18, 0.20, 0.29],
        vertical_spacing=0.05,
        horizontal_spacing=0.06,
    )

    kpi_items = [
        ("Star Model R²", meta["model_star"]["r2"], ""),
        ("Sentiment Model R²", meta["model_sentiment"]["r2"], ""),
        ("Mean Rating Inflation (gap)", meta["mean_gap"], " stars"),
    ]
    kpi_x = [(0.0, 0.30), (0.35, 0.65), (0.70, 1.0)]
    for (title, val, suf), (x0, x1) in zip(kpi_items, kpi_x):
        color = C["star"] if "Star" in title else (C["sentiment"] if "Sentiment" in title else C["gap"])
        fig.add_trace(go.Indicator(
            mode="number", value=val,
            number={"font": {"size": 28, "color": color},
                    "valueformat": "+.4f" if "gap" in title.lower() else ".4f",
                    "suffix": suf},
            title={"text": f"<b>{title}</b>", "font": {"size": 11, "color": C["text"]}},
            domain={"x": [x0, x1], "y": [0.96, 1.0]},
        ))

    for col_i, (shap_s, color) in enumerate([
        (shap_star, C["star"]), (shap_sent, C["sentiment"]), (shap_gap, C["gap"]),
    ], start=1):
        top20 = shap_s.nlargest(20).sort_values()
        clean = pretty_feature_name(pd.Series(top20.index)).tolist()
        fig.add_trace(go.Bar(
            x=top20.values, y=clean, orientation="h", marker_color=color, marker_opacity=0.85,
            showlegend=False, hovertemplate="%{y}<br>Mean |SHAP|: %{x:.5f}<extra></extra>",
        ), row=2, col=col_i)
        fig.update_xaxes(title_text="Mean |SHAP|", row=2, col=col_i)

    group_colors = {
        "recipe": C["recipe"], "reviewer": C["reviewer"],
        "review_context": C["review_ctx"], "sentiment": C["sentiment"], "other": C["neutral"],
    }
    for col_i, decomp in enumerate([decomp_star, decomp_sent, decomp_gap], start=1):
        d = decomp.sort_values("pct_of_total", ascending=True)
        fig.add_trace(go.Bar(
            x=d["pct_of_total"],
            y=d["feature_class"].str.replace("_", " ").str.title(),
            orientation="h",
            marker_color=[group_colors.get(g, C["neutral"]) for g in d["feature_class"]],
            text=[f"{v:.1f}%" for v in d["pct_of_total"]],
            textposition="inside", textfont=dict(color="white", size=10),
            showlegend=False,
            hovertemplate="%{y}<br>%{x:.2f}% of total SHAP<extra></extra>",
        ), row=3, col=col_i)
        fig.update_xaxes(title_text="% of Total |SHAP|", row=3, col=col_i)

    models = ["Star Rating", "Sentiment", "Gap (Inflation)"]
    r2_vals = [meta["model_star"]["r2"], meta["model_sentiment"]["r2"], meta["model_gap"]["r2"]]
    fig.add_trace(go.Bar(
        x=models, y=r2_vals, marker_color=[C["star"], C["sentiment"], C["gap"]],
        text=[f"R²={v:.4f}" for v in r2_vals], textposition="inside",
        textfont=dict(color="white", size=11), showlegend=False,
        hovertemplate="%{x}<br>R²=%{y:.4f}<extra></extra>",
    ), row=4, col=1)
    fig.update_yaxes(title_text="R²", row=4, col=1, range=[0, max(r2_vals) * 1.3])

    fig.add_trace(go.Histogram(
        x=joint["Rating"].values, nbinsx=5, marker_color=C["star"], marker_opacity=0.8,
        showlegend=False, hovertemplate="Rating: %{x}<br>Count: %{y:,}<extra></extra>",
    ), row=4, col=2)
    fig.update_xaxes(title_text="Star Rating", row=4, col=2)
    fig.update_yaxes(title_text="Count", row=4, col=2)

    gap_sample = joint["gap"].sample(min(100_000, len(joint)), random_state=RANDOM_STATE)
    fig.add_trace(go.Histogram(
        x=gap_sample.values, nbinsx=40, marker_color=C["gap"], marker_opacity=0.8,
        showlegend=False, hovertemplate="Gap: %{x:.2f}<br>Count: %{y:,}<extra></extra>",
    ), row=4, col=3)
    fig.update_xaxes(title_text="Gap (Star − Sentiment)", row=4, col=3)
    fig.update_yaxes(title_text="Count", row=4, col=3)

    rel_colors_map = {"Reliable": C["recipe"], "Moderate": C["sentiment"], "Unreliable": C["reviewer"]}
    fig.add_trace(go.Bar(
        x=cat_rel["category"], y=cat_rel["pearson_r"],
        marker_color=[rel_colors_map.get(l, C["neutral"]) for l in cat_rel["reliability_label"]],
        text=[f"r={v:+.3f}<br>gap={g:+.2f}" for v, g in zip(cat_rel["pearson_r"], cat_rel["mean_gap"])],
        textposition="outside", textfont=dict(size=9, color=C["text"]),
        customdata=cat_rel[["n_reviews", "mean_star", "mean_sentiment", "reliability_label"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>Pearson r: %{y:.4f}<br>n reviews: %{customdata[0]:,}<br>"
            "Mean star: %{customdata[1]:.2f}<br>Mean sentiment: %{customdata[2]:.2f}<br>"
            "Reliability: %{customdata[3]}<extra></extra>"
        ),
        showlegend=False,
    ), row=5, col=1)
    fig.update_xaxes(title_text="Recipe Category", row=5, col=1, tickangle=-30)
    fig.update_yaxes(title_text="Pearson r (Star vs Sentiment)", row=5, col=1, range=[-0.1, 1.05])

    rel_counts = cat_rel["reliability_label"].value_counts()
    fig.add_trace(go.Bar(
        x=rel_counts.index.tolist(), y=rel_counts.values.tolist(),
        marker_color=[rel_colors_map.get(k, C["neutral"]) for k in rel_counts.index],
        text=rel_counts.values.tolist(), textposition="outside", textfont=dict(color=C["text"]),
        showlegend=False, hovertemplate="%{x}: %{y}<extra></extra>",
    ), row=5, col=3)
    fig.update_yaxes(title_text="# Categories", row=5, col=3)

    add_grouped_shap_bar(fig, comp_df, row=6, col=1)
    add_shap_ternary(fig, comp_df, row=6, col=2)
    add_shap_ridge_violin(fig, comp_df, sv_star, sv_sent, sv_gap, X_samp_star, X_samp_sent, X_samp_gap, row=6, col=3)

    fig.update_layout(
        title={
            "text": (
                "<b>Per-Review Sentiment Bridge Analysis</b><br>"
                "<sup>Star rating vs sentiment text signal — what drives each, and where do they diverge?</sup>"
            ),
            "x": 0.5, "xanchor": "center", "font": {"size": 17, "color": C["text"]},
        },
        height=3000,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font={"color": C["text"], "family": "Inter, sans-serif"},
        margin=dict(t=100, b=60, l=190, r=40),
        barmode="group",
    )
    axis_style = dict(
        gridcolor=C["grid"], zerolinecolor=C["grid"],
        linecolor="rgba(255,255,255,0.15)", tickfont={"size": 9},
    )
    for key in fig.layout:
        if key.startswith("xaxis") or key.startswith("yaxis"):
            fig.layout[key].update(axis_style)
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=10, color="#B0B8C8"))

    out = os.path.join(OUTPUT_DIR, "plot_feature_review_level.html")
    fig.write_html(out, include_plotlyjs="cdn", config={"displayModeBar": True}, full_html=True)
    print(f"    Saved → {out}")


# ─────────────────────────────────────────────────────────────
# 12. CSV EXPORTS
# ─────────────────────────────────────────────────────────────
def save_csvs(joint: pd.DataFrame, cat_rel: pd.DataFrame,
              decomp_star, decomp_sent, decomp_gap,
              shap_star, shap_sent, shap_gap,
              pred_star, pred_sent, pred_gap):
    print("📄  Saving CSVs...")

    out_df = joint[["ReviewId", "RecipeId", "AuthorId", "Rating", "sentiment_scaled", "gap"]].copy()
    out_df["pred_star"] = pred_star
    out_df["pred_sentiment"] = pred_sent
    out_df["pred_gap"] = pred_gap
    out_df["residual_star"] = out_df["Rating"] - out_df["pred_star"]
    out_df.to_csv(os.path.join(OUTPUT_DIR, "review_level_residuals.csv"), index=False)

    decomp_all = pd.concat([
        decomp_star.assign(model="star"),
        decomp_sent.assign(model="sentiment"),
        decomp_gap.assign(model="gap"),
    ])
    decomp_all.to_csv(os.path.join(OUTPUT_DIR, "shap_class_decomposition.csv"), index=False)

    pd.DataFrame({
        "shap_star": shap_star,
        "shap_sentiment": shap_sent,
        "shap_gap": shap_gap,
    }).sort_values("shap_star", ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "shap_importance_comparison.csv")
    )

    cat_rel.to_csv(os.path.join(OUTPUT_DIR, "category_reliability.csv"), index=False)
    print(f"    Saved 4 CSVs → {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n══════════════════════════════════════════════════════")
    print("  Per-Review Sentiment Bridge Pipeline")
    print("══════════════════════════════════════════════════════\n")

    recipes, reviews = load_data()

    # recipes, reviews = recipes.head(10000), reviews.head(30000)

    recipes, cat_cols = clean_recipes(recipes)
    reviews = clean_reviews(reviews, recipes)
    reviews = score_sentiment(reviews)
    reviews = engineer_reviewer_features(reviews)
    joint = build_joint(reviews, recipes)

    X_base, X_with_sent, y_star, y_sent, y_gap, feat_groups = build_feature_matrix(joint, cat_cols)

    model_star, r2_star, rmse_star, pred_star = train_lgb(X_with_sent, y_star, "Star Rating")
    model_sent, r2_sent, rmse_sent, pred_sent = train_lgb(X_base, y_sent, "Sentiment")
    model_gap, r2_gap, rmse_gap, pred_gap = train_lgb(X_with_sent, y_gap, "Gap")

    sv_star, shap_star, X_samp_star = compute_shap(model_star, X_with_sent, "Star")
    sv_sent, shap_sent, X_samp_sent = compute_shap(model_sent, X_base, "Sentiment")
    sv_gap, shap_gap, X_samp_gap = compute_shap(model_gap, X_with_sent, "Gap")

    decomp_star = decompose_shap_by_group(shap_star, feat_groups)
    decomp_sent = decompose_shap_by_group(shap_sent, feat_groups)
    decomp_gap = decompose_shap_by_group(shap_gap, feat_groups)

    print("\n  SHAP Variance Decomposition:")
    for label, decomp in [("Star", decomp_star), ("Sentiment", decomp_sent), ("Gap", decomp_gap)]:
        print(f"\n  [{label}]")
        print(decomp.to_string(index=False))

    cat_rel = category_reliability(joint)

    payload, comp_df = save_json(
        r2_star, rmse_star, r2_sent, rmse_sent, r2_gap, rmse_gap,
        shap_star, shap_sent, shap_gap,
        feat_groups,
        decomp_star, decomp_sent, decomp_gap,
        cat_rel, joint,
        sv_star, sv_sent, sv_gap,
        X_samp_star, X_samp_sent, X_samp_gap,
    )

    build_html(
        payload, shap_star, shap_sent, shap_gap,
        decomp_star, decomp_sent, decomp_gap,
        cat_rel, joint,
        feat_groups,
        sv_star, sv_sent, sv_gap,
        X_samp_star, X_samp_sent, X_samp_gap,
        comp_df=comp_df,
    )

    save_csvs(
        joint, cat_rel,
        decomp_star, decomp_sent, decomp_gap,
        shap_star, shap_sent, shap_gap,
        pred_star, pred_sent, pred_gap,
    )

    print("\n══════════════════════════════════════════════════════")
    print("  Final Summary")
    print("══════════════════════════════════════════════════════")
    print(f"\n  Model R²:")
    print(f"    Star Rating : {r2_star:.4f}")
    print(f"    Sentiment   : {r2_sent:.4f}")
    print(f"    Gap         : {r2_gap:.4f}")
    print(f"\n  Rating Inflation:")
    print(f"    Mean gap (star − sentiment) : {joint['gap'].mean():+.4f} stars")
    print(f"    Positive gap (inflation)    : {(joint['gap'] > 0).mean()*100:.1f}% of reviews")
    print(f"\n  Category Reliability:")
    print(f"    Most reliable   : {cat_rel.iloc[-1]['category']} (r={cat_rel.iloc[-1]['pearson_r']:+.3f})")
    print(f"    Least reliable  : {cat_rel.iloc[0]['category']} (r={cat_rel.iloc[0]['pearson_r']:+.3f})")
    print(f"\n✅  Pipeline complete.")
    print(f"   Output: {os.path.abspath(OUTPUT_DIR)}")
    print("   Files:")
    print("     plot_feature_review_level.html    ← dashboard")
    print("     plot_feature_review_level.json    ← pre-computed web app data + standalone figure specs")
    print("     review_level_residuals.csv        ← per-review predictions")
    print("     shap_class_decomposition.csv      ← variance decomposition")
    print("     shap_importance_comparison.csv    ← SHAP per feature × 3 models")
    print("     category_reliability.csv          ← category reliability map")
    print()


if __name__ == "__main__":
    main()
