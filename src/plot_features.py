#!/usr/bin/env python3
"""
Functions to build the feature importance figures using LGBM and VADER.
The figure's data and metadata are exported as a standalone JSON.
No processed data being output!
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
from scipy.stats import pearsonr, spearmanr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

DB_PATH = "./data/tables/food_recipe.db"
OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_REVIEWER_REVIEWS = 3
MIN_RECIPE_REVIEWS = 5
TOP_N_CATEGORIES = 30
RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 100_000

LGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=-1,
)

C = {
    "star": "#4C9BE8",
    "sentiment": "#F28B30",
    "gap": "#9B59B6",
    "recipe": "#2ECC71",
    "reviewer": "#E74C3C",
    "review_ctx": "#1612F3",
    "neutral": "#7F8C8D",
    "bg": "#FFFFFF",
    "text": "#000000",
    "grid": "rgba(255,255,255,0.07)",
    "zero": "rgba(255,80,80,0.55)",
}

TERNARY_BG = "rgba(173, 216, 250, 0.28)"
TERNARY_LINE = "#1A5A8A"
TERNARY_GRID = "rgba(26,  90, 138, 0.20)"

GROUP_COLOR_MAP = {
    "recipe": C["recipe"],
    "reviewer": C["reviewer"],
    "review_context": C["review_ctx"],
    "sentiment": C["sentiment"],
    "other": C["neutral"],
}

GROUP_LABELS = {
    "recipe": "Recipe",
    "reviewer": "Reviewer",
    "review_context": "Review Context",
    "sentiment": "Sentiment",
    "other": "Other",
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

APP_FIGURE_KEYS = {
    "feature_importance_main": "ternary_feature_role",
    "feature_reliability": "category_reliability",
    "feature_ridge": "ridge_shap_distribution",
    "feature_grouped": "grouped_cross_model_shap",
    "feature_rating_distribution": "rating_distribution",
    "feature_decomp_combined": "decomp_combined",
}

# ##################
# Helpers
# ##################
def hex_to_rgba(h: str, a: float = 1.0) -> str:
    """Convert a hex color to plotly rgba so we can control opacity separately"""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def pretty_feature_name(s: pd.Series) -> pd.Series:
    """Clean internal feature names into labels that are easier to read in the figures"""
    return (
        s.astype(str)
        .str.replace("cat_", "", regex=False)
        .str.replace("log_", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )


def _count_list_items(series: pd.Series) -> pd.Series:
    """Count list-like entries stored as strings such as ingredient or keyword arrays"""
    def _count(val):
        if pd.isna(val) or str(val).strip() == "":
            return 0
        val = re.sub(r"[\[\]\"']", "", str(val))
        return len([x for x in val.split(",") if x.strip()])
    return series.apply(_count)


def _count_steps(series: pd.Series) -> pd.Series:
    """Estimate the number of meaningful instruction steps from free-text instructions"""
    def _count(val):
        if pd.isna(val) or str(val).strip() == "":
            return 0
        val = re.sub(r"[\[\]\"']", "", str(val))
        # split on sentence boundaries or pipe separators because both appear in scraped instruction text
        parts = re.split(r"(?<=[.!?])\s+|\|", val)
        # require a minimum length so tiny fragments do not become fake steps
        return max(1, len([p for p in parts if len(p.strip()) > 10]))
    return series.apply(_count)


def _technique_score(series: pd.Series) -> pd.Series:
    """Count how often technique-related cooking verbs appear in the instruction text"""
    pattern = "|".join(TECHNIQUE_KEYWORDS)
    return series.fillna("").str.lower().str.count(pattern)


def apply_single_figure_layout(
    fig: go.Figure,
    # title: str = "",
    *,
    height: int = 600,
    showlegend: bool = False,
    barmode: str | None = None,
) -> go.Figure:
    fig.update_layout(
        title=None,
        margin=dict(t=70, b=50, l=90, r=30),
        height=height,
        showlegend=showlegend,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
    """Convert a plotly figure into plain json-safe data for the web app"""
    return json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))

# ##################
# Feature Pipeline
# ##################
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """load the raw recipe and review tables from sqlite"""
    print("Loading data...")
    conn = sqlite3.connect(DB_PATH)
    recipes = pd.read_sql("SELECT * FROM recipes", conn)
    reviews = pd.read_sql("SELECT * FROM reviews", conn)
    conn.close()
    print(f"    Recipes : {len(recipes):,}")
    print(f"    Reviews : {len(reviews):,}")
    return recipes, reviews


def clean_recipes(recipes: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean recipe fields and engineer recipe-level predictors for the models"""
    print("Cleaning recipes...")
    df = recipes.copy()

    for col in NUTRITION_COLS + TIME_COLS + ["AggregatedRating", "ReviewCount", "RecipeServings"]:
        if col in df.columns:
            # coerce invalid strings to nan so later numeric operations behave predictably
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ReviewCount"] = df["ReviewCount"].fillna(0)
    # very low-review recipes are removed so recipe-level signals are not dominated by extremely sparse examples
    df = df[df["ReviewCount"] >= MIN_RECIPE_REVIEWS].copy()

    for col in NUTRITION_COLS + TIME_COLS:
        if col in df.columns:
            # clip extreme outliers because a few pathological values can dominate tree splits and feature scales
            cap = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=cap)

    for col in NUTRITION_COLS:
        df[col] = df[col].fillna(df[col].median())

    for col in TIME_COLS:
        # time variables are strongly right-skewed so log1p makes them easier for the model to use
        df[f"log_{col}"] = np.log1p(df[col].fillna(0))

    df["ingredient_count"] = _count_list_items(df["RecipeIngredientParts"])
    df["keyword_count"] = _count_list_items(df["Keywords"])
    df["instruction_steps"] = _count_steps(df["RecipeInstructions"])
    df["instruction_length"] = df["RecipeInstructions"].fillna("").apply(
        lambda x: len(re.sub(r"[\[\]\"']", "", str(x)))
    )
    df["technique_score"] = _technique_score(df["RecipeInstructions"])

    df["calorie_density"] = df["Calories"] / df["RecipeServings"].clip(lower=1).fillna(1)
    df["protein_calorie_ratio"] = df["ProteinContent"] / df["Calories"].clip(lower=1)
    df["fat_sugar_combo"] = df["FatContent"] * df["SugarContent"]
    df["fat_protein_combo"] = df["FatContent"] * df["ProteinContent"]

    df["RecipeCategory"] = df["RecipeCategory"].fillna("Unknown").str.strip()
    top_cats = df["RecipeCategory"].value_counts().nlargest(50).index
    df["RecipeCategory_clean"] = df["RecipeCategory"].where(
        df["RecipeCategory"].isin(top_cats), other="Other"
    )
    # collapse rare categories into other to keep the dummy matrix from exploding in width
    cat_dummies = pd.get_dummies(df["RecipeCategory_clean"], prefix="cat", drop_first=True)
    df = pd.concat([df, cat_dummies], axis=1)

    print(f"    Recipes after filter: {len(df):,}")
    return df, list(cat_dummies.columns)


def clean_reviews(reviews: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    """Clean review rows and keep only reviews that still match the filtered recipe table"""
    print("Cleaning reviews...")
    df = reviews.copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating", "RecipeId", "AuthorId"])
    # ratings outside the expected star range are treated as invalid records
    df = df[df["Rating"].between(1, 5)]

    valid_recipe_ids = set(recipes["RecipeId"].astype(str))
    df["RecipeId"] = df["RecipeId"].astype(str)
    # this keeps the review table aligned with the already filtered recipe table
    df = df[df["RecipeId"].isin(valid_recipe_ids)]

    for col in ["DateSubmitted", "DateModified"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"    Reviews after filter: {len(df):,}")
    return df


def score_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    """Score each review with vader and remap the compound score onto the 1 to 5 star scale"""
    print("Scoring sentiment (VADER)...")
    analyzer = SentimentIntensityAnalyzer()
    texts = reviews["Review"].fillna("").astype(str).tolist()
    scores = [analyzer.polarity_scores(t)["compound"] for t in texts] # performing the sentiment analyses -> ouputting the sentiment score

    reviews = reviews.copy()
    reviews["sentiment_raw"] = scores
    # remapping to the star scale makes the sentiment and rating directly comparable when computing the gap
    reviews["sentiment_scaled"] = (np.array(scores) + 1) / 2 * 4 + 1
    print(f"    Sentiment scored for {len(reviews):,} reviews.")
    return reviews


def engineer_reviewer_features(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    ((This docstring is created by ChatGPT))
    Construct reviewer-level behavioral features using a leave-one-out strategy to avoid target leakage.

    This function aggregates each reviewer's historical rating behavior and derives features such as
    mean rating, rating variability, and activity level. All statistics are computed in a
    leave-one-out manner, meaning the current review is excluded from its own feature calculation.
    This prevents leakage where the model could trivially learn the target from features derived using
    the same data point.

    Features created:
    - reviewer_total_reviews: total number of reviews written by the reviewer,
    - reviewer_loo_mean: mean rating of the reviewer excluding the current review,
    - reviewer_loo_std: standard deviation of ratings excluding the current review,
    - reviewer_log_reviews: log-scaled review count to reduce skew in activity levels,
    """
    print("Engineering reviewer features (leave-one-out)...")
    rev_stats = (
        reviews.groupby("AuthorId")["Rating"]
        .agg(
            reviewer_total_reviews="count",
            reviewer_sum_rating="sum",
            reviewer_sum_sq_rating=lambda x: (x**2).sum(),
        )
        .reset_index()
    )

    df = reviews.merge(rev_stats, on="AuthorId", how="left")
    # leave-one-out averages prevent a review from making its own target easier to predict
    df["reviewer_loo_mean"] = (
        (df["reviewer_sum_rating"] - df["Rating"])
        / (df["reviewer_total_reviews"] - 1).clip(lower=1)
    )

    n = df["reviewer_total_reviews"]
    s = df["reviewer_sum_rating"]
    sq = df["reviewer_sum_sq_rating"]
    r = df["Rating"]
    loo_n = (n - 1).clip(lower=1)
    loo_s = s - r
    loo_sq = sq - r**2
    # compute leave-one-out variance from aggregated sums instead of an expensive per-row group recalculation
    loo_var = (loo_sq - loo_s**2 / loo_n.clip(lower=1)) / (loo_n - 1).clip(lower=1)
    df["reviewer_loo_std"] = np.sqrt(loo_var.clip(lower=0))
    df["reviewer_log_reviews"] = np.log1p(df["reviewer_total_reviews"])
    df["reviewer_loo_std"] = df["reviewer_loo_std"].fillna(0)
    df["reviewer_loo_mean"] = df["reviewer_loo_mean"].fillna(df["Rating"].mean())
    # extremely sparse reviewers do not have stable history-based features so we drop them
    df = df[df["reviewer_total_reviews"] >= MIN_REVIEWER_REVIEWS]

    print(f"    Reviews after reviewer filter: {len(df):,}")
    return df


def build_joint(reviews: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    """
    ((This docstring is created by ChatGPT))
    Merge review-level and recipe-level datasets and engineer contextual interaction features.

    This function constructs a unified dataframe where each row represents a single review
    enriched with both recipe attributes and contextual signals about when and how the review
    was written.

    Engineered feature groups:
    - time-based:
        - days_since_pub: time difference between review submission and recipe publication
        - log_days_since_pub: log-transformed version to reduce skew
    - lifecycle / sequence:
        - review_position: order of the review within a recipe
        - log_review_position: log-transformed version
    - text:
        - review_length: character length of the review text
        - log_review_length: log-transformed version
    - reviewer-category interaction:
        - reviewer_category_familiarity: number of prior reviews by the same author in the same category
        - log_cat_familiarity: log-transformed version
    - target engineering:
        - gap: difference between rating and sentiment

    Notes:
    - only a slim subset of recipe columns is used to keep the merge efficient
    - log1p is used to stabilize highly skewed distributions
    - chronological sorting ensures that cumulative counts (e.g. familiarity, position) are valid
    - missing publication dates are handled by defaulting to zero
    """
    print("Joining reviews with recipe features...")
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

    # keep only the fields we need so the merge stays lighter and easier to reason about
    recipes_slim = recipes[["RecipeId"] + recipe_feature_cols].copy()
    recipes_slim["RecipeId"] = recipes_slim["RecipeId"].astype(str)
    joint = reviews.merge(recipes_slim, on="RecipeId", how="inner")

    if "DatePublished" in joint.columns:
        joint["DatePublished"] = pd.to_datetime(joint["DatePublished"], errors="coerce")
        joint["days_since_pub"] = (
            joint["DateSubmitted"] - joint["DatePublished"]
        ).dt.days.clip(lower=0)
        # timing since publication is highly skewed so we log-transform it just like the time fields
        joint["log_days_since_pub"] = np.log1p(joint["days_since_pub"].fillna(0))
    else:
        joint["log_days_since_pub"] = 0.0

    joint = joint.sort_values(["RecipeId", "DateSubmitted"])
    # order reviews within each recipe so we can capture whether a review came early or late in the lifecycle
    joint["review_position"] = joint.groupby("RecipeId").cumcount() + 1
    joint["log_review_position"] = np.log1p(joint["review_position"])

    joint["review_length"] = joint["Review"].fillna("").str.len()
    joint["log_review_length"] = np.log1p(joint["review_length"])

    # this counts how many earlier reviews the same author wrote in the same category
    cat_familiarity = (
        joint.groupby(["AuthorId", "RecipeCategory"])
        .cumcount()
        .rename("reviewer_category_familiarity")
    )
    joint["reviewer_category_familiarity"] = cat_familiarity.values
    joint["log_cat_familiarity"] = np.log1p(joint["reviewer_category_familiarity"])
    # positive gap means the written sentiment sounds less positive than the assigned star rating
    joint["gap"] = joint["Rating"] - joint["sentiment_scaled"]

    print(f"    Joint dataframe: {len(joint):,} rows")
    return joint


def build_feature_matrix(joint: pd.DataFrame, cat_cols: list[str]):
    """
    Construct model-ready feature matrices and organize features into conceptual groups.
    Feature groups:
    - recipe_features:
        intrinsic properties of the recipe (nutrition, time, structure, category indicators)
    - reviewer_features:
        behavioral statistics of the reviewer (leave-one-out mean/std, activity, familiarity)
    - review_context_features:
        situational context of the review (timing, position, length)

    Outputs:
    - X_base:
        feature matrix without sentiment (used for predicting rating and gap)
    - X_with_sentiment:
        same as X_base but includes sentiment_scaled as an additional predictor
        (used for modeling interactions between sentiment and other features)
    - y_star:
        target variable for rating prediction
    - y_sentiment:
        target variable for sentiment prediction
    - y_gap:
        target variable for disagreement between rating and sentiment
    - feat_groups:
        dictionary mapping each feature to its conceptual group

    Notes:
    - median imputation is used as a simple, robust default suitable for tree-based models,
    - constant columns are removed since they provide no predictive signal and can distort importance metrics,
    """
    print("Building feature matrix...")
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
        "reviewer_loo_mean", "reviewer_loo_std", "reviewer_log_reviews", "log_cat_familiarity",
    ]
    review_ctx_features = ["log_days_since_pub", "log_review_position", "log_review_length"]

    all_features = (
        [f for f in recipe_features if f in joint.columns]
        + [f for f in reviewer_features if f in joint.columns]
        + [f for f in review_ctx_features if f in joint.columns]
    )

    for col in all_features:
        if joint[col].dtype in [np.float64, np.int64, float, int]:
            # median imputation is a simple robust default for tree models and avoids dropping rows here
            joint[col] = joint[col].fillna(joint[col].median())

    X_base = joint[all_features].astype(float)
    # constant columns carry no signal and can confuse downstream summaries so we remove them
    X_base = X_base.loc[:, X_base.nunique() > 1]

    X_with_sentiment = X_base.copy()
    if "sentiment_scaled" in joint.columns:
        X_with_sentiment["sentiment_scaled"] = joint["sentiment_scaled"].values

    y_star = joint["Rating"].values.astype(float)
    y_sentiment = joint["sentiment_scaled"].values.astype(float)
    y_gap = joint["gap"].values.astype(float)

    feat_groups: dict[str, str] = {}
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

# ##################
# Modeling and summaries
# ##################

def train_lgb(X: pd.DataFrame, y: np.ndarray, label: str):
    """Train a lightgbm regressor and report whole-dataset fit metrics for the exported summary"""
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    print(f"Training LightGBM [{label}]...")
    # keep a validation split for early stopping so the large tree ensemble does not overtrain unnecessarily
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    # predictions are computed on the full matrix because the exported summary is descriptive rather than a strict benchmark
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"    [{label}] R²={r2:.4f}  RMSE={rmse:.4f}  best_iter={model.best_iteration_}")
    return model, r2, rmse, y_pred

def compute_shap(model, X: pd.DataFrame, label: str, sample_size: int = SHAP_SAMPLE_SIZE):
    """Compute SHAP values on a capped sample so interpretation stays feasible on large datasets"""
    print(f"Computing SHAP [{label}] on {min(len(X), sample_size):,} samples...")
    if len(X) > sample_size:
        # shap can be expensive so we subsample deterministically for reproducible figure content
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    # mean absolute shap is used because we care about overall influence strength regardless of sign
    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_sample.columns, name=f"shap_{label}")
    print(f"    Top feature: {mean_abs.idxmax()} ({mean_abs.max():.5f})")
    return shap_vals, mean_abs, X_sample

def decompose_shap_by_group(mean_abs_shap: pd.Series, feat_groups: dict[str, str]) -> pd.DataFrame:
    """Sum mean absolute SHAP values within each feature family for high-level decomposition plots"""
    rows = []
    total = mean_abs_shap.sum()
    group_sums: dict[str, float] = {}
    for feat, shap_val in mean_abs_shap.items():
        g = feat_groups.get(feat, "other")
        group_sums[g] = group_sums.get(g, 0) + shap_val

    for g, s in group_sums.items():
        rows.append({
            "feature_class": g,
            "total_shap": round(s, 6),
            "pct_of_total": round(s / total * 100, 2) if total > 0 else 0,
        })
    return pd.DataFrame(rows).sort_values("total_shap", ascending=False)

def build_shap_comparison_table(
    shap_star: pd.Series,
    shap_sent: pd.Series,
    shap_gap: pd.Series,
    feat_groups: dict[str, str],
    top_n: int = 80,
    rank_by: str = "mean",
) -> pd.DataFrame:
    """Combine SHAP summaries from the three models into one comparison table"""
    df = pd.concat(
        [
            shap_star.rename("star"),
            shap_sent.rename("sentiment"),
            shap_gap.rename("gap"),
        ],
        axis=1,
    ).fillna(0.0)

    df["raw_sum"] = df[["star", "sentiment", "gap"]].sum(axis=1)
    norm = df[["star", "sentiment", "gap"]].copy()
    for c in ["star", "sentiment", "gap"]:
        s = norm[c].sum()
        # normalize within each model so models with larger absolute shap scales can still be compared visually
        norm[c] = norm[c] / s if s > 0 else 0.0

    df["star_norm"] = norm["star"]
    df["sentiment_norm"] = norm["sentiment"]
    df["gap_norm"] = norm["gap"]

    if rank_by == "mean":
        df["rank_score"] = df[["star_norm", "sentiment_norm", "gap_norm"]].mean(axis=1)
    elif rank_by == "max":
        df["rank_score"] = df[["star_norm", "sentiment_norm", "gap_norm"]].max(axis=1)
    else:
        df["rank_score"] = df["raw_sum"]

    df["feature_group"] = [feat_groups.get(f, "other") for f in df.index]

    # ternary coordinates show how each feature splits its influence across the three prediction tasks
    tri_sum = df[["star", "sentiment", "gap"]].sum(axis=1).replace(0, np.nan)
    df["tri_star"] = df["star"] / tri_sum
    df["tri_sentiment"] = df["sentiment"] / tri_sum
    df["tri_gap"] = df["gap"] / tri_sum

    df = (
        df.sort_values("rank_score", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return df

def category_reliability(joint: pd.DataFrame, top_n: int = TOP_N_CATEGORIES) -> pd.DataFrame:
    """Measure how strongly written sentiment and star ratings agree within each recipe category"""
    print(f"\nComputing category reliability map (top {top_n})...")
    top_cats = joint["RecipeCategory"].value_counts().nlargest(top_n).index.tolist()

    results = []
    for cat in top_cats:
        sub = joint[joint["RecipeCategory"] == cat]
        # skip tiny categories because correlation estimates are too noisy there
        if len(sub) < 50:
            continue

        r_pearson, _ = pearsonr(sub["Rating"], sub["sentiment_scaled"])
        r_spearman, _ = spearmanr(sub["Rating"], sub["sentiment_scaled"])
        mean_gap = sub["gap"].mean()
        std_gap = sub["gap"].std()
        mean_star = sub["Rating"].mean()
        mean_sent = sub["sentiment_scaled"].mean()

        # the thresholds are heuristic labels meant for interpretation in the web app rather than formal inference
        label = "Reliable" if r_pearson >= 0.5 else "Moderate" if r_pearson >= 0.3 else "Unreliable"
        results.append({
            "category": cat,
            "n_reviews": len(sub),
            "pearson_r": round(r_pearson, 4),
            "spearman_rho": round(r_spearman, 4),
            "mean_gap": round(mean_gap, 4),
            "std_gap": round(std_gap, 4),
            "mean_star": round(mean_star, 4),
            "mean_sentiment": round(mean_sent, 4),
            "reliability_label": label,
        })
        print(f"    {cat:<35} n={len(sub):>7,}  r={r_pearson:+.3f}  gap={mean_gap:+.3f}")

    return pd.DataFrame(results).sort_values("pearson_r")

# ##################
# For app figures only
# ##################

def add_grouped_shap_bar(fig, comp_df: pd.DataFrame, row: int, col: int):
    """Add grouped bars comparing normalized shap importance across the three models"""
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
    """Add the ternary scatter that shows each feature's role split across rating sentiment and gap"""
    d = comp_df.copy()
    d["feature_pretty"] = pretty_feature_name(d["feature"])

    total_raw = d[["star", "sentiment", "gap"]].sum(axis=1)
    denom = max(float(total_raw.max()), 1e-12)
    # marker size encodes total raw importance so both direction and magnitude are visible at once
    d["marker_size"] = 12 + 36 * (total_raw / denom)

    for grp in d["feature_group"].unique():
        subset = d[d["feature_group"] == grp]
        color = GROUP_COLOR_MAP.get(grp, C["neutral"])
        label = GROUP_LABELS.get(grp, grp.title())

        fig.add_trace(
            go.Scatterternary(
                a=subset["tri_star"],
                b=subset["tri_sentiment"],
                c=subset["tri_gap"],
                mode="markers",
                name=label,
                legendgroup=grp,
                showlegend=True,
                marker=dict(
                    size=subset["marker_size"].tolist(),
                    color=color,
                    opacity=1.0,
                    line=dict(color="black", width=1.8),
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
                    "Group: %{customdata[1]}<br><br>"
                    "<b>Raw mean |SHAP|</b><br>"
                    "Rating   : %{customdata[2]:.5f}<br>"
                    "Sentiment: %{customdata[3]:.5f}<br>"
                    "Gap       : %{customdata[4]:.5f}<br><br>"
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

    axis_common = dict(
        tickfont=dict(size=11, color=TERNARY_LINE),
        linecolor=TERNARY_LINE,
        linewidth=3,
        gridcolor=TERNARY_GRID,
        gridwidth=1.5,
        tickcolor=TERNARY_LINE,
        showgrid=True,
        ticks="outside",
        dtick=0.2,
        min=0,
    )

    fig.update_ternaries(
        bgcolor=TERNARY_BG,
        aaxis=dict(**axis_common, title=dict(text="<b>Rating</b>", font=dict(size=14, color="black", family="Inter, sans-serif"))),
        baxis=dict(**axis_common, title=dict(text="<b>Sentiment</b>", font=dict(size=14, color="black", family="Inter, sans-serif"))),
        caxis=dict(**axis_common, title=dict(text="<b>Gap</b>", font=dict(size=14, color="black", family="Inter, sans-serif"))),
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
    """Add violin traces that approximate a ridge plot for per-sample shap distributions"""
    d = comp_df.copy()
    d["feature_pretty"] = pretty_feature_name(d["feature"])
    d = d.sort_values("rank_score", ascending=True)

    idx_star = {f: i for i, f in enumerate(X_samp_star.columns)}
    idx_sent = {f: i for i, f in enumerate(X_samp_sent.columns)}
    idx_gap = {f: i for i, f in enumerate(X_samp_gap.columns)}

    # cap the number of samples to keep the figure responsive while still showing the overall distribution shape
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

def make_combined_decomposition_figure(
    decomp_star: pd.DataFrame,
    decomp_sent: pd.DataFrame,
    decomp_gap: pd.DataFrame,
) -> go.Figure:
    """Build the three-panel decomposition figure for rating sentiment and gap"""
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

def make_rating_histogram(joint: pd.DataFrame) -> go.Figure:
    """Build a simple histogram of the observed star ratings"""
    fig = go.Figure(go.Histogram(
        x=joint["Rating"].values, nbinsx=5, marker_color=C["star"], marker_opacity=0.8,
        hovertemplate="Rating: %{x}<br>Count: %{y:,}<extra></extra>", showlegend=False,
    ))
    fig.update_xaxes(title_text="Star Rating")
    fig.update_yaxes(title_text="Count")
    return apply_single_figure_layout(fig, "Star Rating Distribution", height=450)

def make_category_reliability_figure(cat_rel: pd.DataFrame) -> go.Figure:
    """Build the category reliability bar chart from the precomputed correlation table"""
    d = cat_rel.copy().sort_values("pearson_r", ascending=False).reset_index(drop=True)
    d["category_pretty"] = d["category"].astype(str).str.replace("_", " ", regex=False)

    r_min = float(d["pearson_r"].min())
    r_max = float(d["pearson_r"].max())
    # guard against a zero range so color normalization never divides by zero
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
        marker=dict(color=bar_colors, line=dict(color="black", width=1.4)),
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

    fig.add_vline(x=0, line_width=1.4, line_color="rgba(0,0,0,0.65)")
    fig.update_layout(
        margin=dict(t=30, b=30, l=55, r=55),
        height=max(520, 72 + 46 * len(d)),
        bargap=0.28,
        font=dict(size=16, color=C["text"], family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        title_text="Pearson 𝑟 (star rating vs sentiment)",
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
        showline=False,
        ticks="",
        showticklabels=False,
        tickfont=dict(size=15, color=C["text"]),
        automargin=True,
    )
    return fig

def make_grouped_shap_figure(comp_df: pd.DataFrame) -> go.Figure:
    """Wrapper for the grouped shap comparison figure"""
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "bar"}]])
    add_grouped_shap_bar(fig, comp_df, row=1, col=1)
    return apply_single_figure_layout(fig, "Grouped Cross-Model SHAP Comparison", height=750, showlegend=True, barmode="group")

def make_ternary_figure(comp_df: pd.DataFrame) -> go.Figure:
    """Wrapper for the ternary feature-role figure"""
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "ternary"}]])
    add_shap_ternary(fig, comp_df, row=1, col=1)
    fig.update_layout(
        title=dict(x=0.5, xanchor="center", font=dict(size=16, color=C["text"], family="Inter, sans-serif")),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.0)",
            bordercolor="rgba(0,0,0,0.0)",
            borderwidth=1,
            font=dict(size=11, color=C["text"]),
            itemsizing="constant",
            x=0.65, y=0.95, xanchor="left",
        ),
        height=700,
        margin=dict(t=30, b=30, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="Inter, sans-serif"),
    )
    return fig

def make_ridge_figure(
    comp_df: pd.DataFrame,
    sv_star: np.ndarray,
    sv_sent: np.ndarray,
    sv_gap: np.ndarray,
    X_samp_star: pd.DataFrame,
    X_samp_sent: pd.DataFrame,
    X_samp_gap: pd.DataFrame,
) -> go.Figure:
    """Wrapper for the ridge-style shap distribution figure"""
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "violin"}]])
    add_shap_ridge_violin(fig, comp_df, sv_star, sv_sent, sv_gap, X_samp_star, X_samp_sent, X_samp_gap, row=1, col=1)
    return apply_single_figure_layout(fig, "Ridge-Style SHAP Distribution by Feature", height=850, showlegend=False)

# ##################
# Exporting
# ##################

def build_app_payload(
    r2_star,
    rmse_star,
    r2_sent,
    rmse_sent,
    r2_gap,
    rmse_gap,
    shap_star: pd.Series,
    shap_sent: pd.Series,
    shap_gap: pd.Series,
    feat_groups: dict[str, str],
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
) -> dict:
    """Assemble the final json payload consumed by the static web app"""
    print("\nBuilding app payload...")

    meta = {
        "model_star": {"r2": round(r2_star, 4), "rmse": round(rmse_star, 4)},
        "model_sentiment": {"r2": round(r2_sent, 4), "rmse": round(rmse_sent, 4)},
        "model_gap": {"r2": round(r2_gap, 4), "rmse": round(rmse_gap, 4)},
        "n_reviews": int(len(joint)),
        "n_recipes": int(joint["RecipeId"].nunique()),
        "n_reviewers": int(joint["AuthorId"].nunique()),
        "mean_gap": round(float(joint["gap"].mean()), 4),
        "mean_star": round(float(joint["Rating"].mean()), 4),
        "mean_sentiment": round(float(joint["sentiment_scaled"].mean()), 4),
    }

    # keep only the top features for app figures so the payload stays lighter and the plots remain readable
    comp_df = build_shap_comparison_table(
        shap_star=shap_star,
        shap_sent=shap_sent,
        shap_gap=shap_gap,
        feat_groups=feat_groups,
        top_n=15,
        rank_by="mean",
    )

    standalone_figures = {
        "ternary_feature_role": figure_to_payload(make_ternary_figure(comp_df)),
        "category_reliability": figure_to_payload(make_category_reliability_figure(cat_rel)),
        "ridge_shap_distribution": figure_to_payload(
            make_ridge_figure(comp_df, sv_star, sv_sent, sv_gap, X_samp_star, X_samp_sent, X_samp_gap)
        ),
        "grouped_cross_model_shap": figure_to_payload(make_grouped_shap_figure(comp_df)),
        "rating_distribution": figure_to_payload(make_rating_histogram(joint)),
        "decomp_combined": figure_to_payload(make_combined_decomposition_figure(decomp_star, decomp_sent, decomp_gap)),
    }

    return {
        "meta": meta,
        "cross_model_shap_comparison": comp_df.to_dict(orient="records"),
        "category_reliability": cat_rel.to_dict(orient="records"),
        "standalone_figures": standalone_figures,
        "webapp_panels": {
            "feature_importance_main": "ternary_feature_role",
            "feature_importance_alt": [
                "ridge_shap_distribution",
                "grouped_cross_model_shap",
            ],
            "insight_figures": [
                "decomp_combined",
                "rating_distribution",
                "category_reliability",
            ],
        },
    }

def save_app_json(payload: dict) -> str:
    """write the final payload to disk for later use by the web app"""
    out = os.path.join(OUTPUT_DIR, "plot_features.json")
    with open(out, "w") as f:
        json.dump(payload, f, default=str, indent=2)
    print(f"    Saved → {out}")
    return out

# ##################
# MAIN
# ##################
def main():
    """run the end-to-end feature importance pipeline and export the web app payload"""
    print("\n============================")
    print("  Feature Importance Pipeline")
    print("=================================\n")

    recipes, reviews = load_data()
    # useful for quick debugging runs when the full dataset is too slow
    # recipes, reviews = recipes.head(10000), reviews.head(50000)
    recipes, cat_cols = clean_recipes(recipes)
    reviews = clean_reviews(reviews, recipes)
    reviews = score_sentiment(reviews)
    reviews = engineer_reviewer_features(reviews)
    joint = build_joint(reviews, recipes)

    X_base, X_with_sent, y_star, y_sent, y_gap, feat_groups = build_feature_matrix(joint, cat_cols)

    model_star, r2_star, rmse_star, _ = train_lgb(X_with_sent, y_star, "Star Rating")
    model_sent, r2_sent, rmse_sent, _ = train_lgb(X_base, y_sent, "Sentiment")
    model_gap, r2_gap, rmse_gap, _ = train_lgb(X_with_sent, y_gap, "Gap")

    sv_star, shap_star, X_samp_star = compute_shap(model_star, X_with_sent, "Star")
    sv_sent, shap_sent, X_samp_sent = compute_shap(model_sent, X_base, "Sentiment")
    sv_gap, shap_gap, X_samp_gap = compute_shap(model_gap, X_with_sent, "Gap")

    decomp_star = decompose_shap_by_group(shap_star, feat_groups)
    decomp_sent = decompose_shap_by_group(shap_sent, feat_groups)
    decomp_gap = decompose_shap_by_group(shap_gap, feat_groups)
    cat_rel = category_reliability(joint)

    payload = build_app_payload(
        r2_star, rmse_star,
        r2_sent, rmse_sent,
        r2_gap, rmse_gap,
        shap_star, shap_sent, shap_gap,
        feat_groups,
        decomp_star, decomp_sent, decomp_gap,
        cat_rel, joint,
        sv_star, sv_sent, sv_gap,
        X_samp_star, X_samp_sent, X_samp_gap,
    )
    save_app_json(payload)

    print("\n================================")
    print("  Final Summary")
    print("==================================")
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
    print("     plot_features.json    ← pre-computed web app data")
    print()


if __name__ == "__main__":
    main()
