from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import plotly.io as pio

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DB_PATH = PROJECT_ROOT / "data" / "tables" / "food_recipe.db"
OUTPUT_DIR = PROJECT_ROOT / "plots" / "popularity_outputs"

RANDOM_STATE = 42
N_SPLITS = 5
TOP_CATEGORIES = 40
MIN_RECIPES_PER_AUTHOR = 2
PRED_RATING_BINS = 35
POP_PERCENTILE_BINS = 50

NUMERIC_FEATURES = [
    "PrepTime",
    "CookTime",
    "TotalTime",
    "RecipeServings",
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent",
    "ingredient_count",
    "instruction_count",
    "keyword_count",
    "description_len",
    "instruction_char_len",
]

CATEGORY_FEATURE = "RecipeCategory_top"
TARGET_COL = "observed_rating"
COUNT_COL = "observed_review_count"

POPULARITY_METRICS = {
    "prolificness": "author_recipe_count",
    "reach": "author_total_reviews",
    "prestige": "author_prestige_bayes",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_parse_list(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip().strip("'\"") for p in text.split(",")]
    return [p for p in parts if p]


def robust_qcut(series: pd.Series, q: int = 5) -> pd.Series:
    ranked = series.rank(method="first")
    try:
        return pd.qcut(ranked, q=q, labels=[f"Q{i}" for i in range(1, q + 1)])
    except ValueError:
        unique_n = ranked.nunique(dropna=True)
        q_eff = max(1, min(q, unique_n))
        return pd.qcut(ranked, q=q_eff, labels=[f"Q{i}" for i in range(1, q_eff + 1)])


# -----------------------------------------------------------------------------
# Data loading and preparation
# -----------------------------------------------------------------------------
def load_tables(db_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        recipes = pd.read_sql_query("SELECT * FROM recipes", conn)
        reviews = pd.read_sql_query("SELECT * FROM reviews", conn)

    return recipes, reviews


def prepare_recipe_features(recipes: pd.DataFrame) -> pd.DataFrame:
    df = recipes.copy()

    for col in [
        "PrepTime",
        "CookTime",
        "TotalTime",
        "RecipeServings",
        "Calories",
        "FatContent",
        "SaturatedFatContent",
        "CholesterolContent",
        "SodiumContent",
        "CarbohydrateContent",
        "FiberContent",
        "SugarContent",
        "ProteinContent",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for text_col in ["Description", "RecipeInstructions", "Keywords", "RecipeIngredientParts"]:
        if text_col not in df.columns:
            df[text_col] = ""
        df[text_col] = df[text_col].fillna("")

    df["ingredient_count"] = df["RecipeIngredientParts"].map(lambda x: len(safe_parse_list(x)))
    df["instruction_count"] = df["RecipeInstructions"].map(lambda x: len(safe_parse_list(x)))
    df["keyword_count"] = df["Keywords"].map(lambda x: len(safe_parse_list(x)))
    df["description_len"] = df["Description"].astype(str).str.len()
    df["instruction_char_len"] = df["RecipeInstructions"].astype(str).str.len()

    df["RecipeCategory"] = df["RecipeCategory"].fillna("Unknown").astype(str)
    top_cats = df["RecipeCategory"].value_counts().head(TOP_CATEGORIES).index
    df[CATEGORY_FEATURE] = np.where(df["RecipeCategory"].isin(top_cats), df["RecipeCategory"], "Other")

    keep_cols = [
        "RecipeId",
        "AuthorId",
        "AuthorName",
        "Name",
        "RecipeCategory",
        CATEGORY_FEATURE,
    ] + [c for c in NUMERIC_FEATURES if c in df.columns]

    return df[keep_cols].copy()


def prepare_review_aggregates(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy()
    for col in ["RecipeId", "AuthorId", "Rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["RecipeId", "Rating"]).copy()
    df["RecipeId"] = df["RecipeId"].astype("int64")

    agg = (
        df.groupby("RecipeId")
        .agg(
            observed_rating=("Rating", "mean"),
            observed_review_count=("Rating", "size"),
            rating_std=("Rating", lambda s: float(np.std(s, ddof=0))),
        )
        .reset_index()
    )
    return agg


def merge_recipe_review_data(recipes: pd.DataFrame, recipe_features: pd.DataFrame, review_agg: pd.DataFrame) -> pd.DataFrame:
    df = recipe_features.merge(review_agg, on="RecipeId", how="left")

    # Fallback to recipe-level aggregated rating only if the recipe has no review-row rating.
    recipe_rating_fallback = recipes[["RecipeId", "AggregatedRating", "ReviewCount"]].copy()
    recipe_rating_fallback["AggregatedRating"] = pd.to_numeric(recipe_rating_fallback["AggregatedRating"], errors="coerce")
    recipe_rating_fallback["ReviewCount"] = pd.to_numeric(recipe_rating_fallback["ReviewCount"], errors="coerce")

    df = df.merge(recipe_rating_fallback, on="RecipeId", how="left")
    df["observed_rating"] = df["observed_rating"].fillna(df["AggregatedRating"])
    df["observed_review_count"] = df["observed_review_count"].fillna(df["ReviewCount"])
    df["rating_std"] = df["rating_std"].fillna(0.0)

    df = df.dropna(subset=["RecipeId", "AuthorId", TARGET_COL]).copy()
    df[COUNT_COL] = pd.to_numeric(df[COUNT_COL], errors="coerce").fillna(0).astype(int)
    df = df[df[COUNT_COL] > 0].copy()

    return df


# -----------------------------------------------------------------------------
# Popularity metrics
# -----------------------------------------------------------------------------
def compute_author_metrics(recipe_df: pd.DataFrame) -> pd.DataFrame:
    global_mean = recipe_df[TARGET_COL].mean()
    shrink_k = 10.0

    author = (
        recipe_df.groupby(["AuthorId", "AuthorName"], dropna=False)
        .agg(
            author_recipe_count=("RecipeId", "size"),
            author_total_reviews=(COUNT_COL, "sum"),
            author_raw_mean_rating=(TARGET_COL, "mean"),
            author_rating_std=(TARGET_COL, lambda s: float(np.std(s, ddof=0))),
            author_total_rating_points=(TARGET_COL, lambda s: float(np.sum(s))),
        )
        .reset_index()
    )

    author["author_prestige_bayes"] = (
        author["author_raw_mean_rating"] * author["author_recipe_count"] + global_mean * shrink_k
    ) / (author["author_recipe_count"] + shrink_k)

    for metric_name, metric_col in POPULARITY_METRICS.items():
        author[f"{metric_name}_quintile"] = robust_qcut(author[metric_col], q=5).astype(str)
        author[f"{metric_name}_percentile"] = author[metric_col].rank(pct=True)

    return author


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------
def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in NUMERIC_FEATURES if c in feature_df.columns]
    categorical_cols = [CATEGORY_FEATURE]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor


def fit_models_with_cv(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + [CATEGORY_FEATURE]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].to_numpy()
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    preprocessor = build_preprocessor(X)

    ols_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    df = df.copy()
    df["pred_ols"] = cross_val_predict(ols_model, X, y, cv=cv, n_jobs=1)
    df["pred_rf"] = cross_val_predict(rf_model, X, y, cv=cv, n_jobs=1)
    df["resid_ols"] = df[TARGET_COL] - df["pred_ols"]
    df["resid_rf"] = df[TARGET_COL] - df["pred_rf"]

    return df


# -----------------------------------------------------------------------------
# Precomputed outputs for the web app
# -----------------------------------------------------------------------------
def build_quintile_residuals(recipe_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for metric_name in POPULARITY_METRICS:
        qcol = f"{metric_name}_quintile"
        for model_key in ["ols", "rf"]:
            rcol = f"resid_{model_key}"
            tmp = recipe_df[["RecipeId", "Name", "AuthorId", "AuthorName", qcol, rcol, TARGET_COL, "RecipeCategory"]].copy()
            tmp = tmp.rename(columns={qcol: "popularity_quintile", rcol: "residual"})
            tmp["metric"] = metric_name
            tmp["model"] = model_key
            frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


def build_gradient_grids(recipe_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for metric_name in POPULARITY_METRICS:
        pcol = f"{metric_name}_percentile"
        for model_key in ["ols", "rf"]:
            rcol = f"resid_{model_key}"
            pred_col = f"pred_{model_key}"

            tmp = recipe_df[[pcol, pred_col, rcol]].copy().dropna()
            tmp["pop_bin"] = pd.cut(
                tmp[pcol],
                bins=np.linspace(0, 1, POP_PERCENTILE_BINS + 1),
                include_lowest=True,
            )

            y_min = float(np.floor(tmp[pred_col].min() * 4) / 4)
            y_max = float(np.ceil(tmp[pred_col].max() * 4) / 4)
            tmp["pred_bin"] = pd.cut(
                tmp[pred_col],
                bins=np.linspace(y_min, y_max, PRED_RATING_BINS + 1),
                include_lowest=True,
            )

            g = (
                tmp.groupby(["pop_bin", "pred_bin"], observed=False)
                .agg(mean_residual=(rcol, "mean"), count=(rcol, "size"))
                .reset_index()
            )

            g["metric"] = metric_name
            g["model"] = model_key
            g["pop_mid"] = g["pop_bin"].map(lambda x: x.mid if pd.notna(x) else np.nan)
            g["pred_mid"] = g["pred_bin"].map(lambda x: x.mid if pd.notna(x) else np.nan)

            # Convert interval columns to strings, or drop them entirely.
            g["pop_bin_label"] = g["pop_bin"].astype(str)
            g["pred_bin_label"] = g["pred_bin"].astype(str)
            g = g.drop(columns=["pop_bin", "pred_bin"])

            parts.append(g)

    return pd.concat(parts, ignore_index=True)


def build_rank_shift(recipe_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    base = recipe_df[["RecipeId", "Name", "AuthorId", "AuthorName", TARGET_COL, "RecipeCategory"]].copy()
    base["raw_rank"] = base[TARGET_COL].rank(method="dense", ascending=False)

    for model_key in ["ols", "rf"]:
        pred_col = f"pred_{model_key}"
        tmp = base.copy()
        tmp["corrected_score"] = recipe_df[pred_col]
        tmp["corrected_rank"] = tmp["corrected_score"].rank(method="dense", ascending=False)
        tmp["rank_shift"] = tmp["raw_rank"] - tmp["corrected_rank"]
        tmp["model"] = model_key
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def save_violin_html(quintile_df: pd.DataFrame) -> Path:
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Prolificness — OLS",
            "Reach — OLS",
            "Prestige — OLS",
            "Prolificness — Random Forest",
            "Reach — Random Forest",
            "Prestige — Random Forest",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    metric_order = ["prolificness", "reach", "prestige"]
    model_order = ["ols", "rf"]
    row_map = {"ols": 1, "rf": 2}
    col_map = {"prolificness": 1, "reach": 2, "prestige": 3}

    for model_key in model_order:
        for metric_name in metric_order:
            row, col = row_map[model_key], col_map[metric_name]
            sdf = quintile_df[(quintile_df["metric"] == metric_name) & (quintile_df["model"] == model_key)]
            for q in sorted(sdf["popularity_quintile"].dropna().unique()):
                qdf = sdf[sdf["popularity_quintile"] == q]
                fig.add_trace(
                    go.Violin(
                        x=qdf["popularity_quintile"],
                        y=qdf["residual"],
                        name=q,
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=False,
                        points=False,
                        hovertemplate=(
                            "Quintile=%{x}<br>Residual=%{y:.3f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title="Author Popularity Bias — Residual Distribution by Popularity Quintile",
        height=900,
        width=1500,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Actual rating − expected rating", zeroline=True, zerolinewidth=1.2)
    fig.update_xaxes(title_text="Popularity quintile")

    out = OUTPUT_DIR / "author_popularity_violin.html"
    fig.write_html(out, include_plotlyjs="cdn")

    json_str = pio.to_json(fig, pretty=False)
    (OUTPUT_DIR / "author_popularity_violin.json").write_text(json_str, encoding="utf-8")
    return out


def save_gradient_html(grid_df: pd.DataFrame) -> Path:
    metric_order = ["prolificness", "reach", "prestige"]
    model_order = ["ols", "rf"]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Prolificness — OLS",
            "Reach — OLS",
            "Prestige — OLS",
            "Prolificness — Random Forest",
            "Reach — Random Forest",
            "Prestige — Random Forest",
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )

    zmax = np.nanpercentile(np.abs(grid_df["mean_residual"].to_numpy()), 95)
    zmax = float(max(zmax, 0.05))

    for i, model_key in enumerate(model_order, start=1):
        for j, metric_name in enumerate(metric_order, start=1):
            sdf = grid_df[(grid_df["metric"] == metric_name) & (grid_df["model"] == model_key)].copy()
            pivot = sdf.pivot(index="pred_mid", columns="pop_mid", values="mean_residual").sort_index()
            fig.add_trace(
                go.Heatmap(
                    x=pivot.columns,
                    y=pivot.index,
                    z=pivot.values,
                    zmin=-zmax,
                    zmax=zmax,
                    coloraxis="coloraxis",
                    hovertemplate=(
                        "Popularity percentile=%{x:.2f}<br>Predicted rating=%{y:.2f}<br>Mean residual=%{z:.3f}<extra></extra>"
                    ),
                ),
                row=i,
                col=j,
            )

    fig.update_layout(
        title="Author Popularity Bias Gradient Field",
        height=900,
        width=1500,
        template="plotly_white",
        coloraxis=dict(colorscale="RdBu_r", colorbar=dict(title="Mean residual")),
    )
    fig.update_xaxes(title_text="Author popularity percentile")
    fig.update_yaxes(title_text="Predicted rating")

    out = OUTPUT_DIR / "author_popularity_gradient.html"
    fig.write_html(out, include_plotlyjs="cdn")

    json_str = pio.to_json(fig, pretty=False)
    (OUTPUT_DIR / "author_popularity_gradient.json").write_text(json_str, encoding="utf-8")


    return out


def save_rank_shift_html(rank_df: pd.DataFrame, top_n: int = 20) -> Path:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "OLS — Most underrated by raw rating",
            "OLS — Most reputation-inflated",
            "Random Forest — Most underrated by raw rating",
            "Random Forest — Most reputation-inflated",
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    for row, model_key in enumerate(["ols", "rf"], start=1):
        sdf = rank_df[rank_df["model"] == model_key].copy()
        up = sdf.nlargest(top_n, "rank_shift").sort_values("rank_shift")
        down = sdf.nsmallest(top_n, "rank_shift").sort_values("rank_shift")

        fig.add_trace(
            go.Bar(
                x=up["rank_shift"],
                y=up["Name"],
                orientation="h",
                showlegend=False,
                customdata=np.stack([up["AuthorName"], up["RecipeCategory"]], axis=1),
                hovertemplate="%{y}<br>Rank shift=%{x:.0f}<br>Author=%{customdata[0]}<br>Category=%{customdata[1]}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=down["rank_shift"],
                y=down["Name"],
                orientation="h",
                showlegend=False,
                customdata=np.stack([down["AuthorName"], down["RecipeCategory"]], axis=1),
                hovertemplate="%{y}<br>Rank shift=%{x:.0f}<br>Author=%{customdata[0]}<br>Category=%{customdata[1]}<extra></extra>",
            ),
            row=row,
            col=2,
        )

    fig.update_layout(
        title="Ranking Shifts After Removing Feature-Adjusted Rating Inflation",
        height=1000,
        width=1500,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Raw rank − corrected rank")

    out = OUTPUT_DIR / "author_popularity_rank_shift.html"
    fig.write_html(out, include_plotlyjs="cdn")
    
    json_str = pio.to_json(fig, pretty=False)
    (OUTPUT_DIR / "author_popularity_rank_shift.json").write_text(json_str, encoding="utf-8")
    return out


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def save_data_outputs(recipe_df: pd.DataFrame, author_df: pd.DataFrame, quintile_df: pd.DataFrame, grid_df: pd.DataFrame, rank_df: pd.DataFrame) -> None:
    recipe_out = OUTPUT_DIR / "author_popularity_recipe_level.parquet"
    author_out = OUTPUT_DIR / "author_popularity_author_level.parquet"
    quintile_out = OUTPUT_DIR / "author_popularity_quintile_residuals.parquet"
    grid_out = OUTPUT_DIR / "author_popularity_gradient_grid.parquet"
    rank_out = OUTPUT_DIR / "author_popularity_rank_shift.parquet"

    grid_df = grid_df.copy()
    for col in grid_df.columns:
        if str(grid_df[col].dtype).startswith("interval"):
            grid_df[col] = grid_df[col].astype(str)

    recipe_df.to_parquet(recipe_out, index=False)
    author_df.to_parquet(author_out, index=False)
    quintile_df.to_parquet(quintile_out, index=False)
    grid_df.to_parquet(grid_out, index=False)
    rank_df.to_parquet(rank_out, index=False)

    # JSON copies are easier to consume directly in the web app when parquet is inconvenient.
    recipe_df.head(5000).to_json(OUTPUT_DIR / "author_popularity_recipe_level.preview.json", orient="records")
    quintile_df.to_json(OUTPUT_DIR / "author_popularity_quintile_residuals.json", orient="records")
    grid_df.to_json(OUTPUT_DIR / "author_popularity_gradient_grid.json", orient="records")
    rank_df.to_json(OUTPUT_DIR / "author_popularity_rank_shift.json", orient="records")

    metadata = {
        "db_path": str(DB_PATH),
        "n_recipe_rows": int(len(recipe_df)),
        "n_author_rows": int(len(author_df)),
        "models": ["ols", "rf"],
        "popularity_metrics": POPULARITY_METRICS,
        "numeric_features": [c for c in NUMERIC_FEATURES if c in recipe_df.columns],
        "category_feature": CATEGORY_FEATURE,
        "target_col": TARGET_COL,
        "cv_splits": N_SPLITS,
        "top_categories_kept": TOP_CATEGORIES,
    }
    (OUTPUT_DIR / "author_popularity_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_pipeline() -> Dict[str, Path]:
    ensure_output_dir()
    recipes, reviews = load_tables(DB_PATH)
    recipes = recipes.head(10_000)
    reviews = reviews.head(50_000)
    recipe_features = prepare_recipe_features(recipes)
    review_agg = prepare_review_aggregates(reviews)
    recipe_df = merge_recipe_review_data(recipes, recipe_features, review_agg)

    # Keep all recipes with at least one usable rating, but drop authors with only one rated recipe
    # because the author-level popularity and portfolio prestige are not stable there.
    author_counts = recipe_df.groupby("AuthorId")["RecipeId"].size()
    keep_authors = author_counts[author_counts >= MIN_RECIPES_PER_AUTHOR].index
    recipe_df = recipe_df[recipe_df["AuthorId"].isin(keep_authors)].copy()

    author_df = compute_author_metrics(recipe_df)
    recipe_df = recipe_df.merge(author_df, on=["AuthorId", "AuthorName"], how="left")
    recipe_df = fit_models_with_cv(recipe_df)

    quintile_df = build_quintile_residuals(recipe_df)
    grid_df = build_gradient_grids(recipe_df)
    rank_df = build_rank_shift(recipe_df)

    save_data_outputs(recipe_df, author_df, quintile_df, grid_df, rank_df)

    html_paths = {
        "violin_html": save_violin_html(quintile_df),
        "gradient_html": save_gradient_html(grid_df),
        "rank_shift_html": save_rank_shift_html(rank_df),
    }

    summary = {
        "n_recipes_modeled": int(len(recipe_df)),
        "n_authors_modeled": int(len(author_df)),
        "mean_rating": float(recipe_df[TARGET_COL].mean()),
        "ols_residual_mean": float(recipe_df["resid_ols"].mean()),
        "rf_residual_mean": float(recipe_df["resid_rf"].mean()),
    }
    (OUTPUT_DIR / "author_popularity_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return html_paths


if __name__ == "__main__":
    paths = run_pipeline()
    print("Saved outputs:")
    for key, value in paths.items():
        print(f"  {key}: {value}")
