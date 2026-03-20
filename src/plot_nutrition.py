# ============================================================
# Visualization 3 — The Nutritional Landscape
# Export-first pipeline for web-app replotting
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# USER SETTINGS
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RECIPE_FILE = BASE_DIR / "data" / "tables" / "food_recipe.db"
OUTPUT_DIR = BASE_DIR / "plots" / "nutritional_landscape_outputs"

TOP_N_CATEGORIES = 8
N_CLUSTERS = 7
MIN_REVIEWS = 2
MAX_MISSING_ALLOWED = 2
RANDOM_STATE = 42

# web-app / export settings
EXPORT_PARQUET = False   # True if pyarrow is installed and you want smaller/faster files
WRITE_HTML = True
WRITE_PNG = True         # Requires kaleido installed
WRITE_PLOTLY_JSON = True

DISCRETE_CLUSTER_COLORS = px.colors.sample_colorscale(
    "Turbo",
    [i / (N_CLUSTERS - 1) if N_CLUSTERS > 1 else 0 for i in range(N_CLUSTERS)]
)
CATEGORY_COLORS = px.colors.qualitative.Set2
PCA_MARKER_SIZE = 7.5
PCA_MARKER_OPACITY = 0.74
TRADEOFF_MARKER_OPACITY = 0.68

# -----------------------------
# EXPECTED COLUMNS
# -----------------------------
NUTRITION_COLS = [
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent",
]

META_COLS = [
    "RecipeId",
    "Name",
    "RecipeCategory",
    "AggregatedRating",
    "ReviewCount",
    "RecipeServings",
]


# -----------------------------
# PATH HELPERS
# -----------------------------
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def figure_dir(root_output_dir: str | Path, fig_key: str) -> Path:
    path = Path(root_output_dir) / "figures" / fig_key
    ensure_dir(path)
    return path

def safe_write_json(obj, path: str | Path):
    def _json_default(x):
        if isinstance(x, Path):
            return str(x)
        raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)

def safe_read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_dataframe(df: pd.DataFrame, path_no_ext: str | Path, use_parquet: bool = False):
    path_no_ext = Path(path_no_ext)

    if use_parquet:
        try:
            out = path_no_ext.with_suffix(".parquet")
            df.to_parquet(out, index=False)
            return str(out)
        except Exception:
            pass

    out = path_no_ext.with_suffix(".csv")
    df.to_csv(out, index=False)
    return str(out)


def load_exported_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# -----------------------------
# DATA LOADING
# -----------------------------
def load_recipes_from_sqlite(db_path, table="recipes"):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)

    query = f"""
        SELECT
            RecipeId,
            Name,
            RecipeCategory,
            AggregatedRating,
            ReviewCount,
            RecipeServings,
            Calories,
            FatContent,
            SaturatedFatContent,
            CholesterolContent,
            SodiumContent,
            CarbohydrateContent,
            FiberContent,
            SugarContent,
            ProteinContent
        FROM {table}
        WHERE
            Calories IS NOT NULL
            AND FatContent IS NOT NULL
            AND CarbohydrateContent IS NOT NULL
            AND ProteinContent IS NOT NULL
            AND SugarContent IS NOT NULL
            AND SodiumContent IS NOT NULL
            AND FiberContent IS NOT NULL
            AND CholesterolContent IS NOT NULL
            AND SaturatedFatContent IS NOT NULL
            AND ReviewCount >= {MIN_REVIEWS}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# -----------------------------
# CLEANING / FEATURES
# -----------------------------
def clean_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def basic_recipe_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in META_COLS:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = NUTRITION_COLS + ["AggregatedRating", "ReviewCount", "RecipeServings"]
    df = clean_numeric_columns(df, numeric_cols)

    missing_count = df[NUTRITION_COLS].isna().sum(axis=1)
    df = df.loc[missing_count <= MAX_MISSING_ALLOWED].copy()

    for col in NUTRITION_COLS:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    if "ReviewCount" in df.columns:
        df = df[(df["ReviewCount"].fillna(0) >= MIN_REVIEWS)].copy()

    df = df[df["Calories"].fillna(0) > 0].copy()
    for col in NUTRITION_COLS:
        df = df[df[col].fillna(0) >= 0].copy()

    df["RecipeCategory"] = df["RecipeCategory"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["RecipeCategory"] == "", "RecipeCategory"] = "Unknown"

    df["Name"] = df["Name"].fillna("Unknown Recipe").astype(str)

    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    df["ProteinPer100Cal"] = 100 * df["ProteinContent"] / (df["Calories"] + eps)
    df["FiberPer100Cal"] = 100 * df["FiberContent"] / (df["Calories"] + eps)
    df["SugarPer100Cal"] = 100 * df["SugarContent"] / (df["Calories"] + eps)
    df["SodiumPer100Cal"] = 100 * df["SodiumContent"] / (df["Calories"] + eps)
    df["FatPer100Cal"] = 100 * df["FatContent"] / (df["Calories"] + eps)
    df["CarbPer100Cal"] = 100 * df["CarbohydrateContent"] / (df["Calories"] + eps)

    df["SugarShareOfCarbs"] = df["SugarContent"] / (df["CarbohydrateContent"] + eps)
    df["SatFatShareOfFat"] = df["SaturatedFatContent"] / (df["FatContent"] + eps)

    df["NutritionBalanceScore"] = (
        1.5 * df["ProteinPer100Cal"]
        + 1.5 * df["FiberPer100Cal"]
        - 1.0 * df["SugarPer100Cal"]
        - 0.002 * df["SodiumPer100Cal"]
    )

    return df

def select_features_for_landscape(df: pd.DataFrame) -> list[str]:
    features = [
        "Calories",
        "FatContent",
        "SaturatedFatContent",
        "SodiumContent",
        "CarbohydrateContent",
        "FiberContent",
        "SugarContent",
        "ProteinContent",
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SugarPer100Cal",
        "SodiumPer100Cal",
        "SugarShareOfCarbs",
        "SatFatShareOfFat",
    ]
    return [f for f in features if f in df.columns]

def transform_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()

    skewed_cols = [
        "Calories",
        "FatContent",
        "SaturatedFatContent",
        "SodiumContent",
        "CarbohydrateContent",
        "FiberContent",
        "SugarContent",
        "ProteinContent",
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SugarPer100Cal",
        "SodiumPer100Cal",
    ]

    for col in skewed_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])

    for col in ["SugarShareOfCarbs", "SatFatShareOfFat"]:
        if col in X.columns:
            X[col] = X[col].clip(lower=0, upper=X[col].quantile(0.99))

    return X

def compute_pca_and_clusters(df: pd.DataFrame, feature_cols: list[str], n_clusters: int = 5):
    X_raw = transform_features(df, feature_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=10,
    )
    clusters = gmm.fit_predict(X_scaled)
    cluster_probs = gmm.predict_proba(X_scaled).max(axis=1)

    out = df.copy()
    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]
    out["Cluster"] = clusters.astype(str)
    out["ClusterConfidence"] = cluster_probs

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=["PC1_loading", "PC2_loading"]
    ).reset_index(names="Feature")

    explained = pd.DataFrame({
        "Component": ["PC1", "PC2"],
        "ExplainedVarianceRatio": pca.explained_variance_ratio_
    })

    return out, loadings, explained, scaler, pca, gmm

def get_top_categories(df: pd.DataFrame, n: int = 8) -> list[str]:
    return df["RecipeCategory"].value_counts().head(n).index.tolist()

def summarize_category_spread(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    sub = df[df["RecipeCategory"].isin(categories)].copy()

    summary = (
        sub.groupby("RecipeCategory")
        .agg(
            n_recipes=("RecipeId", "count"),
            median_calories=("Calories", "median"),
            median_protein_per_100cal=("ProteinPer100Cal", "median"),
            median_fiber_per_100cal=("FiberPer100Cal", "median"),
            median_sugar_per_100cal=("SugarPer100Cal", "median"),
            median_sodium_per_100cal=("SodiumPer100Cal", "median"),
            calories_iqr=("Calories", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            sugar_iqr=("SugarContent", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            sodium_iqr=("SodiumContent", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        )
        .sort_values("n_recipes", ascending=False)
        .reset_index()
    )
    return summary


# -----------------------------
# STYLING
# -----------------------------
def apply_food_theme(
    fig: go.Figure,
    title: str | None = None,
    subtitle: str | None = None,
    *,
    show_title: bool = False,
    legend_y: float = -0.05,
    legend_x: float = 0.5,
    legend_xanchor: str = "center",
    legend_yanchor: str = "top",
    margin: dict | None = None,
) -> go.Figure:
    fig.update_layout(
        title=None if not show_title else dict(
            text=title if subtitle is None else f"{title}<br><sup>{subtitle}</sup>",
            x=0.02,
            xanchor="left",
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        font=dict(size=14, color="#243447"),
        margin=margin or dict(l=60, r=40, t=30, b=110),
        legend=dict(
            title=None,
            orientation="h",
            yanchor=legend_yanchor,
            y=legend_y,
            xanchor=legend_xanchor,
            x=legend_x,
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(0,0,0,0.10)",
            borderwidth=1,
            tracegroupgap=8,
            itemsizing="constant",
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="rgba(0,0,0,0.20)",
        ticks="outside",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="rgba(0,0,0,0.20)",
        ticks="outside",
    )
    return fig

def build_discrete_colorscale(colors: list[str]) -> list[list[float | str]]:
    n = len(colors)
    if n <= 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]

    stepped = []
    for i, color in enumerate(colors):
        left = i / n
        right = (i + 1) / n
        stepped.append([left, color])
        stepped.append([right, color])
    return stepped

def add_discrete_cluster_colorbar(fig: go.Figure, cluster_order: list[str], colors: list[str]) -> go.Figure:
    n = len(cluster_order)

    fig.update_layout(
        coloraxis=dict(
            colorscale=build_discrete_colorscale(colors[:n]),
            cmin=-0.5,
            cmax=n - 0.5,
            colorbar=dict(
                title=dict(text="Cluster", side="right"),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.1,
                yanchor="top",
                len=0.62,
                thickness=25,
                tickmode="array",
                tickvals=list(range(n)),
                ticktext=[str(c) for c in cluster_order],
                outlinecolor="black",
                outlinewidth=1,
                ticklen=6,
                tickcolor="rgba(0,0,0,0.65)",
            ),
        )
    )
    return fig

def compute_pca_axis_ranges(plot_df: pd.DataFrame, pad_frac: float = 0.06) -> dict:
    x_min, x_max = plot_df["PC1"].min(), plot_df["PC1"].max()
    y_min, y_max = plot_df["PC2"].min(), plot_df["PC2"].max()

    x_pad = (x_max - x_min) * pad_frac if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * pad_frac if y_max > y_min else 1.0

    return {
        "x_range": [float(x_min - x_pad), float(x_max + x_pad)],
        "y_range": [float(y_min - y_pad), float(y_max + y_pad)],
    }

def apply_shared_pca_layout(fig: go.Figure, meta: dict, *, use_colorbar: bool = False) -> go.Figure:
    explained_pc1 = meta["explained_variance"]["PC1"]
    explained_pc2 = meta["explained_variance"]["PC2"]

    fig.update_traces(
        marker=dict(
            size=PCA_MARKER_SIZE,
            opacity=PCA_MARKER_OPACITY,
            line=dict(width=0.35, color="rgba(255,255,255,0.35)"),
        )
    )

    fig.update_xaxes(
        title=f"PC1 ({explained_pc1 * 100:.1f}% variance explained)",
        side="top",
        range=meta["x_range"],
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks="",
        automargin=True,
        title_standoff=18,
    )

    fig.update_yaxes(
        title=f"PC2 ({explained_pc2 * 100:.1f}% variance explained)",
        range=meta["y_range"],
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks="",
        automargin=True,
        title_standoff=18,
    )

    fig.update_layout(
        showlegend=not use_colorbar,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        font=dict(size=14, color="#243447"),
        margin=dict(l=85, r=60, t=50, b=50),
    )

    return fig

# -----------------------------
# EXPORT / RE-PLOT CORE
# -----------------------------
def export_figure_bundle(
    fig_key: str,
    plot_df: pd.DataFrame,
    metadata: dict,
    fig: go.Figure,
    output_dir: str,
    use_parquet: bool = False,
):
    fdir = figure_dir(output_dir, fig_key)

    data_path = export_dataframe(plot_df, os.path.join(fdir, "plot_data"), use_parquet=use_parquet)
    meta_path = os.path.join(fdir, "metadata.json")
    safe_write_json(metadata, meta_path)

    if WRITE_PLOTLY_JSON:
        fig.write_json(os.path.join(fdir, "figure.plotly.json"))

    if WRITE_HTML:
        fig.write_html(
            os.path.join(fdir, "figure.html"),
            include_plotlyjs="cdn",
            full_html=True
        )

    if WRITE_PNG:
        try:
            fig.write_image(os.path.join(fdir, "figure.png"), scale=4, width=1800, height=1100)
        except Exception:
            pass

    return {
        "fig_key": fig_key,
        "data_path": data_path,
        "metadata_path": meta_path,
        "plotly_json_path": os.path.join(fdir, "figure.plotly.json") if WRITE_PLOTLY_JSON else None,
        "html_path": os.path.join(fdir, "figure.html") if WRITE_HTML else None,
        "png_path": os.path.join(fdir, "figure.png") if WRITE_PNG else None,
    }


# -----------------------------
# FIGURE BUILDERS
# -----------------------------
def build_fig_pca_landscape(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    cluster_order = meta["cluster_order"]
    cluster_to_num = {cluster: i for i, cluster in enumerate(cluster_order)}
    plot_df = plot_df.copy()
    plot_df["ClusterNum"] = plot_df["Cluster"].astype(str).map(cluster_to_num)

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="ClusterNum",
        color_continuous_scale=build_discrete_colorscale(DISCRETE_CLUSTER_COLORS[: len(cluster_order)]),
        hover_name="Name",
        hover_data={
            "RecipeCategory": True,
            "Calories": ":.1f",
            "ProteinContent": ":.1f",
            "CarbohydrateContent": ":.1f",
            "SugarContent": ":.1f",
            "SodiumContent": ":.1f",
            "AggregatedRating": ":.2f",
            "ReviewCount": True,
            "PC1": ":.3f",
            "PC2": ":.3f",
            "ClusterNum": False,
            "Cluster": True,
            "ClusterConfidence": ":.3f",
            "CholesterolContent": ":.1f",
            "FiberContent": ":.1f",
        },
        render_mode="webgl",
    )

    fig = apply_shared_pca_layout(fig, meta, use_colorbar=True)

    fig.update_traces(
        customdata=plot_df[
            [
                "RecipeCategory",
                "Calories",
                "ProteinContent",
                "CarbohydrateContent",
                "SugarContent",
                "SodiumContent",
                "AggregatedRating",
                "ReviewCount",
                "Cluster",
                "ClusterConfidence",
                "CholesterolContent",
                "FiberContent",
            ]
        ].to_numpy(),
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Category: %{customdata[0]}<br>"
            "Calories: %{customdata[1]:.1f} kcal<br>"
            "Protein: %{customdata[2]:.1f} g<br>"
            "Carbohydrates: %{customdata[3]:.1f} g<br>"
            "Fiber: %{customdata[11]:.1f} g<br>"
            "Sugar: %{customdata[4]:.1f} g<br>"
            "Cholesterol: %{customdata[10]:.1f} mg<br>"
            "Sodium: %{customdata[5]:.1f} mg<br>"
            "Rating: %{customdata[6]:.2f}<br>"
            "Reviews: %{customdata[7]}<br>"
            "Cluster: %{customdata[8]}<br>"
            "Confidence: %{customdata[9]:.3f}<br>"
            "PC1: %{x:.3f}<br>"
            "PC2: %{y:.3f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        margin=dict(
            l=120,
            r=70,
            t=0,
            b=0
        ),
        coloraxis_colorbar=dict(
            y=-0.2,
            # yanchor="bottom"
        ),
    )

    return add_discrete_cluster_colorbar(fig, cluster_order, DISCRETE_CLUSTER_COLORS)

def build_fig_pca_categories(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    category_order = meta["category_order"]
    color_map = {
        c: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        for i, c in enumerate(category_order)
    }

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="RecipeCategory",
        category_orders={"RecipeCategory": category_order},
        color_discrete_map=color_map,
        hover_name="Name",
        hover_data={
            "RecipeCategory": True,
            "Calories": ":.1f",
            "ProteinPer100Cal": ":.1f",
            "CarbPer100Cal": ":.1f",
            "SugarPer100Cal": ":.1f",
            "SodiumPer100Cal": ":.1f",
            "AggregatedRating": ":.2f",
            "ReviewCount": True,
            "PC1": ":.3f",
            "PC2": ":.3f",
            "Cluster": True,
            "FiberPer100Cal": ":.1f",
        },
        render_mode="webgl",
    )

    fig = apply_shared_pca_layout(fig, meta, use_colorbar=False)

    fig.update_traces(
        customdata=plot_df[
            [
                "RecipeCategory",
                "Calories",
                "ProteinPer100Cal",
                "CarbPer100Cal",
                "SugarPer100Cal",
                "SodiumPer100Cal",
                "AggregatedRating",
                "ReviewCount",
                "Cluster",
                "FiberPer100Cal",
            ]
        ].to_numpy(),
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Category: %{customdata[0]}<br>"
            "Calories: %{customdata[1]:.1f} kcal<br>"
            "Protein/100cal: %{customdata[2]:.1f} g<br>"
            "Carb/100cal: %{customdata[3]:.1f} g<br>"
            "Fiber/100cal: %{customdata[9]:.1f} g<br>"
            "Sugar/100cal: %{customdata[4]:.1f} g<br>"
            "Sodium/100cal: %{customdata[5]:.1f} mg<br>"
            "Rating: %{customdata[6]:.2f}<br>"
            "Reviews: %{customdata[7]}<br>"
            "Cluster: %{customdata[8]}<br>"
            "PC1: %{x:.3f}<br>"
            "PC2: %{y:.3f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        legend=dict(
            title=None,
            orientation="h",
            x=0.5,
            y=1.12,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            tracegroupgap=8,
        ),
        margin=dict(l=78, r=62, t=110, b=120),
    )

    fig.update_xaxes(
        showline=True,
        linecolor="rgba(0,0,0,0.35)",
        linewidth=1.2,
        ticks="outside",
        side="bottom",
    )
    fig.update_yaxes(
        showline=True,
        linecolor="rgba(0,0,0,0.35)",
        linewidth=1.2,
        ticks="outside",
    )

    return fig

def build_fig_category_spread(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    category_order = meta["category_order"]

    fig = px.violin(
        plot_df,
        x="ProteinPer100Cal",
        y="RecipeCategory",
        color="RecipeCategory",
        category_orders={"RecipeCategory": category_order},
        color_discrete_sequence=CATEGORY_COLORS,
        box=True,
        points="suspectedoutliers",
        hover_data={
            "Name": True,
            "ProteinPer100Cal": ":.2f",
            "Calories": ":.1f",
            "ProteinContent": ":.1f",
            "AggregatedRating": ":.2f",
            "ReviewCount": True,
        },
    )

    fig.update_traces(meanline_visible=True, opacity=0.78, marker=dict(size=4, opacity=0.45))
    fig.update_xaxes(title="Protein per 100 Calories")
    fig.update_yaxes(title="Recipe Category")

    return apply_food_theme(
        fig,
        legend_y=1.05,
        legend_yanchor="bottom",
        margin=dict(l=60, r=40, t=95, b=80),
    )

def build_fig_tradeoff(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    plot_df = plot_df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["SodiumPer100Cal", "ProteinPer100Cal", "AggregatedRating", "RecipeCategory"])
    plot_df = plot_df[plot_df["SodiumPer100Cal"] > 0].copy()

    category_order = meta["category_order"]
    color_map = {
        c: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        for i, c in enumerate(category_order)
    }

    fig = px.scatter(
        plot_df,
        x="SodiumPer100Cal",
        y="ProteinPer100Cal",
        color="RecipeCategory",
        size="RatingSize",
        category_orders={"RecipeCategory": category_order},
        color_discrete_map=color_map,
        hover_name="Name",
        hover_data={
            "SodiumPer100Cal": ":.2f",
            "ProteinPer100Cal": ":.2f",
            "AggregatedRating": ":.2f",
            "ReviewCount": True,
            "Calories": ":.1f",
            "ProteinContent": ":.1f",
            "SugarContent": ":.1f",
            "SodiumContent": ":.1f",
            "NutritionBalanceScore": ":.2f",
            "RatingSize": False,
        },
        render_mode="webgl",
    )

    fig.update_traces(
        marker=dict(
            opacity=TRADEOFF_MARKER_OPACITY,
            sizemin=5,
            line=dict(width=0.6, color="rgba(255,255,255,0.55)"),
        )
    )

    fig.update_xaxes(title="Sodium per 100 Calories", type="log")
    fig.update_yaxes(title="Protein per 100 Calories")

    return apply_food_theme(fig, margin=dict(l=60, r=40, t=30, b=118))

def build_fig_cluster_heatmap(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    plot_df = plot_df.copy()
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)
    plot_df["Feature"] = plot_df["Feature"].astype(str)

    x_left = meta["content_feature_order"]
    x_right = meta["per100cal_feature_order"]
    y = [str(v) for v in meta["cluster_order"]]

    zmat = (
        plot_df.pivot(index="Cluster", columns="Feature", values="ZMedian")
        .reindex(index=y)
    )

    raw_med = (
        plot_df.pivot(index="Cluster", columns="Feature", values="MedianRaw")
        .reindex(index=y)
    )

    z_left = zmat.reindex(columns=x_left)
    z_right = zmat.reindex(columns=x_right)

    raw_left = raw_med.reindex(columns=x_left)
    raw_right = raw_med.reindex(columns=x_right)

    zabs = np.nanmax(np.abs(zmat.values))
    if not np.isfinite(zabs) or zabs == 0:
        zabs = 1.0

    def make_hover(raw_df, z_df):
        hover = np.empty(z_df.shape, dtype=object)
        for i, cluster in enumerate(z_df.index):
            for j, feat in enumerate(z_df.columns):
                hover[i, j] = (
                    f"Cluster: {cluster}<br>"
                    f"Feature: {feat}<br>"
                    f"Z-scored median: {z_df.iloc[i, j]:.2f}<br>"
                    f"Raw median: {raw_df.iloc[i, j]:.2f}"
                )
        return hover

    hover_left = make_hover(raw_left, z_left)
    hover_right = make_hover(raw_right, z_right)

    text_left = np.vectorize(lambda v: f"{v:.2f}")(z_left.values)
    text_right = np.vectorize(lambda v: f"{v:.2f}")(z_right.values)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z_left.values,
            x=x_left,
            y=y,
            hovertext=hover_left,
            hoverinfo="text",
            text=text_left,
            texttemplate="%{text}",
            textfont=dict(color="white", size=16),
            colorscale="RdBu",
            zmid=0,
            zmin=-zabs,
            zmax=zabs,
            xaxis="x",
            yaxis="y",
            showscale=False,
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=z_right.values,
            x=x_right,
            y=y,
            hovertext=hover_right,
            hoverinfo="text",
            text=text_right,
            texttemplate="%{text}",
            textfont=dict(color="white", size=14),
            colorscale="RdBu",
            zmid=0,
            zmin=-zabs,
            zmax=zabs,
            xaxis="x2",
            yaxis="y2",
            colorbar=dict(
                title=dict(text="Z-score", side="right"),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=1.08,
                yanchor="bottom",
                len=0.62,
                thickness=25,
                tickmode="auto",
                ticks="outside",
                ticklabelposition="outside top",
                outlinecolor="black",
                outlinewidth=1,
                ticklen=6,
                tickcolor="rgba(0,0,0,0.65)",
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=14, color="#243447"),
        margin=dict(l=30, r=40, t=80, b=80),

        xaxis=dict(
            domain=[0.00, 0.48],
            title="Content",
            tickangle=-20,
            showgrid=False,
            zeroline=False,
            side="bottom",
            showline=False,
            linecolor="black",
            linewidth=1.5,
        ),
        xaxis2=dict(
            domain=[0.50, 1.00],
            title="Per 100 Calories",
            tickangle=-20,
            showgrid=False,
            zeroline=False,
            side="bottom",
            showline=False,
            linecolor="black",
            linewidth=1.5,
        ),
        yaxis=dict(
            title="",              # remove it from the axis
            autorange="reversed",
            automargin=False,
            showline=False,
            linecolor="black",
            linewidth=1.5,
        ),
        yaxis2=dict(
            autorange="reversed",
            matches="y",
            showticklabels=False,
            showline=False,
            linecolor="black",
            linewidth=1.5,
        ),
        shapes=[
            # left heatmap border drawn in paper coordinates
            dict(type="rect", xref="paper", yref="paper",
                 x0=0.00, x1=0.48, y0=0.0, y1=1.0,
                 line=dict(color="black", width=1.5), layer="above"),
            # right heatmap border drawn in paper coordinates
            dict(type="rect", xref="paper", yref="paper",
                 x0=0.50, x1=1.00, y0=0.0, y1=1.0,
                 line=dict(color="black", width=1.5), layer="above"),
        ]
    )


    fig.add_annotation(
        text="Nutrition Cluster",
        xref="paper", yref="paper",
        x=-0.02,   # tweak this value: less negative = closer to the ticks
        y=0.5,
        showarrow=False,
        textangle=-90,
        font=dict(size=15, color="#243447"),
        xanchor="center",
        yanchor="middle",
    )


    return fig

def build_fig_cluster_categories(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    plot_df = plot_df.copy()
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)
    plot_df["Category"] = plot_df["Category"].astype(str)

    cluster_order = [str(c) for c in meta["cluster_order"]]
    category_order = meta["category_order"]

    color_map = {
        cat: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        for i, cat in enumerate(category_order)
    }

    fig = go.Figure()

    for cat in category_order:
        sub = plot_df[plot_df["Category"] == cat].copy()

        x_vals = []
        y_vals = []
        text_vals = []
        custom_counts = []

        for cluster in cluster_order:
            row = sub[sub["Cluster"] == cluster]
            if len(row) == 0:
                x_vals.append(0.0)
                y_vals.append(cluster)
                text_vals.append("")
                custom_counts.append(0)
            else:
                prop = float(row["Proportion"].iloc[0])
                cnt = int(row["Count"].iloc[0])
                x_vals.append(prop)
                y_vals.append(cluster)
                text_vals.append(f"{prop:.1%}" if prop >= 0.06 else "")
                custom_counts.append(cnt)

        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=cat,
                orientation="h",
                marker=dict(
                    color=color_map[cat],
                    line=dict(color="white", width=0.8),
                ),
                text=text_vals,
                textposition="inside",
                customdata=np.array(custom_counts).reshape(-1, 1),
                hovertemplate=(
                    "Cluster: %{y}<br>"
                    f"Category: {cat}<br>"
                    "Share within cluster: %{x:.1%}<br>"
                    "Recipe count: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=14, color="#243447"),
        margin=dict(l=90, r=40, t=75, b=70),
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )

    fig.update_xaxes(
        title="Share of Recipes within Cluster",
        tickformat=".0%",
        range=[0, 1],
        showgrid=False,
        zeroline=False,
        showline=False,
    )
    fig.update_yaxes(
        title="Cluster",
        categoryorder="array",
        categoryarray=cluster_order[::-1],
        showgrid=False,
        zeroline=False,
        showline=False,
    )

    return fig

def build_fig_loadings(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df["Feature"],
            y=plot_df["PC1_loading"],
            name="PC1 loading",
            marker=dict(color="#f4a3a3"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=plot_df["Feature"],
            y=plot_df["PC2_loading"],
            name="PC2 loading",
            marker=dict(color="#9ec5fe"),
        )
    )

    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Feature", tickangle=-35)
    fig.update_yaxes(title="Loading")

    fig = apply_food_theme(fig)

    fig.update_layout(
        legend=dict(
            title=None,
            orientation="h",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(l=60, r=30, t=30, b=95),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        showgrid=False,
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.18)",
        zerolinewidth=1,
    )

    return fig


# -----------------------------
# FIGURE EXPORTERS
# -----------------------------
def export_fig_pca_landscape(df_landscape: pd.DataFrame, explained_df: pd.DataFrame, output_dir: str):
    fig_key = "nutrition_pca_landscape"

    plot_df = df_landscape[
        [
            "RecipeId", "Name", "RecipeCategory",
            "AggregatedRating", "ReviewCount",
            "Calories", "ProteinContent", "CarbohydrateContent", "SugarContent", "SodiumContent",
            "CholesterolContent", "FiberContent",
            "PC1", "PC2", "Cluster", "ClusterConfidence"
        ]
    ].copy()

    evr = dict(zip(explained_df["Component"], explained_df["ExplainedVarianceRatio"]))
    axis_ranges = compute_pca_axis_ranges(plot_df)

    meta = {
        "fig_key": fig_key,
        "fig_type": "scatter",
        "title": "The Nutritional Landscape of Recipes",
        "x_col": "PC1",
        "y_col": "PC2",
        "color_col": "Cluster",
        "hover_name_col": "Name",
        "explained_variance": evr,
        "cluster_order": sorted(plot_df["Cluster"].astype(str).unique(), key=lambda x: int(x)),
        "x_range": axis_ranges["x_range"],
        "y_range": axis_ranges["y_range"],
    }

    fig = build_fig_pca_landscape(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)

def export_fig_pca_categories(
    df_landscape: pd.DataFrame,
    explained_df: pd.DataFrame,
    top_categories: list[str],
    output_dir: str
):
    fig_key = "nutrition_pca_categories"

    plot_df = df_landscape[df_landscape["RecipeCategory"].isin(top_categories)].copy()
    plot_df = plot_df[
        [
            "RecipeId", "Name", "RecipeCategory",
            "AggregatedRating", "ReviewCount",
            "Calories", "ProteinPer100Cal", "SugarPer100Cal", "SodiumPer100Cal",
            "CarbPer100Cal", "FatPer100Cal", "FiberPer100Cal",
            "PC1", "PC2", "Cluster"
        ]
    ]

    evr = dict(zip(explained_df["Component"], explained_df["ExplainedVarianceRatio"]))
    axis_ranges = compute_pca_axis_ranges(df_landscape)

    meta = {
        "fig_key": fig_key,
        "fig_type": "scatter",
        "title": "Category Labels vs Nutritional Reality",
        "x_col": "PC1",
        "y_col": "PC2",
        "color_col": "RecipeCategory",
        "hover_name_col": "Name",
        "explained_variance": evr,
        "category_order": top_categories,
        "top_categories": top_categories,
        "x_range": axis_ranges["x_range"],
        "y_range": axis_ranges["y_range"],
    }

    fig = build_fig_pca_categories(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)

def export_fig_category_spread(df_landscape: pd.DataFrame, top_categories: list[str], output_dir: str):
    fig_key = "nutrition_category_spread"

    plot_df = df_landscape[df_landscape["RecipeCategory"].isin(top_categories)].copy()
    plot_df = plot_df[
        [
            "RecipeId", "Name", "RecipeCategory",
            "AggregatedRating", "ReviewCount",
            "Calories", "ProteinContent", "ProteinPer100Cal"
        ]
    ]

    order = (
        plot_df.groupby("RecipeCategory")["ProteinPer100Cal"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    meta = {
        "fig_key": fig_key,
        "fig_type": "violin",
        "title": "Category Nutritional Spread",
        "x_col": "ProteinPer100Cal",
        "y_col": "RecipeCategory",
        "color_col": "RecipeCategory",
        "category_order": order,
        "metric": "ProteinPer100Cal",
        "metric_label": "Protein per 100 Calories",
    }

    fig = build_fig_category_spread(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)

def export_fig_tradeoff(df_landscape: pd.DataFrame, top_categories: list[str], output_dir: str):
    fig_key = "nutrition_tradeoff"

    plot_df = df_landscape[df_landscape["RecipeCategory"].isin(top_categories)].copy()

    rating = plot_df["AggregatedRating"].fillna(plot_df["AggregatedRating"].median())
    rating_min = rating.min()
    rating_max = rating.max()

    if pd.isna(rating_min) or pd.isna(rating_max) or rating_min == rating_max:
        plot_df["RatingSize"] = 10.0
    else:
        plot_df["RatingSize"] = 8 + 14 * (rating - rating_min) / (rating_max - rating_min)

    plot_df = plot_df[
        [
            "RecipeId", "Name", "RecipeCategory",
            "AggregatedRating", "ReviewCount",
            "Calories", "ProteinContent", "SugarContent", "SodiumContent",
            "ProteinPer100Cal", "SodiumPer100Cal",
            "NutritionBalanceScore", "Cluster", "RatingSize"
        ]
    ].copy()

    meta = {
        "fig_key": fig_key,
        "fig_type": "scatter",
        "title": "Nutritional Trade-offs",
        "x_col": "SodiumPer100Cal",
        "y_col": "ProteinPer100Cal",
        "color_col": "RecipeCategory",
        "size_col": "RatingSize",
        "category_order": top_categories,
        "x_scale": "log",
    }

    fig = build_fig_tradeoff(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)

def export_fig_cluster_heatmap(df_landscape: pd.DataFrame, output_dir: str):
    fig_key = "nutrition_cluster_heatmap"

    content_cols = [
        "Calories",
        "FatContent",
        "SugarContent",
        "ProteinContent",
        "FiberContent",
        "SodiumContent",
    ]

    per100cal_cols = [
        "Calories",
        "FatPer100Cal",
        "SugarPer100Cal",
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SodiumPer100Cal",
    ]

    all_profile_cols = list(dict.fromkeys(content_cols + per100cal_cols))

    cluster_profile = df_landscape.groupby("Cluster")[all_profile_cols].median().copy()
    cluster_profile = cluster_profile.sort_index(key=lambda x: x.astype(int))

    std = cluster_profile.std(ddof=0).replace(0, 1.0)
    cluster_profile_z = (cluster_profile - cluster_profile.mean()) / std

    long_rows = []
    for cluster in cluster_profile.index:
        for feat in all_profile_cols:
            long_rows.append({
                "Cluster": str(cluster),
                "Feature": feat,
                "MedianRaw": float(cluster_profile.loc[cluster, feat]),
                "ZMedian": float(cluster_profile_z.loc[cluster, feat]),
            })

    plot_df = pd.DataFrame(long_rows)

    meta = {
        "fig_key": fig_key,
        "fig_type": "double_heatmap",
        "title": "Nutrition Cluster Profiles",
        "content_feature_order": content_cols,
        "per100cal_feature_order": per100cal_cols,
        "cluster_order": [str(x) for x in cluster_profile.index.tolist()],
        "value_col": "ZMedian",
        "raw_value_col": "MedianRaw",
    }

    fig = build_fig_cluster_heatmap(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)

def export_fig_cluster_categories(df_landscape: pd.DataFrame, output_dir: str):
    fig_key = "nutrition_cluster_categories"

    if "RecipeCategory" not in df_landscape.columns:
        return None

    top_n = 5

    df = df_landscape.copy()
    df["Cluster"] = df["Cluster"].astype(str)

    counts = (
        df.groupby(["Cluster", "RecipeCategory"])
        .size()
        .reset_index(name="Count")
    )

    counts["Proportion"] = counts.groupby("Cluster")["Count"].transform(
        lambda x: x / x.sum()
    )

    counts = (
        counts.sort_values(
            ["Cluster", "Proportion", "RecipeCategory"],
            ascending=[True, False, True]
        )
        .groupby("Cluster", group_keys=False)
        .head(top_n)
        .copy()
    )

    plot_df = counts.rename(columns={"RecipeCategory": "Category"})

    cluster_order = sorted(plot_df["Cluster"].unique(), key=int)

    category_order = (
        plot_df.groupby("Category")["Proportion"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    meta = {
        "fig_key": fig_key,
        "fig_type": "cluster_category_composition",
        "title": "Dominant Recipe Categories within Each Cluster",
        "cluster_order": cluster_order,
        "category_order": category_order,
        "top_n_per_cluster": top_n,
    }

    fig = build_fig_cluster_categories(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)

def export_fig_pca_loadings(loadings_df: pd.DataFrame, output_dir: str):
    fig_key = "nutrition_pca_loadings"

    plot_df = loadings_df.copy()
    plot_df["abs_loading_strength"] = (
        plot_df["PC1_loading"].abs() + plot_df["PC2_loading"].abs()
    )
    plot_df = plot_df.sort_values("abs_loading_strength", ascending=False).reset_index(drop=True)

    meta = {
        "fig_key": fig_key,
        "fig_type": "bar",
        "title": "PCA Loadings",
        "x_col": "Feature",
        "y_cols": ["PC1_loading", "PC2_loading"],
    }

    fig = build_fig_loadings(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)


# -----------------------------
# RE-PLOTTING API
# -----------------------------
def replot_exported_nutrition_figure(fig_key: str, output_dir: str) -> go.Figure:
    fdir = figure_dir(output_dir, fig_key)

    metadata = safe_read_json(os.path.join(fdir, "metadata.json"))

    csv_path = os.path.join(fdir, "plot_data.csv")
    parquet_path = os.path.join(fdir, "plot_data.parquet")
    if os.path.exists(parquet_path):
        plot_df = load_exported_dataframe(parquet_path)
    elif os.path.exists(csv_path):
        plot_df = load_exported_dataframe(csv_path)
    else:
        raise FileNotFoundError(f"No exported plot data found for figure '{fig_key}'.")

    if fig_key == "nutrition_pca_landscape":
        return build_fig_pca_landscape(plot_df, metadata)
    if fig_key == "nutrition_pca_categories":
        return build_fig_pca_categories(plot_df, metadata)
    if fig_key == "nutrition_category_spread":
        return build_fig_category_spread(plot_df, metadata)
    if fig_key == "nutrition_tradeoff":
        return build_fig_tradeoff(plot_df, metadata)
    if fig_key == "nutrition_cluster_heatmap":
        return build_fig_cluster_heatmap(plot_df, metadata)
    if fig_key == "nutrition_pca_loadings":
        return build_fig_loadings(plot_df, metadata)
    if fig_key == "nutrition_cluster_categories":
        return build_fig_cluster_categories(plot_df, metadata)

    raise ValueError(f"Unknown figure key: {fig_key}")


# -----------------------------
# REPORTING
# -----------------------------
def print_pca_report(loadings_df: pd.DataFrame, explained_df: pd.DataFrame):
    explained = explained_df.set_index("Component")["ExplainedVarianceRatio"]

    print("\n" + "=" * 70)
    print("PCA EXPLAINED VARIANCE")
    print("=" * 70)
    print(explained.round(4))

    for comp in ["PC1_loading", "PC2_loading"]:
        comp_label = comp.replace("_loading", "")
        print("\n" + "=" * 70)
        print(f"TOP POSITIVE / NEGATIVE LOADINGS FOR {comp_label}")
        print("=" * 70)
        print(loadings_df[["Feature", comp]].sort_values(comp, ascending=False).head(6).round(4))
        print(loadings_df[["Feature", comp]].sort_values(comp, ascending=True).head(6).round(4))


def print_category_summary(summary: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CATEGORY SPREAD SUMMARY")
    print("=" * 70)
    print(summary.round(2).to_string(index=False))


def print_best_tradeoff_examples(df: pd.DataFrame, n: int = 10):
    cols_to_show = [
        "Name",
        "RecipeCategory",
        "Calories",
        "ProteinContent",
        "FiberContent",
        "SugarContent",
        "SodiumContent",
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SugarPer100Cal",
        "SodiumPer100Cal",
        "NutritionBalanceScore",
        "AggregatedRating",
        "ReviewCount",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]

    best = (
        df.sort_values(
            ["NutritionBalanceScore", "AggregatedRating", "ReviewCount"],
            ascending=[False, False, False]
        )
        .head(n)[cols_to_show]
        .copy()
    )

    worst = (
        df.sort_values(
            ["NutritionBalanceScore", "AggregatedRating", "ReviewCount"],
            ascending=[True, False, False]
        )
        .head(n)[cols_to_show]
        .copy()
    )

    print("\n" + "=" * 70)
    print("EXAMPLE RECIPES WITH STRONGER NUTRITIONAL TRADE-OFFS")
    print("=" * 70)
    print(best.round(2).to_string(index=False))

    print("\n" + "=" * 70)
    print("EXAMPLE RECIPES WITH WEAKER NUTRITIONAL TRADE-OFFS")
    print("=" * 70)
    print(worst.round(2).to_string(index=False))


# -----------------------------
# MANIFEST
# -----------------------------
def write_manifest(output_dir: str, entries: list[dict], global_meta: dict):
    manifest = {
        "visualization": "nutritional_landscape",
        "global_meta": global_meta,
        "figures": entries,
    }
    safe_write_json(manifest, os.path.join(output_dir, "nutrition_figure_manifest.json"))


# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    # 1. Load
    df = load_recipes_from_sqlite(RECIPE_FILE)

    # 2. Clean
    df = basic_recipe_cleaning(df)

    # 3. Features
    df = add_derived_features(df)
    feature_cols = select_features_for_landscape(df)

    # 4. PCA + clusters
    df_landscape, loadings_df, explained_df, scaler, pca, gmm = compute_pca_and_clusters(
        df=df,
        feature_cols=feature_cols,
        n_clusters=N_CLUSTERS
    )

    # 5. Top categories and summaries
    top_categories = get_top_categories(df_landscape, n=TOP_N_CATEGORIES)
    category_summary = summarize_category_spread(df_landscape, top_categories)

    # 6. Reporting
    print_pca_report(loadings_df, explained_df)
    print_category_summary(category_summary)
    print_best_tradeoff_examples(df_landscape, n=10)

    # 7. Save global processed tables
    export_dataframe(df_landscape, os.path.join(OUTPUT_DIR, "recipes_with_nutritional_landscape"), EXPORT_PARQUET)
    export_dataframe(loadings_df, os.path.join(OUTPUT_DIR, "pca_loadings"), EXPORT_PARQUET)
    export_dataframe(category_summary, os.path.join(OUTPUT_DIR, "category_spread_summary"), EXPORT_PARQUET)

    scaler_meta = {
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "gmm_weights": gmm.weights_.tolist(),
        "gmm_means": gmm.means_.tolist(),
        "gmm_covariances": gmm.covariances_.tolist(),
        "random_state": RANDOM_STATE,
        "n_clusters": N_CLUSTERS,
    }
    safe_write_json(scaler_meta, os.path.join(OUTPUT_DIR, "model_metadata.json"))

    # 8. Export figures individually
    manifest_entries = []
    manifest_entries.append(export_fig_pca_landscape(df_landscape, explained_df, OUTPUT_DIR))
    manifest_entries.append(export_fig_pca_categories(df_landscape, explained_df, top_categories, OUTPUT_DIR))
    manifest_entries.append(export_fig_category_spread(df_landscape, top_categories, OUTPUT_DIR))
    manifest_entries.append(export_fig_tradeoff(df_landscape, top_categories, OUTPUT_DIR))
    manifest_entries.append(export_fig_cluster_heatmap(df_landscape, OUTPUT_DIR))
    manifest_entries.append(export_fig_pca_loadings(loadings_df, OUTPUT_DIR))
    manifest_entries.append(export_fig_cluster_categories(df_landscape, OUTPUT_DIR))

    # 9. Write global manifest
    global_meta = {
        "recipe_file": str(RECIPE_FILE),
        "output_dir": str(OUTPUT_DIR),
        "top_n_categories": TOP_N_CATEGORIES,
        "top_categories": top_categories,
        "n_clusters": N_CLUSTERS,
        "min_reviews": MIN_REVIEWS,
        "max_missing_allowed": MAX_MISSING_ALLOWED,
        "feature_cols": feature_cols,
        "n_recipes_after_cleaning": int(len(df_landscape)),
    }
    write_manifest(OUTPUT_DIR, manifest_entries, global_meta)

    print("\nDone.")
    print(f"All outputs saved in: {OUTPUT_DIR}")
    print("Use `replot_exported_nutrition_figure(fig_key, OUTPUT_DIR)` to rebuild any figure later.")


if __name__ == "__main__":
    main()