#!/usr/bin/env python3
"""
Functions to build the cooking-time duration figure used by the web app.

The pipeline:
1. loads only the recipe/review columns needed for this visualization,
2. cleans and filters time-related fields,
3. aggregates review statistics per recipe,
4. builds an intermediate windrose-style payload, and
5. exports the final Plotly JSON payload for the app.

No processed table is exported; only the figure payload is written out.
"""

from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "tables" / "food_recipe.db"
OUT_DIR = BASE_DIR / "plots"
DEFAULT_WINDROSE_OUTPUT_DIR = OUT_DIR
JSON_FILENAME = "plot_duration.json"

MAX_TOTAL_TIME_SEC = 172_800 # hard cap to exclude extreme durations (> 48 hours).
TIME_CONSISTENCY_TOL = 60 # allowed mismatch between PrepTime + CookTime and TotalTime

TOTAL_TIME_BIN_EDGES_SEC = np.array([
    0,
    15 * 60,
    30 * 60,
    45 * 60,
    60 * 60,
    90 * 60,
    120 * 60,
    240 * 60,
    MAX_TOTAL_TIME_SEC,
], dtype=float)
TOTAL_TIME_BIN_LABELS = [
    "0–15 min",
    "15–30 min",
    "30–45 min",
    "45–60 min",
    "1–1.5 h",
    "1.5–2 h",
    "2–4 h",
    ">4 h",
]
N_TOTAL_TIME_SECTORS = len(TOTAL_TIME_BIN_LABELS)
ANGLE_TOTAL_TIME = "total_time"

TOP_CATEGORIES_PER_WEDGE = 4
CATEGORY_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
    "#FF9DA6", "#9D755D", "#BAB0AC", "#2E91E5", "#E15F99", "#1CA71C",
    "#FB0D0D", "#DA16FF", "#222A2A", "#B68100", "#750D86", "#EB663B",
]

APP_USED_FIGURES = {
    "windrose_total_time_population": "windrose_total_time_population",
}
WEBAPP_PANELS = {
    "duration_main": "windrose_total_time_population",
}


# ##################
# Data loading / preparation
# ##################

# load only the recipe and review columns needed for the duration figure.
def load_duration_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    print("Loading data from SQLite ...")
    con = sqlite3.connect(DB_PATH)

    recipes = pd.read_sql_query(
        """
        SELECT RecipeId,
               PrepTime, CookTime, TotalTime,
               RecipeCategory
        FROM recipes
        """,
        con,
    )

    reviews = pd.read_sql_query(
        """
        SELECT RecipeId, Rating
        FROM reviews
        WHERE Rating IS NOT NULL
        """,
        con,
    )

    con.close()
    print(f"    recipes loaded : {len(recipes):,}")
    print(f"    reviews loaded : {len(reviews):,}")
    return recipes, reviews

# clean recipe time columns and assign each recipe to a total-time sector
def clean_recipe_times(recipes: pd.DataFrame) -> pd.DataFrame:

    print("Cleaning and filtering recipes ...")
    df = recipes.copy()
    n0 = len(df)

    for col in ["PrepTime", "CookTime", "TotalTime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["PrepTime", "CookTime", "TotalTime"])
    df = df[(df["PrepTime"] > 0) & (df["CookTime"] > 0) & (df["TotalTime"] > 0)]

    df = df[df["TotalTime"] <= MAX_TOTAL_TIME_SEC] # remove extreme total times that are likely bad records

    deviation = (df["PrepTime"] + df["CookTime"] - df["TotalTime"]).abs()
    df = df[deviation <= TIME_CONSISTENCY_TOL].copy() # keep only rows where TotalTime is approximately PrepTime + CookTime

    df["sector_total_time"] = pd.cut(
        df["TotalTime"],
        bins=TOTAL_TIME_BIN_EDGES_SEC,
        labels=False,
        include_lowest=True,
        right=True,
    ).fillna(N_TOTAL_TIME_SECTORS - 1).astype(int)

    print(f"    kept {len(df):,} / {n0:,} recipes")
    return df

def attach_recipe_review_stats(recipes: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:

    print("Aggregating review statistics ...")
    review_stats = (
        reviews.groupby("RecipeId")
        .agg(mean_rating_r=("Rating", "mean"), review_count_r=("Rating", "count"))
        .reset_index()
    )

    out = recipes.merge(review_stats, on="RecipeId", how="inner")
    print(f"    recipes with at least 1 valid review: {len(out):,}")
    return out


# ##################
# Figure builder helpers
# ##################

def build_total_time_category_population_payload(
    recipes_out: pd.DataFrame,
    *,
    top_n: int = TOP_CATEGORIES_PER_WEDGE,
    total_time_labels: list[str] | None = None,
    total_time_edges: list[float] | None = None,
) -> dict:
    """
    Build the intermediate payload for the total-time windrose figure

    Each angular sector corresponds to one total-time bin. The wedge radius is
    proportional to sqrt(N_bin / N_total), so sector area scales with the share
    of recipes in that time regime. Within each wedge, the top recipe
    categories are stacked as radial segments

    Returns:
    dict
        A plotting payload containing per-sector geometry, category segments,
        and figure metadata.
    """
    work = recipes_out.copy()

    work["RecipeCategory"] = work["RecipeCategory"].fillna("").astype(str).str.strip()
    work = work[(work["RecipeCategory"] != "") & (work["RecipeCategory"].str.lower() != "nan")].copy()
    if work.empty:
        raise ValueError("No valid RecipeCategory values available for population windrose.")

    labels = total_time_labels or TOTAL_TIME_BIN_LABELS
    edges = total_time_edges or TOTAL_TIME_BIN_EDGES_SEC.tolist()
    n_bins = len(labels)
    total_recipes = int(len(work))

    sector_rows: list[dict] = []
    all_categories: set[str] = set()

    for sector_idx in range(n_bins):
        group = work[work["sector_total_time"] == sector_idx].copy()
        recipe_count = int(len(group))

        # radius is scaled by sqrt(share) so wedge area, not radius itself, reflects the population fraction.
        radius = float(np.sqrt(recipe_count / total_recipes)) if recipe_count > 0 else 0.0

        # keep only the top categories within this wedge
        cat_counts = group["RecipeCategory"].value_counts().head(top_n)
        segments = []
        for cat, count in cat_counts.items():
            count_i = int(count)
            frac = count_i / recipe_count if recipe_count else 0.0

            # each segment gets a fraction of the wedge radius proportional to its share within the wedge
            seg_r = radius * frac
            segments.append({
                "category": cat,
                "count": count_i,
                "fraction_in_wedge": float(frac),
                "r_segment": float(seg_r),
            })
            all_categories.add(cat)

        # compute review-weighted average rating for this wedge
        # review count acts as the weight so recipes with more reviews have
        # more influence on the wedge-level rating summary
        weights = pd.to_numeric(group.get("review_count_r", 0), errors="coerce").fillna(0.0).clip(lower=0.0)
        ratings = pd.to_numeric(group.get("mean_rating_r", np.nan), errors="coerce")
        valid = ratings.notna()
        if recipe_count and valid.any() and float(weights[valid].sum()) > 0:
            weighted_rating = float(np.average(ratings[valid], weights=weights[valid]))
        elif recipe_count and valid.any():
            # fallback to unweighted mean if weights are all zero
            weighted_rating = float(ratings[valid].mean())
        else:
            weighted_rating = float("nan")

        total_reviews = int(weights.sum()) if recipe_count else 0
        avg_time_min = (
            float(pd.to_numeric(group.get("TotalTime", np.nan), errors="coerce").mean() / 60.0)
            if recipe_count else float("nan")
        )

        sector_rows.append({
            "sector_idx": sector_idx,
            "label": labels[sector_idx],
            "angle_deg": float(sector_idx * (360 / n_bins)),
            "sector_width_deg": float(360 / n_bins),
            "recipe_count": recipe_count,
            "radius": radius,
            "weighted_avg_rating": weighted_rating,
            "review_count": total_reviews,
            "mean_total_time_min": avg_time_min,
            "segments": segments,
        })

    radial_max = max((row["radius"] for row in sector_rows), default=1.0)
    rating_values = [row["weighted_avg_rating"] for row in sector_rows if np.isfinite(row["weighted_avg_rating"])]

    return {
        "metadata": {
            "angle_mode": ANGLE_TOTAL_TIME,
            "angle_axis_title": "Total time",
            "hover_axis_label": "Total time bin",
            "bin_labels": labels,
            "bin_edges": edges,
            "bin_edge_units": "seconds",
            "n_sectors_total": n_bins,
            "sector_width_deg": 360 / n_bins,
            "angular_direction": "clockwise",
            "rotation_deg": 90,
            "top_n_categories": int(top_n),
            "total_recipes_used": total_recipes,
            "radial_max_main": float(radial_max),
            "rating_min": float(min(rating_values)) if rating_values else 1.0,
            "rating_max": float(max(rating_values)) if rating_values else 5.0,
            "all_categories": sorted(all_categories),
        },
        "sectors": sector_rows,
    }

def build_total_time_category_population_figure(plot_data: dict) -> go.Figure:
    """
    Convert the intermediate total-time payload into the final Plotly figure

    The main stacked Barpolar traces encode category composition within each
    time wedge, while the outer ring encodes the wedge-level weighted average
    rating
    """
    meta = plot_data["metadata"]
    sectors = plot_data["sectors"]
    if not sectors:
        raise ValueError("No sectors in plot payload — cannot build figure.")

    all_categories = meta.get("all_categories", [])
    color_map = {
        cat: CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
        for i, cat in enumerate(all_categories)
    }

    theta = [sector["angle_deg"] for sector in sectors]
    width = [meta["sector_width_deg"]] * len(sectors)

    # These values control the overall radial geometry of the figure.
    # radial_max_main is intentionally fixed here for display stability across runs.
    radial_max_main = 0.20  # float(meta.get("radial_max_main", 1.0))
    ring_gap = min(0.03, radial_max_main * 0.01)
    ring_base = radial_max_main + ring_gap
    ring_thickness_min = min(0.018, radial_max_main * 0.05)
    ring_thickness_span = min(0.025, radial_max_main * 0.05)

    traces: list[go.Barpolar] = []

    # One stacked Barpolar trace per category.
    # Categories absent from a wedge get r=0 for that wedge.
    for cat in all_categories:
        r_vals = []
        hover = []
        for sector in sectors:
            seg = next((seg for seg in sector["segments"] if seg["category"] == cat), None)
            if seg is None:
                r_vals.append(0.0)
                hover.append("")
                continue
            r_vals.append(float(seg["r_segment"]))
            hover.append(
                f"<b>{meta['hover_axis_label']}: {sector['label']}</b><br>"
                f"Category: {cat}<br>"
                f"Recipes in category segment: {seg['count']:,}<br>"
                f"Segment share of wedge: {seg['fraction_in_wedge'] * 100:.1f}%<br>"
                f"Recipes in wedge: {sector['recipe_count']:,}<br>"
                f"Wedge radius: {sector['radius']:.3f}<br>"
                f"Weighted avg rating: {sector['weighted_avg_rating']:.3f} ★<br>"
                f"Reviews in wedge: {sector['review_count']:,}"
            )

        traces.append(go.Barpolar(
            r=r_vals,
            theta=theta,
            width=width,
            name=cat,
            marker_color=color_map[cat],
            marker_line=dict(color="black", width=1.2),
            opacity=0.95,
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        ))

    rating_min = float(meta.get("rating_min", 1.0))
    rating_max = float(meta.get("rating_max", 5.0))
    ring_r = []
    ring_hover = []
    ring_color = []

    # build the outer rating ring,
    # ring thickness varies slightly with rating, and ring color also encodes rating
    for sector in sectors:
        rating = sector.get("weighted_avg_rating")
        if np.isfinite(rating):
            norm = 0.5 if rating_max <= rating_min else (float(rating) - rating_min) / (rating_max - rating_min)
            ring_r.append(ring_thickness_min + ring_thickness_span * norm)
            ring_color.append(float(rating))
            ring_hover.append(
                f"<b>{meta['hover_axis_label']}: {sector['label']}</b><br>"
                f"Weighted avg rating: {float(rating):.3f} ★<br>"
                f"Reviews in wedge: {sector['review_count']:,}<br>"
                f"Recipes in wedge: {sector['recipe_count']:,}"
            )
        else:
            ring_r.append(0.0)
            ring_color.append(rating_min)
            ring_hover.append(
                f"<b>{meta['hover_axis_label']}: {sector['label']}</b><br>"
                "No review-based rating available"
            )

    traces.append(go.Barpolar(
        r=ring_r,
        theta=theta,
        width=width,
        base=[ring_base] * len(sectors),
        marker=dict(
            color=ring_color,
            colorscale="Blues",
            cmin=1.0,
            cmax=5.0,
            line=dict(color="black", width=0.8),
            showscale=False,
        ),
        opacity=0.95,
        hovertext=ring_hover,
        hoverinfo="text",
        showlegend=False,
    ))

    max_range = ring_base + ring_thickness_min + ring_thickness_span + ring_gap * 0.85
    tick_vals = np.linspace(0, radial_max_main, 4)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=None,
        polar=dict(
            bargap=0.04,
            angularaxis=dict(
                tickmode="array",
                tickvals=theta,
                ticktext=[sector["label"] for sector in sectors],
                direction="clockwise",
                rotation=90,
                tickfont=dict(size=13),
                showline=True,
                linewidth=1.4,
                linecolor="black",
                gridcolor="black",
                griddash="solid",
                gridwidth=1.2,
            ),
            radialaxis=dict(
                range=[0, max_range],
                tickvals=tick_vals.tolist(),
                ticktext=[f"{t:.2f}" for t in tick_vals],
                tickfont=dict(size=12, color="rgba(110,110,110,0.95)"),
                showline=True,
                linewidth=1.7,
                gridcolor="black",
                griddash="solid",
                gridwidth=1.5,
                linecolor="black",
                angle=67.5,
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=30, l=28, r=28),
        autosize=True,
        barmode="stack",
    )
    return fig


# ##################
# Export writer
# ##################

class _PathAwarePlotlyEncoder(PlotlyJSONEncoder):
    """
    Plotly JSON encoder with extra support for pathlib.Path objects

    Standard json.dump cannot serialize Path directly, so paths are converted
    to strings before export
    """
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def export_plotly_payload(payload: dict, output_dir: str | Path, filename: str = JSON_FILENAME) -> Path:
    """
    Write the final Plotly payload to disk and return the output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, cls=_PathAwarePlotlyEncoder)
    return out_path


# ##################
# Full payload builder
# ##################

def build_duration_payload(recipes_out: pd.DataFrame) -> dict:
    """
    Build the final app payload containing the duration figure as Plotly JSON
    """
    intermediate = build_total_time_category_population_payload(
        recipes_out,
        top_n=TOP_CATEGORIES_PER_WEDGE,
        total_time_labels=TOTAL_TIME_BIN_LABELS,
        total_time_edges=TOTAL_TIME_BIN_EDGES_SEC.tolist(),
    )
    fig = build_total_time_category_population_figure(intermediate)

    return {
        "standalone_figures": {
            APP_USED_FIGURES["windrose_total_time_population"]: fig.to_plotly_json(),
        },
        "webapp_panels": dict(WEBAPP_PANELS),
        "meta": {
            "angle_mode": ANGLE_TOTAL_TIME,
            "n_total_time_sectors": N_TOTAL_TIME_SECTORS,
            "top_categories_per_wedge": TOP_CATEGORIES_PER_WEDGE,
        },
    }


# ##################
# main()
# ##################

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recipes, reviews = load_duration_data()
    recipes = clean_recipe_times(recipes)
    recipes_out = attach_recipe_review_stats(recipes, reviews)

    print("Building and exporting app payload ...")
    payload = build_duration_payload(recipes_out)
    payload_path = export_plotly_payload(payload, OUT_DIR, JSON_FILENAME)
    print(f"    JSON → {payload_path}")
    print("\nDone.")
if __name__ == "__main__":
    main()
