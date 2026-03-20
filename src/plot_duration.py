"""
Effort–Reward windrose pipeline.

This version supports two angular encodings:
  1. cook_fraction  — CookTime / (PrepTime + CookTime)
  2. total_time     — TotalTime split into 16 global bins

For each angular mode, the pipeline can export both rating/hour and WER payloads,
plus per-category payload collections for quick re-plotting in the web app.
"""

import json
import re
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "tables" / "food_recipe.db"
OUT_DIR = BASE_DIR / "plots" / "cooking_time_outputs"
DEFAULT_WINDROSE_OUTPUT_DIR = OUT_DIR

N_SECTORS = 16
MIN_RECIPES_PER_SECTOR = 10
MIN_REVIEWS_PER_SECTOR = 30
MAX_TOTAL_TIME_SEC = 172_800
TIME_CONSISTENCY_TOL = 60

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

ANGLE_COOK_FRACTION = "cook_fraction"
ANGLE_TOTAL_TIME = "total_time"
ANGLE_MODES = (ANGLE_COOK_FRACTION, ANGLE_TOTAL_TIME)

TONE_THRESHOLDS = {
    "strongly_positive": 0.50,
    "mildly_positive": 0.05,
    "neutral_mod": -0.05,
    "mildly_negative": -0.50,
}

MODIFICATION_KEYWORDS = re.compile(
    r"\b(added?|add|substitut\w*|swap\w*|replac\w*|next time|doubled?|"
    r"halved?|used? instead|omit\w*|skip\w*|modify|modif\w*|tweak\w*|"
    r"adjust\w*|change\w*|forgot|less \w+|more \w+|cut back|extra)\b",
    re.IGNORECASE,
)

TONE_GROUPS = [
    "strongly_positive",
    "mildly_positive",
    "modification",
    "mildly_negative",
    "strongly_negative",
]

COLORS = {
    "strongly_positive": "#1D9E75",
    "mildly_positive": "#8ECFC9",
    "modification": "#EF9F27",
    "mildly_negative": "#F0997B",
    "strongly_negative": "#D85A30",
}

TONE_DISPLAY = {
    "strongly_positive": "Strongly positive",
    "mildly_positive": "Mildly positive",
    "modification": "Modification / neutral",
    "mildly_negative": "Mildly negative",
    "strongly_negative": "Strongly negative",
}


def sector_label_fraction(sector_idx: int, n: int = N_SECTORS) -> str:
    step = 100 / n
    lo = round(sector_idx * step)
    hi = round((sector_idx + 1) * step)
    return f"{lo}–{hi}%"


def format_duration_seconds(seconds: float) -> str:
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{int(round(minutes))} min"
    hours = minutes / 60.0
    if hours < 10:
        return f"{hours:.1f} h"
    return f"{int(round(hours))} h"


def total_time_bin_labels(edges: np.ndarray) -> list[str]:
    if len(edges) == len(TOTAL_TIME_BIN_EDGES_SEC) and np.allclose(edges, TOTAL_TIME_BIN_EDGES_SEC):
        return TOTAL_TIME_BIN_LABELS.copy()

    labels = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == len(edges) - 2 and hi >= MAX_TOTAL_TIME_SEC:
            labels.append(f">{format_duration_seconds(float(lo))}")
        else:
            lo_txt = format_duration_seconds(float(lo))
            hi_txt = format_duration_seconds(float(hi))
            labels.append(f"{lo_txt}–{hi_txt}")
    return labels


def classify_tone(compound: float, text: str) -> str:
    has_mod = bool(MODIFICATION_KEYWORDS.search(text or ""))
    if has_mod and compound > -0.20:
        return "modification"
    if compound >= TONE_THRESHOLDS["strongly_positive"]:
        return "strongly_positive"
    if compound >= TONE_THRESHOLDS["mildly_positive"]:
        return "mildly_positive"
    if compound <= TONE_THRESHOLDS["mildly_negative"]:
        return "strongly_negative"
    if compound <= TONE_THRESHOLDS["neutral_mod"]:
        return "mildly_negative"
    return "modification"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[1/5] Loading data from SQLite …")
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
        SELECT RecipeId, Rating, Review
        FROM reviews
        WHERE Review IS NOT NULL AND TRIM(Review) != ''
          AND Rating IS NOT NULL
        """,
        con,
    )
    con.close()
    print(f"    recipes loaded : {len(recipes):,}")
    print(f"    reviews loaded : {len(reviews):,}")
    return recipes, reviews


def clean_recipes(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Cleaning and filtering recipes …")
    n0 = len(df)
    drop_log: dict[str, int] = {}

    for col in ["PrepTime", "CookTime", "TotalTime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["PrepTime", "CookTime", "TotalTime"])
    drop_log["null time"] = before - len(df)

    before = len(df)
    df = df[(df["PrepTime"] > 0) & (df["CookTime"] > 0) & (df["TotalTime"] > 0)]
    drop_log["non-positive time"] = before - len(df)

    before = len(df)
    df = df[df["TotalTime"] <= MAX_TOTAL_TIME_SEC]
    drop_log[f"TotalTime > {MAX_TOTAL_TIME_SEC}s"] = before - len(df)

    before = len(df)
    deviation = (df["PrepTime"] + df["CookTime"] - df["TotalTime"]).abs()
    df = df[deviation <= TIME_CONSISTENCY_TOL].copy()
    drop_log[f"time inconsistency > {TIME_CONSISTENCY_TOL}s"] = before - len(df)

    denom = df["PrepTime"] + df["CookTime"]
    df["cook_fraction"] = (df["CookTime"] / denom).clip(0.0, 1.0)
    df["sector_cook_fraction"] = (
        np.floor(df["cook_fraction"] * N_SECTORS).clip(0, N_SECTORS - 1).astype(int)
    )

    time_edges = TOTAL_TIME_BIN_EDGES_SEC.copy()
    # Include the maximum in the last bin.
    df["sector_total_time"] = pd.cut(
        df["TotalTime"],
        bins=time_edges,
        labels=False,
        include_lowest=True,
        right=True,
    ).fillna(N_TOTAL_TIME_SECTORS - 1).astype(int)

    df.attrs["total_time_edges"] = time_edges.tolist()
    df.attrs["total_time_labels"] = total_time_bin_labels(time_edges)

    n1 = len(df)
    print(f"    kept {n1:,} / {n0:,} recipes  ({n0 - n1:,} dropped)")
    for reason, count in drop_log.items():
        if count:
            print(f"      − {count:>7,}  {reason}")
    return df


def score_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Scoring sentiment with VADER …")
    print(f"    processing {len(reviews):,} reviews — this may take a minute …")

    # reviews = reviews.head(10_000).copy()
    analyzer = SentimentIntensityAnalyzer()
    compounds = []
    chunk_size = 100_000
    for start in range(0, len(reviews), chunk_size):
        chunk = reviews["Review"].iloc[start:start + chunk_size]
        compounds.extend(analyzer.polarity_scores(str(text))["compound"] for text in chunk)
        print(
            f"    … {min(start + chunk_size, len(reviews)):,} / {len(reviews):,} done",
            end="\r",
        )
    print()

    reviews["compound"] = compounds
    reviews["tone"] = [classify_tone(c, t) for c, t in zip(reviews["compound"], reviews["Review"])]
    print(f"    tone distribution (5 groups): {reviews['tone'].value_counts().to_dict()}")
    return reviews


def _angle_config(angle_mode: str, recipes: pd.DataFrame) -> dict:
    if angle_mode == ANGLE_COOK_FRACTION:
        return {
            "angle_mode": ANGLE_COOK_FRACTION,
            "sector_col": "sector_cook_fraction",
            "labels": [sector_label_fraction(i) for i in range(N_SECTORS)],
            "annotation_top": "← more prep",
            "annotation_bottom": "← more cook",
            "hover_axis_label": "Cook fraction bin",
            "angle_axis_title": "Cook fraction",
            "bin_edges": [i / N_SECTORS for i in range(N_SECTORS + 1)],
            "bin_edge_units": "fraction",
        }
    if angle_mode == ANGLE_TOTAL_TIME:
        return {
            "angle_mode": ANGLE_TOTAL_TIME,
            "sector_col": "sector_total_time",
            "labels": recipes.attrs.get("total_time_labels", [str(i) for i in range(N_TOTAL_TIME_SECTORS)]),
            "annotation_top": "← shorter total time",
            "annotation_bottom": "← longer total time",
            "hover_axis_label": "Total time bin",
            "angle_axis_title": "Total time",
            "bin_edges": recipes.attrs.get("total_time_edges", []),
            "bin_edge_units": "seconds",
        }
    raise ValueError(f"Unsupported angle_mode: {angle_mode}")


def _weighted_tone_fraction(group: pd.DataFrame) -> dict[str, float]:
    weights = pd.to_numeric(group.get("review_count_scored", 0), errors="coerce").fillna(0.0)
    total_w = float(weights.sum())
    out: dict[str, float] = {}
    if total_w > 0:
        for tone in TONE_GROUPS:
            values = pd.to_numeric(group.get(tone, 0), errors="coerce").fillna(0.0)
            out[tone] = float((values * weights).sum() / total_w)
    else:
        for tone in TONE_GROUPS:
            values = pd.to_numeric(group.get(tone, 0), errors="coerce").fillna(0.0)
            out[tone] = float(values.mean()) if len(values) else 0.0
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


def _make_plot_payload_from_sector_table(
    sector_table: pd.DataFrame,
    *,
    metric_col: str,
    metric_label: str,
    angle_cfg: dict,
    n_recipes_after_cleaning: int,
    n_reviews_scored: int,
) -> dict:
    sector_table = sector_table.sort_values("sector_idx").reset_index(drop=True)
    r_total = sector_table[metric_col].to_numpy(dtype=float)
    r_max = float(r_total.max()) * 1.15 if len(r_total) else 1.0
    tick_vals = [round(r_max * k / 4, 4) for k in range(5)]
    labels = angle_cfg["labels"]
    n_angle_bins = len(labels)

    sectors_out = []
    for _, row in sector_table.iterrows():
        r_t = float(row[metric_col])
        seg = {tone: round(float(row[tone]) * r_t, 6) for tone in TONE_GROUPS}
        idx = int(row["sector_idx"])
        sectors_out.append({
            "sector_idx": idx,
            "label": labels[idx],
            "angle_deg": float(idx * (360 / n_angle_bins)),
            "sector_width_deg": float(360 / n_angle_bins),
            "r_total": round(r_t, 6),
            "segments": seg,
            "tone_fractions": {t: round(float(row[t]), 6) for t in TONE_GROUPS},
            "recipe_count": int(row["recipe_count"]),
            "review_count": int(row["review_count"]),
            "mean_rating": round(float(row["mean_rating"]), 4),
            "mean_total_time_min": round(float(row["mean_total_time_min"]), 2),
            "mean_review_count": round(float(row["mean_review_count"]), 2),
            "mean_rating_per_hour": round(float(row["mean_rating_per_hour"]), 4),
            "mean_wer": round(float(row["mean_wer"]), 4),
        })

    return {
        "metadata": {
            "metric_col": metric_col,
            "metric_label": metric_label,
            "angle_mode": angle_cfg["angle_mode"],
            "angle_axis_title": angle_cfg["angle_axis_title"],
            "hover_axis_label": angle_cfg["hover_axis_label"],
            "annotation_top": angle_cfg["annotation_top"],
            "annotation_bottom": angle_cfg["annotation_bottom"],
            "bin_labels": labels,
            "bin_edges": angle_cfg["bin_edges"],
            "bin_edge_units": angle_cfg["bin_edge_units"],
            "n_sectors_total": n_angle_bins,
            "n_sectors_plotted": int(len(sector_table)),
            "sector_width_deg": 360 / n_angle_bins,
            "angular_direction": "clockwise",
            "rotation_deg": 90,
            "tone_groups": TONE_GROUPS,
            "tone_colors": COLORS,
            "tone_display_names": TONE_DISPLAY,
            "radial_max": round(r_max, 4),
            "radial_tick_vals": tick_vals,
            "radial_tick_text": [str(v) for v in tick_vals],
            "min_recipes_per_sector": MIN_RECIPES_PER_SECTOR,
            "min_reviews_per_sector": MIN_REVIEWS_PER_SECTOR,
            "time_consistency_tol_sec": TIME_CONSISTENCY_TOL,
            "vader_thresholds": TONE_THRESHOLDS,
            "rating_source": "individual reviewer Rating from reviews table",
            "n_recipes_after_cleaning": int(n_recipes_after_cleaning),
            "n_reviews_scored": int(n_reviews_scored),
        },
        "sectors": sectors_out,
    }


def aggregate(recipes: pd.DataFrame, reviews: pd.DataFrame) -> tuple[dict[str, dict], pd.DataFrame]:
    print("[4/5] Aggregating to sector × tone …")

    recipe_cols = [
        "RecipeId", "RecipeCategory", "PrepTime", "CookTime", "TotalTime",
        "cook_fraction", "sector_cook_fraction", "sector_total_time",
    ]
    recipe_sectors = recipes[recipe_cols].copy()

    reviews_valid = reviews[reviews["Rating"].between(1, 5)].copy()
    per_recipe_stats = (
        reviews_valid.groupby("RecipeId")
        .agg(mean_rating_r=("Rating", "mean"), review_count_r=("Rating", "count"))
        .reset_index()
    )
    recipe_sectors = recipe_sectors.merge(per_recipe_stats, on="RecipeId", how="inner")
    print(f"    recipes with at least 1 valid review: {len(recipe_sectors):,}")

    hours = (recipe_sectors["TotalTime"] / 3600).replace(0, np.nan)
    recipe_sectors["rating_per_hour"] = recipe_sectors["mean_rating_r"] / hours
    recipe_sectors["wer"] = (
        recipe_sectors["mean_rating_r"] * np.log1p(recipe_sectors["review_count_r"]) / hours
    )

    recipe_tone = (
        reviews_valid.groupby(["RecipeId", "tone"]).size()
        .unstack(fill_value=0)
        .reindex(columns=TONE_GROUPS, fill_value=0)
        .astype(float)
    )
    recipe_tone["review_count_scored"] = recipe_tone.sum(axis=1)
    has_rev = recipe_tone["review_count_scored"] > 0
    recipe_tone.loc[has_rev, TONE_GROUPS] = (
        recipe_tone.loc[has_rev, TONE_GROUPS]
        .div(recipe_tone.loc[has_rev, "review_count_scored"], axis=0)
    )
    recipe_tone = recipe_tone.reset_index()

    recipes_out = recipe_sectors.merge(recipe_tone, on="RecipeId", how="left")
    recipes_out["sector_label_cook_fraction"] = recipes_out["sector_cook_fraction"].apply(sector_label_fraction)
    total_labels = recipes.attrs.get("total_time_labels", [str(i) for i in range(N_TOTAL_TIME_SECTORS)])
    recipes_out["sector_label_total_time"] = recipes_out["sector_total_time"].map(lambda i: total_labels[int(i)])

    for col in ["cook_fraction", "mean_rating_r", "rating_per_hour", "wer"] + TONE_GROUPS:
        if col in recipes_out.columns:
            recipes_out[col] = recipes_out[col].round(4)

    payloads: dict[str, dict] = {}
    n_reviews_scored = int(len(reviews_valid.merge(recipe_sectors[["RecipeId"]], on="RecipeId", how="inner")))

    for angle_mode in ANGLE_MODES:
        angle_cfg = _angle_config(angle_mode, recipes)
        sector_col = angle_cfg["sector_col"]
        rev_joined = reviews_valid.merge(
            recipe_sectors[["RecipeId", sector_col]].rename(columns={sector_col: "sector_idx"}),
            on="RecipeId",
            how="inner",
        )

        sector_stats = (
            recipe_sectors.groupby(sector_col)
            .agg(
                mean_rating=("mean_rating_r", "mean"),
                median_rating=("mean_rating_r", "median"),
                recipe_count=("mean_rating_r", "count"),
                mean_total_time_min=("TotalTime", lambda x: x.mean() / 60),
                mean_review_count=("review_count_r", "mean"),
                mean_rating_per_hour=("rating_per_hour", "mean"),
                mean_wer=("wer", "mean"),
            )
            .reset_index()
            .rename(columns={sector_col: "sector_idx"})
        )

        tone_counts = (
            rev_joined.groupby(["sector_idx", "tone"]).size()
            .unstack(fill_value=0)
            .reindex(columns=TONE_GROUPS, fill_value=0)
            .astype(float)
        )
        tone_counts["review_count"] = tone_counts.sum(axis=1)
        tone_fractions = tone_counts[TONE_GROUPS].div(tone_counts["review_count"], axis=0)
        tone_fractions["review_count"] = tone_counts["review_count"]
        tone_fractions = tone_fractions.reset_index()

        agg = sector_stats.merge(tone_fractions, on="sector_idx", how="left")
        agg = agg[(agg["recipe_count"] >= MIN_RECIPES_PER_SECTOR) & (agg["review_count"] >= MIN_REVIEWS_PER_SECTOR)].copy()
        print(f"    {angle_mode}: {len(agg)} / {len(angle_cfg['labels'])} sectors retained after minimum-count filter")

        payloads[f"{angle_mode}__rating_per_hour"] = _make_plot_payload_from_sector_table(
            agg,
            metric_col="mean_rating_per_hour",
            metric_label="Rating / hour",
            angle_cfg=angle_cfg,
            n_recipes_after_cleaning=len(recipe_sectors),
            n_reviews_scored=n_reviews_scored,
        )
        payloads[f"{angle_mode}__wer"] = _make_plot_payload_from_sector_table(
            agg,
            metric_col="mean_wer",
            metric_label="Weighted engagement rate",
            angle_cfg=angle_cfg,
            n_recipes_after_cleaning=len(recipe_sectors),
            n_reviews_scored=n_reviews_scored,
        )

    print(f"    per-recipe rows: {len(recipes_out):,}")
    return payloads, recipes_out


def build_plot_payload_from_recipe_frame(
    recipes_frame: pd.DataFrame,
    *,
    angle_mode: str,
    metric: str,
    total_time_labels: list[str] | None = None,
    total_time_edges: list[float] | None = None,
) -> dict | None:
    if recipes_frame.empty:
        return None

    if metric not in {"wer", "rating_per_hour"}:
        raise ValueError("metric must be 'wer' or 'rating_per_hour'")

    if angle_mode == ANGLE_COOK_FRACTION:
        sector_col = "sector_cook_fraction"
        labels = [sector_label_fraction(i) for i in range(N_SECTORS)]
        angle_cfg = {
            "angle_mode": ANGLE_COOK_FRACTION,
            "labels": labels,
            "annotation_top": "← more prep",
            "annotation_bottom": "← more cook",
            "hover_axis_label": "Cook fraction bin",
            "angle_axis_title": "Cook fraction",
            "bin_edges": [i / N_SECTORS for i in range(N_SECTORS + 1)],
            "bin_edge_units": "fraction",
        }
    elif angle_mode == ANGLE_TOTAL_TIME:
        if not total_time_labels or not total_time_edges:
            raise ValueError("total_time_labels and total_time_edges are required for total_time mode")
        sector_col = "sector_total_time"
        angle_cfg = {
            "angle_mode": ANGLE_TOTAL_TIME,
            "labels": total_time_labels,
            "annotation_top": "← shorter total time",
            "annotation_bottom": "← longer total time",
            "hover_axis_label": "Total time bin",
            "angle_axis_title": "Total time",
            "bin_edges": total_time_edges,
            "bin_edge_units": "seconds",
        }
    else:
        raise ValueError(f"Unsupported angle_mode: {angle_mode}")

    metric_col = "mean_wer" if metric == "wer" else "mean_rating_per_hour"
    metric_label = "Weighted engagement rate" if metric == "wer" else "Rating / hour"

    sector_rows: list[dict] = []
    for sector_idx, group in recipes_frame.groupby(sector_col):
        recipe_count = int(len(group))
        review_count = int(pd.to_numeric(group["review_count_scored"], errors="coerce").fillna(0).sum())
        if recipe_count < MIN_RECIPES_PER_SECTOR or review_count < MIN_REVIEWS_PER_SECTOR:
            continue
        tone_frac = _weighted_tone_fraction(group)
        sector_rows.append({
            "sector_idx": int(sector_idx),
            "recipe_count": recipe_count,
            "review_count": review_count,
            "mean_rating": float(pd.to_numeric(group["mean_rating_r"], errors="coerce").mean()),
            "mean_total_time_min": float(pd.to_numeric(group["TotalTime"], errors="coerce").mean() / 60.0),
            "mean_review_count": float(pd.to_numeric(group["review_count_r"], errors="coerce").mean()),
            "mean_rating_per_hour": float(pd.to_numeric(group["rating_per_hour"], errors="coerce").mean()),
            "mean_wer": float(pd.to_numeric(group["wer"], errors="coerce").mean()),
            **tone_frac,
        })

    if not sector_rows:
        return None

    sector_table = pd.DataFrame(sector_rows)
    return _make_plot_payload_from_sector_table(
        sector_table,
        metric_col=metric_col,
        metric_label=metric_label,
        angle_cfg=angle_cfg,
        n_recipes_after_cleaning=len(recipes_frame),
        n_reviews_scored=int(pd.to_numeric(recipes_frame["review_count_scored"], errors="coerce").fillna(0).sum()),
    )


def build_category_payloads(
    recipes_out: pd.DataFrame,
    *,
    total_time_labels: list[str],
    total_time_edges: list[float],
) -> tuple[dict[str, dict], list[str]]:
    work = recipes_out.copy()
    work["RecipeCategory"] = work["RecipeCategory"].fillna("").astype(str).str.strip()
    work = work[(work["RecipeCategory"] != "") & (work["RecipeCategory"].str.lower() != "nan")]
    categories = sorted(work["RecipeCategory"].unique().tolist())

    payloads: dict[str, dict] = {
        "cook_fraction__wer": {},
        "total_time__wer": {},
        "cook_fraction__rating_per_hour": {},
        "total_time__rating_per_hour": {},
    }

    for cat in categories:
        sub = work[work["RecipeCategory"] == cat].copy()
        payloads["cook_fraction__wer"][cat] = build_plot_payload_from_recipe_frame(sub, angle_mode=ANGLE_COOK_FRACTION, metric="wer")
        payloads["total_time__wer"][cat] = build_plot_payload_from_recipe_frame(
            sub,
            angle_mode=ANGLE_TOTAL_TIME,
            metric="wer",
            total_time_labels=total_time_labels,
            total_time_edges=total_time_edges,
        )
        payloads["cook_fraction__rating_per_hour"][cat] = build_plot_payload_from_recipe_frame(sub, angle_mode=ANGLE_COOK_FRACTION, metric="rating_per_hour")
        payloads["total_time__rating_per_hour"][cat] = build_plot_payload_from_recipe_frame(
            sub,
            angle_mode=ANGLE_TOTAL_TIME,
            metric="rating_per_hour",
            total_time_labels=total_time_labels,
            total_time_edges=total_time_edges,
        )

    return payloads, categories


def build_figure(plot_data: dict) -> go.Figure:
    meta = plot_data["metadata"]
    sectors = plot_data["sectors"]
    if not sectors:
        raise ValueError("No sectors in plot payload — cannot build figure.")
    df = pd.DataFrame(sectors)

    traces = []
    for tone in TONE_GROUPS:
        r_vals = [s["segments"][tone] for s in sectors]
        hover = []
        for s in sectors:
            hover.append(
                f"<b>{meta['hover_axis_label']}: {s['label']}</b><br>"
                f"Recipes : {s['recipe_count']:,}<br>"
                f"Reviews : {s['review_count']:,}<br>"
                f"─────────────────────<br>"
                f"Mean rating       : {s['mean_rating']:.3f} ★<br>"
                f"Rating / hour     : {s['mean_rating_per_hour']:.3f}<br>"
                f"WER               : {s['mean_wer']:.3f}<br>"
                f"Mean time         : {s['mean_total_time_min']:.1f} min<br>"
                f"─────────────────────<br>" +
                "<br>".join(f"{TONE_DISPLAY[t]}: {s['tone_fractions'][t] * 100:.1f}%" for t in TONE_GROUPS)
            )
        traces.append(go.Barpolar(
            r=r_vals,
            theta=df["angle_deg"].tolist(),
            width=[meta["sector_width_deg"]] * len(df),
            name=TONE_DISPLAY[tone],
            legendgroup=tone,
            marker_color=COLORS[tone],
            marker_line=dict(color="black", width=1.0),
            opacity=0.90,
            hovertext=hover,
            hoverinfo="text",
        ))

    annotations = [
        dict(text=f"<b>{meta['annotation_top']}</b>", xref="paper", yref="paper", x=0.5, y=1.00,
             showarrow=False, font=dict(size=10, color="rgba(120,120,120,0.85)"), align="center"),
        dict(text=f"<b>{meta['annotation_bottom']}</b>", xref="paper", yref="paper", x=0.5, y=0.02,
             showarrow=False, font=dict(size=10, color="rgba(120,120,120,0.85)"), align="center"),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=None,
        annotations=annotations,
        polar=dict(
            bargap=0.05,
            angularaxis=dict(
                tickmode="array",
                tickvals=df["angle_deg"].tolist(),
                ticktext=df["label"].tolist(),
                direction="clockwise",
                rotation=90,
                tickfont=dict(size=8),
                showline=True,
                linewidth=1.4,
                linecolor="black",
                gridcolor="black",
                griddash="solid",
                gridwidth=1.2,
            ),
            radialaxis=dict(
                range=[0, meta["radial_max"]],
                tickvals=meta["radial_tick_vals"],
                ticktext=meta["radial_tick_text"],
                tickfont=dict(size=8, color="rgba(130,130,130,0.9)"),
                showline=True,
                linewidth=1.4,
                gridcolor="black",
                griddash="solid",
                gridwidth=1.2,
                linecolor="black",
                angle=45,
            ),
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5,
            font=dict(size=10), tracegroupgap=0, itemsizing="constant",
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=24, b=86, l=30, r=42),
        width=640,
        height=640,
        barmode="stack",
    )
    return fig




TOP_CATEGORIES_PER_WEDGE = 4
CATEGORY_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
    "#FF9DA6", "#9D755D", "#BAB0AC", "#2E91E5", "#E15F99", "#1CA71C",
    "#FB0D0D", "#DA16FF", "#222A2A", "#B68100", "#750D86", "#EB663B",
]


def build_total_time_category_population_payload(
    recipes_out: pd.DataFrame,
    *,
    top_n: int = TOP_CATEGORIES_PER_WEDGE,
    total_time_labels: list[str] | None = None,
    total_time_edges: list[float] | None = None,
) -> dict:
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
        radius = float(np.sqrt(recipe_count / total_recipes)) if recipe_count > 0 else 0.0

        cat_counts = group["RecipeCategory"].value_counts().head(top_n)
        segments = []
        for cat, count in cat_counts.items():
            count_i = int(count)
            frac = count_i / recipe_count if recipe_count else 0.0
            seg_r = radius * frac
            segments.append({
                "category": cat,
                "count": count_i,
                "fraction_in_wedge": float(frac),
                "r_segment": float(seg_r),
            })
            all_categories.add(cat)

        weights = pd.to_numeric(group.get("review_count_r", 0), errors="coerce").fillna(0.0).clip(lower=0.0)
        ratings = pd.to_numeric(group.get("mean_rating_r", np.nan), errors="coerce")
        valid = ratings.notna()
        if recipe_count and valid.any() and float(weights[valid].sum()) > 0:
            weighted_rating = float(np.average(ratings[valid], weights=weights[valid]))
        elif recipe_count and valid.any():
            weighted_rating = float(ratings[valid].mean())
        else:
            weighted_rating = float("nan")

        total_reviews = int(weights.sum()) if recipe_count else 0
        avg_time_min = float(pd.to_numeric(group.get("TotalTime", np.nan), errors="coerce").mean() / 60.0) if recipe_count else float("nan")

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
            # "outer_ring_note": "Thin outer ring encodes review-count-weighted average rating.",
            "all_categories": sorted(all_categories),
        },
        "sectors": sector_rows,
    }


def build_total_time_category_population_figure(plot_data: dict) -> go.Figure:
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
    radial_max_main = float(meta.get("radial_max_main", 1.0))
    ring_gap = max(0.03, radial_max_main * 0.06)
    ring_base = radial_max_main + ring_gap
    ring_thickness_min = max(0.018, radial_max_main * 0.05)
    ring_thickness_span = max(0.025, radial_max_main * 0.08)

    traces: list[go.Barpolar] = []

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

    max_range = ring_base + ring_thickness_min + ring_thickness_span + ring_gap * 0.55
    tick_vals = np.linspace(0, radial_max_main, 4)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=None,
        # annotations=[
        #     dict(
        #         text="<b>Thin outer ring = weighted average rating</b>",
        #         xref="paper", yref="paper", x=0.5, y=1.08,
        #         showarrow=False, align="center",
        #         font=dict(size=11, color="rgba(100,100,100,0.9)")
        #     ),
        # ],
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


def build_and_export(payloads: dict[str, dict], out_dir: Path) -> None:
    print("[5/5] Building Plotly windrose figures and exporting …")
    html_targets = {
        "cook_fraction__rating_per_hour": "windrose_rating_per_hour.html",
        "cook_fraction__wer": "windrose_wer.html",
        "total_time__rating_per_hour": "windrose_total_time_rating_per_hour.html",
        "total_time__wer": "windrose_total_time_wer.html",
    }
    for key, filename in html_targets.items():
        fig = build_figure(payloads[key])
        html_path = out_dir / filename
        fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            full_html=True,
            config={"displayModeBar": True, "toImageButtonOptions": {"format": "svg", "filename": filename[:-5]}},
        )
        print(f"    HTML → {html_path}")


def _payload_json_filename(angle_mode: str, metric: str) -> str:
    if angle_mode == ANGLE_COOK_FRACTION:
        return "windrose_plot_wer.json" if metric == "wer" else "windrose_plot_rating_per_hour.json"
    return "windrose_plot_total_time_wer.json" if metric == "wer" else "windrose_plot_total_time_rating_per_hour.json"


def _category_json_filename(angle_mode: str, metric: str) -> str:
    if angle_mode == ANGLE_COOK_FRACTION:
        return "windrose_plot_wer_by_category.json" if metric == "wer" else "windrose_plot_rating_per_hour_by_category.json"
    return "windrose_plot_total_time_wer_by_category.json" if metric == "wer" else "windrose_plot_total_time_rating_per_hour_by_category.json"


def replot_exported_windrose(
    *,
    fig_key: str = "windrose_wer",
    category: str | None = None,
    output_dir: str | Path | None = None,
):
    output_dir = Path(output_dir or DEFAULT_WINDROSE_OUTPUT_DIR)
    if fig_key == "windrose_total_time_population":
        path = output_dir / "windrose_plot_total_time_population.json"
        if not path.exists():
            raise FileNotFoundError(f"Windrose payload file not found: {path}")
        if category:
            raise ValueError("Category filtering is not supported for windrose_total_time_population.")
        with open(path, "r", encoding="utf-8") as f:
            plot_data = json.load(f)
        return build_total_time_category_population_figure(plot_data)

    mapping = {
        "windrose_wer": (ANGLE_COOK_FRACTION, "wer"),
        "windrose_rating_per_hour": (ANGLE_COOK_FRACTION, "rating_per_hour"),
        "windrose_total_time_wer": (ANGLE_TOTAL_TIME, "wer"),
        "windrose_total_time_rating_per_hour": (ANGLE_TOTAL_TIME, "rating_per_hour"),
    }
    if fig_key not in mapping:
        raise ValueError(f"Unsupported fig_key: {fig_key}")
    angle_mode, metric = mapping[fig_key]

    if category:
        path = output_dir / _category_json_filename(angle_mode, metric)
        if not path.exists():
            raise FileNotFoundError(f"Category payload file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payloads = json.load(f)
        plot_data = payloads.get(category)
        if not plot_data:
            raise ValueError(f"No windrose data for category: {category}")
    else:
        path = output_dir / _payload_json_filename(angle_mode, metric)
        if not path.exists():
            raise FileNotFoundError(f"Windrose payload file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            plot_data = json.load(f)

    return build_figure(plot_data)


def load_windrose_category_options(output_dir: str | Path | None = None) -> list[str]:
    output_dir = Path(output_dir or DEFAULT_WINDROSE_OUTPUT_DIR)
    options_path = output_dir / "windrose_category_options.json"
    if options_path.exists():
        with open(options_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    fallback = output_dir / "windrose_plot_wer_by_category.json"
    if fallback.exists():
        with open(fallback, "r", encoding="utf-8") as f:
            data = json.load(f)
        return sorted(data.keys())
    return []


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recipes, reviews = load_data()
    recipes = clean_recipes(recipes)
    reviews = score_sentiment(reviews)
    payloads, recipes_out = aggregate(recipes, reviews)

    total_time_edges = recipes.attrs.get("total_time_edges", [])
    total_time_labels = recipes.attrs.get("total_time_labels", [])
    category_payloads, categories = build_category_payloads(
        recipes_out,
        total_time_labels=total_time_labels,
        total_time_edges=total_time_edges,
    )

    exports = {
        "cook_fraction__rating_per_hour": OUT_DIR / "windrose_plot_rating_per_hour.json",
        "cook_fraction__wer": OUT_DIR / "windrose_plot_wer.json",
        "total_time__rating_per_hour": OUT_DIR / "windrose_plot_total_time_rating_per_hour.json",
        "total_time__wer": OUT_DIR / "windrose_plot_total_time_wer.json",
    }
    for key, path in exports.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payloads[key], f, indent=2, ensure_ascii=False, default=float)
        print(f"    JSON → {path}")

    category_exports = {
        "cook_fraction__rating_per_hour": OUT_DIR / "windrose_plot_rating_per_hour_by_category.json",
        "cook_fraction__wer": OUT_DIR / "windrose_plot_wer_by_category.json",
        "total_time__rating_per_hour": OUT_DIR / "windrose_plot_total_time_rating_per_hour_by_category.json",
        "total_time__wer": OUT_DIR / "windrose_plot_total_time_wer_by_category.json",
    }
    for key, path in category_exports.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(category_payloads[key], f, indent=2, ensure_ascii=False, default=float)
        print(f"    JSON → {path}")

    recipes_json_path = OUT_DIR / "windrose_data_recipes.json"
    recipes_out.to_json(recipes_json_path, orient="records", indent=2, force_ascii=False, default_handler=float)
    print(f"    JSON → {recipes_json_path}")

    category_options_path = OUT_DIR / "windrose_category_options.json"
    with open(category_options_path, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)
    print(f"    JSON → {category_options_path}")

    metadata_path = OUT_DIR / "windrose_angle_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"total_time_edges": total_time_edges, "total_time_labels": total_time_labels}, f, indent=2, ensure_ascii=False)
    print(f"    JSON → {metadata_path}")

    population_payload = build_total_time_category_population_payload(
        recipes_out,
        top_n=TOP_CATEGORIES_PER_WEDGE,
        total_time_labels=total_time_labels,
        total_time_edges=total_time_edges,
    )
    population_payload_path = OUT_DIR / "windrose_plot_total_time_population.json"
    with open(population_payload_path, "w", encoding="utf-8") as f:
        json.dump(population_payload, f, indent=2, ensure_ascii=False, default=float)
    print(f"    JSON → {population_payload_path}")

    population_fig = build_total_time_category_population_figure(population_payload)
    population_html_path = OUT_DIR / "windrose_total_time_population.html"
    population_fig.write_html(
        str(population_html_path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "toImageButtonOptions": {"format": "svg", "filename": "windrose_total_time_population"}},
    )
    print(f"    HTML → {population_html_path}")

    build_and_export(payloads, OUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
