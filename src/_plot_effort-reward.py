"""
windrose_pipeline.py
────────────────────
Effort-Reward Windrose: Prep/Cook Fraction × Rating × Sentiment Tone

Pipeline:
  1. Load recipes + reviews from SQLite
  2. Clean & filter time columns
  3. Compute prep fraction → assign angular sector (16 bins × 22.5°)
  4. Run VADER sentiment on review text → classify into 5 tone groups
  5. Aggregate to (sector × sentiment_zone) → export JSON
  6. Build two interactive Plotly windrose HTML files:
       windrose_rating_per_hour.html   — radial = rating / hour
       windrose_wer.html               — radial = weighted engagement rate

Outputs (to /project_folder/plots/cooking_time_outputs/):
  windrose_plot_rating_per_hour.json  — plot-ready data for rating/hour chart
  windrose_plot_wer.json              — plot-ready data for WER chart
  windrose_data_recipes.json          — one record per recipe (client-side filtering)
  windrose_rating_per_hour.html       — self-contained interactive Plotly figure
  windrose_wer.html                   — self-contained interactive Plotly figure

Rating source: individual reviewer Rating from the reviews table, averaged per recipe
  (mean_rating_r), replacing AggregatedRating from the recipes table.
  review_count is also derived from the reviews table (actual row count per RecipeId).

Time-data policy: recipes are dropped entirely if any of PrepTime, CookTime, or TotalTime
  is NULL, zero, or negative, OR if PrepTime + CookTime deviates from TotalTime by more
  than TIME_CONSISTENCY_TOL seconds. No imputation is performed.

Three independent windrose dimensions:
  • Angle    → cook fraction bin (0% cook … 100% cook, 16 × 22.5° sectors)
  • Radius   → mean rating/hour  OR  mean weighted engagement rate
  • Bar split → fraction of reviews in each of 5 sentiment tone groups

Configurable constants are grouped at the top of the file.
"""

# ── standard library ──────────────────────────────────────────────────────────
import json
# import os
import re
import sqlite3
import warnings
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — change these without touching the pipeline logic
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH   = BASE_DIR  / "data" / "tables" / "food_recipe.db"
OUT_DIR   = BASE_DIR / "plots" / "cooking_time_outputs"
DEFAULT_WINDROSE_OUTPUT_DIR = OUT_DIR

# Sector count — must divide 360 evenly. 16 → 22.5° per sector.
N_SECTORS = 16

# Minimum number of recipes a sector must contain to be rendered.
MIN_RECIPES_PER_SECTOR = 10

# Minimum number of reviews a sector must contain for sentiment bars.
MIN_REVIEWS_PER_SECTOR = 30

# Rating column — kept for reference only; actual ratings come from reviews table.
RATING_COL = "AggregatedRating"

# Time sanity ceiling: recipes with TotalTime > this (seconds) are dropped.
# 48 h = 172 800 s  — keeps extreme outliers out.
MAX_TOTAL_TIME_SEC = 172_800

# Maximum tolerated absolute difference (seconds) between TotalTime and
# PrepTime + CookTime before a recipe is considered inconsistent and dropped.
# 60 s covers rounding artefacts in the original dataset.
TIME_CONSISTENCY_TOL = 60

# VADER compound score boundaries for 5 tone groups (compound in [-1, +1]).
#   strongly_positive  :  c >= 0.50
#   mildly_positive    :  0.05 <= c < 0.50  (AND no modification keyword)
#   modification       :  keyword present AND c > -0.20
#                         OR  -0.05 <= c < 0.05  (true neutral)
#   mildly_negative    : -0.50 < c <= -0.05  (AND no keyword)
#   strongly_negative  :  c <= -0.50
TONE_THRESHOLDS = {
    "strongly_positive": 0.50,
    "mildly_positive":   0.05,
    "neutral_mod":      -0.05,
    "mildly_negative":  -0.50,
}

# Modification keywords — nudge a review into "modification" tone when matched
# AND compound > -0.20 (not strongly negative).
MODIFICATION_KEYWORDS = re.compile(
    r"\b(added?|add|substitut\w*|swap\w*|replac\w*|next time|doubled?|"
    r"halved?|used? instead|omit\w*|skip\w*|modify|modif\w*|tweak\w*|"
    r"adjust\w*|change\w*|forgot|less \w+|more \w+|cut back|extra)\b",
    re.IGNORECASE,
)

# 5 tone groups in stacking order (bottom → top of each wedge)
TONE_GROUPS = [
    "strongly_positive",
    "mildly_positive",
    "modification",
    "mildly_negative",
    "strongly_negative",
]

COLORS = {
    "strongly_positive": "#1D9E75",   # deep teal
    "mildly_positive":   "#8ECFC9",   # light teal
    "modification":      "#EF9F27",   # amber
    "mildly_negative":   "#F0997B",   # light coral
    "strongly_negative": "#D85A30",   # deep coral
}

TONE_DISPLAY = {
    "strongly_positive": "Strongly positive",
    "mildly_positive":   "Mildly positive",
    "modification":      "Modification / neutral",
    "mildly_negative":   "Mildly negative",
    "strongly_negative": "Strongly negative",
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sector_label(sector_idx: int, n: int = N_SECTORS) -> str:
    """Percentage-range label for a sector — no prose, just numbers."""
    step = 100 / n
    lo = round(sector_idx * step)
    hi = round((sector_idx + 1) * step)
    return f"{lo}–{hi}%"


def classify_tone(compound: float, text: str) -> str:
    """
    Map VADER compound score + modification keyword presence to one of 5 tone groups.

    Decision order
    ──────────────
    1. Modification keyword present AND compound > -0.20
       → "modification"  (e.g. "I added garlic and loved it" scores high but is mod-intent)
    2. compound >= 0.50  → "strongly_positive"
    3. compound >= 0.05  → "mildly_positive"
    4. compound <= -0.50 → "strongly_negative"
    5. compound <= -0.05 → "mildly_negative"
    6. -0.05 < compound < 0.05  → "modification"  (genuinely neutral)
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[1/5] Loading data from SQLite …")
    con = sqlite3.connect(DB_PATH)

    # ReviewCount is intentionally excluded — actual review counts and ratings
    # are derived from the reviews table to avoid relying on pre-aggregated values.
    recipes = pd.read_sql_query(
        """
        SELECT RecipeId,
               PrepTime, CookTime, TotalTime,
               RecipeCategory
        FROM   recipes
        """,
        con,
    )

    reviews = pd.read_sql_query(
        """
        SELECT RecipeId, Rating, Review
        FROM   reviews
        WHERE  Review IS NOT NULL AND TRIM(Review) != ''
          AND  Rating IS NOT NULL
        """,
        con,
    )

    con.close()
    print(f"    recipes loaded : {len(recipes):,}")
    print(f"    reviews loaded : {len(reviews):,}")
    return recipes, reviews


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN & FILTER TIME COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def clean_recipes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strict drop-only cleaning — no imputation is performed at any stage.

    A recipe is dropped if ANY of the following are true:
      • PrepTime, CookTime, or TotalTime is NULL
      • PrepTime, CookTime, or TotalTime is <= 0
      • TotalTime > MAX_TOTAL_TIME_SEC
      • |PrepTime + CookTime − TotalTime| > TIME_CONSISTENCY_TOL

    AggregatedRating is NOT used here — ratings come from the reviews table.
    """
    print("[2/5] Cleaning and filtering recipes …")
    n0 = len(df)
    drop_log: dict[str, int] = {}

    # Cast to numeric (guard against stray strings)
    for col in ["PrepTime", "CookTime", "TotalTime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop NULL in any time column ──────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["PrepTime", "CookTime", "TotalTime"])
    drop_log["null time"] = before - len(df)

    # ── Drop non-positive values in any time column ───────────────────────────
    before = len(df)
    df = df[(df["PrepTime"] > 0) & (df["CookTime"] > 0) & (df["TotalTime"] > 0)]
    drop_log["non-positive time"] = before - len(df)

    # ── Drop extreme TotalTime outliers ───────────────────────────────────────
    before = len(df)
    df = df[df["TotalTime"] <= MAX_TOTAL_TIME_SEC]
    drop_log[f"TotalTime > {MAX_TOTAL_TIME_SEC}s"] = before - len(df)

    # ── Drop time-inconsistent rows ───────────────────────────────────────────
    before = len(df)
    deviation = (df["PrepTime"] + df["CookTime"] - df["TotalTime"]).abs()
    df = df[deviation <= TIME_CONSISTENCY_TOL]
    drop_log[f"time inconsistency > {TIME_CONSISTENCY_TOL}s"] = before - len(df)

    # ── Cook fraction & sector assignment ─────────────────────────────────────
    denom = df["PrepTime"] + df["CookTime"]
    df = df.copy()
    df["cook_fraction"] = (df["CookTime"] / denom).clip(0.0, 1.0)
    df["sector"] = (
        np.floor(df["cook_fraction"] * N_SECTORS)
        .clip(0, N_SECTORS - 1)
        .astype(int)
    )

    n1 = len(df)
    print(f"    kept {n1:,} / {n0:,} recipes  ({n0 - n1:,} dropped)")
    for reason, count in drop_log.items():
        if count:
            print(f"      − {count:>7,}  {reason}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — VADER SENTIMENT ON REVIEWS
# ══════════════════════════════════════════════════════════════════════════════

def score_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Scoring sentiment with VADER …")
    print(f"    processing {len(reviews):,} reviews — this may take a minute …")

    # reviews = reviews.head(100_000)
    analyzer = SentimentIntensityAnalyzer()
    chunk_size = 100_000
    compounds = []
    for start in range(0, len(reviews), chunk_size):
        chunk = reviews["Review"].iloc[start : start + chunk_size]
        compounds.extend(
            analyzer.polarity_scores(str(text))["compound"]
            for text in chunk
        )
        print(f"    … {min(start + chunk_size, len(reviews)):,} / "
              f"{len(reviews):,} done", end="\r")

    print()
    reviews = reviews.copy()
    reviews["compound"] = compounds
    reviews["tone"] = [
        classify_tone(c, t)
        for c, t in zip(reviews["compound"], reviews["Review"])
    ]

    tone_counts = reviews["tone"].value_counts()
    print(f"    tone distribution (5 groups): {tone_counts.to_dict()}")
    return reviews


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — AGGREGATE TO (sector × tone)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(
    recipes: pd.DataFrame,
    reviews: pd.DataFrame,
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Returns
    -------
    agg_rph : dict
        Plot-ready aggregated data for the rating-per-hour windrose.
    agg_wer : dict
        Plot-ready aggregated data for the weighted-engagement-rate windrose.
    recipes_out : pd.DataFrame
        Per-recipe frame for windrose_data_recipes.json.
    """
    print("[4/5] Aggregating to sector × tone …")

    # ── Working frame: recipes with time/sector columns only ──────────────────
    recipe_sectors = recipes[
        ["RecipeId", "sector", "cook_fraction",
         "PrepTime", "CookTime", "TotalTime", "RecipeCategory"]
    ].copy()

    # ── Compute per-recipe mean rating and review count from reviews table ─────
    # This replaces AggregatedRating and ReviewCount from the recipes table.
    # Only reviews that survived the sentiment filter (have Rating) are used.
    reviews_valid = reviews[reviews["Rating"].between(1, 5)].copy()
    per_recipe_stats = (
        reviews_valid
        .groupby("RecipeId")
        .agg(
            mean_rating_r  = ("Rating", "mean"),    # mean individual reviewer rating
            review_count_r = ("Rating", "count"),   # actual review count in DB
        )
        .reset_index()
    )

    # Merge per-recipe stats onto recipe_sectors (inner: only recipes with reviews)
    recipe_sectors = recipe_sectors.merge(per_recipe_stats, on="RecipeId", how="inner")
    print(f"    recipes with at least 1 valid review: {len(recipe_sectors):,}")

    # ── Per-recipe efficiency metrics ──────────────────────────────────────────
    # Both metrics use mean_rating_r (from reviews table) and TotalTime.
    #
    # rating_per_hour  = mean_rating_r / (TotalTime / 3600)
    #
    # weighted_engagement_rate (WER):
    #   WER = mean_rating_r × log1p(review_count_r) / (TotalTime / 3600)
    #   log1p compresses the dynamic range of review_count_r so that a recipe
    #   with 10000 reviews does not dominate over one with 10 reviews.
    hours = (recipe_sectors["TotalTime"] / 3600).replace(0, np.nan)
    recipe_sectors["rating_per_hour"] = recipe_sectors["mean_rating_r"] / hours
    recipe_sectors["wer"] = (
        recipe_sectors["mean_rating_r"]
        * np.log1p(recipe_sectors["review_count_r"])
        / hours
    )

    # ── Join sector label onto reviews for tone counting ──────────────────────
    rev_joined = reviews_valid.merge(
        recipe_sectors[["RecipeId", "sector"]],
        on="RecipeId",
        how="inner",
    )

    # ── Sector-level aggregation ───────────────────────────────────────────────
    sector_stats = (
        recipe_sectors
        .groupby("sector")
        .agg(
            mean_rating          = ("mean_rating_r",   "mean"),
            median_rating        = ("mean_rating_r",   "median"),
            recipe_count         = ("mean_rating_r",   "count"),
            mean_total_time_min  = ("TotalTime",        lambda x: x.mean() / 60),
            mean_review_count    = ("review_count_r",  "mean"),
            mean_rating_per_hour = ("rating_per_hour", "mean"),
            mean_wer             = ("wer",             "mean"),
        )
        .reset_index()
    )

    # ── Sector-level tone fractions (5 groups) ────────────────────────────────
    tone_counts = (
        rev_joined
        .groupby(["sector", "tone"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TONE_GROUPS, fill_value=0)
        .astype(float)
    )
    tone_counts["review_count"] = tone_counts.sum(axis=1)
    tone_fractions = tone_counts[TONE_GROUPS].div(tone_counts["review_count"], axis=0)
    tone_fractions["review_count"] = tone_counts["review_count"]
    tone_fractions = tone_fractions.reset_index()

    # ── Merge and filter ───────────────────────────────────────────────────────
    agg = sector_stats.merge(tone_fractions, on="sector", how="left")
    agg_filtered = agg[
        (agg["recipe_count"]  >= MIN_RECIPES_PER_SECTOR) &
        (agg["review_count"]  >= MIN_REVIEWS_PER_SECTOR)
    ].copy()

    agg_filtered["label"]        = agg_filtered["sector"].apply(sector_label)
    agg_filtered["angle_deg"]    = agg_filtered["sector"] * (360 / N_SECTORS)
    agg_filtered["sector_width"] = 360 / N_SECTORS

    n_kept = len(agg_filtered)
    print(f"    {n_kept} / {N_SECTORS} sectors retained after minimum-count filter")

    # ── Build plot-ready JSON payloads ─────────────────────────────────────────
    # Each payload contains everything the web app needs to re-draw the chart:
    # sector geometry, tone segment r-values, colors, radial grid, and metadata.

    def make_plot_payload(metric_col: str, metric_label: str) -> dict:
        r_total   = agg_filtered[metric_col].values
        r_max     = float(r_total.max()) * 1.15
        tick_vals = [round(r_max * k / 4, 4) for k in range(5)]

        sectors_out = []
        for _, row in agg_filtered.iterrows():
            r_t = float(row[metric_col])
            seg = {}
            for tone in TONE_GROUPS:
                seg[tone] = round(float(row[tone]) * r_t, 6)
            sectors_out.append({
                "sector_idx"         : int(row["sector"]),
                "label"              : row["label"],
                "angle_deg"          : float(row["angle_deg"]),
                "sector_width_deg"   : float(row["sector_width"]),
                "r_total"            : round(r_t, 6),
                "segments"           : seg,               # r-value per tone segment
                "tone_fractions"     : {t: round(float(row[t]), 6) for t in TONE_GROUPS},
                "recipe_count"       : int(row["recipe_count"]),
                "review_count"       : int(row["review_count"]),
                "mean_rating"        : round(float(row["mean_rating"]), 4),
                "mean_total_time_min": round(float(row["mean_total_time_min"]), 2),
                "mean_review_count"  : round(float(row["mean_review_count"]), 2),
                "mean_rating_per_hour": round(float(row["mean_rating_per_hour"]), 4),
                "mean_wer"           : round(float(row["mean_wer"]), 4),
            })

        return {
            "metadata": {
                "metric_col"               : metric_col,
                "metric_label"             : metric_label,
                "n_sectors_total"          : N_SECTORS,
                "n_sectors_plotted"        : n_kept,
                "sector_width_deg"         : 360 / N_SECTORS,
                "angular_direction"        : "clockwise",
                "rotation_deg"             : 90,
                "tone_groups"              : TONE_GROUPS,
                "tone_colors"              : COLORS,
                "tone_display_names"       : TONE_DISPLAY,
                "radial_max"               : round(r_max, 4),
                "radial_tick_vals"         : tick_vals,
                "radial_tick_text"         : [str(v) for v in tick_vals],
                "min_recipes_per_sector"   : MIN_RECIPES_PER_SECTOR,
                "min_reviews_per_sector"   : MIN_REVIEWS_PER_SECTOR,
                "time_consistency_tol_sec" : TIME_CONSISTENCY_TOL,
                "vader_thresholds"         : TONE_THRESHOLDS,
                "rating_source"            : "individual reviewer Rating from reviews table",
                "n_recipes_after_cleaning" : int(len(recipe_sectors)),
                "n_reviews_scored"         : int(len(rev_joined)),
            },
            "sectors": sectors_out,
        }

    agg_rph = make_plot_payload("mean_rating_per_hour", "Rating / hour")
    agg_wer = make_plot_payload("mean_wer",             "Weighted engagement rate")

    # ── Per-recipe output frame ────────────────────────────────────────────────
    # Per-recipe tone fractions (from individual reviews)
    recipe_tone = (
        reviews_valid
        .groupby(["RecipeId", "tone"])
        .size()
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
    recipes_out["sector_label"] = recipes_out["sector"].apply(sector_label)

    for col in ["cook_fraction", "mean_rating_r", "rating_per_hour", "wer"] + TONE_GROUPS:
        if col in recipes_out.columns:
            recipes_out[col] = recipes_out[col].round(4)

    print(f"    per-recipe rows: {len(recipes_out):,}")
    return agg_rph, agg_wer, recipes_out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — BUILD PLOTLY WINDROSE + EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(plot_data: dict) -> go.Figure:
    """
    Build a single windrose figure from a plot-ready payload dict.
    No dropdown, no explanation annotations — only the chart elements.
    Canvas size is derived from the data so whitespace is minimal.
    """
    meta     = plot_data["metadata"]
    sectors  = plot_data["sectors"]
    df       = pd.DataFrame(sectors)

    if df.empty:
        raise ValueError("No sectors in plot payload — cannot build figure.")

    r_max        = meta["radial_max"]
    tick_vals    = meta["radial_tick_vals"]
    tick_text    = meta["radial_tick_text"]
    sector_width = meta["sector_width_deg"]
    metric_label = meta["metric_label"]

    # ── One Barpolar trace per tone group ─────────────────────────────────────
    traces = []
    for tone in TONE_GROUPS:
        r_vals = [s["segments"][tone] for s in sectors]
        hover  = [
            (
                f"<b>{s['label']} cook</b><br>"
                f"Recipes : {s['recipe_count']:,}<br>"
                f"Reviews : {s['review_count']:,}<br>"
                f"─────────────────────<br>"
                f"Mean rating       : {s['mean_rating']:.3f} ★<br>"
                f"Rating / hour     : {s['mean_rating_per_hour']:.3f}<br>"
                f"WER               : {s['mean_wer']:.3f}<br>"
                f"Mean time         : {s['mean_total_time_min']:.1f} min<br>"
                f"─────────────────────<br>"
                + "<br>".join(
                    f"{TONE_DISPLAY[t]}: {s['tone_fractions'][t]*100:.1f}%"
                    for t in TONE_GROUPS
                )
            )
            for s in sectors
        ]
        traces.append(
            go.Barpolar(
                r            = r_vals,
                theta        = df["angle_deg"].tolist(),
                width        = [sector_width] * len(df),
                name         = TONE_DISPLAY[tone],
                legendgroup  = tone,
                marker_color = COLORS[tone],
                marker_line  = dict(color="rgba(255,255,255,0.18)", width=0.4),
                opacity      = 0.90,
                hovertext    = hover,
                hoverinfo    = "text",
            )
        )

    fig = go.Figure(data=traces)

    # ── Compass annotations (minimal — no explanation text) ───────────────────
    annotations = [
        dict(
                text="<b>← more prep</b>",
                xref="paper", yref="paper",
                x=0.5, y=1.00,
                showarrow=False,
                font=dict(size=10, color="rgba(120,120,120,0.85)"),
                align="center",
            ),
        dict(
                text="<b>← more cook</b>",
                xref="paper", yref="paper",
                x=0.5, y=-0.0,
                showarrow=False,
                font=dict(size=10, color="rgba(120,120,120,0.85)"),
                align="center",
            ),
    ]

    fig.update_layout(
        title = None,
        annotations = annotations,
        polar = dict(
            bargap = 0.05,
            angularaxis = dict(
                tickmode  = "array",
                tickvals  = df["angle_deg"].tolist(),
                ticktext  = df["label"].tolist(),
                direction = "clockwise",
                rotation  = 90,           # sector 0 (0–6% cook) at top
                tickfont  = dict(size=8),
                linecolor = "rgba(150,150,150,0.25)",
                gridcolor = "rgba(150,150,150,0.10)",
            ),
            radialaxis = dict(
                range     = [0, r_max],
                tickvals  = tick_vals,
                ticktext  = tick_text,
                tickfont  = dict(size=8, color="rgba(130,130,130,0.9)"),
                gridcolor = "rgba(150,150,150,0.20)",
                linecolor = "rgba(150,150,150,0.20)",
                angle     = 45,
            ),
        ),
        legend = dict(
            orientation   = "h",
            yanchor       = "top",
            y             = -0.06,
            xanchor       = "center",
            x             = 0.5,
            font          = dict(size=10),
            tracegroupgap = 0,
            itemsizing    = "constant",
        ),
        showlegend    = True,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        # Tight margins — legend sits just below the polar area.
        # Right margin is wider only to accommodate radial tick labels at 45°.
        margin        = dict(t=28, b=110, l=40, r=60),
        width         = 680,
        height        = 720,
        barmode       = "stack",
    )

    return fig



def _weighted_tone_fraction(frame: pd.DataFrame, tone: str) -> float:
    weights = pd.to_numeric(frame.get("review_count_scored", 0), errors="coerce").fillna(0.0)
    values  = pd.to_numeric(frame.get(tone, 0), errors="coerce").fillna(0.0)
    total_w = float(weights.sum())
    if total_w <= 0:
        return float(values.mean()) if len(values) else 0.0
    return float((values * weights).sum() / total_w)


def build_plot_payload_from_recipe_frame(
    recipes_frame: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    *,
    category_name: str | None = None,
) -> dict:
    """Create a plot-ready payload from a per-recipe frame."""
    if recipes_frame.empty:
        return {
            "metadata": {
                "metric_col": metric_col,
                "metric_label": metric_label,
                "category_name": category_name,
                "n_sectors_total": N_SECTORS,
                "n_sectors_plotted": 0,
                "sector_width_deg": 360 / N_SECTORS,
                "angular_direction": "clockwise",
                "rotation_deg": 90,
                "tone_groups": TONE_GROUPS,
                "tone_colors": COLORS,
                "tone_display_names": TONE_DISPLAY,
                "radial_max": 1.0,
                "radial_tick_vals": [0, 0.25, 0.5, 0.75, 1.0],
                "radial_tick_text": ["0", "0.25", "0.5", "0.75", "1.0"],
                "min_recipes_per_sector": MIN_RECIPES_PER_SECTOR,
                "min_reviews_per_sector": MIN_REVIEWS_PER_SECTOR,
                "time_consistency_tol_sec": TIME_CONSISTENCY_TOL,
                "rating_source": "individual reviewer Rating from reviews table",
                "n_recipes_after_cleaning": 0,
                "n_reviews_scored": 0,
            },
            "sectors": [],
        }

    sectors_out = []
    grouped = recipes_frame.groupby("sector", sort=True)
    for sector_idx, g in grouped:
        recipe_count = int(len(g))
        review_count = int(pd.to_numeric(g.get("review_count_scored", 0), errors="coerce").fillna(0).sum())
        if recipe_count < MIN_RECIPES_PER_SECTOR or review_count < MIN_REVIEWS_PER_SECTOR:
            continue

        metric_value = float(pd.to_numeric(g[metric_col], errors="coerce").dropna().mean())
        if not np.isfinite(metric_value):
            continue

        tone_fractions = {tone: _weighted_tone_fraction(g, tone) for tone in TONE_GROUPS}
        tone_sum = sum(tone_fractions.values())
        if tone_sum > 0:
            tone_fractions = {k: v / tone_sum for k, v in tone_fractions.items()}

        sectors_out.append({
            "sector_idx": int(sector_idx),
            "label": sector_label(int(sector_idx)),
            "angle_deg": float(int(sector_idx) * (360 / N_SECTORS)),
            "sector_width_deg": float(360 / N_SECTORS),
            "r_total": round(metric_value, 6),
            "segments": {tone: round(tone_fractions[tone] * metric_value, 6) for tone in TONE_GROUPS},
            "tone_fractions": {tone: round(float(tone_fractions[tone]), 6) for tone in TONE_GROUPS},
            "recipe_count": recipe_count,
            "review_count": review_count,
            "mean_rating": round(float(pd.to_numeric(g["mean_rating_r"], errors="coerce").dropna().mean()), 4),
            "mean_total_time_min": round(float(pd.to_numeric(g["TotalTime"], errors="coerce").dropna().mean() / 60), 2),
            "mean_review_count": round(float(pd.to_numeric(g["review_count_r"], errors="coerce").dropna().mean()), 2),
            "mean_rating_per_hour": round(float(pd.to_numeric(g["rating_per_hour"], errors="coerce").dropna().mean()), 4),
            "mean_wer": round(float(pd.to_numeric(g["wer"], errors="coerce").dropna().mean()), 4),
        })

    sectors_out = sorted(sectors_out, key=lambda x: x["sector_idx"])
    if sectors_out:
        r_max = max(s["r_total"] for s in sectors_out) * 1.15
    else:
        r_max = 1.0
    tick_vals = [round(r_max * k / 4, 4) for k in range(5)]

    return {
        "metadata": {
            "metric_col": metric_col,
            "metric_label": metric_label,
            "category_name": category_name,
            "n_sectors_total": N_SECTORS,
            "n_sectors_plotted": len(sectors_out),
            "sector_width_deg": 360 / N_SECTORS,
            "angular_direction": "clockwise",
            "rotation_deg": 90,
            "tone_groups": TONE_GROUPS,
            "tone_colors": COLORS,
            "tone_display_names": TONE_DISPLAY,
            "radial_max": round(float(r_max), 4),
            "radial_tick_vals": tick_vals,
            "radial_tick_text": [str(v) for v in tick_vals],
            "min_recipes_per_sector": MIN_RECIPES_PER_SECTOR,
            "min_reviews_per_sector": MIN_REVIEWS_PER_SECTOR,
            "time_consistency_tol_sec": TIME_CONSISTENCY_TOL,
            "rating_source": "individual reviewer Rating from reviews table",
            "n_recipes_after_cleaning": int(len(recipes_frame)),
            "n_reviews_scored": int(pd.to_numeric(recipes_frame.get("review_count_scored", 0), errors="coerce").fillna(0).sum()),
        },
        "sectors": sectors_out,
    }


def build_category_payloads(recipes_out: pd.DataFrame) -> tuple[dict, dict, list[str]]:
    """Build plot-ready payloads for every RecipeCategory."""
    cat_series = (
        recipes_out["RecipeCategory"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    valid_categories = sorted(c for c in cat_series.unique().tolist() if c and c.lower() != "nan")

    wer_payloads = {}
    rph_payloads = {}
    for category in valid_categories:
        subset = recipes_out.loc[cat_series == category].copy()
        wer_payloads[category] = build_plot_payload_from_recipe_frame(
            subset,
            "wer",
            "Weighted engagement rate",
            category_name=category,
        )
        rph_payloads[category] = build_plot_payload_from_recipe_frame(
            subset,
            "rating_per_hour",
            "Rating / hour",
            category_name=category,
        )

    return rph_payloads, wer_payloads, valid_categories


def replot_exported_windrose(
    fig_key: str = "windrose_wer",
    *,
    category: str | None = None,
    output_dir: str | Path = DEFAULT_WINDROSE_OUTPUT_DIR,
) -> go.Figure:
    """Rebuild a windrose figure from exported JSON payload(s)."""
    output_dir = Path(output_dir)
    fig_map = {
        "windrose_wer": output_dir / "windrose_plot_wer.json",
        "windrose_rating_per_hour": output_dir / "windrose_plot_rating_per_hour.json",
    }
    by_category_map = {
        "windrose_wer": output_dir / "windrose_plot_wer_by_category.json",
        "windrose_rating_per_hour": output_dir / "windrose_plot_rating_per_hour_by_category.json",
    }
    if fig_key not in fig_map:
        raise ValueError(f"Unknown fig_key: {fig_key}")

    payload_path = by_category_map[fig_key] if category else fig_map[fig_key]
    if not payload_path.exists():
        raise FileNotFoundError(f"Windrose export not found: {payload_path}")

    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if category:
        if category not in payload:
            raise ValueError(f"Category not found in windrose export: {category}")
        payload = payload[category]

    return build_figure(payload)


def load_windrose_category_options(output_dir: str | Path = DEFAULT_WINDROSE_OUTPUT_DIR) -> list[str]:
    output_dir = Path(output_dir)
    index_path = output_dir / "windrose_category_options.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("categories", [])

    by_cat_path = output_dir / "windrose_plot_wer_by_category.json"
    if by_cat_path.exists():
        with open(by_cat_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return sorted(payload.keys())

    return []

def build_and_export(
    agg_rph : dict,
    agg_wer : dict,
    out_dir : Path,
) -> None:
    print("[5/5] Building Plotly windrose figures and exporting …")

    for payload, stem in [
        (agg_rph, "windrose_rating_per_hour"),
        (agg_wer, "windrose_wer"),
    ]:
        fig      = build_figure(payload)
        html_path = out_dir / f"{stem}.html"
        fig.write_html(
            str(html_path),
            include_plotlyjs = "cdn",
            full_html        = True,
            config           = {
                "displayModeBar"      : True,
                "toImageButtonOptions": {"format": "svg", "filename": stem},
            },
        )
        print(f"    HTML → {html_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recipes, reviews            = load_data()
    recipes                     = clean_recipes(recipes)
    reviews                     = score_sentiment(reviews)
    agg_rph, agg_wer, recipes_out = aggregate(recipes, reviews)
    category_rph, category_wer, windrose_categories = build_category_payloads(recipes_out)

    # ── Export 1: plot-ready JSON for rating-per-hour chart ───────────────────
    rph_json_path = OUT_DIR / "windrose_plot_rating_per_hour.json"
    with open(rph_json_path, "w", encoding="utf-8") as f:
        json.dump(agg_rph, f, indent=2, ensure_ascii=False, default=float)
    print(f"    JSON (rating/hour)  → {rph_json_path}")

    # ── Export 2: plot-ready JSON for WER chart ────────────────────────────────
    wer_json_path = OUT_DIR / "windrose_plot_wer.json"
    with open(wer_json_path, "w", encoding="utf-8") as f:
        json.dump(agg_wer, f, indent=2, ensure_ascii=False, default=float)
    print(f"    JSON (WER)          → {wer_json_path}")

    wer_cat_json_path = OUT_DIR / "windrose_plot_wer_by_category.json"
    with open(wer_cat_json_path, "w", encoding="utf-8") as f:
        json.dump(category_wer, f, indent=2, ensure_ascii=False, default=float)
    print(f"    JSON (WER by cat)   → {wer_cat_json_path}")

    rph_cat_json_path = OUT_DIR / "windrose_plot_rating_per_hour_by_category.json"
    with open(rph_cat_json_path, "w", encoding="utf-8") as f:
        json.dump(category_rph, f, indent=2, ensure_ascii=False, default=float)
    print(f"    JSON (R/H by cat)   → {rph_cat_json_path}")

    categories_json_path = OUT_DIR / "windrose_category_options.json"
    with open(categories_json_path, "w", encoding="utf-8") as f:
        json.dump({"categories": windrose_categories}, f, indent=2, ensure_ascii=False)
    print(f"    JSON (categories)   → {categories_json_path}")

    # ── Export 3: per-recipe JSON ──────────────────────────────────────────────
    recipes_json_path = OUT_DIR / "windrose_data_recipes.json"
    recipes_out.to_json(
        recipes_json_path,
        orient        = "records",
        indent        = 2,
        force_ascii   = False,
        default_handler = float,
    )
    print(f"    JSON (per-recipe)   → {recipes_json_path}")

    # ── Export 4 & 5: HTML figures ────────────────────────────────────────────
    build_and_export(agg_rph, agg_wer, OUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()