from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import plotly.graph_objects as go
import plotly.io as pio

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "tables" / "food_recipe.db"
OUTDIR = BASE_DIR / "plots" / "complexity_bloom_outputs"

EXPORT_RECIPES = "complexity_recipes.json"
EXPORT_META = "complexity_meta.json"
EXPORT_SEARCH = "complexity_search.json"

_BASE_PALETTE = [
    "#D4537E", "#1D9E75", "#BA7517", "#7F77DD",
    "#378ADD", "#D85A30", "#3B6D11", "#0F6E56",
    "#9B4F96", "#C0392B", "#16A085", "#8E44AD",
    "#2C3E50", "#E67E22", "#27AE60", "#2980B9",
]

_FONT = "'DM Sans', sans-serif"
_MUTED = "#7090a8"

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=_FONT, color=_MUTED, size=12),
    dragmode=False,
    autosize=True,
)


def _cat_color(name: str, index: int) -> str:
    if index < len(_BASE_PALETTE):
        return _BASE_PALETTE[index]
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return _BASE_PALETTE[h % len(_BASE_PALETTE)]


def _rgba(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _parse_list_col(val) -> list[str]:
    if pd.isna(val) or val is None:
        return []
    s = str(val).strip()
    if s.startswith("c("):
        s = s[1:]
    try:
        result = ast.literal_eval(s)
        if isinstance(result, (list, tuple)):
            return [str(x) for x in result]
        return [str(result)]
    except Exception:
        s = s.strip("[]()\"'")
        return [x.strip().strip("\"'") for x in s.split(",") if x.strip()]


def _seconds_to_minutes(val) -> float:
    try:
        v = float(val)
        if v <= 0 or np.isnan(v):
            return np.nan
        return np.clip(v / 60.0, 1.0, 2880.0)
    except Exception:
        return np.nan


def _query_db() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT
            r.RecipeId,
            r.Name,
            r.RecipeCategory,
            r.TotalTime,
            r.RecipeIngredientParts,
            r.RecipeInstructions,
            r.AggregatedRating,
            r.ReviewCount,
            r.Keywords
        FROM recipes r
        WHERE
            r.AggregatedRating IS NOT NULL
            AND r.ReviewCount IS NOT NULL
            AND r.ReviewCount > 0
            AND r.TotalTime IS NOT NULL
            AND r.RecipeCategory IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def _compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["_time_min"] = df["TotalTime"].apply(_seconds_to_minutes)
    df = df.dropna(subset=["_time_min"])

    df["_ing_count"] = df["RecipeIngredientParts"].apply(lambda x: len(_parse_list_col(x)))
    df["_step_count"] = df["RecipeInstructions"].apply(lambda x: len(_parse_list_col(x)))
    df["_keywords"] = df["Keywords"].apply(
        lambda x: ", ".join(_parse_list_col(x)[:5]) if pd.notna(x) else ""
    )
    n_rows = len(df)
    for out_col, src_col in (("time_n", "_time_min"), ("ing_n", "_ing_count"), ("steps_n", "_step_count")):
        df[out_col] = rankdata(df[src_col]) / n_rows

    df["AggregatedRating"] = pd.to_numeric(df["AggregatedRating"], errors="coerce")
    df["ReviewCount"] = pd.to_numeric(df["ReviewCount"], errors="coerce")
    df = df.dropna(subset=["AggregatedRating", "ReviewCount"])

    sat_raw = df["AggregatedRating"] * np.log1p(df["ReviewCount"])
    df["sat_n"] = rankdata(sat_raw) / n_rows

    df["RecipeCategory"] = df["RecipeCategory"].astype(str).str.strip()
    df.loc[
        df["RecipeCategory"].str.lower().isin(["nan", "none", "null", ""]),
        "RecipeCategory"
    ] = "Uncategorised"

    keep = [
        "RecipeId", "Name", "RecipeCategory", "time_n", "ing_n", "steps_n", "sat_n",
        "AggregatedRating", "ReviewCount", "_keywords",
    ]
    return df[keep].reset_index(drop=True)


def _build_meta(df: pd.DataFrame) -> dict:
    cats = sorted([c for c in df["RecipeCategory"].unique().tolist() if isinstance(c, str)])
    color_map = {cat: _cat_color(cat, i) for i, cat in enumerate(cats)}

    cat_stats = []
    for cat in cats:
        sub = df[df["RecipeCategory"] == cat]
        cat_stats.append({
            "name": cat,
            "color": color_map[cat],
            "count": int(len(sub)),
            "mean_sat": round(float(sub["sat_n"].mean()), 4),
            "mean_ing": round(float(sub["ing_n"].mean()), 4),
            "mean_time": round(float(sub["time_n"].mean()), 4),
            "mean_steps": round(float(sub["steps_n"].mean()), 4),
        })

    return {
        "categories": cats,
        "color_map": color_map,
        "cat_stats": cat_stats,
        "total_recipes": int(len(df)),
    }


def _save_png(fig: go.Figure, path: Path, width: int = 1600, height: int = 1200, scale: int = 2) -> None:
    orig_paper = fig.layout.paper_bgcolor
    orig_plot = fig.layout.plot_bgcolor
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f5f9fd")
    try:
        pio.write_image(fig, str(path), format="png", width=width, height=height, scale=scale)
    finally:
        fig.update_layout(paper_bgcolor=orig_paper, plot_bgcolor=orig_plot)


def compute_and_export(max_recipes: int | None = 100000, min_reviews: int = 5) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = _query_db()
    df = df[df["ReviewCount"] >= min_reviews].copy()
    df = _compute_scores(df)

    # if max_recipes is not None and len(df) > max_recipes:
    #     cats = df["RecipeCategory"].unique()
    #     n_each = max(1, max_recipes // len(cats))
    #     sampled = (
    #         df.groupby("RecipeCategory", group_keys=False)
    #         .apply(lambda g: g.sample(min(len(g), n_each), random_state=42), include_groups=False)
    #     )
    #     remaining = max_recipes - len(sampled)
    #     if remaining > 0:
    #         leftover = df[~df.index.isin(sampled.index)]
    #         extra = leftover.sample(min(remaining, len(leftover)), random_state=42)
    #         sampled = pd.concat([sampled, extra])
    #     df = sampled.reset_index(drop=True)

    meta = _build_meta(df)

    records = []
    search_records = []
    for _, row in df.iterrows():
        rec = {
            "id": int(row["RecipeId"]),
            "name": str(row["Name"]),
            "category": str(row["RecipeCategory"]),
            "time_n": round(float(row["time_n"]), 4),
            "ing_n": round(float(row["ing_n"]), 4),
            "steps_n": round(float(row["steps_n"]), 4),
            "sat_n": round(float(row["sat_n"]), 4),
            "rating": round(float(row["AggregatedRating"]), 2),
            "reviews": int(row["ReviewCount"]),
            "keywords": str(row["_keywords"]),
        }
        records.append(rec)
        search_records.append({"id": rec["id"], "name": rec["name"], "category": rec["category"]})

    search_payload = {
        "categories": meta["categories"],
        "recipes": sorted(search_records, key=lambda r: (r["name"].lower(), r["id"])),
    }

    (OUTDIR / EXPORT_RECIPES).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / EXPORT_META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTDIR / EXPORT_SEARCH).write_text(json.dumps(search_payload, ensure_ascii=False), encoding="utf-8")


def _load_exports() -> tuple[list[dict], dict]:
    rec_path = OUTDIR / EXPORT_RECIPES
    meta_path = OUTDIR / EXPORT_META
    if not rec_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Export files not found in {OUTDIR}. Run compute_and_export() first.")
    return json.loads(rec_path.read_text(encoding="utf-8")), json.loads(meta_path.read_text(encoding="utf-8"))


def load_search_payload() -> dict:
    path = OUTDIR / EXPORT_SEARCH
    if not path.exists():
        records, meta = _load_exports()
        return {
            "categories": meta["categories"],
            "recipes": [{"id": r["id"], "name": r["name"], "category": r["category"]} for r in records],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _complexity(r: dict, wt: float, wi: float, ws: float) -> float:
    total = wt + wi + ws or 1.0
    return (r["time_n"] * wt + r["ing_n"] * wi + r["steps_n"] * ws) / total


def _find_golden_recipe(recs: Iterable[dict], wt: float, wi: float, ws: float) -> int | None:
    best_id = None
    best_score = None
    for r in recs:
        cpx = _complexity(r, wt, wi, ws)
        score = 0.72 * r["sat_n"] - 0.28 * cpx + 0.03 * np.log1p(max(r.get("reviews", 0), 0))
        if best_score is None or score > best_score:
            best_score = score
            best_id = r["id"]
    return best_id


def resolve_recipe(records: list[dict], recipe_id: int | None = None, recipe_name: str | None = None) -> dict | None:
    if recipe_id is not None:
        for r in records:
            if int(r["id"]) == int(recipe_id):
                return r
    if recipe_name:
        needle = recipe_name.strip().lower()
        for r in records:
            if r["name"].strip().lower() == needle:
                return r
    return None


def plot_bloom_from_export(
    category: str | None = None,
    wt: float = 100,
    wi: float = 35,
    ws: float = 25,
    canvas_w: float = 700,
    canvas_h: float = 620,
    highlight_recipe_id: int | None = None,
    show_golden: bool = False,
) -> go.Figure:
    records, meta = _load_exports()
    color_map = meta["color_map"]
    categories = meta["categories"]

    if category is None:
        category = categories[0]
    if category not in color_map:
        raise ValueError(f"Category '{category}' not found.")

    recs = [r for r in records if r["category"] == category]
    col = color_map[category]
    cr, cg, cb = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)

    golden_id = _find_golden_recipe(recs, wt, wi, ws) if show_golden else None

    cx, cy = canvas_w / 2, canvas_h / 2 - 10
    R = min(canvas_w, canvas_h) * 0.37
    theta = np.linspace(0, np.pi, 90)
    traces: list[go.Scatter] = []

    for f in (0.33, 0.66, 1.0):
        traces.append(go.Scatter(
            x=cx + R * f * np.cos(theta), y=cy + R * f * np.sin(theta),
            mode="lines", line=dict(color="rgba(46,136,204,0.5)", width=1.0, dash="dot"),
            hoverinfo="none", showlegend=False,
        ))
    for i in range(9):
        a = i / 8 * np.pi
        traces.append(go.Scatter(
            x=[cx, cx + np.cos(a) * R], y=[cy, cy + np.sin(a) * R],
            mode="lines", line=dict(color="rgba(46,136,204,0.5)", width=1.0),
            hoverinfo="none", showlegend=False,
        ))

    max_reviews = max((r["reviews"] for r in recs), default=1)
    warm = np.array([249, 203, 66])
    cool = np.array([int(cr * 0.45 + 55), int(cg * 0.30), int(cb * 0.70)])

    for r in recs:
        c = _complexity(r, wt, wi, ws)
        angle = np.pi - c * np.pi
        dist = r["sat_n"] * R * 0.94 + R * 0.03
        pw = np.sqrt(r["reviews"] / max_reviews) * R * 0.10 + R * 0.030
        ph = dist * 0.20 + 4
        px, py = cx + np.cos(angle) * dist, cy + np.sin(angle) * dist

        t = r["sat_n"]
        rgb = (cool + (warm - cool) * t).astype(int)
        alpha = 0.50 + t * 0.38

        k = np.linspace(0, 2 * np.pi, 25)
        lx = pw * np.cos(k)
        ly = ph * np.sin(k)
        rot = angle + np.pi / 2
        ex = px + lx * np.cos(rot) - ly * np.sin(rot)
        ey = py + lx * np.sin(rot) + ly * np.cos(rot)

        is_selected = highlight_recipe_id is not None and int(r["id"]) == int(highlight_recipe_id)
        is_golden = golden_id is not None and int(r["id"]) == int(golden_id)
        line_color = f"rgba({int(rgb[0])},{int(rgb[1])},{int(rgb[2])},0.150)"
        line_width = 0.3

        if is_golden:
            line_color = "rgba(249,203,66,0.95)"
            line_width = 3.0
        if is_selected:
            line_color = "rgba(11,51,82,0.95)"
            line_width = 3.2

        tag = []
        if is_selected:
            tag.append("Selected recipe")
        if is_golden:
            tag.append("Golden recipe")
        tag_html = f"<br><b>{' · '.join(tag)}</b>" if tag else ""

        tooltip = (
            f"<b>{r['name']}</b><br>"
            f"Complexity: {c * 100:.0f}%<br>"
            f"Satisfaction: {r['sat_n'] * 100:.0f}%<br>"
            f"Rating: {r['rating']:.1f}★ · {r['reviews']:,} reviews"
            f"{tag_html}"
            + (f"<br><i>{r['keywords']}</i>" if r.get("keywords") else "")
        )

        traces.append(go.Scatter(
            x=ex, y=ey, mode="lines", fill="toself",
            fillcolor=f"rgba({int(rgb[0])},{int(rgb[1])},{int(rgb[2])},{alpha:.3f})",
            line=dict(color=line_color, width=line_width),
            text=tooltip, hoverinfo="text",
            hoverlabel=dict(bgcolor="#fff", bordercolor="#c5dff4", font=dict(family=_FONT, size=12, color="#16283a")),
            showlegend=False,
        ))

    annotations = [
        dict(x=cx - R + 20, y=cy + 7, xref="x", yref="y", showarrow=False, text="Simple", font=dict(size=17, color="rgba(46,136,204,1.0)", family=_FONT)),
        dict(x=cx + R - 20, y=cy + 7, xref="x", yref="y", showarrow=False, text="Complex", font=dict(size=17, color="rgba(46,136,204,1.0)", family=_FONT)),
        dict(x=cx + R * 0.25 + 8, y=cy - 7, xref="x", yref="y", showarrow=False, text="Low sat.", font=dict(size=17, color="rgba(46,136,204,1.0)", family=_FONT)),
        dict(x=cx + R * 0.90 + 8, y=cy - 7, xref="x", yref="y", showarrow=False, text="High sat.", font=dict(size=17, color="rgba(46,136,204,1.0)", family=_FONT)),
        dict(x=cx, y=cy + R + 10, xref="x", yref="y", showarrow=False, text=f"<b>{category}</b>", font=dict(size=17, color=col, family=_FONT)),
    ]

    all_x, all_y = [], []
    for trace in traces:
        all_x.extend(trace.x)
        all_y.extend(trace.y)

    pad = 20
    x_min, x_max = min(all_x) - pad, max(all_x) + pad
    y_min, y_max = min(all_y) - pad, max(all_y) + pad

    layout = go.Layout(
        **_BASE_LAYOUT,
        xaxis=dict(visible=False, range=[x_min, x_max]),
        yaxis=dict(visible=False, range=[y_min, y_max], scaleanchor="x", scaleratio=1),
        showlegend=False,
        annotations=annotations,
        margin=dict(t=0, b=0, l=10, r=10),
    )
    return go.Figure(data=traces, layout=layout)


def plot_constellation_from_export(
    wt: float = 100,
    wi: float = 100,
    ws: float = 100,
    gem_sat_thresh: float = 0.70,
    gem_cpx_thresh: float = 0.40,
    canvas_w: float = 800,
    canvas_h: float = 600,
) -> go.Figure:
    records, meta = _load_exports()
    color_map = meta["color_map"]
    categories = meta["categories"]

    cx, cy = canvas_w / 2, canvas_h / 2
    R = min(canvas_w, canvas_h) * 0.36

    positioned = []
    for r in records:
        c = _complexity(r, wt, wi, ws)
        angle = np.pi + c * np.pi
        dist = r["sat_n"] * R * 0.92 + R * 0.04
        x = cx + np.cos(angle) * dist
        y = cy + np.sin(angle) * dist
        positioned.append({**r, "cx": x, "cy": y, "cpx": c})

    traces = []
    theta = np.linspace(np.pi, 2*np.pi, 90)
    for f in (0.33, 0.66, 1.0):
        traces.append(go.Scatter(
            x=cx + R * f * np.cos(theta), y=cy + R * f * np.sin(theta),
            mode="lines", line=dict(color="rgba(46,136,204,0.4)", width=1.0, dash="dot"),
            hoverinfo="none", showlegend=False,
        ))

    for cat in categories:
        pts = [p for p in positioned if p["category"] == cat]
        col = color_map[cat]
        xs, ys, sizes, colors, texts = [], [], [], [], []
        for p in pts:
            is_gem = p["sat_n"] > gem_sat_thresh and p["cpx"] < gem_cpx_thresh
            xs.append(p["cx"])
            ys.append(p["cy"])
            sizes.append(4 + np.sqrt(p["reviews"] / 300) * 10)
            colors.append("#f9cb42" if is_gem else _rgba(col, 0.8))
            texts.append(
                f"<b>{p['name']}</b><br>Category: {cat}<br>"
                f"Complexity: {p['cpx'] * 100:.0f}%<br>Satisfaction: {p['sat_n'] * 100:.0f}%<br>"
                f"Rating: {p['rating']:.1f}★ · {p['reviews']:,} reviews"
            )
        traces.append(go.Scatter(
            x=xs, y=ys, mode="markers", name=cat,
            marker=dict(size=sizes, color=colors, line=dict(width=0)),
            text=texts, hoverinfo="text",
            hoverlabel=dict(bgcolor="#fff", bordercolor="#c5dff4", font=dict(family=_FONT, size=12, color="#16283a")),
        ))

    all_x = []
    all_y = []
    for trace in traces:
        all_x.extend(trace.x)
        all_y.extend(trace.y)

    # find the tight bounding box
    pad = 20   # small padding in data units so nothing touches the edge
    x_min, x_max = min(all_x) - pad, max(all_x) + pad
    y_min, y_max = min(all_y) - pad, max(all_y) + pad

    layout = go.Layout(
        **_BASE_LAYOUT,
        xaxis=dict(visible=False, range=[x_min, x_max]),
        yaxis=dict(visible=False, range=[y_min, y_max],  # inverted: large→small = top→bottom
                scaleanchor="x", scaleratio=1),
        showlegend=False,
        annotations=annotations,
        margin=dict(t=0, b=0, l=10, r=10),
    )
    return go.Figure(data=traces, layout=layout)



def plot_bin_from_export(
    wt: float = 40,
    wi: float = 35,
    ws: float = 25,
    grid_n: int = 20,
) -> go.Figure:
    records, meta = _load_exports()
    color_map = meta["color_map"]
    categories = meta["categories"]

    cpx = np.array([_complexity(r, wt, wi, ws) for r in records])
    sat = np.array([r["sat_n"] for r in records])
    cats_idx = {cat: i for i, cat in enumerate(categories)}
    cat_ids = np.array([cats_idx.get(r["category"], 0) for r in records])

    gx_arr = np.clip((cpx * grid_n).astype(int), 0, grid_n - 1)
    gy_arr = np.clip((sat * grid_n).astype(int), 0, grid_n - 1)

    counts = np.zeros((grid_n, grid_n), dtype=int)
    cat_counts = np.zeros((grid_n, grid_n, len(categories)), dtype=int)

    for i in range(len(records)):
        counts[gy_arr[i], gx_arr[i]] += 1
        cat_counts[gy_arr[i], gx_arr[i], cat_ids[i]] += 1

    max_count = counts.max() if counts.max() > 0 else 1

    cell_w = 1.0 / grid_n
    cell_h = 1.0 / grid_n
    gap = cell_w * 0.06

    traces: list[go.Scatter] = []

    for cat in categories:
        col = color_map[cat]
        traces.append(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=cat,
            marker=dict(size=8, color=col, symbol="square"),
            showlegend=True,
        ))

    for gy in range(grid_n):
        for gx in range(grid_n):
            n = counts[gy, gx]
            if n == 0:
                continue

            dom_ci = int(cat_counts[gy, gx].argmax())
            dom_cat = categories[dom_ci]
            col_hex = color_map[dom_cat]
            r_c = int(col_hex[1:3], 16)
            g_c = int(col_hex[3:5], 16)
            b_c = int(col_hex[5:7], 16)

            alpha = 0.10 + 0.85 * (n / max_count) ** 0.45

            x0 = gx * cell_w + gap
            x1 = (gx + 1) * cell_w - gap
            y0 = gy * cell_h + gap
            y1 = (gy + 1) * cell_h - gap

            px = [x0, x1, x1, x0, x0]
            py = [y0, y0, y1, y1, y0]

            top3 = sorted(
                [(cat_counts[gy, gx, ci], categories[ci]) for ci in range(len(categories)) if cat_counts[gy, gx, ci] > 0],
                reverse=True,
            )[:3]
            breakdown = "<br>".join(f"  {name}: {cnt}" for cnt, name in top3)
            cx_val = (gx + 0.5) * cell_w
            cy_val = (gy + 0.5) * cell_h
            tip = (
                f"<b>{n} recipe{'s' if n != 1 else ''}</b><br>"
                f"complexity ≈ {cx_val:.2f} · satisfaction ≈ {cy_val:.2f}<br>"
                f"{breakdown}"
            )

            traces.append(go.Scatter(
                x=px, y=py,
                mode="lines", fill="toself",
                fillcolor=f"rgba({r_c},{g_c},{b_c},{alpha:.3f})",
                line=dict(color="rgba(0,0,0,0)", width=0),
                text=tip, hoverinfo="text",
                hoverlabel=dict(
                    bgcolor="#fff", bordercolor="#c5dff4",
                    font=dict(family=_FONT, size=12, color="#16283a"),
                ),
                showlegend=False,
            ))

    label_style = dict(size=11, color=_MUTED, family=_FONT)
    annotations = [
        dict(x=0.01,  y=0.01,  xref="x", yref="y", showarrow=False,
             text="Simple · Low sat.",    font=label_style, xanchor="left",  yanchor="bottom"),
        dict(x=0.99,  y=0.01,  xref="x", yref="y", showarrow=False,
             text="Complex · Low sat.",   font=label_style, xanchor="right", yanchor="bottom"),
        dict(x=0.01,  y=0.99,  xref="x", yref="y", showarrow=False,
             text="Simple · High sat.",   font=label_style, xanchor="left",  yanchor="top"),
        dict(x=0.99,  y=0.99,  xref="x", yref="y", showarrow=False,
             text="Complex · High sat.",  font=label_style, xanchor="right", yanchor="top"),
    ]

    n_total = int(counts.sum())
    layout = go.Layout(
        **_BASE_LAYOUT,
        xaxis=dict(
            title=dict(text="Complexity", font=dict(family=_FONT, size=13, color=_MUTED)),
            range=[0, 1],
            showgrid=True, gridcolor="rgba(46,136,204,0.08)", gridwidth=1,
            zeroline=False,
            tickfont=dict(family=_FONT, size=11, color=_MUTED),
            tickformat=".2f",
        ),
        yaxis=dict(
            title=dict(text="Satisfaction", font=dict(family=_FONT, size=13, color=_MUTED)),
            range=[0, 1],
            showgrid=True, gridcolor="rgba(46,136,204,0.08)", gridwidth=1,
            zeroline=False,
            tickfont=dict(family=_FONT, size=11, color=_MUTED),
            tickformat=".2f",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.12,
            font=dict(family=_FONT, size=11, color=_MUTED),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        annotations=annotations,
        margin=dict(t=40, b=100, l=60, r=20),
        title=dict(
            text=f"Recipe Landscape — {n_total:,} recipes · {grid_n}×{grid_n} grid",
            font=dict(family=_FONT, size=14, color=_MUTED),
            x=0.5, xanchor="center",
        ),
    )

    return go.Figure(data=traces, layout=layout)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export complexity bloom data from Food.com DB.")
    parser.add_argument("--max-recipes", type=int, default=8000)
    parser.add_argument("--min-reviews", type=int, default=5)
    args = parser.parse_args()
    max_r = args.max_recipes if args.max_recipes > 0 else None
    compute_and_export(max_recipes=max_r, min_reviews=args.min_reviews)
