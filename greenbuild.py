"""
GreenBuild Analyzer - compare wall materials by cost, embodied carbon and durability.

!! The numbers in MATERIALS are PLACEHOLDERS for demonstration. Replace them with
real data (local prices in ETB, embodied carbon from ICE / EC3 / published studies)
before you present any result. Keep a "source" for every row.

Sections
  1. Data & cleaning          3. Simulation (weights + data uncertainty)
  2. Scoring & lifecycle      4. Tests        5. Streamlit app (UI)

Run the app   :  streamlit run greenbuild.py
Run the tests :  python greenbuild.py test
"""
from __future__ import annotations

import html
import sys

import numpy as np
import pandas as pd

APP_NAME = "GreenBuild Analyzer"
APP_VERSION = "1.0"

# ==================================================================
# 1. DATA & CLEANING
# ==================================================================
# Functional unit: 1 m2 of finished wall.
MATERIALS = pd.DataFrame(
    [
        # material,                       cost_etb_m2, co2_kg_m2, life_years, source
        ("Hollow concrete block",             1100,       45,        50,  "placeholder"),
        ("Fired clay brick",                  1300,       75,        60,  "placeholder"),
        ("Compressed stabilized earth block",  900,       20,        40,  "placeholder"),
        ("Rammed earth",                       800,       15,        50,  "placeholder"),
        ("Adobe (mud brick) + plaster",        600,        8,        25,  "placeholder"),
        ("Bamboo panel wall",                  700,       10,        20,  "placeholder"),
        ("Stone masonry",                     1500,       15,        80,  "placeholder"),
        ("Timber frame + infill",             1600,       30,        40,  "placeholder"),
    ],
    columns=["material", "cost_etb_m2", "co2_kg_m2", "life_years", "source"],
)
REQUIRED = ["material", "cost_etb_m2", "co2_kg_m2", "life_years"]
NUMERIC = ["cost_etb_m2", "co2_kg_m2", "life_years"]

PRESETS = {
    "Balanced": (5, 5, 5),
    "Budget first": (9, 3, 3),
    "Climate first": (3, 9, 3),
    "Durability first": (3, 3, 9),
}


def clean_data(raw: pd.DataFrame, default_source: str = "user upload"):
    """Validate a user table. Returns (clean_df, notes). Raises ValueError if columns missing."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if "source" not in df.columns:
        df["source"] = default_source
    notes = []

    df["material"] = df["material"].astype("string").str.strip()
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["source"] = df["source"].astype("string").fillna("").str.strip()
    df.loc[df["source"] == "", "source"] = "unspecified"

    incomplete = df["material"].isna() | (df["material"] == "") | df[NUMERIC].isna().any(axis=1)
    if incomplete.any():
        notes.append(f"{int(incomplete.sum())} incomplete row(s) ignored.")
    df = df[~incomplete]
    bad = (df["cost_etb_m2"] <= 0) | (df["life_years"] <= 0) | (df["co2_kg_m2"] < 0)
    if bad.any():
        notes.append(f"{int(bad.sum())} row(s) with zero/negative cost or lifespan, or negative carbon, ignored.")
    df = df[~bad]
    dup = df["material"].duplicated()
    if dup.any():
        notes.append(f"{int(dup.sum())} duplicate material name(s) ignored (first kept).")
    df = df[~dup]
    df = df[REQUIRED + ["source"]].reset_index(drop=True)
    df["material"] = df["material"].astype(str)
    df["source"] = df["source"].astype(str)
    return df, notes


# ==================================================================
# 2. SCORING & LIFECYCLE
# ==================================================================
def minmax(s: pd.Series, higher_is_better: bool) -> pd.Series:
    """Scale to 0-1 where 1 is always best."""
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(1.0, index=s.index)
    scaled = (s - s.min()) / rng
    return scaled if higher_is_better else 1 - scaled


def builds_needed(life, horizon: float, mode: str):
    """How many times the wall is built over the study period.
    mode 'whole'   : whole rebuilds (ceil), e.g. 20-year wall over 50 years -> 3
    mode 'prorata' : only the years actually used count, e.g. 50/20 = 2.5"""
    life = np.asarray(life, dtype=float)
    return np.ceil(horizon / life - 1e-9) if mode == "whole" else horizon / life


def add_metrics(df, area, horizon, mode, basis):
    """Add project-level and lifecycle columns. basis: 'initial' or 'lifecycle'."""
    d = df.copy()
    d["builds"] = builds_needed(d["life_years"], horizon, mode)
    d["init_cost"] = d["cost_etb_m2"] * area
    d["init_co2"] = d["co2_kg_m2"] * area
    d["lc_cost"] = d["init_cost"] * d["builds"]
    d["lc_co2"] = d["init_co2"] * d["builds"]
    d["cost_per_year"] = d["init_cost"] / d["life_years"]
    d["co2_per_year"] = d["init_co2"] / d["life_years"]
    d["eval_cost"] = d["lc_cost"] if basis == "lifecycle" else d["init_cost"]
    d["eval_co2"] = d["lc_co2"] if basis == "lifecycle" else d["init_co2"]
    return d


def score(df: pd.DataFrame, w_cost: float, w_co2: float, w_life: float) -> pd.Series:
    """Weighted 0-1 score using eval_cost / eval_co2 / life_years."""
    total = (w_cost + w_co2 + w_life) or 1
    return (w_cost * minmax(df["eval_cost"], False)
            + w_co2 * minmax(df["eval_co2"], False)
            + w_life * minmax(df["life_years"], True)) / total


def pareto_mask(cost, co2) -> np.ndarray:
    """True where no other option is at least as good on both and better on one."""
    c, e = np.asarray(cost, float), np.asarray(co2, float)
    keep = np.ones(len(c), dtype=bool)
    for i in range(len(c)):
        dom = (c <= c[i]) & (e <= e[i]) & ((c < c[i]) | (e < e[i]))
        keep[i] = not dom.any()
    return keep


def explain_best(f: pd.DataFrame) -> str:
    """One plain-language sentence on why the top option leads."""
    b = f.iloc[0]
    if len(f) == 1:
        return f"**{b['material']}** is the only option that meets your limits."
    parts = []
    for col, label in [("eval_cost", "cost"), ("eval_co2", "embodied carbon"), ("life_years", "lifespan")]:
        med = f[col].median()
        if med:
            diff = (b[col] - med) / med * 100
            if abs(diff) >= 1:
                parts.append(f"{abs(diff):.0f}% {'lower' if diff < 0 else 'higher'} {label}")
    tail = (" than the median option" if parts else "")
    return f"**{b['material']}** leads with " + (", ".join(parts) if parts else "a balanced profile") + tail + "."


# ==================================================================
# 3. SIMULATION (priority weights and data uncertainty)
# ==================================================================
def _mm_rows(x: np.ndarray, higher_is_better: bool) -> np.ndarray:
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    rng = hi - lo
    safe = np.where(rng == 0, 1.0, rng)
    scaled = (x - lo) / safe
    out = scaled if higher_is_better else 1 - scaled
    return np.where(rng == 0, 1.0, out)


def simulate(df, area, horizon, mode, basis, weights=None, uncertainty=0.0, n=2000, seed=0):
    """Monte-Carlo ranking. weights=None -> random priorities; else fixed (cost, co2, life).
    uncertainty=0.2 perturbs every input by up to +/-20%. Returns pct_first and avg_rank."""
    rng = np.random.default_rng(seed)
    m = len(df)
    if weights is None:
        W = rng.dirichlet([1, 1, 1], size=n)
    else:
        w = np.asarray(weights, float)
        w = w / w.sum() if w.sum() else np.ones(3) / 3
        W = np.tile(w, (n, 1))

    def jitter():
        return 1 + rng.uniform(-uncertainty, uncertainty, size=(n, m)) if uncertainty > 0 else np.ones((n, m))

    cost = df["cost_etb_m2"].values[None, :] * jitter()
    co2 = df["co2_kg_m2"].values[None, :] * jitter()
    life = np.maximum(df["life_years"].values[None, :] * jitter(), 1e-9)
    if basis == "lifecycle":
        b = builds_needed(life, horizon, mode)
        ec, ee = cost * area * b, co2 * area * b
    else:
        ec, ee = cost * area, co2 * area
    S = (W[:, [0]] * _mm_rows(ec, False) + W[:, [1]] * _mm_rows(ee, False)
         + W[:, [2]] * _mm_rows(life, True))
    winners = S.argmax(axis=1)
    ranks = (-S).argsort(axis=1).argsort(axis=1) + 1
    out = pd.DataFrame({"material": df["material"].values,
                        "pct_first": np.bincount(winners, minlength=m) / n * 100,
                        "avg_rank": ranks.mean(axis=0)})
    return out.sort_values(["pct_first", "avg_rank"], ascending=[False, True]).reset_index(drop=True)


def build_summary(inputs: dict, f: pd.DataFrame, placeholder: bool) -> str:
    lines = [f"# {APP_NAME} - summary", ""]
    if placeholder:
        lines += ["> **Warning:** some values are PLACEHOLDERS, not sourced data.", ""]
    lines += ["## Project", *[f"- {k}: {v}" for k, v in inputs.items()], "", "## Top options"]
    for i, (_, r) in enumerate(f.head(3).iterrows(), 1):
        lines.append(f"{i}. **{r['material']}** - score {r['score']:.2f}, "
                     f"cost {r['eval_cost']:,.0f} ETB, carbon {r['eval_co2']:,.0f} kg CO2e, "
                     f"life {r['life_years']:.0f} years (source: {r['source']})")
    lines += ["", explain_best(f), "",
              "Method: criteria scaled 0-1 (1 = best), combined with the chosen priority weights. "
              "Embodied carbon covers production only."]
    return "\n".join(lines)


# ==================================================================
# 4. TESTS
# ==================================================================
def run_tests() -> bool:
    passed = total = 0

    def check(name, cond, detail=""):
        nonlocal passed, total
        total += 1
        passed += bool(cond)
        print("PASS" if cond else "FAIL", name, detail)

    s = pd.Series([10.0, 20.0, 30.0])
    check("minmax higher-is-better", list(minmax(s, True)) == [0, .5, 1])
    check("minmax lower-is-better", list(minmax(s, False)) == [1, .5, 0])
    check("minmax constant series", list(minmax(pd.Series([5, 5]), True)) == [1, 1])

    check("whole rebuilds", builds_needed([20, 50, 80], 50, "whole").tolist() == [3, 1, 1])
    check("pro-rata builds", np.allclose(builds_needed([20, 50, 100], 50, "prorata"), [2.5, 1, .5]))

    d = add_metrics(MATERIALS, 100, 50, "whole", "lifecycle")
    check("initial cost = cost x area", d.loc[0, "init_cost"] == 110000)
    check("lifecycle cost uses builds", d.loc[5, "lc_cost"] == 700 * 100 * 3)  # bamboo, 20y -> 3 builds
    check("basis initial uses initial", add_metrics(MATERIALS, 100, 50, "whole", "initial")
          .loc[5, "eval_cost"] == 70000)

    d["score"] = score(d, 5, 5, 5)
    check("scores within 0-1", d["score"].between(0, 1).all())
    cost_only = score(d, 1, 0, 0)
    check("cost-only winner is cheapest", d.loc[cost_only.idxmax(), "material"] == d.loc[d["eval_cost"].idxmin(), "material"])
    check("zero weights do not crash", score(d, 0, 0, 0).notna().all())

    pm = pareto_mask([1, 2, 3, 2], [3, 2, 1, 4])
    check("pareto front", pm.tolist() == [True, True, True, False], str(pm.tolist()))

    sim = simulate(MATERIALS, 100, 50, "whole", "lifecycle")
    check("random-weight shares sum to 100", abs(sim["pct_first"].sum() - 100) < 1e-6)
    sim2 = simulate(MATERIALS, 100, 50, "whole", "lifecycle", weights=(5, 5, 5), uncertainty=0.2)
    check("uncertainty shares sum to 100", abs(sim2["pct_first"].sum() - 100) < 1e-6)
    check("avg rank between 1 and n", sim2["avg_rank"].between(1, len(MATERIALS)).all())
    fixed = simulate(MATERIALS, 100, 50, "whole", "initial", weights=(5, 5, 5), uncertainty=0.0, n=50)
    check("no uncertainty -> one deterministic winner", (fixed["pct_first"] == 100).sum() == 1)

    bad = pd.DataFrame({"Material": ["A", "A", "B", "", "C", "D"],
                        "cost_etb_m2": [100, 100, "x", 50, 80, -5],
                        "CO2_kg_m2": [1, 1, 1, 1, 1, 1], "life_years": [10, 10, 10, 10, 0, 10]})
    try:
        clean_data(bad.rename(columns={"CO2_kg_m2": "co2_kg_m2"}))
        ok = True
    except ValueError:
        ok = False
    cl, notes = clean_data(bad.rename(columns={"CO2_kg_m2": "co2_kg_m2"}))
    check("clean_data keeps only valid unique rows", ok and cl["material"].tolist() == ["A"], str(notes))
    try:
        clean_data(pd.DataFrame({"material": ["A"]}))
        check("missing columns raise", False)
    except ValueError:
        check("missing columns raise", True)

    f = d.sort_values("score", ascending=False).reset_index(drop=True)
    check("explain_best mentions top material", f.loc[0, "material"] in explain_best(f))
    check("summary renders", "Top options" in build_summary({"Area": "100 m2"}, f, True))

    print(f"\nTOTAL: {passed}/{total} passed ({passed / total:.0%})")
    return passed == total


# ==================================================================
# 5. STREAMLIT APP (UI)
# ==================================================================
st = px = go = None   # bound inside main()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--g:#0B7A3B;--g2:#14A44D;--gd:#095F2E;--ink:#0F1F17;--muted:#5B6B62;--line:#E3EAE6;}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif; }
.stApp { background:#F4F7F5; }
.block-container { padding-top:1.2rem; max-width:1240px; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid var(--line); }
#MainMenu, footer { visibility:hidden; }
.hero { background:linear-gradient(120deg,#095F2E 0%,#0B7A3B 55%,#14A44D 100%); border-radius:18px;
  padding:1.35rem 1.7rem; display:flex; justify-content:space-between; align-items:center;
  margin-bottom:1.1rem; box-shadow:0 8px 24px rgba(11,122,59,.22); gap:1rem; }
.stApp .hero .hero-title { color:#fff; font-size:1.8rem; font-weight:700; line-height:1.2; }
.stApp .hero .hero-sub { color:#DDF5E7; font-size:.97rem; margin-top:.2rem; }
.stApp .hero .pill { background:rgba(255,255,255,.18); color:#fff; border:1px solid rgba(255,255,255,.4);
  padding:.3rem .8rem; border-radius:999px; font-size:.72rem; font-weight:600; letter-spacing:.08em; white-space:nowrap; }
.brand { display:flex; align-items:center; gap:.6rem; margin:.2rem 0 .6rem; }
.brand .logo { width:38px; height:38px; border-radius:11px; background:linear-gradient(135deg,#0B7A3B,#14A44D);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:1.1rem; }
.brand .name { font-weight:700; font-size:1.05rem; color:var(--ink); line-height:1.1; }
.brand .tag { font-size:.72rem; color:var(--muted); }
.kpi { background:#fff; border:1px solid var(--line); border-left:5px solid var(--g2); border-radius:14px;
  padding:.9rem 1.1rem; box-shadow:0 1px 2px rgba(16,24,40,.05); height:100%; }
.kpi.amber { border-left-color:#F59E0B; } .kpi.blue { border-left-color:#2563EB; } .kpi.grey { border-left-color:#9CA3AF; }
.kpi .lbl { font-size:.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
.kpi .val { font-size:1.25rem; font-weight:700; color:var(--ink); line-height:1.3; }
.kpi .hint { font-size:.8rem; color:var(--muted); }
.section-title { font-weight:700; color:var(--ink); font-size:1.05rem; margin:.4rem 0 .5rem; }
.callout { background:#ECFDF3; border:1px solid #A6E9C2; color:#0F1F17; border-radius:12px; padding:.7rem 1rem; margin:.6rem 0; }
.warnbox { background:#FFF8E6; border:1px solid #FCDD8A; color:#7A5600; border-radius:12px;
  padding:.6rem .9rem; font-size:.88rem; margin-bottom:1rem; }
</style>
"""

COLORS = ["#0B7A3B", "#F59E0B", "#2563EB", "#DC2626", "#8B5CF6", "#0891B2", "#65A30D", "#DB2777"]


def esc(x) -> str:
    return html.escape(str(x))


def _new_width_api() -> bool:
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
        return (major, minor) >= (1, 50)
    except Exception:
        return False


def stretch() -> dict:
    return {"width": "stretch"} if _new_width_api() else {"use_container_width": True}


def hero(title, subtitle, pill="DEMO"):
    st.markdown(f"<div class='hero'><div><div class='hero-title'>{esc(title)}</div>"
                f"<div class='hero-sub'>{esc(subtitle)}</div></div><span class='pill'>{esc(pill)}</span></div>",
                unsafe_allow_html=True)


def kpi(col, label, value, hint="", tone=""):
    col.markdown(f"<div class='kpi {tone}'><div class='lbl'>{esc(label)}</div>"
                 f"<div class='val'>{esc(value)}</div><div class='hint'>{esc(hint)}</div></div>",
                 unsafe_allow_html=True)


def section(title):
    st.markdown(f"<div class='section-title'>{esc(title)}</div>", unsafe_allow_html=True)


def style_fig(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui, sans-serif", size=13),
                      legend=dict(orientation="h", y=-0.2), title_font_size=15)
    return fig


def main():
    global st, px, go
    import plotly.express as _px
    import plotly.graph_objects as _go
    import streamlit as _st
    st, px, go = _st, _px, _go

    st.set_page_config(page_title=APP_NAME, page_icon="🌱", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("reset_n", 0)

    # ------------------------------------------------ sidebar inputs
    with st.sidebar:
        st.markdown(f"<div class='brand'><div class='logo'>G</div><div><div class='name'>{APP_NAME}</div>"
                    "<div class='tag'>Wall materials: cost · carbon · durability</div></div></div>",
                    unsafe_allow_html=True)
        st.header("Your project")
        area = st.number_input("Wall area (m²)", min_value=1.0, value=100.0, step=10.0)
        budget = st.number_input("Upfront budget (ETB)", min_value=0.0, value=150000.0, step=5000.0)
        min_life = st.slider("Minimum lifespan (years)", 0, 80, 20)
        horizon = st.slider("Study period (years)", 10, 100, 50, 5,
                            help="Length of time the building must keep standing.")
        basis_label = st.radio("Compare on", ["Initial build", "Lifecycle (with rebuilds)"],
                               help="Lifecycle counts the cost and carbon of rebuilding short-lived walls.")
        basis = "lifecycle" if basis_label.startswith("Lifecycle") else "initial"
        mode_label = st.radio("Rebuild counting", ["Whole rebuilds", "Pro-rata (years used)"],
                              disabled=(basis == "initial"))
        mode = "whole" if mode_label.startswith("Whole") else "prorata"

        st.header("Priorities")
        preset = st.selectbox("Preset", ["Custom"] + list(PRESETS))
        locked = preset != "Custom"
        pw = PRESETS.get(preset, (5, 5, 5))
        w_cost = st.slider("Low cost", 0, 10, pw[0], disabled=locked)
        w_co2 = st.slider("Low carbon", 0, 10, pw[1], disabled=locked)
        w_life = st.slider("Durability", 0, 10, pw[2], disabled=locked)

        st.header("Data")
        up = st.file_uploader("Upload your own CSV (optional)", type="csv")
        st.caption(f"v{APP_VERSION}")

    # ------------------------------------------------ data
    base, notes = MATERIALS.copy(), []
    uploaded = False
    if up is not None:
        try:
            base, notes = clean_data(pd.read_csv(up))
            uploaded = True
        except Exception as e:
            st.sidebar.error(f"Could not use that CSV: {e}")
            base = MATERIALS.copy()

    placeholder_now = None  # decided after edits
    hero("Compare wall materials", "Cost, embodied carbon and durability, and how sure the ranking is",
         "PLACEHOLDER DATA" if (not uploaded and (base["source"].str.lower() == "placeholder").any()) else "YOUR DATA")

    with st.expander("✏️ Edit material data"):
        st.caption("Change numbers, add rows (+), and fill the *source* column. Edits apply immediately.")
        key = f"ed_{up.name if up is not None else 'default'}_{st.session_state['reset_n']}"
        edited = st.data_editor(base, num_rows="dynamic", key=key, hide_index=True, **stretch())
        if st.button("Reset to original data"):
            st.session_state["reset_n"] += 1
            st.rerun()
    try:
        data, more = clean_data(edited)
        notes += more
    except ValueError as e:
        st.error(f"Data problem: {e}")
        st.stop()
    for n in notes:
        st.info(n)
    if data.empty:
        st.error("No valid materials to analyse. Add at least one complete row.")
        st.stop()

    n_ph = int((data["source"].str.lower() == "placeholder").sum())
    if n_ph:
        st.markdown(f"<div class='warnbox'><b>Placeholder data:</b> {n_ph} of {len(data)} materials use "
                    "placeholder numbers, not sourced values. Replace them (with a source for each row) "
                    "before drawing conclusions.</div>", unsafe_allow_html=True)

    # ------------------------------------------------ compute
    df = add_metrics(data, area, horizon, mode, basis)
    df["within_budget"] = df["init_cost"] <= budget
    df["meets_life"] = df["life_years"] >= min_life
    feasible = df[df["within_budget"] & df["meets_life"]].copy()
    excluded = df[~(df["within_budget"] & df["meets_life"])]

    if feasible.empty:
        st.error("No material meets your budget and lifespan limits. Raise the budget or lower the "
                 "minimum lifespan.")
        st.dataframe(df[["material", "init_cost", "life_years", "within_budget", "meets_life"]],
                     hide_index=True, **stretch())
        st.stop()

    feasible["score"] = score(feasible, w_cost, w_co2, w_life)
    feasible = feasible.sort_values("score", ascending=False).reset_index(drop=True)
    feasible["rank"] = feasible.index + 1
    feasible["pareto"] = np.where(pareto_mask(feasible["eval_cost"], feasible["eval_co2"]),
                                  "Pareto-efficient", "Dominated")
    best = feasible.iloc[0]
    cheapest = feasible.loc[feasible["eval_cost"].idxmin()]
    greenest = feasible.loc[feasible["eval_co2"].idxmin()]
    unit = "lifecycle" if basis == "lifecycle" else "upfront"

    c = st.columns(4)
    kpi(c[0], "Best overall", best["material"], f"score {best['score']:.2f}")
    kpi(c[1], f"Cheapest ({unit})", cheapest["material"], f"{cheapest['eval_cost']:,.0f} ETB", "blue")
    kpi(c[2], f"Lowest carbon ({unit})", greenest["material"], f"{greenest['eval_co2']:,.0f} kg CO₂e")
    kpi(c[3], "Options in play", f"{len(feasible)} of {len(df)}", "after budget & lifespan limits", "amber")
    st.markdown(f"<div class='callout'>💡 {explain_best(feasible)}</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Ranking", "Trade-offs", "Compare", "Robustness", "Data & method"])

    # ---- Ranking
    with tabs[0]:
        section("Ranked options")
        show = feasible[["rank", "material", "score", "eval_cost", "eval_co2", "life_years", "builds",
                         "cost_per_year", "co2_per_year", "source"]]
        st.dataframe(show, hide_index=True, **stretch(), column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f"),
            "eval_cost": st.column_config.NumberColumn(f"Cost, {unit} (ETB)", format="%,.0f"),
            "eval_co2": st.column_config.NumberColumn(f"Carbon, {unit} (kg CO₂e)", format="%,.0f"),
            "life_years": st.column_config.NumberColumn("Life (yrs)", format="%.0f"),
            "builds": st.column_config.NumberColumn("Builds in period", format="%.1f"),
            "cost_per_year": st.column_config.NumberColumn("Cost / year", format="%,.0f"),
            "co2_per_year": st.column_config.NumberColumn("CO₂e / year", format="%,.0f")})
        fig = px.bar(feasible, x="score", y="material", orientation="h", color="score",
                     color_continuous_scale="Greens", title="Overall score (higher is better)")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        fig.update_xaxes(title="", range=[0, 1]); fig.update_yaxes(title="")
        st.plotly_chart(style_fig(fig, 90 + 42 * len(feasible)), **stretch())
        if not excluded.empty:
            with st.expander(f"Excluded by your constraints ({len(excluded)})"):
                ex = excluded[["material", "init_cost", "life_years", "within_budget", "meets_life"]]
                st.dataframe(ex, hide_index=True, **stretch())

    # ---- Trade-offs
    with tabs[1]:
        section(f"Cost vs carbon ({unit}); bubble size = lifespan")
        fig = px.scatter(feasible, x="eval_cost", y="eval_co2", size="life_years", color="material",
                         symbol="pareto", text="material", hover_data={"score": ":.2f"},
                         color_discrete_sequence=COLORS,
                         labels={"eval_cost": "Cost (ETB)", "eval_co2": "Embodied carbon (kg CO₂e)"})
        fig.update_traces(textposition="top center")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 460), **stretch())
        pf = feasible.loc[feasible["pareto"] == "Pareto-efficient", "material"].tolist()
        st.markdown(f"**Pareto-efficient options:** {', '.join(pf)}")
        st.caption("Bottom-left is cheap and low-carbon. A Pareto-efficient option cannot be beaten on "
                   "both cost and carbon by any other option.")
        section("How many times is each wall built?")
        fig = px.bar(feasible.sort_values("builds"), x="builds", y="material", orientation="h",
                     color_discrete_sequence=["#14A44D"],
                     title=f"Builds needed over {horizon} years ({'whole rebuilds' if mode == 'whole' else 'pro-rata'})")
        fig.update_xaxes(title=""); fig.update_yaxes(title="")
        st.plotly_chart(style_fig(fig, 90 + 38 * len(feasible)), **stretch())

    # ---- Compare
    with tabs[2]:
        names = feasible["material"].tolist()
        pick = st.multiselect("Choose up to 4 materials", names, default=names[:3], max_selections=4)
        if len(pick) < 2:
            st.info("Pick at least two materials to compare.")
        else:
            sub = feasible[feasible["material"].isin(pick)].copy()
            cost_s = minmax(feasible["eval_cost"], False)
            co2_s = minmax(feasible["eval_co2"], False)
            life_s = minmax(feasible["life_years"], True)
            axes = ["Low cost", "Low carbon", "Durability"]
            fig = go.Figure()
            for i, m in enumerate(pick):
                idx = feasible.index[feasible["material"] == m][0]
                vals = [cost_s[idx], co2_s[idx], life_s[idx]]
                fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=axes + axes[:1], fill="toself",
                                              name=m, line=dict(color=COLORS[i % len(COLORS)])))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
                              title="Strengths at a glance (outer edge = best)")
            a, b = st.columns([1, 1.1], gap="large")
            a.plotly_chart(style_fig(fig, 400), **stretch())
            with b:
                side = sub.set_index("material")[["eval_cost", "eval_co2", "life_years", "builds", "score"]].T
                side.index = [f"Cost, {unit} (ETB)", f"Carbon, {unit} (kg CO₂e)", "Life (yrs)",
                              "Builds in period", "Score"]
                st.dataframe(side.round(2), **stretch())
            section("Savings against a baseline")
            base_name = st.selectbox("Baseline material", names,
                                     index=int(feasible["eval_co2"].values.argmax()))
            base_row = feasible[feasible["material"] == base_name].iloc[0]
            sv = feasible[feasible["material"] != base_name].copy()
            sv["co2_saved"] = base_row["eval_co2"] - sv["eval_co2"]
            sv["co2_saved_pct"] = sv["co2_saved"] / base_row["eval_co2"] * 100 if base_row["eval_co2"] else 0.0
            sv["cost_diff"] = sv["eval_cost"] - base_row["eval_cost"]
            if sv.empty:
                st.caption("No other option to compare.")
            else:
                fig = px.bar(sv.sort_values("co2_saved"), x="co2_saved", y="material", orientation="h",
                             color="co2_saved", color_continuous_scale="RdYlGn",
                             title=f"Carbon saved vs {base_name} (kg CO₂e, {unit})")
                fig.update_layout(coloraxis_showscale=False)
                fig.update_xaxes(title=""); fig.update_yaxes(title="")
                st.plotly_chart(style_fig(fig, 90 + 38 * len(sv)), **stretch())
                st.dataframe(sv[["material", "co2_saved", "co2_saved_pct", "cost_diff"]], hide_index=True,
                             **stretch(), column_config={
                                 "co2_saved": st.column_config.NumberColumn("CO₂e saved (kg)", format="%,.0f"),
                                 "co2_saved_pct": st.column_config.NumberColumn("Saved (%)", format="%.0f"),
                                 "cost_diff": st.column_config.NumberColumn("Cost difference (ETB)", format="%,.0f")})

    # ---- Robustness
    with tabs[3]:
        st.write("How much do the results depend on your priorities and on the accuracy of the "
                 "input numbers? Each test runs 2,000 simulated scenarios.")
        base_df = feasible[["material", "cost_etb_m2", "co2_kg_m2", "life_years"]].reset_index(drop=True)
        left, right = st.columns(2, gap="large")
        with left:
            section("1. If your priorities change")
            sw = simulate(base_df, area, horizon, mode, basis, weights=None)
            fig = px.bar(sw, x="pct_first", y="material", orientation="h",
                         title="% of random priority mixes where it ranks #1",
                         color_discrete_sequence=["#0B7A3B"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            fig.update_xaxes(title="", range=[0, 100]); fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, 90 + 38 * len(sw)), **stretch())
            top = sw.iloc[0]
            if top["pct_first"] >= 60:
                st.success(f"{top['material']} wins in {top['pct_first']:.0f}% of mixes: a robust choice.")
            else:
                st.warning(f"No clear winner (best is {top['pct_first']:.0f}%). The answer depends on your values.")
        with right:
            section("2. If the input data is off")
            unc = st.slider("Possible error in every number (±%)", 0, 50, 20, 5)
            su = simulate(base_df, area, horizon, mode, basis,
                          weights=(w_cost, w_co2, w_life), uncertainty=unc / 100)
            fig = px.bar(su, x="pct_first", y="material", orientation="h",
                         title=f"% of runs ranked #1 with ±{unc}% data error",
                         color_discrete_sequence=["#F59E0B"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            fig.update_xaxes(title="", range=[0, 100]); fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, 90 + 38 * len(su)), **stretch())
            st.dataframe(su.round(1), hide_index=True, **stretch(), column_config={
                "pct_first": st.column_config.NumberColumn("% ranked #1", format="%.1f"),
                "avg_rank": st.column_config.NumberColumn("Average rank", format="%.2f")})
        if n_ph:
            st.caption("These tests describe the placeholder numbers, so they show how the method behaves, "
                       "not a real-world recommendation.")

    # ---- Data & method
    with tabs[4]:
        a, b = st.columns([1, 1], gap="large")
        with a:
            section("Method")
            st.markdown(
                "- **Functional unit:** 1 m² of finished wall.\n"
                "- **Constraints first:** options over the upfront budget, or below the minimum lifespan, "
                "are excluded.\n"
                "- **Scoring:** cost, carbon and lifespan are each scaled 0–1 (1 = best), then combined "
                "with your priority weights.\n"
                "- **Lifecycle view:** a short-lived wall must be rebuilt during the study period, which "
                "adds cost and carbon. Choose whole rebuilds or pro-rata.\n"
                "- **Robustness:** random priorities and ± data error show how stable the winner is.")
        with b:
            section("Limits")
            st.markdown(
                "- Embodied carbon covers **production only** (no transport, use-phase or end-of-life).\n"
                "- Maintenance, thermal performance and residual value are not modelled.\n"
                "- Results are only as good as the input data and its sources.")
        section("Data used")
        st.dataframe(data, hide_index=True, **stretch())
        inputs = {"Wall area": f"{area:,.0f} m²", "Upfront budget": f"{budget:,.0f} ETB",
                  "Minimum lifespan": f"{min_life} years", "Study period": f"{horizon} years",
                  "Compared on": basis_label, "Weights (cost/carbon/durability)": f"{w_cost}/{w_co2}/{w_life}"}
        d1, d2, d3 = st.columns(3)
        d1.download_button("Data template (CSV)", MATERIALS.to_csv(index=False), "materials_template.csv",
                           "text/csv", **stretch())
        d2.download_button("Ranking results (CSV)",
                           feasible.drop(columns=["pareto"]).to_csv(index=False), "ranking.csv",
                           "text/csv", **stretch())
        d3.download_button("Summary (Markdown)", build_summary(inputs, feasible, bool(n_ph)),
                           "summary.md", "text/markdown", **stretch())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sys.exit(0 if run_tests() else 1)
    main()
