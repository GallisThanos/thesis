import os
import tempfile
from typing import Literal, cast

import google.generativeai as genai
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from feature_extraction.config import Config
from feature_extraction.pipeline import (
    average_features,
    build_llm_phrasing_prompt,
    compare_features,
    extract_features_from_file,
    interpret,
    std_features,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audio Feature Analysis Tool",
    page_icon="🎛️",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">

<style>

/* ───────── GLOBAL ───────── */
html, body, .stApp {
    background-color: #0d0d0f;
    color: #c8cdd6;
    font-family: 'IBM Plex Mono', monospace;
}

#MainMenu, header, footer {
    visibility: hidden;
}

/* ───────── HERO ───────── */
.hero {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2028;
    margin-bottom: 2rem;
}

.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin: 0;
}

.hero h1 span { color: #00e5c0; }

.hero p {
    font-size: 0.75rem;
    color: #555a66;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ───────── SECTION LABELS ───────── */
.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00e5c0;
    margin: 2rem 0 0.6rem 0;
}

/* ───────── SIDEBAR ───────── */
section[data-testid="stSidebar"] {
    background: #111217 !important;
    border-right: 1px solid #1e2028;
}

section[data-testid="stSidebar"] * {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #c8cdd6 !important;
}

/* ───────── INPUTS ───────── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #16181f !important;
    border: 1px solid #272932 !important;
    border-radius: 4px !important;
    color: #c8cdd6 !important;
}

/* ───────── SLIDER (STABLE) ───────── */
[data-testid="stSlider"] div[role="slider"] {
    background: #00e5c0 !important;
}

/* ───────── FILE UPLOADER ───────── */
section[data-testid="stFileUploader"] {
    background: #111217 !important;
    border: 1px dashed #272932 !important;
    border-radius: 6px !important;
    padding: 0.5rem !important;
}

section[data-testid="stFileUploader"] button {
    background: #1a1c24 !important;
    border: 1px solid #272932 !important;
    color: #c8cdd6 !important;
}

/* ───────── BUTTON ───────── */
button[kind="primary"] {
    background: #00e5c0 !important;
    color: #0d0d0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em;
    border-radius: 4px !important;
}

button[kind="primary"]:hover {
    opacity: 0.85;
}

/* ───────── CHECKBOX ───────── */
[data-testid="stCheckbox"] label {
    color: #777d8a !important;
    font-size: 0.72rem;
}

/* ───────── METRIC CARDS ───────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 0.6rem;
}

.metric-card {
    background: #111217;
    border: 1px solid #1e2028;
    border-radius: 6px;
    padding: 0.9rem 1rem;
    position: relative;
}

.metric-card::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
}

.metric-card.higher::before { background: #00e5c0; }
.metric-card.lower::before { background: #ff4d6d; }
.metric-card.similar::before { background: #272932; }

.metric-name {
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: #555a66;
}

.metric-value {
    font-size: 0.95rem;
}

.metric-value.higher { color: #00e5c0; }
.metric-value.lower { color: #ff4d6d; }
.metric-value.similar { color: #555a66; }

/* ───────── ALERTS ───────── */
[data-testid="stAlert"] {
    background: #111217 !important;
    border: 1px solid #272932 !important;
    border-radius: 6px !important;
}

/* ───────── TEXT FIXES ───────── */
p, span, label {
    color: #d0d4dc !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# ── Gemini setup ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

# ── Chart theme ───────────────────────────────────────────────────────────────
DARK_BG = "#111217"
GRID_COLOR = "#1e2028"
TEXT_COLOR = "#555a66"
TEAL = "#00e5c0"
RED = "#ff4d6d"


def apply_dark_style():
    mpl.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_BG,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": "#ffffff",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "legend.facecolor": "#16181f",
            "legend.edgecolor": GRID_COLOR,
            "legend.fontsize": 8,
            "font.family": "monospace",
            "text.color": TEXT_COLOR,
        }
    )


def normalize_features(ref: dict, user: dict) -> tuple[dict, dict]:
    norm_ref, norm_user = {}, {}
    for key in ref:
        r, u = ref[key], user[key]
        max_val = max(abs(r), abs(u)) + 1e-9
        norm_ref[key] = r / max_val
        norm_user[key] = u / max_val
    return norm_ref, norm_user


def plot_bar(ref: dict, user: dict) -> plt.Figure:
    apply_dark_style()
    keys = list(ref.keys())
    ref_vals = [ref[k] for k in keys]
    user_vals = [user[k] for k in keys]
    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 0.2, ref_vals, width=0.38, color=TEAL, alpha=0.75, label="Reference", zorder=3)
    ax.bar(x + 0.2, user_vals, width=0.38, color=RED, alpha=0.75, label="Your Track", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=40, ha="right")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_title("Normalized Feature Comparison", pad=12)
    ax.legend(framealpha=0.4)
    fig.tight_layout()
    return fig


def plot_radar(ref: dict, user: dict) -> plt.Figure:
    apply_dark_style()
    keys = list(ref.keys())
    ref_vals = [ref[k] for k in keys] + [ref[keys[0]]]
    user_vals = [user[k] for k in keys] + [user[keys[0]]]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"polar": True})
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.plot(angles, ref_vals, color=TEAL, linewidth=1.8, label="Reference")
    ax.fill(angles, ref_vals, color=TEAL, alpha=0.08)
    ax.plot(angles, user_vals, color=RED, linewidth=1.8, label="Your Track")
    ax.fill(angles, user_vals, color=RED, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys, size=6.5, color=TEXT_COLOR)
    ax.tick_params(axis="y", colors=GRID_COLOR, labelsize=6)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.set_title("Radar View", pad=14, color="#ffffff", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), framealpha=0.3)
    fig.tight_layout()
    return fig


def plot_zscore_bar(zscores: dict, config: Config) -> plt.Figure:
    """Visualise z-scores with severity bands."""
    apply_dark_style()
    keys = list(zscores.keys())
    vals = [zscores[k] for k in keys]
    colors = []
    for v in vals:
        az = abs(v)
        if az < config.z_ignore:
            colors.append("#272932")
        elif az < config.z_mild:
            colors.append("#f0c060")
        elif az < config.z_moderate:
            colors.append(TEAL if v > 0 else RED)
        else:
            colors.append(TEAL if v > 0 else RED)

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, vals, color=colors, alpha=0.8, zorder=3)
    for thresh, label in [
        (config.z_ignore, "ignore"),
        (config.z_mild, "mild"),
        (config.z_moderate, "moderate"),
    ]:
        ax.axhline(thresh, color="#ffffff18", linewidth=0.8, linestyle="--")
        ax.axhline(-thresh, color="#ffffff18", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=40, ha="right")
    ax.axhline(0, color=GRID_COLOR, linewidth=1)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_title("Z-Score Deviation from Reference (positive = your track is higher)", pad=12)
    fig.tight_layout()
    return fig


def stream_llm_phrasing(interpretation: dict, ref_avg: dict, user_features: dict):
    prompt = build_llm_phrasing_prompt(interpretation, ref_avg, user_features)
    if gemini_model is None:
        raise RuntimeError("Set GOOGLE_API_KEY to enable Gemini phrasing.")
    response = gemini_model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <h1>Audio Feat<span>ure Analysis</span></h1>
    <p>Audio Feature Analysis &amp; Reference Matching</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="section-label">Engine Settings</div>', unsafe_allow_html=True)
    sr = st.selectbox("Sample Rate", [22050, 44100, 48000], index=1)
    avg_method = st.selectbox("Averaging Method", ["mean", "median", "trimmed"])
    momentary = st.slider("Momentary Window (sec)", 0.2, 1.0, 0.4)
    trim_db = st.slider("Silence Threshold (dB)", 20, 60, 40)

    st.markdown('<div class="section-label">Z-Score Thresholds</div>', unsafe_allow_html=True)
    z_ignore = st.slider("Ignore below |z|", 0.1, 1.0, 0.5, step=0.05)
    z_mild = st.slider("Mild above |z|", 0.5, 1.5, 1.0, step=0.05)
    z_moderate = st.slider("Moderate above |z|", 1.0, 3.0, 2.0, step=0.1)

    st.markdown('<div class="section-label">AI Suggestions</div>', unsafe_allow_html=True)
    enable_ai = st.checkbox(
        "Enable after analysis", value=False, disabled=not bool(GEMINI_API_KEY), key="enable_ai_cb"
    )
    st.caption(
        "When enabled, Gemini rephrases the rule-based output only — it adds no new decisions."
    )

config = Config(
    sr=sr,
    avg_method=cast(Literal["mean", "median", "trimmed"], avg_method),
    momentary_window=momentary,
    trim_db=trim_db,
    z_ignore=z_ignore,
    z_mild=z_mild,
    z_moderate=z_moderate,
)

col_ref, col_usr = st.columns(2)
with col_ref:
    st.markdown('<div class="section-label">Reference Tracks</div>', unsafe_allow_html=True)
    ref_files = st.file_uploader(
        "ref", type=["wav"], accept_multiple_files=True, label_visibility="collapsed"
    )
with col_usr:
    st.markdown('<div class="section-label">Your Track</div>', unsafe_allow_html=True)
    user_file = st.file_uploader("usr", type=["wav"], label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("▶  Analyze")

# ── Analysis ──────────────────────────────────────────────────────────────────
if run:
    if not ref_files or not user_file:
        st.warning("Please upload at least one reference track and your track.")
        st.stop()

    with st.spinner("Extracting features…"):
        ref_features_list = []
        for file in ref_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file.read())
                path = tmp.name
            ref_features_list.append(extract_features_from_file(path, config))
            os.remove(path)

        ref_avg = average_features(ref_features_list, config)
        ref_std = std_features(ref_features_list)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(user_file.read())
            user_path = tmp.name
        user_features = extract_features_from_file(user_path, config)
        os.remove(user_path)

    comparison = compare_features(ref_avg, user_features)
    interpretation = interpret(ref_avg, ref_std, user_features, len(ref_features_list), config)
    zscores = interpretation["zscores"]
    severities = interpretation["severities"]
    confidence = interpretation["confidence"]

    # ── Feature cards ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Feature Differences</div>', unsafe_allow_html=True)
    if len(ref_features_list) == 1:
        st.info(
            "Single reference uploaded — z-scores fall back to scale-normalised deltas. Add more references for statistically meaningful thresholds."
        )

    cards_html = '<div class="metric-grid">'
    for key, val in comparison.items():
        sev_key, direction = severities[key]
        conf = confidence.get(key, "medium")
        diff = val["difference"]

        if sev_key == "ignore":
            card_cls = badge_cls = "similar"
            label = "On target"
            badge = "≈"
        elif direction == "higher":
            card_cls = badge_cls = "higher"
            label = f"+{diff:.3f}"
            badge = sev_key
        else:
            card_cls = badge_cls = "lower"
            label = f"{diff:.3f}"
            badge = sev_key

        cards_html += f"""
        <div class="metric-card {card_cls}">
            <div class="metric-name">{key.replace("_", " ")}</div>
            <div class="metric-value {card_cls}">{label}
                <span class="metric-badge badge-{badge_cls}">{badge}</span>
            </div>
            <div class="confidence-{conf}">ref confidence: {conf}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Visualizations</div>', unsafe_allow_html=True)
    norm_ref, norm_user = normalize_features(ref_avg, user_features)

    chart_col, radar_col = st.columns([2, 1])
    with chart_col:
        st.pyplot(plot_bar(norm_ref, norm_user))
    with radar_col:
        st.pyplot(plot_radar(norm_ref, norm_user))

    st.markdown('<div class="section-label">Z-Score Analysis</div>', unsafe_allow_html=True)
    st.pyplot(plot_zscore_bar(zscores, config))

    # ── Rule-based interpretation ─────────────────────────────────────────
    st.markdown('<div class="section-label">Rule-Based Diagnosis</div>', unsafe_allow_html=True)
    st.caption("Deterministic — same inputs always produce the same output.")

    has_any_advice = any(interpretation["advice"][d] for d in interpretation["priority_order"])

    if not has_any_advice:
        st.success(
            "Your track is within target range on all measured features relative to the reference(s). No significant corrections indicated."
        )
    else:
        for domain in interpretation["priority_order"]:
            domain_advice = interpretation["advice"][domain]
            if not domain_advice:
                continue
            st.markdown(f"### {domain}")
            for point in domain_advice:
                st.markdown(f"- {point}")

        st.markdown("### Priority Order")
        for i, domain in enumerate(interpretation["priority_order"], 1):
            if interpretation["advice"][domain]:
                st.markdown(f"{i}. **{domain}**")

    # ── AI phrasing (optional) ────────────────────────────────────────────
    if enable_ai and has_any_advice:
        st.markdown(
            '<div class="section-label">AI — Rephrased Suggestions</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="ai-caption">Gemini rephrases the rule-based output above. '
            "No new decisions are introduced.</div>",
            unsafe_allow_html=True,
        )
        try:
            st.write_stream(stream_llm_phrasing(interpretation, ref_avg, user_features))
        except Exception as e:
            st.error(f"Could not generate suggestions: {e}")
