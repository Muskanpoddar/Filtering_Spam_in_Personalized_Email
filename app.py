"""Streamlit web app for the personalized email spam detection capstone demo.

Launch with:
    streamlit run app.py

Tabs:
  1. Project Overview     - What this is, datasets, architecture diagram
  2. Model Comparison     - Side-by-side metrics + interactive charts
  3. Live Classifier      - Paste any email -> scores from all 4 models
  4. Feedback Loop        - Personalization layer in action
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODELS_DIR, RESULTS_DIR
from src.features.structural import STRUCTURAL_FEATURE_NAMES, extract_batch, extract_structural
from src.models.baseline_nb import TfidfNBClassifier
from src.models.distilbert import DistilBertClassifier
from src.models.hybrid import HybridClassifier
from src.models.personalized import PersonalizedSpamDetector
from src.personalization.user_profile import UserProfile
from src.preprocessor import ParsedEmail

MODEL_DISPLAY = {
    "baseline": "TF-IDF + Naive Bayes",
    "distilbert": "DistilBERT",
    "hybrid": "DistilBERT + Structural",
    "personalized": "Personalized (Hybrid + Profile)",
}
MODEL_COLORS = {
    "baseline": "#3B82F6",
    "distilbert": "#10B981",
    "hybrid": "#F59E0B",
    "personalized": "#EF4444",
}

# Brand palette \u2014 used across hero, sidebar, badges, etc.
PALETTE = {
    "ink":       "#0F172A",   # near-black text
    "ink_soft":  "#334155",
    "muted":     "#64748B",
    "line":      "#E2E8F0",
    "page":      "#F8FAFC",   # page background
    "card":      "#FFFFFF",
    "brand":     "#4F46E5",   # indigo
    "brand_alt": "#8B5CF6",   # violet
    "brand_dk":  "#1E1B4B",
    "accent":    "#06B6D4",   # cyan
    "good":      "#10B981",
    "warn":      "#F59E0B",
    "bad":       "#EF4444",
    "spam_bg":   "#FEF2F2",
    "ham_bg":    "#ECFDF5",
}

st.set_page_config(
    page_title="SpamGuard \u00b7 Personalized Spam Detection",
    page_icon="\U0001F6E1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Global styles --------------------------------------------------------
# Single CSS block injected once per page. Streamlit re-runs the script on
# each interaction but this only affects DOM, not state.
st.markdown(
    """
    <style>
      /* Inter font (CDN) with system fallback */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"], .stMarkdown, .stTextInput input,
      .stTextArea textarea, .stButton button, .stRadio, .stMetric {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif !important;
      }

      /* Page background + text */
      .stApp { background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%); }
      .main .block-container { padding-top: 2rem; max-width: 1280px; }

      /* Hide Streamlit's default chrome (keep the top header so the
         running status and theme toggle remain accessible) */
      #MainMenu, footer, .stDeployButton { visibility: hidden; height: 0; }

      /* ---- Hero / titles ---- */
      .sg-hero {
        background: radial-gradient(120% 120% at 0% 0%, #4F46E5 0%, #8B5CF6 45%, #0E1B4B 100%);
        color: #fff; padding: 1.75rem 2rem; border-radius: 18px;
        box-shadow: 0 18px 40px -18px rgba(79,70,229,.55);
        margin-bottom: 1.25rem; position: relative; overflow: hidden;
      }
      .sg-hero::before {
        content: ''; position: absolute; inset: 0;
        background: radial-gradient(60% 80% at 100% 0%, rgba(6,182,212,.35), transparent 60%),
                    radial-gradient(50% 70% at 0% 100%, rgba(236,72,153,.30), transparent 60%);
        pointer-events: none;
      }
      .sg-hero h1 { color: #fff !important; font-weight: 800; margin: 0; font-size: 1.85rem; letter-spacing: -.01em; }
      .sg-hero p  { color: rgba(255,255,255,.85); margin: .35rem 0 0; font-size: 1.02rem; }
      .sg-hero .pill {
        display: inline-block; padding: .2rem .65rem; border-radius: 999px;
        background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25);
        font-size: .78rem; margin-right: .4rem; backdrop-filter: blur(4px);
      }

      /* ---- Sidebar ---- */
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E1B4B 0%, #0F172A 100%);
        color: #E2E8F0;
      }
      [data-testid="stSidebar"] .stMarkdown h1,
      [data-testid="stSidebar"] .stMarkdown h2,
      [data-testid="stSidebar"] .stMarkdown h3 { color: #fff; }
      [data-testid="stSidebar"] label { color: #CBD5E1 !important; font-weight: 500; }
      [data-testid="stSidebar"] .stRadio > label { padding: .35rem 0; }
      [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.08); }

      .sg-side-brand {
        font-weight: 800; font-size: 1.25rem; color: #fff; letter-spacing: -.01em;
        margin-bottom: .25rem; display: flex; align-items: center; gap: .5rem;
      }
      .sg-side-brand .dot {
        width: .6rem; height: .6rem; border-radius: 50%;
        background: linear-gradient(135deg, #06B6D4, #8B5CF6);
        box-shadow: 0 0 12px rgba(139,92,246,.6);
      }

      /* ---- Section headers ---- */
      h2, h3 { letter-spacing: -.01em; }
      .stMarkdown h2 { font-size: 1.55rem; font-weight: 700; color: #0F172A; }
      .stMarkdown h3 { font-size: 1.18rem; font-weight: 700; color: #1E293B; }

      /* ---- KPI tiles (used on Overview) ---- */
      .sg-kpi {
        background: #fff; padding: 1.05rem 1.15rem; border-radius: 14px;
        border: 1px solid #E2E8F0; box-shadow: 0 6px 18px -12px rgba(15,23,42,.18);
        height: 100%;
      }
      .sg-kpi .label { font-size: .78rem; color: #64748B; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
      .sg-kpi .value { font-size: 1.55rem; font-weight: 800; color: #0F172A; margin-top: .25rem; }
      .sg-kpi .sub   { font-size: .82rem; color: #475569; margin-top: .15rem; }
      .sg-kpi.brand { background: linear-gradient(135deg, #4F46E5, #8B5CF6); color: #fff; border: none; }
      .sg-kpi.brand .label, .sg-kpi.brand .sub { color: rgba(255,255,255,.85); }
      .sg-kpi.brand .value { color: #fff; }

      /* ---- Cards ---- */
      .sg-card {
        background: #fff; border: 1px solid #E2E8F0; border-radius: 14px;
        padding: 1.1rem 1.2rem; box-shadow: 0 6px 18px -12px rgba(15,23,42,.18);
      }
      .sg-card h4 { margin: 0 0 .25rem; font-weight: 700; color: #0F172A; }
      .sg-card .meta { color: #64748B; font-size: .85rem; }

      /* ---- Verdict tiles (Live classifier) ---- */
      .sg-verdict {
        border-radius: 14px; padding: 1rem 1.1rem; height: 100%;
        border: 1px solid; transition: transform .18s ease;
      }
      .sg-verdict:hover { transform: translateY(-2px); }
      .sg-verdict .name { font-size: .82rem; font-weight: 600; opacity: .9; }
      .sg-verdict .label { font-size: 1.7rem; font-weight: 800; letter-spacing: -.01em; }
      .sg-verdict .score { font-size: .92rem; margin-top: .2rem; }
      .sg-verdict .breakdown { font-size: .75rem; opacity: .75; margin-top: .35rem; }
      .sg-verdict.spam { background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%); border-color: #FCA5A5; color: #991B1B; }
      .sg-verdict.ham  { background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%); border-color: #6EE7B7; color: #065F46; }

      /* ---- Buttons ---- */
      .stButton button {
        background: linear-gradient(135deg, #4F46E5, #8B5CF6) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important;
        font-weight: 600 !important; padding: .55rem .9rem !important;
        box-shadow: 0 8px 22px -14px rgba(79,70,229,.6) !important;
        transition: transform .15s ease, box-shadow .15s ease !important;
      }
      .stButton button:hover { transform: translateY(-1px); box-shadow: 0 12px 26px -14px rgba(79,70,229,.75) !important; }
      .stButton button:active { transform: translateY(0); }
      .stButton button:focus { outline: 2px solid rgba(79,70,229,.35) !important; outline-offset: 2px; }
      /* Ghost look on sample / secondary buttons */
      .stButton button.sg-ghost {
        background: #fff !important; color: #4F46E5 !important;
        border: 1px solid #C7D2FE !important; box-shadow: none !important;
      }
      /* Danger / success variants via key suffix */
      .stButton button[data-testid="baseButton-secondary"] {
        background: #fff !important; color: #0F172A !important;
        border: 1px solid #E2E8F0 !important; box-shadow: none !important;
      }

      /* ---- Input styling ---- */
      .stTextInput input, .stTextArea textarea {
        background: #fff !important; border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important; color: #0F172A !important;
      }
      .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #818CF8 !important; box-shadow: 0 0 0 3px rgba(129,140,248,.18) !important;
      }

      /* ---- DataFrames / dataframe border-radius ---- */
      .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid #E2E8F0; }

      /* ---- Pills / chips ---- */
      .sg-chip {
        display: inline-block; padding: .22rem .65rem; border-radius: 999px;
        font-size: .78rem; font-weight: 600; line-height: 1.4;
        background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE;
      }
      .sg-chip.spam { background: #FEF2F2; color: #991B1B; border-color: #FECACA; }
      .sg-chip.ham  { background: #ECFDF5; color: #065F46; border-color: #BBF7D0; }

      /* ---- Architecture flow ---- */
      .sg-flow {
        background: #fff; border: 1px solid #E2E8F0; border-radius: 16px;
        padding: 1.2rem 1.4rem; box-shadow: 0 6px 18px -12px rgba(15,23,42,.18);
      }
      .sg-flow .node {
        flex: 1; min-width: 130px; text-align: center;
        background: #0F172A; color: #fff; padding: .7rem .6rem; border-radius: 10px;
        border: 1px solid rgba(255,255,255,.06);
      }
      .sg-flow .node.alt { background: linear-gradient(135deg, #1D4ED8, #3B82F6); }
      .sg-flow .node.brand { background: linear-gradient(135deg, #4F46E5, #8B5CF6); }
      .sg-flow .node.warn { background: linear-gradient(135deg, #B45309, #F59E0B); }
      .sg-flow .node.good { background: linear-gradient(135deg, #065F46, #10B981); }
      .sg-flow .node.danger { background: linear-gradient(135deg, #9F1239, #EF4444); }
      .sg-flow .arrow { font-size: 1.4rem; color: #94A3B8; padding: 0 .25rem; }
      .sg-flow .small { font-size: .72rem; opacity: .8; }
      .sg-flow .b { font-weight: 700; font-size: .9rem; }
      .sg-flow .legend { color: #475569; font-size: .85rem; margin-top: .8rem; }

      /* ---- Misc ---- */
      .sg-muted { color: #64748B; }

      /* ---- Radio buttons (default label color is too light on the gradient) ---- */
      [data-testid="stRadio"] label, [data-testid="stRadio"] p,
      [data-testid="stRadio"] span, [data-testid="stRadio"] div[role="radiogroup"] label {
        color: #0F172A !important;
        font-weight: 500 !important;
      }
      [data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: .82rem !important;
        text-transform: uppercase;
        letter-spacing: .04em;
      }

      /* ---- Sidebar nav links (page-link list) ---- */
      [data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] a span,
      [data-testid="stSidebarNav"] a p, [data-testid="stSidebarNav"] li span,
      [data-testid="stSidebarNav"] li {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        font-size: .95rem !important;
      }
      [data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNav"] a:hover span {
        color: #FFFFFF !important;
        background: rgba(255,255,255,.10) !important;
      }
      [data-testid="stSidebarNav"] a[aria-current="page"],
      [data-testid="stSidebarNav"] a[aria-current="page"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
      }

      /* ---- Sidebar header & subheader text ---- */
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] li, [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
      }
      .sg-foot {
        margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #E2E8F0;
        color: #64748B; font-size: .82rem; text-align: center;
      }
      .stCaption, .stMarkdown small { color: #64748B !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading baseline model...")
def load_baseline():
    path = MODELS_DIR / "baseline_nb"
    if not (path / "nb.joblib").exists():
        return None
    return TfidfNBClassifier.load(str(path))


@st.cache_resource(show_spinner="Loading DistilBERT model...")
def load_distilbert():
    path = MODELS_DIR / "distilbert"
    if not (path / "config.json").exists():
        return None
    return DistilBertClassifier.load(str(path))


@st.cache_resource(show_spinner="Loading hybrid model...")
def load_hybrid():
    path = MODELS_DIR / "hybrid"
    if not (path / "hybrid.pt").exists():
        return None
    return HybridClassifier.load(str(path))


@st.cache_resource(show_spinner="Loading personalized detector...")
def load_personalized():
    path = MODELS_DIR / "personalized"
    if not (path / "hybrid.pt").exists():
        return None
    return PersonalizedSpamDetector.load(str(path))


def get_models() -> dict:
    return {
        "baseline": load_baseline(),
        "distilbert": load_distilbert(),
        "hybrid": load_hybrid(),
        "personalized": load_personalized(),
    }


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    """Load per-dataset comparison metrics.

    Each file is shaped `{dataset_name: {model_name: metrics_dict}}`; we
    unwrap the outer key so callers receive `{dataset_name: {model: ...}}`.
    """
    out: dict = {}
    for name, fname in [
        ("spamassassin", "comparison_metrics.json"),
        ("enron_spam", "enron_validation_metrics.json"),
    ]:
        p = RESULTS_DIR / fname
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        # Accept both wrapper and flat shapes.
        if name in payload and isinstance(payload[name], dict):
            payload = payload[name]
        elif len(payload) == 1:
            payload = next(iter(payload.values()))
        out[name] = payload
    return out


def score_email(parsed: ParsedEmail, models: dict) -> dict[str, dict]:
    """Score an email with all available models, return {name: {proba, decision}}."""
    emails = [parsed]
    struct = extract_batch(emails)
    out = {}
    if models["baseline"] is not None:
        p = float(models["baseline"].predict_proba(emails)[0])
        out["baseline"] = {"proba": p, "decision": "SPAM" if p >= 0.5 else "HAM"}
    if models["distilbert"] is not None:
        p = float(models["distilbert"].predict_proba(emails)[0])
        out["distilbert"] = {"proba": p, "decision": "SPAM" if p >= 0.5 else "HAM"}
    if models["hybrid"] is not None:
        p = float(models["hybrid"].predict_proba(emails, struct)[0])
        out["hybrid"] = {"proba": p, "decision": "SPAM" if p >= 0.5 else "HAM"}
    if models["personalized"] is not None:
        det = models["personalized"]
        base = float(det.predict_proba_raw(emails, struct)[0])
        bias = float(det.profile.personalization_bias(parsed.sender or "", parsed.body or ""))
        blended = float(det.predict_proba(emails, struct)[0])
        out["personalized"] = {
            "proba": blended,
            "base_proba": base,
            "bias": bias,
            "decision": "SPAM" if blended >= det.threshold else "HAM",
        }
    return out


def _kpi(label: str, value: str, sub: str = "", brand: bool = False) -> str:
    cls = "sg-kpi brand" if brand else "sg-kpi"
    return (
        f"<div class='{cls}'>"
        f"<div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>"
        + (f"<div class='sub'>{sub}</div>" if sub else "")
        + "</div>"
    )


def render_overview():
    st.markdown(
        """
        <div class="sg-hero">
          <div>
            <span class="pill">Capstone</span>
            <span class="pill">Hybrid DistilBERT</span>
            <span class="pill">User Personalization</span>
          </div>
          <h1>SpamGuard \u2014 Personalized Spam Detection</h1>
          <p>Context-aware email triage that fuses DistilBERT semantics, 20 structural signals,
             and a per-user feedback loop \u2014 dropping false positives on the mail you actually care about.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_payload = load_metrics()
    sa = metrics_payload.get("spamassassin") or {}
    personalized_metrics = sa.get("personalized") or {}
    fpr_text = f"{personalized_metrics['false_positive_rate']:.3f}" if personalized_metrics else "\u2014"
    f1_text = f"{personalized_metrics['f1']:.3f}" if personalized_metrics else "\u2014"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_kpi("Primary dataset", "SpamAssassin", "7,953 emails \u00b7 public corpus"), unsafe_allow_html=True)
    c2.markdown(_kpi("Validation", "Enron-Spam", "33,716 emails \u00b7 out-of-distribution"), unsafe_allow_html=True)
    c3.markdown(_kpi("Models compared", "4", "TF-IDF, BERT, hybrid, personalized"), unsafe_allow_html=True)
    c4.markdown(_kpi("Personalized F1", f1_text, f"False-positive rate {fpr_text}", brand=True), unsafe_allow_html=True)

    st.markdown("### What this project does")
    st.markdown(
        """
        <div class="sg-card">
          <p style="margin:0 0 .5rem;color:#334155;">
            <b>SpamGuard</b> classifies incoming mail as <b>Spam</b> or <b>Ham</b> using three
            cooperating layers:
          </p>
          <ol style="margin:0 0 0 1.1rem;color:#334155;">
            <li><b>DistilBERT</b> extracts contextual semantic embeddings of subject + body.</li>
            <li><b>20 structural features</b> capture signals DistilBERT misses: sender, URLs,
                HTML ratio, attachments, length, capitalization, suspicious-subject pattern,
                free-mail domain, reply-to mismatch.</li>
            <li><b>Personalization layer</b> maintains a per-user profile (trusted/blocked
                senders, sender history, keyword preferences) and <i>biases</i> the prediction.
                Feedback updates the profile in-place.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Architecture")
    st.markdown(
        """
        <div class="sg-flow">
          <div style="display:flex;flex-wrap:wrap;gap:.6rem;align-items:center;">
            <div class="node"><div class="small">INPUT</div><div class="b">Raw email</div></div>
            <div class="arrow">\u2192</div>
            <div class="node"><div class="small">PREPROCESSOR</div><div class="b">RFC822 \u00b7 HTML strip</div></div>
            <div class="arrow">\u2192</div>
            <div class="node alt"><div class="small">SEMANTIC</div><div class="b">DistilBERT</div></div>
            <div class="arrow">+</div>
            <div class="node warn"><div class="small">STRUCTURAL</div><div class="b">20 hand-crafted features</div></div>
            <div class="arrow">\u2192</div>
            <div class="node brand"><div class="small">FUSION MLP</div><div class="b">Hybrid score</div></div>
            <div class="arrow">\u2295</div>
            <div class="node danger"><div class="small">PERSONALIZATION</div><div class="b">User profile bias</div></div>
            <div class="arrow">\u2192</div>
            <div class="node good"><div class="small">OUTPUT</div><div class="b">Spam / Ham</div></div>
          </div>
          <div class="legend">
            <b>Hybrid score</b> = MLP([DistilBERT[CLS] ; structural]).
            <b>Personalized</b> = (1 \u2212 \u03b1) \u00b7 hybrid + \u03b1 \u00b7 bias,
            where \u03b1 is the personalization strength and bias comes from trusted/blocked senders,
            sender reputation, and keyword history.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Models compared")
    comp = {
        "Model": list(MODEL_DISPLAY.values()),
        "Strength": [
            "Fast, transparent, bag-of-words baseline",
            "Contextual semantic understanding",
            "Adds 20 hand-crafted structural features",
            "Learns from each user's feedback history",
        ],
        "Weakness": [
            "Ignores context, can't generalize across vocab",
            "Misses URL/attachment/sender signals",
            "Heavier, more expensive to train",
            "Requires labelled feedback from each user",
        ],
        "Use when": [
            "You need an interpretable, low-cost first pass",
            "Pure semantic classification suffices",
            "You have a labelled dataset with sender signals",
            "You want long-running accuracy per recipient",
        ],
    }
    st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)


def metric_chip(value, color):
    return f"<span style='background:{color};color:white;padding:.15rem .55rem;border-radius:8px;font-weight:600;'>{value}</span>"


def render_comparison():
    st.markdown("## \U0001F4CA Model Comparison")

    metrics = load_metrics()
    if not metrics:
        st.warning("No comparison metrics found. Run `python -m src.compare` first.")
        return

    dataset_choice = st.radio(
        "Evaluation set",
        list(metrics.keys()),
        format_func=lambda k: "SpamAssassin (primary)" if k == "spamassassin" else "Enron-Spam (secondary validation)",
        horizontal=True,
    )
    data = metrics[dataset_choice]

    df = pd.DataFrame([
        {
            "Model": MODEL_DISPLAY.get(k, k),
            "Color": MODEL_COLORS.get(k, "#64748B"),
            "Accuracy": v["accuracy"],
            "Precision": v["precision"],
            "Recall": v["recall"],
            "F1": v["f1"],
            "FPR": v["false_positive_rate"],
            "AUC": v["auc"],
        }
        for k, v in data.items()
    ]).sort_values("F1", ascending=False).reset_index(drop=True)

    # KPI row — best metric on this dataset
    p1, p2, p3, p4 = st.columns(4)
    best_f1 = df.iloc[df["F1"].idxmax()]
    best_acc = df.iloc[df["Accuracy"].idxmax()]
    lowest_fpr = df.iloc[df["FPR"].idxmin()]
    best_auc = df.iloc[df["AUC"].idxmax()]
    p1.markdown(_kpi("Best F1", f"{best_f1['F1']:.3f}", best_f1["Model"], brand=True), unsafe_allow_html=True)
    p2.markdown(_kpi("Highest Accuracy", f"{best_acc['Accuracy']:.3f}", best_acc["Model"]), unsafe_allow_html=True)
    p3.markdown(_kpi("Lowest FPR", f"{lowest_fpr['FPR']:.3f}", lowest_fpr["Model"]), unsafe_allow_html=True)
    p4.markdown(_kpi("Best AUC", f"{best_auc['AUC']:.3f}", best_auc["Model"]), unsafe_allow_html=True)

    st.markdown("### Metrics table")
    styled = (
        df.style
        .format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "FPR": "{:.4f}",
            "AUC": "{:.4f}",
        })
        .background_gradient(cmap="RdYlGn", subset=["Accuracy", "Precision", "Recall", "F1", "AUC"])
        .background_gradient(cmap="RdYlGn_r", subset=["FPR"])
        .hide(subset=["Color"], axis="columns")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("### Per-model scorecards")
    model_cols = st.columns(len(df))
    for col, row in zip(model_cols, df.itertuples()):
        # Horizontal mini-bar for F1 inside the card.
        f1_pct = max(0.0, min(1.0, float(row.F1)))
        col.markdown(
            f"<div class='sg-card'>"
            f"<div style='display:flex;align-items:center;gap:.55rem;'>"
            f"<span style='width:.7rem;height:.7rem;border-radius:50%;background:{row.Color};display:inline-block;box-shadow:0 0 0 3px {row.Color}22;'></span>"
            f"<h4 style='margin:0;font-size:.95rem;'>{row.Model}</h4>"
            f"</div>"
            f"<div style='margin-top:.65rem;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:.78rem;color:#64748B;margin-bottom:.25rem;'>"
            f"<span>F1</span><span><b style='color:#0F172A;'>{row.F1:.3f}</b></span></div>"
            f"<div style='height:6px;background:#F1F5F9;border-radius:99px;overflow:hidden;'>"
            f"<div style='height:100%;width:{f1_pct*100:.1f}%;background:linear-gradient(90deg,{row.Color},{row.Color}cc);border-radius:99px;'></div>"
            f"</div></div>"
            f"<div style='margin-top:.7rem;display:grid;grid-template-columns:1fr 1fr;gap:.35rem .8rem;font-size:.83rem;color:#334155;'>"
            f"<div><span style='color:#94A3B8;'>Accuracy</span> <b style='float:right;'>{row.Accuracy:.3f}</b></div>"
            f"<div><span style='color:#94A3B8;'>Precision</span> <b style='float:right;'>{row.Precision:.3f}</b></div>"
            f"<div><span style='color:#94A3B8;'>Recall</span> <b style='float:right;'>{row.Recall:.3f}</b></div>"
            f"<div><span style='color:#94A3B8;'>FPR</span> <b style='float:right;'>{row.FPR:.3f}</b></div>"
            f"<div style='grid-column:1/-1;border-top:1px solid #E2E8F0;padding-top:.4rem;margin-top:.2rem;'>"
            f"<span style='color:#94A3B8;'>AUC</span> <b style='float:right;'>{row.AUC:.3f}</b></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Head-to-head: Recall vs False-Positive Rate")
    st.caption(
        "A good spam filter sits in the top-left of this chart "
        "&mdash; <b>high recall</b> catches spam, <b>low FPR</b> protects legitimate mail."
    )
    # Grouped bar: each model has two bars side-by-side, Recall (green) + FPR (red).
    # FPR is shown as 1 - FPR so it points the same direction as Recall (higher = better).
    h2h = df.copy()
    h2h["1 - FPR"] = 1.0 - h2h["FPR"]
    melted = h2h.melt(
        id_vars=["Model", "Color"],
        value_vars=["Recall", "1 - FPR"],
        var_name="Metric",
        value_name="Score",
    )
    # Friendlier label for the legend
    melted["Metric"] = melted["Metric"].replace({"1 - FPR": "1 − FPR (clean-mail recall)"})
    bar_colors = {"Recall": "#10B981", "1 − FPR (clean-mail recall)": "#F59E0B"}
    fig2 = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        color_discrete_map=bar_colors,
        barmode="group",
        text="Score",
        range_y=(0, 1.08),
    )
    fig2.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        textfont=dict(color="#0F172A", size=13,
                      family="Inter, sans-serif"),
        marker=dict(line=dict(width=0), opacity=1.0),
        hovertemplate="<b>%{x}</b><br>%{y:.3f}<extra></extra>",
    )
    # Force the marker colors per trace — px color_discrete_map can drop on
    # transparent streamlit canvases, leaving the bars pale. Pin them solid.
    for trace, color in zip(fig2.data, ["#10B981", "#F59E0B"]):
        trace.marker.color = color
        trace.marker.opacity = 1.0
        trace.marker.line = dict(width=0)
    fig2.update_layout(
        height=440,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, l=20, r=20, b=70),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color="#0F172A", family="Inter, sans-serif"),
            title=None,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(15,23,42,.06)",
            tickfont=dict(color="#64748B", family="Inter, sans-serif"),
            title=dict(text="<b>Score</b> (higher is better)", font=dict(color="#334155", family="Inter, sans-serif")),
            range=[0, 1.05],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(family="Inter, sans-serif", size=12, color="#334155"),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Per-metric breakdown")
    csel, _ = st.columns([1, 3])
    metric_to_plot = csel.selectbox(
        "Metric",
        ["F1", "Accuracy", "Precision", "Recall", "FPR", "AUC"],
        index=0,
        label_visibility="collapsed",
    )
    color_map = {row["Model"]: row["Color"] for _, row in df.iterrows()}
    fig = px.bar(
        df,
        x="Model",
        y=metric_to_plot,
        color="Model",
        color_discrete_map=color_map,
        text=metric_to_plot,
        range_y=(0, 1.05) if metric_to_plot != "FPR" else (0, max(0.6, df["FPR"].max() * 1.2)),
    )
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        textfont=dict(color="#0F172A", size=13,
                      family="Inter, sans-serif"),
        marker=dict(line=dict(width=0), opacity=1.0),
        hovertemplate=f"<b>%{{x}}</b><br>{metric_to_plot}=%{{y:.3f}}<extra></extra>",
    )
    # Pin each bar to its model color at full opacity (otherwise the
    # transparent canvas can wash the color out).
    for trace, color in zip(fig.data, [c for c in color_map.values()]):
        trace.marker.color = color
        trace.marker.opacity = 1.0
        trace.marker.line = dict(width=0)
    fig.update_layout(
        height=420,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title=metric_to_plot,
        margin=dict(t=20, l=20, r=20, b=60),
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#0F172A", family="Inter, sans-serif")),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(15,23,42,.06)",
            tickfont=dict(color="#64748B", family="Inter, sans-serif"),
        ),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Confusion matrices")
    cols = st.columns(min(4, len(data)))
    for col, (k, v) in zip(cols, data.items()):
        cm = v["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            x=["Pred HAM", "Pred SPAM"],
            y=["True HAM", "True SPAM"],
            color_continuous_scale=[[0, "#FFFFFF"], [1, "#4F46E5"]],
            text_auto=True,
            aspect="equal",
            zmin=0,
        )
        fig_cm.update_traces(textfont=dict(size=18, color="#0F172A", family="Inter, sans-serif"))
        fig_cm.update_layout(
            title=dict(
                text=f"<b>{MODEL_DISPLAY.get(k, k)}</b>",
                x=0.05, xanchor="left",
                font=dict(size=13, color="#0F172A", family="Inter, sans-serif"),
            ),
            height=300,
            margin=dict(t=50, l=10, r=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#334155", size=11),
            coloraxis_showscale=False,
            xaxis=dict(
                tickfont=dict(color="#0F172A", size=11,
                              family="Inter, sans-serif"),
                automargin=True,
            ),
            yaxis=dict(
                tickfont=dict(color="#0F172A", size=11,
                              family="Inter, sans-serif"),
                automargin=True,
            ),
        )
        col.plotly_chart(
            fig_cm, use_container_width=True,
            config={"displayModeBar": False, "displaylogo": False,
                    "responsive": True},
        )


def _classify_now():
    """Read the current classifier form values, run the 4 models, and stash
    the result so the rest of `render_classifier` can render it."""
    parsed = ParsedEmail()
    parsed.sender = st.session_state.get("cls_sender", "")
    parsed.subject = st.session_state.get("cls_subject", "")
    body = st.session_state.get("cls_body", "")
    parsed.body = body
    parsed.raw_body = body
    parsed.char_count = len(body)
    parsed.word_count = len(body.split())
    parsed.urls = []

    models = get_models()
    if all(v is None for v in models.values()):
        st.session_state["cls_result"] = None
        st.session_state["cls_result_error"] = "No trained models found in `models/`. Run `python -m src.train` first."
        return

    try:
        st.session_state["cls_result"] = score_email(parsed, models)
        st.session_state["cls_result_error"] = None
        st.session_state["cls_last_sender"] = parsed.sender
        st.session_state["cls_last_subject"] = parsed.subject
        st.session_state["cls_last_body"] = body
        # Invalidate any cached explanations — they belong to the old email.
        st.session_state.pop("cls_explanations", None)
    except Exception as exc:  # noqa: BLE001 — surface any model failure to the UI
        st.session_state["cls_result"] = None
        st.session_state["cls_result_error"] = f"Scoring failed: {exc}"


# ---------------------------------------------------------------------------
# Token / feature-level explainability
# ---------------------------------------------------------------------------
def _explain_baseline(nb_model, body: str, subject: str, top_k: int = 12):
    """Per-token contribution to the spam score from TF-IDF + Naive Bayes.

    contribution = tfidf(token) * (log P(token|spam) - log P(token|ham))
    """
    import re

    if nb_model is None or not getattr(nb_model.featurizer, "fitted", False):
        return None
    vec = nb_model.featurizer.vectorizer
    log_prob = nb_model.model.feature_log_prob_  # shape (2, V)
    spam_minus_ham = log_prob[1] - log_prob[0]   # shape (V,)

    parsed = ParsedEmail(subject=subject, body=body, raw_body=body)
    X = nb_model.featurizer.transform([parsed])
    if X.shape[1] == 0:
        return None
    row = X.toarray()[0]
    nonzero_idx = np.where(row > 0)[0]
    if len(nonzero_idx) == 0:
        return None

    id_to_token = {i: t for t, i in vec.vocabulary_.items()}
    contribs = []
    for i in nonzero_idx:
        token = id_to_token.get(i, "")
        if not token or " " in token and token.startswith(" "):
            token_disp = token.strip()
        else:
            token_disp = token
        tfidf = float(row[i])
        log_odds = float(spam_minus_ham[i])
        contribs.append((token_disp, tfidf, log_odds, tfidf * log_odds))
    contribs.sort(key=lambda r: abs(r[3]), reverse=True)
    return contribs[:top_k]


def _explain_text_saliency(db_model, body: str, subject: str, top_k: int = 12):
    """Gradient-based saliency over the DistilBERT embedding layer.

    Single forward + backward pass; sums subword importance back to the
    original word boundaries so the UI can show word-level bars.
    """
    import torch

    if db_model is None:
        return None
    parsed = ParsedEmail(subject=subject, body=body, raw_body=body)
    text = f"{parsed.subject or ''} \n {parsed.body or ''}"[:8000]
    if not text.strip():
        return None
    db_model._ensure_loaded()
    enc = db_model.tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=db_model.max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(db_model.device)
    attn = enc["attention_mask"].to(db_model.device)

    db_model.model.zero_grad(set_to_none=True)
    db_model.model.eval()
    emb_layer = db_model.model.get_input_embeddings()
    with torch.enable_grad():
        embeddings = emb_layer(input_ids).clone().detach().requires_grad_(True)
        out = db_model.model(
            inputs_embeds=embeddings, attention_mask=attn
        )
        p_spam = torch.softmax(out.logits, dim=-1)[0, 1]
        p_spam.backward()
    if embeddings.grad is None:
        return None
    # Signed gradient × input: positive = pushes the spam logit up,
    # negative = pushes it down. Magnitude = bar length.
    saliency = (embeddings.grad[0] * embeddings.detach()[0]).sum(dim=-1)
    saliency = saliency.detach().cpu().numpy()

    tokens = db_model.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    # Skip [CLS], [SEP], [PAD] — they're not user-visible.
    skip_ids = {
        db_model.tokenizer.cls_token_id,
        db_model.tokenizer.sep_token_id,
        db_model.tokenizer.pad_token_id,
    }
    word_sal: dict[str, float] = {}
    word_order: list[str] = []
    for tok, score, tid in zip(tokens, saliency, input_ids[0].tolist()):
        if tid in skip_ids:
            continue
        # DistilBERT uses ##subword marker.
        clean = tok.replace("##", "")
        if tok.startswith("##"):
            if word_order:
                word_order[-1] += clean
                word_sal[word_order[-1]] = word_sal.get(word_order[-1], 0.0) + float(score)
                continue
        word_order.append(clean)
        word_sal[clean] = float(score)
    if not word_sal:
        return None
    # Keep the top-k by absolute influence. The caller decides how to colour.
    items = sorted(word_sal.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return items  # list of (word, signed_saliency)


def _explain_structural_saliency(hybrid_model, parsed: ParsedEmail,
                                 struct_vec: np.ndarray):
    """Gradient-based saliency over the 20 standardized structural features.

    `struct_vec` must be the un-scaled 20-dim feature vector; the scaler is
    applied inside the model and gradients are taken with respect to the
    scaled input so the numbers are comparable across features.
    """
    import torch

    if hybrid_model is None or hybrid_model.scaler is None:
        return None
    parsed_texts = [parsed]
    X_scaled = hybrid_model.scaler.transform(struct_vec)
    if X_scaled.shape[0] == 0:
        return None
    struct_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
    # We need an actual forward pass to get gradients. Reuse the model's
    # forward, but supply the structural tensor directly.
    enc = hybrid_model.tokenizer(
        [_build_text_for_explain(parsed)],
        truncation=True,
        padding=True,
        max_length=hybrid_model.max_length,
        return_tensors="pt",
    )
    hybrid_model.model.eval()
    hybrid_model.model.zero_grad(set_to_none=True)
    input_ids = enc["input_ids"].to(hybrid_model.device)
    attn = enc["attention_mask"].to(hybrid_model.device)
    struct_tensor = struct_tensor.to(hybrid_model.device).requires_grad_(True)
    with torch.enable_grad():
        logits = hybrid_model.model(input_ids, attn, struct_tensor)
        p_spam = torch.softmax(logits, dim=-1)[0, 1]
        p_spam.backward()
    if struct_tensor.grad is None:
        return None
    # signed contribution: positive = pushes toward spam.
    signed = (struct_tensor.grad[0] * struct_tensor[0]).detach().cpu().numpy()
    return signed


def _build_text_for_explain(parsed: ParsedEmail) -> str:
    """Mirror `_build_text` from src.models.distilbert without the import."""
    subject = (parsed.subject or "").strip()
    body = (parsed.body or "").strip()
    sender = (parsed.sender or "").strip()
    head = f"[SUBJ] {subject} [FROM] {sender} [BODY]"
    return f"{head} {body}"[:8000]


@st.cache_data(show_spinner=False)
def _compute_explanations_v2(body, subject, sender):
    """Compute all three explanations and stash in one cache slot.

    The args are content (strings), so re-runs for the same email are free;
    switching emails recomputes. Heavy work (one DistilBERT forward+backward
    for text saliency, one hybrid forward+backward for structural saliency)
    is unavoidable.
    """
    parsed = ParsedEmail()
    parsed.sender = sender or ""
    parsed.subject = subject or ""
    parsed.body = body or ""
    parsed.raw_body = body or ""
    parsed.char_count = len(body or "")
    parsed.word_count = len((body or "").split())
    struct_vec = extract_structural(parsed).reshape(1, -1)

    models = get_models()
    base_contribs = _explain_baseline(models["baseline"], body or "", subject or "")
    text_sal = _explain_text_saliency(models["distilbert"], body or "", subject or "")
    struct_signed = _explain_structural_saliency(models["hybrid"], parsed, struct_vec)
    return {
        "baseline": base_contribs,
        "distilbert": text_sal,
        "structural_signed": struct_signed.tolist() if struct_signed is not None else None,
        "struct_vec": struct_vec.flatten().tolist(),
    }


def _set_sample(label: str, sender: str, subject: str, body: str):
    """Sample-button callback: load the canned email, then request a classify.

    The actual scoring is deferred to the main render loop so it runs
    inside a `st.spinner`. Running heavy CPU work (DistilBERT forward
    pass) inside a callback blocks Streamlit's rerun and makes the user
    click the button multiple times before the result appears.
    """
    del label  # label is part of the button row, not used downstream
    st.session_state["cls_sender"] = sender
    st.session_state["cls_subject"] = subject
    st.session_state["cls_body"] = body
    st.session_state["cls_needs_run"] = True
    # Invalidate any previous result so we don't briefly show stale scores.
    st.session_state["cls_result"] = None
    st.session_state["cls_result_error"] = None


def render_classifier():
    st.markdown("## \U0001F50E Live Email Classifier")
    st.caption("Paste any email below to see what each of the 4 models thinks.")

    # Seed session state with defaults on first render.
    defaults = {
        "cls_sender": "newsletter@team.example.com",
        "cls_subject": "Quarterly project update",
        "cls_body": (
            "Hi team,\n\n"
            "Attached are the latest slides for the Q4 review meeting. "
            "Please have a look before Friday and let me know if anything needs clarification.\n\n"
            "Best,\nJohn"
        ),
        "cls_result": None,
        "cls_result_error": None,
        "cls_last_sender": "",
        "cls_last_subject": "",
        "cls_last_body": "",
        "cls_needs_run": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Inputs bind to session_state via `key=` so the sample buttons can mutate them.
    c1, c2 = st.columns([2, 1])
    c1.text_input("From (sender address)", key="cls_sender")
    c2.text_input("Subject", key="cls_subject")
    body = st.text_area("Email body", key="cls_body", height=200)

    # "Classify" button — just sets a flag; the actual scoring happens in the
    # main body below (where `st.spinner` is allowed to render).
    def _request_classify():
        st.session_state["cls_needs_run"] = True

    st.button(
        "\U0001F50D Classify this email",
        use_container_width=True,
        on_click=_request_classify,
    )

    # Sample buttons: clicking one updates the three inputs + triggers scoring.
    st.markdown("### \U0001F4A1 Or try a sample")
    samples = [
        ("Phishing prize", "lottery@cheap-deals.click",
         "WIN A FREE iPHONE 15 PRO!!!",
         "Congratulations! Click http://cheap-deals.click/claim?id=42 to claim your prize. Limited time!"),
        ("Legit work", "manager@company.com",
         "Project review meeting",
         "Hi, please find the latest slides attached for our review on Friday at 3 PM."),
        ("Bank scam", "security@bankk-secure-login.tk",
         "URGENT verify your account",
         "Dear user your account will be deactivated. Verify immediately: http://bankk-secure-login.tk/login"),
        ("Friend chat", "alex@friend.com",
         "Lunch tomorrow?",
         "Hey! Want to grab lunch tomorrow at the new Thai place near the office?"),
    ]
    sample_cols = st.columns(4)
    for col, (label, s, sub, b) in zip(sample_cols, samples):
        col.button(
            label,
            key=f"sample_{label}",
            use_container_width=True,
            on_click=_set_sample,
            args=(label, s, sub, b),
        )

    # Drain the pending-flag queue: if a button (sample or Classify) was just
    # clicked, run the heavy scoring here in the main render loop so the
    # spinner can show progress and Streamlit can complete the rerun cleanly.
    if st.session_state.get("cls_needs_run", False):
        st.session_state["cls_needs_run"] = False
        with st.spinner("Scoring across 4 models..."):
            _classify_now()

    # If a sample button was clicked on a previous render, scoring already ran.
    result = st.session_state.get("cls_result")
    err = st.session_state.get("cls_result_error")
    if err:
        st.error(err)
        return
    if result is None:
        st.info("Paste an email above (or pick a sample) and press **Classify this email** to see what each model thinks.")
        return

    sender = st.session_state["cls_last_sender"]
    subject = st.session_state["cls_last_subject"]
    body_disp = st.session_state["cls_last_body"]
    scores = result

    st.markdown(
        f"<div class='sg-card' style='margin-top:1rem;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap;'>"
        f"<div>"
        f"<h4 style='margin:0;font-size:1.1rem;'>&#128236; {subject or '(no subject)'}</h4>"
        f"<div style='color:#64748B;font-size:.85rem;margin-top:.2rem;'>"
        f"From: <code style='background:#F1F5F9;padding:.05rem .35rem;border-radius:4px;'>{sender or '(unknown)'}</code>"
        f" &nbsp;\u00b7&nbsp; {len(body_disp.split())} words &nbsp;\u00b7&nbsp; {len(body_disp)} chars"
        f"</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(scores))
    for col, (k, info) in zip(cols, scores.items()):
        proba = info["proba"]
        decision = info["decision"]
        verdict_cls = "spam" if decision == "SPAM" else "ham"
        body_html = (
            f"<div class='sg-verdict {verdict_cls}'>"
            f"<div class='name'>{MODEL_DISPLAY[k]}</div>"
            f"<div class='label'>{decision}</div>"
            f"<div class='score'>score <b>{proba:.3f}</b></div>"
        )
        if k == "personalized" and "bias" in info:
            body_html += (
                f"<div class='breakdown'>hybrid {info['base_proba']:.3f} \u00b7 "
                f"profile bias {info['bias']:.3f}</div>"
            )
        body_html += "</div>"
        col.markdown(body_html, unsafe_allow_html=True)

    if "personalized" in scores:
        det = load_personalized()
        st.markdown("### Why personalized thinks what it thinks")
        c1, c2, c3 = st.columns(3)
        c1.markdown(_kpi("Hybrid score", f"{scores['personalized']['base_proba']:.3f}",
                         "DistilBERT + structural"), unsafe_allow_html=True)
        c2.markdown(_kpi("Profile bias", f"{scores['personalized']['bias']:.3f}",
                         "Sender reputation + keywords"), unsafe_allow_html=True)
        c3.markdown(_kpi("Final blended", f"{scores['personalized']['proba']:.3f}",
                         "(1\u2212\u03b1)\u00b7hybrid + \u03b1\u00b7bias", brand=True), unsafe_allow_html=True)
        st.caption(
            f"Profile \u2014 trusted senders: {len(det.profile.trusted_senders)} \u00b7 "
            f"blocked senders: {len(det.profile.blocked_senders)} \u00b7 "
            f"feedback events: {len(det.profile.feedback)} \u00b7 "
            f"tracked senders: {len(det.profile.sender_stats)}"
        )

    with st.expander("Structural features extracted", expanded=False):
        parsed = ParsedEmail()
        parsed.sender = sender
        parsed.subject = subject
        parsed.body = body_disp
        parsed.raw_body = body_disp
        parsed.char_count = len(body_disp)
        parsed.word_count = len(body_disp.split())
        feat_vec = extract_structural(parsed)
        fdf = pd.DataFrame({
            "feature": STRUCTURAL_FEATURE_NAMES,
            "value": [round(float(v), 4) for v in feat_vec.tolist()],
        })
        st.dataframe(fdf, use_container_width=True, hide_index=True, height=300)

    # ---- Why did each model say what it said? (token / feature saliency) ----
    st.markdown("### Why each model scored it that way")
    st.markdown(
        "<div style='color:#475569;font-size:.88rem;line-height:1.4;"
        "margin:-0.2rem 0 0.6rem 0;'>"
        "Token importance from the TF-IDF baseline, gradient saliency from "
        "DistilBERT, and signed structural-feature contributions from the "
        "hybrid head. Red pushes toward SPAM, green toward HAM."
        "</div>",
        unsafe_allow_html=True,
    )
    with st.spinner("Computing explanations (one forward+backward pass per model)..."):
        expl = _compute_explanations_v2(body_disp, subject, sender)

    expl_cols = st.columns(3)

    def _card_title(text: str) -> None:
        st.markdown(
            f"<div style='color:{PALETTE['ink']};font-weight:700;"
            f"font-size:.95rem;margin:0 0 .35rem 0;'>{text}</div>",
            unsafe_allow_html=True,
        )

    def _card_caption(text: str) -> None:
        st.markdown(
            f"<div style='color:{PALETTE['ink_soft']};font-size:.78rem;"
            f"line-height:1.35;margin-top:.35rem;'>{text}</div>",
            unsafe_allow_html=True,
        )

    plotly_cfg = {"displayModeBar": False, "displaylogo": False,
                  "responsive": True}

    # 1) TF-IDF baseline: top tokens by signed contribution.
    with expl_cols[0]:
        _card_title("TF-IDF + Naive Bayes")
        base = expl.get("baseline")
        if not base:
            st.info("No in-vocabulary tokens, or model not loaded.")
        else:
            peak = max(abs(c) for _, _, _, c in base) or 1.0
            rows = []
            for tok, tfidf, lo, contrib in base:
                rows.append({
                    "token": tok,
                    "tfidf": tfidf,
                    "log-odds": lo,
                    "toward": "SPAM" if contrib > 0 else "HAM",
                })
            bdf = pd.DataFrame(rows)
            st.dataframe(
                bdf.style.format({"tfidf": "{:.2f}", "log-odds": "{:.2f}"}),
                use_container_width=True, hide_index=True, height=300,
            )
            _card_caption(
                f"Top tokens by |TF-IDF × log P(t|spam) − log P(t|ham)|. "
                f"Max contribution: {peak:.2f}."
            )

    # 2) DistilBERT: top words by signed gradient × input saliency.
    with expl_cols[1]:
        _card_title("DistilBERT (text)")
        sal = expl.get("distilbert")
        if not sal:
            st.info("No saliency available (model not loaded, or empty input).")
        else:
            words = [w for w, _ in sal]
            scores = [s for _, s in sal]
            # Divergent: positive = pushes spam logit up (red),
            # negative = pushes it down (green).
            colors = [PALETTE["bad"] if s > 0 else PALETTE["good"] for s in scores]
            fig = go.Figure(go.Bar(
                x=scores[::-1], y=words[::-1], orientation="h",
                marker=dict(color=colors[::-1], opacity=1.0,
                            line=dict(color="rgba(15,23,42,.15)", width=0.5)),
                hovertemplate="<b>%{y}</b><br>signed Δ %{x:+.3f}<extra></extra>",
            ))
            peak = max(abs(s) for s in scores) or 1.0
            fig.update_layout(
                height=300, margin=dict(t=10, l=10, r=10, b=30),
                xaxis=dict(
                    title=dict(text="signed contribution to spam logit",
                               font=dict(color=PALETTE["ink_soft"], size=11)),
                    zeroline=True, zerolinecolor=PALETTE["muted"],
                    range=[-peak * 1.15, peak * 1.15],
                    showgrid=True, gridcolor="rgba(15,23,42,.06)",
                    tickfont=dict(color=PALETTE["ink_soft"], size=11),
                ),
                yaxis=dict(
                    tickfont=dict(color=PALETTE["ink"], size=11,
                                  family="Inter, sans-serif"),
                    automargin=True,
                ),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=PALETTE["ink_soft"],
                          size=11),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config=plotly_cfg)
            _card_caption(
                "Signed gradient × input summed across subwords. Red pushes "
                "the spam logit up, green pushes it down."
            )

    # 3) Hybrid: signed structural-feature contributions.
    with expl_cols[2]:
        _card_title("Hybrid (structural features)")
        signed = expl.get("structural_signed")
        if not signed:
            st.info("No structural saliency available (hybrid not loaded).")
        else:
            feats = STRUCTURAL_FEATURE_NAMES
            pairs = sorted(
                zip(feats, signed), key=lambda kv: abs(kv[1]), reverse=True
            )[:14]
            labels = [p[0] for p in pairs][::-1]
            values = [p[1] for p in pairs][::-1]
            colors = [PALETTE["bad"] if v > 0 else PALETTE["good"]
                      for v in values]
            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker=dict(color=colors, opacity=1.0,
                            line=dict(color="rgba(15,23,42,.15)", width=0.5)),
                hovertemplate="<b>%{y}</b><br>signed Δ %{x:+.3f}<extra></extra>",
            ))
            fig.update_layout(
                height=300, margin=dict(t=10, l=10, r=10, b=30),
                xaxis=dict(title=dict(text="signed contribution to spam logit",
                                      font=dict(color=PALETTE["ink_soft"], size=11)),
                           zeroline=True, zerolinecolor=PALETTE["muted"],
                           showgrid=True, gridcolor="rgba(15,23,42,.06)",
                           tickfont=dict(color=PALETTE["ink_soft"], size=11)),
                yaxis=dict(
                    tickfont=dict(color=PALETTE["ink"], size=10,
                                  family="Inter, sans-serif"),
                    automargin=True,
                ),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=PALETTE["ink_soft"],
                          size=11),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config=plotly_cfg)
            _card_caption(
                "Top 14 of 20 structural features. Positive pushes the spam "
                "logit up; negative pushes it down."
            )


def render_feedback():
    st.markdown(
        """
        <div class="sg-hero" style="padding:1.2rem 1.5rem;">
          <h1 style="font-size:1.45rem;">&#128257; Feedback Loop</h1>
          <p>Label each one \u2014 the personalization layer remembers trusted senders, sender
             reputation, and keyword preferences, and reuses them to bias future predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    det = load_personalized()
    if det is None:
        st.error("No personalized detector found. Run training first.")
        return

    samples = [
        {
            "label": "Phishing prize", "sender": "lottery@cheap-deals.click",
            "subject": "WIN A FREE iPHONE 15 PRO!!!",
            "body": ("Congratulations! Click http://cheap-deals.click/claim?id=42 to claim your prize. "
                     "FREE!! Limited time. http://bit.ly/free-iphone"),
            "truth": 1,
        },
        {
            "label": "Legit work mail", "sender": "manager@company.com",
            "subject": "Quarterly review meeting",
            "body": ("Hi team, just a reminder that our quarterly review meeting is scheduled for "
                     "Friday at 3 PM. Please bring your status updates and any blockers."),
            "truth": 0,
        },
        {
            "label": "Bank phishing", "sender": "security@bankk-secure-login.tk",
            "subject": "URGENT verify your account",
            "body": ("Dear user your account will be deactivated. Please verify your details immediately: "
                     "http://bankk-secure-login.tk/login"),
            "truth": 1,
        },
        {
            "label": "Friend lunch", "sender": "alex@friend.com",
            "subject": "Lunch tomorrow?",
            "body": "Hey! Want to grab lunch tomorrow at the new Thai place near the office?",
            "truth": 0,
        },
    ]

    if "feedback_round" not in st.session_state:
        st.session_state.feedback_round = 0
        st.session_state.feedback_history = []

    # KPI tiles for current profile state
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(_kpi("Round", f"#{st.session_state.feedback_round}", "current sample"), unsafe_allow_html=True)
    k2.markdown(_kpi("Feedback so far", f"{len(st.session_state.feedback_history)}", "labels given"), unsafe_allow_html=True)
    k3.markdown(
        _kpi("Trusted senders", str(len(det.profile.trusted_senders)),
             f"+ {len(det.profile.trusted_domains)} domains"),
        unsafe_allow_html=True,
    )
    k4.markdown(
        _kpi("Blocked senders", str(len(det.profile.blocked_senders)),
             f"+ {len(det.profile.blocked_domains)} domains"),
        unsafe_allow_html=True,
    )

    st.markdown("### Email to label")
    email = samples[st.session_state.feedback_round % len(samples)]
    parsed = ParsedEmail()
    parsed.sender = email["sender"]
    parsed.subject = email["subject"]
    parsed.body = email["body"]
    parsed.raw_body = email["body"]
    parsed.char_count = len(email["body"])
    parsed.word_count = len(email["body"].split())

    models = get_models()
    with st.spinner("Scoring..."):
        scores = score_email(parsed, models)

    st.markdown(
        f"<div class='sg-card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap;'>"
        f"<div style='flex:1;'>"
        f"<h4 style='margin:0;font-size:1.05rem;'>&#128236; {email['subject']}</h4>"
        f"<div style='color:#64748B;font-size:.85rem;margin-top:.25rem;'>"
        f"From <code style='background:#F1F5F9;padding:.05rem .35rem;border-radius:4px;'>{email['sender']}</code>"
        f"</div>"
        f"<div style='color:#334155;margin-top:.7rem;font-size:.92rem;line-height:1.55;'>{email['body']}</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Current model opinions")
    cols = st.columns(len(scores))
    for col, (k, info) in zip(cols, scores.items()):
        proba = info["proba"]
        decision = info["decision"]
        verdict_cls = "spam" if decision == "SPAM" else "ham"
        body_html = (
            f"<div class='sg-verdict {verdict_cls}' style='text-align:center;padding:1rem;'>"
            f"<div class='name'>{MODEL_DISPLAY[k]}</div>"
            f"<div class='label' style='font-size:1.4rem;'>{decision}</div>"
            f"<div class='score'>{proba:.3f}</div>"
        )
        if k == "personalized" and "bias" in info:
            body_html += (
                f"<div class='breakdown'>hybrid {info['base_proba']:.3f} \u00B7 "
                f"bias {info['bias']:.3f}</div>"
            )
        body_html += "</div>"
        col.markdown(body_html, unsafe_allow_html=True)

    st.markdown("### Your verdict")
    b1, b2, b3 = st.columns([1, 1, 1])
    if b1.button("&#128308; Mark as SPAM", key="fb_spam", use_container_width=True):
        det.detector_submit_feedback(parsed, label=1)
        st.session_state.feedback_history.append((email["label"], 1))
        st.session_state.feedback_round += 1
        st.rerun()
    if b2.button("&#128994; Mark as HAM (legit)", key="fb_ham", use_container_width=True):
        det.detector_submit_feedback(parsed, label=0)
        st.session_state.feedback_history.append((email["label"], 0))
        st.session_state.feedback_round += 1
        st.rerun()
    if b3.button("\u23ED Reset profile", key="fb_reset", use_container_width=True):
        det.profile = UserProfile()
        det.save(str(MODELS_DIR / "personalized"))
        st.session_state.feedback_history = []
        st.session_state.feedback_round = 0
        st.rerun()

    if st.session_state.feedback_history:
        st.markdown("### Feedback history")
        hdf = pd.DataFrame(
            st.session_state.feedback_history,
            columns=["Email", "Your verdict"],
        )
        hdf["Your verdict"] = hdf["Your verdict"].map({0: "&#128994; HAM", 1: "&#128308; SPAM"})
        st.dataframe(hdf, use_container_width=True, hide_index=True)


_PAGE_OPTIONS = [
    "Overview",
    "Model Comparison",
    "Live Classifier",
    "Feedback Loop",
]


def main():
    st.sidebar.markdown(
        """
        <div class="sg-side-brand">
          <span class="dot"></span> SpamGuard
        </div>
        <div style="color:#94A3B8;font-size:.78rem;margin-bottom:1.25rem;">
          Context-aware personalized spam detection
        </div>
        """,
        unsafe_allow_html=True,
    )

    labels = list(_PAGE_OPTIONS)
    choice = st.sidebar.radio("Navigate", labels, label_visibility="collapsed")
    page = choice

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="color:#CBD5E1;font-size:.85rem;line-height:1.7;">
          <b style="color:#fff;">How to use</b><br/>
          1. <b>Overview</b> for the project summary<br/>
          2. <b>Model Comparison</b> to see metrics<br/>
          3. <b>Live Classifier</b> to score an email<br/>
          4. <b>Feedback Loop</b> to teach the profile
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="color:#94A3B8;font-size:.72rem;line-height:1.55;">
          <b style="color:#CBD5E1;">Stack</b><br/>
          DistilBERT \u00b7 20 structural features<br/>
          MultinomialNB baseline \u00b7 user profile
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Overview":
        render_overview()
    elif page == "Model Comparison":
        render_comparison()
    elif page == "Live Classifier":
        render_classifier()
    elif page == "Feedback Loop":
        render_feedback()

    st.markdown(
        """
        <div class="sg-foot">
          Built with Streamlit \u00b7 DistilBERT \u00b7 scikit-learn \u00b7 NumPy \u00b7 Plotly<br/>
          SpamGuard \u2014 personalized email spam detection \u00b7 capstone demo
        </div>
        """,
        unsafe_allow_html=True,
    )


def _attach_submit_feedback(self, parsed: ParsedEmail, label: int):
    """Attach a convenient method to PersonalizedSpamDetector for the demo."""
    self.profile.record(parsed.sender or "", int(label), confidence=1.0,
                         note="streamlit_demo")
    self.save(str(MODELS_DIR / "personalized"))


PersonalizedSpamDetector.detector_submit_feedback = _attach_submit_feedback


if __name__ == "__main__":
    main()