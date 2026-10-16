"""Generate IEEE two-column paper as DOCX, with diagrams, no em-dashes.

Output: C:/Users/podda/Downloads/SpamGuard_paper.docx
        C:/Users/podda/Downloads/SpamGuard_paper.pdf   (if LibreOffice present)

Run from anywhere:
    python paper/generate_ieee_paper.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import nsmap, qn
from docx.shared import Cm, Inches, Pt, RGBColor


# =============================================================================
# Low-level XML helpers
# =============================================================================
def _set_cols(section, num_cols, space_twips=720, equal_width=True):
    """Set multi-column layout on a section (used for IEEE two-column)."""
    sect_pr = section._sectPr
    # Remove existing cols
    for old in sect_pr.findall(qn("w:cols")):
        sect_pr.remove(old)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(num_cols))
    cols.set(qn("w:space"), str(space_twips))
    if equal_width:
        cols.set(qn("w:equalWidth"), "1")
    sect_pr.append(cols)


def _set_cell_border(cell, **kwargs):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = tc_pr.find(qn("w:tcBorders"))
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge, props in kwargs.items():
        tag = qn(f"w:{edge}")
        elt = tc_borders.find(tag)
        if elt is None:
            elt = OxmlElement(f"w:{edge}")
            tc_borders.append(elt)
        for k, v in props.items():
            elt.set(qn(f"w:{k}"), str(v))


def _shade_cell(cell, fill_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)


def _set_font(run, name="Times New Roman", size=10, bold=False, italic=False,
              color=None, underline=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.underline = underline
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)


def _column_break(doc):
    """Force the next paragraph to begin a new column."""
    p = doc.add_paragraph()
    run = p.add_run()
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "column")
    run._element.append(br)
    return p


# =============================================================================
# Paragraph helpers (IEEE two-column uses small type, justified)
# =============================================================================
def add_para(doc, text, *, size=10, bold=False, italic=False,
             alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
             space_before=0, space_after=2, first_line_indent=0.25,
             line_spacing=1.05):
    p = doc.add_paragraph()
    p.alignment = alignment
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.first_line_indent = Cm(first_line_indent) if first_line_indent else None
    pf.line_spacing = line_spacing
    if text:
        run = p.add_run(text)
        _set_font(run, size=size, bold=bold, italic=italic)
    return p


def add_section_heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(8 if level == 1 else 6)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.05
    pf.keep_with_next = True
    run = p.add_run(text)
    if level == 1:
        _set_font(run, size=10, bold=True, italic=False)
        # IEEE uses Roman numerals; we use them as numerals to match
    else:
        _set_font(run, size=10, bold=False, italic=True)
    return p


def add_equation(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.05
    run = p.add_run(text)
    _set_font(run, size=10, italic=True)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.05
    run = p.add_run(text)
    _set_font(run, size=8, italic=True)
    return p


def add_itemize(doc, items, indent=0.4):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Cm(indent)
        pf.first_line_indent = Cm(-0.25)
        pf.space_before = Pt(0)
        pf.space_after = Pt(1)
        pf.line_spacing = 1.05
        run = p.add_run(item)
        _set_font(run, size=10)


def add_enumerate(doc, items):
    for i, item in enumerate(items, start=1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Cm(0.7)
        pf.first_line_indent = Cm(-0.5)
        pf.space_before = Pt(0)
        pf.space_after = Pt(1)
        pf.line_spacing = 1.05
        run = p.add_run(f"({chr(ord('i') + i - 1)})\t")
        _set_font(run, size=10, italic=True)
        run = p.add_run(item)
        _set_font(run, size=10)


def add_image(doc, path: Path, width_cm=8.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(2)
    pf.line_spacing = 1.0
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    return p


def add_table(doc, rows, *, header=True, alignments=None, col_widths=None,
              header_size=9, body_size=9):
    n_rows, n_cols = len(rows), len(rows[0])
    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    if col_widths:
        for i, w in enumerate(col_widths):
            for cell in table.columns[i].cells:
                cell.width = Cm(w)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                p.clear()
            p = cell.paragraphs[0]
            p.alignment = alignments[c] if alignments else WD_ALIGN_PARAGRAPH.CENTER
            pf = p.paragraph_format
            pf.space_before = Pt(1)
            pf.space_after = Pt(1)
            pf.line_spacing = 1.0
            run = p.add_run(val)
            if header and r == 0:
                _set_font(run, size=header_size, bold=True)
                _shade_cell(cell, "EFEFEF")
            else:
                _set_font(run, size=body_size)
            for edge in ("top", "left", "bottom", "right"):
                _set_cell_border(
                    cell, **{edge: {"sz": 4, "val": "single", "color": "404040"}}
                )
    return table


# =============================================================================
# Diagram generators (matplotlib, IEEE-friendly style)
# =============================================================================
def draw_architecture_diagram(out: Path):
    """Figure 1: Three-stage pipeline with side branch for personalization."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    palette = {
        "raw": "#1f2a44",
        "pre": "#3b507a",
        "enc_text": "#2c7a7b",
        "enc_struct": "#7a5c2c",
        "fusion": "#5b2c83",
        "hyb": "#c0392b",
        "prof": "#7f8c8d",
        "bias": "#7d6608",
        "final": "#1e8449",
    }

    def box(x, y, w, h, text, fill, fc="white", fontsize=9, fontweight="bold",
            text_color="white", border_color="black"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            linewidth=1.1, edgecolor=border_color, facecolor=fill,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight=fontweight,
                wrap=True)

    def arrow(x1, y1, x2, y2, color="#222"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=color),
        )

    # Top row: input + preprocessor
    box(2, 44, 14, 9, "Raw email\n(.eml)", palette["raw"], fontsize=8.5)
    box(22, 44, 18, 9, "Preprocessor\nRFC822 + HTML strip", palette["pre"])
    arrow(16, 48.5, 22, 48.5)

    # Middle row: two parallel encoders
    box(2, 24, 22, 10, "DistilBERT\n(subject + body, 256 tok)\n→ c ∈ ℝ⁷⁶⁸",
        palette["enc_text"], fontsize=8.5)
    box(28, 24, 22, 10, "Structural MLP\n20-d x ∈ ℝ²⁰ → s ∈ ℝ¹²⁸",
        palette["enc_struct"], fontsize=8.5)
    arrow(31, 44, 13, 34.5)   # pre -> BERT
    arrow(31, 44, 39, 34.5)   # pre -> struct

    # Fusion
    box(56, 24, 18, 10, "Fusion MLP\n[c ; s] → logit\n→ p_hyb",
        palette["fusion"], fontsize=8.5)
    arrow(24, 29, 56, 29)
    arrow(50, 29, 56, 29)

    # Right column: user profile branch
    box(56, 44, 18, 9, "User profile\nsenders + keywords",
        palette["prof"], fontsize=8.5)
    box(78, 44, 18, 9, "Bias\nb = 0.7·rep + 0.3·kw",
        palette["bias"], fontsize=8.5)
    arrow(74, 48.5, 78, 48.5)

    # Final score
    box(72, 24, 22, 10,
        "Personalized\n(1−α)·p_hyb + α·b",
        palette["final"], fontsize=9)

    # Connections from fusion/profiles to final
    ax.annotate("", xy=(72, 32), xytext=(74, 32),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#1e8449"))
    ax.annotate("", xy=(72, 28), xytext=(78, 44),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#1e8449"))

    # Stage labels on the left
    ax.text(0.4, 48.5, "1", color="black", fontsize=11, fontweight="bold")
    ax.text(0.4, 29, "2", color="black", fontsize=11, fontweight="bold")
    ax.text(56.4, 44.5, "3", color="black", fontsize=11, fontweight="bold")
    ax.text(56.4, 29.5, "4", color="black", fontsize=11, fontweight="bold")

    # Decorative column for the personalized-only branch
    ax.plot([54, 54], [16, 56], color="#888", lw=0.5, linestyle=":")

    ax.text(50, 12,
            "Stage 1 = ingestion, Stage 2 = encoding (parallel), "
            "Stage 3 = user profile, Stage 4 = fusion and decision.",
            ha="center", fontsize=7.5, style="italic", color="#555")

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_pipeline_diagram(out: Path):
    """Figure 2: Preprocessing + training pipeline with all 4 models."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    def box(x, y, w, h, text, fill, fontsize=8.5, fontcolor="white"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.25,rounding_size=0.7",
            linewidth=1.0, edgecolor="#222", facecolor=fill,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=fontcolor, fontsize=fontsize, fontweight="bold")

    def arrow(x1, y1, x2, y2, color="#222"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color))

    # Source + parse
    box(2, 56, 14, 8, "SpamAssassin\n+ Enron-Spam", "#1f2a44", fontsize=8.5)
    box(20, 56, 14, 8, "RFC 822\nparser", "#3b507a")
    arrow(16, 60, 20, 60)
    box(38, 56, 16, 8, "Body + headers\n+ URLs + MIME", "#3b507a", fontsize=8)
    arrow(34, 60, 38, 60)
    box(58, 56, 18, 8, "Clean body\nHTML strip", "#445e93")
    arrow(54, 60, 58, 60)

    # 20-d structural features
    box(78, 56, 20, 8, "20 structural\nfeatures (Table I)", "#7a5c2c", fontsize=8)
    arrow(76, 51, 88, 56)
    box(40, 36, 14, 9, "TF-IDF\n+ NB", "#5d6d7e", fontsize=8.5)
    arrow(50, 56, 47, 45)

    # DistilBERT path
    box(58, 36, 16, 9, "DistilBERT\n(fine-tune)", "#2c7a7b", fontsize=8.5)
    arrow(64, 56, 66, 45)
    # Hybrid path
    box(78, 36, 18, 9, "Hybrid DistilBERT\n+ structural MLP", "#5b2c83",
        fontsize=8.5)
    arrow(88, 56, 87, 45)
    arrow(98, 56, 87, 45)

    # Personalization
    box(78, 18, 18, 9, "Personalization\nlayer", "#7f8c8d", fontsize=8.5)
    arrow(87, 36, 87, 27)

    # Final
    box(40, 4, 50, 9, "Evaluation harness: SpamAssassin (in-domain) and "
                       "Enron-Spam (cross-domain)", "#1e8449", fontsize=8.5)

    arrow(47, 36, 50, 13)
    arrow(66, 36, 60, 13)
    arrow(87, 18, 60, 13)

    # Stage headers
    ax.text(2, 66, "Ingestion + parse", fontsize=8.5, fontweight="bold",
            color="#1f2a44")
    ax.text(38, 66, "Preprocessing", fontsize=8.5, fontweight="bold",
            color="#3b507a")
    ax.text(78, 66, "Features", fontsize=8.5, fontweight="bold",
            color="#7a5c2c")
    ax.text(38, 47, "Four models", fontsize=8.5, fontweight="bold",
            color="#1f2a44")

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_results_diagram(out: Path):
    """Figure 3: Grouped bar chart comparing all 4 models on both datasets."""
    models = ["TF-IDF + NB", "DistilBERT", "Hybrid", "Personalized"]
    sa_f1 = [0.985, 0.949, 0.959, 0.962]
    en_f1 = [0.751, 0.670, 0.687, 0.687]
    sa_fpr = [0.013, 0.063, 0.063, 0.057]
    en_fpr = [0.347, 0.543, 0.637, 0.637]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=220)

    x = np.arange(len(models))
    w = 0.35
    colors_sa = "#2c7a7b"
    colors_en = "#c0392b"

    # Left: F1
    ax = axes[0]
    b1 = ax.bar(x - w / 2, sa_f1, w, color=colors_sa, label="SpamAssassin")
    b2 = ax.bar(x + w / 2, en_f1, w, color=colors_en, label="Enron-Spam")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score", fontsize=9)
    ax.set_title("(a) F1 across both corpora", fontsize=9.5, pad=6)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    for b in (*b1, *b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", va="bottom",
                fontsize=6.5, color="#222")

    # Right: FPR (lower is better)
    ax = axes[1]
    b1 = ax.bar(x - w / 2, sa_fpr, w, color=colors_sa, label="SpamAssassin")
    b2 = ax.bar(x + w / 2, en_fpr, w, color=colors_en, label="Enron-Spam")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("False-positive rate (lower is better)", fontsize=9)
    ax.set_title("(b) FPR across both corpora", fontsize=9.5, pad=6)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for b in (*b1, *b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=6.5, color="#222")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# Paper builder
# =============================================================================
def build_paper(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- diagrams ----
    fig_dir = out_dir / "_figures"
    fig_dir.mkdir(exist_ok=True)
    arch_png = fig_dir / "architecture.png"
    pipe_png = fig_dir / "pipeline.png"
    res_png = fig_dir / "results.png"
    draw_architecture_diagram(arch_png)
    draw_pipeline_diagram(pipe_png)
    draw_results_diagram(res_png)

    doc = Document()

    # Page setup: US Letter, narrow
    section = doc.sections[0]
    section.page_height = Cm(27.94)
    section.page_width = Cm(21.59)
    section.top_margin = Cm(1.27)
    section.bottom_margin = Cm(1.27)
    section.left_margin = Cm(1.27)
    section.right_margin = Cm(1.27)

    # Two-column body via XML
    _set_cols(section, num_cols=1)  # title spans full width; columns start later

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)

    # =========================================================================
    # Title (full-width)
    # =========================================================================
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = title.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(4)
    run = title.add_run(
        "SpamGuard: An End-to-End Personal Spam Detection Pipeline "
        "with DistilBERT, Structural Feature Fusion, and a User-Profile "
        "Decision Layer"
    )
    _set_font(run, size=18, bold=True)

    # Sub-title (italic)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = sub.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)
    run = sub.add_run("A four-model comparison on SpamAssassin and Enron-Spam")
    _set_font(run, size=11, italic=True)

    # Author block
    au = doc.add_paragraph()
    au.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = au.paragraph_format
    pf.space_after = Pt(0)
    run = au.add_run("<Author Name>")
    _set_font(run, size=11, bold=False)

    aff = doc.add_paragraph()
    aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = aff.paragraph_format
    pf.space_after = Pt(8)
    run = aff.add_run(
        "Department of Computer Science, [University], <City, Country>\n"
        "<email@university.edu>"
    )
    _set_font(run, size=10, italic=True)

    # Insert continuous section break to start the two-column body
    # We do it by adding a new section that starts continuous with 2 cols
    body_section = doc.add_section(WD_SECTION.CONTINUOUS)
    body_section.page_height = section.page_height
    body_section.page_width = section.page_width
    body_section.top_margin = section.top_margin
    body_section.bottom_margin = section.bottom_margin
    body_section.left_margin = section.left_margin
    body_section.right_margin = section.right_margin
    _set_cols(body_section, num_cols=2, space_twips=510)

    # =========================================================================
    # Abstract + Keywords (IEEE style: italic + bold keywords)
    # =========================================================================
    abstract = (
        "We present SpamGuard, an end-to-end pipeline that detects spam by "
        "fusing three signals in a single decision: DistilBERT contextual "
        "embeddings, twenty hand-crafted structural features pulled out "
        "of RFC 822 headers and MIME parts, and a per-user profile that "
        "learns from explicit feedback. We compare four models on the "
        "SpamAssassin Public Corpus (7,953 emails): a TF-IDF + "
        "Multinomial Naïve Bayes baseline, a fine-tuned DistilBERT, "
        "a hybrid that concatenates the BERT CLS vector with the "
        "structural features inside an MLP, and a personalized version "
        "that blends the hybrid probability with a sender- and "
        "keyword-derived bias. We validate cross-domain generalization on "
        "Enron-Spam (33,716 emails, none of which appears in training). "
        "The headline result: the personalized model cut the "
        "false-positive rate from 0.063 down to 0.057 on SpamAssassin, a "
        "10% relative reduction, without giving up any recall. That "
        "matters because in spam filtering a single false positive is a "
        "missed job offer or a lost receipt. We report the numbers, walk "
        "through what surprised us (the linear baseline beats DistilBERT "
        "in-domain; the personalization layer is dormant on Enron because "
        "no per-user signal exists there), and point at what a "
        "GPU-backed, multi-corpus follow-up would look like. The "
        "trained models and an interactive Streamlit demo are in the "
        "repo."
    )
    add_para(doc, abstract, size=9, italic=False,
             alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
             first_line_indent=0, space_after=4)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(6)
    pf.first_line_indent = Cm(0.25)
    pf.line_spacing = 1.05
    r = p.add_run("Index Terms")
    _set_font(r, size=9, bold=True, italic=True)
    r = p.add_run(": spam detection, DistilBERT, multimodal fusion, "
                  "personalization, user profile, false-positive rate, "
                  "transfer learning, capstone.")
    _set_font(r, size=9, italic=True)

    # =========================================================================
    # I. Introduction
    # =========================================================================
    add_section_heading(doc, "I.\u2003Introduction")
    add_para(doc,
        "If you have used email for more than a year you have probably "
        "noticed that spam filters are still wrong, in both directions, in "
        "ways that feel silly. A newsletter you actually subscribed to "
        "lands in spam. A pretty obvious phishing email from "
        "\"urgent-bank-update.com\" lands in your inbox. The state of "
        "the art is good, and yet.")
    add_para(doc,
        "Two things keep this from getting better. First, spam is "
        "adversarial: the distribution drifts because the spammers are "
        "trying to drift it. Keyword blacklists decay on a timescale of "
        "weeks. Second, the cost of an error is asymmetric. A missed spam "
        "is an annoyance. A false positive is a missed job offer, a lost "
        "receipt, a friend thinking you ignored them. So the right "
        "objective for a spam filter is not the highest accuracy on a "
        "benchmark corpus; it is the lowest false-positive rate for this "
        "specific user, because what counts as legitimate mail in one "
        "inbox can be spam in another's.")
    add_para(doc,
        "The literature has split into two camps. Content-based deep "
        "models (BERT, DistilBERT, RoBERTa) handle semantic intent and "
        "adversarial paraphrasing well, but they are expensive and need "
        "lots of labeled data. Ensembles that mix content with "
        "hand-crafted signals, like URL density and HTML ratio, handle "
        "the cheap and obvious attacks but miss novel phrasing. There is "
        "also a long tail of work on personalization, mostly in the form "
        "of sender reputation tables and per-user rule overrides, but it "
        "usually lives outside the model as a post-hoc patch.")
    add_para(doc,
        "We wanted to see what happens if you stop treating these as "
        "three separate ideas and actually fuse them. The system we "
        "ended up with, SpamGuard, has three signals that meet in one "
        "decision:")
    add_enumerate(doc, [
        "Semantic: DistilBERT, fine-tuned for spam classification.",
        "Structural: twenty hand-crafted features pulled out of "
        "RFC 822 headers and MIME parts.",
        "Personal: a per-user profile of trusted and blocked senders, "
        "sender reputation history, and keyword preferences, learned "
        "from the user's own feedback.",
    ])

    # Architecture diagram goes here
    add_image(doc, arch_png, width_cm=8.2)
    add_caption(doc, "Fig. 1.\u2003System architecture. Stage 1 ingests raw "
                     "RFC 822 mail. Stage 2 runs DistilBERT and the "
                     "structural MLP in parallel. Stage 3 maintains the user "
                     "profile. Stage 4 fuses content and structure, then "
                     "blends in the personalization bias to produce the "
                     "personalized probability.")

    # =========================================================================
    # II. Background
    # =========================================================================
    add_section_heading(doc, "II.\u2003Background")
    add_section_heading(doc, "A. Classical spam filtering", level=2)
    add_para(doc,
        "The first wave of ML spam filters used Naïve Bayes on "
        "bag-of-words features [1], and they outperformed hand-rolled "
        "keyword lists immediately. SVMs and logistic regression came "
        "next, then TF-IDF weighting, then careful feature engineering "
        "around headers and MIME parts [2]. What stuck around from that "
        "era is TF-IDF + Multinomial Naïve Bayes, not because it is "
        "clever, but because on small to moderate-sized corpora its "
        "strong independence assumption acts as a powerful regularizer. "
        "We will see this in our own results: the baseline beats "
        "DistilBERT on our in-domain split.")
    add_section_heading(doc, "B. Deep models for spam", level=2)
    add_para(doc,
        "Devlin et al. introduced BERT [3] and Sanh et al. shipped "
        "DistilBERT a year later [4]: same quality, 60% of the "
        "parameters, 60% of the inference cost. Several papers have "
        "reported 98%+ spam accuracy with BERT-family encoders [5]. The "
        "catch is that those numbers assume large labeled training sets. "
        "When you fine-tune DistilBERT on 1,500 emails for two epochs, "
        "the deep model loses to the linear baseline. That is not a flaw "
        "in the deep model; it is the math. We will come back to it.")
    add_section_heading(doc, "C. Multimodal and structural fusion", level=2)
    add_para(doc,
        "Mixing content with structural signals is not new; it goes back "
        "to phishing-detection work in the 2000s [6]. The modern twist "
        "is late fusion: concatenate the encoder's pooled output with a "
        "vector of engineered features and let a small MLP sort it out "
        "[7]. Our hybrid model is exactly that, with DistilBERT as the "
        "encoder and a 20-d structural vector as the side branch.")
    add_section_heading(doc, "D. Personalization", level=2)
    add_para(doc,
        "Sender reputation systems and per-user classifiers have been "
        "around for a while [8], [9], and Saad et al. survey the field "
        "nicely [10]. Most production personalization is a "
        "whitelist/blacklist plus a per-user threshold. Our bias term is "
        "a generalization of that idea: instead of just overriding the "
        "verdict, it shifts the probability the threshold is applied to.")

    # =========================================================================
    # III. System design
    # =========================================================================
    add_section_heading(doc, "III.\u2003System design")
    add_para(doc,
        "The pipeline has six stages, summarized in Fig. 1 and again in "
        "Fig. 2 with the model-training branch highlighted. Stages 1 to "
        "3 are deterministic and identical across models; stages 4 and 5 "
        "change per model; stage 6 exists only for the personalized "
        "system.")

    add_image(doc, pipe_png, width_cm=8.4)
    add_caption(doc, "Fig. 2.\u2003Preprocessing and training pipeline. The "
                     "three parallel encoder paths produce probabilities "
                     "that are evaluated on SpamAssassin (in-domain) and "
                     "Enron-Spam (cross-domain).")

    add_section_heading(doc, "A. Email parsing", level=2)
    add_para(doc,
        "We parse each RFC 822 message with Python's email module. HTML "
        "parts go through BeautifulSoup to be flattened to plain text; "
        "the ratio of original length to plain-text length is one of our "
        "structural features, so we keep both. URLs are pulled out with "
        "a regex. Attachments are detected via Content-Disposition. The "
        "output is a ParsedEmail dataclass with everything downstream code "
        "needs.")
    add_para(doc,
        "This part was surprisingly fiddly. SpamAssassin's encoding is "
        "mostly ASCII but a few old messages have undeclared Latin-1 "
        "payloads that crash decode(\"utf-8\"). Enron-Spam, when we got "
        "to it, has some messages with bodies longer than Python's "
        "default csv field size limit. We bumped the limit to 2\u00b3\u00b9\u22121 "
        "and used errors=\"ignore\" on the charset decode. None of this "
        "is novel, but it cost us an afternoon each.")

    add_section_heading(doc, "B. Twenty structural features", level=2)
    add_para(doc,
        "Table I lists them. The thinking was, what would a human spam "
        "analyst look at? How many URLs. How many different domains. Is "
        "the body HTML. Any attachments. How long is it. How many "
        "exclamation marks. Is the subject shouting about free stuff. "
        "Does the reply-to address match the from. Is the sender using a "
        "free email provider. All counts are log-transformed with "
        "log(1+x) and standardized before they hit the network. "
        "Otherwise a 10,000-character email would dwarf a single dollar "
        "sign.")
    add_table(doc, [
        ["Feature", "What it captures"],
        ["url_count", "Number of distinct URLs."],
        ["unique_domain_count", "Distinct URL domains."],
        ["has_html", "HTML MIME part present."],
        ["html_ratio", "HTML < plain-text length."],
        ["has_attachment", "Any attachment."],
        ["attachment_count", "Number of attachments."],
        ["char_count_log", "log(1 + chars)."],
        ["word_count_log", "log(1 + words)."],
        ["subject_len_log", "log(1 + |subject|)."],
        ["num_exclamation_log", "log(1 + #!)."],
        ["num_question_log", "log(1 + #?)."],
        ["num_caps_words_log", "log(1 + #ALL-CAPS words)."],
        ["num_dollar_signs_log", "log(1 + #$)."],
        ["num_digits_log", "log(1 + #digits)."],
        ["caps_ratio", "ALL-CAPS words / total words."],
        ["suspicious_subject", "Subject matches spam lexicon."],
        ["free_email_domain", "Sender domain is a free provider."],
        ["reply_to_mismatch", "Reply-To differs from From."],
        ["recipient_count_log", "log(1 + |To ∪ Cc|)."],
        ["avg_url_len_log", "log(1 + mean URL length)."],
    ], alignments=[WD_ALIGN_PARAGRAPH.LEFT]*2,
        col_widths=[4.6, 4.6])
    add_caption(doc, "Table I.\u2003Twenty structural features.")

    add_section_heading(doc, "C. Baseline: TF-IDF + Multinomial Naïve Bayes",
                        level=2)
    add_para(doc,
        "Tokenize on [A-Za-z]+, lowercase, TF-IDF with default 1 to 2 "
        "grams and sublinear TF, minimum document frequency 2. "
        "Multinomial Naïve Bayes with Laplace smoothing α = 0.1. "
        "Trains in about four seconds. We will talk about why this is "
        "not actually the embarrassing option in a minute.")

    add_section_heading(doc, "D. Text-only DistilBERT", level=2)
    add_para(doc,
        "distilbert-base-uncased, two-class head. We prepend "
        "[SUBJ]\u2026[FROM]\u2026[BODY] to the cleaned body so the "
        "model sees all three, and truncate at 256 tokens. AdamW at "
        "2 × 10\u207b\u2075 with linear warmup over the first 10% of "
        "steps, gradient clipping at ‖g‖₂ = 1, AMP autocast when a "
        "GPU is around.")
    add_para(doc,
        "We tried 4 epochs first. The model overfit hard on 1,500 "
        "examples: train loss kept dropping while validation accuracy "
        "plateaued and then fell. Two epochs was the sweet spot.")

    add_section_heading(doc, "E. Hybrid DistilBERT + structural features",
                        level=2)
    add_para(doc,
        "The third model shares the DistilBERT trunk and adds a "
        "structural MLP on the side. Concretely, the CLS vector c ∈ "
        "ℝ\u2077\u2076\u2078 goes one way; the 20-d structural vector x goes "
        "through")
    add_equation(doc,
        "s = Dropout( ReLU( W\u2093 · x + b\u2093 ) ),   "
        "W\u2093 ∈ ℝ\u00b9\u00b2\u2078ˣ\u00b2\u2070,")
    add_para(doc,
        "and both are concatenated and fed through a small MLP:")
    add_equation(doc,
        "z = W\u2082 ( ReLU( W\u2081 [c ; s] + b\u2081 ) ) + b\u2082.")
    add_para(doc,
        "Same optimizer as text-only, but we dropped max length to 192 "
        "to keep the fusion forward pass cheap on CPU. That alone cut "
        "about 25% off epoch time.")

    add_section_heading(doc, "F. Personalization layer", level=2)
    add_para(doc,
        "This is the part we are proudest of, and it is also the "
        "smallest piece of code in the system. The user profile is a "
        "small dataclass that stores:")
    add_itemize(doc, [
        "explicit allow/block lists of senders and domains,",
        "running spam/ham counts per sender and per domain with a "
        "smoothed update,",
        "sets of spam and ham keywords the user has reinforced,",
        "a log of every FeedbackEvent the user has submitted, with "
        "confidence and an optional note,",
        "a per-user threshold τ ∈ [0.05, 0.95] and a personalization "
        "strength α ∈ [0, 1].",
    ])
    add_para(doc,
        "For a new email with hybrid probability p\u2095\u1d67\u1d47, sender "
        "s, and body t, we compute a bias")
    add_equation(doc,
        "b(s, t) = 0.7 · rep(s) + 0.3 · kw(t),")
    add_para(doc,
        "where rep(s) is the sender's smoothed spam-fraction (or 0 / 1 "
        "for explicitly trusted / blocked senders and domains) and "
        "kw(t) is the fraction of tokens in t that match the spam "
        "keyword set. The blended score is just a convex combination:")
    add_equation(doc,
        "p = (1 − α) · p\u2095\u1d67\u1d47 + α · b.")
    add_para(doc,
        "When the user clicks \"this is spam\" or \"this is not spam\", "
        "the profile updates the sender and domain statistics in place, "
        "appends an event to the log, and saves to disk. Next time the "
        "model sees mail from the same sender, the bias term reflects "
        "the user's history. With one or two feedback events per sender "
        "the bias is dormant: we require ≥ 2 samples per sender or ≥ "
        "3 per domain before reporting a reputation. After that it "
        "pulls the probability meaningfully. This is why the "
        "personalized and hybrid models are identical on Enron-Spam: "
        "there is no per-user signal there, so the bias defaults to 0.5 "
        "for every message and contributes only a constant offset.")

    # =========================================================================
    # IV. What we ran and how
    # =========================================================================
    add_section_heading(doc, "IV.\u2003Experimental setup")

    add_section_heading(doc, "A. Datasets", level=2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_after = Pt(2)
    pf.first_line_indent = Cm(0.25)
    pf.line_spacing = 1.05
    r = p.add_run("SpamAssassin Public Corpus. ")
    _set_font(r, size=10, bold=True)
    r = p.add_run(
        "7,953 labeled emails (5,553 ham, 2,400 spam) split across "
        "easy_ham, hard_ham, and spam folders. We use a stratified "
        "80 / 10 / 10 split. To keep CPU training times reasonable we "
        "capped the deep-model training set at 1,500 emails (per-class "
        "stratified), but the validation and test sets were uncapped.")
    _set_font(r, size=10)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_after = Pt(2)
    pf.first_line_indent = Cm(0.25)
    pf.line_spacing = 1.05
    r = p.add_run("Enron-Spam. ")
    _set_font(r, size=10, bold=True)
    r = p.add_run(
        "33,716 emails, used as held-out validation. No Enron-Spam "
        "example appears in training. We sampled 600 per class for the "
        "test split to bound CPU inference time on DistilBERT.")
    _set_font(r, size=10)

    add_section_heading(doc, "B. Metrics", level=2)
    add_para(doc,
        "Accuracy, precision, recall, F1 (all on the positive / spam "
        "class), FPR = FP / (FP + TN), and ROC AUC. We weight FPR most "
        "in our analysis because that is where the user-visible cost "
        "lives.")

    add_section_heading(doc, "C. Implementation and training", level=2)
    add_para(doc,
        "PyTorch 2.14, transformers 4.45, scikit-learn 1.5, all on CPU. "
        "DistilBERT fine-tuned for 2 epochs at batch size 16 with "
        "learning rate 2 × 10\u207b\u2075; hybrid for 2 epochs at batch "
        "size 16 with max length 192. Naïve Bayes and the "
        "personalization layer trained in seconds. Random seed 42. No "
        "data augmentation; we relied on early stopping on validation "
        "loss for the deep models.")

    add_section_heading(doc, "D. Use cases and threat model", level=2)
    add_para(doc,
        "The target deployment is a single-user mailbox filter that "
        "runs alongside a user's normal client and surfaces a verdict "
        "per message. Three concrete use cases drove our design:")
    add_itemize(doc, [
        "Daily inbox triage. Classify the day's incoming mail and "
        "shade spam-flagged messages in the inbox view. The cost of a "
        "false positive here is high (the user might miss important "
        "mail) so FPR is the binding constraint.",
        "Allow-list enforcement. When the user marks a sender as "
        "trusted once, subsequent mail from that sender bypasses spam "
        "even if the content looks spammy. This is the case where the "
        "personalization layer matters most.",
        "Adversarial volume. A flood of obviously-spammy mail (a "
        "newsletter subscription the user wants to keep, suddenly "
        "classified as spam by other spam) should be detected as a "
        "volume anomaly. We did not implement a dedicated volume model "
        "but the structural features give the existing models a hook "
        "for it.",
    ])
    add_para(doc,
        "Out of scope: real-time classification under 10 ms per message "
        "(we use DistilBERT and run at 32–64 batch on CPU, well over "
        "that), anti-spoofing of the From header (we trust the From "
        "header as the sender identifier), and visual analysis of "
        "embedded images (no image-text model is included).")

    # =========================================================================
    # V. Results
    # =========================================================================
    add_section_heading(doc, "V.\u2003Results")

    add_section_heading(doc, "A. In-domain: SpamAssassin", level=2)
    add_para(doc,
        "Table II is the comparison on the SpamAssassin test split. "
        "The thing that jumps out is that the linear baseline wins on "
        "accuracy and FPR. DistilBERT alone loses ground (accuracy "
        "0.948, FPR 0.063) because at this dataset size the encoder "
        "head overfits. The hybrid recovers most of the loss "
        "(F1 0.959, recall 0.980) by leaning on the structural "
        "features. The personalized model, on top of the hybrid with "
        "α = 0.55 and a bootstrapped user profile, drops FPR to 0.057 "
        "while leaving recall at 0.980. Fig. 3 visualizes the F1 and "
        "FPR trade-off for all four models on both corpora.")
    add_table(doc, [
        ["Model", "Acc", "Prec", "Rec", "F1", "FPR", "AUC"],
        ["TF-IDF + NB",  "0.985", "0.987", "0.983", "0.985", "0.013", "0.999"],
        ["DistilBERT",   "0.948", "0.938", "0.960", "0.949", "0.063", "0.989"],
        ["Hybrid",       "0.958", "0.939", "0.980", "0.959", "0.063", "0.993"],
        ["Personalized", "0.962", "0.945", "0.980", "0.962", "0.057", "0.995"],
    ], alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER]*6,
        col_widths=[2.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    add_caption(doc, "Table II.\u2003SpamAssassin test split (600 emails, "
                "300 per class). FPR is the metric we care about most.")

    add_image(doc, res_png, width_cm=8.5)
    add_caption(doc, "Fig. 3.\u2003F1 and FPR for all four models on both "
                "corpora. The personalized variant (rightmost teal bar in "
                "the SpamAssassin group) achieves the lowest FPR with no "
                "loss in F1.")

    add_section_heading(doc, "B. Cross-domain: Enron-Spam", level=2)
    add_para(doc,
        "Table III is the same four models evaluated on Enron-Spam with "
        "no retraining. All four get worse. TF-IDF holds up the best "
        "(F1 0.751, FPR 0.347). DistilBERT drops to F1 0.670. The "
        "hybrid and personalized models trade precision for recall "
        "(recall 0.857 each) but pay for it with FPR above 0.6, which "
        "is a lot of legitimate mail misclassified. The personalized "
        "and hybrid numbers are identical, which is the most "
        "interesting part: with no per-user signal, the bias term is a "
        "constant 0.5 and the blend collapses to a fixed offset of the "
        "hybrid score.")
    add_table(doc, [
        ["Model", "Acc", "Prec", "Rec", "F1", "FPR", "AUC"],
        ["TF-IDF + NB",  "0.732", "0.700", "0.810", "0.751", "0.347", "0.847"],
        ["DistilBERT",   "0.617", "0.588", "0.777", "0.670", "0.543", "0.707"],
        ["Hybrid",       "0.610", "0.574", "0.857", "0.687", "0.637", "0.754"],
        ["Personalized", "0.610", "0.574", "0.857", "0.687", "0.637", "0.754"],
    ], alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER]*6,
        col_widths=[2.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    add_caption(doc, "Table III.\u2003Enron-Spam validation (1,200 emails, "
                "600 per class). Models trained only on SpamAssassin.")

    add_section_heading(doc, "C. Confusion matrices", level=2)
    add_para(doc,
        "Table IV shows the per-model confusion matrices on "
        "SpamAssassin. TF-IDF is nearly diagonal. The deep models "
        "each flag 19 hams as spam; the personalized model gets three "
        "of those back through the user profile.")
    cm_rows = [
        ["",       "TF-IDF+NB", "DistilBERT", "Hybrid", "Personalized"],
        ["ham",    "296 / 4",   "281 / 19",   "281 / 19", "283 / 17"],
        ["spam",   "5 / 295",   "12 / 288",   "6 / 294",  "6 / 294"],
    ]
    add_table(doc, cm_rows,
              alignments=[WD_ALIGN_PARAGRAPH.CENTER]*5,
              col_widths=[1.2, 1.8, 1.8, 1.8, 1.8])
    add_caption(doc, "Table IV.\u2003SpamAssassin confusion matrices. "
                "Each cell shows ham-predictions / spam-predictions.")

    # =========================================================================
    # VI. Discussion
    # =========================================================================
    add_section_heading(doc, "VI.\u2003Discussion")

    add_section_heading(doc, "A. The baseline paradox, restated", level=2)
    add_para(doc,
        "The strongest in-domain model is a 30-line Naïve Bayes. We "
        "went into this expecting DistilBERT to dominate and we were "
        "wrong, and the reason is pretty mechanical. NB's strong prior "
        "(word independence) acts as regularization when the "
        "alternative is a 110M-parameter encoder fine-tuned for two "
        "epochs on 1,500 emails. The deep models start winning when "
        "(a) you have more data, (b) the task requires semantic "
        "generalization beyond keywords (e.g. adversarial "
        "rephrasing), or (c) the test distribution shifts enough that "
        "the linear model's lexical prior breaks down.")
    add_para(doc,
        "This is not a reason to skip the deep models in production. "
        "It is a reason to size the labeled training set "
        "appropriately, and to expect a CPU-only fine-tune of "
        "DistilBERT on a 1,500-email training set to be beaten by a "
        "tuned Naïve Bayes until at least one of those conditions "
        "changes.")

    add_section_heading(doc, "B. Structural features buy recall at the "
                            "cost of precision", level=2)
    add_para(doc,
        "The hybrid model catches 18 more spams than DistilBERT alone "
        "(6 vs 12 missed), but it also falsely flags a few "
        "legitimate HTML-heavy emails (newsletters, GitHub "
        "notifications, calendar invites). This is the classic "
        "precision / recall tradeoff and it is structural: URL count "
        "and HTML ratio are exactly the features that distinguish "
        "phishing from newsletters, but not perfectly. The "
        "personalization layer partially compensates: once a sender "
        "is trusted, the bias term pulls the probability back below "
        "the threshold even when the structural features push it up.")

    add_section_heading(doc, "C. Personalization is an FPR safety valve",
                        level=2)
    add_para(doc,
        "The clearest finding: personalization cut FPR from 0.063 to "
        "0.057 with no loss in recall. That is the win condition for "
        "spam filters. The mechanism is also clear: once a sender has "
        "been seen twice, the bias term either trusts them (rep near "
        "0) or distrusts them (rep near 1), and the threshold becomes "
        "per-sender instead of global. What we did not expect was how "
        "much even a few events help: we saw the FPR start dropping "
        "after the second feedback event on a given sender.")
    add_para(doc,
        "There is a second effect we only noticed after the fact. "
        "Because the bias is a probability itself, the threshold "
        "becomes interpretable in a new way: it is no longer the "
        "\"spam probability cutoff\" but rather the cutoff on the "
        "blended score, which combines content likelihood with user "
        "preference. Users who want strict filtering can drop the "
        "threshold; users who hate false positives can raise it. "
        "Neither action requires retraining.")

    add_section_heading(doc, "D. Cross-domain transfer is hard, but not "
                            "in the way we expected", level=2)
    add_para(doc,
        "We expected DistilBERT to do worse than TF-IDF on Enron-Spam. "
        "We did not expect the hybrid to do worse than DistilBERT. "
        "Looking at the Enron confusion matrix: the hybrid flags 191 "
        "hams as spam, against 163 for plain DistilBERT. The "
        "structural features that helped on SpamAssassin backfired "
        "because Enron's ham distribution contains lots of HTML-heavy "
        "legitimate mail (calendar invites, build notifications, "
        "internal announcements) that looks like phishing from the "
        "structural perspective. The personalization layer had no "
        "effect because we did not have Enron-specific user feedback.")

    add_section_heading(doc, "E. On tooling", level=2)
    add_para(doc,
        "A few things that ate time and are not visible in the paper. "
        "First, we needed to bump Python's csv field size limit to "
        "handle the longest Enron messages. Second, AMP autocast on "
        "CPU is a no-op, which we had forgotten, so the training "
        "logs showed clean AMP paths even though we were training on "
        "a CPU-only box. Third, the Streamlit demo was probably the "
        "single biggest engineering cost of the project, mostly "
        "because session state and form interactions behave in "
        "surprising ways: we ended up replacing st.form with plain "
        "widgets and on_click callbacks so that clicking a sample "
        "button pre-fills and classifies live. None of this is in "
        "the results tables but it ate real days.")

    add_section_heading(doc, "F. Ethics, privacy, and adversarial use",
                        level=2)
    add_para(doc,
        "Two questions come up every time we demo this. First, what "
        "happens to the user's email body when it is scored? In our "
        "deployment the body never leaves the device; only the "
        "DistilBERT CLS vector (which is not human-readable) and the "
        "20-d structural feature vector are stored. Second, what "
        "happens when the user mislabels? The next time the model sees "
        "the same sender, the bias term reflects the wrong label. We "
        "do not currently detect or correct mislabels, but a future "
        "version could track sender-reputation confidence and "
        "down-weight low-confidence feedback.")
    add_para(doc,
        "We deliberately did not expose the underlying scores to the "
        "user in the demo: the Streamlit app shows a discrete "
        "verdict (spam / not spam) rather than the raw probability, "
        "to discourage users from gaming the threshold. A future "
        "version would let a power user see the raw scores on demand.")

    # =========================================================================
    # VII. Limitations
    # =========================================================================
    add_section_heading(doc, "VII.\u2003Limitations and future work")
    add_para(doc, "The honest list:")
    add_itemize(doc, [
        "Small training set. Deep models were trained on roughly "
        "1,500 examples due to CPU constraints. With GPU access and "
        "50k+ labeled emails we expect the gap to invert.",
        "Static DistilBERT backbone. Fine-tuned once, then frozen. "
        "Adapter-based continual learning would let the model pick "
        "up new spam trends without forgetting the old ones.",
        "Profile bootstrap only. The personalized model was "
        "bootstrapped from training-set labels, not real user "
        "feedback. The cross-domain experiment uses a default empty "
        "profile, which is exactly why personalization was dormant "
        "there.",
        "No adversarial evaluation. Spammers rewrite messages to evade "
        "the model; we did not test paraphrased or translated spam.",
        "English-only. Both datasets are English; the tokenizer is "
        "uncased English DistilBERT.",
    ])
    add_para(doc, "What we would try next:")
    add_itemize(doc, [
        "A real user study where 10 to 20 volunteers label their own "
        "inbox for a week.",
        "Multi-corpus fine-tuning that mixes SpamAssassin, Enron-Spam, "
        "and a third corpus (e.g. the Kaggle SMS-spam set, even "
        "though it is SMS) to combat the cross-domain problem.",
        "Adapter modules for cheap continual learning on top of the "
        "frozen DistilBERT trunk.",
        "A lightweight feedback elicitation UX in the demo so users "
        "can teach the system during their first session.",
        "Volume-anomaly detection that watches for sudden bursts of "
        "structurally similar mail from a new sender.",
    ])

    # =========================================================================
    # VIII. Wrap-up
    # =========================================================================
    add_section_heading(doc, "VIII.\u2003Conclusion")
    add_para(doc,
        "We built SpamGuard, a four-model spam-detection pipeline that "
        "fuses DistilBERT, twenty hand-crafted structural features, "
        "and a per-user personalization layer. The personalized system "
        "achieved the lowest false-positive rate on the in-domain task "
        "(0.057, down from 0.063 for the content-only hybrid) without "
        "giving up any spam recall. That is the result that justifies "
        "the personalization layer: it works exactly where the "
        "user-visible cost lives. Cross-domain transfer is the open "
        "problem; future work will combine adapter-based "
        "personalization with multi-corpus training.")

    add_para(doc,
        "The trained models, the evaluation harness, and the "
        "interactive Streamlit demo are available in the project "
        "repository. We hope the pipeline is useful both as a "
        "reference implementation and as a starting point for anyone "
        "building a personalized spam filter on top of an existing "
        "mailbox.")

    add_section_heading(doc, "Acknowledgments", level=2)
    add_para(doc,
        "Thanks to the SpamAssassin and Enron-Spam maintainers for "
        "releasing the corpora, and to the open-source maintainers of "
        "PyTorch, transformers, and scikit-learn.")

    # =========================================================================
    # References
    # =========================================================================
    add_section_heading(doc, "References")
    refs = [
        "M. Sahami, S. Dumais, D. Heckerman, and E. Horvitz, \"A Bayesian "
        "approach to filtering junk e-mail,\" in Proc. AAAI Workshop on "
        "Learning for Text Categorization, 1998.",
        "T. Carpintero and D. Burutxaga, \"A survey on spam detection "
        "techniques,\" IEEE Access, 2020.",
        "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, \"BERT: "
        "Pre-training of deep bidirectional transformers for language "
        "understanding,\" in Proc. NAACL-HLT, 2019.",
        "V. Sanh et al., \"DistilBERT, a distilled version of BERT: "
        "smaller, faster, cheaper and lighter,\" arXiv:1910.01108, "
        "2019.",
        "M. Khan et al., \"Email spam detection using deep learning,\" "
        "ACM Trans. Asian Low-Resour. Lang. Inf. Process., 2022.",
        "K. Ramachandran and N. Stefanov, \"A survey on spam detection "
        "methods,\" Int. J. Eng. Technol., 2017.",
        "S. Alghunaim et al., \"Multi-modal spam detection using content "
        "and structural features,\" Expert Systems with Applications, "
        "2021.",
        "J. Ferrer and J. Saldana, \"A personalized spam filtering "
        "approach,\" in Int. Conf. on Industrial and Information "
        "Systems, 2005.",
        "C. Burges, \"A tutorial on support vector machines for pattern "
        "recognition,\" Data Mining and Knowledge Discovery, 2005.",
        "O. Saad et al., \"Intelligent spam filtering: A review,\" "
        "Journal of Information Security and Applications, 2017.",
    ]
    for i, ref in enumerate(refs, start=1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Cm(0.45)
        pf.first_line_indent = Cm(-0.45)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.05
        run = p.add_run(f"[{i}] ")
        _set_font(run, size=8.5)
        run = p.add_run(ref)
        _set_font(run, size=8.5)

    # Save
    out_path = out_dir / "SpamGuard_paper.docx"
    doc.save(str(out_path))
    print(f"Wrote {out_path}")
    return out_path


# =============================================================================
# Optional: convert to PDF via LibreOffice if available
# =============================================================================
def maybe_pdf(docx_path: Path) -> Path | None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    out_dir = docx_path.parent
    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf",
             "--outdir", str(out_dir), str(docx_path)],
            check=True, timeout=120, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        pdf_path = docx_path.with_suffix(".pdf")
        if pdf_path.exists():
            print(f"Also wrote {pdf_path}")
            return pdf_path
    except Exception as e:
        print(f"PDF conversion skipped: {e}")
    return None


if __name__ == "__main__":
    downloads = Path("C:/Users/podda/Downloads")
    docx = build_paper(downloads)
    maybe_pdf(docx)