"""Humanize the Poddar & Lakshmi IEEE paper and re-emit it as DOCX.

Preserves:
  * IEEE two-column format (Times New Roman, 10pt body)
  * All 4 figures (extracted from the source PDF)
  * All 5 tables (Table I through Table V) with original content
  * All 23 references
  * Section structure and math notation

Humanizes:
  * Rewrites the prose in a natural, varied voice
  * Removes every em-dash and double-hyphen
  * Avoids AI-detection tells (no "this paper presents", no "it is
    worth noting", no perfect parallelism, varied sentence lengths)

Output:
  C:/Users/podda/Downloads/IEEE_Paper_Personalized_Email_Spam_Filtering_humanized.docx
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


# =============================================================================
# Low-level XML helpers
# =============================================================================
def _set_cols(section, num_cols, space_twips=510, equal_width=True):
    sect_pr = section._sectPr
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
              color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
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


# =============================================================================
# Paragraph / table helpers
# =============================================================================
def add_para(doc, text="", *, size=10, bold=False, italic=False,
             alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
             space_before=0, space_after=2, first_line_indent=0.25,
             line_spacing=1.05):
    p = doc.add_paragraph()
    p.alignment = alignment
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if first_line_indent:
        pf.first_line_indent = Cm(first_line_indent)
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
        _set_font(run, size=10, bold=True)
    else:
        _set_font(run, size=10, italic=True)
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


def add_itemize(doc, items, indent=0.4, size=10):
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
        _set_font(run, size=size)


def add_enumerate(doc, items, size=10):
    for i, item in enumerate(items, start=1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Cm(0.7)
        pf.first_line_indent = Cm(-0.5)
        pf.space_before = Pt(0)
        pf.space_after = Pt(1)
        pf.line_spacing = 1.05
        run = p.add_run(f"({i})\t")
        _set_font(run, size=size, italic=True)
        run = p.add_run(item)
        _set_font(run, size=size)


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
              header_size=8.5, body_size=8.5, header_fill="EFEFEF"):
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
                _shade_cell(cell, header_fill)
            else:
                _set_font(run, size=body_size)
            for edge in ("top", "left", "bottom", "right"):
                _set_cell_border(
                    cell, **{edge: {"sz": 4, "val": "single", "color": "404040"}}
                )
    return table


# =============================================================================
# Helpers for inline runs (so we can do bold / italic / math mixed in prose)
# =============================================================================
def add_rich_para(doc, runs, *, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
                  first_line_indent=0.25, space_after=2, line_spacing=1.05):
    """runs: list of (text, {kwargs})."""
    p = doc.add_paragraph()
    p.alignment = alignment
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    if first_line_indent:
        pf.first_line_indent = Cm(first_line_indent)
    pf.line_spacing = line_spacing
    for text, kw in runs:
        run = p.add_run(text)
        _set_font(run, size=kw.get("size", 10),
                  bold=kw.get("bold", False),
                  italic=kw.get("italic", False))
    return p


# =============================================================================
# Build the paper
# =============================================================================
def build_paper(fig_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = Document()

    section = doc.sections[0]
    section.page_height = Cm(27.94)
    section.page_width = Cm(21.59)
    section.top_margin = Cm(1.27)
    section.bottom_margin = Cm(1.27)
    section.left_margin = Cm(1.27)
    section.right_margin = Cm(1.27)
    _set_cols(section, num_cols=1)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)

    # ---- Title block (full width) ----
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = title.paragraph_format
    pf.space_after = Pt(4)
    run = title.add_run(
        "Filtering Spam in Personalized Email Systems: A Context-Aware "
        "Hybrid Framework Using DistilBERT Semantic Embeddings, "
        "Structural Features and User Feedback"
    )
    _set_font(run, size=18, bold=True)

    # ---- Author 1 ----
    au1 = doc.add_paragraph()
    au1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = au1.paragraph_format
    pf.space_after = Pt(0)
    run = au1.add_run("Muskan Poddar")
    _set_font(run, size=11, bold=False)
    aff1 = doc.add_paragraph()
    aff1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = aff1.paragraph_format
    pf.space_after = Pt(0)
    for ln in [
        "Department of Computer Science and Engineering",
        "Amrita School of Computing, Amrita Vishwa Vidyapeetham",
        "Chennai, India",
    ]:
        run = aff1.add_run(ln + "\n")
        _set_font(run, size=10, italic=True)
    roll = doc.add_paragraph()
    roll.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = roll.paragraph_format
    pf.space_after = Pt(4)
    run = roll.add_run("Roll No. CH.SC.U4CSE23266")
    _set_font(run, size=10, italic=True)

    # ---- Author 2 ----
    au2 = doc.add_paragraph()
    au2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = au2.paragraph_format
    pf.space_after = Pt(0)
    run = au2.add_run("Sri Lakshmi")
    _set_font(run, size=11)
    aff2 = doc.add_paragraph()
    aff2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = aff2.paragraph_format
    pf.space_after = Pt(0)
    for ln in [
        "Department of Computer Science and Engineering",
        "Amrita School of Computing, Amrita Vishwa Vidyapeetham",
        "Chennai, India",
    ]:
        run = aff2.add_run(ln + "\n")
        _set_font(run, size=10, italic=True)
    roll = doc.add_paragraph()
    roll.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = roll.paragraph_format
    pf.space_after = Pt(8)
    run = roll.add_run("Roll No. CH.SC.U4CSE23230")
    _set_font(run, size=10, italic=True)

    # ---- Switch to two-column body ----
    body_section = doc.add_section(WD_SECTION.CONTINUOUS)
    body_section.page_height = section.page_height
    body_section.page_width = section.page_width
    body_section.top_margin = section.top_margin
    body_section.bottom_margin = section.bottom_margin
    body_section.left_margin = section.left_margin
    body_section.right_margin = section.right_margin
    _set_cols(body_section, num_cols=2, space_twips=510)

    # =========================================================================
    # Abstract (humanized; no em-dash)
    # =========================================================================
    add_rich_para(doc, [
        ("Abstract", {"bold": True, "italic": True}),
        (": ", {"bold": True}),
        ("Most spam filters are trained once and then applied to every "
         "recipient in exactly the same way, even though what counts as "
         "spam depends on who is reading the message. A newsletter that one "
         "user wants is noise to another. A message from an unfamiliar "
         "domain may be phishing for a student and a legitimate enquiry "
         "for a researcher. A single global classifier cannot resolve "
         "this, and filters that ignore the recipient also fail to react "
         "quickly when spam campaigns shift. We built a context-aware, "
         "personalized spam filtering framework that draws on three "
         "complementary signals: (i) a 768-dimensional contextual "
         "embedding of the subject and body produced by a pre-trained "
         "DistilBERT encoder; (ii) structural features pulled out of "
         "sender, URL, attachment, HTML, subject and header metadata, "
         "including SPF and DKIM authentication results; and (iii) "
         "user-specific context such as trusted senders, sender "
         "reputation, interaction behaviour and explicit feedback. A "
         "hybrid classifier fuses the first two into a spam probability, "
         "and a personalization layer blends that probability with a "
         "user score through an adaptive weight that depends on how "
         "much feedback the user has provided, which also handles the "
         "cold-start case. A feedback loop updates user profiles, sender "
         "reputation and per-user decision thresholds, with optional "
         "periodic model refresh to follow concept drift. We describe "
         "the architecture, data flow, mathematical formulation and an "
         "evaluation protocol built on the Enron and SpamAssassin "
         "corpora, with baselines, ablations and an adaptation study. "
         "The framework targets a higher F1 score and a lower "
         "false-positive rate than text-only or non-personalized filters "
         "while remaining light enough to run on practical hardware.",
         {}),
    ], first_line_indent=0)

    # ---- Index Terms ----
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0.25)
    pf.line_spacing = 1.05
    pf.space_after = Pt(6)
    r = p.add_run("Index Terms: ")
    _set_font(r, size=9, bold=True, italic=True)
    r = p.add_run("spam filtering, personalization, DistilBERT, hybrid "
                  "classifier, structural features, feedback learning, "
                  "concept drift, email security.")
    _set_font(r, size=9, italic=True)

    # =========================================================================
    # I. Introduction
    # =========================================================================
    add_section_heading(doc, "I.\u2003Introduction")
    add_para(doc,
        "Email is still one of the most widely used channels for personal, "
        "academic and professional communication, and it remains just as "
        "attractive to attackers. Unsolicited bulk mail, phishing and "
        "malware delivery waste user time, consume network resources and "
        "expose recipients to fraud [1], [2]. Filtering has moved on from "
        "hand-written rules and blacklists through statistical learning "
        "to, more recently, transformer language models [3], [4].")
    add_para(doc,
        "Even so, most published and deployed filters share one "
        "assumption: whether a message is spam is independent of its "
        "recipient. In practice the boundary is user-dependent. A "
        "promotional newsletter is noise to one recipient and a valued "
        "subscription to another; a message from an unfamiliar domain "
        "may be phishing for a student yet a legitimate enquiry for a "
        "researcher. A single global classifier must therefore trade "
        "between two kinds of error, letting spam through (false "
        "negatives) and diverting legitimate mail (false positives). "
        "The latter is usually more costly, because users rarely inspect "
        "their spam folders.")
    add_para(doc,
        "Two further limitations motivate this work. First, content-only "
        "models ignore structural evidence such as sender authentication "
        "results, URL characteristics, attachment types and HTML "
        "composition, which are hard for an attacker to forge "
        "consistently alongside fluent text. Second, static models "
        "degrade as spam campaigns change, a form of concept drift [5], "
        "and they do not exploit the most direct signal available: what "
        "each user does with their mail.")
    add_para(doc,
        "This paper proposes a context-aware personalized spam filtering "
        "framework that addresses these gaps; Fig. 1 gives an overview "
        "of the architecture. Its contributions are:")
    add_enumerate(doc, [
        "A hybrid representation that fuses a 768-dimensional DistilBERT "
        "embedding of the subject and body [6] with a structural feature "
        "vector extracted from email metadata.",
        "A personalization layer that blends the global spam probability "
        "with a user-specific score built from sender reputation, "
        "interaction behaviour and explicit preferences, using an "
        "adaptive weight that reduces to the global model for new users.",
        "A feedback-driven adaptation mechanism that updates user "
        "profiles, sender reputation and per-user decision thresholds, "
        "with optional periodic model refresh to follow concept drift.",
        "A complete system architecture and data-flow design, together "
        "with an evaluation protocol containing baselines, ablations "
        "and a personalization study.",
    ])
    add_para(doc,
        "The rest of the paper is organized as follows. Section II "
        "reviews related work. Section III formalizes the problem. "
        "Section IV details the proposed methodology and Section V the "
        "system design. Section VI describes the experimental design, "
        "Section VII discusses expected benefits, limitations and "
        "deployment considerations, and Section VIII concludes.")

    # =========================================================================
    # II. Related work
    # =========================================================================
    add_section_heading(doc, "II.\u2003Related work")

    add_section_heading(doc, "A. Rule-based and classical machine learning", level=2)
    add_para(doc,
        "Early filters relied on manually defined keyword rules, "
        "heuristics and blacklists. They are simple and fast but need "
        "constant manual maintenance and cannot recognize novel or "
        "obfuscated messages. Statistical filters replaced hand-written "
        "rules with models learned from labelled mail. Sahami et al. "
        "introduced a Bayesian approach to junk-mail filtering [7], "
        "Androutsopoulos et al. evaluated naive Bayes on public corpora "
        "[3], and Metsis et al. compared variants of the naive Bayes "
        "family [8]. Systematic reviews summarize the resulting body of "
        "work on naive Bayes, support vector machines, neural networks "
        "and ensembles [1], [9]. Later comparative studies report that "
        "random forests and logistic regression are strong on TF-IDF "
        "features [10], but these models depend on feature engineering, "
        "perform differently across datasets and have a limited grasp of "
        "context.")

    add_section_heading(doc, "B. Deep learning and transformer models", level=2)
    add_para(doc,
        "Deep models learn feature representations directly from text. "
        "Convolutional and recurrent architectures capture local and "
        "sequential patterns [11], and the survey of Dada et al. "
        "identifies deep learning as a promising direction while noting "
        "its need for large labelled datasets and higher computational "
        "cost [2]. Self-attention [12] and pre-trained encoders such as "
        "BERT [4] further improved contextual understanding. Sahmoud "
        "and Mikki applied BERT to spam detection and obtained higher "
        "classification performance than classical baselines [13]. "
        "Labonne and Moran benchmarked transformer and large language "
        "models for few-shot spam detection [14], and Shirvani and "
        "Ghasemshirazi paired BERT with FLAN-T5 [15] for zero-shot, "
        "adaptive detection [16]. The recurring drawback is cost: such "
        "models need substantial memory and compute and are harder to "
        "fine-tune and deploy. DistilBERT, obtained by knowledge "
        "distillation, is reported to be 40% smaller and 60% faster than "
        "BERT while retaining about 97% of its language-understanding "
        "performance [6], which motivates its use here. Table I "
        "summarizes the key studies.")
    add_table(doc, [
        ["Year", "Study", "Technique / model", "Dataset / scope",
         "Key findings", "Limitations"],
        ["2008", "Cormack [1]", "Statistical and ML approaches "
         "(naive Bayes, SVM)", "Multiple public datasets (Enron, "
         "Nazario)", "Comprehensive review of filtering techniques; ML "
         "effective in real-world scenarios", "Spam patterns evolve; "
         "continuous updates required for sustained performance"],
        ["2009", "Guzella and Caminhas [9]", "Naive Bayes, SVM, neural "
         "networks, ensembles", "Public datasets (Enron, "
         "SpamAssassin)", "Good accuracy at low complexity; ensembles "
         "improve performance", "Struggles with new patterns; limited "
         "contextual understanding"],
        ["2019", "Dada et al. [2]", "Traditional ML and deep learning",
         "Enron, Ling-Spam, SpamAssassin", "Deep learning improves on "
         "classical ML; promising research direction", "Requires large "
         "labelled datasets; higher computational cost"],
        ["2021", "Saxena et al. [10]", "NB, KNN, SVM, LR, decision tree, "
         "random forest", "Public email datasets (Enron, "
         "SpamAssassin)", "Random forest and logistic regression "
         "strong; classical ML remains effective", "Feature engineering "
         "required; results vary across datasets"],
        ["2022", "Sahmoud and Mikki [13]", "BERT (transformer)", "Enron, "
         "SpamAssassin, TREC", "Captures contextual meaning; higher "
         "performance than classical ML", "High compute and memory "
         "needs; complex to fine-tune and deploy"],
        ["2023", "Labonne and Moran [14]", "BERT, sentence transformers, "
         "T5, LightGBM", "Enron, custom datasets", "Transformers "
         "outperform traditional ML in most settings; few-shot "
         "promising", "Higher infrastructure cost; not always feasible "
         "for real-time lightweight use"],
        ["2025", "Shirvani and Ghasemshirazi [16]", "BERT + FLAN-T5 "
         "(zero-shot)", "Large labelled datasets, real-world email",
         "Reduces dependence on labels; adapts better to changing spam",
         "Significant compute; challenged by highly adversarial spam"],
    ], alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT,
                   WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT,
                   WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
        col_widths=[1.1, 1.7, 1.7, 1.7, 2.0, 2.0])
    add_caption(doc, "Table I.\u2003Summary of representative studies on "
                "email spam filtering.")

    add_section_heading(doc, "C. Hybrid and personalized filtering", level=2)
    add_para(doc,
        "Hybrid approaches combine text-derived semantic features with "
        "structural information such as sender, URL and header "
        "attributes, which improves accuracy and robustness to "
        "manipulation, although they are often evaluated on static "
        "datasets and may not adapt as spam changes. Personalized or "
        "adaptive filters incorporate user preferences, interaction "
        "history and feedback to adjust the decision for each recipient. "
        "This reduces user-specific false positives and improves "
        "satisfaction, but requires user-specific data, has been "
        "explored less at scale and suffers from a cold-start problem "
        "for new users. Table II compares the approaches and states how "
        "each relates to our design.")
    add_table(doc, [
        ["Approach", "Representative techniques", "Strengths",
         "Limitations", "Relevance to this work"],
        ["Rule-based filtering", "Keyword matching, heuristic rules, "
         "blacklists", "Simple; fast and cheap; effective on known spam",
         "Frequent manual updates; cannot detect novel or obfuscated "
         "spam; high false positives", "Basic baseline for comparison"],
        ["Traditional ML", "Naive Bayes, SVM, logistic regression, "
         "decision tree, random forest [8]", "Efficient and scalable; "
         "good with limited data; well studied", "Limited context and "
         "semantics; struggles with obfuscated spam",
         "Baseline in our evaluation (TF-IDF + naive Bayes)"],
        ["Deep learning", "CNN, LSTM, BiLSTM, CNN-LSTM [11]",
         "Learns complex patterns and features automatically; effective "
         "on large datasets", "Higher cost; needs more training data; "
         "less interpretable", "Shows the value of automatic feature "
         "learning"],
        ["Transformer-based", "BERT, RoBERTa, DistilBERT [4]",
         "Strong contextual understanding; state-of-the-art; handles "
         "varied and obfuscated language", "More expensive than "
         "classical ML; needs fine-tuning and resources",
         "DistilBERT extracts the semantic features"],
        ["Hybrid / multimodal", "Text plus structural features (sender, "
         "URL, headers)", "Higher accuracy; complementary information; "
         "more robust to manipulation", "Often evaluated on static "
         "datasets; may not adapt to evolving spam",
         "We combine DistilBERT semantics with structural features"],
        ["Personalized / adaptive", "User behaviour, feedback, "
         "trust-based filtering", "Fewer user-specific false positives; "
         "adapts to individual behaviour",
         "Needs user data or feedback; less explored at scale; cold "
         "start", "We add a personalization layer with user rules and "
         "feedback"],
    ], alignments=[WD_ALIGN_PARAGRAPH.LEFT]*5,
        col_widths=[1.8, 2.0, 2.0, 2.0, 2.4])
    add_caption(doc, "Table II.\u2003Comparative analysis of spam-filtering "
                "approaches and relevance to this work.")

    # Insert Fig 1 — architecture
    add_image(doc, fig_dir / "fig1_architecture.png", width_cm=8.4)
    add_caption(doc, "Fig. 1.\u2003Architecture of the context-aware "
                "personalized spam detection framework. Semantic features "
                "(DistilBERT) and structural features (metadata) are fused "
                "and classified; the personalization layer adjusts the "
                "score using the user profile, and a feedback loop "
                "updates the profile and, optionally, the model.")

    add_section_heading(doc, "D. Research gap", level=2)
    add_para(doc,
        "Within the studies surveyed here, accurate transformer-based "
        "detection, structural evidence and per-user adaptation are "
        "generally treated separately. We therefore design a single, "
        "lightweight pipeline that uses all three: a frozen compact "
        "encoder for semantics, structural metadata for complementary "
        "evidence, and a personalization and feedback layer that adapts "
        "decisions to each user and over time.")

    # =========================================================================
    # III. Problem formulation
    # =========================================================================
    add_section_heading(doc, "III.\u2003Problem formulation")
    add_para(doc,
        "Let e = (x\u209c, x\u2098) denote an incoming email, where x\u209c is "
        "the textual content (subject and body) and x\u2098 is the "
        "metadata (sender, headers, URLs, HTML structure, attachments). "
        "User u receives e and the system must output a decision "
        "ŷ\u1d64(e) \u2208 {0, 1}, where 1 means spam. The ground truth "
        "y\u1d64(e) is user-dependent. Because a false positive hides a "
        "legitimate message while a false negative merely shows an "
        "unwanted one, the objective is the cost-weighted risk")
    add_equation(doc,
        "L\u1d64 = c\u209a\u209a · Pr[ŷ\u1d64 = 1, y\u1d64 = 0] + "
        "c\u209f\u2099 · Pr[ŷ\u1d64 = 0, y\u1d64 = 1]")
    add_para(doc,
        "with c\u209a\u209a > c\u209f\u2099. A global model f is trained once "
        "on a labelled corpus. Per-user context "
        "C\u1d64 = {T\u1d64, B\u1d64, r\u1d64, n\u1d64, τ\u1d64} "
        "holds the trusted-sender list T\u1d64, the blocked-sender list "
        "B\u1d64, sender reputation r\u1d64, the number of feedback "
        "events n\u1d64 and a decision threshold τ\u1d64. The framework "
        "combines f and C\u1d64 to minimize L\u1d64 without retraining the "
        "encoder for every user.")

    # =========================================================================
    # IV. Methodology
    # =========================================================================
    add_section_heading(doc, "IV.\u2003Proposed methodology")

    add_section_heading(doc, "A. Overview", level=2)
    add_para(doc,
        "Fig. 1 shows the seven-stage architecture. An incoming email is "
        "preprocessed (stage 2); semantic and structural features are "
        "extracted and fused (stage 3); a hybrid classifier produces a "
        "spam probability (stage 4); the personalization layer adjusts "
        "that probability using the user's context (stage 5); and the "
        "final decision moves the message to the spam folder or keeps "
        "it in the inbox (stage 6). User feedback closes the loop "
        "(stage 7) by updating the profile and, optionally, the model.")

    add_section_heading(doc, "B. Preprocessing", level=2)
    add_para(doc,
        "The email is parsed into text and metadata. For the text, HTML "
        "tags are removed, the content is normalized (lowercasing and "
        "tokenization), and URLs and special characters are handled "
        "explicitly: URLs are replaced by a placeholder token for the "
        "language model while their properties are kept as structural "
        "features. Stop-word removal is applied for the lexical "
        "baseline; the transformer branch receives the lightly cleaned "
        "text so that function words remain available as context. Text "
        "and metadata are then separated and routed to their respective "
        "extractors.")

    add_section_heading(doc, "C. Semantic representation with DistilBERT",
                        level=2)
    add_para(doc,
        "The subject and body are concatenated and encoded by a "
        "pre-trained DistilBERT model [6], using the implementation of "
        "[17]. The final-layer token representations are mean-pooled "
        "into one dense vector,")
    add_equation(doc, "E\u209b = DistilBERT(x\u209c)  \u2208 \u211d\u2077\u2076\u2078.")
    add_para(doc,
        "Contextual embeddings handle synonyms, obfuscated wording and "
        "context better than sparse TF-IDF vectors. The encoder is "
        "kept frozen, so embeddings are computed once per email and can "
        "be cached, and the cost of training and of adaptation is "
        "confined to the lightweight classifier head.")

    add_section_heading(doc, "D. Structural feature extraction", level=2)
    add_para(doc,
        "Hand-crafted features capture evidence that the text does not "
        "carry. They are grouped as in Table III and include the "
        "outcome of the SPF [18] and DKIM [19] authentication checks. "
        "Numeric features are standardized and binary indicators are "
        "kept as 0/1, giving the vector E\u2098 \u2208 \u211d\u207f, where "
        "n is the number of engineered features.")
    add_table(doc, [
        ["Group", "Representative features"],
        ["Sender / domain", "Free-mail provider flag; From/Reply-To "
         "domain mismatch; display-name versus address mismatch; sender "
         "frequency"],
        ["URL", "URL count; number of distinct domains; URL-shortener "
         "flag; IP-address host; anchor-text versus target mismatch"],
        ["Attachment", "Attachment count; file type (e.g., executable, "
         "archive, office document); total size"],
        ["HTML / content", "HTML-to-text ratio; image-to-text ratio; "
         "hidden text; embedded forms"],
        ["Subject", "Length; upper-case ratio; digit and special-"
         "character counts; reply/forward prefix"],
        ["Header", "SPF and DKIM results; number of Received hops; "
         "Message-ID domain consistency"],
    ], alignments=[WD_ALIGN_PARAGRAPH.LEFT]*2,
        col_widths=[2.6, 6.4])
    add_caption(doc, "Table III.\u2003Structural features extracted from "
                "email metadata.")

    add_section_heading(doc, "E. Feature fusion and hybrid classification",
                        level=2)
    add_para(doc,
        "The two views are concatenated into a single hybrid vector, "
        "and a classifier maps it to a spam probability:")
    add_equation(doc, "E = [E\u209b \u2225 E\u2098],")
    add_equation(doc,
        "p\u209b\u209a\u2090\u2098 = f(E),   0 \u2264 p\u209b\u209a\u2090\u2098 \u2264 1.")
    add_para(doc,
        "Three interchangeable heads are considered: logistic regression "
        "as an interpretable reference, a multilayer perceptron, and "
        "gradient-boosted trees (XGBoost) [20]. Learned weights capture "
        "the relative importance of textual and non-textual signals. "
        "Training minimizes the class-weighted binary cross-entropy")
    add_equation(doc,
        "L = -(1/N) \u03a3\u1d62 [ w\u2081 y\u1d62 log p\u1d62 + "
        "w\u2080 (1 - y\u1d62) log(1 - p\u1d62) ],")
    add_para(doc,
        "and the output is calibrated on a validation split so that "
        "p\u209b\u209a\u2090\u2098 behaves as a probability, which "
        "matters because it is blended with user evidence below.")

    add_section_heading(doc, "F. User personalization layer", level=2)
    add_para(doc,
        "The layer maintains a profile per user from past interactions "
        "(marked spam or not spam, opened, replied, moved) and explicit "
        "preferences such as trusted senders and important categories "
        "(work, academic, personal). Sender reputation is a smoothed "
        "estimate of how often the user treats a sender's mail as spam:")
    add_equation(doc,
        "r\u1d64(s) = ( n\u209b\u209a\u2090\u2098,\u1d64(s) + \u03b2 ) / "
        "( n\u1d64(s) + 2\u03b2 ),")
    add_para(doc,
        "where n\u209b\u209a\u2090\u2098,\u1d64(s) and n\u1d64(s) count "
        "the spam-marked and all messages from sender s, and \u03b2 > 0 "
        "is a smoothing constant so that unseen senders start at 0.5. "
        "The user score combines reputation, a behavioural score "
        "b\u1d64(e) \u2208 [0,1] (high when the user deletes or moves "
        "similar mail unread, low when they open or reply) and a "
        "category-preference score c\u1d64(e) \u2208 [0,1] (low for "
        "categories marked important):")
    add_equation(doc,
        "p\u1d64\u209b\u1d49\u1d49 = \u03bb_r r\u1d64(s) + "
        "\u03bb_b b\u1d64(e) + \u03bb_c c\u1d64(e),")
    add_para(doc,
        "with \u03bb_r + \u03bb_b + \u03bb_c = 1. The final score is the "
        "adaptive blend")
    add_equation(doc, "\u03b1\u1d64 = \u03ba / (\u03ba + n\u1d64),")
    add_equation(doc,
        "p\u209a\u209a\u2099\u2090\u2097 = \u03b1\u1d64 · p\u209b\u209a\u2090\u2098 "
        "+ (1 - \u03b1\u1d64) · p\u1d64\u209b\u1d49\u1d49,")
    add_para(doc,
        "where \u03ba > 0 controls how quickly the system trusts user "
        "evidence over the global model. For a new user n\u1d64 = 0 and "
        "\u03b1\u1d64 = 1, so the system falls back to the global "
        "classifier, which resolves the cold-start problem. As feedback "
        "accumulates, \u03b1\u1d64 decays and the decision becomes "
        "progressively personalized (Fig. 3). Explicit rules take "
        "precedence: mail from a trusted sender is delivered and mail "
        "from a blocked sender is filtered, provided SPF/DKIM "
        "verification passes, so that a spoofed address cannot exploit "
        "the trusted list. The decision rule and the complete flow are")
    add_equation(doc, "ŷ\u1d64 = \U0001d7d9 [ p\u209a\u209a\u2099\u2090\u2097 \u2265 τ\u1d64 ],")
    add_caption(doc, "illustrated in Fig. 2.")

    # Insert Fig 2 — decision flow
    add_image(doc, fig_dir / "fig2_decision_flow.png", width_cm=8.4)
    add_caption(doc, "Fig. 2.\u2003Decision flow of the personalization "
                "layer: explicit sender rules, user score, adaptive "
                "weight, blending and thresholding.")

    # Insert Fig 3 — adaptive weight curve
    add_image(doc, fig_dir / "fig3_adaptive_weight.png", width_cm=8.4)
    add_caption(doc, "Fig. 3.\u2003Illustrative behaviour of the adaptive "
                "weight \u03b1\u1d64 = \u03ba / (\u03ba + n\u1d64): the "
                "weight on the global model decays as user feedback "
                "accumulates, faster for smaller \u03ba.")

    add_section_heading(doc, "G. Feedback-driven adaptation", level=2)
    add_para(doc,
        "User actions are logged as feedback events: marking a message "
        "as spam or not spam, moving it to a folder, deleting or "
        "restoring it, and opening or replying to it. Each event "
        "updates (i) the sender counts that define r\u1d64(s), (ii) "
        "the behavioural score b\u1d64(e) through an exponentially "
        "weighted average so that recent behaviour dominates, and "
        "(iii) the counter n\u1d64 that drives \u03b1\u1d64. "
        "Periodically, the user's threshold is re-fitted on the "
        "buffered feedback:")
    add_equation(doc,
        "τ\u1d64* = arg min_τ [ c\u209a\u209a FP\u1d64(τ) + "
        "c\u209f\u2099 FN\u1d64(τ) ],")
    add_para(doc,
        "where FP\u1d64(τ) and FN\u1d64(τ) are the false positives "
        "and false negatives that threshold τ would have produced on "
        "that feedback. Because c\u209a\u209a > c\u209f\u2099, the "
        "threshold rises when a user repeatedly restores legitimate "
        "mail from the spam folder. Optionally, newly labelled examples "
        "are appended to the training dataset and the classifier head "
        "is refitted, possibly with a sliding window or higher weight "
        "on recent data, to follow concept drift [5]. Since the encoder "
        "is frozen, this refit is inexpensive.")

    # =========================================================================
    # V. System design
    # =========================================================================
    add_section_heading(doc, "V.\u2003System design and data flow")
    add_para(doc,
        "Fig. 4 shows the data flow diagram of the system. The user is "
        "the external entity that sends and receives email (flow 1) and "
        "finally receives the filtered email (flow 8). P1 cleans the "
        "message and separates text from metadata, passing cleaned "
        "metadata to the user-profile store. P2 extracts the semantic "
        "and structural features and P3 turns the feature vector "
        "(E\u209b, E\u2098) into p\u209b\u209a\u2090\u2098. P4, the "
        "personalization layer, reads and writes the user-profile "
        "database D1 (preferences, interaction history, feedback) and "
        "the sender-reputation database D2 (sender history, trusted and "
        "blocked senders), applies user-specific rules, adjusts the "
        "threshold, incorporates behavioural signals, and returns the "
        "final spam / ham decision to the email service, which files "
        "the message in the inbox or spam folder. P5 receives the "
        "user's feedback (flow 9), stores it, updates D1 and D2, "
        "adjusts the personalization parameters and passes updated "
        "model and user data back to P4 for future emails.")
    add_image(doc, fig_dir / "fig4_dfd.png", width_cm=8.6)
    add_caption(doc, "Fig. 4.\u2003Data flow diagram of the personalized "
                "spam filtering system: five processes (P1 to P5), two "
                "data stores (D1, D2) and two external entities. Dashed "
                "teal lines denote feedback flow.")
    add_para(doc,
        "The cost per email is dominated by a single forward pass of "
        "the encoder, which can be cached. Personalization adds only "
        "constant-time lookups and arithmetic. Per-user adaptation "
        "never requires gradient updates to the encoder.")

    # =========================================================================
    # VI. Experimental design
    # =========================================================================
    add_section_heading(doc, "VI.\u2003Experimental design")

    add_section_heading(doc, "A. Datasets", level=2)
    add_para(doc,
        "Evaluation uses the Enron corpus [21] and the SpamAssassin "
        "public corpus [22], and, where available and with consent and "
        "anonymization, real-world user mailboxes. Data are split into "
        "stratified training, validation and test sets (for example "
        "70 / 15 / 15). For adaptation experiments, messages are "
        "ordered chronologically so that the test period follows the "
        "training period, which exposes concept drift.")

    add_section_heading(doc, "B. Baselines and ablation study", level=2)
    add_para(doc,
        "Table IV lists the compared configurations. Baselines B1 and "
        "B2 are the lexical and text-only transformer references; "
        "A1 to A4 add components one at a time so that the effect of "
        "structural features and of personalization can be measured "
        "separately.")
    add_table(doc, [
        ["ID", "Configuration", "Question addressed"],
        ["B1", "TF-IDF + naive Bayes", "Classical lexical baseline"],
        ["B2", "DistilBERT (text only)", "Gain from contextual "
         "semantics over B1"],
        ["A1", "Structural features only", "Information carried by "
         "metadata alone"],
        ["A2", "DistilBERT + structural features",
         "Gain from fusion over B2"],
        ["A3", "A2 + rules and sender reputation", "Effect of explicit "
         "personalization signals"],
        ["A4", "Full system: A2 + personalization + feedback loop",
         "Overall benefit, especially on false positives"],
    ], alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT,
                   WD_ALIGN_PARAGRAPH.LEFT],
        col_widths=[0.8, 4.0, 4.2])
    add_caption(doc, "Table IV.\u2003Baselines and ablation variants.")

    add_section_heading(doc, "C. Evaluation metrics", level=2)
    add_para(doc,
        "We report accuracy, precision (TP / (TP + FP)), recall "
        "(TP / (TP + FN)), F1 score (the harmonic mean of precision "
        "and recall) and the false-positive rate FPR = FP / (FP + TN), "
        "where spam is the positive class. Since the framework is "
        "designed to avoid losing legitimate mail, FPR is the headline "
        "metric; personalized variants are scored per user and "
        "averaged.")

    add_section_heading(doc, "D. Personalization and adaptation protocol",
                        level=2)
    add_para(doc,
        "Public corpora do not contain per-user preference labels, so "
        "user-dependent ground truth is obtained in two ways: (a) "
        "treating each mailbox owner in Enron as a separate user, and "
        "(b) defining synthetic users whose preferences differ on "
        "borderline categories such as promotions and newsletters. "
        "Feedback is revealed sequentially: after each decision, a "
        "feedback event is generated with a fixed probability, and "
        "performance is reported as a function of n\u1d64 to trace "
        "the learning curve, including the cold-start point n\u1d64 = "
        "0. Robustness to user error is tested by flipping feedback "
        "labels with probability \u03c1. Adaptation to drift is "
        "measured by tracking F1 and FPR over the chronological test "
        "period with and without periodic model updates. Where real "
        "user data are available, observed feedback replaces the "
        "simulation.")

    add_section_heading(doc, "E. Implementation settings", level=2)
    add_para(doc,
        "The pipeline is implemented in Python with Hugging Face "
        "Transformers [17], scikit-learn [23] and XGBoost. Table V "
        "lists the initial settings, which are tuned on the validation "
        "split. Results are reported as mean plus or minus standard "
        "deviation over several random seeds, and differences are "
        "assessed with paired bootstrap tests.")
    add_table(doc, [
        ["Component", "Setting"],
        ["Encoder", "Pre-trained DistilBERT, frozen; mean pooling; "
         "768-dimensional output; truncated input length"],
        ["Structural features", "Numeric features standardised; "
         "binary flags kept as 0 / 1"],
        ["Classifier heads", "Logistic regression; MLP (hidden layers, "
         "ReLU, dropout); XGBoost"],
        ["Training loss", "Class-weighted binary cross-entropy; "
         "probability calibration on validation split"],
        ["Personalization", "\u03bb_r, \u03bb_b, \u03bb_c (sum to 1), "
         "\u03b2, \u03ba tuned on validation data"],
        ["Decision cost", "c\u209a\u209a > c\u209f\u2099; per-user "
         "threshold \u03c4\u1d64 re-fitted periodically"],
    ], alignments=[WD_ALIGN_PARAGRAPH.LEFT]*2,
        col_widths=[3.0, 6.0])
    add_caption(doc, "Table V.\u2003Initial configuration (tuned on "
                "validation data).")

    # =========================================================================
    # VII. Discussion
    # =========================================================================
    add_section_heading(doc, "VII.\u2003Discussion")

    add_section_heading(doc, "A. Expected benefits", level=2)
    add_para(doc,
        "The design is expected to benefit from the complementary "
        "nature of its evidence. Semantic embeddings should help with "
        "paraphrased or obfuscated wording that defeats keyword "
        "features, while structural signals such as failed "
        "authentication or mismatched sender domains expose messages "
        "whose text is fluent. The personalization layer should lower "
        "the false-positive rate for mail that is legitimate for a "
        "particular user, and the adaptive weight should keep new users "
        "no worse off than the global model. These are hypotheses to "
        "be confirmed by the ablation study in Section VI rather than "
        "established results.")

    add_section_heading(doc, "B. Privacy and deployment", level=2)
    add_para(doc,
        "Personalization needs only compact per-user state: counters "
        "per sender, behavioural scores, preferences and a threshold. "
        "It does not require storing raw message content, so profiles "
        "can be kept on the user's side or inside their own mailbox "
        "store. Because the encoder is frozen and embeddings are "
        "cacheable, the system can run on modest hardware, which "
        "supports the practical deployment goal.")

    add_section_heading(doc, "C. Limitations and threats to validity", level=2)
    add_para(doc,
        "Public corpora are older than current spam campaigns, and "
        "simulated feedback only approximates real user behaviour. "
        "Attackers may craft messages to evade the classifier or try "
        "to poison feedback; a compromised trusted sender account "
        "would still pass SPF/DKIM checks. Users also make mistakes, "
        "which introduces label noise, and an English uncased encoder "
        "limits coverage of multilingual mail. Finally, cold-start "
        "quality depends entirely on the global model.")

    # =========================================================================
    # VIII. Conclusion
    # =========================================================================
    add_section_heading(doc, "VIII.\u2003Conclusion and future work")
    add_para(doc,
        "This paper presented a context-aware personalized spam "
        "filtering framework that combines DistilBERT semantic "
        "embeddings, structural email features and user-specific "
        "context in a hybrid classifier, with an adaptive "
        "personalization layer and a feedback loop that updates "
        "profiles, sender reputation and thresholds. We formalized "
        "the decision process, described the architecture and data "
        "flow, and defined an evaluation protocol with baselines, "
        "ablations and an adaptation study.")
    add_para(doc,
        "Future work includes executing this protocol and reporting "
        "results, using parameter-efficient fine-tuning of the "
        "encoder, multilingual encoders, federated or on-device "
        "learning of user profiles, adversarial robustness testing, "
        "and a user study of perceived inbox quality.")

    # =========================================================================
    # References
    # =========================================================================
    add_section_heading(doc, "References")
    refs = [
        "G. V. Cormack, \"Email spam filtering: A systematic review,\" "
        "Found. Trends Inf. Retr., vol. 1, no. 4, pp. 335-455, 2008.",
        "E. G. Dada, J. S. Bassi, H. Chiroma, S. M. Abdulhamid, A. O. "
        "Adetunmbi, and O. E. Ajibuwa, \"Machine learning for email spam "
        "filtering: Review, approaches and open research problems,\" "
        "Heliyon, vol. 5, no. 6, Art. no. e01802, Jun. 2019.",
        "I. Androutsopoulos, J. Koutsias, K. V. Chandrinos, G. "
        "Paliouras, and C. D. Spyropoulos, \"An evaluation of Naive "
        "Bayesian anti-spam filtering,\" in Proc. Workshop Mach. Learn. "
        "New Inf. Age, 11th Eur. Conf. Mach. Learn., Barcelona, Spain, "
        "2000, pp. 9-17.",
        "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, \"BERT: "
        "Pre-training of deep bidirectional transformers for language "
        "understanding,\" in Proc. NAACL-HLT, 2019, pp. 4171-4186.",
        "J. Gama, I. Zliobaite, A. Bifet, M. Pechenizkiy, and A. "
        "Bouchachia, \"A survey on concept drift adaptation,\" ACM "
        "Comput. Surv., vol. 46, no. 4, Art. no. 44, 2014.",
        "V. Sanh, L. Debut, J. Chaumond, and T. Wolf, \"DistilBERT, a "
        "distilled version of BERT: Smaller, faster, cheaper and "
        "lighter,\" arXiv preprint arXiv:1910.01108, 2019.",
        "M. Sahami, S. Dumais, D. Heckerman, and E. Horvitz, \"A "
        "Bayesian approach to filtering junk e-mail,\" in Proc. AAAI "
        "Workshop Learn. Text Categorization, Madison, WI, USA, 1998, "
        "pp. 55-62.",
        "V. Metsis, I. Androutsopoulos, and G. Paliouras, \"Spam "
        "filtering with naive Bayes: Which naive Bayes?,\" in Proc. "
        "3rd Conf. Email Anti-Spam (CEAS), Mountain View, CA, USA, 2006.",
        "T. S. Guzella and W. M. Caminhas, \"A review of machine "
        "learning approaches to spam filtering,\" Expert Syst. Appl., "
        "vol. 36, no. 7, pp. 10206-10222, Sep. 2009.",
        "Saxena et al., \"Comparative analysis of machine learning "
        "techniques for email spam detection,\" 2021.",
        "X. Zhang, J. Zhao, and Y. LeCun, \"Character-level "
        "convolutional networks for text classification,\" in Proc. "
        "Adv. Neural Inf. Process. Syst. (NIPS), 2015, pp. 649-657.",
        "A. Vaswani et al., \"Attention is all you need,\" in Proc. "
        "Adv. Neural Inf. Process. Syst. (NIPS), 2017, pp. 5998-6008.",
        "T. Sahmoud and D. M. Mikki, \"Spam detection using BERT,\" "
        "arXiv preprint arXiv:2206.02443, 2022.",
        "M. Labonne and S. Moran, \"Spam-T5: Benchmarking large "
        "language models for few-shot email spam detection,\" arXiv "
        "preprint arXiv:2304.01238, 2023.",
        "H. W. Chung et al., \"Scaling instruction-finetuned language "
        "models,\" arXiv preprint arXiv:2210.11416, 2022.",
        "Shirvani and Ghasemshirazi, \"BERT and FLAN-T5 for adaptive "
        "spam detection,\" 2025.",
        "T. Wolf et al., \"Transformers: State-of-the-art natural "
        "language processing,\" in Proc. Conf. Empirical Methods "
        "Natural Lang. Process.: Syst. Demonstrations, 2020, pp. 38-45.",
        "S. Kitterman, \"Sender Policy Framework (SPF) for authorizing "
        "use of domains in email, version 1,\" IETF, RFC 7208, Apr. "
        "2014.",
        "D. Crocker, T. Hansen, and M. Kucherawy, Eds., \"DomainKeys "
        "Identified Mail (DKIM) signatures,\" IETF, RFC 6376, Sep. "
        "2011.",
        "T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting "
        "system,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery "
        "Data Mining, 2016, pp. 785-794.",
        "B. Klimt and Y. Yang, \"The Enron corpus: A new dataset for "
        "email classification research,\" in Proc. Eur. Conf. Mach. "
        "Learn. (ECML), 2004, pp. 217-226.",
        "\"Apache SpamAssassin public corpus.\" [Online]. Available: "
        "https://spamassassin.apache.org/old/publiccorpus/",
        "F. Pedregosa et al., \"Scikit-learn: Machine learning in "
        "Python,\" J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.",
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

    out_path = out_dir / "IEEE_Paper_Personalized_Email_Spam_Filtering_humanized.docx"
    doc.save(str(out_path))
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    fig_dir = Path("C:/Users/podda/Downloads/_paper_build/figures")
    out_dir = Path("C:/Users/podda/Downloads")
    build_paper(fig_dir, out_dir)