# SpamGuard — Context-Aware Personalized Email Spam Detection

A capstone project that classifies incoming mail as **Spam** or **Ham** using three cooperating layers: a fine-tuned **DistilBERT** encoder for semantic understanding, **20 hand-crafted structural features** (sender, URLs, HTML, attachments, length, capitalization, SPF/DKIM) for evidence the transformer misses, and a **personalization layer** that learns from each user's feedback to lower their specific false-positive rate.

The repo ships a four-tab **Streamlit demo** for live classification, model comparison, and an interactive feedback loop. An IEEE-format research paper (also in this repo) describes the architecture, problem formulation, experimental design and evaluation protocol in detail.

---

## Highlights

- **Four models compared on the same data**:
  1. **TF-IDF + Multinomial Naive Bayes** — interpretable lexical baseline
  2. **DistilBERT** — fine-tuned contextual classifier
  3. **Hybrid (DistilBERT + structural)** — multimodal fusion MLP
  4. **Personalized (Hybrid + user profile)** — final blended score
- **20 structural features** including SPF/DKIM authentication, URL shortener flag, HTML-to-text ratio, attachment type, free-mail provider, sender/display-name mismatch
- **Adaptive personalization weight** `α_u = κ / (κ + n_u)` so the system falls back to the global model for new users and trusts user evidence as feedback accumulates
- **Feedback loop** that updates trusted/blocked senders, sender reputation, keyword preferences and per-user decision threshold
- **Per-prediction explanations** (token saliency for the baseline, gradient × input for DistilBERT, signed structural-feature contributions for the hybrid) so a reviewer can ask "why did you flag this?" and get a defensible answer

---

## Streamlit Demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`. The app has four tabs:

| Tab | What it shows |
| --- | --- |
| **Project Overview** | Hero, KPIs, architecture flow diagram, four-model comparison table |
| **Model Comparison** | Metrics table, per-model scorecards, Recall vs FPR scatter, head-to-head bars, confusion matrices, per-metric breakdown |
| **Live Classifier** | Paste any email (or pick a sample), get all 4 model verdicts, see the personalization math, plus token / feature saliency for the verdict |
| **Feedback Loop** | Round-by-round labelling. Trusted/blocked senders, sender reputation and personalization strength all update in real time |

---

## Project layout

```
.
├── app.py                    # Streamlit demo (4 tabs)
├── requirements.txt
├── data/
│   ├── raw/                  # downloaded SpamAssassin + Enron archives
│   └── processed/            # per-message CSVs with parsed structural features
├── src/
│   ├── preprocessor.py       # RFC822 parsing, HTML stripping, cleaning
│   ├── data_loader.py        # SpamAssassin / Enron loaders, CSV IO
│   ├── features/
│   │   ├── structural.py     # 20 hand-crafted features
│   │   └── tfidf_features.py # TF-IDF vectorizer wrapper
│   ├── models/
│   │   ├── baseline_nb.py    # TF-IDF + MultinomialNB
│   │   ├── distilbert.py     # fine-tuned DistilBERT
│   │   ├── hybrid.py         # DistilBERT + structural fusion MLP
│   │   └── personalized.py   # Hybrid + user-profile decision layer
│   ├── personalization/
│   │   └── user_profile.py   # trusted/blocked senders, sender reputation,
│   │                         # keyword prefs, feedback log
│   ├── train.py              # end-to-end training pipeline
│   ├── evaluate.py           # accuracy, precision, recall, F1, FPR, AUC
│   └── compare.py            # comparison + plots
├── scripts/
│   ├── build_datasets.py
│   ├── run_all.py
│   ├── demo.py
│   ├── finish_personalized.py
│   └── generate_comparison_metrics.py
├── tests/test_smoke.py
├── paper/                    # IEEE LaTeX source, DOCX, and figures
└── results/                  # metrics JSON, reports, plots
```

---

## Datasets

- **SpamAssassin public corpus** — primary training set (`data/raw/spamassassin/`, ~5,000 ham + ~3,000 spam).
- **Enron-Spam** — out-of-distribution validation (`data/raw/enron_spam/`, ~33,000 messages).
- **Optional real mailbox data** — the feedback loop is designed to consume an anonymized per-user mailbox with consent; the public corpora are used to bootstrap the global model.

Both are downloaded and parsed by `scripts/build_datasets.py` into per-message CSVs in `data/processed/`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate           # or: .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

> PyTorch's CUDA wheels are large. If you don't have a GPU, install the CPU build (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) — the project runs end-to-end on CPU, just slower for the deep models.

---

## Run the full pipeline

```bash
python scripts/build_datasets.py --dataset all
python -m src.train --csv data/processed/spamassassin.csv --epochs-distilbert 2 --epochs-hybrid 2
python scripts/generate_comparison_metrics.py --csv data/processed/spamassassin.csv
python scripts/generate_comparison_metrics.py --csv data/processed/enron_spam.csv \
    --out results/enron_validation_metrics.json
streamlit run app.py
```

This will:
1. Download the corpora into `data/raw/`.
2. Parse them into `data/processed/*.csv`.
3. Train all four models on SpamAssassin.
4. Write metrics to `results/comparison_metrics.json` and `results/enron_validation_metrics.json`.
5. Launch the Streamlit demo at `http://localhost:8501`.

To share a temporary public URL, you can front the Streamlit server with a Cloudflare quick tunnel:

```bash
./cloudflared tunnel --url http://localhost:8501
```

---

## Personalization layer

For each user `u`, the personalized detector maintains:

- `T_u` — trusted senders / domains
- `B_u` — blocked senders / domains
- `r_u(s)` — sender reputation (Laplace-smoothed spam ratio)
- `b_u(e)` — behavioural score from open / reply / delete events
- `c_u(e)` — category-preference score
- `n_u` — number of feedback events
- `τ_u` — per-user decision threshold (re-fitted on buffered feedback)

The adaptive weight is

```
α_u = κ / (κ + n_u)
```

so a brand-new user gets `α = 1` (no personalization, fall back to the global model) and the system gradually trusts user evidence as `n_u` grows. The final score is

```
p_final = (1 − α_u) · p_hybrid + α_u · p_user
ŷ = 1[p_final ≥ τ_u]
```

Sender rules take precedence: mail from a trusted sender is always delivered, mail from a blocked sender is always filtered, **provided SPF/DKIM verification passes** so a spoofed address can't exploit the trust.

---

## Metrics reported

- Accuracy, Precision, Recall, F1, **False Positive Rate**, ROC AUC
- Confusion matrix per model
- Per-token contributions (TF-IDF baseline), per-word gradient saliency (DistilBERT), per-feature signed contributions (Hybrid)

The headline metric for spam filters is **FPR** — losing legitimate mail is the failure users actually complain about. The personalized layer is designed to lower FPR for each user without sacrificing recall.

---

## Research paper

An IEEE conference-format paper describing the architecture, mathematical formulation, system design, experimental protocol, and discussion is in [`paper/`](paper/). Both LaTeX source and rendered DOCX are included.

---

## License

MIT — research / educational use.
