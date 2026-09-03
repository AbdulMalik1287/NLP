# Demonstrability: Concrete Numbers and the Live Demo

The proposal (§28) refuses to promise scores before experimentation. Correct for the paper —
useless for a review meeting. This doc adds the missing layer:

1. a **fast path to the first real numbers** (day 10, no annotation needed),
2. the **exact result tables** to fill, and
3. **expected ranges from published literature** — these are *sanity bands, not claims*.

> **Read the ranges this way:** if a run lands inside the band, the pipeline is probably
> working. If it lands far outside, suspect a bug (leakage, label mismatch, tokenizer,
> class imbalance) *before* reporting a finding. Every number in the paper is the measured
> one. Nothing here is a promised outcome.

---

## 1. Fast path — first defensible numbers in ~10 working days

Skips annotation entirely. Uses only datasets that already have labels.

| Day | Action | Output — a number you can show |
|---|---|---|
| 1–2 | Pull CFPB, keep rows with narratives, map `product`/`issue` to ~15–20 coarse intents | dataset size, class histogram, e.g. "48,213 narratives, 18 intents, imbalance 31:1" |
| 3 | Split by domain proxy; MinHash dedup | "1,847 near-duplicates removed (3.8%)" |
| 4 | TF-IDF + LogisticRegression, TF-IDF + SVM | first macro-F1 on the board |
| 5 | mBERT fine-tune | macro-F1 vs. TF-IDF delta |
| 6 | XLM-R fine-tune, in-domain | the headline in-domain number |
| 7 | Leave-one-domain-out run | **the cross-domain drop — the core research number** |
| 8 | MASSIVE hi-IN zero-shot: train EN, test HI | **the cross-language drop** |
| 9 | FAISS index + TF-IDF retrieval baseline, P@5 / MRR | retrieval comparison table |
| 10 | mT5 zero-shot summaries + SHAP on 50 examples | screenshots for slides |

After day 10 you have a filled results table and a working pipeline. Contrastive learning,
annotation, and the Hinglish set then improve a system that already produces numbers,
instead of being the thing standing between you and any number at all.

---

## 2. Result tables to fill

### T1 — Intent classification, in-domain (RQ baseline)

| Model | Params | Accuracy | Macro-F1 | Weighted-F1 | Train time |
|---|---|---|---|---|---|
| Majority class | 0 | | | | — |
| TF-IDF + LogReg | ~1M feat | | | | |
| TF-IDF + SVM | ~1M feat | | | | |
| BERT-base (EN only) | 110M | | | | |
| mBERT | 178M | | | | |
| XLM-R base | 278M | | | | |
| XLM-R + contrastive | 278M | | | | |

*Expected band, macro-F1, ~15–20 coarse classes, ~50k examples:*
TF-IDF+LogReg **0.55–0.70** · mBERT **0.68–0.80** · XLM-R **0.70–0.84**.
Transformer over TF-IDF is typically **+8 to +15 F1**. If TF-IDF beats XLM-R, the
fine-tune is broken (LR too high, wrong pooling, frozen weights) — not a finding.
Fine-grained ontologies (50+ classes) sit 10–20 points lower; BANKING77-style
77-class setups report ~0.90+ accuracy only because that data is clean and balanced.

### T2 — Cross-domain transfer (**the central number, H2/RQ1**)

| Train domains | Held-out | n_train | n_test | In-domain F1 | Cross-domain F1 | **Δ** |
|---|---|---|---|---|---|---|
| Bank+Telecom+Ecom | Education | | | | | |
| Bank+Telecom+Edu | E-commerce | | | | | |
| Telecom+Ecom+Edu | Banking | | | | | |
| single domain only | (each) | | | | | |

*Expected band:* zero-shot to an unseen domain drops **10–30 macro-F1 points**.
Multi-domain training beats single-domain on the held-out domain by **+5 to +15** (H2).
A Δ near 0 means domain leakage — check the split. A Δ over ~40 means the ontology
does not actually transfer, which is itself a reportable finding.

### T3 — Cross-language transfer (H1/RQ2)

| Train | Test | XLM-R F1 | mBERT F1 | BERT-EN F1 |
|---|---|---|---|---|
| EN | EN | | | |
| EN | HI (zero-shot) | | | |
| EN | Hinglish (zero-shot) | | | |
| EN+HI | Hinglish | | | |

*Expected band:* XLM-R EN→HI zero-shot drops **5–15 points**; mBERT drops more
(that gap *is* H1). EN→Hinglish drops hardest, **15–30 points** — romanised
code-mixed text is out-of-distribution for both encoders. Adding HI to training
recovers part of the Hinglish gap (H5).

### T4 — Limited-data adaptation (RQ1, Ablation 6)

| Target-domain examples | 0 | 25 | 50 | 100 | 250 |
|---|---|---|---|---|---|
| Macro-F1 | | | | | |

*Expected shape:* steep gain from 0→50, flattening by 250. **Plot this curve** — it is
the single most persuasive slide in the deck, and it directly answers "is cross-domain
transfer worth anything when you have almost no data for the new domain."

### T5 — Semantic retrieval (H4/RQ4)

| Method | P@5 | R@5 | MRR | Index build | Query latency |
|---|---|---|---|---|---|
| TF-IDF cosine | | | | | |
| XLM-R mean-pool | | | | | |
| XLM-R + contrastive | | | | | |

*Expected band on a small hand-labelled similarity set (~200 queries):*
TF-IDF P@5 **0.30–0.50**; dense embeddings **+0.10 to +0.25** over it; contrastive adds
**+0.05 to +0.15** on top. Off-the-shelf XLM-R mean-pooling is a *weak* sentence encoder —
if it loses to TF-IDF before contrastive training, that is expected, not a bug (see SimCSE).

### T6 — Summarization (RQ5)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| Lead-1 sentence baseline | | | | |
| mT5 zero-shot (XLSum ckpt) | | | | |
| mT5 fine-tuned on 300–500 pairs | | | | |

*Expected band:* zero-shot ROUGE-L **0.15–0.25** on out-of-domain complaint text;
fine-tuning on a few hundred pairs typically adds **+3 to +8 ROUGE-L**. Report human
factuality on ~50 summaries (count hallucinated facts) — reviewers trust that over ROUGE.

### T7 — Explanation faithfulness (RQ6/H7)

Deletion-based, no human study required:

| Metric | Value |
|---|---|
| Mean confidence, original input | e.g. 0.91 |
| Mean confidence, top-5 SHAP tokens removed | e.g. 0.42 |
| **Confidence drop (faithfulness)** | **0.49** |
| Confidence, 5 random tokens removed (control) | e.g. 0.85 |
| Top-token overlap, EN vs. HI paraphrase | e.g. 0.61 |

*Expected:* removing top SHAP tokens must hurt far more than removing random tokens.
If the two are similar, the explanations are not faithful — a legitimate negative result,
and a better contribution than a vague "SHAP looks reasonable."

### T8 — System numbers (put these on the poster)

| Metric | Value |
|---|---|
| Encoder params / disk | 278M / ~1.1 GB |
| mT5-base params | 580M |
| Index size | e.g. 48,213 × 768 float32 = ~148 MB |
| Fine-tune time, 1 MIG slice (1g.24gb) | e.g. 8 min/epoch, 5 epochs |
| Inference latency, classify (GPU) | ~5–15 ms |
| Inference latency, classify (CPU) | ~50–150 ms |
| FAISS top-5 lookup, flat index | ~1–5 ms |
| mT5 summary generation | ~0.5–2 s ← **the demo bottleneck** |
| End-to-end API response | ~1–3 s |

Latency ranges are hardware expectations, to be replaced with measured values.
If the live demo feels slow, it is mT5 generation — cache summaries for the demo inputs.

---

## 3. The live demo

One screen. Paste a complaint, get every component's output at once, each with a number.

```
INPUT   Mera recharge successful hai but internet abhi tak activate nahi hua.

Language      Hinglish            confidence 0.94        [12 ms]
Domain        Telecom             confidence 0.88
Intent        activation_failure  confidence 0.91        [9 ms]
              2nd  service_unavailable   0.05
              3rd  payment_failed        0.02

Why this intent (SHAP)
   recharge +0.31 ######
   activate +0.28 #####
   nahi     +0.19 ###
   hua      +0.04 |
   Mera     -0.02 |

Similar complaints (FAISS, cosine)          [3 ms]
   0.89  "Recharge done but net not working since morning"      TELECOM  EN
   0.85  "Paisa cut gaya, internet start nahi hua"              TELECOM  HINGLISH
   0.71  "Plan activated but no data service"                   TELECOM  EN

Summary (mT5)                                [1.4 s]
   "Customer reports a successful recharge but internet service
    has not been activated."
```

**Demo must-haves**

- Confidence on every prediction, and the **top-3** intents, not just the winner
- Per-component **latency in ms**, live
- A **language toggle**: same complaint in EN / HI / Hinglish, three predictions side by
  side. Consistent intent across all three *is* the cross-lingual claim, visible in one click
- A **domain badge** on retrieved neighbours — showing a Hinglish telecom query pulling an
  English banking neighbour with the same intent demonstrates cross-domain transfer better
  than any table
- Prepared failure case. Show one input the model gets wrong and explain why. Reviewers
  trust a demo with a known failure far more than a demo that only ever succeeds

**Demo safety:** pre-load models at startup, pre-compute the index, cache summaries for
5–10 canned inputs, and keep a screen-recorded fallback. Never fine-tune or build an index
live.

---

## 4. Slide-ready headline numbers

Fill these six sentences and the presentation writes itself:

1. "We unified **N** complaints across **D** domains and **3** languages into **K** intents."
2. "In-domain macro-F1: **__**. Cross-domain: **__**. The transfer gap is **__** points."
3. "XLM-R beats mBERT by **__** F1 on Hindi and **__** on Hinglish."
4. "Contrastive training adds **__** F1 cross-domain and **__** P@5 on retrieval."
5. "With just **50** labelled examples from an unseen domain we recover **__%** of in-domain performance."
6. "Removing the top-5 SHAP tokens drops confidence by **__**, versus **__** for random tokens."

Sentence 5 is the one people remember. Prioritise T4.

---

## 5. Guardrails

- Report **mean ± std over 3 seeds** for every headline number. Single-seed transformer
  results move 1–3 F1 points on their own; do not report a 1-point win from one run.
- Always show the **majority-class** and **TF-IDF** floor. A big number means nothing
  without the floor beneath it.
- Freeze the test set before the final comparison (§17). Fill these tables **once**.
- Negative results stay in. "Contrastive learning did not help cross-domain transfer,
  and here is the embedding analysis showing why" is a real paper. Silent deletion is not.
