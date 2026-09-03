# Progress & Handover

**Project:** Cross-Domain Complaint Understanding via Multilingual Encoders (B.Tech capstone, IIIT Nagpur, 16 weeks)
**Last updated:** 2026-09-03
**Status:** Week 1 of 16. Data pipeline built and run; T1 complete on both baselines and transformers. Ontology v1 is the next blocking item.

Read this first, then `docs/audit/CFPB_AUDIT.md` (what the data actually is) and
`docs/DEMO_AND_METRICS.md` (what numbers we owe and what counts as a broken run).

---

## 1. Where the project stands

**Done**
- Prerequisites collected: environment, dataset inventory, literature list (`docs/`)
- CFPB corpus audited against the live API — the numbers in §3 are measured, not estimated
- Preprocessing pipeline built and run over all 17.5M rows → **73,643-row dataset, 16 intents**
- Intent ontology v0.1 (`configs/intent_ontology_v0.json`) — 85 rules, 7.3% of rows unmapped
- TF-IDF baselines measured (T1 floor)
- Blackwell node set up end to end, verified working

**In flight**
- nothing; T1 is complete (see §2)

**Not started**
- Everything multilingual (Hindi, Hinglish) — see §6, this is the critical path
- Contrastive learning (the paper's actual research contribution)
- Retrieval, summarization, SHAP
- The 1,000–1,500-example annotation set

---

## 2. Numbers so far

### T1 — in-domain intent classification (51,555 train / 11,044 test, 16 intents)

| Model | macro-F1 | Notes |
|---|---|---|
| Majority class | 0.0079 | floor |
| TF-IDF + LinearSVC | **0.5223** | 3 seeds, std 0.0000, fit 17 s |
| TF-IDF + LogReg | **0.5254** | 3 seeds, std 0.0000, fit 193 s |
| XLM-R base *(laptop pilot)* | 0.5126 | **discarded — see below** |
| XLM-R base *(node, corrected)* | **0.5419** | clears the floor by +0.020 |
| mBERT *(node, corrected)* | **0.5263** | ties the floor (+0.001) |

Seed std is 0.0000 for the TF-IDF models because both are deterministic convex solvers on
fixed splits. Real variance only appears with the transformers, which is where the
3-seed rule matters.

## T1 — in-domain intent classification, FINAL (51,555 train / 11,044 test, 16 intents)

| Model | macro-F1 | Accuracy | vs floor | Notes |
|---|---|---|---|---|
| Majority class | 0.0079 | 0.068 | — | |
| TF-IDF + LinearSVC | 0.5223 | 0.526 | — | the floor; fit 17 s |
| TF-IDF + LogReg | 0.5254 | 0.526 | +0.003 | fit 193 s |
| mBERT | **0.5263** | 0.527 | **+0.001** | 6 ep, 61 min, MIG slice |
| **XLM-R base** | **0.5419** | 0.542 | **+0.020** | 6 ep, 66 min, MIG slice |
| XLM-R base *(laptop pilot, discarded)* | 0.5126 | 0.517 | -0.010 | 3 ep, truncated, unweighted |

Node config: bf16, `max_len 512`, `--class-weights balanced`, embeddings unfrozen,
batch 32, seed 13. Identical splits to the baselines (md5-verified), so the comparison
is like for like.

**Read this honestly.** XLM-R beats the TF-IDF floor by **+0.020 macro-F1**, and mBERT
essentially ties it (+0.001). Both are far below the 0.70-0.84 band predicted in
`DEMO_AND_METRICS.md`. A 278M-parameter multilingual encoder buying two points over
bag-of-words is a weak result, and the likeliest cause is **label noise in ontology
v0.1**, not the models:

- Both models fail on the *same* classes, in the same order: `debt_threats` (0.43/0.44),
  `account_management` (0.44/0.45), `reporting_incorrect_information` (0.42/0.46),
  `reporting_investigation_dispute` (0.48). The TF-IDF baselines fail on those too.
- Those are exactly the pairs whose CFPB issue strings overlap semantically. Our intents
  come from keyword rules over `issue || sub_issue`, so any complaint sitting between two
  issue labels gets an arbitrary intent.
- Validation converged (0.5367 / 0.5363 / 0.5379 over the last three epochs), so this is
  not underfitting any more. The ceiling is in the labels.

XLM-R over mBERT (+0.016) is the expected direction, but on English-only data this says
nothing about H1 — H1 is a cross-*language* claim and needs Hindi/Hinglish to test.

**Next action this implies:** ontology v1 before more model work. Merge or sharpen the
confusable intent pairs, then re-run T1. Chasing model tweaks against noisy labels is
wasted compute.

### The discarded pilot — read this before trusting any transformer number

The first XLM-R run scored **0.5126, below the TF-IDF floor**. Per the rule written into
`docs/DEMO_AND_METRICS.md` before any model ran, that means a broken run, not a finding.
Three causes, all methodological:

1. **Underfit.** Val macro-F1 was still climbing at the final epoch (0.4645 → 0.5047 →
   0.5157) and train loss still falling. Three epochs was too few.
2. **Truncation asymmetry.** TF-IDF reads the whole narrative; the encoder read
   `max_len=256` tokens, which truncates **40% of the test set**. The baseline had
   strictly more information than the model.
3. **Class-weight asymmetry.** Baselines used `class_weight="balanced"`; the transformer
   used plain cross-entropy. Macro-F1 punishes that on smaller intents.

The in-flight sweep fixes all three: 6 epochs, `max_len 512`, `--class-weights balanced`,
embeddings unfrozen, bf16. **If XLM-R still loses to TF-IDF after this, that is a real
result worth investigating — but check the confounds again first.**

---

## 3. Findings that shape the whole project

From `docs/audit/CFPB_AUDIT.md`. Each one changed a decision.

| # | Finding | Consequence |
|---|---|---|
| F1 | Credit reporting is **65% of narratives**; raw imbalance **1183:1** | Never sample CFPB randomly. Pipeline caps per intent; final imbalance is 2.77:1 |
| F2 | 21 product labels are **versioned duplicates** of ~13 concepts | Canonicalise before splitting, or "cross-domain" splits leak the same domain twice and inflate transfer results |
| F3 | **78.7%** of narratives carry `XXXX` redactions (14.4 runs/doc) — names, dates, **amounts** | CFPB cannot be the summarization eval source; those pairs must be annotated by hand |
| F4 | **23%** templated near-duplicates (credit-repair form letters) | Biggest leakage risk in the project. Dedup runs before splitting; 11.4% dropped in the real run |
| F5 | **0.00% Devanagari**, 0.01% non-ASCII | CFPB is the English/finance anchor only. It cannot serve RQ2 or true cross-domain |

Corpus totals: **17,516,902 complaints, 3,846,901 with narratives (21.96%)**. The
proposal's "~7M / ~1.5M" estimate was stale.

**Sub-domain trick we're using:** CFPB's own products act as pseudo-domains for a first
leave-one-out transfer experiment (12 folds already generated). Label it honestly as
*intra-financial* transfer — weaker than true cross-domain, but it produces a real
number without waiting on other datasets.

---

## 4. Repository map

```
configs/    product_canonical.json   21 CFPB labels -> 13 canonical products
            intent_ontology_v0.json  ordered keyword rules -> 16 intents
            presets.json             hardware presets (laptop / MIG / full node)
scripts/    audit_cfpb.py            corpus audit via the CFPB API
            prep_cfpb.py             zip -> deduped, capped, split dataset (stdlib only)
            train_baselines.py       TF-IDF floors
            train_transformer.py     encoder fine-tuning
            check_env.py             GPU/torch compatibility gate — run before any node job
            node_env.sh              contains all caches inside the project dir
            cleanup_node.sh          removes the project from the shared node
docs/       audit/CFPB_AUDIT.md      the measured dataset audit
            DEMO_AND_METRICS.md      result tables T1-T8, expected bands, demo spec
            DATASETS.md              every candidate dataset, links + licences
            LITERATURE.md            ~50 papers by component, P0/P1/P2
            BLACKWELL.md             node setup (on the `blackwell` branch)
```

**Branches are hardware-specific and this matters:**
- `main` — 6 GB laptop: torch cu121, fp16, frozen embeddings, batch 16
- `blackwell` — the college node: torch cu128, bf16, presets, Slurm template

---

## 5. Environment

**Laptop:** RTX 2060 6 GB. Python **3.11** venv (system 3.14 has no torch/faiss wheels).
XLM-R base fits *only* with embeddings frozen (192M of 278M params are the 250k-vocab
embedding matrix). Peak VRAM measured at 5.10 GB of 6.4 GB — no headroom for anything else.

**Node:** 2× RTX PRO 6000 Blackwell, exposed as **MIG 1g.24gb slices (25.4 GB each)**,
driver 580.173.02, Slurm (`gpu` partition), 24 cores, 377 GB RAM.

> **The trap:** Blackwell is **sm_120**. torch cu121/cu126 builds carry no sm_120 kernels.
> They install, import, and report `cuda.is_available() == True`, then die at the first
> matmul with *"no kernel image is available for execution on the device"*. Use the
> **cu128** build (`torch==2.8.0`) and run `scripts/check_env.py` before queueing anything.

**Shared-node etiquette — please keep this.** Everything lives under `~/nlp`: venv,
HF cache, data, checkpoints, logs. `scripts/node_env.sh` redirects `HF_HOME`,
`XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `TORCH_HOME` and `MPLCONFIGDIR` inside it, so nothing
pollutes the shared home or `/tmp`. When finished: `bash scripts/cleanup_node.sh --all`.
The node is ~80% full and other people use it.

---

## 6. Plan

The proposal's 16-week plan is in `docs/proposal.pdf` §26. What matters operationally:

**Two independent axes.** Cross-**domain** transfer (RQ1/H2) needs no Indic data at all —
it is measurable in English today. Cross-**language** transfer (RQ2/H1/H5) is the half
that needs Hindi and Hinglish. Do not serialise them.

**Sequencing that follows from that:**

| When | Work | Blocks on |
|---|---|---|
| Now | Finish T1, then T2 (12 LODO folds × 3 seeds) | nothing |
| Now, in parallel | **Start the annotation set** | nothing — and it is calendar-bound |
| Week 3–4 | MASSIVE hi-IN: train EN → test HI = first real T3 number | a download, no annotation |
| Week 5–7 | Contrastive learning (H3) — the actual research contribution | node, large batches |
| Week 8+ | Retrieval (T5), summarization (T6), SHAP (T7) | trained encoder |

**Why annotation must start now.** Nothing public contains Hinglish *complaints* —
L3Cube/GLUECoS are Hinglish but not complaints; CFPB is complaints but not Hinglish.
That intersection is empty, which is why the proposal asks for 1,000–1,500 hand-annotated
examples. Two annotators, agreement measurement, guideline revision — weeks of wall-clock
no GPU can compress. Discovering in week 10 that the set is too small leaves no recovery
room, and the demo input in proposal §22 is Hinglish, so it is a presentation dependency
too.

**Fallback if annotation collapses:** MASSIVE covers 51 languages with real labels. The
cross-language paper survives; the Indian/Hinglish angle drops from core contribution to
case study. Weaker, not fatal.

---

## 7. Open decisions — these need a human

1. **Government and Education domains have no usable public complaint corpus.** Either
   annotate them or drop them from Experiment B. Undecided. Blocks the final domain set.
2. **Ontology v1.** Weakest classes in *both* models are the semantically overlapping
   pairs: `reporting_incorrect_information` (0.41) vs `reporting_investigation_dispute`
   (0.44), `account_management` (0.43) vs `account_access` (0.64). That is an ontology
   boundary problem, not a model problem. Merge them or sharpen the definitions.
3. **7.3% of rows are unmapped** by the ontology on the full corpus. Acceptable now;
   revisit at v1.
4. `Credit card or prepaid card` was merged into `credit_card` rather than `prepaid_card`.
   Defensible (the merged label is mostly credit cards) but reversible in one config line.
5. **Annotation tooling** not chosen — Label Studio or Doccano.
6. **Tracking** not chosen — MLflow or W&B. Pick one before the sweeps multiply.

---

## 8. Traps already hit — do not rediscover these

| Trap | Symptom | Fix |
|---|---|---|
| CFPB API custom User-Agent | HTTP 403 from the CDN | use a curl-style UA |
| CFPB API has **no offset paging** | `frm`/`from`/`offset` silently ignored, every page identical; `search_after` → 424. An early audit produced 3,000 "docs" that were 99 records repeated 30× | stratify over `date_received` day windows |
| MinHash per-permutation hashing | ~25k blake2b calls/doc, hours on the full corpus | hash once to 64 bits, affine permutations, numpy-vectorised |
| Windows CRLF | `$'\r': command not found` when node scripts run | `.gitattributes` forces LF for `.sh`/`.sbatch`/`.py` |
| `results/` in `.gitignore` | also matched `docs/results/`, silently excluding result files | anchor patterns: `/results/` |
| Blackwell sm_120 | plausible-looking CUDA failure at first matmul | cu128 torch + `check_env.py` |
| Two jobs per MIG slice | `nvidia-smi` shows ~21 GB used while torch reports 10 GB peak (caching allocator) | one training job per slice |

---

## 9. How to run everything

```bash
# laptop, from repo root
py -3.11 -m venv .venv && .venv/Scripts/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python scripts/audit_cfpb.py --days 150 --out-dir data/audit         # corpus audit
curl -O https://files.consumerfinance.gov/ccdb/complaints.csv.zip     # 1.43 GB
python scripts/prep_cfpb.py --zip complaints.csv.zip --out-dir data/processed
python scripts/train_baselines.py --data-dir data/processed --out results/baselines
python scripts/train_baselines.py --lodo data/processed/lodo --out results/baselines_lodo

# node (blackwell branch)
git checkout blackwell
python3 -m venv ~/nlp/.venv && . scripts/node_env.sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-blackwell.txt
python scripts/check_env.py            # must print sm_120 ... OK

python scripts/train_transformer.py --preset mig_24gb \
    --model FacebookAI/xlm-roberta-base --epochs 6 --max-len 512 \
    --class-weights balanced --tag indomain --seed 13 --out results/transformers
```

`prep_cfpb.py` is stdlib-only and streams the zip, so it runs on a login node with a bare
interpreter and no GPU. It is deterministic given the same zip and `--seed`, but **the CFPB
file is refreshed daily** — keep the zip you actually used if you need to reproduce a split.

---

## 10. If you change one thing first

Run the LODO sweep (`data/processed/lodo/`, 12 folds already generated). It produces T2,
the cross-domain transfer gap, which is the single number the research question turns on —
and unlike everything multilingual, it needs no new data.
