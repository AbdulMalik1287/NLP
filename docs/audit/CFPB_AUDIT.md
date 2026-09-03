# CFPB Consumer Complaint Database — Dataset Audit

**Run:** 2026-09-03 · `scripts/audit_cfpb.py --days 150 --per-window 25`
**Source:** CFPB search API v1 (`consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`)
**Raw output:** `docs/audit/cfpb_aggregates.json`, `docs/audit/cfpb_sample_stats.json`

Aggregate counts are **exact, over the whole corpus** (server-side aggregations).
Text statistics come from a **7,250-document stratified sample** (150 random days,
2015-01-01 to now, asc+desc heads per day).

---

## 1. Headline numbers

| Metric | Value |
|---|---|
| Total complaints | **17,516,902** |
| **With free-text narrative** | **3,846,901 (21.96%)** |
| Product categories (raw) | 21 buckets (~10 after canonicalisation) |
| Issue categories (raw) | **173** |
| Sub-issues seen in sample | 202 |
| Submission channel, narrative subset | Web, 100% |
| Median narrative length | **131 words** (mean 200.6, p90 425) |
| Narratives >256 words | 23.6% |
| Narratives >512 words | 6.9% |
| Language | English only — 0.01% non-ASCII, **0.00% Devanagari** |

The proposal's estimate of "~7M complaints, ~1.5M with narratives" is **stale and low**.
Corrected in `docs/DATASETS.md`.

---

## 2. Five findings that change the plan

### F1 — This is a credit-reporting corpus, not a general banking corpus

Product distribution over all 3.85M narratives:

| Product | Narratives | Share |
|---|---|---|
| Credit reporting or other personal consumer reports | 1,672,265 | 43.5% |
| Credit reporting, credit repair services, … (old label) | 807,499 | 21.0% |
| Debt collection | 443,933 | 11.5% |
| Checking or savings account | 188,564 | 4.9% |
| Mortgage | 146,950 | 3.8% |
| Credit card | 128,424 | 3.3% |
| Money transfer, virtual currency, or money service | 121,549 | 3.2% |
| Credit card or prepaid card | 108,683 | 2.8% |
| Student loan | 63,000 | 1.6% |
| Vehicle loan or lease | 53,434 | 1.4% |
| *(11 smaller/legacy buckets)* | ~112,600 | ~2.9% |

**Credit reporting alone is 65.3%** of all narratives. Two issue labels
("Incorrect information on your report" 1,205,337 and "Improper use of your report"
664,169) cover ~48% of the corpus by themselves.

*Consequence:* do **not** train on a random CFPB sample — the model will learn
"dispute a credit report" and little else. **Cap per class** (e.g. 5,000/intent) and
sample deliberately across products. The natural imbalance ratio in the sample is
**1183:1** between the largest and smallest product.

### F2 — Label vocabulary is versioned, not stable

CFPB renamed categories over the years, so the same concept appears under several keys:

- `Credit card` **and** `Credit card or prepaid card` **and** `Prepaid card`
- `Payday loan` / `Payday loan, title loan, or personal loan` / `Payday loan, title loan, personal loan, or advance loan`
- `Bank account or service` (retired) vs `Checking or savings account` (current)
- `Credit reporting` (retired) vs the two longer credit-reporting labels
- `Money transfers` vs `Money transfer, virtual currency, or money service`

*Consequence:* a canonicalisation map is **mandatory before any split**, otherwise
"cross-domain" splits leak the same domain under two names — which would silently
inflate the T2 transfer result. Canonicalisation is now Day-1 work, and 21 raw
product buckets collapse to roughly 10 canonical ones.

### F3 — 78.7% of narratives contain `XXXX` redactions (mean 14.4 runs per doc)

CFPB scrubs names, dates, account numbers and **amounts**. Real example from the sample:

> "I submited a auto loan application on XXXX/XXXX/2013 at XXXX of XXXX. The deal was
> XXXX finaliized that same day…"

*Consequences, all three components:*
- **Classification** — `XXXX` becomes a high-frequency token. Normalise to a single
  `<REDACTED>` sentinel; do not leave raw runs, and check it does not become a
  SHAP-salient token (it will otherwise pollute T7).
- **Summarization** — the proposal's motivating example ("₹399 was charged") depends on
  amounts that CFPB has removed. **CFPB cannot supply the summarization eval set.**
  The 300–500 manual complaint-summary pairs must come from your own annotation.
- **Retrieval** — near-identical redaction patterns create spurious similarity.

Sample text also contains raw typos ("submited", "finaliized"). That is genuine user
text and should be kept, not corrected.

### F4 — 23% of documents are templated near-duplicates

623 duplicate groups by first-200-characters over 7,250 sampled docs, affecting
**22.98%** of them. These are credit-repair form letters filed en masse.

*Consequence:* this is the **largest leakage risk in the project** (§17). MinHash
dedup is not optional and must run *before* splitting. Expect to drop roughly a fifth
of the corpus. Report the exact figure — "we removed N templated duplicates (X%)" is a
credible methodology slide.

### F5 — Zero multilingual signal

0.01% of narratives contain any non-ASCII character; **0.00% contain Devanagari**.

*Consequence:* CFPB contributes **English, one domain (finance)**. It cannot support
RQ2 (cross-language) or Experiment B (cross-domain) on its own. It is the *in-domain
English anchor*, nothing more. Every cross-language number must come from MASSIVE hi-IN
plus your own annotated Hindi/Hinglish set; every cross-domain number needs at least
two more domains from other sources.

---

## 3. What CFPB can and cannot deliver

| Project requirement | CFPB verdict |
|---|---|
| In-domain English intent classification (T1) | ✅ Ample — millions of labelled narratives |
| Intent ontology grounding (§8) | ✅ 173 issues + 202 sub-issues to map |
| Cross-domain transfer (T2, Exp. B) | ⚠️ Only *within* finance (credit-reporting vs debt-collection vs mortgage as pseudo-domains) |
| Cross-language transfer (T3) | ❌ None — English only |
| Hinglish robustness (H5) | ❌ None |
| Summarization pairs (T6) | ❌ Redaction destroys the factual detail |
| Retrieval corpus (T5) | ✅ Good, after dedup |
| Licence | ✅ US public domain, redistributable |

**Sub-domain trick worth using:** CFPB's own products can act as *pseudo-domains* for a
first leave-one-out experiment — train on debt collection + mortgage + credit card,
test on credit reporting. That gets a real T2 number in week 1 without waiting for any
other dataset. Label it honestly as *intra-financial* transfer, weaker than true
cross-domain transfer, and treat it as a pilot for the real thing.

---

## 4. Preprocessing decisions this audit settles

1. `max_length = 256` tokens, head+tail truncation — covers 76% intact, 512 costs ~2× compute for the remaining 17%
2. Normalise `X{2,}` runs → single `<REDACTED>` token
3. Canonicalise 21 product labels → ~10; freeze the map in `configs/product_canonical.yaml`
4. MinHash (`datasketch`) dedup at threshold 0.8 **before** splitting; log the drop count
5. Cap per intent at ~5,000; report both capped and natural-distribution results
6. Drop narratives under ~20 words (sample min is 2 words — those are unusable)
7. Split by **canonical product**, never randomly, for any transfer experiment

## 5. API notes for whoever runs this next

- A custom `User-Agent` gets **HTTP 403** from the CDN. Use a curl-style UA.
- **There is no offset paging.** `frm`, `from` and `offset` are silently ignored — every
  page returns identical hits — and `search_after` returns **HTTP 424**. An earlier run of
  this audit produced 3,000 "documents" that were 99 unique records repeated 30×.
  The script now stratifies over `date_received_min`/`max` day windows instead.
- Aggregation blocks nest as `{"doc_count": N, "<field>": {"buckets": [...]}}`.
- Aggregate counts are server-side and exact; only the text statistics are sampled.
- For the full 3.85M narratives, use the bulk CSV (~2–3 GB) rather than the API:
  https://files.consumerfinance.gov/ccdb/complaints.csv.zip

## 6. Next actions

- [ ] Write `configs/product_canonical.yaml` (F2)
- [ ] Draft intent ontology v0 by mapping the 173 issues onto the §8 categories
- [ ] Pull the bulk CSV for the real training run; API sampling was for the audit only
- [ ] Run MinHash dedup, record the exact drop percentage (F4)
- [ ] Start the intra-financial leave-one-product-out pilot → first T2 number
- [ ] Audit MASSIVE hi-IN next — it is the only near-term source of a real T3 number
