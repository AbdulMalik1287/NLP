# Dataset Inventory (candidates)

Week-1 deliverable per §26. Nothing here is downloaded yet. Every row must be
**licence-verified** before use, and licences below are recorded from public docs —
re-check at download time. The repo ships preparation scripts, never restricted data.

Selection criteria (§15A): public accessibility, licence, real complaint *narratives*
(not just ratings/labels), domain relevance, label quality, language info,
compatibility with the unified intent ontology.

---

## A. Complaint / grievance data by domain

### Banking & finance — the anchor domain
| Dataset | Link | Size / notes | Licence |
|---|---|---|---|
| **CFPB Consumer Complaint Database** | https://www.consumerfinance.gov/data-research/consumer-complaints/ | ~7M complaints, ~1.5M with free-text narratives; product/sub-product/issue/sub-issue labels; English, US, financial. **Best single source of real complaint narratives.** | US public domain (open data) |
| CFPB bulk CSV / API | https://cfpb.github.io/api/ccdb/ | filtered download + API | same |
| **BANKING77** | https://huggingface.co/datasets/PolyAI/banking77 | 13k online-banking queries, 77 fine-grained intents. Intent-ontology reference + in-domain baseline. | CC-BY-4.0 |
| RBI / Ombudsman annual reports | https://rbi.org.in/Scripts/AnnualReportPublications.aspx | aggregate stats, no narratives — background only | — |

### Telecom / e-commerce / general customer support
| Dataset | Link | Size / notes | Licence |
|---|---|---|---|
| **Customer Support on Twitter** | https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter | 3M tweets, 20+ brands (Apple, Amazon, Uber, telcos, airlines). Real complaints, unlabelled → needs our ontology annotation. | CC BY-NC-SA 4.0 (**non-commercial — fine for research, check redistribution**) |
| Twitter Complaints (Preoţiuc-Pietro et al. 2019) | https://github.com/danielpreotiuc/complaints-social-media | ~3.4k tweets labelled complaint / not-complaint, 9 domains | research use, see repo |
| Complaint severity (Jin & Aletras) | https://github.com/mali-git/complaint_severity (see paper for canonical link) | severity-annotated extension of the above | research use |
| Amazon Reviews 2023 | https://amazon-reviews-2023.github.io/ | huge; low-star reviews as weak e-commerce complaint proxy — **weak labels, use carefully** | research use |
| Consumer Complaint / e-commerce sets on Kaggle | https://www.kaggle.com/search?q=consumer+complaints+in%3Adatasets | mixed quality; audit individually | varies |

### Government / public services (India)
| Source | Link | Notes |
|---|---|---|
| **CPGRAMS / PGPORTAL** | https://pgportal.gov.in/ | central public-grievance portal. Public **reports** exist; per-grievance text is **not** bulk-downloadable → likely manual/scraped subset or synthetic-adjacent, **check ToS before scraping** |
| National Consumer Helpline (INGRAM) | https://consumerhelpline.gov.in/ | grievance categories + some public dashboards |
| data.gov.in — grievance datasets | https://www.data.gov.in/catalogs (search "grievance") | mostly aggregate counts; occasional narrative sets | 
| Swachhata / MyGov complaint feeds | https://www.mygov.in/ | civic complaints, short text |

**Reality check:** government-domain narrative data is the weakest link. If no usable
public source is found in Week 1, make Government the domain covered by the
project-specific annotation set (§15C) or drop it from Experiment B and say so.

### Education
No good public complaint corpus is known. Expect this domain to come from
project-specific annotation, or be dropped. Decide in Week 1, do not stall on it.

---

## B. Multilingual / Indian-language + code-mixed resources

Per §15B these are **not complaint datasets** — they support language-ID, multilingual
representation, and Hinglish robustness, and must not be labelled as complaints.

| Resource | Link | Use |
|---|---|---|
| **MASSIVE** (51 languages, incl. hi-IN) | https://huggingface.co/datasets/AmazonScience/massive | parallel intent classification across languages → **cross-language transfer testbed with real labels** |
| MTOP | https://huggingface.co/datasets/facebook/mtop | multilingual task-oriented intents (en/de/es/fr/th/hi in some versions) |
| **IndicCorp / IndicNLP Suite (AI4Bharat)** | https://ai4bharat.iitm.ac.in/resources | Hindi + Indic monolingual corpora, IndicGLUE benchmark |
| IndicGLUE | https://huggingface.co/datasets/ai4bharat/indic_glue | Indian-language eval tasks |
| **GLUECoS** (Hindi-English code-mixed benchmark) | https://github.com/microsoft/GLUECoS | the standard code-mixed evaluation suite |
| **LinCE** (Linguistic Code-switching Eval) | https://ritual.uh.edu/lince/ | code-switching LID + tasks; hi-en included |
| **L3Cube HingCorpus / HingBERT** | https://github.com/l3cube-pune/code-mixed | large real Hinglish corpus + pretrained Hinglish models |
| Dakshina | https://github.com/google-research-datasets/dakshina | romanised ↔ native script pairs for 12 South Asian langs → **transliteration handling for Hinglish** |
| Samanantar / IITB en-hi parallel corpus | https://www.cfilt.iitb.ac.in/iitb_parallel/ | paraphrase/translation pairs for contrastive positives |
| WiLI-2018 / fastText lid.176 | https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin | LID baseline model (see ENVIRONMENT.md) |

## C. Summarization

| Resource | Link | Use |
|---|---|---|
| **XL-Sum** | https://huggingface.co/datasets/csebuetnlp/xlsum | 44 languages incl. Hindi; mT5 pretraining/compat reference (§13) |
| mT5 XL-Sum checkpoint | https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum | zero-shot summarization warm start |
| Project complaint-summary set | — | **to create**: 300–500 manual complaint→summary pairs (§13) |

## D. Project-specific annotation (§15C)

To be created, since naturally-occurring multilingual complaint data is scarce:
- **~1,000–1,500 complaint examples**, meaningful proportions of English / Hindi / Hinglish, multiple domains
- Schema: `complaint_id, domain, language, intent, text` (+ `similar_complaint_id`, `summary` on subsets)
- 2 annotators on a substantial subset, **Cohen's κ** reported, guidelines revised before the test split is frozen
- Suggested tools: Label Studio (https://labelstud.io/) or Doccano (https://github.com/doccano/doccano)

---

## Leakage rules to enforce at prep time (§17)
Exact duplicates across splits · near-duplicate paraphrases · translated versions of the
same complaint in different splits · source-specific leakage · metadata label leakage ·
synthetic examples in the eval set. Use MinHash/`datasketch` + embedding-similarity sweeps.
Freeze the test set before final model comparison.

## Week-1 audit table to fill in
| dataset | domain | language(s) | #examples | has narratives | label scheme | maps to our ontology? | licence | redistributable | decision |
|---|---|---|---|---|---|---|---|---|---|
