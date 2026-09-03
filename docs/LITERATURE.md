# Literature Base

Reading list grouped by component. **Verify every arXiv ID / venue against the live page
before citing in the paper** — links are given so a click confirms it.

Priority key: **P0** = read before writing any code · P1 = read before that component starts · P2 = context.

---

## 1. Multilingual encoders (core)

| P | Paper | Link |
|---|---|---|
| **P0** | Conneau et al., *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R), ACL 2020 | https://arxiv.org/abs/1911.02116 |
| **P0** | Devlin et al., *BERT*, NAACL 2019 (BERT + mBERT baselines) | https://arxiv.org/abs/1810.04805 |
| P1 | Conneau & Lample, *Cross-lingual Language Model Pretraining* (XLM) | https://arxiv.org/abs/1901.07291 |
| P1 | Pires et al., *How Multilingual is Multilingual BERT?*, ACL 2019 | https://arxiv.org/abs/1906.01502 |
| P1 | K et al., *Cross-Lingual Ability of Multilingual BERT* | https://arxiv.org/abs/1912.07840 |
| P1 | Khanuja et al., *MuRIL: Multilingual Representations for Indian Languages* | https://arxiv.org/abs/2103.10730 |
| P1 | Feng et al., *LaBSE: Language-agnostic BERT Sentence Embedding* | https://arxiv.org/abs/2007.01852 |
| P2 | Hu et al., *XTREME* (cross-lingual transfer benchmark) | https://arxiv.org/abs/2003.11080 |
| P2 | Ruder et al., *A Survey of Cross-lingual Word Embedding Models* | https://arxiv.org/abs/1706.04902 |

## 2. Complaint understanding (the actual task)

| P | Paper | Link |
|---|---|---|
| **P0** | Preotiuc-Pietro et al., *Automatically Identifying Complaints in Social Media*, ACL 2019 | https://arxiv.org/abs/1906.03890 |
| **P0** | Jin & Aletras, *Modeling the Severity of Complaints in Social Media*, NAACL 2021 | https://arxiv.org/abs/2103.12428 |
| P1 | Jin & Aletras, *Complaint Identification in Social Media with Transformer Networks*, COLING 2020 | https://aclanthology.org/2020.coling-main.157/ |
| P1 | Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (BANKING77) | https://arxiv.org/abs/2003.04807 |
| P1 | Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction* (CLINC150) | https://arxiv.org/abs/1909.02027 |
| P2 | FitzGerald et al., *MASSIVE: 1M-Example Multilingual NLU Dataset* | https://arxiv.org/abs/2204.08582 |

## 3. Cross-domain transfer & limited-data adaptation

| P | Paper | Link |
|---|---|---|
| **P0** | Ramponi & Plank, *Neural Unsupervised Domain Adaptation in NLP — A Survey*, COLING 2020 | https://arxiv.org/abs/2006.00632 |
| P1 | Gururangan et al., *Dont Stop Pretraining: Adapt LM to Domains and Tasks*, ACL 2020 | https://arxiv.org/abs/2004.10964 |
| P1 | Tunstall et al., *SetFit: Efficient Few-Shot Learning Without Prompts* | https://arxiv.org/abs/2209.11055 |
| P1 | Houlsby et al., *Parameter-Efficient Transfer Learning for NLP* (adapters) | https://arxiv.org/abs/1902.00751 |
| P2 | Hu et al., *LoRA: Low-Rank Adaptation* | https://arxiv.org/abs/2106.09685 |

## 4. Contrastive representation learning (the research extension)

| P | Paper | Link |
|---|---|---|
| **P0** | Gao et al., *SimCSE: Simple Contrastive Learning of Sentence Embeddings*, EMNLP 2021 | https://arxiv.org/abs/2104.08821 |
| **P0** | Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019 | https://arxiv.org/abs/1908.10084 |
| P1 | Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020 | https://arxiv.org/abs/2004.11362 |
| P1 | Gunel et al., *Supervised Contrastive Learning for Pre-trained LM Fine-Tuning*, ICLR 2021 | https://arxiv.org/abs/2011.01403 |
| P1 | Reimers & Gurevych, *Making Monolingual Sentence Embeddings Multilingual via Knowledge Distillation* | https://arxiv.org/abs/2004.09813 |
| P2 | Oord et al., *Representation Learning with Contrastive Predictive Coding* (InfoNCE) | https://arxiv.org/abs/1807.03748 |
| P2 | Robinson et al., *Contrastive Learning with Hard Negative Samples* | https://arxiv.org/abs/2010.04592 |

## 5. Code-mixing / Hinglish

| P | Paper | Link |
|---|---|---|
| **P0** | Khanuja et al., *GLUECoS: Evaluation Benchmark for Code-Switched NLP*, ACL 2020 | https://arxiv.org/abs/2004.12376 |
| P1 | Aguilar et al., *LinCE: Centralized Benchmark for Linguistic Code-switching Evaluation*, LREC 2020 | https://arxiv.org/abs/2005.04322 |
| P1 | Nayak & Joshi, *L3Cube-HingCorpus and HingBERT* | https://arxiv.org/abs/2204.08398 |
| P1 | Roark et al., *Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset* | https://arxiv.org/abs/2007.01176 |
| P2 | CALCS shared tasks on code-switching (LinCE portal) | https://ritual.uh.edu/lince/ |

## 6. Language identification

| P | Paper | Link |
|---|---|---|
| **P0** | Joulin et al., *Bag of Tricks for Efficient Text Classification* (fastText) | https://arxiv.org/abs/1607.01759 |
| P1 | Joulin et al., *FastText.zip: Compressing Text Classification Models* (lid.176.ftz) | https://arxiv.org/abs/1612.03651 |
| P2 | Caswell et al., *Language ID in the Wild* (LID failure modes at scale) | https://arxiv.org/abs/2010.14571 |

## 7. Semantic retrieval

| P | Paper | Link |
|---|---|---|
| **P0** | Johnson et al., *Billion-scale Similarity Search with GPUs* (FAISS) | https://arxiv.org/abs/1702.08734 |
| P1 | Karpukhin et al., *Dense Passage Retrieval for Open-Domain QA*, EMNLP 2020 | https://arxiv.org/abs/2004.04906 |
| P1 | Thakur et al., *BEIR: Heterogeneous Benchmark for Zero-shot IR* | https://arxiv.org/abs/2104.08663 |
| P2 | Robertson & Zaragoza, *BM25 and Beyond* | https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf |

## 8. Multilingual summarization

| P | Paper | Link |
|---|---|---|
| **P0** | Xue et al., *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*, NAACL 2021 | https://arxiv.org/abs/2010.11934 |
| P1 | Hasan et al., *XL-Sum: Large-Scale Multilingual Abstractive Summarization*, ACL-IJCNLP 2021 Findings | https://arxiv.org/abs/2106.13822 |
| P1 | Lin, *ROUGE: A Package for Automatic Evaluation of Summaries*, 2004 | https://aclanthology.org/W04-1013/ |
| P1 | Zhang et al., *BERTScore: Evaluating Text Generation with BERT*, ICLR 2020 | https://arxiv.org/abs/1904.09675 |
| P2 | Maynez et al., *On Faithfulness and Factuality in Abstractive Summarization*, ACL 2020 | https://arxiv.org/abs/2005.00661 |

## 9. Explainability (SHAP on the intent classifier only)

| P | Paper | Link |
|---|---|---|
| **P0** | Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP), NeurIPS 2017 | https://arxiv.org/abs/1705.07874 |
| P1 | Ribeiro et al., *Why Should I Trust You? Explaining the Predictions of Any Classifier* (LIME), KDD 2016 | https://arxiv.org/abs/1602.04938 |
| P1 | DeYoung et al., *ERASER: Benchmark to Evaluate Rationalized NLP Models*, ACL 2020 — deletion-based faithfulness | https://arxiv.org/abs/1911.03429 |
| P1 | Jacovi & Goldberg, *Towards Faithfully Interpretable NLP Systems*, ACL 2020 | https://arxiv.org/abs/2004.03685 |
| P2 | Jain & Wallace, *Attention is not Explanation*, NAACL 2019 | https://arxiv.org/abs/1902.10186 |
| P2 | Bastings & Filippova, *The elephant in the interpretability room* | https://arxiv.org/abs/2010.05607 |

## 10. Evaluation & annotation methodology

| P | Paper | Link |
|---|---|---|
| P1 | Artstein & Poesio, *Inter-Coder Agreement for Computational Linguistics*, CL 2008 | https://aclanthology.org/J08-4004/ |
| P1 | Cohen, *A Coefficient of Agreement for Nominal Scales*, 1960 (kappa, the metric we report) | https://doi.org/10.1177/001316446002000104 |
| P1 | Dror et al., *The Hitchhikers Guide to Testing Statistical Significance in NLP*, ACL 2018 | https://aclanthology.org/P18-1128/ |
| P2 | Gebru et al., *Datasheets for Datasets* — template for our dataset documentation | https://arxiv.org/abs/1803.09010 |
| P2 | Mitchell et al., *Model Cards for Model Reporting* | https://arxiv.org/abs/1810.03993 |

---

## Search entry points for the literature review

- ACL Anthology: https://aclanthology.org/ — search "complaint", "code-mixed intent", "cross-domain intent"
- Papers with Code, intent classification: https://paperswithcode.com/task/intent-classification
- Semantic Scholar citation-graph walk outward from XLM-R and Preotiuc-Pietro 2019
- Hugging Face model/dataset cards usually cite the canonical paper — cheap citation check
