# Environment Prerequisites

## Python

**Use Python 3.11.** The machine currently has **3.14.3**, which has no PyTorch/FAISS wheels —
create a separate 3.11 environment, do not use the system interpreter.

```bash
# conda (preferred on the GPU node)
conda create -n complaint python=3.11 -y
conda activate complaint

# or venv, if a 3.11 interpreter is on PATH
py -3.11 -m venv .venv && .venv\Scripts\activate      # Windows
python3.11 -m venv .venv && source .venv/bin/activate # Linux/AI node
```

## Install order

Torch must come from the CUDA index *before* the rest, otherwise pip pulls a CPU build.

```bash
pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Laptop-only work (dataset audit, EDA, TF-IDF baselines, API skeleton):

```bash
pip install -r requirements-cpu.txt
```

## GPU / cluster (from the proposal, §24)

| Resource | Provided |
|---|---|
| GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition |
| MIG | 16 instances of 1g.24gb |
| RAM | ~306 GB |
| Stack | CUDA, GPU-enabled PyTorch, NVIDIA NeMo, vLLM/TGI, Slurm, Kubernetes |

Notes:
- Blackwell needs a **recent CUDA runtime**; if `cu121` wheels fail on the node, use the
  newest `cu124`/`cu126` torch build the driver supports. Verify with `nvidia-smi` +
  `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
- A single **1g.24gb MIG slice fits `xlm-roberta-base` fine-tuning** (batch 16–32, seq 128–256).
  `xlm-roberta-large` and mT5-base want a bigger slice or gradient accumulation + fp16/bf16.
- vLLM/TGI/NeMo are **not needed** — this project has no LLM/generation-serving layer.
- FAISS runs on CPU; the dataset is far below the scale that needs GPU indexing.
- Run everything through Slurm (`sbatch`), not on the login node. Get a template
  `run.sbatch` from the lab before Week 2.

## Models to pull (Hugging Face)

```
FacebookAI/xlm-roberta-base        # primary encoder
FacebookAI/xlm-roberta-large       # if VRAM allows
google-bert/bert-base-uncased      # English-only baseline (H1 control)
google-bert/bert-base-multilingual-cased   # mBERT baseline
google/mt5-base                    # summarization
google/muril-base-cased            # optional Indian-language comparison
sentence-transformers/LaBSE        # optional retrieval baseline
csebuetnlp/mT5_multilingual_XLSum  # summarization warm start
```

Language-ID model (fastText, not on HF):
```
https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin   # 126 MB, full
https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz   # 917 KB, compressed
```

Pre-download on a login node with network access:
```bash
export HF_HOME=/path/with/space/.cache/huggingface
huggingface-cli download FacebookAI/xlm-roberta-base
```

## Accounts / access needed before Week 1 closes

- [ ] Hugging Face account + `huggingface-cli login` token (some datasets are gated)
- [ ] Kaggle account + `~/.kaggle/kaggle.json` (several complaint datasets are Kaggle-only)
- [ ] Slurm/AI-node SSH access confirmed, quota + scratch path known
- [ ] MLflow tracking URI **or** a W&B account — pick one, don't run both
- [ ] GitHub repo (this one) + branch protection on `main` if the team is >1 person

## Reproducibility rules (§25)

Every run records: dataset version, preprocessing version, model checkpoint,
hyperparameters, random seed, training config, train/val/test split, eval-script version.
Fix seeds for `random`, `numpy`, `torch`, and set `transformers.set_seed()`.

Target repo layout once code starts:
```
data/  src/  models/  configs/  experiments/  evaluation/  api/  ui/  notebooks/  docs/
```

## Corporate-proxy note

If pip fails TLS verification behind an intercepting proxy, use the system CA store
(`pip install --use-feature=truststore`) rather than `--trusted-host`, and set
`REQUESTS_CA_BUNDLE` for `huggingface_hub`.
