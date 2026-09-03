# Running on the RTX PRO 6000 Blackwell node

This branch (`blackwell`) carries the node-specific setup. `main` stays pinned to the
6 GB laptop config. Only four things differ, so merging between them is cheap:
`requirements-blackwell.txt`, `run_blackwell.sbatch`, `configs/presets.json`,
`scripts/check_env.py`, plus the `--preset`/`--precision` flags in the trainer.

---

## 1. The one trap that will waste your afternoon

**RTX PRO 6000 Blackwell (GB202) is compute capability `sm_120`.**

A torch built against CUDA 12.1 or 12.6 — including the `torch==2.4.1+cu121` pin on
`main` — has **no sm_120 kernels**. It installs cleanly, imports cleanly, and
`torch.cuda.is_available()` returns `True`. Then the first matmul dies with:

```
CUDA error: no kernel image is available for execution on the device
```

The fix is a **cu128** build:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-blackwell.txt
python scripts/check_env.py        # run this BEFORE queueing anything
```

`check_env.py` prints the arch list, compares it against the installed device, and runs
a real bf16 matmul — because an arch list alone can still lie. The trainer now performs
the same check at startup and exits with a clear message rather than a CUDA fault.

If the cluster driver is older than ~570 and you cannot upgrade it, skip pip wheels and
use the NGC container: `nvcr.io/nvidia/pytorch:25.06-py3` or newer.

## 2. Use bf16, not fp16

Blackwell has native bf16. It needs no `GradScaler`, will not produce inf/NaN loss
scaling stalls, and costs nothing in speed. Every node preset sets `precision: bf16`;
the laptop preset stays on fp16 because Turing (RTX 2060) has no bf16.

## 3. Presets

```bash
python scripts/train_transformer.py --preset node_full  --model FacebookAI/xlm-roberta-base
python scripts/train_transformer.py --preset mig_24gb   --model FacebookAI/xlm-roberta-large
```

| Preset | Target | Batch | Precision | Freeze embeddings |
|---|---|---|---|---|
| `laptop_6gb` | RTX 2060 6 GB | 16 ×2 accum | fp16 | **yes** (required to fit) |
| `mig_24gb` | one 1g.24gb MIG slice | 32 | bf16 | no |
| `node_full` | full 96 GB card | 128 | bf16 | no |
| `node_full_large` | xlm-roberta-large, 96 GB | 64 | bf16 | no |

Explicit CLI flags always beat the preset.

## 4. What the node unlocks that the laptop cannot do

| Experiment | Laptop (6 GB) | Blackwell |
|---|---|---|
| TF-IDF baselines | ✅ seconds | ✅ |
| mBERT / XLM-R **base** fine-tune | ✅ ~40 min/run | ✅ minutes |
| XLM-R **large** fine-tune | ❌ OOM (~14 GB needed) | ✅ |
| **mT5-base** summarization fine-tune | ❌ OOM | ✅ |
| Contrastive, large in-batch negatives | ⚠️ batch ≤32 | ✅ batch 256–512 |
| Full 3.85M-row corpus, no cap | ❌ RAM-bound | ✅ |
| 3 seeds × 5 models × 6 LODO folds | ⚠️ days | ✅ parallel across MIG slices |

The contrastive line is the important one. In-batch negatives are the mechanism SimCSE
depends on, and their quality scales with batch size. Batch 32 on the laptop is a
compromised version of H3; batch 256+ on the node is the real experiment.

## 5. Parallelism: use MIG slices, not one big job

The proposal lists **16 MIG instances of 1g.24gb**. Almost nothing here needs a whole
96 GB card. Run the experiment matrix as many small concurrent jobs instead:

```bash
for seed in 13 42 7; do
  for model in FacebookAI/xlm-roberta-base google-bert/bert-base-multilingual-cased; do
    sbatch run_blackwell.sbatch scripts/train_transformer.py \
      --preset mig_24gb --model "$model" --seed "$seed" --tag indomain
  done
done
```

Six runs land at once on six slices. The full T1 table finishes in the time one
sequential run would take.

## 6. Data on the node

Do not re-download per job. Once, on a login node with network access:

```bash
export HF_HOME=$SCRATCH/.cache/huggingface
huggingface-cli download FacebookAI/xlm-roberta-base
huggingface-cli download FacebookAI/xlm-roberta-large
huggingface-cli download google-bert/bert-base-multilingual-cased
huggingface-cli download google/mt5-base
curl -O https://files.consumerfinance.gov/ccdb/complaints.csv.zip   # 1.43 GB
python scripts/prep_cfpb.py --zip complaints.csv.zip --out-dir data/processed
```

`prep_cfpb.py` is stdlib-only and streams the zip, so it runs on a login node without
the venv and without a GPU. With node RAM (~306 GB) you can raise `--per-product` far
above the laptop default of 20,000 — the audit's finding F1 caps still apply, but you
are no longer RAM-bound.

## 7. Checklist before the first real run

- [ ] `python scripts/check_env.py` prints `sm_120 ... OK` and the bf16 smoke matmul passes
- [ ] `nvidia-smi` inside a Slurm job shows the GPU (not just on the login node)
- [ ] `HF_HOME` points at scratch, models pre-downloaded
- [ ] `--preset` matches what Slurm actually allocated (MIG slice vs full card)
- [ ] `logs/` and `results/` exist and are writable
- [ ] One `--limit-train 2000` smoke run finishes before you queue the full matrix
