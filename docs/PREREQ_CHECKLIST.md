# Week-1 Prerequisite Checklist

## Environment

- [ ] Python 3.11 env created (system 3.14 is unusable — no torch/faiss wheels)
- [ ] `torch` CUDA build installs and `torch.cuda.is_available()` is True on the AI node
- [ ] `pip install -r requirements.txt` clean on the node
- [ ] `pip install -r requirements-cpu.txt` clean on the laptop
- [ ] Slurm job submits and sees a GPU (`sbatch` hello-world running `nvidia-smi`)
- [ ] `HF_HOME` / cache path on a disk with space; models pre-downloaded
- [ ] `lid.176.bin` downloaded and loads in fastText
- [ ] Config + tracking harness decided (Hydra + MLflow **or** W&B — pick one, not both)

## Access

- [ ] Hugging Face token (`huggingface-cli login`)
- [ ] Kaggle API key at `~/.kaggle/kaggle.json`
- [ ] AI-node SSH access + storage quota and scratch path known
- [ ] GitHub repo created, whole team has push access

## Data audit — the real Week-1 work

- [ ] CFPB complaint database inspected: narrative coverage, label distribution, size
- [ ] Twitter customer-support corpus licence read (CC BY-NC-SA — check redistribution)
- [ ] Preotiuc-Pietro and Jin & Aletras complaint sets obtained
- [ ] BANKING77 + MASSIVE (hi-IN) pulled as ontology and cross-language references
- [ ] Government-domain source found **or** decision recorded to annotate/drop it
- [ ] Education-domain source found **or** decision recorded to annotate/drop it
- [ ] Hinglish resources checked: GLUECoS, LinCE, L3Cube HingCorpus, Dakshina
- [ ] Audit table in `DATASETS.md` filled, one row per candidate
- [ ] Licence text saved per dataset under `docs/licences/`

## Design decisions to freeze

- [ ] Final domain set for Experiment B (leave-one-domain-out)
- [ ] Unified intent ontology v0, derived from actual label distributions (proposal §8)
- [ ] Annotation tool chosen (Label Studio or Doccano) + guidelines v0 drafted
- [ ] Split strategy + leakage checks (MinHash near-dup, translation-pair detection)

## Reading

- [ ] All **P0** papers in `LITERATURE.md` read, notes folded into the literature-review draft
- [ ] arXiv IDs and venues in `LITERATURE.md` verified against the live pages
