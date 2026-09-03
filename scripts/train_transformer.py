"""Fine-tune a transformer encoder for complaint-intent classification (T1/T2/T3).

Tuned for a 6 GB GPU (RTX 2060). The memory trick that makes XLM-R fit:

    xlm-roberta-base is 278M params, but 192M of those are the 250k-vocab
    embedding matrix. Freezing word embeddings drops grads + Adam state for
    69% of the model, taking training from ~5.9 GB (OOM-prone) to ~3.4 GB.

Defaults: fp16 AMP, frozen embeddings, seq 256, batch 16. Override for a bigger card.

Usage:
    python scripts/train_transformer.py --model FacebookAI/xlm-roberta-base
    python scripts/train_transformer.py --model google-bert/bert-base-multilingual-cased
    python scripts/train_transformer.py --train data/processed/lodo/train_wo_mortgage.jsonl \
        --test data/processed/lodo/test_mortgage.jsonl --tag lodo_mortgage
"""

import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class ComplaintDS(Dataset):
    def __init__(self, rows, tok, label2id, max_len):
        self.rows, self.tok, self.l2i, self.max_len = rows, tok, label2id, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=self.max_len,
                       padding=False)
        enc["labels"] = self.l2i[r["intent"]]
        return enc


def collate(batch, tok):
    labels = torch.tensor([b.pop("labels") for b in batch], dtype=torch.long)
    out = tok.pad(batch, return_tensors="pt")
    out["labels"] = labels
    return out


def freeze_embeddings(model, verbose=True):
    """Freeze word/position embeddings: the bulk of the params, none of the task signal."""
    frozen = total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if ".embeddings.word_embeddings" in name or ".embeddings.position_embeddings" in name \
                or ".embeddings.token_type_embeddings" in name:
            p.requires_grad = False
            frozen += p.numel()
    if verbose:
        print("  params total=%.1fM frozen=%.1fM trainable=%.1fM"
              % (total / 1e6, frozen / 1e6, (total - frozen) / 1e6))
    return total, frozen


@torch.no_grad()
def evaluate(model, loader, device, id2label, amp, amp_dtype=torch.float16):
    model.eval()
    preds, golds = [], []
    t0 = time.time()
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            logits = model(**batch).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        golds.extend(labels.cpu().tolist())
    y_pred = [id2label[i] for i in preds]
    y_true = [id2label[i] for i in golds]
    labels_sorted = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro",
                                   labels=labels_sorted, zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted",
                                      labels=labels_sorted, zero_division=0), 4),
        "eval_seconds": round(time.time() - t0, 1),
        "n": len(y_true),
        "per_class": classification_report(y_true, y_pred, labels=labels_sorted,
                                           output_dict=True, zero_division=0),
    }


def apply_preset(a, ap):
    """Preset values fill in only what the user did not pass explicitly."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs", "presets.json")
    with open(path, encoding="utf-8") as f:
        presets = json.load(f)["presets"]
    if a.preset not in presets:
        raise SystemExit("unknown preset %r; available: %s" % (a.preset, sorted(presets)))
    cfg = presets[a.preset]
    explicit = {act.dest for act in ap._actions
                if any(opt in sys.argv for opt in act.option_strings)}
    for k, v in cfg.items():
        if k == "comment":
            continue
        if k == "freeze_embeddings":
            if "no_freeze_embeddings" not in explicit:
                a.no_freeze_embeddings = not v
            continue
        if hasattr(a, k) and k not in explicit:
            setattr(a, k, v)
    print("preset %s: %s" % (a.preset, cfg.get("comment", "")))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--test", default="data/processed/test.jsonl")
    ap.add_argument("--out", default="results/transformers")
    ap.add_argument("--tag", default="indomain")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--no-freeze-embeddings", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16",
                    help="bf16 on Blackwell/Ampere+: no GradScaler, better stability")
    ap.add_argument("--preset", default="", help="named preset from configs/presets.json")
    ap.add_argument("--limit-train", type=int, default=0, help="debug: truncate train set")
    ap.add_argument("--save-model", default="")
    ap.add_argument("--class-weights", choices=["none", "balanced"], default="none",
                    help="balanced matches the TF-IDF baselines, which use class_weight=balanced")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="stop if val macro-F1 has not improved for N epochs (0 = off)")
    ap.add_argument("--select-best", action="store_true",
                    help="report test on the best-val-macro-F1 epoch, not the last epoch")
    a = ap.parse_args()

    if a.preset:
        a = apply_preset(a, ap)
    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (not a.no_amp) and device == "cuda" and a.precision != "fp32"
    amp_dtype = torch.bfloat16 if a.precision == "bf16" else torch.float16
    if a.precision == "bf16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 requested but this GPU does not support it; use --precision fp16")
    print("device=%s precision=%s amp=%s model=%s" % (device, a.precision, amp, a.model))
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        arches = torch.cuda.get_arch_list()
        if ("sm_%d%d" % cap) not in arches:
            raise SystemExit(
                "torch %s has no kernels for sm_%d%d (%s). Blackwell needs a cu128 build; "
                "run scripts/check_env.py." % (torch.__version__, cap[0], cap[1],
                                               torch.cuda.get_device_name(0)))

    tr = read_jsonl(a.train)
    va = read_jsonl(a.val) if os.path.exists(a.val) else []
    te = read_jsonl(a.test)
    if a.limit_train:
        tr = tr[:a.limit_train]
    labels = sorted({r["intent"] for r in tr} | {r["intent"] for r in te})
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    print("train=%d val=%d test=%d labels=%d" % (len(tr), len(va), len(te), len(labels)))

    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = AutoConfig.from_pretrained(a.model, num_labels=len(labels),
                                     id2label=i2l, label2id=l2i)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, config=cfg).to(device)
    total, frozen = (0, 0)
    if not a.no_freeze_embeddings:
        total, frozen = freeze_embeddings(model)

    def mk(rows, shuffle):
        return DataLoader(ComplaintDS(rows, tok, l2i, a.max_len), batch_size=a.batch_size,
                          shuffle=shuffle, collate_fn=lambda b: collate(b, tok),
                          num_workers=0, pin_memory=(device == "cuda"))

    dl_tr, dl_te = mk(tr, True), mk(te, False)
    dl_va = mk(va, False) if va else None

    if a.class_weights == "balanced":
        counts = collections.Counter(r["intent"] for r in tr)
        w = torch.tensor([len(tr) / (len(labels) * counts[l]) for l in labels],
                         dtype=torch.float32, device=device)
        print("  class weights: min=%.3f max=%.3f" % (w.min().item(), w.max().item()))
    else:
        w = None
    criterion = torch.nn.CrossEntropyLoss(weight=w)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    steps = max(1, (len(dl_tr) // a.grad_accum) * a.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * a.warmup_ratio), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and a.precision == "fp16"))

    import copy
    hist = []
    best_val, best_epoch, best_state, stale = -1.0, 0, None, 0
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        running, t_ep = 0.0, time.time()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl_tr, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch.pop("labels")
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                logits = model(**batch).logits
                loss = criterion(logits.float(), target) / a.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * a.grad_accum
            if step % a.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 100 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                print("\r  ep%d %d/%d loss=%.4f peakVRAM=%.2fGB"
                      % (ep, step, len(dl_tr), running / step, mem), end="", flush=True)
        print()
        ep_stats = {"epoch": ep, "train_loss": round(running / max(1, len(dl_tr)), 4),
                    "epoch_seconds": round(time.time() - t_ep, 1)}
        if dl_va:
            ep_stats["val"] = evaluate(model, dl_va, device, i2l, amp, amp_dtype)
            vf1 = ep_stats["val"]["macro_f1"]
            print("  ep%d val macro-F1=%s" % (ep, vf1))
            if vf1 > best_val:
                best_val, best_epoch, stale = vf1, ep, 0
                if a.select_best:
                    best_state = copy.deepcopy(
                        {k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
        hist.append(ep_stats)
        if a.early_stop_patience and stale >= a.early_stop_patience:
            print("  early stop: no val improvement for %d epochs (best ep%d=%.4f)"
                  % (stale, best_epoch, best_val))
            break

    if a.select_best and best_state is not None:
        print("  loading best-val checkpoint from epoch %d (val macro-F1 %.4f)"
              % (best_epoch, best_val))
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_stats = evaluate(model, dl_te, device, i2l, amp, amp_dtype)
    print("TEST macro-F1=%s acc=%s (best val ep%d=%.4f)"
          % (test_stats["macro_f1"], test_stats["accuracy"], best_epoch, best_val))

    out = {
        "tag": a.tag,
        "model": a.model,
        "seed": a.seed,
        "config": vars(a),
        "params_total_M": round(total / 1e6, 1) if total else None,
        "params_frozen_M": round(frozen / 1e6, 1) if total else None,
        "n_train": len(tr), "n_test": len(te), "n_labels": len(labels),
        "history": hist,
        "best_val_macro_f1": round(best_val, 4),
        "best_val_epoch": best_epoch,
        "selected_checkpoint": ("best_val_epoch_%d" % best_epoch) if a.select_best else "last_epoch",
        "test": test_stats,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if device == "cuda" else None,
        "total_seconds": round(time.time() - t_start, 1),
    }
    os.makedirs(a.out, exist_ok=True)
    safe = a.model.split("/")[-1]
    path = os.path.join(a.out, "%s_%s_seed%d.json" % (a.tag, safe, a.seed))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("written %s" % path)

    if a.save_model:
        model.save_pretrained(a.save_model)
        tok.save_pretrained(a.save_model)
        print("model saved to %s" % a.save_model)


if __name__ == "__main__":
    main()
