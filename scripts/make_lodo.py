"""Rebuild leave-one-domain-out folds from the three split files.

Each fold gets its own validation set carved from the TRAINING domains only.
Using the global val.jsonl would leak the held-out domain into model selection,
which is exactly the confound the T2 experiment exists to measure.

    python scripts/make_lodo.py --data-dir data/processed --out-dir data/processed/lodo
"""

import argparse
import collections
import json
import os
import random
import re


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="data/processed/lodo")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    rows = []
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        rows.extend(read(os.path.join(a.data_dir, name)))
    print("pooled %d rows" % len(rows))

    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    by_prod = collections.Counter(r["product"] for r in rows)
    manifest = {}

    for prod, n in sorted(by_prod.items(), key=lambda kv: -kv[1]):
        if n < a.min_test:
            print("  skip %-22s only %d rows" % (prod, n))
            continue
        held = [r for r in rows if r["product"] == prod]
        rest = [r for r in rows if r["product"] != prod]

        # stratified val split over the training domains only
        by_intent = collections.defaultdict(list)
        for r in rest:
            by_intent[r["intent"]].append(r)
        tr, va = [], []
        for _, grp in sorted(by_intent.items()):
            rng.shuffle(grp)
            k = max(1, int(len(grp) * a.val_frac))
            va.extend(grp[:k])
            tr.extend(grp[k:])
        rng.shuffle(tr)
        rng.shuffle(va)

        safe = re.sub(r"[^a-z0-9]+", "_", prod.lower()).strip("_")
        write(os.path.join(a.out_dir, "train_wo_%s.jsonl" % safe), tr)
        write(os.path.join(a.out_dir, "val_wo_%s.jsonl" % safe), va)
        write(os.path.join(a.out_dir, "test_%s.jsonl" % safe), held)
        manifest[safe] = {
            "held_out_product": prod,
            "train": len(tr), "val": len(va), "test": len(held),
            "test_intents": len({r["intent"] for r in held}),
            "train_intents": len({r["intent"] for r in tr}),
        }
        print("  %-22s train=%6d val=%5d test=%6d (%d intents in test)"
              % (safe, len(tr), len(va), len(held), manifest[safe]["test_intents"]))

    with open(os.path.join(a.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("%d folds written to %s" % (len(manifest), a.out_dir))


if __name__ == "__main__":
    main()
