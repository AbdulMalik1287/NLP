"""CFPB bulk CSV -> unified complaint-intent dataset.

Implements the preprocessing decisions from docs/audit/CFPB_AUDIT.md:
  F1 imbalance   -> per-product reservoir sampling, then a per-intent cap
  F2 label drift -> canonical product map (configs/product_canonical.json)
  F3 redaction   -> X{2,} runs collapse to a single <REDACTED> sentinel
  F4 duplicates  -> exact hash + MinHash/LSH near-duplicate removal BEFORE splitting
  F5 English     -> no language column is written; CFPB is the English anchor only

Stdlib only: streams the zip, never loads the full 3.85M rows into RAM.

Usage:
    python scripts/prep_cfpb.py --zip data/raw/complaints.csv.zip --out-dir data/processed
"""

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import zipfile

XX_RUN = re.compile(r"X{2,}")
WS = re.compile(r"\s+")

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(os.path.dirname(HERE), "configs")


def load_json(name):
    with open(os.path.join(CONFIGS, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- normalising

def normalise_text(t):
    t = XX_RUN.sub("<REDACTED>", t)
    t = WS.sub(" ", t).strip()
    return t


def canon_product(raw, cmap):
    return cmap["map"].get(raw, cmap.get("fallback", "other"))


def map_intent(issue, sub_issue, rules):
    """First matching ordered rule wins. Rules match on issue, then sub_issue."""
    hay = ((issue or "") + " || " + (sub_issue or "")).lower()
    for rule in rules["rules"]:
        for pat in rule["contains"]:
            if pat in hay:
                return rule["intent"]
    return None


# ---------------------------------------------------------------- dedup

def shingles(text, k=5):
    w = text.lower().split()
    if len(w) < k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


class MinHasher:
    """Dependency-free MinHash.

    Each shingle is hashed ONCE to 64 bits, then the n_perm permutations are cheap
    affine maps (a*h + b mod P) over those values. Hashing per permutation instead
    (64 x ~400 blake2b calls per document) made the full corpus run take hours.
    Uses numpy when it is importable and falls back to pure Python so the script
    still runs on a login node with a bare interpreter.
    """

    P = (1 << 61) - 1  # Mersenne prime

    def __init__(self, n_perm=64, bands=16, seed=13):
        assert n_perm % bands == 0
        self.n_perm, self.bands = n_perm, bands
        self.rows = n_perm // bands
        rng = random.Random(seed)
        self.a = [rng.randrange(1, self.P) for _ in range(n_perm)]
        self.b = [rng.randrange(0, self.P) for _ in range(n_perm)]
        try:
            import numpy as np
            self.np = np
            self.a_np = np.array(self.a, dtype=np.uint64)
            self.b_np = np.array(self.b, dtype=np.uint64)
        except ImportError:
            self.np = None

    @staticmethod
    def _h64(s):
        return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")

    def signature(self, sh):
        if not sh:
            return None
        hs = [self._h64(s) & 0x1FFFFFFFFFFFFFFF for s in sh]
        if self.np is not None:
            np = self.np
            h = np.array(hs, dtype=np.uint64)
            # (a[:,None] * h[None,:] + b[:,None]) % P, done in uint64
            m = (self.a_np[:, None] * h[None, :] + self.b_np[:, None]) % np.uint64(self.P)
            return m.min(axis=1).tolist()
        return [min((a * x + b) % self.P for x in hs)
                for a, b in zip(self.a, self.b)]

    def band_keys(self, sig):
        return [
            hashlib.blake2b(
                (",".join(map(str, sig[b * self.rows:(b + 1) * self.rows]))).encode(),
                digest_size=8,
            ).hexdigest()
            for b in range(self.bands)
        ]


def dedup(rows, n_perm, bands, sample_shingles, log_every=5000):
    """Exact-hash pass, then banded MinHash LSH. Keeps the first occurrence."""
    seen_exact = set()
    kept, n_exact = [], 0
    for r in rows:
        h = hashlib.blake2b(r["text"].encode("utf-8"), digest_size=16).hexdigest()
        if h in seen_exact:
            n_exact += 1
            continue
        seen_exact.add(h)
        kept.append(r)

    mh = MinHasher(n_perm, bands)
    buckets = collections.defaultdict(list)
    out, n_near = [], 0
    for i, r in enumerate(kept, 1):
        sh = shingles(r["text"])
        if len(sh) > sample_shingles:  # bound the cost on very long complaints
            sh = set(sorted(sh)[:sample_shingles])
        sig = mh.signature(sh)
        if sig is None:
            out.append(r)
            continue
        keys = mh.band_keys(sig)
        if any(buckets[k] for k in keys):
            n_near += 1
            continue
        for k in keys:
            buckets[k].append(1)
        out.append(r)
        if i % log_every == 0:
            sys.stderr.write("\r  dedup %d/%d kept=%d" % (i, len(kept), len(out)))
            sys.stderr.flush()
    sys.stderr.write("\n")
    return out, n_exact, n_near


# ---------------------------------------------------------------- main passes

def stream_rows(zip_path, cmap, per_product, min_words, seed, limit_rows=0):
    """Reservoir-sample up to per_product narratives per canonical product."""
    rng = random.Random(seed)
    res = collections.defaultdict(list)
    counts = collections.Counter()
    seen = kept_candidates = 0
    t0 = time.time()

    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            rdr = csv.DictReader(text)
            cols = {c.lower().strip(): c for c in (rdr.fieldnames or [])}
            col_narr = cols.get("consumer complaint narrative")
            col_prod = cols.get("product")
            col_sub = cols.get("sub-product")
            col_issue = cols.get("issue")
            col_subissue = cols.get("sub-issue")
            col_date = cols.get("date received")
            col_id = cols.get("complaint id")
            if not col_narr:
                raise SystemExit("no narrative column; found: %s" % rdr.fieldnames)

            for row in rdr:
                seen += 1
                if limit_rows and seen > limit_rows:
                    break
                t = (row.get(col_narr) or "").strip()
                if not t:
                    continue
                t = normalise_text(t)
                if len(t.split()) < min_words:
                    continue
                prod = canon_product(row.get(col_prod) or "", cmap)
                counts[prod] += 1
                kept_candidates += 1
                rec = {
                    "id": row.get(col_id) or "",
                    "text": t,
                    "product_raw": row.get(col_prod) or "",
                    "product": prod,
                    "sub_product": row.get(col_sub) or "",
                    "issue": row.get(col_issue) or "",
                    "sub_issue": row.get(col_subissue) or "",
                    "date": (row.get(col_date) or "")[:10],
                }
                bucket = res[prod]
                if len(bucket) < per_product:
                    bucket.append(rec)
                else:  # reservoir replacement keeps the sample uniform
                    j = rng.randrange(counts[prod])
                    if j < per_product:
                        bucket[j] = rec
                if seen % 200000 == 0:
                    sys.stderr.write(
                        "\r  scanned %d rows, narratives kept-eligible %d, %.0fs"
                        % (seen, kept_candidates, time.time() - t0))
                    sys.stderr.flush()
    sys.stderr.write("\n")
    rows = [r for b in res.values() for r in b]
    rng.shuffle(rows)
    return rows, seen, kept_candidates, counts


def split_rows(rows, seed, val=0.15, test=0.15):
    """Stratified by intent, so rare intents still appear in every split."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["intent"]].append(r)
    tr, va, te = [], [], []
    for _, group in sorted(by.items()):
        rng.shuffle(group)
        n = len(group)
        n_te = max(1, int(n * test)) if n >= 3 else 0
        n_va = max(1, int(n * val)) if n >= 3 else 0
        te.extend(group[:n_te])
        va.extend(group[n_te:n_te + n_va])
        tr.extend(group[n_te + n_va:])
    for s in (tr, va, te):
        rng.shuffle(s)
    return tr, va, te


def write_jsonl(path, rows, fields=("id", "text", "intent", "product", "issue", "date")):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in fields}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/complaints.csv.zip")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--per-product", type=int, default=20000)
    ap.add_argument("--cap-per-intent", type=int, default=5000)
    ap.add_argument("--min-words", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-perm", type=int, default=64)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--max-shingles", type=int, default=400)
    ap.add_argument("--limit-rows", type=int, default=0, help="debug: stop after N CSV rows")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    cmap = load_json("product_canonical.json")
    onto = load_json("intent_ontology_v0.json")
    stats = {"params": vars(a)}

    print("[1/5] streaming %s ..." % a.zip)
    rows, seen, eligible, prod_counts = stream_rows(
        a.zip, cmap, a.per_product, a.min_words, a.seed, a.limit_rows)
    stats["csv_rows_scanned"] = seen
    stats["narratives_eligible"] = eligible
    stats["narratives_by_product_full"] = dict(prod_counts.most_common())
    stats["sampled_before_dedup"] = len(rows)
    print("      scanned %d rows, %d eligible narratives, sampled %d" % (seen, eligible, len(rows)))

    print("[2/5] dedup (exact + MinHash LSH)...")
    rows, n_exact, n_near = dedup(rows, a.n_perm, a.bands, a.max_shingles)
    stats["dropped_exact_duplicates"] = n_exact
    stats["dropped_near_duplicates"] = n_near
    stats["after_dedup"] = len(rows)
    base = stats["sampled_before_dedup"] or 1
    stats["duplicate_drop_pct"] = round(100.0 * (n_exact + n_near) / base, 2)
    print("      dropped %d exact + %d near (%.2f%%), %d remain"
          % (n_exact, n_near, stats["duplicate_drop_pct"], len(rows)))

    print("[3/5] intent mapping...")
    mapped, unmapped_issues = [], collections.Counter()
    for r in rows:
        intent = map_intent(r["issue"], r["sub_issue"], onto)
        if intent is None:
            unmapped_issues[r["issue"]] += 1
            continue
        r["intent"] = intent
        mapped.append(r)
    stats["mapped"] = len(mapped)
    stats["unmapped"] = sum(unmapped_issues.values())
    stats["unmapped_coverage_pct"] = round(
        100.0 * stats["unmapped"] / (len(rows) or 1), 2)
    stats["top_unmapped_issues"] = unmapped_issues.most_common(25)
    print("      mapped %d, unmapped %d (%.2f%%)"
          % (len(mapped), stats["unmapped"], stats["unmapped_coverage_pct"]))

    print("[4/5] capping at %d per intent..." % a.cap_per_intent)
    rng = random.Random(a.seed)
    by_intent = collections.defaultdict(list)
    for r in mapped:
        by_intent[r["intent"]].append(r)
    capped = []
    for intent, group in sorted(by_intent.items()):
        rng.shuffle(group)
        capped.extend(group[:a.cap_per_intent])
    rng.shuffle(capped)
    dist = collections.Counter(r["intent"] for r in capped)
    stats["final_n"] = len(capped)
    stats["intent_distribution"] = dict(dist.most_common())
    stats["n_intents"] = len(dist)
    stats["imbalance_ratio"] = round(max(dist.values()) / min(dist.values()), 2) if dist else 0
    stats["product_distribution"] = dict(
        collections.Counter(r["product"] for r in capped).most_common())
    print("      final %d rows, %d intents, imbalance %s"
          % (len(capped), len(dist), stats["imbalance_ratio"]))

    print("[5/5] splits...")
    tr, va, te = split_rows(capped, a.seed)
    write_jsonl(os.path.join(a.out_dir, "train.jsonl"), tr)
    write_jsonl(os.path.join(a.out_dir, "val.jsonl"), va)
    write_jsonl(os.path.join(a.out_dir, "test.jsonl"), te)
    stats["splits"] = {"train": len(tr), "val": len(va), "test": len(te)}

    # leave-one-product-out files for the intra-financial transfer pilot (T2)
    lodo_dir = os.path.join(a.out_dir, "lodo")
    os.makedirs(lodo_dir, exist_ok=True)
    prods = sorted({r["product"] for r in capped})
    lodo_stats = {}
    for p in prods:
        held = [r for r in capped if r["product"] == p]
        rest = [r for r in capped if r["product"] != p]
        if len(held) < 200 or len(rest) < 1000:
            continue
        safe = re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_")
        write_jsonl(os.path.join(lodo_dir, "train_wo_%s.jsonl" % safe), rest)
        write_jsonl(os.path.join(lodo_dir, "test_%s.jsonl" % safe), held)
        lodo_stats[p] = {"train": len(rest), "test": len(held),
                         "test_intents": len({r["intent"] for r in held})}
    stats["lodo"] = lodo_stats

    with open(os.path.join(a.out_dir, "prep_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("      train=%d val=%d test=%d, %d LODO folds" % (len(tr), len(va), len(te), len(lodo_stats)))
    print("written to %s" % a.out_dir)


if __name__ == "__main__":
    main()
