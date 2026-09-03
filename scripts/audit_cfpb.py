"""CFPB Consumer Complaint Database — Week-1 dataset audit.

Stdlib only (no pandas/torch needed). Two passes:
  1. aggregation pass — exact counts over the whole corpus via the search API aggs
  2. sample pass      — stratified narrative sample for length/language/quality stats

API quirks this script works around (verified 2026-09-03):
  * a custom User-Agent string gets 403 from the CDN; a curl-style UA works
  * NO offset paging. `frm`, `from` and `offset` are all silently ignored and every
    page returns the same hits; `search_after` returns HTTP 424. Deep paging is
    therefore impossible, so the sample is stratified over date windows instead:
    for each sampled day we take the head of both the asc and desc sort.

Usage:
    python audit_cfpb.py --days 120 --per-window 25 --out-dir data/audit
"""

import argparse
import collections
import datetime as dt
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.parse
import urllib.request

API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
UA = {"User-Agent": "curl/8.0", "Accept": "application/json"}  # custom UA strings get 403


def fetch(params, retries=3, timeout=120):
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - audit script: retry then surface
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("failed: %s (%s)" % (url, last))


def find_buckets(block):
    """Agg blocks look like {"doc_count": N, "<field>": {"buckets": [...]}}."""
    if not isinstance(block, dict):
        return []
    if "buckets" in block:
        return block["buckets"]
    for v in block.values():
        if isinstance(v, dict) and "buckets" in v:
            return v["buckets"]
    return []


def sub_buckets(bucket):
    for k, v in bucket.items():
        if isinstance(v, dict) and "buckets" in v:
            return k, v["buckets"]
    return None, []


def aggregation_pass():
    out = {}
    out["total_complaints"] = fetch({"size": 0, "no_aggs": "true"})["hits"]["total"]["value"]
    out["total_with_narrative"] = fetch(
        {"size": 0, "no_aggs": "true", "has_narrative": "true"}
    )["hits"]["total"]["value"]

    full = fetch({"size": 0, "has_narrative": "true"})
    aggs = full.get("aggregations", {})
    dist = {}
    for field, block in aggs.items():
        bs = find_buckets(block)
        if not bs:
            continue
        rows = []
        for b in bs:
            row = {"key": b.get("key"), "count": b.get("doc_count")}
            subname, subs = sub_buckets(b)
            if subs:
                row["sub_field"] = subname
                row["sub"] = [{"key": s.get("key"), "count": s.get("doc_count")} for s in subs]
            rows.append(row)
        dist[field] = rows
    out["distributions"] = dist
    out["narrative_coverage_pct"] = round(
        100.0 * out["total_with_narrative"] / out["total_complaints"], 2
    )
    return out


def sampled_days(n_days, start="2015-01-01", end=None, seed=13):
    end = end or (dt.date.today() - dt.timedelta(days=7)).isoformat()
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    span = (e - s).days
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(span), min(n_days, span)))
    return [(s + dt.timedelta(days=d)).isoformat() for d in picks]


def sample_pass(n_days, per_window):
    """No offset paging exists, so stratify: each sampled day, asc head + desc head."""
    seen = {}
    days = sampled_days(n_days)
    for i, day in enumerate(days, 1):
        for order in ("created_date_desc", "created_date_asc"):
            try:
                r = fetch({
                    "size": per_window,
                    "no_aggs": "true",
                    "has_narrative": "true",
                    "date_received_min": day,
                    "date_received_max": day,
                    "sort": order,
                })
            except RuntimeError as exc:
                sys.stderr.write("\n  warn %s %s: %s\n" % (day, order, exc))
                continue
            for h in r["hits"]["hits"]:
                seen[h["_id"]] = h["_source"]
        sys.stderr.write("\r  day %d/%d (%s) unique=%d" % (i, len(days), day, len(seen)))
        sys.stderr.flush()
        time.sleep(0.15)
    sys.stderr.write("\n")
    return list(seen.values())


XX_RUN = re.compile(r"X{2,}")
NON_ASCII = re.compile(r"[^\x00-\x7F]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def analyse(docs):
    stats = {"n_sampled": len(docs)}
    lengths_ch, lengths_wd = [], []
    redaction_docs = redaction_tokens = 0
    non_ascii_docs = devanagari_docs = empty = 0
    prod = collections.Counter()
    sub = collections.Counter()
    issue = collections.Counter()
    sub_issue = collections.Counter()
    pair = collections.Counter()
    years = collections.Counter()
    firsts = collections.Counter()

    for d in docs:
        t = (d.get("complaint_what_happened") or "").strip()
        if not t:
            empty += 1
            continue
        lengths_ch.append(len(t))
        lengths_wd.append(len(t.split()))
        hits = XX_RUN.findall(t)
        if hits:
            redaction_docs += 1
            redaction_tokens += len(hits)
        if NON_ASCII.search(t):
            non_ascii_docs += 1
        if DEVANAGARI.search(t):
            devanagari_docs += 1
        firsts[t[:200]] += 1
        prod[d.get("product") or "?"] += 1
        sub[d.get("sub_product") or "?"] += 1
        issue[d.get("issue") or "?"] += 1
        sub_issue[d.get("sub_issue") or "?"] += 1
        pair[(d.get("product") or "?", d.get("issue") or "?")] += 1
        dr = d.get("date_received") or ""
        if len(dr) >= 4:
            years[dr[:4]] += 1

    n = len(lengths_ch)

    def pct(x, base=n):
        return round(100.0 * x / base, 2) if base else 0.0

    stats["n_with_text"] = n
    stats["n_empty_text"] = empty
    if n:
        srt_c, srt_w = sorted(lengths_ch), sorted(lengths_wd)
        stats["chars"] = {
            "min": srt_c[0], "max": srt_c[-1],
            "mean": round(statistics.mean(lengths_ch), 1),
            "median": statistics.median(lengths_ch),
            "p90": srt_c[int(0.9 * n) - 1],
        }
        stats["words"] = {
            "min": srt_w[0], "max": srt_w[-1],
            "mean": round(statistics.mean(lengths_wd), 1),
            "median": statistics.median(lengths_wd),
            "p90": srt_w[int(0.9 * n) - 1],
            "over_256_words_pct": pct(sum(1 for w in lengths_wd if w > 256)),
            "over_512_words_pct": pct(sum(1 for w in lengths_wd if w > 512)),
        }
        stats["redaction"] = {
            "docs_with_XX_runs_pct": pct(redaction_docs),
            "mean_runs_per_doc": round(redaction_tokens / n, 2),
        }
        stats["language_signals"] = {
            "docs_with_non_ascii_pct": pct(non_ascii_docs),
            "docs_with_devanagari_pct": pct(devanagari_docs),
            "note": "CFPB is US English. Near-zero Devanagari is the expected result and "
                    "is precisely why the project still needs its own Hindi/Hinglish set.",
        }
        dup_docs = sum(v for v in firsts.values() if v > 1)
        stats["exact_dupes_first200"] = {
            "groups": sum(1 for v in firsts.values() if v > 1),
            "affected_docs_pct": pct(dup_docs),
        }
    stats["label_space"] = {
        "n_products": len(prod), "n_sub_products": len(sub),
        "n_issues": len(issue), "n_sub_issues": len(sub_issue),
        "n_product_issue_pairs": len(pair),
    }
    stats["top_products"] = prod.most_common(20)
    stats["top_issues"] = issue.most_common(30)
    stats["top_sub_issues"] = sub_issue.most_common(30)
    stats["top_product_issue_pairs"] = [
        {"product": k[0], "issue": k[1], "count": v} for k, v in pair.most_common(30)
    ]
    stats["years"] = sorted(years.items())
    if prod:
        stats["product_imbalance_ratio_sample"] = round(
            prod.most_common(1)[0][1] / min(prod.values()), 1
        )
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120, help="date windows to stratify over")
    ap.add_argument("--per-window", type=int, default=25, help="docs per sort order per day")
    ap.add_argument("--out-dir", default="data/audit")
    ap.add_argument("--skip-sample", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("[1/2] aggregation pass (whole corpus)...")
    agg = aggregation_pass()
    print("      total=%(total_complaints)d with_narrative=%(total_with_narrative)d "
          "(%(narrative_coverage_pct)s%%)" % agg)
    with open(os.path.join(a.out_dir, "cfpb_aggregates.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    if a.skip_sample:
        return
    print("[2/2] stratified sample: %d days x %d x 2 sorts..." % (a.days, a.per_window))
    docs = sample_pass(a.days, a.per_window)
    stats = analyse(docs)
    stats["source"] = ("CFPB search API v1, has_narrative=true, stratified over %d random "
                       "days 2015-01-01..now, asc+desc heads (no offset paging exists)"
                       % a.days)
    with open(os.path.join(a.out_dir, "cfpb_sample_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(a.out_dir, "cfpb_sample_head.jsonl"), "w", encoding="utf-8") as f:
        for d in docs[:300]:
            f.write(json.dumps({
                "product": d.get("product"),
                "sub_product": d.get("sub_product"),
                "issue": d.get("issue"),
                "sub_issue": d.get("sub_issue"),
                "date_received": d.get("date_received"),
                "text": (d.get("complaint_what_happened") or "")[:1500],
            }, ensure_ascii=False) + "\n")
    print("      %d unique docs, %d with text" % (stats["n_sampled"], stats["n_with_text"]))
    print("written to %s" % a.out_dir)


if __name__ == "__main__":
    main()
