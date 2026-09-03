"""TF-IDF baselines for complaint-intent classification (table T1 / T2 floor).

CPU only, sklearn. These are the floor every transformer result must beat; if a
transformer loses to these, the fine-tune is broken, not the finding.

Usage:
    python scripts/train_baselines.py --data-dir data/processed --out results/baselines
    python scripts/train_baselines.py --lodo data/processed/lodo --out results/baselines_lodo
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.svm import LinearSVC


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def xy(rows):
    return [r["text"] for r in rows], [r["intent"] for r in rows]


def majority_baseline(y_tr, y_te, labels):
    top = max(set(y_tr), key=y_tr.count)
    pred = [top] * len(y_te)
    return {
        "model": "majority_class",
        "majority_label": top,
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "macro_f1": round(f1_score(y_te, pred, average="macro", labels=labels,
                                   zero_division=0), 4),
        "weighted_f1": round(f1_score(y_te, pred, average="weighted", labels=labels,
                                      zero_division=0), 4),
    }


def evaluate(name, clf, Xtr, y_tr, Xte, y_te, labels, seed, t_fit):
    t0 = time.time()
    pred = clf.predict(Xte)
    return {
        "model": name,
        "seed": seed,
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "macro_f1": round(f1_score(y_te, pred, average="macro", labels=labels,
                                   zero_division=0), 4),
        "weighted_f1": round(f1_score(y_te, pred, average="weighted", labels=labels,
                                      zero_division=0), 4),
        "fit_seconds": round(t_fit, 1),
        "predict_seconds": round(time.time() - t0, 1),
        "per_class": classification_report(y_te, pred, labels=labels, output_dict=True,
                                           zero_division=0),
    }, pred


def run_pair(train_rows, test_rows, seed, max_features, out_dir, tag):
    X_tr_raw, y_tr = xy(train_rows)
    X_te_raw, y_te = xy(test_rows)
    labels = sorted(set(y_tr) | set(y_te))

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True, strip_accents="unicode")
    t0 = time.time()
    Xtr = vec.fit_transform(X_tr_raw)
    Xte = vec.transform(X_te_raw)
    vec_seconds = round(time.time() - t0, 1)

    results = [majority_baseline(y_tr, y_te, labels)]
    preds = {}

    t0 = time.time()
    lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                            n_jobs=-1, random_state=seed)
    lr.fit(Xtr, y_tr)
    r, p = evaluate("tfidf_logreg", lr, Xtr, y_tr, Xte, y_te, labels, seed, time.time() - t0)
    results.append(r)
    preds["tfidf_logreg"] = p

    t0 = time.time()
    sv = LinearSVC(C=0.5, class_weight="balanced", random_state=seed)
    sv.fit(Xtr, y_tr)
    r, p = evaluate("tfidf_linearsvc", sv, Xtr, y_tr, Xte, y_te, labels, seed, time.time() - t0)
    results.append(r)
    preds["tfidf_linearsvc"] = p

    out = {
        "tag": tag,
        "seed": seed,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_labels": len(labels),
        "labels": labels,
        "vectorizer": {"max_features": max_features, "ngram_range": [1, 2],
                       "vocab_size": len(vec.vocabulary_), "fit_seconds": vec_seconds},
        "results": results,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "%s_seed%d.json" % (tag, seed)), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    best = max(r for r in results if "macro_f1" in r)
    cm_model = "tfidf_linearsvc"
    cm = confusion_matrix(y_te, preds[cm_model], labels=labels)
    np.savetxt(os.path.join(out_dir, "%s_seed%d_cm_%s.csv" % (tag, seed, cm_model)),
               cm, fmt="%d", delimiter=",", header=",".join(labels), comments="")
    return out


def summarise(all_runs):
    by_model = {}
    for run in all_runs:
        for r in run["results"]:
            by_model.setdefault(r["model"], []).append(r["macro_f1"])
    lines = ["| model | macro-F1 mean | std | runs |", "|---|---|---|---|"]
    for m, vals in sorted(by_model.items(), key=lambda kv: -np.mean(kv[1])):
        lines.append("| %s | %.4f | %.4f | %d |"
                     % (m, float(np.mean(vals)), float(np.std(vals)), len(vals)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--lodo", default="", help="run leave-one-domain-out folds from this dir")
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 7])
    ap.add_argument("--max-features", type=int, default=200000)
    a = ap.parse_args()

    runs = []
    if a.lodo:
        for train_path in sorted(glob.glob(os.path.join(a.lodo, "train_wo_*.jsonl"))):
            dom = re.sub(r"^train_wo_|\.jsonl$", "", os.path.basename(train_path))
            test_path = os.path.join(a.lodo, "test_%s.jsonl" % dom)
            if not os.path.exists(test_path):
                continue
            tr, te = read_jsonl(train_path), read_jsonl(test_path)
            print("[LODO] held-out=%s train=%d test=%d" % (dom, len(tr), len(te)))
            for seed in a.seeds:
                runs.append(run_pair(tr, te, seed, a.max_features, a.out, "lodo_%s" % dom))
                print("   seed %d macro-F1: %s" % (seed, {
                    r["model"]: r["macro_f1"] for r in runs[-1]["results"]}))
    else:
        tr = read_jsonl(os.path.join(a.data_dir, "train.jsonl"))
        te = read_jsonl(os.path.join(a.data_dir, "test.jsonl"))
        print("[in-domain] train=%d test=%d" % (len(tr), len(te)))
        for seed in a.seeds:
            runs.append(run_pair(tr, te, seed, a.max_features, a.out, "indomain"))
            print("   seed %d macro-F1: %s" % (seed, {
                r["model"]: r["macro_f1"] for r in runs[-1]["results"]}))

    table = summarise(runs)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "SUMMARY.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# Baseline results\n\n%s\n" % table)
    print("\n" + table)
    print("\nwritten to %s" % a.out)


if __name__ == "__main__":
    main()
