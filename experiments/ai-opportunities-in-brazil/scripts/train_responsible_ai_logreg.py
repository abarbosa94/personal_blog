"""Train and evaluate a local TF-IDF logistic-regression RAI classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from conference_pipeline.responsible_ai_context import classify_context
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline


def model() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.98,
            sublinear_tf=True, max_features=50_000,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, max_features=50_000,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            C=2.0, class_weight="balanced", max_iter=2_000,
            solver="liblinear", random_state=20250730,
        )),
    ])


def read_rows(
    validation: Path, validation_context: Path, audit: Path, confirmation: Path
) -> list[dict[str, str]]:
    contexts = {}
    with validation_context.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            contexts[(row["venue"], row["paper_id"])] = row["abstract"]
    rows = []
    with validation.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "source_set": "validation86", "venue": row["venue"],
                "paper_id": row["paper_id"], "title": row["title"],
                "abstract": contexts[(row["venue"], row["paper_id"])],
                "label": row["manual_label"],
            })
    for source_set, path, label_field in (
        ("audit158", audit, "audit_manual_label"),
        ("confirmation60", confirmation, "confirmation_manual_label"),
    ):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "source_set": source_set, "venue": row["venue"],
                    "paper_id": row["paper_id"], "title": row["title"],
                    "abstract": row["abstract"], "label": row[label_field],
                })
    return rows


def feature_text(row: dict[str, str]) -> str:
    dimensions = classify_context(row["title"], row["abstract"])
    rule_tokens = " ".join(f"RULE_DIM_{value}" for value in dimensions)
    return (
        f"VENUE_{row['venue']} "
        f"{'RULE_POSITIVE' if dimensions else 'RULE_NEGATIVE'} {rule_tokens}\n"
        f"TITLE: {row['title']}\nABSTRACT: {row['abstract']}"
    )


def scores(y: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0,
    }


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = []
    for threshold in np.linspace(0.2, 0.8, 121):
        metric = scores(y, (probability >= threshold).astype(int))
        candidates.append((threshold, metric))
    passing = [
        item for item in candidates
        if item[1]["precision"] >= 0.9 and item[1]["recall"] >= 0.9
    ]
    pool = passing or candidates
    threshold, _ = max(
        pool,
        key=lambda item: (
            min(item[1]["precision"], item[1]["recall"]),
            item[1]["f1"],
            -abs(item[0] - 0.5),
        ),
    )
    return float(threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-context", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(
        args.validation, args.validation_context, args.audit, args.confirmation
    )
    text = np.array([feature_text(row) for row in rows])
    y = np.array([row["label"] == "positive" for row in rows], dtype=int)
    groups = np.array([row["venue"] for row in rows])

    stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=20250730)
    stratified_prob = cross_val_predict(
        model(), text, y, cv=stratified, method="predict_proba", n_jobs=1
    )[:, 1]
    venue_cv = GroupKFold(n_splits=len(set(groups)))
    venue_prob = cross_val_predict(
        model(), text, y, groups=groups, cv=venue_cv,
        method="predict_proba", n_jobs=1,
    )[:, 1]
    stratified_pred = (stratified_prob >= 0.5).astype(int)
    venue_pred = (venue_prob >= 0.5).astype(int)
    development = np.array([row["source_set"] != "confirmation60" for row in rows])
    confirmation = ~development
    development_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=20250730
    )
    development_prob = cross_val_predict(
        model(), text[development], y[development], cv=development_cv,
        method="predict_proba", n_jobs=1,
    )[:, 1]
    selected_threshold = choose_threshold(y[development], development_prob)
    holdout_model = model().fit(text[development], y[development])
    confirmation_prob = holdout_model.predict_proba(text[confirmation])[:, 1]
    confirmation_pred = (confirmation_prob >= selected_threshold).astype(int)
    metrics = {
        "records": len(rows),
        "positive": int(y.sum()),
        "negative": int((1 - y).sum()),
        "threshold": 0.5,
        "stratified_5fold": scores(y, stratified_pred),
        "leave_one_venue_out": scores(y, venue_pred),
        "fresh_confirmation60": scores(y[confirmation], confirmation_pred),
        "development_threshold_selection": {
            "threshold": selected_threshold,
            **scores(
                y[development],
                (development_prob >= selected_threshold).astype(int),
            ),
        },
        "leave_one_venue_out_by_venue": {
            venue: scores(y[groups == venue], venue_pred[groups == venue])
            for venue in sorted(set(groups))
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_rows = []
    for index, row in enumerate(rows):
        holdout_index = int(np.sum(confirmation[:index])) if confirmation[index] else None
        output_rows.append({
            **row,
            "truth": int(y[index]),
            "stratified_probability": stratified_prob[index],
            "stratified_prediction": int(stratified_pred[index]),
            "venue_holdout_probability": venue_prob[index],
            "venue_holdout_prediction": int(venue_pred[index]),
            "fresh_holdout_probability": (
                confirmation_prob[holdout_index] if holdout_index is not None else ""
            ),
            "fresh_holdout_prediction": (
                int(confirmation_pred[holdout_index]) if holdout_index is not None else ""
            ),
        })
    with args.predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    fitted = model().fit(text, y)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": fitted, "threshold": selected_threshold,
        "training_records": len(rows), "venues": sorted(set(groups)),
    }, args.model_output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
