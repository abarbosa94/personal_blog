"""Leakage-safe local TF-IDF + LightGBM classifier for Responsible AI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from conference_pipeline.responsible_ai_context import classify_context


RANDOM_STATE = 20250730
PARAMETERS = (
    {"num_leaves": 7, "min_child_samples": 10, "learning_rate": 0.04, "n_estimators": 250},
    {"num_leaves": 15, "min_child_samples": 10, "learning_rate": 0.04, "n_estimators": 250},
    {"num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.05, "n_estimators": 200},
)


def estimator(parameters: dict[str, object]) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.98,
            sublinear_tf=True, max_features=20_000,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, max_features=20_000,
        )),
    ])
    classifier = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=8,
        reg_lambda=1.0,
        **parameters,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def feature_text(row: dict[str, str]) -> str:
    dimensions = classify_context(row["title"], row["abstract"])
    rule_tokens = " ".join(f"RULE_DIM_{value}" for value in dimensions)
    return (
        f"{'RULE_POSITIVE' if dimensions else 'RULE_NEGATIVE'} {rule_tokens}\n"
        f"TITLE: {row['title']}\nABSTRACT: {row['abstract']}"
    )


def reviewed_rows(
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
                "dimensions": row["manual_dimensions"],
                "notes": row["review_notes"],
            })
    for source_set, path, label_field, dimensions_field, notes_field in (
        ("audit158", audit, "audit_manual_label", "audit_manual_dimensions", "audit_notes"),
        (
            "confirmation60", confirmation, "confirmation_manual_label",
            "confirmation_manual_dimensions", "confirmation_notes",
        ),
    ):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "source_set": source_set, "venue": row["venue"],
                    "paper_id": row["paper_id"], "title": row["title"],
                    "abstract": row["abstract"], "label": row[label_field],
                    "dimensions": row[dimensions_field], "notes": row[notes_field],
                })
    return rows


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    values: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["venue"], row["paper_id"] or row["title"])
        current = values.get(key)
        if current:
            if (
                current["label"] != row["label"]
                or set(current["dimensions"].split("|")) != set(row["dimensions"].split("|"))
            ):
                raise ValueError(f"Conflicting reviewed duplicate: {key}")
            current["source_set"] += f"|{row['source_set']}"
            continue
        values[key] = {
            **row,
            "scope_status": (
                "borderline" if "borderline" in row["notes"].casefold() else "core"
            ),
        }
    return sorted(values.values(), key=lambda row: (row["venue"], row["paper_id"], row["title"]))


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0,
    }


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, dict[str, object]]:
    candidates = []
    for threshold in np.linspace(0.1, 0.9, 161):
        score = metric(y, (probability >= threshold).astype(int))
        candidates.append((float(threshold), score))
    passing = [
        item for item in candidates
        if item[1]["precision"] >= 0.9 and item[1]["recall"] >= 0.9
    ]
    pool = passing or candidates
    return max(
        pool,
        key=lambda item: (
            min(item[1]["precision"], item[1]["recall"]),
            item[1]["f1"],
            -abs(item[0] - 0.5),
        ),
    )


def strata(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([f"{row['venue']}|{row['label']}" for row in rows])


def select_inner(
    text: np.ndarray, y: np.ndarray, joint_strata: np.ndarray
) -> tuple[dict[str, object], float, list[dict[str, object]]]:
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for parameters in PARAMETERS:
        probability = cross_val_predict(
            estimator(parameters), text, y, cv=inner.split(text, joint_strata),
            method="predict_proba", n_jobs=1,
        )[:, 1]
        threshold, score = choose_threshold(y, probability)
        results.append({
            "parameters": parameters, "threshold": threshold, "metrics": score
        })
    best = max(
        results,
        key=lambda item: (
            min(item["metrics"]["precision"], item["metrics"]["recall"]),
            item["metrics"]["f1"],
        ),
    )
    return best["parameters"], float(best["threshold"]), results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-context", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    raw = reviewed_rows(
        args.validation, args.validation_context, args.audit, args.confirmation
    )
    rows = deduplicate(raw)
    if len(raw) != 304 or len(rows) != 298:
        raise ValueError(f"Expected 304 judgments/298 unique, got {len(raw)}/{len(rows)}")
    text = np.array([feature_text(row) for row in rows])
    y = np.array([row["label"] == "positive" for row in rows], dtype=int)
    joint = strata(rows)

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probabilities = np.zeros(len(rows))
    predictions = np.zeros(len(rows), dtype=int)
    fold_details = []
    for fold, (train, test) in enumerate(outer.split(text, joint), start=1):
        parameters, threshold, inner_results = select_inner(
            text[train], y[train], joint[train]
        )
        fitted = estimator(parameters).fit(text[train], y[train])
        probability = fitted.predict_proba(text[test])[:, 1]
        prediction = (probability >= threshold).astype(int)
        probabilities[test] = probability
        predictions[test] = prediction
        fold_details.append({
            "fold": fold, "train": len(train), "test": len(test),
            "parameters": parameters, "threshold": threshold,
            "metrics": metric(y[test], prediction),
            "inner_candidates": inner_results,
        })

    overall = metric(y, predictions)
    per_venue = {
        venue: metric(
            y[np.array([row["venue"] == venue for row in rows])],
            predictions[np.array([row["venue"] == venue for row in rows])],
        )
        for venue in sorted({row["venue"] for row in rows})
    }
    final_parameters, final_threshold, final_candidates = select_inner(text, y, joint)
    fitted = estimator(final_parameters).fit(text, y)
    payload = {
        "classifier": "responsible-ai-lightgbm-v1",
        "model": fitted,
        "threshold": final_threshold,
        "parameters": final_parameters,
        "training_records": len(rows),
        "training_sha256": hashlib.sha256(
            "\n".join(
                f"{row['venue']}|{row['paper_id']}|{row['label']}|{row['dimensions']}"
                for row in rows
            ).encode()
        ).hexdigest(),
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_output)
    metrics = {
        "judgments": len(raw), "unique_papers": len(rows),
        "duplicates_collapsed": len(raw) - len(rows),
        "positive": int(y.sum()), "negative": int((1 - y).sum()),
        "nested_out_of_fold": overall, "per_venue": per_venue,
        "outer_folds": fold_details,
        "final_parameters": final_parameters,
        "final_threshold": final_threshold,
        "final_inner_candidates": final_candidates,
    }
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with args.ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = []
    for index, row in enumerate(rows):
        output.append({
            **row, "truth": int(y[index]), "probability": probabilities[index],
            "prediction": int(predictions[index]),
            "correct": int(predictions[index]) == int(y[index]),
        })
    with args.predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({
        "nested_out_of_fold": overall,
        "per_venue": per_venue,
        "final_parameters": final_parameters,
        "final_threshold": final_threshold,
    }, indent=2))


if __name__ == "__main__":
    main()
