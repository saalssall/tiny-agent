"""Train and evaluate the command risk classifier.

The model is a character n-gram TF-IDF vectorizer feeding logistic regression.
Evaluation uses cross-validation grouped by template, so every held-out fold
contains command shapes the model did not train on.

The number that matters most is how many risky commands would be auto-approved
at the deployment threshold. The script exits non-zero if that is above zero or
if recall on the risky class drops below --min-recall, so CI catches regressions.

The trained model is exported as plain JSON weights so the app can score
commands in pure Python (see RiskModel in tiny_agent/risk.py). Before saving,
the script checks that the pure-Python scorer reproduces scikit-learn's
probabilities on every training command.

Usage:
    python ml/train.py              # evaluate, then train on everything and save the model
    python ml/train.py --no-save    # evaluate only (used in CI)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline, make_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiny_agent.risk import RiskModel, match_rule

DATA = Path(__file__).with_name("commands.csv")
MODEL_OUT = Path(__file__).resolve().parents[1] / "tiny_agent" / "risk_model.json"

# Probability of "safe" the model must reach before a command is auto-approved.
THRESHOLD = 0.85
NGRAM_RANGE = (2, 5)
RISKY, SAFE = 1, 0


def load_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    commands, labels, groups = [], [], []
    with DATA.open(newline="") as fh:
        for row in csv.DictReader(fh):
            commands.append(row["command"])
            labels.append(RISKY if row["label"] == "risky" else SAFE)
            groups.append(int(row["template"]))
    return commands, np.array(labels), np.array(groups)


def build_pipeline() -> Pipeline:
    """Character n-grams catch flags and paths; whole-word tokens pin down the program name."""
    features = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE, sublinear_tf=True, lowercase=False),
        TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, lowercase=False),
    )
    return make_pipeline(features, LogisticRegression(C=8.0, class_weight="balanced", max_iter=5000))


def evaluate(commands: list[str], y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Grouped cross-validation. Prints a report and returns the headline metrics."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    proba_risky = cross_val_predict(
        build_pipeline(), commands, y, cv=cv, groups=groups, method="predict_proba"
    )[:, 1]
    predicted = (proba_risky >= 0.5).astype(int)

    print("=== 5-fold grouped cross-validation (held-out templates) ===")
    print(classification_report(y, predicted, target_names=["safe", "risky"], digits=3))
    tn, fp, fn, tp = confusion_matrix(y, predicted).ravel()
    print(f"confusion: true-safe={tn} false-risky={fp} false-safe={fn} true-risky={tp}\n")

    # What the deployed gate does: hard rules first, then auto-approve only when P(safe) >= THRESHOLD.
    rule_hit = np.array([match_rule(command) is not None for command in commands])
    model_auto = (1 - proba_risky) >= THRESHOLD
    gate_auto = model_auto & ~rule_hit
    n_safe, n_risky = int(np.sum(y == SAFE)), int(np.sum(y == RISKY))
    safe_auto = int(np.sum(gate_auto & (y == SAFE)))
    risky_auto = int(np.sum(gate_auto & (y == RISKY)))
    print(f"=== deployed gate: hard rules, then model at P(safe) >= {THRESHOLD:.2f} ===")
    print(
        f"hard rules catch:             {int(np.sum(rule_hit & (y == RISKY)))}/{n_risky} risky "
        f"(and wrongly stop {int(np.sum(rule_hit & (y == SAFE)))}/{n_safe} safe)"
    )
    print(f"model alone would auto-approve {int(np.sum(model_auto & (y == RISKY)))} risky; rules block them")
    print(f"safe commands auto-approved:  {safe_auto}/{n_safe}")
    print(f"risky commands auto-approved: {risky_auto}/{n_risky}   <- must be 0\n")

    worst = np.argsort(-np.abs(proba_risky - y))[:12]
    print("=== largest errors (label -> P(risky)) ===")
    for i in worst:
        if (proba_risky[i] >= 0.5) != bool(y[i]):
            label = "risky" if y[i] else "safe"
            print(f"  {label:5s} -> {proba_risky[i]:.2f}   {commands[i]}")
    print()

    recall_risky = tp / (tp + fn)
    return {"recall_risky": recall_risky, "risky_auto_approved": risky_auto, "safe_auto_approved": safe_auto}


def export(pipeline: Pipeline, n_samples: int) -> dict:
    """Turn the fitted pipeline into the JSON bundle RiskModel understands."""
    char_vec, word_vec = (t for _, t in pipeline.named_steps["featureunion"].transformer_list)
    clf = pipeline.named_steps["logisticregression"]
    assert list(clf.classes_) == [SAFE, RISKY]
    return {
        "format": 1,
        "threshold": THRESHOLD,
        "ngram_range": list(NGRAM_RANGE),
        "char": {
            "vocabulary": {k: int(v) for k, v in char_vec.vocabulary_.items()},
            "idf": char_vec.idf_.tolist(),
        },
        "word": {
            "vocabulary": {k: int(v) for k, v in word_vec.vocabulary_.items()},
            "idf": word_vec.idf_.tolist(),
        },
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_samples": n_samples,
    }


def verify_export(bundle: dict, pipeline: Pipeline, commands: list[str]) -> float:
    """Return the largest disagreement between pure-Python scoring and scikit-learn."""
    model = RiskModel(bundle)
    sk = pipeline.predict_proba(commands)[:, 1]
    ours = np.array([model.p_risky(c) for c in commands])
    return float(np.max(np.abs(sk - ours)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-save", action="store_true", help="evaluate only; do not write the model file")
    parser.add_argument(
        "--min-recall", type=float, default=0.80, help="minimum model-only recall on the risky class"
    )
    args = parser.parse_args()

    commands, y, groups = load_data()
    print(
        f"{len(commands)} commands, {int(np.sum(y == SAFE))} safe / {int(np.sum(y == RISKY))} risky, "
        f"{len(set(groups))} templates\n"
    )

    metrics = evaluate(commands, y, groups)
    ok = metrics["recall_risky"] >= args.min_recall and metrics["risky_auto_approved"] == 0
    print(
        f"recall(risky)={metrics['recall_risky']:.3f}  risky auto-approved={metrics['risky_auto_approved']}  "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        return 1
    if args.no_save:
        return 0

    pipeline = build_pipeline().fit(commands, y)
    bundle = export(pipeline, len(commands))
    gap = verify_export(bundle, pipeline, commands)
    print(f"pure-Python scorer vs scikit-learn: max probability gap {gap:.2e}")
    if gap > 1e-6:
        print("FAIL: exported model does not reproduce scikit-learn; not saving")
        return 1
    MODEL_OUT.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"saved model to {MODEL_OUT} ({MODEL_OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
