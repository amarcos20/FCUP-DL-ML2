"""Tabular Baseline for Heart Murmur Detection (FCUP ML2).

Trains and evaluates classical classifiers on patient-aggregated acoustic features
using 5-Fold Patient-Level Cross-Validation and the official PhysioNet Weighted Accuracy.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features_tabular import get_patient_dataset
from src.metrics import CLASSES, report, weighted_accuracy


def evaluate_model_cv(clf_name="rf", tune_thresholds=True):
    """Run 5-Fold CV on dev folds (0..4) and report weighted accuracy."""
    df = get_patient_dataset()

    # Features: exclude metadata and labels
    meta_cols = ["patient_id", "murmur", "group", "fold"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Only dev set (folds 0..4); test set (fold == -1) remains untouched
    dev_mask = df["fold"] >= 0
    dev_df = df[dev_mask].copy()

    X = dev_df[feature_cols].values
    y = dev_df["murmur"].values
    folds = dev_df["fold"].values

    print(f"\n=======================================================")
    print(f"  Evaluating Baseline: {clf_name.upper()} (5-Fold Patient-Level CV)")
    print(f"  Dev set: {len(dev_df)} patients, {len(feature_cols)} features")
    print(f"=======================================================\n")

    oof_probs = np.zeros((len(dev_df), len(CLASSES)))
    fold_was = []

    for k in range(5):
        train_idx = folds != k
        val_idx = folds == k

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Classifier instantiation
        if clf_name.lower() == "rf":
            clf = RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                class_weight="balanced",
                random_state=42 + k,
                n_jobs=-1,
            )
        elif clf_name.lower() == "hgb":
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                class_weight="balanced",
                random_state=42 + k,
            )
        elif clf_name.lower() == "lr":
            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42 + k,
            )
        else:
            raise ValueError(f"Unknown classifier: {clf_name}")

        clf.fit(X_train_scaled, y_train)

        # Probabilities mapped to CLASSES order: ['Present', 'Unknown', 'Absent']
        classes_in_clf = list(clf.classes_)
        prob_val = clf.predict_proba(X_val_scaled)
        ordered_probs = np.zeros((len(val_idx.nonzero()[0]), len(CLASSES)))
        for c_idx, c_name in enumerate(CLASSES):
            if c_name in classes_in_clf:
                ordered_probs[:, c_idx] = prob_val[:, classes_in_clf.index(c_name)]

        oof_probs[val_idx] = ordered_probs
        fold_preds = [CLASSES[i] for i in np.argmax(ordered_probs, axis=1)]
        wa = weighted_accuracy(y_val, fold_preds)
        fold_was.append(wa)
        print(f"  Fold {k}: Weighted Accuracy = {wa:.4f}")

    print(f"\n--- Cross-Validation Summary ---")
    print(f"Mean Fold WA: {np.mean(fold_was):.4f} +/- {np.std(fold_was):.4f}")

    # Standard Argmax Predictions
    oof_preds_standard = [CLASSES[i] for i in np.argmax(oof_probs, axis=1)]
    rep_standard = report(y, oof_preds_standard)
    print(f"\nOverall Out-of-Fold Weighted Accuracy (Standard Argmax): {rep_standard['weighted_accuracy']:.4f}")
    print("Patient-Level Confusion Matrix (rows = predicted, cols = true):")
    print(pd.DataFrame(rep_standard["confusion"], index=CLASSES, columns=CLASSES))
    print("Per-class recall:", {k: round(v, 3) for k, v in rep_standard["recall"].items()})

    # Optional: Clinical Cost-Sensitive Threshold Tuning
    if tune_thresholds:
        print(f"\n--- Clinical Cost-Sensitive Decision Adjustment ---")
        # Since Present has weight 5, Unknown has weight 3, Absent has weight 1,
        # multiplying probabilities by utility weights often dramatically improves clinical utility!
        weights = np.array([5.0, 3.0, 1.0])
        weighted_probs = oof_probs * weights
        oof_preds_weighted = [CLASSES[i] for i in np.argmax(weighted_probs, axis=1)]
        rep_weighted = report(y, oof_preds_weighted)
        print(f"Weighted Accuracy with Class Weight Utility: {rep_weighted['weighted_accuracy']:.4f}")
        print("Confusion Matrix with Utility Weighting:")
        print(pd.DataFrame(rep_weighted["confusion"], index=CLASSES, columns=CLASSES))
        print("Per-class recall:", {k: round(v, 3) for k, v in rep_weighted["recall"].items()})

    return rep_standard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "hgb", "lr"])
    args = parser.parse_args()

    evaluate_model_cv(clf_name=args.model)
