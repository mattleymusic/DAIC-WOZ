#!/usr/bin/env python3
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc, average_precision_score,
    precision_recall_curve
)
from sklearn.metrics import balanced_accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def try_read_features(paths):
    last_err = None
    for p in paths:
        p = Path(p)
        if p.exists():
            print(f"Loading features from: {p}")
            return pd.read_csv(p)
        last_err = f"Missing: {p}"
    raise FileNotFoundError(f"Could not find features CSV in any of: {paths}. Last error: {last_err}")

def detect_id_column(df):
    candidates = ["patient_id", "participant", "Patient_ID", "pid", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find an ID column in features. Checked: {candidates}")

def aggregate_patient_level(features_df, id_col, agg="median"):
    # Keep only numeric columns for aggregation
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    # Ensure id col is in frame
    if id_col not in features_df.columns:
        raise KeyError(f"ID column {id_col} not present in features")
    # Some numeric cols may be empty; groupby will ignore gracefully
    if agg == "median":
        agg_df = features_df.groupby(id_col)[numeric_cols].median().reset_index()
    elif agg == "mean":
        agg_df = features_df.groupby(id_col)[numeric_cols].mean().reset_index()
    else:
        raise ValueError("agg must be 'median' or 'mean'")
    return agg_df, numeric_cols

def load_labels(meta_csv):
    meta = pd.read_csv(meta_csv)
    required = {"participant", "target", "subset"}
    if not required.issubset(meta.columns):
        raise KeyError(f"meta_info.csv must have columns {required}, got {meta.columns.tolist()}")
    return meta

def align_features_labels(agg_df, id_col, labels_df):
    # labels use 'participant'
    merged = agg_df.merge(labels_df, left_on=id_col, right_on="participant", how="inner")
    # Drop any rows with NaNs after merge (should be none in X after imputation later)
    return merged

def split_by_subset(df):
    subsets = {}
    for s in ["train", "dev", "test"]:
        part = df[df["subset"] == s].copy()
        if not part.empty:
            subsets[s] = part
    # Fallbacks if train/dev missing
    if "train" not in subsets and "dev" in subsets:
        # Use dev as train if train missing
        subsets["train"] = subsets["dev"]
        print("No 'train' subset found. Using 'dev' as 'train'.")
    if "dev" not in subsets and "train" in subsets:
        # Carve a small validation via stratified split if possible
        print("No 'dev' subset found. Proceeding without a dev set.")
    return subsets

def build_pipelines(random_state=42):
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ]
    logreg = Pipeline(steps=base_steps + [
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"))
    ])
    rf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400, random_state=random_state, n_jobs=-1, class_weight="balanced_subsample"
        ))
    ])
    # Tree boosting candidate
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                random_state=random_state
            ))
        ])
    except Exception:  # pragma: no cover
        hgb = None
    grids = {
        "logreg": {
            "pipeline": logreg,
            "param_grid": {
                "clf__C": [0.05, 0.1, 0.5, 1.0, 2.0],
                "clf__penalty": ["l2"]
            }
        },
        "rf": {
            "pipeline": rf,
            "param_grid": {
                "clf__n_estimators": [300, 500, 800],
                "clf__max_depth": [None, 8, 16, 32],
                "clf__min_samples_leaf": [1, 2, 4]
            }
        }
    }
    if 'hgb' not in locals() or hgb is None:
        return grids
    # Add HGB grid
    grids["hgb"] = {
        "pipeline": hgb,
        "param_grid": {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [None, 3, 5],
            "clf__max_iter": [300, 600],
            "clf__min_samples_leaf": [20, 50]
        }
    }
    return grids

def _predict_scores(model, X):
    """Return positive class probabilities when available, otherwise normalized decision scores."""
    y_prob = None
    # Predict probabilities or decision scores for ROC/PR
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Map decision scores to [0,1] via rank-based scaling if needed (rare)
        smin, smax = np.min(scores), np.max(scores)
        y_prob = (scores - smin) / (smax - smin + 1e-8)
    return y_prob

def _find_best_threshold(y_true, y_prob, metric: str = "f1"):
    """Find threshold maximizing the chosen metric on a validation set.

    Supported metrics: 'f1' (default).
    """
    if y_prob is None:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)  # align lengths
    if metric == "f1":
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
        idx = int(np.nanargmax(f1))
        return float(thresholds[idx])
    if metric == "youden":
        fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        idx = int(np.argmax(j))
        return float(roc_thr[idx])
    if metric == "bal_acc":
        # Evaluate balanced accuracy across thresholds from PR curve grid
        best_ba = -1.0
        best_thr = 0.5
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            ba = balanced_accuracy_score(y_true, y_pred)
            if ba > best_ba:
                best_ba = ba
                best_thr = float(thr)
        return best_thr
    # Default
    return 0.5

def evaluate(model, X, y, label, outdir: Path, threshold: float | None = None):
    y_prob = _predict_scores(model, X)
    if threshold is None or y_prob is None:
        y_pred = model.predict(X)
    else:
        y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y, y_prob) if y_prob is not None else None
    ap = average_precision_score(y, y_prob) if y_prob is not None else None

    # Save confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(3.8, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(outdir / f"confusion_matrix_{label}.png", dpi=160)
    plt.close()

    # Save ROC curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y, y_prob)
        plt.figure(figsize=(3.8, 3.2))
        plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
        plt.plot([0,1], [0,1], "k--", alpha=0.5)
        plt.title(f"ROC - {label}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"roc_{label}.png", dpi=160)
        plt.close()
        # Save PR curve
        pr_prec, pr_rec, _thr = precision_recall_curve(y, y_prob)
        plt.figure(figsize=(3.8, 3.2))
        plt.plot(pr_rec, pr_prec, label=f"AP={ap:.3f}")
        plt.title(f"PR Curve - {label}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"pr_{label}.png", dpi=160)
        plt.close()

    # Save classification report
    report = classification_report(y, y_pred, digits=4)
    (outdir / f"classification_report_{label}.txt").write_text(report)

    return {
        "subset": label,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
        "average_precision": ap,
        "support": int(np.sum(y == 1)),
        "threshold": threshold if threshold is not None else 0.5
    }

def save_feature_list(feature_cols, outdir: Path):
    (outdir / "feature_columns.txt").write_text("\n".join(feature_cols))

def main():
    # -----------------------------
    # Hard-coded parameters (edit here)
    # -----------------------------
    RANDOM_STATE = 42
    AGGREGATION = "median"  # 'median' or 'mean'
    FEATURES_PRIMARY = "data/features/all_patients_clean_combined.csv"
    FEATURES_FALLBACK = "data/features/clean_egemaps/all_patients_clean_combined.csv"
    META_CSV = "data/daic_woz/meta_info.csv"
    MODEL_ROOT = Path("data/models")
    MODEL_NAME = f"egemaps_clean_{AGGREGATION}_{time.strftime('%Y%m%d_%H%M%S')}"
    OUTDIR = ensure_dir(MODEL_ROOT / MODEL_NAME)
    CV_FOLDS = 5
    SCORING = "roc_auc"
    # New knobs
    THRESHOLD_METRIC = "youden"  # 'f1', 'youden', 'bal_acc'
    REFIT_ON_TRAIN_DEV = False    # Avoid threshold drift by default

    print(f"Saving models and results to: {OUTDIR}")

    # -----------------------------
    # Load data
    # -----------------------------
    features_df = try_read_features([FEATURES_PRIMARY, FEATURES_FALLBACK])
    # Make best effort to coerce non-numeric feature columns later
    id_col = detect_id_column(features_df)
    # Coerce potential numeric strings to numeric (without touching id col)
    for c in features_df.columns:
        if c != id_col:
            features_df[c] = pd.to_numeric(features_df[c], errors="ignore")

    # Aggregate to patient-level
    agg_df, numeric_cols = aggregate_patient_level(features_df, id_col=id_col, agg=AGGREGATION)
    print(f"Aggregated to patient-level: {agg_df.shape[0]} patients, {len(numeric_cols)} numeric features")

    # Load labels
    labels = load_labels(META_CSV)
    data = align_features_labels(agg_df, id_col=id_col, labels_df=labels)
    print(f"After join with labels: {data.shape[0]} patients")

    # Prepare splits
    subsets = split_by_subset(data)
    if "train" not in subsets:
        raise RuntimeError("No training subset available after merge.")
    train_df = subsets["train"]
    dev_df = subsets.get("dev", None)
    test_df = subsets.get("test", None)

    feature_cols = [c for c in agg_df.columns if c != id_col]
    save_feature_list(feature_cols, OUTDIR)

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").values
    y_train = train_df["target"].astype(int).values

    # -----------------------------
    # Model selection via CV on train
    # -----------------------------
    grids = build_pipelines(random_state=RANDOM_STATE)
    best_model_name, best_estimator, best_cv_score = None, None, -np.inf

    for name, spec in grids.items():
        print(f"Tuning model: {name}")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(
            estimator=spec["pipeline"],
            param_grid=spec["param_grid"],
            scoring=SCORING,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True
        )
        gs.fit(X_train, y_train)
        cv_results = pd.DataFrame(gs.cv_results_)
        cv_results.to_csv(OUTDIR / f"cv_results_{name}.csv", index=False)
        print(f"Best {name} CV {SCORING}: {gs.best_score_:.4f} with params: {gs.best_params_}")

        if gs.best_score_ > best_cv_score:
            best_cv_score = gs.best_score_
            best_estimator = gs.best_estimator_
            best_model_name = name

    if best_estimator is None:
        raise RuntimeError("No model was selected during CV.")

    print(f"Selected model: {best_model_name} (CV {SCORING}={best_cv_score:.4f})")

    # -----------------------------
    # Evaluate on dev (if available) and tune threshold
    # -----------------------------
    metrics_list = []
    tuned_threshold = 0.5
    if dev_df is not None and not dev_df.empty:
        X_dev = dev_df[feature_cols].apply(pd.to_numeric, errors="coerce").values
        y_dev = dev_df["target"].astype(int).values
        # Find best threshold on dev according to chosen metric
        y_dev_prob = _predict_scores(best_estimator, X_dev)
        tuned_threshold = _find_best_threshold(y_dev, y_dev_prob, metric=THRESHOLD_METRIC)
        m_dev = evaluate(best_estimator, X_dev, y_dev, "dev", OUTDIR, threshold=tuned_threshold)
        metrics_list.append(m_dev)

    # -----------------------------
    # Refit on train+dev, then evaluate on test
    # -----------------------------
    if REFIT_ON_TRAIN_DEV:
        if dev_df is not None and not dev_df.empty:
            train_plus_dev = pd.concat([train_df, dev_df], axis=0, ignore_index=True)
        else:
            train_plus_dev = train_df
        X_trd = train_plus_dev[feature_cols].apply(pd.to_numeric, errors="coerce").values
        y_trd = train_plus_dev["target"].astype(int).values
        best_estimator.fit(X_trd, y_trd)

    if test_df is not None and not test_df.empty:
        X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").values
        y_test = test_df["target"].astype(int).values
        m_test = evaluate(best_estimator, X_test, y_test, "test", OUTDIR, threshold=tuned_threshold)
        metrics_list.append(m_test)

        # Save per-patient predictions
        y_test_prob = _predict_scores(best_estimator, X_test)
        y_test_pred = (y_test_prob >= tuned_threshold).astype(int) if y_test_prob is not None else best_estimator.predict(X_test)
        preds_test = pd.DataFrame({
            "participant": test_df["participant"].values,
            "y_true": y_test,
            "y_prob": y_test_prob,
            "y_pred": y_test_pred
        })
        preds_test.to_csv(OUTDIR / "predictions_test.csv", index=False)
    # Save dev predictions if available
    if dev_df is not None and not dev_df.empty:
        y_dev_prob = _predict_scores(best_estimator, X_dev)
        y_dev_pred = (y_dev_prob >= tuned_threshold).astype(int) if y_dev_prob is not None else best_estimator.predict(X_dev)
        preds_dev = pd.DataFrame({
            "participant": dev_df["participant"].values,
            "y_true": y_dev,
            "y_prob": y_dev_prob,
            "y_pred": y_dev_pred
        })
        preds_dev.to_csv(OUTDIR / "predictions_dev.csv", index=False)

    # -----------------------------
    # Save artifacts
    # -----------------------------
    # Save pipeline (includes imputer/scaler/model)
    joblib.dump(best_estimator, OUTDIR / "best_model_pipeline.joblib")
    # Save meta/config
    config = {
        "random_state": RANDOM_STATE,
        "aggregation": AGGREGATION,
        "features_csv_used": (
            str(Path(FEATURES_PRIMARY)) if Path(FEATURES_PRIMARY).exists()
            else str(Path(FEATURES_FALLBACK))
        ),
        "meta_csv": META_CSV,
        "feature_columns_count": len(feature_cols),
        "model_name": best_model_name,
        "cv_folds": CV_FOLDS,
        "scoring": SCORING,
        "selected_cv_score": float(best_cv_score),
        "tuned_threshold": float(tuned_threshold),
        "threshold_metric": THRESHOLD_METRIC,
        "refit_on_train_dev": REFIT_ON_TRAIN_DEV
    }
    (OUTDIR / "config.json").write_text(json.dumps(config, indent=2))

    # Save metrics
    if metrics_list:
        pd.DataFrame(metrics_list).to_json(OUTDIR / "metrics.json", orient="records", indent=2)

    # Optional: export simple feature importances if RF or coefficients if LR
    try:
        clf = best_estimator.named_steps.get("clf", None)
        if isinstance(clf, RandomForestClassifier):
            importances = clf.feature_importances_
            imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
            imp_df.to_csv(OUTDIR / "feature_importances.csv", index=False)
        elif isinstance(clf, LogisticRegression):
            coefs = clf.coef_.ravel()
            coef_df = pd.DataFrame({"feature": feature_cols, "coef": coefs}).sort_values("coef", key=np.abs, ascending=False)
            coef_df.to_csv(OUTDIR / "feature_coefficients.csv", index=False)
    except Exception as e:
        print(f"Skipping feature importance export: {e}")

    print("Done.")

if __name__ == "__main__":
    main()