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

def load_labels(meta_csv):
    meta = pd.read_csv(meta_csv)
    required = {"participant", "target", "subset"}
    if not required.issubset(meta.columns):
        raise KeyError(f"meta_info.csv must have columns {required}, got {meta.columns.tolist()}")
    return meta

def align_features_labels(features_df, id_col, labels_df):
    # Merge features with labels, keeping all chunks
    merged = features_df.merge(labels_df, left_on=id_col, right_on="participant", how="inner")
    # Drop any rows with NaNs after merge
    merged = merged.dropna(subset=["target", "subset"])
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
            y_pred_thr = (y_prob >= thr).astype(int)
            ba = balanced_accuracy_score(y_true, y_pred_thr)
            if ba > best_ba:
                best_ba = ba
                best_thr = float(thr)
        return best_thr
    # Default
    return 0.5

def predict_patient_level(model, X_chunks, patient_ids, threshold=0.5):
    """Predict at chunk level and aggregate to patient level using majority vote."""
    # Get chunk-level predictions
    y_chunk_prob = _predict_scores(model, X_chunks)
    y_chunk_pred = (y_chunk_prob >= threshold).astype(int) if y_chunk_prob is not None else model.predict(X_chunks)
    
    # Create DataFrame with chunk predictions
    chunk_preds = pd.DataFrame({
        'patient_id': patient_ids,
        'chunk_pred': y_chunk_pred,
        'chunk_prob': y_chunk_prob if y_chunk_prob is not None else np.nan
    })
    
    # Aggregate to patient level using majority vote
    patient_preds = chunk_preds.groupby('patient_id').agg({
        'chunk_pred': lambda x: (x.sum() > len(x)/2).astype(int),  # Majority vote
        'chunk_prob': 'mean'  # Average probability
    }).reset_index()
    
    patient_preds.columns = ['patient_id', 'patient_pred', 'patient_prob']
    
    return patient_preds, chunk_preds

def evaluate(model, X_chunks, y_true, patient_ids, label, outdir: Path, threshold: float | None = None):
    """Evaluate model using chunk-level predictions aggregated to patient level."""
    # Get patient-level predictions using majority vote
    patient_preds, chunk_preds = predict_patient_level(model, X_chunks, patient_ids, threshold)
    
    # Merge with true labels (assuming patient_ids in y_true correspond to patient_preds)
    # This assumes y_true is already at patient level
    if len(y_true) != len(patient_preds):
        print(f"Warning: Mismatch between y_true length ({len(y_true)}) and patient predictions ({len(patient_preds)})")
        print("Proceeding with available data...")
    
    # Use patient-level predictions for evaluation
    y_pred = patient_preds['patient_pred'].values
    y_prob = patient_preds['patient_prob'].values
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if y_prob is not None and not np.any(np.isnan(y_prob)) else None
    ap = average_precision_score(y_true, y_prob) if y_prob is not None and not np.any(np.isnan(y_prob)) else None

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3.8, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(outdir / f"confusion_matrix_{label}.png", dpi=160)
    plt.close()

    # Save ROC curve
    if y_prob is not None and not np.any(np.isnan(y_prob)):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
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
        pr_prec, pr_rec, _thr = precision_recall_curve(y_true, y_prob)
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
    report = classification_report(y_true, y_pred, digits=4)
    (outdir / f"classification_report_{label}.txt").write_text(report)

    # Save chunk-level predictions for analysis
    chunk_preds.to_csv(outdir / f"chunk_predictions_{label}.csv", index=False)
    patient_preds.to_csv(outdir / f"patient_predictions_{label}.csv", index=False)

    return {
        "subset": label,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
        "average_precision": ap,
        "support": int(np.sum(y_true == 1)),
        "threshold": threshold if threshold is not None else 0.5,
        "n_chunks": len(X_chunks),
        "n_patients": len(y_true)
    }

def save_feature_list(feature_cols, outdir: Path):
    (outdir / "feature_columns.txt").write_text("\n".join(feature_cols))

def main():
    # -----------------------------
    # Hard-coded parameters (edit here)
    # -----------------------------
    RANDOM_STATE = 42
    FEATURES_PRIMARY = "data/features/all_patients_clean_combined.csv"
    FEATURES_FALLBACK = "data/features/clean_egemaps/all_patients_clean_combined.csv"
    META_CSV = "data/daic_woz/meta_info.csv"
    MODEL_ROOT = Path("data/models")
    MODEL_NAME = f"egemaps_clean_majority_vote_{time.strftime('%Y%m%d_%H%M%S')}"
    OUTDIR = ensure_dir(MODEL_ROOT / MODEL_NAME)
    CV_FOLDS = 2
    SCORING = "roc_auc"
    # New knobs
    THRESHOLD_METRIC = "youden"  # 'f1', 'youden', 'bal_acc'
    REFIT_ON_TRAIN_DEV = False    # Avoid threshold drift by default

    print(f"Saving models and results to: {OUTDIR}")

    # -----------------------------
    # Load data
    # -----------------------------
    features_df = try_read_features([FEATURES_PRIMARY, FEATURES_FALLBACK])
    id_col = detect_id_column(features_df)
    
    # Note: We'll filter to numeric columns later after loading labels

    # Load labels
    labels = load_labels(META_CSV)
    data = align_features_labels(features_df, id_col=id_col, labels_df=labels)
    print(f"After join with labels: {data.shape[0]} chunks from {data[id_col].nunique()} patients")

    # Prepare splits
    subsets = split_by_subset(data)
    if "train" not in subsets:
        raise RuntimeError("No training subset available after merge.")
    train_df = subsets["train"]
    dev_df = subsets.get("dev", None)
    test_df = subsets.get("test", None)

    # Filter to only numeric columns for features
    feature_cols = []
    for c in data.columns:
        if c not in [id_col, "target", "subset"]:
            # Check if column is numeric or can be converted to numeric
            try:
                pd.to_numeric(data[c])
                feature_cols.append(c)
            except (ValueError, TypeError):
                print(f"Skipping non-numeric column: {c}")
                continue
    
    print(f"Selected {len(feature_cols)} numeric feature columns")
    save_feature_list(feature_cols, OUTDIR)

    # For training, we need to create a custom CV strategy that works with chunk-level data
    # We'll use GroupKFold to ensure chunks from the same patient stay together
    from sklearn.model_selection import GroupKFold
    
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].astype(int).values
    train_patient_ids = train_df[id_col].values

    # -----------------------------
    # Model selection via CV on train (using GroupKFold to keep patient chunks together)
    # -----------------------------
    grids = build_pipelines(random_state=RANDOM_STATE)
    best_model_name, best_estimator, best_cv_score = None, None, -np.inf

    for name, spec in grids.items():
        print(f"Tuning model: {name}")
        # Use GroupKFold to ensure chunks from same patient stay together
        cv = GroupKFold(n_splits=CV_FOLDS)
        gs = GridSearchCV(
            estimator=spec["pipeline"],
            param_grid=spec["param_grid"],
            scoring=SCORING,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True
        )
        gs.fit(X_train, y_train, groups=train_patient_ids)
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
        X_dev = dev_df[feature_cols].values
        dev_patient_labels = dev_df.groupby(id_col)["target"].first().reset_index()
        y_dev = dev_patient_labels["target"].astype(int).values
        dev_patient_ids = dev_df[id_col].values
        
        # Find best threshold on dev according to chosen metric
        # First get chunk-level predictions and aggregate to patient level
        y_dev_prob = _predict_scores(best_estimator, X_dev)
        # Aggregate chunk probabilities to patient level (mean of probabilities)
        dev_chunk_preds = pd.DataFrame({
            'patient_id': dev_patient_ids,
            'chunk_prob': y_dev_prob
        })
        dev_patient_probs = dev_chunk_preds.groupby('patient_id')['chunk_prob'].mean().values
        
        # Now find threshold using patient-level probabilities
        tuned_threshold = _find_best_threshold(y_dev, dev_patient_probs, metric=THRESHOLD_METRIC)
        m_dev = evaluate(best_estimator, X_dev, y_dev, dev_patient_ids, "dev", OUTDIR, threshold=tuned_threshold)
        metrics_list.append(m_dev)

    # -----------------------------
    # Refit on train+dev, then evaluate on test
    # -----------------------------
    if REFIT_ON_TRAIN_DEV:
        if dev_df is not None and not dev_df.empty:
            train_plus_dev = pd.concat([train_df, dev_df], axis=0, ignore_index=True)
        else:
            train_plus_dev = train_df
        X_trd = train_plus_dev[feature_cols].values
        y_trd = train_plus_dev["target"].values
        best_estimator.fit(X_trd, y_trd)

    if test_df is not None and not test_df.empty:
        X_test = test_df[feature_cols].values
        test_patient_labels = test_df.groupby(id_col)["target"].first().reset_index()
        y_test = test_patient_labels["target"].astype(int).values
        test_patient_ids = test_df[id_col].values
        
        m_test = evaluate(best_estimator, X_test, y_test, test_patient_ids, "test", OUTDIR, threshold=tuned_threshold)
        metrics_list.append(m_test)

    # -----------------------------
    # Save artifacts
    # -----------------------------
    # Save pipeline (includes imputer/scaler/model)
    joblib.dump(best_estimator, OUTDIR / "best_model_pipeline.joblib")
    # Save meta/config
    config = {
        "random_state": RANDOM_STATE,
        "prediction_method": "majority_vote",
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