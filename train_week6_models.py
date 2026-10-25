"""
HealthConnect Rwanda Week 6 - train and evaluate feasible models.
Run from the project root:
    python train_week6_models.py

Uses prepared datasets in Data/Processed_Datasets and writes results into
Models/ and Reports/. Does not modify original or processed datasets.
The outputs are educational analyses, not clinically validated systems.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "Processed_Datasets"
MODELS = ROOT / "Models"
REPORTS = ROOT / "Reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
ID_COLS = {
    "AppointmentID", "PatientID", "DoctorID", "FacilityID", "DepartmentID",
    "InsuranceID", "BillingID", "ClaimID", "AdmissionID", "DiagnosisID",
    "FeedbackID"
}
DATE_COLS = {"Date"}


def load_csv(name):
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run prepare_week6_datasets.py first.")
    return pd.read_csv(path)


def clean_features(X):
    X = X.copy()
    for col in DATE_COLS.intersection(X.columns):
        parsed = pd.to_datetime(X[col], errors="coerce")
        X[col + "_month"] = parsed.dt.month
        X[col + "_dayofweek"] = parsed.dt.dayofweek
        X = X.drop(columns=[col])
    X = X.drop(columns=[c for c in ID_COLS if c in X.columns], errors="ignore")
    # Keep category fields as strings, and numeric columns numeric.
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("Unknown").astype(str)
    return X


def build_pipeline(X, estimator, scale_numeric=False):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(("cat", cat_pipe, categorical))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocess", prep), ("model", estimator)])


def safe_split(X, y, classification=False):
    stratify = y if classification and y.value_counts().min() >= 2 else None
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )


def classification_task(dataset, target, task_name, positive_label=None):
    df = load_csv(dataset)
    if target not in df:
        raise ValueError(f"{target} not found in {dataset}")
    y = df[target]
    X = clean_features(df.drop(columns=[target]))
    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    if y.nunique() < 2:
        raise ValueError(f"{task_name}: fewer than two target classes.")

    X_train, X_test, y_train, y_test = safe_split(X, y, True)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE,
        stratify=y_train if y_train.value_counts().min() >= 2 else None,
    )

    candidates = {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    candidate_scores = {}
    for name, estimator in candidates.items():
        candidate_pipe = build_pipeline(X_fit, estimator, scale_numeric=(name == "Logistic Regression"))
        candidate_pipe.fit(X_fit, y_fit)
        val_pred = candidate_pipe.predict(X_val)
        candidate_scores[name] = float(
            f1_score(y_val, val_pred, average="macro", zero_division=0)
        )

    best_name = max(candidate_scores, key=candidate_scores.get)
    best_estimator = candidates[best_name]
    pipe = build_pipeline(
        X_train, best_estimator, scale_numeric=(best_name == "Logistic Regression")
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    # Keep a simple baseline score so a selected model is not judged in isolation.
    baseline = build_pipeline(X_train, DummyClassifier(strategy="most_frequent"))
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    result = {
        "task": task_name,
        "dataset": dataset,
        "target": target,
        "rows": int(len(df)),
        "test_rows": int(len(y_test)),
        "classes": [str(v) for v in sorted(y.unique(), key=str)],
        "selected_model": best_name,
        "validation_f1_macro": candidate_scores[best_name],
        "candidate_validation_f1_macro": candidate_scores,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "baseline_accuracy": float(accuracy_score(y_test, baseline_pred)),
        "baseline_f1_macro": float(f1_score(y_test, baseline_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=pipe.named_steps["model"].classes_).tolist(),
        "confusion_matrix_labels": [str(v) for v in pipe.named_steps["model"].classes_],
        "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True),
        "note": "Dataset labels are proxies/observed alternatives; not clinical validation."
    }
    try:
        classes = list(pipe.named_steps["model"].classes_)
        if len(classes) == 2:
            positive_index = classes.index(positive_label) if positive_label in classes else 1
            probs = pipe.predict_proba(X_test)[:, positive_index]
            result["roc_auc"] = float(roc_auc_score(
                (y_test == classes[positive_index]).astype(int), probs
            ))
        else:
            probs = pipe.predict_proba(X_test)
            result["roc_auc_ovr_macro"] = float(roc_auc_score(
                y_test, probs, multi_class="ovr", average="macro", labels=classes
            ))
    except Exception as exc:
        result["roc_auc_note"] = f"Could not calculate ROC-AUC: {exc}"

    safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in task_name).strip("_")
    joblib.dump(pipe, MODELS / f"{safe_name}.joblib")
    return result


def regression_task(dataset, target, task_name):
    df = load_csv(dataset)
    if target not in df:
        raise ValueError(f"{target} not found in {dataset}")
    y = pd.to_numeric(df[target], errors="coerce")
    X = clean_features(df.drop(columns=[target]))
    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    X_train, X_test, y_train, y_test = safe_split(X, y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE
    )

    candidates = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Decision Tree": DecisionTreeRegressor(
            min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=1.0,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=1.0,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    candidate_scores = {}
    for name, estimator in candidates.items():
        candidate_pipe = build_pipeline(X_fit, estimator)
        candidate_pipe.fit(X_fit, y_fit)
        val_pred = candidate_pipe.predict(X_val)
        candidate_scores[name] = float(mean_absolute_error(y_val, val_pred))

    best_name = min(candidate_scores, key=candidate_scores.get)
    pipe = build_pipeline(X_train, candidates[best_name])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    baseline = build_pipeline(X_train, DummyRegressor(strategy="mean"))
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    nonzero = np.abs(y_test.to_numpy()) > 1e-9
    mape = float(np.mean(np.abs(
        (y_test.to_numpy()[nonzero] - pred[nonzero]) /
        y_test.to_numpy()[nonzero]
    )) * 100) if nonzero.any() else None
    result = {
        "task": task_name, "dataset": dataset, "target": target,
        "rows": int(len(df)), "test_rows": int(len(y_test)),
        "selected_model": best_name,
        "validation_MAE": candidate_scores[best_name],
        "candidate_validation_MAE": candidate_scores,
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": rmse, "R2": float(r2_score(y_test, pred)),
        "MAPE_percent": mape,
        "baseline_MAE": float(mean_absolute_error(y_test, baseline_pred)),
        "baseline_RMSE": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
        "baseline_R2": float(r2_score(y_test, baseline_pred)),
        "note": "Random split; assess prediction-time feature availability before interpreting."
    }
    safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in task_name).strip("_")
    joblib.dump(pipe, MODELS / f"{safe_name}.joblib")
    return result


def clustering_task():
    df = load_csv("patient_clustering.csv")
    # Identifiers are excluded. Categorical features one-hot encoded; impute missing values.
    X = clean_features(df.drop(columns=["PatientID"], errors="ignore"))
    pipe = build_pipeline(
        X,
        KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10),
    )
    labels = pipe.fit_predict(X)
    out = df.copy()
    out["PatientSegment"] = labels
    out.to_csv(DATA / "patient_segments.csv", index=False)
    numeric = out.select_dtypes(include=["number"]).columns.tolist()
    profile = out.groupby("PatientSegment")[numeric].mean(numeric_only=True).round(3)
    profile.to_csv(REPORTS / "patient_cluster_profiles.csv")
    joblib.dump(pipe, MODELS / "patient_clustering_kmeans.joblib")
    return {
        "task": "Patient clustering",
        "rows": int(len(df)),
        "clusters": 4,
        "cluster_sizes": {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        "profile_file": "Reports/patient_cluster_profiles.csv",
        "note": "Exploratory patient segments; assess stability and clinical usefulness before use."
    }


def main():
    results = []
    # Required tasks that can be modelled with current labels/data.
    results.append(classification_task(
        "patient_satisfaction_model.csv", "SatisfactionClass",
        "Patient satisfaction classification (rating-derived)"
    ))
    results.append(classification_task(
        "high_risk_severity_proxy.csv", "HighRiskProxy",
        "High-risk patient classification (severity proxy)"
    ))
    results.append(classification_task(
        "claims_approved_pending_alternative.csv", "ClaimStatusBinary",
        "Insurance claim outcome (Approved vs Pending alternative)"
    ))
    results.append(regression_task(
        "billing_amount_model.csv", "Amount", "Billing amount prediction"
    ))
    results.append(regression_task(
        "length_of_stay_model.csv", "LengthOfStayDays", "Length of stay prediction"
    ))
    results.append(clustering_task())

    # Save one machine-readable metrics file and a readable summary.
    with open(REPORTS / "week6_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    rows = []
    for r in results:
        row = {k: r.get(k) for k in [
            "task", "rows", "test_rows", "selected_model", "validation_f1_macro",
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "baseline_accuracy", "baseline_f1_macro", "roc_auc", "roc_auc_ovr_macro",
            "validation_MAE", "MAE", "RMSE", "R2", "MAPE_percent",
            "baseline_MAE", "baseline_RMSE", "baseline_R2", "clusters", "note"
        ] if k in r}
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPORTS / "week6_baseline_metrics.csv", index=False)

    lines = ["HealthConnect Rwanda Week 6 - model comparison results",
             "Educational baseline results only; not clinically validated.", ""]
    for r in results:
        lines.append(f"## {r['task']}")
        for k, v in r.items():
            if k not in {"classification_report", "confusion_matrix"}:
                lines.append(f"- {k}: {v}")
        if "confusion_matrix" in r:
            lines.append(f"- confusion_matrix_labels: {r['confusion_matrix_labels']}")
            lines.append(f"- confusion_matrix: {r['confusion_matrix']}")
        lines.append("")
    lines += [
        "## Tasks not honestly trainable from current labels",
        "- Hospital admission prediction: no reliable negative labels; do not infer non-admission from unmatched records.",
        "- Three-class insurance outcome: no Rejected labels in the source data.",
        "- Facility clustering is produced separately by build_week6_facility_clustering.py using direct FacilityID record counts; its exploratory limits are documented.",
        "- Anomaly review and permutation feature importance are produced by week6_explainability_anomalies.py; SHAP/LIME is not evidenced by these outputs."
    ]
    (REPORTS / "Week6_Baseline_Model_Report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Model comparison and training complete.")
    print("Metrics: Reports/week6_baseline_metrics.csv")
    print("Report: Reports/Week6_Baseline_Model_Report.txt")
    print("Models: Models/*.joblib")
    print("Patient segments: Data/Processed_Datasets/patient_segments.csv")
    print("Review warnings/limitations before treating any result as meaningful.")


if __name__ == "__main__":
    main()