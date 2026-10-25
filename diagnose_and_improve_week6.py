"""Diagnose Week 6 target feasibility and compare simple model candidates.

Run from the project root:
    python diagnose_and_improve_week6.py

This script does not modify original datasets or overwrite existing models.
It writes diagnostic results to Reports/ and saves any candidate models separately.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score, mean_absolute_percentage_error
)

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "Data" / "Original_Datasets"
PROCESSED = ROOT / "Data" / "Processed_Datasets"
REPORTS = ROOT / "Reports"
MODELS = ROOT / "Models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
CANDIDATE_DIR = MODELS / "Diagnostic_Candidates"
CANDIDATE_DIR.mkdir(exist_ok=True)

ID_HINTS = ("id", "identifier", "patientid", "appointmentid", "doctorid",
            "facilityid", "departmentid", "admissionid", "claimid", "invoiceid")

def locate_file(folder, name):
    exact = folder / name
    if exact.exists():
        return exact
    target = name.lower()
    for p in folder.iterdir() if folder.exists() else []:
        if p.is_file() and p.name.lower() == target:
            return p
    return None

def remove_ids_and_dates(df, target):
    X = df.drop(columns=[target]).copy()
    # IDs are linkage keys, not generalizable predictors.
    id_cols = [c for c in X.columns if any(h in str(c).lower() for h in ID_HINTS)]
    X = X.drop(columns=id_cols, errors="ignore")
    # Convert dates to simple calendar features; do not pass raw date strings.
    for col in list(X.columns):
        if "date" in str(col).lower() or "time" in str(col).lower():
            parsed = pd.to_datetime(X[col], errors="coerce")
            if parsed.notna().mean() >= 0.5:
                X[col + "_month"] = parsed.dt.month
                X[col + "_dayofweek"] = parsed.dt.dayofweek
                X = X.drop(columns=[col])
    # Convert bools to categorical-compatible strings.
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(str)
    return X

def build_preprocessor(X, scale_numeric=False):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(num_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols)
    ])

def summarize_relationships():
    lines = ["HealthConnect Rwanda Week 6 - target and linkage diagnostics", ""]
    workbook = locate_file(ORIGINAL, "HC_Clinical_Transactions.xlsx")
    if workbook:
        try:
            xls = pd.ExcelFile(workbook)
            lines.append(f"Clinical transactions workbook: {workbook.name}")
            lines.append("Sheets: " + ", ".join(xls.sheet_names))
            sheet_dfs = {}
            for sheet in xls.sheet_names:
                d = pd.read_excel(workbook, sheet_name=sheet)
                sheet_dfs[sheet.lower()] = d
                lines.append(f"- {sheet}: {len(d)} rows; columns: {', '.join(map(str, d.columns))}")
            appt = sheet_dfs.get("appointments")
            adm = sheet_dfs.get("admissions")
            if appt is not None and adm is not None:
                lines.append("")
                lines.append("Appointment/admission linkage checks (counts only; no labels are generated):")
                for key in ("AppointmentID", "PatientID"):
                    a_col = next((c for c in appt.columns if str(c).lower() == key.lower()), None)
                    m_col = next((c for c in adm.columns if str(c).lower() == key.lower()), None)
                    if a_col and m_col:
                        aset = set(appt[a_col].dropna().astype(str))
                        mset = set(adm[m_col].dropna().astype(str))
                        matched = len(aset & mset)
                        lines.append(
                            f"- {key}: unique appointment-side IDs={len(aset)}, "
                            f"unique admission-side IDs={len(mset)}, shared unique IDs={matched}"
                        )
                        if key.lower() == "patientid":
                            lines.append("  Note: patient-level overlap alone does not prove which appointment led to an admission.")
                    else:
                        lines.append(f"- {key}: cannot compare; column absent from one or both sheets.")
        except Exception as exc:
            lines.append(f"Could not inspect clinical transactions workbook: {exc}")
    else:
        lines.append("Clinical transactions workbook not found; admission linkage not rechecked.")

    claims = locate_file(ORIGINAL, "Insurance_Claims.csv")
    if claims:
        try:
            cdf = pd.read_csv(claims)
            status_col = next((c for c in cdf.columns if str(c).lower() in ("status", "claimstatus", "claim_status")), None)
            lines.append("")
            lines.append(f"Claims source: {claims.name}; rows={len(cdf)}")
            lines.append("Columns: " + ", ".join(map(str, cdf.columns)))
            if status_col:
                lines.append("Claim status counts: " + str(cdf[status_col].fillna("<missing>").value_counts(dropna=False).to_dict()))
                statuses = set(cdf[status_col].dropna().astype(str).str.strip().str.lower())
                lines.append("Rejected status present: " + str(any("reject" in s for s in statuses)))
        except Exception as exc:
            lines.append(f"Could not inspect claims source: {exc}")

    workbook2 = locate_file(ORIGINAL, "HC_Clinical_Transactions.xlsx")
    if workbook2:
        try:
            lab = pd.read_excel(workbook2, sheet_name="Laboratory_Tests")
            result_col = next((c for c in lab.columns if str(c).lower() in ("result", "testresult", "resultvalue")), None)
            lines.append("")
            if result_col:
                numeric = pd.to_numeric(lab[result_col], errors="coerce")
                lines.append(f"Lab result numeric parse: {numeric.notna().sum()} / {len(lab)} numeric values.")
                lines.append(f"Lab result sample values: {lab[result_col].dropna().astype(str).head(8).tolist()}")
                lines.append("Do not apply clinical thresholds without units and validated reference ranges.")
        except Exception as exc:
            lines.append(f"Could not inspect lab result field: {exc}")

    return lines

def evaluate_classification(task, filename, target, multiclass=False, exclude_extra=None):
    path = locate_file(PROCESSED, filename)
    if not path:
        return [f"\n## {task}\nPrepared dataset not found: {filename}"], []
    df = pd.read_csv(path)
    lines = [f"\n## {task}", f"Dataset: {filename}; rows={len(df)}; columns={len(df.columns)}"]
    if target not in df:
        return lines + [f"Target column '{target}' not found."], []
    y = df[target]
    valid = y.notna()
    df = df.loc[valid].copy()
    y = df[target]
    lines.append("Target counts: " + str(y.astype(str).value_counts(dropna=False).to_dict()))
    lines.append(f"Unique target classes: {y.nunique()}")
    if y.nunique() < 2:
        return lines + ["Cannot train: fewer than two target classes."], []

    X = remove_ids_and_dates(df, target)
    if exclude_extra:
        X = X.drop(columns=[c for c in exclude_extra if c in X], errors="ignore")
    # Drop columns with only one nonmissing value.
    X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True) <= 1], errors="ignore")
    lines.append("Predictor columns used: " + ", ".join(map(str, X.columns)))
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError as exc:
        return lines + [f"Split failed: {exc}"], []

    pre = build_preprocessor(X, scale_numeric=True)
    candidates = [
        ("Dummy majority baseline", DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(
            n_estimators=150, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        )),
    ]
    rows = []
    for name, estimator in candidates:
        pipe = Pipeline([("preprocess", pre), ("model", estimator)])
        try:
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            row = {
                "task": task, "model": name, "rows": len(df), "test_rows": len(yte),
                "accuracy": accuracy_score(yte, pred),
                "balanced_accuracy": balanced_accuracy_score(yte, pred),
                "precision_macro": precision_score(yte, pred, average="macro", zero_division=0),
                "recall_macro": recall_score(yte, pred, average="macro", zero_division=0),
                "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
                "roc_auc": np.nan,
                "error": ""
            }
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(Xte)
                try:
                    if y.nunique() == 2:
                        row["roc_auc"] = roc_auc_score(yte, proba[:, 1])
                    else:
                        row["roc_auc"] = roc_auc_score(
                            yte, proba, multi_class="ovr", average="macro",
                            labels=pipe.named_steps["model"].classes_
                        )
                except Exception:
                    pass
            rows.append(row)
            if name != "Dummy majority baseline":
                cm = confusion_matrix(yte, pred, labels=pipe.named_steps["model"].classes_)
                cm_path = REPORTS / f"diagnostic_confusion_{task.lower().replace(' ', '_').replace('/', '_')}_{name.lower().replace(' ', '_')}.csv"
                pd.DataFrame(cm, index=pipe.named_steps["model"].classes_,
                             columns=pipe.named_steps["model"].classes_).to_csv(cm_path)
                # Save only separately, never overwrite existing models.
                safe_task = "".join(ch.lower() if ch.isalnum() else "_" for ch in task).strip("_")
                joblib.dump(pipe, CANDIDATE_DIR / f"{safe_task}_{name.lower().replace(' ', '_')}.joblib")
        except Exception as exc:
            rows.append({"task": task, "model": name, "rows": len(df),
                         "test_rows": len(yte), "error": repr(exc)})
    lines.append("Results are in Reports/Week6_Diagnostic_Model_Comparison.csv.")
    return lines, rows

def evaluate_regression(task, filename, target):
    path = locate_file(PROCESSED, filename)
    if not path:
        return [f"\n## {task}\nPrepared dataset not found: {filename}"], []
    df = pd.read_csv(path)
    lines = [f"\n## {task}", f"Dataset: {filename}; rows={len(df)}; columns={len(df.columns)}"]
    if target not in df:
        return lines + [f"Target column '{target}' not found."], []
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.loc[df[target].notna()].copy()
    y = df[target]
    X = remove_ids_and_dates(df, target)
    X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True) <= 1], errors="ignore")
    lines.append(f"Target range: min={y.min():.4f}; median={y.median():.4f}; max={y.max():.4f}")
    lines.append("Predictor columns used: " + ", ".join(map(str, X.columns)))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = build_preprocessor(X, scale_numeric=True)
    candidates = [
        ("Dummy mean baseline", DummyRegressor(strategy="mean")),
        ("Random Forest", RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1
        )),
    ]
    rows = []
    for name, estimator in candidates:
        pipe = Pipeline([("preprocess", pre), ("model", estimator)])
        try:
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            nonzero = np.abs(yte.to_numpy()) > 1e-9
            mape = mean_absolute_percentage_error(yte.to_numpy()[nonzero], pred[nonzero]) * 100 if nonzero.any() else np.nan
            rows.append({
                "task": task, "model": name, "rows": len(df), "test_rows": len(yte),
                "MAE": mean_absolute_error(yte, pred),
                "RMSE": mean_squared_error(yte, pred) ** 0.5,
                "R2": r2_score(yte, pred),
                "MAPE_percent_nonzero_targets": mape,
                "error": ""
            })
            if name != "Dummy mean baseline":
                safe_task = "".join(ch.lower() if ch.isalnum() else "_" for ch in task).strip("_")
                joblib.dump(pipe, CANDIDATE_DIR / f"{safe_task}_random_forest.joblib")
        except Exception as exc:
            rows.append({"task": task, "model": name, "rows": len(df),
                         "test_rows": len(yte), "error": repr(exc)})
    lines.append("Results are in Reports/Week6_Diagnostic_Model_Comparison.csv.")
    return lines, rows

def main():
    report = summarize_relationships()
    all_rows = []

    classification_tasks = [
        ("Patient satisfaction classification", "patient_satisfaction_model.csv", "SatisfactionClass", False, None),
        ("High-risk classification (severity proxy)", "high_risk_severity_proxy.csv", "HighRiskProxy", False, None),
        # Exclude Amount if it duplicates ClaimAmount; checked dynamically below.
    ]
    for task, filename, target, multiclass, exclude in classification_tasks:
        lines, rows = evaluate_classification(task, filename, target, multiclass, exclude)
        report.extend(lines)
        all_rows.extend(rows)

    claims_path = locate_file(PROCESSED, "claims_approved_pending_alternative.csv")
    claims_exclude = []
    if claims_path:
        cdf = pd.read_csv(claims_path)
        if "ClaimAmount" in cdf and "Amount" in cdf:
            a = pd.to_numeric(cdf["ClaimAmount"], errors="coerce")
            b = pd.to_numeric(cdf["Amount"], errors="coerce")
            same = len(a) == len(b) and a.notna().equals(b.notna()) and np.allclose(
                a.fillna(-999999).to_numpy(), b.fillna(-999999).to_numpy()
            )
            if same:
                claims_exclude.append("Amount")
                report += ["", "Claims feature check: Amount duplicates ClaimAmount exactly; Amount excluded from this diagnostic model to reduce redundant predictors."]
    lines, rows = evaluate_classification(
        "Insurance claim outcome (Approved vs Pending alternative)",
        "claims_approved_pending_alternative.csv", "ClaimStatusBinary", False, claims_exclude
    )
    report.extend(lines)
    all_rows.extend(rows)

    for task, filename, target in [
        ("Billing amount prediction", "billing_amount_model.csv", "Amount"),
        ("Length of stay prediction", "length_of_stay_model.csv", "LengthOfStayDays"),
    ]:
        lines, rows = evaluate_regression(task, filename, target)
        report.extend(lines)
        all_rows.extend(rows)

    metrics_path = REPORTS / "Week6_Diagnostic_Model_Comparison.csv"
    pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

    report += [
        "",
        "## Interpretation rules",
        "- Compare each fitted model with the dummy baseline on the same holdout split.",
        "- A candidate is not an improvement unless it beats the relevant dummy baseline on appropriate metrics.",
        "- Do not select a model using accuracy alone; inspect macro F1, balanced accuracy, ROC-AUC and confusion matrices for classification.",
        "- For regression, compare MAE/RMSE/R² with the mean baseline. Negative R² is worse than predicting the test-set mean.",
        "- These are diagnostic holdout comparisons, not final clinical validation or deployment approval.",
        "- The script does not create admission labels or rejected claims labels.",
        "- Candidate models are saved separately in Models/Improved_Candidates; existing model files are not overwritten.",
    ]
    report_path = REPORTS / "Week6_Target_and_Model_Diagnostics.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("Completed diagnostic run.")
    print(f"Comparison CSV: {metrics_path.relative_to(ROOT)}")
    print(f"Diagnostic report: {report_path.relative_to(ROOT)}")
    print(f"Separate diagnostic candidates: {CANDIDATE_DIR.relative_to(ROOT)}")
    print("Original datasets and existing model files were not modified.")

if __name__ == "__main__":
    main()
