"""
HealthConnect Rwanda Week 6 - explainability and anomaly review.
Run from the project root:
    python week6_explainability_anomalies.py

Creates feature-importance summaries for the baseline models and rule-based
anomaly review files using available source columns. These are exploratory
flags, not proof of fraud, disease, or clinical abnormality.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
ORIG = ROOT / "Data" / "Original_Datasets"
PROC = ROOT / "Data" / "Processed_Datasets"
MODELS = ROOT / "Models"
REPORTS = ROOT / "Reports"
REPORTS.mkdir(exist_ok=True)

ID_COLS = {
    "AppointmentID", "PatientID", "DoctorID", "FacilityID", "DepartmentID",
    "InsuranceID", "BillingID", "ClaimID", "AdmissionID", "DiagnosisID",
    "FeedbackID", "LabID", "PrescriptionID", "ReferralID"
}


def csv(name):
    return pd.read_csv(ORIG / name)


def sheet(name):
    return pd.read_excel(ORIG / "HC_Clinical_Transactions.xlsx", sheet_name=name)


def prepare_x(df):
    X = df.copy()
    for c in ["Date"]:
        if c in X.columns:
            dt = pd.to_datetime(X[c], errors="coerce")
            X[c + "_month"] = dt.dt.month
            X[c + "_dayofweek"] = dt.dt.dayofweek
            X = X.drop(columns=[c])
    X = X.drop(columns=[c for c in ID_COLS if c in X.columns], errors="ignore")
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].fillna("Unknown").astype(str)
    return X


def anomaly_summary(name, df, flagged, reason):
    flagged = flagged.copy()
    flagged["AnomalyReason"] = reason
    flagged.to_csv(REPORTS / name, index=False)
    return {
        "file": name,
        "source_rows": int(len(df)),
        "flagged_rows": int(len(flagged)),
        "reason": reason
    }


def run_anomalies():
    summaries = []

    # Unusual billing amounts: IQR rule. Keep only non-PII billing fields.
    billing = csv("HC_Billing.csv")
    amt = pd.to_numeric(billing["Amount"], errors="coerce")
    q1, q3 = amt.quantile([0.25, 0.75])
    iqr = q3 - q1
    bill_mask = (amt < q1 - 1.5 * iqr) | (amt > q3 + 1.5 * iqr)
    bill_out = billing.loc[bill_mask, ["BillingID", "AppointmentID", "Amount", "Status"]].copy()
    summaries.append(anomaly_summary(
        "anomaly_unusual_billing.csv", billing, bill_out,
        f"Amount outside 1.5×IQR bounds ({q1 - 1.5*iqr:.2f}, {q3 + 1.5*iqr:.2f}); investigate, not proof of error"
    ))

    # Claims: extreme amounts by IQR; missing/unknown status and repeated claim IDs.
    claims = csv("Insurance_Claims.csv")
    ca = pd.to_numeric(claims["ClaimAmount"], errors="coerce")
    cq1, cq3 = ca.quantile([0.25, 0.75])
    ciqr = cq3 - cq1
    claim_mask = (ca < cq1 - 1.5 * ciqr) | (ca > cq3 + 1.5 * ciqr) | claims["Status"].isna()
    claim_out = claims.loc[claim_mask, ["ClaimID", "BillingID", "Provider", "ClaimAmount", "Status"]].copy()
    summaries.append(anomaly_summary(
        "anomaly_claims_review.csv", claims, claim_out,
        f"ClaimAmount outside 1.5×IQR bounds ({cq1 - 1.5*ciqr:.2f}, {cq3 + 1.5*ciqr:.2f}) or missing status; no Rejected class in source"
    ))

    # Lab results are categorical (e.g. Normal/Abnormal), not numeric.
    # Summarize category frequencies without assigning clinical meaning.
    labs = sheet("Laboratory_Tests")
    lab_result = labs["Result"].fillna("Missing").astype(str).str.strip()
    lab_counts = (
        lab_result.value_counts(dropna=False)
        .rename_axis("ResultCategory")
        .reset_index(name="RowCount")
    )
    lab_counts["ReviewNote"] = (
        "Category frequency only; clinical reference ranges and units are unavailable. "
        "This is not a clinical abnormality assessment."
    )
    lab_counts.to_csv(REPORTS / "anomaly_lab_result_categories.csv", index=False)

    # Remove the obsolete numeric-outlier output from earlier script versions.
    old_lab_report = REPORTS / "anomaly_lab_numeric_outliers.csv"
    if old_lab_report.exists():
        old_lab_report.unlink()

    summaries.append({
        "file": "anomaly_lab_result_categories.csv",
        "source_rows": int(len(labs)),
        "flagged_rows": 0,
        "reason": (
            f"Categorical results summarized across {lab_result.nunique(dropna=False)} "
            "values; no clinical abnormality is inferred"
        )
    })
    nonnumeric_count = int(pd.to_numeric(labs["Result"], errors="coerce").isna().sum())

    # Prescribing pattern: frequent medication strings are not clinically interpreted.
    prescriptions = sheet("Prescriptions")
    med_counts = prescriptions["Medication"].value_counts(dropna=False)
    common = med_counts[med_counts > med_counts.quantile(0.95)].index
    frequent = prescriptions[prescriptions["Medication"].isin(common)][
        ["PrescriptionID", "AppointmentID", "Medication", "Dosage"]
    ].copy()
    summaries.append(anomaly_summary(
        "anomaly_frequent_medications.csv", prescriptions, frequent,
        "Medication appears among medication values above the 95th percentile of frequency; frequency alone is not suspicious"
    ))

    # Duplicate patient IDs and exact duplicate records. Do not export names/contact details.
    patients = csv("HC_Patients.csv")
    dup_id = patients[patients["PatientID"].duplicated(keep=False)]
    dup_exact = patients[patients.duplicated(keep=False)]
    duplicate_report = pd.DataFrame([
        {"check": "Duplicate PatientID", "flagged_rows": int(len(dup_id)),
         "interpretation": "Identifier duplication check; audit previously found unique PatientID values."},
        {"check": "Exact duplicate patient rows", "flagged_rows": int(len(dup_exact)),
         "interpretation": "Exact-row duplication check only; does not identify different IDs belonging to the same person."}
    ])
    duplicate_report.to_csv(REPORTS / "anomaly_duplicate_patient_summary.csv", index=False)
    summaries.append({
        "file": "anomaly_duplicate_patient_summary.csv",
        "source_rows": int(len(patients)),
        "flagged_rows": int(len(dup_id) + len(dup_exact)),
        "reason": "Duplicate identifier and exact-row counts; no personal details exported"
    })

    # Facility volume: descriptive only; do not assume all facilities have equal size.
    appts = sheet("Appointments")
    vol = appts.groupby("FacilityID").size().rename("AppointmentCount").reset_index()
    v = vol["AppointmentCount"]
    q1, q3 = v.quantile([0.25, 0.75])
    iqr = q3 - q1
    vol_out = vol[(v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)].copy()
    summaries.append(anomaly_summary(
        "anomaly_facility_appointment_volume.csv", vol, vol_out,
        f"Facility appointment count outside 1.5×IQR bounds ({q1 - 1.5*iqr:.2f}, {q3 + 1.5*iqr:.2f}); investigate context"
    ))

    # Beds: status counts, as equipment has no availability/status column.
    beds = csv("HC_Beds.csv")
    beds["Status"] = beds["Status"].fillna("Missing").astype(str)
    beds.groupby("Status").size().rename("BedCount").reset_index().to_csv(
        REPORTS / "bed_status_summary.csv", index=False
    )

    # Prescriptions duplicated by AppointmentID + Medication + Dosage.
    dup_rx = prescriptions[prescriptions.duplicated(
        subset=["AppointmentID", "Medication", "Dosage"], keep=False
    )][["PrescriptionID", "AppointmentID", "Medication", "Dosage"]]
    summaries.append(anomaly_summary(
        "anomaly_repeated_prescription_patterns.csv", prescriptions, dup_rx,
        "Repeated appointment-medication-dosage combination; may be legitimate and needs review"
    ))

    report = [
        "HealthConnect Rwanda Week 6 - exploratory anomaly review",
        "Flags use statistical/structural rules only. They are not proof of fraud, prescribing error, disease, or clinical abnormality.",
        f"Laboratory results not parseable as numeric: {nonnumeric_count} of {len(labs)}.",
        "Laboratory Result values are summarized as categories only; no units or clinical reference ranges are available, so this is not a clinical abnormality assessment.",
        "The dataset contains no Rejected insurance claims, so suspicious-claim checks cannot validate rejection detection.",
        "Medical equipment has no status/availability field in the inspected schema; bed status is summarized separately.",
        ""
    ]
    for s in summaries:
        report.append(f"- {s['file']}: {s['flagged_rows']} flagged rows / {s['source_rows']} source rows. {s['reason']}")
    (REPORTS / "Week6_Anomaly_Review.txt").write_text("\n".join(report), encoding="utf-8")
    return summaries


def run_explainability():
    # task dataset, target, model filename
    tasks = [
        ("patient_satisfaction_model.csv", "SatisfactionClass",
         "patient_satisfaction_classification__rating_derived.joblib", "accuracy"),
        ("high_risk_severity_proxy.csv", "HighRiskProxy",
         "high_risk_patient_classification__severity_proxy.joblib", "accuracy"),
        ("claims_approved_pending_alternative.csv", "ClaimStatusBinary",
         "insurance_claim_outcome__approved_vs_pending_alternative.joblib", "accuracy"),
        ("billing_amount_model.csv", "Amount",
         "billing_amount_prediction.joblib", "neg_mean_absolute_error"),
        ("length_of_stay_model.csv", "LengthOfStayDays",
         "length_of_stay_prediction.joblib", "neg_mean_absolute_error"),
    ]
    output = []
    for dataset, target, model_file, scoring in tasks:
        path = MODELS / model_file
        if not path.exists():
            output.append({"task": model_file, "status": "model file not found"})
            continue
        df = pd.read_csv(PROC / dataset)
        y = df[target]
        X = prepare_x(df.drop(columns=[target]))
        valid = y.notna()
        X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
        # A reproducible sample bounds runtime for permutation importance.
        sample_n = min(500, len(X))
        sample = X.sample(n=sample_n, random_state=42)
        ys = y.loc[sample.index]
        model = joblib.load(path)
        try:
            result = permutation_importance(
                model, sample, ys, scoring=scoring, n_repeats=3,
                random_state=42, n_jobs=-1
            )
            imp = pd.DataFrame({
                "Feature": sample.columns,
                "ImportanceMean": result.importances_mean,
                "ImportanceStd": result.importances_std,
            }).sort_values("ImportanceMean", ascending=False)
            safe = model_file.replace(".joblib", "")
            imp.to_csv(REPORTS / f"feature_importance_{safe}.csv", index=False)

            all_zero = (
                len(imp) > 0
                and np.isfinite(imp["ImportanceMean"].to_numpy()).all()
                and np.allclose(imp["ImportanceMean"].to_numpy(), 0.0, atol=1e-12)
            )
            if all_zero:
                status = "uninformative_zero_importance"
                note = (
                    "All mean permutation importances are zero. Permuting the sampled "
                    "features did not measurably change the model score; this explanation "
                    "is uninformative."
                )
            else:
                status = "ok"
                note = (
                    "Permutation importance on a sample; association with model predictions, "
                    "not causal explanation. Rankings are exploratory and should be interpreted "
                    "alongside baseline performance."
                )

            output.append({
                "task": model_file,
                "status": status,
                "sample_rows": int(sample_n),
                "top_features": imp.head(10).to_dict(orient="records"),
                "file": f"Reports/feature_importance_{safe}.csv",
                "note": note
            })
        except Exception as exc:
            output.append({"task": model_file, "status": "failed", "error": str(exc)})
    (REPORTS / "week6_feature_importance_summary.json").write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "HealthConnect Rwanda Week 6 - permutation feature importance",
        "Interpretation: importance describes impact on this fitted model's score, not causation.",
        "Baseline models are weak; feature rankings should be treated as exploratory.",
        "A zero-importance result is labeled uninformative, not a successful explanation.",
        ""
    ]
    for item in output:
        lines.append(f"## {item['task']}")
        lines.append(f"- status: {item['status']}")
        if item.get("file"):
            lines.append(f"- file: {item['file']}")
            for f in item.get("top_features", [])[:10]:
                lines.append(f"- {f['Feature']}: mean importance {f['ImportanceMean']:.6f} (std {f['ImportanceStd']:.6f})")
        if item.get("error"):
            lines.append(f"- error: {item['error']}")
        if item.get("note"):
            lines.append(f"- note: {item['note']}")
        lines.append("")
    (REPORTS / "Week6_Explainability_Report.txt").write_text("\n".join(lines), encoding="utf-8")
    return output


def main():
    print("Running anomaly checks...")
    run_anomalies()
    print("Running permutation feature importance...")
    run_explainability()
    print("\nCompleted.")
    print("Anomaly report: Reports/Week6_Anomaly_Review.txt")
    print("Explainability report: Reports/Week6_Explainability_Report.txt")
    print("Review the reports for limitations and failed or uninformative importance calculations.")


if __name__ == "__main__":
    main()
