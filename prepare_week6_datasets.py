"""
HealthConnect Rwanda Week 6 - targeted dataset preparation.
Run from the project root:
    python prepare_week6_datasets.py

Reads original source files and writes task-specific datasets to
Data/Processed_Datasets. Original datasets are not modified.

Important:
- Admission prediction is not labelled here because unmatched admissions do not
  prove non-admission.
- Claims retain only observed Approved/Pending outcomes; no labels are fabricated.
- High-risk classification is explicitly a severity-based proxy.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "Data" / "Original_Datasets"
PROCESSED = ROOT / "Data" / "Processed_Datasets"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS = ROOT / "Reports"
REPORTS.mkdir(parents=True, exist_ok=True)
PREPROCESSING_NOTES = []

WORKBOOK = ORIGINAL / "HC_Clinical_Transactions.xlsx"


def read_csv(filename):
    path = ORIGINAL / filename
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")
    return pd.read_csv(path)


def read_sheet(sheet):
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"Required workbook not found: {WORKBOOK}")
    return pd.read_excel(WORKBOOK, sheet_name=sheet)


def merge_patient_features(df, appointments, patients):
    """Join only through verified PatientID relationships."""
    appointment_cols = [
        c for c in ["AppointmentID", "PatientID", "DoctorID", "FacilityID",
                    "DepartmentID", "Date", "VisitType"]
        if c in appointments.columns
    ]
    patient_cols = [
        c for c in ["PatientID", "Age", "Gender", "MaritalStatus",
                    "Occupation", "Education", "InsuranceID", "BloodGroup",
                    "HeightCm", "WeightKg", "ChronicCondition", "BMI"]
        if c in patients.columns
    ]
    out = df.merge(appointments[appointment_cols], on="AppointmentID",
                   how="inner", validate="many_to_one")
    out = out.merge(patients[patient_cols], on="PatientID", how="left",
                    validate="many_to_one")
    return out


def clean_records(df, dataset_name, target=None):
    """Apply conservative, documented cleaning without deleting valid outliers."""
    out = df.copy()
    start_rows = len(out)

    # Convert +/- infinity to missing values so the training pipeline can impute
    # feature values. Do not silently replace missing target labels.
    out = out.replace([np.inf, -np.inf], np.nan)

    # Correct only clearly impossible values in known numeric fields. These are
    # feature values unless the field is the task target; invalid targets are
    # excluded below because they cannot be trained against reliably.
    invalid_rules = {
        "Age": lambda s: (s < 0) | (s > 120),
        "HeightCm": lambda s: (s <= 0) | (s > 250),
        "WeightKg": lambda s: (s <= 0) | (s > 500),
        "BMI": lambda s: (s <= 0) | (s > 100),
        "Amount": lambda s: s < 0,
        "ClaimAmount": lambda s: s < 0,
        "LengthOfStayDays": lambda s: s < 0,
    }
    invalid_counts = {}
    for col, invalid_mask in invalid_rules.items():
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            mask = invalid_mask(out[col]) & out[col].notna()
            count = int(mask.sum())
            if count:
                out.loc[mask, col] = np.nan
                invalid_counts[col] = count

    # Keep only valid known target labels. This avoids inventing a class or
    # training on a missing/invalid outcome.
    if "Rating" in out.columns and dataset_name == "patient_satisfaction_model.csv":
        valid = out["Rating"].isin([1, 2, 3, 4, 5])
        out = out.loc[valid].copy()
    if "Severity" in out.columns and dataset_name == "high_risk_severity_proxy.csv":
        valid = out["Severity"].astype(str).str.strip().str.casefold().isin(
            {"mild", "moderate", "severe"}
        )
        out = out.loc[valid].copy()
    if target and target in out.columns:
        out = out.loc[out[target].notna()].copy()
        if target in out.columns and pd.api.types.is_numeric_dtype(out[target]):
            out = out.loc[np.isfinite(out[target])].copy()

    # Remove exact duplicate records while IDs are still present in the task
    # dataset, so separate clinical events with different IDs are retained.
    before_dedup = len(out)
    out = out.drop_duplicates().copy()
    duplicates_removed = before_dedup - len(out)

    # Detect potential IQR outliers for review. These are reported, not deleted,
    # because extreme clinical/billing values may be genuine observations.
    outlier_counts = {}
    numeric_cols = out.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        values = out[col].dropna()
        if len(values) < 4 or values.nunique() < 2:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
            if count:
                outlier_counts[col] = count

    PREPROCESSING_NOTES.append(
        f"{dataset_name}: input_rows={start_rows}; output_rows={len(out)}; "
        f"exact_duplicates_removed={duplicates_removed}; "
        f"invalid_numeric_values_set_missing={invalid_counts or 'none'}; "
        f"potential_IQR_outliers_retained_for_review={outlier_counts or 'none'}"
    )
    return out


def save(df, filename, description):
    path = PROCESSED / filename
    df.to_csv(path, index=False)
    print(f"{filename}: {len(df):,} rows, {len(df.columns)} columns - {description}")


def save_preprocessing_summary():
    lines = [
        "HEALTHCONNECT RWANDA - WEEK 6 PREPROCESSING SUMMARY",
        "====================================================",
        "Original source files are read-only; only task-specific processed datasets are written.",
        "Missing feature values are imputed in train_week6_models.py (median for numeric, most-frequent for categorical; object missing values may be marked Unknown by clean_features).",
        "Categorical features are one-hot encoded in the model pipeline. StandardScaler is used for Logistic Regression pipelines; tree-based models are not scaled.",
        "Exact duplicate rows are removed within each task dataset before identifier columns are dropped. Rows with different event IDs are retained.",
        "Positive and negative infinity values are converted to missing values across the dataset; rows with missing targets are excluded, while missing feature values are handled by the model pipeline imputers. Clearly impossible numeric values in documented fields are set to missing, and invalid task targets are excluded rather than fabricated.",
        "Potential numerical outliers are flagged using the 1.5*IQR rule and retained for review; they are not automatically deleted because extreme clinical and billing values may be valid.",
        "Join keys are validated in merge_patient_features using many_to_one validation. Only records linked to known appointments are used for the appointment-linked tasks.",
        "Claims remain Approved/Pending only; no Rejected labels are fabricated. High-risk remains a severity-based proxy, not a clinically validated risk label.",
        "",
        "PER-DATASET COUNTS",
        "-------------------",
        *PREPROCESSING_NOTES,
        "",
        "Note: this summary records the rules applied by this script. It is not evidence of clinical validation or accounting reconciliation."
    ]
    (REPORTS / "Week6_Preprocessing_Summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    appointments = read_sheet("Appointments")
    admissions = read_sheet("Admissions")
    feedback = read_csv("HC_Patient_Feedback.csv")
    claims = read_csv("Insurance_Claims.csv")
    billing = read_csv("HC_Billing.csv")
    patients = read_csv("HC_Patients.csv")
    diagnoses = read_csv("HC_Diagnoses.csv")

    # Common patient-level features; omit direct identifiers and contact details.
    safe_patient_columns = [
        c for c in ["PatientID", "Age", "Gender", "MaritalStatus", "Occupation",
                    "Education", "InsuranceID", "BloodGroup", "HeightCm",
                    "WeightKg", "ChronicCondition", "BMI"]
        if c in patients.columns
    ]

    # 1. Satisfaction: only feedback records linked to a known appointment.
    # Rating mapping is documented and reproducible: 1-2 Low, 3 Medium, 4-5 High.
    satisfaction = feedback.loc[
        feedback["AppointmentID"].isin(appointments["AppointmentID"])
    ].copy()
    satisfaction["SatisfactionClass"] = satisfaction["Rating"].map(
        {1: "Low", 2: "Low", 3: "Medium", 4: "High", 5: "High"}
    )
    satisfaction = merge_patient_features(
        satisfaction, appointments, patients
    )
    satisfaction = clean_records(
        satisfaction, "patient_satisfaction_model.csv", target="SatisfactionClass"
    )
    satisfaction = satisfaction.drop(
        columns=[c for c in ["FeedbackID", "Comment", "Rating"] if c in satisfaction.columns]
    )
    save(satisfaction, "patient_satisfaction_model.csv",
         "feedback linked to known appointments; rating-derived 3-class target")

    # 2. High-risk proxy: Severe diagnosis vs Mild/Moderate.
    # Keep diagnosis records with a valid appointment link, and explicitly name
    # the proxy target rather than claiming clinically validated risk.
    risk = diagnoses.loc[
        diagnoses["AppointmentID"].isin(appointments["AppointmentID"])
    ].copy()
    risk["Severity"] = risk["Severity"].astype(str).str.strip().str.title()
    risk = risk.loc[risk["Severity"].isin(["Mild", "Moderate", "Severe"])].copy()
    risk["HighRiskProxy"] = (risk["Severity"] == "Severe").astype("int8")
    risk = merge_patient_features(risk, appointments, patients)
    risk = clean_records(risk, "high_risk_severity_proxy.csv", target="HighRiskProxy")
    risk = risk.drop(
        columns=[c for c in ["DiagnosisID", "Diagnosis", "Severity"] if c in risk.columns]
    )
    save(risk, "high_risk_severity_proxy.csv",
         "appointment-linked diagnoses; Severe=1, Mild/Moderate=0 proxy target")

    # 3. Billing amount: only bills linked to known appointments. Amount is target.
    # BillingID and billing Status are excluded from predictors to reduce leakage.
    bill = billing.loc[
        billing["AppointmentID"].isin(appointments["AppointmentID"])
    ].copy()
    bill = merge_patient_features(bill, appointments, patients)
    bill = clean_records(bill, "billing_amount_model.csv", target="Amount")
    bill = bill.drop(
        columns=[c for c in ["BillingID", "Status"] if c in bill.columns]
    )
    save(bill, "billing_amount_model.csv",
         "appointment-linked bills; Amount is the regression target")

    # 4. Length of stay: use only admission records linked to known appointments.
    # Admission IDs are identifiers, not predictors.
    los = admissions.loc[
        admissions["AppointmentID"].isin(appointments["AppointmentID"])
    ].copy()
    los = merge_patient_features(los, appointments, patients)
    los = clean_records(los, "length_of_stay_model.csv", target="LengthOfStayDays")
    los = los.drop(columns=[c for c in ["AdmissionID"] if c in los.columns])
    save(los, "length_of_stay_model.csv",
         "only admissions linked to appointments; LengthOfStayDays is target")

    # 5. Claims: preserve only observed classes; this is a binary alternative,
    # not the required three-class task. Billing join is by BillingID only.
    claims_model = claims.loc[
        claims["Status"].isin(["Approved", "Pending"])
    ].copy()
    claims_model = claims_model.merge(
        billing[["BillingID", "Amount", "AppointmentID"]],
        on="BillingID", how="left", validate="many_to_one"
    )
    claims_model["ClaimStatusBinary"] = claims_model["Status"]
    claims_model = clean_records(claims_model, "claims_approved_pending_alternative.csv", target="ClaimStatusBinary")
    claims_model = claims_model.drop(
        columns=[c for c in ["ClaimID", "BillingID", "Status"] if c in claims_model.columns]
    )
    save(claims_model, "claims_approved_pending_alternative.csv",
         "binary alternative only; source has no Rejected labels")

    # 6. Patient clustering data: one row per patient, no direct contact details.
    patient_cluster = patients[safe_patient_columns].copy()
    patient_cluster = clean_records(patient_cluster, "patient_clustering.csv")
    save(patient_cluster, "patient_clustering.csv",
         "patient-level clustering features; no target label")

    # 7. A record of blocked/unavailable target scope, saved for documentation.
    limitations = pd.DataFrame([
        ["Hospital admission classification",
         "Not labelled in this script",
         "Admission sheet appointment IDs overlap only partially with Appointments; unmatched records cannot safely be labelled non-admissions."],
        ["Insurance claim outcome (3 classes)",
         "Not trainable with current labels",
         "Observed claim statuses are Approved and Pending; no Rejected records were found."],
        ["High-risk classification",
         "Proxy target provided",
         "Severe diagnosis is used as a transparent proxy, not a clinically validated risk label."],
        ["Facility clustering",
         "Requires further validation",
         "Facility-to-department linkage rates in the audit were about 5%; avoid unsupported facility-level conclusions."],
    ], columns=["Task", "Status", "Limitation"])
    save(limitations, "week6_target_limitations.csv",
         "records target constraints and assumptions")
    save_preprocessing_summary()

    print("\nPreparation complete. Original datasets were not modified.")
    print("Preprocessing summary: Reports/Week6_Preprocessing_Summary.txt")
    print("Review the target definitions and row counts before model training.")


if __name__ == "__main__":
    main()


