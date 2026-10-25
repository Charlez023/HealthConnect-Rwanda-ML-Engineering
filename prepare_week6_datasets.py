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
import pandas as pd

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "Data" / "Original_Datasets"
PROCESSED = ROOT / "Data" / "Processed_Datasets"
PROCESSED.mkdir(parents=True, exist_ok=True)

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


def save(df, filename, description):
    path = PROCESSED / filename
    df.to_csv(path, index=False)
    print(f"{filename}: {len(df):,} rows, {len(df.columns)} columns - {description}")


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
    risk["HighRiskProxy"] = (risk["Severity"] == "Severe").astype("int8")
    risk = merge_patient_features(risk, appointments, patients)
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
    claims_model = claims_model.drop(
        columns=[c for c in ["ClaimID", "BillingID", "Status"] if c in claims_model.columns]
    )
    save(claims_model, "claims_approved_pending_alternative.csv",
         "binary alternative only; source has no Rejected labels")

    # 6. Patient clustering data: one row per patient, no direct contact details.
    patient_cluster = patients[safe_patient_columns].copy()
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

    print("\nPreparation complete. Original datasets were not modified.")
    print("Review the target definitions and row counts before model training.")


if __name__ == "__main__":
    main()


