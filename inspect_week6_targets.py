import pandas as pd
from pathlib import Path

root = Path("Data/Original_Datasets")
wb = root / "HC_Clinical_Transactions.xlsx"

appointments = pd.read_excel(wb, sheet_name="Appointments")
admissions = pd.read_excel(wb, sheet_name="Admissions")
feedback = pd.read_csv(root / "HC_Patient_Feedback.csv")
claims = pd.read_csv(root / "Insurance_Claims.csv")
billing = pd.read_csv(root / "HC_Billing.csv")
patients = pd.read_csv(root / "HC_Patients.csv")
diagnoses = pd.read_csv(root / "HC_Diagnoses.csv")

ids = set(appointments["AppointmentID"])

def report(name, df, columns):
    print(f"\n--- {name} ---")
    for col in columns:
        print(f"\n{col} counts:")
        print(df[col].value_counts(dropna=False).head(20).to_string())
    print("\nMissing values:")
    print(df[columns].isna().sum().to_string())

print("Appointments:", len(appointments))
print("Unique appointment IDs:", appointments["AppointmentID"].nunique())
print("Appointment dates:", appointments["Date"].min(), "to", appointments["Date"].max())
print("Unique patients:", patients["PatientID"].nunique())
print("Appointment patient IDs matching patient records:",
      appointments["PatientID"].isin(set(patients["PatientID"])).sum())

report("ADMISSIONS", admissions, ["LengthOfStayDays"])
print("Admission rows matching appointments:",
      admissions["AppointmentID"].isin(ids).sum())
print("Unique admission appointment IDs:", admissions["AppointmentID"].nunique())

report("FEEDBACK", feedback, ["Rating"])
print("Feedback rows matching appointments:",
      feedback["AppointmentID"].isin(ids).sum())

report("CLAIMS", claims, ["Status", "Provider"])
print("Claim BillingIDs matching billing:",
      claims["BillingID"].isin(set(billing["BillingID"])).sum())

report("BILLING", billing, ["Amount", "Status"])
print("Billing rows matching appointments:",
      billing["AppointmentID"].isin(ids).sum())

report("DIAGNOSES", diagnoses, ["Severity"])
print("Diagnosis rows matching appointments:",
      diagnoses["AppointmentID"].isin(ids).sum())

report("PATIENTS", patients, [
    "Age", "Gender", "ChronicCondition", "BMI", "Status"
])
