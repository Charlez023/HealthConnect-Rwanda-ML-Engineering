"""Targeted Week 6 data-readiness audit for HealthConnect Rwanda.
Run from the project root:
    python profile_week6_data.py

This is a focused audit of assignment-critical relationships and target availability,
not a generic testing suite. It does not modify the original datasets.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "Original_Datasets"
OUT = ROOT / "Reports" / "Data_Readiness_Audit.txt"
CLINICAL_FILE = DATA / "HC_Clinical_Transactions.xlsx"
FACILITY_FILE = DATA / "HC_Facility_Master_Data.xlsx"

def read_csv(name):
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")
    return pd.read_csv(path)

def read_sheet(path, sheet):
    if not path.exists():
        raise FileNotFoundError(f"Required workbook not found: {path}")
    return pd.read_excel(path, sheet_name=sheet)

def money(value):
    return f"{value:,.2f}"

def main():
    lines = []
    def add(text=""):
        lines.append(str(text))

    # Source inventory
    files = sorted(p for p in DATA.iterdir() if p.is_file())
    add("HEALTHCONNECT RWANDA - WEEK 6 DATA READINESS AUDIT")
    add("=" * 58)
    add(f"Original data files found: {len(files)}")
    for p in files:
        add(f" - {p.name}")

    # Load only tables needed for the known assignment-critical gaps.
    appointments = read_sheet(CLINICAL_FILE, "Appointments")
    clinical_sheets = {
        name: read_sheet(CLINICAL_FILE, name)
        for name in ["Diagnoses", "Laboratory_Tests", "Prescriptions", "Admissions", "Referrals"]
    }
    billing = read_csv("HC_Billing.csv")
    payments = read_csv("HC_Payments.csv")
    diagnoses_csv = read_csv("HC_Diagnoses.csv")
    feedback = read_csv("HC_Patient_Feedback.csv")
    claims = read_csv("Insurance_Claims.csv")
    revenue = read_csv("HC_Revenue_Summary.csv")
    departments_csv = read_csv("HC_Departments.csv")
    doctors = read_csv("HC_Doctor.csv")
    nurses = read_csv("HC_Nurse.csv")
    beds = read_csv("HC_Beds.csv")

    add("\n1. TABLE SIZES")
    add("-" * 58)
    for name, df in [
        ("Appointments worksheet", appointments),
        *[(f"{k} worksheet", v) for k, v in clinical_sheets.items()],
        ("HC_Diagnoses.csv", diagnoses_csv),
        ("HC_Billing.csv", billing),
        ("HC_Payments.csv", payments),
        ("HC_Patient_Feedback.csv", feedback),
        ("Insurance_Claims.csv", claims),
        ("HC_Revenue_Summary.csv", revenue),
        ("HC_Departments.csv", departments_csv),
        ("HC_Doctor.csv", doctors),
        ("HC_Nurse.csv", nurses),
        ("HC_Beds.csv", beds),
    ]:
        add(f"{name}: {len(df):,} rows")

    appt_ids = set(appointments["AppointmentID"].dropna().astype(str))
    add("\n2. APPOINTMENT LINKAGE")
    add("-" * 58)
    add(f"Appointments distinct IDs: {appointments['AppointmentID'].nunique():,}")
    add(f"Appointments ID range: {appointments['AppointmentID'].min()} to {appointments['AppointmentID'].max()}")
    for name, df in [
        ("Clinical Transactions / Diagnoses", clinical_sheets["Diagnoses"]),
        ("Clinical Transactions / Laboratory_Tests", clinical_sheets["Laboratory_Tests"]),
        ("Clinical Transactions / Prescriptions", clinical_sheets["Prescriptions"]),
        ("Clinical Transactions / Admissions", clinical_sheets["Admissions"]),
        ("Clinical Transactions / Referrals", clinical_sheets["Referrals"]),
        ("HC_Diagnoses.csv", diagnoses_csv),
        ("HC_Billing.csv", billing),
        ("HC_Patient_Feedback.csv", feedback),
    ]:
        ids = df["AppointmentID"].dropna().astype(str)
        unmatched = int((~ids.isin(appt_ids)).sum())
        add(f"{name}: {len(df):,} rows; {ids.nunique():,} distinct appointment IDs; "
            f"{unmatched:,} rows ({unmatched / max(len(df), 1):.1%}) have no matching appointment")

    add("\n3. INSURANCE CLAIM TARGET")
    add("-" * 58)
    add(f"Claims rows: {len(claims):,}")
    add("Status counts:")
    for status, count in claims["Status"].fillna("<MISSING>").value_counts(dropna=False).items():
        add(f" - {status}: {count:,}")
    add(f"Missing claim statuses: {int(claims['Status'].isna().sum()):,}")

    add("\n4. BILLING / PAYMENT / REVENUE RECONCILIATION")
    add("-" * 58)
    add(f"Total billed (HC_Billing.Amount): RWF {money(pd.to_numeric(billing['Amount'], errors='coerce').sum())}")
    add(f"Total recorded payments (HC_Payments.AmountPaid): RWF {money(pd.to_numeric(payments['AmountPaid'], errors='coerce').sum())}")
    add(f"Revenue Summary total (Revenue column): RWF {money(pd.to_numeric(revenue['Revenue'], errors='coerce').sum())}")
    payment_by_bill = payments.groupby("BillingID", dropna=False)["AmountPaid"].sum()
    bill_amounts = billing.set_index("BillingID")["Amount"]
    recon = bill_amounts.to_frame("Billed").join(payment_by_bill.rename("Paid"), how="left")
    recon["Paid"] = recon["Paid"].fillna(0)
    recon["Difference_Billed_Minus_Paid"] = recon["Billed"] - recon["Paid"]
    underpaid = int((recon["Difference_Billed_Minus_Paid"] > 0).sum())
    overpaid = int((recon["Difference_Billed_Minus_Paid"] < 0).sum())
    equal = int((recon["Difference_Billed_Minus_Paid"] == 0).sum())
    add(f"Billing IDs: {billing['BillingID'].nunique():,}; payment BillingIDs: {payments['BillingID'].nunique():,}")
    add(f"Bills where recorded payments < billed amount: {underpaid:,}")
    add(f"Bills where recorded payments > billed amount: {overpaid:,}")
    add(f"Bills where recorded payments = billed amount: {equal:,}")
    add("Note: these are arithmetic comparisons, not a verified accounting definition of revenue.")

    add("\n5. FACILITY / DEPARTMENT LINKAGE")
    add("-" * 58)
    dept_facility = departments_csv[["DepartmentID", "FacilityID"]].drop_duplicates()
    for name, staff in [("Doctors", doctors), ("Nurses", nurses)]:
        if {"FacilityID", "DepartmentID"}.issubset(staff.columns):
            merged = staff[["FacilityID", "DepartmentID"]].merge(
                dept_facility, on="DepartmentID", how="left", suffixes=("_record", "_department")
            )
            known = merged["FacilityID_department"].notna() & merged["FacilityID_record"].notna()
            matches = int((merged.loc[known, "FacilityID_record"].astype(str) ==
                           merged.loc[known, "FacilityID_department"].astype(str)).sum())
            compared = int(known.sum())
            add(f"{name}: {matches:,}/{compared:,} rows with a department facility available match the record FacilityID "
                f"({matches / compared:.1%} of comparable rows)" if compared else
                f"{name}: no comparable facility/department links.")
    if {"FacilityID", "DepartmentID"}.issubset(appointments.columns):
        merged = appointments[["FacilityID", "DepartmentID"]].merge(
            dept_facility, on="DepartmentID", how="left", suffixes=("_record", "_department")
        )
        known = merged["FacilityID_department"].notna() & merged["FacilityID_record"].notna()
        matches = int((merged.loc[known, "FacilityID_record"].astype(str) ==
                       merged.loc[known, "FacilityID_department"].astype(str)).sum())
        compared = int(known.sum())
        add(f"Appointments: {matches:,}/{compared:,} rows with a department facility available match the record FacilityID "
            f"({matches / compared:.1%} of comparable rows)" if compared else
            "Appointments: no comparable facility/department links.")
    add("Facility master attributes: inspect the supplied workbook sheets; this audit does not invent facility names or locations.")

    add("\n6. BED STATUS AND EQUIPMENT LIMITATION")
    add("-" * 58)
    add(f"Bed rows: {len(beds):,}")
    for status, count in beds["Status"].fillna("<MISSING>").value_counts(dropna=False).items():
        add(f" - Bed status {status}: {count:,}")
    add("Equipment availability: the supplied equipment schema has no availability/status field unless the source schema changes.")

    add("\n7. DUPLICATE / MISSING IDENTIFIER CHECKS")
    add("-" * 58)
    checks = [
        ("Appointments", appointments, "AppointmentID"),
        ("HC_Billing", billing, "BillingID"),
        ("HC_Payments", payments, "PaymentID"),
        ("Insurance_Claims", claims, "ClaimID"),
        ("HC_Patient_Feedback", feedback, "FeedbackID"),
        ("HC_Patients", read_csv("HC_Patients.csv"), "PatientID"),
    ]
    for name, df, col in checks:
        add(f"{name}.{col}: missing={int(df[col].isna().sum()):,}; duplicate values={int(df[col].duplicated().sum()):,}")

    add("\nINTERPRETATION")
    add("-" * 58)
    add("This audit identifies data constraints; it does not clean or alter source data.")
    add("Use task-specific modelling populations and only valid joins. Keep unmatched records for tasks where their available fields remain sufficient.")
    add("Do not fabricate rejected claims, missing appointments, facility attributes, bed occupancy, or equipment availability.")
    add("Confirm target definitions and prediction timing before feature engineering, especially for high-risk and admission prediction.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audit complete. Report saved to: {OUT}")
    print("\n".join(lines[:14]))
    print(f"\nFull report: {OUT}")

if __name__ == "__main__":
    main()
