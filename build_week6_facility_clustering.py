"""
Build facility-level operational clusters using only direct FacilityID records.

Run from the project root:
    python build_week6_facility_clustering.py

No facility names, ownership, location, or department-to-facility joins are inferred.
Counts describe records in the supplied files; they do not measure service quality,
actual utilization, or equipment availability.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "Data" / "Original_Datasets"
PROCESSED = ROOT / "Data" / "Processed_Datasets"
MODELS = ROOT / "Models"
REPORTS = ROOT / "Reports"
for folder in (PROCESSED, MODELS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)


def read_csv(name):
    return pd.read_csv(ORIGINAL / name)


def count_by_facility(df, feature_name):
    if "FacilityID" not in df.columns:
        return pd.Series(dtype="int64", name=feature_name)
    return df.groupby("FacilityID").size().rename(feature_name)


def main():
    clinical = pd.read_excel(ORIGINAL / "HC_Clinical_Transactions.xlsx",
                             sheet_name="Appointments")
    doctors = read_csv("HC_Doctor.csv")
    nurses = read_csv("HC_Nurse.csv")
    departments = read_csv("HC_Departments.csv")
    equipment = read_csv("HC_Medical_Equipment.csv")
    rooms = read_csv("HC_Rooms.csv")
    beds = read_csv("HC_Beds.csv")

    # Use only direct FacilityID groupings. Do not join appointments to departments.
    counts = [
        count_by_facility(clinical, "AppointmentRecords"),
        count_by_facility(doctors, "DoctorRecords"),
        count_by_facility(nurses, "NurseRecords"),
        count_by_facility(departments, "DepartmentRecords"),
        count_by_facility(equipment, "EquipmentRecords"),
        count_by_facility(rooms, "RoomRecords"),
    ]

    # Beds have RoomID rather than FacilityID; use the room table's direct RoomID
    # relationship to assign a bed to a facility. No department relationship used.
    if {"RoomID", "FacilityID"}.issubset(rooms.columns) and {"RoomID"}.issubset(beds.columns):
        beds_with_facility = beds.merge(
            rooms[["RoomID", "FacilityID"]].drop_duplicates(),
            on="RoomID", how="left", validate="many_to_one"
        )
        counts.append(count_by_facility(beds_with_facility, "BedRecords"))

    all_ids = set()
    for series in counts:
        all_ids.update(series.index.dropna().astype(str))
    facility = pd.DataFrame({"FacilityID": sorted(all_ids)})
    for series in counts:
        facility = facility.merge(
            series.rename_axis("FacilityID").reset_index(),
            on="FacilityID", how="left", validate="one_to_one"
        )
    feature_cols = [c for c in facility.columns if c != "FacilityID"]
    facility[feature_cols] = facility[feature_cols].fillna(0).astype(float)
    facility["StaffRecords"] = facility.get("DoctorRecords", 0) + facility.get("NurseRecords", 0)
    facility["RecordedResourceFootprint"] = facility[
        ["StaffRecords", "DepartmentRecords", "EquipmentRecords", "RoomRecords", "BedRecords"]
    ].sum(axis=1)

    dataset_path = PROCESSED / "facility_clustering.csv"
    facility.to_csv(dataset_path, index=False)

    model_features = [
        "AppointmentRecords", "DoctorRecords", "NurseRecords",
        "DepartmentRecords", "EquipmentRecords", "RoomRecords", "BedRecords"
    ]
    X = facility[model_features].astype(float)
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_scaled = preprocessor.fit_transform(X)
    # Three groups are used for exploratory operational segmentation, not ranking.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    facility["FacilitySegment"] = labels
    segments_path = PROCESSED / "facility_segments.csv"
    facility.to_csv(segments_path, index=False)
    joblib.dump(
        {"preprocessor": preprocessor, "model": kmeans, "feature_columns": model_features},
        MODELS / "facility_clustering_kmeans.joblib"
    )

    profile_cols = model_features + ["StaffRecords", "RecordedResourceFootprint"]
    profiles = facility.groupby("FacilitySegment")[profile_cols].mean().round(2)
    profiles["FacilityCount"] = facility.groupby("FacilitySegment").size()
    profiles.to_csv(REPORTS / "facility_cluster_profiles.csv")

    median_appointments = float(facility["AppointmentRecords"].median())
    median_resources = float(facility["RecordedResourceFootprint"].median())
    lines = [
        "HealthConnect Rwanda Week 6 - Facility Clustering Review",
        "",
        "Method: KMeans with three exploratory clusters using standardized counts",
        "from direct FacilityID records in appointments, doctors, nurses, departments,",
        "equipment, rooms and beds. Beds are assigned through the direct RoomID-to-room",
        "relationship. Appointment DepartmentID is not joined to department FacilityID.",
        "",
        "Limitations:",
        "- No verified facility names, ownership, type or province were inferred.",
        "- Record counts are not proof of active staffing, equipment availability, bed occupancy,",
        "  quality of care, workload, or facility performance.",
        "- Clusters are exploratory and should not be used alone to reallocate resources.",
        "- Department/facility mismatches remain a documented source-data limitation.",
        "- Staffing, department, equipment, room and bed counts are identical or nearly identical",
        "  across these facilities; the resulting groups are driven mainly by appointment-record",
        "  volume and should be treated as a preliminary segmentation only.",
        "",
        "Cluster profiles (mean counts unless otherwise indicated):",
    ]
    for segment_id, row in profiles.iterrows():
        ids = facility.loc[facility["FacilitySegment"] == segment_id, "FacilityID"].tolist()
        appointments_mean = float(row["AppointmentRecords"])
        resources_mean = float(row["RecordedResourceFootprint"])
        lines.append(
            f"- Segment {segment_id}: {int(row['FacilityCount'])} facilities; "
            f"FacilityIDs={', '.join(ids)}; mean appointment records={appointments_mean:.1f}; "
            f"mean recorded resource footprint={resources_mean:.1f}."
        )
        if appointments_mean >= median_appointments and resources_mean < median_resources:
            recommendation = (
                "Review staffing, room and equipment planning against verified workload and "
                "service hours; appointment counts alone do not establish overcrowding."
            )
        elif appointments_mean < median_appointments and resources_mean >= median_resources:
            recommendation = (
                "Check whether recorded resources are active and compare with verified utilization "
                "before considering any resource reallocation."
            )
        elif appointments_mean < median_appointments and resources_mean < median_resources:
            recommendation = (
                "Validate the completeness of facility records and compare service scope and patient "
                "needs before making resource-planning decisions."
            )
        else:
            recommendation = (
                "Review facility-specific workload and resource utilization; the cluster describes "
                "recorded counts, not service quality or capacity adequacy."
            )
        lines.append(f"  Planning consideration: {recommendation}")

    lines += [
        "",
        "Overall interpretation: the clusters group facilities by the supplied operational record counts.",
        "Validate records with facility administrators and add reliable utilization, service-scope and",
        "outcome measures before drawing stronger conclusions.",
    ]
    report_path = REPORTS / "Week6_Facility_Clustering_Report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "task": "Facility clustering",
        "facility_rows": int(len(facility)),
        "cluster_count": 3,
        "cluster_sizes": {str(k): int(v) for k, v in facility["FacilitySegment"].value_counts().sort_index().items()},
        "feature_columns": model_features,
        "dataset": "Data/Processed_Datasets/facility_clustering.csv",
        "segments": "Data/Processed_Datasets/facility_segments.csv",
        "profiles": "Reports/facility_cluster_profiles.csv",
        "report": "Reports/Week6_Facility_Clustering_Report.txt",
        "limitation": "Exploratory record-count clusters; not facility performance ratings."
    }
    (REPORTS / "week6_facility_clustering_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Facility clustering outputs created.")
    print("Dataset: Data/Processed_Datasets/facility_clustering.csv")
    print("Segments: Data/Processed_Datasets/facility_segments.csv")
    print("Profiles: Reports/facility_cluster_profiles.csv")
    print("Report: Reports/Week6_Facility_Clustering_Report.txt")


if __name__ == "__main__":
    main()
