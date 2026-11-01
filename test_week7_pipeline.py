"""
HealthConnect Rwanda Week 7 - ML Pipeline Reliability Tests

Tests:
- Existing saved supervised model pipelines without retraining.
- Prepared dataset structure and target integrity.
- Week 6 data-cleaning and merge functions using controlled test data.

Run from the project root:
    python test_week7_pipeline.py

This script does not rerun dataset preparation or model training.
"""

from pathlib import Path
import importlib.util
import json
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "Processed_Datasets"
MODELS = ROOT / "Models"
REPORTS = ROOT / "Reports"

REPORTS.mkdir(parents=True, exist_ok=True)

PREPARATION_SCRIPT = ROOT / "prepare_week6_datasets.py"

ID_COLS = {
    "AppointmentID", "PatientID", "DoctorID", "FacilityID",
    "DepartmentID", "InsuranceID", "BillingID", "ClaimID",
    "AdmissionID", "DiagnosisID", "FeedbackID"
}

TASKS = [
    {
        "name": "Patient satisfaction",
        "data": "patient_satisfaction_model.csv",
        "target": "SatisfactionClass",
        "model": "patient_satisfaction_classification__rating_derived.joblib",
        "type": "classification",
    },
    {
        "name": "High-risk severity proxy",
        "data": "high_risk_severity_proxy.csv",
        "target": "HighRiskProxy",
        "model": "high_risk_patient_classification__severity_proxy.joblib",
        "type": "classification",
    },
    {
        "name": "Insurance claim outcome",
        "data": "claims_approved_pending_alternative.csv",
        "target": "ClaimStatusBinary",
        "model": "insurance_claim_outcome__approved_vs_pending_alternative.joblib",
        "type": "classification",
    },
    {
        "name": "Billing amount",
        "data": "billing_amount_model.csv",
        "target": "Amount",
        "model": "billing_amount_prediction.joblib",
        "type": "regression",
    },
    {
        "name": "Length of stay",
        "data": "length_of_stay_model.csv",
        "target": "LengthOfStayDays",
        "model": "length_of_stay_prediction.joblib",
        "type": "regression",
    },
]

results = []


def record(task, test, status, detail=""):
    results.append({
        "task": task,
        "test": test,
        "status": status,
        "detail": str(detail),
    })

    print(f"[{status}] {task} - {test}: {detail}")


def run_check(task, test, check_function):
    """Run an individual test without stopping the remaining tests."""
    try:
        detail = check_function()
        record(task, test, "PASS", detail)

    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        record(task, test, "FAIL", detail)


def clean_features(X):
    """Apply the same feature cleaning used by the Week 6 training script."""
    X = X.copy()

    if "Date" in X.columns:
        parsed = pd.to_datetime(X["Date"], errors="coerce")
        X["Date_month"] = parsed.dt.month
        X["Date_dayofweek"] = parsed.dt.dayofweek
        X = X.drop(columns=["Date"])

    X = X.drop(
        columns=[c for c in ID_COLS if c in X.columns],
        errors="ignore"
    )

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("Unknown").astype(str)

    return X


# ---------------------------------------------------------
# SECTION 1: DATA PREPARATION MODULE
# ---------------------------------------------------------

preparation_module = None


def load_preparation_module():
    global preparation_module

    if not PREPARATION_SCRIPT.exists():
        raise FileNotFoundError(
            f"Preparation script not found: {PREPARATION_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "week6_preparation",
        PREPARATION_SCRIPT
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not load the preparation script.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    preparation_module = module

    return "prepare_week6_datasets.py loaded successfully"


run_check(
    "Data preparation",
    "Preparation module loading",
    load_preparation_module
)


# ---------------------------------------------------------
# SECTION 2: PREPARED DATASET VALIDATION
# ---------------------------------------------------------

def test_prepared_dataset(task):
    path = DATA / task["data"]

    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset missing: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Prepared dataset contains no records.")

    if task["target"] not in df.columns:
        raise ValueError(
            f"Expected target column missing: {task['target']}"
        )

    if df[task["target"]].isna().any():
        raise ValueError(
            f"Target column contains missing values: {task['target']}"
        )

    if pd.api.types.is_numeric_dtype(df[task["target"]]):
        if not np.isfinite(df[task["target"]]).all():
            raise ValueError("Numeric target contains non-finite values.")

    return (
        f"{len(df)} records; target '{task['target']}' present "
        "with no missing or non-finite values"
    )


for task in TASKS:
    run_check(
        task["name"],
        "Prepared dataset integrity",
        lambda current_task=task: test_prepared_dataset(current_task)
    )


# ---------------------------------------------------------
# SECTION 3: DATA CLEANING FUNCTION TESTS
# ---------------------------------------------------------

def test_invalid_values_and_missing_targets():
    if preparation_module is None:
        raise RuntimeError("Preparation module could not be loaded.")

    synthetic = pd.DataFrame({
        "Age": [-1, 30, np.inf, 40, -1],
        "Feature": [1, 2, 3, 4, 1],
        "Target": [1, 2, 3, np.nan, 1],
    })

    # Rows 0 and 4 are exact duplicates.
    # Row 3 has a missing target.
    cleaned = preparation_module.clean_records(
        synthetic,
        "synthetic_cleaning_test.csv",
        target="Target"
    )

    if len(cleaned) != 3:
        raise AssertionError(
            f"Expected 3 rows after cleaning; got {len(cleaned)}."
        )

    if cleaned["Target"].isna().any():
        raise AssertionError("Missing target values were not removed.")

    if cleaned["Age"].iloc[0] == -1:
        raise AssertionError("Impossible negative age was not corrected.")

    if not cleaned["Age"].isna().any():
        raise AssertionError(
            "Invalid numeric feature values were not converted to missing."
        )

    return (
        "Invalid ages and infinity handled; missing target removed; "
        "exact duplicate removed"
    )


run_check(
    "Data cleaning",
    "Invalid values, missing targets and duplicates",
    test_invalid_values_and_missing_targets
)


def test_outlier_retention():
    if preparation_module is None:
        raise RuntimeError("Preparation module could not be loaded.")

    synthetic = pd.DataFrame({
        "Amount": [10, 11, 12, 13, 10000],
        "Target": [1, 2, 3, 4, 5],
    })

    cleaned = preparation_module.clean_records(
        synthetic,
        "synthetic_outlier_test.csv",
        target="Target"
    )

    if len(cleaned) != len(synthetic):
        raise AssertionError(
            "Potential outlier records were unexpectedly removed."
        )

    if 10000 not in cleaned["Amount"].values:
        raise AssertionError("The extreme value was not retained.")

    return "All 5 records retained, including the extreme Amount value"


run_check(
    "Data cleaning",
    "Potential outlier retention",
    test_outlier_retention
)

def test_numeric_text_validation():
    if preparation_module is None:
        raise RuntimeError("Preparation module could not be loaded.")

    synthetic = pd.DataFrame({
        "Age": ["150", "30", "inf", "-inf"],
        "WeightKg": ["600", "70", "80", "90"],
        "Target": ["A", "B", "A", "B"],
    })

    cleaned = preparation_module.clean_records(
        synthetic,
        "synthetic_numeric_text_test.csv",
        target="Target"
    )

    if not pd.isna(cleaned.loc[0, "Age"]):
        raise AssertionError(
            "Age value '150' stored as text was not identified as invalid."
        )

    if not pd.isna(cleaned.loc[0, "WeightKg"]):
        raise AssertionError(
            "WeightKg value '600' stored as text was not identified as invalid."
        )

    if pd.isna(cleaned.loc[1, "Age"]) or cleaned.loc[1, "Age"] != 30:
        raise AssertionError(
            "Valid numeric text in Age was not converted correctly."
        )

    return "Invalid numeric text detected; valid numeric text preserved"


run_check(
    "Data cleaning",
    "Numeric values stored as text",
    test_numeric_text_validation
)


def test_text_form_infinity_handling():
    if preparation_module is None:
        raise RuntimeError("Preparation module could not be loaded.")

    synthetic = pd.DataFrame({
        "Age": ["inf", "-inf", "35"],
        "Target": ["A", "B", "A"],
    })

    cleaned = preparation_module.clean_records(
        synthetic,
        "synthetic_text_infinity_test.csv",
        target="Target"
    )

    if not cleaned["Age"].isna().iloc[:2].all():
        raise AssertionError(
            "Text-form infinity values were not converted to missing."
        )

    if cleaned["Age"].iloc[2] != 35:
        raise AssertionError(
            "Valid numeric text was not preserved as a numeric value."
        )

    return "Text-form positive and negative infinity handled correctly"


run_check(
    "Data cleaning",
    "Text-form infinity handling",
    test_text_form_infinity_handling
)


# ---------------------------------------------------------
# SECTION 4: DATA MERGE FUNCTION TESTS
# ---------------------------------------------------------

def test_valid_patient_feature_merge():
    if preparation_module is None:
        raise RuntimeError("Preparation module could not be loaded.")

    source = pd.DataFrame({
        "AppointmentID": [101, 102],
        "ClaimAmount": [100, 200],
    })

    appointments = pd.DataFrame({
        "AppointmentID": [101, 102],
        "PatientID": [1, 2],
        "VisitType": ["General", "Follow-up"],
    })

    patients = pd.DataFrame({
        "PatientID": [1, 2],
        "Age": [30, 45],
    })

    merged = preparation_module.merge_patient_features(
        source,
        appointments,
        patients
    )

    if len(merged) != 2:
        raise AssertionError(
            f"Expected 2 merged records; got {len(merged)}."
        )

    if "Age" not in merged.columns:
        raise AssertionError("Patient features were not included.")

    if merged["Age"].isna().any():
        raise AssertionError("Expected patient features are missing.")

    return "Valid appointment and patient relationships merged successfully"


run_check(
    "Data merging",
    "Valid patient feature merge",
    test_valid_patient_feature_merge
)


def test_duplicate_appointment_key_rejection():
    if preparation_module is None:
        raise RuntimeError("Preparation module could not be loaded.")

    source = pd.DataFrame({
        "AppointmentID": [101],
        "ClaimAmount": [100],
    })

    # Duplicate AppointmentID violates the expected many-to-one relationship.
    appointments = pd.DataFrame({
        "AppointmentID": [101, 101],
        "PatientID": [1, 2],
    })

    patients = pd.DataFrame({
        "PatientID": [1, 2],
        "Age": [30, 45],
    })

    try:
        preparation_module.merge_patient_features(
            source,
            appointments,
            patients
        )

    except pd.errors.MergeError:
        return "Duplicate appointment keys correctly rejected"

    raise AssertionError(
        "Merge accepted duplicate appointment keys unexpectedly."
    )


run_check(
    "Data merging",
    "Duplicate appointment key rejection",
    test_duplicate_appointment_key_rejection
)


# ---------------------------------------------------------
# SECTION 5: EXISTING SAVED MODEL PIPELINE TESTS
# ---------------------------------------------------------

for task in TASKS:
    name = task["name"]
    current_test = "Pipeline test execution"

    try:
        data_path = DATA / task["data"]
        model_path = MODELS / task["model"]

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset missing: {data_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model missing: {model_path}")

        current_test = "Dataset loading"
        df = pd.read_csv(data_path)

        if task["target"] not in df.columns:
            raise ValueError(
                f"Target column missing: {task['target']}"
            )

        if df.empty:
            raise ValueError(
                f"Dataset contains no rows: {task['data']}"
            )

        current_test = "Model loading"
        model = joblib.load(model_path)

        record(
            name,
            "Model loading",
            "PASS",
            task["model"]
        )

        current_test = "Feature preparation"
        X = clean_features(df.drop(columns=[task["target"]]))
        X = X.replace([np.inf, -np.inf], np.nan)

        if X.empty or len(X.columns) == 0:
            raise ValueError(
                "No input features available after cleaning."
            )

        sample = (
            X.sample(n=min(20, len(X)), random_state=42)
            .copy()
            .reset_index(drop=True)
        )

        current_test = "Valid input prediction"
        predictions = model.predict(sample)

        if len(predictions) != len(sample):
            raise AssertionError(
                "Prediction count does not match input row count."
            )

        if task["type"] == "classification":
            current_test = "Prediction class validation"

            if hasattr(model, "steps") and model.steps:
                final_estimator = model.steps[-1][1]
            else:
                final_estimator = model

            if not hasattr(final_estimator, "classes_"):
                raise AttributeError(
                    "Final classification estimator does not expose classes_."
                )

            valid_classes = set(final_estimator.classes_)
            predicted_classes = set(predictions)

            if not predicted_classes.issubset(valid_classes):
                raise AssertionError(
                    "Predictions contain classes not seen during training."
                )

        else:
            current_test = "Regression output validation"
            numeric_predictions = np.asarray(
                predictions,
                dtype=float
            )

            if not np.isfinite(numeric_predictions).all():
                raise AssertionError(
                    "Regression predictions contain non-finite values."
                )

        record(
            name,
            "Valid input prediction",
            "PASS",
            f"{len(predictions)} predictions produced"
        )

        # Test handling of an unseen categorical value.
        current_test = "Unseen category handling"

        categorical = sample.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if categorical:
            unusual = sample.copy()
            tested_feature = categorical[0]
            unusual[tested_feature] = "__WEEK7_UNSEEN_CATEGORY__"

            try:
                unusual_predictions = model.predict(unusual)

                if len(unusual_predictions) == len(unusual):
                    record(
                        name,
                        "Unseen category handling",
                        "PASS",
                        f"Tested feature: {tested_feature}"
                    )
                else:
                    record(
                        name,
                        "Unseen category handling",
                        "FAIL",
                        "Unexpected prediction count"
                    )

            except Exception as exc:
                record(
                    name,
                    "Unseen category handling",
                    "FAIL",
                    f"{type(exc).__name__}: {exc}"
                )

        else:
            record(
                name,
                "Unseen category handling",
                "N/A",
                "No categorical input feature in the sample"
            )

        # Test rejection of a missing required feature.
        current_test = "Missing feature rejection"

        feature_to_remove = sample.columns[0]
        reduced = sample.drop(columns=[feature_to_remove])

        try:
            model.predict(reduced)

            record(
                name,
                "Missing feature rejection",
                "FAIL",
                f"Pipeline accepted input without feature: {feature_to_remove}"
            )

        except ValueError as exc:
            record(
                name,
                "Missing feature rejection",
                "PASS",
                f"Rejected missing feature '{feature_to_remove}' "
                f"(ValueError: {exc})"
            )

        except Exception as exc:
            record(
                name,
                "Missing feature rejection",
                "FAIL",
                f"Unexpected exception: {type(exc).__name__}: {exc}"
            )

    except Exception as exc:
        error_detail = (
            f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        )

        record(
            name,
            current_test,
            "FAIL",
            error_detail
        )

        print(f"\n--- Full traceback for {name} ---")
        traceback.print_exc()
        print("--- End traceback ---\n")


# ---------------------------------------------------------
# SECTION 6: FACILITY CLUSTERING RELIABILITY TESTS
# ---------------------------------------------------------

FACILITY_DATA_PATH = DATA / "facility_clustering.csv"
FACILITY_SEGMENTS_PATH = DATA / "facility_segments.csv"
FACILITY_MODEL_PATH = MODELS / "facility_clustering_kmeans.joblib"

FACILITY_FEATURES = [
    "AppointmentRecords",
    "DoctorRecords",
    "NurseRecords",
    "DepartmentRecords",
    "EquipmentRecords",
    "RoomRecords",
    "BedRecords",
]

facility_data = None
facility_segments = None
facility_artifact = None


def load_facility_clustering_data():
    global facility_data

    if not FACILITY_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Facility clustering dataset missing: {FACILITY_DATA_PATH}"
        )

    df = pd.read_csv(FACILITY_DATA_PATH)

    if df.empty:
        raise ValueError("Facility clustering dataset contains no rows.")

    required = ["FacilityID"] + FACILITY_FEATURES
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Facility clustering dataset missing columns: {missing}")

    if df["FacilityID"].isna().any():
        raise ValueError("FacilityID contains missing values.")

    if df["FacilityID"].astype(str).duplicated().any():
        raise ValueError("FacilityID contains duplicate facility records.")

    numeric = df[FACILITY_FEATURES].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Facility clustering features contain missing or non-numeric values.")

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Facility clustering features contain non-finite values.")

    facility_data = df.copy()
    return f"{len(df)} facilities; required ID and 7 numeric model features are valid"


run_check(
    "Facility clustering",
    "Facility dataset integrity",
    load_facility_clustering_data
)


def load_facility_segments():
    global facility_segments

    if facility_data is None:
        raise RuntimeError("Facility clustering dataset did not load successfully.")

    if not FACILITY_SEGMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Facility segments dataset missing: {FACILITY_SEGMENTS_PATH}"
        )

    segments = pd.read_csv(FACILITY_SEGMENTS_PATH)
    if segments.empty:
        raise ValueError("Facility segments dataset contains no rows.")

    required = ["FacilityID", "FacilitySegment"]
    missing = [column for column in required if column not in segments.columns]
    if missing:
        raise ValueError(f"Facility segments dataset missing columns: {missing}")

    if segments["FacilityID"].isna().any() or segments["FacilityID"].astype(str).duplicated().any():
        raise ValueError("Facility segments contain missing or duplicate FacilityID values.")

    source_ids = set(facility_data["FacilityID"].astype(str))
    segment_ids = set(segments["FacilityID"].astype(str))
    if source_ids != segment_ids:
        raise AssertionError(
            "Facility IDs do not match between facility_clustering.csv and facility_segments.csv."
        )

    if segments["FacilitySegment"].isna().any():
        raise ValueError("FacilitySegment contains missing assignments.")

    facility_segments = segments.copy()
    return f"{len(segments)} segment assignments; facility IDs match the clustering dataset"


run_check(
    "Facility clustering",
    "Segment dataset integrity and ID alignment",
    load_facility_segments
)


def load_facility_model_artifact():
    global facility_artifact

    if not FACILITY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Facility clustering model missing: {FACILITY_MODEL_PATH}"
        )

    artifact = joblib.load(FACILITY_MODEL_PATH)
    if not isinstance(artifact, dict):
        raise TypeError("Expected the saved facility model artifact to be a dictionary.")

    required = {"preprocessor", "model", "feature_columns"}
    missing = required.difference(artifact)
    if missing:
        raise KeyError(f"Facility model artifact missing keys: {sorted(missing)}")

    facility_artifact = artifact
    return "Saved preprocessing and KMeans components loaded without retraining"


run_check(
    "Facility clustering",
    "Saved clustering artifact loading",
    load_facility_model_artifact
)


def test_facility_model_feature_contract():
    if facility_artifact is None or facility_data is None:
        raise RuntimeError("Facility data or saved model artifact is unavailable.")

    saved_features = list(facility_artifact["feature_columns"])
    if saved_features != FACILITY_FEATURES:
        raise AssertionError(
            f"Saved feature columns differ from expected features. Found: {saved_features}"
        )

    missing = [column for column in saved_features if column not in facility_data.columns]
    if missing:
        raise AssertionError(f"Saved model features missing from dataset: {missing}")

    return f"Saved model and facility dataset agree on all {len(saved_features)} features"


run_check(
    "Facility clustering",
    "Saved model feature contract",
    test_facility_model_feature_contract
)


def transform_facility_features():
    if facility_artifact is None or facility_data is None:
        raise RuntimeError("Facility data or saved model artifact is unavailable.")

    features = facility_artifact["feature_columns"]
    X = facility_data[features].apply(pd.to_numeric, errors="raise")
    transformed = facility_artifact["preprocessor"].transform(X)

    if transformed.shape[0] != len(facility_data):
        raise AssertionError("Preprocessor changed the number of facility records.")

    if not np.isfinite(np.asarray(transformed, dtype=float)).all():
        raise AssertionError("Preprocessed facility features contain non-finite values.")

    return transformed


def test_facility_preprocessing():
    transformed = transform_facility_features()
    return (
        f"Saved preprocessor transformed {transformed.shape[0]} facilities "
        f"into {transformed.shape[1]} finite features"
    )


run_check(
    "Facility clustering",
    "Saved preprocessing transformation",
    test_facility_preprocessing
)


def test_facility_cluster_predictions():
    if facility_artifact is None or facility_data is None:
        raise RuntimeError("Facility data or saved model artifact is unavailable.")

    transformed = transform_facility_features()
    model = facility_artifact["model"]
    predictions = np.asarray(model.predict(transformed))

    if len(predictions) != len(facility_data):
        raise AssertionError("Cluster prediction count does not match facility count.")

    if not hasattr(model, "cluster_centers_"):
        raise AttributeError("Saved clustering model has no cluster_centers_.")

    valid_labels = set(range(len(model.cluster_centers_)))
    if not set(predictions.tolist()).issubset(valid_labels):
        raise AssertionError("Model produced cluster labels outside its fitted clusters.")

    if facility_segments is None:
        raise RuntimeError("Facility segment assignments are unavailable.")

    predicted = pd.DataFrame({
        "FacilityID": facility_data["FacilityID"].astype(str).to_numpy(),
        "PredictedSegment": predictions,
    })
    stored = facility_segments[["FacilityID", "FacilitySegment"]].copy()
    stored["FacilityID"] = stored["FacilityID"].astype(str)
    comparison = predicted.merge(stored, on="FacilityID", how="inner", validate="one_to_one")

    if len(comparison) != len(facility_data):
        raise AssertionError("Could not compare predictions for every facility.")

    if not np.array_equal(
        comparison["PredictedSegment"].to_numpy(),
        comparison["FacilitySegment"].to_numpy()
    ):
        raise AssertionError("Saved facility segment labels differ from model predictions.")

    return (
        f"{len(predictions)} valid cluster assignments; predictions match "
        "the saved FacilitySegment labels"
    )


run_check(
    "Facility clustering",
    "Cluster prediction and saved-label consistency",
    test_facility_cluster_predictions
)


# ---------------------------------------------------------
# SECTION 7: SAVE TEST RESULTS
# ---------------------------------------------------------

report_path = REPORTS / "Week7_ML_Pipeline_Test_Results.csv"

pd.DataFrame(results).to_csv(
    report_path,
    index=False
)

summary = {
    "total_checks": len(results),
    "passed": sum(r["status"] == "PASS" for r in results),
    "failed": sum(r["status"] == "FAIL" for r in results),
    "not_applicable": sum(r["status"] == "N/A" for r in results),
    "results_file": str(report_path.relative_to(ROOT)),
}

summary_path = REPORTS / "Week7_ML_Pipeline_Test_Summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nWeek 7 pipeline test summary")
print(json.dumps(summary, indent=2))