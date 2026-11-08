
"""
HealthConnect Rwanda - Week 8 Integrated ML Pipeline

Run from the project root:
    python run_week8_pipeline.py --task patient_satisfaction --input path/to/input.csv

This script reuses existing Week 6 trained model pipelines.
It does not retrain models or modify original datasets.

Educational predictions only. Not clinically validated.
"""

from pathlib import Path
from datetime import datetime
import argparse
import json
import logging
import re
import sys

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODELS = ROOT / "Models"
REPORTS = ROOT / "Reports"
OUTPUTS = REPORTS / "Week8_Predictions"
LOGS = ROOT / "Logs"

ID_COLS = {
    "AppointmentID", "PatientID", "DoctorID", "FacilityID",
    "DepartmentID", "InsuranceID", "BillingID", "ClaimID",
    "AdmissionID", "DiagnosisID", "FeedbackID"
}

DATE_COLS = {"Date"}


# Existing model artifacts and their prediction types.
TASKS = {
    "patient_satisfaction": {
        "artifact": "patient_satisfaction_classification__rating_derived.joblib",
        "kind": "classification",
        "description": "Rating-derived patient satisfaction classification"
    },
    "high_risk_severity_proxy": {
        "artifact": "high_risk_patient_classification__severity_proxy.joblib",
        "kind": "classification",
        "description": "Severity-based high-risk proxy classification"
    },
    "insurance_claims": {
        "artifact": "insurance_claim_outcome__approved_vs_pending_alternative.joblib",
        "kind": "classification",
        "description": "Approved versus Pending claims alternative"
    },
    "billing_amount": {
        "artifact": "billing_amount_prediction.joblib",
        "kind": "regression",
        "description": "Billing amount prediction"
    },
    "length_of_stay": {
        "artifact": "length_of_stay_prediction.joblib",
        "kind": "regression",
        "description": "Length of stay prediction"
    },
    "patient_clustering": {
        "artifact": "patient_clustering_kmeans.joblib",
        "kind": "clustering",
        "description": "Exploratory patient segmentation"
    },
    "facility_clustering": {
        "artifact": "facility_clustering_kmeans.joblib",
        "kind": "facility_clustering",
        "description": "Exploratory facility segmentation"
    }
}


# ---------------------------------------------------------
# 2. LOGGING CONFIGURATION
# ---------------------------------------------------------

def configure_logging():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicate logging.
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # Always enable console logging.
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

    logger.addHandler(console_handler)

    # Enable file logging when the log directory is accessible.
    try:

        LOGS.mkdir(
            parents=True,
            exist_ok=True
        )

        file_handler = logging.FileHandler(
            LOGS / "week8_pipeline.log",
            encoding="utf-8"
        )

        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )
        )

        logger.addHandler(file_handler)

    except OSError as error:

        logger.warning(
            "File logging unavailable. "
            "Console logging will continue: %s",
            error
        )


# ---------------------------------------------------------
# 3. DATA PREPARATION
# ---------------------------------------------------------

def clean_features(data):

    X = data.copy()

    # Reproduce the date transformations used during training.
    for col in DATE_COLS.intersection(X.columns):

        parsed = pd.to_datetime(X[col], errors="coerce")

        X[col + "_month"] = parsed.dt.month
        X[col + "_dayofweek"] = parsed.dt.day_of_week

        invalid_dates = (
            X[col].notna() & parsed.isna()
        ).sum()

        if invalid_dates:
            logging.warning(
                "%s invalid date values were converted to missing values.",
                invalid_dates
            )

        X = X.drop(columns=[col])

    # Remove identifier columns from model features.
    X = X.drop(
        columns=[c for c in ID_COLS if c in X.columns],
        errors="ignore"
    )

    # Match categorical missing-value handling used during Week 6 training.
    # Supports both object and pandas string columns.
    for col in X.columns:

        if (
            pd.api.types.is_object_dtype(X[col].dtype)
            or pd.api.types.is_string_dtype(X[col].dtype)
        ):

            X[col] = X[col].fillna("Unknown").astype(str)

    return X


# ---------------------------------------------------------
# 4. INPUT VALIDATION
# ---------------------------------------------------------

def load_input(input_path):

    path = Path(input_path).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Input path is not a file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Only CSV input files are currently supported."
        )

    data = pd.read_csv(path, encoding="utf-8-sig")

    if data.empty:
        raise ValueError(
            "Input dataset contains no records."
        )

    if data.columns.duplicated().any():
        raise ValueError(
            "Input dataset contains duplicate column names."
        )

    if data.columns.isna().any():
        raise ValueError(
            "Input dataset contains missing column names."
        )

    if any(
        not isinstance(col, str) or not col.strip()
        for col in data.columns
    ):
        raise ValueError(
            "Input dataset contains empty or invalid column names."
        )

    logging.info(
        "Loaded %s input records from %s",
        len(data),
        path.name
    )

    return data, path


def validate_model_features(X, model_pipeline):

    if not hasattr(model_pipeline, "named_steps"):
        raise TypeError(
            "Saved artifact is not a compatible sklearn Pipeline."
        )

    if "preprocess" not in model_pipeline.named_steps:
        raise ValueError(
            "Saved model does not contain the expected preprocessing step."
        )

    preprocessor = model_pipeline.named_steps["preprocess"]

    expected = getattr(
        preprocessor,
        "feature_names_in_",
        None
    )

    if expected is None:
        raise ValueError(
            "Cannot establish the model's expected input features."
        )

    expected = list(expected)

    missing = [
        col for col in expected
        if col not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required model features: {missing}"
        )

    # Identify numerical columns from the fitted preprocessing pipeline.
    numeric_columns = []

    for name, transformer, columns in preprocessor.transformers_:

        if columns is None or isinstance(columns, str):
            continue

        if name == "num":
            numeric_columns.extend(list(columns))

    X = X[expected].copy()

    # Validate numeric inputs without changing missing values.
    # The fitted preprocessing pipeline applies its training-time imputer.
    for col in numeric_columns:

        original = X[col]
        converted = pd.to_numeric(original, errors="coerce")

        invalid = (
            original.notna() & converted.isna()
        ).sum()

        if invalid:
            raise ValueError(
                f"{col} contains {invalid} non-numeric values."
            )

        infinite = np.isinf(
            converted.dropna().to_numpy(dtype=float)
        ).sum()

        if infinite:
            raise ValueError(
                f"{col} contains infinite numeric values."
            )

        X[col] = converted

    logging.info(
        "Input validation passed. Expected features: %s",
        len(expected)
    )

    return X


# ---------------------------------------------------------
# 5. MODEL LOADING
# ---------------------------------------------------------

def load_model(task):

    if task not in TASKS:
        raise ValueError(
            f"Unsupported task: {task}"
        )

    artifact_path = MODELS / TASKS[task]["artifact"]

    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}"
        )

    model = joblib.load(artifact_path)

    logging.info(
        "Loaded model artifact: %s",
        artifact_path.name
    )

    return model, artifact_path


# ---------------------------------------------------------
# 6. IDENTIFIER PRESERVATION
# ---------------------------------------------------------

def get_output_base(data):

    identifiers = [
        col for col in data.columns
        if col in ID_COLS
    ]

    output = pd.DataFrame({
        "source_row": np.arange(1, len(data) + 1)
    })

    for col in identifiers:
        output[col] = data[col].values

    return output


# ---------------------------------------------------------
# 7. SUPERVISED MODEL PREDICTION
# ---------------------------------------------------------

def run_supervised_model(task, data, model):

    X = clean_features(data)

    X = validate_model_features(X, model)

    predictions = model.predict(X)

    if len(predictions) != len(data):
        raise ValueError(
            "Prediction count does not match input record count."
        )

    output = get_output_base(data)

    output["prediction"] = predictions

    kind = TASKS[task]["kind"]

    if kind == "classification":

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X)

            estimator = model.named_steps["model"]
            classes = estimator.classes_

            if len(probabilities) != len(data):
                raise ValueError(
                    "Probability count does not match input record count."
                )

            if probabilities.shape[1] != len(classes):
                raise ValueError(
                    "Probability columns do not match model classes."
                )

            output["predicted_probability"] = (
                probabilities.max(axis=1)
            )

            for index, class_name in enumerate(classes):

                safe_class = re.sub(
                    r"[^A-Za-z0-9]+",
                    "_",
                    str(class_name)
                ).strip("_")

                output[
                    f"probability_{safe_class}"
                ] = probabilities[:, index]

    logging.info(
        "Generated %s predictions for %s.",
        len(predictions),
        task
    )

    return output


# ---------------------------------------------------------
# 8. PATIENT CLUSTERING
# ---------------------------------------------------------

def run_patient_clustering(data, model):

    X = clean_features(
        data.drop(
            columns=["PatientID"],
            errors="ignore"
        )
    )

    X = validate_model_features(X, model)

    clusters = model.predict(X)

    if len(clusters) != len(data):
        raise ValueError(
            "Cluster assignment count does not match input record count."
        )

    output = get_output_base(data)

    output["prediction"] = clusters

    logging.info(
        "Generated patient cluster assignments."
    )

    return output


# ---------------------------------------------------------
# 9. FACILITY CLUSTERING
# ---------------------------------------------------------

def run_facility_clustering(data, artifact):

    required_keys = {
        "preprocessor",
        "model",
        "feature_columns"
    }

    if not isinstance(artifact, dict):
        raise TypeError(
            "Facility clustering artifact must be a dictionary."
        )

    if not required_keys.issubset(artifact):
        raise ValueError(
            "Facility clustering artifact is incomplete."
        )

    features = artifact["feature_columns"]

    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError(
            "Facility clustering feature columns are invalid."
        )

    missing = [
        col for col in features
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing facility clustering features: {missing}"
        )

    X = data[list(features)].copy()

    for col in features:

        original = X[col]
        converted = pd.to_numeric(original, errors="coerce")

        invalid = (
            original.notna() & converted.isna()
        ).sum()

        if invalid:
            raise ValueError(
                f"{col} contains {invalid} non-numeric values."
            )

        if np.isinf(
            converted.dropna().to_numpy(dtype=float)
        ).any():
            raise ValueError(
                f"{col} contains infinite numeric values."
            )

        X[col] = converted

    transformed = artifact["preprocessor"].transform(X)

    clusters = artifact["model"].predict(transformed)

    if len(clusters) != len(data):
        raise ValueError(
            "Facility cluster assignment count does not match input record count."
        )

    output = get_output_base(data)

    output["prediction"] = clusters

    logging.info(
        "Generated facility cluster assignments."
    )

    return output


# ---------------------------------------------------------
# 10. OUTPUT VALIDATION AND PATH SAFETY
# ---------------------------------------------------------

def resolve_output_path(output_path, task, input_path):

    if output_path:

        destination = Path(output_path).expanduser()

        if not destination.is_absolute():
            destination = ROOT / destination

    else:

        destination = (
            OUTPUTS / f"{task}_predictions.csv"
        )

    destination = destination.resolve()
    input_path = input_path.resolve()

    if destination == input_path:
        raise ValueError(
            "Output path cannot overwrite the input dataset."
        )

    if destination.suffix.lower() != ".csv":
        raise ValueError(
            "Prediction output path must have a .csv extension."
        )

    summary_path = destination.with_suffix(".summary.json")

    if summary_path == input_path:
        raise ValueError(
            "Output summary path cannot overwrite the input dataset."
        )

    if destination == summary_path:
        raise ValueError(
            "Prediction output and summary paths cannot be identical."
        )

    return destination


# ---------------------------------------------------------
# 11. MAIN PIPELINE
# ---------------------------------------------------------

def run_pipeline(task, input_path, output_path=None):

    configure_logging()

    logging.info("Starting Week 8 ML pipeline.")
    logging.info("Selected task: %s", task)

    if task not in TASKS:
        raise ValueError(
            f"Unsupported task: {task}"
        )

    data, resolved_input = load_input(input_path)

    model, artifact_path = load_model(task)

    kind = TASKS[task]["kind"]

    if kind == "facility_clustering":

        predictions = run_facility_clustering(
            data,
            model
        )

    elif kind == "clustering":

        predictions = run_patient_clustering(
            data,
            model
        )

    else:

        predictions = run_supervised_model(
            task,
            data,
            model
        )

    # Enforce output record consistency.
    if len(predictions) != len(data):
        raise ValueError(
            "Prediction count does not match input record count."
        )

    # Resolve and validate the output destination.
    destination = resolve_output_path(
        output_path,
        task,
        resolved_input
    )

    summary_path = destination.with_suffix(".summary.json")

    # Create the output directory only after validation.
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save prediction results.
    predictions.to_csv(
        destination,
        index=False,
        encoding="utf-8"
    )

    # Save a machine-readable execution summary.
    summary = {
        "task": task,
        "description": TASKS[task]["description"],
        "input_file": str(resolved_input.relative_to(ROOT))
            if resolved_input.is_relative_to(ROOT)
            else str(resolved_input),
        "model_artifact": str(artifact_path.relative_to(ROOT)),
        "output_file": str(destination.relative_to(ROOT))
            if destination.is_relative_to(ROOT)
            else str(destination),
        "input_rows": int(len(data)),
        "output_rows": int(len(predictions)),
        "prediction_type": kind,
        "execution_status": "SUCCESS",
        "timestamp": datetime.now().astimezone().isoformat(),
        "clinical_validation": False
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    logging.info("Pipeline completed successfully.")
    logging.info("Prediction output: %s", destination)
    logging.info("Execution summary: %s", summary_path)

    return summary


# ---------------------------------------------------------
# 12. COMMAND-LINE INTERFACE
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="HealthConnect Week 8 Integrated ML Pipeline"
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASKS.keys()),
        help="Select the ML task."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file."
    )

    parser.add_argument(
        "--output",
        required=False,
        help="Optional path for the prediction output CSV."
    )

    args = parser.parse_args()

    try:

        summary = run_pipeline(
            task=args.task,
            input_path=args.input,
            output_path=args.output
        )

        print("\nML pipeline execution summary")
        print("--------------------------------")
        print(f"Task: {summary['task']}")
        print(f"Input rows: {summary['input_rows']}")
        print(f"Output rows: {summary['output_rows']}")
        print(f"Status: {summary['execution_status']}")
        print(f"Output: {summary['output_file']}")

    except Exception as error:

        if not logging.getLogger().handlers:
            configure_logging()

        logging.exception(
            "ML pipeline execution failed: %s",
            error
        )

        print(f"\nPipeline failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()