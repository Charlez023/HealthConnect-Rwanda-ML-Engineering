# HealthConnect Rwanda - Machine Learning Engineering

**Project:** AI-Powered Clinical Risk Prediction and Healthcare Intelligence System

**Programme:** AnalystLab Africa Experience Lab

**Track:** Machine Learning Engineering

---

## 1. Project Overview

HealthConnect Rwanda is a machine learning engineering project focused on developing exploratory healthcare prediction and intelligence solutions using the supplied healthcare datasets.

The project covers three phases:

- **Week 6:** Data preparation, feature engineering, model training, evaluation, clustering and explainability.
- **Week 7:** Pipeline testing, reliability improvements and cross-track validation.
- **Week 8:** Integration of available saved machine learning models into a reusable prediction pipeline with logging, error handling and reproducibility documentation.

The objective is to establish a structured machine learning workflow that supports model execution, prediction generation, evaluation, traceability and engineering handover.

The resulting engineering package provides a foundation for demonstrating how machine learning workflows can support healthcare intelligence while maintaining transparency about data quality, model limitations and responsible use.

## 2. Project Structure

| Directory/File | Description |
|---|---|
| `Data/Original_Datasets/` | Original healthcare datasets, preserved unchanged. |
| `Data/Processed_Datasets/` | Prepared datasets for modelling and prediction. |
| `Notebook/` | Week 6 machine learning notebook. |
| `Models/` | Saved machine learning model artifacts. |
| `Reports/` | Evaluation, explainability, clustering, testing and prediction reports. |
| `Logs/` | Pipeline execution logs. |
| `Presentation/` | Project presentation materials. |
| `App/` | Application files, where included. |
| `References/` | Supplied assignment PDFs. |
| `prepare_week6_datasets.py` | Dataset preparation script. |
| `train_week6_models.py` | Model training and comparison script. |
| `run_week8_pipeline.py` | Integrated prediction pipeline. |
| `test_week7_pipeline.py` | Week 7 pipeline test script. |
| `requirements.txt` | Project dependencies and pinned package versions. |

## 3. Data Preparation and Limitations

The original datasets were preserved, and separate prepared datasets were generated for machine learning tasks.

Data inspection identified several limitations affecting model development and interpretation:

- **Insurance claims:** The original dataset contains only Approved and Pending statuses. No Rejected class is available. Therefore, the project uses a binary Approved-versus-Pending classification alternative.
- **High-risk classification:** Diagnosis severity is used as a proxy target, with Severe represented as the positive class and Mild/Moderate as the negative class. This is not a clinically validated risk outcome.
- **Patient satisfaction:** Available patient rating data is used as the prediction target.
- **Length of stay:** Only 586 admission records could be linked to the available appointment records.
- **Facility relationships:** Weak relationships between facility and department records limit reliable joins and analysis.
- **Other operational data:** Bed records are marked Available, and sufficient equipment availability information is not supplied.

These limitations affect the completeness and predictive reliability of the resulting models. Missing outcomes and unsupported labels were not artificially generated.

Original datasets remain separate from the processed modelling datasets.

Further details are documented in:

`Reports/Data_Readiness_Audit.txt`

## 4. Machine Learning and Evaluation

The Week 6 modelling workflow covers:

- Classification and regression.
- Patient clustering and facility clustering.
- Model comparison against baseline models.
- Permutation feature importance.
- SHAP explainability.
- Anomaly detection and review.

The facility clustering workflow uses direct `FacilityID`-level record counts and the `RoomID`-to-room relationship for bed counts. It produces three exploratory clusters, largely reflecting differences in appointment volume rather than facility quality.

Explainability findings describe model behaviour and feature relationships, not causation. Detected anomalies require contextual review and should not automatically be interpreted as errors or fraudulent activity.

Model evaluation results and supporting analysis are available in the `Reports/` directory.

Some models perform close to or worse than their baseline alternatives. These results are documented transparently rather than presented as evidence of successful predictive performance.

The models are exploratory and should not be interpreted as clinically validated prediction systems.

## 5. Week 7 - Testing and Reliability

Week 7 focused on validating and improving the existing machine learning pipeline.

Testing covered data preparation, numeric conversion, missing values, duplicates, data merging, preprocessing compatibility, saved model loading, prediction generation and facility clustering.

### 5.1 Verified Improvement

The `clean_records()` function in `prepare_week6_datasets.py` was improved to:

- Convert numeric values stored as text using `pd.to_numeric(..., errors="coerce")`.
- Handle positive and negative infinity.
- Record numeric conversion failures.

The prepared datasets were regenerated, and the affected tests passed successfully without changing their previous row counts.

### 5.2 Test Results

| Test result | Count |
|---|---:|
| Total checks | 38 |
| Passed | 38 |
| Failed | 0 |
| Not applicable | 0 |

Run the test suite from the project root:

```powershell
python test_week7_pipeline.py
```

**Evidence files:**

- `Reports/Week7_ML_Pipeline_Test_Results.csv`
- `Reports/Week7_ML_Pipeline_Test_Summary.json`
- `Reports/Week7_ML_Pipeline_Testing_and_Reliability_Report.md`

### 5.3 HC-POD Cross-Track Validation

Week 7 also included independent validation of findings from the Data Analytics track as part of the broader HC-POD integration work.

Key findings included:

- The insurance claims dataset contains 21,028 Pending and 21,008 Approved records, with no Rejected records.
- Only 5,000 of 60,000 billing records linked to available appointments.
- Only 1,756 of 21,064 feedback records linked to available appointments.
- Linked facility rating averages ranged from approximately 2.76 to 3.19.
- ANOVA and Kruskal-Wallis tests did not provide statistically significant evidence of differences in facility ratings.

These findings do not provide sufficient evidence to rank facilities by patient satisfaction.

The reported dashboard correction involving readmission counts could not be independently verified using the supplied source-data archive.

Detailed findings are available in:

`Reports/Week7_Cross_Track_Testing_Evidence.md`

## 6. Week 8 - Integrated Prediction Pipeline

Week 8 delivered a reusable prediction pipeline that integrates the available saved model artifacts and supports seven prediction tasks.

The pipeline uses existing trained models. It does not retrain models or modify original datasets during prediction execution.

### 6.1 Environment Setup

The project uses Python 3.11.9.

Create a virtual environment from the project root:

```powershell
python -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check installed package compatibility:

```powershell
python -m pip check
```

The `requirements.txt` file contains the project's pinned package versions.

### 6.2 Supported Prediction Tasks

The integrated pipeline supports the following seven prediction tasks:

| Task | Prepared input dataset | Records |
|---|---|---:|
| `patient_satisfaction` | `patient_satisfaction_model.csv` | 1,756 |
| `high_risk_severity_proxy` | `high_risk_severity_proxy.csv` | 5,000 |
| `insurance_claims` | `claims_approved_pending_alternative.csv` | 42,036 |
| `billing_amount` | `billing_amount_model.csv` | 5,000 |
| `length_of_stay` | `length_of_stay_model.csv` | 586 |
| `patient_clustering` | `patient_clustering.csv` | 20,000 |
| `facility_clustering` | `facility_clustering.csv` | 20 |

All prepared input datasets are located in:

`Data/Processed_Datasets/`

### 6.3 Running Predictions

Execute the following command from the project root:

```powershell
python run_week8_pipeline.py --task TASK_NAME --input INPUT_FILE
```

Example:

```powershell
python run_week8_pipeline.py --task patient_satisfaction --input Data/Processed_Datasets/patient_satisfaction_model.csv
```

An optional output location can be specified using `--output`.

### 6.4 Prediction Outputs

Prediction CSV files and their corresponding JSON execution summaries are saved in:

`Reports/Week8_Predictions/`

The execution summaries record:

- Selected prediction task.
- Input and output record counts.
- Model artifact used.
- Prediction type.
- Execution status.
- Execution timestamp.
- Clinical validation status.

All seven supported prediction tasks completed successfully during the recorded Week 8 pipeline execution.

Their generated prediction record counts matched the corresponding input record counts.

### 6.5 Integrated Pipeline Workflow

The integrated prediction workflow follows this general sequence:

1. Receive the selected prediction task and input dataset.
2. Validate the input file and required model artifact.
3. Load the corresponding saved model.
4. Validate the required input features.
5. Apply the preprocessing and prediction workflow.
6. Generate predictions or clustering results.
7. Validate prediction output counts.
8. Save prediction results and execution summaries.
9. Record execution information and errors in the pipeline logs.

This workflow provides a reusable engineering structure for executing the available machine learning models.

## 7. Reproducibility, Logging and Error Handling

### 7.1 Reproducibility

The project includes documented environment setup instructions, pinned dependencies in `requirements.txt`, saved model artifacts and a reusable prediction pipeline.

The model training workflow defines a fixed random state of `42` for supported stochastic operations and data splits where the value is applied. This helps improve reproducibility but does not guarantee identical results across different software environments.

The project provides the following resources for reproducing the workflow:

- Python environment setup instructions.
- Dependency installation instructions.
- Saved model artifacts.
- Prepared input datasets.
- A reusable integrated prediction script.
- Defined prediction task names and input paths.
- Execution logs.
- Prediction CSV files and JSON execution summaries.

The Week 8 pipeline was successfully executed across all seven supported prediction tasks. Execution logs, prediction files and JSON summaries provide evidence of successful execution and matching input and output record counts.

These resources provide the documented instructions and project artifacts needed to set up and execute the workflow again.

**Verification scope:** The available execution evidence establishes successful pipeline execution. Independent execution in a separate clean temporary environment has not been established as part of the available verification evidence.

### 7.2 Logging

The Week 8 pipeline uses Python logging to record execution information.

Logging includes:

- Pipeline execution events.
- Prediction task information.
- Successful execution messages.
- Input and output validation information.
- Error details.

The primary execution log is stored in:

`Logs/week8_pipeline.log`

The pipeline configures INFO-level console and file logging.

If file-logging setup encounters an `OSError`, the pipeline continues with console logging.

### 7.3 Error Handling

The integrated pipeline incorporates the following engineering safeguards:

- Input file validation.
- Model artifact validation.
- Required feature validation.
- Prediction and output record-count checks.
- Output path validation.
- Explicit exception handling.
- Execution logging and error reporting.
- Console logging fallback when file-logging setup fails.
- Top-level exception handling for unexpected execution failures.

These safeguards improve execution reliability, traceability and maintainability.

## 8. Responsible AI and Deployment Considerations

The project uses supplied synthetic or educational healthcare data. Its models are intended for learning, analysis, engineering development and demonstration.

Successful pipeline execution and reproducibility do not establish predictive accuracy, clinical reliability or suitability for real-world healthcare decisions.

The following principles guided the project:

1. Preserve original datasets.
2. Use valid joins and task-specific modelling datasets.
3. Avoid fabricated labels and unsupported assumptions.
4. Distinguish verified findings from proxies and documented limitations.
5. Consider patient confidentiality, data protection and potential model bias.
6. Do not use exploratory predictions as the sole basis for clinical or operational decisions.

### 8.1 Proposed Deployment Architecture

The proposed deployment architecture is conceptual and is not an implemented clinical application.

A potential deployment workflow could connect hospital systems to a web dashboard and API, validate incoming data, apply preprocessing and trained models, return predictions with explanations, and route results for human review.

The current engineering package establishes the reusable prediction pipeline and supporting model artifacts for demonstration.

The project does not claim a production clinical deployment.

### 8.2 Data Science Model Coordination

No separate final model artifact was supplied by the Data Science track for the Week 8 integration.

The ML Engineering pipeline therefore integrates the available saved model artifacts for all seven supported prediction tasks.

The integrated workflow, its model inputs, prediction outputs and engineering validation are documented in this repository.

This handover accurately reflects the available model artifacts and the engineering integration completed. It does not claim receipt, approval or direct participation by the Data Science track in the final integration.

## 9. Known Limitations

The following limitations remain relevant to interpreting the project results:

1. **Insurance claims:** No Rejected class exists in the supplied source dataset. The available classification task distinguishes Approved from Pending claims only.
2. **High-risk classification:** Diagnosis severity is a proxy target rather than a clinically validated risk outcome.
3. **Patient satisfaction:** Available ratings provide limited linked records for modelling and analysis.
4. **Length of stay:** The modelling dataset contains only 586 linked admission records.
5. **Facility analysis:** Weak facility relationships limit the reliability of some facility-level comparisons.
6. **Model performance:** Some models perform close to or below baseline alternatives and require further improvement before practical predictive use.
7. **Explainability:** Feature importance and SHAP results describe model behaviour and do not establish causation.
8. **Anomaly detection:** Identified anomalies require contextual review and are not automatically evidence of errors or fraud.
9. **Clinical validation:** No clinical validation or real-world deployment approval has been established.
10. **Reproducibility evidence:** Successful recorded execution is available, but independent clean-environment execution has not been established in the available evidence.

These limitations are documented to ensure that the engineering outputs are interpreted within their actual scope.

## 10. Final Integration and Handover

The Week 8 engineering package brings together:

- Prepared machine learning datasets.
- Saved classification, regression and clustering models.
- An integrated prediction pipeline.
- Documented input and output requirements.
- Pinned dependencies and environment setup instructions.
- Pipeline logging and error handling.
- Week 7 testing and reliability evidence.
- Cross-track validation findings.
- Prediction outputs and execution summaries.
- Known limitations and responsible AI considerations.
- Project documentation and presentation materials.

The integrated pipeline provides a common execution workflow for the seven supported prediction tasks.

The available saved model artifacts were used because no separate final Data Science model artifact was supplied for integration.

The package is prepared for engineering handover and demonstration within its documented scope.

## 11. Conclusion

The HealthConnect Rwanda ML Engineering project establishes a structured workflow covering data preparation, model development, evaluation, pipeline testing and integrated prediction execution.

Week 6 delivered the modelling, evaluation, clustering and explainability workflows. Week 7 strengthened pipeline reliability through targeted testing, data-cleaning improvements and cross-track validation. Week 8 delivered a reusable prediction pipeline supporting seven tasks, with logging, error handling, output validation and reproducibility documentation.

The Week 7 test suite recorded 38 passing checks with no failures. All seven Week 8 prediction tasks also completed successfully during the recorded pipeline execution, with matching input and output record counts.

The completed engineering package provides the required foundation for final integration handover and demonstration within its documented scope.

The models remain provisional and are not clinically validated. Their predictions must not be treated as established clinical outcomes or used as the sole basis for real-world healthcare decisions.