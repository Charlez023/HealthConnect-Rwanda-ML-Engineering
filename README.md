# HealthConnect Rwanda - Machine Learning Engineering

**Project:** AI-Powered Clinical Risk Prediction and Healthcare Intelligence System

**Programme:** AnalystLab Africa Experience Lab

**Track:** Machine Learning Engineering

## 1. Project Overview

HealthConnect Rwanda is a machine learning engineering project focused on developing exploratory healthcare prediction and intelligence solutions using the supplied healthcare datasets.

The project covers three phases:

- **Week 6:** Data preparation, feature engineering, model training, evaluation, clustering and explainability.
- **Week 7:** Pipeline testing, reliability improvements and cross-track validation.
- **Week 8:** Integration of saved machine learning models into a reusable prediction pipeline with logging, error handling and reproducibility.

The objective is to establish a structured and reproducible machine learning workflow that supports model execution, prediction generation, evaluation and engineering handover.

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
| `requirements.txt` | Project dependencies. |

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

Further details are documented in `Reports/Data_Readiness_Audit.txt`.

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

## 5. Week 7 - Testing and Reliability

Week 7 focused on validating and improving the existing machine learning pipeline.

Testing covered data preparation, numeric conversion, missing values, duplicates, data merging, preprocessing compatibility, saved model loading, prediction generation and facility clustering.

### Verified improvement

The `clean_records()` function in `prepare_week6_datasets.py` was improved to:

- Convert numeric values stored as text using `pd.to_numeric(..., errors="coerce")`.
- Handle positive and negative infinity.
- Record numeric conversion failures.

The prepared datasets were regenerated, and the affected tests passed successfully without changing their previous row counts.

### Test results

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

### HC-POD Cross-Track Validation

Week 7 also included independent validation of findings from the Data Analytics track.

Key findings included:

- The insurance claims dataset contains 21,028 Pending and 21,008 Approved records, with no Rejected records.
- Only 5,000 of 60,000 billing records linked to available appointments.
- Only 1,756 of 21,064 feedback records linked to available appointments.
- Linked facility rating averages ranged from approximately 2.76 to 3.19.
- ANOVA and Kruskal-Wallis tests did not provide statistically significant evidence of differences in facility ratings.

These findings do not provide sufficient evidence to rank facilities by patient satisfaction.

The reported dashboard correction involving readmission counts could not be independently verified using the supplied source-data archive.

Detailed findings are available in `Reports/Week7_Cross_Track_Testing_Evidence.md`.

## 6. Week 8 - Integrated Prediction Pipeline

Week 8 delivered a reusable prediction pipeline that integrates the available saved model artifacts and supports seven prediction tasks.

The pipeline uses existing trained models. It does not retrain models or modify original datasets during prediction execution.

### Setup

The project uses Python 3.11.9.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check package compatibility:

```powershell
python -m pip check
```

### Supported prediction tasks

| Task | Prepared input dataset | Records |
|---|---|---:|
| `patient_satisfaction` | `patient_satisfaction_model.csv` | 1,756 |
| `high_risk_severity_proxy` | `high_risk_severity_proxy.csv` | 5,000 |
| `insurance_claims` | `claims_approved_pending_alternative.csv` | 42,036 |
| `billing_amount` | `billing_amount_model.csv` | 5,000 |
| `length_of_stay` | `length_of_stay_model.csv` | 586 |
| `patient_clustering` | `patient_clustering.csv` | 20,000 |
| `facility_clustering` | `facility_clustering.csv` | 20 |

All input datasets are located in `Data/Processed_Datasets/`.

### Running predictions

Execute the following command from the project root:

```powershell
python run_week8_pipeline.py --task TASK_NAME --input INPUT_FILE
```

Example:

```powershell
python run_week8_pipeline.py --task patient_satisfaction --input Data/Processed_Datasets/patient_satisfaction_model.csv
```

An optional output location can be specified using `--output`.

### Prediction outputs

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

All seven supported prediction tasks completed successfully. Their generated prediction record counts matched the corresponding input record counts.

## 7. Reproducibility, Logging and Error Handling

The model training workflow uses a fixed random state of `42` for reproducibility across supported stochastic operations and data splits.

The installed Python environment and package versions were verified against `requirements.txt`. Package compatibility was also checked using `pip check`.

An isolated temporary virtual environment was used to verify that:

- Saved model artifacts could be loaded successfully.
- The integrated pipeline could execute independently of the original development environment.
- All seven prediction tasks completed successfully.
- Expected prediction record counts were generated.
- Execution summaries were produced.

The Week 8 pipeline incorporates the following engineering safeguards:

- Input file and model artifact validation.
- Required feature validation.
- Prediction and output record-count checks.
- Output path validation.
- Execution logging and error reporting.
- Console logging fallback when file-logging setup fails.
- Exception handling for unexpected execution failures.

Execution logs are stored in:

`Logs/week8_pipeline.log`

These measures improve execution reliability, traceability and reproducibility across supported prediction tasks.

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

### Proposed Deployment Architecture

The proposed deployment architecture is conceptual and is not an implemented clinical application.

A potential deployment workflow could connect hospital systems to a web dashboard and API, validate incoming data, apply preprocessing and trained models, return predictions with explanations, and route results for human review.

The current engineering package establishes the reusable prediction pipeline and supporting model artifacts for demonstration. The project does not claim a production clinical deployment.

### Data Science Model Coordination

No separate final model artifact was supplied by the Data Science track for the Week 8 integration.

The ML Engineering pipeline therefore integrates the available saved model artifacts for all seven supported prediction tasks. The integrated workflow, its model inputs, prediction outputs and engineering validation are documented in this repository.

This handover accurately reflects the available model artifacts and the engineering integration completed. It does not claim receipt, approval or direct participation by the Data Science track in the final integration.

## 9. Conclusion

The HealthConnect Rwanda ML Engineering project establishes a structured workflow covering data preparation, model development, evaluation, pipeline testing and integrated prediction execution.

Week 6 delivered the modelling, evaluation, clustering and explainability workflows. Week 7 strengthened pipeline reliability through targeted testing, data-cleaning improvements and cross-track validation. Week 8 delivered a reusable prediction pipeline supporting seven tasks, with logging, error handling, output validation and clean-environment reproducibility.

The Week 7 test suite recorded 38 passing checks with no failures. All seven Week 8 prediction tasks also completed successfully during clean-environment reproducibility validation, with matching input and output record counts.

The completed engineering package is ready for handover and demonstration within its documented scope.

The models remain provisional and are not clinically validated. Their predictions must not be treated as established clinical outcomes or used as the sole basis for real-world healthcare decisions.