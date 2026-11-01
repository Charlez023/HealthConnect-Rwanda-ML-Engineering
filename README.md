# HealthConnect Rwanda - Machine Learning Engineering

**Project:** AI-Powered Clinical Risk Prediction and Healthcare Intelligence System
**Programme:** AnalystLab Africa Experience Lab - Machine Learning Engineering.

## Project Overview

This project addresses the Week 6 and Week 7 Machine Learning Engineering requirements in the supplied assignment PDFs.

Week 6 covered data preparation and feature engineering, classification and regression tasks, patient and facility clustering, anomaly detection, explainable AI, model comparison, and a proposed deployment architecture.

Week 7 focused on testing and improving the reliability of the existing ML pipeline through data-processing validation, preprocessing checks, model integration testing, error handling, dependency verification, and cross-track validation.

The models and findings are exploratory. Their results should be interpreted alongside the documented data limitations and evaluation metrics; they are not evidence of clinically validated prediction performance.

## Project Structure

- `Data/Original_Datasets/` - original supplied CSV and Excel files; preserve these unchanged.
- `Data/Processed_Datasets/` - cleaned and feature-engineered datasets.
- `Notebook/` - the Week 6 machine learning notebook.
- `Models/` - trained model artifacts, where applicable.
- `Reports/` - data readiness, model evaluation, model comparison, explainability, anomaly detection, clustering, deployment architecture, and Week 7 testing reports.
- `Presentation/` - project presentation.
- `App/` - application files, if included in the deliverables.
- `References/` - supplied assignment PDFs.
- `requirements.txt` - project Python dependencies.
- `test_week7_pipeline.py` - targeted Week 7 pipeline test script.

## Data Readiness and Limitations

The `Reports/Data_Readiness_Audit.txt` report documents the main constraints identified during data inspection:

- Only 586 admission records link to the 5,000 available appointment records.
- Claims contain Approved and Pending labels, but no Rejected class.
- Facility and department relationships are weak, with matches in only about 5% of comparable records.
- Bed records are marked Available, and equipment availability status is not supplied.
- Billing, payment, and revenue-summary totals do not reconcile directly.
- Appointment references and feedback records have limited linkage.

These constraints affect which prediction tasks can be supported and how results should be interpreted. Hospital admission prediction and three-class claims prediction cannot be reliably trained without suitable labels.

Do not fabricate admission outcomes, rejected claims, facility names or attributes, ownership, locations, bed occupancy, or equipment availability.

## Machine Learning and Evaluation

The project includes classification and regression tasks, patient clustering, facility clustering, anomaly analysis, and model explainability. Evaluation reports and metrics should be used to compare model performance against appropriate baselines.

Some models perform close to or worse than their baselines. These results should be reported honestly rather than presented as successful predictive systems. The high-risk task uses a constructed severity proxy and must not be described as a clinically validated risk outcome.

## Facility Clustering

The facility clustering workflow uses direct `FacilityID`-level record counts and the `RoomID`-to-room relationship for bed counts. It avoids unreliable appointment-to-department joins and does not infer missing facility attributes.

The exploratory analysis produces three facility clusters and their profiles. Because resource counts are similar across many facilities, the clusters largely reflect differences in appointment volume and should not be interpreted as measures of facility quality or clinical performance.

To regenerate the facility clustering outputs, run this command from the project root if the script and its dependencies are available:

```bash
python build_week6_facility_clustering.py
```

## Explainable AI and Anomaly Analysis

The project includes permutation feature importance and SHAP explainability work to examine how features influence model outputs. The findings describe model behaviour, not causation. Feature importance from weak models should be treated as exploratory, and explanations do not establish clinical relevance.

The high-risk target is a constructed severity proxy, not a validated clinical outcome. Explainability results should therefore be interpreted with this limitation in mind.

The project also includes anomaly analysis to identify unusual patterns in the available data. Such patterns require contextual review and should not automatically be treated as errors, fraud, or clinical problems.

### SHAP Workflow

If the SHAP script is present in the project, run it from the project root:

```bash
python week6_shap_explainability.py
```

The required packages are included in `requirements.txt`. If they are not installed in the active Python environment, install the project dependencies as described below.

The documented outputs are:

- `Reports/Week6_SHAP_Explainability_Report.txt`
- `Reports/Week6_SHAP_Feature_Importance.csv`
- `Reports/Week6_SHAP_Feature_Importance.png`
- `Reports/Week6_SHAP_Individual_Explanations.csv`

These output paths apply when the script generates the corresponding files successfully.

## Week 7 - ML Pipeline Testing and Reliability

Week 7 focused on validating the existing Week 6 pipeline rather than rebuilding or retraining the models.

The testing covered:

- Data preparation and prepared dataset target integrity.
- Data cleaning, missing values, invalid numeric values, duplicates, and outlier retention.
- Numeric values stored as text and positive or negative infinity.
- Data merging and duplicate appointment-key handling.
- Loading and testing saved classification and regression pipelines.
- Valid predictions, unseen categorical values, and missing-feature handling.
- Facility clustering dataset integrity, feature compatibility, preprocessing, and predictions.

### Verified Improvement

Testing identified an issue in the data-cleaning process involving numeric values stored as text. The `clean_records()` function in `prepare_week6_datasets.py` was updated to convert the relevant numeric fields using `pd.to_numeric(..., errors="coerce")`, handle positive and negative infinity, and record numeric conversion failures.

The prepared datasets were regenerated, and the affected tests were rerun successfully. Their row counts remained consistent with the previous preparation run.

### Pipeline Test Results

The final test run recorded:

- Total checks: 38
- Passed: 38
- Failed: 0
- Not applicable: 0

The test suite is available in:

```bash
python test_week7_pipeline.py
```

The generated evidence files are:

- `Reports/Week7_ML_Pipeline_Test_Results.csv`
- `Reports/Week7_ML_Pipeline_Test_Summary.json`

The tests validate the selected pipeline components and saved artifacts. They do not establish clinical validity or guarantee that every possible input condition has been covered.

## Dependency Management and Reproducibility

The project uses Python 3.11.9. Its third-party dependencies are documented in the root-level `requirements.txt` file.

To install the listed dependencies in an active Python environment:

```bash
python -m pip install -r requirements.txt
```

To check installed dependency compatibility:

```bash
python -m pip check
```

The recorded dependency check returned:

```text
No broken requirements found.
```

A clean installation test in a fresh environment has not yet been performed. Therefore, the dependency check should not be interpreted as proof of complete fresh-environment reproducibility.

## HC-POD Cross-Track Validation

Week 7 included cross-track testing using findings from the Data Analytics track.

The review independently confirmed:

- The claims dataset contains 21,028 Pending and 21,008 Approved records, with no Rejected records.
- Of 60,000 billing records, 5,000 link to the available appointment records, leaving 55,000 unmatched (91.67%).
- Of 21,064 feedback records, 1,756 link to the available appointment records.
- The linked facility rating averages range from approximately 2.76 to 3.19.
- The ANOVA and Kruskal-Wallis tests did not provide statistically significant evidence of differences in facility ratings.

These results do not support ranking facilities by satisfaction ratings. The reported dashboard correction involving readmission counts was recorded as an external finding but could not be independently verified using the supplied source-data archive.

Detailed methods and results are documented in:

`Reports/Week7_Cross_Track_Testing_Evidence.md`

## Week 7 Documentation and Evidence

The Week 7 reports include:

- `Reports/Week7_ML_Pipeline_Testing_and_Reliability_Report.md` - testing scope, methods, improvements, results, dependencies, and remaining limitations.
- `Reports/Week7_Project_Summary.md` - summary of the Week 7 work and outcomes.
- `Reports/Week7_Cross_Track_Testing_Evidence.md` - independently verified cross-track findings and their implications.
- `Reports/Week7_ML_Pipeline_Test_Results.csv` - individual test results.
- `Reports/Week7_ML_Pipeline_Test_Summary.json` - test execution summary.

## Proposed Deployment Architecture

The deployment architecture is a conceptual proposal, not a deployed clinical application. A future implementation could connect hospital systems to a web dashboard and API, validate incoming data, apply preprocessing and trained models, return predictions with explanations, and route results for human review.

Before deployment, the system would require stronger data relationships, suitable outcome labels, further validation, privacy and security review, monitoring, and appropriate clinical and organisational approval. Predictions should support qualified human review rather than replace professional judgement.

## Responsible AI and Working Rules

1. Keep the original datasets unchanged.
2. Build only what the assignment requires.
3. Avoid unnecessary tests, redundant experiments, and duplicate deliverables.
4. Use task-specific modelling datasets and valid joins.
5. Distinguish verified results, assumptions, constructed proxies, and limitations.
6. Do not claim a task is complete until its required output has been produced and checked.
7. Protect patient confidentiality and assess potential bias before any real-world use.
8. Do not use exploratory predictions as the sole basis for clinical or operational decisions.

## Conclusion

This project demonstrates a machine learning engineering workflow using the supplied HealthConnect Rwanda datasets. Week 6 established the data preparation, modelling, evaluation, clustering, and explainability workflows. Week 7 strengthened the reliability of the existing pipeline through targeted testing, a verified data-cleaning improvement, dependency checks, and cross-track validation.

The final test run recorded 38 passed checks and no failures. The project's remaining data and modelling limitations have also been documented to support responsible interpretation and future improvements. Further data improvements and validation are necessary before the models could be considered for real-world healthcare use.
