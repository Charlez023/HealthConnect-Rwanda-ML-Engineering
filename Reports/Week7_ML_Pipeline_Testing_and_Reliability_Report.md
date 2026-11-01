# HealthConnect Rwanda - Week 7 ML Pipeline Testing and Reliability Report

## 1. Introduction

This report documents the testing, validation, and reliability improvements carried out on the HealthConnect Rwanda Machine Learning Engineering project during Week 7.

The work builds on the existing Week 6 pipeline. It focuses on validating data preparation, preprocessing, saved model pipelines, model inputs and outputs, clustering components, and dependency configuration.

The objective was to identify and correct verified reliability issues while preserving the original datasets, existing models, and completed project work.

## 2. Testing Scope

The following components were included in the testing process:

- Week 6 dataset preparation module.
- Prepared datasets and target columns.
- Data cleaning and preprocessing functions.
- Appointment and patient feature merging.
- Saved classification and regression pipelines.
- Unseen categorical values and missing model features.
- Facility clustering dataset, preprocessing, and saved model.
- Python dependencies and project setup.

Testing was performed against the existing project components. Model training was not repeated as part of the reliability tests.

## 3. Testing Approach

A dedicated test script, `test_week7_pipeline.py`, was created to run targeted checks against the existing Week 6 components.

The tests covered:

### 3.1 Data Preparation and Dataset Validation

- Confirmed that the Week 6 preparation module could be loaded.
- Checked that prepared datasets existed and contained records.
- Verified the presence of expected target columns.
- Checked for missing and non-finite target values.

### 3.2 Data Cleaning and Preprocessing

Controlled test data was used to examine:

- Invalid numeric values.
- Missing target values.
- Duplicate records.
- Potential outlier retention.
- Numeric values stored as text.
- Unparseable numeric values.
- Positive and negative infinity values.

### 3.3 Data Merging

The merge function was tested using controlled data to confirm:

- Valid appointment and patient records could be joined.
- Expected patient features were retained.
- Duplicate appointment keys were rejected when they violated the expected many-to-one relationship.

### 3.4 Saved Model Pipeline Integration

The five existing supervised model pipelines were tested without retraining.

The checks covered:

- Loading the saved model artifacts.
- Loading the corresponding prepared datasets.
- Generating predictions from valid input records.
- Checking classification predictions against valid target classes.
- Checking regression predictions for finite values.
- Handling previously unseen categorical values.
- Rejecting inputs with missing required features.

### 3.5 Facility Clustering

The facility clustering components were checked for:

- Dataset integrity.
- Alignment between facility records and saved segment IDs.
- Successful loading of the saved clustering artifact.
- Consistency of the expected feature inputs.
- Successful preprocessing transformations.
- Agreement between generated predictions and saved segment labels.

## 4. Issues Identified and Improvements

### 4.1 Numeric Values Stored as Text

**Issue identified:**

The data-cleaning function did not explicitly convert known numeric fields stored as text into numeric values before applying numeric cleaning rules.

This could allow invalid or unparseable values to pass through preprocessing inconsistently.

**Improvement implemented:**

The `clean_records()` function in `prepare_week6_datasets.py` was updated to convert known numeric fields using `pd.to_numeric(..., errors="coerce")`.

The affected fields include:

- Age
- HeightCm
- WeightKg
- BMI
- Amount
- ClaimAmount
- LengthOfStayDays

Values that could not be converted were treated as missing. Positive and negative infinity values were also converted to missing values.

The preprocessing summary was updated to record numeric conversion failures.

**Verification:**

The dataset preparation script was rerun, followed by the Week 7 pipeline test script. The targeted cleaning checks passed.

### 4.2 Missing and Invalid Values

The controlled cleaning tests confirmed that missing target values were removed, invalid values were handled, and duplicate records were removed as expected.

Potential outliers were retained rather than automatically discarded.

### 4.3 Duplicate Merge Keys

The merge validation test confirmed that duplicate appointment keys were rejected when they violated the expected many-to-one relationship.

No change to the merge function was required.

## 5. Test Results

The final execution of `test_week7_pipeline.py` produced the following results:

| Test Result | Count |
|---|---:|
| Total checks | 38 |
| Passed | 38 |
| Failed | 0 |
| Not applicable | 0 |

**Overall result: All 38 checks passed.**

The successful results confirm that the tested data preparation, cleaning, merging, saved supervised model pipelines, and facility clustering components behaved as expected under the tested conditions.

The test outputs were saved to:

- `Reports/Week7_ML_Pipeline_Test_Results.csv`
- `Reports/Week7_ML_Pipeline_Test_Summary.json`

## 6. Dependency and Reproducibility Verification

A root-level `requirements.txt` file was added to document the project's Python dependencies.

The project uses Python 3.11.9.

The dependency file includes:

- NumPy
- Pandas
- OpenPyXL
- Scikit-learn
- SciPy
- Joblib
- Matplotlib
- SHAP
- IPykernel

The dependencies were checked using:

`python -m pip check`

The command returned:

`No broken requirements found.`

This confirms that the installed environment had no reported broken package requirements at the time of checking.

A fresh environment installation test was not performed. Therefore, a completely independent clean-install reproduction has not yet been established.

## 7. Cross-Track Testing

Cross-track testing was conducted with the Data Analytics track.

The review independently confirmed the following findings:

1. **Insurance claims:** The source contains 21,028 Pending and 21,008 Approved records, with no Rejected labels.
2. **Billing linkage:** 55,000 of 60,000 billing records (91.67%) could not be matched to the available 5,000 appointment records.
3. **Facility satisfaction:** Average facility ratings ranged from approximately 2.76 to 3.19. Statistical tests did not show significant differences in the matched ratings.

The reported dashboard readmission-count correction was recorded but could not be independently reproduced because the supplied materials did not include the dashboard implementation or its testing evidence.

Further details are documented in:

`Reports/Week7_Cross_Track_Testing_Evidence.md`

## 8. Remaining Limitations

Although all 38 pipeline checks passed, some project limitations remain:

- The claims dataset has no Rejected class, so a three-class claims model cannot be validated.
- A large proportion of billing records cannot be linked to the available appointment records.
- Many patient feedback records cannot be linked to the available appointment records.
- Facility ratings do not provide statistically significant evidence for ranking facilities.
- The billing amount model has weak predictive performance. This is a modelling limitation and was not established as a pipeline reliability defect by these tests.
- The classification models also have documented predictive limitations. Passing pipeline tests does not establish predictive accuracy or clinical validity.
- A fresh environment installation test has not been completed.

## 9. Conclusion

The Week 7 testing exercise validated the tested components of the existing HealthConnect Rwanda ML pipeline.

A numeric conversion weakness in the data-cleaning function was identified and corrected. The preparation process was rerun, and the relevant tests passed afterward.

The final test execution recorded 38 passed checks, with no failures or non-applicable checks. Dependency compatibility was also verified in the existing Python environment.

The work improved the reliability and documentation of the existing pipeline without rebuilding the project, fabricating missing labels, or unnecessarily retraining the models.

The resulting test scripts, machine-readable results, dependency file, and cross-track evidence provide supporting documentation for the Week 7 ML Pipeline Testing and Reliability Package.