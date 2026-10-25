# HealthConnect Rwanda
## Week 6 — Executive Summary

### 1. Project Overview

HealthConnect Rwanda is a healthcare machine learning project that explores how available healthcare data can support operational analysis and decision-making. The project examines patient satisfaction, patient risk indicators, insurance claims, billing amounts, length of stay, and facility segmentation.

The objective is to develop baseline machine learning models, evaluate their performance, identify patterns in the available data, and document the requirements for potential future deployment.

This work is exploratory. The models have not been clinically validated and should not be used independently to make clinical or financial decisions.

### 2. Data Preparation and Readiness

The project uses the supplied HealthConnect Rwanda datasets, including patient, appointment, admission, claims, billing, laboratory, and facility-related records.

Data preparation involved examining available fields, preparing model-ready datasets, and documenting data quality and relationship limitations. These checks helped identify where the available data was insufficient to support reliable prediction.

Key limitations included:

- Only 586 of the 5,000 admission records could be linked to the available appointment IDs. Unmatched appointments could not safely be labeled as non-admissions.
- The claims data contains Approved and Pending outcomes but no Rejected examples. Therefore, a three-class claims model could not be responsibly trained using the supplied labels.
- Only 1,756 feedback records matched appointments, limiting the sample available for patient satisfaction modeling.
- Relationships between facility, department, doctor, and nurse records were incomplete.
- Laboratory results were recorded as Normal or Abnormal without units or clinical reference ranges.
- Billing, payment, and revenue summaries could not be directly reconciled using the supplied records.

These limitations were documented rather than addressed through fabricated data or unsupported assumptions.

### 3. Machine Learning Approach and Results

The project developed and evaluated exploratory baseline models for several prediction tasks.

**Patient satisfaction classification:** The model used available rating-derived labels and achieved 37.78% test accuracy with a macro F1-score of 0.3635. Performance was limited and did not establish reliable prediction of patient satisfaction.

**High-risk patient classification:** A severity-proxy classification task achieved 50.20% test accuracy and a macro F1-score of 0.4742. Because the labels are proxies rather than clinically verified outcomes, the model cannot be treated as a validated clinical risk assessment tool.

**Insurance claims classification:** An alternative Approved-versus-Pending task achieved approximately 50.02% test accuracy and an ROC-AUC of approximately 0.50, indicating little demonstrated predictive value. The absence of Rejected examples also prevented a defensible three-class classification task.

**Billing amount prediction:** The selected mean-prediction baseline had a mean absolute error (MAE) of approximately RWF 70,207 and an R² of -0.0023. It did not improve on the mean-prediction baseline.

**Length of stay prediction:** The model achieved an MAE of approximately 2.47 days and an R² of -0.1554. The negative R² indicates that it performed worse than predicting the test-set mean.

Overall, the results demonstrate why models must be evaluated against appropriate baselines rather than assuming that training a model produces a useful predictor.

### 4. Facility Clustering and Explainability

Facility clustering grouped 20 facilities into three exploratory clusters using available facility-level appointment counts and selected resource records.

Resource counts were very similar across facilities, so the resulting groups were influenced mainly by differences in appointment-record volumes. These clusters should not be interpreted as facility quality rankings, measures of clinical performance, or sufficient evidence for resource allocation decisions.

The project includes permutation feature importance and SHAP explainability work to examine how features influence model outputs. However, explanations must be interpreted alongside model performance. Explaining a weak model does not make its predictions reliable, and feature importance does not establish causation.

Statistical and structural anomaly flags support further investigation but do not, by themselves, prove fraud, data errors, or clinical abnormalities.

### 5. Proposed Deployment Approach

The proposed deployment architecture describes how a future system could connect a hospital dashboard and Hospital Information System (HIS) to a backend API, input validation and preprocessing, a trained model, prediction and explanation outputs, and human review.

A practical implementation would require user authentication, appropriate access controls, secure handling of patient information, input validation, model-performance monitoring, and safeguards against unnecessary exposure of identifiable data. Model updates would also require evaluation and documented approval.

This architecture is a proposal, not evidence of a deployed production system. Further data preparation, model evaluation, privacy and security review, clinical validation, and relevant clinical and operational approval would be required before deployment could be considered.

### 6. Key Findings and Recommendations

The project demonstrates a structured machine learning workflow using the supplied healthcare datasets. It covers data preparation, baseline model development, evaluation, patient and facility clustering, feature-importance analysis, anomaly review, and documentation of limitations.

However, current results do not justify operational deployment. Several classification models perform poorly, while billing and length-of-stay prediction fail to outperform their mean baselines. Missing target labels and incomplete dataset relationships also limit the prediction tasks that can be responsibly developed.

Recommended next steps are:

1. Obtain reliable, consistently linked admission and appointment records with verified outcome labels.
2. Obtain representative claims data containing the required outcome classes, including rejected claims where appropriate.
3. Improve dataset relationships and document field definitions, units, and valid values.
4. Reassess model performance using suitable validation strategies and meaningful baseline comparisons.
5. Review feature explanations and flagged anomalies with appropriate subject-matter experts.
6. Assess potential bias and protect patient confidentiality throughout future analysis.
7. Complete security, privacy, and clinical governance reviews before considering deployment.

### 7. Conclusion

HealthConnect Rwanda provides an exploratory foundation for examining machine learning applications in healthcare operations. Its main outcome is a documented workflow showing what can be modeled with the available data, how the baseline models performed, and where data limitations prevent defensible conclusions.

Further progress depends on improving data quality, obtaining reliable labels, and demonstrating meaningful predictive performance. Until these requirements are met and appropriate validation is completed, the models should remain exploratory and should not be used to make real-world clinical, insurance, or billing decisions.