# HealthConnect Rwanda — Machine Learning Engineering

**Project:** AI-Powered Clinical Risk Prediction and Healthcare Intelligence System  
**Programme:** AnalystLab Africa Experience Lab — Week 6  
**Deadline:** Sunday, 25 October 2026, 11:59 PM WAT

## Project Overview

This project addresses the Week 6 Machine Learning Engineering requirements in the supplied assignment PDF. It covers data preparation and feature engineering, four classification tasks, two regression tasks, patient and facility clustering, anomaly detection, explainable AI, model comparison, and a proposed deployment architecture.

The models and findings are exploratory. Their results should be interpreted alongside the documented data limitations and evaluation metrics; they are not evidence of clinically validated prediction performance.

## Project Structure

- `Data/Original_Datasets/` — original supplied CSV and Excel files; preserve these unchanged.
- `Data/Processed_Datasets/` — cleaned and feature-engineered datasets.
- `Notebook/` — the Week 6 machine learning notebook.
- `Models/` — trained model artifacts, where applicable.
- `Reports/` — data readiness, model evaluation, model comparison, explainability, anomaly detection, clustering, executive summary, and deployment architecture reports.
- `Presentation/` — project presentation.
- `App/` — application files, if included in the deliverables.
- `References/` — supplied assignment PDF.

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

### SHAP workflow

If the SHAP script is present in the project, run it from the project root:

```bash
python week6_shap_explainability.py
```

If SHAP is not installed in the active Python environment, install the required packages:

```bash
pip install shap matplotlib
```

The documented outputs are:

- `Reports/Week6_SHAP_Explainability_Report.txt`
- `Reports/Week6_SHAP_Feature_Importance.csv`
- `Reports/Week6_SHAP_Feature_Importance.png`
- `Reports/Week6_SHAP_Individual_Explanations.csv`

These output paths apply when the script generates the corresponding files successfully.

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

This project demonstrates a machine learning engineering workflow using the supplied HealthConnect Rwanda datasets. Its value lies in the implemented preparation and modelling work, evaluation against baselines, clustering and explainability analysis, and clear documentation of data limitations. Further data improvements and validation are necessary before the models could be considered for real-world healthcare use.