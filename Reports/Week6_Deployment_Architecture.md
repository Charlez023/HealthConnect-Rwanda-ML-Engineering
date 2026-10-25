# HealthConnect Rwanda — Proposed ML Deployment Architecture

## Architecture Flow

```text
Hospital Staff
     |
     v
Web Interface / Hospital Dashboard
     |
     v
Hospital Information System (HIS)
     |
     v
Backend API (FastAPI)
     |
     v
Input Validation and Preprocessing
     |
     v
Trained ML Model
     |
     v
Prediction and Explanation
     |
     v
API Response to Dashboard
     |
     v
Human Review and Decision
     |
     v
Performance Monitoring and Model Retraining
```

## Components

1. **Data Sources:** The system would use available patient, appointment, admission, laboratory, claims, billing, and facility records. Data availability and linkage limitations must be considered before using these sources for predictions.

2. **Hospital Dashboard:** Provides authorized hospital staff with an interface to submit permitted patient or operational data and review model outputs.

3. **Hospital Information System (HIS) Integration:** A proposed API connection would allow the system to exchange permitted data with the hospital's existing information system, subject to compatibility and authorization.

4. **Backend API (FastAPI):** Receives prediction requests, validates inputs, calls the appropriate model, and returns results to the dashboard.

5. **Input Validation and Preprocessing:** Checks required fields, data types, missing values, and unexpected inputs. It applies the same preprocessing steps used during model training.

6. **Model-Serving Layer:** Loads the selected saved model and generates predictions for supported tasks. Only models that pass further evaluation should be considered for operational use.

7. **Prediction and Explanation:** Returns the prediction and, where supported, information about influential features. Outputs should include appropriate limitations and should not be treated as confirmed clinical findings.

8. **Human Review:** Healthcare professionals review model outputs alongside relevant clinical information. The system does not replace professional judgment or independently make clinical decisions.

## Security, Monitoring, and Maintenance

- Authenticate users and apply role-based access controls.
- Protect patient information in transit and at rest.
- Minimize the collection and retention of identifiable patient information.
- Monitor input-data quality, missing values, prediction errors, and changes in data patterns.
- Evaluate model performance using appropriate metrics and newly verified outcomes when available.
- Retrain models only when sufficient suitable data are available and evaluation shows a justified need.
- Version models and document changes before deploying updated versions.
- Require human oversight before any clinical action.

## Deployment Status and Limitations

This is a **proposed architecture**, not a deployed production system. The current models are exploratory baselines and have not been clinically validated. Some prediction tasks are limited by missing target labels, weak relationships between datasets, and poor baseline performance.

The architecture is therefore a proposed future implementation. Deployment would require improved data quality, further model evaluation, clinical validation, privacy and security review, compatible HIS integration, and appropriate approval from healthcare stakeholders.