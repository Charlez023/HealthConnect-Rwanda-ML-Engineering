# Week 7 Cross-Track Testing Evidence

## 1. Overview

As part of the Week 7 cross-track testing activity, findings from the Data Analytics track were reviewed and independently checked against the HealthConnect Rwanda source datasets.

The purpose was to verify shared data issues, assess their implications for the ML pipeline, and document any limitations that could affect model development and interpretation.

**Tracks involved:**
- Machine Learning Engineering
- Data Analytics

## 2. Findings and Verification Results

### 2.1 Insurance Claims Status Distribution

**Data Analytics finding:**  
The insurance claims dataset contains 21,028 Pending claims and 21,008 Approved claims, with no Rejected records across the four providers.

**ML Engineering verification:** Confirmed.

The `Insurance_Claims.csv` dataset was inspected. It contains 42,036 records with the following status distribution:

| Claim Status | Number of Records |
|---|---:|
| Pending | 21,028 |
| Approved | 21,008 |
| Rejected | 0 |
| **Total** | **42,036** |

**Implication for ML:**  
The claims classification task remains an Approved-versus-Pending alternative. A Rejected class cannot be developed or evaluated because no such labelled records exist in the source dataset. No labels were fabricated.

### 2.2 Billing and Appointment Linkage

**Data Analytics finding:**  
The appointments table contains 5,000 records, while billing records reference appointments beyond the available appointment range. Consequently, 91.7% of billing records cannot be linked to a facility, department, or appointment date.

**ML Engineering verification:** Confirmed.

The `HC_Billing.csv` dataset was joined to the Appointments sheet in `HC_Clinical_Transactions.xlsx` using `AppointmentID`.

The verification produced the following results:

| Measurement | Records |
|---|---:|
| Total billing records | 60,000 |
| Total appointment records | 5,000 |
| Billing records successfully matched | 5,000 |
| Billing records without a matching appointment | 55,000 |
| Unmatched billing records | 91.67% |

The unmatched billing records also had no corresponding FacilityID, DepartmentID, or appointment Date through this relationship.

**Implication for ML:**  
The incomplete linkage limits the use of billing records for analyses requiring appointment-level facility, department, or date information. The existing billing amount model should not be interpreted as having complete facility-level or appointment-level coverage.

### 2.3 Patient Satisfaction and Facility Ratings

**Data Analytics finding:**  
Facility ratings range from approximately 2.76 to 3.19, with no statistically significant difference between facilities (p = 0.80). Therefore, facilities should not be ranked based on these ratings.

**ML Engineering verification:** Confirmed.

The `HC_Patient_Feedback.csv` dataset was joined to the Appointments sheet using `AppointmentID`.

The join produced:

| Measurement | Records |
|---|---:|
| Total patient feedback records | 21,064 |
| Feedback records matched to appointments | 1,756 |
| Feedback records without a matching appointment | 19,308 |

The 1,756 matched feedback records correspond to the row count of the prepared patient satisfaction modelling dataset.

Facility-level average ratings were calculated using the matched feedback records. The results ranged from **2.76 to 3.19** across 20 facilities.

Two statistical tests were performed:

| Statistical Test | Test Statistic | P-value |
|---|---:|---:|
| One-way ANOVA | 0.7349 | 0.7851 |
| Kruskal-Wallis | 13.7999 | 0.7952 |

Both tests produced p-values of approximately 0.80, above the 0.05 significance threshold.

**Implication for ML:**  
There is no statistically significant evidence of differences in average facility ratings in the matched data. Facility rankings based on these ratings are therefore not justified. The large number of unmatched feedback records is also an important data limitation.

### 2.4 Dashboard Readmission Count Correction

**Data Analytics finding:**  
Dashboard testing reportedly identified a blank-record issue that inflated readmission counts, showing 9 instead of 8. The issue was reportedly corrected.

**ML Engineering verification:** Not independently verified.

The supplied dataset archive did not contain the dashboard implementation, underlying dashboard query, or the Data Analytics testing log needed to reproduce this finding.

The correction is therefore recorded as a finding reported by the Data Analytics track, rather than an independently confirmed result.

## 3. Impact on the ML Engineering Pipeline

The verified findings have the following implications:

1. **Claims classification:** Retain the Approved-versus-Pending alternative. Do not introduce a Rejected class without genuine labelled records.
2. **Billing data:** Document the 91.67% unmatched billing records as a limitation when interpreting appointment-linked or facility-level analyses.
3. **Patient satisfaction:** Retain the rating-derived target while avoiding unsupported facility rankings.
4. **Data linkage:** Clearly document the limited appointment linkage affecting billing and patient feedback records.
5. **Model reliability:** These findings identify source-data limitations. They do not, by themselves, establish defects in the existing model pipelines.

## 4. Remaining Issues and Limitations

- 55,000 billing records cannot be linked to the available appointment records.
- 19,308 patient feedback records cannot be linked to the available appointment records.
- The source claims dataset contains no Rejected labels.
- Facility ratings do not show statistically significant differences in the matched feedback data.
- The reported dashboard readmission correction remains independently unverified.

## 5. Verification Scope

The verification used the supplied HealthConnect source-data archive, including:

- `Insurance_Claims.csv`
- `HC_Billing.csv`
- `HC_Patient_Feedback.csv`
- `HC_Clinical_Transactions.xlsx` - Appointments sheet

The checks involved status distribution analysis, appointment-level joins, unmatched-record counts, facility-level rating calculations, and statistical tests.

The findings in this report distinguish independently verified results from findings reported by the Data Analytics track but not independently reproduced.

## 6. Conclusion

The cross-track review confirmed three major Data Analytics findings concerning insurance claims, billing-to-appointment linkage, and facility satisfaction ratings.

These findings provide additional evidence of data limitations that must be considered when interpreting HealthConnect's ML outputs. They have been documented without fabricating missing labels, changing the original datasets, or unnecessarily retraining existing models.