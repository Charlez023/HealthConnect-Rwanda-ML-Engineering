"""Build Week 6 model comparison and evaluation outputs from existing baseline results.

Run from the HealthConnect Rwanda project root:
    python build_week6_model_evaluation.py

This script reads existing reports and processed data only. It does not retrain models
or modify original datasets.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "Reports"
DATA = ROOT / "Data" / "Processed_Datasets"

metrics_path = REPORTS / "week6_baseline_metrics.csv"
if not metrics_path.exists():
    raise FileNotFoundError(
        f"Could not find {metrics_path}. Run train_week6_models.py first."
    )

metrics = pd.read_csv(metrics_path)
if metrics.empty:
    raise ValueError(f"{metrics_path} contains no model results.")

# Normalize column names only for identifying fields; preserve original columns in output.
def norm(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())

normalized = {norm(c): c for c in metrics.columns}

def find_col(candidates):
    for candidate in candidates:
        key = norm(candidate)
        if key in normalized:
            return normalized[key]
    return None

model_col = find_col(["model", "task", "model_name", "name", "target"])
metric_cols = []
for c in metrics.columns:
    if norm(c) in {
        "accuracy", "precision", "macroprecision", "recall", "macrorecall",
        "f1", "f1score", "macrof1", "rocauc", "auc", "mae", "rmse", "r2",
        "mape", "testrows", "testsize", "nrows", "rows", "samples"
    }:
        metric_cols.append(c)

# Keep all original metric fields, even if their names differ from common conventions.
comparison_path = REPORTS / "Week6_Model_Comparison.csv"
metrics.to_csv(comparison_path, index=False)

limitations = []
limitations_path = DATA / "week6_target_limitations.csv"
if limitations_path.exists():
    try:
        lim = pd.read_csv(limitations_path)
        limitations = lim.fillna("").astype(str).to_dict("records")
    except Exception as exc:
        limitations = [{"note": f"Could not parse target limitations file: {exc}"}]

report_lines = [
    "HealthConnect Rwanda Week 6 - Baseline Model Evaluation",
    "",
    "This report summarizes existing baseline metrics; it does not retrain models.",
    "Results are exploratory and should not be interpreted as clinically validated.",
    "",
    "## Results",
]
for _, row in metrics.iterrows():
    label = str(row[model_col]) if model_col else str(row.iloc[0])
    report_lines.append(f"\n### {label}")
    for col in metrics.columns:
        if col == model_col:
            continue
        value = row[col]
        if pd.isna(value):
            value_text = "not reported"
        elif isinstance(value, float):
            value_text = f"{value:.4f}" if abs(value) < 1000 else f"{value:,.2f}"
        else:
            value_text = str(value)
        report_lines.append(f"- {col}: {value_text}")

report_lines += [
    "",
    "## Interpretation and limitations",
    "- Baseline classification scores near chance indicate weak predictive performance.",
    "- Negative regression R² values indicate worse performance than predicting the test-set mean.",
    "- The claims model is only an Approved-versus-Pending alternative; the source has no Rejected labels.",
    "- The high-risk target is a severity-derived proxy, not a clinically validated risk label.",
    "- Satisfaction classes are derived from feedback ratings, not a separately supplied clinical outcome.",
    "- Length-of-stay results use only appointment-linked admissions, a limited subset of the available admission records.",
    "- Model metrics and feature-importance rankings do not establish causation or clinical usefulness.",
    "- Do not use these baseline models to make real patient-care, insurance, or billing decisions.",
]
if limitations:
    report_lines += ["", "## Recorded target limitations"]
    for item in limitations:
        parts = [f"{k}: {v}" for k, v in item.items() if v.strip()]
        if parts:
            report_lines.append("- " + "; ".join(parts))

report_path = REPORTS / "Week6_Model_Evaluation_Report.txt"
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

print("Completed.")
print(f"Model comparison: {comparison_path.relative_to(ROOT)}")
print(f"Evaluation report: {report_path.relative_to(ROOT)}")
print(f"Rows summarized: {len(metrics)}")
print(f"Metric columns copied: {', '.join(metric_cols) if metric_cols else 'all existing columns preserved'}")
