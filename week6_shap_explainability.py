"""
Week 6 SHAP explainability for the exploratory high-risk severity proxy model.
Run from project root: python week6_shap_explainability.py

This fits a reproducible decision-tree model on the existing proxy-labelled dataset.
It does not treat the proxy as a clinically validated risk label.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from train_week6_models import load_csv, clean_features, build_pipeline, DATA, MODELS, REPORTS, RANDOM_STATE

warnings.filterwarnings("ignore")
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

DATASET = "high_risk_severity_proxy.csv"
TARGET = "HighRiskProxy"
MAX_EXPLAIN_ROWS = 100


def main():
    df = load_csv(DATASET)
    y = df[TARGET]
    X = clean_features(df.drop(columns=[TARGET]))
    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y if y.value_counts().min() >= 2 else None,
    )
    pipe = build_pipeline(
        X_train,
        DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE),
    )
    pipe.fit(X_train, y_train)

    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    X_sample = X_test.iloc[:min(MAX_EXPLAIN_ROWS, len(X_test))].copy()
    transformed = preprocessor.transform(X_sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names, index=X_sample.index)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_df)
    # SHAP versions differ: binary-class outputs can be list, 2-D array, or 3-D array.
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1 if len(shap_values) > 1 else 0])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1 if values.shape[2] > 1 else 0]
    if values.ndim != 2:
        raise RuntimeError(f"Unexpected SHAP output shape: {values.shape}")

    mean_abs = np.abs(values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap_value": mean_abs,
    }).sort_values("mean_absolute_shap_value", ascending=False)
    importance.to_csv(REPORTS / "Week6_SHAP_Feature_Importance.csv", index=False)

    # Keep a compact, readable top-15 global importance plot.
    top = importance.head(15).sort_values("mean_absolute_shap_value")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["mean_absolute_shap_value"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Encoded feature")
    plt.title("SHAP feature importance — high-risk severity proxy")
    plt.tight_layout()
    plt.savefig(REPORTS / "Week6_SHAP_Feature_Importance.png", dpi=160)
    plt.close()

    # Save a small set of individual explanations using the largest absolute SHAP feature.
    prediction = pipe.predict(X_sample)
    probability = pipe.predict_proba(X_sample)
    class_values = list(model.classes_)
    positive_index = class_values.index(1) if 1 in class_values else min(1, len(class_values)-1)
    base_values = np.asarray(explainer.expected_value)
    if base_values.ndim:
        base_value = float(base_values[positive_index])
    else:
        base_value = float(base_values)
    rows = []
    for i in range(min(10, len(X_sample))):
        top_idx = np.argsort(np.abs(values[i]))[::-1][:5]
        row = {
            "sample_index": int(i),
            "actual_proxy_label": str(y_test.loc[X_sample.index[i]]),
            "predicted_proxy_label": str(prediction[i]),
            "predicted_probability_class_1": float(probability[i, positive_index]),
            "base_value": base_value,
        }
        for rank, j in enumerate(top_idx, start=1):
            row[f"feature_{rank}"] = str(feature_names[j])
            row[f"feature_{rank}_shap_value"] = float(values[i, j])
            row[f"feature_{rank}_encoded_value"] = float(transformed_df.iloc[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPORTS / "Week6_SHAP_Individual_Explanations.csv", index=False)

    joblib.dump(pipe, MODELS / "high_risk_severity_proxy_shap_tree.joblib")
    report = [
        "HealthConnect Rwanda Week 6 — SHAP Explainability",
        "",
        "Model: Decision Tree classifier fitted on the existing HighRiskProxy target.",
        f"Dataset: Data/Processed_Datasets/{DATASET}",
        f"Rows used: {len(X)}; held-out test rows: {len(X_test)}; rows explained: {len(X_sample)}.",
        "",
        "Outputs:",
        "- Reports/Week6_SHAP_Feature_Importance.csv: global mean absolute SHAP values.",
        "- Reports/Week6_SHAP_Feature_Importance.png: top 15 encoded features.",
        "- Reports/Week6_SHAP_Individual_Explanations.csv: up to 10 individual explanations.",
        "- Models/high_risk_severity_proxy_shap_tree.joblib: fitted pipeline used for these explanations.",
        "",
        "Interpretation and limitations:",
        "- SHAP values explain this fitted model's output; they do not establish causation.",
        "- HighRiskProxy is a constructed severity proxy, not a clinically validated outcome.",
        "- Encoded categorical levels appear as separate features; importance can be split across levels.",
        "- Existing baseline performance is weak, so explanations describe a weak exploratory model and must not guide clinical decisions.",
        "- This implements SHAP; LIME is not implemented by this script.",
    ]
    (REPORTS / "Week6_SHAP_Explainability_Report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("SHAP outputs written to Reports/Week6_SHAP_*")


if __name__ == "__main__":
    main()
