import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

input_dir = "result_all/Fall"
output_dir = "result_calibrated"

os.makedirs(output_dir, exist_ok=True)

results = ['F00', 'F01', 'F02', 'F03', 'F05', 'F06', 'F10', 'F17', 'F32', 'F40', 'F41', 'F43']

for file in os.listdir(input_dir):
    if not file.endswith(".csv"):
        continue

    disease_name = file.replace(".csv", "")
    file_path = os.path.join(input_dir, file)

    print(f"\nProcessing: {disease_name}")

    df = pd.read_csv(file_path)

    y = df["label"].values
    scores = df["score"].values.reshape(-1, 1)

    X_train, X_calib, y_train, y_calib = train_test_split(
        scores, y, test_size=0.3, stratify=y, random_state=42
    )

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(X_calib, y_calib)

    calibrated_prob = calibrator.predict_proba(scores)[:, 1]
    df["calibrated_score"] = calibrated_prob

    auc_before = roc_auc_score(y, scores)
    auc_after = roc_auc_score(y, calibrated_prob)

    brier_before = brier_score_loss(y, scores)
    brier_after = brier_score_loss(y, calibrated_prob)

    print(f"AUC before: {auc_before:.4f} | after: {auc_after:.4f}")
    print(f"Brier before: {brier_before:.4f} | after: {brier_after:.4f}")

    results.append({
        "disease": disease_name,
        "auc_before": auc_before,
        "auc_after": auc_after,
        "brier_before": brier_before,
        "brier_after": brier_after
    })

    out_path = os.path.join(output_dir, f"{disease_name}_calibrated.csv")
    df.to_csv(out_path, index=False)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

print("\nAll diseases processed!")