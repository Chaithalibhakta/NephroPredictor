
import pandas as pd

# Original Random Forest confusion matrix:
# [[TN, FP],
#  [FN, TP]]
rf_tn, rf_fp = 52953, 1
rf_fn, rf_tp = 75, 52879

# Original XGBoost confusion matrix
xgb_tn, xgb_fp = 52850, 104
xgb_fn, xgb_tp = 119, 52835


def calculate_metrics(tn, fp, fn, tp, accuracy, roc_auc):
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-score": f1 * 100,
        "ROC-AUC": roc_auc,
    }


results = pd.DataFrame([
    {
        "Model": "Random Forest",
        **calculate_metrics(
            rf_tn, rf_fp, rf_fn, rf_tp,
            accuracy=99.93,
            roc_auc=99.99
        )
    },
    {
        "Model": "XGBoost",
        **calculate_metrics(
            xgb_tn, xgb_fp, xgb_fn, xgb_tp,
            accuracy=99.79,
            roc_auc=100.00
        )
    }
])

print("\nNEPHROPREDICTOR - ORIGINAL RECORDED METRICS")
print("=" * 85)

for _, row in results.iterrows():
    print(f"\n{row['Model']}")
    print("-" * 40)

    for metric in [
        "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"
    ]:
        print(f"{metric:<12}: {row[metric]:.2f}%")

# Save a copy of the calculated metrics.
results.to_csv(
    "classification/model_comparison_metrics.csv",
    index=False
)

print("\nMetrics saved to:")
print("classification/model_comparison_metrics.csv")

print("\nNo models were trained or modified by this script.")

