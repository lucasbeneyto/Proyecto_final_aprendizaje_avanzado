import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


def evaluate_model():
    """
    Evalúa el modelo final guardado sobre el conjunto de test.
    """
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    best_model = joblib.load("results/models/best_model.pkl")
    best_model_name = joblib.load("results/models/best_model_name.pkl")

    X_test = joblib.load("results/models/X_test.pkl")
    y_test = joblib.load("results/models/y_test.pkl")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": best_model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("results/tables/final_model_metrics_from_script.csv", index=False)

    predictions_df = X_test.copy()
    predictions_df["y_real"] = y_test.values
    predictions_df["y_pred"] = y_pred
    predictions_df["y_proba_dropout"] = y_proba

    predictions_df.to_csv("results/tables/best_model_predictions_from_script.csv", index=False)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Dropout", "Dropout"],
        yticklabels=["No Dropout", "Dropout"]
    )
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión - {best_model_name}")
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix_from_script.png", dpi=300)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"Curva ROC - {best_model_name}")
    plt.tight_layout()
    plt.savefig("results/figures/roc_curve_from_script.png", dpi=300)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title(f"Curva Precision-Recall - {best_model_name}")
    plt.tight_layout()
    plt.savefig("results/figures/pr_curve_from_script.png", dpi=300)
    plt.close()

    print("Modelo evaluado:", best_model_name)
    print(metrics_df)
    print()
    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Dropout", "Dropout"]
    ))

    return metrics_df


if __name__ == "__main__":
    evaluate_model()
