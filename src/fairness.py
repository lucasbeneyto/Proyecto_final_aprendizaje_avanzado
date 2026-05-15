import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from scipy import sparse


def explain_model():
    """
    Genera explicabilidad global del modelo final usando SHAP.
    """
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/xai", exist_ok=True)

    best_model = joblib.load("results/models/best_model.pkl")
    best_model_name = joblib.load("results/models/best_model_name.pkl")

    X_train = joblib.load("results/models/X_train.pkl")
    X_test = joblib.load("results/models/X_test.pkl")

    preprocessor = best_model.named_steps["preprocessor"]
    model = best_model.named_steps["model"]

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if sparse.issparse(X_train_transformed):
        X_train_transformed = X_train_transformed.toarray()

    if sparse.issparse(X_test_transformed):
        X_test_transformed = X_test_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    feature_names_clean = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names
    ]

    X_train_shap = pd.DataFrame(
        X_train_transformed,
        columns=feature_names_clean,
        index=X_train.index
    )

    X_test_shap = pd.DataFrame(
        X_test_transformed,
        columns=feature_names_clean,
        index=X_test.index
    )

    sample_size = min(500, X_test_shap.shape[0])
    background_size = min(200, X_train_shap.shape[0])

    X_explain = X_test_shap.sample(
        n=sample_size,
        random_state=42
    )

    X_background = X_train_shap.sample(
        n=background_size,
        random_state=42
    )

    explainer = shap.TreeExplainer(
        model,
        data=X_background,
        feature_names=X_train_shap.columns
    )

    shap_values_array = explainer.shap_values(
        X_explain,
        check_additivity=False
    )

    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)

    shap_importance = pd.DataFrame({
        "feature": X_explain.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)

    shap_importance.to_csv(
        "results/tables/shap_feature_importance_from_script.csv",
        index=False
    )

    top_n = 20

    plt.figure(figsize=(10, 7))
    plt.barh(
        shap_importance.head(top_n)["feature"][::-1],
        shap_importance.head(top_n)["mean_abs_shap"][::-1]
    )
    plt.xlabel("Importancia media absoluta SHAP")
    plt.title(f"Top variables más influyentes según SHAP - {best_model_name}")
    plt.tight_layout()
    plt.savefig("results/figures/shap_global_importance_from_script.png", dpi=300)
    plt.close()

    shap.summary_plot(
        shap_values_array,
        X_explain,
        feature_names=X_explain.columns,
        show=False
    )
    plt.tight_layout()
    plt.savefig("results/figures/shap_summary_plot_from_script.png", dpi=300)
    plt.close()

    joblib.dump(shap_importance, "results/xai/shap_importance_from_script.pkl")

    print("Explicabilidad generada para:", best_model_name)
    print(shap_importance.head(20))

    return shap_importance


if __name__ == "__main__":
    explain_model()