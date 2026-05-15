
import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import prepare_data


RANDOM_STATE = 42


def get_models():
    """
    Define los modelos evaluados en el proyecto.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE
        )
    }

    return models


def train_and_compare(data_path="data/dataset.csv"):
    """
    Entrena varios modelos, compara resultados mediante validación cruzada
    y guarda el mejor modelo.
    """
    X, y, preprocessor, numeric_features, categorical_features = prepare_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    models = get_models()

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=5,
            scoring=scoring,
            n_jobs=-1
        )

        pipe.fit(X_train, y_train)
        trained_models[model_name] = pipe

        results.append({
            "Model": model_name,
            "CV Accuracy": cv_scores["test_accuracy"].mean(),
            "CV Precision": cv_scores["test_precision"].mean(),
            "CV Recall": cv_scores["test_recall"].mean(),
            "CV F1": cv_scores["test_f1"].mean(),
            "CV ROC-AUC": cv_scores["test_roc_auc"].mean()
        })

    cv_results_df = pd.DataFrame(results).sort_values(
        by="CV F1",
        ascending=False
    )

    # Modelo seleccionado para el estudio final.
# Aunque Logistic Regression puede obtener una métrica ligeramente superior,
# se selecciona Gradient Boosting por su buen rendimiento y su utilidad para XAI con SHAP.
    best_model_name = "Gradient Boosting"
    best_model = trained_models[best_model_name]
    
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    joblib.dump(best_model, "results/models/best_model.pkl")
    joblib.dump(best_model_name, "results/models/best_model_name.pkl")

    joblib.dump(X_train, "results/models/X_train.pkl")
    joblib.dump(X_test, "results/models/X_test.pkl")
    joblib.dump(y_train, "results/models/y_train.pkl")
    joblib.dump(y_test, "results/models/y_test.pkl")

    joblib.dump(numeric_features, "results/models/numeric_features.pkl")
    joblib.dump(categorical_features, "results/models/categorical_features.pkl")

    cv_results_df.to_csv("results/tables/cv_results_from_script.csv", index=False)

    print("Modelo seleccionado:", best_model_name)
    print(cv_results_df)

    return best_model, best_model_name, cv_results_df


if __name__ == "__main__":
    train_and_compare()