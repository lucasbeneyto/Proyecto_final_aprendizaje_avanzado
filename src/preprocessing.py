import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(data_path):
    """
    Carga el dataset desde un archivo CSV.
    """
    return pd.read_csv(data_path, sep=",")


def create_binary_target(df, target_col="Target"):
    """
    Convierte el problema original multiclase en un problema binario:
    Dropout = 1
    No Dropout = 0

    Se consideran como No Dropout las clases Graduate y Enrolled.
    """
    df = df.copy()
    df["Target_bin"] = (df[target_col] == "Dropout").astype(int)
    return df


def split_features_target(df, target_col="Target", binary_target_col="Target_bin"):
    """
    Separa variables predictoras y variable objetivo.
    Elimina la columna Target original para evitar data leakage.
    """
    X = df.drop(columns=[target_col, binary_target_col])
    y = df[binary_target_col]

    return X, y


def get_feature_types(X):
    """
    Define variables numéricas y categóricas.

    Aunque muchas variables estén codificadas como números,
    algunas representan categorías. Esta separación permite
    aplicar escalado a numéricas y one-hot encoding a categóricas.
    """
    categorical_features = [
        "Marital status",
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
        "International"
    ]

    categorical_features = [
        col for col in categorical_features if col in X.columns
    ]

    numeric_features = [
        col for col in X.columns if col not in categorical_features
    ]

    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """
    Construye el preprocesador común para los modelos:
    - StandardScaler para variables numéricas.
    - OneHotEncoder para variables categóricas.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor


def prepare_data(data_path):
    """
    Función completa de preparación de datos.
    """
    df = load_data(data_path)
    df = create_binary_target(df)

    X, y = split_features_target(df)
    numeric_features, categorical_features = get_feature_types(X)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    return X, y, preprocessor, numeric_features, categorical_features
