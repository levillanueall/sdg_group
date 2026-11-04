from numbers import Number

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from src.config.model_config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    EXCLUDED_COLUMNS
)


class ChurnModelTrainer:

    def __init__(self):

        self.feature_columns = FEATURE_COLUMNS
        self.target_column = TARGET_COLUMN

        self.model = None

    def train(self, dataframe: pd.DataFrame) -> dict:

        # Seleccionar variables según la configuración global
        if FEATURE_COLUMNS is None:

            selected_columns = [
                col for col in dataframe.columns
                if col not in EXCLUDED_COLUMNS
            ]

        else:

            selected_columns = [
                col for col in FEATURE_COLUMNS
                if col not in EXCLUDED_COLUMNS
            ]

        if not selected_columns:
            raise ValueError("No hay variables predictoras.")

        missing_columns = [
            col for col in selected_columns
            if col not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Variables inexistentes: {missing_columns}"
            )

        # Variables predictoras
        X = dataframe[selected_columns].copy()

        # Variable objetivo
        y = pd.to_numeric(
            dataframe[TARGET_COLUMN],
            errors="raise"
        ).astype(int)

        if not y.isin([0, 1]).all():
            raise ValueError("churn debe contener únicamente 0 y 1.")

        if X.empty or len(X.columns) == 0:
            raise ValueError("No hay variables predictoras.")

        # Identificar variables numéricas y categóricas.
        numeric_columns = []
        categorical_columns = []

        for column in X.columns:

            non_null = X[column].dropna()

            is_numeric = (
                pd.api.types.is_numeric_dtype(X[column])
                or (
                    not non_null.empty
                    and isinstance(non_null.iloc[0], Number)
                )
            )

            if is_numeric:
                numeric_columns.append(column)

                X[column] = pd.to_numeric(
                    X[column],
                    errors="coerce"
                )

            else:
                categorical_columns.append(column)

                X[column] = (
                    X[column]
                    .astype("string")
                    .str.strip()
                    .replace("", pd.NA)
                    .fillna("MISSING")
                    .astype(str)
                )

        # Preprocesamiento numérico.
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Preprocesamiento según tipo de variable.
        transformers = []

        if numeric_columns:
            transformers.append((
                "numeric",
                numeric_transformer,
                numeric_columns
            ))

        if categorical_columns:
            transformers.append((
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=30
                ),
                categorical_columns
            ))

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        # Pipeline completo.
        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ])

        # División entrenamiento / prueba.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Entrenamiento.
        self.model.fit(X_train, y_train)

        # Predicciones.
        predictions = self.model.predict(X_test)

        probabilities = self.model.predict_proba(X_test)[:, 1]

        # Evaluación.
        metrics = {
            "roc_auc": roc_auc_score(y_test, probabilities),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            )
        }

        self.feature_columns = X.columns.tolist()

        return {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "features": self.feature_columns,
            "numeric_features": numeric_columns,
            "categorical_features": categorical_columns,
            "metrics": metrics
        }