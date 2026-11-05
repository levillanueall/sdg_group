
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
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

from src.modeling.churn_preprocessor import (
    ChurnPreprocessor
)

from xgboost import XGBClassifier


class ChurnModelTrainer:

    def __init__(self):

        self.feature_columns = FEATURE_COLUMNS
        self.target_column = TARGET_COLUMN
        self.model = None

    
    @staticmethod
    def build_pipeline(
        algorithm: str = "logistic_regression",
        random_state: int = 42,
        model_params: dict | None = None
    ) -> Pipeline:

        # Parámetros adicionales del experimento
        overrides = dict(model_params or {})

        # 1. Seleccionar algoritmo
        if algorithm == "logistic_regression":

            params = {
                "max_iter": 1000,
                "random_state": random_state
            }

            params.update(overrides)

            classifier = LogisticRegression(**params)

        elif algorithm == "xgboost":

            params = {
                "objective": "binary:logistic",
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "random_state": random_state,
                "n_jobs": 4
            }

            params.update(overrides)

            classifier = XGBClassifier(**params)

        else:
            raise ValueError(
                f"Algoritmo no soportado: {algorithm}"
            )

        # 2. Construir pipeline compartido
        return Pipeline([
            (
                "preprocessor",
                ChurnPreprocessor()
            ),
            (
                "classifier",
                classifier
            )
        ])


    
    
    def train(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        algorithm: str = "logistic_regression",
        model_params: dict | None = None
    ) -> dict:


        # 1. Seleccionar variables predictoras
        if self.feature_columns is None:
            selected_columns = [
                col for col in train.columns
                if col not in EXCLUDED_COLUMNS
                and col != self.target_column
            ]
        else:
            selected_columns = [
                col for col in self.feature_columns
                if col not in EXCLUDED_COLUMNS
                and col != self.target_column
            ]

        if not selected_columns:
            raise ValueError(
                "No existen variables predictoras."
            )

        # 2. Validar columnas
        required_columns = (
            selected_columns + [self.target_column]
        )

        for name, dataframe in (
            ("Train", train),
            ("Validation", validation)
        ):
            missing = [
                col for col in required_columns
                if col not in dataframe.columns
            ]

            if missing:
                raise ValueError(
                    f"Columnas faltantes en {name}: {missing}"
                )

        # 3. Separar X e y
        X_train = train[selected_columns].copy()
        X_validation = validation[selected_columns].copy()

        y_train = pd.to_numeric(
            train[self.target_column],
            errors="raise"
        )

        y_validation = pd.to_numeric(
            validation[self.target_column],
            errors="raise"
        )

        for name, y in (
            ("Train", y_train),
            ("Validation", y_validation)
        ):
            if y.isna().any() or not y.isin([0, 1]).all():
                raise ValueError(
                    f"Target invalido en {name}."
                )

            if y.nunique() != 2:
                raise ValueError(
                    f"Se requieren ambas clases en {name}."
                )

        y_train = y_train.astype(int)
        y_validation = y_validation.astype(int)

        
        # 4. Construir pipeline segun el algoritmo solicitado
        self.model = self.build_pipeline(
            algorithm=algorithm,
            random_state=42,
            model_params=model_params
        )


        # 5. Entrenar exclusivamente con Train
        self.model.fit(X_train, y_train)

        # 6. Predecir sobre Validation
        predictions = self.model.predict(
            X_validation
        )

        probabilities = self.model.predict_proba(
            X_validation
        )[:, 1]

        # 7. Calcular métricas compartidas
        metrics = ChurnMetrics.calculate(
            y_true=y_validation,
            predictions=predictions,
            probabilities=probabilities
        )

        # 8. Recuperar información del preprocesamiento
        preprocessor = self.model.named_steps[
            "preprocessor"
        ]

        self.feature_columns = selected_columns

        # 9. Consolidar resultados
        return {
            "algorithm": algorithm,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "features": self.feature_columns,
            "numeric_features": preprocessor.numeric_columns_,
            "categorical_features": preprocessor.categorical_columns_,
            "metrics": metrics
        }




class ChurnMetrics:

    @staticmethod
    def calculate(
        y_true,
        predictions,
        probabilities
    ) -> dict:

        y_true = np.asarray(y_true)
        predictions = np.asarray(predictions)
        probabilities = np.asarray(probabilities)

        # Validar dimensiones
        if any(
            values.ndim != 1
            for values in (
                y_true,
                predictions,
                probabilities
            )
        ):
            raise ValueError(
                "Las entradas deben ser unidimensionales."
            )

        if not (
            len(y_true)
            == len(predictions)
            == len(probabilities)
        ):
            raise ValueError(
                "Las predicciones no coinciden "
                "con la cantidad de registros."
            )

        # Calcular métricas
        return {
            "roc_auc": float(
                roc_auc_score(y_true, probabilities)
            ),
            "precision": float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            )
        }
