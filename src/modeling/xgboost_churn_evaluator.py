
import pandas as pd

from src.config.model_config import (
    TARGET_COLUMN,
    EXCLUDED_COLUMNS
)

from src.modeling.churn_model import (
    ChurnModelTrainer,
    ChurnMetrics
)

class XGBoostChurnEvaluator:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None

    def evaluate(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame
    ) -> dict:

        # 1. Identificar variables predictoras
        features = [
            column for column in train.columns
            if column not in EXCLUDED_COLUMNS
            and column != TARGET_COLUMN
        ]

        if not features:
            raise ValueError(
                "No se encontraron variables predictoras."
            )

        # 2. Validar columnas
        missing = [
            column
            for column in features + [TARGET_COLUMN]
            if column not in validation.columns
        ]

        if missing:
            raise ValueError(
                f"Columnas faltantes: {missing}"
            )

        # 3. Separar X e y
        X_train = train[features].copy()
        X_validation = validation[features].copy()

        y_train = pd.to_numeric(
            train[TARGET_COLUMN],
            errors="raise"
        )

        y_validation = pd.to_numeric(
            validation[TARGET_COLUMN],
            errors="raise"
        )

        for y in (y_train, y_validation):
            if y.isna().any() or not y.isin([0, 1]).all():
                raise ValueError(
                    "churn debe contener únicamente 0 y 1."
                )

        y_train = y_train.astype(int)
        y_validation = y_validation.astype(int)


        # 4. Construir modelo usando el constructor centralizado
        self.model = ChurnModelTrainer.build_pipeline(
            algorithm="xgboost",
            random_state=self.random_state
        )


        # 5. Entrenar
        self.model.fit(X_train, y_train)

        # 6. Predecir
        probabilities = self.model.predict_proba(
            X_validation
        )[:, 1]

        predictions = self.model.predict(
            X_validation
        )

        
        # 7. Calcular métricas compartidas
        metrics = ChurnMetrics.calculate(
            y_true=y_validation,
            predictions=predictions,
            probabilities=probabilities
        )

        # 8. Consolidar resultados
        return {
            "experiment": "xgboost_98",
            "n_features": len(features),
            "train_rows": len(train),
            "validation_rows": len(validation),
            **metrics
        }

