
import pandas as pd

from src.config.model_config import (
    TARGET_COLUMN,
    EXCLUDED_COLUMNS
)

from src.modeling.churn_model import (
    ChurnModelTrainer,
    ChurnMetrics
)

class FeatureSelectionEvaluator:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def evaluate(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        feature_sets: dict
    ) -> pd.DataFrame:

        # 1. Validar variable objetivo
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

        results = []

        # 2. Evaluar cada conjunto de variables
        for name, features in feature_sets.items():

            if not features or len(features) != len(set(features)):
                raise ValueError(
                    f"Variables vacías o duplicadas: {name}"
                )

            if set(features).intersection(EXCLUDED_COLUMNS):
                raise ValueError(
                    f"Variables excluidas en: {name}"
                )

            missing = [
                col for col in features
                if col not in train.columns
                or col not in validation.columns
            ]

            if missing:
                raise ValueError(
                    f"Variables faltantes en {name}: {missing}"
                )

            X_train = train[features].copy()
            X_validation = validation[features].copy()

            
            # 3. Construir modelo mediante el constructor centralizado
            model = ChurnModelTrainer.build_pipeline(
                algorithm="logistic_regression",
                random_state=self.random_state
            )


            # 4. Entrenar modelo
            model.fit(X_train, y_train)

            # 5. Generar predicciones
            predictions = model.predict(X_validation)

            probabilities = model.predict_proba(
                X_validation
            )[:, 1]

            
            # 6. Calcular metricas con componente compartido
            metrics = ChurnMetrics.calculate(
                y_true=y_validation,
                predictions=predictions,
                probabilities=probabilities
            )

            # 7. Consolidar resultados
            results.append({
                "experiment": name,
                "n_features": len(features),
                "train_rows": len(train),
                "validation_rows": len(validation),
                **metrics
            })


        return pd.DataFrame(results)
