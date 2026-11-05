
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from src.config.model_config import (
    TARGET_COLUMN,
    EXCLUDED_COLUMNS
)

from src.modeling.correlation_feature_selector import (
    CorrelationFeatureSelector
)

from src.modeling.feature_selection_evaluator import (
    FeatureSelectionEvaluator
)

from src.modeling.xgboost_churn_evaluator import (
    XGBoostChurnEvaluator
)

class CrossValidationFeatureEvaluator:

    def __init__(
        self,
        n_splits: int = 5,
        threshold: float = 0.95,
        random_state: int = 42
    ):
        if n_splits < 2:
            raise ValueError(
                "n_splits debe ser al menos 2."
            )

        self.n_splits = n_splits
        self.threshold = threshold
        self.random_state = random_state

    def evaluate(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        # 1. Validar la variable objetivo
        if TARGET_COLUMN not in dataframe.columns:
            raise ValueError(
                "No se encontró la variable churn."
            )

        y = pd.to_numeric(
            dataframe[TARGET_COLUMN],
            errors="raise"
        ).astype(int)

        if not y.isin([0, 1]).all():
            raise ValueError(
                "churn debe contener únicamente 0 y 1."
            )

        if y.value_counts().min() < self.n_splits:
            raise ValueError(
                "No hay suficientes registros por clase."
            )

        # 2. Configurar validación cruzada
        cross_validator = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        baseline_features = [
            column for column in dataframe.columns
            if column not in EXCLUDED_COLUMNS
        ]

        results = []

        # 3. Ejecutar los folds
        for fold, (train_idx, val_idx) in enumerate(
            cross_validator.split(dataframe, y),
            start=1
        ):

            train = dataframe.iloc[
                train_idx
            ].reset_index(drop=True)

            validation = dataframe.iloc[
                val_idx
            ].reset_index(drop=True)

            print(f"\nProcesando fold {fold}...")

            # 4. Seleccionar variables solo en Train
            selector = CorrelationFeatureSelector(
                threshold=self.threshold
            )

            selection = selector.analyze(train)

            selected_features = selection[
                "selected_features"
            ]

            removed_features = selection[
                "candidates_to_remove"
            ]

            # 5. Comparar ambos modelos
            evaluator = FeatureSelectionEvaluator(
                random_state=self.random_state
            )

            fold_results = evaluator.evaluate(
                train=train,
                validation=validation,
                feature_sets={
                    "baseline": baseline_features,
                    "correlation_095": selected_features
                }
            )

            fold_results.insert(
                0, "fold", fold
            )

            
            fold_results["candidates_to_remove"] = (
                len(removed_features)
            )

            # 6. Entrenar XGBoost con las 98 variables
            xgboost_evaluator = XGBoostChurnEvaluator(
                random_state=self.random_state
            )

            xgboost_results = xgboost_evaluator.evaluate(
                train=train,
                validation=validation
            )

            # Agregar información del fold
            xgboost_results["fold"] = fold
            xgboost_results["candidates_to_remove"] = 0

            # Integrar XGBoost a la comparación
            fold_results = pd.concat(
                [
                    fold_results,
                    pd.DataFrame([xgboost_results])
                ],
                ignore_index=True
            )

            results.append(fold_results)


        # 6. Consolidar resultados
        return pd.concat(
            results,
            ignore_index=True
        )
