
import pandas as pd

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)

from src.config.model_config import (
    TARGET_COLUMN,
    EXCLUDED_COLUMNS
)

from src.modeling.churn_model import ChurnModelTrainer


class XGBoostHyperparameterTuner:

    def __init__(
        self,
        n_iter: int = 20,
        n_splits: int = 3,
        random_state: int = 42
    ):
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.random_state = random_state

        self.search = None
        self.best_model = None

    def tune(
        self,
        development: pd.DataFrame
    ) -> dict:

        # 1. Validar los datos
        if TARGET_COLUMN not in development.columns:
            raise ValueError(
                "No se encontró la variable churn."
            )

        features = [
            column for column in development.columns
            if column not in EXCLUDED_COLUMNS
        ]

        if not features:
            raise ValueError(
                "No existen variables predictoras."
            )

        X = development[features].copy()

        y = pd.to_numeric(
            development[TARGET_COLUMN],
            errors="raise"
        )

        if y.isna().any() or not y.isin([0, 1]).all():
            raise ValueError(
                "churn debe contener únicamente 0 y 1."
            )

        y = y.astype(int)

        if y.value_counts().min() < self.n_splits:
            raise ValueError(
                "No hay suficientes registros por clase."
            )

        
        # 2. Construir modelo usando el constructor centralizado
        model = ChurnModelTrainer.build_pipeline(
            algorithm="xgboost",
            random_state=self.random_state
        )



        # 5. Definir espacio de búsqueda
        param_distributions = {
            "classifier__n_estimators": [
                150, 250, 300, 450, 600
            ],
            "classifier__max_depth": [
                3, 4, 5, 6
            ],
            "classifier__learning_rate": [
                0.01, 0.03, 0.05, 0.1
            ],
            "classifier__min_child_weight": [
                1, 3, 5, 10
            ],
            "classifier__subsample": [
                0.7, 0.8, 0.9, 1.0
            ],
            "classifier__colsample_bytree": [
                0.7, 0.8, 0.9, 1.0
            ],
            "classifier__gamma": [
                0, 0.1, 0.5, 1.0
            ],
            "classifier__reg_lambda": [
                0.1, 1.0, 5.0, 10.0
            ]
        }

        # 6. Configurar validación cruzada
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        # 7. Configurar búsqueda aleatoria
        self.search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=self.random_state,
            n_jobs=1,
            verbose=2,
            refit=True,
            return_train_score=False,
            error_score="raise"
        )

        # 8. Ejecutar optimización
        self.search.fit(X, y)

        # 9. Recuperar mejor modelo
        self.best_model = self.search.best_estimator_

        # 10. Consolidar resultados
        cv_results = pd.DataFrame(
            self.search.cv_results_
        ).sort_values(
            by="rank_test_score"
        ).reset_index(drop=True)

        return {
            "best_params": self.search.best_params_,
            "best_cv_roc_auc": self.search.best_score_,
            "n_features": len(features),
            "n_iter": self.n_iter,
            "n_splits": self.n_splits,
            "cv_results": cv_results
        }
