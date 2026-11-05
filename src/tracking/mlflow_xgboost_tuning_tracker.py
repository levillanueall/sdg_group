
import json
from math import isclose
from pathlib import Path

import mlflow
import pandas as pd


class MLflowXGBoostTuningTracker:

    def __init__(
        self,
        tracking_uri: str = "http://127.0.0.1:5000",
        experiment_name: str = "SDG_Churn_Model_Comparison"
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def log_from_files(
        self,
        results_dir: str | Path
    ) -> dict:

        # 1. Localizar los archivos
        results_dir = Path(results_dir)

        summary_path = (
            results_dir / "xgboost_tuning_summary.json"
        )

        cv_results_path = (
            results_dir / "xgboost_tuning_cv_results.csv"
        )

        for path in [summary_path, cv_results_path]:
            if not path.is_file():
                raise FileNotFoundError(
                    f"No se encontró: {path}"
                )

        # 2. Cargar los resultados
        with open(
            summary_path,
            "r",
            encoding="utf-8"
        ) as file:
            summary = json.load(file)

        cv_results = pd.read_csv(cv_results_path)

        # 3. Validar consistencia
        if len(cv_results) != summary["n_iter"]:
            raise ValueError(
                "El número de configuraciones no coincide."
            )

        best_rows = cv_results[
            cv_results["rank_test_score"] == 1
        ]

        if len(best_rows) != 1:
            raise ValueError(
                "No se identificó una única mejor configuración."
            )

        best_row = best_rows.iloc[0]

        best_auc = float(
            summary["best_cv_roc_auc"]
        )

        if not isclose(
            float(best_row["mean_test_score"]),
            best_auc,
            abs_tol=1e-8
        ):
            raise ValueError(
                "El mejor ROC-AUC no coincide entre archivos."
            )

        # 4. Configurar MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # 5. Registrar el experimento
        with mlflow.start_run(
            run_name="xgboost_randomized_search_20x3"
        ) as run:

            # Parámetros de la búsqueda
            search_params = {
                "algorithm": summary["algorithm"],
                "search_method": "RandomizedSearchCV",
                "scoring": summary["scoring"],
                "n_iter": summary["n_iter"],
                "n_splits": summary["n_splits"],
                "n_features": summary["n_features"],
                "development_rows": summary[
                    "development_rows"
                ],
                "reserved_test_rows": summary[
                    "reserved_test_rows"
                ],
                "random_state": 42,
                "test_set_used": False
            }

            mlflow.log_params(search_params)

            # Registrar mejores hiperparámetros
            best_params = {
                key.replace("classifier__", ""): value
                for key, value in
                summary["best_params"].items()
            }

            mlflow.log_params(best_params)

            # Métricas de la búsqueda
            mlflow.log_metrics({
                "tuning_best_cv_roc_auc": best_auc,
                "tuning_best_cv_roc_auc_std": float(
                    best_row["std_test_score"]
                ),
                "configurations_evaluated": len(
                    cv_results
                )
            })

            # Guardar archivos originales
            mlflow.log_artifact(
                str(summary_path),
                artifact_path="tuning"
            )

            mlflow.log_artifact(
                str(cv_results_path),
                artifact_path="tuning"
            )

            # Etiquetas para trazabilidad
            mlflow.set_tags({
                "project": "sdg_churn",
                "experiment_type": "hyperparameter_tuning",
                "validation": "inner_cv_3fold",
                "source": "existing_tuning_results",
                "model_artifact_saved": "false",
                "test_evaluated": "false"
            })

            return {
                "run_id": run.info.run_id,
                "best_cv_roc_auc": best_auc,
                "best_params": best_params
            }
