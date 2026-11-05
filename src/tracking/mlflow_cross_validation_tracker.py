
from pathlib import Path

import mlflow
import pandas as pd


class MLflowCrossValidationTracker:

    def __init__(
        self,
        tracking_uri: str = "http://127.0.0.1:5000",
        experiment_name: str = "SDG_Churn_Model_Comparison"
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def log_from_csv(
        self,
        results_dir: str | Path
    ) -> list[dict]:

        results_dir = Path(results_dir)

        files = {
            "folds": results_dir / "cross_validation_folds.csv",
            "summary": results_dir / "cross_validation_summary.csv",
            "differences": (
                results_dir / "cross_validation_differences.csv"
            )
        }

        # 1. Validar los archivos
        for name, path in files.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"No existe el archivo {name}: {path}"
                )

        # 2. Leer resultados
        folds = pd.read_csv(files["folds"])
        summary = pd.read_csv(files["summary"])

        experiments = {
            "baseline": "LogisticRegression_L2",
            "correlation_095": "LogisticRegression_L2",
            "xgboost_98": "XGBClassifier"
        }

        # 3. Verificar resultados completos
        for experiment in experiments:

            summary_rows = summary[
                summary["experiment"] == experiment
            ]

            fold_rows = folds[
                folds["experiment"] == experiment
            ]

            if len(summary_rows) != 1:
                raise ValueError(
                    f"Resumen inválido: {experiment}"
                )

            if (
                len(fold_rows) != 5
                or fold_rows["fold"].nunique() != 5
            ):
                raise ValueError(
                    f"Folds incompletos: {experiment}"
                )

        # 4. Configurar MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        logged_runs = []

        # 5. Registrar cada experimento
        for experiment, algorithm in experiments.items():

            experiment_summary = summary[
                summary["experiment"] == experiment
            ].iloc[0]

            experiment_folds = folds[
                folds["experiment"] == experiment
            ].sort_values("fold")

            with mlflow.start_run(
                run_name=f"cv5_{experiment}"
            ) as run:

                # Parámetros generales
                params = {
                    "algorithm": algorithm,
                    "cv_method": "StratifiedKFold",
                    "n_splits": 5,
                    "random_state": 42,
                    "train_rows_per_fold": 64000,
                    "validation_rows_per_fold": 16000,
                    "test_set_used": False
                }

                if experiment == "correlation_095":
                    params.update({
                        "feature_selection": "pearson",
                        "correlation_threshold": 0.95
                    })

                if experiment == "xgboost_98":
                    params.update({
                        "n_estimators": 300,
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "tree_method": "hist"
                    })

                mlflow.log_params(params)

                # Métricas agregadas
                metrics = {
                    "cv_roc_auc_mean": float(
                        experiment_summary["roc_auc_mean"]
                    ),
                    "cv_roc_auc_std": float(
                        experiment_summary["roc_auc_std"]
                    ),
                    "cv_precision_mean": float(
                        experiment_summary["precision_mean"]
                    ),
                    "cv_recall_mean": float(
                        experiment_summary["recall_mean"]
                    ),
                    "cv_f1_mean": float(
                        experiment_summary["f1_mean"]
                    ),
                    "cv_n_features_mean": float(
                        experiment_summary["n_features_mean"]
                    )
                }

                mlflow.log_metrics(metrics)

                # Métricas individuales por fold
                for _, row in experiment_folds.iterrows():

                    fold = int(row["fold"])

                    for metric in [
                        "roc_auc",
                        "precision",
                        "recall",
                        "f1"
                    ]:
                        mlflow.log_metric(
                            f"fold_{metric}",
                            float(row[metric]),
                            step=fold
                        )

                # Adjuntar los resultados originales
                for path in files.values():
                    mlflow.log_artifact(
                        str(path),
                        artifact_path="reports"
                    )

                mlflow.set_tags({
                    "project": "sdg_churn",
                    "experiment_type": "model_comparison",
                    "source": "existing_cv_results",
                    "model_artifact_saved": "false"
                })

                logged_runs.append({
                    "experiment": experiment,
                    "run_id": run.info.run_id,
                    "roc_auc_mean": metrics[
                        "cv_roc_auc_mean"
                    ]
                })

        return logged_runs
