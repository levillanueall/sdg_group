
import os

from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

class MLflowExperimentTracker:

    
    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None
    ):

        self.tracking_uri = (
            tracking_uri
            or os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000"
            )
        )

        self.experiment_name = (
            experiment_name
            or os.getenv(
                "MLFLOW_EXPERIMENT_NAME",
                "SDG_Churn_Model_Comparison"
            )
        )

    def log_experiment(
        self,
        run_name: str,
        params: dict | None = None,
        metrics: dict | None = None,
        model=None,
        artifacts: list | None = None,
        tags: dict | None = None,
        model_name: str = "churn_model"
    ) -> dict:

        # 1. Validar los artefactos
        artifact_paths = [
            Path(path) for path in (artifacts or [])
        ]

        for path in artifact_paths:
            if not path.is_file():
                raise FileNotFoundError(
                    f"No existe el artefacto: {path}"
                )


        # Configurar servidor y experimento
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        # 2. Crear una ejecucion en MLflow
        with mlflow.start_run(run_name=run_name) as run:

            # 3. Registrar parametros
            if params:
                mlflow.log_params(params)

            # 4. Registrar metricas
            if metrics:
                mlflow.log_metrics({
                    key: float(value)
                    for key, value in metrics.items()
                })

            # 5. Registrar archivos
            for path in artifact_paths:
                mlflow.log_artifact(
                    str(path),
                    artifact_path="reports"
                )

            # 6. Registrar modelo (opcional)
            model_uri = None

            if model is not None:
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name=model_name
                )

                model_uri = model_info.model_uri

            # 7. Registrar etiquetas
            if tags:
                mlflow.set_tags(tags)

            return {
                "run_id": run.info.run_id,
                "model_uri": model_uri,
                "status": "success"
            }


    
    
    def log_training(
        self,
        training_result: dict,
        model,
        run_name: str | None = None,
        tags: dict | None = None
    ) -> dict:



        # 1. Identificar el algoritmo entrenado
        algorithm = training_result["algorithm"]

        # 2. Construir los parametros generales
        params = {
            "algorithm": algorithm,
            "train_rows": training_result["train_rows"],
            "validation_rows": training_result["validation_rows"],
            "features": ",".join(
                training_result["features"]
            ),
            "imputer": "median",
            "scaler": "StandardScaler"
        }

        # 3. Obtener el clasificador entrenado
        classifier = model.named_steps["classifier"]

        # 4. Recuperar sus hiperparametros efectivos
        classifier_params = classifier.get_params(
            deep=False
        )

        # 5. Agregar los hiperparametros a MLflow
        params.update({
            f"classifier__{name}": value
            for name, value in classifier_params.items()
            if value is not None
            and isinstance(
                value,
                (str, int, float, bool)
            )
        })

        # 6. Registrar el experimento
        
        return self.log_experiment(
            run_name=run_name or f"{algorithm}_all_features",
            params=params,
            metrics=training_result["metrics"],
            model=model,
            tags=tags,
            model_name="churn_all_features"
        )


    
    def select_best_model(self, run_group_id: str) -> dict:

        mlflow.set_tracking_uri(self.tracking_uri)

        client = MlflowClient()

        experiment = client.get_experiment_by_name(
            self.experiment_name
        )

        if experiment is None:
            raise ValueError(
                f"Experimento inexistente: {self.experiment_name}"
            )

        # Consultar solo los candidatos de esta ejecución
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                f"tags.run_group_id = '{run_group_id}' "
                "AND tags.split_version = "
                "'customer_id_stratified_v1' "
                "AND tags.evaluation_role = 'candidate'"
            ),
            max_results=1000
        )

        # Considerar únicamente ejecuciones exitosas
        candidates = [
            run for run in runs
            if run.info.status == "FINISHED"
            and "roc_auc" in run.data.metrics
        ]

        expected_models = {
            "Logistic Regression - Baseline",
            "Logistic Regression - Feature Engineering",
            "XGBoost Optimizado - Baseline",
            "XGBoost Optimizado - Feature Engineering"
        }

        available_models = {
            run.info.run_name for run in candidates
        }

        missing_models = expected_models - available_models

        if missing_models:
            raise ValueError(
                f"Modelos candidatos faltantes: {missing_models}"
            )

        # Seleccionar el mejor ROC-AUC de Validation
        best_run = max(
            candidates,
            key=lambda run: run.data.metrics["roc_auc"]
        )

        # Recuperar el modelo almacenado en MLflow
        logged_models = client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                f"source_run_id = '{best_run.info.run_id}'"
            ),
            max_results=10
        )

        if not logged_models:
            raise ValueError(
                "El modelo ganador no está almacenado en MLflow."
            )

        model_uri = f"models:/{logged_models[0].model_id}"

        return {
            "run_group_id": run_group_id,
            "run_id": best_run.info.run_id,
            "model_name": best_run.info.run_name,
            "model_uri": model_uri,
            "validation_roc_auc": best_run.data.metrics["roc_auc"],
            "features": best_run.data.params["features"].split(",")
        }

    
    def load_model(self, model_uri: str):

        if not model_uri:
            raise ValueError(
                "No se proporcionó el URI del modelo."
            )

        # Configurar conexión a MLflow
        mlflow.set_tracking_uri(self.tracking_uri)

        # Recuperar el pipeline entrenado
        model = mlflow.sklearn.load_model(
            model_uri
        )

        return model

