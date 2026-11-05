
import os

from pathlib import Path

import mlflow
import mlflow.sklearn


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
        run_name: str | None = None
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
            model_name="churn_all_features"
        )
