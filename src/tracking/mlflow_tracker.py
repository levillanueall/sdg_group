import mlflow
import mlflow.sklearn


class MLflowExperimentTracker:

    def log_training(
        self,
        training_result: dict,
        model
    ) -> dict:

        with mlflow.start_run(
            run_name="logistic_regression_all_features"
        ) as run:

            # Registrar parámetros del experimento
            mlflow.log_params({
                "algorithm": "LogisticRegression",
                "train_rows": training_result["train_rows"],
                "test_rows": training_result["test_rows"],
                "features": ",".join(training_result["features"]),
                "imputer": "median",
                "scaler": "StandardScaler"
            })

            # Registrar las métricas obtenidas
            mlflow.log_metrics(
                training_result["metrics"]
            )

            # Registrar el Pipeline entrenado
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="churn_all_features"
            )

            return {
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "status": "success"
            }