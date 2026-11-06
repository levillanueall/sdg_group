
from pathlib import Path
import subprocess
import sys

import pendulum
from airflow.sdk import dag, task

import os
from airflow.sdk import dag, task, get_current_context


PROJECT_DIR = Path("/opt/airflow/project")


def get_components():
    """
    Reutiliza los componentes y el orquestador
    existentes en main.py.
    """
    project_path = str(PROJECT_DIR)

    if project_path not in sys.path:
        sys.path.insert(0, project_path)

    from main import main

    return main()


@dag(
    dag_id="sdg_churn_pipeline",
    schedule=None,
    start_date=pendulum.datetime(
        2026, 11, 5, tz="America/Mexico_City"
    ),
    catchup=False,
    max_active_runs=1,
    tags=["sdg", "churn", "mlops"]
)
def sdg_churn_pipeline():

    @task
    def bronze_customers():
        orchestrator, *_ = get_components()

        result = orchestrator.run(
            stage="bronze",
            file_path=str(PROJECT_DIR / "data/dataset.csv"),
            table_name="customers_raw"
        )

        print(result)

    @task
    def bronze_descriptions():
        orchestrator, *_ = get_components()

        result = orchestrator.run(
            stage="bronze",
            file_path=str(
                PROJECT_DIR / "data/data_descriptions.csv"
            ),
            table_name="data_descriptions"
        )

        print(result)

    @task
    def silver_ingestion():
        (
            orchestrator,
            silver_transformation,
            *_,
        ) = get_components()

        column_types = (
            silver_transformation.build_column_types()
        )

        transformed_batches = (
            silver_transformation.transform_batches(
                batch_size=1000
            )
        )

        result = orchestrator.run(
            stage="silver",
            transformed_batches=transformed_batches,
            column_types=column_types,
            schema="silver",
            table_name="customers"
        )

        print(result)

    @task
    def validate_silver():
        (
            orchestrator,
            silver_transformation,
            *_,
        ) = get_components()

        column_types = (
            silver_transformation.build_column_types()
        )

        validations = [
            (
                "structure",
                orchestrator.run(
                    stage="validate_silver",
                    column_types=column_types
                )
            ),
            (
                "identifiers",
                orchestrator.run(
                    stage="validate_silver_ids"
                )
            ),
            (
                "values",
                orchestrator.run(
                    stage="validate_silver_values",
                    column_types=column_types,
                    batch_size=1000,
                    max_examples=10
                )
            )
        ]

        for name, result in validations:
            print(name, result["is_valid"])

            if not result["is_valid"]:
                raise RuntimeError(
                    f"Validacion Silver fallida: {name}"
                )

    @task
    def gold_ingestion():
        orchestrator, *_ = get_components()

        result = orchestrator.run(
            stage="gold"
        )

        print(result)

    @task
    def validate_gold():
        orchestrator, *_ = get_components()

        result = orchestrator.run(
            stage="validate_gold"
        )

        print(result)

        if not result["is_valid"]:
            raise RuntimeError(
                "Gold no supero las validaciones"
            )

    
    @task
    def train_and_track():

        # Identificador único de esta ejecución del DAG
        context = get_current_context()
        run_group_id = context["ti"].run_id

        # Compartir identificador con main.py
        env = os.environ.copy()
        env["SDG_RUN_GROUP_ID"] = run_group_id

        print(f"Grupo MLflow: {run_group_id}")

        # Ejecutar entrenamiento existente
        subprocess.run(
            [sys.executable, "-X", "utf8", "main.py"],
            cwd=str(PROJECT_DIR),
            env=env,
            check=True
        )

        return run_group_id

    
    @task
    def select_best_model(run_group_id: str):

        orchestrator, *_ = get_components()

        best_model = orchestrator.run(
            stage="select_best_model",
            run_group_id=run_group_id
        )

        print("\n--- MEJOR MODELO SELECCIONADO ---")
        print(f"Modelo: {best_model['model_name']}")
        print(f"Run ID: {best_model['run_id']}")
        print(f"Model URI: {best_model['model_uri']}")
        print(
            f"ROC-AUC Validation: "
            f"{best_model['validation_roc_auc']:.6f}"
        )

        return best_model

    
    @task
    def evaluate_test(best_model: dict):

        # Verificar que el modelo pertenece a esta ejecución
        context = get_current_context()
        run_group_id = context["ti"].run_id

        if best_model["run_group_id"] != run_group_id:
            raise ValueError(
                "El modelo seleccionado no pertenece "
                "a la ejecución actual de Airflow."
            )

        orchestrator, *_ = get_components()

        # 1. Recuperar Gold
        gold_dataset = orchestrator.run(
            stage="load_gold",
            batch_size=1000
        )

        # 2. Reconstruir las mismas particiones
        datasets = orchestrator.run(
            stage="split_data",
            dataframe=gold_dataset
        )

        test_dataset = datasets["test"]

        # 3. Recuperar el modelo ganador desde MLflow
        model = orchestrator.run(
            stage="load_model",
            model_uri=best_model["model_uri"]
        )

        # 4. Evaluación final sin reentrenamiento
        result = orchestrator.run(
            stage="evaluate_test",
            model=model,
            test=test_dataset,
            features=best_model["features"]
        )

        # 5. Mostrar resultados en los logs de Airflow
        print("\n--- EVALUACION FINAL TEST ---")
        print(f"Modelo: {best_model['model_name']}")
        print(f"Run ID: {best_model['run_id']}")
        print(f"Registros Test: {result['test_rows']}")
        print(
            f"ROC-AUC Validation: "
            f"{best_model['validation_roc_auc']:.6f}"
        )

        for metric, value in result["metrics"].items():
            print(f"Test {metric}: {value:.6f}")

        return {
            "model_name": best_model["model_name"],
            "run_id": best_model["run_id"],
            "test_rows": result["test_rows"],
            "metrics": result["metrics"]
        }



    
    
    # Dependencias entre tareas

    training_task = train_and_track()

    best_model_task = select_best_model(
        training_task
    )

    evaluate_test(
        best_model_task
    )

    (
        bronze_customers()
        >> bronze_descriptions()
        >> silver_ingestion()
        >> validate_silver()
        >> gold_ingestion()
        >> validate_gold()
        >> training_task
    )



sdg_churn_pipeline()
