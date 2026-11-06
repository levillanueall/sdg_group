
from pathlib import Path
import subprocess
import sys

import pendulum
from airflow.sdk import dag, task


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
        """
        Reutiliza el entrenamiento y el registro
        en MLflow existentes en main.py.
        """
        subprocess.run(
            [sys.executable, "-X", "utf8", "main.py"],
            cwd=str(PROJECT_DIR),
            check=True
        )

    # Dependencias entre tareas
    (
        bronze_customers()
        >> bronze_descriptions()
        >> silver_ingestion()
        >> validate_silver()
        >> gold_ingestion()
        >> validate_gold()
        >> train_and_track()
    )


sdg_churn_pipeline()
