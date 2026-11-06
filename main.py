
import os
import sys

from src.data.repositories.postgres_repository import PostgresRepository
from src.data.validators.data_validator import DataValidator
from src.data.ingestion.bronze_ingestion import BronzeIngestion
from src.data.ingestion.silver_ingestion import SilverIngestion
from src.transformations.silver_transformation import SilverTransformation
from src.data.validators.silver_validator import SilverValidator
from src.transformations.data_type_transformer import DataTypeTransformer
from src.orchestration.data_orchestrator import DataOrchestrator
from src.data.ingestion.gold_ingestion import GoldIngestion
from src.eda.churn_eda import ChurnEDA
from src.data.validators.gold_validator import GoldValidator
from src.data.readers.gold_reader import GoldDatasetReader
from src.modeling.churn_model import ChurnModelTrainer
from src.tracking.mlflow_tracker import MLflowExperimentTracker
from src.modeling.xgboost_hyperparameter_tuner import (
    XGBoostHyperparameterTuner
)
from src.modeling.data_splitter import ChurnDataSplitter
from src.config.model_config import FEATURE_TYPES


# ============================================================
# 1. INICIALIZACION DE COMPONENTES
# ============================================================

def main():

    # --------------------------------------------------------
    # BASE DE DATOS
    # --------------------------------------------------------

    
    repository = PostgresRepository(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=os.getenv("POSTGRES_DB", "sdg_churn"),
        user=os.getenv("POSTGRES_USER", "sdg_user"),
        password=os.getenv("POSTGRES_PASSWORD", "sdg_password")
    )


    # --------------------------------------------------------
    # BRONZE
    # --------------------------------------------------------

    validator = DataValidator()

    bronze_ingestion = BronzeIngestion(
        repository=repository,
        validator=validator
    )

    # --------------------------------------------------------
    # SILVER
    # --------------------------------------------------------

    transformer = DataTypeTransformer()

    silver_transformation = SilverTransformation(
        repository=repository,
        transformer=transformer
    )

    silver_ingestion = SilverIngestion(
        repository=repository
    )

    silver_validator = SilverValidator(
        repository=repository
    )

    # --------------------------------------------------------
    # GOLD
    # --------------------------------------------------------

    gold_ingestion = GoldIngestion(
        repository=repository
    )

    gold_validator = GoldValidator(
        repository=repository
    )

    gold_reader = GoldDatasetReader(
        repository=repository
    )

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    churn_eda = ChurnEDA(
        repository=repository
    )

    # --------------------------------------------------------
    # MODELADO
    # --------------------------------------------------------

    churn_model = ChurnModelTrainer()

    data_splitter = ChurnDataSplitter(
        test_size=0.20,
        validation_size=0.20,
        random_state=42
    )

    xgboost_tuner = XGBoostHyperparameterTuner(
        n_iter=20,
        n_splits=3,
        random_state=42
    )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    mlflow_tracker = MLflowExperimentTracker()

    # --------------------------------------------------------
    # DATA ORCHESTRATOR
    # --------------------------------------------------------

    def run_gold_pipeline():

        # Obtener los tipos de las columnas existentes
        column_types = silver_transformation.build_column_types()

        # Leer los datos de Silver
        gold_batches = repository.read_batches(
            schema="silver",
            table="customers",
            batch_size=1000
        )

        # Ejecutar Gold Ingestion
        return gold_ingestion.ingest(
            batches=gold_batches,
            column_types=column_types,
            table_name="customers"
        )

    data_orchestrator = DataOrchestrator(
        stages={

            # Ingestion
            "bronze": bronze_ingestion.ingest,
            "silver": silver_ingestion.ingest,
            "gold": run_gold_pipeline,

            # Validaciones
            "validate_silver": (
                silver_validator.validate_structure
            ),
            "validate_silver_ids": (
                silver_validator.validate_identifiers
            ),
            "validate_silver_values": (
                silver_validator.validate_values
            ),
            "validate_gold": (
                gold_validator.validate_structure
            ),

            # EDA
            "eda_target": churn_eda.analyze_target,
            "eda_missing": churn_eda.analyze_missing_values,

            # Lectura y separacion
            "load_gold": gold_reader.load,
            "split_data": data_splitter.split,

            # Modelado
            "train_model": churn_model.train,
            "tune_model": xgboost_tuner.tune,

            # MLflow
            "track_model": mlflow_tracker.log_training
        }
    )

    return (
        data_orchestrator,
        silver_transformation,
        repository,
        gold_reader,
        churn_model
    )


# ============================================================
# 2. PREPARACION DE DATOS
# ============================================================

def run_data_preparation(
    orchestrator,
    silver_transformation,
    repository
):
    """
    Ejecuta Bronze, Silver, Gold,
    validaciones y EDA.

    No se ejecuta automaticamente durante el modelado.
    """

    # --------------------------------------------------------
    # BRONZE - CLIENTES
    # --------------------------------------------------------

    bronze_result = orchestrator.run(
        stage="bronze",
        file_path="data/dataset.csv",
        table_name="customers_raw"
    )

    print(bronze_result)

    # --------------------------------------------------------
    # BRONZE - DESCRIPCIONES
    # --------------------------------------------------------

    descriptions_result = orchestrator.run(
        stage="bronze",
        file_path="data/data_descriptions.csv",
        table_name="data_descriptions"
    )

    print(descriptions_result)

    # --------------------------------------------------------
    # TRANSFORMACIONES SILVER
    # --------------------------------------------------------

    column_types = (
        silver_transformation.build_column_types()
    )

    transformed_batches = (
        silver_transformation.transform_batches(
            batch_size=1000
        )
    )

    # --------------------------------------------------------
    # SILVER INGESTION
    # --------------------------------------------------------

    silver_result = orchestrator.run(
        stage="silver",
        transformed_batches=transformed_batches,
        column_types=column_types,
        schema="silver",
        table_name="customers"
    )

    print(silver_result)

    # --------------------------------------------------------
    # GOLD INGESTION
    # --------------------------------------------------------

    gold_result = orchestrator.run(
        stage="gold"
    )

    print(gold_result)

    # --------------------------------------------------------
    # VALIDACIONES SILVER
    # --------------------------------------------------------

    validation_result = orchestrator.run(
        stage="validate_silver",
        column_types=column_types
    )

    print(validation_result)

    identifiers_result = orchestrator.run(
        stage="validate_silver_ids"
    )

    print(identifiers_result)

    values_result = orchestrator.run(
        stage="validate_silver_values",
        column_types=column_types,
        batch_size=1000,
        max_examples=10
    )

    print(values_result)

    # --------------------------------------------------------
    # VALIDACIONES GOLD
    # --------------------------------------------------------

    gold_validation_result = orchestrator.run(
        stage="validate_gold"
    )

    print(gold_validation_result)

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    target_result = orchestrator.run(
        stage="eda_target"
    )

    print(target_result)

    missing_result = orchestrator.run(
        stage="eda_missing"
    )

    for column_result in missing_result:
        print(column_result)


# ============================================================
# 3. EJECUCION PRINCIPAL - MODELADO
# ============================================================

if __name__ == "__main__":

    (
        orchestrator,
        silver_transformation,
        repository,
        gold_reader,
        churn_model
    ) = main()

    # --------------------------------------------------------
    # EJECUCION INDEPENDIENTE DE GOLD
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        if len(sys.argv) != 2 or sys.argv[1] != "gold":
            raise ValueError("Etapa no soportada")

        print("\n--- EJECUTANDO GOLD ---", flush=True)

        gold_result = orchestrator.run(
            stage="gold"
        )

        print(gold_result)

        validation_result = orchestrator.run(
            stage="validate_gold"
        )

        print(validation_result)

        if not validation_result["is_valid"]:
            raise RuntimeError(
                "Gold no superó las validaciones"
            )

        sys.exit(0)

    # --------------------------------------------------------
    # CARGAR GOLD
    # --------------------------------------------------------

    print("\n--- CARGANDO GOLD ---", flush=True)

    gold_dataset = orchestrator.run(
        stage="load_gold",
        batch_size=1000
    )

    print(
        f"Dataset de entrenamiento: {gold_dataset.shape}"
    )

    # --------------------------------------------------------
    # SEPARAR TRAIN, VALIDATION Y TEST
    # --------------------------------------------------------

    datasets = orchestrator.run(
        stage="split_data",
        dataframe=gold_dataset
    )

    print("\n--- PARTICIONES ---")

    print(f"Train: {datasets['train'].shape}")
    print(f"Validation: {datasets['validation'].shape}")
    print(f"Test reservado: {datasets['test'].shape}")

    # --------------------------------------------------------
    # CONFIGURACION DE EXPERIMENTOS
    # --------------------------------------------------------

    
    experiments = [
        {
            "name": "Logistic Regression - Baseline",
            "algorithm": "logistic_regression",
            "model_params": None,
            "use_engineered_features": False
        },
        {
            "name": "Logistic Regression - Feature Engineering",
            "algorithm": "logistic_regression",
            "model_params": None,
            "use_engineered_features": True
        }
    ]


    # --------------------------------------------------------
    # ENTRENAMIENTO Y TRACKING
    # --------------------------------------------------------

    experiment_results = []

    for experiment in experiments:

        print(
            f"\n--- ENTRENANDO: {experiment['name']} ---",
            flush=True
        )

        # Seleccionar variables para el experimento
        excluded_features = (
            []
            if experiment["use_engineered_features"]
            else list(FEATURE_TYPES)
        )

        experiment_train = datasets["train"].drop(
            columns=excluded_features
        )

        experiment_validation = datasets["validation"].drop(
            columns=excluded_features
        )

        print(
            f"Variables utilizadas: "
            f"{experiment_train.shape[1] - 1}"
        )

        # ----------------------------------------------------
        # ENTRENAMIENTO MEDIANTE EL ORQUESTADOR
        # ----------------------------------------------------

        training_result = orchestrator.run(
            stage="train_model",
            train=experiment_train,
            validation=experiment_validation,
            algorithm=experiment["algorithm"],
            model_params=experiment["model_params"]
        )

        # ----------------------------------------------------
        # METRICAS DE VALIDACION
        # ----------------------------------------------------

        print("\n--- METRICAS VALIDATION ---")

        for metric, value in (
            training_result["metrics"].items()
        ):
            print(f"{metric}: {value:.6f}")

        # ----------------------------------------------------
        # REGISTRO EN MLFLOW
        # ----------------------------------------------------

        print(
            "\n--- REGISTRANDO EN MLFLOW ---",
            flush=True
        )

        tracking_result = orchestrator.run(
            stage="track_model",
            training_result=training_result,
            model=churn_model.model,
            run_name=experiment["name"]
        )

        print(
            f"Run ID: {tracking_result['run_id']}"
        )

        print(
            f"Model URI: {tracking_result['model_uri']}"
        )

        # ----------------------------------------------------
        # CONSOLIDAR RESULTADOS
        # ----------------------------------------------------

        experiment_results.append({
            "name": experiment["name"],
            "algorithm": experiment["algorithm"],
            "metrics": training_result["metrics"].copy(),
            "run_id": tracking_result["run_id"],
            "model_uri": tracking_result["model_uri"]
        })

    # --------------------------------------------------------
    # COMPARACION FINAL
    # --------------------------------------------------------

    print("\n--- COMPARACION FINAL ---")

    for experiment in experiment_results:

        print(f"\nModelo: {experiment['name']}")

        for metric, value in experiment["metrics"].items():
            print(f"{metric}: {value:.6f}")

        print(
            f"Run ID: {experiment['run_id']}"
        )

    # --------------------------------------------------------
    # FINALIZACION
    # --------------------------------------------------------

    print("\nTest permanece reservado.")
    print("EXPERIMENTOS COMPLETADOS")
