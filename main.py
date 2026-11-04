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

import mlflow

def main():

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("sdg_churn")


    repository = PostgresRepository(
        host="localhost",
        port=5433,
        database="sdg_churn",
        user="sdg_user",
        password="sdg_password"
    )

    validator = DataValidator()

    bronze_ingestion = BronzeIngestion(
        repository=repository,
        validator=validator
    )

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

    churn_eda = ChurnEDA(
        repository=repository
    )

    gold_ingestion = GoldIngestion(
        repository=repository
    )

    gold_validator = GoldValidator(
        repository=repository
)
    
    data_orchestrator = DataOrchestrator(
        stages={
            "bronze": bronze_ingestion.ingest,
            "silver": silver_ingestion.ingest,
            "validate_silver": silver_validator.validate_structure,
            "validate_silver_ids": silver_validator.validate_identifiers,
            "validate_silver_values": silver_validator.validate_values,
            "eda_target": churn_eda.analyze_target,
            "eda_missing": churn_eda.analyze_missing_values,
            "gold": gold_ingestion.ingest,
            "validate_gold": gold_validator.validate_structure,
        }
    )

    return data_orchestrator, silver_transformation, repository


if __name__ == "__main__":
    orchestrator, silver_transformation, repository = main()

    # Cargar los clientes en Bronze.
    bronze_result = orchestrator.run(
        stage="bronze",
        file_path="data/dataset.csv",
        table_name="customers_raw"
    )

    print(bronze_result)

    # Cargar las descripciones en Bronze.
    descriptions_result = orchestrator.run(
        stage="bronze",
        file_path="data/data_descriptions.csv",
        table_name="data_descriptions"
    )

    print(descriptions_result)

    # Obtener las definiciones y preparar los lotes transformados.
    column_types = silver_transformation.build_column_types()

    transformed_batches = silver_transformation.transform_batches(
        batch_size=1000
    )

    # Cargar los lotes transformados en Silver.
    silver_result = orchestrator.run(
        stage="silver",
        transformed_batches=transformed_batches,
        column_types=column_types,
        schema="silver",
        table_name="customers"
    )

    print(silver_result)

    gold_batches = repository.read_batches(
        schema="silver",
        table="customers",
        batch_size=1000
    )

    gold_result = orchestrator.run(
        stage="gold",
        batches=gold_batches,
        column_types=column_types,
        table_name="customers"
    )

    print(gold_result)

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

    # Validar estructura y consistencia entre Silver y Gold.
    gold_validation_result = orchestrator.run(
        stage="validate_gold"
    )

    print(gold_validation_result)

    target_result = orchestrator.run(
        stage="eda_target"
    )

    print(target_result)

    
    missing_result = orchestrator.run(
        stage="eda_missing"
    )

    for column_result in missing_result:
        print(column_result)