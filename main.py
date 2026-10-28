
from src.data.repositories.postgres_repository import PostgresRepository
from src.data.validators.data_validator import DataValidator
from src.data.ingestion.bronze_ingestion import BronzeIngestion
from src.orchestration.data_orchestrator import DataOrchestrator


def main():

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

    data_orchestrator = DataOrchestrator(
        stages={
            "bronze": bronze_ingestion.ingest
        }
    )

    return data_orchestrator


if __name__ == "__main__":
    orchestrator = main()

    result = orchestrator.run(
        stage="bronze",
        file_path="data/dataset.csv",
        table_name="customers_raw"
    )

    print(result)

    descriptions_result = orchestrator.run(
        stage="bronze",
        file_path="data/data_descriptions.csv",
        table_name="data_descriptions"
    )

    print(descriptions_result)