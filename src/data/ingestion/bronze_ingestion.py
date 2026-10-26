from src.data.readers.csv_reader import CSVReader
from src.data.validators.data_validator import DataValidator
from src.data.repositories.postgres_repository import PostgresRepository


class BronzeIngestion:

    def __init__(
        self,
        repository: PostgresRepository,
        validator: DataValidator
    ):
        self.repository = repository
        self.validator = validator

    def create_table(
        self,
        file_path: str,
        table_name: str,
        schema: str = "bronze"
    ) -> dict:

        
        reader = CSVReader(file_path)
        headers = reader.read_headers()

        validation = self.validator.validate_headers(headers)

        if not validation["is_valid"]:
            raise ValueError(
                f"Encabezados inválidos: {validation['errors']}"
            )

        self.repository.create_table(
            schema=schema,
            table=table_name,
            columns=headers
        )

        return {
            "schema": schema,
            "table": table_name,
            "columns": len(headers),
            "status": "created_or_existing"
        }