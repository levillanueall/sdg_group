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

    def ingest(
        self,
        file_path: str,
        table_name: str,
        schema: str = "bronze",
        batch_size: int = 1000
    ) -> dict:

        if batch_size <= 0:
            raise ValueError("batch_size debe ser mayor que cero.")

        # 1. Leer y validar encabezados
        reader = CSVReader(file_path)
        headers = reader.read_headers()

        validation = self.validator.validate_headers(headers)

        if not validation["is_valid"]:
            raise ValueError(validation["errors"])

        # 2. Validar la estructura de todas las filas
        total_rows = 0

        for row in reader.read_rows():
            total_rows += 1
            result = self.validator.validate_row(
                row,
                len(headers)
            )

            if not result["is_valid"]:
                raise ValueError(
                    f"Error en fila {total_rows}: {result['errors']}"
                )

        # 3. Crear la tabla
        self.repository.create_schema(schema)
        
        self.repository.create_table(
            schema=schema,
            table=table_name,
            columns=headers
        )

        # 4. Iniciar la carga transaccional
        inserted_rows = 0
        batch = []

        with self.repository.connect() as connection:

            self.repository.truncate_table(
                connection,
                schema,
                table_name
            )

            for row in reader.read_rows():

                batch.append(row)

                if len(batch) >= batch_size:
                    inserted_rows += self.repository.insert_batch(
                        connection,
                        schema,
                        table_name,
                        headers,
                        batch
                    )
                    batch = []

            # Insertar el último lote incompleto
            if batch:
                inserted_rows += self.repository.insert_batch(
                    connection,
                    schema,
                    table_name,
                    headers,
                    batch
                )

        return {
            "schema": schema,
            "table": table_name,
            "validated_rows": total_rows,
            "inserted_rows": inserted_rows,
            "status": "success"
        }