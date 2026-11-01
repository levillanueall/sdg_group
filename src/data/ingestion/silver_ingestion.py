from collections.abc import Iterable

from src.data.repositories.postgres_repository import PostgresRepository


class SilverIngestion:

    def __init__(
        self,
        repository: PostgresRepository
    ):
        self.repository = repository

    def ingest(
        self,
        transformed_batches: Iterable[list[dict]],
        column_types: dict[str, str],
        schema: str = "silver",
        table_name: str = "customers"
    ) -> dict:

        columns = list(column_types)

        self.repository.create_schema(schema)

        self.repository.create_typed_table(
            schema=schema,
            table=table_name,
            column_types=column_types
        )

        inserted_rows = 0

        with self.repository.connect() as connection:
            self.repository.truncate_table(
                connection=connection,
                schema=schema,
                table=table_name
            )

            for batch in transformed_batches:
                inserted_rows += self.repository.insert_batch(
                    connection=connection,
                    schema=schema,
                    table=table_name,
                    columns=columns,
                    rows=batch
                )

        return {
            "schema": schema,
            "table": table_name,
            "inserted_rows": inserted_rows,
            "status": "success"
        }