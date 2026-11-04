from collections.abc import Iterable

from src.data.repositories.postgres_repository import PostgresRepository


class GoldIngestion:

    def __init__(self, repository: PostgresRepository):
        self.repository = repository

    def ingest(
        self,
        batches: Iterable[list[dict]],
        column_types: dict[str, str],
        table_name: str = "customers"
    ) -> dict:
        schema = "gold"
        columns = list(column_types)
        inserted_rows = 0

        self.repository.create_schema(schema=schema)

        self.repository.create_typed_table(
            schema=schema,
            table=table_name,
            column_types=column_types
        )

        with self.repository.connect() as connection:
            self.repository.truncate_table(
                connection=connection,
                schema=schema,
                table=table_name
            )

            for batch in batches:
                if not batch:
                    continue

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