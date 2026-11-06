from collections.abc import Iterable
from src.features.feature_engineering import FeatureEngineering
from src.config.model_config import FEATURE_TYPES

from src.data.repositories.postgres_repository import PostgresRepository


class GoldIngestion:

    def __init__(self, repository: PostgresRepository):
        self.repository = repository
        self.feature_engineering = FeatureEngineering()

    def ingest(
        self,
        batches: Iterable[list[dict]],
        column_types: dict[str, str],
        table_name: str = "customers"
    ) -> dict:

        schema = "gold"

        gold_column_types = {
            **column_types,
            **FEATURE_TYPES
        }

        columns = list(gold_column_types)
        inserted_rows = 0

        self.repository.create_schema(schema=schema)

        self.repository.create_typed_table(
            schema=schema,
            table=table_name,
            column_types=gold_column_types
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

                enriched_batch = self.feature_engineering.transform_batch(batch)

                inserted_rows += self.repository.insert_batch(
                    connection=connection,
                    schema=schema,
                    table=table_name,
                    columns=columns,
                    rows=enriched_batch
                )

        return {
            "schema": schema,
            "table": table_name,
            "inserted_rows": inserted_rows,
            "status": "success"
        }