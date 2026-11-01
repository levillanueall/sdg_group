from src.data.repositories.postgres_repository import PostgresRepository
from src.transformations.data_type_transformer import DataTypeTransformer
from collections.abc import Iterator

class SilverTransformation:

    def __init__(
        self,
        repository: PostgresRepository,
        transformer: DataTypeTransformer
    ):
        self.repository = repository
        self.transformer = transformer

    def build_column_types(self) -> dict[str, str]:
        column_types = {}

        for batch in self.repository.read_batches(
            schema="bronze",
            table="data_descriptions"
        ):
            for row in batch:
                definition = self.transformer.get_silver_column_definition(
                    variable=row["Variable"],
                    meaning=row["Significado"]
                )

                column_types[definition["column"]] = definition["type"]

        return column_types

    def transform_batches(
        self,
        batch_size: int = 1000
    ) -> Iterator[list[dict]]:

        column_types = self.build_column_types()

        for batch in self.repository.read_batches(
            schema="bronze",
            table="customers_raw",
            batch_size=batch_size
        ):
            transformed_batch = [
                self.transformer.transform_row(
                    row=row,
                    column_types=column_types
                )
                for row in batch
            ]

            yield transformed_batch