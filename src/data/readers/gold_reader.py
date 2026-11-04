import pandas as pd

from src.data.repositories.postgres_repository import PostgresRepository
from src.config.model_config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    EXCLUDED_COLUMNS
)


class GoldDatasetReader:

    def __init__(self, repository: PostgresRepository):
        self.repository = repository

    def load(self, batch_size: int = 1000) -> pd.DataFrame:

        records = []
        columns = None

        batches = self.repository.read_batches(
            schema="gold",
            table="customers",
            batch_size=batch_size
        )

        for batch in batches:
            for row in batch:

                if columns is None:

                    if TARGET_COLUMN not in row:
                        raise ValueError(
                            f"No existe la columna objetivo: {TARGET_COLUMN}"
                        )

                    if FEATURE_COLUMNS is None:
                        features = [
                            col for col in row.keys()
                            if col not in EXCLUDED_COLUMNS
                        ]
                    else:
                        features = [
                            col for col in FEATURE_COLUMNS
                            if col not in EXCLUDED_COLUMNS
                        ]

                    if not features:
                        raise ValueError(
                            "No hay variables predictoras."
                        )

                    columns = features + [TARGET_COLUMN]

                    missing = [
                        col for col in columns
                        if col not in row
                    ]

                    if missing:
                        raise ValueError(
                            f"Columnas inexistentes en Gold: {missing}"
                        )

                records.append({
                    col: row[col]
                    for col in columns
                })

        if not records:
            raise ValueError("No hay registros en Gold.")

        return pd.DataFrame.from_records(
            records,
            columns=columns
        )