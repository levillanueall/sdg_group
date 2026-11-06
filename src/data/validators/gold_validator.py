from src.data.repositories.postgres_repository import PostgresRepository
from src.config.model_config import FEATURE_TYPES

class GoldValidator:

    def __init__(self, repository: PostgresRepository):
        self.repository = repository

    def validate_structure(
        self,
        silver_schema: str = "silver",
        silver_table: str = "customers",
        gold_schema: str = "gold",
        gold_table: str = "customers"
    ) -> dict:

        silver_rows = self.repository.count_rows(
            schema=silver_schema,
            table=silver_table
        )

        gold_rows = self.repository.count_rows(
            schema=gold_schema,
            table=gold_table
        )

        silver_columns = self.repository.get_table_columns(
            schema=silver_schema,
            table=silver_table
        )

        gold_columns = self.repository.get_table_columns(
            schema=gold_schema,
            table=gold_table
        )

        silver_names = set(silver_columns)
        gold_names = set(gold_columns)

        expected_gold_names = silver_names | set(FEATURE_TYPES)

        type_mismatches = {}

        for column in silver_names & gold_names:
            expected_type = silver_columns[column]
            actual_type = gold_columns[column]

            if expected_type != actual_type:
                type_mismatches[column] = {
                    "expected": expected_type,
                    "actual": actual_type
                }

        checks = {
            "row_count": silver_rows > 0 and silver_rows == gold_rows,
            "columns": gold_names == expected_gold_names,
            "types": not type_mismatches
        }

        return {
            "is_valid": all(checks.values()),
            "checks": checks,
            "row_counts": {
                "silver": silver_rows,
                "gold": gold_rows
            },
            "column_counts": {
                "silver": len(silver_columns),
                "gold": len(gold_columns)
            },
            "column_differences": {
                "missing_in_gold": sorted(expected_gold_names - gold_names),
                "extra_in_gold": sorted(gold_names - expected_gold_names)
            },
            "type_mismatches": type_mismatches
        }