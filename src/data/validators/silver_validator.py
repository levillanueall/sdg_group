from src.data.repositories.postgres_repository import PostgresRepository
from decimal import Decimal


class SilverValidator:

    def __init__(self, repository: PostgresRepository):
        self.repository = repository

    def validate_structure(
        self,
        column_types: dict[str, str],
        bronze_schema: str = "bronze",
        bronze_table: str = "customers_raw",
        silver_schema: str = "silver",
        silver_table: str = "customers"
    ) -> dict:

        bronze_rows = self.repository.count_rows(
            schema=bronze_schema,
            table=bronze_table
        )

        silver_rows = self.repository.count_rows(
            schema=silver_schema,
            table=silver_table
        )

        bronze_columns = self.repository.get_table_columns(
            schema=bronze_schema,
            table=bronze_table
        )

        silver_columns = self.repository.get_table_columns(
            schema=silver_schema,
            table=silver_table
        )

        bronze_names = set(bronze_columns)
        silver_names = set(silver_columns)
        defined_names = set(column_types)

        postgres_types = {
            "string": "text",
            "numeric": "numeric"
        }

        type_mismatches = {}

        for column, logical_type in column_types.items():
            if logical_type not in postgres_types:
                raise ValueError(
                    f"Tipo no soportado para {column}: {logical_type}"
                )

            expected_type = postgres_types[logical_type]
            actual_type = silver_columns.get(column)

            if actual_type != expected_type:
                type_mismatches[column] = {
                    "expected": expected_type,
                    "actual": actual_type
                }

        checks = {
            "row_count": bronze_rows > 0 and bronze_rows == silver_rows,
            "columns": bronze_names == silver_names == defined_names,
            "types": not type_mismatches
        }

        return {
            "is_valid": all(checks.values()),
            "checks": checks,
            "row_counts": {
                "bronze": bronze_rows,
                "silver": silver_rows
            },
            "column_counts": {
                "bronze": len(bronze_columns),
                "silver": len(silver_columns),
                "defined": len(column_types)
            },
            "column_differences": {
                "missing_in_silver": sorted(bronze_names - silver_names),
                "extra_in_silver": sorted(silver_names - bronze_names),
                "missing_definitions": sorted(bronze_names - defined_names),
                "extra_definitions": sorted(defined_names - bronze_names)
            },
            "type_mismatches": type_mismatches
        }
    
    def _inspect_identifiers(
        self,
        schema: str,
        table: str,
        key_column: str = "Customer_ID"
    ) -> dict:

        identifiers = set()
        duplicate_ids = set()
        missing_ids = 0
        total_rows = 0

        for batch in self.repository.read_batches(
            schema=schema,
            table=table
        ):
            for row in batch:
                total_rows += 1
                value = row[key_column]

                if value is None or not value.strip():
                    missing_ids += 1
                    continue

                identifier = value.strip()

                if identifier in identifiers:
                    duplicate_ids.add(identifier)
                else:
                    identifiers.add(identifier)

        return {
            "total_rows": total_rows,
            "identifiers": identifiers,
            "missing_ids": missing_ids,
            "duplicate_ids": sorted(duplicate_ids)
        }
    
    
    def validate_identifiers(
        self,
        bronze_schema: str = "bronze",
        bronze_table: str = "customers_raw",
        silver_schema: str = "silver",
        silver_table: str = "customers",
        key_column: str = "Customer_ID"
    ) -> dict:

        bronze = self._inspect_identifiers(
            schema=bronze_schema,
            table=bronze_table,
            key_column=key_column
        )

        silver = self._inspect_identifiers(
            schema=silver_schema,
            table=silver_table,
            key_column=key_column
        )

        missing_in_silver = (
            bronze["identifiers"] - silver["identifiers"]
        )

        extra_in_silver = (
            silver["identifiers"] - bronze["identifiers"]
        )

        checks = {
            "row_count": (
                bronze["total_rows"] > 0
                and bronze["total_rows"] == silver["total_rows"]
            ),
            "no_missing_ids": (
                bronze["missing_ids"] == 0
                and silver["missing_ids"] == 0
            ),
            "no_duplicate_ids": (
                not bronze["duplicate_ids"]
                and not silver["duplicate_ids"]
            ),
            "same_identifiers": (
                not missing_in_silver
                and not extra_in_silver
            )
        }

        return {
            "is_valid": all(checks.values()),
            "checks": checks,
            "unique_id_counts": {
                "bronze": len(bronze["identifiers"]),
                "silver": len(silver["identifiers"])
            },
            "missing_ids": {
                "bronze": bronze["missing_ids"],
                "silver": silver["missing_ids"]
            },
            "duplicate_ids": {
                "bronze": bronze["duplicate_ids"],
                "silver": silver["duplicate_ids"]
            },
            "missing_in_silver": sorted(missing_in_silver),
            "extra_in_silver": sorted(extra_in_silver)
        }

    def _expected_value(
        self,
        value: str | None,
        column_type: str
    ):
        if column_type == "string":
            return None if value is None else value.strip()

        if column_type == "numeric":
            if value is None or not value.strip():
                return None

            normalized = value.strip().replace(",", ".")
            return Decimal(normalized)

        raise ValueError(
            f"Tipo no soportado: {column_type}"
        )

    
    def validate_values(
        self,
        column_types: dict[str, str],
        batch_size: int = 1000,
        max_examples: int = 10
    ) -> dict:

        if not column_types:
            raise ValueError("El mapa de tipos está vacío.")

        if batch_size <= 0:
            raise ValueError("batch_size debe ser mayor que cero.")

        if max_examples < 0:
            raise ValueError("max_examples no puede ser negativo.")

        compared_rows = 0
        compared_values = 0
        differences_by_column = {
            column: 0 for column in column_types
        }
        examples = []

        for batch in self.repository.read_batches(
            schema="bronze",
            table="customers_raw",
            batch_size=batch_size
        ):
            identifiers = []

            for row in batch:
                identifier = row["Customer_ID"]

                if identifier is None or not identifier.strip():
                    raise ValueError(
                        "Existe un Customer_ID vacío en Bronze."
                    )

                identifiers.append(identifier.strip())

            silver_rows = self.repository.read_by_ids(
                schema="silver",
                table="customers",
                identifiers=identifiers
            )

            silver_by_id = {}

            for row in silver_rows:
                identifier = row["Customer_ID"]

                if identifier in silver_by_id:
                    raise ValueError(
                        f"Customer_ID duplicado en Silver: {identifier}"
                    )

                silver_by_id[identifier] = row

            for bronze_row in batch:
                identifier = bronze_row["Customer_ID"].strip()

                if identifier not in silver_by_id:
                    raise ValueError(
                        f"Customer_ID ausente en Silver: {identifier}"
                    )

                silver_row = silver_by_id[identifier]
                compared_rows += 1

                for column, column_type in column_types.items():
                    original = bronze_row[column]

                    expected = self._expected_value(
                        value=original,
                        column_type=column_type
                    )

                    actual = silver_row[column]
                    compared_values += 1

                    if expected != actual:
                        differences_by_column[column] += 1

                        if len(examples) < max_examples:
                            examples.append({
                                "Customer_ID": identifier,
                                "column": column,
                                "bronze": original,
                                "expected": expected,
                                "actual": actual
                            })

        total_differences = sum(differences_by_column.values())

        return {
            "is_valid": (
                compared_rows > 0
                and total_differences == 0
            ),
            "compared_rows": compared_rows,
            "compared_values": compared_values,
            "total_differences": total_differences,
            "differences_by_column": {
                column: count
                for column, count in differences_by_column.items()
                if count > 0
            },
            "examples": examples
        }