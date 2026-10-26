class DataValidator:

    def validate_headers(self, headers: list[str]) -> dict:

        errors = []

        if not headers:
            errors.append("El archivo no contiene encabezados.")
        else:
            if any(not header.strip() for header in headers):
                errors.append("Existen encabezados vacíos.")

            if len(headers) != len(set(headers)):
                errors.append("Existen encabezados duplicados.")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }
    
    
    def validate_row(
        self,
        row: dict,
        expected_columns: int
    ) -> dict:

        errors = []

        if None in row:
            errors.append(
                "El registro contiene columnas adicionales."
            )

        missing_columns = [
            column
            for column, value in row.items()
            if column is not None and value is None
        ]

        if missing_columns:
            errors.append(
                f"Faltan valores para las columnas: {missing_columns}"
            )

        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }
    
    def validate_identifiers(
        self,
        rows,
        key_column: str
    ) -> dict:

        seen_ids = set()
        duplicate_ids = set()
        missing_ids = 0
        total_rows = 0

        for row in rows:
            total_rows += 1

            customer_id = row.get(key_column)

            if customer_id is None or not customer_id.strip():
                missing_ids += 1
                continue

            if customer_id in seen_ids:
                duplicate_ids.add(customer_id)
            else:
                seen_ids.add(customer_id)

        return {
            "is_valid": missing_ids == 0 and len(duplicate_ids) == 0,
            "total_rows": total_rows,
            "missing_ids": missing_ids,
            "duplicate_ids": sorted(duplicate_ids)
        }