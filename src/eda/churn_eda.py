from src.data.repositories.postgres_repository import PostgresRepository


class ChurnEDA:

    def __init__(self, repository: PostgresRepository):
        self.repository = repository

    def analyze_target(self) -> dict:

        counts = self.repository.get_value_counts(
            schema="silver",
            table="customers",
            column="churn"
        )

        total_rows = sum(row["count"] for row in counts)

        null_count = 0
        unexpected_values = []
        distribution = []

        for row in counts:
            value = row["value"]
            count = row["count"]

            if value is None:
                null_count += count
            elif value not in (0, 1):
                unexpected_values.append(value)

            percentage = (
                count / total_rows * 100
                if total_rows > 0
                else 0.0
            )

            distribution.append({
                "value": value,
                "count": count,
                "percentage": round(percentage, 2)
            })

        return {
            "total_rows": total_rows,
            "null_count": null_count,
            "unexpected_values": unexpected_values,
            "distribution": distribution
        }
    
    
    def analyze_missing_values(self) -> list[dict]:

        columns = self.repository.get_table_columns(
            schema="silver",
            table="customers"
        )

        results = []

        for column, data_type in columns.items():
            if column in ("Customer_ID", "churn"):
                continue

            counts = self.repository.get_missing_counts(
                schema="silver",
                table="customers",
                column=column
            )

            total_rows = counts["total_rows"]
            null_count = counts["null_count"]
            empty_count = counts["empty_count"]
            missing_count = null_count + empty_count

            percentage_factor = (
                100 / total_rows
                if total_rows > 0
                else 0.0
            )

            results.append({
                "column": column,
                "data_type": data_type,
                "total_rows": total_rows,
                "null_count": null_count,
                "null_percentage": round(
                    null_count * percentage_factor, 2
                ),
                "empty_count": empty_count,
                "empty_percentage": round(
                    empty_count * percentage_factor, 2
                ),
                "missing_count": missing_count,
                "missing_percentage": round(
                    missing_count * percentage_factor, 2
                )
            })

        return sorted(
            results,
            key=lambda item: item["missing_percentage"],
            reverse=True
        )