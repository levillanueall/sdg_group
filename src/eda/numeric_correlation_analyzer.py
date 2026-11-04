from numbers import Number

import pandas as pd

from src.config.model_config import EXCLUDED_COLUMNS


class NumericCorrelationAnalyzer:

    def __init__(self, method: str = "pearson"):
        self.method = method

    def analyze(self, dataframe: pd.DataFrame) -> dict:

        # Excluir ID y variable objetivo
        X = dataframe.drop(
            columns=EXCLUDED_COLUMNS,
            errors="ignore"
        )

        # Identificar variables numéricas
        numeric_columns = []

        for column in X.columns:

            non_null = X[column].dropna()

            if (
                pd.api.types.is_numeric_dtype(X[column])
                or (
                    not non_null.empty
                    and isinstance(non_null.iloc[0], Number)
                )
            ):
                numeric_columns.append(column)

        if not numeric_columns:
            raise ValueError(
                "No se encontraron variables numéricas."
            )

        # Convertir a tipos numéricos de pandas
        numeric_data = X[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Calcular matriz de correlación
        correlation_matrix = numeric_data.corr(
            method=self.method
        )

        # Generar pares de variables sin duplicados
        pairs = []

        for i, column_1 in enumerate(numeric_columns):
            for column_2 in numeric_columns[i + 1:]:

                correlation = correlation_matrix.loc[
                    column_1, column_2
                ]

                if pd.notna(correlation):
                    pairs.append({
                        "variable_1": column_1,
                        "variable_2": column_2,
                        "correlation": correlation,
                        "abs_correlation": abs(correlation)
                    })

        correlation_pairs = pd.DataFrame(
            pairs,
            columns=[
                "variable_1",
                "variable_2",
                "correlation",
                "abs_correlation"
            ]
        )

        correlation_pairs = correlation_pairs.sort_values(
            by="abs_correlation",
            ascending=False
        ).reset_index(drop=True)

        return {
            "numeric_columns": numeric_columns,
            "correlation_matrix": correlation_matrix,
            "correlation_pairs": correlation_pairs
        }