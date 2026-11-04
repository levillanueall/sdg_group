from numbers import Number

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif

from src.config.model_config import (
    TARGET_COLUMN,
    EXCLUDED_COLUMNS
)


class ChurnAssociationAnalyzer:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def analyze(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        y = pd.to_numeric(
            dataframe[TARGET_COLUMN],
            errors="raise"
        ).astype(int)

        if not y.isin([0, 1]).all():
            raise ValueError(
                "La variable objetivo debe contener 0 y 1."
            )

        numeric_columns = []

        for column in dataframe.columns:

            if column in EXCLUDED_COLUMNS:
                continue

            non_null = dataframe[column].dropna()

            if (
                pd.api.types.is_numeric_dtype(dataframe[column])
                or (
                    not non_null.empty
                    and isinstance(non_null.iloc[0], Number)
                )
            ):
                numeric_columns.append(column)

        X = dataframe[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.replace([np.inf, -np.inf], np.nan)

        valid_columns = X.columns[
            X.nunique(dropna=True) > 1
        ]

        X = X[valid_columns]

        if X.empty:
            raise ValueError(
                "No hay variables numéricas válidas."
            )

        pearson = X.corrwith(y, method="pearson")

        X_imputed = X.fillna(X.median())

        mutual_information = mutual_info_classif(
            X_imputed.to_numpy(dtype=float),
            y,
            discrete_features=False,
            random_state=self.random_state
        )

        results = pd.DataFrame({
            "feature": X.columns,
            "pearson_correlation": pearson.values,
            "mutual_information": mutual_information
        })

        results["abs_pearson"] = (
            results["pearson_correlation"].abs()
        )

        return results.sort_values(
            by="mutual_information",
            ascending=False
        ).reset_index(drop=True)