
from numbers import Number

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.utils.validation import check_is_fitted


class ChurnPreprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):

        if (
            not isinstance(X, pd.DataFrame)
            or X.empty
            or X.shape[1] == 0
        ):
            raise ValueError(
                "X debe contener variables predictoras."
            )

        self.feature_columns_ = X.columns.tolist()

        self.numeric_columns_ = []
        self.categorical_columns_ = []

        # Identificar tipos usando entrenamiento
        for column in self.feature_columns_:

            non_null = X[column].dropna()

            is_numeric = (
                pd.api.types.is_numeric_dtype(X[column])
                or (
                    not non_null.empty
                    and isinstance(non_null.iloc[0], Number)
                )
            )

            if is_numeric:
                self.numeric_columns_.append(column)
            else:
                self.categorical_columns_.append(column)

        transformers = []

        # Preprocesamiento numérico
        if self.numeric_columns_:
            transformers.append((
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]),
                self.numeric_columns_
            ))

        # Preprocesamiento categórico
        if self.categorical_columns_:
            transformers.append((
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=30
                ),
                self.categorical_columns_
            ))

        self.transformer_ = ColumnTransformer(
            transformers=transformers
        )

        # Ajustar únicamente con entrenamiento
        self.transformer_.fit(
            self._normalize(X)
        )

        return self

    def transform(self, X: pd.DataFrame):

        check_is_fitted(
            self,
            "transformer_"
        )

        prepared = self._normalize(X)

        return self.transformer_.transform(
            prepared
        )

    def get_feature_names_out(
        self,
        input_features=None
    ):

        check_is_fitted(
            self,
            "transformer_"
        )

        return self.transformer_.get_feature_names_out()

    def _normalize(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Se esperaba un DataFrame de pandas."
            )

        missing = [
            column for column in self.feature_columns_
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Columnas faltantes: {missing}"
            )

        prepared = X[self.feature_columns_].copy()

        # Normalizar datos numéricos
        for column in self.numeric_columns_:

            prepared[column] = (
                pd.to_numeric(
                    prepared[column],
                    errors="coerce"
                )
                .astype("float64")
            )

        # Normalizar datos categóricos
        for column in self.categorical_columns_:

            prepared[column] = (
                prepared[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
                .fillna("MISSING")
                .astype(str)
            )

        return prepared
