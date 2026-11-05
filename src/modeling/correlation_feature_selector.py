
import pandas as pd

from src.config.model_config import TARGET_COLUMN, EXCLUDED_COLUMNS
from src.eda.numeric_correlation_analyzer import NumericCorrelationAnalyzer


class CorrelationFeatureSelector:

    def __init__(self, threshold: float = 0.95):
        if not 0 < threshold < 1:
            raise ValueError("El threshold debe estar entre 0 y 1.")

        self.threshold = threshold

    def analyze(self, train_df: pd.DataFrame) -> dict:

        if TARGET_COLUMN not in train_df.columns:
            raise ValueError("No existe la variable objetivo.")

        features = [
            col for col in train_df.columns
            if col not in EXCLUDED_COLUMNS
        ]

        analyzer = NumericCorrelationAnalyzer(method="pearson")
        results = analyzer.analyze(train_df)

        numeric_columns = results["numeric_columns"]
        correlation_matrix = results["correlation_matrix"]

        numeric_data = train_df[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        target = pd.to_numeric(
            train_df[TARGET_COLUMN],
            errors="raise"
        )

        target_correlation = (
            numeric_data.corrwith(target)
            .abs()
            .fillna(0.0)
        )

        coverage = numeric_data.notna().mean()

        ranked_features = sorted(
            numeric_columns,
            key=lambda col: (
                -target_correlation[col],
                -coverage[col],
                col
            )
        )

        selected_numeric = []
        dropped = {}

        for feature in ranked_features:

            conflicts = []

            for selected in selected_numeric:
                correlation = correlation_matrix.loc[
                    feature, selected
                ]

                if (
                    pd.notna(correlation)
                    and abs(correlation) >= self.threshold
                ):
                    conflicts.append(
                        (selected, abs(correlation))
                    )

            if conflicts:
                retained, correlation = max(
                    conflicts,
                    key=lambda item: item[1]
                )

                dropped[feature] = {
                    "correlated_with": retained,
                    "abs_correlation": correlation
                }

            else:
                selected_numeric.append(feature)

        selected_set = set(selected_numeric)

        # Las categóricas no se eliminan en esta etapa
        categorical_columns = [
            col for col in features
            if col not in numeric_columns
        ]

        selected_set.update(categorical_columns)

        selected_features = [
            col for col in features
            if col in selected_set
        ]

        report = []

        for feature in features:

            removal = dropped.get(feature)

            report.append({
                "feature": feature,
                "type": (
                    "numeric"
                    if feature in numeric_columns
                    else "categorical"
                ),
                "decision": (
                    "candidate_to_remove"
                    if removal
                    else "keep"
                ),
                "correlated_with": (
                    removal["correlated_with"]
                    if removal else None
                ),
                "abs_pair_correlation": (
                    removal["abs_correlation"]
                    if removal else None
                ),
                "abs_target_correlation": (
                    target_correlation.get(feature, None)
                )
            })

        return {
            "selected_features": selected_features,
            "candidates_to_remove": list(dropped.keys()),
            "selection_report": pd.DataFrame(report)
        }
