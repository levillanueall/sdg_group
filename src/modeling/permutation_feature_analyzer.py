
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance


class PermutationFeatureAnalyzer:

    def __init__(
        self,
        n_repeats: int = 5,
        random_state: int = 42,
        sample_size: int = 5000
    ):
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.sample_size = sample_size

    def analyze(
        self,
        model,
        X_validation: pd.DataFrame,
        y_validation: pd.Series
    ) -> pd.DataFrame:

        if X_validation.empty:
            raise ValueError(
                "El conjunto de validación está vacío."
            )

        if len(X_validation) != len(y_validation):
            raise ValueError(
                "X_validation e y_validation no coinciden."
            )

        n_samples = min(
            self.sample_size,
            len(X_validation)
        )

        rng = np.random.RandomState(
            self.random_state
        )

        indices = rng.choice(
            len(X_validation),
            size=n_samples,
            replace=False
        )

        X_sample = X_validation.iloc[
            indices
        ].reset_index(drop=True)

        y_sample = y_validation.iloc[
            indices
        ].reset_index(drop=True)

        print(
            f"Observaciones para permutación: {n_samples}"
        )

        results = permutation_importance(
            estimator=model,
            X=X_sample,
            y=y_sample,
            scoring="roc_auc",
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=1
        )

        importance_df = pd.DataFrame({
            "feature": X_sample.columns,
            "importance_mean": results.importances_mean,
            "importance_std": results.importances_std
        })

        return importance_df.sort_values(
            by="importance_mean",
            ascending=False
        ).reset_index(drop=True)
