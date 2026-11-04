import pandas as pd

from sklearn.pipeline import Pipeline


class LogisticCoefficientAnalyzer:

    def analyze(self, model: Pipeline) -> pd.DataFrame:

        # Recuperar los componentes del modelo entrenado.
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        # Obtener los nombres después de las transformaciones.
        feature_names = preprocessor.get_feature_names_out()

        # Obtener los coeficientes de la regresión logística.
        coefficients = classifier.coef_[0]

        if len(feature_names) != len(coefficients):
            raise ValueError(
                "El número de variables no coincide "
                "con el número de coeficientes."
            )

        # Construir tabla de análisis.
        results = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients
        })

        results["abs_coefficient"] = (
            results["coefficient"].abs()
        )

        results["direction"] = results["coefficient"].apply(
            lambda value: (
                "positive" if value > 0
                else "negative" if value < 0
                else "neutral"
            )
        )

        return results.sort_values(
            by="abs_coefficient",
            ascending=False
        ).reset_index(drop=True)