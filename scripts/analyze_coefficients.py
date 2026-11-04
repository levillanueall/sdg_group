from pathlib import Path

import mlflow
import mlflow.sklearn

from src.modeling.coefficient_analyzer import (
    LogisticCoefficientAnalyzer
)


MODEL_URI = "models:/m-52d83afbb16c499cb2cead392e0107c8"

OUTPUT_PATH = Path(
    "results/model_analysis/logistic_regression_coefficients.csv"
)


def main():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    model = mlflow.sklearn.load_model(MODEL_URI)

    analyzer = LogisticCoefficientAnalyzer()

    results = analyzer.analyze(model)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print(f"Coeficientes analizados: {len(results)}")

    print("\nTop 20 coeficientes:")
    print(results.head(20).to_string(index=False))

    print(f"\nResultados guardados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()