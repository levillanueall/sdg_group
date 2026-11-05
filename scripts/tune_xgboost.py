
import json
from pathlib import Path

import pandas as pd

from src.data.repositories.postgres_repository import (
    PostgresRepository
)
from src.data.readers.gold_reader import GoldDatasetReader
from src.modeling.data_splitter import ChurnDataSplitter
from src.modeling.xgboost_hyperparameter_tuner import (
    XGBoostHyperparameterTuner
)


OUTPUT_DIR = Path("results/model_analysis")


def main():

    # 1. Conectar con PostgreSQL
    repository = PostgresRepository(
        host="localhost",
        port=5433,
        database="sdg_churn",
        user="sdg_user",
        password="sdg_password"
    )

    # 2. Cargar datos desde Gold
    reader = GoldDatasetReader(repository=repository)

    dataframe = reader.load(batch_size=1000)

    print(f"Dataset Gold: {dataframe.shape}")

    # 3. Reservar Test
    splitter = ChurnDataSplitter(
        test_size=0.20,
        validation_size=0.20,
        random_state=42
    )

    datasets = splitter.split(dataframe)

    # 4. Reunir Train y Validation
    development = pd.concat(
        [
            datasets["train"],
            datasets["validation"]
        ],
        ignore_index=True
    )

    print(f"Desarrollo: {development.shape}")
    print(f"Test reservado: {datasets['test'].shape}")

    # 5. Inicializar optimizador
    tuner = XGBoostHyperparameterTuner(
        n_iter=20,
        n_splits=3,
        random_state=42
    )

    # 6. Ejecutar optimización
    print("\n--- OPTIMIZACIÓN XGBOOST ---")

    results = tuner.tune(
        development=development
    )

    # 7. Preparar resultados
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cv_results_path = (
        OUTPUT_DIR / "xgboost_tuning_cv_results.csv"
    )

    summary_path = (
        OUTPUT_DIR / "xgboost_tuning_summary.json"
    )

    # 8. Exportar resultados de cada configuración
    results["cv_results"].to_csv(
        cv_results_path,
        index=False,
        encoding="utf-8"
    )

    # 9. Consolidar resumen
    summary = {
        "algorithm": "XGBClassifier",
        "scoring": "roc_auc",
        "development_rows": len(development),
        "reserved_test_rows": len(datasets["test"]),
        "n_features": int(results["n_features"]),
        "n_iter": int(results["n_iter"]),
        "n_splits": int(results["n_splits"]),
        "best_cv_roc_auc": float(
            results["best_cv_roc_auc"]
        ),
        "best_params": {
            key: (
                value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in results["best_params"].items()
        }
    }

    # 10. Exportar resumen en JSON
    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # 11. Mostrar mejores resultados
    print("\n--- RESULTADOS XGBOOST ---")

    print(
        f"Mejor ROC-AUC CV: "
        f"{summary['best_cv_roc_auc']:.6f}"
    )

    print("\nMejores hiperparámetros:")

    for parameter, value in summary["best_params"].items():
        print(f"{parameter}: {value}")

    print("\n--- ARCHIVOS GENERADOS ---")
    print(cv_results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
