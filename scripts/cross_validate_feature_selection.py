
from pathlib import Path

import pandas as pd

from src.data.repositories.postgres_repository import (
    PostgresRepository
)
from src.data.readers.gold_reader import GoldDatasetReader
from src.modeling.data_splitter import ChurnDataSplitter
from src.modeling.cross_validation_feature_evaluator import (
    CrossValidationFeatureEvaluator
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

    # 2. Cargar dataset Gold
    reader = GoldDatasetReader(repository=repository)

    dataframe = reader.load(batch_size=1000)

    print(f"Dataset Gold: {dataframe.shape}")

    # 3. Reservar el conjunto Test
    splitter = ChurnDataSplitter(
        test_size=0.20,
        validation_size=0.20,
        random_state=42
    )

    datasets = splitter.split(dataframe)

    # 4. Reunir Train + Validation
    development = pd.concat(
        [
            datasets["train"],
            datasets["validation"]
        ],
        ignore_index=True
    )

    print(f"Desarrollo: {development.shape}")
    print(f"Test reservado: {datasets['test'].shape}")

    # 5. Ejecutar validación cruzada
    evaluator = CrossValidationFeatureEvaluator(
        n_splits=5,
        threshold=0.95,
        random_state=42
    )

    results = evaluator.evaluate(
        dataframe=development
    )

    # 6. Calcular estadísticas por experimento
    summary = (
        results
        .groupby("experiment", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            n_features_mean=("n_features", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean")
        )
    )

    # 7. Comparar ROC-AUC fold por fold
    auc_comparison = results.pivot(
        index="fold",
        columns="experiment",
        values="roc_auc"
    )

    auc_differences = pd.DataFrame({
        "feature_selection_vs_baseline": (
            auc_comparison["correlation_095"]
            - auc_comparison["baseline"]
        ),
        "xgboost_vs_baseline": (
            auc_comparison["xgboost_98"]
            - auc_comparison["baseline"]
        )
    })

    # 8. Exportar resultados
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_DIR / "cross_validation_folds.csv",
        index=False,
        encoding="utf-8"
    )

    summary.to_csv(
        OUTPUT_DIR / "cross_validation_summary.csv",
        index=False,
        encoding="utf-8"
    )

    auc_differences.to_csv(
        OUTPUT_DIR / "cross_validation_differences.csv",
        index=True,
        encoding="utf-8"
    )

    # 9. Mostrar resumen
    print("\n--- RESUMEN CROSS-VALIDATION ---")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # 10. Mostrar diferencias por fold
    print("\n--- DIFERENCIAS ROC-AUC POR FOLD ---")
    print(
        auc_differences.to_string(
            float_format=lambda x: f"{x:+.6f}"
        )
    )

    # 11. Mostrar diferencias promedio
    print("\n--- DIFERENCIAS PROMEDIO ROC-AUC ---")
    print(
        auc_differences.mean().to_string(
            float_format=lambda x: f"{x:+.6f}"
        )
    )

    # 12. Confirmar archivos generados
    print("\n--- ARCHIVOS GENERADOS ---")
    print(OUTPUT_DIR / "cross_validation_folds.csv")
    print(OUTPUT_DIR / "cross_validation_summary.csv")
    print(OUTPUT_DIR / "cross_validation_differences.csv")


if __name__ == "__main__":
    main()
