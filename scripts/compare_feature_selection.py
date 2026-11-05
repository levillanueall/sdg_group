
from pathlib import Path

from src.config.model_config import EXCLUDED_COLUMNS
from src.data.repositories.postgres_repository import PostgresRepository
from src.data.readers.gold_reader import GoldDatasetReader
from src.modeling.data_splitter import ChurnDataSplitter
from src.modeling.correlation_feature_selector import (
    CorrelationFeatureSelector
)
from src.modeling.feature_selection_evaluator import (
    FeatureSelectionEvaluator
)


OUTPUT_PATH = Path(
    "results/model_analysis/feature_selection_comparison.csv"
)


def main():

    # 1. Cargar dataset Gold
    repository = PostgresRepository(
        host="localhost",
        port=5433,
        database="sdg_churn",
        user="sdg_user",
        password="sdg_password"
    )

    reader = GoldDatasetReader(repository=repository)

    dataframe = reader.load(batch_size=1000)

    print(f"Dataset original: {dataframe.shape}")

    # 2. Generar las mismas particiones
    splitter = ChurnDataSplitter(
        test_size=0.20,
        validation_size=0.20,
        random_state=42
    )

    datasets = splitter.split(dataframe)

    train = datasets["train"]
    validation = datasets["validation"]

    print(f"Train: {train.shape}")
    print(f"Validation: {validation.shape}")
    print(f"Test reservado: {datasets['test'].shape}")

    # 3. Definir variables del baseline
    baseline_features = [
        column for column in train.columns
        if column not in EXCLUDED_COLUMNS
    ]

    # 4. Seleccionar variables sobre Train
    selector = CorrelationFeatureSelector(
        threshold=0.95
    )

    selection_results = selector.analyze(train)

    selected_features = selection_results[
        "selected_features"
    ]

    candidates = selection_results[
        "candidates_to_remove"
    ]

    if not selected_features:
        raise ValueError(
            "La selección no produjo variables."
        )

    if not set(selected_features).issubset(
        set(baseline_features)
    ):
        raise ValueError(
            "La selección contiene variables inválidas."
        )

    print("\n--- FEATURE SELECTION ---")
    print(f"Baseline: {len(baseline_features)} variables")
    print(f"Seleccionadas: {len(selected_features)} variables")
    print(f"Candidatas a eliminar: {len(candidates)}")

    # 5. Definir experimentos
    feature_sets = {
        "baseline": baseline_features,
        "correlation_095": selected_features
    }

    # 6. Entrenar y evaluar ambos modelos
    evaluator = FeatureSelectionEvaluator(
        random_state=42
    )

    results = evaluator.evaluate(
        train=train,
        validation=validation,
        feature_sets=feature_sets
    )

    # 7. Exportar resultados
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n--- COMPARACIÓN DE MODELOS ---")
    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    baseline_auc = results.loc[
        results["experiment"] == "baseline",
        "roc_auc"
    ].iloc[0]

    selected_auc = results.loc[
        results["experiment"] == "correlation_095",
        "roc_auc"
    ].iloc[0]

    print(
        f"\nDiferencia ROC-AUC: "
        f"{selected_auc - baseline_auc:+.6f}"
    )

    print(f"\nReporte guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
