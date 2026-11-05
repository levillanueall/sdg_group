
from pathlib import Path

from src.data.repositories.postgres_repository import PostgresRepository
from src.data.readers.gold_reader import GoldDatasetReader
from src.modeling.data_splitter import ChurnDataSplitter
from src.modeling.correlation_feature_selector import (
    CorrelationFeatureSelector
)


OUTPUT_DIR = Path("results/model_analysis")


def main():

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

    splitter = ChurnDataSplitter(
        test_size=0.20,
        validation_size=0.20,
        random_state=42
    )

    datasets = splitter.split(dataframe)

    train = datasets["train"]

    print(f"Train: {train.shape}")
    print(f"Validation: {datasets['validation'].shape}")
    print(f"Test reservado: {datasets['test'].shape}")

    selector = CorrelationFeatureSelector(
        threshold=0.95
    )

    results = selector.analyze(train)

    selected_features = results["selected_features"]
    candidates = results["candidates_to_remove"]
    report = results["selection_report"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR / "correlation_feature_selection.csv"
    )

    report.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print("\n--- RESULTADOS ---")
    print(f"Variables originales: {dataframe.shape[1] - 1}")
    print(f"Variables conservadas: {len(selected_features)}")
    print(f"Candidatas a eliminar: {len(candidates)}")

    print("\nVariables candidatas a eliminar:")
    for feature in candidates:
        print(f" - {feature}")

    print(f"\nReporte guardado en: {output_path}")


if __name__ == "__main__":
    main()
