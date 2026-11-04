from pathlib import Path

from src.data.repositories.postgres_repository import PostgresRepository
from src.data.readers.gold_reader import GoldDatasetReader
from src.eda.numeric_correlation_analyzer import NumericCorrelationAnalyzer


OUTPUT_DIR = Path("results/model_analysis")


def main():

    repository = PostgresRepository(
        host="localhost",
        port=5433,
        database="sdg_churn",
        user="sdg_user",
        password="sdg_password"
    )

    gold_reader = GoldDatasetReader(repository=repository)

    dataframe = gold_reader.load(batch_size=1000)

    print(f"Dataset: {dataframe.shape}")

    analyzer = NumericCorrelationAnalyzer(method="pearson")

    results = analyzer.analyze(dataframe)

    correlation_matrix = results["correlation_matrix"]
    correlation_pairs = results["correlation_pairs"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    correlation_matrix.to_csv(
        OUTPUT_DIR / "numeric_correlation_matrix.csv",
        encoding="utf-8"
    )

    correlation_pairs.to_csv(
        OUTPUT_DIR / "numeric_correlation_pairs.csv",
        index=False,
        encoding="utf-8"
    )

    high_correlation = correlation_pairs[
        correlation_pairs["abs_correlation"] >= 0.85
    ]

    print(
        f"Variables numéricas: "
        f"{len(results['numeric_columns'])}"
    )

    print(
        f"Pares analizados: {len(correlation_pairs)}"
    )

    print(
        f"Pares con correlación >= 0.85: "
        f"{len(high_correlation)}"
    )

    print("\nTop 20 correlaciones:")
    print(correlation_pairs.head(20).to_string(index=False))


if __name__ == "__main__":
    main()