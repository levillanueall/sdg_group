from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config.model_config import TARGET_COLUMN
from src.data.repositories.postgres_repository import PostgresRepository
from src.data.readers.gold_reader import GoldDatasetReader
from src.eda.churn_association_analyzer import ChurnAssociationAnalyzer


OUTPUT_DIR = Path("results/model_analysis")


def main():

    repository = PostgresRepository(
        host="localhost",
        port=5433,
        database="sdg_churn",
        user="sdg_user",
        password="sdg_password"
    )

    gold_reader = GoldDatasetReader(
        repository=repository
    )

    dataframe = gold_reader.load(batch_size=1000)

    print(f"Dataset original: {dataframe.shape}")

    train_data, _ = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=42,
        stratify=dataframe[TARGET_COLUMN]
    )

    print(f"Dataset de entrenamiento: {train_data.shape}")

    analyzer = ChurnAssociationAnalyzer(
        random_state=42
    )

    results = analyzer.analyze(train_data)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR / "churn_numeric_associations.csv"
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"\nVariables analizadas: {len(results)}")

    print("\nTop 15 Mutual Information:")
    print(
        results.head(15).to_string(index=False)
    )

    print("\nTop 15 correlación absoluta con churn:")
    print(
        results.sort_values(
            "abs_pearson",
            ascending=False
        ).head(15).to_string(index=False)
    )

    print(f"\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    main()