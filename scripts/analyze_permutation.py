from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.config.model_config import (
    TARGET_COLUMN,
    EXCLUDED_COLUMNS
)
from src.data.repositories.postgres_repository import PostgresRepository
from src.data.readers.gold_reader import GoldDatasetReader
from src.modeling.data_splitter import ChurnDataSplitter
from src.modeling.permutation_feature_analyzer import (
    PermutationFeatureAnalyzer
)


MODEL_URI = "models:/m-52d83afbb16c499cb2cead392e0107c8"

OUTPUT_PATH = Path(
    "results/model_analysis/permutation_feature_importance.csv"
)


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

    print(f"Dataset original: {dataframe.shape}")

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

    X_train = train.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore"
    )

    y_train = train[TARGET_COLUMN].astype(int)

    X_validation = validation.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore"
    )

    y_validation = validation[TARGET_COLUMN].astype(int)

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    original_model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    model = clone(original_model)

    print("\nEntrenando modelo con 64,000 registros...")

    model.fit(X_train, y_train)

    predictions = model.predict_proba(
        X_validation
    )[:, 1]

    baseline_auc = roc_auc_score(
        y_validation,
        predictions
    )

    print(f"\nValidation ROC-AUC: {baseline_auc:.6f}")

    analyzer = PermutationFeatureAnalyzer(
        n_repeats=5,
        random_state=42
    )

    importance_df = analyzer.analyze(
        model=model,
        X_validation=X_validation,
        y_validation=y_validation
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\nTop 20 variables por importancia:")
    print(
        importance_df.head(20).to_string(index=False)
    )

    print(
        f"\nResultados guardados en: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()