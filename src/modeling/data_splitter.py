import pandas as pd

from sklearn.model_selection import train_test_split

from src.config.model_config import TARGET_COLUMN


class ChurnDataSplitter:

    def __init__(
        self,
        test_size: float = 0.20,
        validation_size: float = 0.20,
        random_state: int = 42
    ):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

    def split(self, dataframe: pd.DataFrame) -> dict:

        if TARGET_COLUMN not in dataframe.columns:
            raise ValueError(
                f"No se encontró la variable {TARGET_COLUMN}"
            )

        row_hash = pd.util.hash_pandas_object(
            dataframe,
            index=False
        )

        ordered_data = (
            dataframe
            .assign(_row_hash=row_hash)
            .sort_values("_row_hash", kind="stable")
            .drop(columns="_row_hash")
            .reset_index(drop=True)
        )

        train_validation, test = train_test_split(
            ordered_data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=ordered_data[TARGET_COLUMN]
        )

        train, validation = train_test_split(
            train_validation,
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=train_validation[TARGET_COLUMN]
        )

        return {
            "train": train.reset_index(drop=True),
            "validation": validation.reset_index(drop=True),
            "test": test.reset_index(drop=True)
        }