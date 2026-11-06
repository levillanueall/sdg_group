import pandas as pd

from sklearn.model_selection import train_test_split

from src.config.model_config import TARGET_COLUMN, ID_COLUMN


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


        # Validar el identificador del cliente
        if ID_COLUMN not in dataframe.columns:
            raise ValueError(
                f"No existe la columna {ID_COLUMN}"
            )

        if dataframe[ID_COLUMN].isna().any():
            raise ValueError(
                "Existen identificadores nulos."
            )

        if dataframe[ID_COLUMN].duplicated().any():
            raise ValueError(
                "Existen identificadores duplicados."
            )

        # Ordenamiento estable por Customer_ID
        ordered_data = (
            dataframe
            .sort_values(ID_COLUMN, kind="stable")
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