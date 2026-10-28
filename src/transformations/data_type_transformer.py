
from decimal import Decimal, InvalidOperation


class DataTypeTransformer:

    def to_numeric(self, value: str):
        if value is None or not value.strip():
            return None

        normalized = value.strip().replace(",", ".")

        try:
            return Decimal(normalized)
        except InvalidOperation:
            raise ValueError(
                f"Valor numérico inválido: {value}"
            )

    def to_integer(self, value: str):
        numeric_value = self.to_numeric(value)

        if numeric_value is None:
            return None

        if numeric_value != numeric_value.to_integral_value():
            raise ValueError(
                f"El valor no es un entero: {value}"
            )

        return int(numeric_value)

    def to_text(self, value: str):
        if value is None:
            return None

        return value.strip()
