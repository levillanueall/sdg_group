
from decimal import Decimal, InvalidOperation


class DataTypeTransformer:

    DESCRIPTION_VARIABLE_ALIASES = {
        "Rev_Mean": "rev_Mean",
        "text_formatnew_cell": "new_cell",
        "text_formatcrclscod": "crclscod",
        "checkasl_flag": "asl_flag",
        "text_formatprizm_social_one": "prizm_social_one",
        "text_formatarea": "area",
        "text_formatdualband": "dualband",
        "text_formatrefurb_new": "refurb_new",
        "text_formathnd_webcap": "hnd_webcap",
        "text_formatownrent": "ownrent",
        "text_formatdwlltype": "dwlltype",
        "text_formatmarital": "marital",
        "text_formatinfobase": "infobase",
        "text_formatHHstatin": "HHstatin",
        "text_formatdwllsize": "dwllsize",
        "text_formatethnic": "ethnic",
        "text_formatkid0_2": "kid0_2",
        "text_formatkid3_5": "kid3_5",
        "text_formatkid6_10": "kid6_10",
        "text_formatkid11_15": "kid11_15",
        "text_formatkid16_17": "kid16_17",
        "checkcreditcd": "creditcd",
        "vpn_keyCustomer_ID": "Customer_ID",
    }

    def normalize_description_variable(self, variable: str) -> str:
        return self.DESCRIPTION_VARIABLE_ALIASES.get(variable, variable)
    
    STRING_DESCRIPTIONS = {
        "new cell phone user",
        "credit class code",
        "account spending limit",
        "social group letter only",
        "geogrpahic area",
        "dualband",
        "handset: refurbished or new",
        "handset web capability",
        "home owner/renter status",
        "dwelling unit type",
        "marital status",
        "infobase match",
        "premier household status indicator",
        "dwelling size",
        "ethnicity roll-up code",
        "child 0 - 2 years of age in household",
        "child 3 - 5 years of age in household",
        "child 6 - 10 years of age in household",
        "child 11 - 15 years of age in household",
        "child 16 - 17 years of age in household",
        "credit card indicator",
        "customer id",
    }

    def get_silver_column_definition(
        self,
        variable: str,
        meaning: str
    ) -> dict[str, str]:
        normalized_meaning = " ".join(meaning.casefold().split())

        column_type = (
            "string"
            if normalized_meaning in self.STRING_DESCRIPTIONS
            else "numeric"
        )

        return {
            "column": self.normalize_description_variable(variable),
            "type": column_type,
        }

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
    
    def transform_row(
        self,
        row: dict,
        column_types: dict[str, str]
    ) -> dict:

        transformed_row = {}

        for column, value in row.items():
            column_type = column_types[column]

            if column_type == "string":
                transformed_row[column] = self.to_text(value)

            elif column_type == "numeric":
                transformed_row[column] = self.to_numeric(value)

            else:
                raise ValueError(
                    f"Tipo no soportado para {column}: {column_type}"
                )

        return transformed_row
