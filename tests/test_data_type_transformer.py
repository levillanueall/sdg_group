
import unittest
from decimal import Decimal

from src.transformations.data_type_transformer import DataTypeTransformer


class TestDataTypeTransformer(unittest.TestCase):

    def setUp(self):
        self.transformer = DataTypeTransformer()

    def test_numeric(self):
        self.assertEqual(
            self.transformer.to_numeric("23,9975"),
            Decimal("23.9975")
        )

    def test_scientific_notation(self):
        self.assertEqual(
            self.transformer.to_numeric("7,11E-15"),
            Decimal("7.11E-15")
        )

    def test_integer(self):
        self.assertEqual(
            self.transformer.to_integer("61"),
            61
        )

    def test_missing_value(self):
        self.assertIsNone(
            self.transformer.to_numeric("")
        )

    def test_invalid_numeric(self):
        with self.assertRaises(ValueError):
            self.transformer.to_numeric("ABC")

    def test_invalid_integer(self):
        with self.assertRaises(ValueError):
            self.transformer.to_integer("23,5")

    def test_text(self):
        self.assertEqual(
            self.transformer.to_text("  CHICAGO AREA  "),
            "CHICAGO AREA"
        )


if __name__ == "__main__":
    unittest.main()
