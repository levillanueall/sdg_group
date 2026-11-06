
from src.config.model_config import FEATURE_TYPES
from decimal import Decimal


class FeatureEngineering:
    
    @staticmethod
    def _safe_ratio(numerator, denominator):
        if (
            numerator is None
            or denominator is None
            or denominator <= 0
        ):
            return None

        return numerator / denominator

    def transform_row(self, row: dict) -> dict:

        features = row.copy()

        # 1. Antigüedad del equipo relativa a la permanencia
        eqpdays = row["eqpdays"]
        months = row["months"]

        if (
            eqpdays is not None
            and eqpdays >= 0
            and eqpdays != 99999
            and months is not None
        ):
            features["device_tenure_ratio"] = self._safe_ratio(
                eqpdays,
                months * Decimal("30.44")
            )
        else:
            features["device_tenure_ratio"] = None

        # 2. Tendencia de consumo
        usage_ratio = self._safe_ratio(
            row["avg3mou"],
            row["avg6mou"]
        )

        features["usage_trend"] = (
            usage_ratio - 1
            if usage_ratio is not None
            else None
        )

        # 3. Proporción de llamadas fallidas
        features["failed_call_rate"] = self._safe_ratio(
            row["drop_blk_Mean"],
            row["attempt_Mean"]
        )

        # 4. Proporción de ingresos por excedentes
        features["overage_revenue_ratio"] = self._safe_ratio(
            row["ovrrev_Mean"],
            row["rev_Mean"]
        )

        # 5. Duración promedio de llamadas a soporte
        features["care_avg_duration"] = self._safe_ratio(
            row["cc_mou_Mean"],
            row["custcare_Mean"]
        )

        # 6. Tendencia de facturación
        revenue_ratio = self._safe_ratio(
            row["avg3rev"],
            row["avg6rev"]
        )

        features["revenue_trend"] = (
            revenue_ratio - 1
            if revenue_ratio is not None
            else None
        )

        return features

    def transform_batch(self, batch: list[dict]) -> list[dict]:

        return [
            self.transform_row(row)
            for row in batch
        ]
