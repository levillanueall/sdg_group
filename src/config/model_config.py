
TARGET_COLUMN = "churn"

ID_COLUMN = "Customer_ID"

EXCLUDED_COLUMNS = [
    TARGET_COLUMN,
    ID_COLUMN
]

FEATURE_COLUMNS = None

FEATURE_TYPES = {
    "device_tenure_ratio": "numeric",
    "usage_trend": "numeric",
    "failed_call_rate": "numeric",
    "overage_revenue_ratio": "numeric",
    "care_avg_duration": "numeric",
    "revenue_trend": "numeric",
}
