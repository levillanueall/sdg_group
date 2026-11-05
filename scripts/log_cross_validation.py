
from pathlib import Path

from src.tracking.mlflow_cross_validation_tracker import (
    MLflowCrossValidationTracker
)


RESULTS_DIR = Path("results/model_analysis")


def main():

    # 1. Inicializar el tracker
    tracker = MLflowCrossValidationTracker(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="SDG_Churn_Model_Comparison"
    )

    # 2. Registrar experimentos existentes
    logged_runs = tracker.log_from_csv(
        results_dir=RESULTS_DIR
    )

    # 3. Mostrar los experimentos registrados
    print("\n--- EXPERIMENTOS REGISTRADOS ---")

    for run in logged_runs:

        print(
            f"\nExperimento: {run['experiment']}"
            f"\nRun ID: {run['run_id']}"
            f"\nROC-AUC: {run['roc_auc_mean']:.6f}"
        )

    print("\nRegistro de experimentos finalizado.")


if __name__ == "__main__":
    main()
