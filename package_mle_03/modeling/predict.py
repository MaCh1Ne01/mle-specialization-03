from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from package_mle_03.utils.helpers import *

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_test_data.joblib",
    fitted_models_path: Path = MODELS_DIR
):

    # ---- Loading test datasets----
    logger.info("Loading datasets to predict...")
    train_test_data = joblib.load(features_path)
    X_test = train_test_data["X_test"]
    logger.success("Loading datasets to predict complete.")
    # -----------------------------------------

    # ---- Loading fitted models ----
    logger.info("Loading fitted models...")
    model_01 = joblib.load(fitted_models_path/f"{MODEL_01_NAME}_model.joblib")
    model_02 = joblib.load(fitted_models_path/f"{MODEL_02_NAME}_model.joblib")
    model_03 = joblib.load(fitted_models_path/f"{MODEL_03_NAME}_model.joblib")
    model_04 = joblib.load(fitted_models_path/f"{MODEL_04_NAME}_model.joblib")
    logger.success("Loading fitted models complete.")
    # -----------------------------------------

    # ---- Predicting ----
    logger.info("Predicting...")
    predicting_regression_model(model=model_01, model_name=MODEL_01_NAME, X=X_test, init_path=".")
    predicting_regression_model(model=model_02, model_name=MODEL_02_NAME, X=X_test, init_path=".")
    predicting_regression_model(model=model_03, model_name=MODEL_03_NAME, X=X_test, init_path=".")
    predicting_regression_model(model=model_04, model_name=MODEL_04_NAME, X=X_test, init_path=".")
    logger.success("Predictions complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
