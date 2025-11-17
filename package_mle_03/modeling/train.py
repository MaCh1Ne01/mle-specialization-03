from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from package_mle_03.utils.helpers import *
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_test_data.joblib",
    fitted_models_path: Path = MODELS_DIR / "fitted_models.joblib"
):
    
    # ---- Loading processed train datasets ----
    logger.info("Loading processed train datasets...")
    train_test_data = joblib.load(features_path)
    X_train = train_test_data["X_train"]
    X_test = train_test_data["X_test"]
    y_train = train_test_data["y_train"]
    y_test = train_test_data["y_test"]
    X_train_sample = train_test_data["X_train_sample"]
    y_train_sample = train_test_data["y_train_sample"]
    logger.success("Loading processed train datasets complete.")
    # -----------------------------------------

    # ---- DagsHub Integration ----
    logger.info("Connecting with DagsHub...")
    dagshub.init(repo_owner=REPOSITORY_OWNER, repo_name=REPOSITORY_NAME, mlflow=True)
    logger.success("Connection with DagsHub succesfully.")
    # -----------------------------------------

    # ---- MLflow Integration ----
    logger.info("Connecting with MLflow...")
    mlflow.set_tracking_uri(MLFLOW_DAGSHUB_URL)
    setting_experiment()
    mlflow.autolog(disable=True,)
    logger.success("Connection with MLflow succesfully.")
    # -----------------------------------------

    # ---- Models ----
    logger.info("Initializing models...")
    base_model = DummyRegressor(strategy="mean")
    model_01 = ElasticNet(random_state=SEED, max_iter=5000)
    search_space_model_01 = {
    "alpha": Continuous(0.001, 1.0, distribution="log-uniform"),
    "l1_ratio": Continuous(0.1, 0.9),
    "selection": Categorical(["cyclic", "random"])
    }
    model_02 = XGBRegressor(random_state=SEED, verbosity=0)
    search_space_model_02 = {
    "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
    "max_depth": Integer(3, 10),
    "n_estimators": Integer(50, 300),
    "subsample": Continuous(0.6, 1.0)
    }
    model_03 = LGBMRegressor(random_state=SEED, verbose=-1)
    search_space_model_03 = {
    "learning_rate": Continuous(0.01, 0.3, distribution="uniform"),
    "n_estimators": Integer(100, 500), 
    "num_leaves": Integer(31, 127)
    }
    model_04 = RidgeModelStackingEnsemble(base_models=[model_01,model_02,model_03], alpha=1.0, random_state=SEED)
    logger.success("Initializing models complete.")
    # -----------------------------------------

    # ---- Fitting and Saving Models ----
    logger.info("Fitting and Saving models...")
    executing_and_saving_regression_model(model=base_model, model_name=BASE_MODEL_NAME, X=X_train, y=y_train, X_sample=None, y_sample=None,\
                                                                       X_test=X_test, y_test=y_test, label_data=TESTING_DATA_LABEL, search_space=None, init_path=".")
    executing_and_saving_regression_model(model=model_01, model_name=MODEL_01_NAME, X=X_train, y=y_train, X_sample=X_train_sample, y_sample=y_train_sample,\
                                                         X_test=X_test, y_test=y_test, label_data=TESTING_DATA_LABEL, search_space=search_space_model_01, init_path=".")
    executing_and_saving_regression_model(model=model_02, model_name=MODEL_02_NAME, X=X_train, y=y_train, X_sample=X_train_sample, y_sample=y_train_sample,\
                                                         X_test=X_test, y_test=y_test, label_data=TESTING_DATA_LABEL, search_space=search_space_model_02, init_path=".")
    executing_and_saving_regression_model(model=model_03, model_name=MODEL_03_NAME, X=X_train, y=y_train, X_sample=X_train_sample, y_sample=y_train_sample,\
                                                         X_test=X_test, y_test=y_test, label_data=TESTING_DATA_LABEL, search_space=search_space_model_03, init_path=".")
    model_04 = RidgeModelStackingEnsemble(base_models=[model_01,model_02,model_03], alpha=1.0, random_state=SEED)
    executing_and_saving_regression_model(model=model_04, model_name=MODEL_04_NAME, X=X_train, y=y_train, X_sample=None, y_sample=None,\
                                                                   X_test=X_test, y_test=y_test, label_data=TESTING_DATA_LABEL, search_space=None, init_path=".")
    logger.success("Fitting and Saving models complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
