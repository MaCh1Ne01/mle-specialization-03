from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from feast import FeatureStore
from package_mle_03.utils.helpers import *

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "splitted_data.joblib",
    output_path: Path = PROCESSED_DATA_DIR / "train_test_data.joblib"
):
    
    # ---- Feature Engineering ----

    # ---- Loading splitted datasets ----
    logger.info("Loading datasets...")
    splitted_data = joblib.load(input_path)
    X_train = splitted_data["X_train"]
    y_train = splitted_data["y_train"]
    X_test = splitted_data["X_test"]
    y_test = splitted_data["y_test"]
    X_train_sample = splitted_data["X_train_sample"]
    y_train_sample = splitted_data["y_train_sample"]
    logger.success("Loading datasets complete.")
    # -----------------------------------------

    # ---- Date Features ----
    logger.info("Date Features...")
    extracting_temporary_features(dataframe=X_train)
    extracting_temporary_features(dataframe=X_test)
    extracting_temporary_features(dataframe=X_train_sample)
    logger.success("Date Features transformation complete.")
    # -----------------------------------------

    # ---- Nominal Encoding ----
    logger.info("Nominal Encoding...")
    X_train, X_test, X_train_sample = safe_target_encoding_nominal_features(features_train=X_train, target_train=y_train, features_test=X_test,\
                                                                        features_train_sample=X_train_sample, nominal_features=NOMINAL_FEATURES, alpha=10)
    logger.success("Nominal Encoding complete.")
    # -----------------------------------------

    # ---- Scaling ----
    logger.info("Scaling...")
    X_train, X_test, X_train_sample = scaling_numerical_features(method=SCALING_METHOD, features_train=X_train, features_test=X_test,\
                                                                 features_train_sample=X_train_sample, features_to_scale=X_train.columns)
    logger.success("Scaling complete.")
    # -----------------------------------------

    # ---- Serving features with Feast ----
    logger.info("Serving features with Feast...")
    """
    Simulating the Team 1 feature load
    """
    ids_to_retrieve_features = writing_feature_table(dataframe=pd.concat([X_train, y_train], axis=1), file_name="companies_stock_feature_table",\
                                                     init_path=".")
    logger.success("Serving features complete.")
    # -----------------------------------------

    # ---- Retrieving features with Feast ----
    logger.info("Retrieving features with Feast...")
    """
    Simulating the Team 2 feature retrieve
    """
    fs = FeatureStore("."+FEAST_REPOSITORY_PATH)
    entity_df = pd.DataFrame.from_dict({ID_FEATURE: ids_to_retrieve_features, "event_timestamp": [pd.Timestamp.now()] * len(ids_to_retrieve_features)})
    total_features = fs.get_historical_features(entity_df=entity_df, features=fs.get_feature_service("companies_stock_feature_service")).to_df()
    y_train = total_features[TARGET]
    X_train = total_features.drop([TARGET], axis=1)
    logger.success("Retrieving features complete.")
    # -----------------------------------------

    # ---- Saving processed datasets ----
    logger.info("Saving artifacts...")
    processed_data = {
        "X_train": X_train.drop(columns=EXCLUDED_FEATURES),
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_sample": X_train_sample,
        "y_train_sample": y_train_sample
    }
    joblib.dump(processed_data, output_path)
    logger.success("Saving artifacts complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
