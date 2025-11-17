from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from package_mle_03.utils.helpers import *

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "stock_details_5_years.csv",
    output_path: Path = PROCESSED_DATA_DIR / "splitted_data.joblib"
):

    # ---- Loading and Cleaning ----
    logger.info("Loading dataset...")
    df_companies_stock_raw = pd.read_csv(input_path)
    logger.success("Loading dataset complete.")

    logger.info("Cleaning dataset...")
    df_companies_stock = df_companies_stock_raw.copy()
    df_companies_stock.drop(DROP_FEATURES, axis=1, inplace=True)
    df_companies_stock = renaming_features(dataframe=df_companies_stock)
    df_companies_stock = stripping_object_features(dataframe=df_companies_stock)
    logger.success("Cleaning dataset complete.")
    # -----------------------------------------

    # ---- Data Splitting ----
    logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test, X_train_sample, y_train_sample = temporal_split_dataset_by_entity(temporal_feature="Date",\
                                                                                                        dataframe=df_companies_stock,\
                                                                                                    test_percentage=TEST_PERCENTAGE,\
                                                                                                        stratify_feature=STRATIFY_FEATURE,\
                                                                                                            with_sample=True)
    logger.success("Splitting dataset complete.")
    # -----------------------------------------

    # ---- Saving datasets ----
    logger.info("Saving datasets...")
    splitted_data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_sample": X_train_sample,
        "y_train_sample": y_train_sample
    }
    joblib.dump(splitted_data, output_path)
    logger.success("Saving datasets complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
