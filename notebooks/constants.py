TARGET = "Close"
CUTOFF_DATE = "2023-09-15"
TEST_PERCENTAGE = 0.2
SAMPLE_PERCENTAGE = 0.01
SEED = 2
DROP_FEATURES = []
SCALING_METHOD = "Standard"
REPOSITORY_OWNER = "MaCh1Ne01"
REPOSITORY_NAME = "mle-specialization-03"
MLFLOW_DAGSHUB_URL = "https://dagshub.com/MaCh1Ne01/mle-specialization-03.mlflow"
EXPERIMENT_NAME = "Companies Stock Forecasting - MLE 03 Project"
OBJECT_TO_NUMERICAL_FEATURES = ["Year_Birth","Teenhome"]
INITIAL_OBJECT_FEATURES = ["Date","Company"]
INITIAL_NUMERICAL_FEATURES = ["Open","High","Low","Close","Volume","Dividends","Stock_Splits"]
NOMINAL_FEATURES = ["Company"]
ORDINAL_FEATURES = ["Education"]
NUMERICAL_FEATURES = ["Customer_Age","Income","Kidhome","Teenhome","Customer_Tenure","Recency","MntWines","MntFruits","MntMeatProducts",\
                      "MntFishProducts","MntSweetProducts","MntGoldProds","NumDealsPurchases","NumWebPurchases","NumCatalogPurchases",\
                      "NumStorePurchases","NumWebVisitsMonth","AcceptedCmp2","Response"]
DATE_FEATURES_TRANSFORM = [("Year_Birth","years"),("Dt_Customer","days")]
DATE_FEATURES_RENAMED = {"Year_Birth": "Customer_Age", "Dt_Customer": "Customer_Tenure"}
DATE_FEATURES = ["Date"]
ID_FEATURE = "companies_stock_id"
EXCLUDED_FEATURES = [ID_FEATURE] + ["event_timestamp","created"]
STRATIFY_FEATURE = "Company"
CUSTOM_ORDER = {
    "Education": ["Basic","2n Cycle","Graduation","Master","PhD"]
}
FEAST_REPOSITORY_PATH = "/feast_service/fs_mle_03/feature_repo/"
BASE_MODEL_NAME = "Linear Regression"
MODEL_01_NAME = "LightGBM Regressor"
MODEL_02_NAME = "XGBoost Regressor"
MODEL_03_NAME = "Ridge Stacking Ensemble"
TRAINING_DATA_LABEL = "Training Data"
TESTING_DATA_LABEL = "Testing Data"