import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
import uuid
import re
import time
import joblib
from datetime import datetime
from typing import List, Dict
import logging
from mlflow.data import from_pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, OPTICS, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClusterMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn_genetic import GASearchCV
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn_genetic.space import Categorical, Integer, Continuous
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from constants import *

def renaming_features(dataframe:pd.DataFrame):
    dataframe.columns = dataframe.columns.str.replace(" ", "_")
    return dataframe


def extracting_temporary_features(dataframe:pd.DataFrame):
    if dataframe is not None:
        for c in DATE_FEATURES:
            dataframe[c] = pd.to_datetime(dataframe[c], utc=True)
            dataframe[f"{c}_day_of_week"] = dataframe[c].dt.dayofweek
            dataframe[f"{c}_month"] = dataframe[c].dt.month
            dataframe[f"{c}_quarter"] = dataframe[c].dt.quarter
            dataframe[f"{c}_is_month_end"] = dataframe[c].dt.is_month_end.astype(int)
            dataframe[f"{c}_day_of_year"] = dataframe[c].dt.dayofyear
            
            dataframe[f"{c}_month_sin"] = np.sin(2 * np.pi * dataframe[f"{c}_month"] / 12)
            dataframe[f"{c}_month_cos"] = np.cos(2 * np.pi * dataframe[f"{c}_month"] / 12)
            dataframe[f"{c}_day_sin"] = np.sin(2 * np.pi * dataframe[f"{c}_day_of_week"] / 7)
            dataframe[f"{c}_day_cos"] = np.cos(2 * np.pi * dataframe[f"{c}_day_of_week"] / 7)
    
            dataframe.drop(c, axis=1, inplace=True)
        print(f"Features {DATE_FEATURES} were replaced by 9 temporary features each.")
    else:
        print("The dataset is empty.")


def transforming_date_features(dataframe:pd.DataFrame):
    for f in DATE_FEATURES_TRANSFORM:
        if f[1] == "years":
            dataframe[f[0]] = pd.to_datetime(CUTOFF_DATE).year - dataframe[f[0]]
        elif f[1] == "days":
            dataframe[f[0]] = (pd.to_datetime(CUTOFF_DATE) - pd.to_datetime(dataframe[f[0]])).dt.days
    for key, value in DATE_FEATURES_RENAMED.items():
        print(f"Column {key} renamed to: {value}")    
    dataframe.rename(columns=DATE_FEATURES_RENAMED, inplace=True)


def casting_numerical_features(dataframe:pd.DataFrame):
    for f in OBJECT_TO_NUMERICAL_FEATURES:
        dataframe[f] = dataframe[f].astype(float)
    print(f"Casting to float type the columns: {OBJECT_TO_NUMERICAL_FEATURES}")
    return dataframe


def stripping_object_features(dataframe:pd.DataFrame):
    for f in INITIAL_OBJECT_FEATURES:
        dataframe[f] = dataframe[f].apply(lambda x: str(x).strip())
    print(f"Stripping the columns: {INITIAL_OBJECT_FEATURES}")
    return dataframe


def dropping_invalid_year_birth_rows(dataframe:pd.DataFrame):
    new_dataframe = dataframe[~dataframe["Year_Birth"].apply(lambda x: not re.match(r'^[12]', str(x).strip()))]
    print(f"Invalid rows dropped: {dataframe.shape[0] - new_dataframe.shape[0]}")
    return new_dataframe


def graphing_correlation_matrix(dataframe:pd.DataFrame, zoom:int=1):
    plt.figure(figsize=(4*zoom, 3*zoom))
    sns.heatmap(
        dataframe.corr(),
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )
    plt.title("Correlation Matrix")

    
def object_features_report(dataframe:pd.DataFrame):
    type_features = dataframe.select_dtypes(include=["object"]).columns

    print("********************Unique Counts********************")
    uniq_counts = dataframe[type_features].nunique()
    print(uniq_counts)

    print("\n********************Unique Values********************")
    for f in type_features:
        uniq_values = dataframe[f].unique()
        print(f"Feature '{f}': {uniq_values}")

    print("\n********************Value Counts********************")
    for f in type_features:
        val_counts = dataframe[f].value_counts()
        print(f"Feature '{f}' values: {val_counts}\n")


def split_dataset(target_feature:str, dataframe:pd.DataFrame, test_percentage:float, seed:int, stratify_feature:str):
    if test_percentage != 0:
        X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_feature, axis=1),dataframe[target_feature],
                                                        test_size=test_percentage, random_state=seed,
                                                        stratify=dataframe[stratify_feature])
        
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")
        return X_train, X_test, y_train, y_test
    else:
        print(f"X_train: {dataframe.shape}")
        return dataframe, None, None, None


def temporal_split_dataset_by_entity(temporal_feature:str, dataframe:pd.DataFrame, test_percentage:float, stratify_feature:str, with_sample:bool):
    train_list, test_list, train_list_sample = [], [], []
    
    for s in dataframe[stratify_feature].unique():
        s_data = dataframe[dataframe[stratify_feature] == s].sort_values(temporal_feature)
        split_idx = int(len(s_data) * (1 - test_percentage))
        train_s = s_data.iloc[:split_idx]
        test_s = s_data.iloc[split_idx:]
        
        train_list.append(train_s)
        test_list.append(test_s)

        if with_sample:
            split_sample_idx = int(len(s_data) * SAMPLE_PERCENTAGE)
            train_sample_s = s_data.iloc[:split_sample_idx]
            train_list_sample.append(train_sample_s)

    if with_sample:
        X_train, X_test, y_train, y_test, X_train_sample, y_train_sample = pd.concat(train_list).drop(TARGET, axis=1), pd.concat(test_list).drop(TARGET, axis=1),\
                                            pd.concat(train_list)[TARGET], pd.concat(test_list)[TARGET], pd.concat(train_list_sample).drop(TARGET, axis=1), pd.concat(train_list_sample)[TARGET]
    else:
        X_train, X_test, y_train, y_test = pd.concat(train_list).drop(TARGET, axis=1), pd.concat(test_list).drop(TARGET, axis=1),\
                                            pd.concat(train_list)[TARGET], pd.concat(test_list)[TARGET]

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    if with_sample:
        print(f"X_train_sample: {X_train_sample.shape}")
        print(f"y_train_sample: {y_train_sample.shape}")

    if with_sample:
        return X_train, X_test, y_train, y_test, X_train_sample, y_train_sample
    else:
        return X_train, X_test, y_train, y_test, None, None


def safe_target_encoding_nominal_features(features_train:pd.DataFrame, target_train:pd.Series, features_test:pd.DataFrame, features_train_sample:pd.DataFrame, nominal_features:List, alpha:int):
    """
    Target encoding with smoothing
    """
    global_mean = target_train.mean()
    train_combined = features_train.copy()
    train_combined[TARGET] = target_train

    for f in nominal_features:
        group_stats = train_combined.groupby(f)[TARGET].agg(["mean", "count"])
        group_stats["encoded"] = (group_stats["mean"] * group_stats["count"] + 
                                 global_mean * alpha) / (group_stats["count"] + alpha)
        
        encoding_map = group_stats["encoded"].to_dict()     
        features_train[f] = features_train[f].map(encoding_map)
        features_test[f] = features_test[f].map(encoding_map)      
        features_test[f] = features_test[f].fillna(global_mean)

        if features_train_sample is not None:
            features_train_sample[f] = features_train_sample[f].map(encoding_map) 

    print(f"Target Encoding with smoothing (alpha={alpha}) done.")

    return features_train, features_test, features_train_sample

    
def encoding_nominal_features(features_train:pd.DataFrame, features_test:pd.DataFrame, nominal_features:List):
    oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")

    oh_encoder.fit(features_train[nominal_features])
    X_train_encoded = oh_encoder.transform(features_train[nominal_features])

    if features_test:
        X_test_encoded = oh_encoder.transform(features_test[nominal_features])

    feature_names = oh_encoder.get_feature_names_out(input_features=nominal_features)
    X_train_df = pd.DataFrame(X_train_encoded, columns=feature_names)

    if features_test:
        X_test_df = pd.DataFrame(X_test_encoded, columns=feature_names)

    left_features = [f for f in features_train.columns if f not in nominal_features]
    X_train = pd.concat([features_train[left_features].reset_index(drop=True), X_train_df], axis=1)
    
    if features_test:
        X_test = pd.concat([features_test[left_features].reset_index(drop=True), X_test_df], axis=1)

    print("One Hot Encoding with handle_unknown='ignore' and drop='first' done.")

    if features_test:
        return X_train, X_test
    else:
        return X_train, None


def encoding_ordinal_features(features_train:pd.DataFrame, features_test:pd.DataFrame, ordinal_features:List, categories_list:Dict):
    ord_encoder = OrdinalEncoder(categories=categories_list, handle_unknown="use_encoded_value", unknown_value=-1)

    ord_encoder.fit(features_train[ordinal_features])
    features_train[ordinal_features] = ord_encoder.transform(features_train[ordinal_features])
    if features_test:
        features_test[ordinal_features]= ord_encoder.transform(features_test[ordinal_features])

    print("Ordinal Encoding with handle_unknown='use_encoded_value' and unknown_value=-1 done.")
    if features_test:
        return features_train, features_test
    else:
        return features_train, None


def scaling_numerical_features(method:str, features_train:pd.DataFrame, features_test:pd.DataFrame, features_train_sample:pd.DataFrame, features_to_scale:List):
    if method == "Min Max":
        scaler = MinMaxScaler()
    elif method == "Robust":
        scaler = RobustScaler()
    elif method == "Standard":
        scaler = StandardScaler()
    scaler.fit(features_train[features_to_scale])
    features_train[features_to_scale] = scaler.transform(features_train[features_to_scale])
    if features_test is not None:
        features_test[features_to_scale] = scaler.transform(features_test[features_to_scale])
    if features_train_sample is not None:
        features_train_sample[features_to_scale] = scaler.transform(features_train_sample[features_to_scale])
    print(f"{method} Scaling done.")
    
    return features_train, features_test, features_train_sample
    

def writing_feature_table(dataframe:pd.DataFrame, file_name:str, init_path:str):
    feature_table = dataframe.copy()
    if not feature_table.empty:
        feature_table[ID_FEATURE] = [str(uuid.uuid4()) for _ in range(feature_table.shape[0])]
        feature_table["event_timestamp"] = [datetime.now() for _ in range(feature_table.shape[0])]
        time.sleep(1)
        feature_table["created"] = [datetime.now() for _ in range(feature_table.shape[0])]
        feature_table.to_parquet(f"{init_path}{FEAST_REPOSITORY_PATH}data/{file_name}.parquet", index=False)
        print(f"Feature Table in {FEAST_REPOSITORY_PATH}data/{file_name}.parquet")
        return feature_table[ID_FEATURE].tolist()
    else:
        raise Exception("Feature table doesn't exist.")


def predicting_regression_model(model:any, model_name:str, X:pd.DataFrame, init_path:str=".."):
    y_pred = model.predict(X)
    print(f"{model_name} model: ({X.shape[0]} values were predicted.")
    pd.DataFrame(y_pred, columns=[TARGET]).to_csv(f"{init_path}/data/processed/{model_name}_predictions.csv", index=False)
    return y_pred


def executing_and_saving_clustering_model(model:any, model_name:str, X:pd.DataFrame, label_data:str):   
    with mlflow.start_run(run_name=f"{model_name} Model Run") as run:
        labels = model.fit_predict(X)
        mlflow.log_params(model.get_params())
        joblib.dump(model, f"../models/{model_name}_model.joblib")
        # Filtering only clusterized points
        valid_mask = labels != -1
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        if len(np.unique(labels_valid)) > 1:  # At least 2 clusters are required
            silhouette = silhouette_score(X_valid, labels_valid)
            calinski_harabasz = calinski_harabasz_score(X_valid, labels_valid)
            davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
            mlflow.log_metrics(
                {
                    "Silhouette Score": silhouette,
                    "Calinski-Harabasz Score": calinski_harabasz,
                    "Davies-Bouldin Score:": davies_bouldin
                }   
            )
        else:
            raise ValueError("Only 1 cluster exists.")
    print(f"**********{model_name} Metrics ({label_data}):**********")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
    return silhouette, calinski_harabasz, davies_bouldin, labels


def executing_and_saving_regression_model(model:any, model_name:str, X:pd.DataFrame, y:pd.DataFrame, X_sample:pd.DataFrame, y_sample:pd.DataFrame,\
                                          X_test:pd.DataFrame, y_test:pd.DataFrame, label_data:str, search_space:dict, init_path:str=".."):   
    with mlflow.start_run(run_name=f"{model_name} Model Run") as run:
        logging.getLogger("mlflow").setLevel(logging.CRITICAL)
        
        if search_space is not None:
            evolution_regressor = GASearchCV(
                estimator=model,
                scoring="neg_root_mean_squared_error",
                param_grid=search_space,
                population_size=4,
                generations=4,
                cv=3,
                n_jobs=1,
                verbose=True,
            )
            evolution_regressor.fit(X_sample,y_sample)
            model = evolution_regressor.best_estimator_
            model.fit(X,y)
        else:
            model.fit(X,y)
            
        if model_name.lower().split()[0] == "lightgbm":
            mlflow.lightgbm.log_model(model, f"{model_name}")
        elif model_name.lower().split()[0] == "xgboost":
            mlflow.xgboost.log_model(model, f"{model_name}")
        else:
            mlflow.sklearn.log_model(model, f"{model_name}")
        mlflow.log_input(from_pandas(X, name=f"{label_data} Features"), context=f"{label_data}")
        y_pred = model.predict(X_test)
        mlflow.log_params(model.get_params())
        joblib.dump(model, f"{init_path}/models/{model_name}_model.joblib")
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        smape = 100 * np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred)))
        
        mlflow.log_metrics(
            {
                "Mean Absolute Error": mae,
                "Root Mean Squared Error": rmse,
                "Square R": r2,
                "Mean Absolute Percentage Error": mape,
                "Symmetric Mean Absolute Percentage Error": smape
            }   
        )

    print(f"**********{model_name} Metrics ({label_data}):**********")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Square R: {r2:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
    print(f"Symmetric Mean Absolute Percentage Error: {smape:.4f}")
    return (mae, rmse, r2, mape, smape), model


def getting_principal_components(X:pd.DataFrame):
    pca_model = PCA(n_components=2, random_state=SEED)
    return pd.DataFrame(pca_model.fit_transform(X),columns=["PC01", "PC02"])

def getting_K_elbow_method(dataframe:pd.DataFrame): 
    inertias = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=SEED)
        kmeans.fit(dataframe)
        inertias.append(kmeans.inertia_)  
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.xticks(k_range)
    plt.show()
    

def visualizing_regression_model_performance(model:any, model_name:str, y:pd.DataFrame, predictions:pd.DataFrame, label_data:str):
    plt.figure(figsize=(6, 4))
    plt.scatter(y, predictions, alpha=0.5, label=label_data, s=10)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", linewidth=2, label="Perfect Prediction")
    plt.xlabel("Real Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"{model_name} - Real vs Predicted Prices - {label_data}")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualizing_clustering_model_performance(model_name:str, components_dataframe:pd.DataFrame, labels:np.ndarray, label_data:str):
    sns.scatterplot(x="PC01", y="PC02", data=pd.concat([components_dataframe, pd.Series(labels, name='cluster')], axis=1), hue="cluster", palette="tab10")
    plt.title(f"{model_name} - Clustering - {label_data}")
    plt.legend()
    plt.show()

def graphing_regression_models_results(base_model_metrics:tuple, model_01_metrics:tuple, model_02_metrics:tuple):
    models_results = {
        "Model": [BASE_MODEL_NAME, MODEL_01_NAME, MODEL_02_NAME],
        "Root Mean Squared Error": [base_model_metrics[0], model_01_metrics[0], model_02_metrics[0]],
        "Squared R": [base_model_metrics[1], model_01_metrics[1], model_02_metrics[1]]
    }

    df_results = pd.DataFrame(models_results).sort_values("Root Mean Squared Error", ascending=False)

    plt.figure(figsize=(8, 4))
    barplot = sns.barplot(
        x="Root Mean Squared Error", 
        y="Model", 
        data=df_results,
        hue="Model",
        palette="Reds_r",
        legend=False,
        order=df_results["Model"].tolist(),
        hue_order=df_results["Model"].tolist()
    )

    for i, row in df_results.reset_index(drop=True).iterrows():
        plt.text(
            x=row["Root Mean Squared Error"] / 2,
            y=i,
            s=f"RMSE = {row['Root Mean Squared Error']:.4f}",
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )

    plt.title("Comparison of RMSE between models")
    plt.xlabel("Root Mean Squared Error (Lower is better)")
    plt.ylabel("Model")


    df_results = pd.DataFrame(models_results).sort_values("Squared R", ascending=False)

    plt.figure(figsize=(8, 4))
    barplot = sns.barplot(
        x="Squared R", 
        y="Model", 
        data=df_results,
        hue="Model",
        palette="Greens_r",
        legend=False,
        order=df_results["Model"].tolist(),        # Dataframe order
        hue_order=df_results["Model"].tolist()     # Same order to colors
    )

    for i, row in df_results.reset_index(drop=True).iterrows():
        plt.text(
            x=row["Squared R"] / 2,
            y=i,
            s=f"R² = {row['Squared R']:.4f}",
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )

    plt.title("Comparison of Squared R between models")
    plt.xlabel("Squared R (Closer to 1 is better)")
    plt.ylabel("Model")


def graphing_clustering_models_results(base_model_metrics:tuple, model_01_metrics:tuple, model_02_metrics:tuple):
    models_results = {
        "Model": [BASE_MODEL_NAME, MODEL_01_NAME, MODEL_02_NAME],
        "Silhouette Score": [base_model_metrics[0], model_01_metrics[0], model_02_metrics[0]],
        "Calinski-Harabasz Score": [base_model_metrics[1], model_01_metrics[1], model_02_metrics[1]],
        "Davies-Bouldin Score": [base_model_metrics[2], model_01_metrics[2], model_02_metrics[2]]
    }

    df_results = pd.DataFrame(models_results).sort_values("Silhouette Score", ascending=False)

    plt.figure(figsize=(8, 4))
    barplot = sns.barplot(
        x="Silhouette Score", 
        y="Model", 
        data=df_results,
        hue="Model",
        palette="Reds_r",
        legend=False,
        order=df_results["Model"].tolist(),
        hue_order=df_results["Model"].tolist()
    )

    for i, row in df_results.reset_index(drop=True).iterrows():
        plt.text(
            x=row["Silhouette Score"] / 2,
            y=i,
            s=f"{row['Silhouette Score']:.4f}",
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )

    plt.title("Comparison of Silhouette Score between models")
    plt.xlabel("Silhouette Score (Closer to 1 is better)")
    plt.ylabel("Model")


    df_results = pd.DataFrame(models_results).sort_values("Calinski-Harabasz Score", ascending=False)

    plt.figure(figsize=(8, 4))
    barplot = sns.barplot(
        x="Calinski-Harabasz Score", 
        y="Model", 
        data=df_results,
        hue="Model",
        palette="Greens_r",
        legend=False,
        order=df_results["Model"].tolist(),
        hue_order=df_results["Model"].tolist()
    )

    for i, row in df_results.reset_index(drop=True).iterrows():
        plt.text(
            x=row["Calinski-Harabasz Score"] / 2,
            y=i,
            s=f"{row['Calinski-Harabasz Score']:.4f}",
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )

    plt.title("Comparison of Calinski-Harabasz Score between models")
    plt.xlabel("Calinski-Harabasz Score (Higher is better)")
    plt.ylabel("Model")


    df_results = pd.DataFrame(models_results).sort_values("Davies-Bouldin Score", ascending=False)

    plt.figure(figsize=(8, 4))
    barplot = sns.barplot(
        x="Davies-Bouldin Score", 
        y="Model", 
        data=df_results,
        hue="Model",
        palette="Blues_r",
        legend=False,
        order=df_results["Model"].tolist(),
        hue_order=df_results["Model"].tolist()
    )

    for i, row in df_results.reset_index(drop=True).iterrows():
        plt.text(
            x=row["Davies-Bouldin Score"] / 2,
            y=i,
            s=f"{row['Davies-Bouldin Score']:.4f}",
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )

    plt.title("Comparison of Davies-Bouldin Score between models")
    plt.xlabel("Davies-Bouldin Score (Closer to 0 is better)")
    plt.ylabel("Model")


class KMeansGMMEnsemble(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.ensamble_model = None 
        
    def fit(self, X, y=None):
        """
        Aplying K-Means
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit_predict(X)
        
        """
        Aplying GMM initializating the means with K-Means
        """
        self.ensamble_model = GaussianMixture(
            n_components=self.n_clusters,
            means_init=kmeans.cluster_centers_,
            n_init=1, # Only need one initialization
            random_state=self.random_state
        )   
        self.ensamble_model.fit(X)
        return self

    def predict(self, X):
        if self.ensamble_model is None:
            raise ValueError("Must execute fit() first.")
        return self.ensamble_model.predict(X)
    
    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

    def get_params(self, deep=True):
        return {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "n_init": 1
        }
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class OPTICSKMeansEnsemble(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=None, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.optics_model_ = None
        self.kmeans_model_ = None
        self.labels_ = None
        self.core_sample_indices_ = None
        
    def fit(self, X, y=None):
        """
        Applying OPTICS to detect outliers
        """
        self.optics_model_ = OPTICS(min_samples=30, xi=0.005, min_cluster_size=0.15)
        optics_labels = self.optics_model_.fit_predict(X)
        non_outlier_mask = optics_labels != -1
        X_filtered = X[non_outlier_mask]
        self.core_sample_indices_ = np.where(non_outlier_mask)[0]
        
        """
        Applying K-Means on clusterized data
        """
        self.kmeans_model_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans_labels_filtered = self.kmeans_model_.fit_predict(X_filtered)
        self.labels_ = np.full(X.shape[0], -1)
        self.labels_[non_outlier_mask] = kmeans_labels_filtered
        outlier_mask = optics_labels == -1
        if np.any(outlier_mask):
            outlier_indices = np.where(outlier_mask)[0]
            # Calculating outlier distances to cluster centroids
            distances = self.kmeans_model_.transform(X[outlier_mask])
            # Setting to closest cluster
            closest_clusters = np.argmin(distances, axis=1)
            self.labels_[outlier_mask] = closest_clusters
        return self

    def predict(self, X):
        """
        To preserve consistency, DBSCAN detects outliers first and then, K-Means gets clusters
        """
        if self.optics_model_ is None or self.kmeans_model_ is None:
            raise ValueError("Must execute fit() first.")
        final_labels = self.kmeans_model_.predict(X)
        return final_labels
    
    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_
    
    def get_params(self):
        return {
            "optics_params": self.optics_model_.get_params(),
            "kmeans_params": self.kmeans_model_.get_params()
        }
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
        

class AgglomerativeKMeansEnsemble(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=None, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.agg_model_ = None
        self.kmeans_model_ = None
        self.labels_ = None
        
    def fit(self, X, y=None):
        """
        Applying Agglomerative Clustering
        """
        self.agg_model_ = AgglomerativeClustering(n_clusters=self.n_clusters)
        agg_labels = self.agg_model_.fit_predict(X)
        
        """
        Using Agglomerative labels as features for K-Means
        """
        X_with_agg = np.column_stack([X, agg_labels.reshape(-1, 1)])
        
        """
        Applying K-Means on new dataset
        """
        self.kmeans_model_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels_ = self.kmeans_model_.fit_predict(X_with_agg)
        
        return self

    def predict(self, X):
        """
        Predict using the trained ensemble
        """
        if self.agg_model_ is None or self.kmeans_model_ is None:
            raise ValueError("Must execute fit() first.")
        
        # First get Agglomerative predictions for new data
        agg_labels = self.agg_model_.fit_predict(X)
        
        # Create enhanced feature space
        X_with_agg = np.column_stack([X, agg_labels.reshape(-1, 1)])
        
        # Get final K-Means predictions
        return self.kmeans_model_.predict(X_with_agg)
    
    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_
    
    def get_params(self, deep=True):
        return {
            "agg_params": self.agg_model_.get_params(),
            "kmeans_params": self.kmeans_model_.get_params()
        }
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class RidgeModelStackingEnsemble(BaseEstimator, RegressorMixin):  
    def __init__(self, base_models, alpha=1.0, random_state=SEED):
        self.base_models = base_models
        self.alpha = alpha
        self.random_state = random_state
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)

        if not self.base_models:
            raise ValueError("There're not base models.")
        
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        self.meta_features_ = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(self.base_models):
            if hasattr(model, "predict"):
                self.meta_features_[:, i] = model.predict(X)
            else:
                raise ValueError(f"{i} model hasn't predict method.")
        
        self.meta_regressor_ = Ridge(
            alpha=self.alpha, 
            random_state=self.random_state
        )
        self.meta_regressor_.fit(self.meta_features_, y)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)

        if hasattr(self, "feature_names_in_"):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(self.base_models):
            meta_features[:, i] = model.predict(X)
            
        return self.meta_regressor_.predict(meta_features)
    
    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)
    
    def get_params(self, deep=True):
        return {
            "base_models": self.base_models,
            "alpha": self.alpha,
            "random_state": self.random_state
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
        

def setting_experiment():
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is not None:
            mlflow.set_experiment(EXPERIMENT_NAME)
            print(f"Experiment {EXPERIMENT_NAME} already exists. Using existing one.")
            print(f"ID: {experiment.experiment_id}")
        else:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            mlflow.set_experiment(EXPERIMENT_NAME)
            print(f"Using new experiment created: {EXPERIMENT_NAME}")
            print(f"ID: '{experiment_id}'")
    except MlflowException as e:
        print(f"Error setting experiment: {e}.")


def preparing_prophet_datasets(X_train:pd.DataFrame, y_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame, temporal_feature:str, target:str, objective_feature:str, objective_value:str):
    X_train[temporal_feature] = pd.to_datetime(X_train[temporal_feature], utc=True).dt.tz_localize(None)
    X_test[temporal_feature] = pd.to_datetime(X_test[temporal_feature], utc=True).dt.tz_localize(None)
    
    train_index = X_train[objective_feature]==objective_value
    ts_training_data = pd.concat([X_train[train_index][temporal_feature], y_train[train_index]], axis=1)
    ts_training_prophet = ts_training_data.rename(columns={temporal_feature: "ds", target: "y"})
    
    test_index = X_test[objective_feature]==objective_value
    ts_testing_data = pd.concat([X_test[test_index][temporal_feature], y_test[test_index]], axis=1)
    ts_testing_prophet = ts_testing_data.rename(columns={temporal_feature: "ds", target: "y"})

    return ts_training_data, ts_testing_data, ts_training_prophet, ts_testing_prophet


def printing_regression_model_metrics(model_name:str, real:pd.DataFrame, predictions:pd.DataFrame, label_data:str):
    mae = mean_absolute_error(real, predictions)
    rmse = np.sqrt(mean_squared_error(real, predictions))
    r2 = r2_score(real, predictions)
    mape = np.mean(np.abs((real - predictions) / real)) * 100
    smape = 100 * np.mean(2 * np.abs(predictions - real) / (np.abs(real) + np.abs(predictions)))
    
    print(f"**********{model_name} Metrics ({label_data}):**********")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Square R: {r2:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
    print(f"Symmetric Mean Absolute Percentage Error: {smape:.4f}")


def graphing_temporal_serie(model:any, model_name:str, predictions_detail:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame, y_pred:pd.DataFrame, temporal_feature:str, target:str):
    model.plot(predictions_detail)
    plt.scatter(X_test, y_test, color="green", s=10, label="Test Real")
    plt.scatter(X_test, y_pred, color="orange", s=10, label="Test Predicted")
    plt.title(f"{model_name} Forecasting", fontsize=14, fontweight="bold")
    plt.xlabel(temporal_feature)
    plt.ylabel(target)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()


def graphing_temporal_series_components(model:any, model_name:str, predictions_detail:pd.DataFrame):
    model.plot_components(predictions_detail)
    plt.suptitle(f"{model_name} Temporal Serie Components", fontweight="bold")
    plt.tight_layout()
    plt.show()