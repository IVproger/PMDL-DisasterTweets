import zenml
from zenml.client import Client
import pandas as pd
from omegaconf import OmegaConf
from src.transformations import feature_extractor, clear_columns, generate_embeddings
from src.utils import init_hydra
import numpy as np
from sklearn.pipeline import Pipeline
import os

def read_datastore() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Read the sample data.
    """
    cfg = OmegaConf.load("configs/data_description.yaml")
    version = open("configs/data_version.txt", "r").read().strip()
    
    train_path = cfg.data.sample_train_path
    test_path = cfg.data.sample_test_path

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, test_data, version
    

def load_features(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, version: str, name="features_target_train_test") -> None:
    """
    Save the features_target and target as artifact.
    """
    zenml.save_artifact(data=[X_train, y_train, X_test], name=name, tags=[version])

def fetch_features(name: str, version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch the features and target from the artifact store.

    Args:
        name (str): The name of the artifact.
        version (str): The version of the artifact.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): The features DataFrame.
            - y (pd.DataFrame): The target DataFrame.
    """
    
    client = Client()
    lst = client.list_artifact_versions(name=name, tag=version, sort_by="version").items
    lst.reverse()
    
    # Load the latest version of artifact
    artifact = lst[0].load()
    X_train = artifact[0]
    y_train = artifact[1]
    X_test = artifact[2]
        
    return X_train, y_train, X_test

def preprocess_data(X: pd.DataFrame) -> np.ndarray:
    """
    Preprocess the data using a pipeline of transformations.

    Args:
        X (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    all_features = X.columns.tolist()

    pipeline = Pipeline([
        ('extract_features', feature_extractor(features=all_features)),
        ('clean_columns', clear_columns(features=all_features)),
    ])

    transformed_df = pipeline.fit_transform(X)
    
    cfg = init_hydra('embeders_description')
    
    embeddings = generate_embeddings(X=transformed_df, embeder_path=cfg.production.embeder_path,column_name='text')
    
    return embeddings

    