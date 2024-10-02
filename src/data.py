import zenml
from zenml.client import Client
import pandas as pd
from omegaconf import OmegaConf
import os
import numpy as np

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
    

def load_features(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, version: str) -> None:
    """
    Save the features_target and target as artifact.
    """
    zenml.save_artifact(data=[X_train, y_train, X_test], name="features_target_train_test", tags=[version])

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