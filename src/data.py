import zenml
from zenml.client import Client
import pandas as pd
from omegaconf import OmegaConf
import os
import numpy as np

def read_datastore(train=True) -> tuple[pd.DataFrame, str]:
    """
    Read the sample data.
    """
    cfg = OmegaConf.load("configs/data_description.yaml")
    version = open("configs/data_version.txt", "r").read().strip()
    
    data_path = os.path.join('', cfg.data.sample_train_path if train else cfg.data.sample_test_path)
    data = pd.read_csv(data_path)
    
    return data, version
    

def load_features(X: np.ndarray, y: pd.Series, version: str) -> None:
    """
    Save the features_target and target as artifact.
    """
    zenml.save_artifact(data=[X,y], name="features_target", tags=[version])

def fetch_features(name: str, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    X = artifact[0]
    y = artifact[1]
        
    return X, y