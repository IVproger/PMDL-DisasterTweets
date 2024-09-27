import zenml
from zenml.client import Client
import pandas as pd


def load_features(X: pd.DataFrame, y: pd.DataFrame, version: str) -> None:
    """
    Save the features and target as artifact.
    """
    zenml.save_artifact(data=[X,y], name="features", tags=[version])

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