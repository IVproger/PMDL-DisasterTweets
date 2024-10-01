from typing import Annotated
import pandas as pd
import numpy as np
from zenml import pipeline, step, ArtifactConfig
from src.transformations import feature_extractor, clear_columns, merge_columns, generate_embeddings
from src.data import read_datastore, load_features, fetch_features
from src.utils import init_hydra
from sklearn.pipeline import Pipeline

@step(enable_cache=False)
def extract() -> (
    tuple[
        Annotated[
            pd.DataFrame,
            ArtifactConfig(name="extracted_data", tags=["data_preparation"]),
        ],
        Annotated[str, ArtifactConfig(name="data_version", tags=["data_preparation"])],
    ]
):
    """
    Extract the sample data from the data store, and its version.
    """
    df, version = read_datastore()
    print("Extracted from datastore", version)
    return df, version

@step(enable_cache=False)
def transform(
    df: pd.DataFrame) -> (
    tuple[
        Annotated[
            pd.DataFrame,
            ArtifactConfig(name="transformed_data", tags=["data_preparation"]),
        ],
        Annotated[
            pd.Series, 
            ArtifactConfig(name="target", tags=["data_preparation"])],
    ]
    ):
    # Split to features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    all_features = X.columns.tolist()
    new_feature_name = 'merged_text'

    # Create the pipeline
    transformation_pipeline = Pipeline([
        ('extract_features', feature_extractor(features=all_features)),
        ('clean_columns', clear_columns(features=all_features)),
        ('merge_columns', merge_columns(features=all_features, new_feature_name=new_feature_name)),
    ])
    
    # Fit and transform the data 
    transformed_X = transformation_pipeline.fit_transform(X)
    print("Transformation completed successfuly")
    return transformed_X, y

@step(enable_cache=False)
def obtain_embeddings(
    X: pd.DataFrame, embeder_path: str, column_name: str
) -> (Annotated[
        np.ndarray, 
        ArtifactConfig(name="features_target", tags=["data_preparation"])
    ]):
    
    # Generate embeddings
    embeddings = generate_embeddings(embeder_path=embeder_path, X=X, column_name=column_name)
    print("Embeddings generation completed")
    return embeddings

@step(enable_cache=False)
def load(
    X: np.ndarray, y: pd.Series, version: str
) -> tuple[
    Annotated[np.ndarray, ArtifactConfig(name="embedings", tags=["data_preparation"])],
    Annotated[pd.Series, ArtifactConfig(name="target", tags=["data_preparation"])],
]:
    """
    Load the features and target as artifact.
    """
    load_features(X, y, version=version)
    print("Loaded into features store", version)
    return X, y

@pipeline()
def prepare_data_pipeline():
    df, version = extract()
    X, y = transform(df)
    cfg = init_hydra('embeders_description')
    X = obtain_embeddings(X=X, embeder_path=cfg.production.embeder_path,column_name='merged_text')
    X, y = load(X, y, version)


if __name__ == "__main__":
    run = prepare_data_pipeline()