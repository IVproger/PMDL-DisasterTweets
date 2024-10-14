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
            ArtifactConfig(name="extracted_data_train", tags=["data_preparation"]),
        ],
        Annotated[
            pd.DataFrame,
            ArtifactConfig(name="extracted_data_test", tags=["data_preparation"]),
        ],
        Annotated[str, ArtifactConfig(name="data_version", tags=["data_preparation"])],
    ]
):
    """
    Extract the sample data from the data store, and its version.
    """
    df_train, df_test, version = read_datastore()
    print("Extracted from datastore", version)
    return df_train, df_test, version

@step(enable_cache=False)
def transform(
    df: pd.DataFrame,
    train=True) -> (
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
        
    if train:
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
    else:
        X = df
        y = pd.Series()
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
    X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, version: str
) -> tuple[
    Annotated[np.ndarray, ArtifactConfig(name="embedings_train", tags=["data_preparation"])],
    Annotated[pd.Series, ArtifactConfig(name="target_train", tags=["data_preparation"])],
    Annotated[np.ndarray, ArtifactConfig(name="embedings_test", tags=["data_preparation"])]
]:
    """
    Load the features and target as artifact.
    """
    load_features(X_train, y_train, X_test, version=version)
    print("Loaded into features store", version)
    return X_train, y_train, X_test

@pipeline()
def prepare_data_pipeline():
    df_train, df_test, version = extract()
    
    X_train, y_train = transform(df_train)
    X_test, _ = transform(df_test,train=False)
    
    cfg = init_hydra('embeders_description')
    X_train = obtain_embeddings(X=X_train, embeder_path=cfg.production.embeder_path,column_name='merged_text')
    X_test = obtain_embeddings(X=X_test, embeder_path=cfg.production.embeder_path,column_name='merged_text')
    
    X_train, y_train, X_test = load(X_train, y_train, X_test, version)


if __name__ == "__main__":
    run = prepare_data_pipeline()