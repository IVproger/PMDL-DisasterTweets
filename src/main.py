from src.utils import init_hydra
from src.data import preprocess_data
import pandas as pd
import os

def make_prediction_ml(model_name: str, text: str) -> str:
    """
    Make a prediction whether the input text is a real disaster or not.

    Args:
        model_name (str): The name of the model file.
        user_text (str): The raw input text.

    Returns:
        str: The prediction result ("Real Disaster" or "Not").
    """
    # Make the pd.DataFrame from the user input
    user_df = pd.DataFrame([text], columns=['text'])
    
    # Preprocess the data
    user_embeddings = preprocess_data(user_df)
    
    # Load the model configuration
    cfg = init_hydra("models_description.yaml")
    models_dir = cfg.production.models_path
    model_path = os.path.join(models_dir, model_name)
    
    # Read the model from the pkl file
    model = pd.read_pickle(model_path)
    
    # Make the prediction
    prediction = model.predict(user_embeddings)
    
    # Return the prediction 
    return prediction[0]

def make_prediction_dl(model_name: str, text: str) -> str:
    """
    Make a prediction whether the input text is a real disaster or not using a deep learning model.

    Args:
        model_name (str): The name of the model file.
        user_text (str): The raw input text.

    Returns:
        str: The prediction result ("Real Disaster" or "Not").
    """
    pass
