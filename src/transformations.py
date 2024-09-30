from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import demoji
import nltk

# TODO: the resulted merged text has a white space at the beginning, need to remove it

# --- Download resources only once at the top-level ---

# Ensure that necessary NLTK and demoji resources are available
nltk.download('punkt', quiet=True)  # Tokenizer
nltk.download('stopwords', quiet=True)  # Stopwords
stopword = set(stopwords.words('english'))  # Loaded once and cached
stemmer = SnowballStemmer('english')
if not demoji.last_downloaded_timestamp():
    demoji.download_codes()  # Only download codes once, skip if already done

# --- Feature Extractor Transformer ---

class feature_extractor(BaseEstimator, TransformerMixin):
    """
    Extracts specific columns from a DataFrame.

    Args:
        features: List of feature names to extract from the DataFrame.
    """
    def __init__(self, features: list[str]):
        self.features = features

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        # Keep only the specified columns
        existing_features = [feature for feature in self.features if feature in X.columns]
        X_transformed = X[existing_features].copy()
        return X_transformed


# --- Clear Columns Transformer ---
class clear_columns(BaseEstimator, TransformerMixin):
    """
    Cleans the specified text columns in a DataFrame by applying several cleaning steps:
    - Removes non-alphabetic characters.
    - Removes short words (less than 3 characters).
    - Removes URLs.
    - Replaces emoji with descriptive text.
    - Removes special characters like hashtags and HTML tags.
    - Applies stemming and removes stopwords.

    Args:
        features: List of features to clean.
    """
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, data):
        # Apply cleaning to each feature in the list
        for feature in self.features:
            if feature in data.columns:
                data[feature] = data[feature].apply(self.clean_text)
        return data

    def clean_text(self, text):
        """Cleans a single text entry using multiple regex and NLP techniques."""
        if pd.isnull(text):
            return text  # Return if null value

        # Convert to string if not already
        text = str(text)

        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove short words (1-2 characters)
        text = re.sub(r'\b\w{1,2}\b', '', text)
        # Remove URLs
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', '', text)
        # Replace emojis with their textual description
        text = demoji.replace_with_desc(text)
        # Remove @mentions and #hashtags
        text = re.sub(r'[@|#][^\s]+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)

        # Tokenize, stem, and remove stopwords
        value_list = text.split()
        value = ' '.join([stemmer.stem(word) for word in value_list if word not in stopword])

        return value


# --- Merge Columns Transformer ---
class merge_columns(BaseEstimator, TransformerMixin):
    """
    Merges multiple text columns into one, with an option to drop the original columns.

    Args:
        features: List of features to merge.
        new_feature_name: The name of the new merged column.
        drop_original: If True, drops the original columns after merging unless the new column name matches one of the old ones.
    """
    def __init__(self, features, new_feature_name='merged_text', drop_original=True):
        self.features = features
        self.new_feature_name = new_feature_name
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        # Merge the columns into one
        X[self.new_feature_name] = X[self.features].apply(self.merge_text, axis=1)

        # Drop original columns if requested, except if the new feature name matches one of the originals
        if self.drop_original:
            features_to_drop = [f for f in self.features if f != self.new_feature_name]
            if features_to_drop:
                X = X.drop(columns=features_to_drop)

        return X

    def merge_text(self, row):
        """Merges the content of the specified columns into a single string."""
        return ' '.join([str(row[feature]) for feature in self.features if pd.notnull(row[feature])])

def generate_embeddings(embeder_path: str, X: pd.DataFrame, column_name: str):
    """
    Generates embeddings for a specified column in a DataFrame using a SentenceTransformer model.

    Args:
        embeder_path (str): Path to the SentenceTransformer model.
        X (pd.DataFrame): The input DataFrame containing the text data.
        column_name (str): The name of the column to generate embeddings for.

    Returns:
        np.ndarray: The generated embeddings for the specified column.
    """
    model = SentenceTransformer(embeder_path)
    
    return model.encode(X[column_name].values.tolist())
    
    
    
    