from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import hashlib
import demoji
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')
stemmer = SnowballStemmer('english')
stopword = stopwords.words('english')
demoji.download_codes()


class feature_extractor(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str]):
        """
        Args:
            features: List of features (columns) to keep.
        """
        super().__init__()
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        existing_features = [feature for feature in self.features if feature in X.columns]

        X_transformed = X[existing_features].copy()

        return X_transformed


class clear_columns(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        demoji.download_codes() 
        for feature in self.features:
            if feature in data.columns:
                data[feature] = data[feature].apply(self.clean_text)
        return data

    def clean_text(self, text):
        if pd.isnull(text):
            return text
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\b\w{1,2}\b', '', text)
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', '', text)
        text = demoji.replace_with_desc(text)
        text = re.sub(r'[@|#][^\s]+', '', text)
        text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)
        value_list = text.split()
        value = ' '.join([stemmer.stem(i) for i in value_list if i not in stopword])
        return value


class merge_columns(BaseEstimator, TransformerMixin):
    def __init__(self, features, new_feature_name='merged_text', drop_original=True):
        """
        Args:
            features: List of features (columns) to merge.
            new_feature_name: Name of the new merged column.
            drop_original: Whether to drop the original columns after merging, 
                           unless the new feature name is one of the original column names.
        """
        self.features = features
        self.new_feature_name = new_feature_name
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, data):
        # Merge the columns into one
        data[self.new_feature_name] = data[self.features].apply(self.merge_text, axis=1)

        # Drop the original columns only if the new feature name is different
        if self.drop_original:
            features_to_drop = [f for f in self.features if f != self.new_feature_name]
            if features_to_drop:
                data = data.drop(columns=features_to_drop)

        return data

    def merge_text(self, row):
        """Merges the columns specified in `features` for each row."""
        merged_text = ' '.join([str(row[feature]) for feature in self.features if pd.notnull(row[feature])])
        return merged_text
