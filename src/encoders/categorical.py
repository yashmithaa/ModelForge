import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class CategoricalEncoder:
    def _init_(self, encoding_type='onehot'):
        if encoding_type not in ['onehot', 'label']:
            raise ValueError("encoding_type should be either 'onehot' or 'label'")
        self.encoding_type = encoding_type
        self.encoder = None
        logging.info(f"categorical encoder initialized")
    
    def fit(self, X):
        if self.encoding_type == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(X)
        elif self.encoding_type == 'label':
            self.encoder = {}
            for column in X.columns:
                le = LabelEncoder()
                le.fit(X[column])
                self.encoder[column] = le
                
    def transform(self, X):
        if self.encoding_type == 'onehot':
            return pd.DataFrame(self.encoder.transform(X), columns=self.encoder.get_feature_names_out())
        else:
            transformed_data = X.copy()
            for column in X.columns:
                transformed_data[column] = self.encoder[column].transform(X[column])
            return transformed_data
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    