import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class NumericalEncoder:
    def _init_(self, config):
        self.config = config['preprocessing']['numerical']
        self.scalers = {}
        logging.info(f"numeric encoder initialized")

    def fit(self, data):
        for feature in self.config:
            name = feature['name']
            scaler_type = feature.get('scale', 'standard')
            
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {scaler_type}")
            
            scaler.fit(data[[name]])
            self.scalers[name] = scaler

    def transform(self, data):
        for name, scaler in self.scalers.items():
            data[name] = scaler.transform(data[[name]])
        return data
