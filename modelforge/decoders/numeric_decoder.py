import logging

class NumericalDecoder:
    def _init_(self, config):
        self.config = config['preprocessing']['numerical']
        self.scalers = {}
        logging.info(f"numerical decoder initialized")

    def fit(self, scalers):
        self.scalers = scalers

    def inverse_transform(self, data):
        for name, scaler in self.scalers.items():
            data[name] = scaler.inverse_transform(data[[name]])
        return data