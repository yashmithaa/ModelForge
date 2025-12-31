import logging

class CategoricalDecoder:
    def _init_(self, encoding_type='onehot'):
        if encoding_type not in ['onehot', 'label']:
            raise ValueError("encoding_type should be either 'onehot' or 'label'")
        self.encoding_type = encoding_type
        self.encoder = None
        self.column_names = None
        logging.info(f"categorical decoder initialized")

    def fit(self, encoder, column_names):
        self.encoder = encoder
        self.column_names = column_names

    def inverse_transform(self, X):
        if self.encoding_type == 'onehot':
            inverse_transformed_data = self.encoder.inverse_transform(X)
            return pd.DataFrame(inverse_transformed_data, columns=self.column_names)
        else:
            transformed_data = X.copy()
            for column in self.column_names:
                transformed_data[column] = self.encoder[column].inverse_transform(X[column])
            return transformed_data
