import yaml
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import string
import numpy as np
import json
import logging

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.pretty import pprint
import sys

from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Loader:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.data = None

    def load_config(self, config_path):
        logging.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Config loaded successfully")
        return config

    def load_dataset(self):
        dataset_config = self.config['dataset']
        logging.info(f"Loading dataset from {dataset_config['path']}")
        self.data = pd.read_csv(dataset_config['path'], delimiter=dataset_config['delimiter'])
        logging.info("Dataset loaded successfully")
        return self.data

class DataCleaner:
    def __init__(self, config):
        self.config = config

    def clean_data(self, data):
        logging.info("Cleaning data")
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        logging.info("Data cleaned successfully")
        return data

class TextPreprocessor:
    def __init__(self, config):
        self.config = config['preprocessing']['text']

    def preprocess_text(self, text):
        if self.config.get('lower_case'):
            text = text.lower()
        if self.config.get('remove_punctuation'):
            text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = self.tokenize_text(text)
        if self.config.get('remove_stopwords'):
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        if self.config.get('stemming'):
            stemmer = nltk.PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

    def tokenize_text(self, text):
        method = self.config['tokenization']['method']
        if method == 'word':
            tokens = word_tokenize(text)
        elif method == 'sentence':
            tokens = nltk.sent_tokenize(text)
        else:
            raise ValueError(f"Unsupported tokenization method: {method}")
        return tokens

    def preprocess_dataset(self, data):
        logging.info("Preprocessing dataset")
        data['text'] = data['text'].apply(lambda x: self.preprocess_text(x))
        logging.info("Dataset preprocessed successfully")
        return data

class DataSplitter:
    def __init__(self, config):
        self.config = config['preprocessing']['split']
        self.train_data = None
        self.test_data = None
        self.validation_data = None

    def split_data(self, data):
        train_percent = self.config['train']
        test_percent = self.config['test']
        validation_percent = self.config['validation']
        random_seed = self.config.get('random_seed', None)

        # Calculate the sizes for each split
        test_size = test_percent / (test_percent + validation_percent)
        validation_size = validation_percent / (test_percent + validation_percent)

        # Shuffle the data
        logging.info("Shuffling data")
        data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # First split: into training and remaining data (test + validation)
        self.train_data, remaining_data = train_test_split(data, test_size=(test_percent + validation_percent), random_state=random_seed)

        # Second split: remaining data into test and validation sets
        self.test_data, self.validation_data = train_test_split(remaining_data, test_size=test_size, random_state=random_seed)

        self.save_hdf5()

        logging.info("Data split successfully")
        return self.train_data, self.test_data, self.validation_data

    def save_hdf5(self):
        logging.info("Saving datasets to HDF5 files")
        self.train_data.to_hdf(f'dataset.training.hdf5', key='train', mode='w')
        logging.info("Training set saved to dataset.training.hdf5")
        self.test_data.to_hdf(f'dataset.test.hdf5', key='test', mode='w')
        logging.info("Test set saved to dataset.test.hdf5")
        self.validation_data.to_hdf(f'dataset.validation.hdf5', key='validation', mode='w')
        logging.info("Validation set saved to dataset.validation.hdf5")

class ParallelCNN(tf.keras.Model):
    def __init__(self, config):
        super(ParallelCNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=config['params']['vocab_size'],
                                                   output_dim=config['params']['embedding_size'])
        self.convs = [
            tf.keras.layers.Conv2D(filters=config['params']['num_filters'],
                                   kernel_size=(k, config['params']['embedding_size']),
                                   activation='relu') for k in config['params']['filter_sizes']
        ]
        self.fc_layers = [
            tf.keras.layers.Dense(units=config['params']['fc_size'], activation='relu')
            for _ in range(config['params']['num_fc_layers'])
        ]
        self.dropout = tf.keras.layers.Dropout(rate=config['params']['dropout'])
        self.output_layer = tf.keras.layers.Dense(units=config['params']['output_size'])

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.expand_dims(x, -1)
        x = [tf.squeeze(conv(x), axis=2) for conv in self.convs]
        x = [tf.reduce_max(input_tensor=i, axis=1) for i in x]
        x = tf.concat(x, axis=1)
        for fc in self.fc_layers:
            x = self.dropout(fc(x))
        x = self.output_layer(x)
        return x

class StackedCNN(tf.keras.Model):
    def __init__(self, config):
        super(StackedCNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=config['params']['vocab_size'],
                                                   output_dim=config['params']['embedding_size'])
        self.convs = [
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=config['params']['num_filters'],
                                       kernel_size=(k, config['params']['embedding_size']),
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=config['params']['num_filters'],
                                       kernel_size=(k, 1),
                                       activation='relu')
            ]) for k in config['params']['filter_sizes']
        ]
        self.fc_layers = [
            tf.keras.layers.Dense(units=config['params']['fc_size'], activation='relu')
            for _ in range(config['params']['num_fc_layers'])
        ]
        self.dropout = tf.keras.layers.Dropout(rate=config['params']['dropout'])
        self.output_layer = tf.keras.layers.Dense(units=config['params']['output_size'])

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.expand_dims(x, -1)
        x = [tf.squeeze(conv(x), axis=2) for conv in self.convs]
        x = [tf.reduce_max(input_tensor=i, axis=1) for i in x]
        x = tf.concat(x, axis=1)
        for fc in self.fc_layers:
            x = self.dropout(fc(x))
        x = self.output_layer(x)
        return x

class RNNEncoder(tf.keras.Model):
    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        self.config = config['params']
        self.embedding_size = self.config['embedding_size']
        self.hidden_size = self.config['state_size']
        self.output_size = self.config['output_size']
        self.num_layers = self.config['num_layers']
        self.bidirectional = self.config['bidirectional']
        self.cell_type = self.config['cell_type']
        self.representation = self.config['representation']
        self.recurrent_dropout = self.config['recurrent_dropout']
        self.recurrent_initializer = self.config['recurrent_initializer']
        self.use_bias = self.config['use_bias']
        self.weights_initializer = self.config['weights_initializer']
        self.unit_forget_bias = self.config['unit_forget_bias']
        self.reduce_output = self.config['reduce_output']
        self.num_fc_layers = self.config['num_fc_layers']
        self.norm = self.config['norm']
        self.vocab_size = self.config['vocab_size']

        # Embedding layer
        self.embedding = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size)

        # RNN cell
        if self.cell_type == 'rnn':
            self.rnn = layers.SimpleRNN(self.hidden_size, 
                                        return_sequences=True, 
                                        return_state=True, 
                                        recurrent_dropout=self.recurrent_dropout)
        elif self.cell_type == 'lstm':
            self.rnn = layers.LSTM(self.hidden_size, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_dropout=self.recurrent_dropout)
        elif self.cell_type == 'gru':
            self.rnn = layers.GRU(self.hidden_size, 
                                  return_sequences=True, 
                                  return_state=True, 
                                  recurrent_dropout=self.recurrent_dropout)
        else:
            raise ValueError("Unsupported cell type")

        # Fully connected layers
        self.fc_layers = [layers.Dense(self.hidden_size, activation='relu') for _ in range(self.num_fc_layers)]
        self.output_layer = layers.Dense(self.output_size)

    def call(self, inputs):
        x = self.embedding(inputs)
        rnn_out, state = self.rnn(x)
        if self.reduce_output:
            x = tf.reduce_mean(rnn_out, axis=1)
        else:
            x = rnn_out
        for fc in self.fc_layers:
            x = fc(x)
        x = self.output_layer(x)
        return x

class NumericalEncoder:
    def __init__(self, config):
        if 'numerical' in config['preprocessing']:
            self.config = config['preprocessing']['numerical']
            self.scalers = {
                'standard': StandardScaler,
                'minmax': MinMaxScaler
            }
            self.encoders = {
                'onehot': OneHotEncoder,
                'label': LabelEncoder
            }
        else:
            self.config = None

    def encode(self, data, columns):
        if self.config is None:
            logging.warning("No numerical preprocessing configuration found. Skipping numerical encoding.")
            return data
        
        for column in columns:
            if self.config.get(column) is None:
                logging.warning(f"No configuration found for column '{column}'. Skipping encoding.")
                continue

            if self.config[column]['method'] == 'scaling':
                scaler = self.scalers[self.config[column]['type']]()
                data[column] = scaler.fit_transform(data[[column]])
            elif self.config[column]['method'] == 'encoding':
                encoder = self.encoders[self.config[column]['type']]()
                if self.config[column]['type'] == 'label':
                    data[column] = encoder.fit_transform(data[column])
                else:
                    encoded = encoder.fit_transform(data[[column]]).toarray()
                    for i in range(encoded.shape[1]):
                        data[f'{column}_{i}'] = encoded[:, i]
                    data.drop(columns=[column], inplace=True)
        return data

class TextClassificationModel:
    def __init__(self, config):
        self.config = config['model']
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

    def create_encoder(self):
        if self.config['encoder'] == 'cnn_parallel':
            return ParallelCNN(self.config)
        elif self.config['encoder'] == 'cnn_stacked':
            return StackedCNN(self.config)
        elif self.config['encoder'] == 'rnn':
            return RNNEncoder(self.config)
        else:
            raise ValueError(f"Unsupported encoder type: {self.config['encoder']}")

    def create_decoder(self):
        # Implement decoder creation logic based on config if needed
        pass

    def compile(self):
        self.encoder.compile(optimizer=self.config['optimizer'], loss=self.config['loss'])

    def train(self, train_data, validation_data):
        history = self.encoder.fit(train_data, validation_data, epochs=self.config['epochs'], batch_size=self.config['batch_size'])
        return history

    def evaluate(self, test_data):
        return self.encoder.evaluate(test_data)
    
    def predict(self, inputs):
        return self.encoder.predict(inputs)

    def save_model(self, filepath):
        self.encoder.save(filepath)
    
    def load_model(self, filepath):
        self.encoder = tf.keras.models.load_model(filepath)

class ModelTrainer:
    def __init__(self, config, model, train_data, validation_data, test_data):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

    def train_and_evaluate(self):
        logging.info("Starting training")
        self.model.compile()
        history = self.model.train(self.train_data, self.validation_data)
        logging.info("Training completed")

        logging.info("Evaluating model")
        test_loss, test_acc = self.model.evaluate(self.test_data)
        logging.info(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
        return history, test_loss, test_acc

    def save_model(self, filepath):
        self.model.save_model(filepath)
        logging.info(f"Model saved at {filepath}")
    
    def load_model(self, filepath):
        self.model.load_model(filepath)
        logging.info(f"Model loaded from {filepath}")

def main(config_path):
    logging.info("Starting main function")

    loader = Loader(config_path)
    config = loader.config
    data = loader.load_dataset()

    cleaner = DataCleaner(config)
    cleaned_data = cleaner.clean_data(data)

    text_preprocessor = TextPreprocessor(config)
    preprocessed_data = text_preprocessor.preprocess_dataset(cleaned_data)

    numerical_encoder = NumericalEncoder(config)
    encoded_data = numerical_encoder.encode(preprocessed_data, columns=['numerical_column_1', 'numerical_column_2'])

    splitter = DataSplitter(config)
    train_data, test_data, validation_data = splitter.split_data(encoded_data)

    model = TextClassificationModel(config)
    trainer = ModelTrainer(config, model, train_data, validation_data, test_data)

    history, test_loss, test_acc = trainer.train_and_evaluate()

    if config['save_model']:
        trainer.save_model(config['model_save_path'])

    logging.info("Main function completed")

if __name__ == '__main__':
    main('config.yaml')
