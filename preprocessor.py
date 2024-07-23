import yaml
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import string
import numpy as np
import json

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
from scipy.special import softmax

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
# nltk.download('punkt')
# nltk.download('stopwords')

class Loader:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.data = None

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_dataset(self):
        dataset_config = self.config['dataset']
        self.data = pd.read_csv(dataset_config['path'], delimiter=dataset_config['delimiter'])
        return self.data
    
class DataCleaner:
    def __init__(self, config):
        self.config = config

    def clean_data(self, data):
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
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
        data['text'] = data['text'].apply(lambda x: self.preprocess_text(x))
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
        data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # First split: into training and remaining data (test + validation)
        self.train_data, remaining_data = train_test_split(data, test_size=(test_percent + validation_percent), random_state=random_seed)

        # Second split: remaining data into test and validation sets
        self.test_data, self.validation_data = train_test_split(remaining_data, test_size=test_size, random_state=random_seed)

        self.save_hdf5()

        return self.train_data, self.test_data, self.validation_data
    
    def save_hdf5(self):
        self.train_data.to_hdf(f'dataset.training.hdf5', key='train', mode='w')
        print("\nWriting preprocessed training set to dataset.training.hdf5")
        self.test_data.to_hdf(f'dataset.test.hdf5', key='test', mode='w')
        print("Writing preprocessed test set to dataset.test.hdf5")
        self.validation_data.to_hdf(f'dataset.validation.hdf5', key='validation', mode='w')
        print("Writing preprocessed validation set to dataset.validation.hdf5\n")

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
        self.config=config['params']
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
                                        dropout=self.recurrent_dropout, 
                                        recurrent_initializer=self.recurrent_initializer, 
                                        use_bias=self.use_bias)
        elif self.cell_type == 'lstm':
            self.rnn = layers.LSTM(self.hidden_size, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   dropout=self.recurrent_dropout, 
                                   recurrent_initializer=self.recurrent_initializer, 
                                   unit_forget_bias=self.unit_forget_bias, 
                                   use_bias=self.use_bias)
        elif self.cell_type == 'gru':
            self.rnn = layers.GRU(self.hidden_size, 
                                  return_sequences=True, 
                                  return_state=True, 
                                  dropout=self.recurrent_dropout, 
                                  recurrent_initializer=self.recurrent_initializer, 
                                  use_bias=self.use_bias)

        self.dropout = layers.Dropout(rate=config.get('dropout', 0.0))

        # Fully connected layers
        self.fc_layers = []
        if self.num_fc_layers > 0:
            input_dim = self.hidden_size * (2 if self.bidirectional else 1)
            for _ in range(self.num_fc_layers):
                self.fc_layers.append(layers.Dense(self.output_size))
                input_dim = self.output_size

        # Regularization
        if self.norm:
            self.regularizer = layers.LayerNormalization()
        else:
            self.regularizer = None

    def call(self, x):
        x = self.embedding(x)
        
        if self.cell_type == 'lstm':
            output, hidden_state, cell_state = self.rnn(x)
        else:
            output, hidden_state = self.rnn(x)
        
        output = self.dropout(output)

        # Apply representation type
        if self.representation == 'dense':
            output = output
        elif self.representation == 'sparse':
            output = tf.sparse.to_dense(output)

        # Reduce output
        if self.reduce_output == 'sum':
            output = tf.reduce_sum(output, axis=1)
        elif self.reduce_output == 'mean':
            output = tf.reduce_mean(output, axis=1)
        elif self.reduce_output == 'last':
            output = output[:, -1, :]

        # Apply fully connected layers
        for fc in self.fc_layers:
            output = fc(output)

        # Apply regularizer
        if self.regularizer:
            output = self.regularizer(output)

        return output
    
    def encode_data(self, data):
        encoded_data = self.call(data)
        return encoded_data
    
class CategoricalEncoder:
    def _init_(self, encoding_type='onehot'):
        if encoding_type not in ['onehot', 'label']:
            raise ValueError("encoding_type should be either 'onehot' or 'label'")
        self.encoding_type = encoding_type
        self.encoder = None
    
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
    
# Numerical Encoder Class
class NumericalEncoder:
    def _init_(self, config):
        self.config = config['preprocessing']['numerical']
        self.scalers = {}

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
    
class Model:
    def __init__(self, config):
        self.config = config
        self.MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.MODEL)

    def polarity_scores_roberta(self, example):
        encoded_text = self.tokenizer(example, return_tensors='tf')
        output = self.model(encoded_text)
        scores = output.logits[0].numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
        return scores_dict
    
    def roberta(self, data, num_samples=5):
        samples = data.head(num_samples)
        results = []
        for index, row in samples.iterrows():
            text = row['text']
            scores = self.polarity_scores_roberta(text)
            results.append((text, scores))
        return results
    
    def print_results(self, results):
        table = Table(title="Results")
        table.add_column("Text", justify="left")
        table.add_column("Negative", justify="right")
        table.add_column("Neutral", justify="right")
        table.add_column("Positive", justify="right")
        for text, scores in results:
            table.add_row(text, f"{scores['roberta_neg']:.4f}", f"{scores['roberta_neu']:.4f}", f"{scores['roberta_pos']:.4f}")
        console = Console()
        console.print(table)      

def main():
    config_path = 'pcnn.yaml'
    console = Console()

    # Load data and config
    loader = Loader(config_path)
    config = loader.load_config(config_path)
    print("\nUser specified config file\n")
    pprint(config)
    
    data = loader.load_dataset()

    # clean the data
    cleaner = DataCleaner(config)
    data = cleaner.clean_data(data)

    
    md = Markdown('# Preprocessing')
    console.print(md)
    # Preprocess data
    preprocessor = TextPreprocessor(config)
    data = preprocessor.preprocess_dataset(data)
    #print(f"Preprocessed data looks like,\n{data.head(5)}\n") #just to verify

    # Split data
    splitter = DataSplitter(config)
    train_set, validation_set, test_set = splitter.split_data(data)

    table = Table(title=f"Dataset statistics\nTotal datset: {len(train_set)+len(validation_set)+len(test_set)}")
    table.add_column("Dataset", style = "Cyan")
    table.add_column("Size (in Rows)")
    table.add_column("Size (in memeory)")
    table.add_row("Train set", str(len(train_set)), f"{(sys.getsizeof(train_set) / (1024 * 1024)):.2f} Mb")
    table.add_row("Validation set", str(len(validation_set)), f"{(sys.getsizeof(validation_set) / (1024 * 1024)):.2f} Mb")
    table.add_row("Test set", str(len(test_set)), f"{(sys.getsizeof(test_set) / (1024 * 1024)):.2f} Mb")

    
    console.print(table)

    

    for feature in config['input_features']:
        if feature['encoder'] == 'parallel_cnn':
            encoder = ParallelCNN(feature)
            
        if feature['encoder'] == 'roberta':
            model = Model(config)
            results = model.roberta(test_set,num_samples=5)
            model.print_results(results)
            
        if feature['encoder'] == 'stacked_cnn':
            encoder = StackedCNN(feature)
            
        if feature['encoder'] == 'rnn':
            encoder = RNNEncoder(feature)
            # Just for verification
            example_data = tf.random.uniform((32, 10), dtype=tf.int32, maxval=feature['params']['vocab_size'])
            encoded_data = encoder.encode_data(example_data)
            print("Encoded Data:")
            print(encoded_data)

    print(encoder)
    



if __name__ == "__main__":
    main()
