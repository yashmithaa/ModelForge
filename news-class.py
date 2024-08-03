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


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
# nltk.download('punkt')
# nltk.download('stopwords')

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import random

from torch.nn.utils.rnn import pad_sequence

import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import subprocess
import threading
import time
import webbrowser


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
        self.data = pd.read_csv(dataset_config['path'], delimiter=dataset_config['delimiter'], encoding='utf-8', encoding_errors='ignore')
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

        logging.info("Data split successfully")
        return self.train_data, self.test_data, self.validation_data
    
    def save_hdf5(self):
        logging.info("Saving datasets to HDF5 files")
        self.train_data.to_hdf(f'preprocessed-data/dataset.training.hdf5', key='train', mode='w')
        print("\nWriting preprocessed training set to preprocessed-data/dataset.training.hdf5")
        self.test_data.to_hdf(f'preprocessed-data/dataset.test.hdf5', key='test', mode='w')
        print("Writing preprocessed test set to preprocessed-data/dataset.test.hdf5")
        self.validation_data.to_hdf(f'preprocessed-data/dataset.validation.hdf5', key='validation', mode='w')
        print("Writing preprocessed validation set to preprocessed-data/dataset.validation.hdf5\n")

class ParallelCNN(nn.Module):
    def __init__(self, input_feature_config):
        super(ParallelCNN, self).__init__()
        self.vocab_size = input_feature_config['params']['vocab_size']
        self.embedding_size = input_feature_config['params']['embedding_size']
        self.num_filters = input_feature_config['params']['num_filters']
        self.filter_sizes = input_feature_config['params']['filter_sizes']
        self.fc_size = input_feature_config['params']['fc_size']
        self.dropout = input_feature_config['params']['dropout']

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (fs, self.embedding_size)) for fs in self.filter_sizes
        ])

        self.dropout_layer = nn.Dropout(self.dropout)

        # Calculate the output size after the convolution and pooling layers
        self.conv_output_size = len(self.filter_sizes) * self.num_filters

        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, self.fc_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc_size, input_feature_config['params']['output_size'])
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        conv_outputs = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]
        pooled_outputs = [F.max_pool1d(output, output.size(2)).squeeze(2) for output in conv_outputs]
        cat_output = torch.cat(pooled_outputs, 1)
        cat_output = self.dropout_layer(cat_output)
        fc_output = self.fc_layers(cat_output)
        return fc_output
    
class RNNEncoder(nn.Module):
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
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)

        # RNN cell
        if self.cell_type == 'rnn':
            self.rnn = nn.RNN(self.embedding_size, 
                              self.hidden_size, 
                              num_layers=self.num_layers, 
                              bidirectional=self.bidirectional, 
                              dropout=self.recurrent_dropout, 
                              batch_first=True)
        elif self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size, 
                               self.hidden_size, 
                               num_layers=self.num_layers, 
                               bidirectional=self.bidirectional, 
                               dropout=self.recurrent_dropout, 
                               batch_first=True)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size, 
                              self.hidden_size, 
                              num_layers=self.num_layers, 
                              bidirectional=self.bidirectional, 
                              dropout=self.recurrent_dropout, 
                              batch_first=True)

        self.dropout = nn.Dropout(p=config.get('dropout', 0.0))

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        if self.num_fc_layers > 0:
            input_dim = self.hidden_size * (2 if self.bidirectional else 1)
            for _ in range(self.num_fc_layers):
                self.fc_layers.append(nn.Linear(input_dim, self.output_size))
                input_dim = self.output_size

        # Regularization
        if self.norm:
            self.regularizer = nn.LayerNorm(self.output_size)
        else:
            self.regularizer = None

    def call(self, x):
        x = self.embedding(x)
        
        if self.cell_type == 'lstm':
            output, (hidden_state, cell_state) = self.rnn(x)
        else:
            output, hidden_state = self.rnn(x)
        
        output = self.dropout(output)

        # Apply representation type
        if self.representation == 'dense':
            output = output
        elif self.representation == 'sparse':
            output = torch.sparse.FloatTensor(output)

        # Reduce output
        if self.reduce_output == 'sum':
            output = torch.sum(output, dim=1)
        elif self.reduce_output == 'mean':
            output = torch.mean(output, dim=1)
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

class Combiner(nn.Module):
    def __init__(self, config):
        super(Combiner, self).__init__()
        self.config = config
        self.combiner_type = config['combiner']['type']
        self.output_size = config['combiner']['output_size']

        if self.combiner_type == 'concat':
            input_size = sum([feature['params']['output_size'] for feature in config['input_features']])
            self.combiner = nn.Linear(input_size, self.output_size)
        elif self.combiner_type == 'sum':
            input_size = config['input_features'][0]['params']['output_size']
            self.combiner = nn.Linear(input_size, self.output_size)
        else:
            raise ValueError(f"Unsupported combiner type: {self.combiner_type}")

    def forward(self, encoder_outputs):
        if self.combiner_type == 'concat':
            combined_output = torch.cat(encoder_outputs, dim=-1)
        elif self.combiner_type == 'sum':
            combined_output = torch.sum(torch.stack(encoder_outputs), dim=0)
            
        return self.combiner(combined_output)
    
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

 
class Model:
    def __init__(self, config):
        self.config = config
        self.MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)

    def polarity_scores_roberta(self, example):
        encoded_text = self.tokenizer(example, return_tensors='pt')
        output = self.model(**encoded_text)
        scores = output.logits[0].detach().numpy()
        scores = F.softmax(torch.tensor(scores), dim=-1).numpy()
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

# this is just a sample rnn decoder - to check if combiner works - need to add the actual RNNdecoder
class RNNDecoder(nn.Module):
    def __init__(self, config):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(config['decoder']['vocab_size'], config['decoder']['embedding_size'])
        self.lstm = nn.LSTM(config['decoder']['embedding_size'], config['decoder']['hidden_size'], batch_first=True)
        self.fc = nn.Linear(config['decoder']['hidden_size'], config['decoder']['vocab_size'])
        self.dropout = nn.Dropout(config['decoder']['dropout'])

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        return x, hidden
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_feedforward, max_seq_len, num_classes, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.PositionalEncoding(d_model, max_seq_len)
        self.encoder = self.TransformerEncoder(num_layers, d_model, num_heads, dim_feedforward, dropout)
        self.decoder = self.TransformerDecoder(num_layers, d_model, num_heads, dim_feedforward, dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        memory = self.encoder(src, src_mask)

        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask)

        output = output.mean(dim=1)  # Aggregate over sequence length
        output = self.fc(output)
        return output

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            self.encoding = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
            self.encoding[:, 0::2] = torch.sin(position * div_term)
            self.encoding[:, 1::2] = torch.cos(position * div_term)
            self.encoding = self.encoding.unsqueeze(0)

        def forward(self, x):
            return x + self.encoding[:, :x.size(1)].detach()

    class TransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        def forward(self, src, mask=None):
            src2 = self.self_attn(src, src, src, attn_mask=mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src

    class TransformerEncoder(nn.Module):
        def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout):
            super().__init__()
            self.layers = nn.ModuleList([
                TransformerModel.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])

        def forward(self, src, mask=None):
            for layer in self.layers:
                src = layer(src, mask)
            return src

    class TransformerDecoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, dim_feedforward, dropout):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model)
            )
            self.layer_norm1 = nn.LayerNorm(d_model)
            self.layer_norm2 = nn.LayerNorm(d_model)
            self.layer_norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
            # Self-attention
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.layer_norm1(tgt)

            # Multi-head attention
            tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.layer_norm2(tgt)

            # Feed-forward network
            tgt2 = self.ffn(tgt)
            tgt = tgt + self.dropout(tgt2)
            tgt = self.layer_norm3(tgt)

            return tgt

    class TransformerDecoder(nn.Module):
        def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout):
            super().__init__()
            self.layers = nn.ModuleList([
                TransformerModel.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])

        def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
            for layer in self.layers:
                tgt = layer(tgt, memory, tgt_mask, memory_mask)
            return tgt

    class LayerNormalization(nn.Module):
        def __init__(self, parameters_shape, eps=1e-5):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(parameters_shape))
            self.beta = nn.Parameter(torch.zeros(parameters_shape))
            self.eps = eps

        def forward(self, input):
            dims = [-(i + 1) for i in range(len(input.size()) - 1)]
            mean = input.mean(dim=dims, keepdim=True)
            var = ((input - mean) ** 2).mean(dim=dims, keepdim=True)
            std = (var + self.eps).sqrt()
            y = (input - mean) / std
            out = self.gamma * y + self.beta
            return out
        

class ModelArch(nn.Module):
    def __init__(self, config):
        super(ModelArch, self).__init__()
        self.encoder = ParallelCNN(config['input_features'][0])
        self.decoder = nn.Linear(config['input_features'][0]['params']['output_size'], config['output_features'][0]['num_classes'])

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output
    
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, config):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.config = config

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_tokens = [self.vocab.get(token, self.vocab["<UNK>"]) for token in text.split()]
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return text_tensor, label_tensor
    
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts_padded, labels
    
class ModelTrainer:
    def __init__(self, model, train_data, val_data, vocab, config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.vocab = vocab
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training'][0]['learning_rate'])
        
        self.train_loader = DataLoader(TextDataset(train_data[0], train_data[1], vocab, config), batch_size=config['training'][0]['batch_size'], shuffle=True,collate_fn=collate_fn)
        self.val_loader = DataLoader(TextDataset(val_data[0], val_data[1], vocab, config), batch_size=config['training'][0]['batch_size'], shuffle=False,collate_fn=collate_fn)
        
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        
        for texts, labels in self.train_loader:
            texts, labels = texts.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(texts)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
        
        return total_loss / len(self.train_loader), correct_predictions / len(self.train_loader.dataset)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for texts, labels in self.val_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                
                outputs = self.model(texts)
                
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
        
        return total_loss / len(self.val_loader), correct_predictions / len(self.val_loader.dataset)
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate()

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
class Evaluator:
    def __init__(self, model, test_texts, test_labels, label_encoder, vocab, device='cpu'):
        self.model = model
        self.test_texts = test_texts
        self.test_labels = test_labels
        self.label_encoder = label_encoder
        self.vocab = vocab
        self.device = device

    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        encoded_texts = self.encode_texts(self.test_texts)

                # Pad sequences
        padded_texts = pad_sequence([torch.tensor(seq) for seq in encoded_texts], batch_first=True, padding_value=self.vocab["<UNK>"])

        with torch.no_grad():
            inputs = padded_texts.to(self.device)
            outputs = self.model(inputs)
            _, predictions = torch.max(outputs, 1)
        
        # Decode labels
        predicted_labels = predictions.cpu().numpy()
        true_labels = self.test_labels

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        
        

        table = Table(title=f"Results")
        table.add_column("Metrics", style = "Cyan")
        table.add_column("Results")
        
        table.add_row("Accuracy", str(accuracy))
        table.add_row("Precision", str(precision))
        table.add_row("Recall", str(recall))
        table.add_row("F1 Score", str(f1))
        console = Console()
        console.print(table)

        
        print(classification_report(true_labels, predicted_labels, target_names=self.label_encoder.classes_))

        self.print_samples(self.test_texts, true_labels, predicted_labels)

    def encode_texts(self, texts):
        encoded_texts = []
        for text in texts:
            encoded_text = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()]
            encoded_texts.append(encoded_text)
        return encoded_texts

    def print_samples(self, texts, true_labels, predicted_labels, num_samples=3):
        print("\nSample Predictions:")
        samples = random.sample(list(zip(texts, true_labels, predicted_labels)), num_samples)
        for text, true_label, pred_label in samples:
            print(f"Text: {text}")
            print(f"Expected: {self.label_encoder.inverse_transform([true_label])[0]}")
            print(f"Predicted: {self.label_encoder.inverse_transform([pred_label])[0]}\n")


def start_mlflow_server():
    subprocess.run([
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "file:./mlruns",
        "--host", "0.0.0.0"
    ], check=True)

def main():
    # Start MLflow server in a background thread
    server_thread = threading.Thread(target=start_mlflow_server)
    server_thread.daemon = True
    server_thread.start()

    # Give the server some time to start
    time.sleep(10)  # Adjust this sleep time as needed

    webbrowser.open("http://localhost:5000")

    if len(sys.argv) != 2:
        print("Usage: python script.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    #config_path = 'config.yaml'
    console = Console()

    logging.info("Starting main function")

    # Load data and config
    loader = Loader(config_path)
    config = loader.load_config(config_path)
    print("\nUser specified config file\n")
    pprint(config)
    
    # Initializing Mlflow and starting a new run
    mlflow.set_tracking_uri("http://localhost:5000")  # MLflow tracking server URI
    mlflow.set_experiment("NewsClassificationExperiment")  # Set experiment name
    
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("config", config_path)
        mlflow.set_tag("Training Info", "News Classification")

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

     # Create vocabulary
    all_text = ' '.join(train_set['text'])
    all_words = all_text.split()
    unique_words = list(set(all_words))
    max_vocab_size = config['input_features'][0]['params']['vocab_size']
    if len(unique_words) > max_vocab_size:
        unique_words = unique_words[:max_vocab_size - 1]  # Reserve one index for "<UNK>"
    vocab = {word: idx + 1 for idx, word in enumerate(unique_words)}
    vocab["<UNK>"] = 0

        # Prepare data for training
    label_encoder = LabelEncoder()
    train_texts = train_set['text'].tolist()
    train_labels = label_encoder.fit_transform(train_set['category'].tolist())
    val_texts = validation_set['text'].tolist()
    val_labels = label_encoder.transform(validation_set['category'].tolist())
    test_texts = test_set['text'].tolist()
    test_labels = label_encoder.transform(test_set['category'].tolist())
    

    for feature in config['input_features']:
        if feature['encoder'] == 'roberta':
            model = Model(config)
            results = model.roberta(test_set,num_samples=5)
            model.print_results(results)
        elif feature['encoder'] == 'transformer':
            model = TransformerModel(vocab_size=config['model']['vocab_size'],
                    d_model=config['model']['d_model'],
                    num_heads=config['model']['num_heads'],
                    num_layers=config['model']['num_layers'],
                    dim_feedforward=config['model']['dim_feedforward'],
                    max_seq_len=config['model']['max_seq_len'],
                    num_classes=config['model']['num_classes'],
                    dropout=config['model']['dropout'])
            print(model)
        else:
            model = ModelArch(config)
            print(model)

        mlflow.log_params(config['input_features'][0]['params'])

    md = Markdown('# Training')
    console.print(md)

    trainer = ModelTrainer(model, (train_texts, train_labels), (val_texts, val_labels), vocab, config)
    trainer.train(config['training'][0]['num_epochs']) 
    
    md = Markdown('# Evaluation Metrics')
    console.print(md)
    evaluator = Evaluator(model, test_texts, test_labels, label_encoder, vocab)
    evaluator.evaluate()

    # Save model using MLflow
    mlflow.pytorch.log_model(model, "modelforge")
    

    

if __name__ == "__main__":
    main()
