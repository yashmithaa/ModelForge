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

import torch
import torch.nn as nn
import torch.nn.functional as F
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

class ParallelCNN(nn.Module):
    def __init__(self, config):
        super(ParallelCNN, self).__init__()
        self.embedding = nn.Embedding(config['params']['vocab_size'], config['params']['embedding_size'])
        self.convs = nn.ModuleList([
            nn.Conv2d(1, config['params']['num_filters'], (k, config['params']['embedding_size'])) for k in config['params']['filter_sizes']
        ])
        self.fc_layers = nn.ModuleList([
            nn.Linear(config['params']['num_filters'] * len(config['params']['filter_sizes']), config['params']['fc_size'])
            for _ in range(config['params']['num_fc_layers'])
        ])
        self.dropout = nn.Dropout(config['params']['dropout'])
        self.output_layer = nn.Linear(config['params']['fc_size'], config['params']['output_size'])

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        for fc in self.fc_layers:
            x = self.dropout(F.relu(fc(x)))
        x = self.output_layer(x)
        return x
    
class StackedCNN(nn.Module):
    def __init__(self, config):
        super(StackedCNN, self).__init__()
        self.embedding = nn.Embedding(config['params']['vocab_size'], config['params']['embedding_size'])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, config['params']['num_filters'], (k, config['params']['embedding_size'])),
                nn.ReLU(),
                nn.Conv2d(config['params']['num_filters'], config['params']['num_filters'], (k, 1))
            ) for k in config['params']['filter_sizes']
        ])
        self.fc_layers = nn.ModuleList([
            nn.Linear(config['params']['num_filters'] * len(config['params']['filter_sizes']), config['params']['fc_size'])
            for _ in range(config['params']['num_fc_layers'])
        ])
        self.dropout = nn.Dropout(config['params']['dropout'])
        self.output_layer = nn.Linear(config['params']['fc_size'], config['params']['output_size'])

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        for fc in self.fc_layers:
            x = self.dropout(F.relu(fc(x)))
        x = self.output_layer(x)
        return x

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
    print(encoder)



if __name__ == "__main__":
    main()
