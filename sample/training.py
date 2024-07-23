import yaml
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import string
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.special import softmax
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.pretty import pprint
import sys
import os

# nltk.download('punkt')
# nltk.download('stopwords')


class Loader:
    def _init_(self, config_path):
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
    def _init_(self, config):
        self.config = config

    def clean_data(self, data):
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        return data


class TextPreprocessor:
    def _init_(self, config):
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
    def _init_(self, config):
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
        self.train_data.to_hdf('dataset.training.hdf5', key='train', mode='w')
        print("\nWriting preprocessed training set to dataset.training.hdf5")
        self.test_data.to_hdf('dataset.test.hdf5', key='test', mode='w')
        print("Writing preprocessed test set to dataset.test.hdf5")
        self.validation_data.to_hdf('dataset.validation.hdf5', key='validation', mode='w')
        print("Writing preprocessed validation set to dataset.validation.hdf5\n")


class SentimentDataset(Dataset):
    def _init_(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


class Model:
    def _init_(self, config):
        self.config = config
        self.MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.MODEL)
        self.model = RobertaForSequenceClassification.from_pretrained(self.MODEL, num_labels=3)

    def train(self, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_dataloader)}")
            self.evaluate(val_dataloader)

        self.save_model()

    def evaluate(self, dataloader):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.eval()
        preds, true_labels, texts = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                batch_texts = batch['text']

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                texts.extend(batch_texts)

        accuracy = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average='weighted')
        precision = precision_score(true_labels, preds, average='weighted')
        recall = recall_score(true_labels, preds, average='weighted')
        conf_matrix = confusion_matrix(true_labels, preds)

        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Print a few test cases
        print("\nSample Predictions:")
        for i in range(min(5, len(texts))):  # Print up to 5 test cases
            print(f"Text: {texts[i]}")
            print(f"Actual Sentiment: {self.config['label_map'][true_labels[i]]}")
            print(f"Predicted Sentiment: {self.config['label_map'][preds[i]]}\n")

    def save_model(self):
        model_dir = 'models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.model.state_dict(), os.path.join(model_dir, 'trained_model.pt'))
        print("Model saved to models/trained_model.pt")


class Trainer:
    def _init_(self, config):
        self.config = config
        self.tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    def prepare_dataloaders(self, train_data, val_data, test_data):
        label_map = self.config['label_map']
        train_labels = train_data['sentiment'].map(label_map).values
        val_labels = val_data['sentiment'].map(label_map).values
        test_labels = test_data['sentiment'].map(label_map).values

        train_dataset = SentimentDataset(train_data['text'].tolist(), train_labels, self.tokenizer, max_length=128)
        val_dataset = SentimentDataset(val_data['text'].tolist(), val_labels, self.tokenizer, max_length=128)
        test_dataset = SentimentDataset(test_data['text'].tolist(), test_labels, self.tokenizer, max_length=128)

        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader

    def run(self):
        console = Console()
        md = Markdown('# Training and Evaluation')
        console.print(md)

        model = Model(self.config)
        train_dataloader, val_dataloader, test_dataloader = self.prepare_dataloaders(
            train_data, val_data, test_data
        )

        # Train the model
        model.train(train_dataloader, val_dataloader, epochs=self.config['training']['epochs'])

        # Evaluate the model on the test set
        model.evaluate(test_dataloader)


def main():
    config_path = 'config.yaml'
    console = Console()

    # Load data and config
    loader = Loader(config_path)
    config = loader.load_config(config_path)
    print("\nUser specified config file\n")
    pprint(config)
    
    data = loader.load_dataset()


    # Clean the data
    cleaner = DataCleaner(config)
    data = cleaner.clean_data(data)

    md = Markdown('# Preprocessing')
    console.print(md)

    # Preprocess data
    preprocessor = TextPreprocessor(config)
    data = preprocessor.preprocess_dataset(data)

    # Split data
    splitter = DataSplitter(config)
    train_data, val_data, test_data = splitter.split_data(data)

    table = Table(title=f"Dataset statistics\nTotal dataset: {len(train_data) + len(val_data) + len(test_data)}")
    table.add_column("Dataset", style="cyan")
    table.add_column("Size (in Rows)")
    table.add_column("Size (in Memory)")
    table.add_row("Train set", str(len(train_data)), f"{(sys.getsizeof(train_data) / (1024 * 1024)):.2f} Mb")
    table.add_row("Validation set", str(len(val_data)), f"{(sys.getsizeof(val_data) / (1024 * 1024)):.2f} Mb")
    table.add_row("Test set", str(len(test_data)), f"{(sys.getsizeof(test_data) / (1024 * 1024)):.2f} Mb")

    console.print(table)

    # Train and evaluate the model
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()