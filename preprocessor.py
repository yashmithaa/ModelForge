import yaml
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import string
from rich.console import Console
from rich.table import Table

from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score
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
        self.config = config['training']['split']
        self.train_data = None
        self.test_data = None
        self.validation_data = None

    def split_data(self, data):
        train_percent = self.config['train']
        test_percent = self.config['test']
        validation_percent = self.config['validation']

        # Calculate the sizes for each split
        test_size = test_percent / (test_percent + validation_percent)
        validation_size = validation_percent / (test_percent + validation_percent)

        # First split: into training and remaining data (test + validation)
        self.train_data, remaining_data = train_test_split(data, test_size=(test_percent + validation_percent))

        # Second split: remaining data into test and validation sets
        self.test_data, self.validation_data = train_test_split(remaining_data, test_size=test_size)

        return self.train_data, self.test_data, self.validation_data
    
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
    config_path = 'config.yaml'

    # Load data and config
    loader = Loader(config_path)
    config = loader.load_config(config_path)
    data = loader.load_dataset()

    # clean the data
    cleaner = DataCleaner(config)
    data = cleaner.clean_data(data)

    # Preprocess data
    preprocessor = TextPreprocessor(config)
    data = preprocessor.preprocess_dataset(data)
    print(f"Preprocessed data looks like,\n{data.head(5)}\n")
    # Split data
    splitter = DataSplitter(config)
    train_set, validation_set, test_set = splitter.split_data(data)

    table = Table(title=f"Dataset total{len(train_set)+len(validation_set)+len(test_set)}")
    table.add_column("Train set")
    table.add_column("Validation set")
    table.add_column("Test set")
    table.add_row(str(len(train_set)),str(len(validation_set)),str(len(test_set)))
    console = Console()
    console.print(table)

    model = Model(config)
    results = model.roberta(test_set,num_samples=5)
    model.print_results(results)


if __name__ == "__main__":
    main()
