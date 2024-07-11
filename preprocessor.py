import yaml
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import string

nltk.download('punkt')
nltk.download('stopwords')

class Preprocessor:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.data = None
        self.train_data = None
        self.test_data = None
        self.validation_data = None
    
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def load_dataset(self):
        dataset_config = self.config['dataset']
        self.data = pd.read_csv(dataset_config['path'], delimiter=dataset_config['delimiter'])
    
    def preprocess_text(self, text):
        config = self.config['preprocessing']['text']
        if config.get('lower_case'):
            text = text.lower()
        if config.get('remove_punctuation'):
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        tokens = self.tokenize_text(text)
        if config.get('remove_stopwords'):
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        if config.get('stemming'):
            stemmer = nltk.PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

    def tokenize_text(self, text):
        config = self.config['preprocessing']['text']
        method = config['tokenization']['method']
        if method == 'word':
            tokens = word_tokenize(text)
        elif method == 'sentence':
            tokens = nltk.sent_tokenize(text)
        else:
            raise ValueError(f"Unsupported tokenization method: {method}")
        return tokens

    def preprocess_dataset(self):
        self.data['text'] = self.data['text'].apply(lambda x: self.preprocess_text(x))

    def split_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_dataset() before split_data().")
        
        train_percent = self.config['training']['split']['train']
        test_percent = self.config['training']['split']['test']
        validation_percent = self.config['training']['split']['validation']

        # Calculate the sizes for each split
        test_size = test_percent / (test_percent + validation_percent)
        validation_size = validation_percent / (test_percent + validation_percent)

        # First split: into training and remaining data (test + validation)
        self.train_data, remaining_data = train_test_split(self.data, test_size=(test_percent + validation_percent))

        # Second split: remaining data into test and validation sets
        self.test_data, self.validation_data = train_test_split(remaining_data, test_size=test_size)
    
    def get_data_sizes(self):
        if self.data is None or self.train_data is None or self.test_data is None or self.validation_data is None:
            raise ValueError("Data not properly split. Ensure split_data() has been called.")
        
        return {
            "total": len(self.data),
            "train": len(self.train_data),
            "test": len(self.test_data),
            "validation": len(self.validation_data)
        }

def main():
    config_path = 'config.yaml'
    pipeline = Preprocessor(config_path)
    pipeline.load_dataset()
    pipeline.preprocess_dataset()

    pipeline.split_data()
    
    
    data_sizes = pipeline.get_data_sizes()
    
    # Print to verify
    print(pipeline.data.head(5))
    print(f"Total size of data: {data_sizes['total']}")
    print(f"Training set size: {data_sizes['train']}")
    print(f"Test set size: {data_sizes['test']}")
    print(f"Validation set size: {data_sizes['validation']}")

if __name__ == "__main__":
    main()
