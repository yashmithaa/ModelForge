import nltk
import string
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

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