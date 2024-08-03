import yaml
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import string
import numpy as np
import json

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.pretty import pprint
import sys

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
        data['source_text'] = data['source_text'].apply(lambda x: self.preprocess_text(x))
        data['target_text'] = data['target_text'].apply(lambda x: self.preprocess_text(x))
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
        print("\nWriting preprocessed training set cache to dataset.training.hdf5")
        self.test_data.to_hdf(f'dataset.test.hdf5', key='test', mode='w')
        print("Writing preprocessed test set cache to dataset.test.hdf5")
        self.validation_data.to_hdf(f'dataset.validation.hdf5', key='validation', mode='w')
        print("Writing preprocessed validation set cache to dataset.validation.hdf5\n")

class RNNEncoder(tf.keras.Model):
    def __init__(self, config, vocab_size):
        super(RNNEncoder, self).__init__()
        self.config = config['model']

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
        self.activation = self.config['activation']
        self.unit_forget_bias = self.config['unit_forget_bias']
        self.reduce_output = self.config['reduce_output']
        self.num_fc_layers = self.config['num_fc_layers']
        self.norm = self.config['norm']

        # Embedding layer
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_size)

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

        # Wrap RNN with Bidirectional if specified
        if self.bidirectional:
            self.rnn = layers.Bidirectional(self.rnn)
        else:
            self.rnn = self.rnn

        self.dropout = layers.Dropout(rate=config.get('dropout', 0.0))

        # Fully connected layers
        self.fc_layers = []
        if self.num_fc_layers > 0:
            input_dim = self.hidden_size * (2 if self.bidirectional else 1)
            for _ in range(self.num_fc_layers):
                self.fc_layers.append(layers.Dense(self.output_size, activation=self.activation))
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
            return output, hidden_state, cell_state
        else:
            output, hidden_state = self.rnn(x)
            return output, hidden_state

    def encode_data(self, data):
        encoded_output, hidden_state, cell_state = self.call(data)
        return encoded_output, hidden_state, cell_state


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, attention_size):
        super(AttentionLayer, self).__init__()
        self.attention_size = attention_size
        self.W = layers.Dense(attention_size)
        self.U = layers.Dense(attention_size)
        self.v = layers.Dense(1)

    def call(self, encoder_outputs, decoder_hidden):
        ###print(f"encoder_outputs shape: {encoder_outputs.shape}")
        ###print(f"decoder_hidden shape: {decoder_hidden.shape}")

        # Ensure decoder_hidden is shaped correctly
        if len(decoder_hidden.shape) == 2:
            decoder_hidden = tf.expand_dims(decoder_hidden, 1)
        
        # Calculate attention scores
        score = self.v(tf.nn.tanh(self.W(encoder_outputs) + self.U(decoder_hidden)))
        ###print(f"score shape: {score.shape}")

        # Convert scores to weights
        attention_weights = tf.nn.softmax(score, axis=1)
        ###print(f"attention_weights shape: {attention_weights.shape}")

        # Compute context vector
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
    
class AttentionRNNDecoder(tf.keras.Model):
    def __init__(self, config, vocab_size):
        super(AttentionRNNDecoder, self).__init__()
        self.config = config['model']
        self.embedding_size = self.config['embedding_size']
        self.hidden_size = self.config['state_size']
        self.attention_size = self.config['attention_size']
        self.num_layers = self.config['num_layers']
        self.cell_type = self.config['cell_type']
        self.use_bias = self.config['use_bias']
        self.recurrent_dropout = self.config['recurrent_dropout']
        self.recurrent_initializer = self.config['recurrent_initializer']
        self.unit_forget_bias = self.config['unit_forget_bias']
        self.activation = self.config['activation']
        self.output_size = self.config['output_size']
        
        # Embedding layer
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_size)
        
        # RNN cell
        if self.cell_type == 'lstm':
            self.rnn = layers.LSTM(self.hidden_size, return_sequences=True, return_state=True, dropout=self.recurrent_dropout)
        elif self.cell_type == 'gru':
            self.rnn = layers.GRU(self.hidden_size, return_sequences=True, return_state=True)
        elif self.cell_type == 'rnn':
            self.rnn = layers.SimpleRNN(self.hidden_size, return_sequences=True, return_state=True)
        
        # Attention layer
        self.attention = AttentionLayer(self.attention_size)

        # Output layer
        self.fc = layers.Dense(vocab_size, activation='softmax')

    def call(self, x, encoder_outputs, decoder_hidden):
        x = self.embedding(x)
        
        # Decode the data
        if self.cell_type == 'lstm':
            output, hidden_state, cell_state = self.rnn(x, initial_state=decoder_hidden)
        else:
            output, hidden_state = self.rnn(x, initial_state=decoder_hidden)
        
        # Apply attention
        context_vector, attention_weights = self.attention(encoder_outputs, hidden_state)
        
        # Expand dimensions of context_vector to match output shape
        context_vector = tf.expand_dims(context_vector, 1)  # Shape: (batch_size, 1, hidden_size)
        context_vector = tf.tile(context_vector, [1, tf.shape(output)[1], 1])  # Shape: (batch_size, seq_length, hidden_size)
        
        # Concatenate context vector with RNN output
        output = tf.concat([output, context_vector], axis=-1)
        
        # Final output layer
        output = self.fc(output)

        return output, hidden_state, attention_weights

class Trainer:
    def __init__(self, config, encoder, decoder, train_data, validation_data):
        self.config = config['training']
        self.cell_type = config['model']['cell_type']
        self.encoder = encoder
        self.decoder = decoder
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = Adam(learning_rate=self.config['learning_rate'])
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.source_vectorizer = TextVectorization(max_tokens=None, output_mode='int')
        self.target_vectorizer = TextVectorization(max_tokens=None, output_mode='int')
        self.source_vectorizer.adapt(train_data['source_text'])
        self.target_vectorizer.adapt(train_data['target_text'])

    def prepare_dataset(self, data):
        source_texts = self.source_vectorizer(data['source_text'])
        target_texts = self.target_vectorizer(data['target_text'])
        dataset = tf.data.Dataset.from_tensor_slices((source_texts, target_texts))
        dataset = dataset.batch(self.config['batch_size'])
        return dataset

    def calculate_accuracy(self, true_labels, predictions):
        predicted_classes = tf.argmax(predictions, axis=-1)
        correct_predictions = tf.cast(tf.equal(predicted_classes, true_labels), tf.float32)
        accuracy = tf.reduce_mean(correct_predictions)
        return accuracy

    def train_step(self, source_texts, target_texts):
        with tf.GradientTape() as tape:
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder.encode_data(source_texts)
            decoder_initial_state = [encoder_hidden, encoder_cell]
            decoder_output, _, _ = self.decoder(target_texts, encoder_outputs, decoder_initial_state)
            loss = self.loss_fn(target_texts, decoder_output)
            accuracy = self.calculate_accuracy(target_texts, decoder_output)

        gradients = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))

        return loss, accuracy

    def validate_step(self, source_texts, target_texts):
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder.encode_data(source_texts)
        decoder_initial_state = [encoder_hidden, encoder_cell]
        decoder_output, _, _ = self.decoder(target_texts, encoder_outputs, decoder_initial_state)
        loss = self.loss_fn(target_texts, decoder_output)
        accuracy = self.calculate_accuracy(target_texts, decoder_output)

        return loss, accuracy

    def train(self):
        train_dataset = self.prepare_dataset(self.train_data)
        validation_dataset = self.prepare_dataset(self.validation_data)

        for epoch in range(self.config['epochs']):
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            epoch_loss = 0.0
            epoch_accuracy = 0.0

            for source_texts, target_texts in train_dataset:
                loss, accuracy = self.train_step(source_texts, target_texts)
                epoch_loss += loss
                epoch_accuracy += accuracy

            train_loss = epoch_loss / len(train_dataset)
            train_accuracy = epoch_accuracy / len(train_dataset)

            print(f"Train loss: {train_loss:.3f}, Train accuracy: {train_accuracy:.3f}")

            validation_loss = 0.0
            validation_accuracy = 0.0

            for source_texts, target_texts in validation_dataset:
                loss, accuracy = self.validate_step(source_texts, target_texts)
                validation_loss += loss
                validation_accuracy += accuracy

            validation_loss /= len(validation_dataset)
            validation_accuracy /= len(validation_dataset)

            print(f"Validation loss: {validation_loss:.3f}, Validation accuracy: {validation_accuracy:.3f}")


def main():
    config_path = 'rnn_config.yaml'
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
    data['text'] = data['text'].apply(lambda x: preprocessor.preprocess_text(x))
    data['ctext'] = data['ctext'].apply(lambda x: preprocessor.preprocess_text(x))
    
    # Create source and target columns for the summarization task
    data['source_text'] = data['text']  # Source text for summarization
    data['target_text'] = data['ctext']  # Target summary text

    source_texts = data['source_text']
    target_texts = data['target_text']

    source_vectorizer = TextVectorization(max_tokens=None, output_mode='int')
    target_vectorizer = TextVectorization(max_tokens=None, output_mode='int')

    source_vectorizer.adapt(source_texts)
    target_vectorizer.adapt(target_texts)

    source_vocab_size = len(source_vectorizer.get_vocabulary())
    target_vocab_size = len(target_vectorizer.get_vocabulary())

    # Split data
    splitter = DataSplitter(config)
    train_set, validation_set, test_set = splitter.split_data(data)

    table = Table(title=f"Dataset statistics\nTotal dataset: {len(train_set)+len(validation_set)+len(test_set)}")
    table.add_column("Dataset", style="cyan")
    table.add_column("Size (in Rows)")
    table.add_column("Size (in memory)")
    table.add_row("Train set", str(len(train_set)), f"{(sys.getsizeof(train_set) / (1024 * 1024)):.2f} MB")
    table.add_row("Validation set", str(len(validation_set)), f"{(sys.getsizeof(validation_set) / (1024 * 1024)):.2f} MB")
    table.add_row("Test set", str(len(test_set)), f"{(sys.getsizeof(test_set) / (1024 * 1024)):.2f} MB")

    console.print(table)

    rnn_encoder = RNNEncoder(config, source_vocab_size)
    rnn_decoder = AttentionRNNDecoder(config, target_vocab_size)

    # Initialize the Trainer
    trainer = Trainer(config, rnn_encoder, rnn_decoder, train_set, validation_set)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()


 





# import yaml
# import pandas as pd
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.model_selection import train_test_split
# import string
# import numpy as np
# import json

# import tensorflow as tf
# from tensorflow.keras.layers import TextVectorization
# from tensorflow.keras import layers
# from tensorflow.keras import initializers

# from rich.console import Console
# from rich.table import Table
# from rich.markdown import Markdown
# from rich.pretty import pprint
# import sys

# # nltk.download('punkt')
# # nltk.download('stopwords')

# class Loader:
#     def __init__(self, config_path):
#         self.config = self.load_config(config_path)
#         self.data = None

#     def load_config(self, config_path):
#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)
#         return config

#     def load_dataset(self):
#         dataset_config = self.config['dataset']
#         self.data = pd.read_csv(dataset_config['path'], delimiter=dataset_config['delimiter'])
#         return self.data

# class DataCleaner:
#     def __init__(self, config):
#         self.config = config

#     def clean_data(self, data):
#         data.dropna(inplace=True)
#         data.drop_duplicates(inplace=True)
#         return data
    
# class TextPreprocessor:
#     def __init__(self, config):
#         self.config = config['preprocessing']['text']

#     def preprocess_text(self, text):
#         if self.config.get('lower_case'):
#             text = text.lower()
#         if self.config.get('remove_punctuation'):
#             text = text.translate(str.maketrans('', '', string.punctuation))
#         tokens = self.tokenize_text(text)
#         if self.config.get('remove_stopwords'):
#             stop_words = set(stopwords.words('english'))
#             tokens = [word for word in tokens if word not in stop_words]
#         if self.config.get('stemming'):
#             stemmer = nltk.PorterStemmer()
#             tokens = [stemmer.stem(word) for word in tokens]
#         return ' '.join(tokens)

#     def tokenize_text(self, text):
#         method = self.config['tokenization']['method']
#         if method == 'word':
#             tokens = word_tokenize(text)
#         elif method == 'sentence':
#             tokens = nltk.sent_tokenize(text)
#         else:
#             raise ValueError(f"Unsupported tokenization method: {method}")
#         return tokens

#     def preprocess_dataset(self, data):
#         data['source_text'] = data['source_text'].apply(lambda x: self.preprocess_text(x))
#         data['target_text'] = data['target_text'].apply(lambda x: self.preprocess_text(x))
#         return data

# class DataSplitter:
#     def __init__(self, config):
#         self.config = config['preprocessing']['split']
#         self.train_data = None
#         self.test_data = None
#         self.validation_data = None

#     def split_data(self, data):
#         train_percent = self.config['train']
#         test_percent = self.config['test']
#         validation_percent = self.config['validation']
#         random_seed = self.config.get('random_seed', None)

#         # Calculate the sizes for each split
#         test_size = test_percent / (test_percent + validation_percent)
#         validation_size = validation_percent / (test_percent + validation_percent)

#         # Shuffle the data
#         data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

#         # First split: into training and remaining data (test + validation)
#         self.train_data, remaining_data = train_test_split(data, test_size=(test_percent + validation_percent), random_state=random_seed)

#         # Second split: remaining data into test and validation sets
#         self.test_data, self.validation_data = train_test_split(remaining_data, test_size=test_size, random_state=random_seed)

#         self.save_hdf5()

#         return self.train_data, self.test_data, self.validation_data
    
#     def save_hdf5(self):
#         self.train_data.to_hdf(f'dataset.training.hdf5', key='train', mode='w')
#         print("\nWriting preprocessed training set cache to dataset.training.hdf5")
#         self.test_data.to_hdf(f'dataset.test.hdf5', key='test', mode='w')
#         print("Writing preprocessed test set cache to dataset.test.hdf5")
#         self.validation_data.to_hdf(f'dataset.validation.hdf5', key='validation', mode='w')
#         print("Writing preprocessed validation set cache to dataset.validation.hdf5\n")

# class RNNEncoder(tf.keras.Model):
#     def __init__(self, config, vocab_size):
#         super(RNNEncoder, self).__init__()
#         self.config = config['model']

#         self.embedding_size = self.config['embedding_size']
#         self.hidden_size = self.config['state_size']
#         self.output_size = self.config['output_size']
#         self.num_layers = self.config['num_layers']
#         self.bidirectional = self.config['bidirectional']
#         self.cell_type = self.config['cell_type']
#         self.representation = self.config['representation']
#         self.recurrent_dropout = self.config['recurrent_dropout']
#         self.recurrent_initializer = self.config['recurrent_initializer']
#         self.use_bias = self.config['use_bias']
#         self.activation = self.config['activation']
#         self.unit_forget_bias = self.config['unit_forget_bias']
#         self.reduce_output = self.config['reduce_output']
#         self.num_fc_layers = self.config['num_fc_layers']
#         self.norm = self.config['norm']
#         self.vocab_size = vocab_size

#         # Define layers
#         self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_size)
#         self.rnn = None
#         self.dropout = layers.Dropout(rate=config.get('dropout', 0.0))
#         self.fc_layers = []
#         self.regularizer = None

#     def build(self, input_shape):
#         # Build the RNN cell
#         if self.cell_type == 'rnn':
#             self.rnn = layers.SimpleRNN(self.hidden_size, 
#                                         return_sequences=True, 
#                                         return_state=True, 
#                                         dropout=self.recurrent_dropout, 
#                                         recurrent_initializer=self.recurrent_initializer, 
#                                         use_bias=self.use_bias)
#         elif self.cell_type == 'lstm':
#             self.rnn = layers.LSTM(self.hidden_size, 
#                                    return_sequences=True, 
#                                    return_state=True, 
#                                    dropout=self.recurrent_dropout, 
#                                    recurrent_initializer=self.recurrent_initializer, 
#                                    unit_forget_bias=self.unit_forget_bias, 
#                                    use_bias=self.use_bias)
#         elif self.cell_type == 'gru':
#             self.rnn = layers.GRU(self.hidden_size, 
#                                   return_sequences=True, 
#                                   return_state=True, 
#                                   dropout=self.recurrent_dropout, 
#                                   recurrent_initializer=self.recurrent_initializer, 
#                                   use_bias=self.use_bias)
        
#         # Wrap RNN with Bidirectional if specified
#         if self.bidirectional:
#             self.rnn = layers.Bidirectional(self.rnn)
        
#         # Fully connected layers
#         if self.num_fc_layers > 0:
#             input_dim = self.hidden_size * (2 if self.bidirectional else 1)
#             for _ in range(self.num_fc_layers):
#                 self.fc_layers.append(layers.Dense(self.output_size, activation=self.activation))
#                 input_dim = self.output_size

#         # Regularization
#         if self.norm:
#             self.regularizer = layers.LayerNormalization()

#         super(RNNEncoder, self).build(input_shape)  # Call the parent class build method

#     def call(self, x):
#         x = self.embedding(x)
        
#         if self.cell_type == 'lstm':
#             output, hidden_state, cell_state = self.rnn(x)
#         else:
#             output, hidden_state = self.rnn(x)
        
#         output = self.dropout(output)

#         # Apply representation type
#         if self.representation == 'dense':
#             output = output
#         elif self.representation == 'sparse':
#             output = tf.sparse.from_dense(output)

#         # Reduce output
#         if self.reduce_output == 'sum':
#             output = tf.reduce_sum(output, axis=1)
#         elif self.reduce_output == 'mean':
#             output = tf.reduce_mean(output, axis=1)
#         elif self.reduce_output == 'last':
#             output = output[:, -1, :]

#         # Apply fully connected layers
#         for fc in self.fc_layers:
#             output = fc(output)

#         # Apply regularizer
#         if self.regularizer:
#             output = self.regularizer(output)

#         # Return both output and hidden state
#         if self.cell_type == 'lstm':
#             return output, (hidden_state, cell_state)
#         else:
#             return output, hidden_state
    
#     def encode_data(self, data):
#         encoded_data = self.call(data)
#         return encoded_data


# class AttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, config, hidden_size):
#         super(AttentionLayer, self).__init__()
#         self.config = config
#         self.recurrent_initializer = self.config['recurrent_initializer']
#         self.W = layers.Dense(hidden_size, kernel_initializer=self.recurrent_initializer)
#         self.U = layers.Dense(hidden_size, kernel_initializer=self.recurrent_initializer)
#         self.v = layers.Dense(1, kernel_initializer=self.recurrent_initializer)

#     def call(self, encoder_outputs, decoder_hidden):
#         # Expand dimensions for broadcasting
#         encoder_outputs_expanded = tf.expand_dims(encoder_outputs, 1)  # (batch_size, 1, sequence_length, hidden_size)
#         decoder_hidden_expanded = tf.expand_dims(decoder_hidden, 1)  # (batch_size, 1, hidden_size)

#         # Calculate scores
#         score = self.v(tf.nn.tanh(self.W(encoder_outputs_expanded) + self.U(decoder_hidden_expanded)))  # (batch_size, 1, sequence_length, 1)
#         attention_weights = tf.nn.softmax(score, axis=2)  # (batch_size, 1, sequence_length, 1)
        
#         # Compute context vector
#         context_vector = tf.reduce_sum(attention_weights * encoder_outputs_expanded, axis=2)  # (batch_size, hidden_size)
        
#         # Ensure context_vector and attention_weights shapes are correct
#         print(f"Context Vector Shape: {context_vector.shape}")
#         print(f"Attention Weights Shape: {attention_weights.shape}")
        
#         return context_vector, attention_weights



# class AttentionRNNDecoder(tf.keras.Model):
#     def __init__(self, config, vocab_size):
#         super(AttentionRNNDecoder, self).__init__()
#         self.config = config['model']
        
#         self.attention_size = self.config['attention_size']
#         self.embedding_size = self.config['embedding_size']
#         self.hidden_size = self.config['state_size']
#         self.output_size = self.config['output_size']
#         self.cell_type = self.config['cell_type']
#         self.vocab_size = vocab_size

#         # Initialize attention layer
#         self.attention = AttentionLayer(self.config,self.attention_size)

#         self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_size)
#         self.rnn = None

#     def build(self, input_shape):
#         if self.cell_type == 'rnn':
#             self.rnn = layers.SimpleRNN(self.hidden_size, return_sequences=True, return_state=True)
#         elif self.cell_type == 'lstm':
#             self.rnn = layers.LSTM(self.hidden_size, return_sequences=True, return_state=True)
#         elif self.cell_type == 'gru':
#             self.rnn = layers.GRU(self.hidden_size, return_sequences=True, return_state=True)
#         super(AttentionRNNDecoder, self).build(input_shape)

#     def call(self, x, encoder_outputs, encoder_hidden):
#         x = self.embedding(x)
        
#         if self.cell_type == 'lstm':
#             output, decoder_hidden, _ = self.rnn(x, initial_state=encoder_hidden)
#         else:
#             output, decoder_hidden = self.rnn(x, initial_state=encoder_hidden)

#         # Apply attention
#         context_vector, attention_weights = self.attention(encoder_outputs, decoder_hidden)
#         context_vector = tf.expand_dims(context_vector, 1)  # (batch_size, 1, hidden_size)
#         context_vector = tf.tile(context_vector, [1, tf.shape(output)[1], 1])  # (batch_size, sequence_length, hidden_size)

#         # Concatenate output with context_vector
#         output = tf.concat([output, context_vector], axis=-1)  # (batch_size, sequence_length, hidden_size + hidden_size)

#         # Further processing (if needed)
#         return output, decoder_hidden, attention_weights


# def main():
#     config_path = 'rnn_config.yaml'
#     console = Console()

#     # Load data and config
#     loader = Loader(config_path)
#     config = loader.load_config(config_path)
#     print("\nUser specified config file\n")
#     pprint(config)
    
#     data = loader.load_dataset()

#     # clean the data
#     cleaner = DataCleaner(config)
#     data = cleaner.clean_data(data)

#     md = Markdown('# Preprocessing')
#     console.print(md)
#     # Preprocess data
#     preprocessor = TextPreprocessor(config)
#     data = preprocessor.preprocess_dataset(data)

#     source_texts = data['source_text']
#     target_texts = data['target_text']
#     source_vectorizer = TextVectorization(max_tokens=None, output_mode='int')
#     target_vectorizer = TextVectorization(max_tokens=None, output_mode='int')
#     source_vectorizer.adapt(source_texts)
#     target_vectorizer.adapt(target_texts)
#     source_vocab_size = len(source_vectorizer.get_vocabulary())
#     target_vocab_size = len(target_vectorizer.get_vocabulary())

#     # Split data
#     splitter = DataSplitter(config)
#     train_set, validation_set, test_set = splitter.split_data(data)

#     table = Table(title=f"Dataset statistics\nTotal dataset: {len(train_set)+len(validation_set)+len(test_set)}")
#     table.add_column("Dataset", style="cyan")
#     table.add_column("Size (in Rows)")
#     table.add_column("Size (in memory)")
#     table.add_row("Train set", str(len(train_set)), f"{(sys.getsizeof(train_set) / (1024 * 1024)):.2f} MB")
#     table.add_row("Validation set", str(len(validation_set)), f"{(sys.getsizeof(validation_set) / (1024 * 1024)):.2f} MB")
#     table.add_row("Test set", str(len(test_set)), f"{(sys.getsizeof(test_set) / (1024 * 1024)):.2f} MB")

#     console.print(table)

#     rnn_encoder = RNNEncoder(config, source_vocab_size)
#     rnn_decoder = AttentionRNNDecoder(config, target_vocab_size)

#     # Example input data (replace with your actual input data)
#     example_source_data = tf.random.uniform((32, 10), dtype=tf.int32, maxval=source_vocab_size)  # Example shape (batch_size, sequence_length)
#     example_target_data = tf.random.uniform((32, 10), dtype=tf.int32, maxval=target_vocab_size)  # Example shape (batch_size, sequence_length)

#     # Encode the data
#     encoder_output = rnn_encoder(example_source_data)

#     if isinstance(encoder_output, tuple):
#         if len(encoder_output) == 3:  # LSTM
#             encoder_outputs, encoder_hidden, encoder_cell = encoder_output
#             encoder_hidden = (encoder_hidden, encoder_cell)
#         elif len(encoder_output) == 2:  # GRU or SimpleRNN
#             encoder_outputs, encoder_hidden = encoder_output
#         else:
#             raise ValueError(f"Unexpected number of values returned by encoder: {len(encoder_output)}")
#     else:
#         raise ValueError("Encoder output is not a tuple")

#     # Decode the data
#     decoder_output, decoder_hidden, attention_weights = rnn_decoder(example_target_data, encoder_outputs, encoder_hidden)
#     print("Decoder Output:")
#     print(decoder_output)
#     print("Attention Weights:")
#     print(attention_weights)

# if __name__ == "__main__":
#     main()



# import yaml
# import pandas as pd
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.model_selection import train_test_split
# import string
# import numpy as np
# import json

# import tensorflow as tf
# from tensorflow.keras.layers import TextVectorization
# from tensorflow.keras import layers
# from tensorflow.keras import initializers


# from rich.console import Console
# from rich.table import Table
# from rich.markdown import Markdown
# from rich.pretty import pprint
# import sys

# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
# from scipy.special import softmax
# from tqdm import tqdm

# from sklearn.metrics import accuracy_score, precision_score, recall_score
# global vocab_size


# # nltk.download('punkt')
# # nltk.download('stopwords')

# class Loader:
#     def __init__(self, config_path):
#         self.config = self.load_config(config_path)
#         self.data = None

#     def load_config(self, config_path):
#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)
#         return config

#     def load_dataset(self):
#         dataset_config = self.config['dataset']
#         self.data = pd.read_csv(dataset_config['path'], delimiter=dataset_config['delimiter'])
#         return self.data

# class DataCleaner:
#     def __init__(self, config):
#         self.config = config

#     def clean_data(self, data):
#         data.dropna(inplace=True)
#         data.drop_duplicates(inplace=True)
#         return data
    
# class TextPreprocessor:
#     def __init__(self, config):
#         self.config = config['preprocessing']['text']

#     def preprocess_text(self, text):
#         if self.config.get('lower_case'):
#             text = text.lower()
#         if self.config.get('remove_punctuation'):
#             text = text.translate(str.maketrans('', '', string.punctuation))
#         tokens = self.tokenize_text(text)
#         if self.config.get('remove_stopwords'):
#             stop_words = set(stopwords.words('english'))
#             tokens = [word for word in tokens if word not in stop_words]
#         if self.config.get('stemming'):
#             stemmer = nltk.PorterStemmer()
#             tokens = [stemmer.stem(word) for word in tokens]
#         return ' '.join(tokens)

#     def tokenize_text(self, text):
#         method = self.config['tokenization']['method']
#         if method == 'word':
#             tokens = word_tokenize(text)
#         elif method == 'sentence':
#             tokens = nltk.sent_tokenize(text)
#         else:
#             raise ValueError(f"Unsupported tokenization method: {method}")
#         return tokens

#     def preprocess_dataset(self, data):
#         data['text'] = data['text'].apply(lambda x: self.preprocess_text(x))
#         return data

 
# class DataSplitter:
#     def __init__(self, config):
#         self.config = config['preprocessing']['split']
#         self.train_data = None
#         self.test_data = None
#         self.validation_data = None

#     def split_data(self, data):
#         train_percent = self.config['train']
#         test_percent = self.config['test']
#         validation_percent = self.config['validation']
#         random_seed = self.config.get('random_seed', None)

#         # Calculate the sizes for each split
#         test_size = test_percent / (test_percent + validation_percent)
#         validation_size = validation_percent / (test_percent + validation_percent)

#         # Shuffle the data
#         data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

#         # First split: into training and remaining data (test + validation)
#         self.train_data, remaining_data = train_test_split(data, test_size=(test_percent + validation_percent), random_state=random_seed)

#         # Second split: remaining data into test and validation sets
#         self.test_data, self.validation_data = train_test_split(remaining_data, test_size=test_size, random_state=random_seed)

#         self.save_hdf5()

#         return self.train_data, self.test_data, self.validation_data
    
#     def save_hdf5(self):
#         self.train_data.to_hdf(f'dataset.training.hdf5', key='train', mode='w')
#         print("\nWriting preprocessed training set cache to dataset.training.hdf5")
#         self.test_data.to_hdf(f'dataset.test.hdf5', key='test', mode='w')
#         print("Writing preprocessed test set cache to dataset.test.hdf5")
#         self.validation_data.to_hdf(f'dataset.validation.hdf5', key='validation', mode='w')
#         print("Writing preprocessed validation set cache to dataset.validation.hdf5\n")

# class RNNEncoder(tf.keras.Model):
#     def __init__(self, config,vocab_size):
#         super(RNNEncoder, self).__init__()
#         self.config=config['model']

#         self.embedding_size = self.config['embedding_size']
#         self.hidden_size = self.config['state_size']
#         self.output_size = self.config['output_size']
#         self.num_layers = self.config['num_layers']
#         self.bidirectional = self.config['bidirectional']
#         self.cell_type = self.config['cell_type']
#         self.representation = self.config['representation']
#         self.recurrent_dropout = self.config['recurrent_dropout']
#         self.recurrent_initializer = self.config['recurrent_initializer']
#         self.use_bias = self.config['use_bias']
#         self.activation = self.config['activation']
#         self.unit_forget_bias = self.config['unit_forget_bias']
#         self.reduce_output = self.config['reduce_output']
#         self.num_fc_layers = self.config['num_fc_layers']
#         self.norm = self.config['norm']

#         # Embedding layer
#         self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_size)

#         # RNN cell
#         if self.cell_type == 'rnn':
#             self.rnn = layers.SimpleRNN(self.hidden_size, 
#                                         return_sequences=True, 
#                                         return_state=True, 
#                                         dropout=self.recurrent_dropout, 
#                                         recurrent_initializer=self.recurrent_initializer, 
#                                         use_bias=self.use_bias)
#         elif self.cell_type == 'lstm':
#             self.rnn = layers.LSTM(self.hidden_size, 
#                                    return_sequences=True, 
#                                    return_state=True, 
#                                    dropout=self.recurrent_dropout, 
#                                    recurrent_initializer=self.recurrent_initializer, 
#                                    unit_forget_bias=self.unit_forget_bias, 
#                                    use_bias=self.use_bias)
#         elif self.cell_type == 'gru':
#             self.rnn = layers.GRU(self.hidden_size, 
#                                   return_sequences=True, 
#                                   return_state=True, 
#                                   dropout=self.recurrent_dropout, 
#                                   recurrent_initializer=self.recurrent_initializer, 
#                                   use_bias=self.use_bias)
            
#         # Wrap RNN with Bidirectional if specified
#         if self.bidirectional:
#             self.rnn = layers.Bidirectional(self.rnn)
#         else:
#             self.rnn = self.rnn

#         self.dropout = layers.Dropout(rate=config.get('dropout', 0.0))

#         # Fully connected layers
#         self.fc_layers = []
#         if self.num_fc_layers > 0:
#             input_dim = self.hidden_size * (2 if self.bidirectional else 1)
#             for _ in range(self.num_fc_layers):
#                 self.fc_layers.append(layers.Dense(self.output_size,activation = self.activation))
#                 input_dim = self.output_size

#         # Regularization
#         if self.norm:
#             self.regularizer = layers.LayerNormalization()
#         else:
#             self.regularizer = None

#     def call(self, x):
#         x = self.embedding(x)
        
#         if self.cell_type == 'lstm':
#             output, hidden_state, cell_state = self.rnn(x)
#         else:
#             output, hidden_state = self.rnn(x)
        
#         output = self.dropout(output)

#         # Apply representation type
#         if self.representation == 'dense':
#             output = output
#         elif self.representation == 'sparse':
#             output = tf.sparse.from_dense(output)

#         # Reduce output
#         if self.reduce_output == 'sum':
#             output = tf.reduce_sum(output, axis=1)
#         elif self.reduce_output == 'mean':
#             output = tf.reduce_mean(output, axis=1)
#         elif self.reduce_output == 'last':
#             output = output[:, -1, :]

#         # Apply fully connected layers
#         for fc in self.fc_layers:
#             output = fc(output)

#         # Apply regularizer
#         if self.regularizer:
#             output = self.regularizer(output)

#         return output
    
#     def encode_data(self, data):
#         encoded_data = self.call(data)
#         return encoded_data

# class Model:
#     def __init__(self, config):
#         self.config = config
#         self.MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
#         self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
#         self.model = TFAutoModelForSequenceClassification.from_pretrained(self.MODEL)

#     def polarity_scores_roberta(self, example):
#         encoded_text = self.tokenizer(example, return_tensors='tf')
#         output = self.model(encoded_text)
#         scores = output.logits[0].numpy()
#         scores = softmax(scores)
#         scores_dict = {
#             'roberta_neg': scores[0],
#             'roberta_neu': scores[1],
#             'roberta_pos': scores[2]
#         }
#         return scores_dict
    
#     def roberta(self, data, num_samples=5):
#         samples = data.head(num_samples)
#         results = []
#         for index, row in samples.iterrows():
#             text = row['text']
#             scores = self.polarity_scores_roberta(text)
#             results.append((text, scores))
#         return results
    
#     def print_results(self, results):
#         table = Table(title="Results")
#         table.add_column("Text", justify="left")
#         table.add_column("Negative", justify="right")
#         table.add_column("Neutral", justify="right")
#         table.add_column("Positive", justify="right")
#         for text, scores in results:
#             table.add_row(text, f"{scores['roberta_neg']:.4f}", f"{scores['roberta_neu']:.4f}", f"{scores['roberta_pos']:.4f}")
#         console = Console()
#         console.print(table)


# def main():
#     config_path = 'rnn_config.yaml'
#     console = Console()

#     # Load data and config
#     loader = Loader(config_path)
#     config = loader.load_config(config_path)
#     print("\nUser specified config file\n")
#     pprint(config)
    
#     data = loader.load_dataset()

    
#     # clean the data
#     cleaner = DataCleaner(config)
#     data = cleaner.clean_data(data)

    
#     md = Markdown('# Preprocessing')
#     console.print(md)
#     # Preprocess data
#     preprocessor = TextPreprocessor(config)
#     data = preprocessor.preprocess_dataset(data)

#     text_data = data['text']
#     vectorizer = TextVectorization(max_tokens=None, output_mode='int')
#     vectorizer.adapt(text_data)
#     vocab_size = len(vectorizer.get_vocabulary())
#     #print(f"Preprocessed data looks like,\n{data.head(5)}\n") #just to verify

 
#     # Split data
#     splitter = DataSplitter(config)
#     train_set, validation_set, test_set = splitter.split_data(data)

#     table = Table(title=f"Dataset statistics\nTotal dataset: {len(train_set)+len(validation_set)+len(test_set)}")
#     table.add_column("Dataset", style = "cyan")
#     table.add_column("Size (in Rows)")
#     table.add_column("Size (in memory)")
#     table.add_row("Train set", str(len(train_set)), f"{(sys.getsizeof(train_set) / (1024 * 1024)):.2f} MB")
#     table.add_row("Validation set", str(len(validation_set)), f"{(sys.getsizeof(validation_set) / (1024 * 1024)):.2f} MB")
#     table.add_row("Test set", str(len(test_set)), f"{(sys.getsizeof(test_set) / (1024 * 1024)):.2f} MB")

#     console.print(table)

#     """model = Model(config)
#     results = model.roberta(test_set, num_samples=5)
#     model.print_results(results)"""

    
#     rnn_encoder = RNNEncoder(config, vocab_size)

#     # Example input data (replace with your actual input data)
#     example_data = tf.random.uniform((32, 10), dtype=tf.int32, maxval=vocab_size)  # Example shape (batch_size, sequence_length)

#     # Encode the data
#     encoded_data = rnn_encoder.encode_data(example_data)
#     print("Encoded Data:")
#     print(encoded_data)


# if __name__ == "__main__":
#     main()

# import yaml
# import pandas as pd
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.model_selection import train_test_split
# import string
# from rich.console import Console
# from rich.table import Table

# from transformers import AutoTokenizer
# from transformers import TFAutoModelForSequenceClassification
# from scipy.special import softmax
# from tqdm import tqdm
# # nltk.download('punkt')
# # nltk.download('stopwords')

# class Loader:
#     def __init__(self, config_path):
#         self.config = self.load_config(config_path)
#         self.data = None

#     def load_config(self, config_path):
#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)
#         return config

#     def load_dataset(self):
#         dataset_config = self.config['dataset']
#         self.data = pd.read_csv(dataset_config['path'], delimiter=dataset_config['delimiter'])
#         return self.data
    
# class DataCleaner:
#     def __init__(self, config):
#         self.config = config

#     def clean_data(self, data):
#         data.dropna(inplace=True)
#         data.drop_duplicates(inplace=True)
#         return data
    
# class TextPreprocessor:
#     def __init__(self, config):
#         self.config = config['preprocessing']['text']

#     def preprocess_text(self, text):
#         if self.config.get('lower_case'):
#             text = text.lower()
#         if self.config.get('remove_punctuation'):
#             text = text.translate(str.maketrans('', '', string.punctuation))
        
#         tokens = self.tokenize_text(text)
#         if self.config.get('remove_stopwords'):
#             stop_words = set(stopwords.words('english'))
#             tokens = [word for word in tokens if word not in stop_words]
#         if self.config.get('stemming'):
#             stemmer = nltk.PorterStemmer()
#             tokens = [stemmer.stem(word) for word in tokens]
#         return ' '.join(tokens)

#     def tokenize_text(self, text):
#         method = self.config['tokenization']['method']
#         if method == 'word':
#             tokens = word_tokenize(text)
#         elif method == 'sentence':
#             tokens = nltk.sent_tokenize(text)
#         else:
#             raise ValueError(f"Unsupported tokenization method: {method}")
#         return tokens

#     def preprocess_dataset(self, data):
#         data['text'] = data['text'].apply(lambda x: self.preprocess_text(x))
#         return data

# class DataSplitter:
#     def __init__(self, config):
#         self.config = config['training']['split']
#         self.train_data = None
#         self.test_data = None
#         self.validation_data = None

#     def split_data(self, data):
#         train_percent = self.config['train']
#         test_percent = self.config['test']
#         validation_percent = self.config['validation']

#         # Calculate the sizes for each split
#         test_size = test_percent / (test_percent + validation_percent)
#         validation_size = validation_percent / (test_percent + validation_percent)

#         # First split: into training and remaining data (test + validation)
#         self.train_data, remaining_data = train_test_split(data, test_size=(test_percent + validation_percent))

#         # Second split: remaining data into test and validation sets
#         self.test_data, self.validation_data = train_test_split(remaining_data, test_size=test_size)

#         return self.train_data, self.test_data, self.validation_data
    
# class Model:
#     def __init__(self, config):
#         self.config = config
#         self.MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
#         self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
#         self.model = TFAutoModelForSequenceClassification.from_pretrained(self.MODEL)

#     def polarity_scores_roberta(self, example):
#         encoded_text = self.tokenizer(example, return_tensors='tf')
#         output = self.model(encoded_text)
#         scores = output.logits[0].numpy()
#         scores = softmax(scores)
#         scores_dict = {
#             'roberta_neg': scores[0],
#             'roberta_neu': scores[1],
#             'roberta_pos': scores[2]
#         }
#         return scores_dict
    
#     def roberta(self, data, num_samples=5):
#         samples = data.head(num_samples)
#         results = []
#         for index, row in samples.iterrows():
#             text = row['text']
#             scores = self.polarity_scores_roberta(text)
#             results.append((text, scores))
#         return results
    
#     def print_results(self, results):
#         table = Table(title="Results")
#         table.add_column("Text", justify="left")
#         table.add_column("Negative", justify="right")
#         table.add_column("Neutral", justify="right")
#         table.add_column("Positive", justify="right")
#         for text, scores in results:
#             table.add_row(text, f"{scores['roberta_neg']:.4f}", f"{scores['roberta_neu']:.4f}", f"{scores['roberta_pos']:.4f}")
#         console = Console()
#         console.print(table) 
  
       

# def main():
#     config_path = 'config.yaml'

#     # Load data and config
#     loader = Loader(config_path)
#     config = loader.load_config(config_path)
#     data = loader.load_dataset()

#     # clean the data
#     cleaner = DataCleaner(config)
#     data = cleaner.clean_data(data)

#     # Preprocess data
#     preprocessor = TextPreprocessor(config)
#     data = preprocessor.preprocess_dataset(data)
#     print(f"Preprocessed data looks like,\n{data.head(5)}")
#     # Split data
#     splitter = DataSplitter(config)
#     train_set, validation_set, test_set = splitter.split_data(data)

#     table = Table(title="Dataset")
#     table.add_column("Train set")
#     table.add_column("Validation set")
#     table.add_column("Test set")
#     table.add_row(str(len(train_set)),str(len(validation_set)),str(len(test_set)))
#     console = Console()
#     console.print(table)

#     model = Model(config)
#     results = model.roberta(test_set,num_samples=5)
#     model.print_results(results)

# if __name__ == "__main__":
#     main()