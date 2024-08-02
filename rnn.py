import string
import numpy as np
import yaml
import sys
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
import matplotlib.pyplot as plt
import argparse

class TranslationPreprocessor:
    def __init__(self, path_to_data, config, start_idx=1000, end_idx=20000):
        self.path_to_data = path_to_data
        self.config = config
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pairs = self.load_data()
        self.english_sentences = [self.clean_sentence(pair[0]) for pair in self.pairs]
        self.spanish_sentences = [self.clean_sentence(pair[1]) for pair in self.pairs]

    def load_data(self):
        pairs = []
        with open(self.path_to_data, "r", encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(self.config['dataset']['delimiter'])
                if len(parts) >= 2:
                    pairs.append(parts[:2])
        return pairs[self.start_idx:self.end_idx]

    @staticmethod
    def clean_sentence(sentence):
        lower_case_sent = sentence.lower()
        string_punctuation = string.punctuation + "¡" + '¿'
        return lower_case_sent.translate(str.maketrans('', '', string_punctuation))

    def tokenize(self, sentences):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)
        return tokenizer.texts_to_sequences(sentences), tokenizer

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.preprocessor = TranslationPreprocessor(config['dataset']['path'], config)

    def process_data(self):
        spa_text_tokenized, spa_text_tokenizer = self.preprocessor.tokenize(self.preprocessor.spanish_sentences)
        eng_text_tokenized, eng_text_tokenizer = self.preprocessor.tokenize(self.preprocessor.english_sentences)

        spanish_vocab_size = len(spa_text_tokenizer.word_index) + 1
        english_vocab_size = len(eng_text_tokenizer.word_index) + 1
        max_spanish_len = len(max(spa_text_tokenized, key=len))
        max_english_len = len(max(eng_text_tokenized, key=len))

        spa_pad_sentence = pad_sequences(spa_text_tokenized, max_spanish_len, padding="post")
        eng_pad_sentence = pad_sequences(eng_text_tokenized, max_english_len, padding="post")
        eng_pad_sentence = eng_pad_sentence.reshape(*eng_pad_sentence.shape, 1)

        return (spa_pad_sentence, eng_pad_sentence, spa_text_tokenizer, eng_text_tokenizer,
                spanish_vocab_size, english_vocab_size, max_spanish_len, max_english_len)

    def process_sentence(self, sentence, tokenizer, max_len):
        sentence = self.preprocessor.clean_sentence(sentence)
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])
        pad_sentence = pad_sequences(tokenized_sentence, max_len, padding="post")
        return pad_sentence

class Encoder:
    def __init__(self, config, spanish_vocab_size, max_spanish_len):
        self.config = config
        self.spanish_vocab_size = spanish_vocab_size
        self.max_spanish_len = max_spanish_len

    def build(self):
        encoder_inputs = Input(shape=(self.max_spanish_len,), name='encoder_inputs')
        embedding = Embedding(input_dim=self.spanish_vocab_size, output_dim=self.config['model']['embedding_size'])(encoder_inputs)
        
        rnn_cell = GRU if self.config['model']['cell_type'].lower() == 'gru' else LSTM
        rnn_layer = rnn_cell(self.config['model']['state_size'], return_sequences=False, dropout=self.config['model']['dropout'])

        if self.config['model'].get('bidirectional', False):
            rnn_layer = Bidirectional(rnn_layer)

        encoder_rnn = rnn_layer(embedding)
        
        x = encoder_rnn
        for _ in range(self.config['model'].get('num_fc_layers', 0)):
            x = Dense(self.config['model']['state_size'], activation='relu')(x)
            if self.config['model'].get('dropout', 0) > 0:
                x = Dropout(self.config['model']['dropout'])(x)

        return encoder_inputs, x

class Decoder:
    def __init__(self, config, english_vocab_size, max_english_len):
        self.config = config
        self.english_vocab_size = english_vocab_size
        self.max_english_len = max_english_len

    def build(self, encoder_rnn):
        repeat_vector = RepeatVector(self.max_english_len)(encoder_rnn)
        
        rnn_cell = GRU if self.config['model']['cell_type'].lower() == 'gru' else LSTM
        rnn_layer = rnn_cell(self.config['model']['state_size'], return_sequences=True, dropout=self.config['model']['dropout'])

        if self.config['model'].get('bidirectional', False):
            rnn_layer = Bidirectional(rnn_layer)

        decoder_rnn = rnn_layer(repeat_vector)
        
        x = decoder_rnn
        for _ in range(self.config['model'].get('num_fc_layers', 0)):
            x = TimeDistributed(Dense(self.config['model']['state_size'], activation='relu'))(x)
            if self.config['model'].get('dropout', 0) > 0:
                x = TimeDistributed(Dropout(self.config['model']['dropout']))(x)

        logits = TimeDistributed(Dense(self.english_vocab_size))(x)
        return logits

class Seq2SeqModel:
    def __init__(self, config, spanish_vocab_size, english_vocab_size, max_spanish_len, max_english_len):
        self.config = config
        self.validation_split = self.config['training']['split']['validation']
        self.spanish_vocab_size = spanish_vocab_size
        self.english_vocab_size = english_vocab_size
        self.max_spanish_len = max_spanish_len
        self.max_english_len = max_english_len
        self.model = self.build_model()

    def build_model(self):
        encoder = Encoder(self.config, self.spanish_vocab_size, self.max_spanish_len)
        encoder_inputs, encoder_rnn = encoder.build()

        decoder = Decoder(self.config, self.english_vocab_size, self.max_english_len)
        logits = decoder.build(encoder_rnn)

        model = Model(encoder_inputs, Activation('softmax')(logits))
        model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(self.config['training']['learning_rate']),
                      metrics=['accuracy'])
        return model

    def train(self, spa_pad_sentence, eng_pad_sentence, batch_size, epochs):
        num_samples = len(spa_pad_sentence)
        split_index = int(num_samples * (1 - self.validation_split))
        train_data = (spa_pad_sentence[:split_index], eng_pad_sentence[:split_index])
        val_data = (spa_pad_sentence[split_index:], eng_pad_sentence[split_index:])
        
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            start_time = time.time()
            
            # Training step
            train_loss_metric = tf.keras.metrics.Mean()
            train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

            for batch_start in range(0, len(train_data[0]), batch_size):
                batch_end = min(batch_start + batch_size, len(train_data[0]))
                batch_inputs = train_data[0][batch_start:batch_end]
                batch_targets = train_data[1][batch_start:batch_end]

                with tf.GradientTape() as tape:
                    logits = self.model(batch_inputs, training=True)
                    loss = self.loss_fn(batch_targets, logits)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                train_loss_metric.update_state(loss)
                train_accuracy_metric.update_state(batch_targets, logits)
                
            # Validation step
            val_loss_metric = tf.keras.metrics.Mean()
            val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

            for batch_start in range(0, len(val_data[0]), batch_size):
                batch_end = min(batch_start + batch_size, len(val_data[0]))
                batch_inputs = val_data[0][batch_start:batch_end]
                batch_targets = val_data[1][batch_start:batch_end]

                logits = self.model(batch_inputs, training=False)
                loss = self.loss_fn(batch_targets, logits)

                val_loss_metric.update_state(loss)
                val_accuracy_metric.update_state(batch_targets, logits)

            end_time = time.time()
            duration = end_time - start_time

            history['loss'].append(train_loss_metric.result().numpy())
            history['accuracy'].append(train_accuracy_metric.result().numpy())
            history['val_loss'].append(val_loss_metric.result().numpy())
            history['val_accuracy'].append(val_accuracy_metric.result().numpy())

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Time: {duration:.2f}s")
            print(f"Train Loss: {history['loss'][-1]:.4f} - Accuracy: {history['accuracy'][-1]:.4f}")
            print(f"Validation Loss: {history['val_loss'][-1]:.4f} - Accuracy: {history['val_accuracy'][-1]:.4f}")

        # Save the weights after training
        if 'weights_path' in self.config['training']:
            self.model.save_weights(self.config['training']['weights_path'])
            print(f"Weights saved to {self.config['training']['weights_path']}")

        return history

    def loss_fn(self, targets, logits):
        return sparse_categorical_crossentropy(targets, logits, from_logits=True)

    def predict(self, input_sentence, max_len, tokenizer):
        start_time = time.time()
        processed_sentence = DataProcessor(self.config).process_sentence(input_sentence, tokenizer, max_len)
        #print(f"Processed sentence shape: {processed_sentence.shape}")  # Debug: Check the shape
        logits = self.model.predict(processed_sentence,verbose=0)
        predicted_indices = np.argmax(logits, axis=-1)[0]
        translated_sentence = ' '.join([Seq2SeqModel.index_to_word(tokenizer, idx) for idx in predicted_indices])
        end_time = time.time()
        duration = end_time - start_time
        print(f"Prediction Time: {duration:.2f}s")
        return translated_sentence

    @staticmethod
    def index_to_word(tokenizer, index):
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                return word
        return '<unk>'



def main():
    parser = argparse.ArgumentParser(description='Seq2Seq Model Training and Prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help='Mode: train or predict')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    data_processor = DataProcessor(config)
    spa_pad_sentence, eng_pad_sentence, spa_text_tokenizer, eng_text_tokenizer, spanish_vocab_size, english_vocab_size, max_spanish_len, max_english_len = data_processor.process_data()

    model = Seq2SeqModel(config, spanish_vocab_size, english_vocab_size, max_spanish_len, max_english_len)

    if args.mode == 'train':
        history = model.train(spa_pad_sentence, eng_pad_sentence,
                              batch_size=config['training']['batch_size'],
                              epochs=config['training']['epochs'])

        # Plot the training and validation loss/accuracy
        df = pd.DataFrame({
            'Epoch': range(1, len(history['loss']) + 1),
            'Train Loss': history['loss'],
            'Train Accuracy': history['accuracy'],
            'Validation Loss': history['val_loss'],
            'Validation Accuracy': history['val_accuracy']
        })

        print(df)
        df.plot(x='Epoch', y=['Train Loss', 'Validation Loss'], kind='line', title='Loss')
        plt.show()

        df.plot(x='Epoch', y=['Train Accuracy', 'Validation Accuracy'], kind='line', title='Accuracy')
        plt.show()

    elif args.mode == 'predict':
        if 'weights_path' in config['training']:
            model.model.load_weights(config['training']['weights_path'])
        else:
            print("Weights path not specified in the configuration file.")
            return

        while True:
            input_sentence = input("Enter a sentence to translate (or type 'exit' to quit): ")
            if input_sentence.lower() == 'exit':
                break
            translated_sentence = model.predict(input_sentence, max_spanish_len, spa_text_tokenizer)
            print(f"Translated Sentence: {translated_sentence}")

if __name__ == '__main__':
    main()
