import torch
import torch.nn as nn
import torch.nn.functional as F

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