import torch
import torch.nn as nn
import torch.nn.functional as F

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