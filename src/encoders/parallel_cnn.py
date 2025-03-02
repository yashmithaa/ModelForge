import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelCNN(nn.Module):
    def __init__(self, config):
        super(ParallelCNN, self).__init__()
        self.embedding = nn.Embedding(config['params']['vocab_size'], config['params']['embedding_size'])
        self.convs = nn.ModuleList([
            nn.Conv2d(1, config['params']['num_filters'], (k, config['params']['embedding_size']))
            for k in config['params']['filter_sizes']
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
    