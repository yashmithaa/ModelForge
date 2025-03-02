import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import RNNEncoder, ParallelCNN
from .decoders import RNNDecoder, NumericalDecoder, CategoricalDecoder
from .combiner import Combiner

class ModelArch(nn.Module):
    def __init__(self, config):
        super(ModelArch, self).__init__()
        self.encoders = nn.ModuleList()

        for feature in config['input_features']:
            if feature['encoder'] == 'rnn':
                self.encoders.append(RNNEncoder(feature))
            elif feature['encoder'] == 'parallel_cnn':
                self.encoders.append(ParallelCNN(feature))
            # Add other encoders here as needed

        self.combiner = Combiner(config)
        self.decoder = RNNDecoder(config)
        self.config = config

    def forward(self, encoder_inputs, decoder_input):
        encoder_outputs = []

        for encoder, input in zip(self.encoders, encoder_inputs):
            encoder_outputs.append(encoder(input))

        combined_output = self.combiner(encoder_outputs)
        
        # Initialize the hidden state for the decoder
        batch_size = combined_output.size(0)
        hidden = (torch.zeros(1, batch_size, self.config['decoder']['hidden_size']).to(combined_output.device),
                  torch.zeros(1, batch_size, self.config['decoder']['hidden_size']).to(combined_output.device))
        
        decoder_output, _ = self.decoder(decoder_input, hidden)
        return decoder_output