import torch
import torch.nn as nn
import torch.nn.functional as F

class Combiner(nn.Module):
    def __init__(self, config):
        super(Combiner, self).__init__()
        self.config = config
        self.combiner_type = config['combiner']['type']
        self.output_size = config['combiner']['output_size']

        if self.combiner_type == 'concat':
            input_size = sum([feature['params']['output_size'] for feature in config['input_features']])
            self.combiner = nn.Linear(input_size, self.output_size)
        elif self.combiner_type == 'sum':
            input_size = config['input_features'][0]['params']['output_size']
            self.combiner = nn.Linear(input_size, self.output_size)
        else:
            raise ValueError(f"Unsupported combiner type: {self.combiner_type}")

    def forward(self, encoder_outputs):
        if self.combiner_type == 'concat':
            combined_output = torch.cat(encoder_outputs, dim=-1)
        elif self.combiner_type == 'sum':
            combined_output = torch.sum(torch.stack(encoder_outputs), dim=0)
            
        return self.combiner(combined_output)