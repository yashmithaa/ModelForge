import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_feedforward, max_seq_len, num_classes, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.PositionalEncoding(d_model, max_seq_len)
        self.encoder = self.TransformerEncoder(num_layers, d_model, num_heads, dim_feedforward, dropout)
        self.decoder = self.TransformerDecoder(num_layers, d_model, num_heads, dim_feedforward, dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        memory = self.encoder(src, src_mask)

        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask)

        output = output.mean(dim=1)  # Aggregate over sequence length
        output = self.fc(output)
        return output

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            self.encoding = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
            self.encoding[:, 0::2] = torch.sin(position * div_term)
            self.encoding[:, 1::2] = torch.cos(position * div_term)
            self.encoding = self.encoding.unsqueeze(0)

        def forward(self, x):
            return x + self.encoding[:, :x.size(1)].detach()

    class TransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        def forward(self, src, mask=None):
            src2 = self.self_attn(src, src, src, attn_mask=mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src

    class TransformerEncoder(nn.Module):
        def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout):
            super().__init__()
            self.layers = nn.ModuleList([
                TransformerModel.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])

        def forward(self, src, mask=None):
            for layer in self.layers:
                src = layer(src, mask)
            return src

    class TransformerDecoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, dim_feedforward, dropout):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model)
            )
            self.layer_norm1 = nn.LayerNorm(d_model)
            self.layer_norm2 = nn.LayerNorm(d_model)
            self.layer_norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
            # Self-attention
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.layer_norm1(tgt)

            # Multi-head attention
            tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.layer_norm2(tgt)

            # Feed-forward network
            tgt2 = self.ffn(tgt)
            tgt = tgt + self.dropout(tgt2)
            tgt = self.layer_norm3(tgt)

            return tgt

    class TransformerDecoder(nn.Module):
        def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout):
            super().__init__()
            self.layers = nn.ModuleList([
                TransformerModel.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])

        def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
            for layer in self.layers:
                tgt = layer(tgt, memory, tgt_mask, memory_mask)
            return tgt

    class LayerNormalization(nn.Module):
        def __init__(self, parameters_shape, eps=1e-5):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(parameters_shape))
            self.beta = nn.Parameter(torch.zeros(parameters_shape))
            self.eps = eps

        def forward(self, input):
            dims = [-(i + 1) for i in range(len(input.size()) - 1)]
            mean = input.mean(dim=dims, keepdim=True)
            var = ((input - mean) ** 2).mean(dim=dims, keepdim=True)
            std = (var + self.eps).sqrt()
            y = (input - mean) / std
            out = self.gamma * y + self.beta
            return out
 