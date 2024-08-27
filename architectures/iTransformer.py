import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted
import numpy as np
import matplotlib.pyplot as plt
import random
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TriangularCausalMask2():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

    def update_mask(self, x_enc):
        """
        Update the mask to remove tokens that start with -9 in x_enc.
        
        Args:
            x_enc (torch.Tensor): The input tensor of shape [B, L, D].
        """
        B, L, D = x_enc.shape
        for b in range(B):
            for l in range(L):
                if x_enc[b, l, 0] == -9:  # Assuming the first variate is the one to check
                    self._mask[b, 0, l, :] = 0  # Remove the token in the attention mask

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy

        self.positional_encoding = PositionalEncoding(configs.d_model)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        #attn_mask = TriangularCausalMask2(x_enc.shape[0], x_enc.shape[1], device=x_enc.device)
        #attn_mask.update_mask(x_enc)
        attn_mask = None
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        enc_out = self.positional_encoding(enc_out)
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=attn_mask)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        #if random.randint(1, 100) == 1:
        #    # Plotting the attention matrix and the input
        #    attention_matrix = attns[0].detach().cpu().numpy()  # Assuming attns is a list of attention matrices
        #    input_data = x_enc.detach().cpu().numpy()
        #    plt.figure(figsize=(20, 8))
        #    # Plot attention matrix
        #    plt.subplot(2, 1, 1)
        #    #avg_attention_weights = np.mean(attention_matrix[0, 0, :, :15], axis=0).reshape(1, -1)
        #    #plt.imshow(avg_attention_weights, cmap='viridis', aspect='auto')
        #    plt.imshow(attention_matrix[0, 0, :, :], cmap='viridis', aspect='auto')  # Visualizing the first head of the first batch
        #    plt.colorbar()
        #    plt.title('Attention Matrix')
        #    plt.xlabel('Key Position')
        #    plt.ylabel('Query Position')
        #    # Plot input data
        #    plt.subplot(2, 1, 2)
        #    plt.imshow(input_data[0,:,:], cmap='viridis', aspect='auto')  # Visualizing the first batch
        #    plt.colorbar()
        #    plt.title('Input Data')
        #    plt.xlabel('Feature Dimension')
        #    plt.ylabel('Time Step')
        #    plt.show()

        return dec_out[:,:,:1]


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False