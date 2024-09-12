import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer, FullAttentionCompletelyRemoveMissing
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

class NegativeNineMask:
    def __init__(self, x_enc, num_heads, device="cpu",completly_remove_missing=True):
        """
        Creates a boolean attention mask that masks out tokens where the first value is -9,
        expanded to match the dimensions expected by the attention scores.
        
        Args:
            x_enc (torch.Tensor): Input tensor of shape (B, L, N), where B is the batch size,
                                  L is the sequence length, and N is the number of tokens (features).
            num_heads (int): Number of attention heads.
            device (str): Device on which to create the mask (e.g., "cpu" or "cuda").
        """
        # Check if the first value in each token is -9
        batch_size, seq_len, num_tokens = x_enc.shape
        mask = (x_enc[:, :, 0] == -9).unsqueeze(1).to(device)  # Shape: (B, 1, N)
        if completly_remove_missing:
            mask = mask | mask.transpose(1, 2)  # Logical OR to combine masks, Shape: (B, L, L)
        # Expand the mask to match the expected dimensions [B, H, L, L]
        self._mask = mask.unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len)  # Shape: (B, H, L, L)

    @property
    def mask(self):
        return self._mask

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
        self.mask_ratio = 0.5#configs.mask_ratio  # Add mask ratio for masking percentage
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, configs.d_model))
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
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
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

    def apply_mask_tokens(self, input, embedded_tokens):
        """
        This method applies masking to sequences in embedded_tokens based on the presence of -9 in the input x_enc.
        It replaces the entire sequence with the learnable mask token if the first token in x_enc is -9.
        """
        batch_size, seq_length, embedding_dim = embedded_tokens.shape
        
        # Check if the first token of each sequence in x_enc is -9
        mask_condition = (input[:, 0] == -9)
        
        # Copy the input sequence to avoid modification
        masked_embedded_tokens = embedded_tokens.clone()
        
        # Apply the mask to sequences that meet the condition
        masked_embedded_tokens[mask_condition] = self.mask_token

        return masked_embedded_tokens

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        device = x_enc.device
        attn_mask = NegativeNineMask(x_enc.transpose(1, 2), num_heads=8, device=x_enc.device, completly_remove_missing=False)
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

        # Apply masking to the embedded sequence
        enc_out = self.apply_mask_tokens(x_enc, enc_out)  # Replace some patches with mask tokens
        if False:#random.randint(1, 100) == 1:
            x_enc_np = x_enc.detach().cpu().numpy()
            enc_out_np = enc_out.detach().cpu().numpy()
            
            # Select one random sample from the batch
            #random_index = np.random.randint(x_enc_np.shape[0])
            #sample_input_tokens = x_enc_np[random_index]
            sample_embedded_sequence = enc_out_np[0]
            
            # Plot the input tokens
            plt.figure(figsize=(12, 8))
            
            input_data = x_enc.detach().cpu().numpy()
            plt.subplot(2, 1, 1)
            plt.imshow(input_data[0, :, :], cmap='viridis', aspect='auto')  # Visualizing the first batch
            plt.colorbar()
            plt.title('Input Data')
            plt.xlabel('Time Step')
            plt.ylabel('Feature Dimension')
            
            # Plot the embedded sequence
            plt.subplot(2, 1, 2)
            plt.imshow(sample_embedded_sequence.T, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f'Embedded Sequence Sample')
            plt.ylabel('Embedding Dimension')
            plt.xlabel('Sequence Position')
            
            plt.tight_layout()
            plt.show()
        # Assuming enc_out has shape (batch_size, seq_length, embedding_dim)
        batch_size, seq_length, embedding_dim = enc_out.shape
        if False:
            # Calculate the length of each part
            part_length = seq_length // 4

            # Split the input tensor into four equal parts
            enc_out_part1 = enc_out[:, :part_length, :]
            enc_out_part2 = enc_out[:, part_length:2*part_length, :]
            enc_out_part3 = enc_out[:, 2*part_length:3*part_length, :]
            enc_out_part4 = enc_out[:, 3*part_length:, :]

            # Apply positional encoding to each part
            enc_out_part1 = self.positional_encoding(enc_out_part1)
            enc_out_part2 = self.positional_encoding(enc_out_part2)
            enc_out_part3 = self.positional_encoding(enc_out_part3)
            enc_out_part4 = self.positional_encoding(enc_out_part4)

            # Concatenate the encoded parts back together
            enc_out = torch.cat((enc_out_part1, enc_out_part2, enc_out_part3, enc_out_part4), dim=1)
        else:
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

        if False:#random.randint(1, 100) == 1:
            # Plotting the attention matrix and the input
            attention_matrix = attns[0].detach().cpu().numpy()  # Assuming attns is a list of attention matrices
            input_data = x_enc.detach().cpu().numpy()
            
            # Average the attention weights over all heads
            avg_attention_weights = np.mean(attention_matrix[:, :, :, :], axis=1)  # Averaging over the heads dimension
            
            # Ensure the dimensions align for matrix multiplication
            # Transpose input_data to match the dimensions
            input_data_transposed = np.transpose(input_data[0, :, :], (1, 0))
            
            # Matrix multiply the input data with the attention matrix
            attention_applied = np.matmul(avg_attention_weights[0, :, :], input_data_transposed)
            
            plt.figure(figsize=(20, 12))
            
            # Plot attention matrix
            plt.subplot(2, 1, 1)
            plt.imshow(avg_attention_weights[0, :, :], cmap='plasma', aspect='auto')  # Visualizing the averaged attention weights of the first batch
            plt.colorbar()
            plt.title('Attention Matrix (Averaged over Heads)')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            
            # Plot input data
            plt.subplot(2, 1, 2)
            plt.imshow(input_data[0, :, :], cmap='viridis', aspect='auto')  # Visualizing the first batch
            plt.colorbar()
            plt.title('Input Data')
            plt.xlabel('Time Step')
            plt.ylabel('Feature Dimension')
            
            # Plot attention applied to input data
            #plt.subplot(3, 1, 3)
            #plt.imshow(attention_applied.T, cmap='inferno', aspect='auto')  # Visualizing the attention applied to the input data with swapped axes
            #plt.colorbar()
            #plt.title('Attention Applied to Input Data')
            #plt.xlabel('Query Position')
            #plt.ylabel('Feature Dimension')
            
            plt.show()

        return dec_out[:,:,:1]


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False