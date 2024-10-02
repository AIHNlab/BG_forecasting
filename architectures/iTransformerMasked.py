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
        self.d_model = d_model

        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

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
        
class ReconstructionHead(nn.Module):
    def __init__(self, d_model, pred_len):
        super(ReconstructionHead, self).__init__()
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, enc_out, N):
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        return dec_out
    
class EncoderModel(nn.Module):
    def __init__(self, configs):
        super(EncoderModel, self).__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.n_features = configs.n_features
        self.embed = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.positional_encoding = PositionalEncoding(configs.d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=True),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, attn_mask):
        enc_out = self.embed(x_enc, x_mark_enc)
        #enc_out = self.positional_encoding(enc_out)
        enc_out_parts = torch.chunk(enc_out, self.n_features, dim=1)  # Split along the second dimension (sequence length)

        # Apply positional encoding to each part
        encoded_parts = [self.positional_encoding(part) for part in enc_out_parts]

        # Combine the parts back together
        enc_out = torch.cat(encoded_parts, dim=1)

        enc_out, attns = self.encoder(enc_out, attn_mask=attn_mask)
        return enc_out, attns

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.mask_ratio = 0.5 # NB. Not used
        self.missing_token = nn.Parameter(torch.randn(1, 1, configs.d_model))
        self.mask_token = nn.Parameter(torch.randn(1, 1, configs.d_model))
        #self.missing_token = torch.ones(1, 1, configs.d_model)  # Initialize with ones
        #self.mask_token = torch.zeros(1, 1, configs.d_model)    # Initialize with zeros
        self.encoder_model = EncoderModel(configs)
        self.projector_model = ReconstructionHead(configs.d_model, configs.pred_len)
        self.tb_writer = None
        self.iter_count = 0

    # Should set the embedded token to self.mask_token if the first value of the input is -9
    def apply_mask_tokens(self, input, embedded_tokens, apply_random_tokens=False):
        """
        Replaces the embedded tokens with self.mask_token where the first value of the input is -9
        and also replaces the first embedded token with the mask token.

        Args:
            input (torch.Tensor): The original input tensor x_enc of shape (Batch, Time, Variate).
            embedded_tokens (torch.Tensor): The embedded tokens of shape (Batch, Variate, d_model).

        Returns:
            torch.Tensor: The masked embedded tokens of shape (Batch, Variate, d_model).
        """
        device = embedded_tokens.device
        self.missing_token = self.missing_token.to(device)
        self.mask_token = self.mask_token.to(device)

        # Create a mask for the first token
        random_token_mask = torch.zeros_like(embedded_tokens, dtype=torch.bool)
        #add historty of days and missing days (also to config)
        if apply_random_tokens:
            batch_size, _ , _ = embedded_tokens.shape
            for i in range(batch_size):
                random_numbers = [random.randint(0, 30) for _ in range(8)]
                random_token_mask[i, random_numbers, :] = True

        # Create a mask where the first value along Time is -9 for each Variate
        mask = (input[:, 0, :] == -9)  # Shape: (Batch, Variate)
        mask = mask.unsqueeze(-1)      # Shape: (Batch, Variate, 1)
        mask = mask.expand(-1, -1, embedded_tokens.size(-1))  # Shape: (Batch, Variate, d_model)

        # Combine the masks
        combined_mask = random_token_mask & ~mask
        masked_tokens = combined_mask[:,:,1]
        # Replace embedded tokens with mask_token where random_token_mask is True
        embedded_tokens = torch.where(random_token_mask, self.mask_token, embedded_tokens)

        # Replace embedded tokens with missing_token where mask is True
        embedded_tokens = torch.where(mask, self.missing_token, embedded_tokens)

        return embedded_tokens, masked_tokens

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_masked_tokens=False):
        device = x_enc.device
        attn_mask = NegativeNineMask(x_enc.transpose(1, 2), num_heads=8, device=x_enc.device, completly_remove_missing=False)
        #attn_mask = None
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out, attns = self.encoder_model(x_enc, x_mark_enc, attn_mask=attn_mask)
        enc_out, masked_tokens = self.apply_mask_tokens(x_enc, enc_out, apply_random_tokens=False)
        #masked_tokens = None
        dec_out = self.projector_model(enc_out, N)

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #if random.randint(1, 100) == 1:
        if self.iter_count % 5000 == 0:
            self.plot_attention(x_enc, enc_out, attns, dec_out, masked_tokens)
        if return_masked_tokens:
            return dec_out, masked_tokens
        else:
            return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,return_masked_tokens=False):
        self.iter_count += 1
        if return_masked_tokens:
            dec_out, masked_tokens = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_masked_tokens)
            return dec_out[:, -self.pred_len:, :], masked_tokens
        else:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_masked_tokens)
            return dec_out[:, -self.pred_len:, :]

    def freeze_encoder(self):
        for param in self.encoder_model.parameters():
            param.requires_grad = False

    def plot_attention(self, x_enc, enc_out, attns, dec_out, masked_tokens):
        found = False
        #for i in range(x_enc.shape[0]):
        #    batch = masked_tokens[i]
        #    for j in range(batch.shape[0]):
        #        if batch[j]:
        #            batch_index = i
        #            token = j
        #            found = True
        #            break
        #    if found:
        #        break
        if not found:
            print("No masked tokens found")
            batch_index = 0
            token = 0
        # Convert tensors to numpy arrays
        x_enc_np = x_enc.detach().cpu().numpy()
        enc_out_np = enc_out.detach().cpu().numpy()
        input_data = x_enc_np
        dec_out_np = dec_out.detach().cpu().numpy()

        # Select the first sample from the batch
        sample_embedded_sequence = enc_out_np[batch_index]

        # Prepare the attention matrix
        attention_matrix = attns[batch_index].detach().cpu().numpy()  # Assuming attns is a list of attention matrices

        # Average the attention weights over all heads
        avg_attention_weights = np.mean(attention_matrix, axis=1)  # Averaging over the heads dimension

        # Ensure the dimensions align for matrix multiplication
        # Transpose input_data to match the dimensions
        #input_data_transposed = np.transpose(input_data[0], (1, 0))

        #attention_weights = avg_attention_weights[0, 0][:, np.newaxis]  # Shape: (L_k, 1)
        #input_data_sample = input_data[0]                               # Shape: (L_k, N)

        # Element-wise multiplication
        #attention_applied = attention_weights * input_data_sample.T       # Shape: (L_k, N)
        #attention_applied = attention_applied.T
        input_data_with_nan = np.where(input_data == -9, np.nan, input_data)
        # Now plot all in subplots
        plt.figure(figsize=(20, 12))
        # Determine the common color scale range
        #vmin = np.nanmin(input_data_with_nan)
        #vmax = np.nanmax(input_data_with_nan)
        dec_out_np[batch_index][:, np.isnan(input_data_with_nan[batch_index]).any(axis=0)] = np.nan
        vmin = min(
            np.nanquantile(dec_out_np[batch_index], 0.05),
            np.nanquantile(input_data_with_nan[batch_index], 0.05)
        )
        vmax = max(
            np.nanquantile(dec_out_np[batch_index], 0.95),
            np.nanquantile(input_data_with_nan[batch_index], 0.95)
        )
        # Subplot 1: Input Data
        plt.subplot(2, 2, 1)
        plt.imshow(input_data_with_nan[batch_index], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)  # Visualizing the first batch
        plt.colorbar()
        plt.title('Input Data')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Dimension')

        # Count the number of non-NaN columns
        non_nan_columns = np.sum(~np.isnan(input_data_with_nan[batch_index]).any(axis=0))


        if False:
            # Subplot 2: Embedded Sequence
            plt.subplot(2, 2, 2)
            plt.imshow(sample_embedded_sequence.T, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title('Embedded Sequence Sample')
            plt.ylabel('Embedding Dimension')
            plt.xlabel('Sequence Position')
        else:
            # Subplot 2: dec_out
            plt.subplot(2, 2, 2)
            
            plt.imshow(dec_out_np[batch_index], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title('Decoder Output (dec_out) Sample')
            plt.ylabel('Output Dimension')
            plt.xlabel('Sequence Position')

        # Replace 0 values with NaN in avg_attention_weights
        avg_attention_weights_with_nan = np.where(avg_attention_weights == 0, np.nan, avg_attention_weights)

        # Replace 0 values with NaN in the tiled attention weights for the first token
        tiled_attention_with_nan = np.tile(avg_attention_weights_with_nan[batch_index, token], (input_data[batch_index].shape[0], 1))

        # Subplot 3: Attention for token 0
        if True:
            plt.subplot(2, 2, 3)
            plt.imshow(tiled_attention_with_nan, cmap='viridis', aspect='auto')  # Visualizing the first batch
            plt.colorbar()
            plt.title('Attention Weights for Token '+str(token))
            plt.xlabel('Time Step')
            plt.ylabel('Feature Dimension')
        else:
            plt.subplot(2, 2, 3)
            masked_tokens_np = masked_tokens.detach().cpu().numpy()
            plt.imshow(masked_tokens_np[batch_index].reshape(-1, 1).T, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('Feature Dimension')
            plt.xlabel('Time step')
            plt.ylabel('Mask')

        # Subplot 4: Attention Matrix
        #remove_missing_tokens = True
        #if remove_missing_tokens:
        #    mask = avg_attention_weights_with_nan[0]*avg_attention_weights_with_nan[0].T
        #    mask = np.where(mask == np.nan, np.nan, 1)
        #    #mask = mask.T
        plt.subplot(2, 2, 4)
        #avg_attention_weights[batch_index][non_nan_columns:, :] = np.nan
        avg_attention_weights[batch_index][:, np.isnan(input_data_with_nan[batch_index]).any(axis=0)] = np.nan
        plt.imshow(avg_attention_weights[batch_index], cmap='viridis', aspect='auto')  # Visualizing the averaged attention weights of the first batch
        plt.colorbar()
        plt.title('Attention Matrix (Averaged over Heads)')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')

        # Overlay grid lines
        num_rows, num_cols = avg_attention_weights[batch_index].shape
        plt.xticks(np.arange(-0.5, num_cols, 1), [])
        plt.yticks(np.arange(-0.5, num_rows, 1), [])
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        plt.xlim(-0.5, num_cols - 0.5)
        plt.ylim(num_rows - 0.5, -0.5)

        import io
        import PIL
        from torchvision.transforms import ToTensor
        plt.tight_layout()
        #plt.show()
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Convert the buffer to a tensor
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        #image = image.permute(0, 2, 3, 1)  # Change from (1, 4, 1200, 2000) to (1, 1200, 2000, 4)
        # Add the image to TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_image('Attention Weights', image, self.iter_count)
            #self.iter_count += 1

