import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer, FullAttentionCompletelyRemoveMissing
from .layers.Embed import DataEmbedding_inverted
import numpy as np
import matplotlib.pyplot as plt
import random

class PeriodicityReshape(nn.Module):
    def __init__(self, main_cycle):
        super(PeriodicityReshape, self).__init__()
        if main_cycle < 1:
            raise ValueError(f'Invalid main_cycle: {main_cycle}. Must be >= 1.')
        self.main_cycle = main_cycle
        
    def __assert_seq_len(self, x):
        _, n_steps, _ = x.shape
        seq_too_long = (n_steps % self.main_cycle) # is > 0 if n_steps is not a multiple of main_cycle -> True
        if seq_too_long:
            raise ValueError(f'''Number of steps {n_steps} is not a multiple of the main cycle ({n_steps}%{self.main_cycle}={n_steps%self.main_cycle}).
                             Suggested: Fill the sequence with zeros at the end to make it a multiple of the main cycle.''') 
            
    def apply(self, x, batch_size, n_features):
        self.__assert_seq_len(x)
        x = x.reshape(batch_size, -1, self.main_cycle, n_features).permute(0, 3, 1, 2)
        x = x.reshape(batch_size, -1, self.main_cycle).permute(0, 2, 1)
        return x

    def revert(self,x, batch_size, n_features):
        x = x.permute(0, 2, 1).reshape(batch_size, n_features, -1, self.main_cycle)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, n_features)
        return x

    def forward(self, x, n_features, direction):
        batch_size = x.shape[0]
        if direction == 'apply':
            return self.apply(x, batch_size, n_features)
        elif direction == 'revert':
            return self.revert(x, batch_size, n_features)
        else:
            raise ValueError(f'Invalid direction: {direction}. Use "apply" or "revert".')

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
        Creates a boolean attention mask that masks out tokens where all values are -9,
        expanded to match the dimensions expected by the attention scores.
        
        Args:
            x_enc (torch.Tensor): Input tensor of shape (B, L, N), where B is the batch size,
                                  L is the sequence length, and N is the number of tokens (features).
            num_heads (int): Number of attention heads.
            device (str): Device on which to create the mask (e.g., "cpu" or "cuda").
        """
        # Check if the first value in each token is -9
        batch_size, seq_len, num_tokens = x_enc.shape
        #mask = (x_enc[:, :, 0] == -9).unsqueeze(1).to(device)  # Shape: (B, 1, N)
        mask = torch.all(x_enc == -9, dim=2).unsqueeze(1).to(device)  # Shape: (B, 1, L)
        if completly_remove_missing:
            mask = mask | mask.transpose(1, 2)  # Logical OR to combine masks, Shape: (B, L, L)
        # Expand the mask to match the expected dimensions [B, H, L, L]
        self._mask = mask.unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len)  # Shape: (B, H, L, L)

    @property
    def mask(self):
        return self._mask
    
class ForecastHead(nn.Module):
    def __init__(self, d_model, pred_len):
        super(ForecastHead, self).__init__()
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, enc_out, token_index):
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, token_index]
        return dec_out    
    
class ReconstructionHead(nn.Module):
    def __init__(self, d_model, pred_len):
        super(ReconstructionHead, self).__init__()
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, enc_out, N):
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        return dec_out
    
class AlarmHead(nn.Module):
    def __init__(self, d_model, token_index):
        super(AlarmHead, self).__init__()
        self.token_index = token_index
        self.projector = nn.Linear(d_model, 1, bias=True)

    def forward(self, enc_out, N):
        dec_out = self.projector(enc_out[:, self.token_index])
        return dec_out
    
class ReconstructionHeadFullMLP(nn.Module):
    def __init__(self, d_model, pred_len, main_cycle, n_features, seq_len):
        super(ReconstructionHeadFullMLP, self).__init__()
        self.projector = nn.Linear(d_model, pred_len, bias=True)
        self.periodicity_reshape = PeriodicityReshape(main_cycle)
        self.n_features = n_features
        self.seq_len = seq_len
        self.full_mlp_layer = nn.Linear(seq_len, seq_len, bias=True)

    def forward(self, enc_out, N):
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = self.periodicity_reshape(dec_out, self.n_features, 'revert')
        dec_out = dec_out.transpose(-2, -1)
        dec_out = self.full_mlp_layer(dec_out)
        dec_out = dec_out.transpose(-2, -1)
        dec_out = self.periodicity_reshape(dec_out, self.n_features, 'apply')

        return dec_out

class EncoderModel(nn.Module):
    def __init__(self, configs):
        super(EncoderModel, self).__init__()
        self.seq_len = configs.patch_size
        self.d_model = configs.d_model
        self.n_features = configs.n_features
        self.embed = DataEmbedding_inverted(self.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.positional_encoding = PositionalEncoding(configs.d_model)
        #self.periodicity_reshape = PeriodicityReshape(self.main_cycle)
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
        # Define tokens using ParameterDict for more compact code
        self.categorical_tokens = nn.ParameterDict({
            # diagnosis_type
            "type1": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "type2": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "prediabetes": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "normal": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "unknown_diagnosis_type": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            # biological_sex
            "male": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "female": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "other": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "unknown_biological_sex": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            # insulin treatment
            "open_loop": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "closed_loop": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "hybrid_closed_loop": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "no_insulin": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "unknown_insulin_treatment": nn.Parameter(torch.randn(1, 1, configs.d_model)), 
            # unknown numerical values
            "unknown_age": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "unknown_bmi": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "forecast_token": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "hypoglycemia_token": nn.Parameter(torch.randn(1, 1, configs.d_model)),
            "hyperglycemia_token": nn.Parameter(torch.randn(1, 1, configs.d_model)),
        })

        self.numerical_embeddings = nn.ModuleDict({
            "age": nn.Linear(1, configs.d_model),
            "bmi": nn.Linear(1, configs.d_model)
        })

    def embed_metadata(self, metadata, batch_size):
        # Define metadata types and their corresponding unknown token names
        metadata_types = ["diagnosis_type", "biological_sex", "insulin_treatment"]
        numerical_types = ["age", "bmi"]
        # Check if metadata is provided
        if metadata is None:
            # If no metadata, use all unknown tokens expanded to batch size
            all_tokens = [self.categorical_tokens[f"unknown_{mtype}"].expand(batch_size, 1, self.d_model) 
                        for mtype in metadata_types]
            # Add zeros for numerical values
            # Add unknown tokens for numerical values
            all_tokens.extend([self.categorical_tokens[f"unknown_{ntype}"].expand(batch_size, 1, self.d_model)
                            for ntype in numerical_types])
            return torch.cat(all_tokens, dim=1)
        
        all_tokens = []
        
        # Process each metadata type
        for mtype in metadata_types:
            if mtype in metadata:
                type_tokens = torch.cat([self.categorical_tokens.get(value, self.categorical_tokens[f"unknown_{mtype}"])
                    for value in metadata[mtype]], dim=0)
            else:
                type_tokens = self.categorical_tokens[f"unknown_{mtype}"].expand(batch_size, 1, self.d_model)
            
            all_tokens.append(type_tokens)

        # Process each numerical metadata type
        for ntype in numerical_types:
            if ntype in metadata:
                # Get numerical values and convert to tensor
                values = metadata[ntype].view(batch_size, 1)
                
                # Create a mask for -1 values (treat as unknown)
                is_unknown = (values == -1)
                
                # For non-unknown values, apply linear embedding
                embedded_values = self.numerical_embeddings[ntype](values)
                embedded_values = embedded_values.view(batch_size, 1, self.d_model)
                
                # Replace embeddings for -1 values with unknown token
                unknown_token = self.categorical_tokens[f"unknown_{ntype}"].expand(batch_size, 1, self.d_model)
                embedded_values = torch.where(
                    is_unknown.unsqueeze(-1).expand(-1, -1, self.d_model),
                    unknown_token,
                    embedded_values
                )
                
                all_tokens.append(embedded_values)
            else:
                # Use unknown token if missing
                all_tokens.append(self.categorical_tokens[f"unknown_{ntype}"].expand(batch_size, 1, self.d_model))
        
        # Concatenate all token types together
        return torch.cat(all_tokens, dim=1)


    def forward(self, x_enc, x_mark_enc, attn_mask, apply_mask_tokens_fn=None, metadata=None):
        enc_out = self.embed(x_enc, x_mark_enc)
        if apply_mask_tokens_fn is not None:
            enc_out = apply_mask_tokens_fn(x_enc, enc_out)

        #enc_out = self.positional_encoding(enc_out)
        enc_out_parts = torch.chunk(enc_out, self.n_features, dim=1)  # Split along the second dimension (sequence length)

        # Apply positional encoding to each part
        encoded_parts = [self.positional_encoding(part) for part in enc_out_parts]

        # Combine the parts back together
        enc_out = torch.cat(encoded_parts, dim=1)

        # Embed metadata
        embedded_metadata = self.embed_metadata(metadata, batch_size=x_enc.shape[0])
        if embedded_metadata is not None:
            # Append metadata as an additional time step
            enc_out = torch.cat((enc_out, embedded_metadata), dim=1)
            # Add hypo/hyperglycemia tokens
            hypo_token = self.categorical_tokens["hypoglycemia_token"].expand(enc_out.shape[0], 1, self.d_model)
            hyper_token = self.categorical_tokens["hyperglycemia_token"].expand(enc_out.shape[0], 1, self.d_model)
            forecast_token = self.categorical_tokens["forecast_token"].expand(enc_out.shape[0], 1, self.d_model)
            enc_out = torch.cat((enc_out, forecast_token, hypo_token, hyper_token), dim=1)

            # Extend the attention mask to match the new enc_out shape
            num_metadata_tokens = embedded_metadata.shape[1] + 3  # +2 for the new tokens
            attn_mask._mask = F.pad(attn_mask.mask, (0, num_metadata_tokens, 0, num_metadata_tokens), value=0)
        enc_out, attns = self.encoder(enc_out, attn_mask=attn_mask)
        return enc_out, attns

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.patch_size
        self.pred_len = configs.patch_size
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.reconstruction = configs.reconstruction
        self.mask_ratio = 0.5 # NB. Not used
        self.missing_token = nn.Parameter(torch.randn(1, 1, configs.d_model))
        self.mask_token = nn.Parameter(torch.randn(1, 1, configs.d_model))
        #self.missing_token = torch.ones(1, 1, configs.d_model)  # Initialize with ones
        #self.mask_token = torch.zeros(1, 1, configs.d_model)    # Initialize with zeros
        self.encoder_model = EncoderModel(configs)
        self.reconstruction_projector = ReconstructionHead(configs.d_model, self.pred_len)
        self.variance_projector = ForecastHead(configs.d_model, configs.forecast_steps)
        self.mean_projector = ForecastHead(configs.d_model, configs.forecast_steps)
        self.forecast_projector = ForecastHead(configs.d_model, configs.forecast_steps)

        self.hyperglycemia_projector = AlarmHead(configs.d_model, -1)
        self.hypoglycemia_projector = AlarmHead(configs.d_model, -2)

        #self.projector_model = ReconstructionHeadFullMLP(configs.d_model, self.pred_len, configs.patch_size, configs.n_features, configs.seq_len)
        #self.projector_model_uncertainty = ReconstructionHeadFullMLP(configs.d_model, self.pred_len, configs.patch_size, configs.n_features, configs.seq_len)
        self.tb_writer = None
        self.iter_count = 0

    # Should set the embedded token to self.mask_token if the first value of the input is -9
    def apply_mask_tokens(self, input, embedded_tokens):
        """
        Replaces the embedded tokens with self.mask_token where:
        - first value is -9 (missing values)
        - first value is -6 (tokens to be masked)

        Args:
            input (torch.Tensor): The original input tensor x_enc of shape (Batch, Time, Variate).
            embedded_tokens (torch.Tensor): The embedded tokens of shape (Batch, Variate, d_model).

        Returns:
            torch.Tensor: The masked embedded tokens of shape (Batch, Variate, d_model).
        """
        device = embedded_tokens.device
        self.missing_token = self.missing_token#.to(device)
        self.mask_token = self.mask_token#.to(device)

        # Create mask for tokens where first value is -6
        random_token_mask = (input[:, 0, :] == -6)  # Shape: (Batch, Variate)
        random_token_mask = random_token_mask.unsqueeze(-1)  # Shape: (Batch, Variate, 1)
        random_token_mask = random_token_mask.expand(-1, -1, embedded_tokens.size(-1))  # Shape: (Batch, Variate, d_model)

        # Create a mask where the first value is -9 for each Variate
        #mask = (input[:, 0, :] == -9)  # Shape: (Batch, Variate)
        # create a mask if all of the the values in the token is -9
        mask = torch.all(input == -9, dim=1)
        mask = mask.unsqueeze(-1)      # Shape: (Batch, Variate, 1)
        mask = mask.expand(-1, -1, embedded_tokens.size(-1))  # Shape: (Batch, Variate, d_model)

        # Combine the masks
        #combined_mask = random_token_mask & ~mask
        #masked_tokens = combined_mask[:,:,1]

        # Replace embedded tokens with mask_token where random_token_mask is True
        embedded_tokens = torch.where(random_token_mask, self.mask_token, embedded_tokens)

        # Replace embedded tokens with missing_token where mask is True
        embedded_tokens = torch.where(mask, self.missing_token, embedded_tokens)

        return embedded_tokens

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_variance=False, metadata=None):
        device = x_enc.device
        attn_mask = NegativeNineMask(x_enc.transpose(1, 2), num_heads=8, device=x_enc.device, completly_remove_missing=False)
        #attn_mask = None
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out, attns = self.encoder_model(x_enc, x_mark_enc, attn_mask=attn_mask, apply_mask_tokens_fn=self.apply_mask_tokens, metadata=metadata)
        with torch.no_grad():
            detached_enc_out = enc_out.detach()
        #enc_out, masked_tokens = self.apply_mask_tokens(x_enc, enc_out, apply_random_tokens=False)
        #masked_tokens = None
        if self.reconstruction:
            dec_out_reconstruction = self.reconstruction_projector(enc_out, N)
        else:
            dec_out_reconstruction = self.reconstruction_projector(detached_enc_out, N)
        dec_out_forecast = self.forecast_projector(enc_out, -3)
        dec_out_mean = self.mean_projector(detached_enc_out, -3)
        dec_out_variance = self.variance_projector(detached_enc_out, -3)
        dec_out_hyperglycemia = self.hyperglycemia_projector(enc_out, N)
        dec_out_hypoglycemia = self.hypoglycemia_projector(enc_out, N)

        if self.use_norm:
            dec_out_reconstruction = dec_out_reconstruction * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out_reconstruction = dec_out_reconstruction + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out_forecast = dec_out_forecast * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out_forecast = dec_out_forecast + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out_mean = dec_out_mean * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out_mean = dec_out_mean + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out_variance = dec_out_variance * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out_variance = dec_out_variance + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #if random.randint(1, 100) == 1:
        if self.iter_count % 5000 == 0:
            self.plot_attention(x_enc, enc_out, attns, dec_out_reconstruction)
        return dec_out_reconstruction, dec_out_forecast, dec_out_mean, dec_out_variance, dec_out_hyperglycemia, dec_out_hypoglycemia

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,return_variance=False, metadata=None):
        self.iter_count += 1
        #dec_out, dec_out_uncertainty = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_variance=True, metadata=metadata)
        #return dec_out[:, -self.pred_len:, :], dec_out_uncertainty
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_variance=True, metadata=metadata) 

    def freeze_encoder(self):
        for param in self.encoder_model.parameters():
            param.requires_grad = False

    def plot_attention(self, x_enc, enc_out, attns, dec_out):
        found = False

        if not found:
            print("No masked tokens found")
            batch_index = 0
            token = 95
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


        input_data_with_nan = np.where(input_data == -9, np.nan, input_data)
        input_data_cleaned= np.where(input_data_with_nan == -6, np.nan, input_data)
        # Now plot all in subplots
        plt.figure(figsize=(20, 12))
        # Determine the common color scale range
        #vmin = np.nanmin(input_data_with_nan)
        #vmax = np.nanmax(input_data_with_nan)
        dec_out_np[batch_index][:, np.isnan(input_data_with_nan[batch_index]).all(axis=0)] = np.nan
        vmin = min(
            np.nanquantile(dec_out_np[batch_index], 0.05),
            np.nanquantile(input_data_cleaned[batch_index], 0.05)
        )
        vmax = max(
            np.nanquantile(dec_out_np[batch_index], 0.95),
            np.nanquantile(input_data_cleaned[batch_index], 0.95)
        )
        # Subplot 1: Input Data
        plt.subplot(2, 2, 1)
        plt.imshow(input_data_cleaned[batch_index], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)  # Visualizing the first batch
        plt.colorbar()
        plt.title('Input Data')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Dimension')

        # Count the number of non-NaN columns
        non_nan_columns = np.sum(~np.isnan(input_data_with_nan[batch_index]).all(axis=0))


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
        # Get the original sequence length without metadata tokens
        orig_seq_len = input_data_with_nan[batch_index].shape[1]
        
        # Update this line to only use the first orig_seq_len columns
        plt.subplot(2, 2, 4)
        
        # Only use the part of attention weights that corresponds to the original sequence
        # This handles the case where metadata tokens were added
        attention_visual = avg_attention_weights[batch_index][:orig_seq_len, :orig_seq_len].copy()
        
        # Now apply the mask only to the dimensions we're visualizing
        missing_mask = np.isnan(input_data_with_nan[batch_index]).any(axis=0)
        attention_visual[:, missing_mask] = np.nan
        
        plt.imshow(attention_visual, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Attention Matrix (Averaged over Heads)')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')

        # Overlay grid lines
        num_rows, num_cols = attention_visual.shape
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

