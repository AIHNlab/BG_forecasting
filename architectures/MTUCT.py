"""MT-UCT: Multi-task Transformer with Unified Clinical Tokenizer.

This module implements the model introduced in:
    Strommen, Panagiotou, Brigato, Mougiakakou.
    "Multi-task Transformer with Unified Clinical Tokenizer for Effective
    Blood Glucose Prediction." IEEE J. Biomed. Health Inform.

Naming:
- ``Model``  : the full MT-UCT model (kept as ``Model`` for the trainer's
               ``model_dict[name].Model(args)`` instantiation pattern).
- ``UCT``    : the Unified Clinical Tokenizer + encoder backbone.
- ``TaskHead``: per-task projection head (forecast, alarm, reconstruction,
                uncertainty).

For backwards compatibility, this module is also exposed as
``architectures.BGiTransformer`` and re-exports ``EncoderModel`` as an
alias of ``UCT``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """MT-UCT model: UCT backbone + multiple task-specific heads."""

    def __init__(self, configs):
        super(Model, self).__init__()
        # ``encoder_model`` attribute name is preserved for backwards
        # compatibility with trainer code that accesses
        # ``self.model.encoder_model.parameters()``.
        self.encoder_model = UCT(configs)
        self.forecast_head = TaskHead(configs, configs.pred_len)
        self.imputation_head = TaskHead(configs, configs.seq_len)
        self.alarm_head = TaskHead(configs, configs.pred_len)
        self.uncertainty_forecast_head = TaskHead(configs, configs.pred_len, output_activation=nn.Softplus())
        self.uncertainty_imputation_head = TaskHead(configs, configs.seq_len, output_activation=nn.Softplus())
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            enc_out, means, stdev = self.encoder_model(x_enc, x_mark_enc)
            _, _, N = x_enc.shape
            forecast_output = self.forecast_head(enc_out, means, stdev, N)
            imputation_output = self.imputation_head(enc_out, means, stdev, N)
            alarm_output = self.alarm_head(enc_out, means, stdev, N)
            uncertainty_forecast_output = self.uncertainty_forecast_head(enc_out, means, stdev, N)
            uncertainty_imputation_output = self.uncertainty_imputation_head(enc_out, means, stdev, N)
            return forecast_output, imputation_output, alarm_output, uncertainty_forecast_output, uncertainty_imputation_output
        else:
            enc_out = self.encoder_model(x_enc, x_mark_enc)
            _, _, N = x_enc.shape
            forecast_output = self.forecast_head(enc_out, N)
            imputation_output = self.imputation_head(enc_out, N)
            alarm_output = self.alarm_head(enc_out, N)
            uncertainty_forecast_output = self.uncertainty_forecast_head(enc_out, N)
            uncertainty_imputation_output = self.uncertainty_imputation_head(enc_out, N)
            return forecast_output, imputation_output, alarm_output, uncertainty_forecast_output, uncertainty_imputation_output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        #return dec_out[:, -self.pred_len:, :] #Not sure I understand the point of this slicing...
        return dec_out

    def freeze_encoder(self):
        self.encoder_model.freeze_encoder()


class UCT(nn.Module):
    """Unified Clinical Tokenizer + Transformer encoder backbone.

    Embeds the multivariate clinical input (CGM, CHO, bolus insulin, plus
    demographic/clinical and task tokens) into a shared latent space and
    contextualises them through stacked self-attention encoder layers.
    """

    def __init__(self, configs):
        super(UCT, self).__init__()
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        if self.use_norm:
            return enc_out, means, stdev
        else:
            return enc_out

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False


# Backwards-compatible alias for the pre-rename class name.
EncoderModel = UCT


class TaskHead(nn.Module):
    def __init__(self, configs, output_len, output_activation='linear'):
        super(TaskHead, self).__init__()
        self.pred_len = output_len
        self.use_norm = configs.use_norm
        self.projector = nn.Linear(configs.d_model, output_len, bias=True)
        if output_activation == 'linear':
            self.output_activation = lambda x: x
        else:
            self.output_activation = output_activation

    def forward(self, enc_out, means=None, stdev=None, N=None):
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = self.output_activation(dec_out)
        return dec_out#[:,:,:1]
