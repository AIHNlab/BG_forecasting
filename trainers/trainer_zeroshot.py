import os
import sys
import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sentinel values used by the pipeline for missing/padded/masked data.
_SENTINELS = {0, -5, -6, -8, -9}


def _clean_bg_context(context_np):
    """Replace sentinel/impossible BG values with linear interpolation.

    The pipeline fills missing CGM readings with 0 and pads sequences
    with 0/-8.  Zero-shot models interpret these as real values, causing
    wild prediction spikes.  This function replaces any value in
    ``_SENTINELS`` (or negative) with linearly-interpolated neighbours.
    """
    cleaned = context_np.copy()
    for i in range(cleaned.shape[0]):
        row = cleaned[i]
        bad = np.isin(row, list(_SENTINELS)) | (row < 1)
        if not bad.any():
            continue
        good = ~bad
        if good.sum() < 2:
            # Not enough valid points — fill with the single valid value
            if good.any():
                cleaned[i] = row[good][0]
            continue
        # Linear interpolation over bad positions
        cleaned[i] = np.interp(
            np.arange(len(row)),
            np.where(good)[0],
            row[good],
        )
    return cleaned


class TrainerChronos:
    """Zero-shot baseline using Amazon Chronos-T5 (no training required).

    Chronos tokenises scalar time-series values and feeds them through a
    pre-trained T5 encoder–decoder.  We simply call ``predict()`` on the
    raw CBG channel extracted from the pipeline's SequenceDataset.
    """

    def __init__(self, hp_config, model_path, retrain_model=True):
        from chronos import ChronosPipeline

        self.hp_config = hp_config
        self.forecast_steps = hp_config["forecast_steps"]

        model_name = hp_config.get("chronos_model", "amazon/chronos-t5-base")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    def train(self, train_loader, val_loader=None):
        pass  # zero-shot — nothing to train

    # ------------------------------------------------------------------
    def test(self, test_loader, scaler=None):
        all_preds, all_targets = [], []
        num_samples = self.hp_config.get("chronos_num_samples", 20)

        for batch in test_loader:
            sequences, targets = batch[0], batch[1]
            # sequences: (B, context_len, n_features) — take CBG channel 0
            context = sequences[:, :, 0].float()  # (B, context_len)

            # Replace sentinel/impossible BG values with NaN.
            # Chronos natively skips NaN during tokenization, so this
            # prevents 0/-8/-9 from being interpreted as real readings.
            ctx_np = context.numpy()
            bad = np.isin(ctx_np, list(_SENTINELS)) | (ctx_np < 1)
            ctx_np[bad] = np.nan
            context = torch.from_numpy(ctx_np)

            with torch.no_grad():
                # Chronos predict: context (B, T), returns (B, num_samples, H)
                samples = self.pipeline.predict(
                    inputs=context,
                    prediction_length=self.forecast_steps,
                    num_samples=num_samples,
                    limit_prediction_length=False,
                )
                preds = samples.median(dim=1).values  # (B, H)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

        predictions = np.concatenate(all_preds, axis=0)  # (N, H)
        actuals = np.concatenate(all_targets, axis=0)      # (N, H)

        # Replace sentinel values in targets
        actuals = np.where(np.isin(actuals, [-8, -9]), np.nan, actuals)

        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)
            actuals = scaler.inverse_transform(actuals)

        # Pipeline expects (N, H, 1)
        predictions = np.expand_dims(predictions, axis=-1)
        actuals = np.expand_dims(actuals, axis=-1)
        return predictions, actuals


class TrainerTimeLLM:
    """Zero-shot baseline using TimeLLM with a frozen GPT-2 backbone.

    TimeLLM reprograms a pre-trained LLM (GPT-2) to perform time-series
    forecasting via prompt-augmented patch embeddings.  The model has its
    own instance normalisation layer so raw BG values are expected.
    """

    def __init__(self, hp_config, model_path, retrain_model=True):
        # Add diab-llm to path so we can import its model definition
        _diab_llm_root = os.path.join(_PROJECT_ROOT, "diab-llm")
        if _diab_llm_root not in sys.path:
            sys.path.insert(0, _diab_llm_root)

        from models.time_llm import Model as TimeLLMModel

        self.hp_config = hp_config
        self.forecast_steps = hp_config["forecast_steps"]
        context_len = hp_config["feature_window"] - hp_config["forecast_steps"]

        # Build the config dict that TimeLLM's Model.__init__ expects
        llm_config = {
            "task_name": "short_term_forecast",
            "prediction_length": self.forecast_steps,
            "sequence_length": context_len,
            "enc_in": 1,          # univariate (CBG only)
            "d_model": hp_config.get("d_model", 768),
            "d_ff": hp_config.get("d_ff", 768),
            "n_heads": hp_config.get("n_heads", 8),
            "llm_model": hp_config.get("llm_model", "GPT2"),
            "llm_dim": hp_config.get("llm_dim", 768),
            "llm_layers": hp_config.get("llm_layers", 6),
            "patch_len": hp_config.get("patch_len", 16),
            "stride": hp_config.get("stride", 8),
            "dropout": hp_config.get("dropout", 0.1),
            "prompt_domain": True,
            "content": (
                "Blood glucose concentration measured by a continuous glucose "
                "monitor (CGM) in mg/dL, sampled every 5 minutes."
            ),
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = TimeLLMModel(configs=llm_config).float().to(device)
        self.model.eval()

        # Freeze all parameters (zero-shot inference only)
        for param in self.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    def train(self, train_loader, val_loader=None):
        pass  # zero-shot — nothing to train

    # ------------------------------------------------------------------
    def test(self, test_loader, scaler=None):
        all_preds, all_targets = [], []

        for batch in test_loader:
            sequences, targets = batch[0], batch[1]
            # sequences: (B, context_len, n_features) — take CBG channel 0
            x_enc = sequences[:, :, 0:1].float()  # (B, T, 1)

            # Replace sentinel/impossible BG values with linear interpolation.
            # TimeLLM runs through a neural net that can't handle NaN, so we
            # interpolate over bad positions instead.
            ctx_np = x_enc[:, :, 0].numpy()
            ctx_np = _clean_bg_context(ctx_np)
            x_enc = torch.from_numpy(ctx_np).unsqueeze(-1).float().to(self.device)

            B, T, _ = x_enc.shape

            # TimeLLM's forecast() accepts these but never reads them
            x_mark_enc = torch.zeros(B, T, 5, device=self.device)
            x_dec = torch.zeros(B, T + self.forecast_steps, 1, device=self.device)
            x_mark_dec = torch.zeros(B, T + self.forecast_steps, 5, device=self.device)

            with torch.no_grad():
                out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                # out: (B, forecast_steps, 1)

            all_preds.append(out.cpu().numpy())
            all_targets.append(targets.numpy())

        predictions = np.concatenate(all_preds, axis=0)  # (N, H, 1)
        actuals = np.concatenate(all_targets, axis=0)      # (N, H)

        # Replace sentinel values in targets
        actuals = np.where(np.isin(actuals, [-8, -9]), np.nan, actuals)

        if scaler is not None:
            predictions = scaler.inverse_transform(predictions.squeeze(-1))
            actuals = scaler.inverse_transform(actuals)
            predictions = np.expand_dims(predictions, axis=-1)

        actuals = np.expand_dims(actuals, axis=-1)
        return predictions, actuals
