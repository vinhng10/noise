import numbers
from os import PathLike
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import onnxruntime as ort
import numpy as np
import pandas as pd

from typing import Callable, Tuple
from torch.nn import *
from torch import Tensor
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from torch.nn.modules.normalization import _shape_t
from lightning.pytorch.utilities import grad_norm


###############################################################################
# Layer Definition
class MonteCarloDropout(nn.Dropout):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, True, self.inplace)


class PositionEmbedding(nn.Module):
    def __init__(
        self, max_length: int, embedding_size: int, dropout: float, inplace: bool
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout, inplace)
        self.position_embedding = nn.Embedding(max_length, embedding_size)

    def forward(self, input: Tensor) -> Tensor:
        position_idx = torch.arange(input.shape[0], device=input.device).unsqueeze(1)
        position_embeddings = self.position_embedding(position_idx)
        return self.dropout(input + position_embeddings)


class ReeoLayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        bias: bool = True,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            self.bias = (
                Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
                if bias
                else None
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


###############################################################################
# Transformation Definition
class FunctionalModule(nn.Module):
    def __init__(self, functional: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.functional = functional

    def forward(self, inputs: Tensor) -> Tensor:
        return self.functional(inputs)


class GlobalStatsNormalization(nn.Module):
    def __init__(self, global_stats_path: str | PathLike) -> None:
        super().__init__()
        scaler = np.load(global_stats_path, allow_pickle=True).item()
        self.register_buffer("mean", torch.FloatTensor(scaler.mean_).reshape(1, 1, -1))
        self.register_buffer(
            "inverse_std",
            1 / (torch.FloatTensor(scaler.scale_).reshape(1, 1, -1) + 1e-8),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return (inputs - self.mean) * self.inverse_std


class InverseGlobalStatsNormalization(nn.Module):
    def __init__(self, global_stats_path: str | PathLike) -> None:
        super().__init__()
        scaler = np.load(global_stats_path, allow_pickle=True).item()
        self.register_buffer("mean", torch.FloatTensor(scaler.mean_).reshape(1, 1, -1))
        self.register_buffer("std", torch.FloatTensor(scaler.scale_).reshape(1, 1, -1))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.std + self.mean


###############################################################################
# Model Definition
class Model(pl.LightningModule):
    SUPPORTED_SAMPLING_RATE = 16000

    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        p835_model_path: str,
        p808_model_path: str,
        global_stats_path: str,
    ) -> None:
        super().__init__()

    def _replace(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    nn.Linear(child.in_features, child.out_features, self.hparams.bias),
                )
            elif isinstance(child, nn.MultiheadAttention):
                setattr(
                    module,
                    name,
                    nn.MultiheadAttention(
                        self.hparams.hidden_size,
                        self.hparams.num_heads,
                        dropout=self.hparams.dropout,
                        batch_first=self.hparams.batch_first,
                        bias=self.hparams.bias,
                    ),
                )
            elif isinstance(child, nn.LayerNorm):
                setattr(
                    module,
                    name,
                    ReeoLayerNorm(child.normalized_shape, bias=self.hparams.bias),
                )
            elif isinstance(child, nn.Dropout):
                setattr(module, name, nn.Dropout(child.p, inplace=self.hparams.inplace))
            elif list(child.named_children()):
                self._replace(child)

    def training_step(self, batch, batch_idx):
        noisy_waveforms, clean_waveforms = batch
        clean_spectrograms, _ = self.transform(clean_waveforms)
        enhanced_spectrograms, _ = self.filter(noisy_waveforms)
        loss = F.mse_loss(enhanced_spectrograms, clean_spectrograms)
        self.log(
            f"train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        mos = MOS(self.hparams.p835_model_path, self.hparams.p808_model_path)
        noisy_waveforms, _ = batch
        enhanced_waveforms = self.forward(noisy_waveforms)
        scores = []
        for waveform in enhanced_waveforms.cpu().numpy():
            scores.append(mos(waveform, Model.SUPPORTED_SAMPLING_RATE))
        scores = pd.DataFrame(scores).mean()
        self.log_dict(
            {
                "val_mos_sig": scores["SIG"],
                "val_mos_bak": scores["BAK"],
                "val_mos_ovr": scores["OVRL"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_before_zero_grad(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=False, on_epoch=True, prog_bar=False, logger=True)


class NSNET2(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.stft = Spectrogram(
            n_fft=self.hparams.n_fft,
            win_length=self.hparams.win_length,
            hop_length=self.hparams.hop_length,
            power=None,
        )
        self.scaled_log_spectrogram = nn.Sequential(
            FunctionalModule(lambda x: (x.abs().pow(2) + 1e-8).log10()),
            FunctionalModule(lambda x: x.permute(2, 0, 1)),
            GlobalStatsNormalization(self.hparams.global_stats_path),
        )
        self.inverse_stft = InverseSpectrogram(
            n_fft=self.hparams.n_fft,
            win_length=self.hparams.win_length,
            hop_length=self.hparams.hop_length,
        )
        self.inverse_scaled_log_spectrogram = nn.Sequential(
            InverseGlobalStatsNormalization(self.hparams.global_stats_path),
            FunctionalModule(lambda x: x.permute(1, 2, 0)),
            FunctionalModule(lambda x: (10**x).sqrt()),
        )
        self.ff1 = nn.Linear(257, 512)
        self.gru = nn.GRU(512, 512, 1)
        self.ff2 = nn.Linear(512, 257)

    def forward(self, waveforms: Tensor) -> Tensor:
        spectrograms, angles = self.filter(waveforms)
        waveforms = self.inverse_transform(spectrograms, angles)
        return waveforms

    def filter(self, waveforms: Tensor) -> tuple[Tensor, Tensor]:
        spectrograms, angles = self.transform(waveforms)
        filters = F.relu(self.ff1(spectrograms))
        filters, _ = self.gru(filters)
        filters = self.ff2(filters)
        spectrograms = spectrograms * filters
        return spectrograms, angles

    def transform(self, waveforms: Tensor) -> tuple[Tensor, Tensor]:
        stfts = self.stft(waveforms)
        angles = stfts.angle()
        scaled_log_spectrograms = self.scaled_log_spectrogram(stfts)
        return scaled_log_spectrograms, angles

    def inverse_transform(
        self, scaled_log_spectrograms: Tensor, angles: Tensor
    ) -> Tensor:
        stfts = self.inverse_scaled_log_spectrogram(scaled_log_spectrograms)
        waveforms = self.inverse_stft(stfts * (angles * 1j).exp())
        return waveforms


###############################################################################
class MOS:
    INPUT_LENGTH = 9.01

    def __init__(self, p835_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(p835_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr):
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio, sampling_rate):
        actual_audio_len = len(audio)
        len_samples = int(MOS.INPUT_LENGTH * sampling_rate)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / sampling_rate) - MOS.INPUT_LENGTH) + 1
        hop_len_samples = sampling_rate
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int(
                    (idx + MOS.INPUT_LENGTH) * hop_len_samples
                )
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(
                self.audio_melspec(audio=audio_seg[:-160])
            ).astype("float32")[np.newaxis, :, :]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {
            "len_in_sec": actual_audio_len / sampling_rate,
            "sr": sampling_rate,
            "num_hops": num_hops,
            "OVRL_raw": np.mean(predicted_mos_ovr_seg_raw),
            "SIG_raw": np.mean(predicted_mos_sig_seg_raw),
            "BAK_raw": np.mean(predicted_mos_bak_seg_raw),
            "OVRL": np.mean(predicted_mos_ovr_seg),
            "SIG": np.mean(predicted_mos_sig_seg),
            "BAK": np.mean(predicted_mos_bak_seg),
            "P808_MOS": np.mean(predicted_p808_mos),
        }

        return clip_dict


# class ProbVerticalStackReeoModule(Model):
#     def __init__(
#         self,
#         sequence_length: int,
#         audio_feature_size: int,
#         motion_feature_size: int,
#         hidden_size: int,
#         num_heads: int,
#         feedforward_size: int,
#         dropout: float,
#         inplace: bool,
#         activation: str,
#         num_layers: int,
#         batch_first: bool,
#         norm_first: bool,
#         bias: bool,
#         step: int,
#         noise_prob: float,
#         noise_scale: float,
#         zero_initial_prob: float,
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()
#         self.pos_embedding = PositionEmbedding(
#             sequence_length, hidden_size, 0.1, inplace
#         )
#         self.linear_emdedding = nn.Linear(
#             audio_feature_size + motion_feature_size, hidden_size, bias
#         )
#         layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size,
#             nhead=num_heads,
#             dim_feedforward=feedforward_size,
#             dropout=dropout,
#             activation=activation,
#             batch_first=batch_first,
#             norm_first=norm_first,
#         )
#         norm = ReeoLayerNorm(hidden_size, bias=bias)
#         self.transformer = nn.TransformerEncoder(
#             layer, num_layers=num_layers, norm=norm
#         )
#         self._replace(self.transformer)
#         self.loc = nn.Linear(hidden_size, motion_feature_size * step, bias)
#         self.log_scale = nn.Linear(hidden_size, motion_feature_size * step, bias)
#         self.register_buffer(
#             "target_mask",
#             torch.triu(
#                 torch.full((sequence_length, sequence_length), float("-inf")),
#                 diagonal=1,
#             ),
#         )

#     def forward(
#         self,
#         motions: Tensor,
#         audios: Tensor,
#         src_pad_masks: Tensor = None,
#     ) -> Tuple[Tensor, Tensor]:
#         scale = math.sqrt(self.hparams.hidden_size)
#         inputs = torch.cat((motions, audios), dim=-1)
#         inputs = self.linear_emdedding(inputs) * scale
#         inputs = self.pos_embedding(inputs)
#         outputs = self.transformer(
#             inputs,
#             mask=self.target_mask,
#             src_key_padding_mask=src_pad_masks,
#             is_causal=True,
#         )
#         locs = self.loc(outputs)
#         log_scales = self.log_scale(outputs)
#         return locs.real, log_scales.real

#     def _shared_step(self, batch, phase: str) -> Dict[str, Tensor]:
#         motions, audios = batch
#         step = self.hparams.step
#         if torch.rand(1).item() < self.hparams.zero_initial_prob:
#             motions = motions[step:]
#             audios = audios[step:]
#             targets = _torch_stack_targets(motions, step)
#             if torch.rand(1).item() < self.hparams.noise_prob:
#                 motions += (
#                     torch.randn_like(
#                         motions, dtype=motions.dtype, device=motions.device
#                     )
#                     * self.hparams.noise_scale
#                 )
#                 motions = F.pad(motions[:-step], (0, 0, 0, 0, step, 0))
#         else:
#             targets = _torch_stack_targets_V2(motions, step)
#             motions = motions[:-step]
#             audios = audios[step:]
#             if torch.rand(1).item() < self.hparams.noise_prob:
#                 motions += (
#                     torch.randn_like(
#                         motions, dtype=motions.dtype, device=motions.device
#                     )
#                     * self.hparams.noise_scale
#                 )
#         locs, log_scales = self.forward(motions, audios)
#         loss = -Normal(locs, log_scales.exp()).log_prob(targets).mean()
#         self.log(
#             f"{phase}_loss",
#             loss,
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#         )
#         return loss

#     def _generate(
#         self,
#         motions: Tensor,
#         audios: Tensor,
#         loc_coef: float = 1.1,
#         scale_coef: float = 1.0,
#         mode: str = "inference",
#     ) -> Tensor:
#         step = self.hparams.step
#         feat_size = self.hparams.motion_feature_size
#         for i in range(step, len(audios) + step, step):
#             locs, log_scales = self(motions[:-step], audios)
#             dists = Normal(locs[i - 1], log_scales[i - 1].exp() * scale_coef)
#             if mode == "inference":
#                 predictions = torch.cat(
#                     (
#                         locs[i - 1].reshape(step, -1, feat_size)[:, :, :3] / loc_coef,
#                         dists.sample().reshape(step, -1, feat_size)[:, :, 3:],
#                     ),
#                     dim=-1,
#                 )
#             else:
#                 predictions = dists.sample().reshape(step, -1, feat_size)
#             motions[i : i + step] = predictions
#         return motions

#     def generate(
#         self,
#         audios: Tensor,
#         loc_coef: float = 1.1,
#         scale_coef: float = 1.0,
#         disable_bar: bool = False,
#     ) -> Tensor:
#         step = self.hparams.step
#         sequence_length = self.hparams.sequence_length
#         feat_size = self.hparams.motion_feature_size

#         pad = sequence_length - (len(audios) % sequence_length)

#         audios = F.pad(audios, (0, 0, 0, 0, 0, pad))

#         motions = torch.zeros(
#             (step + len(audios), audios.shape[1], feat_size),
#             device=audios.device,
#             dtype=audios.dtype,
#         )

#         for i in tqdm(
#             range(sequence_length, len(audios) + 1, sequence_length),
#             disable=disable_bar,
#         ):
#             motions[i - sequence_length : i + step] = self._generate(
#                 motions[i - sequence_length : i + step],
#                 audios[i - sequence_length : i],
#                 loc_coef,
#                 scale_coef,
#                 "inference",
#             )
#         return motions[step:-pad]

#     def log_prob(self, motions: Tensor, audios: Tensor) -> Tensor:
#         """
#         Mathematically, the log joint probability of the whole motion sequence
#         should be the sum of log probability of each features and time step
#         (probability chain rule). However, the sum easily becomes very large
#         and make the exponential infinite.
#         To avoid that, the mean is computed as a scaled down of the log joint
#         probability of the motion sequence.
#         """
#         locs, log_scales = self.forward(
#             F.pad(motions[: -self.hparams.step], (0, 0, 0, 0, self.hparams.step, 0)),
#             audios,
#         )
#         # Take only the immediate next motion to compute log probability:
#         log_probs = (
#             Normal(locs, log_scales.exp())
#             .log_prob(
#                 _torch_stack_targets(
#                     torch.cat((motions, audios), dim=-1), self.hparams.step
#                 )
#             )
#             .mean(dim=(0, 2))
#         )
#         return log_probs
