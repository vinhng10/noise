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
        self.stft = Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
        )
        self.scaled_log_spectrogram = nn.Sequential(
            FunctionalModule(lambda x: (x.abs().pow(2) + 1e-8).log10()),
            FunctionalModule(lambda x: x.permute(2, 0, 1)),
            GlobalStatsNormalization(global_stats_path),
        )
        self.inverse_stft = InverseSpectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length
        )
        self.inverse_scaled_log_spectrogram = nn.Sequential(
            InverseGlobalStatsNormalization(global_stats_path),
            FunctionalModule(lambda x: x.permute(1, 2, 0)),
            FunctionalModule(lambda x: (10**x).sqrt()),
        )

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


class NSNET2(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
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


class ConvNet(Model):
    def __init__(self, base_n_filters: int, bias: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Conv2d(1, base_n_filters, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(base_n_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_n_filters, base_n_filters * 2, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(base_n_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_n_filters * 2, base_n_filters * 4, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(base_n_filters * 4),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_n_filters * 4, base_n_filters * 2, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(base_n_filters * 2),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_n_filters * 2, base_n_filters, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(base_n_filters),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(base_n_filters, 1, 3, 1, 1, bias=bias)

    def forward(self, waveforms: Tensor) -> Tensor:
        spectrograms, angles = self.filter(waveforms)
        waveforms = self.inverse_transform(spectrograms, angles)
        return waveforms

    def filter(self, waveforms: Tensor) -> tuple[Tensor, Tensor]:
        spectrograms, angles = self.transform(waveforms)
        n_samples, _, n_frequencies = spectrograms.shape
        filters = self.net(spectrograms.permute(1, 0, 2).unsqueeze(1))
        filters = F.interpolate(
            filters,
            size=(n_samples, n_frequencies),
            mode="bilinear",
            align_corners=True,
        )
        filters = self.output(filters)
        spectrograms = spectrograms * filters.squeeze(1).permute(1, 0, 2)
        return spectrograms, angles


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
