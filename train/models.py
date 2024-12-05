from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from librosa import util as librosa_util
from librosa.util import pad_center
from scipy.signal import get_window
from torch import Tensor
from torch.nn.common_types import _size_2_t
from torchvision.models._utils import _make_divisible
from torchvision.ops.misc import ConvNormActivation, Conv2dNormActivation


# Teacher:
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    sz_b, len_s = seq.size()
    subsequent_mask = (
        1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)
    ).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self,
        d_word_vec=512,
        n_layers=2,
        n_head=8,
        d_k=64,
        d_v=64,
        d_model=512,
        d_inner=2048,
        dropout=0.1,
        n_position=624,
        scale_emb=False,
    ):

        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        if n_position > 0:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        else:
            self.position_enc = lambda x: x
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        # enc_output = self.src_word_emb(src_seq)
        enc_output = src_seq
        if self.scale_emb:
            enc_output *= self.d_model**0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.module(x)


class DepthwiseSeparableConvTranspose(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias,
        is_output=False,
    ):
        super().__init__()
        self.module = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=in_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if not is_output else nn.Identity(),
        )

    def forward(self, x):
        return self.module(x)


class ConvTranspose2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            nn.ConvTranspose2d,
        )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        dw_layer: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if dw_layer is None:
            dw_layer = Conv2dNormActivation

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                )
            )
        layers.extend(
            [
                # dw
                dw_layer(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, max_frames, n_fft, hop_length, win_length, window):
        super().__init__()
        self.max_frames = max_frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.scale = n_fft / hop_length
        self.pad_amount = n_fft // 2
        self.cutoff = (self.n_fft // 2) + 1

        fourier_basis = np.fft.fft(np.eye(self.n_fft))
        fourier_basis = np.vstack(
            [
                np.real(fourier_basis[: self.cutoff, :]),
                np.imag(fourier_basis[: self.cutoff, :]),
            ]
        )
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :]).detach()
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(self.scale * fourier_basis).T[:, None, :]
        ).detach()
        if window is not None:
            assert n_fft >= win_length
            # get window and zero center pad it to n_fft
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=n_fft)
            fft_window = torch.from_numpy(fft_window).float()
            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window
        window_sum = self.window_sumsquare(
            self.window,
            self.max_frames,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            dtype=np.float32,
        )
        window_sum[window_sum == 0] = 1
        window_sum = self.scale / window_sum
        window_sum = torch.FloatTensor(window_sum).detach()

        self.register_buffer("forward_basis", forward_basis)
        self.register_buffer("inverse_basis", inverse_basis)
        self.register_buffer("window_sum", window_sum)

    def forward(self, waveform):
        waveform = F.pad(
            waveform.squeeze(1), (self.pad_amount, self.pad_amount), mode="reflect"
        )

        forward_transform = F.conv1d(
            waveform, self.forward_basis, stride=self.hop_length, padding=0
        )

        real = forward_transform[:, : self.cutoff, :]
        imag = forward_transform[:, self.cutoff :, :]

        log_magnitude = torch.sqrt(real**2 + imag**2).unsqueeze(1).clamp(min=1e-3).log()
        phase = torch.atan2(imag, real)

        return log_magnitude, phase

    def inverse(self, log_magnitude, phase):
        magnitude = log_magnitude.exp().squeeze(1)
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )
        if self.window is not None:
            inverse_transform *= self.window_sum[: inverse_transform.size(-1)]

        inverse_transform = inverse_transform[
            :, None, :, self.pad_amount : -self.pad_amount
        ]
        inverse_transform = inverse_transform.clamp(min=-1.0, max=1.0)
        return inverse_transform

    @staticmethod
    def window_sumsquare(
        window,
        n_frames,
        hop_length=200,
        win_length=800,
        n_fft=800,
        dtype=np.float32,
        norm=None,
    ):
        """
        # from librosa 0.6
        Compute the sum-square envelope of a window function at a given hop length.
        This is used to estimate modulation effects induced by windowing
        observations in short-time fourier transforms.
        Parameters
        ----------
        window : string, tuple, number, callable, or list-like
            Window specification, as in `get_window`
        n_frames : int > 0
            The number of analysis frames
        hop_length : int > 0
            The number of samples to advance between frames
        win_length : [optional]
            The length of the window function.  By default, this matches `n_fft`.
        n_fft : int > 0
            The length of each analysis frame.
        dtype : np.dtype
            The data type of the output
        Returns
        -------
        wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
            The sum-squared envelope of the window function
        """
        if win_length is None:
            win_length = n_fft

        n = n_fft + hop_length * (n_frames - 1)
        x = np.zeros(n, dtype=dtype)

        # Compute the squared window at the desired length
        win_sq = get_window(window, win_length, fftbins=True)
        win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
        win_sq = librosa_util.pad_center(win_sq, size=n_fft)

        # Fill the envelope
        for i in range(n_frames):
            sample = i * hop_length
            x[sample : min(n, sample + n_fft)] += win_sq[
                : max(0, min(n_fft, n - sample))
            ]
        return x


# Base Definition:
class Base(pl.LightningModule):
    SUPPORTED_SAMPLING_RATE = 16000

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.H_padding = 0
        self.W_padding = 0
        self.searching = False
        self.found = False

    def forward(self, waveforms):
        x, vad = self._forward(waveforms)
        # x *= (1e4 * vad[..., None, None]).sigmoid()
        return x

    def _forward(self, x):
        # Search padding if not already done
        if not self.searching and not self.found:
            self._search_padding(x)

        H, W = x.shape[-2:]
        std = x.std() + 1e-3
        x = x / std
        x = F.pad(x, (self.W_padding, 0, self.H_padding, 0), value=self.pad_value)

        # encoder
        downs = []
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            downs.append(x)
        downs = downs[::-1]

        # attention & vad
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.bottleneck_attention(x, None)
        vad = self.vad(x[:, 0, :])
        x = x.permute(0, 2, 1).view(b, c, h, w)

        # decoder
        for i, upsampling_block in enumerate(self.decoder):
            skip_i = downs[i]
            x = x + skip_i[:, :, -x.shape[-2] :, -x.shape[-1] :]
            x = upsampling_block(x)
        x = x[:, :, -H:, -W:] * std

        return x, vad

    @torch.no_grad()
    def _search_padding(self, x):
        self.searching = True
        H, W = x.shape[-2:]
        out = self.forward(x)
        while out.shape[-2] < H:
            self.H_padding += 1
            out = self.forward(x)
        while out.shape[-1] < W:
            self.W_padding += 1
            out = self.forward(x)
        self.found = True

    def _shared_step(self, batch, batch_idx, stage):
        noisy_waveforms, clean_waveforms, _ = batch
        noisy_waveforms = noisy_waveforms.view(-1, 1, 1, 8000)
        clean_waveforms = clean_waveforms.view(-1, 1, 1, 8000)
        enhanced_waveforms, vad = self._forward(noisy_waveforms)
        vad_loss = F.binary_cross_entropy_with_logits(
            vad.squeeze(),
            ((clean_waveforms.abs() > 0).sum(dim=-1) >= 160).float().squeeze(),
        )
        l1_loss = F.smooth_l1_loss(enhanced_waveforms, clean_waveforms, beta=0.5)
        mrstft_loss = self.hparams.mr_stft_lambda * self.multi_resolution_stft_loss(
            enhanced_waveforms, clean_waveforms
        )
        loss = l1_loss + mrstft_loss + vad_loss
        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_waveform_loss": l1_loss,
                f"{stage}_spectral_loss": mrstft_loss,
                f"{stage}_vad_loss": vad_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def stft_mag(self, input, fft_size, hop_length, win_length):
        x_stft = torch.stft(
            input,
            fft_size,
            hop_length,
            win_length,
            window=torch.hann_window(win_length, device=input.device),
            return_complex=True,
        )

        # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
        return x_stft.abs().clamp(min=math.sqrt(1e-7)).transpose(2, 1)

    def multi_resolution_stft_loss(self, x, y):
        loss = 0
        for fft_size, hop_length, win_length in zip(
            self.hparams.fft_sizes,
            self.hparams.hop_lengths,
            self.hparams.win_lengths,
        ):
            x_mag = self.stft_mag(
                x.view(-1, x.shape[-1]), fft_size, hop_length, win_length
            )
            y_mag = self.stft_mag(
                y.view(-1, x.shape[-1]), fft_size, hop_length, win_length
            )

            # Spectral convergence loss:
            # sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            # Log STFT magnitude loss:
            log_mag_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))

            loss += log_mag_loss
        return loss


class BaseSpectral(Base):
    def __init__(
        self,
        max_frames: int,
        n_fft: int,
        hop_length: Optional[int],
        win_length: Optional[int],
        window: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        hop_length = hop_length if hop_length is not None else n_fft // 4
        win_length = win_length if win_length is not None else n_fft
        self.stft = STFT(
            max_frames=max_frames,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

    def forward(self, waveforms):
        # Compute the STFT to get magnitude and phase
        log_magnitudes, phases = self.stft.forward(waveforms)

        # Apply filters
        log_magnitudes, vad = self._forward(log_magnitudes)

        # Reconstruct the waveform from the magnitude and phase
        waveforms = self.stft.inverse(log_magnitudes, phases)
        # waveforms *= (1e4 * vad[..., None, None]).sigmoid()

        return waveforms

    def _forward(self, log_magnitudes):
        log_magnitudes = (log_magnitudes + 2) * 0.5
        filters, vad = self.model._forward(log_magnitudes)
        log_magnitudes = (log_magnitudes * filters).clamp(min=-2.46, max=4)
        log_magnitudes = log_magnitudes * 2 - 2
        return log_magnitudes, vad

    def _shared_step(self, batch, batch_idx, stage):
        noisy_waveforms, clean_waveforms, _ = batch
        noisy_waveforms = noisy_waveforms.view(-1, 1, 1, 8000)
        clean_waveforms = clean_waveforms.view(-1, 1, 1, 8000)
        noisy_log_magnitudes, phases = self.stft.forward(noisy_waveforms)
        clean_log_magnitudes, _ = self.stft.forward(clean_waveforms)
        enhanced_log_magnitudes, vad = self._forward(noisy_log_magnitudes)

        # Spectral loss
        spectral_loss = F.l1_loss(enhanced_log_magnitudes, clean_log_magnitudes)

        # Voice activity detection loss
        vad_loss = F.binary_cross_entropy_with_logits(
            vad.squeeze(),
            ((clean_waveforms.abs() > 0).sum(dim=-1) >= 160).float().squeeze(),
        )

        # Waveform loss
        enhanced_waveforms = self.stft.inverse(enhanced_log_magnitudes, phases)
        waveform_loss = F.smooth_l1_loss(
            enhanced_waveforms, clean_waveforms[:, 0, 0, :], beta=0.5
        )

        loss = waveform_loss + spectral_loss + vad_loss
        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_vad_loss": vad_loss,
                f"{stage}_waveform_loss": waveform_loss,
                f"{stage}_spectral_loss": spectral_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss


# class Discriminator(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int,
#         max_channels: int,
#         kernel_size: int,
#         stride: int,
#         padding: int,
#         encoder_n_layers: int,
#         bias: bool,
#     ) -> None:
#         super().__init__()
#         self.discriminator = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv2d(
#                         in_channels,
#                         hidden_channels,
#                         kernel_size=(1, kernel_size),
#                         stride=(1, stride),
#                         padding=(0, padding),
#                         bias=bias,
#                     ),
#                     nn.ReLU(inplace=True),
#                 )
#             ]
#         )
#         for i in range(encoder_n_layers):
#             in_channels = hidden_channels
#             hidden_channels = min(hidden_channels * 2, max_channels)
#             self.discriminator.append(
#                 DepthwiseSeparableConv(
#                     in_channels, hidden_channels, kernel_size, stride, padding, bias
#                 )
#             )

#         L = 160000
#         for _ in range(len(self.discriminator)):
#             L = 1 + np.floor((L - kernel_size) / stride)
#         self.head = nn.Sequential(nn.Linear(int(L), 1), nn.Sigmoid())

#     def forward(self, x: torch.FloatTensor):
#         for layer in self.discriminator:
#             x = layer(x)
#         x = self.head(x.squeeze(dim=2))
#         x = x.squeeze(dim=2).mean(dim=1, keepdim=True)
#         return x


# class GAN(pl.LightningModule):
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int,
#         max_channels: int,
#         out_channels: int,
#         kernel_size: int,
#         stride: int,
#         padding: int,
#         encoder_n_layers: int,
#         nhead: int,
#         num_layers: int,
#         dropout: float,
#         bias: bool,
#         src_sampling_rate: int,
#         tgt_sampling_rate: int,
#         generator_optimizer: Dict[str, Any],
#         discriminator_optimizer: Dict[str, Any],
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()
#         self.automatic_optimization = False

#         # encoder and decoder
#         self.generator = MobileNetV1(
#             in_channels,
#             hidden_channels,
#             max_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             padding,
#             encoder_n_layers,
#             nhead,
#             num_layers,
#             dropout,
#             bias,
#             src_sampling_rate,
#             tgt_sampling_rate,
#         )
#         self.discriminator = Discriminator(
#             in_channels,
#             hidden_channels,
#             max_channels,
#             kernel_size,
#             stride,
#             padding,
#             encoder_n_layers - 2,
#             bias,
#         )

#     def forward(self, waveforms):
#         x = torchaudio.functional.resample(
#             waveforms,
#             self.hparams.src_sampling_rate,
#             self.hparams.tgt_sampling_rate,
#         )

#         x = self.generator._forward(x)

#         x = torchaudio.functional.resample(
#             x,
#             self.hparams.tgt_sampling_rate,
#             self.hparams.src_sampling_rate,
#         )
#         return x

#     def adversarial_loss(self, y_hat, y):
#         return F.binary_cross_entropy(y_hat, y)

#     def configure_optimizers(self):
#         g_optimizer = torch.optim.AdamW(
#             self.generator.parameters(), **self.hparams.generator_optimizer
#         )
#         d_optimizer = torch.optim.AdamW(
#             self.discriminator.parameters(), **self.hparams.discriminator_optimizer
#         )
#         return [g_optimizer, d_optimizer], []

#     def training_step(self, batch):
#         noisy_waveforms, clean_waveforms, _ = batch
#         B = clean_waveforms.shape[0]

#         g_optimizer, d_optimizer = self.optimizers()

#         # train generator
#         # generate images
#         self.toggle_optimizer(g_optimizer)
#         enhanced_waveforms = self.generator._forward(noisy_waveforms)

#         l1_loss = F.l1_loss(enhanced_waveforms, clean_waveforms)

#         # ground truth result (ie: all fake)
#         # put on GPU because we created this tensor inside training_loop
#         valid = torch.ones(B, 1, device=clean_waveforms.device)

#         # adversarial loss is binary cross-entropy
#         g_loss = self.adversarial_loss(self.discriminator(enhanced_waveforms), valid)
#         self.manual_backward(g_loss + l1_loss)
#         g_optimizer.step()
#         g_optimizer.zero_grad()
#         self.untoggle_optimizer(g_optimizer)

#         # train discriminator
#         # Measure discriminator's ability to classify real from generated samples
#         self.toggle_optimizer(d_optimizer)

#         # how well can it label as real?
#         valid = torch.ones(B, 1, device=clean_waveforms.device)
#         d_real = self.discriminator(clean_waveforms)
#         real_loss = self.adversarial_loss(d_real, valid)

#         # how well can it label as fake?
#         fake = torch.zeros(B, 1, device=clean_waveforms.device)
#         d_fake = self.discriminator(enhanced_waveforms.detach())
#         fake_loss = self.adversarial_loss(d_fake, fake)

#         # discriminator loss is the average of these
#         d_loss = (real_loss + fake_loss) / 2
#         self.manual_backward(d_loss)
#         d_optimizer.step()
#         d_optimizer.zero_grad()
#         self.untoggle_optimizer(d_optimizer)

#         self.log_dict(
#             {
#                 "l1_loss": l1_loss,
#                 "g_loss": g_loss,
#                 "d_loss": d_loss,
#                 "real_acc": d_real.mean(),
#                 "fake_acc": d_fake.mean(),
#             },
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#             sync_dist=True,
#         )


class VADMobileNetV1(Base):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        max_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: _size_2_t,
        encoder_n_layers: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        bias: bool,
        pad_value: float,
        mr_stft_lambda: float,
        fft_sizes: List[int],
        hop_lengths: List[int],
        win_lengths: List[int],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kernel_size = kernel_size
        self.stride = stride
        self.encoder_n_layers = encoder_n_layers
        self.pad_value = pad_value

        # encoder and decoder
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=(0, 0),
                    bias=bias,
                )
            ]
        )

        for i in range(encoder_n_layers):
            in_channels = hidden_channels
            out_channels = hidden_channels
            hidden_channels = min(hidden_channels * 2, max_channels)
            self.encoder.append(
                DepthwiseSeparableConv(
                    in_channels, hidden_channels, kernel_size, stride, padding, bias
                )
            )
            self.decoder.insert(
                0,
                DepthwiseSeparableConvTranspose(
                    hidden_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    0,
                    bias,
                    i == 0,
                ),
            )

        self.bottleneck_attention = TransformerEncoder(
            d_word_vec=hidden_channels,
            n_layers=num_layers,
            n_head=nhead,
            d_k=hidden_channels // nhead,
            d_v=hidden_channels // nhead,
            d_model=hidden_channels,
            d_inner=hidden_channels,
            dropout=dropout,
            n_position=0,
            scale_emb=False,
        )
        self.vad = nn.Linear(hidden_channels, 1)


class VADMobileNetV2(Base):

    def __init__(
        self,
        nhead: int,
        num_layers: int,
        dropout: float,
        width_mult: float = 1.0,
        layer_config: Optional[List[List[int]]] = None,
        kernel_size: _size_2_t = (1, 3),
        stride: _size_2_t = (1, 2),
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        pad_value: float = 0,
        mr_stft_lambda: float = 0.5,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_lengths: List[int] = [50, 120, 240],
        win_lengths: List[int] = [240, 600, 1200],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pad_value = pad_value

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32

        if layer_config is None:
            self.layer_config = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        else:
            self.layer_config = layer_config

        # only check the first element, assuming user knows t,c,n,s are required
        if len(self.layer_config) == 0 or len(self.layer_config[0]) != 4:
            raise ValueError(
                f"layer_config should be non-empty or a 4-element list, got {self.layer_config}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        self.encoder = nn.ModuleList(
            [
                Conv2dNormActivation(
                    1,
                    input_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                )
            ]
        )

        self.decoder = nn.ModuleList(
            [
                ConvTranspose2dNormActivation(
                    input_channel,
                    1,
                    kernel_size=kernel_size,
                    stride=stride,
                    norm_layer=None,
                    activation_layer=None,
                )
            ]
        )

        # building inverted residual blocks
        for t, c, n, s in self.layer_config:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                self.encoder.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        dw_layer=Conv2dNormActivation,
                        norm_layer=norm_layer,
                    )
                )
                self.decoder.insert(
                    0,
                    InvertedResidual(
                        output_channel,
                        input_channel,
                        stride,
                        expand_ratio=t,
                        dw_layer=ConvTranspose2dNormActivation,
                        norm_layer=norm_layer,
                    ),
                )
                input_channel = output_channel

        hidden_channels = self.layer_config[-1][1]
        self.bottleneck_attention = TransformerEncoder(
            d_word_vec=hidden_channels,
            n_layers=num_layers,
            n_head=nhead,
            d_k=hidden_channels // nhead,
            d_v=hidden_channels // nhead,
            d_model=hidden_channels,
            d_inner=hidden_channels,
            dropout=dropout,
            n_position=0,
            scale_emb=False,
        )
        self.vad = nn.Linear(hidden_channels, 1)


class VADSpectralV1(BaseSpectral):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        max_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: _size_2_t,
        encoder_n_layers: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        bias: bool,
        pad_value: float,
        max_frames: int,
        n_fft: int,
        hop_length: Optional[int],
        win_length: Optional[int],
        window: str,
    ):
        super().__init__(max_frames, n_fft, hop_length, win_length, window)
        self.save_hyperparameters()
        self.model = VADMobileNetV1(
            in_channels,
            hidden_channels,
            max_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            encoder_n_layers,
            nhead,
            num_layers,
            dropout,
            bias,
            pad_value,
        )


class VADSpectralV2(BaseSpectral):
    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int],
        win_length: Optional[int],
        window: str,
        nhead: int,
        num_layers: int,
        dropout: float,
        layer_config: List[List[int]],
        pad_value: float,
        max_frames: int,
    ):
        super().__init__(max_frames, n_fft, hop_length, win_length, window)
        self.save_hyperparameters()
        self.model = VADMobileNetV2(
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            layer_config=layer_config,
            pad_value=pad_value,
        )
