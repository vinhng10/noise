from typing import Any, Dict, List
import math
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio


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


# CleanUNet architecture
def padding(x, D, K, S):
    """padding zeroes to x so that denoised audio has the same length"""

    L = x.shape[-1]
    for _ in range(D):
        if L < K:
            L = 1
        else:
            L = 1 + np.ceil((L - K) / S)

    for _ in range(D):
        L = (L - 1) * S + K

    L = int(L)
    x = F.pad(x, (0, L - x.shape[-1]))
    return x


def weight_scaling_init(layer):
    """
    weight rescaling initialization from https://arxiv.org/abs/1911.13254
    """
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    layer.weight.data /= torch.sqrt(alpha)
    layer.bias.data /= torch.sqrt(alpha)


class OriginalCleanUNet(nn.Module):
    """CleanUNet architecture."""

    def __init__(
        self,
        channels_input=1,
        channels_output=1,
        channels_H=64,
        max_H=768,
        encoder_n_layers=8,
        kernel_size=4,
        stride=2,
        tsfm_n_layers=3,
        tsfm_n_head=8,
        tsfm_d_model=512,
        tsfm_d_inner=2048,
    ):
        """
        Parameters:
        channels_input (int):   input channels
        channels_output (int):  output channels
        channels_H (int):       middle channels H that controls capacity
        max_H (int):            maximum H
        encoder_n_layers (int): number of encoder/decoder layers D
        kernel_size (int):      kernel size K
        stride (int):           stride S
        tsfm_n_layers (int):    number of self attention blocks N
        tsfm_n_head (int):      number of heads in each self attention block
        tsfm_d_model (int):     d_model of self attention
        tsfm_d_inner (int):     d_inner of self attention
        """

        super(OriginalCleanUNet, self).__init__()

        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner

        # encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(encoder_n_layers):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(channels_input, channels_H, kernel_size, stride),
                    nn.ReLU(),
                    nn.Conv1d(channels_H, channels_H * 2, 1),
                    nn.GLU(dim=1),
                )
            )
            channels_input = channels_H

            if i == 0:
                # no relu at end
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv1d(channels_H, channels_H * 2, 1),
                        nn.GLU(dim=1),
                        nn.ConvTranspose1d(
                            channels_H, channels_output, kernel_size, stride
                        ),
                    )
                )
            else:
                self.decoder.insert(
                    0,
                    nn.Sequential(
                        nn.Conv1d(channels_H, channels_H * 2, 1),
                        nn.GLU(dim=1),
                        nn.ConvTranspose1d(
                            channels_H, channels_output, kernel_size, stride
                        ),
                        nn.ReLU(),
                    ),
                )
            channels_output = channels_H

            # double H but keep below max_H
            channels_H *= 2
            channels_H = min(channels_H, max_H)

        # self attention block
        self.tsfm_conv1 = nn.Conv1d(channels_output, tsfm_d_model, kernel_size=1)
        self.tsfm_encoder = TransformerEncoder(
            d_word_vec=tsfm_d_model,
            n_layers=tsfm_n_layers,
            n_head=tsfm_n_head,
            d_k=tsfm_d_model // tsfm_n_head,
            d_v=tsfm_d_model // tsfm_n_head,
            d_model=tsfm_d_model,
            d_inner=tsfm_d_inner,
            dropout=0.0,
            n_position=0,
            scale_emb=False,
        )
        self.tsfm_conv2 = nn.Conv1d(tsfm_d_model, channels_output, kernel_size=1)

        # weight scaling initialization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_scaling_init(layer)

    def forward(self, noisy_audio):
        # (B, L) -> (B, C, L)
        if len(noisy_audio.shape) == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        B, C, L = noisy_audio.shape
        assert C == 1

        # normalization and padding
        std = noisy_audio.std(dim=2, keepdim=True) + 1e-3
        noisy_audio /= std
        x = padding(noisy_audio, self.encoder_n_layers, self.kernel_size, self.stride)

        # encoder
        skip_connections = []
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            skip_connections.append(x)
        skip_connections = skip_connections[::-1]

        # attention mask for causal inference; for non-causal, set attn_mask to None
        len_s = x.shape[-1]  # length at bottleneck
        attn_mask = (
            1 - torch.triu(torch.ones((1, len_s, len_s), device=x.device), diagonal=1)
        ).bool()

        x = self.tsfm_conv1(x)  # C 1024 -> 512
        x = x.permute(0, 2, 1)
        x = self.tsfm_encoder(x, src_mask=attn_mask)
        x = x.permute(0, 2, 1)
        x = self.tsfm_conv2(x)  # C 512 -> 1024

        # decoder
        ups = []
        for i, upsampling_block in enumerate(self.decoder):
            ups.append(x)
            skip_i = skip_connections[i]
            x += skip_i[:, :, : x.shape[-1]]
            x = upsampling_block(x)

        x = x[:, :, :L] * std
        return x


# Model Definition:
class Model(pl.LightningModule):
    SUPPORTED_SAMPLING_RATE = 16000

    def training_step(self, batch, batch_idx):
        noisy_waveforms, clean_waveforms, _ = batch
        enhanced_waveforms = self.forward(noisy_waveforms)
        l1_loss = F.l1_loss(enhanced_waveforms, clean_waveforms)
        mrstft_loss = self.hparams.mr_stft_lambda * self.multi_resolution_stft_loss(
            enhanced_waveforms, clean_waveforms
        )
        loss = l1_loss + mrstft_loss
        self.log_dict(
            {
                "train_loss": loss,
                "train_l1_loss": l1_loss,
                "train_mrstft_loss": mrstft_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

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
            sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            # Log STFT magnitude loss:
            log_mag_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))

            loss += sc_loss + log_mag_loss
        return loss


class CleanUNet(Model):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        encoder_n_layers: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        bias: bool,
        mr_stft_lambda: float,
        fft_sizes: List[int],
        hop_lengths: List[int],
        win_lengths: List[int],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(encoder_n_layers):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        hidden_channels,
                        kernel_size=(1, 3),
                        stride=(1, 2),
                        padding=(0, 1),
                        bias=bias,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels * 2,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),
                        bias=bias,
                    ),
                    nn.GLU(dim=1),
                )
            )
            in_channels = hidden_channels

            if i == 0:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(
                            hidden_channels,
                            hidden_channels * 2,
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            padding=(0, 0),
                            bias=bias,
                        ),
                        nn.GLU(dim=1),
                        nn.ConvTranspose2d(
                            hidden_channels,
                            out_channels,
                            kernel_size=(1, 3),
                            stride=(1, 2),
                            padding=(0, 1),
                            output_padding=(0, 1),
                            bias=bias,
                        ),
                    ),
                )
            else:
                self.decoder.insert(
                    0,
                    nn.Sequential(
                        nn.Conv2d(
                            hidden_channels,
                            hidden_channels * 2,
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            padding=(0, 0),
                            bias=bias,
                        ),
                        nn.GLU(dim=1),
                        nn.ConvTranspose2d(
                            hidden_channels,
                            out_channels,
                            kernel_size=(1, 3),
                            stride=(1, 2),
                            padding=(0, 1),
                            output_padding=(0, 1),
                            bias=bias,
                        ),
                        nn.ReLU(),
                    ),
                )
            out_channels = hidden_channels

            # hidden_channels *= 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=bias,
        )
        self.bottleneck_attention = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        weight rescaling initialization from https://arxiv.org/abs/1911.13254
        """
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            w = module.weight.detach()
            alpha = 10.0 * w.std()
            module.weight.data /= torch.sqrt(alpha)
            module.bias.data /= torch.sqrt(alpha)

    def forward(self, waveforms):
        # normalization and padding
        std = waveforms.std(dim=-1, keepdim=True) + 1e-3
        x = waveforms / std

        # encoder
        skip_connections = []
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            skip_connections.append(x)
        skip_connections = skip_connections[::-1]

        x = x.squeeze(2).permute(2, 0, 1)
        x = self.bottleneck_attention(x)
        x = x.permute(1, 2, 0).unsqueeze(2)

        # decoder
        for i, upsampling_block in enumerate(self.decoder):
            skip_i = skip_connections[i]
            x = x + skip_i[:, :, : x.shape[-1]]
            x = upsampling_block(x)

        x = x * std
        return x


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=[1, kernel_size],
                stride=[1, stride],
                padding=[0, padding],
                groups=in_channels,
                bias=bias,
            ),
            # nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
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
                kernel_size=[1, kernel_size],
                stride=[1, stride],
                padding=[0, padding],
                output_padding=[0, output_padding],
                groups=in_channels,
                bias=bias,
            ),
            # nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU() if not is_output else nn.Identity(),
        )

    def forward(self, x):
        return self.module(x)


class MobileNetV1(Model):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        max_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        encoder_n_layers: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        bias: bool,
        src_sampling_rate: int,
        tgt_sampling_rate: int,
        mr_stft_lambda: float,
        fft_sizes: List[int],
        hop_lengths: List[int],
        win_lengths: List[int],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # encoder and decoder
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        hidden_channels,
                        kernel_size=(1, kernel_size),
                        stride=(1, stride),
                        padding=(0, padding),
                        bias=bias,
                    ),
                    nn.ReLU(),
                )
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=(1, kernel_size),
                    stride=(1, stride),
                    padding=(0, padding),
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

    def forward(self, waveforms):
        x = torchaudio.functional.resample(
            waveforms,
            self.hparams.src_sampling_rate,
            self.hparams.tgt_sampling_rate,
        )

        x = self._forward(x)

        x = torchaudio.functional.resample(
            x,
            self.hparams.tgt_sampling_rate,
            self.hparams.src_sampling_rate,
        )
        return x

    def _forward(self, x):
        L = x.shape[-1]
        std = x.std(dim=-1, keepdim=True) + 1e-3
        x = x / std
        x = padding(
            x,
            self.hparams.encoder_n_layers + 1,
            self.hparams.kernel_size,
            self.hparams.stride,
        )

        # encoder
        downs = []
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            downs.append(x)
        downs = downs[::-1]

        x = x.squeeze(2).permute(0, 2, 1)
        x = self.bottleneck_attention(x, None)
        print(x.shape)
        x = x.permute(0, 2, 1).unsqueeze(2)

        # decoder
        ups = []
        for i, upsampling_block in enumerate(self.decoder):
            ups.append(x)
            skip_i = downs[i]
            x = x + skip_i[:, :, :, : x.shape[-1]]
            x = upsampling_block(x)

        x = x[:, :, :, :L] * std
        return x

    def training_step(self, batch, batch_idx):
        noisy_waveforms, clean_waveforms, _ = batch
        enhanced_waveforms = self._forward(noisy_waveforms)
        l1_loss = F.l1_loss(enhanced_waveforms, clean_waveforms)
        mrstft_loss = self.hparams.mr_stft_lambda * self.multi_resolution_stft_loss(
            enhanced_waveforms, clean_waveforms
        )
        loss = l1_loss + mrstft_loss
        self.log_dict(
            {
                "train_loss": loss,
                "train_l1_loss": l1_loss,
                "train_mrstft_loss": mrstft_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss


class KnowledgeDistillation(Model):
    def __init__(
        self,
        student: Dict[str, Any],
        teacher: Dict[str, Any],
        mr_stft_lambda: float,
        fft_sizes: List[int],
        hop_lengths: List[int],
        win_lengths: List[int],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.student = MobileNetV1(
            **student,
            mr_stft_lambda=mr_stft_lambda,
            fft_sizes=fft_sizes,
            hop_lengths=hop_lengths,
            win_lengths=win_lengths,
        )

        self.encoder_matcher = nn.Conv1d(
            student["max_channels"], teacher["max_H"], 1, 1, 3
        )
        # self.decoder_matcher = nn.Conv1d(
        #     student["hidden_channels"], teacher["channels_H"], 1, 1, 63
        # )

        ckpt_path = teacher.pop("ckpt_path")
        checkpoint = torch.load(
            ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.teacher = OriginalCleanUNet(**teacher)
        self.teacher.load_state_dict(checkpoint["model_state_dict"])
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        noisies, cleans, _ = batch
        s_enhanceds, s_downs, s_ups = self.forward(noisies)
        t_enhanceds, t_downs, t_ups = self.teacher(noisies.squeeze(2))

        # Knowledge distillation loss:
        kd_loss = 10 * F.mse_loss(
            self.encoder_matcher(s_downs[-1].squeeze(2)), t_downs[len(s_downs) - 1]
        )
        # kd_loss += F.mse_loss(self.decoder_matcher(s_ups[-1].squeeze(2)), t_ups[-1])

        # Speech enhancement loss:
        l1_loss = F.l1_loss(s_enhanceds, cleans)
        mrstft_loss = self.hparams.mr_stft_lambda * self.multi_resolution_stft_loss(
            s_enhanceds, cleans
        )
        se_loss = l1_loss + mrstft_loss

        # Final loss:
        loss = kd_loss + se_loss

        self.log_dict(
            {
                "train_loss": loss,
                "train_se_loss": se_loss,
                "train_kd_loss": kd_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss
