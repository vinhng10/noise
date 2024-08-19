from typing import List
import math
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model Definition
class Model(pl.LightningModule):
    SUPPORTED_SAMPLING_RATE = 16000

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        encoder_n_layers: int,
        d_model: int,
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
                        kernel_size=(1, 5),
                        stride=(1, 4),
                        padding=(0, 2),
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
                            kernel_size=(1, 5),
                            stride=(1, 4),
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
                            kernel_size=(1, 5),
                            stride=(1, 4),
                            padding=(0, 1),
                            output_padding=(0, 1),
                            bias=bias,
                        ),
                        nn.ReLU(),
                    ),
                )
            out_channels = hidden_channels

            hidden_channels *= 2

        # self.bottleneck_encoder = nn.Linear(out_channels, d_model, bias=bias)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels // 2,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )
        self.bottleneck_attention = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # self.bottleneck_decoder = nn.Linear(d_model, out_channels, bias=bias)

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

        x = x.squeeze(2).permute(0, 2, 1)
        # x = self.bottleneck_encoder(x)
        x = self.bottleneck_attention(x)
        # x = self.bottleneck_decoder(x)
        x = x.permute(0, 2, 1).unsqueeze(2)

        # decoder
        for i, upsampling_block in enumerate(self.decoder):
            skip_i = skip_connections[i]
            x = x + skip_i[:, :, : x.shape[-1]]
            x = upsampling_block(x)

        x = x * std
        return x

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
