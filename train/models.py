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
        n_filters: int,
        mr_stft_lambda: float,
        fft_sizes: List[int],
        hop_lengths: List[int],
        win_lengths: List[int],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, n_filters, 3, 1, 1),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(n_filters, n_filters * 2, 3, 1, 1),
                    nn.BatchNorm1d(n_filters * 2),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(n_filters * 2, n_filters, 3, 1, 1),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(),
                ),
                nn.Conv1d(n_filters, 1, 3, 1, 1),
            ]
        )

    def forward(self, waveforms):
        outputs = waveforms
        for block in self.blocks:
            outputs = block(outputs)
        return outputs

    def training_step(self, batch, batch_idx):
        noisy_waveforms, clean_waveforms = batch
        enhanced_waveforms = self.forward(noisy_waveforms)
        l1_loss = F.l1_loss(enhanced_waveforms, clean_waveforms)
        mrstft_loss = self.hparams.mr_stft_lambda * self.multi_resolution_stft_loss(
            enhanced_waveforms, clean_waveforms
        )
        loss = l1_loss + mrstft_loss
        self.log_dict(
            {"train_loss": loss, "train_l1_loss": l1_loss, "train_mrstft_loss": mrstft_loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def _stft_mag(self, input, fft_size, hop_length, win_length):
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
            x_mag = self._stft_mag(
                x.view(-1, x.shape[2]), fft_size, hop_length, win_length
            )
            y_mag = self._stft_mag(
                y.view(-1, x.shape[2]), fft_size, hop_length, win_length
            )

            # Spectral convergence loss:
            sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            # Log STFT magnitude loss:
            log_mag_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))

            loss += sc_loss + log_mag_loss
        return loss
