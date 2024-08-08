import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model Definition
class Model(pl.LightningModule):
    SUPPORTED_SAMPLING_RATE = 16000

    def __init__(self, n_filters) -> None:
        super().__init__()
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
        mrstft_loss = 0.5 * self.multi_resolution_stft_loss(
            enhanced_waveforms, clean_waveforms
        )
        loss = l1_loss + mrstft_loss
        self.log_dict(
            {"loss": loss, "l1_loss": l1_loss, "mrstft_loss": mrstft_loss},
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def _stft(self, input, fft_size, hop_length, win_length, window):
        x_stft = torch.stft(
            input, fft_size, hop_length, win_length, window, return_complex=False
        )

        # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
        return torch.sqrt(
            torch.clamp(x_stft[..., 0] ** 2 + x_stft[..., 1] ** 2, min=1e-7)
        ).transpose(2, 1)

    def multi_resolution_stft_loss(self, input, target):
        
