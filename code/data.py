import abc
from pathlib import Path
from typing import List, TypeVar, Tuple, Iterable

import librosa
import torch
import warnings
import numpy as np
import lightning.pytorch as pl
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=np.ComplexWarning)
T = TypeVar("T")
PathOrStr = Path | str

################################################################################


class DataModule(pl.LightningDataModule, metaclass=abc.ABCMeta):
    def prepare_data(self):
        """Data operation to perform only on main process."""
        pass

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            collate_fn=self._colllate_fn,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    @abc.abstractstaticmethod
    def _colllate_fn(batch: List[Tuple[Tensor]]) -> Iterable[Tensor]:
        pass


class NSNET2Dataset(Dataset):
    def __init__(
        self,
        split: str,
        data_dir: str,
        sampling_rate: int = 48000,
        n_fft: int = 512,
        win_length: int = 512,
        hop_length: int = 128,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.split = split
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.eps = eps
        self.files = []
        for clean_audio in (self.data_dir / split / "clean").rglob("*.wav"):
            fileid = clean_audio.stem[6:]
            noisy_audio = list(
                (self.data_dir / split / "noisy").glob(f"*{fileid}.wav")
            )[0]
            self.files.append((noisy_audio, clean_audio))

    def log_power_spectrum(self, audio_path: PathOrStr):
        audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
        audio = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )[
            1:-1
        ]  # Exclude 0th and highest (Nyquist) bins
        audio = np.log10(np.abs(audio) ** 2 + self.eps)
        return audio

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        noisy_audio, clean_audio = self.files[index]
        noisy_audio = torch.from_numpy(self.log_power_spectrum(noisy_audio).T)
        clean_audio = torch.from_numpy(self.log_power_spectrum(clean_audio).T)

        return noisy_audio, clean_audio


class NSNET2DataModule(DataModule):
    @staticmethod
    def _colllate_fn(batch: List[Tuple[Tensor]]) -> Iterable[Tensor]:
        noisy_audios = torch.stack([sample[0] for sample in batch], dim=1)
        clean_audios = torch.stack([sample[1] for sample in batch], dim=1)
        return noisy_audios, clean_audios

    def __init__(
        self,
        data_dir: str,
        sampling_rate: int = 48000,
        n_fft: int = 512,
        win_length: int = 512,
        hop_length: int = 128,
        eps: float = 1e-8,
        num_workers: int = 2,
        batch_size: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        """Data operations to perform on every GPUs.

        Parameters
        ----------
        stage : str
            _description_
        """
        if stage == "fit":
            self.trainset = NSNET2Dataset(
                "train",
                self.hparams.data_dir,
                self.hparams.sampling_rate,
                self.hparams.n_fft,
                self.hparams.win_length,
                self.hparams.hop_length,
                self.hparams.eps,
            )
        #     self.valset = ReeoDataset(
        #         "val",
        #         self.hparams.cache_path,
        #         self.hparams.sequence_length,
        #         self.hparams.step,
        #     )

        # elif stage == "validate":
        #     self.valset = ReeoDataset(
        #         "val",
        #         self.hparams.cache_path,
        #         self.hparams.sequence_length,
        #         self.hparams.step,
        #     )

        # elif stage == "test":
        #     self.testset = ReeoDataset(
        #         "test",
        #         self.hparams.cache_path,
        #         self.hparams.sequence_length,
        #         self.hparams.step,
        #     )

        else:
            raise ValueError(f"Stage {stage} is not supported.")
