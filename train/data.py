import abc
import os
import torch
import librosa
import warnings
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from typing import List, TypeVar, Tuple, Iterable
from sklearn.discriminant_analysis import StandardScaler
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=np.ComplexWarning)
T = TypeVar("T")
PathOrStr = Path | str


################################################################################


def log_power_spectrum(
    audio_path: PathOrStr,
    sampling_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    eps: float,
) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=sampling_rate)
    audio = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )  # Exclude 0th and highest (Nyquist) bins ???
    audio = np.log10(np.abs(audio) ** 2 + eps)
    return audio.T


class DataModule(pl.LightningDataModule, metaclass=abc.ABCMeta):
    def prepare_data(self):
        """Data operation to perform only on main process."""
        save_file = Path(os.environ["SM_MODEL_DIR"]) / "scaler.npy"
        if not save_file.exists():
            scaler = StandardScaler()
            for noisy_path in (Path(self.hparams.data_dir) / "train" / "noisy").rglob(
                "*.wav"
            ):
                scaler.partial_fit(
                    log_power_spectrum(
                        noisy_path,
                        self.hparams.sampling_rate,
                        self.hparams.n_fft,
                        self.hparams.hop_length,
                        self.hparams.win_length,
                        self.hparams.eps,
                    )
                )
            np.save(save_file, scaler)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            collate_fn=self._colllate_fn,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            shuffle=False,
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
        sampling_rate: int = 16000,
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
        for noisy_path in (self.data_dir / split / "noisy").rglob("*.wav"):
            fileid = f'fileid_{noisy_path.stem.split("_")[-1]}'
            tmp = list((self.data_dir / split / "clean").glob(f"*{fileid}.wav"))
            clean_path = tmp[0] if len(tmp) > 0 else None
            self.files.append((noisy_path, clean_path))

        self.scaler = np.load(
            Path(os.environ["SM_MODEL_DIR"]) / "scaler.npy", allow_pickle=True
        ).item()

    def standardize(self, audio: np.ndarray) -> np.ndarray:
        # Some frequency spectra have spiky value => clip within (-5, 5):
        return self.scaler.transform(audio).clip(-5, 5)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        noisy_path, clean_path = self.files[index]
        noisy_audio = torch.from_numpy(
            self.standardize(
                log_power_spectrum(
                    noisy_path,
                    self.sampling_rate,
                    self.n_fft,
                    self.hop_length,
                    self.win_length,
                    self.eps,
                )
            )
        )
        if self.split == "train":
            clean_audio = torch.from_numpy(
                self.standardize(
                    log_power_spectrum(
                        clean_path,
                        self.sampling_rate,
                        self.n_fft,
                        self.hop_length,
                        self.win_length,
                        self.eps,
                    )
                )
            )
        else:
            clean_audio = torch.empty_like(noisy_audio)

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

            self.valset = NSNET2Dataset(
                "val",
                self.hparams.data_dir,
                self.hparams.sampling_rate,
                self.hparams.n_fft,
                self.hparams.win_length,
                self.hparams.hop_length,
                self.hparams.eps,
            )

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
