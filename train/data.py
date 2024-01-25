import abc
import os
import torch
import librosa
import warnings
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from typing import Dict, List, TypeVar, Tuple, Iterable
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
    scaler: StandardScaler | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    audio, _ = librosa.load(audio_path, sr=sampling_rate)
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )  # Exclude 0th and highest (Nyquist) bins ???
    spectrogram = np.log10(np.abs(stft) ** 2 + eps).T
    # Some frequency spectra have spiky value => clip within (-5, 5):
    if scaler is not None:
        spectrogram = scaler.transform(spectrogram).clip(-5, 5)
    angle = np.angle(stft)
    return spectrogram, angle


def inverse_log_power_spectrum(
    spectrogram: np.ndarray,
    angle: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    scaler: StandardScaler | None = None,
) -> np.ndarray:
    if scaler is not None:
        spectrogram = scaler.inverse_transform(spectrogram)
    magnitude = np.sqrt(10**spectrogram).T
    audio = librosa.istft(
        magnitude * np.exp(angle * 1j),
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    return audio


class DataModule(pl.LightningDataModule, metaclass=abc.ABCMeta):
    def prepare_data(self):
        """Data operation to perform only on main process."""
        save_file = Path(os.environ["SM_MODEL_DIR"]) / "scaler.npy"
        if not save_file.exists():
            scaler = StandardScaler()
            for noisy_path in (Path(self.hparams.data_dir) / "train" / "noisy").rglob(
                "*.wav"
            ):
                spectrogram, _ = log_power_spectrum(
                    noisy_path,
                    self.hparams.sampling_rate,
                    self.hparams.n_fft,
                    self.hparams.hop_length,
                    self.hparams.win_length,
                    self.hparams.eps,
                )
                scaler.partial_fit(spectrogram)
            np.save(save_file, scaler)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            collate_fn=self._colllate_fn,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            shuffle=False,
            collate_fn=self._colllate_fn,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            persistent_workers=True,
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

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        noisy_path, clean_path = self.files[index]

        noisy_spectrogram, noisy_angle = log_power_spectrum(
            noisy_path,
            self.sampling_rate,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.eps,
            self.scaler,
        )
        noisy_spectrogram = torch.from_numpy(noisy_spectrogram)
        noisy_angle = torch.from_numpy(noisy_angle)

        if self.split == "train":
            clean_spectrogram, clean_angle = log_power_spectrum(
                clean_path,
                self.sampling_rate,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.eps,
                self.scaler,
            )
            clean_spectrogram = torch.from_numpy(clean_spectrogram)
            clean_angle = torch.from_numpy(clean_angle)
        else:
            clean_spectrogram = torch.empty_like(noisy_spectrogram)
            clean_angle = torch.empty_like(noisy_angle)

        return {
            "noisy_spectrogram": noisy_spectrogram,
            "noisy_angle": noisy_angle,
            "clean_spectrogram": clean_spectrogram,
            "clean_angle": clean_angle,
        }


class NSNET2DataModule(DataModule):
    @staticmethod
    def _colllate_fn(batch: List[Dict[str, Tensor]]) -> Iterable[Tensor]:
        noisy_spectrograms = torch.stack(
            [sample["noisy_spectrogram"] for sample in batch], dim=1
        )
        noisy_angles = torch.stack([sample["noisy_angle"] for sample in batch], dim=1)
        clean_spectrograms = torch.stack(
            [sample["clean_spectrogram"] for sample in batch], dim=1
        )
        clean_angles = torch.stack([sample["clean_angle"] for sample in batch], dim=1)
        return noisy_spectrograms, noisy_angles, clean_spectrograms, clean_angles

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
