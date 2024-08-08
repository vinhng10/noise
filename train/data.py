import torch
import warnings
import torchaudio
import numpy as np
import lightning.pytorch as pl
from os import PathLike
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


class Transform:
    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate

    def __call__(self, path: str | PathLike) -> tuple[Tensor, int]:
        waveform, orig_sr = torchaudio.load(path)
        waveform = torchaudio.functional.resample(waveform, orig_sr, self.sampling_rate)
        waveform = torch.mean(waveform, dim=0)
        return waveform


class NoiseDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        sampling_rate: int = 16000,
        num_workers: int = 4,
        batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_transforms = Transform(sampling_rate)
        self.val_transforms = Transform(sampling_rate)

    def get_files(self, data_dir: str | PathLike) -> list[tuple[Path, Path | None]]:
        data_dir = Path(data_dir)
        files = []
        for noisy_path in (data_dir / "noisy").rglob("*.wav"):
            fileid = f'fileid_{noisy_path.stem.split("_")[-1]}'
            clean_path = data_dir / "clean" / f"clean_{fileid}.wav"
            clean_path = clean_path if clean_path.exists() else None
            files.append((noisy_path, clean_path))
        return files

    def prepare_data(self) -> None:
        """Data operation to perform only on main process."""
        data_dir = Path(self.hparams.data_dir)
        self.train_files = self.get_files(data_dir / "train")
        self.val_files = self.get_files(data_dir / "val")

        # scaler_path = Path("/opt/ml/input/data/mos/scaler.npy")
        # save_file = Path(os.environ["SM_MODEL_DIR"]) / "scaler.npy"
        # if not scaler_path.exists():
        #     print("==> Compute new scaler")
        #     scaler = StandardScaler()
        #     for noisy_path in (Path(self.hparams.data_dir) / "train" / "noisy").rglob(
        #         "*.wav"
        #     ):
        #         spectrogram, _ = log_power_spectrum(
        #             noisy_path,
        #             self.hparams.sampling_rate,
        #             self.hparams.n_fft,
        #             self.hparams.hop_length,
        #             self.hparams.win_length,
        #             self.hparams.eps,
        #         )
        #         scaler.partial_fit(spectrogram)
        #     np.save(save_file, scaler)
        # else:
        #     print("==> Use precomputed scaler")
        #     # scaler_path.rename(save_file)

    def setup(self, stage: str) -> None:
        """Data operations to perform on every GPUs.

        Parameters
        ----------
        stage : str
            _description_
        """
        if stage == "fit":
            self.trainset = NoiseDataset(self.train_files, self.train_transforms)
            self.valset = NoiseDataset(self.val_files, self.val_transforms)

        else:
            raise ValueError(f"Stage {stage} is not supported.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valset,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            persistent_workers=True,
        )


class NoiseDataset(Dataset):
    def __init__(self, files: list[tuple[Path, Path | None]], transform: Transform):
        super().__init__()
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        noisy_path, clean_path = self.files[index]
        noisy_waveform = self.transform(noisy_path)

        if clean_path is not None:
            clean_waveform = self.transform(clean_path)
        else:
            clean_waveform = torch.empty_like(noisy_waveform)

        return noisy_waveform, clean_waveform
