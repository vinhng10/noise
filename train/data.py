import boto3
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
    def __init__(self, sampling_rate: int, length: int) -> None:
        self.sampling_rate = sampling_rate
        self.length = length

    def __call__(self, noisy_path, clean_path) -> tuple[Tensor, int]:
        noisy_waveform, noisy_orig_sr = torchaudio.load(str(noisy_path))
        noisy_waveform = torchaudio.functional.resample(
            noisy_waveform, noisy_orig_sr, self.sampling_rate
        )
        noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)
        noisy_waveform = noisy_waveform.unsqueeze(0)

        if clean_path is not None:
            clean_waveform, clean_orig_sr = torchaudio.load(str(clean_path))
            clean_waveform = torchaudio.functional.resample(
                clean_waveform, clean_orig_sr, self.sampling_rate
            )
            clean_waveform = torch.mean(clean_waveform, dim=0, keepdim=True)
            clean_waveform = clean_waveform.unsqueeze(0)
        else:
            clean_waveform = torch.empty_like(noisy_waveform)

        if self.length > 0:
            offset = np.random.randint(
                0, max(noisy_waveform.shape[-1] - self.length, 1)
            )
            noisy_waveform = noisy_waveform[:, :, offset : offset + self.length]
            clean_waveform = clean_waveform[:, :, offset : offset + self.length]

        return noisy_waveform, clean_waveform


class NoiseDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        sampling_rate: int = 16000,
        length: int = 2048,
        num_workers: int = 4,
        batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_transforms = Transform(sampling_rate, length)
        self.val_transforms = Transform(sampling_rate, length)

    def get_files(self, data_dir: str | PathLike) -> list[tuple[Path, Path | None]]:
        data_dir = Path(data_dir)
        files = []
        for noisy_path in (data_dir / "noisy").rglob("*.wav"):
            fileid = f'fileid_{noisy_path.stem.split("_")[-1]}'
            clean_path = data_dir / "clean" / f"clean_{fileid}.wav"
            clean_path = clean_path if clean_path.exists() else None
            new_noisy_path = data_dir / "noisy" / f"noisy_{fileid}.wav"
            noisy_path.rename(new_noisy_path)
            files.append((new_noisy_path, clean_path))
        return files

    def prepare_data(self) -> None:
        """Data operation to perform only on main process."""
        data_dir = Path(self.hparams.data_dir)
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket="db-noise", Prefix="datasets/train/"):
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]

                    # Skip directories (keys ending with '/')
                    if key.endswith("/"):
                        continue

                    file_name = data_dir / "/".join(key.split("/")[1:])
                    file_name.parent.mkdir(parents=True, exist_ok=True)

                    # Download the file
                    s3.download_file("db-noise", key, file_name)

        self.train_files = self.get_files(data_dir / "train")
        # self.val_files = self.get_files(data_dir / "val")

    def setup(self, stage: str) -> None:
        """Data operations to perform on every GPUs.

        Parameters
        ----------
        stage : str
            _description_
        """
        if stage == "fit":
            self.trainset = NoiseDataset(self.train_files, self.train_transforms)
            # self.valset = NoiseDataset(self.val_files, self.val_transforms)
        elif stage == "predict":
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
        noisy_waveform, clean_waveform = self.transform(noisy_path, clean_path)
        return noisy_waveform, clean_waveform, clean_path.stem[6:]
