import random
import boto3
import torch
import librosa
import numpy as np
import lightning.pytorch as pl
from numpy.typing import NDArray
from os import PathLike
from pathlib import Path
from typing import Dict
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from audiomentations import *
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.composition import BaseCompose


################################################################################


class Cut(BaseWaveformTransform):
    def __init__(self, length: int, is_val: bool = False, p: float = 0.5):
        super().__init__(p)
        self.length = length
        self.is_val = is_val

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            if self.is_val:
                # Take segment from the middle for validation
                self.parameters["offset"] = max((len(samples) - self.length) // 2, 0)
            else:
                # Take a random segment for training
                self.parameters["offset"] = np.random.randint(
                    0, max(len(samples) - self.length, 1)
                )

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        if self.length > 0:
            return samples[
                self.parameters["offset"] : self.parameters["offset"] + self.length
            ]
        else:
            return samples


class ToTensor(BaseWaveformTransform):
    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> torch.FloatTensor:
        return torch.from_numpy(samples)[None, None, :].float()


class NoiseDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        sampling_rate: int = 16000,
        length: int = 2048,
        num_samples: int = 120000,
        num_workers: int = 4,
        batch_size: int = 128,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_transforms = Compose(
            [
                Cut(length=length, is_val=False, p=1.0),
                Shift(min_shift=-1.0, max_shift=1.0, rollover=False, p=p),
                AddBackgroundNoise(
                    sounds_path=f"{data_dir}/train/noise",
                    min_snr_db=5.0,
                    max_snr_db=40.0,
                    noise_transform=Compose(
                        [
                            PolarityInversion(),
                            PitchShift(min_semitones=-12, max_semitones=12),
                        ]
                    ),
                    p=p,
                ),
                PolarityInversion(p=p),
                BandPassFilter(
                    min_center_freq=1000,
                    max_center_freq=4000,
                    min_bandwidth_fraction=1,
                    max_bandwidth_fraction=1.99,
                    p=p / 5,
                ),
                AddColorNoise(min_snr_db=10.0, max_snr_db=40.0, p=p),
                RoomSimulator(
                    min_absorption_value=0.075,
                    max_absorption_value=0.4,
                    leave_length_unchanged=True,
                    p=p,
                ),
                BitCrush(
                    min_bit_depth=8,
                    max_bit_depth=12,
                    p=p,
                ),
                ToTensor(p=1.0),
            ]
        )
        self.val_transforms = Compose(
            [Cut(length=8000, is_val=True, p=1.0), ToTensor(p=1.0)]
        )

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
        if not data_dir.exists():
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
                        if not file_name.exists():
                            file_name.parent.mkdir(parents=True, exist_ok=True)

                            # Download the file
                            s3.download_file("db-noise", key, file_name)

    def setup(self, stage: str) -> None:
        """Data operations to perform on every GPUs.

        Parameters
        ----------
        stage : str
            _description_
        """
        data_dir = Path(self.hparams.data_dir)
        self.train_files = self.get_files(data_dir / "train")
        self.val_files = self.get_files(data_dir / "val")

        if stage == "fit":
            self.trainset = NoiseDataset(
                self.train_files,
                self.hparams.num_samples,
                self.hparams.sampling_rate,
                self.train_transforms,
            )
            self.valset = NoiseDataset(
                self.val_files, -1, self.hparams.sampling_rate, self.val_transforms
            )
        elif stage == "predict":
            self.valset = NoiseDataset(
                self.val_files, -1, self.hparams.sampling_rate, self.val_transforms
            )
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
    def __init__(
        self,
        files: list[tuple[Path, Path | None]],
        num_samples: int,
        sampling_rate: int,
        transforms: BaseCompose,
    ):
        super().__init__()
        random.shuffle(files)
        if num_samples > 0:
            self.files = files[:num_samples]
        else:
            self.files = files
        self.num_samples = num_samples
        self.sampling_rate = sampling_rate
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        paths = self.files[index]
        noisy_waveform, _ = librosa.load(paths[0], sr=self.sampling_rate)
        clean_waveform, _ = librosa.load(paths[1], sr=self.sampling_rate)

        noisy_waveform = self.transforms(
            samples=noisy_waveform, sample_rate=self.sampling_rate
        )
        self.transforms.freeze_parameters()
        for t in self.transforms.transforms:
            if t.__class__.__name__ not in ["Cut", "Shift", "ToTensor"]:
                t.parameters["should_apply"] = False
        clean_waveform = self.transforms(
            samples=clean_waveform, sample_rate=self.sampling_rate
        )
        self.transforms.unfreeze_parameters()

        return noisy_waveform, clean_waveform, paths[1].stem[6:]
