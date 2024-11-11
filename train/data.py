import functools
import random
import warnings
import boto3
import torch
import librosa
import numpy as np
import lightning.pytorch as pl
from numpy.typing import NDArray
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from audiomentations import *
from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.composition import BaseCompose
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    find_audio_files_in_paths,
)


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


class JustNoise(BaseWaveformTransform):
    def __init__(
        self,
        sounds_path: Union[List[Path], List[str], Path, str],
        min_snr_db: Optional[float] = None,
        max_snr_db: Optional[float] = None,
        noise_transform: Optional[
            Callable[[NDArray[np.float32], int], NDArray[np.float32]]
        ] = None,
        p: float = 0.5,
        lru_cache_size: int = 2,
    ):
        super().__init__(p)
        self.sound_file_paths = find_audio_files_in_paths(sounds_path)
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]

        assert len(self.sound_file_paths) > 0

        if min_snr_db is not None:
            self.min_snr_db = min_snr_db
        else:
            self.min_snr_db = 3.0  # the default

        if max_snr_db is not None:
            self.max_snr_db = max_snr_db
        else:
            self.max_snr_db = 30.0  # the default

        assert self.min_snr_db <= self.max_snr_db

        self._load_sound = functools.lru_cache(maxsize=lru_cache_size)(
            JustNoise._load_sound
        )
        self.noise_transform = noise_transform

    @staticmethod
    def _load_sound(file_path, sample_rate):
        return load_sound_file(file_path, sample_rate)

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["snr_db"] = random.uniform(self.min_snr_db, self.max_snr_db)
            self.parameters["noise_file_path"] = random.choice(self.sound_file_paths)

            num_samples = len(samples)
            noise_sound, _ = self._load_sound(
                self.parameters["noise_file_path"], sample_rate
            )

            num_noise_samples = len(noise_sound)
            min_noise_offset = 0
            max_noise_offset = max(0, num_noise_samples - num_samples - 1)
            self.parameters["noise_start_index"] = random.randint(
                min_noise_offset, max_noise_offset
            )
            self.parameters["noise_end_index"] = (
                self.parameters["noise_start_index"] + num_samples
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        if self.are_parameters_frozen:
            return np.zeros_like(samples)

        noise_sound, _ = self._load_sound(
            self.parameters["noise_file_path"], sample_rate
        )
        noise_sound = noise_sound[
            self.parameters["noise_start_index"] : self.parameters["noise_end_index"]
        ]

        if self.noise_transform:
            noise_sound = self.noise_transform(noise_sound, sample_rate)

        noise_rms = calculate_rms(noise_sound)
        if noise_rms < 1e-9:
            warnings.warn(
                "The file {} is too silent to be added as noise. Returning the input"
                " unchanged.".format(self.parameters["noise_file_path"])
            )
            return samples

        clean_rms = calculate_rms(samples)

        desired_noise_rms = calculate_desired_noise_rms(
            clean_rms, self.parameters["snr_db"]
        )

        # Adjust the noise to match the desired noise RMS
        noise_sound = noise_sound * (desired_noise_rms / noise_rms)

        # Repeat the sound if it shorter than the input sound
        num_samples = len(samples)
        while len(noise_sound) < num_samples:
            noise_sound = np.concatenate((noise_sound, noise_sound))
        noise_sound = noise_sound[0:num_samples]

        # Return a mix of the input sound and the background noise sound
        return noise_sound

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of AddBackgroundNoise gets discarded when pickling"
            " it. E.g. this means the cache will not be used when using"
            " AddBackgroundNoise together with multiprocessing on Windows"
        )
        del state["_load_sound"]
        return state


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
                Shift(min_shift=-0.5, max_shift=0.5, rollover=False, p=p),
                AddBackgroundNoise(
                    sounds_path=f"{data_dir}/train/noise",
                    min_snr_db=5.0,
                    max_snr_db=40.0,
                    p=p,
                ),
                JustNoise(
                    sounds_path=f"{data_dir}/train/noise",
                    min_snr_db=5.0,
                    max_snr_db=40.0,
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
            [
                Cut(length=length, is_val=False, p=1.0),
                Shift(min_shift=-0.5, max_shift=0.5, rollover=False, p=p),
                AddBackgroundNoise(
                    sounds_path=f"{data_dir}/val/noise",
                    min_snr_db=5.0,
                    max_snr_db=40.0,
                    p=p,
                ),
                JustNoise(
                    sounds_path=f"{data_dir}/val/noise",
                    min_snr_db=5.0,
                    max_snr_db=40.0,
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
            if t.__class__.__name__ not in ["Cut", "Shift", "JustNoise", "ToTensor"]:
                t.parameters["should_apply"] = False
        clean_waveform = self.transforms(
            samples=clean_waveform, sample_rate=self.sampling_rate
        )
        self.transforms.unfreeze_parameters()

        return noisy_waveform, clean_waveform, paths[1].stem[6:]


class VADNoiseDataModule(NoiseDataModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        length = self.hparams.length
        data_dir = self.hparams.data_dir
        p = self.hparams.p

        self.train_transforms = Compose(
            [
                Shift(min_shift=-0.5, max_shift=0.5, rollover=True, p=p),
                AddBackgroundNoise(
                    sounds_path=f"{data_dir}/train/noise",
                    min_snr_db=5.0,
                    max_snr_db=40.0,
                    p=p,
                ),
                Cut(length=length, is_val=False, p=1.0),
                JustNoise(
                    sounds_path=f"{data_dir}/train/noise",
                    min_snr_db=5.0,
                    max_snr_db=40.0,
                    p=p,
                ),
                PolarityInversion(p=p),
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
            [
                Shift(min_shift=-0.5, max_shift=0.5, rollover=True, p=p),
                AddBackgroundNoise(
                    sounds_path=f"{data_dir}/val/noise",
                    min_snr_db=5.0,
                    max_snr_db=40.0,
                    p=p,
                ),
                Cut(length=length, is_val=True, p=1.0),
                JustNoise(
                    sounds_path=f"{data_dir}/val/noise",
                    min_snr_db=5.0,
                    max_snr_db=40.0,
                    p=p,
                ),
                PolarityInversion(p=p),
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

    def get_files(self, data_dir: str | PathLike) -> list[tuple[Path, Path | None]]:
        data_dir = Path(data_dir)
        return list((data_dir / "clean").rglob("*.wav"))

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
            self.trainset = VADNoiseDataset(
                self.train_files,
                self.hparams.num_samples,
                self.hparams.sampling_rate,
                self.train_transforms,
            )
            self.valset = VADNoiseDataset(
                self.val_files, -1, self.hparams.sampling_rate, self.val_transforms
            )
        elif stage == "predict":
            self.valset = VADNoiseDataset(
                self.val_files, -1, self.hparams.sampling_rate, self.val_transforms
            )
        else:
            raise ValueError(f"Stage {stage} is not supported.")


class VADNoiseDataset(Dataset):
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
        path = self.files[index]
        clean_waveform, _ = librosa.load(path, sr=self.sampling_rate)
        noisy_waveform = self.transforms(
            samples=clean_waveform, sample_rate=self.sampling_rate
        )
        self.transforms.freeze_parameters()
        for t in self.transforms.transforms:
            if t.__class__.__name__ not in ["Cut", "Shift", "JustNoise", "ToTensor"]:
                t.parameters["should_apply"] = False
        clean_waveform = self.transforms(
            samples=clean_waveform, sample_rate=self.sampling_rate
        )
        self.transforms.unfreeze_parameters()
        return noisy_waveform, clean_waveform, path.stem[6:]
