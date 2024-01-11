import argparse
import csv
import glob
import os
import random
import shutil
import sys
import tarfile
import zipfile
import librosa
import pandas as pd
import requests
import sagemaker

import numpy as np
import configparser as CP

from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm
from random import shuffle
from audiolib import (
    audioread,
    audiowrite,
    segmental_snr_mixer,
    activitydetector,
    is_clipped,
)


MAXTRIES = 50
MAXFILELEN = 100

np.random.seed(5)
random.seed(5)


def get_dir(cfg, param_name, new_dir_name):
    """Helper function to retrieve directory name if it exists,
    create it if it doesn't exist"""

    if param_name in cfg:
        dir_name = cfg[param_name]
    else:
        dir_name = os.path.join(os.path.dirname(__file__), new_dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def write_log_file(log_dir, log_filename, data):
    """Helper function to write log file"""
    data = zip(*data)
    with open(os.path.join(log_dir, log_filename), mode="w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in data:
            csvwriter.writerow([row])


def str2bool(string):
    return string.lower() in ("yes", "true", "t", "1")


def rename_copyfile(src_path, dest_dir, prefix="", ext="*.wav"):
    srcfiles = glob.glob(f"{src_path}/" + ext)
    for i in range(len(srcfiles)):
        dest_path = os.path.join(dest_dir, prefix + "_" + os.path.basename(srcfiles[i]))
        shutil.copyfile(srcfiles[i], dest_path)


def add_pyreverb(clean_speech, rir):
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0 : clean_speech.shape[0]]

    return reverb_speech


def build_audio(is_clean, params, index, audio_samples_length=-1):
    """Construct an audio signal from source files"""

    fs_output = params["fs"]
    silence_length = params["silence_length"]
    if audio_samples_length == -1:
        audio_samples_length = int(params["audio_length"] * params["fs"])

    output_audio = np.zeros(0)
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    if is_clean:
        source_files = params["cleanfilenames"]
        idx = index
    else:
        if "noisefilenames" in params.keys():
            source_files = params["noisefilenames"]
            idx = index
        # if noise files are organized into individual subdirectories, pick a directory randomly
        else:
            noisedirs = params["noisedirs"]
            # pick a noise category randomly
            idx_n_dir = np.random.randint(0, np.size(noisedirs))
            source_files = list(
                map(
                    lambda f: str(f),
                    Path(noisedirs[idx_n_dir]).rglob(params["audioformat"]),
                )
            )

            random.shuffle(source_files)
            # pick a noise source file index randomly
            idx = np.random.randint(0, np.size(source_files))

    # initialize silence
    silence = np.zeros(int(fs_output * silence_length))

    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    while remaining_length > 0 and tries_left > 0:
        # read next audio file and resample if necessary

        idx = (idx + 1) % np.size(source_files)
        input_audio, fs_input = audioread(source_files[idx])
        if input_audio is None:
            sys.stderr.write("WARNING: Cannot read file: %s\n" % source_files[idx])
            continue
        if fs_input != fs_output:
            input_audio = librosa.resample(input_audio, fs_input, fs_output)

        # if current file is longer than remaining desired length, and this is
        # noise generation or this is training set, subsample it randomly
        if len(input_audio) > remaining_length and (
            not is_clean or not params["is_test_set"]
        ):
            idx_seg = np.random.randint(0, len(input_audio) - remaining_length)
            input_audio = input_audio[idx_seg : idx_seg + remaining_length]

        # check for clipping, and if found move onto next file
        if is_clipped(input_audio):
            clipped_files.append(source_files[idx])
            tries_left -= 1
            continue

        # concatenate current input audio to output audio stream
        files_used.append(source_files[idx])
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)

        # add some silence if we have not reached desired audio length
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio = np.append(output_audio, silence[:silence_len])
            remaining_length -= silence_len

    if tries_left == 0 and not is_clean and "noisedirs" in params.keys():
        print(
            "There are not enough non-clipped files in the "
            + noisedirs[idx_n_dir]
            + " directory to complete the audio build"
        )
        return [], [], clipped_files, idx

    return output_audio, files_used, clipped_files, idx


def gen_audio(is_clean, params, index, audio_samples_length=-1):
    """Calls build_audio() to get an audio signal, and verify that it meets the
    activity threshold"""

    clipped_files = []
    low_activity_files = []
    if audio_samples_length == -1:
        audio_samples_length = int(params["audio_length"] * params["fs"])
    if is_clean:
        activity_threshold = params["clean_activity_threshold"]
    else:
        activity_threshold = params["noise_activity_threshold"]

    while True:
        audio, source_files, new_clipped_files, index = build_audio(
            is_clean, params, index, audio_samples_length
        )

        clipped_files += new_clipped_files
        if len(audio) < audio_samples_length:
            continue

        if activity_threshold == 0.0:
            break

        percactive = activitydetector(audio=audio)
        if percactive > activity_threshold:
            break
        else:
            low_activity_files += source_files

    return audio, source_files, clipped_files, low_activity_files, index


def main_gen(params):
    """Calls gen_audio() to generate the audio signals, verifies that they meet
    the requirements, and writes the files to storage"""

    clean_source_files = []
    clean_clipped_files = []
    clean_low_activity_files = []
    noise_source_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    clean_index = 0
    noise_index = 0
    file_num = params["fileindex_start"]

    while file_num <= params["fileindex_end"]:
        # generate clean speech
        clean, clean_sf, clean_cf, clean_laf, clean_index = gen_audio(
            True, params, clean_index
        )

        noise, noise_sf, noise_cf, noise_laf, noise_index = gen_audio(
            False, params, noise_index, len(clean)
        )

        clean_clipped_files += clean_cf
        clean_low_activity_files += clean_laf
        noise_clipped_files += noise_cf
        noise_low_activity_files += noise_laf

        # mix clean speech and noise
        # if specified, use specified SNR value
        if not params["randomize_snr"]:
            snr = params["snr"]
        # use a randomly sampled SNR value between the specified bounds
        else:
            snr = np.random.randint(params["snr_lower"], params["snr_upper"])

        clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(
            params=params, clean=clean, noise=noise, snr=snr
        )
        # Uncomment the below lines if you need segmental SNR and comment the above lines using snr_mixer
        # clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(params=params,
        #                                                         clean=clean,
        #                                                          noise=noise,
        #                                                         snr=snr)
        # unexpected clipping
        if is_clipped(clean_snr) or is_clipped(noise_snr) or is_clipped(noisy_snr):
            print(
                "Warning: File #"
                + str(file_num)
                + " has unexpected clipping, "
                + "returning without writing audio to disk"
            )
            continue

        clean_source_files += clean_sf
        noise_source_files += noise_sf

        # write resultant audio streams to files
        hyphen = "-"
        clean_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in clean_sf]
        clean_files_joined = hyphen.join(clean_source_filenamesonly)[:MAXFILELEN]
        noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in noise_sf]
        noise_files_joined = hyphen.join(noise_source_filenamesonly)[:MAXFILELEN]

        noisyfilename = (
            clean_files_joined
            + "_"
            + noise_files_joined
            + "_snr"
            + str(snr)
            + "_tl"
            + str(target_level)
            + "_fileid_"
            + str(file_num)
            + ".wav"
        )
        cleanfilename = "clean_fileid_" + str(file_num) + ".wav"
        noisefilename = "noise_fileid_" + str(file_num) + ".wav"

        noisypath = os.path.join(params["noisyspeech_dir"], noisyfilename)
        cleanpath = os.path.join(params["clean_proc_dir"], cleanfilename)
        noisepath = os.path.join(params["noise_proc_dir"], noisefilename)

        audio_signals = [noisy_snr, clean_snr, noise_snr]
        file_paths = [noisypath, cleanpath, noisepath]

        for i in range(len(audio_signals)):
            try:
                audiowrite(file_paths[i], audio_signals[i], params["fs"])
                sess = sagemaker.Session(default_bucket=params["bucket"])
                sess.upload_data(
                    file_paths[i],
                    bucket=params["bucket"],
                    key_prefix=Path(file_paths[i]).parent,
                )
            except Exception as e:
                print(str(e))
        print("===>", "write resultant")

        file_num += 1

    return (
        clean_source_files,
        clean_clipped_files,
        clean_low_activity_files,
        noise_source_files,
        noise_clipped_files,
        noise_low_activity_files,
    )


def main_body():
    """Main body of this file"""

    parser = argparse.ArgumentParser()

    # Configurations: read noisyspeech_synthesizer.cfg and gather inputs
    parser.add_argument(
        "--cfg",
        default="noisyspeech_synthesizer.cfg",
        help="Read noisyspeech_synthesizer.cfg for all the details",
    )
    parser.add_argument("--cfg_str", type=str, default="noisy_speech")
    args = parser.parse_args()

    params = dict()
    params["args"] = args
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f"No configuration file as [{cfgpath}]"

    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    params["cfg"] = cfg._sections[args.cfg_str]
    cfg = params["cfg"]

    clean_dir = os.path.join(os.path.dirname(__file__), "datasets/clean")

    if cfg["speech_dir"] != "None":
        clean_dir = cfg["speech_dir"]
    if not os.path.exists(clean_dir):
        assert False, "Clean speech data is required"

    noise_dir = os.path.join(os.path.dirname(__file__), "datasets/noise")

    if cfg["noise_dir"] != "None":
        noise_dir = cfg["noise_dir"]
    if not os.path.exists:
        assert False, "Noise data is required"

    params["fs"] = int(cfg["sampling_rate"])
    params["audioformat"] = cfg["audioformat"]
    params["audio_length"] = float(cfg["audio_length"])
    params["silence_length"] = float(cfg["silence_length"])
    params["total_hours"] = float(cfg["total_hours"])

    # AWS configs:
    params["bucket"] = str(cfg["bucket"])

    # clean singing speech
    params["use_singing_data"] = int(cfg["use_singing_data"])
    params["clean_singing"] = str(cfg["clean_singing"])
    params["singing_choice"] = int(cfg["singing_choice"])

    # clean emotional speech
    params["use_emotion_data"] = int(cfg["use_emotion_data"])
    params["clean_emotion"] = str(cfg["clean_emotion"])

    # clean mandarin speech
    params["use_mandarin_data"] = int(cfg["use_mandarin_data"])
    params["clean_mandarin"] = str(cfg["clean_mandarin"])

    if cfg["fileindex_start"] != "None" and cfg["fileindex_end"] != "None":
        params["num_files"] = int(cfg["fileindex_end"]) - int(cfg["fileindex_start"])
        params["fileindex_start"] = int(cfg["fileindex_start"])
        params["fileindex_end"] = int(cfg["fileindex_end"])
    else:
        params["num_files"] = int(
            (params["total_hours"] * 60 * 60) / params["audio_length"]
        )
        params["fileindex_start"] = 0
        params["fileindex_end"] = params["num_files"]

    print("Number of files to be synthesized:", params["num_files"])

    params["is_test_set"] = str2bool(cfg["is_test_set"])
    params["clean_activity_threshold"] = float(cfg["clean_activity_threshold"])
    params["noise_activity_threshold"] = float(cfg["noise_activity_threshold"])
    params["snr_lower"] = int(cfg["snr_lower"])
    params["snr_upper"] = int(cfg["snr_upper"])

    params["randomize_snr"] = str2bool(cfg["randomize_snr"])
    params["target_level_lower"] = int(cfg["target_level_lower"])
    params["target_level_upper"] = int(cfg["target_level_upper"])

    if "snr" in cfg.keys():
        params["snr"] = int(cfg["snr"])
    else:
        params["snr"] = int((params["snr_lower"] + params["snr_upper"]) / 2)

    params["noisyspeech_dir"] = get_dir(cfg, "noisy_destination", "noisy")
    params["clean_proc_dir"] = get_dir(cfg, "clean_destination", "clean")
    params["noise_proc_dir"] = get_dir(cfg, "noise_destination", "noise")

    if "speech_csv" in cfg.keys() and cfg["speech_csv"] != "None":
        cleanfilenames = pd.read_csv(cfg["speech_csv"])
        cleanfilenames = cleanfilenames["filename"]
    else:
        # cleanfilenames = glob.glob(os.path.join(clean_dir, params['audioformat']))
        cleanfilenames = []
        for path in Path(clean_dir).rglob("*.wav"):
            cleanfilenames.append(str(path.resolve()))

    shuffle(cleanfilenames)
    #   add singing voice to clean speech
    if params["use_singing_data"] == 1:
        all_singing = []
        for path in Path(params["clean_singing"]).rglob("*.wav"):
            all_singing.append(str(path.resolve()))

        if params["singing_choice"] == 1:  # male speakers
            mysinging = [s for s in all_singing if ("male" in s and "female" not in s)]

        elif params["singing_choice"] == 2:  # female speakers
            mysinging = [s for s in all_singing if "female" in s]

        elif params["singing_choice"] == 3:  # both male and female
            mysinging = all_singing
        else:  # default both male and female
            mysinging = all_singing

        shuffle(mysinging)
        if mysinging is not None:
            all_cleanfiles = cleanfilenames + mysinging
    else:
        all_cleanfiles = cleanfilenames

    #   add emotion data to clean speech
    if params["use_emotion_data"] == 1:
        all_emotion = []
        for path in Path(params["clean_emotion"]).rglob("*.wav"):
            all_emotion.append(str(path.resolve()))

        shuffle(all_emotion)
        if all_emotion is not None:
            all_cleanfiles = all_cleanfiles + all_emotion
    else:
        print("NOT using emotion data for training!")

    #   add mandarin data to clean speech
    if params["use_mandarin_data"] == 1:
        all_mandarin = []
        for path in Path(params["clean_mandarin"]).rglob("*.wav"):
            all_mandarin.append(str(path.resolve()))

        shuffle(all_mandarin)
        if all_mandarin is not None:
            all_cleanfiles = all_cleanfiles + all_mandarin
    else:
        print("NOT using non-english (Mandarin) data for training!")

    params["cleanfilenames"] = all_cleanfiles
    params["num_cleanfiles"] = len(params["cleanfilenames"])
    # If there are .wav files in noise_dir directory, use those
    # If not, that implies that the noise files are organized into subdirectories by type,
    # so get the names of the non-excluded subdirectories
    if "noise_csv" in cfg.keys() and cfg["noise_csv"] != "None":
        noisefilenames = pd.read_csv(cfg["noise_csv"])
        noisefilenames = noisefilenames["filename"]
    else:
        noisefilenames = list(
            map(lambda f: str(f), Path(noise_dir).rglob(params["audioformat"]))
        )

    if len(noisefilenames) != 0:
        shuffle(noisefilenames)
        params["noisefilenames"] = noisefilenames
    else:
        noisedirs = list(map(lambda f: str(f), Path(noise_dir).rglob("*")))
        if cfg["noise_types_excluded"] != "None":
            dirstoexclude = cfg["noise_types_excluded"].split(",")
            for dirs in dirstoexclude:
                noisedirs.remove(dirs)
        shuffle(noisedirs)
        params["noisedirs"] = noisedirs

    # Call main_gen() to generate audio
    (
        clean_source_files,
        clean_clipped_files,
        clean_low_activity_files,
        noise_source_files,
        noise_clipped_files,
        noise_low_activity_files,
    ) = main_gen(params)

    # Create log directory if needed, and write log files of clipped and low activity files
    log_dir = get_dir(cfg, "log_dir", "Logs")

    write_log_file(log_dir, "source_files.csv", clean_source_files + noise_source_files)
    write_log_file(
        log_dir, "clipped_files.csv", clean_clipped_files + noise_clipped_files
    )
    write_log_file(
        log_dir,
        "low_activity_files.csv",
        clean_low_activity_files + noise_low_activity_files,
    )

    # Compute and print stats about percentange of clipped and low activity files
    total_clean = (
        len(clean_source_files)
        + len(clean_clipped_files)
        + len(clean_low_activity_files)
    )
    total_noise = (
        len(noise_source_files)
        + len(noise_clipped_files)
        + len(noise_low_activity_files)
    )
    pct_clean_clipped = round(len(clean_clipped_files) / total_clean * 100, 1)
    pct_noise_clipped = round(len(noise_clipped_files) / total_noise * 100, 1)
    pct_clean_low_activity = round(len(clean_low_activity_files) / total_clean * 100, 1)
    pct_noise_low_activity = round(len(noise_low_activity_files) / total_noise * 100, 1)

    print(
        "Of the "
        + str(total_clean)
        + " clean speech files analyzed, "
        + str(pct_clean_clipped)
        + "% had clipping, and "
        + str(pct_clean_low_activity)
        + "% had low activity "
        + "(below "
        + str(params["clean_activity_threshold"] * 100)
        + "% active percentage)"
    )
    print(
        "Of the "
        + str(total_noise)
        + " noise files analyzed, "
        + str(pct_noise_clipped)
        + "% had clipping, and "
        + str(pct_noise_low_activity)
        + "% had low activity "
        + "(below "
        + str(params["noise_activity_threshold"] * 100)
        + "% active percentage)"
    )


def download(name: str, category: str, parts: list[str], bucket: str):
    download_dir = Path("datasets/downloads") / name
    download_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("datasets") / category
    data_dir.mkdir(parents=True, exist_ok=True)
    ext = "zip" if "zip" in parts[0] else "tgz"

    sess = sagemaker.Session(default_bucket=bucket)

    for part in tqdm(parts, desc="Download file parts"):
        sess.download_data(download_dir, bucket, part)

    parts = sorted(list(download_dir.rglob(f"*")))
    with open(download_dir / f"{name}.{ext}", "wb") as file:
        for part in tqdm(parts, desc="Combine file parts"):
            with open(part, "rb") as part_file:
                shutil.copyfileobj(part_file, file)
            part.unlink()

    print("Extract file:")
    if ext == "zip":
        with zipfile.ZipFile(download_dir / f"{name}.{ext}", "r") as zip:
            zip.extractall(data_dir)
    else:
        with tarfile.open(download_dir / f"{name}.{ext}", "r") as tar:
            tar.extractall(data_dir)


if __name__ == "__main__":
    datasets = {
        "french_speech": {
            "category": "clean",
            "parts": [
                "headset-training/italian_speech.tar.gz.partaa",
                "headset-training/italian_speech.tar.gz.partab",
                "headset-training/italian_speech.tar.gz.partac",
                "headset-training/italian_speech.tar.gz.partad",
                "headset-training/italian_speech.tar.gz.partae",
                "headset-training/italian_speech.tar.gz.partah",
            ],
        },
        "german_speech": {
            "category": "clean",
            "parts": [
                "headset-training/german_speech.tgz.partaa",
                "headset-training/german_speech.tgz.partab",
                "headset-training/german_speech.tgz.partac",
                "headset-training/german_speech.tgz.partad",
                "headset-training/german_speech.tgz.partae",
                "headset-training/german_speech.tgz.partaf",
                "headset-training/german_speech.tgz.partag",
                "headset-training/german_speech.tgz.partah",
                "headset-training/german_speech.tgz.partaj",
                "headset-training/german_speech.tgz.partal",
                "headset-training/german_speech.tgz.partam",
                "headset-training/german_speech.tgz.partan",
                "headset-training/german_speech.tgz.partao",
                "headset-training/german_speech.tgz.partap",
                "headset-training/german_speech.tgz.partaq",
                "headset-training/german_speech.tgz.partar",
                "headset-training/german_speech.tgz.partas",
                "headset-training/german_speech.tgz.partat",
                "headset-training/german_speech.tgz.partau",
                "headset-training/german_speech.tgz.partav",
                "headset-training/german_speech.tgz.partaw",
            ],
        },
        "italian_speech": {
            "category": "clean",
            "parts": [
                "headset-training/italian_speech.tgz.partaa",
                "headset-training/italian_speech.tgz.partab",
                "headset-training/italian_speech.tgz.partac",
                "headset-training/italian_speech.tgz.partad",
            ],
        },
        "russian_speech": {
            "category": "clean",
            "parts": ["headset-training/russian_speech.tgz"],
        },
        "read_speech": {
            "category": "clean",
            "parts": [
                "headset-training/read_speech.tgz.partaa",
                "headset-training/read_speech.tgz.partab",
                "headset-training/read_speech.tgz.partac",
                "headset-training/read_speech.tgz.partad",
                "headset-training/read_speech.tgz.partae",
                "headset-training/read_speech.tgz.partaf",
                "headset-training/read_speech.tgz.partag",
                "headset-training/read_speech.tgz.partah",
                "headset-training/read_speech.tgz.partai",
                "headset-training/read_speech.tgz.partaj",
                "headset-training/read_speech.tgz.partak",
                "headset-training/read_speech.tgz.partal",
                "headset-training/read_speech.tgz.partam",
                "headset-training/read_speech.tgz.partan",
                "headset-training/read_speech.tgz.partao",
                "headset-training/read_speech.tgz.partap",
                "headset-training/read_speech.tgz.partaq",
                "headset-training/read_speech.tgz.partar",
                "headset-training/read_speech.tgz.partas",
                "headset-training/read_speech.tgz.partat",
                "headset-training/read_speech.tgz.partau",
            ],
        },
        "spanish_speech": {
            "category": "clean",
            "parts": [
                "headset-training/spanish_speech.tgz.partaa",
                "headset-training/spanish_speech.tgz.partab",
                "headset-training/spanish_speech.tgz.partac",
                "headset-training/spanish_speech.tgz.partad",
                "headset-training/spanish_speech.tgz.partae",
                "headset-training/spanish_speech.tgz.partaf",
                "headset-training/spanish_speech.tgz.partag",
            ],
        },
        "vctk_wav48_silence_trimmed": {
            "category": "clean",
            "parts": [
                "headset-training/vctk_wav48_silence_trimmed.tgz.partaa",
                "headset-training/vctk_wav48_silence_trimmed.tgz.partab",
                "headset-training/vctk_wav48_silence_trimmed.tgz.partac",
            ],
        },
        "noise_audioset": {
            "category": "noise",
            "parts": [
                "noise-ir/datasets_fullband.noise_fullband.audioset_000.tar.bz2",
                "noise-ir/datasets_fullband.noise_fullband.audioset_001.tar.bz2",
                "noise-ir/datasets_fullband.noise_fullband.audioset_002.tar.bz2",
                "noise-ir/datasets_fullband.noise_fullband.audioset_003.tar.bz2",
                "noise-ir/datasets_fullband.noise_fullband.audioset_004.tar.bz2",
                "noise-ir/datasets_fullband.noise_fullband.audioset_005.tar.bz2",
                "noise-ir/datasets_fullband.noise_fullband.audioset_006.tar.bz2",
            ],
        },
        "noise_freesound": {
            "category": "noise",
            "parts": [
                "noise-ir/datasets_fullband.noise_fullband.freesound_000.tar.bz2",
                "noise-ir/datasets_fullband.noise_fullband.freesound_001.tar.bz2",
            ],
        },
    }

    for name, value in datasets.items():
        download(name, value["category"], value["parts"], "db-noise")

    main_body()
