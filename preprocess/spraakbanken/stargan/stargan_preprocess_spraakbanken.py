'''
Code taken directly from StarGAN-Voice-Conversion repo.

Modified specifically for Spraakbanken audio data.
By August Semrau Andersen for DTU course #02466.
'''


import librosa
import numpy as np
import os, sys
import argparse
import pyworld
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import *
from tqdm import tqdm
from collections import defaultdict
from collections import namedtuple
from sklearn.model_selection import train_test_split
import glob
from os.path import join, basename


def split_data(paths):
    indices = np.arange(len(paths))
    test_size = 0.05
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=1234)
    train_paths = list(np.array(paths)[train_indices])
    test_paths = list(np.array(paths)[test_indices])
    return train_paths, test_paths


def get_spk_world_feats(spk_fold_path, mc_dir_train, mc_dir_test, sample_rate=16000):
    paths = glob.glob(join(spk_fold_path, '*.wav'))
    spk_name = basename(spk_fold_path)
    train_paths, test_paths = split_data(paths)
    f0s = []
    coded_sps = []

    for wav_file in train_paths:
        f0, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        f0s.append(f0)
        coded_sps.append(coded_sp)

    log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
    coded_sps_mean, coded_sps_std = coded_sp_statistics(coded_sps)
    np.savez(join(mc_dir_train, spk_name + '_stats.npz'),
             log_f0s_mean=log_f0s_mean,
             log_f0s_std=log_f0s_std,
             coded_sps_mean=coded_sps_mean,
             coded_sps_std=coded_sps_std)

    for wav_file in tqdm(train_paths):
        wav_nam = spk_name + "_" + basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
        np.save(join(mc_dir_train, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)

    for wav_file in tqdm(test_paths):
        wav_nam = spk_name + "-" + basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
        np.save(join(mc_dir_test, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sample_rate_default = 16000

    # On August's Computer
    # target_wavpath_default = "../../../../speaker_data/Spraakbanken-Corpus/"
    # mc_dir_train_default = '../../../../preprocessed_data/stargan/spraakbanken/mc/train'
    # mc_dir_test_default = '../../../../preprocessed_data/stargan/spraakbanken/mc/test'

    # # On ssh filesystem
    target_wavpath_default = "/work1/s183921/speaker_data/Spraakbanken-Corpus"
    mc_dir_train_default = '/work1/s183921/preprocessed_data/stargan/spraakbanken/mc/train'
    mc_dir_test_default = '/work1/s183921/preprocessed_data/stargan/spraakbanken/mc/test'


    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate.")
    # parser.add_argument("--origin_wavpath", type=str, default=origin_wavpath_default,
    #                     help="The original wav path to resample.")
    parser.add_argument("--target_wavpath", type=str, default=target_wavpath_default,
                        help="The original wav path to resample.")
    parser.add_argument("--mc_dir_train", type=str, default=mc_dir_train_default,
                        help="The directory to store the training features.")
    parser.add_argument("--mc_dir_test", type=str, default=mc_dir_test_default,
                        help="The directory to store the testing features.")
    parser.add_argument("--num_workers", type=int, default=None, help="The number of cpus to use.")

    argv = parser.parse_args()

    sample_rate = argv.sample_rate
    target_wavpath = argv.target_wavpath
    mc_dir_train = argv.mc_dir_train
    mc_dir_test = argv.mc_dir_test
    num_workers = argv.num_workers if argv.num_workers is not None else cpu_count()


    # Defining speakers in the dataset for training StarGAN
    speaker_used = ["r5650072",  # Target speaker
                    "r5650060",
                    "r5650006",
                    "r5650013",
                    "r5650101",
                    "r5650044",
                    "r5650024",
                    "r5650085",
                    "r5650103",
                    "r5650082",
                    "r5650007",
                    "r5650080",
                    "r5650010",
                    "r5650105",
                    "r5650114",
                    "r5650111",
                    "r5650107",
                    "r5650109",
                    "r5650077",
                    "r5650090",
                    "r5650096",
                    "r5650095",
                    "r5650032",
                    "r5650055",
                    "r5650012"]



    ## Next we are to extract the acoustic features (MCEPs, lf0) and compute the corresponding stats (means, stds).
    # Make dirs to contain the MCEPs
    os.makedirs(mc_dir_train, exist_ok=True)
    os.makedirs(mc_dir_test, exist_ok=True)

    num_workers = len(speaker_used)  # cpu_count()
    print("number of workers: ", num_workers)
    executor = ProcessPoolExecutor(max_workers=num_workers)

    work_dir = target_wavpath

    futures = []
    for spk in speaker_used:
        spk_path = os.path.join(work_dir, spk)
        futures.append(executor.submit(partial(get_spk_world_feats, spk_path, mc_dir_train, mc_dir_test, sample_rate)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)
    sys.exit(0)
