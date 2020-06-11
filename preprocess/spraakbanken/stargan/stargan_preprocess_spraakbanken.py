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
        wav_nam = basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
        np.save(join(mc_dir_train, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)

    for wav_file in tqdm(test_paths):
        wav_nam = basename(wav_file)
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
    target_wavpath_default = "/work1/s183921/speaker_data/Spraakbanken-Corpus-Test"
    mc_dir_train_default = '/work1/s183921/preprocessed_data/stargan/spraakbanken/mc-Test-All-1/train'
    mc_dir_test_default = '/work1/s183921/preprocessed_data/stargan/spraakbanken/mc-Test-All-1/test'


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
    # 25 normal spraakbanken
    # speaker_used = ["r5650072",  # Target speaker
    #                 "r5650060",
    #                 "r5650006",
    #                 "r5650013",
    #                 "r5650101",
    #                 "r5650044",
    #                 "r5650024",
    #                 "r5650085",
    #                 "r5650103",
    #                 "r5650082",
    #                 "r5650007",
    #                 "r5650080",
    #                 "r5650010",
    #                 "r5650105",
    #                 "r5650114",
    #                 "r5650111",
    #                 "r5650107",
    #                 "r5650109",
    #                 "r5650077",
    #                 "r5650090",
    #                 "r5650096",
    #                 "r5650095",
    #                 "r5650032",
    #                 "r5650055",
    #                 "r5650012"]

    # 10 big spraakbanken speakers
    # speaker_used = ['r6110050',  # Target Storkoebenhavn M
    #             # 'r6110048',  # Storkoebenhavn F
    #             'r6110013',  # Soenderjylland F
    #             'r6110015',  # Soenderjylland M
    #             'r6610005',  # Fyn F
    #             'r6110034',  # Fyn M
    #             'r6110049',  # Vestjylland F
    #             # 'r6110008',  # Vestjylland M
    #             # 'r6110043',  # Oestjylland F
    #             'r6110009',  # Oestjylland M
    #             'r6110010',  # Nordjylland F
    #             # 'r6110011',  # Nordjylland M
    #             'r6110032',  # VestSydSjaelland F
    #             'r6110044']  # VestSydSjaelland M

    # All big spraakbanken speakers len 42
    speaker_used = [#'r6110005',
    #             'r6110007',
    #             'r6110008',
    #             'r6110009',
    #             'r6110010',
    #             'r6110011',
    #             'r6110012',
    #             'r6110013',
    #             'r6110014',
    #             'r6110015',
    #             'r6110018',
    #             'r6110019',
    #             'r6110020',
    #             'r6110021',
    #             'r6110022',
    #             'r6110023',
    #             'r6110024',
    #             'r6110025',
    #             'r6110026',
    #             'r6110027',
    #             'r6110028',
                'r6110030',
                'r6110031',
                'r6110032',
                'r6110033',
                'r6110034',
                'r6110035',
                'r6110036',
                'r6110037',
                'r6110038',
                'r6110039',
                'r6110040',
                'r6110041',
                'r6110042',
                'r6110043',
                'r6110044',
                'r6110046',
                'r6110047',
                'r6110048',
                'r6110049',
                'r6110050',
                'r6110051']


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
