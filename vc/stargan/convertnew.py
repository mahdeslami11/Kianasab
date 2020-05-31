'''
Code used from StarGAN-Voice-Conversion repo.
This script binds together pre-processing and converting.
Created by August Semrau Andersen for DTU course #02466.
'''

# Imports for preprocess.py
import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial
from tqdm import tqdm
# from preprocess import resample, resample_to_16k
import subprocess

# Imports for convert.py
import torch
from os.path import join, basename
import glob
from data_loader import to_categorical
from convert import load_wav
from model import Generator
from utils import *



def resample(spk, origin_wavpath, target_wavpath):
    wavfiles = [i for i in os.listdir(join(origin_wavpath, spk)) if i.endswith(".wav")]
    for wav in wavfiles:
        folder_to = join(target_wavpath, spk)
        os.makedirs(folder_to, exist_ok=True)
        wav_to = join(folder_to, wav)
        wav_from = join(origin_wavpath, spk, wav)
        subprocess.call(['sox', wav_from, "-r", "16000", wav_to])
    return 0


def resample_to_16k(origin_wavpath, target_wavpath, num_workers=1):
    os.makedirs(target_wavpath, exist_ok=True)
    spk_folders = os.listdir(origin_wavpath)
    print(f"> Using {num_workers} workers!")
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for spk in spk_folders:
        futures.append(executor.submit(partial(resample, spk, origin_wavpath, target_wavpath)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)

'''
Modified function get_spk_world_feats from preprocess.py
Changed so that _stats.npz file is saved also in the test directory.
'''
def get_spk_world_feats(spk_fold_path, mc_dir_test, sample_rate=16000):
    paths = glob.glob(join(spk_fold_path, '*.wav'))
    spk_name = basename(spk_fold_path)
    test_paths = paths
    f0s = []
    coded_sps = []
    # for wav_file in train_paths:
    for wav_file in test_paths:
        f0, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        f0s.append(f0)
        coded_sps.append(coded_sp)
    log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
    coded_sps_mean, coded_sps_std = coded_sp_statistics(coded_sps)
    np.savez(join(mc_dir_test, spk_name + '_stats.npz'),  # Changed mc_dir_train to mc_dir_test
             log_f0s_mean=log_f0s_mean,
             log_f0s_std=log_f0s_std,
             coded_sps_mean=coded_sps_mean,
             coded_sps_std=coded_sps_std)

    # for wav_file in tqdm(train_paths):
    #     wav_nam = basename(wav_file)
    #     f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
    #     normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
    #     np.save(join(mc_dir_train, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)

    for wav_file in tqdm(test_paths):
        wav_nam = basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
        np.save(join(mc_dir_test, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)
    return 0







'''
Modified class TestDataset and function test from convert.py
Changed so that _stats.npz file is loaded from test directory.
'''
class TestDataset(object):

    def __init__(self, config):
        # assert config.trg_spk in config.speakers, f'The trg_spk should be chosen from {config.speakers}, but you choose {trg_spk}.'
        # Source speaker
        self.src_spk = config.src_spk
        self.trg_spk = config.trg_spk
        self.mc_files = sorted(glob.glob(join(config.test_data_dir, f'{config.src_spk}*.npy')))
        print(self.mc_files)
        self.src_spk_stats = np.load(join(config.test_data_dir, f'{config.src_spk}_stats.npz'))  # Changed to test dir
        self.src_wav_dir = f'{config.wav_dir}/{config.src_spk}'

        self.trg_spk_stats = np.load(join(config.test_data_dir, f'{config.trg_spk}_stats.npz'))  # Changed to test dir

        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']

        # Define target speaker from trained speakers
        # self.speakers = config.speakers
        # spk2idx = dict(zip(self.speakers, range(len(self.speakers))))

        self.speakers = ["Stasjon01_210700_r5650072", ####
                        "Stasjon01_190700_r5650060",
                        "Stasjon01_030700_r5650006"]#,
                        # "Stasjon01_050700_r5650013",
                        # "Stasjon01_040800_r5650101",
                        # "Stasjon01_130700_r5650044",
                        # "Stasjon01_070700_r5650024",
                        # "Stasjon01_280700_r5650085",
                        # "Stasjon01_040800_r5650103",
                        # "Stasjon01_280700_r5650082",
                        # "Stasjon01_040700_r5650007",
                        # "Stasjon01_270700_r5650080",
                        # "Stasjon01_040700_r5650010",
                        # "Stasjon01_070800_r5650105",
                        # "Stasjon01_080800_r5650114",
                        # "Stasjon01_080800_r5650111",
                        # "Stasjon01_070800_r5650107",
                        # "Stasjon01_070800_r5650109",
                        # "Stasjon01_270700_r5650077",
                        # "Stasjon01_010800_r5650090",
                        # "Stasjon01_020800_r5650096",
                        # "Stasjon01_020800_r5650095",
                        # "Stasjon01_110700_r5650032",
                        # "Stasjon01_170700_r5650055",
                        # "Stasjon01_050700_r5650012"]
        spk2idx = dict(zip(self.speakers, range(len(self.speakers))))
        self.spk_idx = spk2idx[config.trg_spk]
        spk_cat = to_categorical([self.spk_idx], num_classes=len(self.speakers))
        self.spk_c_trg = spk_cat

    def get_batch_test_data(self, batch_size):
        batch_data = []
        print(self.mc_files)
        for i in range(batch_size):
            print(i)
            mcfile = self.mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data


'''
Modified function test from convert.py
'''
def test(config):
    os.makedirs(join(config.convert_dir, str(config.resume_iters)), exist_ok=True)
    sampling_rate, num_mcep, frame_period = 16000, 32, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator().to(device)
    test_loader = TestDataset(config)
    # Restore model
    print(f'Loading the trained models from step {config.resume_iters}...')
    G_path = join(config.model_save_dir, f'{config.resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    # Read a batch of testdata
    test_wavfiles = test_loader.get_batch_test_data(batch_size=config.num_converted_wavs)
    test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in test_wavfiles]

    with torch.no_grad():
        for idx, wav in enumerate(test_wavs):
            print(len(wav))
            wav_name = basename(test_wavfiles[idx])
            # print(wav_name)
            f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = pitch_conversion(f0=f0,
                                            mean_log_src=test_loader.logf0s_mean_src,
                                            std_log_src=test_loader.logf0s_std_src,
                                            mean_log_target=test_loader.logf0s_mean_trg,
                                            std_log_target=test_loader.logf0s_std_trg)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
            print("Before being fed into G: ", coded_sp.shape)

            coded_sp_norm = (coded_sp - test_loader.mcep_mean_src) / test_loader.mcep_std_src
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)
            spk_conds = torch.FloatTensor(test_loader.spk_c_trg).to(device)
            # print(spk_conds.size())
            coded_sp_converted_norm = G(coded_sp_norm_tensor, spk_conds).data.cpu().numpy()
            coded_sp_converted = np.squeeze(
                coded_sp_converted_norm).T * test_loader.mcep_std_trg + test_loader.mcep_mean_trg
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            print("After being fed into G: ", coded_sp_converted.shape)
            wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted,
                                                     ap=ap, fs=sampling_rate, frame_period=frame_period)
            wav_id = wav_name.split('.')[0]
            librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters),
                                          f'{wav_id}-vcto-{test_loader.trg_spk}.wav'), wav_transformed, sampling_rate)
            if [True, False][0]:
                wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp,
                                                   ap=ap, fs=sampling_rate, frame_period=frame_period)
                librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters), f'cpsyn-{wav_name}'),
                                         wav_cpsyn, sampling_rate)


'''
Combination of preprocess.py and convert.py
'''
if __name__ == '__main__':

    # On SSH
    # sample_rate_default = 16000
    # resume_iters_default = 100000
    # origin_wavpath_default = "/work1/s183921/newspeakers/wav48"
    # target_wavpath_default = "/work1/s183921/newspeakers/stargan/wav16"
    # # mc_dir_train_default = '/work1/s183921/newspeakers/stargan/mc'
    # mc_dir_test_default = '/work1/s183921/newspeakers/stargan/mc'
    # logs_dir_default = '/work1/s183921/newspeakers/stargan/logs'
    # models_dir_default = '/work1/s183921/trained_models/stargan/spraakbanken'
    # converted_dir_default = '/work1/s183921/converted_speakers/stargan'

    # On August's machine
    sample_rate_default = 16000
    resume_iters_default = 8000
    origin_wavpath_default = "../../../newspeakers/wav48"
    target_wavpath_default = "../../../newspeakers/stargan/wav16"
    # mc_dir_train_default = '../../../newspeakers/stargan/mc/'
    mc_dir_test_default = '../../../newspeakers/stargan/mc/'
    logs_dir_default = '../../../newspeakers/stargan/logs'
    models_dir_default = '../../../trained_models/stargan/spraakbanken'
    converted_dir_default = '../../../converted_speakers/stargan'

    # Parser takes inputs for running file as main
    parser = argparse.ArgumentParser()

    # Following allows for changes to preprocess.py
    parser.add_argument("--sample_rate", type=int, default=sample_rate_default, help="Sample rate.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of cpus to use.")
    # Following allows for changes to convert.py
    parser.add_argument('--resume_iters', type=int, default=resume_iters_default, help='step to resume for testing.')
    parser.add_argument('--num_speakers', type=int, default=None, help='dimension of speaker labels')
    # parser.add_argument('--num_converted_wavs', type=int, default=1, help='number of wavs to convert.')
    parser.add_argument('--src_spk', type=str, default=None, help="Source speakers.")
    parser.add_argument('--trg_spk', type=str, default='Stasjon01_210700_r5650072', help='Target speaker (FIXED).')
    parser.add_argument("--speakers", type=str, default=None)  # This is used for TestDataset class

    # Directories of preprocess.py and convert.py
    parser.add_argument("--origin_wavpath", type=str, default=origin_wavpath_default, help="48 kHz wav path.")
    parser.add_argument("--target_wavpath", type=str, default=target_wavpath_default, help="16 kHz wav path.")
    # parser.add_argument("--mc_dir_train", type=str, default=mc_dir_train_default, help="Dir for training features.")
    parser.add_argument("--mc_dir_test", type=str, default=mc_dir_test_default, help="Dir for testing features.")
    # parser.add_argument('--train_data_dir', type=str, default=mc_dir_train_default)
    parser.add_argument('--test_data_dir', type=str, default=mc_dir_test_default)
    parser.add_argument('--wav_dir', type=str, default=target_wavpath_default)
    parser.add_argument('--log_dir', type=str, default=logs_dir_default)
    parser.add_argument('--model_save_dir', type=str, default=models_dir_default)
    parser.add_argument('--convert_dir', type=str, default=converted_dir_default)

    # Parse arguments
    argv = parser.parse_args()

    # Redefine paths in case parsed arguments differ from default
    sample_rate = argv.sample_rate
    origin_wavpath = argv.origin_wavpath
    target_wavpath = argv.target_wavpath
    # mc_dir_train = argv.mc_dir_train
    mc_dir_test = argv.mc_dir_test
    logs_dir_default = argv.log_dir
    models_dir_default = argv.model_save_dir
    converted_dir_default = argv.convert_dir

    # Set num_workers to number og cpus unless specified
    num_workers = argv.num_workers if argv.num_workers is not None else cpu_count()

    '''Usually below statement is the case, but we define default resume-iteration.'''
    # If no model-iteration has been specified, don't run
    if argv.resume_iters is None:
        raise RuntimeError("Please specify the step number for resuming.")

    # Here it is specified which speakers should be converted
    speaker_used = argv.src_spk if argv.src_spk is not None else None

    if speaker_used is None:
        speaker_used = os.listdir(origin_wavpath)
        print(speaker_used)
        argv.src_spk = speaker_used




    # if speaker_used is not None:
    #     speaker_used = speaker_used.split('+')  # Make list of speakers
    #     testDataset_speaker_used = [None] * (len(speaker_used) + 1)
    #     target = str(argv.trg_spk)
    #     testDataset_speaker_used[:-1] = speaker_used
    #     testDataset_speaker_used[-1] = target
    #     argv.speakers = testDataset_speaker_used

    # If no speakers are specified, make it clear that nothing will be converted
    if speaker_used == []:
        raise RuntimeError("No speakers available in wav48 dir - No conversion will take place")


    # Setting number of speakers
    # if not argv.num_speakers:
    argv.num_speakers = len(speaker_used)

    # If the original wav is 48K, first we want to resample to 16K
    resample_to_16k(origin_wavpath, target_wavpath, num_workers=num_workers)

    ## Next extract the acoustic features (MCEPs, lf0) and compute the corresponding stats (means, stds).
    # Make dirs to contain the MCEPs
    # os.makedirs(mc_dir_train, exist_ok=True)
    os.makedirs(mc_dir_test, exist_ok=True)

    num_workers = len(speaker_used)  # cpu_count()
    print("number of workers: ", num_workers)
    executor = ProcessPoolExecutor(max_workers=num_workers)

    work_dir = target_wavpath


    futures = []
    for spk in speaker_used:
        spk_mc_dir_test = mc_dir_test
        spk_path = os.path.join(work_dir, spk)
        # print(spk_path)
        # Do processing
        futures.append(executor.submit(partial(get_spk_world_feats, spk_path, spk_mc_dir_test, sample_rate)))
        # futures.append(executor.submit(partial(get_spk_world_feats, spk_path, mc_dir_train, spk_mc_dir_test, sample_rate)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)
    # raise RuntimeError("Debugging")
    # sys.exit(0)

    '''
    END OF PREPROCESS.PY

    ONTO CONVERT.PY
    '''

    print(argv)

    # If only one speaker should be converted
    if len(speaker_used) == 1:
        argv.src_spk = speaker_used[0]
        argv.num_converted_wavs = len(glob.glob(join(argv.test_data_dir, f'{speaker_used[0]}*.npy')))
        test(argv)

    # If more than one speaker should be converted, test runs that number of times
    else:
        for speaker_to_convert in speaker_used:
            argv.src_spk = speaker_to_convert  # Redifine for only one speaker

            # Redefine number og wavs to convert to suit each speakers count
            argv.num_converted_wavs = len(glob.glob(join(argv.test_data_dir, f'{speaker_to_convert}*.npy')))
            test(argv)





