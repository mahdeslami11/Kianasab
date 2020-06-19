'''
Code used from StarGAN-Voice-Conversion repo.
This script binds together pre-processing and converting.
Created by August Semrau Andersen for DTU course #02466.
'''

# Imports for preprocessing
import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial
from tqdm import tqdm
import subprocess

# Imports for convert.py
import torch
from os.path import join, basename
import glob
from data_loader import to_categorical
# from convert import load_wav
from model import Generator
from utils import *



'''
Modified class TestDataset and function test from convert.py
Changed so that _stats.npz file is loaded from test directory.
'''
class TestDataset(object):

    def __init__(self, config):

        self.src_spk = config.src_spk  # Source speaker changes for every folder
        self.trg_spk = config.trg_spk  # Target speaker is fixed to r6110050
        self.mc_files = sorted(glob.glob(join(config.test_data_dir, f'{config.src_spk}*.npy')))
        self.src_spk_stats = np.load(join(config.test_data_dir, f'{config.src_spk}_stats.npz'))
        self.src_wav_dir = f'{config.wav_dir}/{config.src_spk}'

        # Loads r6110050_stats.npz from mc folder used for training
        self.trg_spk_stats = np.load(join('/work1/s183921/preprocessed_data/stargan/spraakbanken/mc-Test-All-1/train', f'{config.trg_spk}_stats.npz'))
        # self.trg_spk_stats = np.load(join('../../../preprocessed_data/stargan/spraakbanken/mc-Test/train', f'{config.trg_spk}_stats.npz'))  # Changed to test dir



        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']



        # fyn: 4, Oestjylland: 4, Nordjylland: 2, Vestjylland: 3, VestSydSjaelland: 5, Soenderjylland: 2, kbh: 5
        # 12 Female, 13 Male
        # ALl spraakbanken-Test speakers
        self.speakers = ['r6110005',  # Oestjylland F
                    'r6110008',  # Vestjylland M
                    'r6110009',  # Oestjylland M
                    'r6110011',  # Nordjylland M
                    'r6110018',  # Fyn F
                    'r6110019',  # Fyn M
                    'r6110022',  # Vestjylland F
                    'r6110023',  # Oestjylland M
                    'r6110026',  # Vestjylland M
                    'r6110027',  # Storkoebenhavn M
                    'r6110028',  # VestSydSjaelland F
                    'r6110030',  # Storkoebenhavn M
                    'r6110031',  # Nordjylland F
                    'r6110034',  # Fyn M
                    'r6110037',  # VestSydSjaelland F
                    'r6110038',  # Storkoebenhavn F
                    'r6110041',  # Soenderjylland F
                    'r6110042',  # VestSydSjaelland M
                    'r6110043',  # Oestjylland F
                    'r6110044',  # VestSydSjaelland M
                    'r6110046',  # VestSydSjaelland M
                    'r6110048',  # Storkoebenhavn F
                    'r6110049',  # Soenderjylland F
                    'r6110050',  # Target Storkoebenhavn M
                    'r6110051']  # Fyn F


        assert self.trg_spk in self.speakers, f'The trg_spk should be chosen from {self.speakers}, but you choose {self.trg_spk}.'
        spk2idx = dict(zip(self.speakers, range(len(self.speakers))))
        self.spk_idx = spk2idx[config.trg_spk]
        # print(self.spk_idx)
        spk_cat = to_categorical([self.spk_idx], num_classes=len(self.speakers))
        self.spk_c_trg = spk_cat
        # print(self.spk_c_trg)

    def get_batch_test_data(self, batch_size):
        batch_data = []
        # print(self.mc_files)
        for i in range(batch_size):
            # print(i)
            mcfile = self.mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            # print(filename)
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            # print(wavfile_path)
            batch_data.append(wavfile_path)
        return batch_data

def load_wav(wavfile, sr=16000):
    wav, _ = librosa.load(wavfile, sr=sr, mono=True)
    return wav_padding(wav, sr=sr, frame_period=5, multiple=4)  # TODO
    # return wav


'''
Modified function test from convert.py
'''
def test(config):
    # Make individual conversion folders for each speaker
    spk_converted_dir = os.path.join(config.convert_dir, config.src_spk)
    os.makedirs(spk_converted_dir, exist_ok=True)

    sampling_rate, num_mcep, frame_period = 16000, 36, 5
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    G = Generator(num_speakers=25).to(device)  # Now we shouldn't have to change num_speakers in model.py
    test_loader = TestDataset(config)
    # Restore model
    # print(f'Loading the trained models from step {config.resume_iters}...')
    G_path = join(config.model_save_dir, f'{config.resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    # print(G)
    # Read a batch of testdata
    test_wavfiles = test_loader.get_batch_test_data(batch_size=config.num_converted_wavs)
    test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in test_wavfiles]

    with torch.no_grad():
        for idx, wav in enumerate(test_wavs):
            # print(len(wav))
            wav_name = basename(test_wavfiles[idx])
            # print(wav_name)
            f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = pitch_conversion(f0=f0,
                                            mean_log_src=test_loader.logf0s_mean_src,
                                            std_log_src=test_loader.logf0s_std_src,
                                            mean_log_target=test_loader.logf0s_mean_trg,
                                            std_log_target=test_loader.logf0s_std_trg)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
            # print("Before being fed into G: ", coded_sp.shape)

            coded_sp_norm = (coded_sp - test_loader.mcep_mean_src) / test_loader.mcep_std_src
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)
            spk_conds = torch.FloatTensor(test_loader.spk_c_trg).to(device)
            # print(spk_conds.size())
            coded_sp_converted_norm = G(coded_sp_norm_tensor, spk_conds).data.cpu().numpy()
            coded_sp_converted = np.squeeze(
                coded_sp_converted_norm).T * test_loader.mcep_std_trg + test_loader.mcep_mean_trg
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            # print("After being fed into G: ", coded_sp_converted.shape)
            wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted,
                                                     ap=ap, fs=sampling_rate, frame_period=frame_period)
            wav_id = wav_name.split('.')[0]
            # librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters), f'{wav_id}-vcto-{test_loader.trg_spk}.wav'), wav_transformed, sampling_rate)
            librosa.output.write_wav(join(spk_converted_dir, wav_name), wav_transformed, sampling_rate)

            # if [True, False][0]:
            #     wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp,
            #                                        ap=ap, fs=sampling_rate, frame_period=frame_period)
            #     librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters), f'cpsyn-{wav_name}'),
            #                              wav_cpsyn, sampling_rate)


'''
Combination of original preprocess.py and original convert.py
'''
if __name__ == '__main__':

    # On SSH
    sample_rate_default = 16000
    resume_iters_default = 200000
    target_wavpath_default = "/work1/s183921/speaker_data/Spraakbanken-Selected"
    mc_dir_test_default = '/work1/s183921/preprocessed_data/stargan/spraakbanken/mc-retrain'
    logs_dir_default = '/work1/s183921/newspeakers/stargan/logs'
    models_dir_default = '/work1/s183921/trained_models/stargan/spraakbanken-Test-25-Final'
    converted_dir_default = '/work1/s183921/converted_speakers/stargan/Spraakbanken-Selected'

    # On August's machine
    # sample_rate_default = 16000
    # resume_iters_default = 200000
    # target_wavpath_default = "../../../speaker_data/Spraakbanken-Selected"
    # mc_dir_test_default = '../../../preprocessed_data/stargan/spraakbanken/mc-retrain'
    # logs_dir_default = '../../../newspeakers/stargan/logs'
    # models_dir_default = '../../../trained_models/stargan/spraakbanken-Test-25-Final'
    # converted_dir_default = '../../../converted_speakers/stargan/Spraakbanken-Selected'


    # Parser takes inputs for running file as main
    parser = argparse.ArgumentParser()

    # Following allows for changes to preprocessing step
    parser.add_argument("--sample_rate", type=int, default=sample_rate_default, help="Sample rate.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of cpus to use.")
    # Following allows for changes to convert.py
    parser.add_argument('--resume_iters', type=int, default=resume_iters_default, help='step to resume for testing.')
    parser.add_argument('--num_speakers', type=int, default=None, help='dimension of speaker labels')
    # parser.add_argument('--num_converted_wavs', type=int, default=1, help='number of wavs to convert.')
    parser.add_argument('--src_spk', type=str, default=None, help="Source speakers.")
    parser.add_argument('--trg_spk', type=str, default="r6110050", help='Target speaker (FIXED).')  # Chunk trg speaker
    parser.add_argument("--speakers", type=str, default=None)  # This is used for TestDataset class

    # For running multiple iterations
    parser.add_argument("--index", type=int, default=None)  # This is used for TestDataset class


    # Directories of preprocessing and converting
    parser.add_argument("--target_wavpath", type=str, default=target_wavpath_default, help="16 kHz wav path.")
    parser.add_argument("--mc_dir_test", type=str, default=mc_dir_test_default, help="Dir for testing features.")
    parser.add_argument('--test_data_dir', type=str, default=mc_dir_test_default)
    parser.add_argument('--wav_dir', type=str, default=target_wavpath_default)
    parser.add_argument('--log_dir', type=str, default=logs_dir_default)
    parser.add_argument('--model_save_dir', type=str, default=models_dir_default)
    parser.add_argument('--convert_dir', type=str, default=converted_dir_default)

    # Parse arguments
    argv = parser.parse_args()

    # Redefine paths in case parsed arguments differ from default
    sample_rate = argv.sample_rate
    target_wavpath = argv.target_wavpath
    mc_dir_test = argv.mc_dir_test
    logs_dir_default = argv.log_dir
    models_dir_default = argv.model_save_dir
    converted_dir_default = argv.convert_dir

    # Set num_workers to number og cpus unless specified
    num_workers = argv.num_workers if argv.num_workers is not None else cpu_count()



    # Here the interval of speakers to be preprocessed is specified
    speaker_used = os.listdir(target_wavpath)  # First define speaker_used as all speakers in corpus
    # print(speaker_used)
    index = argv.index
    # print(index)
    step_size = 20
    if index is not None:
        if index <= len(speaker_used):
            speaker_used = speaker_used[index - step_size:index]
        elif index - step_size < len(speaker_used) and index > len(speaker_used):
            speaker_used = speaker_used[index - step_size:]
        else:
            raise RuntimeError("No more speakers")



    # print(speaker_used)
    argv.src_spk = speaker_used


    # Setting number of speakerst
    argv.num_speakers = len(speaker_used)



    # ## Next extract the acoustic features (MCEPs, lf0) and compute the corresponding stats (means, stds).
    # # Make dirs to contain the MCEPs
    # os.makedirs(mc_dir_test, exist_ok=True)

    # num_workers = len(speaker_used)  # cpu_count()
    # print("number of workers: ", num_workers)
    executor = ProcessPoolExecutor(max_workers=num_workers)

    work_dir = target_wavpath


    # futures = []
    # for spk in speaker_used:
    #     spk_mc_dir_test = mc_dir_test
    #     spk_path = os.path.join(work_dir, spk)
    #     # Do processing
    #     futures.append(executor.submit(partial(get_spk_world_feats, spk_path, spk_mc_dir_test, sample_rate)))
    #     # futures.append(executor.submit(partial(get_spk_world_feats, spk_path, mc_dir_train, spk_mc_dir_test, sample_rate)))
    # result_list = [future.result() for future in tqdm(futures)]
    # print(result_list)

    '''
    END OF PREPROCESS.PY

    ONTO CONVERT.PY
    '''

    # print(argv)

    # If only one speaker should be converted
    if len(speaker_used) == 1:

        argv.src_spk = speaker_used[0]
        argv.test_data_dir = os.path.join(mc_dir_test, speaker_used[0])
        argv.num_converted_wavs = len(glob.glob(join(argv.test_data_dir, f'{speaker_used[0]}*.npy')))
        test(argv)

    # If more than one speaker should be converted, test runs that number of times
    else:
        for speaker_to_convert in speaker_used:
            argv.src_spk = speaker_to_convert  # Redifine for only one speaker
            argv.test_data_dir = os.path.join(mc_dir_test, speaker_to_convert)
            # Redefine number og wavs to convert to suit each speakers count
            argv.num_converted_wavs = len(glob.glob(join(argv.test_data_dir, f'{speaker_to_convert}*.npy')))
            test(argv)





