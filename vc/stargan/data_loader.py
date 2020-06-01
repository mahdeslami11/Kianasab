'''
Code taken directly from StarGAN-Voice-Conversion repo.
Modified by August Semrau Andersen for DTU course #02466.
'''

from torch.utils import data
import torch
import os
import random
import glob
from os.path import join, basename, dirname, split
import numpy as np

# Below is the accent info for the used 10 speakers.
spk2acc = {"Stasjon01_210700_r5650072":"Copenhagen",  # Target speaker
            "Stasjon01_190700_r5650060":"Copenhagen",
            "Stasjon01_030700_r5650006":"Vestjylland",
            "Stasjon01_050700_r5650013":"Vestjylland",
            "Stasjon01_040800_r5650101":"Vestjylland",
            "Stasjon01_130700_r5650044":"Vestjylland",
            "Stasjon01_070700_r5650024":"VestSydjylland",
            "Stasjon01_280700_r5650085":"VestSydjylland",
            "Stasjon01_040800_r5650103":"VestSydjylland",
            "Stasjon01_280700_r5650082":"VestSydjylland",
            "Stasjon01_040700_r5650007":"Nordjylland",
            "Stasjon01_270700_r5650080":"Nordjylland",
            "Stasjon01_040700_r5650010":"Sonderjylland",
            "Stasjon01_070800_r5650105":"Sonderjylland",
            "Stasjon01_080800_r5650114":"Fyn",
            "Stasjon01_080800_r5650111":"Fyn",
            "Stasjon01_070800_r5650107":"Fyn",
            "Stasjon01_070800_r5650109":"Fyn",
            "Stasjon01_270700_r5650077":"VestSydsjaelland",
            "Stasjon01_010800_r5650090":"VestSydsjaelland",
            "Stasjon01_020800_r5650096":"VestSydsjaelland",
            "Stasjon01_020800_r5650095":"VestSydsjaelland",
            "Stasjon01_110700_r5650032":"Ostjylland",
            "Stasjon01_170700_r5650055":"Ostjylland",
            "Stasjon01_050700_r5650012":"Ostjylland"}
min_length = 256   # Since we slice 256 frames from each utterance when training.
# Build a dict useful when we want to get one-hot representation of speakers.
speakers = ["Stasjon01_210700_r5650072",  # Target speaker
            "Stasjon01_190700_r5650060",
            "Stasjon01_030700_r5650006",
            "Stasjon01_050700_r5650013",
            "Stasjon01_040800_r5650101",
            "Stasjon01_130700_r5650044",
            "Stasjon01_070700_r5650024",
            "Stasjon01_280700_r5650085",
            "Stasjon01_040800_r5650103",
            "Stasjon01_280700_r5650082",
            "Stasjon01_040700_r5650007",
            "Stasjon01_270700_r5650080",
            "Stasjon01_040700_r5650010",
            "Stasjon01_070800_r5650105",
            "Stasjon01_080800_r5650114",
            "Stasjon01_080800_r5650111",
            "Stasjon01_070800_r5650107",
            "Stasjon01_070800_r5650109",
            "Stasjon01_270700_r5650077",
            "Stasjon01_010800_r5650090",
            "Stasjon01_020800_r5650096",
            "Stasjon01_020800_r5650095",
            "Stasjon01_110700_r5650032",
            "Stasjon01_170700_r5650055",
            "Stasjon01_050700_r5650012"]
spk2idx = dict(zip(speakers, range(len(speakers))))

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class MyDataset(data.Dataset):
    """Dataset for MCEP features and speaker labels."""
    def __init__(self, data_dir):
        mc_files = glob.glob(join(data_dir, '*.npy'))
        #mc_files = [i for i in mc_files if basename(i)[:25] in speakers]
        self.mc_files = self.rm_too_short_utt(mc_files)
        self.num_files = len(self.mc_files)
        print("\t Number of training samples: ", self.num_files)
        for f in self.mc_files:
            mc = np.load(f)
            if mc.shape[0] <= min_length:
                print(f)
                raise RuntimeError \
                    (f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!")

    def rm_too_short_utt(self, mc_files, min_length=min_length):
        new_mc_files = []
        for mcfile in mc_files:
            mc = np.load(mcfile)
            if mc.shape[0] > min_length:
                new_mc_files.append(mcfile)
        return new_mc_files

    def sample_seg(self, feat, sample_len=min_length):
        assert feat.shape[0] - sample_len >= 0
        s = np.random.randint(0, feat.shape[0] - sample_len + 1)
        return feat[s: s +sample_len, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        filename = self.mc_files[index]
        spk = basename(filename).split('-')[0]
        spk_idx = spk2idx[spk]
        mc = np.load(filename)
        mc = self.sample_seg(mc)
        mc = np.transpose(mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape to one-hot
        spk_cat = np.squeeze(to_categorical([spk_idx], num_classes=len(speakers)))

        return torch.FloatTensor(mc), torch.LongTensor([spk_idx]).squeeze_(), torch.FloatTensor(spk_cat)


class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, data_dir, wav_dir, src_spk="Stasjon01_070800_r5650109", trg_spk="Stasjon01_210700_r5650072"):
        self.src_spk = src_spk
        self.trg_spk = trg_spk
        self.mc_files = sorted(glob.glob(join(data_dir, '{}*.npy'.format(self.src_spk))))

        self.src_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(src_spk)))
        self.trg_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(trg_spk)))

        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']
        self.src_wav_dir = f'{wav_dir}/{src_spk}'
        self.spk_idx = spk2idx[trg_spk]
        spk_cat = to_categorical([self.spk_idx], num_classes=len(speakers))
        self.spk_c_trg = spk_cat

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        print(batch_size)
        print(len(self.mc_files))
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data

def get_loader(data_dir, batch_size=32, mode='train', num_workers=1):
    print(data_dir)
    dataset = MyDataset(data_dir)
    print(dataset)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader


if __name__ == '__main__':
    loader = get_loader('./data/mc/train')
    data_iter = iter(loader)
    for i in range(10):
        mc, spk_idx, acc_idx, spk_acc_cat = next(data_iter)
        print('- ' *50)
        print(mc.size())
        print(spk_idx.size())
        print(acc_idx.size())
        print(spk_acc_cat.size())
        print(spk_idx.squeeze_())
        print(spk_acc_cat)
        print('- ' *50)
