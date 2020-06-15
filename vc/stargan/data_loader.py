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

# Below is the accent info for the used 25 normal spraakbanken speakers.
# spk2acc = {"r5650072": "Copenhagen",  # Target speaker
#            "r5650060": "Copenhagen",
#            "r5650006": "Vestjylland",
#            "r5650013": "Vestjylland",
#            "r5650101": "Vestjylland",
#            "r5650044": "Vestjylland",
#            "r5650024": "VestSydjylland",
#            "r5650085": "VestSydjylland",
#            "r5650103": "VestSydjylland",
#            "r5650082": "VestSydjylland",
#            "r5650007": "Nordjylland",
#            "r5650080": "Nordjylland",
#            "r5650010": "Sonderjylland",
#            "r5650105": "Sonderjylland",
#            "r5650114": "Fyn",
#            "r5650111": "Fyn",
#            "r5650107": "Fyn",
#            "r5650109": "Fyn",
#            "r5650077": "VestSydsjaelland",
#            "r5650090": "VestSydsjaelland",
#            "r5650096": "VestSydsjaelland",
#            "r5650095": "VestSydsjaelland",
#            "r5650032": "Ostjylland",
#            "r5650055": "Ostjylland",
#            "r5650012": "Ostjylland"}

# Below is the accent info for the used 10 big spraakbanken speakers.
spk2acc = {'chunkxxx': 'Storkoebenhavn',
            'r6110050': 'Storkoebenhavn',
            'r6110013': 'Soenderjylland',
            'r6110015': 'Soenderjylland',
            'r6110034': 'Fyn',
            'r6110049': 'Vestjylland',
            'r6110009': 'Oestjylland',
            'r6110010': 'Nordjylland',
            'r6110032': 'VestSydSjaelland',
            'r6110044': 'VestSydSjaelland'}


# spk2acc = {'r6110005': 'Storkoebenhavn',
#             'r6110007': 'Storkoebenhavn',
#             'r6110008': 'Storkoebenhavn',
#             'r6110009': 'Storkoebenhavn',
#             'r6110010': 'Storkoebenhavn',
#             'r6110011': 'Storkoebenhavn',
#             'r6110012': 'Storkoebenhavn',
#             'r6110013': 'Storkoebenhavn',
#             'r6110014': 'Storkoebenhavn',
#             'r6110015': 'Storkoebenhavn',
#             'r6110018': 'Storkoebenhavn',
#             'r6110019': 'Storkoebenhavn',
#             'r6110020': 'Storkoebenhavn',
#             'r6110021': 'Storkoebenhavn',
#             'r6110022': 'Storkoebenhavn',
#             'r6110023': 'Storkoebenhavn',
#             'r6110024': 'Storkoebenhavn',
#             'r6110025': 'Storkoebenhavn',
#             'r6110026': 'Storkoebenhavn',
#             'r6110027': 'Storkoebenhavn',
#             'r6110028': 'Storkoebenhavn',
#             'r6110030': 'Storkoebenhavn',
#             'r6110031': 'Storkoebenhavn',
#             'r6110032': 'Storkoebenhavn',
#             'r6110033': 'Storkoebenhavn',
#             'r6110034': 'Storkoebenhavn',
#             'r6110035': 'Storkoebenhavn',
#             'r6110036': 'Storkoebenhavn',
#             'r6110037': 'Storkoebenhavn',
#             'r6110038': 'Storkoebenhavn',
#             'r6110039': 'Storkoebenhavn',
#             'r6110040': 'Storkoebenhavn',
#             'r6110041': 'Storkoebenhavn',
#             'r6110042': 'Storkoebenhavn',
#             'r6110043': 'Storkoebenhavn',
#             'r6110044': 'Storkoebenhavn',
#             'r6110046': 'Storkoebenhavn',
#             'r6110047': 'Storkoebenhavn',
#             'r6110048': 'Storkoebenhavn',
#             'r6110049': 'Storkoebenhavn',
#             'r6110050': 'Storkoebenhavn',
#             'r6110051': 'Storkoebenhavn'}

min_length = 256   # Since we slice 256 frames from each utterance when training.

# Build a dict useful when we want to get one-hot representation of speakers.
# 25 normal spraakbanken speakers
# speakers = ["r5650072",  # Target speaker
#             "r5650060",
#             "r5650006",
#             "r5650013",
#             "r5650101",
#             "r5650044",
#             "r5650024",
#             "r5650085",
#             "r5650103",
#             "r5650082",
#             "r5650007",
#             "r5650080",
#             "r5650010",
#             "r5650105",
#             "r5650114",
#             "r5650111",
#             "r5650107",
#             "r5650109",
#             "r5650077",
#             "r5650090",
#             "r5650096",
#             "r5650095",
#             "r5650032",
#             "r5650055",
#             "r5650012"]

# 10 big spraakbanken speakers
speakers = ['chunkxxx',  # Target
            'r6110050',  # Storkoebenhavn M t
            'r6110013',  # Soenderjylland F t
            'r6110015',  # Soenderjylland M t
            'r6110034',  # Fyn M t
            'r6110049',  # Vestjylland F t
            'r6110009',  # Oestjylland M t
            'r6110010',  # Nordjylland F t
            'r6110044',  # VestSydSjaelland M t
            'r6110032']  # VestSydSjaelland F t


# ALl spraakbanken-Test speakers
# speakers = [#'r6110005',  # Fyn F el.   # Oestjylland F
#             # 'r6110007',  # Soenderjylland M el.   # Nordjylland F
#             # 'r6110008',  # Vestjylland M el.   # Nordjylland M
#             # 'r6110009',  # Oestjylland M el.   # Nordjylland M
#             # 'r6110010',  # Nordjylland F el.   # Storkoebenhavn F
#             'r6110011',  # Nordjylland M
#             'r6110012',  # Storkoebenhavn M
#             # 'r6110013',  # Soenderjylland F el.   # Storkoebenhavn M
#             'r6110014',  # Storkoebenhavn M
#             # 'r6110015',  # Soenderjylland M el.   # Storkoebenhavn M
#             'r6110018',  # Fyn F
#             'r6110019',  # Fyn M
#             'r6110020',  # Storkoebenhavn F
#             'r6110021',  # Storkoebenhavn F
#             'r6110022',  # Vestjylland F
#             'r6110023',  # Oestjylland M
#             'r6110024',  # Storkoebenhavn F
#             'r6110025',  # Storkoebenhavn M
#             'r6110026',  # Vestjylland M
#             'r6110027',  # Storkoebenhavn M
#             'r6110028',  # VestSydSjaelland F
#             'r6110030',  # Storkoebenhavn M
#             'r6110031',  # Nordjylland F
#             'r6110032',  # VestSydSjaelland F
#             'r6110033',  # Storkoebenhavn F
#             'r6110034',  # Fyn M
#             'r6110035',  # Storkoebenhavn F
#             'r6110036',  # Storkoebenhavn M
#             'r6110037',  # VestSydSjaelland F
#             'r6110038',  # Storkoebenhavn F
#             'r6110039',  # Storkoebenhavn F
#             # 'r6110040',
#             'r6110041',  # Soenderjylland F
#             'r6110042',  # VestSydSjaelland M
#             'r6110043',  # Oestjylland F
#             'r6110044',  # VestSydSjaelland M
#             'r6110046',  # VestSydSjaelland M
#             # 'r6110047',  # Storkoebenhavn M el.   # Oestjylland M
#             'r6110048',  # Storkoebenhavn F
#             'r6110049',  # Vestjylland F el.   # Soenderjylland F
#             'r6110050',  # Target Storkoebenhavn M
#             'r6110051']  # Fyn F


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
        mc_files = [i for i in mc_files if basename(i)[:8] in speakers]  # Specified
        self.mc_files = self.rm_too_short_utt(mc_files)
        self.num_files = len(self.mc_files)
        print("\t Number of training samples: ", self.num_files)
        for f in self.mc_files:
            mc = np.load(f)
            if mc.shape[0] <= min_length:
                print(f)
                raise RuntimeError(f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!")

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
        spk = basename(filename).split('_')[0]
        spk_idx = spk2idx[spk]
        mc = np.load(filename)
        mc = self.sample_seg(mc)
        mc = np.transpose(mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape to one-hot
        spk_cat = np.squeeze(to_categorical([spk_idx], num_classes=len(speakers)))

        return torch.FloatTensor(mc), torch.LongTensor([spk_idx]).squeeze_(), torch.FloatTensor(spk_cat)


class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, data_dir, wav_dir, src_spk="r6110009", trg_spk="chunkxxx"):  # Specified
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
