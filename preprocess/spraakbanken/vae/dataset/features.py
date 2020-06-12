from tacotron.utils import get_spectrograms, spec_feature_extraction
import numpy as np
import os
from tinydb import TinyDB, Query

def __get_spectrograms(path_list,n_utts_attr, db:TinyDB):
    all_train_data = []
    for i, path in enumerate(sorted(path_list)):
        if i % 500 == 0 or i == len(path_list) - 1:
            print(f'processing {i} files')
        filename = path.strip().split('/')[-1]
        mel, mag = get_spectrograms(path)
        db.insert({'key': filename, 'val': mel.tolist()})
        if dset == 'train' and i < n_utts_attr:
            train.append(mel)

    return all_train_data

######Feature extraction, mean and variance vectors, saved as pickle
def extract(train_path_list, in_test_path_list, out_test_path_list, 
        n_utts_attr, out_dir):

    for dset, path_list in zip(['train', 'in_test', 'out_test'], \
            [train_path_list, in_test_path_list, out_test_path_list]):

        db = TinyDB(os.join(out_dir, f'{dset}.json'))

        print(f'processing {dset} set, {len(path_list)} files')
        all_train_data = __get_spectrograms(path_list, n_utts_attr, db)
        #Extrating mean and standard deviation for training data and saves it in .pkl
        if dset == 'train':
            all_train_data = np.concatenate(all_train_data)
            mean = np.mean(all_train_data, axis=0)
            std = np.std(all_train_data, axis=0)
            attr = {'mean': mean, 'std': std}
            with open(os.path.join(output_dir, 'attr.pkl'), 'wb+') as f:
                pickle.dump(attr, f)
        #Normalizing mel spectrogram data
        Q = Query()
        for record in db.all():
            val = np.asarray(record['val'])
            val = (val-mean) / std 
            db.update({'key': record['key'], 'val': val.tolist()}, Q.key == record['key']})
