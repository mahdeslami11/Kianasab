import json 
import sys
import os
import random
from spectrogramdb import SpectrogramDB

if __name__ == '__main__':
    db_path = sys.argv[1]
    sample_path = sys.argv[2]
    n_samples = int(sys.argv[3])
    segment_size = int(sys.argv[4])

    db = SpectrogramDB(db_path)

    # (utt_id, timestep, neg_utt_id, neg_timestep)
    samples = []

    # filter length > segment_size
    utt_list = db.get_keys()
    utt_list = sorted(list(filter(lambda u : len(db.get_spectrogram(u)) > segment_size, utt_list)))
    print(f'{len(utt_list)} utterances')
    sample_utt_index_list = random.choices(range(len(utt_list)), k=n_samples)

    for i, utt_ind in enumerate(sample_utt_index_list):
        if i % 500 == 0:
            print(f'sample {i} samples')
        utt_id = utt_list[utt_ind]
        spectrogram = db.get_spectrogram(utt_id)
        t = random.randint(0, len(spectrogram) - segment_size)
        samples.append((utt_id, t))

    with open(sample_path, 'w+') as f:
        json.dump(samples, f)

