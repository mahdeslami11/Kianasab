import pickle
import sys
import os

if __name__ == '__main__':
    pkl_path = sys.argv[1]
    output_path = sys.argv[2]
    segment_size = int(sys.argv[3])

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    reduced_data = {key:val for key, val in data.items() if val.shape[0] > segment_size}
    print(f'Number of speakers after reduction: {len(reduced_data)}')
    print(f'Number of utterances after reduction: {len(reduced_data.keys()}')

    with open(output_path, 'wb+') as f:
        pickle.dump(reduced_data, f)
