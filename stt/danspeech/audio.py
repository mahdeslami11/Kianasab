import os
import glob
import shutil

def preprocess(args):
    data_dir = args.meta_data.rsplit(os.sep, 1)[0]
    out_path = args.out_dir
    print(f'Copying files from {data_dir} to {out_path}')

    paths = glob.glob(os.path.join(data_dir, f'*{os.sep}*.wav'))
    print(f'Found {len(paths)} .wav files')

    for i, p in enumerate(paths):
        shutil.copyfile(p, out_path)
        print(f'Copied {i+1}', end='\r') 
