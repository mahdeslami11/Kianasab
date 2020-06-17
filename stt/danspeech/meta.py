import pandas as pd
import json
import os
import glob

def __write_to_csv(p, meta, csv:pd.DataFrame):
    split_p = p.rsplit(os.sep, 1)[1].split('_')
    station = split_p[0]
    substation = split_p[1]
    speaker_id = split_p[2]

    utterances          = {f'{station}_{substation}_{speaker_id}_{key}':val for key,val in meta.items() if key.startswith('u')}
    filename_col        = list(utterances.keys())
    transcription_col   = list(utterances.values())
    gender_col          = [meta['sex']]*len(utterances)
    age_col             = [meta['age']]*len(utterances)
    add_csv             = pd.DataFrame.from_dict({
                                                    'file'          : filename_col,
                                                    'trans'         : transcription_col,
                                                    'gender'        : gender_col,
                                                    'age'           : age_col
                                                })

    return csv.append(add_csv, ignore_index=True)

def preprocess(args):
    csv = pd.DataFrame()
    csv_out = os.path.join(args.out_dir, args.csv)
    json_paths = [os.path.abspath(p) for p in glob.glob(os.path.join(args.meta_data, '*.json'))]
    
    print(f'Preprocessing {len(json_paths)} .json meta data files')
    for p in json_paths:
        print(f'Writing {p} to csv file')
        with open(p, 'r') as f:
            meta = json.loads(f.read())
            csv = __write_to_csv(p, meta, csv)

    if not args.overwrite:
        old_csv = pd.read_csv(csv_out)
        csv = old_csv.append(csv, ignore_index=True)
    
    print(f'Saving all meta data to: {csv_out}')
    csv.to_csv(csv_out, index=False)
