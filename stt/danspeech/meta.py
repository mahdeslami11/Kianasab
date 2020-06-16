import pandas as pd
import json
import os

def __write_to_csv(meta, csv:pd.DataFrame):
    utterances          = {key:val for key,val in meta.items() if key.startswith('u')}
    filename_col        = utterances.keys()
    transcription_col   = utterances.values()
    gender_col          = meta['gender']*len(utterances)
    age_col             = meta['age']*len(utterances)
    add_csv             = pd.DataFrame.from_dict({
                                                    'file'          : filename_col,
                                                    'transcription' : transcription_col,
                                                    'gender'        : gender_col,
                                                    'age'           : age_col
                                                })

    return csv.append(csv, add_csv, ignore_index=True)

def preprocess(args):
    csv = pd.DataFrame()
    csv_out = os.path.join(args.out_dir, args.csv)
    json_paths = [os.path.abspath(p) for p in os.listdir(args.meta_data) if p.endswith('.json')]
    
    print(f'Preprocessing {len(json_paths)} .json meta data files')
    for p in json_paths:
        print(f'Writing {p} to csv file')
        meta = json.loads(p)
        csv = __write_to_csv(meta, csv)

    if not args.overwrite:
        old_csv = pd.read_csv(csv_out)
        csv = old_csv.append(csv, ignore_index=True)
    
    print(f'Saving all meta data to: {csv_out}')
    csv.to_csv(csv_out)
