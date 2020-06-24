'''
Script for extracting meta data from Spraakbanken .spl files as .json files.
The .spl data is important to extract if you want meta data about the speakers 
e.g. for statistical purposes.
'''
import glob

def read_record_states(c, info_dict):
     delimiter = info_dict['delimiter']
     content = c.split(delimiter)
     utterance =  content[2]
     if not 'tavshed' in utterance:
        if len(content) > 5:
            info_dict[content[5]] = utterance


def read_info_states(c, info_dict):
    content = c.split('=')
    if len(content) > 1:
        right_side = content[1]
        delimiter = info_dict['delimiter']
        right_side = right_side.split(delimiter)
        if len(right_side) > 1:
            key = right_side[0]
            val = right_side[1]
            if str.startswith(key, 'age'):
                info_dict['age'] = val 
            elif str.startswith(key, 'sex'):
                info_dict['sex'] = val 
            elif str.startswith(key, 'region of dialect'):
                info_dict['dialect'] = val

def read_system_info(c, info_dict):
    if str.startswith(c, 'delimiter'):
        info_dict['delimiter'] = c.split('=')[1]
    elif str.startswith(c, 'frequency'):
        info_dict['frequency'] = c.split('=')[1]


def read_spl_file(speaker_id:str, spl_file:str):
    '''
    Reads the content of a speaker .spl info file for the speaker with the given id

    :param speaker_id:  Id of the speaker to find and read the .spl info file for

    :returns:           A dictionary with the read speaker info
    '''
    info_dict = {}
    info_dict['speaker'] = speaker_id

    with open(spl_file, 'r') as spl:
        spl = spl.read().lower()
        spl = spl.split('\n\n')
        
        for f in spl:
            content = f.split('\n')
            for c in content[1:]:
                if content[0] == '[system]':
                    read_system_info(c, info_dict)
                elif content[0] == '[info states]':
                    read_info_states(c, info_dict)
                elif content[0] == '[record states]':
                    read_record_states(c, info_dict)

        return info_dict
