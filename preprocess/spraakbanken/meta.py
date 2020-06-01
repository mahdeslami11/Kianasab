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
        right_side = right_side.split(delimiter)[1]
        if len(right_side) > 1:
            if str.startswith(right_side, 'age'):
                info_dict['age'] = right_side.split(delimiter)[1]
            elif str.startswith(right_side, 'sex'):
                info_dict['sex'] = right_side.split(delimiter)[1]
            elif str.startswith(right_side, 'region of dialect'):
                info_dict['dialect'] = right_side.split(delimiter)[1]

def read_system_info(c, info_dict):
    if str.startswith(c, 'delimiter'):
        info_dict['delimiter'] = c.split('=')[1]
    elif str.startswith(c, 'frequency'):
        info_dict['frequency'] = c.split('=')[1]


def read_spl_file(speaker_id:str):
    '''
    Reads the content of a speaker .spl info file for the speaker with the given id

    :param speaker_id:  Id of the speaker to find and read the .spl info file for

    :returns:           A dictionary with the read speaker info
    '''
    info_dict = {}
    info_dict['speaker'] = speaker_id

    #Change the path to fit your own file structure if needed
    spl_file = glob.glob(f'/work1/s183921/speaker_data/Spraakbanken-Raw/*/*/*/data/*/*/*/{speaker_id}.spl')[0]

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
