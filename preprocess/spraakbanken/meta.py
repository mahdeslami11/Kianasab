from os.path import join
from os import listdir

#Find spl file from speaker id
#Split spl file on newlines
#Find titles 
#[System]
    #Delimiter
    #Frequency
#[Info states] 
    #Age
    #Sex
    #Region of Dialect, make this required. Or else remove speaker
#[Record states]
    #Read all lines, they are each an utterance
    #First line is silence if contains 'tavshed'
    #Use package sox to validate audio file
    #Split on [System] Delimiter
        #Remove if split array entry is empty
#Print meta data to json with indent=4

def read_spl_file(fpath:str):
    info = fpath.split('_')
    station = info[0]
    station_id = station[7:].replace('0', '')
    substation = info[1]
    speaker = info[2]

    for f in listdir(join('/work1/s183921/speaker_data/Spraakbanken-Raw',station,substation,'adb_0565','data','scr0565',f'0{station_id}',speaker)):
        with open(f, 'r') as spl:
            spl_content = spl.readlines()
            spl_content = spl_content.split('\n\n')
            print(spl_content[0])

