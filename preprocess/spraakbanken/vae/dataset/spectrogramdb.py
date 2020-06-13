from numpy import ndarray, asarray
from tinydb import TinyDB, Query

class SpectrogramDB(TinyDB):
    def __init__(self, db_path:str):
        self.query = Query()
        super(SpectrogramDB, self).__init__(db_path)

    def insert_spectrogram(self, name, spectrogram:ndarray):
        super().insert({'key': name, 'val': spectrogram.tolist()})

    def get_spectrogram(self, name):
        return asarray(super().search(query.key == name)[0]['val'])

    def update_spectrogram(self, name, spectrogram):
            super().update({'key': name, 'val': spectrogram.tolist()}, 
                            query.key == name)

    def get_keys(self):
        return [record['key'] for record in super().all()]

    def get_vals(self):
        return [asarray(record['val']) for record in super().all()]
