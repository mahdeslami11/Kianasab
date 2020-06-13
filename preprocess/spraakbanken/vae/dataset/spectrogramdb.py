import os
from numpy import ndarray, asarray
from tinydb import TinyDB, Query

class SpectrogramDB():
    def __init__(self, db_path:str, overwrite=False):
        if overwrite and os.path.exists(db_path):
            os.remove(db_path)
        self.query = Query()
        self.db = TinyDB(db_path)

    def insert_spectrogram(self, name, spectrogram:ndarray):
        self.db.insert({'key': name, 'val': spectrogram.tolist()})

    def get_spectrogram(self, name):
        return asarray(self.db.search(query.key == name)[0]['val'])

    def update_spectrogram(self, name, spectrogram):
            self.db.update({'key': name, 'val': spectrogram.tolist()}, 
                            query.key == name)

    def get_keys(self):
        return [record['key'] for record in self.db.all()]

    def get_vals(self):
        return [asarray(record['val']) for record in self.db.all()]
