import sys
from TinyDB import TinyDB, Query

if __name__ == '__main__':
    db_path = sys.argv[1]
    output_path = sys.argv[2]
    segment_size = int(sys.argv[3])

    db = TinyDB(db_path)
    reduced_db = TinyDB(output_path)

    for record in db.all():
        if record['val'].shape[0] > segment_size:
            reduced_db.insert(record)
