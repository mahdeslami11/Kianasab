import glob
from os.path import join
import json
import matplotlib.pyplot as plt

def plot_dialect_distribution(fpath):
    dialects = []
    for meta_data in glob.glob(join(fpath,'*/*.json')):
        with open(meta_data, 'r') as md:
            meta = json.loads(md.read())
            dialects.append(meta['dialect'])
    
    counts = {d:dialects.count(d) for d in dialects}
    plt.bar(counts.keys(), counts.values())
    plt.savefig('plots/dialects.png')

if __name__ == '__main__':
    plt.gcf().set_size_inches(20,20)
    plot_dialect_distribution('/work1/s183921/speaker_data/Spraakbanken-Corpus')
