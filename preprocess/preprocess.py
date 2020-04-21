from logger import Logger
from directory import OutputDir
from splitter import Splitter

def preprocess(data:str):
    log = Logger()
    vctkOutDir = OutputDir('vctk')
    danishOutDir = OutputDir('spraakbanken')
    splitters = [
            VCTKSplitter(log, vctkOutDir),
            SpraakbankenSplitter(log, danishOutDir)
            ]

    log.write(f'# Preprocessing of {data}')
    log.write(f'VCTK preprocessed data is output to: {vctkOutDir.path}')
    log.write(f'Danish preprocessed data is output to: {danishOutDir.path}')

    for s in splitters:
        s.source_target_split(data)

if __name__ == '__main__':
    preprocess()
