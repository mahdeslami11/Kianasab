from abc import ABC, abstractmethod
from logger import Logger
from directory import OutputDir

class Splitter(ABC):
    def __init__(self, log:Logger, outDir:OutputDir):
        self.log = log
        self.outDir = outDir
        super().__init__()

    @abstractmethod
    def source_target_split(self, dataPath:str):
        pass

