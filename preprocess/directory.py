import os
from os.path import abspath, join

class OutputDir():
    def __init__(self, subdir:str):
        self.path = abspath(join(join(join(os.pardir, os.pardir), 'preprocessed_data')), subdir)

        # Create output dir if not exists
        if not os.path.isdir(self.path):
            os.mkdir(self.path)


    def saveFile(self, save_func, file_name):
        '''
        Save function to give a specific function logic for saving any type of content

        :param save_func:   The logic to use for saving content. For a text file e.g.
                            lambda f: with open(f, 'w+') as myfile:
                                        myfile.write(content)
        :param file_name:   Name of the file to which the save_func will save the content
        '''
        save_func(join(self.path, file_name))
