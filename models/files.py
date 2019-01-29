'''
Created on 05/04/2015

@author: Alexandre Yukio Yamashita
'''

from glob import glob
import os


class Files:
    '''
    List of file paths.
    '''
    
    def __init__(self, path, is_all = False):
        if not is_all:
            self.paths = glob(path + '*.jpeg')
            self.paths.extend(glob(path + '*.JPEG'))
            self.paths.extend(glob(path + '*.png'))
            self.paths.extend(glob(path + '*.jpg'))        
            self.paths.extend(glob(path + '*.JPG'))
            self.paths.extend(glob(path + '*.PNG'))
        else:
            self.paths = glob(path + '*')
            
    def remove(self):
        '''
        Remove all files.
        '''
        for f in self.paths:
            os.remove(f)