'''
Created on 06/04/2015

@author: Alexandre Yukio Yamashita
'''

from models.quadrilateral import Quadrilateral


class Quadrilaterals:
    '''
    Rectangle with 4 points. 
    '''
    
    def __init__(self):
        self.data = []
     
    def clean(self):
        '''
        Clean data.
        '''
        self.data = []
    
    def add_quadrilateral(self):
        '''
        Add quadrilateral.
        '''
        quadrilateral = Quadrilateral()
        self.data.append(quadrilateral)
        return quadrilateral
    
    def remove_quadrilateral(self, quadrilateral):
        '''
        Remove quadrilateral.
        '''
        self.data.remove(quadrilateral)
