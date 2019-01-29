'''
Created on 05/04/2015

@author: Alexandre Yukio Yamashita
'''
import ntpath
from os.path import os

from models.logger import Logger
from models.point import Point


class ConfigData:
    '''
    Parses license plate file.
    '''
    
    _logger = Logger()
    
    '''
    Save and read config data file. 
    '''
    def __init__(self, quadrilaterals, image_path=""):
        self.quadrilaterals = quadrilaterals
        
        if image_path != "": 
            self.get_file_path_by_image_path(image_path)
        
    def set_config_path(self, image_path):
        '''
        Set config file path using image_path.
        '''
        self.get_file_path_by_image_path(image_path)
          
    def get_file_path_by_image_path(self, image_path):
        '''
        Get file path using image path.
        '''
        
        self.file_name = ntpath.basename(image_path)
        self.file_name = os.path.splitext(self.file_name)[0]
        self.folder_path = self.save_path = os.path.dirname(os.path.abspath(image_path))
        self.save_path = os.path.join(self.save_path, self.file_name + ".txt")
        return self.save_path
    
    def set_license_plates(self, license_plates_text):
        '''
        Get license plates by text.
        '''
        self.license_plates = [x.strip() for x in license_plates_text.split(',')]
        
        for license_plate in self.license_plates:
            if not license_plate:
                self.license_plates.remove(license_plate)
            
    def save_file(self, quadrilaterals):
        '''
        Save config file.
        '''
        file_data = ""
        
        total = len(quadrilaterals.data)
        if total == 0:
            file_data = "None"
        else:
            for index_quadrilateral in range(total):
                quadrilateral = quadrilaterals.data[index_quadrilateral]
                
                for point in quadrilateral.points:
                    file_data += str(int(point.x)) + "," + str(int(point.y)) + ","
                
                file_data += self.license_plates[index_quadrilateral]
                
                if index_quadrilateral != total - 1:
                    file_data += "\n"
        
        config_file = open(self.save_path, "w")
        config_file.write(file_data)
        config_file.close() 
        
    def read_data(self):
        '''
        Read config data from file.
        '''
        self.license_plates = []
        self.license_plates_text = ""
        self.quadrilaterals.clean()
        self._logger.log(Logger.INFO, "Reading configuration data from " + self.save_path)
        
        if os.path.isfile(self.save_path):
            f = open(self.save_path, "r")
            read_data = []
            
            for line in f:
                read_data.append(line.rstrip('\n'))
            
            if not (len(read_data) == 1 and read_data == "None" or len(read_data) == 0):
                for index_line in range(len(read_data)):
                    line = read_data[index_line]
                        
                    data_list = [x.strip() for x in line.split(',')]
                        
                    total = len(data_list)
                    if total == 9:
                        quadrilateral = self.quadrilaterals.add_quadrilateral()
                        
                        for data_index in range(4):
                            quadrilateral.add_point(Point(int(data_list[data_index * 2]), int(data_list[data_index * 2 + 1])))
                        
                        self.license_plates.append(data_list[8])
                        
                        if index_line == 0:
                            self.license_plates_text = data_list[8]
                        else:
                            self.license_plates_text += "," + data_list[8]                        
            f.close()
            
