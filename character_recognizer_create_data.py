'''
Created on 17/07/2015

@author: Alexandre Yukio Yamashita
         Flavio Nicastro
'''
from ConfigParser import SafeConfigParser
from argparse import ArgumentParser
from numpy.core.defchararray import isalpha, isdigit

from models import character_validator
from models.character_validator import CharacterValidator
from models.config_data import ConfigData
from models.files import Files
from models.image import Image
from models.license_plate_utils import pre_process_license_plate_image
from models.logger import Logger
from models.point import Point
from models.quadrilaterals import Quadrilaterals
from models.rect import Rect
import numpy as np 
from pre_processing import crop_image


if __name__ == '__main__': 
    '''
    Create data to train character recognizer.
    '''
    
    # Parses args.
    arg_parser = ArgumentParser(description='Create data to train character recognizer.')
    arg_parser.add_argument('-c', '--config', dest='config_file', default='config.ini', help='Configuration file')
    args = vars(arg_parser.parse_args())
     
    # Parses configuration file.
    config_parser = SafeConfigParser()
    config_parser.read(args['config_file'])    
    path_original_positive = config_parser.get('data', 'path_original_positive')
    character_original_width = int(config_parser.get('training', 'character_original_width'))
    character_original_height = int(config_parser.get('training', 'character_original_height'))
    path_letter_images = config_parser.get('data', 'path_letter_images')
    path_letter_labels = config_parser.get('data', 'path_letter_labels')
    path_number_images = config_parser.get('data', 'path_number_images')
    path_number_labels = config_parser.get('data', 'path_number_labels')
    image_width = 400
    image_height = 200
    
    # Load and pre process images.
    logger = Logger()
    logger.log(Logger.INFO, "Loading images.")
    positive_images = Files(path_original_positive)
    character_validator = CharacterValidator()
    number_labels = []
    number_images = []
    letter_labels = []
    letter_images = []
    
    for index in range(len(positive_images.paths)):
        file_path = positive_images.paths[index]
        
        # Load image and configuration data.
        image_original = Image(file_path)
        config_data = ConfigData(Quadrilaterals(), file_path)
        config_data.read_data()
        
        # Convert and equalize image.
        image_original.convert_to_gray()
                  
        for quadrilateral in config_data.quadrilaterals.data:
            # Crop image using configuration data.         
            image = crop_image(image_original, quadrilateral, int(image_width), int(image_height))
            image = pre_process_license_plate_image(image, adaptative=True)
            point1 = quadrilateral.points[0]
            point2 = quadrilateral.points[2]
            width =  point2.x -point1.x
            height = point2.y -point1.y
            
            license_plate = Rect([point1.x, point1.y, width, height])
            license_plate.image = image
            license_plate.subrects = image.compute_rectangles_for_characters()
            character_validator.remove_wrong_characters(license_plate)
            license_plate = character_validator.adjust_characters(license_plate, image_original, image_width)
            correct_plate = config_data.license_plates_text
            index = 0
            
            for character in license_plate.subrects:
                index += 1
                
                character_image = license_plate.image.crop(Point(character.x, character.y), Point(character.x + character.w - 1, character.y + character.h - 1))
                
                if character_image.data.size > 0 and character_image.data is not None:
                    image = Image(image=character_image.data)
                    image = image.resize(200, 200)
                    character = chr(image.plot())
                    image = character_image.resize(character_original_width, character_original_height)
                                
                    if isalpha(character) or isdigit(character):        
                        if isalpha(character) or character == "1" or character == "0":
                            if character != "1" and character != "0":
                                letter_labels.append(ord(character))
                                letter_images.append(image.data.flatten())
                                print "Character: " + character
                            else:
                                if character == "1":
                                    character = "i"
                                    
                                if character == "0":
                                    character = "o"
                                    
                                letter_labels.append(ord(character))
                                letter_images.append(image.data.flatten())
                                print "Character: " + character
                                
                        if isdigit(character) or character == "o" or character == "i":
                            if character != "i" and character != "o":
                                number_labels.append(ord(character))
                                number_images.append(image.data.flatten())
                                print "Character: " + character
                            else:
                                if character == "o":
                                    character = "0"
                                    
                                if character == "i":
                                    character = "1"
                                
                                number_labels.append(ord(character))
                                number_images.append(image.data.flatten())
                                print "Character: " + character
                                
                        if len(number_images) > 0:
                            np_number_images = np.array(number_images, np.uint8)
                            np_number_labels = np.array(number_labels, np.float32)
                            np.savetxt(path_number_images, number_images)
                            np.savetxt(path_number_labels, number_labels)
                        
                        if len(letter_images) > 0:
                            np_letter_images = np.array(letter_images, np.uint8)
                            np_letter_labels = np.array(letter_labels, np.float32)
                            np.savetxt(path_letter_images, letter_images)
                            np.savetxt(path_letter_labels, letter_labels)
