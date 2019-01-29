'''
Created on 06/07/2015

@author: Alexandre Yukio Yamashita
'''
from numpy import math

from models.character_validator import CharacterValidator
from models.filter_characters import FilterCharacters
from models.image import Image
from models.license_plate_utils import pre_process_license_plate_image
from models.point import Point


class FilterLicensePlate:
    '''
    Filter of license plates.
    '''
    
    def filter_using_svm(self, license_plates, image, svm_detector, image_width, image_height):
        '''
        Filter misclassified license plates using SVM.
        '''
        filtered_license_plates = []
        
        for license_plate in license_plates:
            license_plate_image = image.crop(Point(license_plate.x, license_plate.y), \
                                             Point(license_plate.x + license_plate.w, \
                                                   license_plate.y + license_plate.h))
            license_plate_image = license_plate_image.resize(image_width, image_height)
            
            if svm_detector.predict(license_plate_image) > 0:
                filtered_license_plates.append(license_plate)
            else:
                license_plate_image_old = Image(image=license_plate_image.data)
                license_plate_image_old.filter_median(size=3)
                
                if svm_detector.predict(license_plate_image_old) > 0:
                    filtered_license_plates.append(license_plate)
                    
        return filtered_license_plates
    
    def get_best_license_plate(self, license_plates, image_original, resize_width=400):
        '''
        Get best license plate.
        '''
        best_license_plate = None
        
        # Detect characters in license plate.
        for license_plate in license_plates:
            character_validator = CharacterValidator()
            image = image_original.crop(Point(license_plate.x, license_plate.y), Point(license_plate.x + license_plate.w, license_plate.y + license_plate.h))
            image = image.resize(resize_width, resize_width * image.height / image.width)
            image = pre_process_license_plate_image(image, adaptative=False)
            license_plate.image = image
            license_plate.subrects = image.compute_rectangles_for_characters()
            character_validator.remove_wrong_characters(license_plate)
            
            # If not enough characters detected, binarize image using adaptative method. 
            if len(license_plate.subrects) < 2:
                image = image_original.crop(Point(license_plate.x, license_plate.y), Point(license_plate.x + license_plate.w, license_plate.y + license_plate.h))
                image = image.resize(resize_width, resize_width * image.height / image.width)
                image = pre_process_license_plate_image(image, True)
                license_plate.image = image
                license_plate.subrects = image.compute_rectangles_for_characters()
                character_validator.remove_wrong_characters(license_plate)
        
        # Find best license plate by total of characters and area. 
        if len(license_plates) > 0:
            license_plates = sorted(license_plates, key=lambda x: len(x.subrects) -math.tanh(x.h*x.w*1.0/(x.image.width*x.image.height)))
            license_plates = license_plates[::-1] 
            best_license_plate = license_plates[0]
            
            filtera = FilterCharacters()
            best_license_plate.subrects = filtera.filter_characters_by_color(best_license_plate.subrects, best_license_plate.image)
            
        return best_license_plate
