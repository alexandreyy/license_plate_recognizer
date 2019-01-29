'''
Created on 15/07/2015

@author: Alexandre Yukio Yamashita
         Flavio Nicastro
'''
from numpy.core.defchararray import isalpha, isdigit

from character_recognizer import CharacterRecognizer
from models.point import Point


class LicensePlateRecognizer:
    '''
    Recognizer characters in license plate image.
    '''
    
    def __init__(self, character_width=15, character_height=15, classifier_type="svm_linear", \
                 path_letter_classifier="classifier/letter_classifier.xml", path_number_classifier="classifier/number_classifier.xml", \
                 path_number_knn_labels_classifier="classifier/number_knn_labels_classifier.xml",
                 path_number_knn_images_classifier="classifier/number_knn_images_classifier.xml",
                 path_letter_knn_labels_classifier="classifier/letter_knn_labels_classifier.xml",
                 path_letter_knn_images_classifier="classifier/letter_knn_images_classifier.xml"):
        self.letter_recognizer = CharacterRecognizer(character_width, character_height, classifier_type)
        self.number_recognizer = CharacterRecognizer(character_width, character_height, classifier_type)
        self.number_recognizer.load(path_number_classifier, path_number_knn_images_classifier, path_number_knn_labels_classifier)
        self.letter_recognizer.load(path_letter_classifier, path_letter_knn_images_classifier, path_letter_knn_labels_classifier)
        
    def predict(self, license_plate):
        plate = ""
        
        if len(license_plate.subrects) == 7:
            for index in range(3):
                character = license_plate.subrects[index]
                character_image = license_plate.image.crop(Point(character.x, character.y), Point(character.x + character.w - 1, character.y + character.h - 1))
            
                if character_image.data.size > 0 and character_image.data is not None:
                    letter = self.letter_recognizer.predict(character_image, True)
                    
                    if isalpha(letter):
                        plate += letter
                    else:
                        plate += "a"
                else:
                    plate += "a"
                
            for index in range(3, 7):
                character = license_plate.subrects[index]
                character_image = license_plate.image.crop(Point(character.x, character.y), Point(character.x + character.w - 1, character.y + character.h - 1))
            
                if character_image.data.size > 0 and character_image.data is not None:
                    digit = self.number_recognizer.predict(character_image)
                    
                    if isdigit(digit):
                        plate += digit
                    else:
                        plate += str(1)
                else:
                    plate += str(1)
                
        else:
            plate = "AAA1111"
        
        return plate
