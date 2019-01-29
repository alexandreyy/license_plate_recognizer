'''
Created on 26/06/2015

@author: Alexandre Yukio Yamashita
         Flavio Nicastro
'''
#from ConfigParser import SafeConfigParser
import sys

from detector import Detector
from license_plate_recognizer import LicensePlateRecognizer
from models.image import Image


if __name__ == '__main__': 
    '''
    Detect and recognize license plate in image.
    '''

    if len(sys.argv) < 2:
        print "Missing image path parameter"
    elif len(sys.argv) > 2:
        print "Too many arguments. lpdetect only takes 1 argument"
    else:
        # Get image path.
        image_path = sys.argv[1]

        # Commented to improve speed.
#         # Parses args.
#         config_file = "config.ini"
#         
#         # Parses configuration file.
#         config_parser = SafeConfigParser()
#         config_parser.read(config_file)
#         cascade_file = config_parser.get('data', 'path_classifier')
#         path_svm_hog_detector = config_parser.get('data', 'path_svm_hog_detector')
#         path_svm_binary_detector = config_parser.get('data', 'path_svm_binary_detector')
#         image_width = int(config_parser.get('training', 'image_width'))
#         image_height = int(config_parser.get('training', 'image_height'))
#         character_width = int(config_parser.get('training', 'character_width'))
#         character_height = int(config_parser.get('training', 'character_height'))
#         path_letter_classifier = config_parser.get('data', 'path_letter_classifier')
#         path_number_classifier = config_parser.get('data', 'path_number_classifier')
#         character_classifier_type = config_parser.get('data', 'character_classifier_type')
#         path_number_knn_labels_classifier = config_parser.get('data', 'path_number_knn_labels_classifier')
#         path_number_knn_images_classifier = config_parser.get('data', 'path_number_knn_images_classifier')
#         path_letter_knn_labels_classifier = config_parser.get('data', 'path_letter_knn_labels_classifier')
#         path_letter_knn_images_classifier = config_parser.get('data', 'path_letter_knn_images_classifier')
#         directory = config_parser.get('testing', 'path_test')
    
        cascade_file = "classifier/cascade.xml"
        path_svm_hog_detector = "classifier/svm_detector_hog.xml"
        path_svm_binary_detector = "classifier/svm_detector_binary.xml"
        image_width = 96
        image_height = 48
        character_width = 25
        character_height = 25
        path_letter_classifier = "classifier/letter_classifier.xml"
        path_number_classifier = "classifier/number_classifier.xml"
        character_classifier_type = "svm_linear_hog"
        path_number_knn_labels_classifier = "classifier/number_knn_labels_classifier.xml"
        path_number_knn_images_classifier = "classifier/number_knn_images_classifier.xml"
        path_letter_knn_labels_classifier = "classifier/letter_knn_labels_classifier.xml"
        path_letter_knn_images_classifier = "classifier/letter_knn_images_classifier.xml"
        
        detector = Detector(cascade_file, path_svm_hog_detector, path_svm_binary_detector, image_width, image_height) 
        recognizer = LicensePlateRecognizer(character_width, character_height, character_classifier_type, \
                                            path_letter_classifier, path_number_classifier, \
                                            path_number_knn_labels_classifier, path_number_knn_images_classifier,\
                                            path_letter_knn_labels_classifier, path_letter_knn_images_classifier)
        
        # Read image.
        error = False
        try:
            image = Image(image_path)
             
            if image.data is None:
                raise ValueError('Error to open image.')
        except:
            print "Error to open image."
              
        # Detect license plates.
        best_license_plate = detector.detect_license_plate(image)
          
        if best_license_plate is not None:
            # Print result.
            plate = recognizer.predict(best_license_plate)
            point1 = best_license_plate.quadrilateral.points[0]
            point2 = best_license_plate.quadrilateral.points[1]
            point3 = best_license_plate.quadrilateral.points[2]
            point4 = best_license_plate.quadrilateral.points[3]
            
            print str(point1.x) + "," + str(point1.y) + "," + \
                  str(point2.x) + "," + str(point2.y) + "," + \
                  str(point3.x) + "," + str(point3.y) + "," + \
                  str(point4.x) + "," + str(point4.y) + "," + plate
                  
            # Plot result.
            #detector.plot_quadrilateral(image)
            #best_license_plate.image.plot_rectangles(best_license_plate.subrects)             
        else:
            print "None"
            