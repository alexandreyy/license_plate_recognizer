'''
Created on 26/06/2015
 
@author: Alexandre Yukio Yamashita
         Flavio Nicastro
'''
from ConfigParser import SafeConfigParser
from argparse import ArgumentParser
import time
import warnings

from detector import Detector
from license_plate_recognizer import LicensePlateRecognizer
from models.config_data import ConfigData
from models.files import Files
from models.image import Image
from models.quadrilaterals import Quadrilaterals


warnings.simplefilter("error")
 
if __name__ == '__main__': 
    '''
    Test license plate recognizer.
    '''
     
    # Parses args.
    arg_parser = ArgumentParser(description='Test license plate recognizer.')
    arg_parser.add_argument('-c', '--config', dest='config_file', default='config.ini', help='Configuration file')
    args = vars(arg_parser.parse_args())
     
    # Parses configuration file.
    config_parser = SafeConfigParser()
    config_parser.read(args['config_file'])
    cascade_file = config_parser.get('data', 'path_classifier')
    path_svm_hog_detector = config_parser.get('data', 'path_svm_hog_detector')
    path_svm_binary_detector = config_parser.get('data', 'path_svm_binary_detector')
    image_width = int(config_parser.get('training', 'image_width'))
    image_height = int(config_parser.get('training', 'image_height'))
    character_width = int(config_parser.get('training', 'character_width'))
    character_height = int(config_parser.get('training', 'character_height'))
    path_letter_classifier = config_parser.get('data', 'path_letter_classifier')
    path_number_classifier = config_parser.get('data', 'path_number_classifier')
    character_classifier_type = config_parser.get('data', 'character_classifier_type')
    path_number_knn_labels_classifier = config_parser.get('data', 'path_number_knn_labels_classifier')
    path_number_knn_images_classifier = config_parser.get('data', 'path_number_knn_images_classifier')
    path_letter_knn_labels_classifier = config_parser.get('data', 'path_letter_knn_labels_classifier')
    path_letter_knn_images_classifier = config_parser.get('data', 'path_letter_knn_images_classifier')
    directory = config_parser.get('testing', 'path_test')

    image_paths = Files(directory)
    image_paths = image_paths.paths
    detector = Detector(cascade_file, path_svm_hog_detector, path_svm_binary_detector, image_width, image_height) 
    recognizer = LicensePlateRecognizer(character_width, character_height, character_classifier_type, \
                                        path_letter_classifier, path_number_classifier, \
                                        path_number_knn_labels_classifier, path_number_knn_images_classifier,\
                                        path_letter_knn_labels_classifier, path_letter_knn_images_classifier)
    
    correct_characters = 0
    total_characters = 0
    correct_plates = 0
    total_plates = 0
    
    for image_path in image_paths:
        # Read image.
        start = time.time()
        
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
        
            # Test result.    
            config_data = ConfigData(Quadrilaterals(), image_path)
            config_data.read_data()
            correct_plate = config_data.license_plates_text
        
            if len(correct_plate) == 7 and len(plate) == 7:
                correct_plate = correct_plate.lower()
                plate = plate.lower()
                
                for index in range(7):
                    if plate[index] == correct_plate[index]:
                        correct_characters += 1
                
                    total_characters += 1
            
                if correct_plate == plate:
                    correct_plates += 1
                
                print "Correct plate: " + correct_plate
        else:
            print "None"
        
        total_plates += 1
        
        if total_characters > 0:
            print 'Character accuracy: ' + str(correct_characters) + "/" + str(total_characters) + " - " + str(correct_characters * 100.0 / total_characters)
            print 'Plate accuracy: ' + str(correct_plates) + "/" + str(total_plates) + " - " + str(correct_plates * 100.0 / total_plates)
        
        end = time.time()
        total_time = end - start
        print "Time: " + str(total_time) + " s"
        print '---------------------------------------------------------'
        
        