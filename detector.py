'''
Created on 26/06/2015

@author: Alexandre Yukio Yamashita
         Flavio Nicastro
'''
from ConfigParser import SafeConfigParser
from argparse import ArgumentParser
import cv2

from detector_haar import DetectorHaar
from detector_svm import SVM
from models import character_validator
from models.character_validator import CharacterValidator
from models.filter_license_plate import FilterLicensePlate
from models.image import Image
from models.license_plate_utils import reescale_license_plate, \
    get_license_plate_quadrilateral


class Detector:
    '''
    Detect license plate in image.
    '''
    
    cascade_classifier = None
    license_plates = None
    svm_detector = None
    image_width = None
    image_height = None
    
    def __init__(self, cascade_file, svm_hog_detector_file, svm_binary_detector_file, image_width, image_height):
        self.cascade_classifier = DetectorHaar(cascade_file)
        self.svm_detector = SVM(svm_hog_detector_file, svm_binary_detector_file)
        self.image_width = image_width
        self.image_height = image_height
        self.resize_width = 400
        
    def detect_license_plate(self, image_original):
        '''
        Detect license plate in image.
        '''
        
        if image_original.data is None:
            raise ValueError('Error to open image.')
        
        # Convert image to gray.
        image_original.convert_to_gray()
         
        # Reescale image to speed up detection and equalize image.
        image_resized = image_original.resize(self.resize_width, self.resize_width * image_original.height / image_original.width)
        
        # Pre process image for detection.
        image_for_detection = Image(image=image_resized.data)
        image_for_detection.smart_equalize()
        
        # Find license_plates using cascade classifier.
        self.license_plates = []
        self.license_plates += self.cascade_classifier.detect_using_cascade_classifier(image_for_detection)
        
        # Find license_plates using rectangles.
        image_for_detection.filter_median(size=3)
        image_for_detection.contrast()  # Increase contrast to improve rectangle detection.
        self.license_plates += image_for_detection.compute_rectangles_for_plates()
        
        # Filter misclassified license plates using SVM.
        lp_filter = FilterLicensePlate()
        self.license_plates = lp_filter.filter_using_svm(self.license_plates, image_for_detection, self.svm_detector, self.image_width, self.image_height)
        
        # Reescale license plates to original size.
        total_license_plates = range(len(self.license_plates))
        for index in total_license_plates:
            self.license_plates[index] = reescale_license_plate(self.license_plates[index], \
                                                                image_original.width, self.resize_width)
        
        # Get best license plate found, computing character positions.
        best_license_plate = lp_filter.get_best_license_plate(self.license_plates, image_original)
        
        if best_license_plate is not None:
            # Adjust size of characters and predict the position of unrecognized characters.
            character_validator = CharacterValidator()
            best_license_plate = character_validator.adjust_characters(best_license_plate, image_original, self.resize_width)
            best_license_plate = get_license_plate_quadrilateral(best_license_plate)
            self.license_plates = [best_license_plate]
        else:
            self.license_plates = []
            
        return best_license_plate    
    
    def plot(self, image):
        '''
        Plot result.
        '''
        
        image_data = image.data
        image = Image(image=image_data)
        
        if self.license_plates != None:
            for license_plate in self.license_plates:
                cv2.rectangle(image.data, (license_plate.x, license_plate.y), (license_plate.x + license_plate.w, license_plate.y + license_plate.h), (255, 0, 0), 2)
        
        image = image.resize(self.resize_width, self.resize_width * image.height / image.width)
        image.plot()        
    
    def plot_quadrilateral(self, image):
        '''
        Plot result.
        '''
        
        image_data = image.data
        image = Image(image=image_data)
        image.data = cv2.cvtColor(image.data, cv2.COLOR_GRAY2RGB)
        image.channels = 3
        
        if self.license_plates != None and len(self.license_plates) > 0:
            point1 = self.license_plates[0].quadrilateral.points[0]
            point2 = self.license_plates[0].quadrilateral.points[1]
            point3 = self.license_plates[0].quadrilateral.points[2]
            point4 = self.license_plates[0].quadrilateral.points[3]
            cv2.line(image.data, (point1.x, point1.y), (point2.x, point2.y), (255, 0, 0), 4)
            cv2.line(image.data, (point2.x, point2.y), (point3.x, point3.y), (255, 0, 0), 4)
            cv2.line(image.data, (point3.x, point3.y), (point4.x, point4.y), (255, 0, 0), 4)
            cv2.line(image.data, (point4.x, point4.y), (point1.x, point1.y), (255, 0, 0), 4)
            
        image = image.resize(self.resize_width, self.resize_width * image.height / image.width)
        image.plot()       
        
if __name__ == '__main__':   
    '''
    Detect license plate in image.
    '''
    
    # Parses args.
    arg_parser = ArgumentParser(description='Detect license plate.')
    arg_parser.add_argument('-c', '--config', dest='config_file', default='config.ini', help='Configuration file')
    arg_parser.add_argument('-i', '--image', dest='image_path', help='Image path', required=True)
    args = vars(arg_parser.parse_args())
    image_path = args['image_path']
    
    # Parses configuration file.
    config_parser = SafeConfigParser()
    config_parser.read(args['config_file'])
    cascade_file = config_parser.get('data', 'path_classifier')
    path_svm_hog_detector = config_parser.get('data', 'path_svm_hog_detector')
    path_svm_binary_detector = config_parser.get('data', 'path_svm_binary_detector')
    image_width = int(config_parser.get('training', 'image_width'))
    image_height = int(config_parser.get('training', 'image_height'))
        
    # Read image.
    image = Image(image_path)
    
    # Detect license plates.
    detector = Detector(cascade_file, path_svm_hog_detector, path_svm_binary_detector, image_width, image_height) 
    detector.detect_license_plates(image)
    detector.plot_quadrilateral(image)