'''
Created on 06/07/2015

@author: Alexandre Yukio Yamashita
'''
import cv2

from models.rect import Rect


class DetectorHaar:
    '''
    Detector based on opencv cascade classifier.
    '''
    
    cascade_classifier = None
    
    def __init__(self, cascade_file):
        self.cascade_classifier = cv2.CascadeClassifier(cascade_file)
    
    def _to_rects(self, cv_results):
        '''
        Convert opencv results to rectangles.
        '''
        return [Rect(result) for result in cv_results]
    
    def detect_using_cascade_classifier(self, image):
        '''
        Detect license plates using cascade classifier.
        '''        
        license_plates = []
        
        for scale in [float(i) / 10 for i in range(11, 15)]:
            for neighbors in range(2, 5):
                license_plates.extend(self._to_rects(self.cascade_classifier.detectMultiScale(image.data, scaleFactor=scale, minNeighbors=neighbors, \
                                                     minSize=(8, 16), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)))
        
        if len(license_plates) > 0:
            license_plates = self._merge_license_plates(license_plates)
        else:
            license_plates = []
        
        return license_plates
    
    def _merge_license_plates(self, license_plates):
        '''
        Merge license plates.
        '''
        input_license_plates = license_plates
        license_plates = []
          
        # if the sizes of these license plates are within .5% of each other, take the
        # one nearest midpoint
        for current_license_plate in input_license_plates:
            new_license_plate = current_license_plate
            
            # Merge license plates.
            for license_plate in input_license_plates:
                if license_plate.intersect(new_license_plate) and abs(new_license_plate.x - license_plate.x) < 5 and abs(new_license_plate.y - license_plate.y) < 5:
                    x_max = max(license_plate.x + license_plate.w, new_license_plate.x + new_license_plate.w)
                    y_max = max(license_plate.y + license_plate.h, new_license_plate.y + new_license_plate.h)
                     
                    x_min = min(license_plate.x, new_license_plate.x)
                    y_min = min(license_plate.y, new_license_plate.y)
                      
                    width = x_max - x_min
                    height = y_max - y_min
                      
                    new_license_plate.x = x_min
                    new_license_plate.y = y_min
                    new_license_plate.w = width
                    new_license_plate.h = height
            
            # Avoid adding twice the same rectangle.
            exists = False
            
            for license_plate in license_plates:    
                if license_plate.equal(new_license_plate):
                    exists = True
                    break
                
            if not exists:
                license_plates.append(new_license_plate)
        
        return license_plates
