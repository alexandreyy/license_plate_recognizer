'''
Created on 02/07/2015

@author: Alexandre Yukio Yamashita
         Flavio Nicastro
'''
from ConfigParser import SafeConfigParser
from argparse import ArgumentParser
import cv2

from models import logger
from models.files import Files
from models.image import Image
from models.logger import Logger
from models.point import Point
import numpy as np


class SVM:
    '''
    SVM classifier based on HOG descriptor and binary image.
    '''
    
    trained_hog_file = None
    trained_binary_file = None
    svm = None
    
    def __init__(self, trained_hog_file=None, trained_binary_file = None):
        self.trained_hog_file = trained_hog_file
        self.trained_binary_file = trained_binary_file
        self.svm_hog = cv2.SVM()
        #self.svm_binary = cv2.SVM()
        
        if trained_hog_file is not None and trained_binary_file is not None:
            self.load(trained_hog_file, trained_binary_file)
                
    def train(self, samples, responses):
        '''
        Train SVM classifier.
        '''
        
        # Set SVM parameters
        params = dict(kernel_type=cv2.SVM_LINEAR,
                       svm_type=cv2.SVM_C_SVC,
                       C=1)
        
        # Get binaries to train
#         binaries = [] 
#         for sample in samples:
#             image = Image(image=sample.data)
#             image.convert_to_gray()
#             image.filter_median(size=3)
#             image.binarize()   
#             binaries.append(image.resize(96, 48))
#             
#         binaries = np.array([np.array(image.data.flatten(), dtype=np.float32) for image in binaries])
#         
        # Convert data to numpy.
        hog = cv2.HOGDescriptor()
        samples = np.array([hog.compute(image.resize(256, 128).data) for image in samples])
        responses = np.array(responses)
        
        # Train SVM.
        self.svm_hog.train(samples, responses, params=params)
        #self.svm_binary.train(binaries, responses, params=params)

    def predict(self, sample):
        '''
        Predict sample label.
        '''
#         image = sample.resize(96, 48)
#         image.convert_to_gray()
#         image.filter_median(size=3)
#         image.binarize()
#         image = np.array(sample.data.flatten(), dtype=np.float32)
#         binary_detected = self.svm_binary.predict(image)
        binary_detected = True
            
        hog = cv2.HOGDescriptor()
        sample = hog.compute(sample.resize(256, 128).data)
        hog_detected = self.svm_hog.predict(sample)
    
        return binary_detected and hog_detected
    
    def load(self, trained_hog_file, trained_binary_file):
        '''
        Load configuration from file.
        '''
        self.svm_hog.load(trained_hog_file)
        #self.svm_binary.load(trained_binary_file)
    
    def save(self, trained_hog_file, trained_binary_file):
        '''
        Save configuration.
        '''
        
        self.svm_hog.save(trained_hog_file)
        #self.svm_binary.save(trained_binary_file)

    
if __name__ == '__main__':   
    '''
    Train SVM detector.
    '''
    
    # Parses args.
    arg_parser = ArgumentParser(description='Train SVM detector.')
    arg_parser.add_argument('-c', '--config', dest='config_file', default='config.ini', help='Configuration file')
    args = vars(arg_parser.parse_args())
    
    # Parses configuration file.
    config_parser = SafeConfigParser()
    config_parser.read(args['config_file'])
    path_pre_processed_positive = config_parser.get('data', 'path_pre_processed_positive')
    path_pre_processed_negative = config_parser.get('data', 'path_pre_processed_negative')
    path_svm_hog_detector = config_parser.get('data', 'path_svm_hog_detector')
    path_svm_binary_detector = config_parser.get('data', 'path_svm_binary_detector')
    image_width = int(config_parser.get('training', 'image_width'))
    image_height = int(config_parser.get('training', 'image_height'))
    
    # Load images for training.
    logger = Logger()
    logger.log(Logger.INFO, "Loading images to train SVM classifier.")
    positive_images = Files(path_pre_processed_positive)
    negative_images = Files(path_pre_processed_negative)
    samples = []
    responses = []
    
    for file_path in positive_images.paths:
        # Load image.
        image = Image(file_path)
        
        # Convert and equalize image.
        image.convert_to_gray()
        
        # Add image for training.
        samples.append(image)
        responses.append(1)
        
    for file_path in negative_images.paths:
        # Load image.
        image = Image(file_path)
        image.convert_to_gray()
        
        # Add negative images for training.
        for i in range(1):
            x = int(np.random.random() * (image.width - image_width - 1))
            y = int(np.random.random() * (image.height - image_height - 1))
            negative = image.crop(Point(x, y), Point(x + image_width, y + image_height))
            samples.append(negative)
            responses.append(0)
        
    # Train SVM.
    logger.log(Logger.INFO, "Training SVM.")
    svm = SVM()
    svm.train(samples, responses)
    
    # Save SVM trained.
    logger.log(Logger.INFO, "Saving SVM configuration.")
    svm.save(path_svm_binary_detector, path_svm_hog_detector)