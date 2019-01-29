'''
Created on 08/07/2015

@author: Alexandre Yukio Yamashita
         Flavio Nicastro
'''
from ConfigParser import SafeConfigParser
from argparse import ArgumentParser
from curses.ascii import isalpha
import cv2
from numpy.core.defchararray import isdigit

from models.image import Image
from models.logger import Logger
import numpy as np


class CharacterRecognizer:
    '''
    Recognize character from binary image.
    '''
    
    def __init__(self, character_width=15, character_height=15, classifier_type="svm_linear"):
        self.character_width = character_width
        self.character_height = character_height
        self.classifier_type = classifier_type
        self.resize = 128
        
        if classifier_type == "knn":
            self.classifier = cv2.KNearest()
        else:
            self.classifier = cv2.SVM()
            
        if self.classifier_type == "svm_linear_hog":                
            winSize = (128,128)
            blockSize = (32,32)
            blockStride = (16,16)
            cellSize = (8,8)
            nbins = 9
            derivAperture = 1
            winSigma = 4.
            histogramNormType = 0
            L2HysThreshold = 2.0000000000000001e-01
            gammaCorrection = 0
            nlevels = 32
            self.hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                    histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        
    def _pre_process(self, image):
        if self.classifier_type == "svm_linear_hog":     
            image = image.resize(self.resize, self.resize)
            image.invert_binary()
            image.filter_gaussian_blur(size=49)
            #image.plot()
        else:
            image = image.resize(self.character_width, self.character_height)
            image.invert_binary()
            
        return image
    
    def train(self, samples, responses):
        '''
        Train classifier.
        '''
        
        # Get binaries to train
        binaries = []
        
        for sample in samples:
            image = Image(image=sample.data)
            image = self._pre_process(image)
            
            if self.classifier_type == "svm_linear_hog":                
                hog_feature = self.hog.compute(image.data)
                binaries.append(hog_feature)
            else:
                binaries.append(image.resize(self.character_width, self.character_height))
        
        if self.classifier_type != "svm_linear_hog":                
            binaries = np.array([np.array(image.data.flatten(), dtype=np.float32) for image in binaries])
        else:
            binaries = np.array([np.array(image.flatten(), dtype=np.float32) for image in binaries])
            
        # Train classifier.
        responses = np.array(responses)
        
        if self.classifier_type == "knn":
            self.samples = binaries
            self.labels = responses
            self.classifier.train(binaries, responses)
        elif self.classifier_type == "svm_rbf":
            params = dict(kernel_type=cv2.SVM_RBF,
                          svm_type=cv2.SVM_C_SVC,
                          C=1)
            self.classifier.train(binaries, responses, params=params)
        else:
            params = dict(kernel_type=cv2.SVM_LINEAR,
                          svm_type=cv2.SVM_C_SVC,
                          C=1)
            self.classifier.train(binaries, responses, params=params)
            
    def load(self, trained_classifier_file, path_images="", path_labels=""):
        '''
        Load configuration from file.
        '''
        
        if self.classifier_type != "knn":
            self.classifier.load(trained_classifier_file)
        else:
            self.samples = np.loadtxt(path_images, np.float32)
            self.labels = np.loadtxt(path_labels, np.float32)
            self.classifier.train(self.samples, self.labels)
            
    def predict(self, sample, is_letter=False):
        '''
        Predict sample label.
        '''
        
        image = Image(image=sample.data)
        image = self._pre_process(image)
        
        if self.classifier_type == "svm_linear_hog":
            image = self.hog.compute(image.data)
            image = np.array(image.flatten(), dtype=np.float32)
        else:
            image = np.array(image.data.flatten(), dtype=np.float32)
        
        if self.classifier_type != "knn":
            label = self.classifier.predict(image)
        else:
            _, results, _, _ = self.classifier.find_nearest(np.array([image]), k = 1)
            label = results[0][0]
                        
        label = chr(int(label))
        
        return label    
    
    def save(self, trained_classifier_file, path_images="", path_labels=""):
        '''
        Save configuration.
        '''
        if self.classifier_type != "knn":
            self.classifier.save(trained_classifier_file)
        else:
            np.savetxt(path_images, self.samples)
            np.savetxt(path_labels, self.labels)
            
if __name__ == '__main__': 
    '''
    Train letter and number recognizers.
    '''
    
    # Parses args.
    arg_parser = ArgumentParser(description='Train character recognizer.')
    arg_parser.add_argument('-c', '--config', dest='config_file', default='config.ini', help='Configuration file')
    args = vars(arg_parser.parse_args())
    
    # Parses configuration file.
    config_parser = SafeConfigParser()
    config_parser.read(args['config_file'])
    character_width = int(config_parser.get('training', 'character_width'))
    character_height = int(config_parser.get('training', 'character_height'))
    character_original_width = int(config_parser.get('training', 'character_original_width'))
    character_original_height = int(config_parser.get('training', 'character_original_height'))
    path_letter_images = config_parser.get('data', 'path_letter_images')
    path_letter_labels = config_parser.get('data', 'path_letter_labels')
    path_letter_classifier = config_parser.get('data', 'path_letter_classifier')
    path_number_images = config_parser.get('data', 'path_number_images')
    path_number_labels = config_parser.get('data', 'path_number_labels')
    path_number_classifier = config_parser.get('data', 'path_number_classifier')
    path_number_knn_labels_classifier = config_parser.get('data', 'path_number_knn_labels_classifier')
    path_number_knn_images_classifier = config_parser.get('data', 'path_number_knn_images_classifier')
    path_letter_knn_labels_classifier = config_parser.get('data', 'path_letter_knn_labels_classifier')
    path_letter_knn_images_classifier = config_parser.get('data', 'path_letter_knn_images_classifier')
    character_classifier_type = config_parser.get('data', 'character_classifier_type')
    
    # Load images and labels.
    logger = Logger()
    logger.log(Logger.INFO, "Loading images to train classifiers.")
   
    number_images = np.loadtxt(path_number_images, np.uint8)
    number_labels = np.loadtxt(path_number_labels, np.float32)
    number_labels = number_labels.reshape((number_labels.size, 1))                
    converted_images = []
    labels = []
    
    for index in range(len(number_images)):
        image = number_images[index]
        reshaped = Image(image=image.reshape((character_original_height, character_original_width)))
        reshaped.binarize(adaptative=True)
        mean_value = np.mean(reshaped.data)
        
        if mean_value < 220:
            if isdigit(chr(int(number_labels[index]))):
                converted_images.append(reshaped)
                labels.append(number_labels[index])
                          
    number_images = converted_images
    number_labels = labels
    
    letter_images = np.loadtxt(path_letter_images, np.uint8)
    letter_labels = np.loadtxt(path_letter_labels, np.float32)
    letter_labels = letter_labels.reshape((letter_labels.size, 1))
    converted_images_l = []
    labels_l = []
    
    for index in range(len(letter_images)):
        image = letter_images[index]
        reshaped = Image(image=image.reshape((character_original_height, character_original_width)))
        reshaped.binarize(adaptative=True)
        mean_value = np.mean(reshaped.data)
        
        if mean_value < 220:
            if isalpha(chr(int(letter_labels[index]))):
                converted_images_l.append(reshaped)
                labels_l.append(letter_labels[index])
            
    letter_images = converted_images_l
    letter_labels = labels_l
    
    # Train classifiers.
    logger.log(Logger.INFO, "Training letter classifier.")
    letter_classifier = CharacterRecognizer(character_width, character_height, character_classifier_type)
    letter_classifier.train(letter_images, letter_labels)
    
    if letter_classifier.classifier_type != "knn":
        letter_classifier.save(path_letter_classifier)
    else:
        letter_classifier.save("", path_letter_knn_images_classifier, path_letter_knn_labels_classifier)
        
    logger.log(Logger.INFO, "Training number classifier.")
    number_classifier = CharacterRecognizer(character_width, character_height, character_classifier_type)
    number_classifier.train(number_images, number_labels)
    
    if number_classifier.classifier_type != "knn":
        number_classifier.save(path_number_classifier)
    else:
        number_classifier.save("", path_number_knn_images_classifier, path_number_knn_labels_classifier)
