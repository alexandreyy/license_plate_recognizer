'''
Created on 17/07/2015

@author: Alexandre Yukio Yamashita
'''
from ConfigParser import SafeConfigParser
from argparse import ArgumentParser
import numpy as np

if __name__ == '__main__': 
    '''
    Merge character data.
    '''
    
    # Parses args.
    arg_parser = ArgumentParser(description='Merge character data.')
    arg_parser.add_argument('-c', '--config', dest='config_file', default='config.ini', help='Configuration file')
    args = vars(arg_parser.parse_args())
     
    # Parses configuration file.
    config_parser = SafeConfigParser()
    config_parser.read(args['config_file'])    
    path_letter_images = config_parser.get('data', 'path_letter_images')
    path_letter_labels = config_parser.get('data', 'path_letter_labels')
    path_number_images = config_parser.get('data', 'path_number_images')
    path_number_labels = config_parser.get('data', 'path_number_labels')

    number_images_1 = np.loadtxt("resources/ocr/number_images_1.data", np.uint8)
    number_labels_1 = np.loadtxt("resources/ocr/number_labels_1.data", np.float32)
    letter_images_1 = np.loadtxt("resources/ocr/letter_images_1.data", np.uint8)
    letter_labels_1 = np.loadtxt("resources/ocr/letter_labels_1.data", np.float32)
    
    number_images_2 = np.loadtxt("resources/ocr/number_images_2.data", np.uint8)
    number_labels_2 = np.loadtxt("resources/ocr/number_labels_2.data", np.float32)
    letter_images_2 = np.loadtxt("resources/ocr/letter_images_2.data", np.uint8)
    letter_labels_2 = np.loadtxt("resources/ocr/letter_labels_2.data", np.float32)
    
    number_images = np.concatenate((number_images_1, number_images_2), axis=0)
    number_labels = np.concatenate((number_labels_1, number_labels_2), axis=0)
    letter_images = np.concatenate((letter_images_1, letter_images_2), axis=0)
    letter_labels = np.concatenate((letter_labels_1, letter_labels_2), axis=0)
     
    np_number_images = np.array(number_images, np.uint8)
    np_number_labels = np.array(number_labels, np.float32)
    np.savetxt(path_number_images, number_images)
    np.savetxt(path_number_labels, number_labels)
                         
    np_letter_images = np.array(letter_images, np.uint8)
    np_letter_labels = np.array(letter_labels, np.float32)
    np.savetxt(path_letter_images, letter_images)
    np.savetxt(path_letter_labels, letter_labels)

    