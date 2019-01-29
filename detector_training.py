'''
Created on 28/04/2015

@author: Alexandre Yukio Yamashita
         Flavio Nicastro
'''
from ConfigParser import SafeConfigParser
from argparse import ArgumentParser
from subprocess import call

from models.files import Files
from models.logger import Logger
from pre_processing import pre_processing


def training(argv):
    '''
    Train license plate detector.
    '''
    # Parses args.
    arg_parser = ArgumentParser(description='Load and plot image.')
    arg_parser.add_argument('-c', '--config', dest='config_file', default='config.ini', help='Configuration file')
    args = vars(arg_parser.parse_args())
    
    # Parses configuration file.
    config_parser = SafeConfigParser()
    config_parser.read(args['config_file'])
    path_pre_processed_negative = config_parser.get('data', 'path_pre_processed_negative')
    total_negative_files = len(Files(path_pre_processed_negative).paths)
    negative_file = "negative.txt"
    positive_file = "positive.vec"
    reserved_memory = str(config_parser.get('training', 'reserved_memory'))
    total_positive_files = config_parser.get('training', 'total_positive_files')
    training_width = config_parser.get('training', 'training_width')
    training_height = config_parser.get('training', 'training_height')
    detector_command = "opencv_traincascade -data classifier -vec " + positive_file + " -bg " + negative_file + " -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos " + str(total_positive_files) + " -numNeg " + str(total_negative_files) + " -w " + str(training_width) + " -h " + str(training_height) + " -mode ALL -precalcValBufSize " + reserved_memory + " -precalcIdxBufSize " + reserved_memory
    
    # Create script to train classifier.
    logger = Logger()
    logger.log(Logger.INFO, "Creating script training.sh")
    training_path = "training.sh"
    training_file = open(training_path, "w")
    training_file.write("#!/bin/bash\n")
    training_file.write(detector_command)
    training_file.close()
    
    # Training classifier.    
    logger.log(Logger.INFO, "Start trainning.")
    call(["sh", training_path])
     
if __name__ == '__main__':
    import sys
    
    # Pre process image files for trainning.
    pre_processing(sys.argv)
    
    # Train license plate detector.
    training(sys.argv)