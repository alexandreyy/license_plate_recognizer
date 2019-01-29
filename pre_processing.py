'''
Created on 01/05/2015

@author: Alexandre Yukio Yamashita
'''
from ConfigParser import SafeConfigParser
from argparse import ArgumentParser
import ntpath
import os
from subprocess import call

from lib.mergevec import merge_vec_files
from models.config_data import ConfigData
from models.files import Files
from models.image import Image
from models.logger import Logger
from models.point import Point
from models.quadrilaterals import Quadrilaterals


def crop_image(image, quadrilateral, width_dest, height_dest):
    '''
    Crop image using parameters of quadrilateral.
    '''  
    # Find crop parameters.
    logger = Logger()
    logger.log(Logger.INFO, "Finding crop parameters.")
    origin = Point(image.width, image.height)
    end = Point(0, 0)
    
    for point in quadrilateral.points:
        if point.x > end.x:
            end.x = point.x
        
        if point.y > end.y:
            end.y = point.y
                            
        if point.x < origin.x:
            origin.x = point.x
                        
        if point.y < origin.y:
            origin.y = point.y
    
    # Adjust dimensions to height be 20% of width.
    scale = height_dest * 1.0 / width_dest
    height = end.y -origin.y
    width = end.x -origin.x
    
    # Check if we will change height or width.
    if width*scale < height:
        # We need to increase width.
        increment = height*2.5 -width        
        end.x += increment/2
        
        if end.x >= image.width:
            increment -= image.width -1 -end.x
            end.x = image.width -1
        else:
            increment /= 2
        
        origin.x -= increment   
        
        if origin.x < 0:
            increment = -origin.x
            origin.x = 0            
            end.x += increment
            
            if end.x >= image.width:
                end.x = image.width -1
        
        logger.log(Logger.INFO, "Increasing width in " + str(end.x -origin.x -width))
    else:
        # We need to increase height.
        increment = width*scale -height        
        end.y += increment/2
        
        if end.y >= image.height:
            increment -= image.height -1 -end.y
            end.y = image.height -1
        else:
            increment /= 2
        
        origin.y -= increment   
        
        if origin.y < 0:
            increment = -origin.y
            origin.y = 0            
            end.y += increment
            
            if end.y >= image.height:
                end.y = image.height -1
        
        logger.log(Logger.INFO, "Increasing height in " + str(end.y -origin.y -height))       
    
    
    # Crop image using points from configuration data.
    cropped = image.crop(Point(origin.x, origin.y), Point(end.x, end.y))
    
    # Resize image.
    resized = cropped.resize(width_dest, height_dest)
    
    return resized

def save_training_images_paths(path_pre_processed_positive, path_pre_processed_negative):
    '''
    Save detector_training image paths in txt files.
    '''
    positive_training_images = Files(path_pre_processed_positive)
    negative_training_images = Files(path_pre_processed_negative)
    
    positive_path = "positive.txt"
    logger = Logger()
    logger.log(Logger.INFO, "Saving positive paths in: " + positive_path)
    
    positive_file = open(positive_path, "w")
    for file_path in positive_training_images.paths:
        base_name = ntpath.basename(file_path)
        save_path = path_pre_processed_positive + base_name
        positive_file.write(save_path + "\n")
    positive_file.close()
    
    negative_path = "negative.txt"
    logger.log(Logger.INFO, "Saving negative paths in: " + negative_path)
    
    negative_file = open(negative_path, "w")
    for file_path in negative_training_images.paths:
        base_name = ntpath.basename(file_path)
        save_path = path_pre_processed_negative + base_name
        negative_file.write(save_path + "\n")
    negative_file.close()
    
def pre_processing(argv):
    '''
    Pre process image files to train detector.
    '''
    # Parses args.
    arg_parser = ArgumentParser(description='Load and plot image.')
    arg_parser.add_argument('-c', '--config', dest='config_file', default='config.ini', help='Configuration file')
    args = vars(arg_parser.parse_args())
    
    # Parses configuration file.
    config_parser = SafeConfigParser()
    config_parser.read(args['config_file'])    
    path_original_positive = config_parser.get('data', 'path_original_positive')
    path_original_negative = config_parser.get('data', 'path_original_negative')
    path_pre_processed_positive = config_parser.get('data', 'path_pre_processed_positive')
    path_pre_processed_negative = config_parser.get('data', 'path_pre_processed_negative')
    path_training_vec = config_parser.get('data', 'path_training_vec')
    total_positive_files = config_parser.get('training', 'total_positive_files')
    image_width = config_parser.get('training', 'image_width')
    image_height = config_parser.get('training', 'image_height')
    training_width = config_parser.get('training', 'training_width')
    training_height = config_parser.get('training', 'training_height')
   
    # Removing old files.
    logger = Logger()
    logger.log(Logger.INFO, "Removing old files.")
    files = Files(path_pre_processed_positive, True)
    files.remove()
    files = Files(path_pre_processed_negative, True)
    files.remove()
    files = Files(path_training_vec, True)
    files.remove()
     
    # Pre process positive images.
    logger.log(Logger.INFO, "Pre-processing positive images.")
    positive_images = Files(path_original_positive)
     
    for file_path in positive_images.paths:
        # Load image and configuration data.
        image = Image(file_path)
        config_data = ConfigData(Quadrilaterals(), file_path)
        config_data.read_data()
       
        # Convert and equalize image.
        image.convert_to_gray()
        image.equalize()
         
        # Get file name and extension.
        base_name = os.path.splitext(ntpath.basename(file_path))
        file_name = base_name[0]
        extension = base_name[1]
         
        index_quadrilateral = 0
        for quadrilateral in config_data.quadrilaterals.data:
            # Crop image using configuration data.         
            license_plate = crop_image(image, quadrilateral, int(image_width), int(image_height))
             
            # Save image.
            save_image_path = os.path.abspath(path_pre_processed_positive + file_name + "_" + str(index_quadrilateral) + extension)
            license_plate.save(save_image_path)
             
            index_quadrilateral += 1
     
    # Pre process negative images.
    logger.log(Logger.INFO, "Pre-processing negative images.")
    negative_images = Files(path_original_negative)
      
    for file_path in negative_images.paths:
        # Load image.
        image = Image(file_path)
          
        # Convert and equalize image.
        image.convert_to_gray()
        image.equalize()
          
        # Get base name.
        base_name = ntpath.basename(file_path)
          
        # Save image.
        save_image_path = os.path.abspath(path_pre_processed_negative + base_name)
        image.save(save_image_path)
      
    # Save detector_training image paths in txt files.
    save_training_images_paths(path_pre_processed_positive, path_pre_processed_negative)
      
    # Generating samples.
    logger.log(Logger.INFO, "Generating samples.")
    call(["perl", 'lib/generate_samples.pl', 
          'positive.txt', 
          'negative.txt', 
          path_training_vec, str(int(total_positive_files) +500), 
          'opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1\ -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w ' + str(training_width) + ' -h ' + str(training_height)])
     
    # Merging data for detector_training.
    path_training_vec_file = "positive.vec"
    logger.log(Logger.INFO, "Merging data for detector_training in: " + path_training_vec_file)
    merge_vec_files(path_training_vec, path_training_vec_file)
    
if __name__ == '__main__':
    '''
    Pre process image files to train detector.
    '''
    
    import sys
    pre_processing(sys.argv)