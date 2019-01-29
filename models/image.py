'''
Created on 05/04/2015

@author: Alexandre Yukio Yamashita
         Flavio Nicastro
'''

from argparse import ArgumentParser
import cv2
import os

from models.logger import Logger
from models.rect import Rect
import numpy as np


class Image:
    '''
    Reads and process image. 
    '''
    
    _logger = Logger()
    data = None  # The image data.
    
    def __init__(self, file_path=None, image=None):
        # Create image from matrix.
        if image is not None:
            self._set_image_data(image)
        
        # Load image if user specified file path.
        elif file_path is not None:
            # Check if file exist.
            if os.path.isfile(file_path):
                # File exists.
                file_path = os.path.abspath(file_path)
            
                # Load image                
                self._logger.log(Logger.INFO, "Loading image " + file_path)
                image_data = cv2.imread(file_path)
                self.file_path = file_path
                
                # If image is in bgr, convert it to rgb.
                if len(image_data.shape) == 3:
                    # Image is in bgr.
                    
                    # Convert image to rgb.
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                
                self._set_image_data(image_data)
            else:
                # File does not exist.
                self._logger.log(Logger.ERROR, "File '" + file_path + "' does not exist.")
    
    def _set_image_data(self, data):
        '''
        Set image data.
        '''
        # Check if image is in rgb or gray scale
        if len(data.shape) == 3:
            # Image is in rgb.
            self.height, self.width, self.channels = data.shape
        else:
            # Image is in gray scale.
            self.height, self.width = data.shape
            self.channels = 1
        
        self.data = data
    
    def filter_median(self, image=None, size=5):
        '''
        Filter image with median.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to filter image.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before filtering.")
                self.convert_to_gray(self.data)
                
            # Filter image.
            self.data = self.data = cv2.medianBlur(self.data, size)            
            
        return self.data
    
    def filter_gaussian_blur(self, image=None, size=5):
        '''
        Filter image with gaussian blur.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to filter image.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before filtering.")
                self.convert_to_gray(self.data)
                
            # Filter image.
            self.data = cv2.GaussianBlur(self.data, (size, size), 0)            
            
        return self.data
    
    def smart_equalize(self, image=None):
        '''
        Equalize image if it's too dark or too light.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to equalize image.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before equalization.")
                self.convert_to_gray(self.data)
                
            # Equalize image if it is too light or too dark.
            mean = np.mean(self.data)
    
            if mean < 70 or mean > 100:
                if mean > 200:
                    self.contrast()
                
                if mean < 20:
                    self.data = self.data.astype(int)
                    self.data = self.data * 2
                    self.data = np.uint8(self.data)
                    
                self.equalize()
                
        return self.data
    
    def equalize(self, image=None):
        '''
        Equalize image.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to equalize image.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before equalization.")
                self.convert_to_gray(self.data)
                
            # Equalize image.
            self._logger.log(Logger.INFO, "Equalizing image.")
            self.data = cv2.equalizeHist(self.data)
            
        return self.data
    
    def convert_to_gray(self, image=None):
        '''
        Convert rgb to gray.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to convert image.")
        elif image is not None:
            self._set_image_data(image)
        
        if self.data is not None:
            # Convert image only if it is in rgb.
            if len(self.data.shape) == 3:
                self._logger.log(Logger.INFO, "Converting image to gray scale.")
                self.data = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
            else:
                self._logger.log(Logger.INFO, "Image is already in gray scale.")
                
            self.channels = 1            
        
        return self.data
    
    def plot(self, image=None):
        '''
        Plot image.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to plot image.")
        elif image is not None:
            self._set_image_data(image)
            
        # Display image if we have data for it.
        if self.data is not None:
            # Convert image to BGR, if it is in rgb.
            if self.channels == 3:
                image = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
            else:
                image = self.data
            
            # Plot image.    
            self._logger.log(Logger.INFO, "Plotting image.")
            cv2.imshow("Image", image)
            return cv2.waitKey()
    
    def resize(self, width, height, image=None):
        '''
        Resize image.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to resize image.")
        elif image is not None:
            self._set_image_data(image)
            
        self._logger.log(Logger.INFO, "Resizing image to: width = " + str(width) + " height = " + str(height))
        
        try:
            if width > 0 and height > 0:
                resized = cv2.resize(self.data, (width, height), interpolation = cv2.INTER_AREA)
            elif self.width > 0 and self.height > 0:
                resized = np.zeros((self.width, self.height), dtype=np.uint8)
            else:
                resized = np.zeros((400, 200), dtype=np.uint8)
        except:
            if width > 0 and height > 0:
                resized = np.zeros((width, height), dtype=np.uint8)
            elif self.width > 0 and self.height > 0:
                resized = np.zeros((self.width, self.height), dtype=np.uint8)
            else:
                resized = np.zeros((400, 200), dtype=np.uint8)
        
        return Image(image = resized)
    
    def crop(self, origin, end, image=None):
        '''
        Crop image.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to crop image.")
        elif image is not None:
            self._set_image_data(image)
        
        if self.data is not None:
            # Correct parameters.
            if origin.x >= self.width:
                origin.x = self.width - 1
            elif origin.x < 0:
                origin.x = 0
            
            if end.x >= self.width:
                end.x = self.width - 1
            elif end.x < 0:
                end.x = 0
            
            if origin.y >= self.height:
                origin.y = self.height - 1
            elif origin.y < 0:
                origin.y = 0
            
            if end.y >= self.height:
                end.y = self.height - 1
            elif end.y < 0:
                end.y = 0
            
            if origin.x > end.x:
                change = end.x
                end.x = origin.x
                origin.x = change
                       
            if origin.y > end.y:
                change = end.y
                end.y = origin.y
                origin.y = change
                
            self._logger.log(Logger.INFO, "Cropping image. Origin: (%d, %d) End: (%d, %d)" \
                % (origin.x, origin.y, end.x, end.y))
            return Image(image = self.data[origin.y:end.y, origin.x:end.x])
    
        
    def invert_binary(self, image=None):
        '''
        Invert binary image.
        '''
        
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to invert image.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to invert it.")
                self.convert_to_gray(self.data)
                self.binary(self.data)
                
            # Invert binary image.
            self._logger.log(Logger.INFO, "Invert binary image.")
            self.data = cv2.bitwise_not(self.data)
        
        return self.data
      
    def compute_morphologic(self, image=None, morphologic=1):
        '''
        Apply morphologic operation in image.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to apply morphologic operation.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before applying morphologic operation.")
                self.convert_to_gray(self.data)
                self.binary(self.data)
                
            # Apply morphologic operation.
            self._logger.log(Logger.INFO, "Apply morphologic operation in image.")
            
            if morphologic == 1:
                self.data = cv2.bitwise_not(self.data)
                element_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                element_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                
                # Erode and dilate.
                temp = self.data
                temp = cv2.erode(temp, element_3)
                temp = cv2.dilate(temp, element_3)
                
                # Dilate and erode.
                temp = cv2.erode(temp, element_3)
                self.data = cv2.dilate(temp, element_5)
                
                self.data = cv2.bitwise_not(self.data)
            else:
                self.data = cv2.bitwise_not(self.data)
                element_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                element_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                
                # Erode and dilate.
                temp = self.data
                temp = cv2.erode(temp, element_5)
                temp = cv2.dilate(temp, element_5)
                
                # Dilate and erode.
                temp = cv2.erode(temp, element_5)
                self.data = cv2.dilate(temp, element_5)
                
                self.data = cv2.bitwise_not(self.data)
                
        return self.data
    
    def compute_rectangles_for_characters(self, image=None):
        '''
        Compute rectangles in image.
        '''        
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to compute rectangles.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before computing rectangles.")
                self.convert_to_gray(self.data)
            
            thresh = cv2.Canny(self.data, 1 ,100)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                ratio = w * 1.0 / h
                
                if ratio > 0.15 and ratio < 1.1 and w*h > 400 and w*h < 7000 and w > 4 and h > 20:
                    filtered_contours.append(Rect([x, y, w, h]))
                
                if w*h < 400:
                    cv2.fillPoly(self.data, pts =[contour], color=(255))
                
                if w*h > 7000:
                    cv2.floodFill(self.data, None, (x , y), 255)
                    
            contours = filtered_contours
            filtered_contours = []
            
            for contour1 in contours:
                inside = False
                
                for contour2 in contours:
                    if contour1.inside(contour2) and contour1.x != contour2.x and contour1.y != contour2.y and \
                       contour1.w != contour2.w and contour1.h != contour2.h:
                        inside = True
                        break
                
                if not inside:
                    exists = False
                        
                    for contour2 in filtered_contours:
                        if contour1.inside(contour2):
                            exists = True
                            break
                    
                    if not exists:
                        filtered_contours.append(contour1)
  
        return filtered_contours
    
    def plot_rectangles(self, rectangles, image=None):
        '''
        Plot rectangles in image.
        '''        
    
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to plot rectangles.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before plotting rectangles.")
                self.convert_to_gray(self.data)
            
            # Draw rectangles.
            image_data = cv2.cvtColor(self.data, cv2.COLOR_GRAY2RGB)
            
            for contour in rectangles:    
                cv2.rectangle(image_data, (contour.x, contour.y), (contour.x + contour.w, contour.y + contour.h), (0, 255, 0), 2)
            
            image = Image(image=image_data)
            return image.plot()
            
    def binarize(self, image=None, adaptative=False):
        '''
        Binarize image.
        '''
        
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to binarize image.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before binarizing image.")
                self.convert_to_gray(self.data)
            
            # Convert to binary.
            _, data2 = cv2.threshold(self.data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if adaptative:
                data1 = cv2.adaptiveThreshold(self.data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2) 
                data1 = cv2.bitwise_not(data1)
                data2 = cv2.bitwise_not(data2)
                self.data = cv2.bitwise_and(data1, data2)
                self.data = cv2.bitwise_not(self.data)
            else:
                self.data = data2
                  
            for x in range(self.width / 3, self.width * 2 / 3):
                if x % 5 == 0:
                    cv2.floodFill(self.data, None, (x , 0), 255)
                    cv2.floodFill(self.data, None, (x , self.height - 1), 255)
                      
        return self.data

    def contrast(self, image=None):
        '''
        Constrast image.
        '''
        
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to contrast image.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before contrasting image.")
                self.convert_to_gray(self.data)
                
            alpha = 2
            self.data = self.data.astype(int)
            self.data = np.power(self.data, alpha)
            self.data = np.multiply(self.data, 1.0 / 255 ** (alpha -1))
            self.data = np.uint8(self.data)                
        return self.data
    
    def compute_edges(self, image=None):
        '''
        Compute edges.
        '''
        
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to compute edges.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before computing edges.")
                self.convert_to_gray(self.data)
            
            self.data = cv2.Canny(self.data,1 ,100)

        return self.data
    
    def compute_rectangles_for_plates(self, image=None):
        '''
        Compute rectangles in image.
        '''
        
        rectangles = []
        
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to compute rectangles.")
        elif image is not None:
            self._set_image_data(image)
            
        if self.data is not None:
            if len(self.data.shape) == 3:                
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before computing rectangles.")
                self.convert_to_gray(self.data)
                
            thresh = cv2.Canny(self.data,1 ,100)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:    
                x, y, w, h = cv2.boundingRect(contour)
                
                min_size = 50
                ratio = w * 1.0 / h
                 
                if ratio > 1.8 and ratio < 5.0 and w > min_size and h > min_size:        
                    rect = Rect([x, y, w, h])
                    rect.haar_recognized = False
                    rectangles.append(rect)
    
        return rectangles
    
    def save(self, file_path=None, image=None):
        '''
        Save image in file path.
        '''
        
        if file_path is None and self.file_path is None:
            self._logger.log(Logger.ERROR, "There is no file path to save image.")
        elif file_path is not None:
            self.file_path = file_path
            
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to save image.")
        elif image is not None:
            self._set_image_data(image)
        
        if self.file_path is not None and self.data is not None:
            image = self.data
            
            if len(self.data.shape) == 3:
                image = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
                        
            self._logger.log(Logger.INFO, "Saving image in " + self.file_path)
            cv2.imwrite(self.file_path, image);
        
if __name__ == '__main__':
    '''
    Load and plot image.
    '''
    
    # Parses args.
    parser = ArgumentParser(description='Load and plot image.')
    parser.add_argument('file_path', help='image file path')
    args = vars(parser.parse_args())
    
    # Load and plot image.
    image = Image(args["file_path"])
    image.plot()
