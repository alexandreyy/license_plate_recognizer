'''
Created on 11/07/2015

@author: Alexandre Yukio Yamashita
'''
import copy

import models.character_validator
from models.image import Image
from models.point import Point
from models.quadrilateral import Quadrilateral
from models.rect import Rect


def normalize_characters_character_height(license_plate, image_original, character_height, m, c, recrop = True):
    '''
    Normalize height of characters.
    '''
    origin = Point(0, 0)
    end = Point(license_plate.image.width -1, license_plate.image.height -1)
    changed_image = False
    
    for index in range(len(license_plate.subrects)):
        new_y = int(license_plate.subrects[index].x * m + c)
        
        if abs(new_y -license_plate.subrects[index].y) > 5:
            license_plate.subrects[index].y = new_y
            
            if license_plate.subrects[index].y < origin.y:
                origin.y = license_plate.subrects[index].y
                changed_image = True
                
        license_plate.subrects[index].h = character_height
        
        if license_plate.subrects[index].h + license_plate.subrects[index].y -1 > end.y:
            end.y = license_plate.subrects[index].y + license_plate.subrects[index].h -1 + 50
            changed_image = True
    
    if changed_image and recrop:
        # Recrop image and compute rectangles again.
        license_plate = recrop_license_plate_image(license_plate, origin, end, image_original)        
        character_validator = models.character_validator.CharacterValidator()
        backup_image = Image(image=license_plate.image.data)
        image = pre_process_license_plate_image(license_plate.image, adaptative=False)
        license_plate.subrects = image.compute_rectangles_for_characters()
        character_validator.remove_wrong_characters(license_plate)
        
        # If not enough characters detected, binarize image using adaptative method. 
        if len(license_plate.subrects) < 2:
            image = pre_process_license_plate_image(backup_image, True)
            license_plate.subrects = image.compute_rectangles_for_characters()
            character_validator.remove_wrong_characters(license_plate)        
        
        validator = models.character_validator.CharacterValidator()
        m, c = validator.get_least_squares(license_plate)
        license_plate, m, c = normalize_characters_character_height(license_plate, image_original, character_height, m, c, False)
    
    return license_plate, m, c

def normalize_characters_character_width(license_plate, image_original, character_width, recrop=True):
    '''
    Normalize width of characters.
    '''

    origin = Point(0, 0)
    end = Point(license_plate.image.width -1, license_plate.image.height -1)
    changed_image = False
    
    for character in license_plate.subrects:
        if character.w < character_width:
            character.x = character.x -(character_width -character.w)/2
            
            if character.x < origin.x:
                origin.x = character.x
                changed_image = True
                
        character.w = character_width
        
        if character.w + character.x -1 > end.x:
            end.x = character.x + character.w -1 + 50
            changed_image = True
    
    if changed_image and recrop:
        # Recrop image and compute rectangles again.
        license_plate = recrop_license_plate_image(license_plate, origin, end, image_original)        
        character_validator = models.character_validator.CharacterValidator()
        backup_image = Image(image=license_plate.image.data)
        image = pre_process_license_plate_image(license_plate.image, adaptative=False)
        license_plate.subrects = image.compute_rectangles_for_characters()
        character_validator.remove_wrong_characters(license_plate)
        
        # If not enough characters detected, binarize image using adaptative method. 
        if len(license_plate.subrects) < 2:
            image = pre_process_license_plate_image(backup_image, True)
            license_plate.subrects = image.compute_rectangles_for_characters()
            character_validator.remove_wrong_characters(license_plate)        
        
        license_plate = normalize_characters_character_width(license_plate, image_original, character_width, False)

    return license_plate

def reescale_license_plate(license_plate, original_image_width, scale_image_width):
    '''
    Reescale license plate to original size.
    '''
    
    # Reescale license plate.
    reescaled_license_plate = Rect([license_plate.x, license_plate.y, license_plate.w, license_plate.h])
    ratio = 1.0 * original_image_width / scale_image_width
    reescaled_license_plate.w = int((reescaled_license_plate.x + reescaled_license_plate.w - reescaled_license_plate.x) * ratio)
    reescaled_license_plate.h = int((reescaled_license_plate.y + reescaled_license_plate.h - reescaled_license_plate.y) * ratio)
    reescaled_license_plate.x = int(reescaled_license_plate.x * ratio)
    reescaled_license_plate.y = int(reescaled_license_plate.y * ratio)
    
    return reescaled_license_plate

def pre_process_license_plate_image(license_plate_image, adaptative=False, morphologic=1):
    '''
    Pre-process license plate image.
    '''
    
    license_plate_image.filter_median(size=3)
    
    if morphologic == 1 or morphologic == 3:
        license_plate_image.binarize(adaptative=adaptative)
        license_plate_image.compute_morphologic(morphologic=morphologic)
        
    return license_plate_image

def recrop_license_plate_image(license_plate, origin, end, image_original, morphologic=1):
    '''
    Recrop image.
    '''
    
    # Reescale crop parameters.
    scale = license_plate.w * 1.0 / license_plate.image.width
    old_width = license_plate.image.width
    
    # Get new width and height. 
    new_width = end.x -origin.x
    new_height = end.y -origin.y
    ratio = license_plate.w * 1.0 / license_plate.h
    
    if int(new_height*ratio + 0.5) > new_width:
        new_width = int(new_height*ratio + 0.5)
    else:
        new_height = int(new_width/ratio + 0.5)
    
    # Find new end.
    end.x = origin.x + new_width -1
    end.y = origin.y + new_height -1
    
    # Compute coordinates in original image.
    origin.x = license_plate.x + int(scale * origin.x + 0.5)
    origin.y = license_plate.y + int(scale * origin.y + 0.5)
    end.x = license_plate.x + int(scale * end.x + 0.5)
    end.y = license_plate.y + int(scale * end.y + 0.5)
    
    # Adjust positions of character.
    for character in license_plate.subrects:
        character.x = int(character.x*scale + 0.5) +license_plate.x -origin.x
        character.y = int(character.y*scale + 0.5) +license_plate.y -origin.y
        character.h = int(character.h*scale + 0.5)
        character.w = int(character.w*scale + 0.5)
        
    # Recrop image.
    license_plate.image = image_original.crop(origin, end)
    scale = old_width*1.0/license_plate.image.width
    license_plate.image = license_plate.image.resize(old_width, int(old_width / ratio))
    license_plate.image = pre_process_license_plate_image(license_plate.image, False, 2)
     
    for character in license_plate.subrects:
        character.x = int(character.x*scale + 0.5)
        character.y = int(character.y*scale + 0.5)
        character.h = int(character.h*scale + 0.5)
        character.w = int(character.w*scale + 0.5)
    
    license_plate.x = origin.x  
    license_plate.y = origin.y
    license_plate.w = end.x -origin.x  
    license_plate.h = end.y -origin.y
        
    return license_plate

def get_license_plate_quadrilateral(best_license_plate):
    '''
    Get license plate with quadrilateral.
    '''
    
    license_plate = copy.deepcopy(best_license_plate)
    
    # Reescale crop parameters.
    scale = license_plate.w * 1.0 / license_plate.image.width
    license_plate.subrects = sorted(license_plate.subrects, key=lambda character: character.x)
    character_validator = models.character_validator.CharacterValidator()
    
    # Adjust positions of character.
    for character in license_plate.subrects:
        character.x = int(character.x*scale + 0.5) +license_plate.x
        character.y = int(character.y*scale + 0.5) +license_plate.y
        character.h = int(character.h*scale + 0.5)
        character.w = int(character.w*scale + 0.5)
    
    quadrilateral = Quadrilateral()
    quadrilateral.add_point(Point(license_plate.subrects[0].x - license_plate.subrects[0].w * 2 / 4, license_plate.subrects[0].y - license_plate.subrects[0].h * 4 / 5))
    quadrilateral.add_point(Point(license_plate.subrects[6].x + license_plate.subrects[0].w * 6 / 4, license_plate.subrects[6].y - license_plate.subrects[0].h * 4 / 5))
    quadrilateral.add_point(Point(license_plate.subrects[6].x + license_plate.subrects[0].w * 6 / 4, license_plate.subrects[6].y + license_plate.subrects[0].h * 5 / 4))
    quadrilateral.add_point(Point(license_plate.subrects[0].x - license_plate.subrects[0].w * 2 / 4, license_plate.subrects[0].y + license_plate.subrects[0].h * 5 / 4))
    best_license_plate.quadrilateral = quadrilateral
    
    return best_license_plate
