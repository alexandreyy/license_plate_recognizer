'''
Created on 06/07/2015

@author: Alexandre Yukio Yamashita.
'''

from models.filter_characters import FilterCharacters
from models.image import Image
from models.license_plate_utils import normalize_characters_character_width, \
    normalize_characters_character_height, recrop_license_plate_image, \
    pre_process_license_plate_image
from models.point import Point
from models.rect import Rect
import numpy as np


class CharacterValidator:
    '''
    Adjust characters found for license plate.
    '''
    
    def _get_x_step(self, license_plate):
        '''
        Get x step to insert new characters.
        '''
        
        if len(license_plate.subrects) > 1:
            if len(license_plate.subrects) < 4:
                # Return mean of widths + bias.
                mean_w = np.mean([rect.w for rect in license_plate.subrects]) 
                return mean_w + mean_w / 15
            else:
                # Calculate min_distances.            
                char_filter = FilterCharacters()
                min_distances, license_plate.subrects = char_filter.calculate_distances_between_characters(license_plate.subrects)
                mean_min_distance = np.mean(min_distances)
                best_interval = 99999999
                 
                # Calculate best_interval based on min_distance.
                for min_distance in min_distances:
                    if abs(mean_min_distance -min_distance) < abs(mean_min_distance -best_interval):
                        best_interval = min_distance
                        
        elif len(license_plate.subrects) == 1:
            best_interval = license_plate.subrects[0].w + license_plate.subrects[0].w/10
        else:
            best_interval = 38
            
        return best_interval
    
    def _get_h_w(self, license_plate):
        '''
        Get height and width to insert and adjust caracters.
        '''
        
        if len(license_plate.subrects) == 1:
            # Return the dimensions of the character detected.
            height = license_plate.subrects[0].h
            width = license_plate.subrects[0].w
        elif len(license_plate.subrects) == 2:
            # Max of heights and widths.
            height = np.mean([license_plate.subrects[0].h, license_plate.subrects[1].h])
            width = np.mean([license_plate.subrects[0].w, license_plate.subrects[1].w])
        elif len(license_plate.subrects) == 3:
            # Find height and width by minimum std.
            std_0_1_h = np.std([license_plate.subrects[0].h, license_plate.subrects[1].h])
            std_0_2_h = np.std([license_plate.subrects[0].h, license_plate.subrects[2].h])
            std_1_2_h = np.std([license_plate.subrects[1].h, license_plate.subrects[2].h])
            std_0_1_w = np.std([license_plate.subrects[0].w, license_plate.subrects[1].w])
            std_0_2_w = np.std([license_plate.subrects[0].w, license_plate.subrects[2].w])
            std_1_2_w = np.std([license_plate.subrects[1].w, license_plate.subrects[2].w])
            
            if std_0_1_w <= std_0_2_w and std_0_1_w <= std_1_2_w:
                width = np.max([license_plate.subrects[0].w, license_plate.subrects[1].w])
            elif std_0_2_w <= std_0_1_w and std_0_2_w <= std_1_2_w:
                width = np.max([license_plate.subrects[0].w, license_plate.subrects[2].w])
            else:
                width = np.max([license_plate.subrects[1].w, license_plate.subrects[2].w])
                
            if std_0_1_h <= std_0_2_h and std_0_1_h <= std_1_2_h:
                height = np.max([license_plate.subrects[0].h, license_plate.subrects[1].h])
            elif std_0_2_h <= std_0_1_h and std_0_2_h <= std_1_2_h:
                height = np.max([license_plate.subrects[0].h, license_plate.subrects[2].h])
            else:
                height = np.max([license_plate.subrects[1].h, license_plate.subrects[2].h])
                
        elif len(license_plate.subrects) > 3:
            # Find height and width by random minimum std.
            best_height_std = 9999999
            best_width_std = 9999999
            best_height = 0
            best_width = 0
            characters = license_plate.subrects
                
            for _ in range(50):
                np.random.shuffle(characters)
                range_w = [characters[index].w for index in range(3)]
                range_h = [characters[index].h for index in range(3)]
                
                current_height_std = np.std(range_h)
                current_width_std = np.std(range_w)
                
                if current_height_std < best_height_std:
                    best_height_std = current_height_std
                    best_height = np.max(range_h)
                
                if current_width_std < best_width_std:
                    best_width_std = current_width_std
                    best_width = np.max(range_w)
            
            height = best_height
            width = best_width
        else:
            height = 58
            width = 35
            
        return int(height), int(width + width / 10)
    
    def get_least_squares(self, license_plate):
        '''
        Get least squares parameters.
        '''
        
        characters = license_plate.subrects
        
        if len(characters) > 2:
            if len(characters) == 3:
                characters_model = characters
            else:
                characters_model = license_plate.best_subrects(3)
            
            # Calculate least squares.
            x = [rect.x for rect in characters_model]
            y = [rect.y for rect in characters_model]
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y)[0]
        elif len(characters) == 2:
            if characters[0].x < characters[1].x:
                x_delta = characters[1].x -characters[0].x
                y_delta = characters[1].y -characters[0].y
            else:
                x_delta = characters[0].x -characters[1].x
                y_delta = characters[0].y -characters[1].y
            
            if x_delta == 0:
                m = 0
            else:
                m = y_delta * 1.0 / x_delta
            
            if m == 0:
                m = 0
                
            c = int(characters[1].y -characters[1].x*m)
        elif len(characters) == 1:
            m = 0
            c = characters[0].y
        else:
            m = 0
            c = license_plate.image.height * 1.0 / 3
        
        return m, c
    
    def _calculate_mean_caracter_values(self, license_plate):
        '''
        Calculate mean of caracter values.
        '''
        
        mean_value = 0
        size = 0
        
        for character in license_plate.subrects:
            character_image = license_plate.image.crop(Point(character.x, character.y), Point(character.x + character.w - 1, character.y + character.h - 1))
            
            if character_image.data.size > 0 and character_image.data is not None:
                mean_value += np.mean(character_image.data)
                size += 1
        
        if size > 0:
            mean_value = int(mean_value/size) 
             
        return mean_value
    
    def _recrop_image(self, license_plate, image_original, width, height, m, c, x_interval):
        '''
        Recrop image if necessary.
        '''
        
        changed_image = False
        origin = Point(0, 0)
        end = Point(license_plate.image.width -1, license_plate.image.height -1)
        
        for character in license_plate.subrects:
            if character.y < origin.y:
                origin.y = character.y
                changed_image = True
            
            if character.x < origin.x:
                origin.x = character.x
                changed_image = True
            
            if character.h + character.y -1 > end.y:
                end.y = character.y + character.h
                changed_image = True
            
            if character.w + character.x -1 > end.x:
                end.x = character.x + character.w
                changed_image = True
             
        if changed_image:
            # Recrop image and compute rectangles again.
            scale = license_plate.w * 1.0 / license_plate.image.width
            license_plate = recrop_license_plate_image(license_plate, origin, end, image_original)
            
            old_image = Image(image=license_plate.image.data)
            image_1 = pre_process_license_plate_image(license_plate.image, True)
            image_2 = pre_process_license_plate_image(old_image, False)
            
            license_plate.image = image_1
            color_mean_1 = self._calculate_mean_caracter_values(license_plate)
            license_plate.image = image_2            
            color_mean_2 = self._calculate_mean_caracter_values(license_plate)
            
            if color_mean_1 < 10 and color_mean_2 > 10:
                color_mean_2 = 119
            elif color_mean_1 > 10 and color_mean_2 < 10: 
                color_mean_1 = 119
            
            if color_mean_1 < 255 and color_mean_2 > 255:
                color_mean_1 = 119
            elif color_mean_1 > 255 and color_mean_2 < 255: 
                color_mean_2 = 119
                    
            t1 = 70
            t2 = 180
             
            color_delta_1 = min(abs(color_mean_1 -t1), abs(color_mean_1 -t2))
            color_delta_2 = min(abs(color_mean_2 -t1), abs(color_mean_2 -t2))
            
            if color_delta_1 < color_delta_2:
                license_plate.image = image_1
            else:
                license_plate.image = image_2
                
            old_list = list(license_plate.subrects)
            
            if len(old_list) > 0:
                height = old_list[0].h
                width = old_list[0].w
                
                if len(old_list) > 2:
                    license_plate.subrects = license_plate.subrects[0:3]
                    m, c = self.get_least_squares(license_plate)
                else:
                    m, c = self.get_least_squares(license_plate)
            else:
                height, width = self._get_h_w(license_plate)
                m, c = self.get_least_squares(license_plate)
            
            x_interval = int(x_interval*scale + 0.5)            
            license_plate.subrects = old_list
            
        #license_plate.image.plot_rectangles(license_plate.subrects)
        
        return license_plate, width, height, m, c, x_interval
    
    def _insert_in_limits(self, license_plate, image_original, width, height, m, c, x_interval):
        '''
        Insert character before or after existent characters.
        '''
        
        license_plate.subrects = sorted(license_plate.subrects, key=lambda character: character.x)        
        new_license_plate_x_1 = license_plate.subrects[0].x -x_interval
        new_license_plate_x_2 = license_plate.subrects[len(license_plate.subrects) -1].x +x_interval
        new_character_1 = Rect([new_license_plate_x_1, int(m * new_license_plate_x_1 + c), width, height])
        new_character_2 = Rect([new_license_plate_x_2, int(m * new_license_plate_x_2 + c) , width, height])
        license_plate.subrects.append(new_character_1)
        license_plate.subrects.append(new_character_2)
        license_plate, width, height, m, c, x_interval = self._recrop_image(license_plate, image_original, width, height, m, c, x_interval)
        
        new_character_1_image = license_plate.image.crop(Point(new_character_1.x, new_character_1.y), Point(new_character_1.x + new_character_1.w - 1, new_character_1.y + new_character_1.h - 1))
        new_character_2_image = license_plate.image.crop(Point(new_character_2.x, new_character_2.y), Point(new_character_2.x + new_character_2.w - 1, new_character_2.y + new_character_2.h - 1))
            
        if new_character_1_image.data.size > 0 and new_character_1_image.data is not None:
            color_mean_1 = np.mean(new_character_1_image.data)
        else:
            color_mean_1 = 0
            
        if new_character_2_image.data.size > 0 and new_character_2_image.data is not None:
            color_mean_2 = np.mean(new_character_2_image.data)
        else:
            color_mean_2 = 0
        
        if color_mean_1 < 10 and color_mean_2 > 10:
            color_mean_2 = 119
        elif color_mean_1 > 10 and color_mean_2 < 10: 
            color_mean_1 = 119
    
        if color_mean_1 < 255 and color_mean_2 > 255:
            color_mean_1 = 119
        elif color_mean_1 > 255 and color_mean_2 < 255: 
            color_mean_2 = 119
            
        t1 = 70
        t2 = 180
         
        color_delta_1 = min(abs(color_mean_1 -t1), abs(color_mean_1 -t2))
        color_delta_2 = min(abs(color_mean_2 -t1), abs(color_mean_2 -t2))
        
        if color_delta_1 < color_delta_2:
            license_plate.subrects.pop()
        else:
            new_character_2 = license_plate.subrects.pop()
            license_plate.subrects.pop()
            license_plate.subrects.append(new_character_2)
        
        #license_plate.image.plot_rectangles(license_plate.subrects)
        
        return license_plate, width, height, m, c, x_interval
    
    def _insert_between_characters(self, license_plate, image_original, width, height, m, c, x_interval):
        '''
        Insert character between existent characters.
        ''' 
         
        license_plate.subrects = sorted(license_plate.subrects, key=lambda character: character.x)
        added = False
        range_character = range(len(license_plate.subrects) - 1)
        
        # Find position between characters,
        for neighbor_character in range_character:
            if neighbor_character == 2 and neighbor_character != len(license_plate.subrects) - 5:
                break
            
            new_license_plate_x = license_plate.subrects[neighbor_character].x + x_interval
            
            if abs(new_license_plate_x - license_plate.subrects[neighbor_character + 1].x) > x_interval / 2:
                new_character = Rect([new_license_plate_x, int(m * new_license_plate_x + c), width, height])
                license_plate.subrects.append(new_character)
                license_plate, width, height, m, c, x_interval = self._recrop_image(license_plate, image_original, width, height, m, c, x_interval)
                new_character_image = license_plate.image.crop(Point(new_character.x, new_character.y), Point(new_character.x + new_character.w - 1, new_character.y + new_character.h - 1))
                                            
                if new_character_image.data.size != 0 and new_character_image.data is not None:
                    color_mean = np.mean(new_character_image.data)
                    
                    if color_mean > 60 and color_mean < 200:
                        return license_plate, True, width, height, m, c, x_interval
                
                license_plate.subrects.pop()
        
        range_index = range(len(license_plate.subrects))
        range_index = range_index[::-1]
        
        for neighbor_character in range_index:
            if neighbor_character == 2 or neighbor_character == len(license_plate.subrects) -3 or neighbor_character == 0:
                break
            
            new_license_plate_x = license_plate.subrects[neighbor_character].x -x_interval
            
            if abs(new_license_plate_x -license_plate.subrects[neighbor_character -1].x) > x_interval/2:
                new_character = Rect([new_license_plate_x, int(m * new_license_plate_x + c), width, height])
                license_plate.subrects.append(new_character)
                license_plate, width, height, m, c, x_interval = self._recrop_image(license_plate, image_original, width, height, m, c, x_interval)

                # Calculate x step.            
                x_interval = self._get_x_step(license_plate)
                height, width = self._get_h_w(license_plate)
                m, c = self.get_least_squares(license_plate)
                
                new_character_image = license_plate.image.crop(Point(new_character.x, new_character.y), Point(new_character.x + new_character.w - 1, new_character.y + new_character.h - 1))
                
                if new_character_image.data.size != 0:
                    color_mean = np.mean(new_character_image.data)
                
                    if color_mean > 60 and color_mean < 200:
                        return license_plate, True, width, height, m, c, x_interval
                    
                license_plate.subrects.pop()
        
        return license_plate, added, width, height, m, c, x_interval
    
    
    def _adjust_characters_too_far(self, license_plate, image_original, width, height, m, c, x_interval):
        '''
        Adjust position of characters too far.   
        '''
         
        # Removes characters too far and add them in other position.
        char_filter = FilterCharacters()
        license_plate.subrects, removed = char_filter.filter_characters_by_distance(license_plate.subrects)
            
        if removed > 0:
            # Adjust license plate to have at least two characters.
            if len(license_plate.subrects) < 2:
                if len(license_plate.subrects) == 0:
                    new_character = Rect([license_plate.image.width/2, license_plate.image.height * 1.0 / 3, width, height])
                    license_plate.subrects.append(new_character)
                    m = 0
                    c = int(license_plate.image.height * 1.0 / 3)
                else:
                    license_plate, width, height, m, c, x_interval = self._insert_in_limits(license_plate, image_original, width, height, m, c, x_interval)
           
            while len(license_plate.subrects) < 7:    
                # Insert new character.
                license_plate, inserted, width, height, m, c, x_interval = self._insert_between_characters(license_plate, image_original, width, height, m, c, x_interval)
                
                if not inserted:
                    license_plate, width, height, m, c, x_interval = self._insert_in_limits(license_plate, image_original, width, height, m, c, x_interval)
              
        return license_plate
    
    def _correct_dot_character(self, license_plate, image_original, width, height, m, c, x_interval):
        '''
        Move dot character to other position.   
        '''
        found_dot = False
        characters = []
        
        for character in license_plate.subrects:
            if character.x < 0:
                character.x = 0
            
            if character.y < 0:
                character.y = 0
            
            if character.x > license_plate.image.width - 1:
                character.x = license_plate.image.width - 1
            
            if character.y > license_plate.image.height - 1:
                character.y = license_plate.image.height - 1
            
            if character.x + character.w > license_plate.image.width - 1:
                character.w = license_plate.image.width - 1 - character.x
            
            if character.y + character.h > license_plate.image.height - 1:
                character.y = license_plate.image.height - 1 - character.y
                        
            if not found_dot:
                character_image = license_plate.image.crop(Point(character.x, character.y), Point(character.x + character.w - 1, character.y + character.h - 1))
                    
                if character_image.data.size > 0 and character_image.data is not None:
                    mean_value = np.mean(character_image.data)
                else:
                    mean_value = 0
                
                if mean_value > 200:
                    found_dot = True
                else:
                    characters.append(character)
            
            else:
                characters.append(character)
                
        if found_dot:
            # Remove dot character and add new character.
            license_plate.subrects = characters
            
            # Insert new character.
            license_plate, inserted, width, height, m, c, x_interval = self._insert_between_characters(license_plate, image_original, width, height, m, c, x_interval)
            
            if not inserted:
                license_plate, width, height, m, c, x_interval = self._insert_in_limits(license_plate, image_original, width, height, m, c, x_interval)
             
        return license_plate
    
    def get_limits_x_y(self, license_plate):
        '''
        Get min and max of x, y.
        '''
        min_x = 99999999
        min_y = 99999999
        max_x = -99999999
        max_y = -99999999
         
        for character in license_plate.subrects:
            if character.x < min_x:
                min_x = character.x
            
            if character.y < min_y:
                min_y = character.y
            
            if character.x > max_x:
                max_x = character.x
            
            if character.y > max_y:
                max_y = character.y
        
        return min_x, min_y, max_x, max_y
    
    def adjust_characters(self, license_plate, image_original, resize_width=400, recropped=False):
        '''
        Adjust size of characters and predict the position of other characters.
        '''
        
        height, width = self._get_h_w(license_plate)
        m, c = self.get_least_squares(license_plate)
        
        # Normalize characters with the same width and height.
        license_plate, m, c = normalize_characters_character_height(license_plate, image_original, height, m, c)
        license_plate = normalize_characters_character_width(license_plate, image_original, width)
        x_interval = self._get_x_step(license_plate)
        
        # Adjust license plate to have at least two characters.
        if len(license_plate.subrects) < 2:
            if len(license_plate.subrects) == 0:
                new_character = Rect([license_plate.image.width/2, license_plate.image.height * 1.0 / 3, width, height])
                license_plate.subrects.append(new_character)
                m = 0
                c = int(license_plate.image.height * 1.0 / 3)
            else:
                license_plate, width, height, m, c, x_interval = self._insert_in_limits(license_plate, image_original, width, height, m, c, x_interval)
        
        # Adjust license plate to have seven characters.
        while len(license_plate.subrects) < 7:
            license_plate, inserted, width, height, m, c, x_interval = self._insert_between_characters(license_plate, image_original, width, height, m, c, x_interval)
            
            if not inserted:
                license_plate, width, height, m, c, x_interval = self._insert_in_limits(license_plate, image_original, width, height, m, c, x_interval)
        
        # Move dot character to other position.
        license_plate = self._correct_dot_character(license_plate, image_original, width, height, m, c, x_interval)
         
#         min_x, min_y, max_x, max_y = self.get_limits_x_y(license_plate)
#         origin = Point(min_x - width * 2, min_y -height)
#         end = Point(max_x + width * 2, max_y)
#
#         recropped = True
#         if not recropped and origin.x < end.x and origin.y < end.y:
#             old_license_plate = copy.deepcopy(license_plate)
#             old_mean = self._calculate_mean_caracter_values(license_plate)
#             
#             try:
#                 license_plate = recrop_license_plate_image(license_plate, origin, end, image_original, 2)
#                 image = pre_process_license_plate_image(license_plate.image, adaptative=True, morphologic=3)
#                 license_plate.image = image
#                 license_plate.subrects = image.compute_rectangles_for_characters()
#                 character_validator = CharacterValidator()
#                 character_validator.remove_wrong_characters(license_plate)
#                 license_plate = self.adjust_characters(license_plate, image_original, resize_width, recropped=True)
#             except:
#                 license_plate = old_license_plate
#             
#             new_mean = self._calculate_mean_caracter_values(license_plate)
#             
#             t = 135
#             if abs(old_mean -t) < abs(new_mean -t):
#                 license_plate = old_license_plate
        
        # Sort characters.
        license_plate.subrects = sorted(license_plate.subrects, key=lambda character: character.x)
            
        return license_plate

    def remove_wrong_characters(self, license_plate):
        '''
        Remove wrong rectangles of characters.
        '''
        
        # Remove rectangles with wrong positions.
        if len(license_plate.subrects) > 0:
            # Filter wrong characters.
            char_filter = FilterCharacters()
            license_plate.subrects = char_filter.filter_characters_by_color(license_plate.subrects, license_plate.image)
            
            if len(license_plate.subrects) > 3:
                license_plate.subrects = char_filter.filter_characters_too_close(license_plate.subrects)
                #license_plate.subrects = char_filter.filter_characters_by_y_std(license_plate.subrects)
                license_plate.subrects = char_filter.filter_characters_by_distance_std(license_plate.subrects)
                license_plate.subrects = char_filter.filter_characters_by_h_std(license_plate.subrects)
                #license_plate.subrects = char_filter.filter_characters_by_x_mean(license_plate.subrects)
                license_plate.subrects = char_filter.filter_characters_using_least_squares(license_plate.subrects, license_plate.best_subrects(3))
                license_plate.subrects = char_filter.filter_extra_characters(license_plate.subrects)
                
        return license_plate
