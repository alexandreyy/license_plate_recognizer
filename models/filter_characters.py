'''
Created on 06/07/2015

@author: Alexandre Yukio Yamashita
'''

from models.point import Point
import numpy as np


class FilterCharacters:
    '''
    Filter of characters.
    '''
    
    def filter_characters_by_color(self, characters, image):
        '''
        Filter characters by color.
        '''
        filtered_characters = []
        
        if len(characters) > 0:
            for character in characters:
                x = character.x - abs(character.w - character.h) / 2            
                character_image = image.crop(Point(x, character.y), Point(character.x + character.h - 1, character.y + character.h - 1))
                
                if len(character_image.data) > 0:
                    color_mean = np.mean(character_image.data)
                    
                    if color_mean > 75 and color_mean < 200:
                        filtered_characters.append(character)
        else:
            filtered_characters = characters
        
        return filtered_characters
    
    def _get_best_width_and_x(self, characters):
        '''
        Get width with minimum std between characters widths and
        x with minimum std. 
        '''
        
        best_width_std = 999999999
        best_width = 999999999
        best_x_std = 999999999
        best_x = 999999999
        len_characters = len(characters)
        
        if len_characters > 3:
            for _ in range(50):
                np.random.shuffle(characters)
                range_w = [characters[index].w for index in range(3)]
                range_x = [characters[index].x for index in range(3)]
                current_width_std = np.std(range_w)
                current_x_std = np.std(range_x)
                
                if current_width_std < best_width_std:
                    best_width_std = current_width_std
                    best_width = np.mean(range_w)
                
                if current_x_std < best_x_std:
                    best_x_std = current_x_std
                    best_x = np.mean(range_x)
        
        elif len_characters > 2:
            for _ in range(50):
                np.random.shuffle(characters)
                range_w = [characters[index].w for index in range(2)]
                range_x = [characters[index].x for index in range(2)]
                current_width_std = np.std(range_w)
                current_x_std = np.std(range_x)
                
                if current_width_std < best_width_std:
                    best_width_std = current_width_std
                    best_width = np.mean(range_w)
                
                if current_x_std < best_x_std:
                    best_x_std = current_x_std
                    best_x = np.mean(range_x)
                    
        elif len_characters > 1:
            best_width = np.mean([character.w for character in characters])
            best_x = np.mean([character.x for character in characters])
        elif len_characters == 1:
            best_width = characters[0].w
            best_x = characters[0].x
        else:
            best_width = 34
            best_x = 48
            
        return best_width, best_x
    
    def filter_extra_characters(self, characters):
        '''
        Filter extra characters by min_distance.
        '''
        # Remove extra rectangles. 
        filtered_characters = characters
        
        while len(filtered_characters) > 7:
            last_index = len(characters) - 1
            index_characters = range(last_index + 1)
            characters = sorted(characters, key=lambda character: character.x)
            best_min_distance = 9999999
            rectangle_to_be_removed = 0
            filtered_characters = []
             
            # Find license plate to be removed. 
            for index_character in index_characters:
                if index_character == 0:
                    min_distance = characters[1].x - characters[0].x
                elif index_character == last_index:
                    min_distance = characters[last_index].x - characters[last_index - 1].x
                else:                    
                    min_distance = min([characters[index_character].x - characters[index_character - 1].x,
                                        characters[index_character + 1].x - characters[index_character].x])
                 
                if min_distance < best_min_distance:
                    best_min_distance = min_distance
                    rectangle_to_be_removed = index_character
             
            for index_character in index_characters:
                if index_character != rectangle_to_be_removed:
                    filtered_characters.append(characters[index_character])
            
            characters = filtered_characters
            
        return filtered_characters
            
    def filter_characters_by_x_mean(self, characters):
        '''
        Filter characters by x mean.
        '''
        filtered_characters = []
        
        if len(characters) > 1:
            # Get width with least std,
            best_width, best_x = self._get_best_width_and_x(characters)
            
            for character in characters:
                if abs(character.x - best_x) < best_width * 5:
                    filtered_characters.append(character)
        else:
            filtered_characters = characters
            
        return filtered_characters
    
    def filter_characters_by_y_std(self, characters):
        '''
        Filter characters by y std.
        '''
        
        if len(characters) > 3:
            # Calculate std of Ys.
            filtered_characters = []
            range_y = [character.y for character in characters]
            std_y = np.std(range_y)
            mean_y = np.mean(range_y)
            
            # Filter characters using std of Ys.
            for index_character in range(len(characters)):     
                delta_y = abs(mean_y - characters[index_character].y)
                
                if not(std_y > 0 and std_y < delta_y and delta_y - std_y > 20):
                    filtered_characters.append(characters[index_character])
        else:
            filtered_characters = characters
            
        return filtered_characters
    
    def filter_characters_by_h_std(self, characters):
        '''
        Filter characters by h std.
        '''
        
        if len(characters) > 3:
            # Calculate std of Hs.
            filtered_characters = []
            range_h = [character.h for character in characters]
            std_h = np.std(range_h)
            mean_h = np.mean(range_h)
            
            # Filter characters using std of Ys.
            for index_character in range(len(characters)):     
                delta_h = abs(mean_h - characters[index_character].h)
                
                if not(std_h > 0 and std_h < delta_h and delta_h - std_h > 9):
                    filtered_characters.append(characters[index_character])
        else:
            filtered_characters = characters
            
        return filtered_characters
    
    def filter_characters_by_distance_std(self, characters):
        '''
        Filter characters by distance std.
        '''
        if len(characters) > 3:
            min_distances, characters = self.calculate_distances_between_characters(characters)
            filtered_characters = []
            mean_min_distance = np.mean(min_distances)
            std_min_distance = np.std(min_distances)
            
            # Filter characters using std of min distances.
            for index_character in range(len(characters)):
                delta_min_distance = abs(mean_min_distance - min_distances[index_character])
                
                if not(std_min_distance > 0 and std_min_distance < delta_min_distance and delta_min_distance - std_min_distance > 10):
                    filtered_characters.append(characters[index_character])
        
        else:
            filtered_characters = characters
            
        return filtered_characters
    
    def filter_characters_by_distance(self, characters):
        '''
        Filter characters by distance.
        '''
        min_distances, characters = self.calculate_distances_between_characters(characters)
        shuffled_min_distances = list(min_distances)        
        best_min_distance = 999999999
        best_min_distance_std = 999999999
        
        for _ in range(20):
            np.random.shuffle(shuffled_min_distances)
            range_min_distance = [shuffled_min_distances[index] for index in range(3)]
            current_min_distance_std = np.std(range_min_distance)
            
            if current_min_distance_std < best_min_distance_std:
                best_min_distance_std = current_min_distance_std
                best_min_distance = np.mean(range_min_distance)
            
        filtered_characters = []
        size_filtered = 0
        
        # Filter characters using min distance.
        for index_character in range(len(characters)):
            delta_min_distance = abs(best_min_distance - min_distances[index_character])
            
            if delta_min_distance < best_min_distance * 2.5:
                filtered_characters.append(characters[index_character])
            else:
                size_filtered += 1
            
        return filtered_characters, size_filtered
    
    def filter_characters_using_least_squares(self, characters, characters_model):
        '''
        Filter characters using least squares.
        '''
        
        if len(characters) > 3:
            # Calculate least squares.
            x = [rect.x for rect in characters_model]
            y = [rect.y for rect in characters_model]
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y)[0]
            
            # Filter characters using least squares.
            filtered_characters = []
            index_characters = range(len(characters))
            
            for index_character in index_characters:     
                predicted_y = characters[index_character].x * m + c
                delta = abs(predicted_y - characters[index_character].y)
                
                if delta < 12:
                    filtered_characters.append(characters[index_character])
        else:
            filtered_characters = characters
            
        return filtered_characters  
    
    def calculate_distances_between_characters(self, characters):
        '''
        Calculate distances between characters.
        '''
        
        last_index = len(characters) - 1
        index_characters = range(last_index + 1)
        characters = sorted(characters, key=lambda character: character.x)
        min_distances = []
        
        for index_character in index_characters:
            if index_character == 0:
                min_distance = characters[1].x - characters[0].x
            elif index_character == last_index:
                min_distance = characters[last_index].x - characters[last_index - 1].x
            else:                    
                min_distance = min([characters[index_character].x - characters[index_character - 1].x,
                                    characters[index_character + 1].x - characters[index_character].x])
            min_distances.append(min_distance)
        
        return min_distances, characters
    
    def filter_characters_too_close(self, characters):
        '''
        Filter characters too close.
        '''
        if len(characters) > 3:
            total_characters_too_close = 2
            
            while total_characters_too_close > 1:
                min_distances, characters = self.calculate_distances_between_characters(characters)
                mean_min_distance = np.mean(min_distances)
                exists_character_too_close = False
                characters_too_close = []
                filtered_characters = []
                
                for index_character in range(len(characters)):
                    if min_distances[index_character] < mean_min_distance * 0.54:
                        characters_too_close.append(index_character)
                        exists_character_too_close = True
                        
                if exists_character_too_close:
                    if len(characters_too_close) < 2:
                        total_characters_too_close = 0
                    else:
                        total_characters_too_close = len(characters_too_close)                        
                        min_area = 999999
                        
                        for index_character in characters_too_close:
                            character_area = characters[index_character].w * characters[index_character].h
                            
                            if character_area < min_area:
                                min_area = character_area
                                min_character = index_character
                        
                        for index_character in range(len(characters)):
                            if index_character != min_character:
                                filtered_characters.append(characters[index_character])
                                
                        total_characters_too_close -= 1
                        characters = filtered_characters
                else:
                    filtered_characters = characters
                    total_characters_too_close = 0
        else:
            filtered_characters = characters
            
        return filtered_characters
