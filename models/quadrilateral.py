'''
Created on 06/04/2015

@author: Alexandre Yukio Yamashita
'''
import math


class Quadrilateral:
    '''
    Quadrilateral with 4 points 2D. 
    '''
    def __init__(self):
        self.points = []
        
    def add_point(self, point):
        '''
        Add point if point quadrilateral is not complete.
        '''
        if self.is_complete():
            return False
        else:
            self.points.append(point)
            self._sort_points()
            return True
        
    def remove_point(self, point):
        '''
        Remove point if found it in quadrilateral.
        '''
        self.points.remove(point)
        self._sort_points()
        
    def is_complete(self):
        '''
        Check if quadrilateral is complete.
        '''
        return len(self.points) == 4
    
    def has_points(self):
        '''
        Check if quadrilateral has points.
        '''
        return len(self.points) != 0
    
    def _mean_x(self):
        '''
        Get mean of point Xs.
        '''
        return sum(point.x for point in self.points) / len(self.points)
    
    def _mean_y(self):
        '''
        Get mean of point Ys.
        '''
        return sum(point.y for point in self.points) / len(self.points)
    
    def _sort_points(self):
        '''
        Sort points in clockwise order.
        '''
        total_points = len(self.points)
                             
        if total_points > 1:
            mean_x = self._mean_x()
            mean_y = self._mean_y()
        
            def key_algorithm(point):
                return (math.atan2(point.y - mean_y, point.x - mean_x) + \
                         2 * math.pi) % (2 * math.pi)
        
            self.points.sort(key=key_algorithm)
            
            if total_points == 3:
                change_point = self.points[1]
                self.points[1] = self.points[2]
                self.points[2] = change_point
            elif total_points == 4:
                change_point = self.points[2]
                self.points[2] = self.points[0]
                self.points[0] = change_point
               
                change_point = self.points[3]
                self.points[3] = self.points[1]
                self.points[1] = change_point
