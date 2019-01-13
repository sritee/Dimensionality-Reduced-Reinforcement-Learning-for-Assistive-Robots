#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:42:30 2018

@author: sritee
"""

import numpy as np
import math

class MountainCar3D:
 
    def __init__(self):
        
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.low = np.array([self.min_position,self.min_position, -self.max_speed,-self.max_speed])
        self.high = np.array([self.max_position,self.max_position, self.max_speed,self.max_speed])

        self.num_actions = 5 #no accelerate, accelerate along x,  decelerate along x, accelerate along y, decelerate along y
        
        #initialize states and velocities to zero initially, call reset function to randomize the x and y pos and set velocity to zero.
        self.x_pos=0
        self.y_pos=0
        self.x_velocity=0
        self.y_velocity=0
        
        self.height=0

        #self.seed()
        self.reset() 
        
    def reset(self):
        
        self.x_pos=np.random.uniform(self.min_position,self.goal_position)
        self.y_pos=np.random.uniform(self.min_position,self.goal_position)
        
        self.x_velocity=np.random.uniform(-self.max_speed,self.max_speed)
        self.y_velocity=np.random.uniform(-self.max_speed,self.max_speed)
        
        
#        self.x_pos=np.random.uniform(-0.6,-0.4)
#        self.y_pos=np.random.uniform(-0.6,-0.4)
#        
#        self.x_velocity=0
#        self.y_velocity=0
        
        self.height= math.sin(3.0*self.x_pos) + math.sin(3.0 * self.y_pos)
        
        return np.array([self.x_pos,self.y_pos,self.x_velocity,self.y_velocity])
    
    def state(self):
        
        return np.array([self.x_pos,self.y_pos,self.x_velocity,self.y_velocity])
    
    def step(self,action):
        
        self.x_velocity += math.cos(3*self.x_pos)*(-0.0025)
        self.y_velocity += math.cos(3*self.y_pos)*(-0.0025)
        
        if action==0:     
            #print('hi')
            pass
        
        elif action==1:
            
            self.x_velocity +=  0.001
            #self.x_velocity = np.clip(self.x_velocity, -self.max_speed, self.max_speed)
            
        elif action==2:
            
            self.x_velocity  += -0.001 
            #self.x_velocity = np.clip(self.x_velocity, -self.max_speed, self.max_speed)
            
        elif action==3:
            
            self.y_velocity += 0.001 
            #self.y_velocity = np.clip(self.y_velocity, -self.max_speed, self.max_speed)
        
        elif action==4:

            self.y_velocity += -0.001
            #self.y_velocity = np.clip(self.y_velocity, -self.max_speed, self.max_speed)
        
        else:
            print('Invalid Action. Exiting!')
            
        self.x_velocity = np.clip(self.x_velocity, -self.max_speed, self.max_speed)
        self.y_velocity = np.clip(self.y_velocity, -self.max_speed, self.max_speed)
        
        self.x_pos = self.x_pos + self.x_velocity
        self.y_pos = self.y_pos + self.y_velocity
        
        self.x_pos = np.clip(self.x_pos, self.min_position, self.max_position)
        self.y_pos = np.clip(self.y_pos, self.min_position, self.max_position)
#        
#        if (self.x_pos==self.min_position and self.x_velocity<0): 
#            self.x_velocity = 0 
#            self.y_velocity= 0
#            
#        if (self.y_pos==self.min_position and self.y_velocity<0): 
#            self.x_velocity = 0 
#            self.y_velocity= 0

        done = bool(self.x_pos >= self.goal_position and self.y_pos>=self.goal_position)
        reward = -1.0
        
        if done:
            reward=0
        
        #self.height= math.sin(3.0*self.x_pos) + math.sin(3.0 * self.y_pos)

        return np.array([self.x_pos,self.y_pos,self.x_velocity,self.y_velocity]), reward, done, {}
        
        