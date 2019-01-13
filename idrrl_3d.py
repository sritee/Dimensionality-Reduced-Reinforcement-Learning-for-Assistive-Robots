#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 20:43:50 2018

@author: sritee
"""
import numpy as np
from tiles3 import tiles,IHT
from MountainCar3D import MountainCar3D
from joblib import load
import matplotlib.pyplot as plt

num_tries=1 #number of trials to average over
steps=[]    #to hold episode step counts 

for tries in range(num_tries):

    #we need not redeclare all of these every trial, but just doing here for ease of use since it's fast anyway.
    
    env=MountainCar3D()
    num_actions=env.num_actions #number of available actions
    num_episodes=4000
   
    observations_dim= 4 #the observations in the environment
    
    num_tilings=16
    alpha=(1/num_tilings) #recommended alpha by Rich Sutton
    tiling_number=10 #num_rows of tiles per tiling. So total number is dimensionality^obs_dim * num_tilings
    
    project_models=[]
    tilings=[] #List of tilings, each for one dimension
    weights=[] #List of weights, one for each dimension
    
    scalers=[]
    
    for k in range(observations_dim): #let us build all the requisite weights
    
        project_models.append(load('./PCA Models/pca_pipeline_{}.joblib'.format(k+1)))
        tilings.append(IHT(num_tilings*np.power(tiling_number,k+1))) #tile coding seperate for each dimension
        
        weights.append(np.zeros([num_tilings*np.power(tiling_number,k+1),num_actions]))
        
    stepcount=np.zeros([num_episodes,1])
    gamma=0.99 #discount factor
    epsilon=0.1 #set exploration parameter as desired
 
    
    def compute_tile_indices(state,dimension):
        
        c=tiles(tilings[dimension],num_tilings,tiling_number*state.flatten()) #use the appropriate tiling as desired
        return c
        
    def normalize(state):
        
        return (0.5*(1+state)) #convert -1 to 1 range to 0 to 1
     
    def compute_value(cs_tiles_list,dimension): #compute values of all action in a state
        #See Equation 3 in Paper
        val=np.zeros(num_actions)
        
        for k in range(dimension+1):   
            val= val+ np.sum(weights[k][cs_tiles_list[k],:],axis=0) + 0 #adding rmax for PAC-MDP guarantee, environment specific

        return val
       
    def updateweights(cs_tiles_list,curaction,delta,dimension):
        
        weights[dimension][cs_tiles_list[dimension],curaction]+= delta*alpha
        
    def epsilon_greedy(cs_tiles_list,dimension,epsilon): #pass a state where agent is eps-greedy, weight matrix w
        
        qval=compute_value(cs_tiles_list,dimension)
        best_action=np.argmax(qval)
        
        action=best_action #by default, take the best action
        
        if np.random.rand(1)< epsilon:
            action=np.random.randint(num_actions)#epsilon greedy
            
        return action, best_action #we are returning both to check if opt policy has converged
    
    def is_converged():
        
        if(((zeta/stepcount[i]>0.7) or cur_dim_it>200) and (cur_dim_it>50*(1+dimension))):
            return 1
        else:
            return 0
    
    dimension=0 #first start learning from dimension 1
    cur_dim_it=0 #number of iterations in current dimension
    num = 0 #for checking if convergence condition holds in 5 consecutive episodes
        
    for i in range(num_episodes):
       
        env.reset()
        curstate=env.state()
        zeta=0 #number of policy changes in the episode
        
        cs_tiles_list=[] #stores the list of curstate tiles (differs by dimension)
        ns_tiles_list=[]
        
        cur_dim_it+=1 #adding one iteration to cur dimension
        
        transformed_state=normalize(project_models[-1].transform(curstate.reshape(1,-1)))
        
        for d in range(dimension+1): #0 based indexing, so dimension + 1
            cs_tiles_list.append(compute_tile_indices(transformed_state[:,:d+1],d))
            
        while True:
            #print(w.shape)
            curaction,best_action=epsilon_greedy(cs_tiles_list,dimension,epsilon)  #epsilon greedy selection
            stepcount[i,0]=stepcount[i,0]+1
            
            nextstate,reward, done, info = env.step(curaction) 
            delta = reward - compute_value(cs_tiles_list,dimension)[curaction];   #The TD Error                    
    
            if done:
               
                print("Episode reduced 3D {} finished after {} timesteps in try {} with zeta {} and dimension {} and curdim {}".format(i,stepcount[i,0],tries,zeta/stepcount[i],dimension,cur_dim_it))
                updateweights(cs_tiles_list,curaction,delta,dimension)
                #alpha=np.maximum(alpha*0.995,0.005)
                break
            
            transformed_state=normalize(project_models[-1].transform(nextstate.reshape(1,-1)))
         
            for d in range(dimension+1): #0 based indexing, so dimension + 1
                ns_tiles_list.append(compute_tile_indices(transformed_state[:,:d+1],d))
 
            
            next_state_val=np.max(compute_value(ns_tiles_list,dimension))
            
            delta=delta+ gamma*np.max(next_state_val)
            
            updateweights(cs_tiles_list,curaction,delta,dimension) #update the weight vector
            
            _,best_action_after_update= epsilon_greedy(cs_tiles_list,dimension,epsilon)
            
            if best_action==best_action_after_update:
                zeta=zeta+1
                
            curstate=nextstate
            cs_tiles_list=ns_tiles_list.copy()
            ns_tiles_list= [] #empty ns_tiles, so we can use append operation later. Otherwise memory will grow.
            
            if stepcount[i]>2000:
                print('failed episode {} in try {}, dim {}, zeta {}, curit {}'.format(i,tries,dimension,zeta/stepcount[i,0],cur_dim_it))
                break
                
        if dimension<observations_dim-1:  #check for transfer of value up to next dimension
            if(is_converged()):
                    num=num+1
                    if num>=2:
                        print('Policy has converged, transferring to next dimension')
                        dimension=dimension+1
                        cur_dim_it=0               
            else:
                    num=0   
                    
    steps.append(stepcount)
    np.save('idrrl_mcar3d.npy',steps)
    
plt.plot(np.array(steps).mean(axis=0))
plt.title('Iterative Dimensionality Reduced Reinforcement Learning')
plt.xlabel('Number of episodes elapsed')
plt.ylabel('Number of Steps to Goal')




