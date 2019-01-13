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
import time
import sys

num_tries=10
steps=[]

for tries in range(num_tries):
#env = gym.make('Acrobot-v1').env # change if desired to required environment, ensure you change num_actions,sample given below
#env=  gym.make('MountainCar-v0').env
#env=gym.make('CartPole-v1').env #change num_actions to 2
    
    env=MountainCar3D()
    num_actions=env.num_actions #number of available actions
    num_episodes=4000
   
    observations_dim= 4 #the observations in the environment
    
    num_tilings=16
    alpha=(1/num_tilings)#change required base alpha
    #alpha=0.1
    tiling_number=10 #num_rows of tiles per tiling. So total number is dimensionality^obs_dim * num_tilings
    
    project_models=[]
    tilings=[]
    weights=[]
    
    scalers=[]
    
    for k in range(observations_dim): #let us build all the requisite weights, we can also do this later, doing here for readability
    
        project_models.append(load('pca_pipeline_{}.joblib'.format(k+1)))
        tilings.append(IHT(num_tilings*np.power(tiling_number,k+1)))
        
        weights.append(np.zeros([num_tilings*np.power(tiling_number,k+1),num_actions])) #only weight number 1 is needed initially, saving memory
        
    stepcount=np.zeros([num_episodes,1])
    gamma=0.99 #discount factor
    epsilon=0.1 #set exploration parameter as desired
    visualize_after_steps=100 #start the display
    
    #val=pipeline.transform(env.low.reshape(1,-1))
    #val1=pipeline.transform(env.high.reshape(1,-1))
    
    def compute_tile_indices(state,dimension):
        
        #reduced_state=pipeline.transform(state)
        c=tiles(tilings[dimension],num_tilings,tiling_number*state.flatten())
        #print(c)
        return c
        
    def normalize(state):
        
        #fix the normalization
        #normstate= (state-scalers[dimension][0])/(scalers[dimension][1] - scalers[dimension][0]) #normalize to 0 to 1
        #print(normstate,temp)
        return (0.5*(1+state))
       #return state
    

    def compute_value(cs_tiles_list,dimension): #compute values of all action in a state
        
        #if dimension==0:
            #return np.sum(weights[0][cs_tiles_list[0],:],axis=0) #no transfer is happening
        #else:
        val=np.zeros(num_actions)
        
        for k in range(dimension+1):   
            val= val+ np.sum(weights[k][cs_tiles_list[k],:],axis=0) + 0 #adding rmax for pac-mdp guarantee

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
                #print(transformed_state.shape)
        for d in range(dimension+1): #0 based indexing, so dimension + 1
            cs_tiles_list.append(compute_tile_indices(transformed_state[:,:d+1],d))
            
        while True:
            #print(w.shape)
            curaction,best_action=epsilon_greedy(cs_tiles_list,dimension,epsilon)  #epsilon greedy selection
            #print(normalize(curstate))
            #if i>visualize_after_steps:
                #env.render()
                #time.sleep(0.01)
            stepcount[i,0]=stepcount[i,0]+1
            
            nextstate,reward, done, info = env.step(curaction) 
            
            #if np.absolute(nextstate[1])>1:
                #sys.exit(0)
            
            delta = reward - compute_value(cs_tiles_list,dimension)[curaction];   #The TD Error                    
    
            if done:
                
                #print(env.state())
                #sys.exit(0)
                
                print("Episode reduced 3D {} finished after {} timesteps in try {} with zeta {} and dimension {} and curdim {}".format(i,stepcount[i,0],tries,zeta/stepcount[i],dimension,cur_dim_it))
                updateweights(cs_tiles_list,curaction,delta,dimension)
                #alpha=np.maximum(alpha*0.995,0.005)
                break
            
            
            transformed_state=normalize(project_models[-1].transform(nextstate.reshape(1,-1)))
            #print(transformed_state.shape)
            for d in range(dimension+1): #0 based indexing, so dimension + 1
                ns_tiles_list.append(compute_tile_indices(transformed_state[:,:d+1],d))
    
            #print(nextaction)
            
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
#                if(zeta/stepcount[i]>0.9):
#                    print('Policy has converged, transferring to next dimension')
#                
#                    if dimension<observations_dim-1:
#                        dimension= dimension+1
                break
        #break
        if dimension<observations_dim-1:  #check for transfer of policy up.
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

#plt.plot(stepcount)    
#plt.title('MountainCar3D - QLearning')   

#np.save('./3D_demos/good_demos.npy',np.array(good_traj))  
#np.save('./3D_demos/bad_demos.npy',np.array(bad_traj))
#np.save('./3D_demos/terrible_demos.npy',np.array(terrible_traj))
#env.monitor.close()


