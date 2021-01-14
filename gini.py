# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:58:27 2019

@author: eeyo
"""
#### own function to calculate gini coefficient. can be used with main script. not very efficient but does the job ####
def gini(population, resource):
    import numpy as np 
    iuse = (population !=0) & (resource != 0) & (resource > 0)
    A = population[iuse]
    B = resource[iuse]
    C = np.divide(B,A)
    
    sorted_capita= C.argsort()
    sorted_A = A[sorted_capita[::1]]
    sorted_B = B[sorted_capita[::1]]
    
    
    cumulative_A = np.cumsum(sorted_A)
    cumulative_B = np.cumsum(sorted_B)
    
    share_cum_A =np.zeros((len(cumulative_A), 1))
    share_cum_B =np.zeros((len(cumulative_B), 1))#
    area_under_curve = np.zeros((len(share_cum_A), 1))
    
    for i in range(0,len(cumulative_A)):
        share_cum_A[i] = cumulative_A[i]/(cumulative_A[-1]) 
        share_cum_B[i] = cumulative_B[i]/(cumulative_B[-1]) 
        
        new_share_A = np.insert(share_cum_A, 0,0)
        new_share_B = np.insert(share_cum_B, 0,0)
        
        for i in range(0,len(area_under_curve)):
            area_under_curve [i] =  (np.add(new_share_B[i+1],new_share_B[i])/2)*(np.subtract(new_share_A[i+1],new_share_A[i]))
   
    
    Area_A = 0.5-sum(area_under_curve)
    Gini_coefficient = 2*Area_A
    return Gini_coefficient
    
