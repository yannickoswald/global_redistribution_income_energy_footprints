# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:08:32 2020

@author: Yannick Oswald mail: y-oswald@web.de, web: collective-mind.org

only usable in the context for income simulations in the paper "Global redistribution of income and household energy"
it is not a very "smart" function but a work-horse so to speak just to avoid lengthy code in the main code and repetition. Simplicity of understanding and overview for authors and readers has been chosen over pythony elegant code.

"""

def footprint_calc(income_vector, elasticities, energy_intensities, normalization):
      import numpy as np 
      e = np.power(income_vector,0.83)*2.6 ## gdp per capita USD ppp constant 2011 to expenditure per capita USD ppp constant 2011 
      
      for i in range(0,len(e)):  ### to make sure that expenditure never exceeds income, it is a simplification of the real world but an important one in our model
         if e[i]>income_vector[i]:
            e[i] = income_vector[i]
            
      ###loading e-intensities     
      #no1_int = [1.35, 1.21, 2.88, 1.30, 40.77, 3.23, 2.41, 2.05, 15.37, 6.52, 1.96, 2.50, 6.86, 1.65]
      no1_int = [1.35, 1.21, 2.88, 1.30, 35.77, 3.23, 2.41, 2.05, 19.37, 6.52, 1.96, 2.50, 6.86, 1.65] ### world average
      no2_int = [1.60, 1.48, 3.16, 1.55, 75.34, 3.63, 2.56, 2.93, 26.05, 8.55, 2.51, 2.91, 8.79, 1.89] ### energy weighted
      no3_int = [2.33, 3.26, 5.87, 5.34, 70.40, 8.83, 9.10, 11.53, 54.61, 16.58, 12.27, 14.49, 21.35, 15.80]### pop weighted
      
      ### setting up elasticities
      if (elasticities == 1):
      ### cross country non-weighted
         c1 = np.power(e,0.54)*16.09082373## expenditure to food 
         c2 = np.power(e,0.89)*0.04730202 ## expenditure to alc&tob 
         c3 = np.power(e,0.88)*0.13298835 ## expenditure to wearables 
         c4 = np.power(e,1.17)*0.02733834 ## expenditure to other housing 
         c5 = np.power(e,0.83)*0.22365838 ## expenditure to heat and electr.
         c6 = np.power(e,1.13)*0.01332764 ## expenditure to hh appli. 
         c7 = np.power(e,1.01)*0.02316099  ## expenditure to health 
         c8 = np.power(e,1.91)*0.00000412 ## expenditure to vehicle purchase 
         c9 = np.power(e,1.65)*0.00009271 ## expenditure to vehicle fuel & maintenance
         c10 = np.power(e,0.84)*0.08001774 ## expenditure other transport 
         c11 = np.power(e,1.21)*0.00594723 ## expenditure to comm. 
         c12 = np.power(e,1.65)*0.00003487 ## expenditure to rec. items 
         c13 = np.power(e,2.05)*0.00000023 ## expenditure to package holiday 
         c14 = np.power(e,1.18)*0.02089814 ## expenditure to edu&finance&luxury 
     
      if (elasticities == 2):
      ### cross country weighted (except package holiday)
         c1 = np.power(e,0.62)*7.75 ## expenditure to food
         c2 = np.power(e,0.91)*0.04 ## expenditure to alc&tob 
         c3 = np.power(e,0.92)*0.11 ## expenditure to wearables 
         c4 = np.power(e,1.24)*0.0127 ## expenditure to other housing 
         c5 = np.power(e,0.88)*0.14972 ## expenditure to heat and electr.
         c6 = np.power(e,1.03)*0.03 ## expenditure to hh appli. 
         c7 = np.power(e,1.04)*0.029 ## expenditure to health 
         c8 = np.power(e,1.60)*0.000073 ## expenditure to vehicle purchase 
         c9 = np.power(e,1.77)*0.000029 ## expenditure to fuel 
         c10 = np.power(e,0.84)*0.076674 ## expenditure other transport 
         c11 = np.power(e,1.26)*0.004269 ## expenditure to comm. 
         c12 = np.power(e,1.56)*0.00011 ## expenditure to rec. items 
         c13 = np.power(e,2.05)*0.00000023 ## expenditure to package holiday ### NON WEIGHTED PARAMETERS KEPT BECAUSE HIGHER STATISTICAL CONFIDENCE 
         c14 = np.power(e,1.25)*0.014520 ## expenditure to edu&finance&luxury 

      if (elasticities == 3):
      ### SENSITIVITY/MONTE-CARLO
          
          #elas1 = np.random.normal(0.62,0) 
          #elas2 = np.random.normal(0.91,0) 
          #elas3 = np.random.normal(0.92,0) 
          #elas4 = np.random.normal(1.24,0) 
          #elas5 = np.random.normal(0.88,0) 
          #elas6 = np.random.normal(1.03,0)
          #elas7 = np.random.normal(1.04,0) 
          #elas8 = np.random.normal(1.60,0) 
          #elas9 = np.random.normal(1.77,0) 
          #elas10 = np.random.normal(0.84,0) 
          #elas11 = np.random.normal(1.26,0) 
          #elas12 = np.random.normal(1.56,0) 
          #elas13 = np.random.normal(2.05,0) 
          #elas14 = np.random.normal(1.25,0) 
          
          elas1 = np.random.normal(0.62,0.013352454) 
          elas2 = np.random.normal(0.91,0.031738452) 
          elas3 = np.random.normal(0.92,0.017664458) 
          elas4 = np.random.normal(1.24,0.028901468) 
          elas5 = np.random.normal(0.88,0.021973621) 
          elas6 = np.random.normal(1.03,0.01786618)
          elas7 = np.random.normal(1.04,0.033131687) 
          elas8 = np.random.normal(1.60,0.050287143) 
          elas9 = np.random.normal(1.77,0.037979913) 
          elas10 = np.random.normal(0.84,0.029026093) 
          elas11 = np.random.normal(1.26,0.028966766) 
          elas12 = np.random.normal(1.56,0.046483377) 
          elas13 = np.random.normal(2.05,0.108192853) 
          elas14 = np.random.normal(1.25,0.02308235) 
     
          ### do another run with constant coefficients
          
          #coeff1 = 7.74574
          #coeff2 = 0.04018
          #coeff3 = 0.11016
          #coeff4 = 0.01275
          #coeff5 = 0.14972
          #coeff6 = 0.03012
          #coeff7 = 0.02858
          #coeff8 = 0.000073
          #coeff9 = 0.000029
          #coeff10 = 0.076674
          #coeff11 = 0.004269
          #coeff12 = 0.000110
          #coeff13 = 0.00000023
          #coeff14 = 0.014520
          
          #### another run with coefficient specific monte carlo, independent of elasticity
          
          #coeff1 = np.exp(np.random.normal(2.047142939,0.111237683))
          #coeff2 = np.exp(np.random.normal(-3.214277588,0.264409645))
          #coeff3 = np.exp(np.random.normal(-2.205864553,0.147160469))
          #coeff4 = np.exp(np.random.normal(-4.362393567,0.240774638))
          #coeff5 = np.exp(np.random.normal(-1.89896887,0.183072883))
          #coeff6 = np.exp(np.random.normal(-3.502542457,0.148840987))
          #coeff7 = np.exp(np.random.normal(-3.555135318,0.276016078))
          #coeff8 = np.exp(np.random.normal(-9.524212437,0.420216226))
          #coeff9 = np.exp(np.random.normal(-10.45007936,0.316470609))
          #coeff10 = np.exp(np.random.normal(-2.568193741,0.241812844))
          #coeff11 = np.exp(np.random.normal(-5.456321307,0.241319746))
          #coeff12 = np.exp(np.random.normal(-9.115501229,0.387247348))
          #coeff13 = np.exp(np.random.normal(-10.0638977,0.929229627))
          #coeff14 = np.exp(np.random.normal(-4.232222795,0.192296269))

          #### default run 
          
          coeff1 = 1759.4*np.exp(-10.53*elas1)*3.161460877 #### the last multiplier is a shift of the exponential fit in order to overlay the fit exactly with the default value of the scaling coefficients chosen
          coeff2 = 1759.4*np.exp(-10.53*elas2)*0.327763356
          coeff3 = 1759.4*np.exp(-10.53*elas3)*1.027528722
          coeff4 = 1759.4*np.exp(-10.53*elas4)*3.270399435
          coeff5 = 1759.4*np.exp(-10.53*elas5)*0.940777699
          coeff6 = 1759.4*np.exp(-10.53*elas6)*0.886216692
          coeff7 = 1759.4*np.exp(-10.53*elas7)*0.950362551
          coeff8 = 1759.4*np.exp(-10.53*elas8)*0.827447472
          coeff9 = 1759.4*np.exp(-10.53*elas9)*2.121461769
          coeff10 = 1759.4*np.exp(-10.53*elas10)*0.308510273
          coeff11 = 1759.4*np.exp(-10.53*elas11)*1.389658011
          coeff12 = 1759.4*np.exp(-10.53*elas12)*0.821745725
          coeff13 = 1759.4*np.exp(-10.53*elas13)*0.30999867
          coeff14 = 1759.4*np.exp(-10.53*elas14)*4.416775739
          

          
          c1 = np.power(e,elas1)*coeff1 ## expenditure to food 
          c2 = np.power(e,elas2)*coeff2 ## expenditure to alc&tob 
          c3 = np.power(e,elas3)*coeff3 ## expenditure to wearables 
          c4 = np.power(e,elas4)*coeff4 ## expenditure to other housing 
          c5 = np.power(e,elas5)*coeff5 ## expenditure to heat and electr.
          c6 = np.power(e,elas6)*coeff6 ## expenditure to hh appli. 
          c7 = np.power(e,elas7)*coeff7 ## expenditure to health 
          c8 = np.power(e,elas8)*coeff8 ## expenditure to vehicle purchase 
          c9 = np.power(e,elas9)*coeff9 ## expenditure to fuel 
          c10 = np.power(e,elas10)*coeff10 ## expenditure other transport 
          c11 = np.power(e,elas11)*coeff11## expenditure to comm. 
          c12 = np.power(e,elas12)*coeff12 ## expenditure to rec. items 
          c13 = np.power(e,elas13)*coeff13 ## expenditure to package holiday ### NON WEIGHTED PARAMETERS KEPT BECAUSE HIGHER STATISTICAL CONFIDENCE 
          c14 = np.power(e,elas14)*coeff14 ## expenditure to edu&finance&luxury 
         
      if (elasticities == 4):
          
         c1 = np.power(e,0.624538654)*2.45060784 ## expenditure to food
         c2 = np.power(e,0.90895678)*0.12262961 ## expenditure to alc&tob 
         c3 = np.power(e,0.921701987)*0.10722844 ## expenditure to wearables 
         c4 = np.power(e,1.236448855)*0.00389883 ## expenditure to other housing 
         c5 = np.power(e,0.884180552)*0.15918423 ## expenditure to heat and electr.
         c6 = np.power(e,1.030792917)*0.03399570 ## expenditure to hh appli. 
         c7 = np.power(e,1.042423963)*0.03007695 ## expenditure to health 
         c8 = np.power(e,1.596135188)*0.00008832 ## expenditure to vehicle purchase 
         c9 = np.power(e,1.773474403)*0.00001365 ## expenditure to fuel 
         c10 = np.power(e,0.841851315)*0.24858605 ## expenditure other transport 
         c11 = np.power(e,1.259057904)*0.00307284 ## expenditure to comm. 
         c12 = np.power(e,1.556664549)*0.00013383 ## expenditure to rec. items 
         c13 = np.power(e,1.479277047)*0.00030231 ## expenditure to package holiday ### NON WEIGHTED PARAMETERS KEPT BECAUSE HIGHER STATISTICAL CONFIDENCE 
         c14 = np.power(e,1.252624264)*0.00328823 ## ex
    
      if (elasticities == 5): #### sensitivity #1 only heat and electricity down drastically
      ### cross country weighted (except package holiday)
         c1 = np.power(e,0.62)*7.75 ## expenditure to food
         c2 = np.power(e,0.91)*0.04 ## expenditure to alc&tob 
         c3 = np.power(e,0.92)*0.11 ## expenditure to wearables 
         c4 = np.power(e,1.24)*0.0127 ## expenditure to other housing 
         c5 = np.power(e,0.7)*1.1 ## expenditure to heat and electr.
         c6 = np.power(e,1.03)*0.03 ## expenditure to hh appli. 
         c7 = np.power(e,1.04)*0.029 ## expenditure to health 
         c8 = np.power(e,1.60)*0.000073 ## expenditure to vehicle purchase 
         c9 = np.power(e,1.77)*0.000029 ## expenditure to fuel 
         c10 = np.power(e,0.84)*0.076674 ## expenditure other transport 
         c11 = np.power(e,1.26)*0.004269 ## expenditure to comm. 
         c12 = np.power(e,1.56)*0.00011 ## expenditure to rec. items 
         c13 = np.power(e,2.05)*0.00000023 ## expenditure to package holiday ### NON WEIGHTED PARAMETERS KEPT BECAUSE HIGHER STATISTICAL CONFIDENCE 
         c14 = np.power(e,1.25)*0.014520 ## expenditure to edu&finance&luxury 
               
      if (elasticities == 6): #### sensitivity #2  heat and electricity and vehicle fuel down drastically
      ### cross country weighted (except package holiday)
         c1 = np.power(e,0.62)*7.75 ## expenditure to food
         c2 = np.power(e,0.91)*0.04 ## expenditure to alc&tob 
         c3 = np.power(e,0.92)*0.11 ## expenditure to wearables 
         c4 = np.power(e,1.24)*0.0127 ## expenditure to other housing 
         c5 = np.power(e,0.7)*1.1 ## expenditure to heat and electr.
         c6 = np.power(e,1.03)*0.03 ## expenditure to hh appli. 
         c7 = np.power(e,1.04)*0.029 ## expenditure to health 
         c8 = np.power(e,1.60)*0.000073 ## expenditure to vehicle purchase 
         c9 = np.power(e,1.2)*0.0057229 ## expenditure to fuel 
         c10 = np.power(e,0.84)*0.076674 ## expenditure other transport 
         c11 = np.power(e,1.26)*0.004269 ## expenditure to comm. 
         c12 = np.power(e,1.56)*0.00011 ## expenditure to rec. items 
         c13 = np.power(e,2.05)*0.00000023 ## expenditure to package holiday ### NON WEIGHTED PARAMETERS KEPT BECAUSE HIGHER STATISTICAL CONFIDENCE 
         c14 = np.power(e,1.25)*0.014520 ## expenditure to edu&finance&luxury 


     
      if (normalization == 1):
          
          sum_c = c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11+c12+c13+c14
      
          c1_share = np.divide(c1,sum_c)
          c2_share = np.divide(c2,sum_c)
          c3_share = np.divide(c3,sum_c)
          c4_share = np.divide(c4,sum_c)
          c5_share = np.divide(c5,sum_c)
          c6_share = np.divide(c6,sum_c)
          c7_share = np.divide(c7,sum_c)
          c8_share = np.divide(c8,sum_c)
          c9_share = np.divide(c9,sum_c)
          c10_share = np.divide(c10,sum_c)
          c11_share = np.divide(c11,sum_c)
          c12_share = np.divide(c12,sum_c)
          c13_share = np.divide(c13,sum_c)
          c14_share = np.divide(c14,sum_c)
      
          c1_normalized = np.multiply(c1_share,e) 
          c2_normalized = np.multiply(c2_share,e) 
          c3_normalized = np.multiply(c3_share,e) 
          c4_normalized = np.multiply(c4_share,e) 
          c5_normalized = np.multiply(c5_share,e) 
          c6_normalized = np.multiply(c6_share,e) 
          c7_normalized = np.multiply(c7_share,e) 
          c8_normalized = np.multiply(c8_share,e) 
          c9_normalized = np.multiply(c9_share,e) 
          c10_normalized = np.multiply(c10_share,e) 
          c11_normalized = np.multiply(c11_share,e) 
          c12_normalized = np.multiply(c12_share,e) 
          c13_normalized = np.multiply(c13_share,e) 
          c14_normalized = np.multiply(c14_share,e) 
          
          
      if (normalization == 0):
          
          c1_normalized = c1
          c2_normalized = c2 
          c3_normalized = c3
          c4_normalized = c4 
          c5_normalized = c5 
          c6_normalized = c6 
          c7_normalized = c7 
          c8_normalized = c8 
          c9_normalized = c9
          c10_normalized = c10 
          c11_normalized = c11 
          c12_normalized = c12 
          c13_normalized = c13 
          c14_normalized = c14 
     
      ###setting up e-intensities
      if (energy_intensities == 1):
         e_intensity1 =  no1_int[0] ## in MJ/$ 
         e_intensity2 = no1_int[1] ## in MJ/$
         e_intensity3 = no1_int[2] ## in MJ/$
         e_intensity4 = no1_int[3] ## in MJ/$
         e_intensity5 = no1_int[4] ## in MJ/$ 
         e_intensity6 = no1_int[5] ## in MJ/$
         e_intensity7 = no1_int[6] ## in MJ/$
         e_intensity8 = no1_int[7]## in MJ/$
         e_intensity9 = no1_int[8] ## in MJ/$  
         e_intensity10 = no1_int[9]## in MJ/$
         e_intensity11 = no1_int[10] ## in MJ/$
         e_intensity12 = no1_int[11] ## in MJ/$
         e_intensity13 = no1_int[12]## in MJ/$
         e_intensity14 = no1_int[13] ## in MJ/$
         
         ######### MONTE CARLO SENSITIVITY ANALYSIS  #####
      if (energy_intensities == 2):
         e_intensity1 =  np.random.normal(no1_int[0],0.09) ## in MJ/$ 
         e_intensity2 = np.random.normal(no1_int[1],0.10)## in MJ/$
         e_intensity3 = np.random.normal(no1_int[2],0.22) ## in MJ/$
         e_intensity4 = np.random.normal(no1_int[3],0.11)## in MJ/$
         e_intensity5 = np.random.normal(no1_int[4],6.35) ## in MJ/$ 
         e_intensity6 = np.random.normal(no1_int[5],0.32) ## in MJ/$
         e_intensity7 = np.random.normal(no1_int[6],0.27) ## in MJ/$
         e_intensity8 = np.random.normal(no1_int[7],0.23)## in MJ/$
         e_intensity9 = np.random.normal(no1_int[8],4.13) ## in MJ/$  
         e_intensity10 = np.random.normal(no1_int[9],0.49)## in MJ/$
         e_intensity11 = np.random.normal(no1_int[10],0.44) ## in MJ/$
         e_intensity12 = np.random.normal(no1_int[11],0.14)## in MJ/$
         e_intensity13 = np.random.normal(no1_int[12],1.76)## in MJ/$
         e_intensity14 = np.random.normal(no1_int[13],0.22) ## in MJ/$
         
         
     
      
      energy_footprints_1 = np.multiply(c1_normalized,e_intensity1)
      energy_footprints_2 = np.multiply(c2_normalized,e_intensity2)
      energy_footprints_3 = np.multiply(c3_normalized,e_intensity3)
      energy_footprints_4 = np.multiply(c4_normalized,e_intensity4)
      energy_footprints_5 = np.multiply(c5_normalized,e_intensity5)
      energy_footprints_6 = np.multiply(c6_normalized,e_intensity6)
      energy_footprints_7 = np.multiply(c7_normalized,e_intensity7)
      energy_footprints_8 = np.multiply(c8_normalized,e_intensity8)
      energy_footprints_9 = np.multiply(c9_normalized,e_intensity9)
      energy_footprints_10 = np.multiply(c10_normalized,e_intensity10)
      energy_footprints_11 = np.multiply(c11_normalized,e_intensity11)
      energy_footprints_12 = np.multiply(c12_normalized,e_intensity12)
      energy_footprints_13 = np.multiply(c13_normalized,e_intensity13)
      energy_footprints_14 = np.multiply(c14_normalized,e_intensity14)
      
      
      
      expenditure_all = np.column_stack((c1_normalized,
                                c2_normalized, 
                                c3_normalized, 
                                c4_normalized,
                                c5_normalized,
                                c6_normalized,
                                c7_normalized,
                                c8_normalized,
                                c9_normalized,
                                c10_normalized,
                                c11_normalized,
                                c12_normalized,
                                c13_normalized,
                                c14_normalized))
      

      e_footprints = np.column_stack((energy_footprints_1,
                                energy_footprints_2, 
                                energy_footprints_3, 
                                energy_footprints_4,
                                energy_footprints_5,
                                energy_footprints_6,
                                energy_footprints_7,
                                energy_footprints_8,
                                energy_footprints_9,
                                energy_footprints_10,
                                energy_footprints_11,
                                energy_footprints_12,
                                energy_footprints_13,
                                energy_footprints_14))
      
      return expenditure_all, e_footprints