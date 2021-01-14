####### EACH SECTION NEEDS TO BE RUN INDIVIDUALLY ##### SECTIONS BEGIN WITH import os
import os
os.getcwd()

os.chdir("your pathway")



                         ## Simulation #0 ##
########## fitting the lognormal model to the income distribution ##########
############################################################################



import pandas as pd 
import math 
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
from gini import *
from lin_fit import *
from footprint_calc import *
from lognorm import *
from scipy.stats import gamma
import math as math
from sympy import Symbol 
from sympy.solvers import solveset
from sympy.solvers import solve
from sympy import erf
from sympy import log
from sympy import sqrt
from sympy import N
from matplotlib import rc
import matplotlib.gridspec as gridspec
from scipy.special import gamma, factorial
############## FIT REAL WORLD CDF ################# 

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', #### color blind friendly list
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

incomeranked = np.genfromtxt('incomeranked.csv', delimiter=',')
cumulativepincome = np.genfromtxt('cumulativepincome.csv', delimiter=',')
incomeranked_lakner = np.genfromtxt('laknerincomeranked.csv', delimiter=',')
cumulativepincome_lakner = np.genfromtxt('laknercumpop.csv', delimiter=',')
incomeranked_alv = np.genfromtxt('incomeranked_alv.csv', delimiter=',')
cumulativepincome_alv = np.genfromtxt('cumulativepincome_alv.csv', delimiter=',')

incomeranked = incomeranked.reshape((379, 1))
cumulativepincome = cumulativepincome.reshape((379, 1))
incomeranked_lakner = incomeranked_lakner.reshape((1179, 1))
cumulativepincome_lakner = cumulativepincome_lakner.reshape((1179, 1))

### LOG-NORMAL MODEL ### equation-based
bins = 1000
q = np.zeros(bins)
probability =  np.linspace((1/bins),1,num=bins);
mu, sigma = 8.723866481,1.259658829 ###(parameters last adjusted on 03/06/2020 see excel sheet "lognormal model derivation")
population_size_real = 7.004E+09
bin_size = population_size_real/bins
p = 0 
from sympy import N



for i in range(1,len(q)+1):
  x = Symbol('x')   
  p = 0+(i/len(q))
  if (i == len(q)):
      solution, = N(sp.solveset((0.5 + 0.5*erf((log(x) - mu)/(sqrt(2)*sigma)))-(1-(1/(10*bins))))) ###0.9999 = approximation to 1 based quantile resolution and appropriate total mean income measured in GDP PPP per capita USD 2011 which was 13592 USD
      q[i-1] = solution
  else:
      solution, = N(sp.solveset((0.5 + 0.5*erf((log(x) - mu)/(sqrt(2)*sigma)))-p))
      q[i-1] = solution
  print("# iteration is " +str(i))

mean_inc = np.zeros(bins) 

for i in range(1,len(q)+1):
    if (i == 1):
       mean_inc[i-1] = (0+q[0])/2
    else:
       mean_inc[i-1] = (q[i-1]+q[i-2])/2
        

plt.figure(num=0)
plt.plot((incomeranked_lakner),(cumulativepincome_lakner), label = "Lakner for 2008 (adjusted)",color = '#377eb8');
plt.plot((incomeranked_alv),(cumulativepincome_alv), label = "Alvaredo et al. for 2011 (adjusted)",color= '#ff7f00');
plt.plot((incomeranked),(cumulativepincome), label = "Oswald et al. estimate for 2011",color = '#4daf4a');
plt.plot(mean_inc, probability, label = "Log-normal model of Alv. data",linestyle="dashed", color = '#984ea3');
plt.show
plt.xlabel('income per capita $ PPP 2011');
plt.ylabel('cumulative population');
plt.legend(loc="lower right",prop={'size': 8}, frameon = False)
plt.xscale('log')
plt.text(2, 0.8, 'Gini coefficient log-normal = 0.63', fontsize=8)
plt.text(2, 0.9, 'Gini coefficient Alv. data = 0.64', fontsize=8)


plt.savefig('fig2.png', dpi = 300)


total_inc = sum(mean_inc*bin_size)
global_mean_gdp = total_inc/population_size_real
 





energy_exp_world = footprint_calc(mean_inc,2,1,1) #### 2,1,1 is standard



## check income gini model vs. real

population_vector = np.full(bins,bin_size)
gini(population_vector,mean_inc)


## check expenditure gini model vs. real
exp_dist= np.multiply(np.sum((energy_exp_world[0]),1),bin_size)
gini(population_vector,exp_dist)

exp_per_capita=np.divide(exp_dist,population_vector)

expenditure_real = np.genfromtxt('expenditure_real_world.csv', delimiter=',')
gini(expenditure_real[:,1],expenditure_real[:,0])


## check energy gini and total energy model vs. real

#total energy
energy_footprints_world_total= sum(np.multiply(np.sum((energy_exp_world[1]),1),bin_size))/(1e12)
#gini 
energy_dist = (np.multiply(np.sum((energy_exp_world[1]),1),bin_size))
gini(population_vector,energy_dist)
energy_per_capita = np.divide(energy_dist,population_vector)

reg_results = lin_fit(exp_per_capita, energy_per_capita)
#reg_results = lin_fit(mean_inc, exp_per_capita)
#hallo = footprint_calc(mean_inc,1,1)

#np.divide(energy_dist,exp_dist)


############## SIMULATIONS ################# 

            ## Simulation #1 ##
########## stretch and compress ##########
##########################################

####### LOG-NORMAL DISTRIBUTION ######## 
#1a
#1 Squeezing the distribution ### the idea is to reshape the real-world
#log-normal distribution by changing its standard deviation to lower and to higher values
import os
os.getcwd()
os.chdir("your pathway)


import pandas as pd 
import math 
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
from gini import *
from lin_fit import *
from footprint_calc import *
from plot_stacked_bar import *
from lognorm import *
from scipy.stats import gamma
import math as math
from sympy import Symbol 
from sympy.solvers import solveset
from sympy import erf
from sympy import log
from sympy import sqrt
from sympy import N
from matplotlib import rc
import matplotlib.gridspec as gridspec
import time


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', #### color blind friendly list
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

iterations = 20
inequality_sim = np.zeros(iterations)
energy_sim = np.zeros(iterations)
mu, sigma = 8.723866481,1.259658829
bins = 1000
q = np.zeros((bins,iterations))
sf = 10 # scale factor to iterate over sigma
mean_inc = np.zeros((bins,iterations)) 
total_energy_footprint = np.zeros(iterations)
total_energy_inequality = np.zeros(iterations)
total_income = np.zeros(iterations)
Gini_coefficient_income = np.zeros(iterations)
population_size_real = 7.004E+09
bin_size = population_size_real/bins
sigma_array = np.zeros(iterations)
per_category = np.zeros((14,iterations))
percent_below_DLE = np.zeros(iterations)
percent_mega_energy_consumers = np.zeros(iterations)
mean_data = float(np.exp(mu+(sigma*sigma)/2)) ### https://en.wikipedia.org/wiki/Log-normal_distribution and https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-with-specified-mean-and-variance.html
standard_dev_data = float(sqrt((np.exp(sigma*sigma)-1)*np.exp(2*mu+sigma*sigma)))
gini_per_category = np.zeros((14,iterations))
top_one_percent_share_category = np.zeros((14,iterations))
top_one_percent_share_total = np.zeros((1,iterations))

fig = plt.figure(figsize = [9.5, 4])
gs = fig.add_gridspec(1,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

 
Gini_coeff_income = math.erf(sigma/2)
#gs.update(wspace=0.4, hspace=0.6) 

plt.subplot(ax1)
for j in range(1,np.size(q, 1)+1):
   standard_dev_sim = standard_dev_data/10+(standard_dev_data/10)*(j-1)
   mu_sim = np.log((mean_data*mean_data)/(np.sqrt(standard_dev_sim*standard_dev_sim+mean_data*mean_data))) ### https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-with-specified-mean-and-variance.html natural log https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
   sigma_sim = np.sqrt(np.log(1 + (standard_dev_sim*standard_dev_sim)/(mean_data*mean_data)))
   #compute bin thresholds 
   for i in range(1,np.size(q, 0)+1):
     x = Symbol('x')   
     p = 0+(i/len(q))
     if (i == len(q)):
          solution, = N(sp.solveset((0.5 + 0.5*erf((log(x) - mu_sim)/(sqrt(2)*(sigma_sim))))-(1-(1/(10*bins))))) ###0.999 = approximation to 1
          q[i-1,j-1] = solution
     else:
          solution, = N(sp.solveset((0.5 + 0.5*erf((log(x) - mu_sim)/(sqrt(2)*(sigma_sim))))-p))
          q[i-1,j-1] = solution
   ### compute average income per bin       
   for i in range(1,np.size(q, 0)+1):
    if (i == 1):
       mean_inc[i-1,j-1] = (0+q[0,j-1])/2
    else:
       mean_inc[i-1,j-1] = (q[i-1,j-1]+q[i-2,j-1])/2
   if (10 == j):
     plt.plot((mean_inc[:,j-1]),np.linspace(1/bins,1,num=bins), label = 'data-based', color = 'red', linewidth = 3.5);
   else:
     plt.plot((mean_inc[:,j-1]),np.linspace(1/bins,1,num=bins), label = '$\u03C3_X$='+str(round(standard_dev_sim)));
   plt.legend(bbox_to_anchor=(1.3, 0.5), loc='center right', fontsize = 7.5, frameon=False)
   plt.xscale('log')
   plt.xlabel('income (GDP) per capita $ PPP',fontsize=10, labelpad=-1);
   plt.ylabel('cumulative population',fontsize=10);
   sigma_array[j-1] = round(standard_dev_sim);
   results =  footprint_calc(mean_inc[:,j-1],2,1,1) ### argument #2 is elasticities argument #3 is energy intensities they can be filled with integer numbers which relate to different values applied, for details view function "footprint_calc". (2,1) is the default setting.
   per_category[:,j-1] = sum(results[1])
   per_category_per_capita = results[1]
   #for i in range(0,14):
    #  test2 = per_category_per_capita[:,i]
     # gini_per_category[i,j-1] = gini(np.full(len(mean_inc), 1), test2)
   total_energy_footprint_per_capita = np.sum(results[1],1)
   total_expenditure_per_capita = np.sum(results[0],1)
   total_expenditure = sum(np.multiply(total_expenditure_per_capita, bin_size))
   total_energy_footprint_per_capita_GJ = (total_energy_footprint_per_capita/1000)
   percent_below_DLE[j-1]  = (sum(total_energy_footprint_per_capita_GJ<26)/bins)*100
   percent_mega_energy_consumers[j-1]  = (sum(total_energy_footprint_per_capita_GJ>270)/bins)*100 ### mega-consumers = people who consume as much energy as the top 20% Americans or more
   total_energy_footprint[j-1] = sum(np.sum(results[1],1)*bin_size)*1e-12 ### in exajoule from megajoule this is why 1e-12
   total_energy_inequality[j-1] = gini(np.full(len(mean_inc), 1),total_energy_footprint_per_capita)
   Gini_coefficient_income[j-1] = gini(np.full(len(mean_inc), 1),mean_inc[:,j-1])
   total_income[j-1] = sum(mean_inc[:,j-1])*bin_size ###
   top_one_percent_share_category[:,j-1] = np.transpose(sum(per_category_per_capita[990:1000,:])/sum(per_category_per_capita))
   top_one_percent_share_total[:,j-1] = (sum(sum(per_category_per_capita[990:1000,:]*bin_size))/10**12)/total_energy_footprint[j-1] 
   plt.xticks(fontsize=9)
   plt.yticks(fontsize=9)
   if (j == iterations):
      plt.text(min(ax1.get_xlim()), 1.15, 'a', fontsize=11)
   print("# iteration is " +str(j))
   fig.tight_layout()



percent_difference_sim =np.subtract(np.divide(total_energy_footprint, total_energy_footprint[9]),1)*100

plt.subplot(ax2)
plt.plot(sigma_array,percent_difference_sim, linewidth = 5);
#plt.yscale('log')
plt.xlabel("$\u03C3_X$", fontsize = 10);
plt.ylabel('net energy change (%)',fontsize = 10);
plt.xticks(fontsize=10);
plt.yticks(fontsize=10);
plt.text(min(Gini_coefficient_income), max(percent_difference_sim)+1.2, 'b', fontsize=11)
plt.plot([sigma_array[9], sigma_array[9]], [min(percent_difference_sim), max(percent_difference_sim)], linestyle = 'dashed', color = 'black', linewidth = 4)
plt.xlim((min(sigma_array), max(sigma_array))) 
plt.ylim((min(percent_difference_sim), max(percent_difference_sim))) 
#plt.plot([40200,40200],[min(percent_difference_sim), max(percent_difference_sim)], linestyle = 'dashed', color = 'black', linewidth = 3)
#plt.plot([13400,13400],[min(percent_difference_sim), max(percent_difference_sim)], linestyle = 'dashed', color = 'black', linewidth = 3)
plt.annotate("data based", xy=(0.6, 0.1), xytext=(0.3, 0),arrowprops=dict(arrowstyle="->"))
axes1 = plt.gca()
axes2 = axes1.twiny()
#axes2.set_xticks([0, 0.4735, 1])
axes2.set_xticks([0, 0.21,0.4735,0.735, 1])
axes2.set_xticklabels([np.round(Gini_coefficient_income[0],2), 0.44, np.round(Gini_coefficient_income[9],2), 0.71, np.round(Gini_coefficient_income[19],2)], fontsize = 9)
# axes2.set_xticks([0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
#axes2.set_xticklabels([np.round(Gini_coefficient_income[0],2), np.round(Gini_coefficient_income[9],2), np.round(Gini_coefficient_income[19],2)], fontsize = 9)
axes2.set_xlabel("Gini coefficient income", fontsize = 10) ### IMPORTANT LABEL MAYBE##
fig.tight_layout()
plt.savefig('fig3.png',bbox_inches = "tight", dpi = 300)


error_margin_income_average = (1-total_income[0]/total_income[19])*100

############ FIGURE 4 PLOTTING ##################

per_category_share = np.zeros((14,iterations))
for j in range(1,iterations+1):
    for i in range(1,15):
        per_category_share[i-1,j-1] = per_category[i-1,j-1]/sum(per_category[:,j-1])
 
file = os.path.join('your pathway')     

dfc = pd.read_excel(file, sheet_name='Sheet1',header=1,usecols="C:I",nrows = 15)
concordance_agg = dfc.to_numpy();

per_category_share_agg= np.zeros((6,iterations))
for j in range(1,iterations+1):
   per_category_share_agg[:,j-1] = np.dot(per_category_share[:,j-1],concordance_agg)

df = pd.DataFrame(np.transpose(per_category_share_agg))
df = df*100


typical_mega = footprint_calc([370000],2,1,1) ### 2,1,1 is default
typical_DLE =footprint_calc([12700],2,1,1)

typical_mega_agg = np.dot(typical_mega[1],concordance_agg)
typical_DLE_agg = np.dot(typical_DLE[1],concordance_agg)

typical_mega_agg_percent = np.zeros(6)
typical_DLE__agg_percent = np.zeros(6)


typical_mega_agg_percent = np.transpose((typical_mega_agg/sum(sum(typical_mega_agg)))*100)
typical_DLE_agg_percent = np.transpose((typical_DLE_agg/sum(sum(typical_DLE_agg)))*100)

N=2
ind = (np.arange(N))

subsistence = (18.1428,6.9795)
heat_elect = (49.4751,17.3443)
health_tech = (8.16245, 5.44703)
transport = (17.0552,58.2235)
edu_recreat_lux = (6.73429,8.0209)           
package_holiday = (0.430082,3.98473)
width = 0.65 
  
def two_scales(ax1, time, data1, data2, c1, c2):
    ax2 = ax1.twinx()
    ax1.plot(time, data1, color=c1, linewidth = 3)
    ax1.set_xlabel('$\u03C3_X$')
    ax1.set_ylabel('per cent < DLE', color = '#377eb8',) 
    ax2.plot(time, data2, color=c2, linewidth = 3)
    ax2.set_ylabel('per cent mega consumers', color = '#ff7f00')
    ax2.margins(x=0)
    ax2.margins(y=0)
    ax1.margins(x=0)
    ax1.margins(y=0)
    return ax1, ax2



fig, [ax1,ax2,ax3] = plt.subplots(nrows= 1,ncols =3, figsize=(14,4),  gridspec_kw={'width_ratios': [2,2,1.2]})
#ax1.legend(loc='center left',labels= ('subsistence', 'heat and electricity', 'health and tech','transport','edu , recreat., lux.', 'package holiday'),fontsize=9,framealpha=0.5)
#set.xlim((min(sigma_array), max(sigma_array))) 
#ax1.ylim((0,100))
#ax1.xlabel("\u03C3")
#ax1.ylabel("per cent of total")
ax1.stackplot(sigma_array, df[0], df[1], df[2], df[3], df[4], df[5], colors = CB_color_cycle)
ax1.legend(loc='center left',labels= ('subsistence', 'heat and electricity', 'health and tech','transport','edu , recreat., lux.', 'package holiday'),fontsize=9,framealpha=0.5)
ax1.set_xlim((min(sigma_array), max(sigma_array))) 
ax1.set_ylim((0,100))
ax1.set_xlabel("$\u03C3_X$")
ax1.set_ylabel("per cent of total")
ax1.text(2000, 111, 'a', fontsize=12)
ax1.plot([26800,26800],[0,100], linestyle = 'dashed', color = 'black', linewidth = 3)
#ax1.plot([13400,13400],[0,100], linestyle = 'dashed', color = 'black', linewidth = 3)
#ax1.plot([2680,2680],[0,100], linestyle = 'dashed', color = 'black', linewidth = 3)
#ax1.plot([40200,40200],[0,100], linestyle = 'dashed', color = 'black', linewidth = 3)
ax1a = ax1.twiny()
ax1a.set_xticks([0, 0.21,0.4735,0.735, 1])
ax1a.set_xticklabels([np.round(Gini_coefficient_income[0],2), 0.44, np.round(Gini_coefficient_income[9],2), 0.71, np.round(Gini_coefficient_income[19],2)], fontsize = 9)
ax1a.set_xlabel("Gini coefficient income", fontsize = 10) ### IMPORTANT LABE
ax2, ax2a = two_scales(ax2, sigma_array, percent_below_DLE, percent_mega_energy_consumers, '#377eb8', '#ff7f00')
ax2.text(0, max(percent_below_DLE)+7, 'b', fontsize=12)
ax2.plot([26800,26800],[min(percent_below_DLE),max(percent_below_DLE)], linestyle = 'dashed', color = 'black', linewidth = 3)
ax2.margins(x=0)
ax2.margins(y=0)
ax2a.set_ylim([0,1.23])
ax2b = ax2.twiny()
ax2b.set_xticks([0, 0.21,0.4735,0.735, 1])
ax2b.set_xticklabels([np.round(Gini_coefficient_income[0],2), 0.44, np.round(Gini_coefficient_income[9],2), 0.71, np.round(Gini_coefficient_income[19],2)], fontsize = 9)
ax2b.set_xlabel("Gini coefficient income", fontsize = 10) ### IMPORTANT LABE
#plt.stackplot(sigma_array, df[0], df[1], df[2], df[3], df[4], df[5])           
p1 = ax3.bar(ind, subsistence, width)
p2 = ax3.bar(ind, heat_elect,width, bottom = subsistence)
p3 = ax3.bar(ind, health_tech,width, bottom = np.add(heat_elect,subsistence))
p4 = ax3.bar(ind, transport,width, bottom = np.add(np.add(heat_elect,subsistence),health_tech),color = '#f781bf')
p5 = ax3.bar(ind, edu_recreat_lux,width, bottom = np.add(np.add(np.add(heat_elect,subsistence),health_tech),transport), color = '#a65628')
p6 = ax3.bar(ind, package_holiday,width, bottom = np.add(np.add(np.add(np.add(heat_elect,subsistence),health_tech),transport),edu_recreat_lux), color = '#984ea3')
ax3.set_ylabel("per cent of total")
ax3.set_xticklabels(('','low consumer','','mega consumer'))
ax3.set_ylim((0,100))
ax3.text(-0.4, 111, 'c', fontsize=12)
fig.tight_layout()  # otherwise the right y-label is slightly clipped  
plt.savefig('fig4.png',bbox_inches = "tight", dpi = 300)



############ FIGURE 5 PLOTTING ##################

metrics = np.genfromtxt('inequality_metrics_fig4.csv', delimiter=',')


fig = plt.figure(figsize = [4.5, 4])
plt.plot(metrics[:,0],metrics[:,1], label = "food", color = CB_color_cycle[0])
plt.plot(metrics[:,0],metrics[:,2], label = "heat and electricity",  color = CB_color_cycle[1])
plt.plot(metrics[:,0],metrics[:,3], label = "vehicle fuel",  color = CB_color_cycle[2])
plt.plot(metrics[:,0],metrics[:,4], label = "vehicle purchasing",  color = CB_color_cycle[3])
plt.plot(metrics[:,0],metrics[:,5], label = "package holiday",  color = CB_color_cycle[4])
#plt.plot(metrics[:,0],metrics[:,6])
plt.plot([0, 1],[0, 1],color = CB_color_cycle[7], linestyle = "--", label = "income = energy")
plt.plot([0.63, 0.63], [0,1], linestyle = 'dashed', color = 'black', linewidth = 2)
plt.xlim((0,1))
plt.ylim((0,1))
plt.xlabel("Gini coefficient income")
plt.ylabel("Gini coefficient energy")
plt.legend(loc = 'upper left' , fontsize = 7)
axes1 = plt.gca()
axes2 = axes1.twiny()
axes2.set_xticks([0.11, 0.63, 0.77])
#axes2.set_xticks(np.arange(0, 3, 1.0))
axes2.set_xticklabels((2680, 26800, 53600), fontsize = 9)
axes2.set_xlabel("$\u03C3_X$", fontsize = 10) ### IMPORTANT LABEL MAYBE##
plt.savefig('fig5.png',bbox_inches = "tight", dpi = 300)





####### extra plots ####

#### income gini vs. energy gini total

plt.plot(Gini_coefficient_income, total_energy_inequality)
plt.plot([0, 1],[0, 1],'r--', label = "income = energy")
plt.xlabel("Gini coefficient income")
plt.ylabel("Gini coefficient energy")

######################################################
################# Gini income vs. Gini 

labels = [
'Food',
'Alcohol and Tobacco',
'Wearables',
'Other housing',
'Heating and Electricity',
'Household Appliances and Services',
'Health',
'Vehicle Purchase',
'Vehicle Fuel and Maintenance',
'Other transport',
'Communication',
'Recreational items',
'Package Holiday',
'Education & Finance & Other Luxury',
]


fig = plt.figure(figsize = [9.5, 4])
gs = fig.add_gridspec(1,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])



#[0,4,7,8,12]
for i in range(0,14):
    ax1.plot(Gini_coefficient_income, per_category[i,:]*(7*0.000001), label = labels[i])
    ax1.set_yscale('log')
    ax1.margins(x=0)
    ax1.margins(y=0)
    ax1.set_xlabel("Gini coefficient income")
    ax1.set_ylabel("Exajoule")
    #ax1.legend( bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 10)


#############share top 1% in categories ####

for i in range(0,14):
    ax2.plot(Gini_coefficient_income, top_one_percent_share_category[i,:]*100, label = labels[i])
    ax2.margins(x=0)
    ax2.margins(y=0)
    ax2.set_xlabel("Gini coefficient income")
    ax2.set_ylabel("top 1% share in per cent")
    ax2.legend(bbox_to_anchor=(-0.5, -0.2), loc='upper left', fontsize = 10,  frameon = False)

plt.savefig('suppfig5.png',bbox_inches = "tight", dpi = 300)


############# SIMULATION #2 ##############################
########## MIN. and MAX. income / Cap and floor ##########
##########################################################


### we will identify income per capita ppp caps and floors by reasoning from the question "How much is enough to pay for the poor"?
### we can for any minimum income find the amount of money needed to pay for it and will then use a bisection algorithm to find the corresponding (minimum) cap that can exactly pay for this. This yields "perfect" redistribution.

#### foundation set up
import os
os.getcwd()
### uni path
os.chdir("your pathway")

import pandas as pd 
import math 
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
from gini import *
from lin_fit import *
from footprint_calc import *
from lognorm import *
from scipy.stats import gamma
import math as math
from sympy import Symbol 
from sympy.solvers import solveset
from sympy import erf
from sympy import log
from sympy import sqrt
from sympy import N
from matplotlib import rc
import matplotlib.gridspec as gridspec


### set up variables needed

bins = 1000
population_size_real = 7.004E+09 ##https://en.wikipedia.org/wiki/World_population
bin_size = population_size_real/bins
pop_segments = np.full((1000,1),bin_size)
    

gdp_ppp_thr = [2.29,
4.30,
8.26,
11.81,
16.97,
27.66,
] ### consumption expenditure poverty thresholds transformed to gdp ppp per capita
cap_floor_pairs = np.zeros((6,2))
percentage_cap_floor = np.zeros((6,2))

for k in range(0,len(gdp_ppp_thr)):
    
    mean_inc = np.genfromtxt('mean_inc_exp2.csv', delimiter=',') ### mean_inc per quantiles. There are 1000 uniform quantiles.
    mean_inc2 = np.genfromtxt('mean_inc_exp2.csv', delimiter=',') 
    mean_inc3 = np.genfromtxt('mean_inc_exp2.csv', delimiter=',') 
    
    min_inc = gdp_ppp_thr[k] ## as daily poverty threshold in USD PPP so for example 1.9$/capita/day (most sense make values between 1.9 $ and 23 $ including 2.97$, 8.44$, 7.4$, 10$,15$ since these are connected to world bank country income classification or poverty thresholds in disscusion https://ourworldindata.org/extreme-poverty#the-share-of-the-world-population-relative-to-various-poverty-lines )
    percentage_cap_floor[k,0] = sum(mean_inc2<min_inc*365)/bins
    mean_inc2[mean_inc2<min_inc*365] = min_inc*365
    money_needed_redist_minbased =  abs(np.dot(np.transpose(pop_segments),(np.subtract(mean_inc,mean_inc2))))
    
    ### first iteration of bisection alograithm to initialize upper and lower bounds as well as money_available variable
    max_inc_lowerbound_start = min(mean_inc)/365
    max_inc_upperbound_start = max(mean_inc)/365
    c = (max_inc_upperbound_start+max_inc_lowerbound_start)/2
    mean_inc3[mean_inc3>c*365] = c*365
    money_available_redist_maxbased = abs(np.dot(np.transpose(pop_segments),(np.subtract(mean_inc,mean_inc3))))
    
    ### leftover iterations of bisection algorithms
    max_inc_lowerbound = max_inc_lowerbound_start 
    max_inc_upperbound = max_inc_upperbound_start 
    Nmax = 10000
    precision_threshold = 0.001 ### accuracy of bisection search to 1% of error
    i = 0
    while i < Nmax:
      i = i + 1
      if abs((money_available_redist_maxbased/money_needed_redist_minbased)-1)<precision_threshold:
          print("Correct income cap found which is at " +str(np.ceil(c))+" $ PPP/capita/day")
          break 
      else: 
        c = (max_inc_upperbound+max_inc_lowerbound)/2
        mean_inc3 = np.genfromtxt('mean_inc_exp2.csv', delimiter=',') 
        percentage_cap_floor[k,1] = abs(sum(mean_inc3<c*365)/bins-1)
        mean_inc3[mean_inc3>c*365] = c*365
        money_available_redist_maxbased = abs(np.dot(np.transpose(pop_segments),(np.subtract(mean_inc,mean_inc3))))
        if (money_available_redist_maxbased/money_needed_redist_minbased)-1>0:
          ### this means that money available is more than money needed so the lower bound is too high
            max_inc_lowerbound = c
            max_inc_upperbound = max_inc_upperbound 
        else:
            max_inc_lowerbound = max_inc_lowerbound
            max_inc_upperbound = c         
    
    
    cap_floor_pairs[k,1]=c
    cap_floor_pairs[k,0]=gdp_ppp_thr[k]




n = ["1.9$ CE","3.2$ CE", "5.5$ CE", "7.4$ CE", "10$ CE", "15$ CE"]   
cap_floor_pairs = np.genfromtxt('caps_and_floors.csv', delimiter=',') #
z = cap_floor_pairs[1:len(n)+1,0]
y = cap_floor_pairs[1:len(n)+1,1]
colors = ["violet", "indigo", "blue", "green", "yellow", "orange", "red"]
plt.scatter(z,y,c = colors[0:len(colors)-1])  
plt.plot(z,y)  
plt.xlabel("gdp income floor $ PPP")
plt.ylabel("gdp income cap $ PPP")
for i in range(0,len(n)):
    plt.annotate(n[i], (z[i]+0.2, y[i]))
#plt.annotate(n[4], (z[4]-11, y[4]-10))
#plt.annotate(n[5], (z[5]-9, y[5]-60))
#plt.savefig('supp fig2.png',bbox_inches = "tight", dpi = 300)


fig = plt.figure()
gs = fig.add_gridspec(1,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
fig.tight_layout()
fig.subplots_adjust(wspace = 1/3)

Gini_coefficients = np.zeros(len(cap_floor_pairs))
Gini_income =np.zeros(len(cap_floor_pairs))
total_energy = np.zeros(len(cap_floor_pairs))
style = ["solid", "dashed", "dotted", "dashdot", "solid", "dashed", "dotted"]
minmaxenergy_cap_floor = np.zeros((len(cap_floor_pairs),2))
income_vectors=np.zeros((len(mean_inc),7))


for i in range(0,len(cap_floor_pairs)):
    mean_inc = np.genfromtxt('mean_inc_exp2.csv', delimiter=',') 
    mean_inc[mean_inc > cap_floor_pairs[i,1]*365] = cap_floor_pairs[i,1]*365  ##### capping
    mean_inc[mean_inc < cap_floor_pairs[i,0]*365] = cap_floor_pairs[i,0]*365  ##### flooring
    income_vectors[:,i] = mean_inc
    
    results = footprint_calc(mean_inc, 2, 1, 1)
    total_energy_footprint_per_capita = np.sum(results[1],1)/1000
    minmaxenergy_cap_floor[i,0] = min(total_energy_footprint_per_capita) 
    minmaxenergy_cap_floor[i,1] = max(total_energy_footprint_per_capita) 
    total_energy_footprint=np.multiply(np.transpose(pop_segments), total_energy_footprint_per_capita)
    energy_vector = np.transpose(total_energy_footprint)
    total_energy[i] = sum(sum(total_energy_footprint))
    
    
    
    Gini_income[i] = gini(pop_segments[:,0], mean_inc)
    population = pop_segments[:,0]
    resource = energy_vector[:,0]
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

    
    for j in range(0,len(cumulative_A)):
        share_cum_A[j] = cumulative_A[j]/(cumulative_A[-1]) 
        share_cum_B[j] = cumulative_B[j]/(cumulative_B[-1]) 
        
    new_share_A = np.insert(share_cum_A, 0,0)
    new_share_B = np.insert(share_cum_B, 0,0)
    plt.subplot(ax2)
    plt.plot(new_share_A,new_share_B, colors[i], linestyle = style[i])
    plt.xlabel("cumulative population")
    plt.ylabel("cumulative energy")
    plt.legend(["none"] + n)
    plt.text(-0.02, 1.07, 'b', fontsize=12)
   # ax = plt.gca()
    #leg = ax.get_legend()
    #eg.legendHandles[i].set_color(colors[i-1])

    if i == len(cap_floor_pairs)-1: 
        plt.plot([0,1],[0,1], linestyle = "solid")
    
    for k in range(0,len(area_under_curve)):
            area_under_curve [k] =  (np.add(new_share_B[k+1],new_share_B[k])/2)*(np.subtract(new_share_A[k+1],new_share_A[k]))
   
    Area_A = 0.5-sum(area_under_curve)
    Gini_coefficients[i] = 2*Area_A




for i in range(0, len(cap_floor_pairs)):
   plt.subplot(ax1)
   plt.plot(income_vectors[:,i], new_share_A[1:1001],colors[i])
   plt.xscale('log')
   plt.xlabel("income (GDP) per capita $ PPP")
   plt.ylabel("cumulative population")
   plt.legend(["none"] + n)
   plt.text(40, 1.07, 'a', fontsize=12)
   
F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*1.5, Size[1]*1, forward=True) #
plt.savefig('fig6.png',bbox_inches = "tight", dpi = 300)
   
   
   


            ## SIMULATION #3 ##
            
#############################################################
###### "MONTE-CARLO"/Stochastic UNCERTAINTY ANALYSIS ########
#############################################################

import os
os.getcwd()
### uni path
os.chdir("your pathway")
### home path
#os.chdir("C:/Users/Home/Dropbox/Bildung/PhD/2. future/Redistribution of income and energy consumption")

import pandas as pd 
import math 
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
from gini import *
from lin_fit import *
from footprint_calc import *
from plot_stacked_bar import *
from lognorm import *
from scipy.stats import gamma
import math as math
from sympy import Symbol 
from sympy.solvers import solveset
from sympy import erf
from sympy import log
from sympy import sqrt
from sympy import N
from matplotlib import rc
import matplotlib.gridspec as gridspec
import time
begin_time = time.time()

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', #### color blind friendly list
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

iterations = 100
inequality_sim = np.zeros(iterations)
energy_sim = np.zeros(iterations)
mu, sigma = 8.723866481,1.259658829
bins = 1000
q = np.zeros((bins,iterations))
sf = 10 # scale factor to iterate over sigma
mean_inc = np.zeros((bins,iterations)) 
total_energy_footprint = np.zeros(iterations)
total_energy_inequality = np.zeros(iterations)
total_income = np.zeros(iterations)
Gini_coefficient_income = np.zeros(iterations)
population_size_real = 7.004E+09
bin_size = population_size_real/bins
sigma_array = np.zeros(iterations)
per_category = np.zeros((14,iterations))
percent_below_DLE = np.zeros(iterations)
percent_mega_energy_consumers = np.zeros(iterations)
mean_data = float(np.exp(mu+(sigma*sigma)/2)) ### https://en.wikipedia.org/wiki/Log-normal_distribution and https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-with-specified-mean-and-variance.html
standard_dev_data = float(sqrt((np.exp(sigma*sigma)-1)*np.exp(2*mu+sigma*sigma)))
gini_per_category = np.zeros((14,iterations))

CARLO = 100
monte_carlo_results_energy_inequality = np.zeros((CARLO,2))
monte_carlo_results_energy_total = np.zeros((CARLO,2))
monte_carlo_results_share_transport = np.zeros((CARLO,2))
monte_carlo_results_share_residential = np.zeros((CARLO,2))
mean_inc_monte_carlo = np.genfromtxt('mean_inc_monte_carlo.csv', delimiter=',') 
file = os.path.join('your pathway')
dfc = pd.read_excel(file, sheet_name='Sheet1',header=1,usecols="C:I",nrows = 15)
concordance_agg = dfc.to_numpy();

for z in range(0,CARLO):
 
   for j in range(1,3):
       
      results =  footprint_calc(mean_inc_monte_carlo[:,j-1],3,2,1) ### monte carlo settings of function
      per_category[:,j-1] = sum(results[1])
      per_category_per_capita = results[1]
      total_energy_footprint_per_capita = np.sum(results[1],1)
      total_expenditure_per_capita = np.sum(results[0],1)
      total_expenditure = sum(np.multiply(total_expenditure_per_capita, bin_size))
      total_energy_footprint_per_capita_GJ = (total_energy_footprint_per_capita/1000)
      total_energy_footprint[j-1] = sum(np.sum(results[1],1)*bin_size)*1e-12 #
      total_energy_inequality[j-1] = gini(np.full(len(mean_inc_monte_carlo), 1),total_energy_footprint_per_capita)

   per_category_share = np.zeros((14,iterations)) 
   for j in range(1,3):
       for i in range(1,15):
          per_category_share[i-1,j-1] = per_category[i-1,j-1]/sum(per_category[:,j-1])

   per_category_share_agg= np.zeros((6,iterations))
   for j in range(1,iterations+1):
      per_category_share_agg[:,j-1] = np.dot(per_category_share[:,j-1],concordance_agg)      
      
   monte_carlo_results_energy_inequality[z,0] = total_energy_inequality[0]
   monte_carlo_results_energy_inequality[z,1] = total_energy_inequality[1]
   monte_carlo_results_share_transport[z,0] = per_category_share_agg[3,0]
   monte_carlo_results_share_transport[z,1] = per_category_share_agg[3,1]
   monte_carlo_results_share_residential[z,0] = per_category_share_agg[1,0]
   monte_carlo_results_share_residential[z,1] = per_category_share_agg[1,1]
   monte_carlo_results_energy_total[z,0] = total_energy_footprint[0]
   monte_carlo_results_energy_total[z,1] = total_energy_footprint[1]
   print("# iteration is " +str(z))
   
   
equal_inquality = monte_carlo_results_energy_inequality[:,0]
unequal_inquality = monte_carlo_results_energy_inequality[:,1]

equal_share = monte_carlo_results_share_transport[:,0]
unequal_share =  monte_carlo_results_share_transport[:,1]

equal_share1 = monte_carlo_results_share_residential[:,0]
unequal_share1 =  monte_carlo_results_share_residential[:,1]

equal_total = monte_carlo_results_energy_total[:,0]
unequal_total =  monte_carlo_results_energy_total[:,1]




fig = plt.figure(figsize = [8, 6])
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[0, 1])
fig.tight_layout()
fig.subplots_adjust(wspace = 1/3)


ax1.boxplot([equal_inquality, unequal_inquality],medianprops = dict(linewidth=2,color='black'))
ax1.set_xticklabels(['equal income', 'unequal income'])
ax1.set_ylabel('Gini coefficient energy',fontsize=11);
ax1.set_ylim(0, 1)
lim_ax1 = ax1.get_ylim()
ax1.text(0.5,lim_ax1[1]+(lim_ax1[1]/100*1.5), 'a', fontsize=12)
ax1.annotate("G = 0.11", xy=(0.7, 0.9), fontsize=12 )
ax1.annotate("G = 0.77", xy=(1.7, 0.9), fontsize=12 )
ax1.axhline(0.71,0.5,1, c= CB_color_cycle[7])
ax1.axhline(0.10,0,0.5, c= CB_color_cycle[7])
ax1.axhline(0.63,0,1, c= 'b',  linestyle='dashed')


ax2.boxplot([equal_share, unequal_share],medianprops = dict(linewidth=3,color='black'))
ax2.set_xticklabels(['equal income', 'unequal income'])
ax2.set_ylabel('share of transport in total energy',fontsize=11);
ax2.set_ylim(0, 0.7)
line1 = ax2.axhline(0.32,0.5,1, c= CB_color_cycle[7])
line2 = ax2.axhline(0.18,0,0.5, c= CB_color_cycle[7])
y1 = equal_share
y2 = unequal_share
x1 = np.random.normal(1, 0.04, size=len(y1))
x2 = np.random.normal(2, 0.04, size=len(y2))
ax2.plot(x1, y1, 'r.', alpha=0.2)
ax2.plot(x2, y2, 'r.', alpha=0.2)
line3 = ax2.axhline(0.26,0,1, c= 'b',  linestyle='dashed')
lim_ax2 = ax2.get_ylim()
ax2.text(0.5, lim_ax2[1]+(lim_ax2[1]/100*1.5), 'c', fontsize=12)
ax2.legend((line1, line3), ('default model run', '2011 parameters'), loc = 'upper left', frameon = False)
#ax2.annotate("high\nuncertainty", xy=(0.6, max(equal_share)+0.06), fontsize=16)


ax3.boxplot([equal_share1, unequal_share1],medianprops = dict(linewidth=3,color='black'))
ax3.set_xticklabels(['equal income', 'unequal income'])
ax3.set_ylabel('share of residential in total energy',fontsize=11);
ax3.set_ylim(0, 0.7);
ax3.axhline(0.37,0.5,1, c= CB_color_cycle[7])
ax3.axhline(0.48,0,0.5, c= CB_color_cycle[7])
y11 = equal_share1
y22 = unequal_share1
x11 = np.random.normal(1, 0.04, size=len(y11))
x22 = np.random.normal(2, 0.04, size=len(y22))
ax3.plot(x11, y11, 'r.', alpha=0.2)
ax3.plot(x22, y22, 'r.', alpha=0.2)
lim_ax3 = ax3.get_ylim()
ax3.text(0.5, lim_ax3[1]+(lim_ax3[1]/100*1.5), 'd', fontsize=12)
ax3.axhline(0.42,0,1, c= 'b',  linestyle='dashed')


ax4.boxplot([equal_total, unequal_total],medianprops = dict(linewidth=3,color='black'))
ax4.set_xticklabels(['equal income', 'unequal income'])
ax4.set_ylabel('total hh energy demand in EJ',fontsize=11);
lim_ax4 = ax4.get_ylim()
txtpos = 300+(lim_ax4[1]/100*2)
ax4.text(0.5, txtpos, 'b', fontsize=12)
ax4.axhline(201,0.5,1, c= CB_color_cycle[7])
ax4.axhline(223,0,0.5, c= CB_color_cycle[7])
ax4.axhline(209,0,1, c= 'b',  linestyle='dashed')
y111 = equal_total
y222 = unequal_total
x111 = np.random.normal(1, 0.04, size=len(y111))
x222 = np.random.normal(2, 0.04, size=len(y222))
ax4.plot(x111, y111, 'r.', alpha=0.2)
ax4.plot(x222, y222, 'r.', alpha=0.2)
ax4.set_ylim(0, 300);
#if lim_ax4[1] > 300: 
    #txtpos = lim_ax4[1]+(lim_ax4[1]/100*1.8) 

plt.savefig('fig7.png',bbox_inches = "tight", dpi = 300)














#############################################################################
######################## SIMULATION #4 ######################################
            
           ###############################################
           ############# Alternative shapes ##############
           ###############################################

############################################################################

import os
os.getcwd()

os.chdir("your pathway)


import pandas as pd 
import math 
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
from gini import *
from lin_fit import *
from footprint_calc import *
from lognorm import *
from scipy.stats import gamma
import math as math
from sympy import Symbol 
from sympy.solvers import solveset
from sympy.solvers import solve
from sympy import erf
from sympy import log
from sympy import sqrt
from sympy import N
from matplotlib import rc
import matplotlib.gridspec as gridspec
from scipy.special import gamma, factorial, erfinv


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', #### color blind friendly list
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

                    ## SIMULATION #4.0 ##                

        ####################################################
             ######## WEIBULL DISTRIBUTION ##############
        #####################################################
        

### LOG-NORMAL MODEL ### equation-based
bins = 1000
q = np.zeros(bins)
probability =  np.linspace((1/bins),1,num=bins);
probability[bins-1] = 1-1/bins*0.1
population_size_real = 7.004E+09
bin_size = population_size_real/bins
population_vector = np.full(bins,bin_size)

j = 12
WB_array_shares = np.zeros((j,14))
WB_array_total = np.zeros((j,1))
WB_array_precent_mega = np.zeros((j,1))
WB_array_percent_below_DLE = np.zeros((j,1))
WB_array_ginis = np.zeros((j,1))
WB_array_mean_inc = np.zeros((1000,j))
WB_array_min_max = np.zeros((2,j))
WB_array_topzeroone =np.zeros((1,j))


mean = 13592


for p in range(0,j):
            k = 1/2+p*1/4 ### shape parameter
            
            helpvalue = gamma(1+1/k)
            
            lamb= mean/helpvalue  ### scale parameter
                  
            bins = 1000000
            probability =  np.linspace((1/bins),1,num=bins);
            probability[bins-1] = 1-1/bins*0.1
            population_size_real = 7.004E+09
            bin_size = population_size_real/bins
            population_vector = np.full(bins,bin_size)
            
            
            
            x = lamb*((-1*np.log(1-probability))**(1/k))
        
            
            x2 = np.pad(x, (1, 0), 'constant')
            x2 = np.delete(x2, bins)
               
            ### compute average income per bin  ####  via numerical integration along the y - axis ####  
               
            integral_bin = np.multiply(x2,population_vector)
            bins2 = 1000
            bin_size2 = population_size_real/bins2
            population_vector2 = np.full(bins2,bin_size2)    
            sum_income_groups= np.add.reduceat(integral_bin, np.arange(0, len(integral_bin), 1000))
            average_income = np.divide(sum_income_groups, population_vector2)
            np.mean(average_income)
            error = mean/np.mean(average_income)
            average_income = np.multiply(average_income,error)
            
            
            
            
            results =  footprint_calc(average_income,2,1,1)
            Gini_income = 1-2**(-1/k)
            
            e_per_category= sum(results[1])
            
            per_category_share = np.zeros((14,1))
            for i in range(0,14):
                    per_category_share[i,0] = e_per_category[i]/sum(e_per_category)
                    
                            
            total_energy_per_capita = np.sum(results[1],1)
            total_energy = sum(np.multiply(total_energy_per_capita,population_vector2))/(10**12)
            total_energy_per_capita_GJ = total_energy_per_capita/1000
            
            
            percent_below_DLE = sum(total_energy_per_capita_GJ <= 26)/10
            percent_mega = sum(total_energy_per_capita_GJ >= 270)/10
            
            
            WB_array_shares[p,:] = np.transpose(per_category_share)
            WB_array_total[p] = total_energy
            WB_array_precent_mega[p] = percent_mega
            WB_array_percent_below_DLE[p] = percent_below_DLE
            WB_array_ginis[p] = Gini_income
            WB_array_mean_inc[:,p]  = average_income
            WB_array_min_max[0,p] = min(total_energy_per_capita_GJ)
            WB_array_min_max[1,p] = max(total_energy_per_capita_GJ)
            WB_array_topzeroone[0,p] = total_energy_per_capita_GJ[999]/sum(total_energy_per_capita_GJ)
            



##### Truncated normal distribution MODEL for comparison cdf ####### needs to be truncated because no Lorenz curves (income inequality measures) can represent negative values

                   ## SIMULATION #4.1 ##
            
        ####################################################
        ######## Truncated normal distribution ##############
        #####################################################
        
        
        

bins = 1000
probability =  np.linspace((1/bins),1,num=bins);
probability[bins-1] = 1-1/bins*0.1
population_size_real = 7.004E+09
bin_size = population_size_real/bins
population_vector = np.full(bins,bin_size)

iterations = 4

tn_array_shares = np.zeros((iterations,14)) ### tn for truncated normal
tn_array_total = np.zeros((iterations,1))
tn_array_precent_mega = np.zeros((iterations,1))
tn_array_percent_below_DLE = np.zeros((iterations,1))
tn_array_ginis = np.zeros((iterations,1))
tn_array_mean_inc = np.zeros((1000,iterations))
tn_array_min_max = np.zeros((2,iterations))
tn_array_topzeroone =np.zeros((1,iterations))

for p in range(0,iterations):

            mu = 13592
            sigma = 1000*(p+1)
            a = 0 
            b = 10**6
            
            
            alpha = (a - mu)/sigma
            beta =  (b - mu)/sigma
            
            Z = (1/2)*(1+math.erf(beta/math.sqrt(2))) - (1/2)*(1+math.erf(alpha/math.sqrt(2)))
            cdf_alpha = (1/2)*(1+math.erf(alpha/math.sqrt(2)))
            
                      
            h = np.add(np.multiply(Z,probability),cdf_alpha) 
            x  = (math.sqrt(2)*erfinv(2*h-1))*sigma+mu
            
            plt.plot(x,probability)
            
                       
            mean_inc = np.zeros(bins) 
                        
            for i in range(1,len(x)+1):
               if (i == 1):
                  mean_inc[i-1] = (0+x[0])/2
               else:
                  mean_inc[i-1] = (x[i-1]+x[i-2])/2
            
            
            np.mean(mean_inc)
            
            Gini_income = gini(population_vector, mean_inc)
            
            
            results = footprint_calc(mean_inc,2,1,1)

            
            e_per_category= sum(results[1])
            
            per_category_share = np.zeros((14,1))
            for i in range(0,14):
                    per_category_share[i,0] = e_per_category[i]/sum(e_per_category)
                    
                            
            total_energy_per_capita = np.sum(results[1],1)
            total_energy = sum(np.multiply(total_energy_per_capita,population_vector))/(10**12)
            total_energy_per_capita_GJ = total_energy_per_capita/1000
            
            
            percent_below_DLE = sum(total_energy_per_capita_GJ <= 26)/10
            percent_mega = sum(total_energy_per_capita_GJ >= 270)/10
            
            
            tn_array_shares[p,:] = np.transpose(per_category_share)
            tn_array_total[p] = total_energy
            tn_array_precent_mega[p] = percent_mega
            tn_array_percent_below_DLE[p] = percent_below_DLE
            tn_array_ginis[p] = Gini_income
            tn_array_mean_inc[:,p]  = mean_inc
            tn_array_min_max[0,p] = min(total_energy_per_capita_GJ)
            tn_array_min_max[1,p] = max(total_energy_per_capita_GJ)
            tn_array_topzeroone[0,p] = total_energy_per_capita_GJ[999]/sum(total_energy_per_capita_GJ)
            print("# of iteration is " + str(p))






                         ## SIMULATION #4.2 ##
            
        #############################################################
                   ########### pareto distribution ##############
        #############################################################



mean = 13592

iterations = 20 

pareto_array_shares = np.zeros((iterations,14)) ### tn for truncated normal
pareto_array_total = np.zeros((iterations,1))
pareto_array_precent_mega = np.zeros((iterations,1))
pareto_array_percent_below_DLE = np.zeros((iterations,1))
pareto_array_ginis = np.zeros((iterations,1))
pareto_array_mean_inc = np.zeros((1000,iterations))
pareto_array_min_max = np.zeros((2,iterations))
pareto_array_topzeroone =np.zeros((1,iterations))

for p in range(0,iterations):
            
            alpha = 5-0.2*p
            x_min = (mean*(alpha-1))/alpha
            
            
            bins = 100000000
            probability =  np.linspace((1/bins),1,num=bins);
            probability[bins-1] = 1-1/bins*0.1
            population_size_real = 7.004E+09
            bin_size = population_size_real/bins
            population_vector = np.full(bins,bin_size)
            
            x = np.divide(x_min,np.power(1-probability,1/alpha))
        
            
            x2 = np.pad(x, (1, 0), 'constant')
            x2 = np.delete(x2, bins)
               
               ### compute average income per bin  ####  via numerical integration along the y - axis ####  
            integral_bin = np.multiply(x2,population_vector)
            bins2 = 1000
            bin_size2 = population_size_real/bins2
            population_vector2 = np.full(bins2,bin_size2)    
            sum_income_groups= np.add.reduceat(integral_bin, np.arange(0, len(integral_bin), 100000))
            average_income = np.divide(sum_income_groups, population_vector2)
            np.mean(average_income)
            error = mean/np.mean(average_income)
            average_income = np.multiply(average_income,error)
            
            Gini_income = 1/(2*alpha-1)
            
            
            results = footprint_calc(average_income,2,1,1)
            e_per_category = sum(results[1])
            
            per_category_share = np.zeros((14,1))
            
            for i in range(0,14):
                    per_category_share[i,0] = e_per_category[i]/sum(e_per_category)
                    
                            
            total_energy_per_capita = np.sum(results[1],1)
            total_energy = sum(np.multiply(total_energy_per_capita,population_vector2))/(10**12)
            total_energy_per_capita_GJ = total_energy_per_capita/1000
            
            pareto_array_min_max[0,p] = min(total_energy_per_capita_GJ)
            pareto_array_min_max[1,p] = max(total_energy_per_capita_GJ)
            percent_below_DLE = sum(total_energy_per_capita_GJ <= 26)/10
            percent_mega = sum(total_energy_per_capita_GJ >= 270)/10
            pareto_array_mean_inc[:,p] = average_income
            pareto_array_shares[p,:] = np.transpose(per_category_share)
            pareto_array_total[p] = total_energy
            pareto_array_precent_mega[p] = percent_mega
            pareto_array_percent_below_DLE[p] = percent_below_DLE
            pareto_array_ginis[p] = Gini_income
            pareto_array_topzeroone[0,p] = total_energy_per_capita_GJ[999]/sum(total_energy_per_capita_GJ)           
            print("# of iteration is " + str(p))



                #plotting inequality vs. other variables over different variables#
                            
#data from lognormal simulation         
                
Gini_income_lognormal = [
            0.110126,
            0.21215,
            0.301693,
            0.3774,
            0.440264,
            0.492235,
            0.535361,
            0.571433,
            0.601899,
            0.627894,
            0.650294,
            0.669779,
            0.686873,
            0.70199,
            0.715453,
            0.727523,
            0.738408,
            0.748278,
            0.757271,
            0.765502,
        ]


energy_total_lognormal = [
            222.387,
            221.254,
            219.634,
            217.801,
            215.952,
            214.194,
            212.574,
            211.107,
            209.784,
            208.595,
            207.524,
            206.556,
            205.678,
            204.879,
            204.148,
            203.477,
            202.858,
            202.285,
            201.752,
            201.256
    ]

    
###subplot total energy demand  
            
plt.plot(tn_array_ginis, tn_array_total, label = 'normal')            
plt.plot(WB_array_ginis, WB_array_total, label = 'Weibull') 
plt.plot(pareto_array_ginis, pareto_array_total, label = 'Pareto')  
plt.plot(Gini_income_lognormal, energy_total_lognormal, label = 'log-normal')
plt.xlim((0.05,0.7))    
plt.xlabel("Gini coefficient income", fontsize = 10);
plt.ylabel('total hh energy in EJ',fontsize = 10);
plt.ylim((200,225)) 
plt.legend()
plt.savefig('suppfig14.png', dpi = 300)

###subplot total energy demand change

plt.plot(tn_array_ginis,  np.divide(tn_array_total,209)-1, label = 'normal')            
plt.plot(WB_array_ginis, np.divide(WB_array_total,209)-1, label = 'Weibull') 
plt.plot(pareto_array_ginis, np.divide(pareto_array_total,209)-1, label = 'Pareto')  
plt.plot(Gini_income_lognormal, np.divide(energy_total_lognormal,209)-1, label = 'log-normal')
plt.xlim((0.05,0.7))    
plt.xlabel("Gini coefficient ", fontsize = 10);
plt.ylabel('total change in hh energy in EJ',fontsize = 10);
#plt.ylim((200,225)) 
plt.legend()
 

np.divide(pareto_array_total,209)-1

###subplot top 0.1% energy consumption as share of total            
plt.plot(WB_array_ginis, np.transpose(WB_array_topzeroone)) 
plt.plot(pareto_array_ginis, np.transpose(pareto_array_topzeroone)) 
plt.xlabel("Gini coefficient", fontsize = 10);
plt.ylabel('top 0.1% share of energy cons.',fontsize = 10);

    
###subplot energy share transport and residential ####

















####################################################################################################################
####################################################################################################################
######## simulaton # 1 again with high resolution numerical integration and quantile function through scipy inv.erf#
####################################################################################################################
####################################################################################################################



import os
os.getcwd()

os.chdir("your pathway"")


import pandas as pd 
import math 
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
from gini import *
from lin_fit import *
from footprint_calc import *
from lognorm import *
from scipy.stats import gamma
import math as math
from sympy import Symbol 
from sympy.solvers import solveset
from sympy.solvers import solve
from sympy import erf
from sympy import log
from sympy import sqrt
from sympy import N
from matplotlib import rc
import matplotlib.gridspec as gridspec
from scipy.special import gamma, factorial, erfinv


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', #### color blind friendly list
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

###### NEW ENGINE ####### put as comments
#bins = 1000000
#probability =  np.linspace((1/bins),1,num=bins);
#probability[bins-1] = 1-1/bins*0.1
#population_size_real = 7.004E+09
#bin_size = population_size_real/bins
#population_vector = np.full(bins,bin_size)

#mu, sigma = 8.723866481,1.259658829 ###



#x =  np.exp(mu+np.multiply(math.sqrt(2*sigma*sigma),erfinv(np.multiply(2,probability)-1)))
#x2 = np.pad(x, (1, 0), 'constant')
#x2 = np.delete(x2, 100000)

#integral_bin = np.multiply(x2,population_vector)

#bins2 = 1000
#bin_size2 = population_size_real/bins2
#population_vector2 = np.full(bins2,bin_size2)


#sum_income_groups= np.add.reduceat(integral_bin, np.arange(0, len(integral_bin), 1000))
#average_income = np.divide(sum_income_groups, population_vector2)


iterations = 20
inequality_sim = np.zeros(iterations)
energy_sim = np.zeros(iterations)
mu, sigma = 8.723866481,1.259658829
bins = 1000
q = np.zeros((bins,iterations))
sf = 10 # scale factor to iterate over sigma
mean_inc = np.zeros((bins,iterations)) 
total_energy_footprint = np.zeros(iterations)
total_energy_inequality = np.zeros(iterations)
total_income = np.zeros(iterations)
Gini_coefficient_income = np.zeros(iterations)
population_size_real = 7.004E+09
bin_size = population_size_real/bins
sigma_array = np.zeros(iterations)
per_category = np.zeros((14,iterations))
percent_below_DLE = np.zeros(iterations)
percent_mega_energy_consumers = np.zeros(iterations)
mean_data = float(np.exp(mu+(sigma*sigma)/2)) ### https://en.wikipedia.org/wiki/Log-normal_distribution and https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-with-specified-mean-and-variance.html
standard_dev_data = float(sqrt((np.exp(sigma*sigma)-1)*np.exp(2*mu+sigma*sigma)))
gini_per_category = np.zeros((14,iterations))

fig = plt.figure(figsize = [9.5, 4])
gs = fig.add_gridspec(1,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])


 
Gini_coeff_income = math.erf(sigma/2)
#gs.update(wspace=0.4, hspace=0.6) 

plt.subplot(ax1)
for j in range(1,np.size(q, 1)+1):
   standard_dev_sim = standard_dev_data/10+(standard_dev_data/10)*(j-1)
   mu_sim = np.log((mean_data*mean_data)/(np.sqrt(standard_dev_sim*standard_dev_sim+mean_data*mean_data))) ### https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-with-specified-mean-and-variance.html natural log https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
   sigma_sim = np.sqrt(np.log(1 + (standard_dev_sim*standard_dev_sim)/(mean_data*mean_data)))
   
   
   ####compute bin thresholds #### high resolution so numerical itnegration is ~ accurate
   
   bins = 1000000
   probability = np.linspace((1/bins),1,num=bins);
   probability[bins-1] = 1-1/bins*0.1
   population_size_real = 7.004E+09
   bin_size = population_size_real/bins
   population_vector = np.full(bins,bin_size)
   x  =  np.exp(mu_sim+np.multiply(math.sqrt(2*sigma_sim*sigma_sim),erfinv(np.multiply(2,probability)-1)))
   x2 = np.pad(x, (1, 0), 'constant')
   x2 = np.delete(x2, 100000)
   
   ### compute average income per bin  ####  via numerical integration along the y - axis ####  
   integral_bin = np.multiply(x2,population_vector)
   bins2 = 1000
   bin_size2 = population_size_real/bins2
   population_vector2 = np.full(bins2,bin_size2)    
   sum_income_groups= np.add.reduceat(integral_bin, np.arange(0, len(integral_bin), 1000))
   average_income = np.divide(sum_income_groups, population_vector2)
   mean_inc[:,j-1] = average_income;     
       
   if (10 == j):
     plt.plot((average_income),np.linspace(1/bins2,1,num=bins2), label = 'data-based', color = 'red', linewidth = 3.5);
   else:
     plt.plot((average_income),np.linspace(1/bins2,1,num=bins2), label = '$\u03C3_X$='+str(round(standard_dev_sim)));
   plt.legend(bbox_to_anchor=(1.3, 0.5), loc='center right', fontsize = 7.5, frameon=False)
   plt.xscale('log')
   plt.xlabel('income (GDP) per capita $ PPP',fontsize=10, labelpad=-1);
   plt.ylabel('cumulative population',fontsize=10);
   sigma_array[j-1] = round(standard_dev_sim);
   results =  footprint_calc(mean_inc[:,j-1],2,1,1) ### argument #2 is elasticities argument #3 is energy intensities they can be filled with integer numbers which relate to different values applied, for details view function "footprint_calc". (2,1) is the default setting.
   per_category[:,j-1] = sum(results[1])
   per_category_per_capita = results[1]
   #for i in range(0,14):
    #  test2 = per_category_per_capita[:,i]
     # gini_per_category[i,j-1] = gini(np.full(len(mean_inc), 1), test2)
   total_energy_footprint_per_capita = np.sum(results[1],1)
   total_expenditure_per_capita = np.sum(results[0],1)
   total_expenditure = sum(np.multiply(total_expenditure_per_capita, bin_size2))
   total_energy_footprint_per_capita_GJ = (total_energy_footprint_per_capita/1000)
   percent_below_DLE[j-1]  = (sum(total_energy_footprint_per_capita_GJ<26)/bins2)*100
   percent_mega_energy_consumers[j-1]  = (sum(total_energy_footprint_per_capita_GJ>270)/bins2)*100 ### mega-consumers = people who consume as much energy as the top 20% Americans or more
   total_energy_footprint[j-1] = sum(np.sum(results[1],1)*bin_size2)*1e-12 ### in exajoule from megajoule this is why 1e-12
   total_energy_inequality[j-1] = gini(np.full(len(mean_inc), 1),total_energy_footprint_per_capita)
   Gini_coefficient_income[j-1] = gini(np.full(len(mean_inc), 1),mean_inc[:,j-1])
   total_income[j-1] = sum(mean_inc[:,j-1])*bin_size2 ###
   plt.xticks(fontsize=9)
   plt.yticks(fontsize=9)
   if (j == iterations):
      plt.text(min(ax1.get_xlim()), 1.15, 'a', fontsize=11)
   print("# iteration is " +str(j))
   fig.tight_layout()


percent_difference_sim =np.subtract(np.divide(total_energy_footprint, total_energy_footprint[9]),1)*100

plt.subplot(ax2)
plt.plot(sigma_array,percent_difference_sim, linewidth = 5);
#plt.yscale('log')
plt.xlabel("$\u03C3_X$", fontsize = 10);
plt.ylabel('net energy change (%)',fontsize = 10);
plt.xticks(fontsize=10);
plt.yticks(fontsize=10);
plt.text(min(Gini_coefficient_income), max(percent_difference_sim)+1.2, 'b', fontsize=11)
plt.plot([sigma_array[9], sigma_array[9]], [min(percent_difference_sim), max(percent_difference_sim)], linestyle = 'dashed', color = 'black', linewidth = 4)
plt.xlim((min(sigma_array), max(sigma_array))) 
plt.ylim((min(percent_difference_sim), max(percent_difference_sim))) 
#plt.plot([40200,40200],[min(percent_difference_sim), max(percent_difference_sim)], linestyle = 'dashed', color = 'black', linewidth = 3)
#plt.plot([13400,13400],[min(percent_difference_sim), max(percent_difference_sim)], linestyle = 'dashed', color = 'black', linewidth = 3)
plt.annotate("data based", xy=(0.6, 0.1), xytext=(0.3, 0),arrowprops=dict(arrowstyle="->"))
axes1 = plt.gca()
axes2 = axes1.twiny()
#axes2.set_xticks([0, 0.4735, 1])
axes2.set_xticks([0, 0.21,0.4735,0.735, 1])
axes2.set_xticklabels([np.round(Gini_coefficient_income[0],2), 0.44, np.round(Gini_coefficient_income[9],2), 0.71, np.round(Gini_coefficient_income[19],2)], fontsize = 9)
# axes2.set_xticks([0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
#axes2.set_xticklabels([np.round(Gini_coefficient_income[0],2), np.round(Gini_coefficient_income[9],2), np.round(Gini_coefficient_income[19],2)], fontsize = 9)
axes2.set_xlabel("Gini coefficient income", fontsize = 10) ### IMPORTANT LABEL MAYBE##
fig.tight_layout()
plt.savefig('fig3.png',bbox_inches = "tight", dpi = 300)


############ FIGURE 4 PLOTTING ##################

per_category_share = np.zeros((14,iterations))
for j in range(1,iterations+1):
    for i in range(1,15):
        per_category_share[i-1,j-1] = per_category[i-1,j-1]/sum(per_category[:,j-1])
 
file = os.path.join('your pathway"')     

dfc = pd.read_excel(file, sheet_name='Sheet1',header=1,usecols="C:I",nrows = 15)
concordance_agg = dfc.to_numpy();

per_category_share_agg= np.zeros((6,iterations))
for j in range(1,iterations+1):
   per_category_share_agg[:,j-1] = np.dot(per_category_share[:,j-1],concordance_agg)

df = pd.DataFrame(np.transpose(per_category_share_agg))
df = df*100


typical_mega = footprint_calc([370000],2,1,1) ### 2,1,1 is default
typical_DLE =footprint_calc([12700],2,1,1)

typical_mega_agg = np.dot(typical_mega[1],concordance_agg)
typical_DLE_agg = np.dot(typical_DLE[1],concordance_agg)

typical_mega_agg_percent = np.zeros(6)
typical_DLE__agg_percent = np.zeros(6)


typical_mega_agg_percent = np.transpose((typical_mega_agg/sum(sum(typical_mega_agg)))*100)
typical_DLE_agg_percent = np.transpose((typical_DLE_agg/sum(sum(typical_DLE_agg)))*100)

N=2
ind = (np.arange(N))

subsistence = (18.1428,6.9795)
heat_elect = (49.4751,17.3443)
health_tech = (8.16245, 5.44703)
transport = (17.0552,58.2235)
edu_recreat_lux = (6.73429,8.0209)           
package_holiday = (0.430082,3.98473)
width = 0.65 
  
def two_scales(ax1, time, data1, data2, c1, c2):
    ax2 = ax1.twinx()
    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('$\u03C3_X$')
    ax1.set_ylabel('per cent < DLE', color = '#377eb8',) 
    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel('per cent mega consumers', color = '#ff7f00')
    ax2.margins(x=0)
    ax2.margins(y=0)
    ax1.margins(x=0)
    ax1.margins(y=0)
    return ax1, ax2



fig, [ax1,ax2,ax3] = plt.subplots(nrows= 1,ncols =3, figsize=(14,4),  gridspec_kw={'width_ratios': [2,2,1.2]})
#ax1.legend(loc='center left',labels= ('subsistence', 'heat and electricity', 'health and tech','transport','edu , recreat., lux.', 'package holiday'),fontsize=9,framealpha=0.5)
#set.xlim((min(sigma_array), max(sigma_array))) 
#ax1.ylim((0,100))
#ax1.xlabel("\u03C3")
#ax1.ylabel("per cent of total")
ax1.stackplot(sigma_array, df[0], df[1], df[2], df[3], df[4], df[5], colors = CB_color_cycle)
ax1.legend(loc='center left',labels= ('subsistence', 'heat and electricity', 'health and tech','transport','edu , recreat., lux.', 'package holiday'),fontsize=9,framealpha=0.5)
ax1.set_xlim((min(sigma_array), max(sigma_array))) 
ax1.set_ylim((0,100))
ax1.set_xlabel("$\u03C3_X$")
ax1.set_ylabel("per cent of total")
ax1.text(2000, 111, 'a', fontsize=12)
ax1.plot([26800,26800],[0,100], linestyle = 'dashed', color = 'black', linewidth = 3)
#ax1.plot([13400,13400],[0,100], linestyle = 'dashed', color = 'black', linewidth = 3)
#ax1.plot([2680,2680],[0,100], linestyle = 'dashed', color = 'black', linewidth = 3)
#ax1.plot([40200,40200],[0,100], linestyle = 'dashed', color = 'black', linewidth = 3)
ax1a = ax1.twiny()
ax1a.set_xticks([0, 0.21,0.4735,0.735, 1])
ax1a.set_xticklabels([np.round(Gini_coefficient_income[0],2), 0.44, np.round(Gini_coefficient_income[9],2), 0.71, np.round(Gini_coefficient_income[19],2)], fontsize = 9)
ax1a.set_xlabel("Gini coefficient income", fontsize = 10) ### IMPORTANT LABE
ax2, ax2a = two_scales(ax2, sigma_array, percent_below_DLE, percent_mega_energy_consumers, '#377eb8', '#ff7f00')
ax2.text(0, max(percent_below_DLE)+7, 'b', fontsize=12)
ax2.plot([26800,26800],[min(percent_below_DLE),max(percent_below_DLE)], linestyle = 'dashed', color = 'black', linewidth = 3)
ax2.margins(x=0)
ax2.margins(y=0)
ax2b = ax2.twiny()
ax2b.set_xticks([0, 0.21,0.4735,0.735, 1])
ax2b.set_xticklabels([np.round(Gini_coefficient_income[0],2), 0.44, np.round(Gini_coefficient_income[9],2), 0.71, np.round(Gini_coefficient_income[19],2)], fontsize = 9)
ax2b.set_xlabel("Gini coefficient income", fontsize = 10) ### IMPORTANT LABE
#plt.stackplot(sigma_array, df[0], df[1], df[2], df[3], df[4], df[5])           
p1 = ax3.bar(ind, subsistence, width)
p2 = ax3.bar(ind, heat_elect,width, bottom = subsistence)
p3 = ax3.bar(ind, health_tech,width, bottom = np.add(heat_elect,subsistence))
p4 = ax3.bar(ind, transport,width, bottom = np.add(np.add(heat_elect,subsistence),health_tech),color = '#f781bf')
p5 = ax3.bar(ind, edu_recreat_lux,width, bottom = np.add(np.add(np.add(heat_elect,subsistence),health_tech),transport), color = '#a65628')
p6 = ax3.bar(ind, package_holiday,width, bottom = np.add(np.add(np.add(np.add(heat_elect,subsistence),health_tech),transport),edu_recreat_lux), color = '#984ea3')
ax3.set_ylabel("per cent of total")
ax3.set_xticklabels(('','low consumer','','mega consumer'))
ax3.set_ylim((0,100))
ax3.text(-0.4, 111, 'c', fontsize=12)
fig.tight_layout()  # otherwise the right y-label is slightly clipped  
plt.savefig('fig4.png',bbox_inches = "tight", dpi = 300)



############ FIGURE 5 PLOTTING ##################

metrics = np.genfromtxt('inequality_metrics_fig4.csv', delimiter=',')


fig = plt.figure(figsize = [4.5, 4])
plt.plot(metrics[:,0],metrics[:,1], label = "food", color = CB_color_cycle[0])
plt.plot(metrics[:,0],metrics[:,2], label = "heat and electricity",  color = CB_color_cycle[1])
plt.plot(metrics[:,0],metrics[:,3], label = "vehicle fuel",  color = CB_color_cycle[2])
plt.plot(metrics[:,0],metrics[:,4], label = "vehicle purchasing",  color = CB_color_cycle[3])
plt.plot(metrics[:,0],metrics[:,5], label = "package holiday",  color = CB_color_cycle[4])
#plt.plot(metrics[:,0],metrics[:,6])
plt.plot([0, 1],[0, 1],color = CB_color_cycle[7], linestyle = "--", label = "income = energy")
plt.plot([0.63, 0.63], [0,1], linestyle = 'dashed', color = 'black', linewidth = 2)
plt.xlim((0,1))
plt.ylim((0,1))
plt.xlabel("Gini coefficient income")
plt.ylabel("Gini coefficient energy")
plt.legend(loc = 'upper left' , fontsize = 7)
axes1 = plt.gca()
axes2 = axes1.twiny()
axes2.set_xticks([0.11, 0.63, 0.77])
#axes2.set_xticks(np.arange(0, 3, 1.0))
axes2.set_xticklabels((2680, 26800, 53600), fontsize = 9)
axes2.set_xlabel("$\u03C3_X$", fontsize = 10) ### IMPORTANT LABEL MAYBE##
plt.savefig('fig5.png',bbox_inches = "tight", dpi = 300)



error_margin_income_average = (1-total_income[0]/total_income[19])*100
