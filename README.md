# global_redistribution_income_energy_footprints
data and code for the paper Oswald et al. (2021) Global redistribution of income and household energy footprints: A computational thought experiment

1) code.py is the main script of code for the paper
2) data contains underlying critical data necessary to reproduce analysis and research process
3) footprint_calc.py is a function to actually compute energy footprints via power laws
4) gini.py computes the gini coefficient
5) several csv files (one xlsx file) that are directly loaded into the mainscript (so they are necessary to run the script also) and that are sometimes not our primary data but adjusted data by other openly accessible sources. The credit for this data thus is completely due to their primary authors which are always properly cited in the paper. But the data is uploaded here so that our script and work is entirely reproducible. Listed in the following
     1. cumulativepincome_alv ||| necessary for Figure 1 population vector of adjusted Alvaredo data ||| originating from World Inequality Lab/ World Inequality Report 2018
     2. incomeranked_alv  ||| to 1. corresponding income vector
     3. laknercumpop ||| necessary for Figure 1 ||| original from Lakner and Milanovic https://openknowledge.worldbank.org/handle/10986/16935
     4. laknerincomeranked ||| to 3. corresponding income vector
     5. inequality_metrics_fig4 ||| data to reproduce fig.4 
     6. expenditure_real_world.xlsx || data to compare expenditure distributions real world vs. model 
