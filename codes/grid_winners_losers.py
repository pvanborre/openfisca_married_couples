import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


year_pre, year_post = 2005, 2006
data_pre = pd.read_csv(f'./excel/{year_pre}/married_25_55_positive_{year_pre}.csv')
data_post = pd.read_csv(f'./excel/{year_post}/married_25_55_positive_{year_post}.csv')


# cf page 48 of the working paper : do each year separately, so will be done in utils_paper.py at the end 

data_pre['share_primary'] = 100 * data_pre['primary_earning']/(data_pre['primary_earning'] + data_pre['secondary_earning'])


print(data_pre)
"""
Relative marriage bonuses/penalties relate the absolute monetary advantage
from filing as a married couple to the total income of the couple, i.e. Tm(y1+y2)−(Ts(y1)+Ts(y2)) / y1+y2
"""

# page 100 paper 
# code reform that computes a primary tax burden that would be as if it was a single man (account for a single declaration)
# but how to allocate dependents ? we allocate them to one spouse only 

