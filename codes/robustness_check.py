import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import click

# robustness check for year 2006 : we do not change primary and secondary earnings for 2006
# but we take as marginal tax rates the corresponding values in 2005

@click.command()
@click.option('-b', '--year_pre', default = 2005, type = int, required = True)
@click.option('-a', '--year_post', default = 2006, type = int, required = True)
def robustness_check(year_pre = None, year_post = None):
    print("Years under consideration", year_pre, year_post)

    data_pre = pd.read_csv(f'./excel/{year_pre}/married_25_55_positive_{year_pre}.csv')
    data_post = pd.read_csv(f'./excel/{year_post}/married_25_55_positive_{year_post}.csv')

    # for the pre year, we only keep the marginal tax rate column and the total earnings associated
    data_pre = data_pre[['total_earning', 'taux_marginal']]
    data_pre = data_pre.sort_values(by='total_earning')

    # for data_post we are going to create a new column taux_marginal (MTR) so we drop the old one (give it a new name to maybe compare it to the new MTR)
    data_post.rename(columns={'taux_marginal': 'old_taux_marginal'}, inplace=True)
    data_post = data_post.sort_values(by='total_earning')



    # find the closest total_earning in data_pre for a given total_earning in data_post
    def find_closest_total_earning(total_earning):
        index = data_pre['total_earning'].searchsorted(total_earning)
        if index == 0:
            return data_pre.iloc[0]['taux_marginal']
        elif index == len(data_pre):
            return data_pre.iloc[-1]['taux_marginal']
        else:
            prev_total_earning = data_pre.iloc[index - 1]['total_earning']
            next_total_earning = data_pre.iloc[index]['total_earning']
            if abs(total_earning - prev_total_earning) < abs(total_earning - next_total_earning):
                return data_pre.iloc[index - 1]['taux_marginal']
            else:
                return data_pre.iloc[index]['taux_marginal']

    # map the taux_marginal values from data_pre to data_post based on the closest total_earning
    data_post['taux_marginal'] = data_post['total_earning'].apply(find_closest_total_earning)


    #data_post['diff'] = data_post['old_taux_marginal'] != data_post['taux_marginal']

    print(data_post)
    data_post.to_csv(f'excel/{year_post}/robustness_married_25_55_positive_{year_post}.csv', index=False)



robustness_check()

# TODO also the reverse operation






















