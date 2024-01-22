import numpy as np
import pandas as pd

from scipy.stats import percentileofscore

import matplotlib.pyplot as plt
import seaborn as sns

import click 

pd.options.display.max_columns = None


# cf page 48 of the working paper : do each year separately
# page 100 paper 


# maps all values between 0 and 2 to 0, all values between 2 and 4 to 2, ..., all values between 98 and 100 to 98
def custom_round(value):
    return int(value  // 2) * 2


def plot_grid(dataset, period):
    weighted_average = dataset.groupby(['percentile_rounded', 'share_rounded']).apply(lambda x: sum(x['bonus'] * x['weight_foyerfiscal']) / sum(x['weight_foyerfiscal'])).reset_index(name='weighted_avg_bonus')
    print(weighted_average)

    pivot_df = weighted_average.pivot(index='share_rounded', columns='percentile_rounded', values='weighted_avg_bonus')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Weighted Avg Bonus'})
    plt.title('Weighted Average Bonus by Income Percentile and Share Primary - {annee}'.format(annee = period))
    plt.xlabel('Share Primary')
    plt.ylabel('Income Percentile')
    plt.show()
    plt.savefig('../outputs/grid_maps/test_grid_map2_{annee}.png'.format(annee = period))
    plt.close()







@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
def grid_winners_losers(annee = None):
    # We load all datasets needed : single tax schedules with or without dependents (assigned either to primary or secondary earner)
    married_dataset = pd.read_csv(f'./excel/{annee}/married_25_55_positive_{annee}.csv')
    tax_single_with_dependents_primary = pd.read_csv(f'excel/{annee}/tax_single_with_dependents_primary_{annee}.csv')
    tax_single_with_dependents_secondary = pd.read_csv(f'excel/{annee}/tax_single_with_dependents_secondary_{annee}.csv')
    tax_single_without_dependents_primary = pd.read_csv(f'excel/{annee}/tax_single_without_dependents_primary_{annee}.csv')
    tax_single_without_dependents_secondary = pd.read_csv(f'excel/{annee}/tax_single_without_dependents_secondary_{annee}.csv')

    # We merge these datasets alltogether
    merged_dataset = pd.merge(married_dataset, tax_single_with_dependents_primary, how='left', on=['idfoy', 'weight_foyerfiscal'])
    merged_dataset = pd.merge(merged_dataset, tax_single_with_dependents_secondary, how='left', on=['idfoy', 'weight_foyerfiscal'])
    merged_dataset = pd.merge(merged_dataset, tax_single_without_dependents_primary, how='left', on=['idfoy', 'weight_foyerfiscal'])
    merged_dataset = pd.merge(merged_dataset, tax_single_without_dependents_secondary, how='left', on=['idfoy', 'weight_foyerfiscal'])

    # counterfactual tax burden is an AVERAGE over two hypothetical tax burdens in which dependents are allocated to either one of the spouses.
    # To emphasis what we want to do, we average : 
    # - T_single_with_dependents(y1) + T_single_without_dependents(y2) that is considering both spouses singles and assigning dependents to the primary earner
    # - T_single_without_dependents(y1) + T_single_with_dependents(y2) that is considering both spouses singles and assigning dependents to the secondary earner
    merged_dataset['average_single_tax_burden'] = (merged_dataset['tax_single_with_dependents_primary'] + merged_dataset['tax_single_without_dependents_secondary'] + merged_dataset['tax_single_without_dependents_primary'] + merged_dataset['tax_single_with_dependents_secondary'])/2

    # we clear the dataset with the previous 4 tax variables not needed anymore
    merged_dataset = merged_dataset.drop(['tax_single_with_dependents_primary', 'tax_single_without_dependents_primary', 'tax_single_with_dependents_secondary', 'tax_single_without_dependents_secondary'], axis=1)


    # Relative marriage bonuses/penalties relate the absolute monetary advantage
    # from filing as a married couple to the total income of the couple, i.e. Tm(y1+y2)−(Ts(y1)+Ts(y2)) / y1+y2
    merged_dataset['bonus'] = 100*(-merged_dataset['ancien_irpp'] + merged_dataset['average_single_tax_burden'])/merged_dataset['total_earning']

    # we add the share of the primary and the gross income percentile
    merged_dataset['share_primary'] = 100 * merged_dataset['primary_earning']/(merged_dataset['primary_earning'] + merged_dataset['secondary_earning'])
    
    # TODO weight this 
    merged_dataset['income_percentile'] = merged_dataset['total_earning'].apply(lambda x: percentileofscore(merged_dataset['total_earning'], x, kind='rank'))

    
    merged_dataset = merged_dataset[['weight_foyerfiscal', 'income_percentile', 'share_primary', 'bonus']]

    merged_dataset['percentile_rounded'] = np.vectorize(custom_round)(merged_dataset['income_percentile'])
    merged_dataset['share_rounded'] = np.vectorize(custom_round)(merged_dataset['share_primary'])
    merged_dataset = merged_dataset.drop(['income_percentile', 'share_primary'], axis=1)

    print(merged_dataset)
    print(np.sum(merged_dataset['bonus'] > 0))

    plot_grid(merged_dataset, annee)
    


grid_winners_losers()


"""
python simulation_creation.py -y 2005
python reform_single_schedule.py -y 2005 -p True
python reform_single_schedule.py -y 2005 -p False
python reform_single_without_dependents.py -y 2005 -p True
python reform_single_without_dependents.py -y 2005 -p False
python grid_winners_losers.py -y 2005
"""



