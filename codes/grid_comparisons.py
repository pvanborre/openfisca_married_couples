import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import click

@click.command()
@click.option('-b', '--year_pre', default = 2005, type = int, required = True)
@click.option('-a', '--year_post', default = 2006, type = int, required = True)
def grid_comparisons(year_pre = None, year_post = None):
    print("Years under consideration", year_pre, year_post)

    data_pre = pd.read_csv(f'./excel/{year_pre}/grid_marriage_bonus_{year_pre}.csv')
    data_post = pd.read_csv(f'./excel/{year_post}/grid_marriage_bonus_{year_post}.csv')


    merged_df = pd.merge(data_pre, data_post, on=['percentile_rounded', 'share_rounded'], suffixes=('_year1', '_year2'))

    merged_df['bonus_difference'] = merged_df['weighted_avg_bonus_year2'] - merged_df['weighted_avg_bonus_year1']

    pivot_diff_df = merged_df.pivot(index='share_rounded', columns='percentile_rounded', values='bonus_difference')



    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_diff_df, annot=True, fmt=".0f", cmap="coolwarm", cbar_kws={'label': 'Bonus Difference'})
    plt.title('Bonus Difference Grid ({Year2} - {Year1})'.format(Year2 = year_post, Year1 = year_pre))
    plt.xlabel('Income Percentile')
    plt.ylabel('Share primary')
    plt.show()
    plt.savefig('../outputs/grid_maps/diff_{Year2}_{Year1}.png'.format(Year2 = year_post, Year1 = year_pre))
    plt.close()

grid_comparisons()