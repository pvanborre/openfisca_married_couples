import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import click
from statsmodels.nonparametric.kernel_regression import KernelReg






def mtr_v2(earning, taux_marginal, weights, period):
    mtr_ratio = taux_marginal/(1-taux_marginal)

    unique_earning = np.unique(earning)
    mean_tax_rates = np.zeros_like(unique_earning, dtype=float)

    for i, unique_value in enumerate(unique_earning):
        indices = np.where(earning == unique_value)
        mean_tax_rate = np.average(mtr_ratio[indices], weights=weights[indices])
        mean_tax_rates[i] = mean_tax_rate

    # plt.scatter(unique_earning, mean_tax_rates, label='MTR ratio')   
    # plt.xlabel('Gross income')
    # plt.ylabel('MTR ratio')
    # plt.title("MTR ratio - {annee}".format(annee = period))
    # plt.legend()
    # plt.show()
    # plt.savefig('../outputs/test_cdf/mtr_ratio3_{annee}.png'.format(annee = period))
    # plt.close()

    return unique_earning, mean_tax_rates

def find_closest_earning_and_tax_rate(original_earnings, average_marginal_tax_rates, period):
    grid_earnings = np.linspace(np.percentile(original_earnings, 1), np.percentile(original_earnings, 99.9), 1000)

    closest_indices = np.argmin(np.abs(original_earnings[:, None] - grid_earnings), axis=0)
    closest_mtr_ratios = average_marginal_tax_rates[closest_indices]

    plt.scatter(grid_earnings, closest_mtr_ratios, label='MTR ratio')   
    plt.xlabel('Gross income')
    plt.ylabel('MTR ratio')
    plt.title("MTR ratio - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/mtr_ratio2_{annee}.png'.format(annee = period))
    plt.close()

    bandwidth = 6000
    kernel_reg = KernelReg(endog=closest_mtr_ratios, exog=grid_earnings, var_type='c', reg_type='ll', bw=[bandwidth], ckertype='gaussian')
    smoothed_y_primary, _ = kernel_reg.fit()
    plt.plot(grid_earnings, smoothed_y_primary, label='MTR')   
    plt.xlabel('Gross income')
    plt.ylabel('MTR')
    plt.title("MTR - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/ratio_exp_{bandwidth}_{annee}.png'.format(bandwidth=bandwidth, annee = period))
    plt.close()

    return closest_mtr_ratios





def compute_intensive_revenue_function(earning, mtr_ratios_grid, weights, elasticity, period):
    """
    Computes the pdf and cdf of earnings, and the intensive revenue function
    """

    earning.sort()

    kde = gaussian_kde(earning, weights=weights)
    x_values = np.linspace(np.percentile(earning, 1), np.percentile(earning, 99.9), 1000)

    pdf = kde(x_values)
    cdf = np.cumsum(pdf) / np.sum(pdf)

    behavioral = - x_values * pdf * elasticity * mtr_ratios_grid
    mechanical = 1 - cdf
    intensive_revenue_function = behavioral + mechanical

    return x_values, cdf, pdf, intensive_revenue_function

def plot_intensive_revenue_function(primary_grid, primary_earning, cdf_primary, pdf_primary, intensive_primary_revenue_function, secondary_grid, secondary_earning, cdf_secondary, pdf_secondary, intensive_secondary_revenue_function, weights, period):
    

    plt.plot(primary_grid, cdf_primary, label='primary') 
    plt.plot(secondary_grid, cdf_secondary, label='secondary')   
    plt.xlabel('Gross income')
    plt.ylabel('CDF')
    plt.title("Cumulative distribution function - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/cdf_{annee}.png'.format(annee = period))
    plt.close()
    
    plt.plot(primary_grid, pdf_primary, label='primary') 
    plt.plot(secondary_grid, pdf_secondary, label='secondary')   
    plt.xlabel('Gross income')
    plt.ylabel('PDF')
    plt.title("Probability distribution function - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/pdf_{annee}.png'.format(annee = period))
    plt.close()

    plt.plot(primary_grid, intensive_primary_revenue_function, label='primary') 
    plt.plot(secondary_grid, intensive_secondary_revenue_function, label='secondary')    
    plt.xlabel('Gross income')
    plt.ylabel('Integral revenue function')
    plt.title("Integral revenue function - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/revenue_function_{annee}.png'.format(annee = period))
    plt.close()

    integral_trap_primary = np.trapz(intensive_primary_revenue_function, primary_grid)
    print("Primary revenue function integral", integral_trap_primary)
    integral_trap_secondary = np.trapz(intensive_secondary_revenue_function, secondary_grid)
    print("Secondary revenue function integral", integral_trap_secondary)

    ratio = integral_trap_primary/integral_trap_secondary
    print("ratio", ratio)

    is_winner = secondary_earning*ratio > primary_earning
    pourcentage_gagnants = 100*np.sum(is_winner*weights)/np.sum(weights)
    print("Pourcentage de gagnants", period, pourcentage_gagnants)






@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
@click.option('-m', '--want_to_mute_decote', default = False, type = bool, required = True)
def launch_utils(annee = None, want_to_mute_decote = None):
    work_df = pd.read_csv(f'./excel/{annee}/married_25_55_{annee}.csv')
    print(work_df)

    
    unique_primary_earning, primary_mean_tax_rates = mtr_v2(earning = work_df['primary_earning'].values, 
                                            taux_marginal = work_df['taux_marginal'].values, 
                                            weights = work_df['wprm'].values,
                                            period = annee)
    
    primary_mtr_ratios_grid = find_closest_earning_and_tax_rate(original_earnings = unique_primary_earning, 
                                                        average_marginal_tax_rates = primary_mean_tax_rates, 
                                                        period = annee)
    
    unique_secondary_earning, secondary_mean_tax_rates = mtr_v2(earning = work_df['secondary_earning'].values, 
                                            taux_marginal = work_df['taux_marginal'].values, 
                                            weights = work_df['wprm'].values,
                                            period = annee)
    
    secondary_mtr_ratios_grid = find_closest_earning_and_tax_rate(original_earnings = unique_secondary_earning, 
                                                        average_marginal_tax_rates = secondary_mean_tax_rates, 
                                                        period = annee)
    

    primary_grid, cdf_primary, pdf_primary, intensive_primary_revenue_function = compute_intensive_revenue_function(earning = work_df['primary_earning'].values, 
                    mtr_ratios_grid = primary_mtr_ratios_grid,
                    weights = work_df['wprm'].values, 
                    elasticity = 0.25,
                    period = annee)
    
    secondary_grid, cdf_secondary, pdf_secondary, intensive_secondary_revenue_function = compute_intensive_revenue_function(earning = work_df['secondary_earning'].values, 
                    mtr_ratios_grid = secondary_mtr_ratios_grid,
                    weights = work_df['wprm'].values, 
                    elasticity = 0.75,
                    period = annee)
    
    plot_intensive_revenue_function(primary_grid, work_df['primary_earning'].values, cdf_primary, pdf_primary, intensive_primary_revenue_function, secondary_grid, work_df['secondary_earning'].values, cdf_secondary, pdf_secondary, intensive_secondary_revenue_function, work_df['wprm'].values, annee)
    

launch_utils()