import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import click
from statsmodels.nonparametric.kernel_regression import KernelReg
import statsmodels.api as sm
from scipy.interpolate import interp1d


def computes_mtr_ratios_knowing_earning(earning, taux_marginal, weights, period):
    """
    This takes as input numpy arrays of size number foyers_fiscaux, primary or secondary earnings, and then the marginal tax rates and the weights
    This computes E(T'/(1-T') | yp/s) for all distinct values of primary earings (or secondary earnings)
    So it returns this numpy array of size unique primary/secondary earnings
    """
    mtr_ratio = taux_marginal/(1-taux_marginal)

    unique_earning = np.unique(earning)
    mean_tax_rates = np.zeros_like(unique_earning, dtype=float)

    for i, unique_value in enumerate(unique_earning):
        indices = np.where(earning == unique_value)
        mean_tax_rate = np.average(mtr_ratio[indices], weights=weights[indices])
        mean_tax_rates[i] = mean_tax_rate

    return unique_earning, mean_tax_rates


def find_closest_earning_and_tax_rate(original_earnings, average_ratios, period):
    """
    This takes as input numpy arrays of size unique primary/secondary earnings computed by the above function, 
    that is distinct primary/secondary earnings and the ratios E(T'/(1-T'))
    The function creates a grid of earnings and for each earning of the grid, find the closest "real" earning and the corresponding MTR ratio
    Then it scatters these ratios.
    Since it is a stairs function with stairs overlapping, we decided to smooth the function using a kernel, and we plot the results
    """


    grid_earnings = np.linspace(np.percentile(original_earnings, 1), np.percentile(original_earnings, 99.9), 1000)

    closest_indices = np.argmin(np.abs(original_earnings[:, None] - grid_earnings), axis=0)
    closest_mtr_ratios = average_ratios[closest_indices]

    plt.scatter(grid_earnings, closest_mtr_ratios, label='MTR ratio')   
    plt.xlabel('Gross income')
    plt.ylabel('MTR ratio')
    plt.title("MTR ratio - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/mtr_ratio2_{annee}.png'.format(annee = period))
    plt.close()

    bandwidth = 5000
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

    # another approach lowess : gives almost the same results as before 
    # lowess = sm.nonparametric.lowess(closest_mtr_ratios, grid_earnings, frac = 0.25)
    # smoothed_y_primary = lowess[:, 1]
    # plt.plot(grid_earnings, smoothed_y_primary, label='MTR')   
    # plt.xlabel('Gross income')
    # plt.ylabel('MTR')
    # plt.title("MTR - {annee}".format(annee = period))
    # plt.legend()
    # plt.show()
    # plt.savefig('../outputs/test_cdf/ratio_exp_lowess_{annee}.png'.format(annee = period))
    # plt.close()



    return smoothed_y_primary





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


def util_extensive_revenue_function(earning, total_earning, tax, sec_earnings, dec_earnings, sec_weights, dec_weights):
    """
    This takes as input numpy arrays of size number foyers_fiscaux, primary or secondary earnings, and then total_earnings, taxes 
    Then other inputs are specific to single and dual earner couples (sec and dec)
    This computes the integrand of the extensive revenue function, that is T/(ym - T) * elast * pdfsec/dec * sharesec/dec
    """
    # TODO first compute the esperance separately because has to be on a grid of primary earnings, then the integrand, then the integration
    # separate in 3 steps
    sec_share = np.sum(sec_weights)/(np.sum(sec_weights) + np.sum(dec_weights))

    sec_earnings.sort()
    sec_kde = gaussian_kde(sec_earnings, weights=sec_weights)
    sec_pdf = sec_kde(earning)

    denominator = total_earning+tax
    denominator[denominator == 0] = 0.001

    # tax is negative (that is why the -tax)
    sec_within_integral = (-tax)/denominator * (0.65 - 0.4 * np.sqrt(total_earning/np.percentile(total_earning, 90))) * sec_pdf * sec_share
    
    dec_earnings.sort()
    dec_kde = gaussian_kde(dec_earnings, weights=dec_weights)
    dec_pdf = dec_kde(earning)

    dec_within_integral = (-tax)/denominator * (0.65 - 0.4 * np.sqrt(total_earning/np.percentile(total_earning, 90))) * dec_pdf * (1-sec_share)
    
    return sec_within_integral, dec_within_integral



def compute_extensive_revenue_function(earning, weights, period):
    """
    Compute extensive revenue function
    """
    return 0






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
    work_df = work_df.drop(['idfoy', 'primary_age', 'secondary_age'], axis = 1)
    print(work_df)
    print()

    # get rid of the bottom 5% of the distribution (in terms of total earnings)
    # TODO not good see what Pierre really meant by this
    # threshold = np.percentile( work_df['total_earning'].values, 5)
    # print("threshold income not taken into account", threshold)
    # work_df = work_df[work_df['total_earning'] > threshold]

    
    
    unique_primary_earning, primary_mean_tax_rates = computes_mtr_ratios_knowing_earning(earning = work_df['primary_earning'].values, 
                                            taux_marginal = work_df['taux_marginal'].values, 
                                            weights = work_df['wprm'].values,
                                            period = annee)
    
    primary_mtr_ratios_grid = find_closest_earning_and_tax_rate(original_earnings = unique_primary_earning, 
                                                        average_ratios = primary_mean_tax_rates, 
                                                        period = annee)
    
    unique_secondary_earning, secondary_mean_tax_rates = computes_mtr_ratios_knowing_earning(earning = work_df['secondary_earning'].values, 
                                            taux_marginal = work_df['taux_marginal'].values, 
                                            weights = work_df['wprm'].values,
                                            period = annee)
    
    secondary_mtr_ratios_grid = find_closest_earning_and_tax_rate(original_earnings = unique_secondary_earning, 
                                                        average_ratios = secondary_mean_tax_rates, 
                                                        period = annee)
    

    elasticity_primary = 0.25

    primary_grid, cdf_primary, pdf_primary, intensive_primary_revenue_function = compute_intensive_revenue_function(earning = work_df['primary_earning'].values, 
                    mtr_ratios_grid = primary_mtr_ratios_grid,
                    weights = work_df['wprm'].values, 
                    elasticity = elasticity_primary,
                    period = annee)
    
    secondary_grid, cdf_secondary, pdf_secondary, intensive_secondary_revenue_function = compute_intensive_revenue_function(earning = work_df['secondary_earning'].values, 
                    mtr_ratios_grid = secondary_mtr_ratios_grid,
                    weights = work_df['wprm'].values, 
                    elasticity = 1 - elasticity_primary,
                    period = annee)
    
    df_dual_earner_couples = pd.read_csv(f'./excel/{annee}/dual_earner_couples_25_55_{annee}.csv')
    df_single_earner_couples = pd.read_csv(f'./excel/{annee}/single_earner_couples_25_55_{annee}.csv')

    primary_sec_within_integral, primary_dec_within_integral = util_extensive_revenue_function(earning = work_df['primary_earning'].values, 
                                    total_earning = work_df['total_earning'].values, 
                                    tax = work_df['ancien_irpp'].values, 
                                    sec_earnings = df_single_earner_couples['primary_earning'].values, 
                                    dec_earnings = df_dual_earner_couples['primary_earning'].values, 
                                    sec_weights = df_single_earner_couples['wprm'].values,
                                    dec_weights = df_dual_earner_couples['wprm'].values)




    
    plot_intensive_revenue_function(primary_grid, work_df['primary_earning'].values, cdf_primary, pdf_primary, intensive_primary_revenue_function, secondary_grid, work_df['secondary_earning'].values, cdf_secondary, pdf_secondary, intensive_secondary_revenue_function, work_df['wprm'].values, annee)
    

launch_utils()