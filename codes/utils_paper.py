import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.integrate import trapz

from statsmodels.nonparametric.kernel_regression import KernelReg

import click
import sys


################################################################################
################# Intensive revenue function ###################################
################################################################################

def computes_mtr_ratios_knowing_earning(earning, taux_marginal, weights):
    """
    This takes as input numpy arrays of size number foyers_fiscaux : primary or secondary earnings, marginal tax rates and weights
    This computes E(T'/(1-T') | y1/2) for all distinct values of primary earings (or secondary earnings)
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



def util_intensive_revenue_function(grid_earnings, original_earnings, average_ratios, name, period):
    """
    This takes as input a grid of earnings 
    and 2 numpy arrays of size unique primary/secondary earnings computed by the above function, 
    that is distinct primary/secondary earnings (original_earnings) and the ratios E(T'/(1-T')) (average_ratios)
    The function fits a gaussian kernel to interpolate these ratios on the grid of earnings, 
    and returns the fitted function (np array of the same size of the grid)
    It also scatters all the ratios and plots the smoothing over the grid 
    """

    bandwidth = 5000
    # define on all FoyersFiscaux
    kernel_reg = KernelReg(endog=average_ratios, exog=original_earnings, var_type='c', reg_type='ll', bw=[bandwidth], ckertype='gaussian')
    # fit on the grid of earnings
    smoothed_y_primary, _ = kernel_reg.fit(grid_earnings)
    
    plt.scatter(original_earnings, average_ratios, label='MTR ratio without smoothing', color = 'lightgreen')
    plt.plot(grid_earnings, smoothed_y_primary, label='MTR ratio with smoothing', color = 'red')   
    plt.xlabel('Gross income')
    plt.ylabel("MTR ratio T'/(1-T')")
    plt.title("MTR ratio - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/mtr_ratio_by_{name}/mtr_ratio_by_{name}_{annee}.png'.format(name = name, annee = period))
    plt.close()

    return smoothed_y_primary





def compute_intensive_revenue_function(grid_earnings, earning, mtr_ratios_grid, weights, elasticity):
    """
    Computes the pdf and cdf of earnings, and the intensive revenue function
    """

    sorted_indices = np.argsort(earning)
    earning_sorted = earning[sorted_indices]
    weights_sorted = weights[sorted_indices]

    kde = gaussian_kde(earning_sorted, weights=weights_sorted)
    pdf = kde(grid_earnings)
    cdf = np.cumsum(pdf) / np.sum(pdf)

    behavioral = - grid_earnings * pdf * elasticity * mtr_ratios_grid
    mechanical = 1 - cdf
    intensive_revenue_function = behavioral + mechanical

    return cdf, pdf, intensive_revenue_function




################################################################################
################# Extensive revenue function ###################################
################################################################################

def computes_tax_ratios_knowing_earning(earning, total_earning, tax, weights, period):
    """
    This takes as input numpy arrays of size number foyers_fiscaux : primary or secondary earnings, total earnings, tax before reform and weights
    This computes E(T/(ym-T) * pelast | y1/2) for all distinct values of primary earings (or secondary earnings)
    To do so, it first computes the participation elasticities pelast and plots it 
    And then it can compute expected means (like in the function computes_mtr_ratios_knowing_earning)
    So it returns this numpy array of size unique primary/secondary earnings
    """
    ########## Computing the participation elasticity ####################

    # 0.65 - 0.4 * np.sqrt(total_earning/np.percentile(total_earning, 90)) is the formula we assume in the paper for the extensive elasticity
    participation_elasticity = 0.65 - 0.4 * np.sqrt(total_earning/np.percentile(total_earning, 90))
    participation_elasticity[total_earning > np.percentile(total_earning, 90)] = 0.25
    
    sorted_indices = np.argsort(total_earning)
    total_earning_sorted = total_earning[sorted_indices]
    participation_elasticity_sorted = participation_elasticity[sorted_indices]
    
    plt.plot(total_earning_sorted, participation_elasticity_sorted)   
    plt.xlabel('Gross income')
    plt.ylabel("Participation elasticity")
    plt.title("Participation elasticity - {annee}".format(annee = period))
    plt.ylim(0, 1)
    plt.show()
    plt.savefig('../outputs/participation_elasticity/participation_elasticity_{annee}.png'.format(annee = period))
    plt.close()

    ########## Computing the expected mean ####################

    # tax is negative (that is why the -tax)
    denominator = total_earning+tax
    denominator[denominator == 0] = 1000 
    # here I can put a random value, this won't affect the extensive revenue function
    # since in the integral from y1 to ymax we don't have the y1 = 0 value (that is equivalent to total_earning = 0)
    
    tax_ratio = (-tax)/denominator * participation_elasticity

    unique_earning = np.unique(earning)
    mean_tax_rates = np.zeros_like(unique_earning, dtype=float)

    for i, unique_value in enumerate(unique_earning):
        indices = np.where(earning == unique_value)
        mean_tax_rate = np.average(tax_ratio[indices], weights=weights[indices])
        mean_tax_rates[i] = mean_tax_rate

    return unique_earning, mean_tax_rates






def util_extensive_revenue_function(grid_earnings, original_earnings, average_ratios, sec_earnings, dec_earnings, sec_weights, dec_weights, period, name):
    """
    This takes as input first a grid of earnings, and then 
    numpy arrays of size number unique foyers_fiscaux : primary or secondary earnings, average ratios E(T/(ym-T) * pelast)
    Then other inputs are specific to single and dual earner couples (sec and dec)
    This computes the integrand of the extensive revenue function on a specific grid of earnings, that is E(T/(ym-T) * pelast) * pdfsec/dec * sharesec/dec 
    To do so, we first fit using a gaussan kernel our E(T/(ym-T) * pelast), and then we compute our pdf 
    """

    if name == "primary":
        bandwidth = 8000
    else:
        bandwidth = 4000

    # define on all FoyersFiscaux
    kernel_reg = KernelReg(endog=average_ratios, exog=original_earnings, var_type='c', reg_type='ll', bw=[bandwidth], ckertype='gaussian')
    # fit on the grid of earnings
    ratio_fit, _ = kernel_reg.fit(grid_earnings)
    plt.scatter(original_earnings, average_ratios, label='Extensive expectation without smoothing', color = 'lightgreen')
    plt.plot(grid_earnings, ratio_fit, label='Extensive expectation with smoothing', color = 'red')   
    plt.xlabel('Gross income')
    plt.ylabel("E(Tm/(ym-Tm) pelast)")
    plt.title("Extensive expectation - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/extensive_expectation_{name}/extensive_expectation_{name}_{annee}.png'.format(name = name, annee = period))
    plt.close()

    # then computes the pdf and the integrand, for both single and dual earner couples
    sec_share = np.sum(sec_weights)/(np.sum(sec_weights) + np.sum(dec_weights))

    sorted_indices_dec = np.argsort(dec_earnings)
    dec_earnings_sorted = dec_earnings[sorted_indices_dec]
    dec_weights_sorted = dec_weights[sorted_indices_dec]

    dec_kde = gaussian_kde(dec_earnings_sorted, weights=dec_weights_sorted)
    dec_pdf = dec_kde(grid_earnings)
    dec_within_integral = ratio_fit * dec_pdf * (1-sec_share)

    if name == "primary":
        sorted_indices_sec = np.argsort(sec_earnings)
        sec_earnings_sorted = sec_earnings[sorted_indices_sec]
        sec_weights_sorted = sec_weights[sorted_indices_sec]

        sec_kde = gaussian_kde(sec_earnings_sorted, weights=sec_weights_sorted)
        sec_pdf = sec_kde(grid_earnings) 
        sec_within_integral = ratio_fit * sec_pdf * sec_share
        return sec_within_integral, dec_within_integral
    
    else: # the single earner secondary has no sense, since all secondary earnings are then 0
        return 0, dec_within_integral



def compute_extensive_revenue_function(grid_earnings, within_integral):
    """
    Compute extensive revenue function : it boils down to the computation of an integral in a cumulative way
    """

    cumulative_integrals = np.zeros_like(grid_earnings) # integrals from y1_0 to all y1's

    for i in range(len(grid_earnings) - 1):
        cumulative_integrals[i + 1] = cumulative_integrals[i] + trapz(within_integral[i:i + 2], grid_earnings[i:i + 2])

    # - integral from y1 to y1_max = integral from y1_0 to y1 - integral from y1_0 to y1_max
    return cumulative_integrals - cumulative_integrals[-1]




################################################################################
################# Altogether ###################################################
################################################################################



def winners_political_economy(primary_grid, primary_earning, primary_mtr_ratios_grid, extensive_primary_revenue_function, secondary_grid, secondary_earning, secondary_mtr_ratios_grid, extensive_secondary_revenue_function, weights, period):
    """
    For different elasticities scenarios we :
        - compute the intensive revenue function and then add the extensive revenue function (this extensive part does not depend on the elasticity)
        - plot the primary and secondary revenue functions
        - compute the integrals of these primary and secondary revenue functions and then deduce the ratio between these 2 integrals
        - with this ratio, compute the percentage of winners of a reform towards individualization in this elasticity scenario

    Then we plot also the lines that separates winners from losers of such a reform
    Finally we also plot pdf and cdf for primary and secondary earnings, that were used to compute the intensive revenue function
    """

    eps1_tab = [0.25, 0.5, 0.75]
    eps2_tab = [0.75, 0.5, 0.25]
    rapport = [0.0]*len(eps1_tab)
    pourcentage_gagnants = [0.0]*len(eps1_tab)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ################################################################################
    ################# Computation of revenues functions ############################
    ################################################################################
    for i in range(len(eps1_tab)):
        cdf_primary, pdf_primary, intensive_primary_revenue_function = compute_intensive_revenue_function(
                    grid_earnings = primary_grid,
                    earning = primary_earning, 
                    mtr_ratios_grid = primary_mtr_ratios_grid,
                    weights = weights, 
                    elasticity = eps1_tab[i])
         
        primary_revenue_function = intensive_primary_revenue_function + extensive_primary_revenue_function

        cdf_secondary, pdf_secondary, intensive_secondary_revenue_function = compute_intensive_revenue_function(
                    grid_earnings = secondary_grid,
                    earning = secondary_earning, 
                    mtr_ratios_grid = secondary_mtr_ratios_grid,
                    weights = weights, 
                    elasticity = eps2_tab[i])
         
        secondary_revenue_function = intensive_secondary_revenue_function + extensive_secondary_revenue_function

        integral_trap_primary = np.trapz(primary_revenue_function, primary_grid)
        print("Primary revenue function integral", integral_trap_primary)
        integral_trap_secondary = np.trapz(secondary_revenue_function, secondary_grid)
        print("Secondary revenue function integral", integral_trap_secondary)

        rapport[i] = integral_trap_primary/integral_trap_secondary
        print('rapport integrales scenario ', i, " ", rapport[i])

        is_winner = secondary_earning*rapport[i] > primary_earning
        pourcentage_gagnants[i] = 100*np.sum(is_winner*weights)/np.sum(weights)
        print("Scenario", i)
        print("Pourcentage de gagnants", period, i, pourcentage_gagnants[i])

        axes[i].plot(primary_grid, primary_revenue_function, label = 'primary ep = {ep}, es = {es}'.format(ep = eps1_tab[i], es = eps2_tab[i]))
        axes[i].plot(secondary_grid, secondary_revenue_function, label = 'secondary ep = {ep}, es = {es}'.format(ep = eps1_tab[i], es = eps2_tab[i]))
        
        axes[i].legend()


        axes[i].set_xlabel('Gross income')
        axes[i].set_ylabel('R function')
        axes[i].set_title('Scenario {}'.format(i))




    plt.tight_layout()  
    plt.show()
    plt.savefig('../outputs/revenue_functions/revenue_functions_{annee}.png'.format(annee = period))
    plt.close()

    ################################################################################
    ################# Separate winners from losers #################################
    ################################################################################
    plt.figure()
    x = np.linspace(0, 600000, 4)
    plt.plot(x, x, c = '#828282')

    green_shades = [(0.0, 1.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.6, 0.0)]
    for i in range(len(eps1_tab)):
        color = green_shades[i]
        plt.plot(x, rapport[i]*x, label = "ep = {ep}, es = {es}".format(ep = eps1_tab[i], es = eps2_tab[i]), color=color)
        plt.annotate(str(round(pourcentage_gagnants[i]))+ " %", xy = (max(secondary_earning)/4*(i+1), max(primary_earning)/2), bbox = dict(boxstyle ="round", fc = color))

    plt.scatter(secondary_earning, primary_earning, s = 0.1, c = '#828282') 

    #plt.axis("equal")

    eps = 5000
    plt.xlim(-eps, max(secondary_earning)) 
    plt.ylim(-eps, max(primary_earning)) 

    plt.grid()
    


    plt.xlabel('Secondary earner')
    plt.ylabel('Primary earner')
    plt.title("Reform towards individual taxation: Political economy - {}".format(period))

    plt.legend()
    plt.show()
    plt.savefig('../outputs/winners_political_economy/winners_political_economy_{annee}.png'.format(annee = period))
    plt.close()

    ################################################################################
    ################# Plot cdf and pdf #############################################
    ################################################################################
    plt.plot(primary_grid, cdf_primary, label='primary') 
    plt.plot(secondary_grid, cdf_secondary, label='secondary')   
    plt.xlabel('Gross income')
    plt.ylabel('CDF')
    plt.title("Cumulative distribution function - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/cdf_primary_secondary/cdf_primary_secondary_{annee}.png'.format(annee = period))
    plt.close()
    
    plt.plot(primary_grid, pdf_primary, label='primary') 
    plt.plot(secondary_grid, pdf_secondary, label='secondary')   
    plt.xlabel('Gross income')
    plt.ylabel('PDF')
    plt.title("Probability distribution function - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/pdf_primary_secondary/pdf_primary_secondary_{annee}.png'.format(annee = period))
    plt.close()

    return rapport

    


################################################################################
################# Welfare ######################################################
################################################################################


def median_share_primary(primary_earning, total_earning, weights, period):
    """
    Here we compute the share of the primary earning compared to the foyer fiscal total earning 
    For each total earning decile we then compute the median of this share
    """
    total_earning[total_earning == 0] = 0.001
    primary_earning[total_earning == 0] = 0.001 #so that we get a share of 100% for the couples who earn both 0 (because since we limited to nonnegative primary and secondary earning, a total earning equals to 0 means that both primary and secondary earnings equal 0)
    share_primary = primary_earning/total_earning * 100 

    earning_sorted = np.sort(total_earning)
    deciles = np.percentile(earning_sorted, np.arange(10, 100, 10)) # gives 9 values in total_earning that correspond to the decile values
    decile_numbers = np.digitize(total_earning, deciles) # gives for each value of total earning the decile (number between 0 and 9) it is associated to

    
    decile_medians = []
    for i in range(10):
        decile_share_primary = share_primary[decile_numbers == i] # we only keep values of the ith decile to get subarrays
        decile_weights = weights[decile_numbers == i]

        # computes the weighted median, i was inspired by https://stackoverflow.com/questions/20601872/numpy-or-scipy-to-calculate-weighted-median
        sorted_indices = np.argsort(decile_share_primary)
        cumulative_weights = np.cumsum(decile_weights[sorted_indices]) # we sort the weights according to earnings, and then build a cumulative tab of weights
        median_index = np.searchsorted(cumulative_weights, 0.5 * cumulative_weights[-1]) # we take the sum of all weights divided by 2, and we look for the indice where it would be inserted without changing the order (that is the median weight position)
        median_index_unsorted = sorted_indices[median_index]
        decile_medians.append(decile_share_primary[median_index_unsorted]) 

    
    decile_medians = np.array(decile_medians)
    print("share of primary for year", period, decile_medians)

    plt.figure()
    plt.scatter(np.arange(1,11), decile_medians, s = 10)
    plt.xticks(np.arange(1,11))
    plt.xlabel('Gross income decile')
    plt.ylabel('Percent')
    plt.title("Median share of primary earner - {annee}".format(annee = period))
    plt.show()
    plt.savefig('../outputs/median_share_primary/median_share_primary_{annee}.png'.format(annee = period))
    plt.close()

    
def main_welfare_graph(primary_earning, secondary_earning, total_earning, weights, slopes_lines, period):
    """
    We plot the same graph as for the winners analysis, that is the lines that would separate winners from losers with different elasticities
    Moreover, we produce a welfare analysis for different weights specifications, that is
    we compute E(welfare_weight * primary_earning) and E(welfare_weight * secondary_earning) and 
    like before, evaluate whether a reform is welfare damaging if the point lies below the line, welfare increasing above the line 
    """
    eps1_tab = [0.25, 0.5, 0.75]
    eps2_tab = [0.75, 0.5, 0.25]

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    # we create a fictive individual that accounts for the bottom 5% of the distribution (that is why we sum the weights)
    # the earnings of this fictive individual are the mean of the bottom 5% earnings 
    # however I'm not convinced about why we do that ? what does it bring ?
    threshold = np.percentile(total_earning, 5)
    print("threshold income not taken into account for welfare", threshold)
    primary_earning = np.append(primary_earning[total_earning > threshold], np.mean(primary_earning[total_earning <= threshold]))
    secondary_earning = np.append(secondary_earning[total_earning > threshold], np.mean(secondary_earning[total_earning <= threshold]))
    weights = np.append(weights[total_earning > threshold], np.sum(weights[total_earning <= threshold]))
    total_earning = np.append(total_earning[total_earning > threshold], np.mean(total_earning[total_earning <= threshold]))

    # equal weights
    welfare_weight = np.ones_like(primary_earning)
    x_equal_weights = np.average(welfare_weight*secondary_earning, weights = weights)
    y_equal_weights = np.average(welfare_weight*primary_earning, weights = weights)
    print("equal weights : ", x_equal_weights, y_equal_weights)
    axes[0].plot(x_equal_weights, y_equal_weights, marker='+', markersize=10, color='red', label = "equal weights")

    # decreasing
    total_earning_modified = np.copy(total_earning)
    total_earning_modified[total_earning == 0] = 1 
    welfare_weight = np.power(total_earning_modified, -0.8)
    x_decreasing = np.average(welfare_weight*secondary_earning, weights = weights)
    y_decreasing = np.average(welfare_weight*primary_earning, weights = weights)
    print("decreasing ", x_decreasing, y_decreasing)
    axes[1].plot(x_decreasing, y_decreasing, marker='+', markersize=10, color='purple', label = "decreasing")

    # Rawlsian
    P5 =  np.percentile(total_earning, 5) # not the real 5% since we already removed the bottom 5% 
    print("P5", P5)
    welfare_weight = 1*(total_earning <= P5)
    x_rawlsian = np.average(welfare_weight*secondary_earning, weights = weights)
    y_rawlsian = np.average(welfare_weight*primary_earning, weights = weights)
    print("rawlsian ", x_rawlsian, y_rawlsian)
    axes[1].plot(x_rawlsian, y_rawlsian, marker='+', markersize=10, color='orange', label = "rawlsian")

    # secondary earner
    total_earning_modified = np.copy(total_earning)
    secondary_earning_modified = np.copy(secondary_earning) 
    secondary_earning_modified[total_earning == 0] = 1
    total_earning_modified[total_earning == 0] = 1 # so that the share secondary/total equals 1 when both incomes are 0
    welfare_weight = secondary_earning_modified/total_earning_modified
    x_secondary = np.average(welfare_weight*secondary_earning, weights = weights)
    y_secondary = np.average(welfare_weight*primary_earning, weights = weights)
    print("secondary earner ", x_secondary, y_secondary)
    axes[0].plot(x_secondary, y_secondary, marker='+', markersize=10, color='blue', label = "secondary")

    # rawslian secondary earner
    welfare_weight = (total_earning <= P5)*secondary_earning_modified/total_earning_modified
    x_rawlsian_secondary = np.average(welfare_weight*secondary_earning, weights = weights)
    y_rawlsian_secondary = np.average(welfare_weight*primary_earning, weights = weights)
    print("rawlsian secondary ", x_rawlsian_secondary, y_rawlsian_secondary)
    axes[1].plot(x_rawlsian_secondary, y_rawlsian_secondary, marker='+', markersize=10, color='pink', label = "rawlsian secondary")




    bornes = [50000, 1000]
    bornes_inf = [500, 20]
    graphs_titles = ["Middle of distribution - {}".format(period), "Bottom of distribution - {}".format(period)]
    
    for j in range(2):
        x = np.linspace(0, bornes[j], 4)
        axes[j].plot(x, x, c = '#828282')

        green_shades = [(0.0, 1.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.6, 0.0)]
        for i in range(len(eps1_tab)):
            color = green_shades[i]
            axes[j].plot(x, slopes_lines[i]*x, label = "ep = {ep}, es = {es}".format(ep = eps1_tab[i], es = eps2_tab[i]), color=color)

        axes[j].scatter(secondary_earning, primary_earning, s = 0.1, c = '#828282') 

        axes[j].set_xlim(-bornes_inf[j], bornes[j]) 
        axes[j].set_ylim(-bornes_inf[j], bornes[j]) 
        axes[j].grid()
        axes[j].set_xlabel('Secondary earner')
        axes[j].set_ylabel('Primary earner')
        axes[j].legend()
        axes[j].set_title(graphs_titles[j])


    plt.show()
    plt.savefig('../outputs/welfare/welfare_{annee}.png'.format(annee = period))
    plt.close()
    



################################################################################
################# Main function ################################################
################################################################################
@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
@click.option('-n', '--want_to_consider_null_earnings', default = False, type = bool, required = True)
@click.option('-r', '--robustness', default = False, type = bool, required = True)
def launch_utils(annee = None, want_to_consider_null_earnings = None, robustness = None):

    if want_to_consider_null_earnings: # if we want to keep null total earnings 
        work_df = pd.read_csv(f'./excel/{annee}/married_25_55_{annee}.csv')

    elif robustness: # if you want to do a robustness check (only implemented for annee == 2006)
        assert (annee == 2006)
        work_df = pd.read_csv(f'./excel/{annee}/robustness_married_25_55_positive_{annee}.csv')
        
    else: # we consider (default) the version where we removed total earning that were equal to 0
        work_df = pd.read_csv(f'./excel/{annee}/married_25_55_positive_{annee}.csv')
    
    df_single_earner_couples = work_df[work_df['secondary_earning'] == 0]
    df_dual_earner_couples = work_df[work_df['secondary_earning'] > 0]
    print(work_df)
    print()

    primary_grid_earnings = np.linspace(np.percentile(work_df['primary_earning'].values, 1), np.percentile(work_df['primary_earning'].values, 99.9), 1000)
    secondary_grid_earnings = np.linspace(np.percentile(work_df['secondary_earning'].values, 1), np.percentile(work_df['secondary_earning'].values, 99.9), 1000)
    
    ################################################################################
    ################# Intensive part ###############################################
    ################################################################################

    # Primary : 2 preliminary steps
    # first computes E(T'/(1-T') | y1) the mean of marginal tax rates ratios, by primary earnings, for all distinct primary earnings
    # then get this ratio on a grid of primary earnings by taking the closest earning and the corresponding ratio, 
    # and then smooth this function using a gaussian kernel
    unique_primary_earning, primary_mean_tax_rates = computes_mtr_ratios_knowing_earning(
                                            earning = work_df['primary_earning'].values, 
                                            taux_marginal = work_df['taux_marginal'].values, 
                                            weights = work_df['weight_foyerfiscal'].values)
    
    primary_mtr_ratios_grid = util_intensive_revenue_function(
                                                        grid_earnings = primary_grid_earnings,
                                                        original_earnings = unique_primary_earning, 
                                                        average_ratios = primary_mean_tax_rates, 
                                                        name = "primary",
                                                        period = annee)
    
    # Secondary : 2 same preliminary steps
    unique_secondary_earning, secondary_mean_tax_rates = computes_mtr_ratios_knowing_earning(
                                            earning = work_df['secondary_earning'].values, 
                                            taux_marginal = work_df['taux_marginal'].values, 
                                            weights = work_df['weight_foyerfiscal'].values)
    
    secondary_mtr_ratios_grid = util_intensive_revenue_function(
                                                        grid_earnings = secondary_grid_earnings,
                                                        original_earnings = unique_secondary_earning, 
                                                        average_ratios = secondary_mean_tax_rates, 
                                                        name = "secondary",
                                                        period = annee)
    
    
    

    
    ################################################################################
    ################# Extensive part ###############################################
    ################################################################################

    # Primary : in 3 steps, first computes E(T/(ym-T) * pelast) by primary earnings for all foyers fiscaux 
    # then computes the inside of the integral on a grid of primary earnings, that is E(T/(ym-T) * pelast) * pdfsec/dec * sharesec/dec 
    # finally integrates this 2 integrands between y1 and y1_max and then minus the result
    unique_primary_earning, primary_mean_tax_rates = computes_tax_ratios_knowing_earning(
                    earning=work_df['primary_earning'].values,
                    total_earning=work_df['total_earning'].values,
                    tax=work_df['ancien_irpp'].values,
                    weights=work_df['weight_foyerfiscal'].values,
                    period = annee)
    
    primary_sec_within_integral, primary_dec_within_integral = util_extensive_revenue_function(
                    grid_earnings = primary_grid_earnings,
                    original_earnings = unique_primary_earning,
                    average_ratios = primary_mean_tax_rates,
                    sec_earnings = df_single_earner_couples['primary_earning'].values, 
                    dec_earnings = df_dual_earner_couples['primary_earning'].values, 
                    sec_weights = df_single_earner_couples['weight_foyerfiscal'].values,
                    dec_weights = df_dual_earner_couples['weight_foyerfiscal'].values, 
                    period = annee,
                    name = "primary")


    primary_extensive_revenue_function = compute_extensive_revenue_function(grid_earnings = primary_grid_earnings, within_integral = primary_sec_within_integral) + compute_extensive_revenue_function(grid_earnings = primary_grid_earnings, within_integral = primary_dec_within_integral)
    #print("primary extensive revenue function", primary_extensive_revenue_function)

    # Secondary : the same 3 steps as for primary
    unique_secondary_earning, secondary_mean_tax_rates = computes_tax_ratios_knowing_earning(
                    earning=work_df['secondary_earning'].values,
                    total_earning=work_df['total_earning'].values,
                    tax=work_df['ancien_irpp'].values,
                    weights=work_df['weight_foyerfiscal'].values,
                    period = annee)
    
    # for single earner couples (sec), the secondary earning is always 0
    secondary_sec_within_integral, secondary_dec_within_integral = util_extensive_revenue_function(
                    grid_earnings = secondary_grid_earnings,
                    original_earnings = unique_secondary_earning,
                    average_ratios = secondary_mean_tax_rates,
                    sec_earnings = df_single_earner_couples['secondary_earning'].values, 
                    dec_earnings = df_dual_earner_couples['secondary_earning'].values, 
                    sec_weights = df_single_earner_couples['weight_foyerfiscal'].values,
                    dec_weights = df_dual_earner_couples['weight_foyerfiscal'].values,
                    period = annee,
                    name = "secondary")


    secondary_extensive_revenue_function = compute_extensive_revenue_function(grid_earnings = secondary_grid_earnings, within_integral = secondary_dec_within_integral) + 0
    

    slopes_lines = winners_political_economy(primary_grid = primary_grid_earnings, 
             primary_earning = work_df['primary_earning'].values, 
             primary_mtr_ratios_grid = primary_mtr_ratios_grid, 
             extensive_primary_revenue_function = primary_extensive_revenue_function, 
             secondary_grid = secondary_grid_earnings, 
             secondary_earning = work_df['secondary_earning'].values,
             secondary_mtr_ratios_grid = secondary_mtr_ratios_grid, 
             extensive_secondary_revenue_function = secondary_extensive_revenue_function, 
             weights = work_df['weight_foyerfiscal'].values, 
             period = annee)
    
    ################################################################################
    ################# Welfare part #################################################
    ################################################################################

    median_share_primary(primary_earning = work_df['primary_earning'].values, 
                         total_earning = work_df['total_earning'].values, 
                         weights = work_df['weight_foyerfiscal'].values, 
                         period = annee)
    
    main_welfare_graph(primary_earning = work_df['primary_earning'].values,
                       secondary_earning = work_df['secondary_earning'].values,
                       total_earning = work_df['total_earning'].values,
                       weights = work_df['weight_foyerfiscal'].values,
                       slopes_lines = slopes_lines,
                       period = annee)
    




def redirect_print_to_file(filename):
    sys.stdout = open(filename, 'a')
    
redirect_print_to_file('output_graphe_15.txt')

launch_utils()

sys.stdout.close()
sys.stdout = sys.__stdout__