import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import click
from statsmodels.nonparametric.kernel_regression import KernelReg
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import sys

################################################################################
################# Intensive revenue function ###################################
################################################################################

def computes_mtr_ratios_knowing_earning(earning, taux_marginal, weights):
    """
    This takes as input numpy arrays of size number foyers_fiscaux, primary or secondary earnings, and then the marginal tax rates and the weights
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



def find_closest_earning_and_tax_rate(grid_earnings, original_earnings, average_ratios, name, period):
    """
    This takes as input numpy arrays of size unique primary/secondary earnings computed by the above function, 
    that is distinct primary/secondary earnings and the ratios E(T'/(1-T'))
    The function creates a grid of earnings and for each earning of the grid, find the closest "real" earning and the corresponding MTR ratio
    Then it scatters these ratios.
    Since it is a stairs function with stairs overlapping, we decided to smooth the function using a kernel, and we plot the results
    """

    closest_indices = np.argmin(np.abs(original_earnings[:, None] - grid_earnings), axis=0)
    closest_mtr_ratios = average_ratios[closest_indices]

    bandwidth = 5000
    kernel_reg = KernelReg(endog=closest_mtr_ratios, exog=grid_earnings, var_type='c', reg_type='ll', bw=[bandwidth], ckertype='gaussian')
    smoothed_y_primary, _ = kernel_reg.fit()
    plt.scatter(grid_earnings, closest_mtr_ratios, label='MTR ratio without smoothing', color = 'lightgreen')
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

def computes_tax_ratios_knowing_earning(earning, total_earning, tax, weights):
    """
    This takes as input numpy arrays of size number foyers_fiscaux, primary or secondary earnings, and then total earnings, tax before reform and weights
    This computes E(T/(ym-T) * pelast | y1/2) for all distinct values of primary earings (or secondary earnings)
    So it returns this numpy array of size unique primary/secondary earnings
    """

    # tax is negative (that is why the -tax)
    denominator = total_earning+tax
    denominator[denominator == 0] = 0.001 # is this the right thing to do ? look at what happens when deno = 0
    tax_ratio = (-tax)/denominator * (0.65 - 0.4 * np.sqrt(total_earning/np.percentile(total_earning, 90)))

    unique_earning = np.unique(earning)
    mean_tax_rates = np.zeros_like(unique_earning, dtype=float)

    for i, unique_value in enumerate(unique_earning):
        indices = np.where(earning == unique_value)
        mean_tax_rate = np.average(tax_ratio[indices], weights=weights[indices])
        mean_tax_rates[i] = mean_tax_rate

    return unique_earning, mean_tax_rates






def util_extensive_revenue_function(grid_earnings, original_earnings, average_ratios, sec_earnings, dec_earnings, sec_weights, dec_weights, name):
    """
    This takes as input numpy arrays of size number unique foyers_fiscaux, primary or secondary unique earnings, and average ratios E(T/(ym-T) * pelast)
    Then other inputs are specific to single and dual earner couples (sec and dec)
    This computes the integrand of the extensive revenue function on a specific grid of earnings, that is E(T/(ym-T) * pelast) * pdfsec/dec * sharesec/dec 
    """

    # defines the grid and gets E(T/(ym-T) * pelast) on this grid (take closest value from the original earnings)
    closest_indices = np.argmin(np.abs(original_earnings[:, None] - grid_earnings), axis=0)
    closest_tax_ratios = average_ratios[closest_indices]

    # then computes the pdf and the integrand, for both single and dual earner couples
    sec_share = np.sum(sec_weights)/(np.sum(sec_weights) + np.sum(dec_weights))

    sorted_indices_dec = np.argsort(dec_earnings)
    dec_earnings_sorted = dec_earnings[sorted_indices_dec]
    dec_weights_sorted = dec_weights[sorted_indices_dec]

    dec_kde = gaussian_kde(dec_earnings_sorted, weights=dec_weights_sorted)
    dec_pdf = dec_kde(grid_earnings)
    dec_within_integral = closest_tax_ratios * dec_pdf * (1-sec_share)

    if name == "primary":
        sorted_indices_sec = np.argsort(sec_earnings)
        sec_earnings_sorted = sec_earnings[sorted_indices_sec]
        sec_weights_sorted = sec_weights[sorted_indices_sec]

        sec_kde = gaussian_kde(sec_earnings_sorted, weights=sec_weights_sorted)
        sec_pdf = sec_kde(grid_earnings) 
        sec_within_integral = closest_tax_ratios * sec_pdf * sec_share
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
        primary_elasticity_maries_pacses =  eps1_tab[i]
        secondary_elasticity_maries_pacses = eps2_tab[i]
        
        cdf_primary, pdf_primary, intensive_primary_revenue_function = compute_intensive_revenue_function(
                    grid_earnings = primary_grid,
                    earning = primary_earning, 
                    mtr_ratios_grid = primary_mtr_ratios_grid,
                    weights = weights, 
                    elasticity = primary_elasticity_maries_pacses)
         
        primary_revenue_function = intensive_primary_revenue_function + extensive_primary_revenue_function

        cdf_secondary, pdf_secondary, intensive_secondary_revenue_function = compute_intensive_revenue_function(
                    grid_earnings = secondary_grid,
                    earning = secondary_earning, 
                    mtr_ratios_grid = secondary_mtr_ratios_grid,
                    weights = weights, 
                    elasticity = secondary_elasticity_maries_pacses)
         
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
        plt.annotate(str(round(pourcentage_gagnants[i]))+ " %", xy = (50000 + 100000*i, 200000), bbox = dict(boxstyle ="round", fc = color))

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

    




################################################################################
################# Main function ################################################
################################################################################
@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
def launch_utils(annee = None):
    work_df = pd.read_csv(f'./excel/{annee}/married_25_55_{annee}.csv')
    
    print(work_df)
    print()
        
    df_dual_earner_couples = pd.read_csv(f'./excel/{annee}/dual_earner_couples_25_55_{annee}.csv')
    df_single_earner_couples = pd.read_csv(f'./excel/{annee}/single_earner_couples_25_55_{annee}.csv')


    # get rid of the bottom 5% of the distribution (in terms of total earnings)
    # TODO not good see what Pierre really meant by this, or only for welfare ?
    # threshold = np.percentile( work_df['total_earning'].values, 5)
    # print("threshold income not taken into account", threshold)
    # work_df = work_df[work_df['total_earning'] > threshold]

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
                                            weights = work_df['wprm'].values)
    
    primary_mtr_ratios_grid = find_closest_earning_and_tax_rate(
                                                        grid_earnings = primary_grid_earnings,
                                                        original_earnings = unique_primary_earning, 
                                                        average_ratios = primary_mean_tax_rates, 
                                                        name = "primary",
                                                        period = annee)
    
    # Secondary : 2 same preliminary steps
    unique_secondary_earning, secondary_mean_tax_rates = computes_mtr_ratios_knowing_earning(
                                            earning = work_df['secondary_earning'].values, 
                                            taux_marginal = work_df['taux_marginal'].values, 
                                            weights = work_df['wprm'].values)
    
    secondary_mtr_ratios_grid = find_closest_earning_and_tax_rate(
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
                    weights=work_df['wprm'].values)
    
    primary_sec_within_integral, primary_dec_within_integral = util_extensive_revenue_function(
                    grid_earnings = primary_grid_earnings,
                    original_earnings = unique_primary_earning,
                    average_ratios = primary_mean_tax_rates,
                    sec_earnings = df_single_earner_couples['primary_earning'].values, 
                    dec_earnings = df_dual_earner_couples['primary_earning'].values, 
                    sec_weights = df_single_earner_couples['wprm'].values,
                    dec_weights = df_dual_earner_couples['wprm'].values, 
                    name = "primary")


    primary_extensive_revenue_function = compute_extensive_revenue_function(grid_earnings = primary_grid_earnings, within_integral = primary_sec_within_integral) + compute_extensive_revenue_function(grid_earnings = primary_grid_earnings, within_integral = primary_dec_within_integral)
    
    # Secondary : the same 3 steps as for primary
    unique_secondary_earning, secondary_mean_tax_rates = computes_tax_ratios_knowing_earning(
                    earning=work_df['secondary_earning'].values,
                    total_earning=work_df['total_earning'].values,
                    tax=work_df['ancien_irpp'].values,
                    weights=work_df['wprm'].values)
    
    # for single earner couples (sec), the secondary earning is always 0
    secondary_sec_within_integral, secondary_dec_within_integral = util_extensive_revenue_function(
                    grid_earnings = secondary_grid_earnings,
                    original_earnings = unique_secondary_earning,
                    average_ratios = secondary_mean_tax_rates,
                    sec_earnings = df_single_earner_couples['secondary_earning'].values, 
                    dec_earnings = df_dual_earner_couples['secondary_earning'].values, 
                    sec_weights = df_single_earner_couples['wprm'].values,
                    dec_weights = df_dual_earner_couples['wprm'].values,
                    name = "secondary")


    secondary_extensive_revenue_function = compute_extensive_revenue_function(grid_earnings = secondary_grid_earnings, within_integral = secondary_dec_within_integral) + 0
    

    winners_political_economy(primary_grid = primary_grid_earnings, 
             primary_earning = work_df['primary_earning'].values, 
             primary_mtr_ratios_grid = primary_mtr_ratios_grid, 
             extensive_primary_revenue_function = primary_extensive_revenue_function, 
             secondary_grid = secondary_grid_earnings, 
             secondary_earning = work_df['secondary_earning'].values,
             secondary_mtr_ratios_grid = secondary_mtr_ratios_grid, 
             extensive_secondary_revenue_function = secondary_extensive_revenue_function, 
             weights = work_df['wprm'].values, 
             period = annee)




def redirect_print_to_file(filename):
    sys.stdout = open(filename, 'a')
    
redirect_print_to_file('output_graphe_15.txt')

launch_utils()

sys.stdout.close()
sys.stdout = sys.__stdout__