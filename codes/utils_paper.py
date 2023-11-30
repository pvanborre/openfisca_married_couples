import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import click


def mtr(total_earning, taux_marginal, period):
    """
    takes as input marginal tax rates and the total earnings of the foyers_fiscal
    computes mtr for the grid x_values of total_earnings we are considering
    """
    sorted_indices = np.argsort(total_earning)
    total_earning = total_earning[sorted_indices]
    taux_marginal = taux_marginal[sorted_indices]

    x_values = np.linspace(np.percentile(total_earning, 1), np.percentile(total_earning, 99.9), 100)
    ipol_MTR = np.interp(x_values, total_earning, taux_marginal)

    plt.plot(x_values, ipol_MTR/(1-ipol_MTR), label='MTR ratio')   
    plt.xlabel('Gross income')
    plt.ylabel('MTR ratio')
    plt.title("MTR ratio - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/mtr_ratio_{annee}.png'.format(annee = period))
    plt.close()

    return ipol_MTR

def earning_adapted_mtr(earning, total_earning, ipol_MTR, period):
    """
    we need to convert the marginal tax rates that we got in the mtr function from mtr on a grid of total_earnings to a grid of primary/secondary earnings
    for this we start from the grid of primary earnings, we look for the primary earning value associated to it, therefore we can retreive the total earning value, 
    and then the total earning sampled value that is the closest and finally the mtr associated
    """
    earning_sampled = np.linspace(np.percentile(earning, 1), np.percentile(earning, 99.9), 100)
    total_earning_sampled = np.linspace(np.percentile(total_earning, 1), np.percentile(total_earning, 99.9), 100)

    output = []
    for i in range(len(earning_sampled)):
        j = np.argmin(np.abs(earning - earning_sampled[i]))
        k = np.argmin(np.abs(total_earning_sampled - total_earning[j]))
        output.append(ipol_MTR[k])

    output = np.array(output)

    plt.plot(earning_sampled, output/(1-output), label='MTR ratio')   
    plt.xlabel('Gross income')
    plt.ylabel('MTR ratio')
    plt.title("MTR ratio - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/mtr_ratio_by_earning_{annee}.png'.format(annee = period))
    plt.close()


    return output







def compute_intensive_revenue_function(earning, ipol_MTR_earning, weights, elasticity, period):
    """
    Computes the pdf and cdf of earnings, and the intensive revenue function
    """

    earning.sort()

    kde = gaussian_kde(earning, weights=weights)
    x_values = np.linspace(np.percentile(earning, 1), np.percentile(earning, 99.9), 100)

    pdf = kde(x_values)
    cdf = np.cumsum(pdf) / np.sum(pdf)

    behavioral = - x_values * pdf * elasticity * ipol_MTR_earning/(1-ipol_MTR_earning)  
    mechanical = 1 - cdf
    intensive_revenue_function = behavioral + mechanical

    return x_values, cdf, pdf, intensive_revenue_function

def plot_intensive_revenue_function(primary_grid, cdf_primary, pdf_primary, intensive_primary_revenue_function, secondary_grid, cdf_secondary, pdf_secondary, intensive_secondary_revenue_function, period):
    

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






@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
@click.option('-m', '--want_to_mute_decote', default = False, type = bool, required = True)
def launch_utils(annee = None, want_to_mute_decote = None):
    work_df = pd.read_csv(f'./excel/{annee}/married_25_55_{annee}.csv')
    print(work_df)

    ipol_MTR = mtr(total_earning = work_df['total_earning'].values,
        taux_marginal = work_df['taux_marginal'].values,
        period = annee)
    
    ipol_MTR_primary = earning_adapted_mtr(earning=work_df['primary_earning'].values,
                        total_earning=work_df['total_earning'].values,
                        ipol_MTR=ipol_MTR,
                        period=annee)
    
    ipol_MTR_secondary = earning_adapted_mtr(earning=work_df['secondary_earning'].values,
                        total_earning=work_df['total_earning'].values,
                        ipol_MTR=ipol_MTR,
                        period=annee)

    primary_grid, cdf_primary, pdf_primary, intensive_primary_revenue_function = compute_intensive_revenue_function(earning = work_df['primary_earning'].values, 
                    ipol_MTR_earning = ipol_MTR_primary,
                    weights = work_df['wprm'].values, 
                    elasticity = 0.25,
                    period = annee)
    
    secondary_grid, cdf_secondary, pdf_secondary, intensive_secondary_revenue_function = compute_intensive_revenue_function(earning = work_df['secondary_earning'].values, 
                    ipol_MTR_earning = ipol_MTR_secondary,
                    weights = work_df['wprm'].values, 
                    elasticity = 0.25,
                    period = annee)
    
    plot_intensive_revenue_function(primary_grid, cdf_primary, pdf_primary, intensive_primary_revenue_function, secondary_grid, cdf_secondary, pdf_secondary, intensive_secondary_revenue_function, annee)
    

launch_utils()