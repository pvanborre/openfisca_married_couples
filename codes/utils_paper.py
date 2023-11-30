import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import click
from statsmodels.nonparametric.kernel_regression import KernelReg



def primary_cdf_pdf(earning, taux_marginal, weights, elasticity, period):
    """
    TODO
    """

    earning.sort()

    kde = gaussian_kde(earning, weights=weights)
    x_values = np.linspace(np.percentile(earning, 1), np.percentile(earning, 99.9), 1000)

    pdf = kde(x_values)
    cdf = np.cumsum(pdf) / np.sum(pdf)

    ipol_MTR_test_case_baseline = np.interp(x_values, earning, taux_marginal)
    

    behavioral = - x_values * pdf * elasticity * ipol_MTR_test_case_baseline/(1-ipol_MTR_test_case_baseline)  
    mechanical = 1 - cdf
    intensive_revenue_function = behavioral + mechanical

    plt.plot(x_values, cdf, label='CDF')   
    plt.xlabel('Gross income')
    plt.ylabel('CDF')
    plt.title("Cumulative distribution function - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/graphe_B13_{annee}.png'.format(annee = period))
    plt.close()

    plt.plot(x_values, pdf, label='PDF')   
    plt.xlabel('Gross income')
    plt.ylabel('PDF')
    plt.title("Probability distribution function - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/graphe_B14_{annee}.png'.format(annee = period))
    plt.close()

    plt.plot(x_values, ipol_MTR_test_case_baseline/(1-ipol_MTR_test_case_baseline), label='MTR ratio')   
    plt.xlabel('Gross income')
    plt.ylabel('MTR ratio')
    plt.title("MTR ratio - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/mtr_ratio_{annee}.png'.format(annee = period))
    plt.close()

    plt.plot(x_values, intensive_revenue_function, label='MTR ratio')   
    plt.xlabel('Gross income')
    plt.ylabel('MTR ratio')
    plt.title("MTR ratio - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/revenue_function_{annee}.png'.format(annee = period))
    plt.close()

    integral_trap_primary = np.trapz(intensive_revenue_function, x_values)
    print("Integral of smoothed_y primary", integral_trap_primary)



@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
@click.option('-m', '--want_to_mute_decote', default = False, type = bool, required = True)
def launch_utils(annee = None, want_to_mute_decote = None):
    work_df = pd.read_csv(f'./excel/{annee}/married_25_55_{annee}.csv')
    print(work_df)

    primary_cdf_pdf(earning = work_df['primary_earning'].values, 
                    taux_marginal = work_df['taux_marginal'].values,
                    weights = work_df['wprm'].values, 
                    elasticity = 0.25,
                    period = annee)

launch_utils()