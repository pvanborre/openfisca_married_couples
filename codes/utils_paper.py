import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import click
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm

def mtr(earning_tab, earning, taux_marginal, period):




    kernel = sm.nonparametric.KDEMultivariate(earning, var_type='c', bw='normal_reference')
    print(kernel.bw)
    mtr_smoothed = lowess(taux_marginal, earning_tab)

    plt.plot(earning_tab[earning_tab<200000], mtr_smoothed[earning_tab<200000], label='MTR')   
    plt.xlabel('Gross income')
    plt.ylabel('MTR')
    plt.title("MTR - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/mtr_kde_{annee}.png'.format(annee = period))
    plt.close()







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


    # expérience ici 
    tab_bandwidth = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for bandwidth in tab_bandwidth:
        kernel_reg = KernelReg(endog=taux_marginal, exog=earning, var_type='c', reg_type='ll', bw=[bandwidth], ckertype='gaussian')
        smoothed_y_primary, _ = kernel_reg.fit()
        plt.plot(earning[earning<200000], smoothed_y_primary[earning<200000], label='MTR')   
        plt.xlabel('Gross income')
        plt.ylabel('MTR')
        plt.title("MTR - {annee}".format(annee = period))
        plt.legend()
        plt.show()
        plt.savefig('../outputs/test_cdf/mtr_{bandwidth}_{annee}.png'.format(bandwidth=bandwidth, annee = period))
        plt.close()





@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
@click.option('-m', '--want_to_mute_decote', default = False, type = bool, required = True)
def launch_utils(annee = None, want_to_mute_decote = None):
    work_df = pd.read_csv(f'./excel/{annee}/married_25_55_{annee}.csv')
    print(work_df)

    mtr(earning_tab = work_df['total_earning'].values, 
        earning = work_df[['total_earning']],
        taux_marginal = work_df['taux_marginal'].values,
        period = annee)

    # primary_cdf_pdf(earning = work_df['primary_earning'].values, 
    #                 taux_marginal = work_df['taux_marginal'].values,
    #                 weights = work_df['wprm'].values, 
    #                 elasticity = 0.25,
    #                 period = annee)

launch_utils()