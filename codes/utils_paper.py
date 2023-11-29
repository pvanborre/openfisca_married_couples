import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import click
from statsmodels.nonparametric.kernel_regression import KernelReg

# TODO commenter ce code 

def esperance_taux_marginal(earning, taux_marginal, weights, period, borne = 0.05):
    """
    TODO
    """
    output = np.zeros_like(earning, dtype=float)
    earning.sort()

    for i in range(len(earning)):
        diff = np.abs(earning - earning[i])
        ir_taux_marginal2 = np.copy(taux_marginal)
        ir_taux_marginal2[diff > borne] = 0
        
        weighted_taux_marginal = ir_taux_marginal2 / (1 - ir_taux_marginal2) * weights
        weighted_diff = (diff <= borne) * weights

        output[i] = np.sum(weighted_taux_marginal) / np.sum(weighted_diff)
    
    plt.scatter(earning, output, label='MTR ratio')   
    plt.xlabel('Gross income')
    plt.ylabel('MTR ratio')
    plt.title("MTR ratio - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/mtr_ratio2_{annee}.png'.format(annee = period))
    plt.close()

    kernel_reg = KernelReg(endog=earning, exog=output, var_type='c', reg_type='ll', bw=[bandwidth], ckertype='gaussian')
    smoothed_y_primary, _ = kernel_reg.fit()
    smoothed_y_primary = smoothed_y_primary[sorted_indices]

    return output

# new_rni_values = np.linspace(rni_values.min(), rni_values.max(), 1000)
# interpolated_cdf = np.interp(new_rni_values, rni_values, cdf1)


def primary_cdf_pdf(earning, esperance_taux_marginal, weights, elasticity, period):
    """
    TODO
    """

    earning.sort()

    kde = gaussian_kde(earning, weights=weights)
    x_values = np.linspace(np.percentile(earning, 1), np.percentile(earning, 99.9), 1000)

    pdf = kde(x_values)
    cdf = np.cumsum(pdf) / np.sum(pdf)

    interpolated_esperance = np.interp(x_values, earning, esperance_taux_marginal)

    behavioral = - x_values * pdf * elasticity * interpolated_esperance  
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

    plt.plot(x_values, interpolated_esperance, label='MTR ratio')   
    plt.xlabel('Gross income')
    plt.ylabel('MTR ratio')
    plt.title("MTR ratio - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/test_cdf/mtr_ratio_{annee}.png'.format(annee = period))
    plt.close()


@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
@click.option('-m', '--want_to_mute_decote', default = False, type = bool, required = True)
def launch_utils(annee = None, want_to_mute_decote = None):
    work_df = pd.read_csv(f'./excel/{annee}/married_25_55_{annee}.csv')
    print(work_df)

    esperance_rapport_mtr = esperance_taux_marginal(earning = work_df['primary_earning'].values,
                                                      taux_marginal = work_df['taux_marginal'].values, 
                                                      weights = work_df['wprm'].values,
                                                      period = annee)

    primary_cdf_pdf(earning = work_df['primary_earning'].values, 
                    esperance_taux_marginal = esperance_rapport_mtr,
                    weights = work_df['wprm'].values, 
                    elasticity = 0.25,
                    period = annee)

launch_utils()