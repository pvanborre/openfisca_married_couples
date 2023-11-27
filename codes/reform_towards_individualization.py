import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import convolve
from statsmodels.nonparametric.kernel_regression import KernelReg

from sklearn.linear_model import Lasso, ElasticNetCV
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tabulate import tabulate


import sys
import click

from openfisca_core.simulation_builder import SimulationBuilder
from openfisca_core.reforms import Reform
from openfisca_core import periods

from openfisca_france import FranceTaxBenefitSystem
from openfisca_france.model.base import *


pandas.options.display.max_columns = None





# tax function Tm1(y1, y2) = Tm0(ym) + τm hm(y1, y2) where h is a reform direction
# reform direction hm(y1, y2) = τ1 y1 + τ2y2  
# on suppose ici pour simplifier que τm = 1 donc nouvel_IRPP = ancien_IRPP + τ1 revenu_principal + τ2 revenu_secondaire

# l'idée est de reprendre la formule de l'impot et d'ajouter un terme tau_1 * y1 + tau_2 * y2 pour les couples mariés ou pacsés (on ne change rien pour les célibataires ou veufs)
# pour cela on cherche d'abord à créer une variable de revenu y pour chaque individu, où on somme ses traitements, salaires, pensions, rentes et ajoute également les autres revenus en faisant un equal splitting au sein du ménage
# ensuite on crée y1 et y2 en comparant le revenu du déclarant principal du foyer fiscal et celui de son conjoint, le max étant y1 (primary_earning), le min étant y2 (secondary earning)
# remarque : ces variables y1 et y2 n'ont de sens que si le couple est marié ou pacsés, dans le cas contraire je choisis que y1 = y2 = 0
# on définit tau_2 = - tau_1 * sum(y1)/sum(y2) où la somme porte sur tous les couples mariés, afin de s'assurer un budget constant pour l'Etat 


# tout d'abord on update la valeur de quelques paramètres (étaient null et des formules demandaient leurs valeurs, qu'on met donc à 0)
def modify_parameters(parameters):
    reform_period = periods.period("2003")
    parameters.impot_revenu.calcul_reductions_impots.divers.intemp.max.update(period = reform_period, value = 0)
    parameters.impot_revenu.calcul_reductions_impots.divers.intemp.pac.update(period = reform_period, value = 0)
    parameters.impot_revenu.calcul_reductions_impots.divers.interets_emprunt_reprise_societe.plafond.update(period = reform_period, value = 0)
    parameters.impot_revenu.calcul_reductions_impots.divers.interets_emprunt_reprise_societe.taux.update(period = reform_period, value = 0)
    return parameters



class vers_individualisation(Reform):
    name = "on code la réforme nouvel impot = ancien impot + tau_1 revenu_1 + tau_2 revenu_2"
    def apply(self):

        # on applique la modification des paramètres pour l'année 2003
        self.modify_parameters(modifier_function = modify_parameters)

        class revenu_individu(Variable):
            # I could have used revenu_categoriel (same modulo a deduction)
            value_type = float
            entity = Individu
            label = "Revenu d'un individu du foyer fiscal"
            definition_period = YEAR

            def formula(individu, period):
                traitements_salaires_pensions_rentes = individu('traitements_salaires_pensions_rentes', period)

                rev_cat_rvcm = individu.foyer_fiscal('revenu_categoriel_capital', period)
                rev_cat_rfon = individu.foyer_fiscal('revenu_categoriel_foncier', period)
                rev_cat_rpns = individu.foyer_fiscal('revenu_categoriel_non_salarial', period)
                rev_cat_pv = individu.foyer_fiscal('revenu_categoriel_plus_values', period)  
                
                # on fait un equal splitting des revenus (problème : si enfants ont des revenus les somme aussi, on suppose que cela n'arrive pas)     
                return traitements_salaires_pensions_rentes + rev_cat_rvcm/2 + rev_cat_rfon/2 + rev_cat_rpns/2 + rev_cat_pv/2
        
        self.add_variable(revenu_individu)


        class revenu_celibataire(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "Revenu d'un célibataire"
            definition_period = YEAR

 
            def formula(foyer_fiscal, period):
                revenu_individu_i = foyer_fiscal.members('revenu_individu', period) # est de taille nb individus
                revenu_declarant_principal = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.DECLARANT_PRINCIPAL) # est de taille nb foyers fiscaux
                revenu_du_conjoint = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.CONJOINT) # est de taille nb foyers fiscaux 

                celibataire_ou_divorce = foyer_fiscal('celibataire_ou_divorce', period)
                veuf = foyer_fiscal('veuf', period)

                return max_(revenu_declarant_principal, revenu_du_conjoint) * (celibataire_ou_divorce | veuf)

        self.add_variable(revenu_celibataire)


        class primary_earning(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "Revenu le plus élevé du foyer fiscal, entre le déclarant principal et son conjoint"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                revenu_individu_i = foyer_fiscal.members('revenu_individu', period) # est de taille nb individus
                revenu_declarant_principal = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.DECLARANT_PRINCIPAL) # est de taille nb foyers fiscaux
                revenu_du_conjoint = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.CONJOINT) # est de taille nb foyers fiscaux 

                maries_ou_pacses = foyer_fiscal('maries_ou_pacses', period)

                return max_(revenu_declarant_principal, revenu_du_conjoint) * maries_ou_pacses

        self.add_variable(primary_earning)

        class secondary_earning(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "Revenu le moins élevé du foyer fiscal, entre le déclarant principal et son conjoint"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                revenu_individu_i = foyer_fiscal.members('revenu_individu', period) # est de taille nb individus
                revenu_declarant_principal = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.DECLARANT_PRINCIPAL) # est de taille nb foyers fiscaux
                revenu_du_conjoint = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.CONJOINT) # est de taille nb foyers fiscaux 

                maries_ou_pacses = foyer_fiscal('maries_ou_pacses', period)

                return min_(revenu_declarant_principal, revenu_du_conjoint) * maries_ou_pacses

        self.add_variable(secondary_earning)

        class primary_age(Variable):
            value_type = int
            entity = FoyerFiscal
            label = "Primary earner age"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                revenu_individu_i = foyer_fiscal.members('revenu_individu', period) # est de taille nb individus
                revenu_declarant_principal = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.DECLARANT_PRINCIPAL) # est de taille nb foyers fiscaux
                revenu_du_conjoint = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.CONJOINT) # est de taille nb foyers fiscaux 

                age_i = foyer_fiscal.members('age', period.last_month) # est de taille nb individus
                age_declarant_principal = foyer_fiscal.sum(age_i, role = FoyerFiscal.DECLARANT_PRINCIPAL) # est de taille nb foyers fiscaux
                age_du_conjoint = foyer_fiscal.sum(age_i, role = FoyerFiscal.CONJOINT) # est de taille nb foyers fiscaux 

                mask = revenu_declarant_principal < revenu_du_conjoint
                age_declarant_principal[mask] = age_du_conjoint[mask]

                maries_ou_pacses = foyer_fiscal('maries_ou_pacses', period)

                return age_declarant_principal * maries_ou_pacses

        self.add_variable(primary_age)

        class secondary_age(Variable):
            value_type = int
            entity = FoyerFiscal
            label = "Secondary earner age"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                revenu_individu_i = foyer_fiscal.members('revenu_individu', period) # est de taille nb individus
                revenu_declarant_principal = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.DECLARANT_PRINCIPAL) # est de taille nb foyers fiscaux
                revenu_du_conjoint = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.CONJOINT) # est de taille nb foyers fiscaux 

                age_i = foyer_fiscal.members('age', period.last_month) # est de taille nb individus
                age_declarant_principal = foyer_fiscal.sum(age_i, role = FoyerFiscal.DECLARANT_PRINCIPAL) # est de taille nb foyers fiscaux
                age_du_conjoint = foyer_fiscal.sum(age_i, role = FoyerFiscal.CONJOINT) # est de taille nb foyers fiscaux 

                mask = revenu_declarant_principal < revenu_du_conjoint
                age_du_conjoint[mask] = age_declarant_principal[mask]

                maries_ou_pacses = foyer_fiscal('maries_ou_pacses', period)

                return age_du_conjoint * maries_ou_pacses

        self.add_variable(secondary_age)

        

class mute_decote(Reform):
    name = "Mute the decote mechanism for couples"
    def apply(self):
        class decote(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "Decote set to 0 for couples"
            definition_period = YEAR

            def formula_2001_01_01(foyer_fiscal, period, parameters):
                ir_plaf_qf = foyer_fiscal('ir_plaf_qf', period)
                decote = parameters(period).impot_revenu.calcul_impot_revenu.plaf_qf.decote

                return numpy.around(max_(0, decote.seuil - ir_plaf_qf) * decote.taux)

            def formula_2014_01_01(foyer_fiscal, period, parameters):
                ir_plaf_qf = foyer_fiscal('ir_plaf_qf', period)
                nb_adult = foyer_fiscal('nb_adult', period)
                taux_decote = parameters(period).impot_revenu.calcul_impot_revenu.plaf_qf.decote.taux
                decote_seuil_celib = parameters(period).impot_revenu.calcul_impot_revenu.plaf_qf.decote.seuil_celib
                decote_seuil_couple = parameters(period).impot_revenu.calcul_impot_revenu.plaf_qf.decote.seuil_couple
                decote_celib = max_(0, decote_seuil_celib - taux_decote * ir_plaf_qf)
                decote_couple = max_(0, decote_seuil_couple - taux_decote * ir_plaf_qf)

                return numpy.around((nb_adult == 1) * decote_celib ) # we mute decote_couple here
            
        self.update_variable(decote)
            
        
 

        

def initialiser_simulation(tax_benefit_system, data_persons):

    sb = SimulationBuilder()
    sb.create_entities(tax_benefit_system)

    # numéro des entités : variables idmen (menage), idfoy (foyer fiscal), idfam (famille)
    # rôles au sein de ces entités : quimen, quifoy et quifam 

    # déclarer les individus
    sb.declare_person_entity('individu', data_persons.noindiv)

    # déclarer les menages
    construire_entite(data_persons, sb, nom_entite = "menage", nom_entite_pluriel = "menages", id_entite = "idmen", id_entite_join = "idmen_original",
                   role_entite = "quimen", nom_role_0 = "personne_de_reference", nom_role_1 = "conjoint", nom_role_2 = "enfant")
    
    # déclarer les foyers fiscaux
    construire_entite(data_persons, sb, nom_entite = "foyer_fiscal", nom_entite_pluriel = "foyers fiscaux", id_entite = "idfoy", id_entite_join = "idfoy",
                   role_entite = "quifoy", nom_role_0 = "declarant_principal", nom_role_1 = "conjoint", nom_role_2 = "personne_a_charge")
    
    # déclarer les familles
    construire_entite(data_persons, sb, nom_entite = "famille", nom_entite_pluriel = "familles", id_entite = "idfam", id_entite_join = "idfam",
                   role_entite = "quifam", nom_role_0 = "demandeur", nom_role_1 = "conjoint", nom_role_2 = "enfant")

    simulation = sb.build(tax_benefit_system)
    return simulation




def construire_entite(data_persons, sb, nom_entite, nom_entite_pluriel, id_entite, id_entite_join, role_entite, nom_role_0, nom_role_1, nom_role_2):

    # il faut bien mettre le bon nombre d'entites avec le .unique()
    # sinon Openfisca croit qu'il y a autant d'entites que d'individus  
    instance = sb.declare_entity(nom_entite, data_persons[id_entite].unique())

    print("nombre de " + nom_entite_pluriel, instance.count)
    print("rôles acceptés par OpenFisca pour les " + nom_entite_pluriel, instance.entity.flattened_roles)


    # join_with_persons accepte comme argument roles un tableau de str, on fait donc les recodages nécéssaires
    data_persons[role_entite] = numpy.select(
        [data_persons[role_entite] == 0, data_persons[role_entite] == 1, data_persons[role_entite] == 2],
        [nom_role_0, nom_role_1, nom_role_2],
        default="anomalie"  
    )

    # On associe chaque personne individuelle à son entité:
    sb.join_with_persons(instance, data_persons[id_entite_join], data_persons[role_entite])

    # on vérifie que les rôles sont bien conformes aux rôles attendus 
    print("rôles de chacun dans son " + nom_entite, instance.members_role)
    assert("anomalie" not in instance.members_role)

    print("\n\n\n\n\n")



def cdf_earnings(earning_condition_maries_pacses):
    """
    This function takes as input a numpy array of earnings (restricted to married couples having positive earnings and age between 25 and 55)
    Then it computes the cumulative distribution of earnings that will have the same size as the input
    """
    counts = numpy.array([numpy.sum(earning_condition_maries_pacses <= y2) for y2 in earning_condition_maries_pacses])

    cdf = numpy.zeros_like(earning_condition_maries_pacses, dtype=float)
    cdf = counts/len(earning_condition_maries_pacses)
    print("check de la cdf", cdf)

    return cdf



def gaussian_kernel(x):
    return 1/numpy.sqrt(2*numpy.pi) * numpy.exp(-1/2 * x * x)

def density_earnings(earning_condition_maries_pacses, period):
    """
    This function takes as input a numpy array of earnings (restricted to married couples having positive earnings and age between 25 and 55) and the period (year) of study
    Then it computes the density of earnings that will have the same size as the input
    """
    
    # Calculate the bandwidth using Silverman's rule (see the paper https://arxiv.org/pdf/1212.2812.pdf top of page 12)
    n = len(earning_condition_maries_pacses)
    estimated_std = numpy.std(earning_condition_maries_pacses, ddof=1)  
    bandwidth = 1.06 * estimated_std * n ** (-1/5)
    print("bandwidth", bandwidth)

    density = numpy.zeros_like(earning_condition_maries_pacses, dtype=float)

    # remarque : il ne faut pas que les foyers fiscaux non mariés ou pacsés portent de densité, on les retire donc puis on les remet
    

    if period not in ['2010', '2011', '2012']:
        # ce code vectorisé marche bien sauf pour ces 3 années où j'ai une memory error
        kernel_values = gaussian_kernel((earning_condition_maries_pacses[:, numpy.newaxis] - earning_condition_maries_pacses) / bandwidth)
        density = (1 / bandwidth) * numpy.mean(kernel_values, axis=1)

    else:
        # pour les 3 années concernées, on revient à un code avec une boucle for 
        for i in range(len(earning_condition_maries_pacses)):
            kernel_values = gaussian_kernel((earning_condition_maries_pacses - earning_condition_maries_pacses[i]) / bandwidth)
            density[i] = numpy.mean(kernel_values) * 1/bandwidth

    
    density /= numpy.sum(density) # attention ne valait pas forcément 1 avant (classique avec les kernels) 
    print("check de la densite", density)

    return density




def esperance_taux_marginal(earning_condition_maries_pacses, taux_marginal_condition_maries_pacses, borne = 0.05):
    """
    This function takes as input a numpy array of earnings and of marginal tax rates (restricted to married couples having positive earnings and age between 25 and 55)
    The borne is a threshold on which we average (we do not consider the mean only on y=y0 but on y in [y0-borne, y0 + borne])
    Then it computes the mean of marginal tax rates for all these intervals [y0-borne, y0 + borne]
    The output is of the same size as the inputs
    """

    output = numpy.zeros_like(earning_condition_maries_pacses, dtype=float)

    for i in range(len(earning_condition_maries_pacses)):
        diff = numpy.abs(earning_condition_maries_pacses - earning_condition_maries_pacses[i])
        ir_taux_marginal2 = numpy.copy(taux_marginal_condition_maries_pacses)
        ir_taux_marginal2[diff > borne] = 0
        output[i] = numpy.sum(ir_taux_marginal2 / (1 - ir_taux_marginal2))/numpy.sum(diff <= borne)

    return output

def moyenne_taux_marginal(primary_earning_condition_maries_pacses, secondary_earning_condition_maries_pacses, taux_marginal_condition_maries_pacses, period, K = 3):
    """
    This function takes as input a numpy array of primary and secondary earnings, and of marginal tax rates (restricted to married couples having positive earnings and age between 25 and 55)
    We perform a convolution in a window of size 2K that is an average of neighbors marginal taxes because we convolve with the vector [1/(2K+1), ..., 1/(2K+1)]
    The output is of the same size as the inputs : we output the total earnings of foyers fiscaux and these average marginal tax rates
    """
    revenu = primary_earning_condition_maries_pacses + secondary_earning_condition_maries_pacses

    sorted_indices = numpy.argsort(revenu)
    revenu = revenu[sorted_indices]
    taux_marginal_condition_maries_pacses = taux_marginal_condition_maries_pacses[sorted_indices]


    ir_taux_marginal_mean = numpy.convolve(taux_marginal_condition_maries_pacses, numpy.ones(2 * K + 1) / (2 * K + 1), mode='same')
    
    # here we fill in the last K values (that we couldn't get thanks to the convolution (slicing window of size K to the right))
    max_value = numpy.max(taux_marginal_condition_maries_pacses[-K:])
    ir_taux_marginal_mean[-K:] = max_value

    plt.plot(revenu, ir_taux_marginal_mean, label='Continuous Data')
    plt.xlabel('Gross earnings')
    plt.ylabel('Average marginal tax rate')
    plt.title("Average marginal tax rates - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/tax_one/graphe_tax_one_{annee}.png'.format(annee = period))
    plt.close()

    return revenu, ir_taux_marginal_mean

    


def couples_elasticity(primary_earning, secondary_earning, maries_ou_pacses, eps1, eps2):
    # lemma 2 formula : we average the elasticities with the shares
    total_earning = primary_earning + secondary_earning
    total_earning[total_earning == 0] = 0.001
    return maries_ou_pacses * (eps1*primary_earning + eps2*secondary_earning) / total_earning


def extensive_partial_revenue_function(base_earning_condition_maries_pacses, other_earning_condition_maries_pacses, secondary_earning_condition_maries_pacses, irpp_condition_maries_pacses):
    """
    This function takes as input numpy array of earnings, and of tax rates (restricted to married couples having positive earnings and age between 25 and 55)
    Then it computes in a cumulative way some integrals that we store in a dictionary partial_integral_values
    partial_integral_values[y] is the integral from 0 to y of ...
    """


    total_earning_condition_maries_pacses = base_earning_condition_maries_pacses + other_earning_condition_maries_pacses

    dual_earner_couples_base_earnings = base_earning_condition_maries_pacses[secondary_earning_condition_maries_pacses > 0]
    dual_earner_couples_total_earnings = total_earning_condition_maries_pacses[secondary_earning_condition_maries_pacses > 0]
    # we take the 9th decile to account for the formula of extensive elasticity 
    dual_earner_couples_total_earnings_sorted = numpy.sort(dual_earner_couples_total_earnings)
    index_9th_decile = int(0.9 * (len(dual_earner_couples_total_earnings_sorted) - 1))
    y90_dual_earner_couple =  dual_earner_couples_total_earnings_sorted[index_9th_decile]

    single_earner_couples_base_earnings = base_earning_condition_maries_pacses[secondary_earning_condition_maries_pacses == 0]
    single_earner_couples_total_earnings = total_earning_condition_maries_pacses[secondary_earning_condition_maries_pacses == 0]
    single_earner_couples_total_earnings_sorted = numpy.sort(single_earner_couples_total_earnings)
    index_9th_decile = int(0.9 * (len(single_earner_couples_total_earnings_sorted) - 1))
    y90_single_earner_couple =  single_earner_couples_total_earnings_sorted[index_9th_decile]

    total_sum = 0
    partial_integral_values = {}
    # trick for the computation : cumulative array fashion 
    # (avoids computing the integral from y1 to y_bar and then from y1' to y_bar since both integrals have a common part)
    # only integrals from 0 to y1_prime stored in this dictionary

    distinct_base_earnings = numpy.sort(numpy.unique(base_earning_condition_maries_pacses))

    for i in range(len(distinct_base_earnings)-1):
        y1_prime = distinct_base_earnings[i]
        next_y1_prime = distinct_base_earnings[i+1]

        condition_sample = (base_earning_condition_maries_pacses >= y1_prime) & (base_earning_condition_maries_pacses < next_y1_prime)
        irpp_sample = irpp_condition_maries_pacses[condition_sample]
        total_earning_sample = total_earning_condition_maries_pacses[condition_sample]
        denominator = total_earning_sample - irpp_sample
        denominator[denominator == 0] = 0.001
        moyenne_dual_earner = numpy.mean(irpp_sample/denominator * (0.65 - 0.4 * numpy.sqrt(total_earning_sample/y90_dual_earner_couple))) 
        moyenne_single_earner = numpy.mean(irpp_sample/denominator * (0.65 - 0.4 * numpy.sqrt(total_earning_sample/y90_single_earner_couple))) 

        condition_dual_sample = (dual_earner_couples_base_earnings >= y1_prime) & (dual_earner_couples_base_earnings < next_y1_prime)
        mass_dual = len(dual_earner_couples_base_earnings[condition_dual_sample])/len(dual_earner_couples_base_earnings)
        condition_single_sample = (single_earner_couples_base_earnings >= y1_prime) & (single_earner_couples_base_earnings < next_y1_prime)
        mass_single = len(single_earner_couples_base_earnings[condition_single_sample])/len(single_earner_couples_base_earnings)

        total_sum += (moyenne_dual_earner * mass_dual + moyenne_single_earner * mass_single)

        partial_integral_values[next_y1_prime] = total_sum 

    partial_integral_values[numpy.min(base_earning_condition_maries_pacses)] = 0 

    return partial_integral_values

def extensive_revenue_function(base_earning_condition_maries_pacses, other_earning_condition_maries_pacses, secondary_earning_condition_maries_pacses, irpp_condition_maries_pacses):
    partial_integral_values = extensive_partial_revenue_function(base_earning_condition_maries_pacses, other_earning_condition_maries_pacses, secondary_earning_condition_maries_pacses, irpp_condition_maries_pacses)
    
    extensive_rev_function = numpy.zeros_like(base_earning_condition_maries_pacses)


    maxi = numpy.max(base_earning_condition_maries_pacses) 

    for i in range(len(base_earning_condition_maries_pacses)):
        extensive_rev_function[i] = partial_integral_values[maxi] - partial_integral_values[base_earning_condition_maries_pacses[i]]
    
    print("extensive", extensive_rev_function)
    return - extensive_rev_function 



def intensive_revenue_function(earning, cdf, density, esperance_taux_marginal, elasticity):

    behavioral = - earning * density * elasticity * esperance_taux_marginal  
    mechanical = 1 - cdf
    return behavioral + mechanical


    


def tracer_et_integrer_revenue_fonctions(primary_income, secondary_income, primary_function, secondary_function):

   # on enleve les outliers 
    threshold = 3

    z_scores = (primary_function - numpy.mean(primary_function)) / numpy.std(primary_function)
    print("number of outliers primary revenue function", numpy.sum(abs(z_scores) > threshold))
    primary_income = primary_income[abs(z_scores) <= threshold]
    primary_function = primary_function[abs(z_scores) <= threshold]

    z_scores = (secondary_function - numpy.mean(secondary_function)) / numpy.std(secondary_function)
    print("number of outliers secondary revenue function", numpy.sum(abs(z_scores) > threshold))
    secondary_income = secondary_income[abs(z_scores) <= threshold]
    secondary_function = secondary_function[abs(z_scores) <= threshold]


    sorted_indices = numpy.argsort(primary_income)
    primary_income = primary_income[sorted_indices]
    primary_function = primary_function[sorted_indices]

    sorted_indices_s = numpy.argsort(secondary_income)
    secondary_income = secondary_income[sorted_indices_s]
    secondary_function = secondary_function[sorted_indices_s]

    print("we begin the kernel regression")

    # 500 for 2018, 1800 for 2017 a mettre sous forme de dictionnaire {year:band}
    bandwidth = 1000

    kernel_reg = KernelReg(endog=primary_function, exog=primary_income, var_type='c', reg_type='ll', bw=[bandwidth], ckertype='gaussian')
    smoothed_y_primary, _ = kernel_reg.fit()
    smoothed_y_primary = smoothed_y_primary[sorted_indices]
    integral_trap_primary = numpy.trapz(smoothed_y_primary, primary_income)
    print("Integral of smoothed_y primary", integral_trap_primary)

    print("end of the first regression")

    kernel_reg = KernelReg(endog=secondary_function, exog=secondary_income, var_type='c', reg_type='ll',bw=[bandwidth], ckertype='gaussian')
    smoothed_y_secondary, _ = kernel_reg.fit()
    smoothed_y_secondary = smoothed_y_secondary[sorted_indices_s]
    integral_trap_secondary = numpy.trapz(smoothed_y_secondary, secondary_income)
    print("Integral of smoothed_y secondary", integral_trap_secondary)

    print("end of the second regression")


    return integral_trap_primary, integral_trap_secondary, primary_income, smoothed_y_primary, secondary_income, smoothed_y_secondary


def graphe14(primary_earning_condition_maries_pacses, secondary_earning_condition_maries_pacses, irpp_condition_maries_pacses, taux_marginal_condition_maries_pacses, cdf_primary_earnings_condition_maries_pacses, cdf_secondary_earnings_condition_maries_pacses, density_primary_earnings_condition_maries_pacses, density_secondary_earnings_condition_maries_pacses, primary_esperance_taux_marginal_condition_maries_pacses, secondary_esperance_taux_marginal_condition_maries_pacses, period):
    """
    This function outputs the most important graph of the paper : Reform towards individual taxation: Political economy
    We are looking at winners and losers of a reform towards individualization among married couples for a given year 
    TBC... TODO 
    """

    # we test the political feasibility for different elasticities scenarios"
    eps1_tab = [0.25, 0.5, 0.75]
    eps2_tab = [0.75, 0.5, 0.25]
    rapport = [0.0]*len(eps1_tab)
    pourcentage_gagnants = [0.0]*len(eps1_tab)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    primary_extensive_revenue_function = extensive_revenue_function(primary_earning_condition_maries_pacses, secondary_earning_condition_maries_pacses, secondary_earning_condition_maries_pacses, irpp_condition_maries_pacses)
    secondary_extensive_revenue_function = extensive_revenue_function(secondary_earning_condition_maries_pacses, primary_earning_condition_maries_pacses, secondary_earning_condition_maries_pacses, irpp_condition_maries_pacses)

    for i in range(len(eps1_tab)):
        primary_elasticity_maries_pacses = eps1_tab[i]
        secondary_elasticity_maries_pacses = eps2_tab[i]
        
        primary_revenue_function = intensive_revenue_function(primary_earning_condition_maries_pacses, cdf_primary_earnings_condition_maries_pacses, density_primary_earnings_condition_maries_pacses, primary_esperance_taux_marginal_condition_maries_pacses, primary_elasticity_maries_pacses) + primary_extensive_revenue_function
        secondary_revenue_function = intensive_revenue_function(secondary_earning_condition_maries_pacses, cdf_secondary_earnings_condition_maries_pacses, density_secondary_earnings_condition_maries_pacses, secondary_esperance_taux_marginal_condition_maries_pacses, secondary_elasticity_maries_pacses) + secondary_extensive_revenue_function

        primary_integral, secondary_integral, primary_income, smoothed_y_primary, secondary_income, smoothed_y_secondary = tracer_et_integrer_revenue_fonctions(primary_earning_condition_maries_pacses, secondary_earning_condition_maries_pacses, primary_revenue_function, secondary_revenue_function)
        rapport[i] = primary_integral/secondary_integral
        print('rapport integrales scenario ', i, " ", rapport[i])

        axes[i].plot(primary_income[primary_income < 200000], smoothed_y_primary[primary_income < 200000], label = 'primary ep = {ep}, es = {es}'.format(ep = eps1_tab[i], es = eps2_tab[i]))
        axes[i].plot(secondary_income[secondary_income < 200000], smoothed_y_secondary[secondary_income < 200000], label = 'secondary ep = {ep}, es = {es}'.format(ep = eps1_tab[i], es = eps2_tab[i]))


        
        axes[i].legend()


        axes[i].set_xlabel('Gross income')
        axes[i].set_ylabel('R function')
        axes[i].set_title('Scenario {}'.format(i))

        tau_1 = 0.1 # comment bien choisir tau_1 ????
        tau_2 = - tau_1 * rapport[i]
        
        nouvel_irpp = -irpp_condition_maries_pacses + tau_1 * primary_earning_condition_maries_pacses/12 + tau_2 * secondary_earning_condition_maries_pacses/12 
        print("IR après réforme scenario ", i, " ", nouvel_irpp)

        # nombre de gagnants
        is_winner = secondary_earning_condition_maries_pacses*rapport[i] > primary_earning_condition_maries_pacses
        pourcentage_gagnants[i] = 100*is_winner.sum()/len(primary_earning_condition_maries_pacses)
        print("Scenario", i)
        print("Pourcentage de gagnants", period, i, pourcentage_gagnants[i])
    
    


    plt.tight_layout()  
    plt.show()
    plt.savefig('../outputs/13/graphe_13_{annee}.png'.format(annee = period))
    plt.close()


    plt.figure()
    x = numpy.linspace(0, 600000, 4)
    plt.plot(x, x, c = '#828282')

    green_shades = [(0.0, 1.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.6, 0.0)]
    for i in range(len(eps1_tab)):
        color = green_shades[i]
        plt.plot(x, rapport[i]*x, label = "ep = {ep}, es = {es}".format(ep = eps1_tab[i], es = eps2_tab[i]), color=color)
        plt.annotate(str(round(pourcentage_gagnants[i]))+ " %", xy = (50000 + 100000*i, 300000), bbox = dict(boxstyle ="round", fc = color))

    plt.scatter(secondary_earning_condition_maries_pacses, primary_earning_condition_maries_pacses, s = 0.1, c = '#828282') 

    #plt.axis("equal")

    eps = 5000
    plt.xlim(-eps, max(secondary_earning_condition_maries_pacses)) 
    plt.ylim(-eps, max(primary_earning_condition_maries_pacses)) 

    plt.grid()
    


    plt.xlabel('Secondary earner')
    plt.ylabel('Primary earner')
    plt.title("Reform towards individual taxation: Political economy - {}".format(period))

    plt.legend()
    plt.show()
    plt.savefig('../outputs/14/graphe_14_{annee}.png'.format(annee = period))
    plt.close()

    


    



@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
@click.option('-m', '--want_to_mute_decote', default = False, type = bool, required = True)
def simulation_reforme(annee = None, want_to_mute_decote = None):
    filename = "../data/{}/openfisca_erfs_fpr_{}.h5".format(annee, annee)
    data_persons_brut = pandas.read_hdf(filename, key = "individu_{}".format(annee))
    data_households_brut =  pandas.read_hdf(filename, key = "menage_{}".format(annee))
    
    data_persons = data_persons_brut.merge(data_households_brut, right_index = True, left_on = "idmen", suffixes = ("", "_x"))
    
    print("want_to_mute", want_to_mute_decote)
    print("Table des personnes")
    print(data_persons, "\n\n\n\n\n")

    #####################################################
    ########### Simulation ##############################
    #####################################################

    tax_benefit_system = FranceTaxBenefitSystem()
    if want_to_mute_decote:
        tax_benefit_system_reforme = vers_individualisation(mute_decote(tax_benefit_system))
    else:
        tax_benefit_system_reforme = vers_individualisation(tax_benefit_system) # chain the 2 reforms
        




    simulation = initialiser_simulation(tax_benefit_system_reforme, data_persons)
    #simulation.trace = True #utile pour voir toutes les étapes de la simulation

    period = str(annee)

    data_households = data_persons.drop_duplicates(subset='idmen', keep='first')

    for ma_variable in data_persons.columns.tolist():
        # variables pouvant entrer dans la simulation 
        if ma_variable not in ["idfam", "idfoy", "idmen", "noindiv", "quifam", "quifoy", "quimen", "wprm", "prest_precarite_hand",
                            "taux_csg_remplacement", "idmen_original", "idfoy_original", "idfam_original",
                            "idmen_original_x", "idfoy_original_x", "idfam_original_x", "wprm", "prest_precarite_hand",
                            "idmen_x", "idfoy_x", "idfam_x"]:
            # variables définies au niveau de l'individu
            if ma_variable not in ["loyer", "zone_apl", "statut_occupation_logement", "taxe_habitation", "logement_conventionne"]:
                simulation.set_input(ma_variable, period, numpy.array(data_persons[ma_variable]))
            # variables définies au niveau du ménage
            else:
                simulation.set_input(ma_variable, period, numpy.array(data_households[ma_variable]))
    
    # TODO commenter tout mon code de manière très très propre

    ancien_irpp = simulation.calculate('impot_revenu_restant_a_payer', period)
    maries_ou_pacses = simulation.calculate('maries_ou_pacses', period)
    ir_taux_marginal = simulation.calculate('ir_taux_marginal', period)
    primary_earning = simulation.calculate('primary_earning', period)
    secondary_earning = simulation.calculate('secondary_earning', period)
    revenu_celib = simulation.calculate('revenu_celibataire', period)
    primary_age = simulation.calculate('primary_age', period)
    secondary_age = simulation.calculate('secondary_age', period)

    # we restrict the foyers fiscaux to those who have positive earnings and in which both spouses are between 25 and 55 years old
    condition = (primary_earning > 0) & (secondary_earning >= 0) & (primary_age >= 25) & (primary_age <= 55) & (secondary_age >= 25) & (secondary_age <= 55)  
    ancien_irpp_condition = ancien_irpp[condition]
    maries_ou_pacses_condition = maries_ou_pacses[condition]
    ir_taux_marginal_condition = ir_taux_marginal[condition]
    primary_earning_condition = primary_earning[condition]
    secondary_earning_condition = secondary_earning[condition]

    # then we restrict to only those who are married (or pacsed)
    ancien_irpp_condition_maries_pacses = ancien_irpp_condition[maries_ou_pacses_condition]
    maries_ou_pacses_condition_maries_pacses = maries_ou_pacses_condition[maries_ou_pacses_condition] # useless i think
    ir_taux_marginal_condition_maries_pacses = ir_taux_marginal_condition[maries_ou_pacses_condition]
    primary_earning_condition_maries_pacses = primary_earning_condition[maries_ou_pacses_condition]
    secondary_earning_condition_maries_pacses = secondary_earning_condition[maries_ou_pacses_condition]








    # TODO individu fictif rassemblant les 5% du bas de la distribution (mais du coup poids associé plus important ?)



    print("beginning of the intermediate functions")

    cdf_primary_earnings = cdf_earnings(earning_condition_maries_pacses = primary_earning_condition_maries_pacses)
    density_primary_earnings = density_earnings(earning_condition_maries_pacses = primary_earning_condition_maries_pacses,
                                                period = period)
    primary_esperance_taux_marginal = esperance_taux_marginal(earning_condition_maries_pacses = primary_earning_condition_maries_pacses,
                                                              taux_marginal_condition_maries_pacses = ir_taux_marginal_condition_maries_pacses)

    cdf_secondary_earnings = cdf_earnings(earning_condition_maries_pacses = secondary_earning_condition_maries_pacses)
    density_secondary_earnings = density_earnings(earning_condition_maries_pacses = secondary_earning_condition_maries_pacses,
                                                  period = period)
    secondary_esperance_taux_marginal = esperance_taux_marginal(earning_condition_maries_pacses = secondary_earning_condition_maries_pacses,
                                                                taux_marginal_condition_maries_pacses = ir_taux_marginal_condition_maries_pacses)
    
    revenu, ir_marginal = moyenne_taux_marginal(primary_earning_condition_maries_pacses = primary_earning_condition_maries_pacses,
                                                secondary_earning_condition_maries_pacses = secondary_earning_condition_maries_pacses,
                                                taux_marginal_condition_maries_pacses = ir_taux_marginal_condition_maries_pacses,
                                                period = period)

    
    print("beginning of the graphs")

    graphe14(primary_earning_condition_maries_pacses = primary_earning_condition_maries_pacses,
             secondary_earning_condition_maries_pacses = secondary_earning_condition_maries_pacses,
             irpp_condition_maries_pacses = ancien_irpp_condition_maries_pacses,
             taux_marginal_condition_maries_pacses = ir_taux_marginal_condition_maries_pacses,
             cdf_primary_earnings_condition_maries_pacses = cdf_primary_earnings,
             cdf_secondary_earnings_condition_maries_pacses = cdf_secondary_earnings,
             density_primary_earnings_condition_maries_pacses = density_primary_earnings,
             density_secondary_earnings_condition_maries_pacses = density_secondary_earnings,
             primary_esperance_taux_marginal_condition_maries_pacses = primary_esperance_taux_marginal,
             secondary_esperance_taux_marginal_condition_maries_pacses = secondary_esperance_taux_marginal,
             period = period)


    graphe16(primary_earning = primary_earning,
            secondary_earning = secondary_earning, 
            maries_ou_pacses = maries_ou_pacses, 
            ancien_irpp = ancien_irpp,
            cdf_primary_earnings = cdf_primary_earnings,
            cdf_secondary_earnings = cdf_secondary_earnings,
            density_primary_earnings = density_primary_earnings,
            density_secondary_earnings = density_secondary_earnings, 
            primary_esperance_taux_marginal = primary_esperance_taux_marginal, 
            secondary_esperance_taux_marginal = secondary_esperance_taux_marginal, 
            period = period)
    
    graph17(primary_earning = primary_earning, 
            secondary_earning = secondary_earning, 
            maries_ou_pacses = maries_ou_pacses,
            period = period)
    
    
    # TODO ici aussi mettre noms variables des fonctions sinon on va s'y perdre
    graphB13(primary_earning, secondary_earning, revenu_celib, maries_ou_pacses, period)
    graphB14(primary_earning, secondary_earning, revenu_celib, maries_ou_pacses, period)
    graphB15(primary_earning, secondary_earning, revenu_celib, maries_ou_pacses, ir_taux_marginal, period)
    
    graphB16(primary_earning, secondary_earning, maries_ou_pacses, period)
    
    graphB17(primary_earning, secondary_earning, maries_ou_pacses, period)
    graphB18(primary_earning, secondary_earning, maries_ou_pacses, period)

    graphB21(primary_earning, secondary_earning, cdf_primary_earnings, cdf_secondary_earnings, period)
    graphB22(primary_earning, secondary_earning, density_primary_earnings, density_secondary_earnings, period)
    
    ma_borne = 500
    primary_esperance_taux_marginal500 = esperance_taux_marginal(earning_condition_maries_pacses = primary_earning_condition_maries_pacses,
                                                              taux_marginal_condition_maries_pacses = ir_taux_marginal_condition_maries_pacses,
                                                              borne = ma_borne)
    secondary_esperance_taux_marginal500 = esperance_taux_marginal(earning_condition_maries_pacses = secondary_earning_condition_maries_pacses,
                                                                taux_marginal_condition_maries_pacses = ir_taux_marginal_condition_maries_pacses,
                                                                borne = ma_borne)
    
    graphB23_B24(primary_earning, maries_ou_pacses, ir_taux_marginal, primary_esperance_taux_marginal500, period, 'primary')
    graphB23_B24(secondary_earning, maries_ou_pacses, ir_taux_marginal, secondary_esperance_taux_marginal500, period, 'secondary')




    

#################################################################################################
########### Graphes de vérification de la robustesse des résultats ##############################
#################################################################################################

def graph17(primary_earning, secondary_earning, maries_ou_pacses, period):
    revenu = primary_earning + secondary_earning
    revenu[revenu == 0] = 0.0001 # sinon division lève une erreur 
    revenu = revenu[maries_ou_pacses] #on retire le reste car on en a pas besoin ici dans les déciles 
    

    part_primary = primary_earning[maries_ou_pacses]/revenu * 100 

    revenu_sorted = numpy.sort(revenu)

    deciles = numpy.percentile(revenu_sorted, numpy.arange(0, 100, 10))

    decile_numbers = numpy.digitize(revenu, deciles) 

    
    decile_means = []
    for i in range(1, 11):
        mask = (decile_numbers == i)
        decile_part_primary = part_primary[mask]
        moyenne_part_primary = numpy.mean(decile_part_primary)
        decile_means.append(moyenne_part_primary)
        
    decile_means = numpy.array(decile_means)
    print("share of primary for year", period, decile_means)
    
    plt.figure()
    plt.scatter(numpy.arange(1,11), decile_means, s = 10)
    plt.xlabel('Gross income decile')
    plt.ylabel('Percent')
    plt.title("Median share of primary earner - {annee}".format(annee = period))
    plt.show()
    plt.savefig('../outputs/17/graphe_17_{annee}.png'.format(annee = period))
    plt.close()

    

def graphB15(primary_earning, secondary_earning, revenu_celib, maries_ou_pacses, ir_taux_marginal, period):

    # couples
    revenu = primary_earning + secondary_earning
    revenu_couples = revenu[maries_ou_pacses]
    mtr_couples = ir_taux_marginal[maries_ou_pacses]

    mtr_couples = mtr_couples[revenu_couples > 0]
    revenu_couples = revenu_couples[revenu_couples > 0]

    # en fait du aux enfants pour un revenu plus élevé le taux d'imposition peut etre plus grand
    # pour avoir une courbe globalement croissante il faut moyenner à un revenu donné
    
    sorted_indices = numpy.argsort(revenu_couples)
    earning_sorted = revenu_couples[sorted_indices]
    ir_marginal_sorted = mtr_couples[sorted_indices]

    sigma = 3.0  
    kernel_size = int(6 * sigma) * 2 + 1
    x_kernel = numpy.linspace(-3 * sigma, 3 * sigma, kernel_size)
    gaussian_kernel = numpy.exp(-x_kernel**2 / (2 * sigma**2)) / (sigma * numpy.sqrt(2 * numpy.pi))
    
    dirac_delta = numpy.zeros_like(x_kernel)
    dirac_delta[(len(x_kernel)//3):] = 0.5

    combined_kernel = gaussian_kernel + dirac_delta
    combined_kernel /= numpy.sum(combined_kernel)

    smoothed_y = convolve(ir_marginal_sorted, combined_kernel, mode='same')

    plt.figure()
    plt.scatter(earning_sorted, ir_marginal_sorted, label='Discrete Data - couples')
    plt.plot(earning_sorted, smoothed_y, label='Smoothed Data - couples')

    

    
    # singles
    revenu_celib = revenu_celib[~maries_ou_pacses]
    mtr_celib = ir_taux_marginal[~maries_ou_pacses]

    mtr_celib = mtr_celib[revenu_celib > 0]
    revenu_celib = revenu_celib[revenu_celib > 0]
    
    sorted_indices = numpy.argsort(revenu_celib)
    earning_sorted = revenu_celib[sorted_indices]
    ir_marginal_sorted = mtr_celib[sorted_indices]

    smoothed_y = convolve(ir_marginal_sorted, combined_kernel, mode='same')

    plt.scatter(earning_sorted, ir_marginal_sorted, label='Discrete Data - singles')
    plt.plot(earning_sorted, smoothed_y, label='Smoothed Data - singles')

    

    
    plt.xlabel('Gross earnings')
    plt.ylabel('MTR')
    plt.title("Effective marginal tax rates - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/B15/graphe_B15_{annee}.png'.format(annee = period))
    plt.close()



def graphB21(primary_earning_maries_pacses, secondary_earning_maries_pacses, cdf_primary, cdf_secondary, period):

    # lines not needed here but take it step by step...
    cdf_primary = cdf_primary[primary_earning_maries_pacses > 0]
    primary_earning_maries_pacses = primary_earning_maries_pacses[primary_earning_maries_pacses > 0]

    primary_sorted_indices = numpy.argsort(primary_earning_maries_pacses)
    primary_earning_sorted = primary_earning_maries_pacses[primary_sorted_indices]
    primary_cdf_sorted = cdf_primary[primary_sorted_indices]

    cdf_secondary = cdf_secondary[secondary_earning_maries_pacses > 0]
    secondary_earning_maries_pacses = secondary_earning_maries_pacses[secondary_earning_maries_pacses > 0]

    secondary_sorted_indices = numpy.argsort(secondary_earning_maries_pacses)
    secondary_earning_sorted = secondary_earning_maries_pacses[secondary_sorted_indices]
    secondary_cdf_sorted = cdf_secondary[secondary_sorted_indices]

    plt.figure()
    plt.plot(primary_earning_sorted[primary_earning_sorted < 400000], primary_cdf_sorted[primary_earning_sorted < 400000], label = 'primary')
    plt.plot(secondary_earning_sorted[secondary_earning_sorted < 400000], secondary_cdf_sorted[secondary_earning_sorted < 400000], label = 'secondary')
    plt.xlabel('Gross income')
    plt.ylabel('CDF')
    plt.title("Cumulative distribution function, primary and secondary earners - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/B21/graphe_B21_{annee}.png'.format(annee = period))
    plt.close()


def graphB22(primary_earning_maries_pacses, secondary_earning_maries_pacses, density_primary_earnings, density_secondary_earnings, period):

    density_primary_earnings = density_primary_earnings[primary_earning_maries_pacses > 0]
    primary_earning_maries_pacses = primary_earning_maries_pacses[primary_earning_maries_pacses > 0]

    primary_sorted_indices = numpy.argsort(primary_earning_maries_pacses)
    primary_earning_sorted = primary_earning_maries_pacses[primary_sorted_indices]
    primary_density_sorted = density_primary_earnings[primary_sorted_indices]

    density_secondary_earnings = density_secondary_earnings[secondary_earning_maries_pacses > 0]
    secondary_earning_maries_pacses = secondary_earning_maries_pacses[secondary_earning_maries_pacses > 0]

    secondary_sorted_indices = numpy.argsort(secondary_earning_maries_pacses)
    secondary_earning_sorted = secondary_earning_maries_pacses[secondary_sorted_indices]
    secondary_density_sorted = density_secondary_earnings[secondary_sorted_indices]

    plt.figure()
    plt.plot(primary_earning_sorted[primary_earning_sorted < 400000], primary_density_sorted[primary_earning_sorted < 400000], label = 'primary')
    plt.plot(secondary_earning_sorted[secondary_earning_sorted < 400000], secondary_density_sorted[secondary_earning_sorted < 400000], label = 'secondary')
    plt.xlabel('Gross income')
    plt.ylabel('PDF')
    plt.title("Probability density function, primary and secondary earners - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/B22/graphe_B22_{annee}.png'.format(annee = period))
    plt.close()


def graphB16(primary_earning, secondary_earning, maries_ou_pacses, period):

    # we take these elasticities for the graph B16 scenario
    eps1 = 0.25
    eps2 = 0.75

    primary_elasticity_simu = primary_elasticity(maries_ou_pacses, eps1)
    secondary_elasticity_simu = secondary_elasticity(maries_ou_pacses, eps2)
    couples_elasticity_simu = couples_elasticity(primary_earning, secondary_earning, maries_ou_pacses, eps1, eps2)

    revenu = primary_earning + secondary_earning     
    revenu = revenu[maries_ou_pacses] #on retire le reste car on en a pas besoin ici dans les déciles  

    primary_elasticity_simu = primary_elasticity_simu[maries_ou_pacses]
    secondary_elasticity_simu = secondary_elasticity_simu[maries_ou_pacses]
    couples_elasticity_simu = couples_elasticity_simu[maries_ou_pacses]

    revenu_sorted = numpy.sort(revenu)
    deciles = numpy.percentile(revenu_sorted, numpy.arange(0, 100, 10))
    decile_numbers = numpy.digitize(revenu, deciles) 

    
    decile_primary = []
    decile_secondary = []
    decile_couples = []

    for i in range(1, 11):
        mask = (decile_numbers == i)
        decile_primary.append(numpy.mean(primary_elasticity_simu[mask]))
        decile_secondary.append(numpy.mean(secondary_elasticity_simu[mask]))
        decile_couples.append(numpy.mean(couples_elasticity_simu[mask]))
        
    decile_primary = numpy.array(decile_primary)
    decile_secondary = numpy.array(decile_secondary)
    decile_couples = numpy.array(decile_couples)

    
    plt.figure()
    plt.plot(numpy.arange(1,11), decile_primary, label = "primary", linestyle='dashed')
    plt.plot(numpy.arange(1,11), decile_secondary, label = "secondary", linestyle='dashed')
    plt.plot(numpy.arange(1,11), decile_couples, label = "couples")
    plt.xlabel('Gross income (deciles)')
    plt.ylabel('Average elasticity')
    plt.title("Average elasticities of couples - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/B16/graphe_B16_{annee}.png'.format(annee = period))
    plt.close()



def graphB13(primary_earning, secondary_earning, revenu_celib, maries_ou_pacses, period):

    # couples 
    earning = primary_earning + secondary_earning
    earnings_maries_pacses = earning[maries_ou_pacses]
    counts = numpy.array([numpy.sum(earnings_maries_pacses <= y2) for y2 in earnings_maries_pacses])

    cdf = numpy.zeros_like(earnings_maries_pacses, dtype=float)
    cdf = counts/len(earnings_maries_pacses)

    cdf = cdf[earnings_maries_pacses > 0]
    earnings_maries_pacses = earnings_maries_pacses[earnings_maries_pacses > 0]
    

    sorted_indices = numpy.argsort(earnings_maries_pacses)
    earning_sorted = earnings_maries_pacses[sorted_indices]
    cdf_sorted = cdf[sorted_indices]

    plt.figure()
    plt.plot(earning_sorted[earning_sorted < 500000], cdf_sorted[earning_sorted < 500000], label = "couples")

    # singles
    revenu_celib = revenu_celib[~maries_ou_pacses]
    counts = numpy.array([numpy.sum(revenu_celib <= y2) for y2 in revenu_celib])
    cdf_celib = numpy.zeros(len(revenu_celib), dtype=float)
    cdf_celib = counts/len(revenu_celib)

    cdf_celib = cdf_celib[revenu_celib > 0]
    revenu_celib = revenu_celib[revenu_celib > 0]
    
    
    sorted_indices = numpy.argsort(revenu_celib)
    earning_celib_sorted = revenu_celib[sorted_indices]
    cdf_celib_sorted = cdf_celib[sorted_indices]


    plt.plot(earning_celib_sorted[earning_celib_sorted < 500000], cdf_celib_sorted[earning_celib_sorted < 500000], label = "singles")    
    plt.xlabel('Gross income')
    plt.ylabel('CDF')
    plt.title("Cumulative distribution function - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/B13/graphe_B13_{annee}.png'.format(annee = period))
    plt.close()



def graphB14(primary_earning, secondary_earning, revenu_celib, maries_ou_pacses, period):

    # IMPORTANT look at scipy.stats.gaussian_kde for pdf estimation with gaussian kernel


    # couples 
    earning = primary_earning + secondary_earning
    earnings_maries_pacses = earning[maries_ou_pacses]

    n = len(earnings_maries_pacses)
    estimated_std = numpy.std(earnings_maries_pacses, ddof=1)  
    bandwidth = 1.06 * estimated_std * n ** (-1/5)
    print("bandwidth", bandwidth)

    density_couple = numpy.zeros_like(earnings_maries_pacses, dtype=float)

    for i in range(len(earnings_maries_pacses)):
        kernel_values = gaussian_kernel((earnings_maries_pacses - earnings_maries_pacses[i]) / bandwidth)
        density_couple[i] = numpy.mean(kernel_values) * 1/bandwidth

    density_couple /= numpy.sum(density_couple)

    density_couple = density_couple[earnings_maries_pacses > 0]
    earnings_maries_pacses = earnings_maries_pacses[earnings_maries_pacses > 0]
    
    sorted_indices = numpy.argsort(earnings_maries_pacses)
    earning_sorted = earnings_maries_pacses[sorted_indices]
    density_couple_sorted = density_couple[sorted_indices]

    plt.figure()
    plt.plot(earning_sorted[earning_sorted < 500000], density_couple_sorted[earning_sorted < 500000], label = "couples")


    # singles 
    revenu_celib = revenu_celib[~maries_ou_pacses]

    # Calculate the bandwidth using Silverman's rule (see the paper https://arxiv.org/pdf/1212.2812.pdf top of page 12)
    n = len(revenu_celib)
    estimated_std = numpy.std(revenu_celib, ddof=1)  
    bandwidth = 1.06 * estimated_std * n ** (-1/5)
    print("bandwidth", bandwidth)

    density_celib = numpy.zeros_like(revenu_celib, dtype=float)

    for i in range(len(revenu_celib)):
        kernel_values = gaussian_kernel((revenu_celib - revenu_celib[i]) / bandwidth)
        density_celib[i] = numpy.mean(kernel_values) * 1/bandwidth

    density_celib /= numpy.sum(density_celib) 

    density_celib = density_celib[revenu_celib > 0]
    revenu_celib = revenu_celib[revenu_celib > 0]
    
    
    sorted_indices = numpy.argsort(revenu_celib)
    earning_celib_sorted = revenu_celib[sorted_indices]
    density_celib_sorted = density_celib[sorted_indices]


    plt.plot(earning_celib_sorted[earning_celib_sorted < 500000], density_celib_sorted[earning_celib_sorted < 500000], label = "singles")    
    plt.xlabel('Gross income')
    plt.ylabel('PDF')
    plt.title("Probability density function - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/B14/graphe_B14_{annee}.png'.format(annee = period))
    plt.close()


def graphB17(primary_earning, secondary_earning, maries_ou_pacses, period):

    primary_earning = primary_earning[maries_ou_pacses]
    secondary_earning = secondary_earning[maries_ou_pacses]
    revenu = primary_earning + secondary_earning

    dual_earner_earning = revenu[secondary_earning > 0]
    single_earner_earning = revenu[secondary_earning == 0]

    # dual earner couples 
    counts = numpy.array([numpy.sum(dual_earner_earning <= y2) for y2 in dual_earner_earning])
    cdf = numpy.zeros_like(dual_earner_earning, dtype=float)
    cdf = counts/len(dual_earner_earning)

    sorted_indices = numpy.argsort(dual_earner_earning)
    earning_sorted = dual_earner_earning[sorted_indices]
    cdf_sorted = cdf[sorted_indices]

    plt.figure()
    plt.plot(earning_sorted[earning_sorted < 500000], cdf_sorted[earning_sorted < 500000], label = "Dual Earner Couples")

    # single earner couples 
    counts = numpy.array([numpy.sum(single_earner_earning <= y2) for y2 in single_earner_earning])
    cdf = numpy.zeros_like(single_earner_earning, dtype=float)
    cdf = counts/len(single_earner_earning)

    cdf = cdf[single_earner_earning > 0]
    single_earner_earning = single_earner_earning[single_earner_earning > 0]

    sorted_indices = numpy.argsort(single_earner_earning)
    earning_sorted = single_earner_earning[sorted_indices]
    cdf_sorted = cdf[sorted_indices]

    plt.plot(earning_sorted[earning_sorted < 500000], cdf_sorted[earning_sorted < 500000], label = "Single Earner Couples")


    plt.xlabel('Gross income')
    plt.ylabel('CDF')
    plt.title("Cumulative distribution function, \n single earner and dual earner couples - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/B17/graphe_B17_{annee}.png'.format(annee = period))
    plt.close()





def graphB18(primary_earning, secondary_earning, maries_ou_pacses, period):

    primary_earning = primary_earning[maries_ou_pacses]
    secondary_earning = secondary_earning[maries_ou_pacses]
    revenu = primary_earning + secondary_earning

    dual_earner_earning = revenu[secondary_earning > 0]
    single_earner_earning = revenu[secondary_earning == 0]

    # dual earner couples 
    n = len(dual_earner_earning)
    estimated_std = numpy.std(dual_earner_earning, ddof=1)  
    bandwidth = 1.06 * estimated_std * n ** (-1/5)

    density_dual = numpy.zeros_like(dual_earner_earning, dtype=float)

    for i in range(len(dual_earner_earning)):
        kernel_values = gaussian_kernel((dual_earner_earning - dual_earner_earning[i]) / bandwidth)
        density_dual[i] = numpy.mean(kernel_values) * 1/bandwidth

    density_dual /= numpy.sum(density_dual)

    sorted_indices = numpy.argsort(dual_earner_earning)
    earning_sorted = dual_earner_earning[sorted_indices]
    pdf_sorted = density_dual[sorted_indices]

    plt.figure()
    plt.plot(earning_sorted[earning_sorted < 200000], pdf_sorted[earning_sorted < 200000], label = "Dual Earner Couples")

    # single earner couples 
    n = len(single_earner_earning)
    estimated_std = numpy.std(single_earner_earning, ddof=1)  
    bandwidth = 1.06 * estimated_std * n ** (-1/5)

    density_single = numpy.zeros_like(single_earner_earning, dtype=float)

    for i in range(len(single_earner_earning)):
        kernel_values = gaussian_kernel((single_earner_earning - single_earner_earning[i]) / bandwidth)
        density_single[i] = numpy.mean(kernel_values) * 1/bandwidth

    density_single /= numpy.sum(density_single)

    density_single = density_single[single_earner_earning > 0]
    single_earner_earning = single_earner_earning[single_earner_earning > 0]

    sorted_indices = numpy.argsort(single_earner_earning)
    earning_sorted = single_earner_earning[sorted_indices]
    pdf_sorted = density_single[sorted_indices]

    plt.plot(earning_sorted[earning_sorted < 200000], pdf_sorted[earning_sorted < 200000], label = "Single Earner Couples")


    plt.xlabel('Gross income')
    plt.ylabel('PDF')
    plt.title("Probability distribution function, \n single earner and dual earner couples - {annee}".format(annee = period))
    plt.legend()
    plt.show()
    plt.savefig('../outputs/B18/graphe_B18_{annee}.png'.format(annee = period))
    plt.close()

def graphB23_B24(earning, maries_ou_pacses, ir_taux_marginal, output, period, nom):

    earning = earning[maries_ou_pacses]
    ir_taux_marginal = ir_taux_marginal[maries_ou_pacses]
    output = output[maries_ou_pacses]

    ir_taux_marginal = ir_taux_marginal[earning > 0]
    output = output[earning > 0]
    earning = earning[earning > 0]

    sorted_indices = numpy.argsort(earning)
    earning_sorted = earning[sorted_indices]
    values = ir_taux_marginal/(1-ir_taux_marginal)
    values_sorted = values[sorted_indices]
    output_sorted = output[sorted_indices]

    sigma = 20.0  
    kernel_size = int(6 * sigma) * 2 + 1
    x_kernel = numpy.linspace(-3 * sigma, 3 * sigma, kernel_size)
    gaussian_kernel = numpy.exp(-x_kernel**2 / (2 * sigma**2)) / (sigma * numpy.sqrt(2 * numpy.pi))

    dirac_delta = numpy.zeros_like(x_kernel)
    dirac_delta[2*(len(x_kernel)//3):] = 0.5

    combined_kernel = gaussian_kernel + dirac_delta
    combined_kernel /= numpy.sum(combined_kernel)

    smoothed_y = convolve(values_sorted, combined_kernel, mode='same')

    plt.figure()
    plt.scatter(earning_sorted, values_sorted, label='Discrete Data')
    plt.scatter(earning_sorted, output_sorted, label = 'calcul a la main sur fenetre')
    plt.plot(earning_sorted, smoothed_y, label='Smoothed Data')

    plt.xlabel('{nom} Earnings'.format(nom = nom))
    plt.ylabel("Tm'/(1-Tm')")
    plt.title("Average marginal tax rates by {nom} earnings - january {annee}".format(annee = period, nom = nom))
    plt.show()
    plt.legend()
    if nom == 'primary':
        plt.savefig('../outputs/B23/graphe_B23_{annee}.png'.format(annee = period))
    else:
        plt.savefig('../outputs/B24/graphe_B24_{annee}.png'.format(annee = period))
    plt.close()



def graphe16(primary_earning, secondary_earning, maries_ou_pacses, ancien_irpp, cdf_primary_earnings, cdf_secondary_earnings, density_primary_earnings, density_secondary_earnings, primary_esperance_taux_marginal, secondary_esperance_taux_marginal, period):

    eps1_tab = [0.25, 0.5, 0.75]
    eps2_tab = [0.75, 0.5, 0.25]
    rapport = [0.0]*len(eps1_tab)


    primary_extensive_revenue_function = extensive_revenue_function(primary_earning, secondary_earning, secondary_earning, ancien_irpp, maries_ou_pacses)
    secondary_extensive_revenue_function = extensive_revenue_function(secondary_earning, primary_earning, secondary_earning, ancien_irpp, maries_ou_pacses)

    for i in range(len(eps1_tab)):
        primary_elasticity_maries_pacses = primary_elasticity(maries_ou_pacses, eps1_tab[i])
        secondary_elasticity_maries_pacses = secondary_elasticity(maries_ou_pacses, eps2_tab[i])
        
        primary_revenue_function = intensive_revenue_function(primary_earning, cdf_primary_earnings, density_primary_earnings, primary_esperance_taux_marginal, maries_ou_pacses, primary_elasticity_maries_pacses) + primary_extensive_revenue_function
        secondary_revenue_function = intensive_revenue_function(secondary_earning, cdf_secondary_earnings, density_secondary_earnings, secondary_esperance_taux_marginal, maries_ou_pacses, secondary_elasticity_maries_pacses) + secondary_extensive_revenue_function

        if i == 0:
            primary_earning_maries_pacses = primary_earning[maries_ou_pacses]
            secondary_earning_maries_pacses = secondary_earning[maries_ou_pacses]

            condition = (primary_earning_maries_pacses >= 0) & (secondary_earning_maries_pacses >= 0)
            primary_earning_maries_pacses = primary_earning_maries_pacses[condition]
            secondary_earning_maries_pacses = secondary_earning_maries_pacses[condition]

        primary_revenue_function = primary_revenue_function[maries_ou_pacses]
        primary_revenue_function = primary_revenue_function[condition]
        secondary_revenue_function = secondary_revenue_function[maries_ou_pacses]
        secondary_revenue_function = secondary_revenue_function[condition]
 
        primary_integral, secondary_integral, primary_income, smoothed_y_primary, secondary_income, smoothed_y_secondary = tracer_et_integrer_revenue_fonctions(primary_earning_maries_pacses, secondary_earning_maries_pacses, primary_revenue_function, secondary_revenue_function)
        rapport[i] = primary_integral/secondary_integral

    plt.figure()

    secondary_earning_maries_pacses = secondary_earning_maries_pacses[primary_earning_maries_pacses > 0]
    primary_earning_maries_pacses = primary_earning_maries_pacses[primary_earning_maries_pacses > 0]
    
    # equal weights
    weight = numpy.ones_like(primary_earning_maries_pacses)
    x_equal_weights = numpy.mean(weight*secondary_earning_maries_pacses)
    y_equal_weights = numpy.mean(weight*primary_earning_maries_pacses)
    print("equal weights : ", x_equal_weights, y_equal_weights)
    plt.plot(x_equal_weights, y_equal_weights, marker='+', markersize=10, color='red', label = "equal weights")
    #plt.scatter(x_equal_weights, y_equal_weights, color='red', marker='x', s=100) #other way to display the cross

    # decreasing
    total_earnings_maries_pacses = primary_earning_maries_pacses + secondary_earning_maries_pacses
    total_earnings_maries_pacses[total_earnings_maries_pacses == 0] = 0.001 #useless line since we restricted > 0 above
    weight = numpy.power(total_earnings_maries_pacses, -0.5)
    x_decreasing = numpy.mean(weight*secondary_earning_maries_pacses)
    y_decreasing = numpy.mean(weight*primary_earning_maries_pacses)
    print("decreasing ", x_decreasing, y_decreasing)
    plt.plot(x_decreasing, y_decreasing, marker='+', markersize=10, color='purple', label = "decreasing")

    # Rawlsian
    total_earnings_maries_pacses = primary_earning_maries_pacses + secondary_earning_maries_pacses
    total_earnings_sorted = numpy.sort(total_earnings_maries_pacses)
    index_5th_percentile = int(0.05 * (len(total_earnings_sorted) - 1))
    P5 =  total_earnings_sorted[index_5th_percentile]
    print("P5", P5)
    weight = 1*(total_earnings_maries_pacses <= P5)
    x_rawlsian = numpy.mean(weight*secondary_earning_maries_pacses)
    y_rawlsian = numpy.mean(weight*primary_earning_maries_pacses)
    print("rawlsian ", x_rawlsian, y_rawlsian)
    plt.plot(x_rawlsian, y_rawlsian, marker='+', markersize=10, color='orange', label = "rawlsian")

    # secondary earner
    total_earnings_maries_pacses = primary_earning_maries_pacses + secondary_earning_maries_pacses
    total_earnings_maries_pacses[total_earnings_maries_pacses == 0] = 0.01
    weight = secondary_earning_maries_pacses/total_earnings_maries_pacses
    x_secondary = numpy.mean(weight*secondary_earning_maries_pacses)
    y_secondary = numpy.mean(weight*primary_earning_maries_pacses)
    print("secondary earner ", x_secondary, y_secondary)
    plt.plot(x_secondary, y_secondary, marker='+', markersize=10, color='blue', label = "secondary")


    # rawslian secondary earner
    total_earnings_maries_pacses = primary_earning_maries_pacses + secondary_earning_maries_pacses
    total_earnings_sorted = numpy.sort(total_earnings_maries_pacses)
    index_5th_percentile = int(0.05 * (len(total_earnings_sorted) - 1))
    P5 =  total_earnings_sorted[index_5th_percentile]
    total_earnings_maries_pacses[total_earnings_maries_pacses == 0] = 0.01
    weight = (total_earnings_maries_pacses <= P5)*secondary_earning_maries_pacses/total_earnings_maries_pacses
    x_rawlsian_secondary = numpy.mean(weight*secondary_earning_maries_pacses)
    y_rawlsian_secondary = numpy.mean(weight*primary_earning_maries_pacses)
    print("rawlsian secondary ", x_rawlsian_secondary, y_rawlsian_secondary)
    plt.plot(x_rawlsian_secondary, y_rawlsian_secondary, marker='+', markersize=10, color='pink', label = "rawlsian secondary")



    
    x = numpy.linspace(0, 30000, 4)
    plt.plot(x, x, c = '#828282')

    green_shades = [(0.0, 1.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.6, 0.0)]
    for i in range(len(eps1_tab)):
        color = green_shades[i]
        plt.plot(x, rapport[i]*x, label = "ep = {ep}, es = {es}".format(ep = eps1_tab[i], es = eps2_tab[i]), color=color)

    plt.scatter(secondary_earning_maries_pacses, primary_earning_maries_pacses, s = 0.1, c = '#828282') 

    eps = 500
    plt.xlim(-eps, 30000) 
    plt.ylim(-eps, 30000) 

    plt.grid()
    


    plt.xlabel('Secondary earner')
    plt.ylabel('Primary earner')
    plt.title("Reform towards individual taxation: Welfare - {}".format(period))

    plt.legend()
    plt.show()
    plt.savefig('../outputs/16/graphe_16_{annee}.png'.format(annee = period))
    plt.close()

    


    



    





# def redirect_print_to_file(filename):
#     sys.stdout = open(filename, 'a')
    
# redirect_print_to_file('output_graphe_15.txt')

simulation_reforme()

# sys.stdout.close()
# sys.stdout = sys.__stdout__