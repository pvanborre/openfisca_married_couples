
import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, PchipInterpolator

import click

from openfisca_core.simulation_builder import SimulationBuilder
from openfisca_core.reforms import Reform

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

class vers_individualisation(Reform):
    name = "on code la réforme nouvel impot = ancien impot + tau_1 revenu_1 + tau_2 revenu_2"
    def apply(self):

        class revenu_individu(Variable):
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



def cdf_earnings(earning, maries_ou_pacses, period, title):
    earnings_maries_pacses = earning[maries_ou_pacses]
    counts = numpy.array([numpy.sum(earnings_maries_pacses <= y2) for y2 in earnings_maries_pacses])

    cdf = numpy.zeros_like(earning, dtype=float)
    cdf[maries_ou_pacses] = counts/len(earnings_maries_pacses)
    print("check de la cdf", cdf)

    # plot here figure B17 cumulative distribution function of gross income
    plt.figure()
    plt.scatter(earning[earning >= 0], cdf[earning >= 0], s = 10)
    plt.xlabel('Revenu annuel')
    plt.title("{type} earnings cumulative distribution function - january {annee}".format(type = title, annee = period))
    plt.show()
    plt.savefig('../outputs/graphe_B17_{type}_{annee}.png'.format(type = title, annee = period))

    return cdf



def gaussian_kernel(x):
    return 1/numpy.sqrt(2*numpy.pi) * numpy.exp(-1/2 * x * x)

def density_earnings(earning, maries_ou_pacses, period, title):
    earnings_maries_pacses = earning[maries_ou_pacses]
    
    # Calculate the bandwidth using Silverman's rule (see the paper https://arxiv.org/pdf/1212.2812.pdf top of page 12)
    n = len(earnings_maries_pacses)
    estimated_std = numpy.std(earnings_maries_pacses, ddof=1)  
    bandwidth = 1.06 * estimated_std * n ** (-1/5)
    print("bandwidth", bandwidth)

    # remarque : il ne faut pas que les foyers fiscaux non mariés ou pacsés portent de densité, on les retire donc puis on les remet
    kernel_values = gaussian_kernel((earnings_maries_pacses[:, numpy.newaxis] - earnings_maries_pacses) / bandwidth)
    density = numpy.zeros_like(earning, dtype=float)
    density[maries_ou_pacses] = (1 / bandwidth) * numpy.mean(kernel_values, axis=1)
    density /= numpy.sum(density) # attention ne valait pas forcément 1 avant (classique avec les kernels) 

    print("check de la densite", density)

    # plot here figure B14 probability density function 
    plt.figure()
    plt.scatter(earning[earning >= 0], density[earning >= 0], s = 10)
    plt.xlabel('Revenu annuel')
    plt.title("{type} earnings probability density function - january {annee}".format(type = title, annee = period))
    plt.show()
    plt.savefig('../outputs/graphe_B14_{type}_{annee}.png'.format(type = title, annee = period))

    return density




def esperance_taux_marginal(earning, ir_taux_marginal, maries_ou_pacses, borne = 0.05):
    output = numpy.zeros_like(earning, dtype=float)
    for i in range(len(earning)):
        diff = numpy.abs(earning - earning[i])
        ir_taux_marginal2 = numpy.copy(ir_taux_marginal)
        ir_taux_marginal2[diff > borne] = 0
        output[i] = numpy.sum(ir_taux_marginal2 / (1 - ir_taux_marginal2))/numpy.sum(diff <= borne)

    return output*maries_ou_pacses



def tax_two_derivative(primary_earning, secondary_earning, ir_taux_marginal):
    revenu = primary_earning + secondary_earning

    sorted_indices = numpy.argsort(revenu)
    earning_sorted = revenu[sorted_indices]
    ir_marginal_sorted = ir_taux_marginal[sorted_indices]

    unique_incomes = numpy.unique(earning_sorted) #unique nécessaire car sinon divisions par 0 dans le gradient
    mean_values = [numpy.mean(ir_marginal_sorted[earning_sorted == i]) for i in unique_incomes]
    tax_two_sorted = numpy.gradient(mean_values, unique_incomes)
    tax_two_original = numpy.interp(revenu, unique_incomes, tax_two_sorted) # obtenir a nouveau valeurs perdues par le unique par une interpolation linéaire

    original_indices = numpy.argsort(sorted_indices)
    return tax_two_original[original_indices]

def primary_elasticity(primary_earning, secondary_earning, maries_ou_pacses, eps1, eps2, ir_taux_marginal, tax_two_derivative):
    # formule en lemma 4 
    denominateur = 1 + tax_two_derivative/(1-ir_taux_marginal) * (eps1*primary_earning + eps2*secondary_earning)
    return maries_ou_pacses * eps1 * 1/denominateur

def secondary_elasticity(primary_earning, secondary_earning, maries_ou_pacses, eps1, eps2, ir_taux_marginal, tax_two_derivative):
    # formule en lemma 4 
    denominateur = 1 + tax_two_derivative/(1-ir_taux_marginal) * (eps1*primary_earning + eps2*secondary_earning)
    return maries_ou_pacses * eps2 * 1/denominateur


def revenue_function(earning, cdf, density, esperance_taux_marginal, maries_ou_pacses, elasticity):

    behavioral = - earning * density * elasticity * esperance_taux_marginal  
    mechanical = 1 - cdf
    return (behavioral + mechanical) * maries_ou_pacses






@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
def simulation_reforme(annee = None):
    filename = "../data/{}/openfisca_erfs_fpr_{}.h5".format(annee, annee)
    data_persons_brut = pandas.read_hdf(filename, key = "individu_{}".format(annee))
    data_households_brut =  pandas.read_hdf(filename, key = "menage_{}".format(annee))
    data_persons = data_persons_brut.merge(data_households_brut, right_index = True, left_on = "idmen", suffixes = ("", "_x"))

    print("Table des personnes")
    print(data_persons, "\n\n\n\n\n")

    #####################################################
    ########### Simulation ##############################
    #####################################################

    tax_benefit_system = FranceTaxBenefitSystem()
    tax_benefit_system_reforme = vers_individualisation(tax_benefit_system)




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
    

    ancien_irpp = simulation.calculate('irpp', period)
    maries_ou_pacses = simulation.calculate('maries_ou_pacses', period)
    ir_taux_marginal = simulation.calculate('ir_taux_marginal', period)
    primary_earning_maries_pacses = simulation.calculate('primary_earning', period)
    secondary_earning_maries_pacses = simulation.calculate('secondary_earning', period)
    

    cdf_primary_earnings = cdf_earnings(primary_earning_maries_pacses, maries_ou_pacses, period, 'primary')
    density_primary_earnings = density_earnings(primary_earning_maries_pacses, maries_ou_pacses, period, 'primary')
    primary_esperance_taux_marginal = esperance_taux_marginal(primary_earning_maries_pacses, ir_taux_marginal, maries_ou_pacses)

    cdf_secondary_earnings = cdf_earnings(secondary_earning_maries_pacses, maries_ou_pacses, period, 'secondary')
    density_secondary_earnings = density_earnings(secondary_earning_maries_pacses, maries_ou_pacses, period, 'secondary')
    secondary_esperance_taux_marginal = esperance_taux_marginal(secondary_earning_maries_pacses, ir_taux_marginal, maries_ou_pacses)
    
    tax_two_derivative_simulation = tax_two_derivative(primary_earning_maries_pacses, secondary_earning_maries_pacses, ir_taux_marginal)

    graphe14(primary_earning_maries_pacses = primary_earning_maries_pacses, 
             secondary_earning_maries_pacses = secondary_earning_maries_pacses,
             maries_ou_pacses = maries_ou_pacses, 
             ancien_irpp = ancien_irpp, 
             ir_taux_marginal = ir_taux_marginal,
             tax_two_derivative_simulation = tax_two_derivative_simulation,
             cdf_primary_earnings = cdf_primary_earnings,
             cdf_secondary_earnings = cdf_secondary_earnings,
             density_primary_earnings = density_primary_earnings,
             density_secondary_earnings = density_secondary_earnings,
             primary_esperance_taux_marginal = primary_esperance_taux_marginal,
             secondary_esperance_taux_marginal = secondary_esperance_taux_marginal,
             period = period)
    
def graphe14(primary_earning_maries_pacses, secondary_earning_maries_pacses, maries_ou_pacses, ancien_irpp, ir_taux_marginal, tax_two_derivative_simulation, cdf_primary_earnings, cdf_secondary_earnings, density_primary_earnings, density_secondary_earnings, primary_esperance_taux_marginal, secondary_esperance_taux_marginal, period):
 
    # Titre graphique : Gagnants et perdants d'une réforme vers l'individualisation de l'impôt, 
    # parmi les couples mariés ou pacsés, en janvier de l'année considérée

    eps1_tab = [0.25, 0.5, 0.75]
    eps2_tab = [0.75, 0.5, 0.25]
    rapport = [0.0]*len(eps1_tab)
    pourcentage_gagnants = [0.0]*len(eps1_tab)

    for i in range(1):
        primary_elasticity_maries_pacses = primary_elasticity(primary_earning_maries_pacses, secondary_earning_maries_pacses, maries_ou_pacses, eps1_tab[i], eps2_tab[i], ir_taux_marginal, tax_two_derivative_simulation)
        secondary_elasticity_maries_pacses = secondary_elasticity(primary_earning_maries_pacses, secondary_earning_maries_pacses, maries_ou_pacses, eps1_tab[i], eps2_tab[i], ir_taux_marginal, tax_two_derivative_simulation)
        
        primary_revenue_function = revenue_function(primary_earning_maries_pacses, cdf_primary_earnings, density_primary_earnings, primary_esperance_taux_marginal, maries_ou_pacses, primary_elasticity_maries_pacses)
        secondary_revenue_function = revenue_function(secondary_earning_maries_pacses, cdf_secondary_earnings, density_secondary_earnings, secondary_esperance_taux_marginal, maries_ou_pacses, secondary_elasticity_maries_pacses)


        primary_earning_maries_pacses = primary_earning_maries_pacses[maries_ou_pacses]
        print("revenu du déclarant principal", primary_earning_maries_pacses)
        secondary_earning_maries_pacses = secondary_earning_maries_pacses[maries_ou_pacses]
        print("revenu du conjoint", secondary_earning_maries_pacses)

        # Statistiques descriptives
        nombre_foyers_maries_pacses = len(primary_earning_maries_pacses)
        print("Nombre de foyers fiscaux dont membres sont mariés ou pacsés", nombre_foyers_maries_pacses)
        print("Proportion de foyers fiscaux dont membres mariés ou pacsés", nombre_foyers_maries_pacses/len(maries_ou_pacses))


        # on enlève les outliers
        condition = (primary_earning_maries_pacses >= 0) & (secondary_earning_maries_pacses >= 0)
        primary_earning_maries_pacses = primary_earning_maries_pacses[condition]
        secondary_earning_maries_pacses = secondary_earning_maries_pacses[condition]
        print("Nombre d'outliers que l'on retire", nombre_foyers_maries_pacses - len(primary_earning_maries_pacses))

        primary_revenue_function = primary_revenue_function[maries_ou_pacses]
        primary_revenue_function = primary_revenue_function[condition]
        secondary_revenue_function = secondary_revenue_function[maries_ou_pacses]
        secondary_revenue_function = secondary_revenue_function[condition]
        ancien_irpp = ancien_irpp[maries_ou_pacses]
        ancien_irpp = ancien_irpp[condition]

        primary_integral = tracer_et_integrer_revenue_fonctions(primary_earning_maries_pacses, primary_revenue_function, 'primary')
        secondary_integral = tracer_et_integrer_revenue_fonctions(secondary_earning_maries_pacses, secondary_revenue_function, 'secondary')
        rapport[i] = primary_integral/secondary_integral
        print('rapport integrales scenario ', i, " ", rapport[i])


        tau_1 = 0.1 # comment bien choisir tau_1 ????
        tau_2 = - tau_1 * rapport[i]
        
        nouvel_irpp = -ancien_irpp + tau_1 * primary_earning_maries_pacses/12 + tau_2 * secondary_earning_maries_pacses/12 
        print("IR après réforme scenario ", i, " ", nouvel_irpp)

        # nombre de gagnants
        is_winner = secondary_earning_maries_pacses*rapport[i] > primary_earning_maries_pacses
        print("Nombre de gagnants", is_winner.sum())
        pourcentage_gagnants[i] = 100*is_winner.sum()/len(primary_earning_maries_pacses)
        print("Pourcentage de gagnants", pourcentage_gagnants[i])



    plt.figure()
    x = numpy.linspace(0, 600000, 4)
    plt.plot(x, x, c = '#828282')
    for i in range(len(eps1_tab)):
        plt.plot(x, rapport[i]*x, label = "ep = {ep}, es = {es}".format(ep = eps1_tab[i], es = eps2_tab[i]))
        plt.annotate(str(pourcentage_gagnants[i])+ " %", xy = (1000000 + 100000*i, 100000), bbox = dict(boxstyle ="round", fc ="0.8"))

    plt.scatter(secondary_earning_maries_pacses, primary_earning_maries_pacses, s = 0.1, c = '#828282') 

    
    plt.grid()
    plt.axis('equal')  

    plt.xlabel('Revenu annuel du secondary earner du foyer fiscal')
    plt.ylabel('Revenu annuel du primary earner du foyer fiscal')
    plt.title("Gagnants et perdants d'une réforme vers l'individualisation de l'impôt, \n parmi les couples mariés ou pacsés français, en janvier {}".format(period))

    plt.legend()
    plt.show()
    plt.savefig('../outputs/graphe_14_{}.png'.format(period))



    

    


def tracer_et_integrer_revenue_fonctions(income, values, title):

    sorted_indices = numpy.argsort(income)
    income = income[sorted_indices]
    values = values[sorted_indices]

    unique_incomes = numpy.unique(income) # on a besoin du unique pour l'interpolation 
    mean_values = [numpy.mean(values[income == i]) for i in unique_incomes]
    
    pchip = PchipInterpolator(unique_incomes, mean_values)
    integral_pchip = pchip.integrate(min(unique_incomes), max(unique_incomes))
    print("Integrale hermite interpolation", integral_pchip)


    x_continuous = numpy.linspace(min(income), max(income), 1000)
   
    plt.figure()
    plt.plot(x_continuous, pchip(x_continuous), label='hermite')
    plt.scatter(income, values, label='Discrete Data', color='red')
    plt.xlabel('Income')
    plt.ylabel(title)
    plt.legend()
    plt.show()
    plt.savefig('../outputs/{}_revenue_function.png'.format(title))

    return integral_pchip


# TODO : plot cumulative distribution function (figure b21 et 22)
# TODO : plot esperance T'/(1-T') (figure b23 et 24)


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


simulation_reforme()