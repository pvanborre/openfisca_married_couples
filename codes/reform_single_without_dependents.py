import numpy
import pandas 

import click

from openfisca_core.simulation_builder import SimulationBuilder
from openfisca_core.reforms import Reform
from openfisca_core import periods

from openfisca_france import FranceTaxBenefitSystem
from openfisca_france.model.base import *


pandas.options.display.max_columns = None


# The goal is to apply to all individuals the tax schedule for singles 

# First, we update the value of some parameters (which were null, and some formulas required their values, so we set them to 0).
def modify_parameters(parameters):
    reform_period = periods.period("2003")
    parameters.impot_revenu.calcul_reductions_impots.divers.intemp.max.update(period = reform_period, value = 0)
    parameters.impot_revenu.calcul_reductions_impots.divers.intemp.pac.update(period = reform_period, value = 0)
    parameters.impot_revenu.calcul_reductions_impots.divers.interets_emprunt_reprise_societe.plafond.update(period = reform_period, value = 0)
    parameters.impot_revenu.calcul_reductions_impots.divers.interets_emprunt_reprise_societe.taux.update(period = reform_period, value = 0)
    return parameters


class single_schedule(Reform):
    name = "reform where everyone is single"
    def apply(self):

        # we apply the parameters modification for year 2003
        self.modify_parameters(modifier_function = modify_parameters)

        # By changing the 2 variables below, we make sure that every individual is single
        class maries_ou_pacses(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "Married or pacses"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return False

        self.replace_variable(maries_ou_pacses)

        class celibataire_ou_divorce(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "Single or divorced"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return True

        self.replace_variable(celibataire_ou_divorce)



        class revenu_individu_couple(Variable):
            # I could have used revenu_categoriel (same modulo a deduction)
            value_type = float
            entity = Individu
            label = "Revenu d'un individu du foyer fiscal, si le couple est marié"
            definition_period = YEAR

            def formula(individu, period):
                traitements_salaires_pensions_rentes = individu('traitements_salaires_pensions_rentes', period)

                rev_cat_rvcm = individu.foyer_fiscal('revenu_categoriel_capital', period)
                rev_cat_rfon = individu.foyer_fiscal('revenu_categoriel_foncier', period)
                rev_cat_rpns = individu.foyer_fiscal('revenu_categoriel_non_salarial', period)
                rev_cat_pv = individu.foyer_fiscal('revenu_categoriel_plus_values', period)  

                # on fait un equal splitting des revenus (problème : si enfants ont des revenus les somme aussi, on suppose que cela n'arrive pas)     
                return traitements_salaires_pensions_rentes + rev_cat_rvcm/2 + rev_cat_rfon/2 + rev_cat_rpns/2 + rev_cat_pv/2
        
        self.add_variable(revenu_individu_couple)

        class primary_earning(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "Revenu le plus élevé du foyer fiscal, entre le déclarant principal et son conjoint"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                revenu_individu_i = foyer_fiscal.members('revenu_individu_couple', period) # est de taille nb individus
                revenu_declarant_principal = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.DECLARANT_PRINCIPAL) # est de taille nb foyers fiscaux
                revenu_du_conjoint = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.CONJOINT) # est de taille nb foyers fiscaux 

                return max_(revenu_declarant_principal, revenu_du_conjoint) 

        self.add_variable(primary_earning)

        class secondary_earning(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "Revenu le moins élevé du foyer fiscal, entre le déclarant principal et son conjoint"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                revenu_individu_i = foyer_fiscal.members('revenu_individu_couple', period) # est de taille nb individus
                revenu_declarant_principal = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.DECLARANT_PRINCIPAL) # est de taille nb foyers fiscaux
                revenu_du_conjoint = foyer_fiscal.sum(revenu_individu_i, role = FoyerFiscal.CONJOINT) # est de taille nb foyers fiscaux 

                return min_(revenu_declarant_principal, revenu_du_conjoint) 

        self.add_variable(secondary_earning)


class without_dependents(Reform):
    name = "reform where nobody has dependents"
    def apply(self):   

        class nb_pac(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "Everyone has no dependents"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return 0

        self.replace_variable(nb_pac)

        class nbptr(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "Everyone has 1 part"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return 1

        self.replace_variable(nbptr)


        class enfant_a_charge(Variable):
            value_type = bool
            entity = Individu
            label = "Nobody is a dependent" # implies nbF, nbG, nbH, nbI equals 0 cf. https://github.com/openfisca/openfisca-france/blob/f770f896dc256a8a7715e002860130a6a0ea9d5c/openfisca_france/model/prelevements_obligatoires/impot_revenu/ir.py#L215
            definition_period = YEAR

            def formula(individu, period):
                return False 

        self.replace_variable(enfant_a_charge)

        class enfant_majeur_celibataire_sans_enfant(Variable):
            value_type = bool
            entity = Individu
            label = 'No adult child without children' # implies nbJ, nombre_enfants_majeurs_celibataires_sans_enfant equals 0
            definition_period = YEAR

            def formula(individu, period):
                return False 
        
        self.replace_variable(enfant_majeur_celibataire_sans_enfant)
            
        # from now on a bit long but we set all measures designed to give more 'demi-parts' to 0
        class caseE(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "No supplementary demi-part"
            end = '2012-12-31'
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return False 
            
        self.replace_variable(caseE)


        class caseF(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "No supplementary demi-part"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return False 
            
        self.replace_variable(caseF)

        class caseG(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "No war widowed pension"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return False 
            
        self.replace_variable(caseG)

        class caseK(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "No supplementary demi-part"
            end = '2011-12-31'
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return False 
            
        self.replace_variable(caseK)

        class caseL(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "No supplementary demi-part"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return False 
            
        self.replace_variable(caseL)

        class caseN(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "You live alone"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return False 
            
        self.replace_variable(caseN)

        class caseP(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "No invalidity pension"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return False 
            
        self.replace_variable(caseP)

        class caseS(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "No war widowed pension"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return False 
            
        self.replace_variable(caseS)

        class caseT(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "Not isolated"
            definition_period = MONTH

            def formula(foyer_fiscal, period):
                return False 
            
        self.replace_variable(caseT)

        class caseW(Variable):
            value_type = bool
            entity = FoyerFiscal
            label = "No war widowed pension"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                return False 
            
        self.replace_variable(caseW)




class no_dependents_for_primary(Reform):
    name = "reform where everyone has the tax schedule for singles, computes Ts(y1)"
    def apply(self):   

        class revenu_categoriel(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "We modify the earning that will be taken into account for the tax computation"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                primary_earning = foyer_fiscal('primary_earning', period)
                return primary_earning

        self.replace_variable(revenu_categoriel)


class no_dependents_for_secondary(Reform):
    name = "reform where everyone has the tax schedule for singles, computes Ts(y2)"
    def apply(self):   

        class revenu_categoriel(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "We modify the earning that will be taken into account for the tax computation"
            definition_period = YEAR

            def formula(foyer_fiscal, period):
                secondary_earning = foyer_fiscal('secondary_earning', period)
                return secondary_earning

        self.replace_variable(revenu_categoriel)

        
#################################################################################################################################################
# General context : OpenFisca considers that there are 4 distinct entities in the population (for more information see the documentation https://openfisca.org/doc/key-concepts/index.html)
# Of course individuals and households, that were initially defined in the ERFS database
# And also foyers fiscaux, that is the French tax unit (each foyer fiscal fills in a tax declaration)
# And families, but we don't use them 
# I wanted to emphasis the fact that foyers fiscaux and families are build by the OpenFisca model, they were not native in the ERFS data
# for more details on this foyers fiscal and families imputation see https://github.com/openfisca/openfisca-france-data/blob/master/openfisca_france_data/erfs_fpr/input_data_builder/step_04_famille.py
#################################################################################################################################################

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



# here we define the entities of the OpenFisca model
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



def simulation_function(tax_benefit_system_reforme, data_persons, annee):

    simulation = initialiser_simulation(tax_benefit_system_reforme, data_persons)
    #simulation.trace = True #utile pour voir toutes les étapes de la simulation
    period = str(annee)

    data_households = data_persons.drop_duplicates(subset='idmen', keep='first')

    for ma_variable in data_persons.columns.tolist():
        # variables pouvant entrer dans la simulation 
        if ma_variable not in ["idfam", "idfoy", "idmen", "noindiv", "quifam", "quifoy", "quimen", "prest_precarite_hand",
                            "taux_csg_remplacement", "idmen_original", "idfoy_original", "idfam_original",
                            "idmen_original_x", "idfoy_original_x", "idfam_original_x", "wprm", "prest_precarite_hand",
                            "idmen_x", "idfoy_x", "idfam_x", "weight_foyerfiscal"]:
            # variables définies au niveau de l'individu
            if ma_variable not in ["loyer", "zone_apl", "statut_occupation_logement", "taxe_habitation", "logement_conventionne"]:
                simulation.set_input(ma_variable, period, numpy.array(data_persons[ma_variable]))
            # variables définies au niveau du ménage
            else:
                simulation.set_input(ma_variable, period, numpy.array(data_households[ma_variable]))
    
    # we compute variables of interest
    revenu_categoriel = simulation.calculate('revenu_categoriel', period)
    impot_revenu_restant_a_payer = simulation.calculate('impot_revenu_restant_a_payer', period)
    

    # we take our original data and keep only the id of the foyer_fiscal and the weight of the household 
    result_df = data_persons.drop_duplicates(subset='idfoy', keep='first')
    result_df = result_df[['idfoy', 'weight_foyerfiscal']]

    # then we add all columns computed in the simulation
    result_df['revenu_categoriel'] = revenu_categoriel
    result_df['impot_revenu_restant_a_payer'] = impot_revenu_restant_a_payer

    return result_df
    # what is important to note is that in the baseline, all individuals in this dataset are not married
    # therefore there is a need to perform a left join with the married dataset



@click.command()
@click.option('-y', '--annee', default = None, type = int, required = True)
@click.option('-p', '--free_dependents_for_primary', default = None, type = bool, required = True)
def simulation_reforme(annee = None, free_dependents_for_primary = None):
    filename = "../data/{}/openfisca_erfs_fpr_{}.h5".format(annee, annee)
    data_persons_brut = pandas.read_hdf(filename, key = "individu_{}".format(annee))
    data_households_brut =  pandas.read_hdf(filename, key = "menage_{}".format(annee))
    
    data_persons = data_persons_brut.merge(data_households_brut, right_index = True, left_on = "idmen", suffixes = ("", "_x"))
    
    # Weight adjustment : wprm weights are for households, whereas we work on foyers fiscaux
    # the idea in OpenFisca France-data is to say that individuals have the weight of their households (what is done in the left join above)
    # and then summing over individuals of the foyer fiscal gives the weight of the foyer fiscal
    sum_wprm_by_idfoy = data_persons.groupby('idfoy')['wprm'].sum().reset_index()
    sum_wprm_by_idfoy = sum_wprm_by_idfoy.rename(columns={'wprm': 'weight_foyerfiscal'})
    data_persons = pandas.merge(data_persons, sum_wprm_by_idfoy, on='idfoy')

    print("Table des personnes")
    print(data_persons, "\n\n\n\n\n")

    #####################################################
    ########### Simulation ##############################
    #####################################################

    tax_benefit_system = FranceTaxBenefitSystem()

    """
    To emphasis what we want to do, we average : 
    - T_single_with_dependents(y1) + T_single_without_dependents(y2) that is considering both spouses singles and assigning dependents to the primary earner
    - T_single_without_dependents(y1) + T_single_with_dependents(y2) that is considering both spouses singles and assigning dependents to the secondary earner
    Here in this file we computed T_single_without_dependents(y1) (if part below) and T_single_without_dependents(y2) (else part below)
    """

    if free_dependents_for_primary:
        tax_benefit_system_no_dependents_primary = no_dependents_for_primary(without_dependents(single_schedule(tax_benefit_system)))
        result_df = simulation_function(tax_benefit_system_no_dependents_primary, data_persons, annee)
        result_df.rename(columns={'impot_revenu_restant_a_payer': 'tax_single_without_dependents_primary'}, inplace=True)

    else:
        tax_benefit_system_no_dependents_secondary = no_dependents_for_secondary(without_dependents(single_schedule(tax_benefit_system)))
        result_df = simulation_function(tax_benefit_system_no_dependents_secondary, data_persons, annee)
        result_df.rename(columns={'impot_revenu_restant_a_payer': 'tax_single_without_dependents_secondary'}, inplace=True)

    print("Dataframe of this simulation")
    print(result_df)

    print("Dataframe of the other simulation, only married couples with some restrictions on age and earnings")
    married_dataset = pandas.read_csv(f'./excel/{annee}/married_25_55_positive_{annee}.csv')
    print(married_dataset)

    print("Merging the two dataframes")
    # keeps only married couples that we want (married couples with some restrictions on age and earnings)
    merged_dataset = pandas.merge(married_dataset, result_df, how='left', on=['idfoy', 'weight_foyerfiscal'])
    print(merged_dataset)
    

    if free_dependents_for_primary:
        merged_dataset = merged_dataset[['idfoy', 'weight_foyerfiscal', 'tax_single_without_dependents_primary']]
        merged_dataset.to_csv(f'excel/{annee}/tax_single_without_dependents_primary_{annee}.csv', index=False)

    else:
        merged_dataset = merged_dataset[['idfoy', 'weight_foyerfiscal', 'tax_single_without_dependents_secondary']]
        merged_dataset.to_csv(f'excel/{annee}/tax_single_without_dependents_secondary_{annee}.csv', index=False)


    
simulation_reforme()

