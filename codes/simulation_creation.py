import numpy
import pandas

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
                decote_celib = max_(0, decote_seuil_celib - taux_decote * ir_plaf_qf)

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
    
    # we compute variables of interest
    ancien_irpp = simulation.calculate('impot_revenu_restant_a_payer', period)
    maries_ou_pacses = simulation.calculate('maries_ou_pacses', period)
    taux_marginal = simulation.calculate('ir_taux_marginal', period)
    primary_earning = simulation.calculate('primary_earning', period)
    secondary_earning = simulation.calculate('secondary_earning', period)
    single_earning = simulation.calculate('revenu_celibataire', period)
    primary_age = simulation.calculate('primary_age', period)
    secondary_age = simulation.calculate('secondary_age', period)

    # we take our original data and keep only the id of the foyer_fiscal and the weight of the household + the age of the first person of the foyer_fiscal
    result_df = data_persons.drop_duplicates(subset='idfoy', keep='first')
    result_df = result_df[['idfoy', 'age', 'wprm']]

    # then we add all columns computed in the simulation
    result_df['ancien_irpp'] = ancien_irpp
    result_df['maries_ou_pacses'] = maries_ou_pacses
    result_df['taux_marginal'] = taux_marginal
    result_df['primary_earning'] = primary_earning
    result_df['secondary_earning'] = secondary_earning
    result_df['single_earning'] = single_earning
    result_df['primary_age'] = primary_age
    result_df['secondary_age'] = secondary_age

    print(result_df)

    # we create a dataframe only for married couples, with positive earnings and in which both spouses are adult
    df_married = result_df[result_df['maries_ou_pacses']]
    df_married = df_married.drop(['age', 'single_earning', 'maries_ou_pacses'], axis=1)
    df_married = df_married[df_married['primary_earning'] >= 0]
    df_married = df_married[df_married['secondary_earning'] >= 0]
    df_married = df_married[df_married['primary_age'] >= 18]
    df_married = df_married[df_married['secondary_age'] >= 18]
    df_married.to_csv(f'excel/{period}/married_adults_{period}.csv', index=False)

    # we restrict to couples in which both spouses are between 25 and 55 years old
    df_married_25_55 = df_married[df_married['primary_age'] >= 25]
    df_married_25_55 = df_married_25_55[df_married_25_55['primary_age'] <= 55]
    df_married_25_55 = df_married_25_55[df_married_25_55['secondary_age'] >= 25]
    df_married_25_55 = df_married_25_55[df_married_25_55['secondary_age'] <= 55]
    df_married_25_55.to_csv(f'excel/{period}/married_25_55_{period}.csv', index=False)

    # we construct single and dual earner couples
    single_earner_couples_25_55 = df_married_25_55[df_married_25_55['secondary_earning'] == 0]
    dual_earner_couples_25_55 = df_married_25_55[df_married_25_55['secondary_earning'] > 0]
    single_earner_couples_25_55.to_csv(f'excel/{period}/single_earner_couples_25_55_{period}.csv', index=False)
    dual_earner_couples_25_55.to_csv(f'excel/{period}/dual_earner_couples_25_55_{period}.csv', index=False)

    # we create a dataframe only for singles, with positive earnings and adult
    df_celib = result_df[~result_df['maries_ou_pacses']]
    df_celib = df_celib.drop(['primary_earning', 'secondary_earning', 'primary_age', 'secondary_age', 'maries_ou_pacses'], axis = 1)
    df_celib = df_celib[df_celib['single_earning'] >= 0]
    df_celib = df_celib[df_celib['age'] >= 18]
    df_celib = df_celib.sort_values(by='single_earning')
    df_celib.to_csv(f'excel/{period}/single_adults_{period}.csv', index=False)

    df_celib_25_55 = df_celib[df_celib['age'] >= 25]
    df_celib_25_55 = df_celib_25_55[df_celib_25_55['age'] <= 25]
    df_celib_25_55 = df_celib_25_55.sort_values(by='single_earning')
    df_celib_25_55.to_csv(f'excel/{period}/singles_25_55_{period}.csv', index=False)
  
  
    
    

    



    






simulation_reforme()

