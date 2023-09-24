
import numpy
import pandas
import matplotlib.pyplot as plt

from openfisca_core.simulation_builder import SimulationBuilder
from openfisca_core.reforms import Reform

from openfisca_france import FranceTaxBenefitSystem
from openfisca_france.model.base import *


pandas.options.display.max_columns = None

annee = 2018

filename = "../data/{}/flat_{}.h5".format(annee, annee)
data_persons = pandas.read_hdf(filename, key = "input")#, stop =11)

 
print("Table des personnes")
print(data_persons, "\n\n\n\n\n")




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


        class irpp(Variable):
            value_type = float
            entity = FoyerFiscal
            label = "Impot sur le revenu réformé"
            definition_period = YEAR

            def formula(foyer_fiscal, period, parameters):

                iai = foyer_fiscal('iai', period)
                credits_impot = foyer_fiscal('credits_impot', period)
                acomptes_ir = foyer_fiscal('acomptes_ir', period)
                contribution_exceptionnelle_hauts_revenus = foyer_fiscal('contribution_exceptionnelle_hauts_revenus', period)
                P = parameters(period).impot_revenu.calcul_impot_revenu.recouvrement

                pre_result = iai - credits_impot - acomptes_ir + contribution_exceptionnelle_hauts_revenus

                
                primary_earning_maries_pacses = foyer_fiscal('primary_earning', period) 
                secondary_earning_maries_pacses = foyer_fiscal('secondary_earning', period) 
                

                tau_1 = 0.1 # comment bien choisir tau_1 ????
                tau_2 = - tau_1 * sum(primary_earning_maries_pacses)/sum(secondary_earning_maries_pacses) 
                # cette déf de tau_2 assure qu'on est à budget de l'Etat constant 
                # car avant on avait SUM(IRPP) et désormais on a SUM(IRPP) + tau_1 SUM(revenu_declarant_principal) + tau_2 SUM(revenu_conjoint)
                # on égalise les deux termes et on trouve l'expression demandée 

                print("tau_2 est", sum(primary_earning_maries_pacses)/sum(secondary_earning_maries_pacses) , "fois plus élevé que tau_1 en valeur absolue") 

                return (
                    (iai > P.seuil) * (
                        (pre_result < P.min)
                        * (pre_result > 0)
                        * iai
                        * 0
                        + ((pre_result <= 0) + (pre_result >= P.min))
                        * (- pre_result)
                        )
                    + (iai <= P.seuil) * (
                        (pre_result < 0)
                        * (-pre_result)
                        + (pre_result >= 0)
                        * 0
                        * iai
                        )
                    + tau_1 * primary_earning_maries_pacses/12 
                    + tau_2 * secondary_earning_maries_pacses/12 
                    )

        self.replace_variable(irpp)


        



tax_benefit_system = FranceTaxBenefitSystem()
tax_benefit_system_reforme = vers_individualisation(tax_benefit_system)




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




#####################################################
########### Simulation ##############################
#####################################################

simulation = initialiser_simulation(tax_benefit_system_reforme, data_persons)
simulation.trace = True #utile pour voir toutes les étapes de la simulation

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
  



total_taxes = simulation.calculate('irpp', period)
print(total_taxes)


maries_ou_pacses = simulation.calculate('maries_ou_pacses', period)

primary_earning_maries_pacses = simulation.calculate('primary_earning', period)
primary_earning_maries_pacses = primary_earning_maries_pacses[maries_ou_pacses]
print("revenu du déclarant principal", primary_earning_maries_pacses)

secondary_earning_maries_pacses = simulation.calculate('secondary_earning', period)
secondary_earning_maries_pacses = secondary_earning_maries_pacses[maries_ou_pacses]
print("revenu du conjoint", secondary_earning_maries_pacses)




#simulation.tracer.print_computation_log()


#####################################################
########### Reproduction graphe 14 ##################
# Titre graphique : Gagnants et perdants d'une réforme vers l'individualisation de l'impôt, parmi les couples mariés ou pacsés, en janvier de l'année considérée
#####################################################

# Statistiques descriptives
nombre_foyers_maries_pacses = len(primary_earning_maries_pacses)
print("Nombre de foyers fiscaux dont membres sont mariés ou pacsés", nombre_foyers_maries_pacses)
print("Proportion de foyers fiscaux dont membres mariés ou pacsés", nombre_foyers_maries_pacses/len(maries_ou_pacses))


# on enlève les outliers
condition = (primary_earning_maries_pacses >= 0) & (secondary_earning_maries_pacses >= 0)
primary_earning_maries_pacses = primary_earning_maries_pacses[condition]
secondary_earning_maries_pacses = secondary_earning_maries_pacses[condition]
print("Nombre d'outliers que l'on retire", nombre_foyers_maries_pacses - len(primary_earning_maries_pacses))

# TODO question (j'aimerais bien ici ajouter les poids wprm ici)
# serait facile à ajouter ici mais dans la def de tau2 dans la réforme serait plus compliqué car il n'existe pas de poids dans la simulation 
rapport = sum(primary_earning_maries_pacses)/sum(secondary_earning_maries_pacses)
print("rapport", rapport)

# nombre de gagnants
is_winner = secondary_earning_maries_pacses*rapport > primary_earning_maries_pacses
print("Nombre de gagnants", is_winner.sum())
pourcentage_gagnants = round(100*is_winner.sum()/len(primary_earning_maries_pacses))
print("Pourcentage de gagnants", pourcentage_gagnants)



plt.figure()
x = numpy.linspace(0, 600000, 4)
plt.plot(x, x, label = "1e bissectrice x -> x")
plt.plot(x, rapport*x, label = "Droite de séparation des foyers fiscaux perdants et gagnants")
plt.scatter(secondary_earning_maries_pacses, primary_earning_maries_pacses, s = 0.1, c = '#828282') 

plt.annotate(str(pourcentage_gagnants)+ " %", xy = (1000000, 100000), bbox = dict(boxstyle ="round", fc ="0.8"))
plt.grid()
plt.axis('equal')  

plt.xlabel('Revenu annuel du secondary earner du foyer fiscal')
plt.ylabel('Revenu annuel du primary earner du foyer fiscal')
plt.title("Gagnants et perdants d'une réforme vers l'individualisation de l'impôt, \n parmi les couples mariés ou pacsés français, en janvier {}".format(period))

plt.legend()
plt.show()
plt.savefig('../outputs/graphe_14_{}.png'.format(period))