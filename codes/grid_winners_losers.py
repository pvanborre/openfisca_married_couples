import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


year_pre, year_post = 2005, 2006
data_pre = pd.read_csv(f'./excel/{year_pre}/married_25_55_positive_{year_pre}.csv')
data_post = pd.read_csv(f'./excel/{year_post}/married_25_55_positive_{year_post}.csv')


# cf page 48 of the working paper : do each year separately, so will be done in utils_paper.py at the end 

data_pre['share_primary'] = 100 * data_pre['primary_earning']/(data_pre['primary_earning'] + data_pre['secondary_earning'])


print(data_pre)
"""
Relative marriage bonuses/penalties relate the absolute monetary advantage
from filing as a married couple to the total income of the couple, i.e. Tm(y1+y2)−(Ts(y1)+Ts(y2)) / y1+y2
"""

# page 100 paper 
# code reform that computes a primary tax burden that would be as if it was a single man (account for a single declaration)
# but how to allocate dependents ? we allocate them to one spouse only 
# counterfactual tax burden is an AVERAGE over two hypothetical tax burdens in which dependents are allocated to either one of the spouses.

# Daniel stata code 6D
# goal is to store a new column in data_pre['bonus'] = (data_pre['taux_marginal'] - TODO)/(data_pre['primary_earning'] + data_pre['secondary_earning'])

# compute info for these fake individuals : for the microsimulation model only need to change marital status for one, change marital status and set all dependents to 0 for the other 
# not that easy : better maybe to used dictionary form or at least look at the IRPP formula again on OF-France
# dico seems the best thing to do to separate individuals without dealing with id_foy etc 
# yes but as I considered at the beginning of the project it is not that easy though so ....
# other idea : do a reform where we change the formula for the IR


"""
Indiv
activite
age
categorie_salarie
categorie_non_salarie
chomage_brut
contrat_de_travail
date_naissance
effectif_entreprise
heures_remunerees_volume
pensions_alimentaires_percues
pensions_invalidite
rag
retraite_brute
ric
rnc
rpns_imposables
salaire_de_base
statut_marital
primes_fonction_publique
traitement_indiciaire_brut

Household
loyer
statut_occupation_logement
taxe_habitation
zone_apl
logement_conventionne



Computed
maries_ou_pacses = False
primary_earning 
secondary_earning 
"""
