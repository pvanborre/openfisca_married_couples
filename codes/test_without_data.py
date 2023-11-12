from openfisca_core.simulation_builder import SimulationBuilder
from openfisca_france import FranceTaxBenefitSystem

TEST_CASE = {
  "individus": {
    "Claude": {
      "salaire_de_base": {
        "2017": 20000
      }
    },
    "Dominique": {
      "salaire_de_base": {
        "2017": 30000
      }
    },
    "Camille": {
    }
  },
  "menages": {
    "menage_1": {
      "personne_de_reference": [
        "Claude"
      ],
      "conjoint": [
        "Dominique"
      ],
      "enfants": [
        "Camille"
      ]
    }
  },
  "familles": {
    "famille_1": {
      "parents": [
        "Claude",
        "Dominique"
      ],
      "enfants": [
        "Camille"
      ]
    }
  },
  "foyers_fiscaux": {
    "foyer_fiscal_1": {
      "declarants": [
        "Claude",
        "Dominique"
      ],
      "personnes_a_charge": [
        "Camille"
      ]
    }
  }
}





tax_benefit_system = FranceTaxBenefitSystem()

simulation_builder = SimulationBuilder()
simulation = simulation_builder.build_from_entities(tax_benefit_system, TEST_CASE)

irpp = simulation.calculate('impot_revenu_restant_a_payer', '2017')

print("irpp", irpp)