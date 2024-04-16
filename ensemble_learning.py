#%% LIBRAIRIES
import functions_other as f
import functions_plot as plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib import reload
from dateutil.rrule import rrule, HOURLY, DAILY, WEEKLY, MONTH

#%% IMPORT data (apprentissage + validation)
DATA_SOR = pd.read_parquet("../../../1_data/11_scada_s2ev/parquet/ECO80_SOR_01-13-2010_01-01-2012.parquet")
reload(f)
## VARIABLES META DU PARC ##
v_turbine = DATA_SOR['ref_turbine_valorem'].unique()
annee = '2011'
mois = [0, 3, 6, 9]

## SEPARATION DES DONNEES ##
data = {'brutes':0, 'standardisées':0}
data_test = {'brutes':0, 'standardisées':0}
data_learning = {'brutes':0, 'standardisées':0}

data['brutes'] = DATA_SOR.copy().sort_values(by='date_heure')
data['brutes'].drop_duplicates(keep = 'first', inplace = True)
# sélection des données de test #
data_test['brutes'] = f.selection_data_mois(data = data['brutes'], annees = ['2010'], mois = list(range(12)))
# sélection des données d'apprentissage #
data_learning['brutes'] = f.selection_data_mois(data = data['brutes'], annees = [annee], mois = mois)
## STANDARDISATION ##
data['standardisées'] = data['brutes'].copy()
data_test['standardisées'] = data_test['brutes'].copy()
data_learning['standardisées'] = data_learning['brutes'].copy()
[data_learning, data, data_test], _, _ = f.std_scaler_manuel(data_learning['brutes'], [data_learning, data, data_test], v_turbine)

#%% GFS RECURSIF
reload(f)
modeles = []
V = list(data_learning['brutes'].columns[3: ])
V.remove('temp_roul_gene2')
while len(V) > 2:
    resultat, _ = f.gfs_main(data_learning = data_learning['standardisées'], sortie = 'temp_roul_gene2', v_turbine = v_turbine, L = 3, V = V)
    for v in resultat.index:
        V.remove(v)
    modeles.append(resultat)
modeles.append(V)