#%% LIBRAIRIES
from tkinter import N
from tkinter.constants import S
import functions_other as f
import functions_plot as plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import operator
import pyarrow as pa
from pyarrow import parquet
from importlib import reload
from sklearn.metrics import mean_absolute_error
from dateutil.rrule import rrule, HOURLY, DAILY, WEEKLY, MONTHLY

#%% ------------------ 0/ IMPORT ----------------------
#%% paramétrisation de l'import ##
reload(f)
## PARCS PRESENTANT UN DEFAUT ##
#v_parc = ['DOSNO', 'LAPLA', 'FOYE']
#v_modele_turbine = ['V90', 'MM92', 'MM92']
#v_début = ['01-01-2021', '04-01-2020', '07-15-2020'] # MM-JJ-AAAA
#v_fin = ['07-01-2022', '07-01-2022', '08-02-2022'] # MM-JJ-AAAA
## PARCS NE PRESENTANT PAS DE DEFAUT ##
v_parc = ['SOR', 'SOULA', 'SANTE', 'TEILL', 'SAINT', 'LESLA', 'CAST', 'ABLAI', 'AZERA'] #manque TERRE (dernière pos) et CORRO (6ème pos)
v_modele_turbine = ['ECO80', 'G87', 'G114', 'G97', 'G90', 'MM122', 'N80', 'N80', 'V100']
v_début = ['01-01-2021']*(11-2) # MM-JJ-AAAA
v_fin = ['07-01-2022']*(11-2) # MM-JJ-AAAA
provenance = "S2EV" # "S2EV" ou "turbiniers"
## import données SCADA 10min ##
meta_data, learning_data, data = f.import_data(v_parc = v_parc, v_modele_turbine = v_modele_turbine, v_début = v_début, v_fin = v_fin, provenance = provenance, is_learning_data = True)

#%% ------------------ 1/ DECLARATION DES VARIABLES & PREPARATION DES DONNEES  ----------------------
reload(f)
# données parc #
v_turbine = dict((parc, 0) for parc in v_parc)
v_composant = dict((parc, 0) for parc in v_parc)
# PARCS PRESENTANT UN DEFAUT : périodes H0, H1 #
#H0 = {v_parc[0]:['2021-01-15', '2022-05-12'], v_parc[1]:['2020-04-01', '2021-12-01'], v_parc[2]:['2020-07-15', '2021-07-15']} # AAAA-MM-JJ # spécificité (à développer pour plusieurs parcs)
#H1 = {v_parc[0]:['2022-05-12', '2022-07-01'], v_parc[1]:['2021-12-01', '2022-04-06'], v_parc[2]:['2021-07-15', '2022-07-01']} # AAAA-MM-JJ # sensibilité (à développer pour plusieurs parcs)
# PARCS NE PRESENTANT PAS DE DEFAUT : périodes H0, H1 #
H0 = {v_parc[i]:['2021-01-01', '2022-07-01'] for i in range(len(v_parc))} # AAAA-MM-JJ # spécificité (à développer pour plusieurs parcs)
H1 = {v_parc[i]:[] for i in range(len(v_parc))} # AAAA-MM-JJ # sensibilité (à développer pour plusieurs parcs)
# données 10min #
data_10min = dict((parc, 0) for parc in v_parc)
YX = dict((parc, dict((composant, pd.DataFrame()) for composant in v_composant)) for parc in v_parc)
# résidus #
TSI = dict((parc, 0) for parc in v_parc)
edp = dict((parc, 0) for parc in v_parc)
# performances #
perfs = dict((parc, pd.DataFrame(index=['specificite', 'sensibilite', 'balanced accuracy', 'MAME'])) for parc in v_parc)

for parc in v_parc:
    print(parc)
    ## VARIABLES META DU PARC ##
    v_turbine[parc] = meta_data[parc]['v_turbine']
    v_composant[parc] = list(learning_data[parc]['seuils'].dropna().index)
    
    ## Erreurs de prédiction & TSI ##
    edp[parc] = dict((composant, pd.DataFrame()) for composant in v_composant[parc])
    TSI[parc] = dict((composant, pd.DataFrame(columns = v_turbine[parc])) for composant in v_composant[parc])

    ## SEPARATION DES DONNEES ##
    data_10min[parc] = {'brutes':0, 'standardisées':0}
    data_10min[parc]['brutes'] = data[parc].reset_index().copy().sort_values(by = 'date_heure')
    data_10min[parc]['brutes'].drop_duplicates(keep = 'first', inplace = True)
    data_10min[parc]['standardisées'] = data_10min[parc]['brutes'].copy()
    
    ## STANDARDISATION ##
    for turbine in v_turbine[parc]:
        data_10min[parc]['standardisées'].loc[data_10min[parc]['standardisées']['ref_turbine_valorem'] == turbine, 'puiss_active_produite':] = \
            (data_10min[parc]['brutes'].loc[data_10min[parc]['brutes']['ref_turbine_valorem'] == turbine, 'puiss_active_produite':] - \
                learning_data[parc]['moyenne'][turbine])/learning_data[parc]['écart-type'][turbine]#"""
    
    # STOCKAGE DES VARIABLES #
    for composant in v_composant[parc]:
        YX[parc][composant] = data_10min[parc]['standardisées'][['ref_turbine_valorem', 'date_heure', learning_data[parc][composant].index.name] + list(learning_data[parc][composant].index[:3])]#"""

#%% -------------------- 2/ CONSTRUCTION DES TSI & MESURE DE PERFORMANCES -----------------------------
reload(f); reload(plot)
cmpsnt = 'roulement 1 génératrice' # composant pour lequel étudier les performances des TSI associés
for parc in v_parc:
    print(parc)
    for composant in v_composant[parc]:
        print(f'composant : {composant}')
        print(f'{learning_data[parc][composant].index.name} = f({list(learning_data[parc][composant].index[:3])})')
        
        ## calcul de l'indicateur ##
        _, _, edp[parc][composant], TSI[parc][composant] = f.residu_mono_multi(YX[parc][composant], list(learning_data[parc][composant].index[:3]), learning_data[parc][composant].index.name, learning_data[parc][composant].iloc[:3, :], v_turbine[parc])
        
        ## filtrage et recentrage ##
        for tur in v_turbine[parc]:
            # recentrage : stand-by (rentre dans la problématique du réapprentissage) #
            TSI[parc][composant][tur] = f.moving_average(TSI[parc][composant][tur]-learning_data[parc][composant].loc['moyenne', tur], 144)
        
        ## calcul des performances pour le composant sélectionné ##
        if composant == cmpsnt:
            for turbine in v_turbine[parc]:
                # Balanced accuracy #
                perfs[parc].loc['specificite', turbine], perfs[parc].loc['sensibilite', turbine], perfs[parc].loc['balanced accuracy', turbine] = f.balanced_accuracy(TSI[parc], cmpsnt, turbine, learning_data[parc]['seuils'], H0[parc], H1[parc])
                # MAME (Mean Absolute Median Error ?) #
                perfs[parc].loc['MAME', turbine] = mean_absolute_error(edp[parc][cmpsnt].loc[H0[parc][0]:H0[parc][1], turbine], edp[parc][cmpsnt].loc[H0[parc][0]:H0[parc][1], 'médiane'])#"""

        ## Prise en compte des indisponibilités ##
        TSI[parc][composant] = TSI[parc][composant].fillna(0)

#%% ------------------ 3/ EXPLOITATION ----------------------
## Configuration ##
reload(plot); reload(f); plt.close('all')
parc = 'LAPLA'

## Affichage des TSI##
#for parc in v_parc:
plot.indicateur_daily_mono(TSI = TSI[parc], v_turbine = ['T3'], v_composant = ['roulement 1 génératrice', 'roulement 2 génératrice'], seuil = learning_data[parc]['seuils'], mois = list(range(18)), annees = ['2021', '2022'], H1=H1[parc])

## Dispersions Statistique (DS) ##
#data_std, data_mean, lines, figs = plot.dispersion_statistique(TSI[parc], v_turbine[parc], ['refroidissement gearbox'], [1, 2, 4, 5, 7, 8, 10, 11], ['2021'], WEEKLY)

## Données brutes régresseurs + sortie ##
#plot.variables_modele(v_turbine = v_turbine[parc], data = data_10min[parc]['standardisées'], data_models = learning_data[parc], composant = 'refroidissement gearbox', mois = list(range(18)), annee = ['2021', '2022'], f = 144)
#plot.variables(v_turbine = v_turbine[parc], data = data_10min[parc]['standardisées'], v_variable = ['temp_huile_multi','temp_roul_multi1','temp_roul_multi2','pression_huile_multi'], mois = list(range(18)), annee = ['2021', '2022'], f = 144)
#plot.ecart_mediane(v_turbine = v_turbine[parc], turbines = v_turbine[parc], data = data_10min[parc]['brutes'], v_variable = ['temp_huile_multi','temp_roul_multi1', 'temp_roul_multi2', 'temp_boite_multi'], f = 144)

## Bilan d'alarmes ##
#data_classe_flotte = plot.heatmaps_alarmes_flotte(TSI, v_turbine, v_composant, v_parc, learning_parameters, [0, 1, 2], [annee], DAILY, False, False) # bool: affichage heatmap parc/turbine

#%% Export TSI
TSI_parquet = pa.Table.from_pandas(TSI[parc]['roulement 2 génératrice'])
writer_data_parquet = parquet.ParquetWriter(f'TSI_{parc}_{v_début[0]}_{v_fin[0]}.parquet', schema = TSI_parquet.schema)
writer_data_parquet.write_table(TSI_parquet)
if writer_data_parquet: writer_data_parquet.close()
