#%% LIBRAIRIES
import fonctions_de_traitement as f
import fonctions_d_affichage as plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.interpolate import PchipInterpolator

#%%
#df = pd.read_parquet(f"../../../1_data/11_scada_S2EV/parquet/ECO80_SOR_01-13-2010_04-01-2012.parquet")#.iloc[:, 1:]
x = pd.read_parquet(f"../../../1_data/11_scada_S2EV/parquet/N131_HOMBL_01-01-2021_07-01-2022.parquet")#.iloc[:, 1:]
x = x[x['ref_turbine_valorem']=='T1']
x = x[(x['date_heure']>'2021-11-10') & (x['date_heure']<'2021-12-10')]['temp_exterieur']
x = pd.Series(data=x.iloc[3800:])
x = x.reset_index(drop=True)
learning_data = pd.read_excel(f"../../../1_data/12_learning/ECO80_S2EV.xlsx", sheet_name=None)

#%% Initialisation des signaux x, x_tild
x_tild = x.copy()#x[x['ref_turbine_valorem']=='T1']['temp_exterieur'].copy()
x_tild[x_tild.sample(frac=0.1).index] = np.nan
x_tild[160:180] = np.nan
x_tild[300:350] = np.nan
id_nan = x_tild[x_tild.isna()].index #index des valeurs imputées
id_notnan = x_tild.dropna().index #index des valeurs disponibles

#%% Imputation de données par interpolation
x_tild = x_tild.interpolate(method='pchip')
x_nan = pd.Series(data = [np.nan]*len(x_tild))
x_notnan = pd.Series(data = [np.nan]*len(x_tild))
x_nan.loc[id_nan] = x_tild.loc[id_nan]
x_notnan.loc[id_notnan] = x_tild.loc[id_notnan]

#%% import du modèle
H0 = ['2010-03-01', '2011-03-01', '2012-01-01'] # AAAA-MM-JJ # apprentissage, (learning (coeffs), test (MAME, spé))
H1 = ['2010-01-01', '2010-03-01'] # AAAA-MM-JJ # défaut
data_learning, data_test, data_full = main.data_layout(learning_data, df, H0, H1)

for key in learning_data:
    learning_data[key] = learning_data[key].set_index(learning_data[key].iloc[:, 0]).iloc[:, 1:]
learning_data, YX_learning, YX_test, YX_full = main.model_generation(learning_data, 'SOR', pd.Series(index=['temp_roul_gene2'], data=['roulement 2 génératrice']), [], ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'], data_learning, data_test, data_full)#list(s_composant.keys())

#%% Création du TSI + seuils
composant = 'roulement 2 génératrice'
v_turbine = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
# déclaration variables #
perfs = dict((composant, pd.DataFrame(index=['specificite', 'MAME', 'faux positifs'])) for composant in learning_data['seuils'].index)
edp_test = dict((composant, 0) for composant in learning_data['seuils'].index)
TSI_learning = dict((composant, pd.DataFrame(columns = v_turbine)) for composant in learning_data['seuils'].index)
TSI_test = dict((composant, pd.DataFrame(columns = v_turbine)) for composant in learning_data['seuils'].index)
TSI_full = dict((composant, pd.DataFrame(columns = v_turbine)) for composant in learning_data['seuils'].index)

# entrainement du modèle : tous les prédicteurs ensemble #
for tur in v_turbine:
    learning_data[composant][tur], _, _ = f.model_learning(tur, YX_learning[composant], list(learning_data[composant].index[: 3]), learning_data[composant].index.name, 'least squares', 3)

# calcul des indicateurs # edp_test : dataframe des edp mono + médiane
_, _, _, TSI_learning[composant] = f.residu_mono_multi(YX_learning[composant], list(learning_data[composant].iloc[: 3].index), learning_data[composant].index.name, learning_data[composant].iloc[: 3], v_turbine)
_, _, edp_test[composant], TSI_test[composant] = f.residu_mono_multi(YX_test[composant], list(learning_data[composant].iloc[: 3].index), learning_data[composant].index.name, learning_data[composant].iloc[: 3], v_turbine)
_, _, edp_test[composant], TSI_full[composant] = f.residu_mono_multi(YX_full[composant], list(learning_data[composant].iloc[: 3].index), learning_data[composant].index.name, learning_data[composant].iloc[: 3], v_turbine)

# (filtrage et )recentrage #
for tur in v_turbine:
    learning_data[composant].loc['moyenne', tur] = TSI_learning[composant][tur].mean()
    TSI_test[composant][tur] = f.moving_average(TSI_test[composant][tur]-TSI_learning[composant][tur].mean(), 144)
    TSI_learning[composant][tur] = f.moving_average(TSI_learning[composant][tur]-TSI_learning[composant][tur].mean(), 144)#"""
    TSI_full[composant][tur] = f.moving_average(TSI_full[composant][tur]-TSI_learning[composant][tur].mean(), 144)#"""

# Seuils de détection #
seuil = pd.Series([TSI_learning[composant].dropna()[tur].quantile(.99) for tur in v_turbine]).median()

#%%
TSI_nan = pd.Series(data = [np.nan]*len(x_tild))
TSI_notnan = pd.Series(data = [np.nan]*len(x_tild))
TSI_nan.loc[id_nan] = TSI_full.loc[id_nan]
TSI_notnan.loc[id_notnan] = TSI_full.loc[id_notnan]

#%% affichage
plt.close('all')
plt.subplot(211)
plt.plot(x, 'b.'), plt.grid()
plt.subplot(212)
plt.plot(x_notnan, 'b.'), plt.plot(x_nan, 'r.'), plt.grid()
#plt.hlines(seuil, xmin = min(TSI_full.index), xmax = max(TSI_full.index), linestyles = 'dashed', colors='red')
#plt.grid()
#print(np.abs(x.mean()-x_tild.mean()))
#print(np.abs(x.std()-x_tild.std()))

#%%
id_nan = TSI_full[composant]['T1'][H0[0]:H0[1]][TSI_full[composant]['T1'][H0[0]:H0[1]].isna()].index
id_notnan = TSI_full[composant]['T1'][H0[0]:H0[1]].dropna().index
TSI = TSI_full[composant]['T1'][H0[0]:H0[1]].copy()
TSI = TSI.astype(float).interpolate(method='polynomial', order=2)
print(np.abs(TSI.mean()-TSI_full[composant]['T1'][H0[0]:H0[1]].mean()))
print(np.abs(TSI.std()-TSI_full[composant]['T1'][H0[0]:H0[1]].std()))
TSI_nan = pd.Series(index = TSI.index, data = [np.nan]*len(TSI))
TSI_notnan = pd.Series(index = TSI.index, data = [np.nan]*len(TSI))
TSI_nan.loc[id_nan] = TSI.loc[id_nan]
TSI_notnan.loc[id_notnan] = TSI.loc[id_notnan]