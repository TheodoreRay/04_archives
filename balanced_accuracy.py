#%% LIBRAIRIES
import functions_other as f
import functions_plot as plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib import reload
from dateutil.rrule import rrule, HOURLY, DAILY, WEEKLY, MONTHLY

#%% IMPORT (TSI + learning_data)
learning_data_SOR = pd.read_excel("../../../1_data/12_learning/ECO80_SOR_S2EV.xlsx", sheet_name = None)
learning_data_ABLAI = pd.read_excel("../../../1_data/12_learning/N80_ABLAI_S2EV.xlsx", sheet_name = None)
learning_data_PDRS = pd.read_excel("../../../1_data/12_learning/MM92_PDRS_S2EV.xlsx", sheet_name = None)
learning_data_FOYE = pd.read_excel("../../../1_data/12_learning/MM92_FOYE_S2EV.xlsx", sheet_name = None)

for key in learning_data_SOR:
    learning_data_SOR[key] = learning_data_SOR[key].set_index(learning_data_SOR[key].iloc[:, 0]).iloc[:, 1:]#"""
for key in learning_data_ABLAI:
    learning_data_ABLAI[key] = learning_data_ABLAI[key].set_index(learning_data_ABLAI[key].iloc[:, 0]).iloc[:, 1:]#"""
for key in learning_data_PDRS:
    learning_data_PDRS[key] = learning_data_PDRS[key].set_index(learning_data_PDRS[key].iloc[:, 0]).iloc[:, 1:]#"""
for key in learning_data_FOYE:
    learning_data_FOYE[key] = learning_data_FOYE[key].set_index(learning_data_FOYE[key].iloc[:, 0]).iloc[:, 1:]#"""

TSI_SOR = pd.read_parquet("TSI_SOR_01-13-2010_03-01-2011.parquet")
TSI_SOR.drop(TSI_SOR[TSI_SOR['T1'] == 0].index, inplace=True) # suppression des dates d'indisponibilité
TSI_ABLAI = pd.read_parquet("TSI_ABLAI_06-20-2020_08-12-2021.parquet")
TSI_ABLAI.drop(TSI_ABLAI[TSI_ABLAI['T2'] == 0].index, inplace=True) # suppression des dates d'indisponibilité
TSI_PDRS = pd.read_parquet("TSI_PDRS_02-06-2018_10-01-2019.parquet")
TSI_PDRS.drop(TSI_PDRS[TSI_PDRS['T1'] == 0].index, inplace=True) # suppression des dates d'indisponibilité
TSI_FOYE = pd.read_parquet("TSI_FOYE_07-15-2020_06-01-2022.parquet")
TSI_FOYE.drop(TSI_FOYE[TSI_FOYE['T3'] == 0].index, inplace=True) # suppression des dates d'indisponibilité

#%% CONFIGURATION DES CLASSIFICATIONS CIBLES
H0H1_theo_SOR = pd.Series(index = TSI_SOR.index, data = [0]*len(TSI_SOR), name = 'T1')
H0H1_theo_ABLAI = pd.Series(index = TSI_ABLAI[:'2021-06-15'].index, data = [0]*len(TSI_ABLAI[:'2021-06-15']), name = 'T2')
H0H1_theo_PDRS = pd.Series(index = TSI_PDRS.index, data = [0]*len(TSI_PDRS), name = 'T1')
H0H1_theo_FOYE = pd.Series(index = TSI_FOYE.index, data = [0]*len(TSI_FOYE), name = 'T3')

H0H1_theo_SOR['2010-01-26':'2010-03-01'] = 1 # ROULEMENT 2 GENERATRICE
H0H1_theo_ABLAI['2020-06-20':'2020-08-12'] = 1 # REFROIDISSEMENT CONVERTISSEUR
#H0H1_theo_ABLAI['2021-06-17':'2021-08-05'] = 1 # REFROIDISSEMENT CONVERTISSEUR
H0H1_theo_PDRS['2018-11-29':'2019-10-01'] = 1 # PALIER ARBRE LENT
H0H1_theo_FOYE['2021-07-15':'2022-06-01'] = 1 # REFROIDISSEMENT MULTIPLICATRICE

#%% GENERATION DES CLASSIFICATIONS EXPERIMENTALES
H0H1_expe_SOR = pd.Series(index = TSI_SOR.index, data = [0]*len(TSI_SOR), name = 'T1')
H0H1_expe_ABLAI = pd.Series(index = TSI_ABLAI[:'2021-06-15'].index, data = [0]*len(TSI_ABLAI[:'2021-06-15']), name = 'T2')
H0H1_expe_PDRS = pd.Series(index = TSI_PDRS.index, data = [0]*len(TSI_PDRS), name = 'T1')
H0H1_expe_FOYE = pd.Series(index = TSI_FOYE.index, data = [0]*len(TSI_FOYE), name = 'T3')

H0H1_expe_SOR.loc[(TSI_SOR['T1'] > learning_data_SOR['seuils'].loc['roulement 2 génératrice', '99%'])] = 1
H0H1_expe_ABLAI.loc[(TSI_ABLAI['T2'] > learning_data_ABLAI['seuils'].loc['cooling convertisseur', '99%'])] = 1
H0H1_expe_PDRS.loc[(np.abs(TSI_PDRS['T1']) > learning_data_PDRS['seuils'].loc['palier arbre lent', '99%'])] = 1
H0H1_expe_FOYE.loc[(TSI_FOYE['T3'] > learning_data_FOYE['seuils'].loc['refroidissement gearbox', '99%'])] = 1

#%% CALCUL DES BALANCED ACCURACY
results = pd.DataFrame(index = ['TP', 'FN', 'sensitivity', 'TN', 'FP', 'specificity', 'balanced accuracy'], columns = ['SOR', 'ABLAI', 'PDRS', 'FOYE'])

results.loc['TP', 'SOR'] = H0H1_expe_SOR.where((H0H1_expe_SOR == 1) & (H0H1_theo_SOR == 1)).dropna().count()
results.loc['FN', 'SOR'] = H0H1_expe_SOR.where((H0H1_expe_SOR == 0) & (H0H1_theo_SOR == 1)).dropna().count()
results.loc['sensitivity', 'SOR'] = results.loc['TP', 'SOR'] / (results.loc['TP', 'SOR'] + results.loc['FN', 'SOR'])
results.loc['TN', 'SOR'] = H0H1_expe_SOR.where((H0H1_expe_SOR == 0) & (H0H1_theo_SOR == 0)).dropna().count()
results.loc['FP', 'SOR'] = H0H1_expe_SOR.where((H0H1_expe_SOR == 1) & (H0H1_theo_SOR == 0)).dropna().count()
results.loc['specificity', 'SOR'] = results.loc['TN', 'SOR'] / (results.loc['TN', 'SOR'] + results.loc['FP', 'SOR'])
results.loc['balanced accuracy', 'SOR'] = (results.loc['sensitivity', 'SOR'] + results.loc['specificity', 'SOR']) / 2

results.loc['TP', 'ABLAI'] = H0H1_expe_ABLAI.where((H0H1_expe_ABLAI == 1) & (H0H1_theo_ABLAI == 1)).dropna().count()
results.loc['FN', 'ABLAI'] = H0H1_expe_ABLAI.where((H0H1_expe_ABLAI == 0) & (H0H1_theo_ABLAI == 1)).dropna().count()
results.loc['sensitivity', 'ABLAI'] = results.loc['TP', 'ABLAI'] / (results.loc['TP', 'ABLAI'] + results.loc['FN', 'ABLAI'])
results.loc['TN', 'ABLAI'] = H0H1_expe_ABLAI.where((H0H1_expe_ABLAI == 0) & (H0H1_theo_ABLAI == 0)).dropna().count()
results.loc['FP', 'ABLAI'] = H0H1_expe_ABLAI.where((H0H1_expe_ABLAI == 1) & (H0H1_theo_ABLAI == 0)).dropna().count()
results.loc['specificity', 'ABLAI'] = results.loc['TN', 'ABLAI'] / (results.loc['TN', 'ABLAI'] + results.loc['FP', 'ABLAI'])
results.loc['balanced accuracy', 'ABLAI'] = (results.loc['sensitivity', 'ABLAI'] + results.loc['specificity', 'ABLAI']) / 2

results.loc['TP', 'PDRS'] = H0H1_expe_PDRS.where((H0H1_expe_PDRS == 1) & (H0H1_theo_PDRS == 1)).dropna().count()
results.loc['FN', 'PDRS'] = H0H1_expe_PDRS.where((H0H1_expe_PDRS == 0) & (H0H1_theo_PDRS == 1)).dropna().count()
results.loc['sensitivity', 'PDRS'] = results.loc['TP', 'PDRS'] / (results.loc['TP', 'PDRS'] + results.loc['FN', 'PDRS'])
results.loc['TN', 'PDRS'] = H0H1_expe_PDRS.where((H0H1_expe_PDRS == 0) & (H0H1_theo_PDRS == 0)).dropna().count()
results.loc['FP', 'PDRS'] = H0H1_expe_PDRS.where((H0H1_expe_PDRS == 1) & (H0H1_theo_PDRS == 0)).dropna().count()
results.loc['specificity', 'PDRS'] = results.loc['TN', 'PDRS'] / (results.loc['TN', 'PDRS'] + results.loc['FP', 'PDRS'])
results.loc['balanced accuracy', 'PDRS'] = (results.loc['sensitivity', 'PDRS'] + results.loc['specificity', 'PDRS']) / 2

results.loc['TP', 'FOYE'] = H0H1_expe_FOYE.where((H0H1_expe_FOYE == 1) & (H0H1_theo_FOYE == 1)).dropna().count()
results.loc['FN', 'FOYE'] = H0H1_expe_FOYE.where((H0H1_expe_FOYE == 0) & (H0H1_theo_FOYE == 1)).dropna().count()
results.loc['sensitivity', 'FOYE'] = results.loc['TP', 'FOYE'] / (results.loc['TP', 'FOYE'] + results.loc['FN', 'FOYE'])
results.loc['TN', 'FOYE'] = H0H1_expe_FOYE.where((H0H1_expe_FOYE == 0) & (H0H1_theo_FOYE == 0)).dropna().count()
results.loc['FP', 'FOYE'] = H0H1_expe_FOYE.where((H0H1_expe_FOYE == 1) & (H0H1_theo_FOYE == 0)).dropna().count()
results.loc['specificity', 'FOYE'] = results.loc['TN', 'FOYE'] / (results.loc['TN', 'FOYE'] + results.loc['FP', 'FOYE'])
results.loc['balanced accuracy', 'FOYE'] = (results.loc['sensitivity', 'FOYE'] + results.loc['specificity', 'FOYE']) / 2

#%% COMPARAISON DES SCENARIOS
plt.close('all')

plt.figure('surchauffe génératrice SOR T1')
plt.subplot(211)
plt.plot(TSI_SOR['T1'], linewidth='1'), plt.ylim(-1, 3), plt.title('TSI'), plt.grid()
plt.subplot(212)
plt.plot(H0H1_theo_SOR, 'k', linewidth='4', zorder = 2,  label = 'H0H1_theo_SOR'), plt.legend(), plt.title('chronogrammes')
plt.plot(H0H1_expe_SOR, 'r--', linewidth='0.4', zorder = 1, label = 'H0H1_expe_SOR'), plt.legend()
plt.fill_between(x = H0H1_expe_SOR.index, y1 = H0H1_expe_SOR.values, color = "r")
plt.ylabel('classe'), plt.grid()

plt.figure('refroidissement convertisseur ABLAI T2')
plt.subplot(211)
plt.plot(TSI_ABLAI.loc[:'2021-06-15', 'T2'], linewidth='1'), plt.ylim(-1, 3), plt.title('TSI'), plt.grid()
plt.subplot(212)
plt.plot(H0H1_theo_ABLAI, 'k', linewidth='4', zorder = 2, label = 'H0H1_theo_ABLAI'), plt.legend(), plt.title('chronogrammes')
plt.plot(H0H1_expe_ABLAI, 'r--', linewidth='0.4', zorder = 1, label = 'H0H1_expe_ABLAI'), plt.legend()
plt.fill_between(x = H0H1_expe_ABLAI.index, y1 = H0H1_expe_ABLAI.values, color = "r")
plt.ylabel('classe'), plt.grid()

plt.figure('palier arbre lent PDRS T1')
plt.subplot(211)
plt.plot(TSI_PDRS['T1'], linewidth='1'), plt.ylim(-1, 3), plt.title('TSI'), plt.grid()
plt.subplot(212)
plt.plot(H0H1_theo_PDRS, 'k', linewidth='4', zorder = 2, label = 'H0H1_theo_PDRS'), plt.legend(), plt.title('chronogrammes')
plt.plot(H0H1_expe_PDRS, 'r--', linewidth='0.4', zorder = 1, label = 'H0H1_expe_PDRS'), plt.legend()
plt.fill_between(x = H0H1_expe_PDRS.index, y1 = H0H1_expe_PDRS.values, color = "r")
plt.ylabel('classe'), plt.grid()

plt.figure('refroidissement gearbox FOYE T3')
plt.subplot(211)
plt.plot(TSI_FOYE['T3'], linewidth='1'), plt.ylim(-1, 3), plt.title('TSI'), plt.grid()
plt.subplot(212)
plt.plot(H0H1_theo_FOYE, 'k', linewidth='4', zorder = 2, label = 'H0H1_theo_FOYE'), plt.legend(), plt.title('chronogrammes')
plt.plot(H0H1_expe_FOYE, 'r--', linewidth='0.4', zorder = 1, label = 'H0H1_expe_FOYE'), plt.legend()
plt.fill_between(x = H0H1_expe_FOYE.index, y1 = H0H1_expe_FOYE.values, color = "r")
plt.ylabel('classe'), plt.grid()
