#%% BANC ESSAI : analyse du TSI par plage de puissance (<>50kW)
# configuration #
composant = 'roulement 1 génératrice'
data_full_T1 = data_full['brutes'][data_full['brutes']['ref_turbine_valorem']=='T1']
# séparation des données résiduelles selon la plage de puissance #
id_inf50 = data_full_T1[data_full_T1['puiss_active_produite'] < 5e4]['date_heure'].unique()
id_sup50 = data_full_T1[data_full_T1['puiss_active_produite'] >= 5e4]['date_heure'].unique()
TSI_inf50 = dict((composant, pd.DataFrame(index = TSI[composant].index, columns = v_turbine, data = np.full([len(TSI[composant]), len(v_turbine)], np.nan))) for composant in dct_models[parc]['seuils'].index)
TSI_sup50 = dict((composant, pd.DataFrame(index = TSI[composant].index, columns = v_turbine, data = np.full([len(TSI[composant]), len(v_turbine)], np.nan))) for composant in dct_models[parc]['seuils'].index)
#TSI_inf50 = pd.Series(index = TSI[composant]['T1'].index, data = [np.nan]*len(TSI[composant]))
#TSI_sup50 = TSI_inf50.copy()
TSI_inf50[composant]['T1'].loc[id_inf50] = TSI[composant]['T1'].loc[id_inf50]
TSI_sup50[composant]['T1'].loc[id_sup50] = TSI[composant]['T1'].loc[id_sup50]
#%% calcul des métriques par plage de puissance #
_, _, FP_inf50, FN_inf50 = f.balanced_accuracy(TSI_inf50, composant, 'T1', dct_models[parc]['seuils'], H0, H1)
_, _, FP_sup50, FN_sup50 = f.balanced_accuracy(TSI_sup50, composant, 'T1', dct_models[parc]['seuils'], H0, H1)
print()
print(f'TFN_inf50 = {round((FN_inf50/len(TSI_inf50[composant]["T1"][H1[0]:H1[1]].dropna()))*100, 2)}%')
print(f'TFP_inf50 = {round((FP_inf50/len(TSI_inf50[composant]["T1"][H0[0]:H0[1]].dropna()))*100, 2)}%')
print(f'TFN_sup50 = {round((FN_sup50/len(TSI_sup50[composant]["T1"][H1[0]:H1[1]].dropna()))*100, 2)}%')
print(f'TFP_sup50 = {round((FP_sup50/len(TSI_sup50[composant]["T1"][H0[0]:H0[1]].dropna()))*100, 2)}%')
#%% affichage #
plt.close('all')
plt.plot(TSI_inf50[composant]['T1'], 'r', label='P<50kW')
plt.plot(TSI_sup50[composant]['T1'], 'b', label='P>50kW')
plt.legend(), plt.grid()
plt.hlines(dct_models[parc]['seuils'].loc[composant, '99%'], xmin = TSI_inf50[composant].index[0], xmax = TSI_inf50[composant].index[-1], linestyles = 'dashed', colors='red')
plt.hlines(dct_models[parc]['seuils'].loc[composant, '1%'], xmin = TSI_inf50[composant].index[0], xmax = TSI_inf50[composant].index[-1], linestyles = 'dashed', colors='red')