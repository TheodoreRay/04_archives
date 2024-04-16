#%% LIBRAIRIES
import seaborn as sns
import fonctions_perso as f
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

#%% IMPORTS
plt.close('all')
data_2011, _ = f.Excel_to_DataFrame()
data_2011 = data_2011.iloc[:, 1:]
#%%
data_2012, _ = f.Excel_to_DataFrame()
data_2012 = data_2012.iloc[:, 1:]
#%%
data_2018, _ = f.Excel_to_DataFrame()
data_2018 = data_2018.iloc[:, 1:]

#%% CONFIGURATION
v = 'temp_roul_gene2'
modele = ['temp_roul_gene1', 'temp_stator', 'puiss_active_produite']
v_turbine = data_2011['ref_turbine_valorem'].unique()
data_in = [[[] for i in range(3)] for j in range(3)]
data_out = [[[] for i in range(3)] for j in range(3)]
data = [data_2011, data_2012, data_2018]
v_coef = [0 for i in range(len(v_turbine)*3*3)]; v_const = v_coef.copy()
RMSE_in = [0 for i in range(len(v_turbine)*3*3)]
RMSE_out = [0 for i in range(len(v_turbine)*3*3)]

#%% CALCUL DES PARAMETRES
ecart_in_out = []; moyenne_in_out = []
## périodes in/out ##
for i, annee in enumerate(['2011', '2012', '2018']):
    data_in[0][i], data_out[0][i] = f.une_saison(data = data[i], debut = f'{annee}-05-01', fin = f'{annee}-08-30', distinction = False) # estival
    data_in[1][i], data_out[1][i] = f.une_saison(data = data[i], debut = f'{annee}-11-20', fin = f'{annee}-02-28', distinction = True) # hivernal
    data_in[2][i], data_out[2][i] = f.un_mois_par_saison(data = data[i], premier_mois = 0) # un mois par saison
for annee in range(3):
    for i, tur in enumerate(v_turbine): 
        ## RMSE IN ##
        v_coef[3*i+3*6*annee], v_const[3*i+3*6*annee], RMSE_in[3*i+3*6*annee] = f.learning_cv(tur, data_in[0][annee], modele, v)
        v_coef[3*i+3*6*annee+1], v_const[3*i+3*6*annee+1], RMSE_in[3*i+3*6*annee+1] = f.learning_cv(tur, data_in[1][annee], modele, v)
        v_coef[3*i+3*6*annee+2], v_const[3*i+3*6*annee+2], RMSE_in[3*i+3*6*annee+2] = f.learning_cv(tur, data_in[2][annee], modele, v)
        ## RMSE OUT ##
        _, y_mes, y_pred, _ = f.residu_mono(tur, data_out[0][annee], modele, v, v_coef[3*i+3*6*annee], v_const[3*i+3*6*annee])
        RMSE_out[3*i+3*6*annee] = np.sqrt(mean_squared_error(y_mes, y_pred))
        _, y_mes, y_pred, _ = f.residu_mono(tur, data_out[1][annee], modele, v, v_coef[3*i+3*6*annee+1], v_const[3*i+3*6*annee+1])
        RMSE_out[3*i+3*6*annee+1] = np.sqrt(mean_squared_error(y_mes, y_pred))
        _, y_mes, y_pred, _ = f.residu_mono(tur, data_out[2][annee], modele, v, v_coef[3*i+3*6*annee+2], v_const[3*i+3*6*annee+2])
        RMSE_out[3*i+3*6*annee+2] = np.sqrt(mean_squared_error(y_mes, y_pred))
        ## ECART IN/OUT ##
        ecart_in_out.append(np.abs(RMSE_in[3*i+3*6*annee]-RMSE_out[3*i+3*6*annee]))
        ecart_in_out.append(np.abs(RMSE_in[3*i+3*6*annee+1]-RMSE_out[3*i+3*6*annee+1]))
        ecart_in_out.append(np.abs(RMSE_in[3*i+3*6*annee+2]-RMSE_out[3*i+3*6*annee+2]))
        ## MOYENNE IN/OUT ##
        moyenne_in_out.append(np.mean([RMSE_in[3*i+3*6*annee], RMSE_out[3*i+3*6*annee]]))
        moyenne_in_out.append(np.mean([RMSE_in[3*i+3*6*annee+1], RMSE_out[3*i+3*6*annee+1]]))
        moyenne_in_out.append(np.mean([RMSE_in[3*i+3*6*annee+2], RMSE_out[3*i+3*6*annee+2]]))

#%% AFFICHAGE - NUAGE DE POINTS
plt.close('all')

## MISE EN FORME DES DONNEES ##
periode = ['estival', 'hivernal', '1 mois par saison']*6*3
df_rmse = pd.DataFrame(dict(RMSE_in = RMSE_in, RMSE_out = RMSE_out, periode = periode))
df = pd.DataFrame(dict(ecart_in_out = ecart_in_out, moyenne_in_out = moyenne_in_out, periode = periode))

## RMSE IN (RMSE OUT)
g = sns.lmplot('RMSE_in', 'RMSE_out', data = df_rmse, hue = 'periode', fit_reg = False, height = 4)
g = g.set(xlim = (1,3.5), ylim = (1,3.5))
# comparaison y = x #
plt.plot(range(5))
plt.show(), plt.grid()

# ECART IN OUT (MOYENNE IN OUT)
sns.lmplot('ecart_in_out', 'moyenne_in_out', data = df, hue='periode', fit_reg = False, height = 6)
# zone de validation #
horizontale_x = [0, 0.5]; horizontale_y = [2, 2]
verticale_x = [0.5, 0.5]; verticale_y = [1, 2]
plt.plot(horizontale_x, horizontale_y, 'r'); plt.plot(verticale_x, verticale_y, 'r')
plt.xlim(0, 3), plt.ylim(1, 7), plt.show(), plt.grid()

#%% AFFICHAGE - DISTRIBUTION DONNEES BRUTES
var = 'puiss_active_produite'
for annee in range(3):
    plt.figure()
    for i, tur in enumerate(v_turbine):
        plt.subplot(3, 2, i+1)
        plt.title(tur)
        data_in[0][annee][data_in[0][annee]['ref_turbine_valorem'] == tur][var].plot.kde(label = 'estival') # estival
        data_in[1][annee][data_in[1][annee]['ref_turbine_valorem'] == tur][var].plot.kde(label = 'hivernal') # hivernal
        plt.legend(), plt.grid()
#%% résidu selon la vitesse de vent
data_in_esti, data_out_esti = f.une_saison(data = data_2011, debut = f'2011-05-01', fin = f'2011-08-30', distinction = False) # estival
v_coef_esti, v_const_esti, RMSE_in_esti = f.learning_cv(tur, data_in_esti, modele, v)
_, y_mes, y_pred, edp = f.residu_mono(tur, data_out_esti, modele, v, v_coef_esti, v_const_esti)
edp_df = pd.Series.to_frame(edp.iloc[:35075])
edp_df['vitesse_vent_nacelle'] = data_out[0][0][data_out[0][0]['ref_turbine_valorem']==tur]['vitesse_vent_nacelle'].values

#%% affichage
plt.close('all')
plt.subplot(221)
plt.plot(edp_df[edp_df['vitesse_vent_nacelle']<12].iloc[:,0])
plt.subplot(222)
edp_df[edp_df['vitesse_vent_nacelle']<12].iloc[:,0].plot.kde(label = 'vitesse faible')
print(edp_df[edp_df['vitesse_vent_nacelle']<12].iloc[:,0].std())
plt.subplot(223)
plt.plot(edp_df[edp_df['vitesse_vent_nacelle']>12].iloc[:,0])
plt.subplot(224)
edp_df[edp_df['vitesse_vent_nacelle']>12].iloc[:,0].plot.kde(label = 'vitesse forte')
print(edp_df[edp_df['vitesse_vent_nacelle']>12].iloc[:,0].std())