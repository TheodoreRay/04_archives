#%%
import f_utilitaires as tools
import matplotlib.pyplot as plt
import f_data_process as process
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
# %% IMPORT DES DONNEES / MISE EN FORME
turbine = 'T1'
modele = ['temp_roul_gene2', 'temp_roul_gene1', 'temp_stator', 'puiss_active_produite', 'temp_transfo']
data, filename = tools.Excel_to_DataFrame()
data = data[data['ref_turbine_valorem'] == turbine].iloc[:, 1:].sort_values(by = 'date_heure')
x_train = data[modele].dropna().drop('temp_roul_gene2', axis=1); y_train = data[modele].dropna()['temp_roul_gene2']
#%%
data_appli, _ = tools.Excel_to_DataFrame()
data_appli = data_appli[data_appli['ref_turbine_valorem'] == turbine].iloc[:, 1:].sort_values(by = 'date_heure')
x_appli = data_appli[modele].dropna().drop('temp_roul_gene2', axis=1); y_appli = data_appli[modele].dropna()['temp_roul_gene2']

#%% APPRENTISSAGE
#SR = SymbolicRegressor(population_size = 5000,
#                           generations = 20, stopping_criteria = 0.1,
#                           p_crossover = 0.7, p_subtree_mutation = 0.1,
#                           p_hoist_mutation = 0.05, p_point_mutation = 0.1,
#                           max_samples = 0.9, verbose=1,
#                           parsimony_coefficient = 0.01, random_state = 0)
#reg_SR = SR.fit(x_train, y_train) #symbolic regressor
#y_pred_SR = reg_SR.predict(x_test)
#score_SR = reg_SR.score(x_test, y_test)
#RMSE_SR = np.sqrt(mean_squared_error(y_test, y_pred_SR))

function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']
ST = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=2,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=3)
reg_ST = ST.fit(x_train, y_train) #symbolic transformer
reg_Lasso = Lasso(max_iter = 1e5).fit(x_train, y_train) # lasso regressor
#%% PREDICTION
# prédiction sans features non linéaires
y_pred_Lasso = reg_Lasso.predict(x_appli)
score_Lasso = reg_Lasso.score(x_appli, y_appli)
RMSE_Lasso = np.sqrt(mean_squared_error(y_appli, y_pred_Lasso))

# ajout de nouvelles features non linéaires
full_data = pd.concat([data, data_appli])
ST_features = ST.transform(full_data[modele].dropna().drop('temp_roul_gene2', axis=1))
new_x_train = np.hstack((x_train, ST_features[:len(x_train)]))
new_x_appli = np.hstack((x_appli, ST_features[len(x_train):]))

reg_Lasso = Lasso(max_iter = 1e5).fit(new_x_train, y_train)
new_y_pred_Lasso = reg_Lasso.predict(new_x_appli)
new_score_Lasso = reg_Lasso.score(new_x_appli, y_appli)
new_RMSE_Lasso = np.sqrt(mean_squared_error(y_appli, new_y_pred_Lasso))
#%% AFFICHAGE
plt.close('all')
edp_Lasso_NL = process.moving_average(y_appli.values - new_y_pred_Lasso, 144)
edp_Lasso = process.moving_average(y_appli.values - y_pred_Lasso, 144)

plt.plot(edp_Lasso_NL, label = 'Lasso regressor avec features non linéaires')
plt.plot(edp_Lasso, label = 'Lasso regressor')

plt.legend()
plt.grid()