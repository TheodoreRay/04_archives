#%% DISTINCTION DES DONNEES SELON LE MODE DE PRODUCTION
df_learning_inf50 = data_learning['standardisées'][data_learning['brutes']['puiss_active_produite'] < 5e4]
df_learning_sup50 = data_learning['standardisées'][data_learning['brutes']['puiss_active_produite'] >= 5e4]
data_learning['standardisées'] = pd.concat(\
    [df_learning_inf50.sample(n=6192*6), \
        df_learning_sup50.sample(n=6192*6)])
df_learning_inf50 = data_learning['brutes'][data_learning['brutes']['puiss_active_produite'] < 5e4]
df_learning_sup50 = data_learning['brutes'][data_learning['brutes']['puiss_active_produite'] >= 5e4]
data_learning['brutes'] = pd.concat(\
    [df_learning_inf50.sample(n=6192*6), \
        df_learning_sup50.sample(n=6192*6)])

#%% CALCUL MAE PAR MODE DE PRODUCTION
composant = 'roulement 1 génératrice'

MAE_inf50 = []
MAE_sup50 = []

id_inf50 = data_full['brutes'][data_full['brutes']['date_heure']>H1[1]][data_full['brutes']['puiss_active_produite'] < 5e4]['date_heure'].unique()
id_sup50 = data_full['brutes'][data_full['brutes']['date_heure']>H1[1]][data_full['brutes']['puiss_active_produite'] >= 5e4]['date_heure'].unique()

for turbine in v_turbine:
    _,y_mes_inf50, y_est_inf50,_ = f.residu_mono(turbine, YX[composant][YX[composant]['date_heure'].isin(id_inf50)], list(learning_data[parc][composant].iloc[: 3].index), learning_data[parc][composant].index.name, learning_data[parc][composant].iloc[: 3][turbine])

    _,y_mes_sup50, y_est_sup50,_ = f.residu_mono(turbine, YX[composant][YX[composant]['date_heure'].isin(id_sup50)], list(learning_data[parc][composant].iloc[: 3].index), learning_data[parc][composant].index.name, learning_data[parc][composant].iloc[: 3][turbine])
    df_inf50 = pd.DataFrame(data = np.array([y_mes_inf50, y_est_inf50]).T, columns=['y_mes', 'y_est']).dropna()
    df_sup50 = pd.DataFrame(data = np.array([y_mes_sup50, y_est_sup50]).T, columns=['y_mes', 'y_est']).dropna()
    
    MAE_inf50.append(mean_absolute_error(df_inf50['y_mes'], df_inf50['y_est']))
    MAE_sup50.append(mean_absolute_error(df_sup50['y_mes'], df_sup50['y_est']))
   
print(np.median(MAE_inf50))
print(np.median(MAE_sup50))