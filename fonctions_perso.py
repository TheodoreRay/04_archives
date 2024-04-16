#%%
import pandas as pd
import os
import numpy as np
import matplotlib
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from tkinter import filedialog, Tk
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from math import *

#%% UTILITAIRES
host = "apps-bdd.valorem.com"
login = "VALEMO"
password = "VALEO&M"
bdd_path = "S2EV"

## ROUTINES DE REQUÊTE SQL ##
def DATA_10MIN_to_Excel(projet, date_debut, date_fin): #si parc complet : projet = "parc1, parc2"
    engine = create_engine(
        'firebird+fdb://{login}:{password}@{host}/{path}'.format(
            login = login, password = password, host = host, path = bdd_path))
    # Paramètres de la requête
    filtre_date = "AND DATE_HEURE >= '{debut}' AND DATE_HEURE < '{fin}'".format(debut=date_debut, fin=date_fin)
    frames = []
    # Importer données avec requête SQL
    if len(projet)>1: SQL_request = f"SELECT * FROM DATA_10MIN WHERE (NOM_PROJET='{projet[0]}' OR NOM_PROJET='{projet[1]}')\
            {filtre_date} ORDER BY DATE_HEURE ASC"
    else: SQL_request = f"SELECT * FROM DATA_10MIN WHERE NOM_PROJET='{projet[0]}'\
        {filtre_date} ORDER BY DATE_HEURE ASC"
    data = pd.read_sql_query(SQL_request, engine)
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # Mise sous forme dataFrame des données
    frames.append(data); result = pd.concat(frames)
    # Unification des références turbine
    if len(projet)>1: 
        L_0 = len(result[result["nom_projet"]==projet[0]]['ref_turbine_valorem'].unique())
        L_1 = len(result[result["nom_projet"]==projet[1]]['ref_turbine_valorem'].unique())
        for num in range(1, L_1+1):
            result.loc[(data['ref_turbine_valorem'] == f'T{num}') & (data['nom_projet'] == projet[1]), 'ref_turbine_valorem'] = f'T{L_0+num}'
        # Enregistrement .xlsx
        filename = f"D10M_{projet[0]}_{projet[1]}_{date_debut[:2]}-{date_debut[8:]}_{date_fin[:2]}-{date_fin[8:]}.xlsx"
        result.to_excel(filename) #"/" n'est pas un caractère autorisé
    else: 
        filename = f"D10M_{projet[0]}_{date_debut[:2]}-{date_debut[8:]}_{date_fin[:2]}-{date_fin[8:]}.xlsx"
        result.to_excel(filename) #"/" n'est pas un caractère autorisé
    return result, filename
def LISTE_ARRETS_to_Excel(projet, deb_arret, fin_arret):
    engine = create_engine(
        'firebird+fdb://{login}:{password}@{host}/{path}'.format(
            login = login,
            password = password,
            host = host,
            path = bdd_path
        )
    )
    # Paramètres de la requête
    filtre_date = "AND DEB_ARRET >= '{debut}' AND FIN_ARRET < '{fin}".format(debut=deb_arret, fin=fin_arret)
    frames = []
    # Importer données avec requête SQL
    SQL_request = "SELECT * FROM LISTE_ARRETS WHERE NOM_PROJET='{nom_projet} '{date}';".format(nom_projet=projet, date= filtre_date)
    data = pd.read_sql_query(SQL_request, engine)
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    frames.append(data)
    result = pd.concat(frames)
    filename = f"LA_{projet}_{deb_arret[:2]}-{deb_arret[8:]}_{fin_arret[:2]}-{fin_arret[8:]}.xlsx"
    result.to_excel(filename) #"/" n'est pas un caractère autorisé
    return result, filename
## IMPORT DES DONNEES ##
def Excel_to_DataFrame():
    # ouverture de la fenêtre d'intéraction
    root = Tk()
    root.destroy()
    root.mainloop()
    import_file_path = filedialog.askopenfilename()
    # obtenir le nom du fichier
    filename = os.path.basename(import_file_path)
    # création du DataFrame
    df = pd.read_excel(import_file_path)
    #df = SQLContext.read.csv("location", import_file_path)
    return df, filename # DataFrame contenant les données du tableau
## DISTINCTION DES JEUX DE DONNEES ##
def un_mois_par_saison(data, premier_mois):
    annee = str(data['date_heure'][0])[:4]
    mois = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '12-31']
    ## données d'entrainement (in_sample) et de validation (out_of_sample)
    data_in = data[(data['date_heure'] >= f'{annee}-{mois[0+premier_mois]}') & (data['date_heure'] < f'{annee}-{mois[1+premier_mois]}')
    | (data['date_heure'] >= f'{annee}-{mois[3+premier_mois]}') & (data['date_heure'] < f'{annee}-{mois[4+premier_mois]}')
    | (data['date_heure'] >= f'{annee}-{mois[6+premier_mois]}') & (data['date_heure'] < f'{annee}-{mois[7+premier_mois]}')
    | (data['date_heure'] >= f'{annee}-{mois[9+premier_mois]}') & (data['date_heure'] < f'{annee}-{mois[10+premier_mois]}')]#"""
    data_out = data.drop(index = data_in.index)
    return data_in, data_out
def une_saison(data, debut, fin, distinction):
    if distinction == False:
        data_in = data[(data['date_heure'] >= debut) & (data['date_heure'] <= fin)]
    if distinction == True:
        data_in = data[(data['date_heure'] >= debut) | (data['date_heure'] <= fin)]
    data_out = data.drop(index = data_in.index)
    return data_in, data_out

#%% DATA PROCESS
## STANDARDISATION ##
def std_scaler_manuel(data_train, data_appli, turbines): #standardisation de l'indicateur
    data_train_scale = data_train.copy()
    data_appli_scale = data_appli.copy()
    for tur in turbines:
        moy = data_train[tur].mean()
        std = data_train[tur].std()
        data_train_scale[tur] = (data_train_scale[tur] - moy)/std
        data_appli_scale[tur] = (data_appli_scale[tur] - moy)/std
    return data_train_scale, data_appli_scale
## FILTRE MOYEN ##
def moving_average(x, n) : #attention au format de x !
    x_avg = np.cumsum(x, dtype=float)
    x_avg[n:] = x_avg[n:] - x_avg[:-n]
    return x_avg[n - 1:] / n

## PERSISTANCE INDICATEUR + DATES DEPASSEMENT ##
def dates_depassement(edp, duree, seuil):
    v_turbine = list(edp.keys())
    date_dep = dict((v_turbine[i], 0) for i in range(len(v_turbine)))
    v_depassement = dict((v_turbine[i], 0) for i in range(len(v_turbine)))
    for tur in v_turbine:
        persistance = 0
        indice_dep = 0
        v_depassement[tur] = edp[tur][(edp[tur]>seuil[tur]) | (edp[tur]<-seuil[tur])].index
        while (persistance != f'{duree} days 00:00:00') and (indice_dep + duree*144<=len(v_depassement[tur])-1):
            persistance = str(v_depassement[tur][indice_dep + duree*144]-v_depassement[tur][indice_dep])
            indice_dep += 1
        if len(v_depassement[tur])>0:
            date_dep[tur] = v_depassement[tur][indice_dep]#"""
    return v_depassement, date_dep
def persistance(edp, turbine, periode, seuil):
    edp_defaut = edp[turbine][periode[0]:periode[1]]
    persistance = (edp_defaut.where(edp_defaut>seuil[turbine]).count())/edp_defaut.count()
    return persistance

#%% DATA LEARNING
##  APPRENTISSAGE ##
def learning(turbine, data, modele, v): # retourne les coefficients de l'équation d'évolution du modèle
    data = data[data['ref_turbine_valorem'] == turbine].copy()
    x_train = data[modele + [v]].dropna().drop(v, axis=1); y_train = data[modele + [v]].dropna()[v]
    if len(y_train) > 1:
        reg = Lasso().fit(x_train, y_train)
        coef = reg.coef_
        const = reg.intercept_
    else:
        coef = np.asarray([0 for x in range(len(modele))])
        const = 0
    return coef, const
def learning_cv(turbine, data, modele, v):
    K = 10; kf = KFold(n_splits = K)
    vRMSE = []; v_coef = [[] for x in range(K)]; v_const = [[] for x in range(K)]
    for x, (train, test) in enumerate(kf.split(data)):
        # établissement des coefficients de régression
        v_coef[x], v_const[x] = learning(turbine, data.iloc[train], modele, v)
        _, y_test_mes, y_test_pred, _ = residu_mono(turbine, data.iloc[test], modele, v, v_coef[x], v_const[x])
        # calcul du RMSE sur la période de test
        Y = pd.concat([y_test_mes, y_test_pred], axis=1).dropna()
        if len(y_test_mes) and len(y_test_pred) > 0:
            RMSE = np.sqrt(mean_squared_error(Y.iloc[:, 0], Y.iloc[:, 1]))
            vRMSE.append(RMSE)
        else: print(f'{modele[-1]} est indisponible pour {turbine}, RMSE non considéré')#; RMSE = 1000#"""
    coef = [np.median([v_coef[x][y] for x in range(len(v_coef))]) for y in range(len(v_coef[0]))]
    const = np.median(v_const)
    RMSE = np.median(vRMSE)
    return coef, const, RMSE
## RESIDUS ##
def residu_mono(turbine, data, modele, v, coef, const):
    t = data[data['ref_turbine_valorem'] == turbine][modele + [v] + ['date_heure']]['date_heure']
    data = data[data['ref_turbine_valorem'] == turbine][modele + [v]].copy()
    #mise en forme des donnees
    data = data[modele + [v]] # on conserve les valeurs NaN pour identifier les données indisponibles
    #data.loc[data[v] == 0, modele] = 0 #x_mes à 0 si y_mes est à 0
    x_mes = data[modele]; y_mes = data[v]
    if (len(x_mes) > 0 and len(y_mes) > 0) and (len(coef) == len(modele)): 
        y_pred = x_mes.dot(coef) + const
        edp_mono = pd.Series(data = y_mes.values - y_pred.values, index = t.values, name = turbine)
    else:
        y_pred = [0]
        edp_mono = pd.Series(data = [0], name = turbine)
    return t, y_mes, y_pred, edp_mono
def residu_mono_multi(data, modele, v, v_coef, v_const, v_turbine):
    # 0) configurations
    t = data.loc[data['ref_turbine_valorem'] == v_turbine[0], 'date_heure']
    edp_parc = pd.DataFrame(data = np.zeros((len(t), len(v_turbine))), index = t, columns = v_turbine)
    edp_final = dict((v_turbine[i], 0) for i in range(len(v_turbine)))
    edp_mono = pd.DataFrame(index = t, columns = v_turbine)
    Y_mes = dict((v_turbine[i], 0) for i in range(len(v_turbine)))
    Y_pred = dict((v_turbine[i], 0) for i in range(len(v_turbine)))
    # 1) residus mono
    for tur in v_turbine:
        _, Y_mes[tur], Y_pred[tur], edp = residu_mono(tur, data, modele, v, v_coef[tur], v_const[tur])
        edp_parc[tur][edp.index] = edp.values
    # 2) residu de reference parc : mediane des residus
    edp_ref = edp_parc.median(axis=1, skipna = True) #skipna=True : ignore les NaN
    for tur in v_turbine:
        edp_mono[tur] = edp_parc[tur].values
        # 3) residu final
        edp_final[tur] = pd.Series(data = edp_mono[tur] - edp_ref.values, index = edp_ref.index)
        edp_final[tur] = edp_final[tur].dropna()# seulement maintenant on peut supprimer les NaN !"""
    return Y_mes, Y_pred, edp_mono, edp_ref, edp_final

#%% SELECTION DE FEATURES
## FONCTIONS ASSOCIEES ##
def maj_du_modele(cp, resultat, variables, gain):
    if len(resultat.columns) == 0:
        ## standardisation ##
        cp['rmse stand'][len(resultat.columns)+1] = (cp['rmse'][len(resultat.columns)+1].mean() - cp['rmse'][len(resultat.columns)+1])/cp['rmse'][len(resultat.columns)+1].std()
        cp['tfa stand'][len(resultat.columns)+1] = (cp['tfa'][len(resultat.columns)+1].mean() - cp['tfa'][len(resultat.columns)+1])/cp['tfa'][len(resultat.columns)+1].std()
        ## calcul des gains ##
        #v_gain = pd.to_numeric(pd.Series(data = data_rmse_stand.iloc[:, len(resultat.columns)]))
        v_gain = pd.to_numeric(pd.Series(data = cp['tfa stand'][len(resultat.columns)+1]))
    elif len(resultat.columns) > 0:
        ## standardisation ##
        cp['rmse stand'][len(resultat.columns)+1] = (resultat.iloc[0, len(resultat.columns)-1] - cp['rmse'][len(resultat.columns)+1])/max(cp['rmse'][len(resultat.columns)+1].std(), 0.1) #max pour éviter %0
        cp['tfa stand'][len(resultat.columns)+1] = (resultat.iloc[1, len(resultat.columns)-1] - cp['tfa'][len(resultat.columns)+1])/max(cp['tfa'][len(resultat.columns)+1].std(), 0.001) #max pour éviter %0
        ## calcul des gains ##
        #rmse uniquement
        v_gain = pd.to_numeric(cp['rmse stand'].iloc[:, len(resultat.columns)])#"""
        """#tfa uniquement
        v_gain = pd.to_numeric(cp['tfa stand'].iloc[:, len(resultat.columns)])#"""
        """#rmse + tfa ##
        v_gain = pd.to_numeric(pd.Series(data = 0.5*(cp['rmse stand'].iloc[:, len(resultat.columns)] + cp['tfa stand'].iloc[:, len(resultat.columns)])))#"""
    # maj du modèle : gain max
    if v_gain.max() > 0:
        variable = v_gain.idxmax(); print(f'nouveau prédicteur : {variable}')
        resultat[variable] = [cp['rmse'].loc[variable, len(resultat.columns)+1], cp['tfa'].loc[variable, len(resultat.columns)+1]]
        variables.pop(variables.index(variable))
        print(f'modèle : {list(resultat.columns)} \n')
        # test d'arrêt de l'algorithme #
        if len(resultat.columns) > 1:
            gain = resultat.iloc[0, len(resultat.columns)-2] - resultat.loc['rmse', variable] # > 0 si apport
            print(f"gain = {gain}")
    else:
        print('max de performances atteint')
        gain = 0 # arrêt de l'algo
    return cp, resultat, variables, gain

def calcul_TFA(edp_H0): # calcul des performances du détecteur (en travaux)
    # taux de fausses alarmes
    FA = 0; TFA = 1; 
    seuil = 4.5 ## A REVOIR ##
    if type(edp_H0) != int:
        for x in edp_H0:
            if (x > seuil) or (x < -seuil): #1.5 : seuil objectif, à revoir un jour
                FA += 1
        TFA = FA/len(edp_H0)
    return TFA
def sous_optimalité(data, v_turbine, turbine, modele, v, modele_parametres, variables, approche): # vérification d'optimalité
    for i in range(1, len(modele)):
        modele_reduit = modele[:i] + modele[i+1:]
        print(modele_reduit)
        print(f'RMSE actuel : {modele_parametres["RMSE"]}')
        ## estimation du RMSE modèle réduit
        if approche == 'turbine':
            _, _, RMSE = learning_cv(turbine, data, modele_reduit, v)
            print(f'RMSE modele reduit : {RMSE}\n')
        elif approche == 'parc':
            v_RMSE = pd.Series(index = v_turbine)
            for tur in v_turbine: 
                _, _, v_RMSE[tur] = learning_cv(tur, data, modele_reduit, v)
            RMSE = np.median(v_RMSE)
            print(f'RMSE modele reduit : {RMSE}\n')
        ## comparaison avec le modèle actuel    
        if RMSE - modele_parametres['RMSE'] < 0:
            print(f'sous-optimalité détectée, {modele[i]} retirée du modèle \n') 
            variables.append(modele[i]); modele.pop(i)
    return variables, modele
# recherche et ajout du prochain meilleur prédicteur au modèle
def prochain_predicteur(variables, v_turbine, turbine, data, v, cp, resultat, approche, gain):
    #count = 0 #compte le nb de variables n'apportant pas de gain de performances #ne sert plus car les variables inutiles sont supprimées automatiquement
    modele = list(resultat.columns); TFA = 1
    for var in variables:
        modele.append(var); print(modele[-1])
        ## données du parc ##
        if approche == 'parc':
            ## estimation RMSE
            v_RMSE = pd.Series(index = v_turbine)
            v_coef = dict((v_turbine[i], 0) for i in range(len(v_turbine))); v_const = v_coef.copy()
            for tur in v_turbine: #partie longue, chercher un moyen d'optimiser
                v_coef[tur], v_const[tur], v_RMSE[tur] = learning_cv(tur, data, modele, v) #estimation plus précise du RMSE
                print(f'RMSE_in_median {tur} = {v_RMSE[tur]}')
            RMSE = np.median(v_RMSE)
            print(f'RMSE_in_median parc = {RMSE} \n')#"""
            ## estimation TFA
            v_TFA = pd.Series(index = v_turbine)
            for tur in v_turbine:
                _, _, _, edp_NBM = residu_mono(tur, data, modele, v, v_coef[tur], v_const[tur])
                v_TFA[tur] = calcul_TFA(edp_NBM)
            TFA = np.median(v_TFA)#"""
        ## données d'une turbine' ##
        elif approche == 'turbine':
            ## estimation RMSE
            coef, const, RMSE = learning_cv(turbine, data, modele, v) #estimation plus précise du RMSE ?
            ## estimation TFA
            _, _, _, edp_NBM = residu_mono(turbine, data, modele, v, coef, const)
            TFA = calcul_TFA(edp_NBM)#"""
        # enregistrement des performances
        cp['rmse'].loc[var, len(modele)] = RMSE
        cp['tfa'].loc[var, len(modele)] = TFA
        #print(f'{var} RMSE = {RMSE}, TFA = {TFA}')
        modele.pop()
    # 2) maj du modèle
    print('\n MISE A JOUR DU MODELE')
    print('--------------------------------')
    cp, resultat, variables, gain = maj_du_modele(cp, resultat, variables, gain)
    print(resultat)
    return variables, cp, resultat, gain
# affichage dee l'évolution des performances (RMSE, TFA) par étape
def affichage_perf(perf, modele):
    plt.figure()
    plt.subplot(211); plt.grid()
    plt.plot(perf['rmse'])
    plt.ylabel('RMSE')
    plt.xticks(ticks = range(len(modele[1:])), labels = modele[1:], color='w')
    plt.yticks(ticks = np.linspace(perf.iloc[0, 0], perf['rmse'][-1], 10).round(4), labels = np.linspace(perf['rmse'][0], perf['rmse'][-1], 10).round(4))
    
    plt.subplot(212); plt.grid()
    y = [perf['tfa'][x]*100 for x in range(len(perf['tfa']))]; plt.plot(y)
    plt.xlabel('taille du modèle'), plt.ylabel('taux de fausses alarmes (%)')
    plt.xticks(ticks = range(len(modele[1:])), labels = modele[1:], rotation = 15, ha='right')
    plt.yticks(ticks = np.linspace(y[0], y[-1], 10).round(4), labels = np.linspace(y[0], y[-1], 10).round(4))
## MAIN : ALGORITHME GFS COMPLET ##
def gfs_main(data, v, v_turbine, turbine, approche, tolerance):
    ### initialisation des variables ###
    resultat = pd.DataFrame(index = ['rmse', 'tfa']) # performances (affichage)
    # supprimer colonne unnamed: 0
    variables = data.columns[2:].tolist(); variables.remove(v) # variables à tester
    ## évolution des performances
    cp_rmse = pd.DataFrame(index = variables)
    cp_rmse_stand = pd.DataFrame(index = variables)
    cp_tfa = pd.DataFrame(index = variables)
    cp_tfa_stand = pd.DataFrame(index = variables)
    cp = {'rmse':cp_rmse, 'rmse stand':cp_rmse_stand, 'tfa':cp_tfa, 'tfa stand':cp_tfa_stand}
    gain = 10 #temporaire, pour arrêt automatique de l'algorithme
    ### construction itérative du modèle optimal ###
    #while len(resultat.columns) < c:
    while gain > tolerance:
        # recherche de la prochaine meilleure feature
        print(f'étape {len(resultat.columns)}')
        variables, cp, resultat, gain = prochain_predicteur(variables, v_turbine, turbine, data, v, cp, resultat, approche, gain)
        ## à revoir : sous optimalité ##
        #if len(modele[1:]) > 1: variables, modele = sous_optimalité(data, vturbine, turbine, modele, v, modele_parametres, variables, approche)
    #resultat = resultat.drop(columns = resultat.columns[len(resultat.columns)-1])
    ## à revoir : affichage perfs ##
    #affichage_perf(resultat, modele)
    return resultat, cp

#%% PLOT
##  DONNEES BRUTES ##
# configuration
def config(font_size, t):
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : font_size}
    matplotlib.rc('font', **font)
    labels = [str(t.iloc[i])[:-9] for i in range(0, len(t), int(len(t)/20))] # 20 dates
    plt.xticks(ticks = t.iloc[::int(len(t)/20)], labels = labels, rotation = 15, ha = 'right')
def annotations(y, dates, textes):
    for x in range(len(dates)):
        plt.annotate(textes[x],
            xy=(dates[x], y[dates[x]]), xycoords='data',
            xytext=(dates[x], y[dates[x]]-y.std()), textcoords='data',
            arrowprops=dict(arrowstyle="fancy", color='black', connectionstyle="arc3"), 
            bbox=dict(boxstyle="round", fc="0.8"),
            ha="center", va="center", size=15
            )
# scatter plot parc + R²
def plot_couple(v_turbine, data, x, y, titre):
    _, ax = plt.subplots()
    for turbine in v_turbine:
        data_red = data[data['ref_turbine_valorem'] == turbine][[x]+[y]].fillna(0)
        ax.scatter(data_red[x], data_red[y], label = f'{turbine} R²={round(scipy.stats.pearsonr(data_red[x], data_red[y])[0], 3)}')
    ax.set(xlabel = x, ylabel = y, title = titre); ax.grid(); ax.legend()
# affichage superposé (turbines) des valeurs d'une variable
def plot_monovariable(turbine_plot, data, variable, titre, filtrage):
    plt.figure(titre); f = 1
    t = data[data['ref_turbine_valorem'] == turbine_plot[0]][[variable]+['date_heure']].fillna(0)['date_heure']
    for turbine in turbine_plot:
        if filtrage: f=144
        y = moving_average(data[data['ref_turbine_valorem']==turbine][variable].fillna(0).values, f)
        plt.plot(t[f-1:], y, label=turbine)
    plt.title(variable)
    config(t, 'dates', '°C')
# subplot par variable des valeurs superposées (turbines)
def plot_multivariable(turbine, data, variables, titre, filtrage):
    plt.figure(titre); f = 1
    t = data[data['ref_turbine_valorem'] == turbine]['date_heure']
    for var in variables:
        if filtrage: f=144
        plt.plot(t[f-1:], moving_average(data[data['ref_turbine_valorem'] == turbine][var].fillna(0).values, f), label = var)
    config(12, t); plt.legend(); plt.grid()
# subplot par variable des valeurs superposées de la variable et de sa valeur médiane parc
def plot_ecart_mediane(v_turbine, turbine, data, variables, titre, filtrage):
    f = 1
    t = data[data['ref_turbine_valorem'] == v_turbine[0]]['date_heure']
    Y = pd.DataFrame(data = np.zeros((len(t), len(v_turbine))), index = t, columns = v_turbine)
    plt.figure(titre)
    for i, var in enumerate(variables):
        # LOCALISATION DU PLOT
        ax = plt.subplot(len(variables), 1, i+1); plt.title(var)
        # EXTRACTION DES DONNEES A AFFICHER
        for tur in v_turbine:
            y = data[data['ref_turbine_valorem'] == tur][var].fillna(0).values
            Y[tur] = y
            if tur == turbine: y_input = Y[tur]
        # CALCUL DE LA REFERENCE : médiane des résidus disponibles à l'instant "temps"
        y_ref = Y.median(axis=1)
        # AFFICHAGE
        if filtrage: 
            f=144
            y_ref.iloc[f-1:] = moving_average(y_ref.values, f) 
            y_input.iloc[f-1:] = moving_average(y_input.values, f) 
            plt.plot(t[f-1:], y_ref.iloc[f-1:], label = 'mediane parc')
            plt.plot(t[f-1:], y_input.iloc[f-1:], label = turbine)
        else:
            plt.plot(t[f-1:], y_ref.values, label = 'mediane parc')
            plt.plot(t[f-1:], y_input, label = turbine)
        if var == variables[-1]:ax.axes.xaxis.set_visible(True)
        else: ax.axes.xaxis.set_visible(False)
        config(12, t)
        plt.legend(); plt.grid()
    return t, y_input

# boxplots
def boxplots_group(data, id):
    plt.figure(id)
    data_group = pd.DataFrame()
    for col in data.columns:
        if id in col:
            Q1 = (data[col].min() + data[col].median()) / 2
            Q3 = (data[col].max() + data[col].median()) / 2
            ratio_Q1 = data[data[col] < Q1].count()[col] / len(data[col])
            ratio_Q3 = data[data[col] > Q3].count()[col] / len(data[col])
            print(col)
            print(f'% données < Q1 = {ratio_Q1}')
            print(f'% données > Q3 = {ratio_Q3} \n')
            data_group = pd.concat([data_group, data[col]], axis = 1)
    data_group.boxplot()
    plt.title(id)
    plt.xticks(ticks = range(1, len(data_group.columns)+1), labels = data_group.columns, rotation=35, ha='right')
    return data_group.columns

# fonction de densité
def plot_fonction_de_densité(data, v_turbine, variables, titre):
    plt.figure(titre)
    for i, var in enumerate(variables):
        for tur in v_turbine:
            ax = plt.subplot(len(variables), 1, i+1); plt.title(var)
            y = pd.Series(data = data[data['ref_turbine_valorem'] == tur][var].dropna().values)
            y.plot.kde(label = tur)
        plt.legend(), plt.grid() 
        plt.xticks(ticks = np.arange(0, 140, 5), labels = np.arange(0, 140, 5), rotation = 35, ha = 'right')

## POUR DIAGNOSTIC ##
def plot_indicateur(sorties, v_turbine, edp, seuil):
    plt.figure('indicateurs')

    if len(v_turbine) == 1:
        for i, s in enumerate(sorties):
            y = edp[v_turbine[0]][s].values; y = np.array(y, dtype=float)
            x = edp[v_turbine[0]][s].index
            plt.plot(x, y, label = s)
            plt.fill_between(x, y, where=(y > seuil[v_turbine[0]][s]) | (y < -seuil[v_turbine[0]][s]), color = 'r')
        config(12, pd.Series(list(edp[v_turbine[0]][sorties[0]].index)))
        
        plt.ylabel(v_turbine[0])
        plt.legend(); plt.grid()

    elif len(v_turbine) > 1:
        for i, s in enumerate(sorties):
            plt.subplot(len(sorties), 1, i+1)
            for tur in v_turbine:
                if len(edp[tur][s].dropna()) > 0: # indicateur existant
                    y = edp[tur][s].values; y = np.array(y, dtype=float)
                    x = edp[tur][s].index
                    plt.plot(x, y, label = tur)
                    plt.fill_between(x, y, where=(y > seuil[tur][s]) | (y < -seuil[tur][s]), color = 'r')
            config(12, pd.Series(list(edp[v_turbine[0]][s].index)))
            plt.ylim(-1, 21) # à revoir
            plt.ylabel(s); plt.grid()
        plt.legend()
def plot_heatmap_indicateur(sorties, v_turbine, edp, seuil):
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 12}
    matplotlib.rc('font', **font)
    for tur in v_turbine:
        plt.figure(tur)
        ##configuration##
        interval = 144*7
        col = [str(edp[tur].index[i*interval]) for i in range(int(len(edp[tur])/interval))]
        df_heatmap_persistance = pd.DataFrame(index = sorties, columns = col)
        df_heatmap_std = pd.DataFrame(index = sorties, columns = col)
        ##remplissage des données Heatmap##
        for i, c in enumerate(df_heatmap_persistance.columns): # 1 colonne = 1 semaine
            for s in sorties:
                ## moyenne hebdomadaire ##
                #df_heatmap.loc[s, c] = np.abs(edp[tur][s][i*interval : (i+1)*interval].mean())
                ## persistance hebdomadaire ##
                df_heatmap_persistance.loc[s, c] = (edp[tur][s].where(edp[tur][s][i*interval : (i+1)*interval] > seuil[tur][s]).count())/edp[tur][s][i*interval : (i+1)*interval].count()
                ## écart-type hebodmadaire ##
                df_heatmap_std.loc[s, c] = np.abs(edp[tur][s][i*interval : (i+1)*interval].std())
        df_heatmap_std = df_heatmap_std.astype(float); df_heatmap_persistance = df_heatmap_persistance.astype(float)
        ##affichage##
        plt.figure(tur)
        ax = plt.subplot(211)
        sns.heatmap(df_heatmap_persistance, cmap = 'BuPu', vmin = 0, vmax = 1, linewidths = .1, linecolor = 'black', annot = False, fmt=".0f", annot_kws={'size':8})
        labels = [str(edp[tur].index[i])[:-9] for i in range(0, len(edp[tur].index), 144*7)]
        plt.title('persistance')
        plt.xticks(ticks = range(len(edp[tur].index[::144*7])), labels = labels, rotation = 15, ha = 'right'); 
        plt.yticks(rotation = 15, ha = 'right'); plt.xlabel('semaine'); plt.ylabel('indicateur')
        ax.axes.xaxis.set_visible(False)
        ax = plt.subplot(212)
        sns.heatmap(df_heatmap_std, cmap = 'Greens', vmin = 0.2, vmax = 2, linewidths = .1, linecolor = 'black', annot = False, fmt=".0f", annot_kws={'size':8})
        labels = [str(edp[tur].index[i])[:-9] for i in range(0, len(edp[tur].index), 144*7)]
        plt.title('écart-type')
        plt.xticks(ticks = range(len(edp[tur].index[::144*7])), labels = labels, rotation = 15, ha = 'right'); 
        plt.yticks(rotation = 15, ha = 'right'); plt.xlabel('semaine'); plt.ylabel('indicateur')
        ax.axes.xaxis.set_visible(True)
    return df_heatmap_persistance, df_heatmap_std
def disponibilite_indicateurs(data, label_sortie, label_predicteurs, v_turbine): #pour modele à 3 variables pour l'instant
    #modele = ['temp_roul_gene2', 'temp_roul_gene1', 'temp_stator', 'puiss_active_produite']
    dispo = dict((v_turbine[i], 0) for i in range(len(v_turbine)))
    t = dict((v_turbine[i], 0) for i in range(len(v_turbine)))
    predicteurs = dict((label_predicteurs[i], 0) for i in range(len(label_predicteurs)))

    for tur in v_turbine:   
        t[tur] = data[data['ref_turbine_valorem']==tur]['date_heure']
        sortie = data[data['ref_turbine_valorem']==tur][label_sortie]
        sortie.loc[~sortie.isnull()] = 1; sortie.loc[sortie.isnull()] = 0
        for x in label_predicteurs:
            predicteurs[x] = data[data['ref_turbine_valorem']==tur][x]
            predicteurs[x].loc[~predicteurs[x].isnull()] = 1; predicteurs[x].loc[predicteurs[x].isnull()] = 0
        dispo[tur] = pd.Series(index = sortie.index, data=np.ones(len(sortie)))
        dispo[tur].loc[(sortie == 0)  | (predicteurs[label_predicteurs[0]] == 0) | (predicteurs[label_predicteurs[1]] == 0)] = 0
        #dispo[tur].loc[(sortie == 0) | (predicteurs == 0)] = 0 # à vérifier
    
    for i, tur in enumerate(v_turbine):
        plt.subplot(6, 1, i+1)
        plt.step(t[tur], dispo[tur]); plt.grid(); plt.ylabel(tur)
    return dispo

## A REVOIR (supprimer ?) ##
def plot_indicateur_desalignement(liste, data, element, typ): 
    #typ = turbine, liste=variables, element=turbine concernée 
    #typ = parc, liste=turbines, element=variable concernée
    f = 144
    titre = []; dist = []
    v_turbine = data['ref_turbine_valorem'].unique()
    t = data[data['ref_turbine_valorem'] == v_turbine[0]]['date_heure'][f-1:]
    for x in range(len(liste)-1):
        for y in range(x+1, len(liste)):
            titre.append(f'distance {liste[x]}/{liste[y]}')
            print(titre[-1])
            if typ == 'turbine':
                dist.append(np.abs(moving_average(data[data['ref_turbine_valorem']==element][liste[x]].values, f) - moving_average(data[data['ref_turbine_valorem']==element][liste[y]].values, f)))
            if typ == 'parc':
                dist.append(np.abs(moving_average(data[data['ref_turbine_valorem']==liste[x]][element].values, f) - moving_average(data[data['ref_turbine_valorem']==liste[y]][element].values, f)))
    #AFFICHAGE
    std = 30 #à revoir
    plt.figure(f'écart simple {element}')
    for i in range(len(dist)):
        plt.subplot(len(dist), 1, i+1), plt.plot(t, dist[i]), plt.grid(), plt.title(titre[i])
        plt.ylim(0, 600)
    plt.figure(f'indicateur désalignement {element}')
    for i in range(len(dist)):
        plt.subplot(len(dist), 1, i+1)
        plot_std(t, dist[i], titre[i], std)
        plt.ylim(0, 300)
def plot_std(t, y, titre, seuil): # pour indicateur désalignement pales 
    v_std = []
    for j in range(7*144, len(y), 144): v_std.append(np.std(y[j-7*144:j]))
    v_std = pd.Series(index = t[7*144::144], data = v_std)
    plt.plot(v_std)
    plt.fill_between(v_std.index, v_std.values, where= np.asarray(v_std) > seuil, color = 'r')
    plt.title(titre), plt.grid()
    return v_std

# stem plot coefficients de corrélations
def plot_cross_corr(data, v):
    data = data.drop(columns = ['ref_turbine_valorem', 'date_heure']).copy()
    ## PEARSON ##
    crosscorr = data.corr(method = "pearson")[v].drop(index = v)
    crosscorr = np.abs(crosscorr.to_list())
    ## LASSO ##
    X_train = data.drop(columns = [v]).iloc[:int(0.8*len(data)), :].fillna(0)
    y_train = data[v].iloc[:int(0.8*len(data))].fillna(0)
    reg_ls = LinearRegression(); reg_ls.fit(X_train, y_train); coef_ls = np.abs(reg_ls.coef_)
    for c in range(len(coef_ls)): coef_ls[c] = min(1, coef_ls[c])
    reg_lasso = Lasso(alpha = 0.1); reg_lasso.fit(X_train, y_train); coef_lasso = np.abs(reg_lasso.coef_)
    reg_ridge = Ridge(); reg_ridge.fit(X_train, y_train); coef_ridge = np.abs(reg_ridge.coef_)
    for c in range(len(coef_ridge)): coef_ridge[c] = min(1, coef_ridge[c])
    reg_elasticnet = ElasticNet(); reg_elasticnet.fit(X_train, y_train); coef_elasticnet = np.abs(reg_elasticnet.coef_)
    # affichage
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 6}
    matplotlib.rc('font', **font)
    plt.bar(range(len(data.columns.drop(v))), crosscorr, label = 'R²')
    plt.bar(range(len(data.columns.drop(v))), coef_ls, bottom = crosscorr, label = 'least square')
    plt.bar(range(len(data.columns.drop(v))), coef_lasso, bottom = crosscorr+coef_ls, label = 'lasso')
    plt.bar(range(len(data.columns.drop(v))), coef_ridge, bottom = crosscorr+coef_ls+coef_lasso, label = 'ridge')
    plt.bar(range(len(data.columns.drop(v))), coef_elasticnet, bottom = crosscorr+coef_ls+coef_lasso+coef_ridge, label = 'elasticnet')
    plt.xticks(ticks = range(len(data.columns.drop(v))), labels = data.columns.drop(v), rotation=15, ha='right')
    plt.legend(), plt.grid()
    #return {'ls':coef_ls, 'lasso':coef_lasso, 'ridge':coef_ridge, 'elasticnet':coef_elasticnet}
def plot_rmse_tfa(data_rmse, data_tfa):
    for etape in range(len(data_rmse.columns)):
        # scatter plot
        plt.figure(f'étape {etape+1} scatter')
        plt.scatter(data_rmse.iloc[:, etape], data_tfa.iloc[:, etape])
        plt.xlabel('rmse'), plt.ylabel('tfa'), plt.grid(), plt.ylim(-3.5, 3.5), plt.xlim(-3.5, 3.5)
        # bar plot
        plt.figure(f'étape {etape+1} bar')
        plt.bar(range(len(data_rmse.index)), data_rmse.iloc[:, etape], label = 'rmse')
        plt.bar(range(len(data_rmse.index)), data_tfa.iloc[:, etape], bottom = data_rmse.iloc[:, etape], color = 'g', label='tfa')
        plt.xticks(ticks = range(len(data_rmse.index)), labels = data_rmse.index, rotation=15, ha = 'right')
        plt.grid(), plt.legend()