
# %% EXPLOITATION
# df doit être déjà filtré sur la turbine + colonnes inutiles supprimées
def inputs_selection(data, v, methode, reduction):
    data = data.dropna()
    y = data[v]
    x = data.drop(columns=[v, 'date_heure'])
    
    # INITIALISATION DES MODELES
    if methode == 'lasso':
        model = Lasso(max_iter = 10000, normalize = True)
        modelcv = LassoCV(cv = 10, max_iter = 100000, normalize = True)    
    elif methode == 'ridge':
        model = Ridge(normalize = True)
        modelcv = RidgeCV(cv = 10, normalize = True)    
    elif methode == 'elasticnet':
        model = ElasticNet(max_iter = 10000, normalize = True)
        modelcv = ElasticNetCV(cv = 10, max_iter = 100000, normalize = True) 
        
    # ENTRAINEMENT
    modelcv.fit(x, y)
    model.set_params(alpha=modelcv.alpha_)
    model.fit(x, y)
    coeff = pd.Series(model.coef_, index=x.columns, name=methode)
    coeff = ((coeff-coeff.min())/(coeff.max()-coeff.min())) #normalisation
    if reduction==True:
        var_set = coeff.sort_values(axis=0, ascending=False)[:4].index.tolist()
    else:
        var_set = coeff.sort_values(axis=0, ascending=False).index.tolist()
    return var_set, coeff

# learning_cv avec data_norm
def learning_cv(turbine, data_norm, data, modele, v):
    K = 10
    kf = KFold(n_splits = K)
    vRMSE = []; v_coef = [[] for x in range(K)]; v_const = [[] for x in range(K)]
    rc_test = [[] for x in range(K)]; rc_train= [[] for x in range(K)]
    # CONCATENER LES y_mes y_pred (plis)
    for x, (train, test) in enumerate(kf.split(data)):
        # établissement des coefficients de régression (données normalisées)
        v_coef[x], v_const[x] = learning(turbine, data_norm.iloc[train], modele, v)
        # dénormalisation des données
        data_test = mean_unnorm(data_norm.iloc[test], data.iloc[test])
        data_train = mean_unnorm(data_norm.iloc[train], data.iloc[train])
        _, y_test_mes, y_test_pred, _ = residu_mono(turbine, data_test, modele, v, v_coef[x], v_const[x])
        _, y_train_mes, y_train_pred, _ = residu_mono(turbine, data_train, modele, v, v_coef[x], v_const[x])
        # calcul des R² (1ère étude de généralisabilité)
        rc_test[x] = y_test_mes.corr(y_test_pred)
        rc_train[x] = y_train_mes.corr(y_train_pred)
        # calcul du RMSE sur la période de test
        if len(y_test_mes) > 0: RMSE = np.sqrt(mean_squared_error(y_test_mes, y_test_pred))
        else: RMSE = 1000; print('une variable est indisponible')
        vRMSE.append(RMSE)
    coef = [np.mean([v_coef[x][y] for x in range(len(v_coef))]) for y in range(len(v_coef[0]))]
    const = np.mean(v_const)
    RMSE = np.median(vRMSE)
    return coef, const, RMSE, rc_test, rc_train

#%% à revoir : étude des modes / interquartiles
#oob : out of bounds (en dehors des bornes)
"""plt.close('all'); turbine = 'T3'
data_mode = data_c.mode(numeric_only = True)

#établissement des bornes
bornes = []
for col in data_c.columns[3:]: #filtre sur puiss_active_produite déjà fait
    mini = data_c[data_c['ref_turbine_valorem'] == turbine][col].min()
    maxi = data_c[data_c['ref_turbine_valorem'] == turbine][col].max()
    med = data_c[data_c['ref_turbine_valorem'] == turbine][col].median()
    Q1 = (med+mini)/2; Q3 = (maxi+med)/2
    LB = Q1-1.5*(Q3-Q1); UB = Q3+1.5*(Q3-Q1)
    bornes.append([LB, UB])
data_bounds = pd.DataFrame(data = bornes, index = data_c.columns[3:], columns = ['lower bound', 'upper bound'])

#études des valeurs oob
nb_outliers = []
for col in data_c.columns[3:]:
    data_lb = data_c[col][data_c[col] < data_bounds.loc[col, 'lower bound']]; nb_data_lb = len(data_lb)
    data_ub = data_c[col][data_c[col] > data_bounds.loc[col, 'upper bound']]; nb_data_ub = len(data_ub)
    data_oob = pd.DataFrame(data = np.transpose(np.asarray([data_lb.values.tolist() + data_ub.values.tolist()]))) 
    #correspondance mode oob / mode signal
    if len(data_oob) > 0 :
        if data_oob.mode()[0][0] == data_mode[col][0]:
            print(f"{data_oob.mode()[0][0]} est le mode indisponible de {col}")
            filtres_FM[col] = data_oob.mode()[0][0]     
    nb_outliers.append(nb_data_lb+nb_data_ub)
    
data_outliers = pd.DataFrame(data = nb_outliers, index = data_c.columns[3:])
plt.plot(data_outliers.index, data_outliers.values), plt.ylim(0, max(data_outliers.values)), plt.grid()
plt.xticks(rotation=35, ticks=range(len(data_outliers.index)), labels = data_outliers.index, ha='right'), plt.xlim(0, len(data_outliers.index))#"""

#%%
def learning_cv(turbine, data, modele, v):
    K = 10; kf = KFold(n_splits = K)
    vRMSE = []; v_coef = [[] for x in range(K)]; v_const = [[] for x in range(K)]
    rc_test = [[] for x in range(K)]; rc_train= [[] for x in range(K)]
    # CONCATENER LES y_mes y_pred (plis)
    for x, (train, test) in enumerate(kf.split(data)):
        # établissement des coefficients de régression
        v_coef[x], v_const[x] = learning(turbine, data.iloc[train], modele, v)
        _, y_test_mes, y_test_pred, _ = residu_mono(turbine, data.iloc[test], modele, v, v_coef[x], v_const[x])
        _, y_train_mes, y_train_pred, _ = residu_mono(turbine, data.iloc[train], modele, v, v_coef[x], v_const[x])
        # calcul du RMSE sur la période de test
        if len(y_test_mes) and len(y_test_pred) > 0: 
            RMSE = np.sqrt(mean_squared_error(y_test_mes, y_test_pred))
            # calcul des R² (1ère étude de généralisabilité)
            rc_test[x] = y_test_mes.corr(y_test_pred)
            rc_train[x] = y_train_mes.corr(y_train_pred)
        else: RMSE = 1000; print('une variable est indisponible')
        vRMSE.append(RMSE)
    coef = [np.mean([v_coef[x][y] for x in range(len(v_coef))]) for y in range(len(v_coef[0]))]
    const = np.mean(v_const)
    RMSE = np.median(vRMSE)
    return coef, const, RMSE, rc_test, rc_train

#%% IMPORT DES DONNEES DE CONSTRUCTION DU NBM
v = 'temp_roul_gene2'
c = 5
K = 10
turbine = 'T2'

#%% FONCTIONS LOCALES
def importer():
    data, filename = tools.Excel_to_DataFrame()
    data = data.iloc[:, 1:].sort_values(by = 'date_heure')
    vturbine = data['ref_turbine_valorem'].unique()
    return vturbine, data, filename

def constructeur(data, vturbine, modele_ref, vcoef, vconst, v):
    edp = pd.DataFrame(columns = vturbine)
    constructeur = pd.DataFrame(index = vturbine, columns = modele_ref[1:]+['constante']+['seuil'])

    for tur in constructeur.index:
        # coefficients de régression
        for i, variable in enumerate(modele_ref[1:]):
            constructeur.loc[tur, variable] = vcoef[tur][i]
        constructeur.loc[tur, 'constante'] = vconst[tur]
        # erreur de prédiction
        _, _, _, edp = process.residu_mono_multi(tur, data, modele_ref, v, vcoef, vconst, vturbine)
        edp[tur] = edp.values

    edp.to_excel(f"EDP_SOR_{v}.xlsx")
    
    # enregistrement du fichier constructeur
    constructeur.to_excel(f"CONSTRUCTEUR_SOR_{v}.xlsx") #nom parc à adapter
    return constructeur

#%% MAIN
#%% import des données
vturbine, data, filename = importer()
#%% vecteurs de coefficients de régression
vcoef = pd.Series(data=[[1] for i in range(len(vturbine))], index = vturbine)
vconst = pd.Series(data=[[1] for i in range(len(vturbine))], index = vturbine)
#%% établissement du NBM
#_, _, modele_ref, rmse, gain_variables = features.GFS(data, v, turbine, c)
#%% calcul des coefficients de régression
for tur in vturbine: _, _, vcoef[tur], vconst[tur], vrmse = process.learning_cv(tur, data, modele_ref, v, K)
#%% création du fichier constructeur (modèle unique)
#constructeur = constructeur(data, vturbine, modele_ref, vcoef, vconst, v)

#%% (temporaire) étude distribution du résidu sur data_train
mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

data_edp_mono = pd.DataFrame(columns = vturbine)
data_edp_mono_multi = pd.DataFrame(columns = vturbine)
for tur in vturbine:
    _, _, _, edp = process.residu_mono(tur, data, modele_ref, v, vcoef[tur], vconst[tur])
    data_edp_mono[tur] = edp[:3000].values
    _, _, _, edp = process.residu_mono_multi(tur, data, modele_ref, v, vcoef, vconst, vturbine)
    data_edp_mono_multi[tur] = edp[:3000]
    print(3*edp[:3000].std())
    
plt.close('all')
data_edp_mono.plot.kde()
plt.grid(); plt.plot(x, stats.norm.pdf(x, mu, sigma), 'o-r', label='distribution normale'); plt.title('distribution edp mono-turbine')
plt.xlim(-10,10), plt.ylim(0,3), plt.legend()
data_edp_mono_multi.plot.kde()
plt.grid(); plt.plot(x, stats.norm.pdf(x, mu, sigma), 'o-r', label='distribution normale'); plt.title('distribution edp mono-multi-turbine')
plt.xlim(-10,10), plt.ylim(0,3), plt.legend()

def meilleur_predicteur(variables, modele, turbine, data, data_fault, v, modele_parametres, data_rmse, data_tfa, perf, seuil):
    gain = modele_parametres['gain'] #gain optimal complexité courante
    for i, var in enumerate(variables):
        modele.append(var)
        # estimation du RMSE
        coef, const, RMSE = process.learning_cv(turbine, data, modele, v) #estimation plus précise du RMSE 
        
        # calcul des performances AVD, TFA
        # REVOIR MONO OU MONO MULTI
        _, _, _, edp_NBM = process.residu_mono(turbine, data, modele, v, coef, const)
        
        temps = data[data['ref_turbine_valorem'] == turbine]['date_heure']
        AVD, TFA = perf_detecteur(temps, 0, edp_NBM, seuil)

        # enregistrement du gain de performances
        if len(modele) == 2: data_rmse.loc[var, len(modele)-1] = RMSE
        else: data_rmse.loc[var, len(modele)-1] = RMSE
            
        # sélection conditionnelle de variable
        modele_parametres = selection_de_feature(perf, AVD, TFA, RMSE, modele_parametres, gain, coef, const, var, i, variables)
        modele.pop()

    # MAJ DU MODELE OPTIMAL
    if len(variables) > 0:
        perf['rmse'].append(modele_parametres['RMSE']); perf['avd'].append(modele_parametres['AVD']); perf['tfa'].append(modele_parametres['TFA'])
        modele.append(variables[modele_parametres['indice']]); variables.pop(modele_parametres['indice'])
        print(f'{modele} \n')
    else: print('maximum de performances atteint')

    return variables, modele, modele_parametres, data_rmse, perf

#%%
def meilleur_predicteur_turbine(variables, modele, turbine, data, data_fault, v, modele_parametres, data_rmse, data_rmse_stand, data_tfa, data_tfa_stand, data_gain, perf):
    gain = modele_parametres['gain'] #gain optimal complexité courante
    # 1) calcul des performances
    print('CALCUL DES PERFORMANCES RMSE/TFA')
    print('--------------------------------')
    for i, var in enumerate(variables):
        modele.append(var)
        # estimation du RMSE
        coef, const, RMSE = process.learning_cv(turbine, data, modele, v) #estimation plus précise du RMSE 
        
        # calcul des performances AVD, TFA
        # REVOIR MONO OU MONO MULTI
        _, _, _, edp_NBM = process.residu_mono(turbine, data, modele, v, coef, const)
        
        #temps = data[data['ref_turbine_valorem'] == turbine]['date_heure']
        TFA = calcul_TFA(edp_NBM)

        # enregistrement des performances
        data_rmse.loc[var, len(modele)-1] = RMSE
        data_tfa.loc[var, len(modele)-1] = TFA
        print(f'{var} RMSE = {RMSE}, TFA = {TFA}')
        #data_gain.loc[var, len(modele)-1] = 0.5*(0.1/TFA + 1.5/RMSE) 

        # sélection conditionnelle de variable
        #modele_parametres = selection_de_feature(perf, AVD, TFA, RMSE, modele_parametres, gain, coef, const, var, i, variables)
        modele.pop()
    
    # 2) maj du modèle
    print('\n MISE A JOUR DU MODELE')
    print('--------------------------------')
    modele, modele_parametres, data_rmse, data_rmse_stand, data_tfa, data_tfa_stand, perf, variables = maj_du_modele(modele, modele_parametres, data_rmse, data_rmse_stand, data_tfa, data_tfa_stand, perf, variables)
        
    """# MAJ DU MODELE OPTIMAL
    if len(variables) > 0:
        perf['rmse'].append(modele_parametres['RMSE']); perf['tfa'].append(modele_parametres['TFA'])
        modele.append(variables[modele_parametres['indice']]); variables.pop(modele_parametres['indice'])
        print(f'{modele} \n')
    else: print('maximum de performances atteint')#"""

    return variables, modele, modele_parametres, data_rmse, data_rmse_stand, data_tfa, data_tfa_stand, data_gain, perf

#%% vérification du critère de sélection
def selection_de_feature(perf, AVD, TFA, RMSE, modele_parametres, gain, coef, const, var, i, variables):
    if len(coef) > 1:
        #if 1.5/RMSE > modele_parametres['gain']:
        if 0.1/TFA > modele_parametres['gain']:    
        #if 0.5*(0.1/TFA + 1.5/RMSE) > modele_parametres['gain']:# and RMSE <= perf['rmse'][-1] and TFA <= perf['tfa'][-1]:
            print(f'{var} apporte un meilleur gain')
            modele_parametres['indice'] = i
            modele_parametres['RMSE'] = RMSE; modele_parametres['TFA'] = TFA
            #modele_parametres['gain'] = 1.5/RMSE
            modele_parametres['gain'] = 0.1/TFA
            #modele_parametres['gain'] = 0.5*(0.1/TFA + 1.5/RMSE)
            print(f'gain {var} = {modele_parametres["gain"]}')
    else:
        if RMSE - modele_parametres['RMSE'] <= 0: # critère principal : RMSE
            print(f'RMSE {var} = {RMSE}')
            modele_parametres['indice'] = i
            modele_parametres['RMSE'] = RMSE; modele_parametres['TFA'] = TFA
            #modele_parametres['gain'] = 1.5/RMSE
            modele_parametres['gain'] = 0.1/TFA
            #modele_parametres['gain'] = 0.5*(0.1/TFA + 1.5/RMSE)
            print(f'gain {var} = {modele_parametres["gain"]}')
    return modele_parametres

def EDP(turbine_plot, vturbine, data, modele, vcoef, vconst, v, typ, filtrage): #adapter format de vcoef et vconst en fonction de typ / #turbine_plot : turbines à afficher
    f = 144; ymax = 0; ymin = 0
    edp = dict((turbine_plot[i], 0) for i in range(len(turbine_plot)))
    for tur in turbine_plot:
        # construction de l'edp
        if typ == 'mono-multi':
            t, _, _, edp[tur] = residu_mono_multi(tur, data, modele, v, vcoef, vconst, vturbine)
            if filtrage: t = t[f-1:]; edp[tur] = moving_average(edp[tur].values, f)
        if typ == 'mono':
            t, _, _, edp[tur] = residu_mono(tur, data, modele, v, vcoef[tur], vconst[tur])
            if filtrage: t = t[f-1:]; edp[tur] = moving_average(edp[tur].values, f)
        # update de ymax/ymin
        if len(edp[tur])>0: 
            if max(edp[tur]) > ymax : ymax = max(edp[tur])
            if min(edp[tur]) < ymin : ymin = min(edp[tur])
        # affichage
        plt.plot(t, edp[tur], label = tur)
        plt.xticks(ticks = t[::int(len(t)/20)], labels = t[::int(len(t)/20)], rotation=35, ha='right')
    # paramètres d'affichage
    #plt.ylim(-8, 8)
    plt.legend(); plt.grid(), plt.title(v)
    return t, edp

def residu_mono_multi(turbine, data, modele, v, vcoef, vconst, vturbine):
    # 0) configurations
    t = data[data['ref_turbine_valorem'] == turbine][modele + [v] + ['date_heure']].fillna(0)['date_heure']#.dt.strftime('%m-j%d-%Hh')
    edp_parc = [[] for i in range(len(vturbine))] # residus monos du parc

    # 1) residus mono
    for i, tur in enumerate(vturbine):
        if tur == turbine: #on enregistre y_mes et y_pred
            _, _, _, edp_parc[i] = residu_mono(tur, data, modele, v, vcoef[tur], vconst[tur])
        else:   
            _, _, _, edp_parc[i] = residu_mono(tur, data, modele, v, vcoef[tur], vconst[tur])
        print(len(edp_parc[i].index))
    edp_mono = edp_parc[vturbine.tolist().index(turbine)].values
    
    # 2) residu de reference parc : mediane des residus disponibles à l'instant "temps"
    edp_ref = []; edp_temps = []
    print('reference début')
    for temps in t.values:
        edp_parc_t = [edp_parc[tur][temps] for tur in range(len(edp_parc)) if temps in edp_parc[tur].index]
        if len(edp_parc_t)==0: edp_parc_t = [0]
        else: edp_temps.append(temps)
        edp_ref.append(np.median(edp_parc_t))
    print('reference fin')
    # 3) residu final
    edp = edp_mono - edp_ref
    edp = pd.Series(index = edp_temps, data = edp)
    return t, edp_mono, edp_ref, edp

def plot_dépassement(edp, seuil, titre, filtrage, turbine): # mise en évidence des zones de dépassement de seuil
    plt.figure(titre); f=1
    if filtrage:
        f=144; y = moving_average(edp.values, f)
    edp_depassement_haut = np.ma.masked_where(y < seuil, y)
    edp_depassement_bas = np.ma.masked_where(y > -seuil, y)
    edp_ok = np.ma.masked_where((y < -seuil) & (y > seuil), y)
    plt.plot(edp.index[f-1:], edp_ok, label=turbine); plt.plot(edp.index[f-1:], edp_depassement_haut, 'xr'); plt.plot(edp.index[f-1:], edp_depassement_bas, 'xr')

def boxplots_group_comparaison(data_ref, data, id):

    plt.figure(id)
    data_group = pd.DataFrame()
    for col in data_ref.columns:
        if (id in col) and (col in data.columns):
            data_group = pd.concat([data_group, data_ref[col]], axis = 1)
            data_group = pd.concat([data_group, data[col]], axis = 1)
    data_group.boxplot()
    plt.title(id)
    plt.xticks(ticks = range(1, len(data_group.columns)+1), labels = data_group.columns, rotation=35, ha='right')
    return data_group.columns

## TRI DES VARIABLES PAR DEPENDANCE ##
# range les variables par MI estimé croissant
def classement(data, v):
    X = data.iloc[:, 2:].dropna().copy(); Y = X[v]
    v_MI = []
    # Mutual Information
    for col in X.columns: v_MI.append(normalized_mutual_info_score(X[col], Y))
    v_MI = pd.Series(data = v_MI, index = X.columns)
    # comparaison avec les autres mesures de corrélation classiques
    v_r = X.corr(method = "pearson")[v]
    v_rho = X.corr(method = "spearman")[v]
    v_kt = X.corr(method = "kendall")[v]
    correlations = pd.DataFrame(data = [v_r, v_rho, v_kt, v_MI], index = ['pearson', 'spearman', 'kendall', 'mutual information']).drop(columns = [v], axis = 1)
    return correlations

## ENRICHISSEMENT DU JEU DE FEATURES ##
def func_carré(x, a, b): return a * x*x + b
def add_quadratique(data, x, y):
    data_red = data[[x, y]].dropna()
    xdata = data_red[x]
    ydata = data_red[y]
    popt, _ = curve_fit(func_carré, xdata, ydata)
    data[f'{x}_quadratique'] = popt[0]*data[x]**2 + popt[1]#"""
    return data
def enrichissement_quadratique(data, v):
    correlations = classement(data, v)
    Q3_MI = (np.median(correlations.loc['mutual information']) + np.max(correlations.loc['mutual information']))/2
    Q3_R = (np.median(correlations.loc['pearson'].dropna()) + np.max(correlations.loc['pearson'].dropna()))/2
    new_features = []
    for col in data.columns[2:].drop(v):
        if (correlations.loc['mutual information', col] > Q3_MI) and (correlations.loc['pearson', col] < Q3_R):
            print(f'relations non-linéaire ({col}, {v})')
            if type(data) != int: # si défaut enregistré
                new_features.append(col)
    return new_features
def add_retard(data, vturbine, delai, v): # ajout d'une feature retardée
    data_delay = pd.Series()
    for turbine in vturbine:
        X = data[data['ref_turbine_valorem'] == turbine][v]
        data_delay_int = pd.concat([pd.DataFrame(data=[0]), X.iloc[:-delai]])
        data_delay = pd.concat([data_delay, data_delay_int])
    data_delay = data_delay.sort_index()
    data_delay.index = range(len(data_delay.index)) #problème de duplications dans l'index (0,0,0...)
    data[f'{v}_retard_{delai}'] = data_delay
    return data
def std_robust_scaler_manuel(data_train, data_appli, turbines): #standardisation de l'indicateur
    data_train_scale = data_train.copy()
    data_appli_scale = data_appli.copy()
    for tur in turbines:
        moy = data_train[tur].median()
        p75 = data_train[tur].describe()['75%']
        p25 = data_train[tur].describe()['25%']
        data_train_scale[tur] = (data_train_scale[tur] - moy)/(p75-p25)
        data_appli_scale[tur] = (data_appli_scale[tur] - moy)/(p75-p25)
    return data_train_scale, data_appli_scale
def std_scaler_fit(data_train, v_turbine): #estimation des paramètres de standardisation
    v_scaler = dict((v_turbine[:6][i], 0) for i in range(len(v_turbine[:6])))
    scaler = StandardScaler()
    for tur in v_turbine[:6]:
        v_scaler[tur] = scaler.fit(data_train.iloc[:,2:][data_train['ref_turbine_valorem'] == tur])
        data_train.iloc[data_train['ref_turbine_valorem'] == tur, 2:] = v_scaler[tur].transform(data_train.iloc[:,2:][data_train['ref_turbine_valorem'] == tur])
    return data_train, v_scaler
def std_scaler_apply(data_apply, v_turbine, scaler): #application de la standardisation
    for tur in v_turbine[:6]:
        data_apply.iloc[data_apply['ref_turbine_valorem']==tur, 2:] = scaler[tur].transform(data_apply.iloc[:,2:][data_apply['ref_turbine_valorem']==tur])
    return data_apply

#%% temporaire : courbe de puissance
plt.close('all')
vturbine = data_c['ref_turbine_valorem'].unique()
for tur in vturbine:
    print(tur)
    data_tur = data_c[data_c['ref_turbine_valorem'] == tur]
    P = data_tur['puiss_active_produite']
    V = data_tur['vitesse_vent_nacelle']
    count = data_tur['puiss_active_produite'][(data_tur['puiss_active_produite'] > 1747500) & (data_tur['puiss_active_produite'] < 1752500)].count()
    print(count)
    plt.scatter(V, P, label=tur)
plt.grid(); plt.legend()

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


#%% temporaire : comparaison seuil théorique/seuil réel
plt.close('all')
import scipy.stats as stats
import matplotlib
font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 22}
matplotlib.rc('font', **font)
turbine = 'T2'
seuil_theo = dict((tur, 3*TSI_bank_in_scale[tur].std()) for tur in v_turbine)
seuil_real = dict((tur, TSI_bank_in_scale[tur].quantile(.99)) for tur in v_turbine)
for c in ['slow shaft']:
    plt.figure(c)
    for tur in v_turbine:
        ## valeurs réelles (résidu en période de fonctionnement normal) ##
        TSI_bank_in_scale[tur][c].plot.kde(label = tur)
        #plt.vlines([-seuil_real[turbine][c], seuil_real[turbine][c]], 0, 1, 'r', label = 'real detection threshold')
        ## valeurs théoriques (distribution Gaussienne) ##
        x = np.linspace(0 - 3*1, 0 + 3*1, 100)
        #plt.vlines(x = [-seuil_theo[turbine][c], seuil_theo[turbine][c]], ymin = 0, ymax = 1, linestyles = 'dashed', colors = 'r', label = 'theoretical detection threshold')
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'b--', linewidth = 10, label = 'Gaussian distribution') 
    plt.grid(), plt.xlim(-5, 5), plt.legend(), plt.ylim(0, 1)

def plot_coefs_lasso(data, v, method):
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 15}
    matplotlib.rc('font', **font)
    data = data.drop(columns = ['ref_turbine_valorem', 'date_heure']).copy()
    X_train = data.drop(columns = [v]).fillna(0)
    y_train = data[v].fillna(0)
    if method == 'pearson':
        v_coef = data.corr(method = "pearson")[v].drop(index = v)
        v_coef = np.abs(v_coef.to_list())
    if method == 'least square':
        reg_ls = LinearRegression(); reg_ls.fit(X_train, y_train); 
        v_coef = np.abs(reg_ls.coef_)
        #for c in range(len(v_coef)): v_coef[c] = min(1, v_coef[c])
    if method == 'lasso':
        reg_lasso = Lasso(); reg_lasso.fit(X_train, y_train)
        v_coef = np.abs(reg_lasso.coef_)
    if method == 'ridge':
        reg_ridge = Lasso(); reg_ridge.fit(X_train, y_train)
        v_coef = np.abs(reg_ridge.coef_)
        #for c in range(len(v_coef)): v_coef[c] = min(1, v_coef[c])
    if method == 'elastic net':
        reg_elasticnet = ElasticNet(); reg_elasticnet.fit(X_train, y_train); 
        v_coef = np.abs(reg_elasticnet.coef_)
    v_coef = pd.Series(data = v_coef, index = X_train.columns)
    v_coef = v_coef[v_coef > 1e-3]; v_coef = v_coef.sort_values(ascending = False)
    plt.figure('coefficients non nuls par ordre décroissant'); v_coef.plot.bar()
    plt.xticks(ticks = range(len(v_coef.index)), labels = v_coef.index, rotation = 10, ha = 'right')
    plt.grid()
    return v_coef

#%% temporaire : comparaison des RMSE pour 3 configurations (2 CV unshuffled, 2 CV shuffled, 10 CV unshuffled)
"""plt.close('all')
rmse_comp = dict((i, pd.DataFrame(index = data.columns[2:])) for i in cp['rmse'].columns)
for i in cp['rmse'].columns:    rmse_comp[i]['0 CV shuffled'] = cp['rmse'][i]
for i in cp['rmse'].columns:
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 10}
    matplotlib.rc('font', **font)
    mean = rmse_comp[i].mean()
    std = rmse_comp[i].std()    
    rmse_comp[i] = (rmse_comp[i] - mean) / std
    ax = rmse_comp[i].plot(kind = 'bar')
    ax.set_title(f'étape {i}')
    plt.grid()#"""

# temporaire : étude sensibilité sélection de features au type (int/float) #
    """data_learning['brutes'] = data_learning['brutes'].dropna()
    data_test['brutes'] = data_test['brutes'].dropna()
    v_temp = ['temp_palier_arbre_lent', 'temp_exterieur', 'temp_interne_nacelle', 'temp_roul_multi1', 'temp_stator', 'temp_transfo', 'temp_convertisseur', 'temp_huile_multi', 'temp_boite_multi', 'temp_hub', 'temp_automatisme', 'temp_transfo_2', 'temp_transfo_3', 'temp_roul_gene2', 'temp_stator1', 'temp_stator2', 'temp_stator3']
    data_learning['brutes'][v_temp] = data_learning['brutes'][v_temp].astype(int)
    data_test['brutes'][v_temp] = data_test['brutes'][v_temp].astype(int)#"""

"""if format == 'NORDEX':
    #data['grid_activepower'] = data['grid_activepower']*1000
    data['date_heure'] = pd.to_datetime(data['date_heure'], format="%Y-%m-%d %H:%M")
    data['date_heure'] = data['date_heure'].dt.tz_localize(None)
    data = data.sort_values(by = 'date_heure')#"""

## CONVERSION FORMATS CONSTRUCTEUR/S2EV ##
"""if format == 'SENVION':
    n_turbine = int(input('nombre de turbines considérées:'))
    n_variables = int(input('nombre de variables considérées:'))
    v_turbine = [f'T{i}' for i in range(1, n_turbine+1)]
    # calcul du n° de la 1ère ligne de données #
    debut = n_variables*len(v_turbine) + 5 + len(v_turbine) + 1
    # redéfinition des labels de colonne #
    columns = data.iloc[debut,:]; columns[0] = 'date_heure'
    data.columns = columns 
    # suppression de l'en tête #
    data = data.iloc[debut+1:,:]
    # ajout de la colonne 'ref_turbine_valorem' #
    data.insert(0, 'ref_turbine_valorem', 0) 
    # transposition du df par turbine #
    data_parc = []
    for i, tur in enumerate(v_turbine):
        data_parc.append(pd.DataFrame(data = pd.concat([data.iloc[:,[0, 1]], data.iloc[:,n_variables*i+2:n_variables*i+2+n_variables]], axis=1)))
        data_parc[-1]['ref_turbine_valorem'] = tur
    data = pd.concat(data_parc)
    # formatage des dates #
    data['date_heure'] = pd.to_datetime(data['date_heure'], format="%d.%m.%Y %H:%M")
    data = data.sort_values(by = 'date_heure')
    if 'Puissance apparente 10min  [kVA] ' in data.columns:
        data['Puissance apparente 10min  [kVA] '] = data['Puissance apparente 10min  [kVA] ']*1000#"""