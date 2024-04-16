#%% 1/ "SCORE" DES TSI 
PM_train = pd.DataFrame(index = list(v_turbine) + ['average', 'étendue'], columns = v_composant)
PM_test = pd.DataFrame(index = list(v_turbine) + ['average', 'étendue'], columns = v_composant)

## 1/ AJOUT DE METAVARIABLES
coef_lasso = f.plot_coefs_lasso(data_in, sorties['rotor'], 15)
V_o = ['puiss_apparente', 'temp_interne_nacelle', 'temp_exterieur', 'i_produite', 'temp_gene', 'vitesse_arbre_lent', 'vitesse_arbre_rapide'] #list(coef_lasso[:8].index)
data_augmentee = f.generation_de_features_non_lineaires(data, V_o)
data_train, data_test = f.une_saison(data = data_augmentee, debut = '2020-09', fin = '2020-12-31', distinction = False) #"""
data_test = data_augmentee[(data_augmentee['date_heure']>'2021-01') & (data_augmentee['date_heure']<'2021-06')]
data_fault = data_augmentee[(data_augmentee['date_heure']>'2020-06') & (data_augmentee['date_heure']<='2020-09')]

#%% 3/ "SCORE" DES TSI 
for tur in v_turbine: 
    PM_train[composant][tur]= np.sqrt((TSI_train[composant][tur]*TSI_train[composant][tur]).mean())
    PM_test[composant][tur]= np.sqrt((TSI_test[composant][tur]*TSI_test[composant][tur]).mean())
PM_train[composant]['average'] = np.median([PM_train[composant][tur] for tur in v_turbine])
PM_test[composant]['average'] = np.median([PM_test[composant][tur] for tur in v_turbine])
PM_train[composant]['étendue'] = np.max([PM_train[composant][tur] for tur in v_turbine]) - np.min([PM_train[composant][tur] for tur in v_turbine])
PM_test[composant]['étendue'] = np.max([PM_test[composant][tur] for tur in v_turbine]) - np.min([PM_test[composant][tur] for tur in v_turbine])

## 3/ TRANSFERT CONDITIONNEL DE COEFFICIENTS ##
if tur == 'T1':
    data_coef[composant][tur], _, _ = f.model_learning(tur, V_train[composant], V_test[composant], models[composant][1], models[composant][0], 'least squares')
else:
    data_coef[composant][tur] = data_coef[composant]['T1']#"""

#%% 4/ AFFICHAGE INDICATEURS
f.plot_indicateur(TSI_test, ['T1'], ['generator'], seuil, '2010-01', '2010-05', 20, 10)#, [{'turbine':'A1', 'composant':'gearbox bearing', 'date':'2020-10-23 03:00:00', 'texte':'changement huile VESTAS'}])
f.distributions(TSI_test, v_turbine, ['gearbox', 'generator'], f'{annee}-01', f'{annee}-12', 15)
df_TFA, NIIT = f.heatmaps_fausses_alarmes(TSI_fault, v_turbine, v_composant, seuil, WEEKLY, 8)

#%% Fonctions
## affichage des fonctions de densité et calcul des critères de performances statistiques ##
def distributions(TSI, v_turbine, v_composant, debut, fin, font_size):
    plt.figure()
    # configuration #
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : font_size}
    matplotlib.rc('font', **font)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'bisque', 'forestgreen', 'royalblue']
    v_std = pd.DataFrame(index = v_turbine, columns = v_composant)
    v_ampmax = pd.DataFrame(index = v_turbine, columns = v_composant)
    for i, composant in enumerate(v_composant):
        plt.subplot(len(v_composant), 1, i+1)
        plt.title(composant)
        for j, turbine in enumerate(v_turbine):
            # affichage des densités #
            y = pd.Series(data = TSI[composant][turbine][(TSI[composant].index>debut) & (TSI[composant].index<fin)].values)
            density = gaussian_kde(y)
            xs = np.linspace(-4, 4, 200)
            plt.plot(xs, density(xs), colors[j], linewidth = 1.5, label = turbine)
            # calcul des critères de performances statistiques #
            v_std.loc[turbine, composant] = y.std()
            v_ampmax.loc[turbine, composant] = np.max(density(xs))
        plt.legend(), plt.ylim(0, 10), plt.grid()
        # calcul de la dispersion et de la moyenne des paramètres #
        v_std.loc['dispersion', composant] = v_std.loc[:, composant].std()
        v_ampmax.loc['dispersion', composant] = v_ampmax.loc[:, composant].std()
        v_std.loc['moyenne', composant] = v_std.loc[:, composant].mean()
        v_ampmax.loc['moyenne', composant] = v_ampmax.loc[:, composant].mean()
    return v_std, v_ampmax

def cahier_des_charges(TSI_test, TSI_fault, seuil, v_turbine, data_models):
    v_ps = {'ps1':0, 'ps2':0, 'ps3':[[] for turbine in v_turbine]} # performances statistiques
    v_po = {'po1':0, 'po2':0, 'po3':0} # performances opérationnelles
    # calcul des performances #
    for composant in data_models.columns:
        print(composant)
        ps1 = 0
        ps2 = 0
        ps3 = [[] for turbine in v_turbine]
        for i, turbine in enumerate(v_turbine):
            ps1 += TSI_test[composant][turbine].std()
            ps2 += TSI_test[composant][turbine].mean()
            ps3[i], _, _, _, _ = linregress(np.linspace(0, len(TSI_test[composant][turbine]), len(TSI_test[composant][turbine])),TSI_test[composant][turbine].values)
    _, v_po['po1'] = heatmaps_fausses_alarmes(TSI_test, v_turbine, [composant], seuil, WEEKLY, 12)
    po3, po2 = performances_detection(TSI_fault, composant, 'T1', seuil, 7, '2020-06-01', '2020-09-01')

    # performances statistiques #
    v_ps['ps1'] = ps1/len(v_turbine); print(f'ps1 = {v_ps["ps1"]}'); 
    v_ps['ps1'] = ps2/len(v_turbine); print(f'ps2 = {v_ps["ps2"]}')
    v_ps['ps1'] = np.mean(ps3); print(f'ps3 = {v_ps["ps3"]}')
    
    # performances opérationnelles #
    print(f'po1 = {v_po["po1"]}')
    v_po['po2'] = po2; print(f'po2 = {v_po["po2"]}')
    v_po['po3'] = po3; print(f'po3 = {v_po["po3"]}')

    return v_ps, v_po

def plot_indicateur(TSI, v_turbine, v_composant, seuil, debut, fin, ticks_number, font_size, fault_cnfg = []):
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : font_size}
    matplotlib.rc('font', **font)
    plt.subplots(len(v_turbine), 1, sharey = True, sharex = True)
    ymax = max([max([TSI[c][tur][(TSI[c].index>debut) & (TSI[c].index<fin)].max() for c in v_composant]) for tur in v_turbine])
    ymin = min([min([TSI[c][tur][(TSI[c].index>debut) & (TSI[c].index<fin)].min() for c in v_composant]) for tur in v_turbine])
    for i, turbine in enumerate(v_turbine):
        #ax = fig.add_subplot(len(v_turbine), 1, i+1)
        plt.subplot(len(v_turbine), 1, i+1)
        for composant in v_composant:
            data = TSI[composant][turbine][(TSI[composant].index>debut) & (TSI[composant].index<fin)]
            if len(TSI[composant][turbine].dropna()) > 0: # indicateur existant
                y = data.values; y = np.array(y, dtype=float)
                x = data.index
                plt.plot(x, y, label = composant)
                plt.fill_between(x, y, where=(y > seuil[composant][turbine]) | (y < -seuil[composant][turbine]))#, color = 'r')
        for cnfg in fault_cnfg:
            if (turbine == cnfg['turbine']) and (composant == cnfg['composant']):  
                annotations(TSI[cnfg['composant']][cnfg['turbine']], [cnfg['date']], [cnfg['texte']])
        plt.grid(True)
        if turbine != v_turbine[-1]:    plt.xlabel('')
        plt.ylim(ymin, ymax), 
        plt.title(turbine), plt.ylabel('TSI value (standardised)')
    config(pd.Series(data = x), ticks_number)
    plt.legend(loc = 'lower right')

# RQ : un point d'indisponibilité signifie que l'indicateur était indisponible au moins les 144 derniers points (cf filtrage moyen)
def disponibilite_indicateurs(TSI, v_turbine, v_composant): 
    dispo = dict((v_composant[c], pd.DataFrame(columns = v_turbine)) for c in range(len(v_composant)))
    dates = list(TSI[v_composant[0]][v_turbine[0]].index)
    for composant in v_composant:
        _, axs = plt.subplots(len(v_turbine), 1, sharey = True, sharex = True)
        for tur, turbine in enumerate(v_turbine): 
            dispo[composant][turbine] = pd.Series(index = dates, data = np.ones(len(TSI[composant][turbine])))
            dispo[composant][turbine].loc[TSI[composant][turbine] == 0] = 0
            axs[tur].step(dates, dispo[composant][turbine])
            axs[tur].grid(); axs[tur].set_ylabel(turbine)
    return dispo

# scatter plot MAE(Kurtosis) #
def plot_metric(cp, s, variables): 
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 25}
    matplotlib.rc('font', **font)
    v = cp['coût'].loc[variables, s].idxmin()
    plt.figure(f'mae(kurt) étape {s}')
    plt.scatter(cp['kurtosis'].loc[variables, s], cp['mae'].loc[variables, s], s=200, 
                c=cp['coût'].loc[variables, s], cmap='RdYlGn_r')
    plt.annotate(v, xy=(cp['kurtosis'].loc[v, s], cp['mae'].loc[v, s]), xycoords='data', 
                xytext=(0.3, 0.1), textcoords='figure fraction',
                arrowprops=dict(facecolor='black', shrink=0.05), bbox=dict(boxstyle="round", fc="0.8"), ha="center", va="center", size=25)
    mplcursors.cursor().connect("add", lambda sel: sel.annotation.set_text(cp['kurtosis'].index[sel.index]))
    plt.xlabel('kurtosis'), plt.ylabel('mae')
    plt.ylim(0, 1), plt.grid()

def sous_optimalité(data, v_turbine, turbine, modele, v, modele_parametres, variables, approche):
    for i in range(1, len(modele)):
        modele_reduit = modele[:i] + modele[i+1:]
        print(modele_reduit)
        print(f'RMSE actuel : {modele_parametres["RMSE"]}')
        ## estimation du RMSE modèle réduit
        if approche == 'turbine':
            _, _, RMSE = model_learning(turbine, data, modele_reduit, v)
            print(f'RMSE modele reduit : {RMSE}\n')
        elif approche == 'parc':
            v_RMSE = pd.Series(index = v_turbine)
            for tur in v_turbine: 
                _, _, v_RMSE[tur] = model_learning(tur, data, modele_reduit, v)
            RMSE = np.median(v_RMSE)
            print(f'RMSE modele reduit : {RMSE}\n')
        ## comparaison avec le modèle actuel    
        if RMSE - modele_parametres['RMSE'] < 0:
            print(f'sous-optimalité détectée, {modele[i]} retirée du modèle \n') 
            variables.append(modele[i]); modele.pop(i)
    return variables, modele

def simu_defaut(data, y, turbine, debut, fin, a):  #simulation de défaut sur Y pour une période donnée
    y_initial = data[data['ref_turbine_valorem']==turbine][y]
    y_sain = data[data['ref_turbine_valorem']==turbine][(data['date_heure'] >= debut) & (data['date_heure'] < fin)][y]
    y_defaut = y_sain + a*np.arange(len(y_sain))
    y_modif = pd.concat([data[data['ref_turbine_valorem']==turbine][y].drop(index = y_sain.index), y_defaut])
    return y_initial, y_modif

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

#%% ----------- AJOUT BOUTONS (nécessite lines, fig) ----------------
#composant = input('composant :') #boutons à activer
"""composant = 'rotor'
def config_turbine_plot(turbine):
            index = list(v_turbine).index(turbine)
            lines[composant][index].set_visible(not lines[composant][index].get_visible())
            figs[composant].canvas.draw()
ax_check = figs[composant].add_axes([0.9, 0.68, 0.05, 0.2])
plot_button = CheckButtons(ax_check, v_turbine, [True]*len(v_turbine))
plot_button.on_clicked(config_turbine_plot)#"""

## Performances de Détection ##
#persistance, date_premiere_detection = f.performances_detection(TSI_test, 'generator', 'T1', seuil, 1, '2020-01-01', '2020-09-01')

"""def integration_defaut(data_H0, turbine, composant, data_models, debut, fin):
    # import des données en défaut #
    if composant == 'refroidissement convertisseur':
        data_H1 = pd.read_excel(f"../../../1_data/11_scada_S2EV/CONV_N131_ABLAI_2020.xlsx", sheet_name=None)#.iloc[:, 1:]
    if composant == 'roulement 2 génératrice':
        data_H1 = pd.read_excel(f"../../../1_data/11_scada_S2EV/GENE_ECO80_SOR_2010.xlsx", sheet_name=None)#.iloc[:, 1:]
    if composant == 'palier arbre lent':
        data_H1 = pd.read_excel(f"../../../1_data/11_scada_S2EV/SLOWSHAFT_MM92_PDRS_2019.xlsx", sheet_name=None)#.iloc[:, 1:]
    for key in data_H1:
        data_H1[key] = data_H1[key].set_index(data_H1[key].iloc[:, 0]).iloc[:, 1:]
    if composant in list(data_models.keys()):
        # établissement des variables à modifier #
        if data_H1['Modèle'].index.name != data_models[composant].index.name:
            data_H1['Défaut'] = data_H1['Défaut'].rename(columns={data_H1['Modèle'].index.name : data_models[composant].index.name})
            data_H1['Bon fonctionnement'] = data_H1['Bon fonctionnement'].rename(columns={data_H1['Modèle'].index.name : data_models[composant].index.name})
            data_H1['Modèle'].index.names = [data_models[composant].index.name]
        X_H0 = list(data_models[composant].index)
        X_H1 = list(data_H1['Modèle'].index) #data_H1['Données'].columns #
        YX = [data_H1['Modèle'].index.name] + list(set(X_H0) & set(X_H1))
        print(f'modélisation {composant} disponible !')
        print(f'variables modèle communes : {YX}')

        # sélection de la période de défaut #
        data_H0H1 = data_H0[(data_H0['date_heure'] >= debut) & (data_H0['date_heure'] < fin)]
        # standardisation des données en défaut #
        [data_H1['Bon fonctionnement'], data_H1['Défaut']], _, _ = std_scaler_manuel(data_H1['Bon fonctionnement'], [data_H1['Bon fonctionnement'], data_H1['Défaut']], [data_H1['Défaut']['ref_turbine_valorem'].iloc[0]])
        # copie des données en défaut #
        data = data_H1['Défaut'][YX].iloc[:len(data_H0H1.loc[data_H0H1['ref_turbine_valorem'] == turbine, YX]), :]
        #data = data - (data.mean() - data_H0H1.loc[data_H0H1['ref_turbine_valorem'] == turbine, YX].mean())
        
        data_H0H1.loc[data_H0H1['ref_turbine_valorem'] == turbine, YX] = data.values
        # ré-assemblage des données
        data_H0H1 = pd.concat([data_H0[data_H0['date_heure'] < debut], data_H0H1, data_H0[data_H0['date_heure'] > fin]])
        data_H0H1.drop_duplicates(keep = 'first', inplace = True)

    else:   
        data_H0H1 = data_H0; print('sortie non disponible, défaut non intégré')

    return data_H0H1#"""

## AFFICHAGE ZONES H0 (vert), H1 (rouge) ##
        # zone H0 #
        """ax.add_patch(Rectangle(xy=(mdates.date2num(np.datetime64(H[0])), ymin), \
            width=mdates.date2num(np.datetime64(H[1]))-mdates.date2num(np.datetime64(H[0])), \
            height=np.abs(ymin)+ymax, edgecolor='green', facecolor='palegreen', lw=3, linestyle='dashed', alpha=.3, label='$H_0$'))
        # zone H1 #
        if len(H1) > 0:
            ax.add_patch(Rectangle(xy=(mdates.date2num(np.datetime64(H1[0])), ymin), \
                width=mdates.date2num(np.datetime64(H1[1]))-mdates.date2num(np.datetime64(H1[0])), \
                height=np.abs(ymin)+ymax, edgecolor='red', facecolor='lightcoral', lw=3, linestyle='dashed', alpha=.3, label='$H_1$'))#"""    
