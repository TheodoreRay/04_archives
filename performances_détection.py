import numpy as np
import fonctions_perso as f
import matplotlib.pyplot as plt

#%% FONCTIONS LOCALES
# calcul du seuil de détection
def seuil(edp_H0):
    s_opti = 0
    std = edp_H0.std()
    range_s = np.arange(std*0.1, 10*std, 0.2*std)
    for s in range_s:
        #_, TFA = perf_detecteur(0, 0, edp_H0, s)
        nii = interventions_inutiles(edp_H0, s)
        #if TFA <= 0.04:
        if nii <= 4:
            s_opti = s
            break
    return s_opti

def interventions_inutiles(edp_H0, seuil):
    nii = 0 # nombre d'interventions inutiles
    for i in range(len(edp_H0)):
        if np.abs(edp_H0[i])>seuil and np.abs(edp_H0[i-1])<seuil:
            nii += 1
    return nii

#%% 0/ CONFIGURATION
## variables input
v = input('variable à surveiller')
turbine = input('turbine concernée')
modele = input('modèle étudié (séparer variables par "," sans espace, dénominations exactes !')
modele = modele.split(",")
## import et mise en forme des données
data, vturbine = f.import_donnees()
# data_ref, _ = import_donnees(v) # comparaison avec un autre parc
data_in, data_out = f.un_mois_par_saison(data = data, premier_mois = 2) # distinction des données
v_coef = dict((vturbine[i], 0) for i in range(len(vturbine))); v_const = v_coef.copy()
for tur in vturbine: v_coef[tur], v_const[tur], _ = f.learning_cv(tur, data_in, modele, v)

#%% 1/ PERFORMANCES DE DETECTION
# import et mise en forme des données de défaillance
data_fault, _ = f.Excel_to_DataFrame()
data_fault = data_fault.iloc[:, 1:].sort_values(by = 'date_heure')

# établissement des seuils
turbine = 'T4'
_, _, _, edp_seuil = f.residu_mono_multi(turbine, data_out, modele, v, v_coef, v_const, vturbine)
edp_seuil = f.moving_average(edp_seuil.values, 144)
seuil = seuil(edp_seuil)
plt.figure(); plt.plot(edp_seuil)

# calcul des performances
_, _, _, edp_H1 = f.residu_mono_multi(turbine, data_fault, modele, v, v_coef, v_const, vturbine)
temps = data_fault[data_fault['ref_turbine_valorem'] == turbine]['date_heure']
#fonction AVD à faire
edp_H1 = f.moving_average(edp_H1.values, 144)
