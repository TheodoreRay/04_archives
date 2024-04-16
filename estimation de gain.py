#%% LIBRAIRIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

#%% CONFIGURATION
plt.close('all')
font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 20}
matplotlib.rc('font', **font)
fig = plt.figure()
df = pd.DataFrame(index=['$F^{generator}_{sor}$','$F^{gearbox}_{foye}$','$F^{gearbox}_{monne}$','$F^{slowshaft}_{pdrs}$','$F^{converter}_{ablai}$','$F^{generator}_{lapla}$','$F^{generator}_{sante}$'], columns=['gain financier (k€)', 'gain énergétique (MWh)'])
ax = fig.add_subplot(111)
ax2 = ax.twinx()
# facteur de charge (moyenne 2020) #
alpha = 0.2635 
# prix de rachat du kWh éolien (€) #
p_kwh = 0.082 
# avance detection time (jours) #
ADT = {df.index[0]:34,df.index[1]:162,df.index[2]:168,df.index[3]:284,df.index[4]:50,df.index[5]:120,df.index[6]:69}
# temps de maintenance corrective (jours) #
t_mc = {df.index[0]:30,df.index[1]:1,df.index[2]:1,df.index[3]:10,df.index[4]:1.7,df.index[5]:1.16,df.index[6]:1}
# coût de maintenance corrective (€) #
c_mc = {df.index[0]:2e5,df.index[1]:0,df.index[2]:0,df.index[3]:8.6e4,df.index[4]:4e4,df.index[5]:0,df.index[6]:0}
# pertes de production durant l'anomalie (kWh) #
E_fault = {df.index[0]:0,df.index[1]:0,df.index[2]:0,df.index[3]:17*2e3*alpha,df.index[4]:3*3.6e3*alpha,df.index[5]:0,df.index[6]:48*2.5e3*alpha}
# puissance nominale du parc (kWh) #
P_n = {df.index[0]:2e3,df.index[1]:2.05e3,df.index[2]:2e3,df.index[3]:2.05e3,df.index[4]:3.6e3,df.index[5]:2.05e3,df.index[6]:2.5e3}

#%% CALCUL DU GAIN
g_ft = 0
for id_defaut in df.index:
    # temps de maintenance préventive (heure) #
    if ADT[id_defaut]>7:
        t_mp = 1/24
    else:
        t_mp = t_mc[id_defaut]/ADT[id_defaut]
    # calcul #
    print(id_defaut)
    g_f = np.round(((t_mc[id_defaut]-t_mp)*P_n[id_defaut]*alpha*p_kwh*24+c_mc[id_defaut]+E_fault[id_defaut]*p_kwh)/1000,2)
    print(f'gain financier = {g_f} k€')
    print(f"soit le prix de l'instrumentation SCADA de {int((g_f)/(10))} éoliennes")
    g_e = int(((t_mc[id_defaut]-t_mp)*P_n[id_defaut]*alpha*24+E_fault[id_defaut])/1000) #+c_sp
    print(f'gain énergétique = {g_e} MWh')
    print(f'soit la consommation électrique annuelle moyenne de {int(g_e/18.796)} foyer de 4 personnes \n')
    df.loc[id_defaut, :] = [g_f, g_e]
    g_ft += g_f
    # affichage #
    df['gain financier (k€)'].plot(kind='bar', color='red', ax=ax, position=1, label='gain financier')
    df['gain énergétique (MWh)'].plot(kind='bar', color='blue', ax=ax2, position=0, label='gain énergétique')
print(f'gain financier total = {np.round(g_ft,2)} k€')
ax.set_ylabel('gain financier (k€)')
ax2.set_ylabel('gain énergétique (MWh)')
plt.grid()
ax.set_title('gains par défaut considéré')
ax.set_xticks(range(len(list(df.index))), list(df.index), rotation=45)