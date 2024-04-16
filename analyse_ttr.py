#%%
from tkinter.constants import S
import functions_plot as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from importlib import reload
from sklearn.metrics import mean_squared_error
from dateutil.rrule import rrule, MONTHLY, WEEKLY

#%% import
#data_grv_2011 = pd.read_excel(f"../../../1_data/11_imports_s2ev/112_Traitées/VESTAS/D10M_GRANDVILLE_2011.xlsx")
#v_turbine_grv = list(data_grv_2011['ref_turbine_valorem'].unique())
data_sqn_2020 = pd.read_excel(f"../../../1_data/12_scada_turbiniers/V110_GENE_SCIPHER.xlsx")
v_turbine_sqn = list(data_sqn_2020['ref_turbine_valorem'].unique())
"""data_sqn_2020 = data_sqn_2020.iloc[:, 1:].copy()
data_sqn_2019 = pd.read_excel(f"../../../1_data/13_autres/TTR/P_SQN_D10M_TEMP_2019.xlsx")
data_sqn_2019 = data_sqn_2019.iloc[:, 1:].copy()
data_sqn_2021 = pd.read_excel(f"../../../1_data/13_autres/TTR/P_SQN_D10M_TEMP_2021.xlsx")
data_sqn_2021 = data_sqn_2021.iloc[:, 1:].copy()#"""

#%% affichage des corrélations
plt.close('all')
#f.plot_couple(v_turbine_grv, data_grv_2011, 'temp_roul_gene1', 'temp_roul_gene2', 'GRV 2011')
f.plot_couple(v_turbine_sqn, data_sqn_2019, 'Bearing Temp (°C)', 'Bearing Rear Temp (°C)', 'LALUZETTE 2022')

#%% évolution R²
plt.close('all')
#data = data_grv_2011.fillna(0)
#data = pd.concat([data_sqn_2019.fillna(0), data_sqn_2020.fillna(0), data_sqn_2021.fillna(0)], axis=0, ignore_index = True)
data = data_sqn_2020.fillna(0)
x = 'Bearing Temp (°C)'
y = 'Bearing Rear Temp (°C)'
#x = 'temp_roul_gene1'
#y = 'temp_roul_gene2'

rc = pd.DataFrame(columns = v_turbine_sqn)
for turbine in ['A1']:
    data_tur = data[data['ref_turbine_valorem'] == turbine]
    dates = list((rrule(WEEKLY, interval = 1, dtstart = min(data_tur['date_heure']), until =  max(data_tur['date_heure']))))
    data_tur = data_tur.set_index('date_heure')
    print(turbine)
    h = 4
    for i, m in enumerate(dates[h:-1]):
        print(m)
        print(dates[i])
        rc.loc[m, turbine] = round(scipy.stats.pearsonr(data_tur[x][dates[i]:m], data_tur[y][dates[i]:m])[0], 3)
    plt.plot(rc[turbine], label = turbine, marker = 'x')
plt.legend(); plt.grid()
