#%% LIBRAIRIES
import functions_other as f
import functions_plot as plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib import reload
from dateutil.rrule import rrule, HOURLY, DAILY, WEEKLY, MONTHLY

#%% Code
plt.close('all')
# import et mise en forme des données
data = pd.read_excel("../../../4_rapports/41_travaux en cours/MAME_SPE_unc.xlsx").set_index('itération')
data_spe = data.iloc[[0, 2, 4, 6], :]
data_MAME = data.iloc[[1, 3, 5, 7], :]

# affichage des histogrammes
plt.figure('spécificité')
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.title(f'T{i}')
    data_spe.iloc[i-1, :].hist(grid=True, bins=10)
    plt.xlim(0.5, 1)
plt.figure('MAME')
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.title(f'T{i}')
    data_MAME.iloc[i-1, :].hist(grid=True, bins=10)
    plt.xlim(0., 0.01)

#%% estimation tau 95 MAME, spécificité
tau95_spe = []
tau95_MAME = []
for i in range(4):
    #tau95_spe.append(data_spe.iloc[i, :].quantile(q=0.95))
    tau95_spe.append(data_spe.iloc[i, :].std())
    #print(data_spe.iloc[i, :].quantile(q=0.95))
    print(data_spe.iloc[i, :].std())
    #tau95_MAME.append(data_MAME.iloc[i, :].quantile(q=0.95))
    tau95_MAME.append(data_MAME.iloc[i, :].std())
    #print(data_MAME.iloc[i, :].quantile(q=0.95))
    print(data_MAME.iloc[i, :].std())
tau95_spe = np.median(tau95_spe)
tau95_MAME = np.median(tau95_MAME)
