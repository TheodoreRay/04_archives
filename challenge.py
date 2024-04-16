#%%
import f_utilitaires as tools
data, filename = tools.Excel_to_DataFrame()
#%%
data_c = data[data['Turbine_ID']=='T07']
# %%
data_H0 = data_c[(data_c['Timestamp']>'2016-04') & (data_c['Timestamp']<'2016-05')]
#%%
data_H0.to_excel('testing_T07.xlsx')