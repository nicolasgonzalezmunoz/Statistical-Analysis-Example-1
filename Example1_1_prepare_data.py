"""
This code is designed to import, clean and prepare data for a statistical analysis
of data. Please run this file first so you get the necessary files for the rest of this work
This is an example script designed to show what the programmer is able to do.
"""
#Import packages
import numpy as np
import pandas as pd
import os #This last package gives us absolute file paths

#Save file names
file_dir="World Bank Data"
exports_file="Exports of goods and services (Current USD).csv"
gdp_file="GDP per capita (Current USD).csv"
pol_stab_file="Political Stability and Absence of Violence.csv"
reg_quality_file="Regulatory Quality.csv"

#Import Data as Pandas DataFrames and Drop Na values
Df_Exports=pd.read_csv(os.path.abspath(file_dir+'/'+exports_file),usecols=[2,3]+list(range(6,16)),skiprows=range(267,272), na_values='..')
Df_GDP=pd.read_csv(os.path.abspath(file_dir+'/'+gdp_file),usecols=[0,1]+list(range(4,2021-1960+5)),skiprows=range(4), na_values='..')
Df_Pol_Stab=pd.read_csv(os.path.abspath(file_dir+'/'+pol_stab_file),usecols=[0,1]+list(range(4,14)),skiprows=range(215,220), na_values='..')
Df_Reg_Quality=pd.read_csv(os.path.abspath(file_dir+'/'+reg_quality_file),usecols=[0,1]+list(range(4,14)),skiprows=range(215,220), na_values='..')

#Drop Columns that are not in all the DataFrames, then all rows with na values
Df_Exports = Df_Exports.drop(columns=Df_Exports.columns[range(2,Df_Exports.shape[1]-10)]).dropna()
Df_GDP = Df_GDP.drop(columns=Df_GDP.columns[range(2,Df_GDP.shape[1]-10)]).dropna()
Df_Pol_Stab = Df_Pol_Stab.drop(columns=Df_Pol_Stab.columns[range(2,Df_Pol_Stab.shape[1]-10)]).dropna()
Df_Reg_Quality = Df_Reg_Quality.drop(columns=Df_Reg_Quality.columns[range(2,Df_Reg_Quality.shape[1]-10)]).dropna()

#All DataFrames have the column "Country Code", so we want to preserve the countries present in all Dfs
Dfs=[Df_Exports,Df_GDP,Df_Pol_Stab,Df_Reg_Quality]
for i in range(3):
    for j in range(i+1,4):
        #Take indexes of both dataframes
        indexes1=np.array(Dfs[i].index)
        indexes2=np.array(Dfs[j].index)
        #Take key values
        codes1=np.array(Dfs[i].loc[:,"Country Code"])
        codes2=np.array(Dfs[j].loc[:,"Country Code"])
        #Get indexes to drop
        drop1=np.where(np.isin(codes1,codes2,invert=True))
        drop2=np.where(np.isin(codes2,codes1,invert=True))
        #Drop data
        Dfs[i].drop(index=indexes1[drop1],inplace=True)
        Dfs[j].drop(index=indexes2[drop2],inplace=True)
    
#sort data by country code, so that each country is allways at the same positions in each dataframe
#then, reindex data
for i in range(4):
    Dfs[i].sort_values(by='Country Code', axis=0, ascending=True, inplace=True)
    Dfs[i].index = range(Dfs[i].shape[0])

#Create directory to save cleaned data
data_dir="Clean Data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

#Save files into the new directory
np.savetxt(os.path.abspath(data_dir+'/'+'Exports_Clean.csv'),np.array(Df_Exports.iloc[:,2:]).astype(np.float),delimiter=',')
np.savetxt(os.path.abspath(data_dir+'/'+'GDP_Clean.csv'),np.array(Df_GDP.iloc[:,2:]).astype(np.float),delimiter=',')
np.savetxt(os.path.abspath(data_dir+'/'+'Pol_Stab_Clean.csv'),np.array(Df_Pol_Stab.iloc[:,2:]).astype(np.float),delimiter=',')
np.savetxt(os.path.abspath(data_dir+'/'+'Reg_Quality_Clean.csv'),np.array(Df_Reg_Quality.iloc[:,2:]).astype(np.float),delimiter=',')
np.savetxt(os.path.abspath(data_dir+'/'+'Country Codes'),np.array(Df_Exports.iloc[:,1]).astype(str),delimiter=',',fmt='%.18s')
np.savetxt(os.path.abspath(data_dir+'/'+'Country Names'),np.array(Df_Exports.iloc[:,0]).astype(str),delimiter=',',fmt='%.18s')
np.savetxt(os.path.abspath(data_dir+'/'+'Columns'),np.array(Df_GDP.columns).astype(str),delimiter=',',fmt='%.18s')