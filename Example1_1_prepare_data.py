"""
This script is dedicated to data preprocessing.

Here we will load the necessary data, drop the unnecessary columns,
drop the rows with NaN values and save some that that we could need 
later.
"""
# Import packages
import os
import numpy as np
import pandas as pd

print('Importing and cleaning data')

# Define file names
file_dir="World Bank Data"
exports_file="Exports of goods and services (Current USD).csv"
gdp_file="GDP per capita (Current USD).csv"
pol_stab_file="Political Stability and Absence of Violence.csv"
reg_quality_file="Regulatory Quality.csv"

# Import Data as Pandas DataFrames and Drop unnecessary columns
Df_Exports=pd.read_csv(os.path.abspath(file_dir+'/'+exports_file),usecols=[2,3]+list(range(6,16)),skiprows=range(267,272), na_values='..')
Df_GDP=pd.read_csv(os.path.abspath(file_dir+'/'+gdp_file),usecols=[0,1]+list(range(4,2021-1960+5)),skiprows=range(4), na_values='..')
Df_Pol_Stab=pd.read_csv(os.path.abspath(file_dir+'/'+pol_stab_file),usecols=[0,1]+list(range(4,14)),skiprows=range(215,220), na_values='..')
Df_Reg_Quality=pd.read_csv(os.path.abspath(file_dir+'/'+reg_quality_file),usecols=[0,1]+list(range(4,14)),skiprows=range(215,220), na_values='..')

# Drop Columns that are not in all the DataFrames, and all rows with na values
Df_Exports = Df_Exports.drop(columns=Df_Exports.columns[range(2,Df_Exports.shape[1]-10)]).dropna()
Df_GDP = Df_GDP.drop(columns=Df_GDP.columns[range(2,Df_GDP.shape[1]-10)]).dropna()
Df_Pol_Stab = Df_Pol_Stab.drop(columns=Df_Pol_Stab.columns[range(2,Df_Pol_Stab.shape[1]-10)]).dropna()
Df_Reg_Quality = Df_Reg_Quality.drop(columns=Df_Reg_Quality.columns[range(2,Df_Reg_Quality.shape[1]-10)]).dropna()

# The column "Country Code" acts as a primary key, so we want to preserve
# only the countires shared by all the datasets
Dfs=[Df_Exports,Df_GDP,Df_Pol_Stab,Df_Reg_Quality]
for i in range(3):
    for j in range(i+1,4):

        # Take indexes of both dataframes
        indexes1=np.array(Dfs[i].index)
        indexes2=np.array(Dfs[j].index)

        # Take key values
        codes1=np.array(Dfs[i].loc[:,"Country Code"])
        codes2=np.array(Dfs[j].loc[:,"Country Code"])

        # Get indexes to drop
        drop1=np.where(np.isin(codes1,codes2,invert=True))
        drop2=np.where(np.isin(codes2,codes1,invert=True))

        # Drop data
        Dfs[i].drop(index=indexes1[drop1],inplace=True)
        Dfs[j].drop(index=indexes2[drop2],inplace=True)
    
# Sort data by country code, so that each country is allways at the same
# positions in each dataframe, then reset index
for i in range(4):
    Dfs[i].sort_values(by='Country Code', axis=0, ascending=True, inplace=True)
    Dfs[i].index = range(Dfs[i].shape[0])

# Create a directory (if it doesn't already exists) to save the tidy data
data_dir="Clean Data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# Save files into the new directory, format it when necessary
np.savetxt(os.path.abspath(data_dir+'/'+'Exports_Clean.csv'),np.array(Df_Exports.iloc[:,2:]).astype(np.float),delimiter=',')
np.savetxt(os.path.abspath(data_dir+'/'+'GDP_Clean.csv'),np.array(Df_GDP.iloc[:,2:]).astype(np.float),delimiter=',')
np.savetxt(os.path.abspath(data_dir+'/'+'Pol_Stab_Clean.csv'),np.array(Df_Pol_Stab.iloc[:,2:]).astype(np.float),delimiter=',')
np.savetxt(os.path.abspath(data_dir+'/'+'Reg_Quality_Clean.csv'),np.array(Df_Reg_Quality.iloc[:,2:]).astype(np.float),delimiter=',')
np.savetxt(os.path.abspath(data_dir+'/'+'Country Codes'),np.array(Df_Exports.iloc[:,1]).astype(str),delimiter=',',fmt='%.18s')
np.savetxt(os.path.abspath(data_dir+'/'+'Country Names'),np.array(Df_Exports.iloc[:,0]).astype(str),delimiter=',',fmt='%.18s')
np.savetxt(os.path.abspath(data_dir+'/'+'Columns'),np.array(Df_GDP.columns).astype(str),delimiter=',',fmt='%.18s')