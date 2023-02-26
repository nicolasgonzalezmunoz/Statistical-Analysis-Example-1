"""
This is the second file of this example. If you have not ran the prepare_data file yet, please do so.
This file is dedicated to data exploration and generate related visualizations.
"""

#Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import os

#Import Clean Data
data_dir = 'Clean Data'
exports = np.genfromtxt(os.path.abspath(data_dir+'/'+'Exports_Clean.csv'), delimiter=',')
gdp = np.genfromtxt(os.path.abspath(data_dir+'/'+'GDP_Clean.csv'), delimiter=',')
pol_stab = np.genfromtxt(os.path.abspath(data_dir+'/'+'Pol_Stab_Clean.csv'), delimiter=',')
reg_quality = np.genfromtxt(os.path.abspath(data_dir+'/'+'Reg_Quality_Clean.csv'), delimiter=',')
timestamps = np.array(np.arange(2012,2022))

#Plot data against each other
data = [gdp, exports, pol_stab, reg_quality]
fig, axs = plt.subplots(2,3, figsize=(16,24))
n_ax = 0 #Counter for axe number
#Iter through data
for i in range(3):
    for j in range(i+1,4):
        for n_col in range(data[0].shape[1]):
            axs[n_ax//3][n_ax%3].plot(data[i][:,n_col], data[j][:,n_col], 'ro', ms=timestamps[n_col]-2011)
        n_ax+=1

#Set labels for axis
labels = ['GDP per capita', 'Exports', 'Political Stability', 'Regulatory Quality']
x_ind = 0   #Indices for labels in each iteration
y_ind = 1
for n_ax in range(6):
    axs[n_ax//3][n_ax%3].set_xlabel(labels[x_ind], fontsize=20)
    axs[n_ax//3][n_ax%3].set_ylabel(labels[y_ind], rotation=90, fontsize=20)
    if n_ax==2:
        x_ind = 1
        y_ind = 2
    elif n_ax==4:
        x_ind = 2
        y_ind = 3
    else:
        y_ind+=1
fig.suptitle('Variables vs each other (dot size=timestamp)', fontsize=30)
plt.tight_layout()
fig.subplots_adjust(top=0.95)

#Make directory to save visualizations
vis_dir = 'Visualizations'
if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)

#Save visualization
vis_file = 'Data Exploration.png'
fig.savefig(os.path.abspath(vis_dir+'/'+vis_file),bbox_inches='tight')



#It seems to be more convinient to replicate the same plot with log(gdp) and log(exports)
data = [np.log(gdp), np.log(exports), pol_stab, reg_quality]
fig2, axs2 = plt.subplots(2,3, figsize=(16,24))
n_ax = 0 #Counter for axe number
#Iter through data
for i in range(3):
    for j in range(i+1,4):
        for n_col in range(data[0].shape[1]):
            if n_ax==3 or n_ax==4:
                axs2[n_ax//3][n_ax%3].plot(data[i][:,n_col], data[j][:,n_col], 'ro', ms=timestamps[n_col]-2011)
            else:
                axs2[n_ax//3][n_ax%3].plot(data[i][:,n_col], data[j][:,n_col], 'ro', ms=timestamps[n_col]-2011)
        n_ax+=1

#Set labels for axis
labels = ['log(GDP per capita)', 'log(Exports)', 'Political Stability', 'Regulatory Quality']
x_ind = 0   #Indices for labels in each iteration
y_ind = 1
for n_ax in range(6):
    axs2[n_ax//3][n_ax%3].set_xlabel(labels[x_ind], fontsize=20)
    axs2[n_ax//3][n_ax%3].set_ylabel(labels[y_ind], rotation=90, fontsize=20)
    if n_ax==2:
        x_ind = 1
        y_ind = 2
    elif n_ax==4:
        x_ind = 2
        y_ind = 3
    else:
        y_ind+=1
fig2.suptitle('Variables vs each other (dot size=timestamp)', fontsize=30)
plt.tight_layout()
fig2.subplots_adjust(top=0.95)
vis_file = 'Data Exploration Log.png'
fig2.savefig(os.path.abspath(vis_dir+'/'+vis_file),bbox_inches='tight')

#Now, let's calculate the correlation matrix of the variables
#First, we'll flatten the variables
gdp = np.log(gdp).flatten()
exports = np.log(exports).flatten()
pol_stab = pol_stab.flatten()
reg_quality = reg_quality.flatten()

#Then we'll build a matrix with each row containing a variable
data = np.stack((gdp, exports, pol_stab, reg_quality),axis=0)

#And finally, calculate the correlation matrix
corr_matrix = np.corrcoef(x=data, rowvar=True)
print('Correlation Matrix: ', corr_matrix)

#Now, we'll plot the correlation matrix as a heat map
fig3, ax3 = plt.subplots()
im = ax3.imshow(corr_matrix)
#Set up the ticks on the plot
ax3.set_xticks(np.arange(len(corr_matrix)), labels=labels)
ax3.set_yticks(np.arange(len(corr_matrix)), labels=labels)

#Set ticks allignment
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#Create annotations on the squares of the heat map
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax3.text(j, i, round(corr_matrix[i, j],2),
                       ha="center", va="center", color="w")

fig3.suptitle("Correlation Matrix")
fig3.tight_layout()
vis_file='Correlation Matrix.png'
fig3.savefig(os.path.abspath(vis_dir+'/'+vis_file),bbox_inches='tight')