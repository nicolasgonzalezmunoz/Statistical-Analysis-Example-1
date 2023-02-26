"""
This is the third file in this example. Please, before you run this file, run the "prepare_data" file
first, and then the "data_exploration" file.
This code is dedicated to the estimation of a variable in function of some available data,
which was previously prepared and explored to do so. Here, we want to predict gdp as a function
of the other 3 variables.
"""
#Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import os

#Import clean data
data_dir = 'Clean Data'
exports = np.genfromtxt(os.path.abspath(data_dir+'/'+'Exports_Clean.csv'), delimiter=',')
gdp = np.genfromtxt(os.path.abspath(data_dir+'/'+'GDP_Clean.csv'), delimiter=',')
pol_stab = np.genfromtxt(os.path.abspath(data_dir+'/'+'Pol_Stab_Clean.csv'), delimiter=',')
reg_quality = np.genfromtxt(os.path.abspath(data_dir+'/'+'Reg_Quality_Clean.csv'), delimiter=',')

#Transform data
log_exports = np.log(exports)
log_gdp = np.log(gdp)

#Flatten arrays
log_gdp = log_gdp.flatten()
log_exports = log_exports.flatten()
pol_stab = pol_stab.flatten()
reg_quality = reg_quality.flatten()

#Initiate linear regression model
#model: log(gdp) = a_0 + a_1*log(exports) + a_2*pol_stab + a_3*reg_quality
model = LinearRegression()

#Split data into train set and test set (without stratification)
log_gdp_train, log_gdp_test, log_exports_train, log_exports_test, pol_train, pol_test, reg_train, reg_test = train_test_split(log_gdp, log_exports, pol_stab, reg_quality, train_size=0.75)
X_train = np.stack((log_exports_train, pol_train, reg_train),axis=1)
X_test = np.stack((log_exports_test, pol_test, reg_test),axis=1)

#Fit model using the test set
model.fit(X=X_train, y=log_gdp_train)
print("Model coeficients: ", model.coef_)
print("Model intercept: ", model.intercept_)
print("Model R2 coeficient on train data: ", model.score(X=X_train, y=log_gdp_train))

#The R2 coeficient of the model is 0.77, so it seems very fine to fit the train data.
#Let's see what happens with the test data
print("Model R2 coeficient on test data: ", model.score(X=X_test, y=log_gdp_test))

#The R2 for the test data is pretty similar to the train data one.
#Now let's make a cross validation to test the stability of the model
scores = cross_val_score(estimator=model, X=X_train, y=log_gdp_train, cv=7)
print("Cross-validation scores: ", scores)
#Again, the results are pretty consistent, so it seems to be a very good model.

#Let's plot the GDP data vs the error on GDP estimation
fig3, ax3 = plt.subplots()
ax3.plot(log_gdp_train, log_gdp_train-model.predict(X_train), "bo", label="Train Data")
ax3.plot(log_gdp_test, log_gdp_test-model.predict(X_test), "ro", label="Test Data")
ax3.set_xlabel("Log(GDP per capita)")
ax3.set_ylabel("Log(GDP per capita) - Estimation(Log(GDP per capita))")
fig3.suptitle("Log(GDP per capita) vs error on its prediction")
fig3.legend()
plt.tight_layout()
vis_dir = "Visualizations"
if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)

vis_file_name = "Log(GDP) Estimation Error.png"
fig3.savefig(os.path.abspath(vis_dir+"/"+vis_file_name), bbox_inches="tight")

#Now, let's finish by saving our model data
model_dir = "Model Data"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
coef_file_name = "Coef_Data.txt"
f = open(os.path.abspath(model_dir+'/'+coef_file_name),'w')
f.write("Model:\tLog(GDP_per_capita) = a_0 + a_1*Log(Exports) + a_2*Political_Stability + a_3*Regulatory_Quality\n")
f.write("Model Intercept:\ta_0 = "+ str(model.intercept_)+'\n')
f.write("Model Coeficients:\t[a_1 a_2 a_3] = "+ str(model.coef_)+'\n')
f.close()

r2_file_name =  "R2_Data.txt"
f2 = open(os.path.abspath(model_dir+'/'+r2_file_name),'w')
f2.write("Model:\tLog(GDP_per_capita) = "+str(model.intercept_)+" + "+ str(model.coef_[0]) +"*Log(Exports) + "+str(model.coef_[1])+"*Political_Stability + "+str(model.coef_[2])+"*Regulatory_Quality\n")
f2.write("Train Set R2 score = "+str(model.score(X=X_train, y=log_gdp_train))+"\n")
f2.write("Test Set R2 score = "+str(model.score(X=X_test, y=log_gdp_test))+"\n")
f2.write("Cross-Validation R2 scores = "+str(scores)+"\n")
f2.close()