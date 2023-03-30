"""
This script is focused on linear regression and statistical analysis.

First, we set the random seed for replicability and load the data, including
the train and test sets, performing some transformations on it. Then, we
create a linear model with sklearn, fit it, calculate its scores and print
some interesting attributes of the model.  We perform a normality test on the
errors and a significance test on a paramter of the model.

Since the model fails the normality test, we try with a polynomial model.
We create some polynomial features and fit the model again, but with no better
results on the normality test. We create some visualizations on the
errors to see what is happening.

-Conclusion:

The linear model seems to be the better model, since the polynomial model
tends to overfit the data.  We obtain an R2 score of about 0.75, with
similar results on the score for the test set and for cross-validation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import shapiro, kstest

from model_selection import get_polynomial_regression_scores, feature_significance, get_VIF, white_test, get_KMeans_scores
from statistical_plots import qq_plot, plot_histogram, plot_scores, plot_cross_val_scores, plot_clusters

np.random.seed(42)

# Import clean data
data_dir = 'Clean Data'
exports = np.genfromtxt(os.path.abspath(data_dir+'/'+'Exports_Clean.csv'), delimiter=',')
gdp = np.genfromtxt(os.path.abspath(data_dir+'/'+'GDP_Clean.csv'), delimiter=',')
pol_stab = np.genfromtxt(os.path.abspath(data_dir+'/'+'Pol_Stab_Clean.csv'), delimiter=',')
reg_quality = np.genfromtxt(os.path.abspath(data_dir+'/'+'Reg_Quality_Clean.csv'), delimiter=',')
train_data = np.genfromtxt(os.path.abspath(data_dir+'/'+'Train_data.csv'), delimiter=',')
test_data = np.genfromtxt(os.path.abspath(data_dir+'/'+'Test_data.csv'), delimiter=',')

# Transform data
log_exports = np.log(exports)
log_gdp = np.log(gdp)

# Flatten arrays
log_gdp = log_gdp.flatten()
log_exports = log_exports.flatten()
pol_stab = pol_stab.flatten()
reg_quality = reg_quality.flatten()

# Split data into train set and test set (without stratification)
X = np.stack((log_exports, pol_stab, reg_quality),axis=1)
X_train = train_data[:,1:]
X_test = test_data[:,1:]
log_gdp_train = train_data[:,0]
log_gdp_test = test_data[:,0]

# We begin with a test on independent variables linear dependence
vif = get_VIF(X)
print('\nResults on Variance Inflation Factor:')
print(vif, '\n')

# Each variable has a VIF lesser than 5, so we can consider them as orthogonal
# and keep them for regression

# Initiate linear regression model and fit it with train data
# model: log(gdp) = a_0 + a_1*log(exports) + a_2*pol_stab + a_3*reg_quality
model = LinearRegression()
model.fit(X=X_train, y=log_gdp_train)

# Compute R2 scores
R2_train_score = model.score(X=X_train, y=log_gdp_train)
R2_test_score = model.score(X=X_test, y=log_gdp_test)
scores = cross_val_score(estimator=model, X=X_train, y=log_gdp_train, cv=7)

# Print the results
print("Linear model info:")
print("Linear model coeficients: ", model.coef_)
print("Linear model intercept: ", model.intercept_)
print("Linear model R2 coeficient on train data: ", R2_train_score)
print("Model R2 coeficient on test data: ", R2_test_score)
print("Cross-validation scores: ", scores, '\n')

# The R2 coeficient using the training set is 0.77, which seems fine
# considering the amount of features used.
# 
# The R2 for the test data is pretty similar to the one of the train data.
# This is also true for the R2 scores for the cross-validation. It seems
# like a good model.

# Now, let's finish by saving our model data
model_dir = "Model Data"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
coef_file_name = "Linear_Model_Coef_Data.csv"
coef_file = open(os.path.abspath(model_dir+'/'+coef_file_name),'w')
for i in np.arange(model.coef_.size+1):
    coef_file.write('a'+str(i))
    if i < model.coef_.size:
        coef_file.write(',')
coef_file.write('\n')
for i in np.arange(model.coef_.size+1):
    if i == 0:
        coef_file.write(str(model.intercept_))
    else:
        coef_file.write(str(model.coef_[i-1]))
    if i < model.coef_.size:
        coef_file.write(',')
coef_file.write('\n')
coef_file.close()

r2_file_name = "Linear_Model_R2_Data.txt"
r2_file = open(os.path.abspath(model_dir+'/'+r2_file_name),'w')
r2_file.write("Model:\tLog(GDP_per_capita) = "+str(model.intercept_)+" + "+ str(model.coef_[0]) +"*Log(Exports) + "+str(model.coef_[1])+"*Political_Stability + "+str(model.coef_[2])+"*Regulatory_Quality\n")
r2_file.write("Train Set R2 score = "+str(model.score(X=X_train, y=log_gdp_train))+"\n")
r2_file.write("Test Set R2 score = "+str(model.score(X=X_test, y=log_gdp_test))+"\n")
r2_file.write("Cross-Validation R2 scores = "+str(scores)+"\n")
r2_file.close()

# We now perform with a significance test
# Make a restricted model
restrict_model = LinearRegression()
X_rest_train = X_train[:,:2]
restrict_model.fit(X=X_rest_train, y=log_gdp_train)

# Make predictions on train data
y_pred_train = model.predict(X_train)
y_rest_pred_train = restrict_model.predict(X_rest_train)

# Make F test
print('Results of the significant test:')
significance = feature_significance(LinearRegression(), X_train, log_gdp_train, test_features=[-1, 0, 1, 2], transformer=None, iterative=True, recursive=True, confidence=0.95)
print(significance, '\n')

# Using the F-test, we are highly confident that all the features (including
# the intercept) are statistically significant

# We now test homoscedasticity of the residuals
print('Results on homoscedasticity test:')
res = log_gdp - model.predict(X)
res_train = log_gdp_train - model.predict(X_train)
res_test = log_gdp_test - model.predict(X_test)
stat, p_value = white_test(res, X)
print('\n')

# The p-value for the homoscedasticity test is almost , so we conclude the
# variance of the errors are not homoscedastic. We will make some plots to
# see what is happening, but first, we will standarize it so any tendence
# becomes more evident.
res_mean = np.mean(res)
res_std = np.std(res)
std_res = (res - res_mean)/res_std
std_res_train = (res_train - res_mean)/res_std
std_res_test = (res_test - res_mean)/res_std

# Initialize the visualizations directory
vis_dir = 'Visualizations'
if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)

# Let's see if the residuals have a linear behaviour as a function of
# the log(GDP per capita)
res_model = LinearRegression()
res_model.fit(X=log_gdp_train.reshape(-1, 1), y=std_res_train)
std_pred_res = res_model.predict(X=log_gdp_train.reshape(-1, 1))
res_R2 = res_model.score(X=log_gdp_train.reshape(-1, 1), y=std_res_train)
print('R2 on errors: ', res_R2)
pred_res_test = res_model.predict(X=log_gdp_test.reshape(-1, 1))
res_R2_test = res_model.score(X=log_gdp_test.reshape(-1, 1), y=std_res_test)
print('R2 on errors on test data: ', res_R2_test, '\n')

# It seems they are mostly non linear with respect of log(GDP per capita).

# Now we will plot the residuals to search for changes in variance
fig, ax = plt.subplots()
ax.plot(log_gdp_train, res_train, 'bo')
ax.set_xlabel('log(GDP per capita)')
ax.set_ylabel('Residuals')
fig.suptitle('oficial data vs residuals')
homoscedasticity_file_name = 'Homoscedasticity.png'
fig.savefig(os.path.abspath(vis_dir+"/"+homoscedasticity_file_name), bbox_inches="tight")

# The plot shows that the the variance on errors decreases at the end,
# while following a non linear behaviour with respect of the 
# log(GDP per capita), where it grows until it stabilizes at a certain
# value of log(GDP per capita) (at about log(GDP per capita) == 8.2).

# Let's try to cluster the residual using KMeans. We aim to divide the data
# into 2 clusters.
data_to_cluster = np.concatenate((log_gdp.reshape(-1, 1), res.reshape(-1, 1)), axis=1)
data_to_cluster_train = np.concatenate((log_gdp_train.reshape(-1, 1), res_train.reshape(-1, 1)), axis=1)
data_to_cluster_test = np.concatenate((log_gdp_test.reshape(-1, 1), res_test.reshape(-1, 1)), axis=1)
std_scal = StandardScaler()
std_scal.fit(data_to_cluster)
transform_data_train = std_scal.transform(data_to_cluster_train)
transform_data_test = std_scal.transform(data_to_cluster_test)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(transform_data_train)
labels_train = kmeans.predict(transform_data_train)
labels_test = kmeans.predict(transform_data_test)
centroids = kmeans.cluster_centers_

unique_labels = np.unique(labels_train)
for label in unique_labels:
    label_index_train = labels_train == label
    label_index_test = labels_test == label
    X_cluster_train = log_gdp_train[label_index_train].reshape(-1, 1)
    X_cluster_test = log_gdp_test[label_index_test].reshape(-1, 1)
    y_cluster_train = std_res_train[label_index_train]
    y_cluster_test = std_res_test[label_index_test]
    res_model = LinearRegression()
    res_model.fit(X=X_cluster_train, y=y_cluster_train)
    std_pred_res = res_model.predict(X=X_cluster_train)
    res_R2 = res_model.score(X=X_cluster_train, y=y_cluster_train)
    print('R2 on errors for clustered linear regression, cluster ', label, ': ', res_R2)
    pred_res_test = res_model.predict(X=X_cluster_test)
    res_R2_test = res_model.score(X=X_cluster_test, y=y_cluster_test)
    print('R2 on errors on test data for clustered linear regression, cluster ', label, ': ', res_R2_test, '\n')

# We now will plot the results
fig_suptitle = 'Residuals clustering'
x_label = 'log(GDP per capita)'
y_label = 'residuals'
fig, ax = plot_clusters(transform_data_train, labels_train, suptitle=fig_suptitle, x_labels=[x_label], y_label=y_label, centroids=centroids)
clust_file_name = 'Residuals_Clustering.png'
fig.savefig(os.path.abspath(vis_dir+"/"+clust_file_name), bbox_inches="tight")

# It seems like we got very good clusters, but the central area is still
# not convincing. Let's see what happens with more clusters.
inertia, sil_scores, sil_test_scores, sil_samples, sil_test_samples, homogeneity_scores, homogeneity_test_scores, labels_array, centroids_list = get_KMeans_scores(data_to_cluster_train, X_test=data_to_cluster_test, min_n_clusters=1, max_n_clusters=10, random_state=42)
fig_suptitle = 'Elbow plot from residual clustering'
fig, ax = plot_scores(inertia, x_label=['Number of clusters'], y_label='Inertia', suptitle=fig_suptitle, styles=['bo-'])
fig_file_name = 'Elbow_residual.png'
fig.savefig(os.path.abspath(vis_dir+"/"+fig_file_name), bbox_inches="tight")

fig_suptitle = 'Plot of silhouette scores from residual clustering'
fig, ax = plot_scores(sil_scores, x_label=['Number of clusters'], y_label='Silhouette scores', suptitle=fig_suptitle, styles=['bo-'])
fig_file_name = 'Silhouette_scores_residual.png'
fig.savefig(os.path.abspath(vis_dir+"/"+fig_file_name), bbox_inches="tight")

# The best clustering is obtained when k = 3.
std_scal = StandardScaler()
std_scal.fit(data_to_cluster)
transform_data_train = std_scal.transform(data_to_cluster_train)
transform_data_test = std_scal.transform(data_to_cluster_test)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(transform_data_train)
labels_train = kmeans.predict(transform_data_train)
labels_test = kmeans.predict(transform_data_test)
centroids = kmeans.cluster_centers_

# We now will plot the results
fig_suptitle = 'Residuals clustering with k = 3'
x_label = 'log(GDP per capita)'
y_label = 'residuals'
fig, ax = plot_clusters(transform_data_train, labels_train, suptitle=fig_suptitle, x_labels=[x_label], y_label=y_label, centroids=centroids)
clust_file_name = 'Residuals_Clustering_Best_k.png'
fig.savefig(os.path.abspath(vis_dir+"/"+clust_file_name), bbox_inches="tight")

# It looks much better. Now we will make a linear regression again.
unique_labels = np.unique(labels_train)
for label in unique_labels:
    label_index_train = labels_train == label
    label_index_test = labels_test == label
    X_cluster_train = log_gdp_train[label_index_train].reshape(-1, 1)
    X_cluster_test = log_gdp_test[label_index_test].reshape(-1, 1)
    y_cluster_train = std_res_train[label_index_train]
    y_cluster_test = std_res_test[label_index_test]
    res_model = LinearRegression()
    res_model.fit(X=X_cluster_train, y=y_cluster_train)
    std_pred_res = res_model.predict(X=X_cluster_train)
    res_R2 = res_model.score(X=X_cluster_train, y=y_cluster_train)
    print('R2 on errors for  best clustering linear regression, cluster ', label, ': ', res_R2)
    pred_res_test = res_model.predict(X=X_cluster_test)
    res_R2_test = res_model.score(X=X_cluster_test, y=y_cluster_test)
    print('R2 on errors on test data for best clustering linear regression, cluster ', label, ': ', res_R2_test, '\n')

# The model only seems better for the third cluster, and only for the test set.
# The heteroscedasticity problem may be produced because the data follows
# an autoregressive behaviour instead of a linear one like we are assuming here.

# Now, perform some  a normality test on the errors (Shapiro-Wilk test)
print('Results of the normality tests on the residuals:')
shap = shapiro(res)
print('\nShapiro-Wilk results: (statistic=', shap[0], ' pvalue=', shap[1], ')')
ks = kstest(res, 'norm')
print(ks,'\n')

# Unfortunately, the errors are not normal with a very high probability.
# This is surely because the data was taken from time series, so the
# relationship between them is probably not linear, but autoregressive.

# Let's plot the errors q-q plot
fig_suptitle = 'Errors Q-Q plot'
fig_qq, ax_qq = qq_plot(res, suptitle=fig_suptitle)
errors_file_name = 'Errors_distribution.png'
fig_qq.savefig(os.path.abspath(vis_dir+"/"+errors_file_name), bbox_inches="tight")

# Plot errors histogram
fig_suptitle = 'Errors histogram'
fig_hist, ax_hist = plot_histogram(res, suptitle=fig_suptitle)
hist_file_name = 'Errors_histogram.png'
fig_hist.savefig(os.path.abspath(vis_dir+"/"+hist_file_name), bbox_inches="tight")

# From the QQ-plot we can see that the erros deviates from the normal
# distribution for values different from zero. Then, in the histogram
# we can see that the errors distribution  is way denser than a normal
# distribution around zero. Then, we expect that there is a very low
# amount of outliers, but the tests computed before may be not very
# accurate.

# Now let's see the outliers wirh the intequartile range criterion
Q1, Q3 = np.percentile(res, [25, 75])
IQR = Q3-Q1
bottom_range = Q1-1.5*IQR
top_range = Q3+1.5*IQR
print('Outliers info:\n')
print('Number of superior outliers', res[res>top_range].size)
print('Number of inferior outliers', res[res<bottom_range].size)
print('Superior outliers on errors: ', res[res>top_range].size/res.size, '%')
print('Inferior outliers on errors: ', res[res<bottom_range].size/res.size, '%\n')

# It seems there are only a few outliers compared to the total size of
# the data, as expected

#Now we compute several polynomial regressions and get their scores
R2_train_scores, R2_test_scores, R2_cv_scores = get_polynomial_regression_scores(X_train, log_gdp_train, X_test, log_gdp_test, max_degree=10, cv=7)

print('Polynomial model info:\n')
print('Polynomial R2 scores on train data: ', R2_train_scores)
print('Polynomial R2 scores on test data: ', R2_test_scores)
print('Polynomial R2 scores on cross-validation data:\n', R2_cv_scores,'\n')

# Plot the results on scores
fig_suptitle = 'R2 scores by degree on polynomial regression'
fig_poly_R2, ax_poly_R2 = plot_scores(R2_train_scores, test_score=R2_test_scores, cross_val_score=R2_cv_scores, suptitle=fig_suptitle)
R2_scores_file_name = 'Poly_scores.png'
fig_poly_R2.savefig(os.path.abspath(vis_dir+"/"+R2_scores_file_name), bbox_inches="tight")

fig_suptitle = 'Cross-validation R2 scores by degree on polynomial regression'
fig_poly_cv, ax_poly_cv = plot_cross_val_scores(R2_cv_scores, suptitle=fig_suptitle)
cv_scores_file_name = 'Poly_cv_scores.png'
fig_poly_cv.savefig(os.path.abspath(vis_dir+"/"+cv_scores_file_name), bbox_inches="tight")

# It seems that the best polynomial regression is the one with degree=1,
# that is, our typical linear model. At least, the relationship between
# the target and the independent variables isn't polynomial.