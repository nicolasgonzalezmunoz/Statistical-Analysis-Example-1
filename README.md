# LinRegExample1
This repository shows an example of some of the most recurrent tasks when doing inference on a model.
In particular, this files produce a statistical analysis on some data downloaded from the World Bank Database.
The scripts:
- Clean and prepare the data.
- Make an exploration of the data, generating some plots in the way.
- Get the correlation matrix between the variables.
- Produce a linear regression model.
- Test the robustness of the model (dividing the data into train and test sets and doing cross-validation) and take R2 scores in several cases.
- Generate tests to validate the main hypotheis of the model, like homoscedasticity and normality of the residuals, non colinearity
of the independent variables, etc.

The code is intented to show some of the tasks the author is able to do in this particular context, so the example is designed to be simple but rather complete.
The final objective here is to see if the value of GDP per capita of a country in a certain year can be explained as a function of the other variables at the same year, assuming a ordinary linear model rather than an autoregressive one, because of the lack of enough timestamps to work in a time series.
(note that, in real life, this model can't be used for estimations since this requires express GDP per capita in a certain timestamp as a function of the other variables in PREVIOUS timestamps).

## Scripts' detail
This repository works on the data in the directory 'World Bank Data' running first the file 'Example1_1_prepare_data.py', then the script 'Example1_2_data_exploration.py'
and finnaly 'Example1_3_statistical_analysis.py' (we will call then Eample1, Example2 and Example3 for short, respectively). The file 'main_Example1.py' runs all the mentioned scripts in order, so we don't have run the files one-by-one. The scripts print some useful information in the terminal, so
we have enough information to make some conclusions.

The 'World Bank Data' directory stores the csv files that are needed for this analysis. It includes data about GDP per capita, Exports, Political Stability and Regulatory quality of several countries. We only use the timestamps and countries that are shared by all the files.

The scripts generate the directories:
- 'Clean Data': Where the data that's ready to be used is stored.
- 'Visualizations': Where all the visualizations are saved.
- 'Model Data': Where we save the coeficients and scores of the model in each of its steps.

## Method
As written in the last section, the scripts only use the data that is shared among all the data files. Despite that the each file contains
a time series, and so they probably have an autoregressive behaviour instead of a simpler linear one, because of the lack of enough timestamps (10 in total), we will omit the dependence on time of the series.

To work on the data, in the file Example1 we will first clean the data, dropping the useless columns and then drop the rows with missing data, 
ending with about 1600 instances. After that, we divide the data into a train set and a test set, with the train set with a size of about the 75% 
of the total number of instances. We save the train and test sets in separeted files, so the analysis remians consisteng.

Then in the file Example2 we load the train and test set files and begin to do some data exploration. We plot each variable vs the others, then
make some transformations on the data, so the data seems to have a more linear behaviour, and plot them again. After that, we compute the correlation 
matrix of the variables and plot it. All the plots are made only using the train set, so we don't generate any unnecessary bias.

Finally, in the file Example3 we begin with the statistical analysis and modeling. The first thing the script do is to test the colinearity of the
independent variables using the Variance Inflation Factor (VIF). If the VIF assciated to a certain variable is greater than 10, we can then
assume a high colinearity of this variable with respect of the others, which makes convinient to discard it. In this particular case, each VIF
is in fact lesser than 5, so we keep them all.

Next we fit a linear model on the train set, validate it using the cross-validation method, and test its capability to generalize using the test data, 
and compute their respective R^2 scores. We get an R^2 score of about 0.76 on the train set, 0.79 on the test set, and a mean of about 0.71. It indicates
that the model is underfitting the data a bit. Before moving to other models, we keep testing the model properties.

The first test that we apply to the model is a significance test on the indepent variables using a F statistic on the sum of squares of the residuals.
Since the p-value obtained for each variable is almost 0, so we reject the null hypothesis of H0: 'The variable is not significant', so we conclude that
each variable is significant for the model.

Then we apply a homoscedasticity test on the residuals of the model using a White test, getting a p-value of 0, and therefore rejecting the null hypothesis
H0: 'The variance of the residuals are constant', concluding that the model has a problem of heteroscedasticity. After generating a plot of the residuals
vs the target instances, we note that there are some clusters with different behaviour with respect of the residuals' variance. We try to solve first generating the clusters using the KMeans algorithm, first with a number of k=2 clusters, and then the best choice from k=1-10, in this case k=3, and then fitting a linear regression on each cluster, with no effect in the results of the test. The method for choosing k=3 in the second search of clusters wasthe criterion of the silhouette scores, choosing the k with the higher score. The additional clusters and the elbow plots are available at the Visualizations folder.

After that, we execute a test on the residuals' normality using the Shapiro-Wilk and the K-S test. In both of them we get a p-value of almost 0, so we reject the normality of the residuals, making the previous tests possibly inaccurate. We plot a Q-Q plot and a historam of the residuals, comparing them
with random data generated with a normal distribution of mean and variance equal to the ones of the residuals. The plots show the existence of very few 
outliers, and a bias of the residuals to 0. The proportion of outliers with respect of the total size of the data is of  about 0.028%.

To finish the analysis, we generate several polynomial models and compute their R^2 scores, obtaining that the most balanced models are the the ones of
degree 1 and 2. Unfortunately, the quadratic model seems to overfit the data, and it also have more variance on the validation scores. Taking all that in consideration, we decide to choose the linear model over the quadratic one.

## Modules used

Note that to execute several rutinary tasks, like the significance test and the Q-Q plot, there are two modules included with the scripts that contain 
several useful functions. They are named 'model_selection' and 'statistical_plots'. As their names indicate, they are modules focused on model selection and plots, respectively. Each module and function are documented, so you can read it in the respective module for more information. In particular, they have the following functions:
### In model_selection

- get_polynomial_regression_scores: fits polynomial features (if any) to a linear model. This linear model can be any regressor class from the sklearn's 
linear_model package that supports the fit and predict methods. It fits polynomial features with a degree in the range given by the user, and returns the
scores for each iteration.
- get_KMeans_scores: similar to get_polynomial_regression_scores, it iterates over a range of n_clusters given by the user, and returns the inertia, silhouette scores and silhouette samples for each iteration. It also returns a homogeneity score that measures the homogeneity in the distribution of the
clusters, taken as the variance of the number of instances per cluster minus the mean number of instances per cluster.
- get_VIF: Gets the Variance Inflation Factor of each feature.
- feature_significance: Takes a linear regressor model and the data as arguments (and optionaly a transformer), and returns a dictionary of nested dictionaries with information about the F tests performed. It has the capability of doing this task recursively, generating new linear models while dropping the the not significant features and testing the remaining ones.
- model_significance: Takes two different models and compares them, getting which is more complex and if it's statistically significant compared to the simpler one.
- white_test: Performs a White test for homoscedasticity.
- bptest: Performs a Breusch-Pagan test for homoscedasticity.

### In statistical_plots

- get_plot_grid: Given the number of desired axes, it returns the n_rows and n_cols such that n_rows*n_cols=n_axes, n_rows<=n_cols and the quantity n_rows and n_cols is minimized.
- get_figsize: given a figsize, it computes a new figure width and heigh such that each axis a size equal to the original figsize.
- plot_one_vs_each: Takes a matrix of features' instances and an index to use as pivot, then plots the feature at the pivot columns and plots it vs the others.
- plot_each_vs_each: Takes a matrix of features' instances and plots each feature vs the others.
- plot_correlation_matrix: Given a correlation matrix and its labels, it plots the matrix as a color map, annotating each cell value in the respective plot's cell.
- qq_plot: Plots a Q-Q plot of the given data vs a random array, generated from a normal distribution. It can optionally the normal distribution quantiles to better compare the data with the distribution.
- plot_histogram: Plots a histogram with the data, and optionally can plot the normal density function over it.
- plot_scores: Given an array of scores from several machine learning models, the function plots them. It also can plot the test and validation scores if given.
- plot_cross_val_scores: Given an array of cross-validation scores from several models, it plots them. It can also optionally plot the mean of the scores.
- plot_silhouette_samples: Given the silhoute samples of one or more KMeans clusters, it generates a knife plot of the samples.
- plot_clusters: Plots each cluster of a clustering model. It optionally can plot the centroids, if given.

Note: Each plot is scaled so that each axes have the desired size. Also a grid is added, and when an axis only accepts a certain type of data, like the degree of a polynomial regression, the ticks are modified so only the necessary info is shown.

## Conclusions
Though this is a toy example, we still want to extract some conclusions, though they may seem natural taking into consideration the limitations of the 
data sets.
- First, the linear model seems to be very good to generalize data. Despite that, it tends to ignore some data, like shown in the cross-validation scores.
In general, it behaves better in this aspect than polynomial models, that tend to overfit data. The overfitting, however, could be solved by regularizing
the quadratic model by, for example, putting some weights to the samples.
- Second, since the linear model failed the homoscedasticity test, the ordinary linear model is no more the best linear unbiased estimator (BLUE), and then
it should be modified to retain its good properties. This possibly implies imposing weights on the data related to the residuals, which could be made
using the statsmodels Python package.
- Third, since the linear model also failed the normality test on the residuals, the other tests performed could be inaccurate. However, depending in the 
number of instances (in this case, N~1610), this could be ignored because the Central Limit Theorem (CLT) affirms that for a enough number of instances, 
the distribution of the residuals approaches a normal distribution. Despite that, the concept of "great number of instances" is a bit ambiguous. Also, the
CLT imposses that the residuals should have bounded mean, which in this case it could be untrue, since the errors show a growing tendency in some cases. In this point we should proceed with caution.
- Finally, remember that we are working with time series, and that we omitted this property because the lack of enough timestamps. The issues mentioned above could be produced because of a mispecification in the behaviour of the data. We modeled it via an ordinary linear regression, but because of the temporal component, it could be better to adjust a autoregressive model (like an ARIMA model).
