# LinRegExample1
This repository shows an example of some of the most recurrent tasks when doing inference on a model. In particular, this files produce a statistical analysis on some data downloaded from the World Bank Database. The scripts:

    Clean and prepare the data.
    Make an exploration of the data, generating some plots in the way.
    Get the correlation matrix between the variables.
    Produce a linear regression model and an ARIMA model.
    Test the robustness of the linear model (dividing the data into train and test sets and doing cross-validation) and take R2 scores in several cases.
    Generate tests to validate the main hypotheis of the model, like homoscedasticity and normality of the residuals, non colinearity of the independent variables, etc.

The code is intented to show some of the tasks the author is able to do in this particular context, so the example is designed to be simple but rather complete. The final objective here is to see if the value of GDP per capita of a country in a certain year can be explained as a function of the other variables at the same year (using linear regression models), or as an autoregressive porcess (using an ARIMA model).

The analysis was built mainly using pandas, seaborn and statsmodels, unlike a previous version of this same repository that was built mainly using numpy, matplotlib and scikit-learn.

# Scripts' detail

This repository works on the data in the directory 'World Bank Data' running the "LinearRegression.ipynb" jupyter notebook. The file exports the processed data into the "Processed Data" folder, as an ETL excercise. The exported data is ready-to-use for analysis, and is exported using data modelling techniques to save as much space as possible, in a SQL database-like structure.

The 'World Bank Data' directory stores the csv files that are needed for this analysis. It includes data about GDP per capita, Exports, Political Stability and Regulatory quality of several countries. We only use the timestamps and countries that are shared by all the files.

All the plots, analysis and conclusions are self-contained into the notebook, so interested people can choose to open directly the notebook or read this readme file.

# Method

As written before, onlythe data that is shared among all the data files is used for analysis. I begin by importing and cleaning the data, dropping unnecesary columns and rows with NaN values directly (this rows are prone to be highly distorted in this dataset). Then, I make a inner join to get the countries that are shared among all the dataframes, and drop the ones who are not. Exploratory analysis is performed to search for outliers, check the correlation between variables, make some feature engineering, and make the necessary transformations. After all the cleansing and exploration, the dataframes are transformed into the long format and merged together. The new, unified dataframe is splitted into a train set and a test set.

Next a linear model is fitted on the train set using the statsmodels package, performing some hypotheses tests, in which the ordinary least squares (OLS) model fails the homoscedasticity and normality tests, trying a General Least Squares (GLS) model instead, based on the residuals obtained from the OLS model. Despite this model being designed to be heteroscedastic-consistent, it fails again both of the tests mentioned before, so I moved on to an ARIMA model.

For the ARIMA model, I used the mean ln(GDP per capita) time series, since the model has to be trained on a single time series. The last 10 timestamps of the said series are taken apart for testing purposes. The model's parameters are determined via differencing the time series until it becomes stationary, which is tested using hypotheses tests, and by analysing the autocorrelation function (AUC) and partial autocorrelation function (PAUC) of the stationarized series. The model is then fitted and tested, getting excelent results.

Taking into consideration the limitations of the statsmodels package, the OLS and GLS models are implemented again, this time using scikit-learn. The models are fitted and tested, getting decent results.

# Conclusions

Though this is a toy example, we still want to extract some conclusions, though they may seem natural taking into consideration the limitations of the data sets.

    The ARIMA model was the best by far in every aspect. It was better at generalization, had an excelent fit and was statistically well-behaved. However, the limitations on the statsmodels package makes it hard to implement this model on a pipeline.
    The best model between the OLS and GLS was the GLS model. Despite it is built from the OLS residuals, it had a better generalization capability and it isn't underfitted nor overfitted, although it could perform better with more data. The OLS model only wins over GLS on R2 score (and just by a little bit).
So, the ARIMA was the best model for this task, but GLS is the best model that can be included into a machine learning pipeline.

