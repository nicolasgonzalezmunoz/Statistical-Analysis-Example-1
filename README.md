# LinRegExample1
This repository shows an example of some of the tasks that I'm used to do in a professional context.
In particular, this files produce an statistical analysis on some data downloaded from the World Bank Database.
The scripts:
- Clean and prepare the data.
- Make an exploration of the data, generating some plots in the way.
- Get the correlation matrix between the variables.
- Produce a linear regression model.
- Test the robustness of the model (dividing the data into train and test sets and doing cross-validation) and take R2 scores in several cases.

The code is intented to show the skills of the author in this particular context, so the example taken is designed to be simple but rather complete. 
Our final objective here is to see if the value of GDP per capita of a country in a certain year can be explained as a function of the other variables at the same year
(note that, in real life, this model can't be used for estimations since this requires express GDP per capita in a certain year as a function of the other variables in PREVIOUS years).

## How the code works
This repository works on the data in the directory 'World Bank Data' running first the file 'Example1_1_prepare_data.py', then the script 'Example1_2_data_exploration.py'
and finnaly 'Example1_3_statistical_analysis.py'. The file 'main_Example1.py' runs all the mentioned scripts in order, so we don't have run the files one-by-one.

The 'World Bank Data' directory stores the csv files that are needed for this analysis. It includes data about GDP per capita, Exports, Political Stability and Regulatory quality.

The scripts generate the directories:
- 'Clean Data': Where the data that's ready to be used is stored.
- 'Visualizations': Where all the visualizations are saved.
- 'Model Data': Where we save the coeficients and scores of the model in each of its steps.

## Conclusions
Though this is a toy example, we still can extract some interesting (and maybe known) conclusions. 
- First, the model seems to be very accurate and robust. We get a high and consistent R2 score with the train set, the test set and the cross-validation sets (in each case of about 0.76).
- Second, though it is a known fact, the regulatory quality is highly correlated to the log of GDP per capita, so according to this model, if a country wants to strengthen its economy,
it may be useful to focus on improving the regulatory quality if this index is low.
- Also, it seems to be very effcient to search for ways to improve political stability if you want to improve your economy (again when the value of this index is low).
- Finally, despite less efficient than the other two methods, it seems to be a good ides to encourage exports from your country if you want your economy to grow (in some cases, it may lead to other, maybe undesirable, effects on the economy, though).
