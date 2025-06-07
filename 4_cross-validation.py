# improve measurement of model performance

import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "data/melb_data.csv" # filepath to the data
data = pd.read_csv(file_path) # read data and put it into a DataFrame

cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt'] # selecting columns to use
X = data[cols_to_use] # store data from columns selected in a variable

y = data.Price # select the prediction target

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

my_pipeline = Pipeline(                                                   # creating a pipeline
    steps=[
        ('preprocessor', SimpleImputer()),                                # creating a imputer to preprocess data
        ('model', RandomForestRegressor(n_estimators=50, random_state=0)) # using random forest regressor as model
    ]
)

from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error') # using cross-validation to split train and validation
# data in multiple ways (in this case 5), so all the data will at some point be used for training or validation, the score is calculated as
# the negative mean absolute error, so it's needed to mutiply by -1

print('MAE scores:\n', scores) # printing MAE scores
print('Average MAE score: ', scores.mean()) # printing the average MAE scores
