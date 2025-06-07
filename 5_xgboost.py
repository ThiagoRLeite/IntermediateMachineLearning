import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "data/melb_data.csv" # filepath to the data
data = pd.read_csv(file_path) # read data and put it into a DataFrame

cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt'] # selecting columns to use
X = data[cols_to_use] # store data from columns selected in a variable

y = data.Price # select the prediction target

X_train, X_valid, y_train, y_valid = train_test_split(X, y) # separate data into training and validation sets

from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=500, early_stopping_rounds=10, learning_rate=0.05, n_jobs=4) # using xgboost regressor
# n_estimators: how many times it will run through the modeling cycle
# early_stopping_rounds: stops iteration when model shows to not improve for specified consecutive rounds
# learning_rate: multiply prediction from each componente model before adding them to the total prediction
# n_jobs: for larger datasets, commonly the number of cores on the machine

my_model.fit(X_train, y_train,
             eval_set=[(X_valid, y_valid)], # eval_set is needed to calculate validation scores when early_stopping_round is used
             verbose=False)

from sklearn.metrics import mean_absolute_error

pred = my_model.predict(X_valid) # predicting validation data
mae = mean_absolute_error(y_valid, pred) # calculating MAE

print('MAE: ', mae)
