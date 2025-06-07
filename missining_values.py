# dealing with missing values

import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "data/melb_data.csv" # filepath to the data
data = pd.read_csv(file_path) # read data and put it into a DataFrame

y = data.Price # select the prediction target
X = data.drop(['Price'], axis=1).select_dtypes(exclude=['object']) # store the data on a variable droping the prediction target and object type columns

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0) # split data into training and validation data

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid): # function to create a model, fit, predict it and return the MAE given the training and validation data
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    return mean_absolute_error(y_valid, prediction)

## approach 1 - drop columns with missing values ##
# simple solution, but usually less accurate

cols_missing = [col for col in X_train.columns
                    if X_train[col].isnull().any()] # make a list with the cols that contains missing values
rX_train = X_train.drop(cols_missing, axis=1) # drop the missing columns from training...
rX_valid = X_valid.drop(cols_missing, axis=1) # ... and validation data
print("MAE 1: ", score_dataset(rX_train, rX_valid, y_train, y_valid)) # print MAE from approach 1

## approach 2 - imputation ##
# fill the missing cells with values based on the others in the same column

from sklearn.impute import SimpleImputer

imputer = SimpleImputer() # creating a imputer
imp_X_train = pd.DataFrame(imputer.fit_transform(X_train)) # doing imputation in the training data
imp_X_valid = pd.DataFrame(imputer.transform(X_valid)) # doing imputation in the validation data

imp_X_train.columns = X_train.columns # copying the column names from data before imputation because it removes them
imp_X_valid.columns = X_valid.columns

print("MAE 2: ", score_dataset(imp_X_train,imp_X_valid,y_train,y_valid)) # print MAE from approach 2

## approach 2 - an extension to imputation ##

X_train_plus = X_train.copy() # create a copy of the original training data
X_valid_plus = X_valid.copy() # create a copy of the original validation data

for col in cols_missing: # adding columns indicating if there was values missing and wich rows were imputed
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

imputer2 = SimpleImputer() # creating a imputer
imp_X_train_plus = pd.DataFrame(imputer2.fit_transform(X_train_plus)) # doing imputation in the training data
imp_X_valid_plus = pd.DataFrame(imputer2.transform(X_valid_plus # doing imputation in the validation data

imp_X_train_plus.columns = X_train_plus.columns # copying the column names from data before imputation because it removes them
imp_X_valid_plus.columns = X_valid_plus.columns

print("MAE 3: ", score_dataset(imp_X_train_plus, imp_X_valid_plus, y_train, y_valid)) # print MAE from approach 3
