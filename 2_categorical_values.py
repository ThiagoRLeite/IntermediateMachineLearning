# dealing with categorical values

import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "data/melb_data.csv" # filepath to the data
data = pd.read_csv(file_path) # read data and put it into a DataFrame

y = data.Price # select the prediction target
X = data.drop(['Price'], axis=1).select_dtypes(exclude=['object']) # store the data on a variable droping the prediction target


X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0) # split data into training and validation data

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid # function to create a model, fit, predict it and return the MAE given the training and validation data
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

cols_missing = [col for col in X_train_full.columns
                    if X_train_full[col].isnull().any()] # make a list with the cols containing missing values

X_train_full.drop(cols_missing, axis=1, inplace=True) # drop columns missing to simplify
X_valid_full.drop(cols_missing, axis=1, inplace=True)

low_card_cols = [cname for cname in X_train_full.columns
                    if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"] # make a list containg only object type columns with a maximum of 10 unique values

numeric_cols = [cname for cname in X_train_full.columns
                    if X_train_full[cname].dtype in ['int64', 'float64']] # make a list contaning all the numeric type columns

cols = low_card_cols + numeric_cols # sum the lists and store in a single one
X_train = X_train_full[cols].copy() # copying only the columns selected from the original training data
X_valid = X_valid_full[cols].copy() # copying only the columns selected from the original validation data

s = (X_train.dtypes == 'object') # create a boolean series indicating if the column of training data is object type or not
ob_cols = list(s[s].index) # create a list only with the names of categorical columns

## approach 1 - drop columns with object type ##
# simple solution, but usually less accurate

drop_X_train = X_train.select_dtypes(exclude=['object']) # drop the object columns from training...
drop_X_valid = X_valid.select_dtypes(exclude=['object']) # ... and validation data

print("MAE 1: ", score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)) # print MAE from approach 1

## approach 2 - ordinal encoder ##
# assign each unique value a different integer, works better with variables where there is a clear ranking between unique values (ordinal variables)

from sklearn.preprocessing import OrdinalEncoder

label_X_train = X_train.copy() # create a copy to avoid changing original data
label_X_valid = X_valid.copy()

ordinal = OrdinalEncoder() # create a ordinal encoder
label_X_train[ob_cols] = ordinal.fit_transform(X_train[ob_cols]) # appy ordinal encoder to each categorical type column
label_X_valid[ob_cols] = ordinal.transform(X_valid[ob_cols])

print("MAE 2: ", score_dataset(label_X_train, label_X_valid, y_train, y_valid)) # print MAE from approach 2

## approach 3 - one-hot encoder ##
# create a column for each unique categorical value and assing boolean values indicating wether that value was on the row or not

from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # create a one-hot encoder
oh_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[ob_cols])) # appy it to each column with categorical data
oh_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[ob_cols]))

oh_cols_train.index = X_train.index # putting index back because one-hot encoder removes it
oh_cols_valid.index = X_valid.index

num_X_train = X_train.drop(ob_cols, axis=1) # remove the categorical columns (one-hot encoding replace them)
num_X_valid = X_valid.drop(ob_cols, axis=1)

oh_X_train = pd.concat([oh_cols_train, num_X_train], axis=1) # concatenate one-hot encoding and numerical columns
oh_X_valid = pd.concat([oh_cols_valid, num_X_valid], axis=1)

oh_X_train.columns = oh_X_train.columns.astype(str) # ensure all column names are string type
oh_X_valid.columns = oh_X_valid.columns.astype(str)


print("MAE 3: ", score_dataset(oh_X_train, oh_X_valid, y_train, y_valid)) # print MAE from approach 3
