# keep your data preprocessing and modeling code organized

import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "data/melb_data.csv" # filepath to the data
data = pd.read_csv(file_path) # read data and put it into a DataFrame


y = data.Price # select the prediction target
X = data.drop(['Price'], axis=1) # store the data on a variable droping the prediction target

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0) # split data into training and validation data

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"] # make a list containg only object type columns with a maximum of 10 unique values

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']] # make a list contaning all the numeric type columns

my_cols = categorical_cols + numerical_cols # sum the lists and store in a single one
X_train = X_train_full[my_cols].copy() # copying only the columns selected from the original training data
X_valid = X_valid_full[my_cols].copy() # copying only the columns selected from the original validation data

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy="constant") # creating a numerical transformer, a imputer using constant strategy

categorical_transformer = Pipeline(steps=[                 # creating a categorical transformer
    ('imputer', SimpleImputer(strategy='most_frequent')),  # imputer that will impute the most frequent value on the missing ones
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # one-hot encoder to allow the processing of categorical data
])

preprocessor = ColumnTransformer(                          # creating the preprocessor for all data
    transformers=[
        ('num', numerical_transformer, numerical_cols),    # using numerical transformer for numerical type columns
        ('cat', categorical_transformer, categorical_cols) # using categorical transformer for object type columns
    ]
)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0) # creating a random forest regressor model

from sklearn.metrics import mean_absolute_error

my_pypeline = Pipeline(steps=[      # creating a pipeline
    ('preprocessor', preprocessor), # first preprocessing the data
    ('model', model)                # inputing it into the model
])

my_pypeline.fit(X_train, y_train) # fitting the model
preds = my_pypeline.predict(X_valid) # predicting validation data
print('MAE: ', mean_absolute_error(y_valid, preds))
