import numpy as np
import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

#READ ME
#This code trains a random forest model based on field, work-experience, and geopraphical region
#To use this for a prediction task, simply import this file and rs.predict (your datas)
#This model perform better when used for non-linear data.

#ALL THE INCOMES ARE LISTED IN USD
income_map = {
    "alabama": 28432,
    "alaska": 40752,
    "arizona": 32751,
    "arkansas": 26981,
    "california": 38574,
    "colorado": 40454,
    "connecticut": 43183,
    "delaware": 37469,
    "florida": 30734,
    "georgia": 32176,
    "hawaii": 38545,
    "idaho": 30628,
    "illinois": 36554,
    "indiana": 30543,
    "iowa": 32572,
    "kansas": 32556,
    "kentucky": 28241,
    "louisiana": 28891,
    "maine": 32245,
    "maryland": 44310,
    "massachusetts": 46176,
    "michigan": 32618,
    "minnesota": 39647,
    "mississippi": 24346,
    "missouri": 30713,
    "montana": 31963,
    "nebraska": 34563,
    "nevada": 33726,
    "new hampshire": 39586,
    "new jersey": 43523,
    "new mexico": 27179,
    "new york": 37715,
    "north carolina": 31546,
    "north dakota": 35014,
    "ohio": 32256,
    "oklahoma": 29178,
    "oregon": 36423,
    "pennsylvania": 35728,
    "rhode island": 36824,
    "south carolina": 30187,
    "south dakota": 32757,
    "tennessee": 29994,
    "texas": 32318,
    "utah": 34223,
    "vermont": 35544,
    "virginia": 39111,
    "washington": 40229,
    "west virginia": 27394,
    "wisconsin": 32609,
    "wyoming": 33584,
    "alberta": int(49900 * 0.74),  # $36,926
    "british columbia": int(43400 * 0.74),  # $32,116
    "manitoba": int(40700 * 0.74),  # $30,118
    "new brunswick": int(38000 * 0.74),  # $28,120
    "newfoundland and labrador": int(39700 * 0.74),  # $29,378
    "nova scotia": int(37300 * 0.74),  # $27,602
    "ontario": int(44800 * 0.74),  # $33,152
    "prince edward island": int(38600 * 0.74),  # $28,564
    "quebec": int(38800 * 0.74),  # $28,712
    "saskatchewan": int(44000 * 0.74),  # $32,560
    "northwest territories": int(57000 * 0.74),  # $42,180
    "nunavut": int(58400 * 0.74),  # $43,416
    "yukon": int(52700 * 0.74)  # $39,998
}

# Load data from CSV
data = pd.read_csv("C:\\Users\\17789\\Downloads\\Experience-Salary.csv")

# Map the 'location' column in the CSV to the corresponding median income
data['median_income'] = data['location'].map(income_map)

# Fill NaN values with the default median income of 35000 (I chose this arbitrarily, maybe use US national median income?)
data['median_income'].fillna(35000, inplace=True)

# Fill missing values in the 'field' column with 'others'
data['field'].fillna('others', inplace=True)

# One-Hot Encode the 'field' column
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity
field_encoded = encoder.fit_transform(data[['field']])

# Combine 'work experience', 'median_income', and the one-hot encoded 'field' into one feature matrix
X = np.column_stack((data['work experience'].values, data['median_income'].values, field_encoded))
y = data['salary'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [5, 10, 25, 50, 75, 100, 150],
    'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],
    'min_samples_split': [2, 5, 6, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Use the best model to predict on the test set
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

# Evaluate the best model
best_mse = mean_squared_error(y_test, y_pred_best)
Stdeviation = (int)(math.sqrt(best_mse))

average_dif = 0
for i in range(len(y_test)):
    average_dif += abs(y_test[i]-y_pred_best[i])
    
print(f"average absolute difference with best model: {average_dif/len(y_test)}")
print(f"Mean Squared Error with best model: {best_mse}")
print(f"Predicted values with best model: {y_pred_best}")
