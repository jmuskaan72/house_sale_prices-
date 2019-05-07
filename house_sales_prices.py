# -*- coding: utf-8 -*-
""" 
Spyder Editor

This is a temporary script file.
"""     
     
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
 
#PART 1:DATA PREPROCESSING       
#importing the dataset  
dataset=pd.read_csv('house_sales.csv')        
dataset.columns
y=dataset.iloc[:, 80].values    
features = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr', 'TotRmsAbvGrd']
X=dataset[features]
X.head()
  
#missing data
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN' ,strategy='median', axis=0)
imputer=imputer.fit(y.reshape(-1,1))
z=y.reshape(-1,1)
z=imputer.transform(z) 
y=z

#splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train ,y_test= train_test_split(X,y, test_size=0.5, random_state=1)

"""#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)""" 

#PART-2 FITIING A SPECIFIC MODEL TO TRAIN SET
# Fitting Decision Tree Regression to the train dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(max_leaf_nodes =None, random_state =1)
regressor.fit(X_train, y_train)

#part -3 MODEL VALIDATION
#set predictions with the train data and checking error 
y_test_pred = regressor.predict(X_test)
print(y_test_pred)
"""MAE  with no max leaf nodes"""
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_test_pred)

#PART-4 underfiitng or overfitting
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(max_leaf_nodes =50, random_state =1)
regressor.fit(X_train, y_train)
y_test_pred = regressor.predict(X_test)
"""MAE with max leaf nodes"""
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_test_pred)

