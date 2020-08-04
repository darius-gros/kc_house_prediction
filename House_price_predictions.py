# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:52:37 2020

@author: dariu
1. Frame problem
Predict house prices: Regression problem
Supervised algorithm
Performance measure: RMSE, MAE
Task: predict exact price of house given features.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

housing = pd.read_csv('kc_house_data.csv')
head = housing.head()

#condition, grade ordinal data
#date, yr_built, yr_renovated, zipcode
#lat, long

housing.info
housing.dtypes #only one object date
housing['date']
housing['date'].value_counts() #many different dates
descript = housing.describe()

housing.hist(bins = 50)
plt.figure(figsize = (40,20))
plt.show()

housing['price'].hist(bins = 20)
plt.show() #few values above 200 000

#stratified sampling to get accuracte test set /training set split
housing['price_cat'] = np.ceil(housing['price']/300000)
housing['price_cat'].where(housing['price_cat']<6, 6.0, inplace = True) #error in stratified sampling if 6.0 not mentioned
housing['price_cat'].hist(bins = 20)
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['price_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
#get percentage
housing['price_cat'].value_counts()/len(housing)

#remove price_cat to get back data to original state
for set in (strat_train_set, strat_test_set):
    set.drop(['price_cat'], axis = 1, inplace = True)


"""2. Data Visualization """

#make copy to play safely with data
housing = strat_train_set.copy()

#visualize geographical info with scatterplot
housing.plot(kind = 'scatter', x = 'long', y = 'lat')
plt.show()
#define alpha to reveal pattern
housing.plot(kind = 'scatter', x = 'long', y = 'lat', alpha = 0.02)
plt.show()

#plot data according to sq ft living & price
housing.plot(kind = 'scatter', x = 'long', y = 'lat', alpha = 0.03,
             s = housing['sqft_living']/100, label = 'sqft_living',
             c = housing['price'], cmap = plt.get_cmap('jet'),
             colorbar = True, sharex = False)
plt.legend()

#look at correlations
corr_matrix = housing.corr()
corr_matrix['price'].sort_values(ascending = False)

#scatter matrix with most relevant components
from pandas.plotting import scatter_matrix

attributes = ['price','sqft_living','grade', 'sqft_above', 'bathrooms']
scatter_matrix(housing[attributes], figsize = (12, 8))

#gonna have to convert grade 
#sqft ft above & sqft living high correlation
#price vs sqft living highest correlation

housing.plot(kind = 'scatter', x = 'sqft_living', y = 'price', alpha = 0.1)
plt.show()

"""3. Data Cleaning"""

housing = strat_train_set.drop('price', axis =1)
housing_labels = strat_train_set['price'].copy()
housing.isna().sum()  #no missing values

#Text & Categorical Attributes
#one categorical object date: drop it for now, low correlation.
housing = housing.drop(['date', 'id'], axis = 1)


#grade : ordinal data
#feature scaling numerical attributes
#join categorical & numerical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from DataFrameSelector import DataFrameSelector

#include imputer if algo used for  future datasets with numerical missing values
housing_num = housing.drop(['grade'], axis = 1)
num_attribs = list(housing_num)
cat_attribs = ['grade']

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy = 'median')),
        ('scaler', StandardScaler())
        ])
    
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('encoder', OneHotEncoder())
        ])

full_pipeline = FeatureUnion(transformer_list = [
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)
        ])

housing_prepared = full_pipeline.fit_transform(housing)

"""Construct Regression Model"""
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)  #rmse : 191 055 not good

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, forest_housing_predictions)
forest_rmse = np.sqrt(forest_mse)    #rmse: 48 596 much better

from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring = 'neg_mean_squared_erro', cv = 10)
forest_rmse_score = np.sqrt(-scores)

def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard Deviation', scores.std())

display_scores(forest_rmse_score)







    










    

