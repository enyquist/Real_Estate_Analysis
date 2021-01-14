import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.callbacks import DeltaXStopper
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pickle
import logging

from utils.RealEstateData import RealEstateData

#######################################################################################################################
# Config Log File
#######################################################################################################################

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create Handlers
c_handler = logging.StreamHandler()
e_handler = logging.FileHandler(filename='logs/error_log.log')
t_handler = logging.FileHandler(filename='logs/training_log.log')

# Create Log Format(s)
f_format = logging.Formatter('%(asctime)s:%(processName)s:%(name)s:%(levelname)s:%(message)s')

# Set handler levels
c_handler.setLevel(logging.INFO)
e_handler.setLevel(logging.ERROR)
t_handler.setLevel(logging.INFO)

# Set handler formats
c_handler.setFormatter(f_format)
e_handler.setFormatter(f_format)
t_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(e_handler)
logger.addHandler(t_handler)

#######################################################################################################################
# Placeholder for user input
#######################################################################################################################

str_city = 'Durham'
str_state_code = 'NC'

#######################################################################################################################
# Initialize City Data
#######################################################################################################################

# def main(str_city, str_state_code):
# durham = RealEstateData(str_city, str_state_code)


# if __name__ == '__main__':
if os.path.exists('01 Misc/durham.pickle'):
    with open('01 Misc/durham.pickle', 'rb') as file:
        df_durham = pickle.load(file)
else:
    df_durham = RealEstateData(str_city, str_state_code).results

    ####################################################################################################################
    # Data Preparation
    ####################################################################################################################

    # ID metadata to drop
    list_drop_columns = [
        'permalink',
        'photos',
        'community',
        'virtual_tours',
        'matterport',
        'primary_photo.href',
        'source.plan_id',
        'source.agents',
        'source.spec_id',
        'description.sub_type',
        'lead_attributes.show_contact_an_agent',
        'other_listings.rdc',
        'primary_photo',
        'products',
        'community.advertisers',
        'community.description.name',
        'location.address.coordinate',
        'other_listings',
        'location.county'
    ]

    df_durham.drop(list_drop_columns, axis=1, inplace=True)

# Parse into unique DataFrame for each type of real estate
df_sf = df_durham[df_durham['description.type'] == 'single_family']

# ID features
list_features = [
    'description.year_built',
    'description.baths_full',
    'description.baths_3qtr',
    'description.baths_half',
    'description.baths_1qtr',
    'description.lot_sqft',
    'description.sqft',
    'description.garage',
    'description.beds',
    'description.stories',
    'location.address.coordinate.lon',
    'location.address.coordinate.lat',
    'list_price'
]

df_sf_features = df_sf[list_features]

# Clean DataFrame
df_sf_features_clean = df_sf_features.fillna(0, axis=1)

# split into input and output elements
X, y = df_sf_features_clean.drop('list_price', axis=1).values, df_sf_features_clean['list_price'].values

# Outlier Detection
clf = IsolationForest(random_state=42).fit_predict(X)

# Mask outliers
mask = np.where(clf == -1)

X = np.delete(X, mask, axis=0)
y = np.delete(y, mask)

# Split data using outlier-free data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Pipeline
regression_pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('feature_selection', SelectFromModel(CatBoostRegressor())),
    ('regressor', CatBoostRegressor())
])

skopt_grid = [
    {  # CatBoostRegressor
        'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
        'feature_selection': Categorical(['passthrough',
                                          SelectFromModel(CatBoostRegressor()),
                                          RFECV(CatBoostRegressor(), step=0.2)]),
        'regressor': Categorical([CatBoostRegressor()]),
        'regressor__depth': Integer(low=4, high=10),
        'regressor__learning_rate': Real(low=0.01, high=0.1),
        'regressor__iterations': Integer(low=500, high=1500),
        'regressor__loss_function': Categorical(['RMSE'])
    },
    {  # GaussianProcessRegressor
        'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
        'feature_selection': Categorical(['passthrough']),
        'regressor': Categorical([GaussianProcessRegressor()]),
        'regressor__kernel': Categorical([DotProduct(), WhiteKernel()])
    },
    {  # SVR with feature selection
        'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
        'feature_selection': Categorical(['passthrough',
                                          SelectFromModel(SVR(kernel='linear')),
                                          RFECV(SVR(kernel='linear'), step=0.2)]),
        'regressor': Categorical([SVR()]),
        'regressor__kernel': Categorical(['linear']),
        'regressor__C': Real(low=1e-6, high=1e6)
    },
    {  # ElasticNet
        'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
        'feature_selection': Categorical(['passthrough',
                                          SelectFromModel(ElasticNet()),
                                          RFECV(ElasticNet(), step=0.2)]),
        'regressor': Categorical([ElasticNet()]),
        'regressor__alpha': Real(low=1, high=1e6),
        'regressor__l1_ratio': Real(low=0, high=1)
    },
    {  # SVR
        'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
        'feature_selection': Categorical(['passthrough']),
        'regressor': Categorical([SVR()]),
        'regressor__kernel': Categorical(['poly', 'rbf']),
        'regressor__C': Real(low=1e-6, high=1e6)
    },
    {  # DecisionTreeRegressor
        'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
        'feature_selection': Categorical(['passthrough',
                                          SelectFromModel(DecisionTreeRegressor()),
                                          RFECV(DecisionTreeRegressor(), step=0.2)]),
        'regressor': Categorical([DecisionTreeRegressor()])
    },
    {  # KNeighborsRegressor
        'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
        'feature_selection': Categorical(['passthrough']),
        'regressor': Categorical([KNeighborsRegressor()]),
        'regressor__n_neighbors': Integer(low=5, high=15),
        'regressor__weights': Categorical(['uniform', 'distance'])
    },
    {  # RandomForestRegressor
        'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
        'feature_selection': Categorical(['passthrough',
                                          SelectFromModel(RandomForestRegressor()),
                                          RFECV(RandomForestRegressor(), step=0.2)]),
        'regressor': Categorical([RandomForestRegressor()])
    },
    {  # MLPRegressor
        'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
        'feature_selection': Categorical(['passthrough']),
        'regressor': Categorical([MLPRegressor()])
    },
    {  # LinearRegression
        'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
        'feature_selection': Categorical(['passthrough',
                                          SelectFromModel(LinearRegression()),
                                          RFECV(LinearRegression(), step=0.2)]),
        'regressor': Categorical([LinearRegression()])
    }
]

skopt = BayesSearchCV(estimator=regression_pipe,
                      search_spaces=skopt_grid,
                      n_iter=100,
                      n_jobs=16,
                      cv=5)

# Train on train data
start = time.perf_counter()

logger.info('Starting Bayesian Search')

skopt.fit(X_train, y_train, callback=DeltaXStopper(0.01))

logger.info('Bayesian Search Complete')

training_time = time.perf_counter() - start

# Capture and save results for compute later
my_df = pd.DataFrame.from_dict(skopt.cv_results_)

my_df.to_csv('01 Misc/bayes_results.csv')

with open('01 Misc/bayes_grid.pickle', 'wb+') as file:
    pickle.dump(skopt, file)

logger.info('Bayes object and cv_results_ saved')

# Cross Validate the score Skopt's best estimator on entire data set
scores = cross_val_score(skopt.best_estimator_, X, y)

# Predict and Test on test data
y_pred = skopt.predict(X_test)

r2 = r2_score(y_test, y_pred)

logger.info('Results from Bayes Search (BS):')
logger.info(f'BS best estimator: {skopt.best_estimator_}')
logger.info(f'BS Validation Score: {skopt.best_score_}')
logger.info(f'BS Best params: {skopt.best_params_}')
logger.info(f'BS Cross Validation Scores: {scores}')
logger.info(f"BS accuracy on data: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
logger.info(f"BS R2 score: %0.2f" % r2)
