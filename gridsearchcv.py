import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, IsolationForest, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pickle

from utils.RealEstateData import RealEstateData
from utils.functions import create_logger

#######################################################################################################################
# Config Log File
#######################################################################################################################

logger = create_logger(e_handler_name='logs/error_log.log', t_handler_name='logs/training_log.log')

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

# Split data using outlier-free data
X_train, X_test, y_train, y_test = train_test_split(np.delete(X, mask, axis=0), np.delete(y, mask), test_size=0.2, random_state=42)

# Define Pipeline
regression_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(CatBoostRegressor())),
    ('regressor', CatBoostRegressor())
])

param_grid = [
    {  # CatBoostRegressor
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough',
                              SelectFromModel(CatBoostRegressor()),
                              RFECV(CatBoostRegressor(), step=0.2)],
        'regressor': [CatBoostRegressor()],
        'regressor__depth': [4, 6, 8, 10],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__iterations': [500, 1000, 1500],
        'regressor__loss_function': ['RMSE']
    },
    {  # GaussianProcessRegressor
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough'],
        'regressor': [GaussianProcessRegressor()],
        'regressor__kernel': [DotProduct() + WhiteKernel()]
    },
    {  # SVR with Feature Selection
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough',
                              SelectFromModel(SVR(kernel='linear')),
                              RFECV(SVR(kernel='linear'), step=0.2)],
        'regressor': [SVR()],
        'regressor__kernel': ['linear'],
        'regressor__C': np.linspace(start=1, stop=100, num=5)
    },
    {  # SVR
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough'],
        'regressor': [SVR()],
        'regressor__kernel': ['poly', 'rbf'],
        'regressor__C': np.linspace(start=1, stop=100, num=5)
    },
    {  # ElasticNet
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough',
                              SelectFromModel(ElasticNet()),
                              RFECV(ElasticNet(), step=0.2)],
        'regressor': [ElasticNet()],
        'regressor__alpha': np.linspace(start=1, stop=100, num=5),
        'regressor__l1_ratio': np.linspace(start=0, stop=1, num=5)
    },
    {  # DecisionTreeRegressor
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough',
                              SelectFromModel(DecisionTreeRegressor()),
                              RFECV(DecisionTreeRegressor(), step=0.2)],
        'regressor': [DecisionTreeRegressor()]
    },
    {  # KNeighborsRegressor
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough'],
        'regressor': [KNeighborsRegressor()],
        'regressor__n_neighbors': [5, 10, 15],
        'regressor__weights': ['uniform', 'distance']
    },
    {  # RandomForestRegressor
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough',
                              SelectFromModel(RandomForestRegressor()),
                              RFECV(RandomForestRegressor(), step=0.2)],
        'regressor': [RandomForestRegressor()]
    },
    {  # MLPRegressor
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough'],
        'regressor': [MLPRegressor()]
    },
    {  # LinearRegression
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough',
                              SelectFromModel(LinearRegression()),
                              RFECV(LinearRegression(), step=0.2)],
        'regressor': [LinearRegression()]
    }
]

# Create Grid
grid = GridSearchCV(estimator=regression_pipe, param_grid=param_grid, cv=5, n_jobs=-1)

# Train Grid on train data
start = time.perf_counter()

logger.info('Starting Exhaustive Grid Search')

grid.fit(X_train, y_train)

logger.info('Grid Search Complete')

training_time = time.perf_counter() - start

# Capture and save results for compute later
my_df = pd.DataFrame.from_dict(grid.cv_results_)

my_df.to_csv('01 Misc/grid_results.csv')

with open('01 Misc/exhaustive_grid.pickle', 'wb+') as file:
    pickle.dump(grid, file)

logger.info('Grid object and cv_results_ saved')

# Cross Validate the score Grid's best estimator on entire data set
scores = cross_val_score(grid.best_estimator_, X, y)

# Predict and Test on test data
y_pred = grid.predict(X_test)

r2 = r2_score(y_test, y_pred)

logger.info('Results from Exhaustive Grid Search (EGS):')
logger.info(f'EGS best estimator: {grid.best_estimator_}')
logger.info(f'EGS Validation Score: {grid.best_score_}')
logger.info(f'EGS Best params: {grid.best_params_}')
logger.info(f'EGS Cross Validation Scores: {scores}')
logger.info(f"EGS accuracy on data: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
logger.info(f"EGS R2 score: %0.2f" % r2)

# GRID CATBOOSTREGRESSOR OUTPUT

# The best estimator across ALL searched params:
#  Pipeline(steps=[('scaler', QuantileTransformer()),
#                 ('feature_selection',
#                  SelectFromModel(estimator=<catboost.core.CatBoostRegressor object at 0x00000200F0EA6EC8>)),
#                 ('regressor',
#                  <catboost.core.CatBoostRegressor object at 0x00000200F0EA6C48>)])

# The best score across ALL searched params on training set:
#   0.8124534438242197

# The best parameters across ALL searched params:
#  {'feature_selection': SelectFromModel(estimator=<catboost.core.CatBoostRegressor object at 0x00000200EFC601C8>),
#  'regressor': <catboost.core.CatBoostRegressor object at 0x00000200EFC603C8>,
#  'regressor__depth': 6,
#  'regressor__iterations': 1000,
#  'regressor__learning_rate': 0.05,
#  'regressor__loss_function': 'RMSE',
#  'scaler': QuantileTransformer()}

# Scores of best_estimator_ on the data: [0.61991392 0.8101451  0.72269698 0.25037767 0.72657418]

# Accuracy on data: 0.63 (+/- 0.39)

# R2 score on test set: 0.61