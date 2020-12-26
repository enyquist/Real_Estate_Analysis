import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from utils.RealEstateData import RealEstateData
from utils.functions import train_my_model, score_my_model, create_logger

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

# Define Pipeline
regression_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(CatBoostRegressor)),
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

logger.info('Starting Regressor Training')

model = train_my_model(my_df=df_sf_features, my_pipeline=regression_pipe, my_param_grid=param_grid)

logger.info('Regressor Training Complete')

list_scores = score_my_model(my_df=df_sf_features, my_model=model)

logger.info('Results from Randomized Grid Search (RGS):')
logger.info(f'RGS best estimator: {model.best_estimator_}')
logger.info(f'RGS Validation Score: {model.best_score_}')
logger.info(f'RGS Best params: {model.best_params_}')
logger.info(f'RGS Cross Validation Scores: {list_scores[0]}')
logger.info(f"RGS accuracy on all data: %0.2f (+/- %0.2f)" % (list_scores[1], list_scores[2]))
logger.info(f"RGS test score: %0.2f" % list_scores[3])
logger.info(f"RGS R2 score: %0.2f" % list_scores[4])

# RANDOMIZEDSEARCHCV OUTPUT

# 2020-12-23 14:02:05,060:MainProcess:root:INFO:Results from Randomized Grid Search (RGS):

# 2020-12-23 14:02:05,063:MainProcess:root:INFO:RGS best estimator: Pipeline(steps=[('scaler', RobustScaler()),
#                 ('feature_selection',
#                  RFECV(estimator=<catboost.core.CatBoostRegressor object at 0x0000020873F919C8>,
#                        step=0.2)),
#                 ('regressor',
#                  <catboost.core.CatBoostRegressor object at 0x000002087505D448>)])

# 2020-12-23 14:02:05,063:MainProcess:root:INFO:RGS Validation Score: 0.6797365755730966

# 2020-12-23 14:02:05,064:MainProcess:root:INFO:
# RGS Best params: {'scaler': RobustScaler(),
# 'regressor__loss_function': 'RMSE',
# 'regressor__learning_rate': 0.05,
# 'regressor__iterations': 1000,
# 'regressor__depth': 6,
# 'regressor': <catboost.core.CatBoostRegressor object at 0x0000020873F51208>,
# 'feature_selection': RFECV(estimator=<catboost.core.CatBoostRegressor object at 0x0000020873F51408>,
#       step=0.2)}

# 2020-12-23 14:02:05,064:MainProcess:root:INFO:RGS Cross Validation Scores:
# [0.74567611 0.82943665 0.74755348 0.53558083 0.78026391]

# 2020-12-23 14:02:05,064:MainProcess:root:INFO:RGS accuracy on all data: 0.73 (+/- 0.20)

# 2020-12-23 14:02:05,064:MainProcess:root:INFO:RGS test score: 0.66

# 2020-12-23 14:02:05,064:MainProcess:root:INFO:RGS R2 score: 0.66
