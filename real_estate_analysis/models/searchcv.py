import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor

from xgboost import XGBRegressor

import pickle

import real_estate_analysis.utils.functions as func


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/error_log.log', t_handler_name='../logs/training_log.log')

    ####################################################################################################################
    # Data
    ####################################################################################################################

    bucket = 're-raw-data'

    # Retrieve data from s3 and format into dataframe
    df = func.create_df_from_s3(bucket=bucket)

    # Parse DataFrame for single family real estate
    df_sf = df[df['description.type'] == 'single_family']

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
        'location.address.state_code',
        'tags',
        'list_price'
    ]

    df_sf_features = df_sf[list_features]

    # Prepare and split data
    X_train, X_test, y_train, y_test = func.prepare_my_data(my_df=df_sf_features)

    ####################################################################################################################
    # Define Pipeline
    ####################################################################################################################

    # Define Pipeline
    regression_pipe = Pipeline([
        ('feature_selection', SelectFromModel(CatBoostRegressor)),
        ('regressor', CatBoostRegressor())
    ])

    # Searched with other algorithms listed below, but removed after finding they performed poorly:
    # GaussianProcessRegressor, SVR, ElasticNet, DecisionTreeRegressor, KNeighborsRegressor,
    # RandomForestRegressor, MLPRegressor, LinearRegression

    param_grid = [
        {  # CatBoostRegressor
            'feature_selection': ['passthrough',
                                  SelectFromModel(CatBoostRegressor())],
            'regressor': [CatBoostRegressor()],
            'regressor__depth': [4, 6, 8, 10],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__iterations': [500, 1000, 1500],
            'regressor__loss_function': ['RMSE'],
            'regressor__od_pval': [10e-2],
            'regressor__logging_level': ['Silent']
        },
        {  # XGBoost
            'feature_selection': ['passthrough',
                                  SelectFromModel(XGBRegressor())],
            'regressor': [XGBRegressor()],
            'regressor__n_estimators': [1000, 1500, 2500],
            'regressor__max_depth': [4, 6, 8],
            'regressor__learning_rate': [0.05, 0.1, 0.3],
            'regressor__booster': ['gbtree']
        }
    ]

    ####################################################################################################################
    # Training
    ####################################################################################################################

    logger.info('Starting Regressor Training')

    model = func.train_my_model(my_pipeline=regression_pipe,
                                my_param_grid=param_grid,
                                x_train=X_train,
                                y_train=y_train,
                                style='grid',
                                n_jobs=15)

    logger.info('Regressor Training Complete')

    ####################################################################################################################
    # Validation
    ####################################################################################################################

    list_scores = func.score_my_model(my_model=model, x_test=X_test, y_test=y_test)

    logger.info('Results from Search:')
    logger.info(f'Search best estimator: {model.best_estimator_}')
    logger.info(f'Search Best params: {model.best_params_}')
    logger.info(f"Search Cross Validation Scores: {list_scores[0]}")
    logger.info(f"Search Validation Score: %0.2f" % model.best_score_)
    logger.info(f"Search accuracy on test data: %0.2f (+/- %0.2f)" % (list_scores[1], list_scores[2]))
    logger.info(f"Search test score: %0.2f" % list_scores[3])


if __name__ == '__main__':
    main()

#######################################################################################################################
# s3 Data
#######################################################################################################################

# 2021-02-04 22:13:18,515:MainProcess:root:INFO:Results from Search:

# 2021-02-04 22:13:18,515:MainProcess:root:INFO:Search best estimator:
# Pipeline(steps=[('feature_selection', 'passthrough'),
#                 ('regressor',
#                  <catboost.core.CatBoostRegressor object at 0x0000021601E61FC8>)])

# 2021-02-04 22:13:18,515:MainProcess:root:INFO:Search Best params:
# {'feature_selection': 'passthrough',
# 'regressor': <catboost.core.CatBoostRegressor object at 0x00000215EFD73448>,
# 'regressor__depth': 8,
# 'regressor__iterations': 1500,
# 'regressor__learning_rate': 0.1,
# 'regressor__logging_level': 'Silent',
# 'regressor__loss_function': 'RMSE',
# 'regressor__od_pval': 0.1}

# 2021-02-04 22:13:18,516:MainProcess:root:INFO:Search Cross Validation Scores:
# [0.77533568 0.8059569  0.76499073 0.65052428 0.79152382]

# 2021-02-04 22:13:18,516:MainProcess:root:INFO:Search Validation Score: 0.78

# 2021-02-04 22:13:18,516:MainProcess:root:INFO:Search accuracy on test data: 0.76 (+/- 0.11)

# 2021-02-04 22:13:18,516:MainProcess:root:INFO:Search test score: 0.80
