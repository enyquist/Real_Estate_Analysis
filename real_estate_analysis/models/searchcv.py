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
                                style='grid')

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

# 2021-01-28 00:07:10,310:MainProcess:root:INFO:Results from Search:

# 2021-01-28 00:07:10,311:MainProcess:root:INFO:Search best estimator:
# Pipeline(steps=[('scaler', StandardScaler()),
#                 ('feature_selection', 'passthrough'),
#                 ('regressor',
#                  <catboost.core.CatBoostRegressor object at 0x00000274B98DF648>)])

# 2021-01-28 00:07:10,311:MainProcess:root:INFO:Search Best params:
# {'feature_selection': 'passthrough',
# 'regressor': <catboost.core.CatBoostRegressor object at 0x0000027572B037C8>,
# 'regressor__depth': 8,
# 'regressor__iterations': 1500,
# 'regressor__learning_rate': 0.1,
# 'regressor__logging_level': 'Silent',
# 'regressor__loss_function': 'RMSE',
# 'regressor__od_pval': 0.1,
# 'scaler': StandardScaler()}

# 2021-01-28 00:07:10,311:MainProcess:root:INFO:Search Cross Validation Scores:
# [0.75494423 0.68898657 0.7131733  0.68125362 0.75666829]

# 2021-01-28 00:07:10,311:MainProcess:root:INFO:Search Validation Score: 0.75

# 2021-01-28 00:07:10,311:MainProcess:root:INFO:Search accuracy on test data: 0.72 (+/- 0.06)

# 2021-01-28 00:07:10,311:MainProcess:root:INFO:Search test score: 0.78
