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

import real_estate_analysis.utils.functions as func


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/error_log.log', t_handler_name='../logs/training_log.log')

    ####################################################################################################################
    # Define Pipeline and variables
    ####################################################################################################################

    bucket = 're-raw-data'

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
                                  SelectFromModel(CatBoostRegressor())],
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
                                  SelectFromModel(SVR(kernel='linear'))],
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
                                  SelectFromModel(ElasticNet())],
            'regressor': [ElasticNet()],
            'regressor__alpha': np.linspace(start=1, stop=100, num=5),
            'regressor__l1_ratio': np.linspace(start=0, stop=1, num=5)
        },
        {  # DecisionTreeRegressor
            'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
            'feature_selection': ['passthrough',
                                  SelectFromModel(DecisionTreeRegressor())],
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
                                  SelectFromModel(RandomForestRegressor())],
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
                                  SelectFromModel(LinearRegression())],
            'regressor': [LinearRegression()]
        }
    ]

    ####################################################################################################################
    # Data
    ####################################################################################################################

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

    logger.info('Starting Regressor Training')

    model = func.train_my_model(my_df=df_sf_features,
                                my_pipeline=regression_pipe,
                                my_param_grid=param_grid,
                                style='grid')

    logger.info('Regressor Training Complete')

    list_scores = func.score_my_model(my_df=df_sf_features, my_model=model)

    logger.info('Results from Search:')
    logger.info(f'Search best estimator: {model.best_estimator_}')
    logger.info(f'Search Best params: {model.best_params_}')
    logger.info(f"Search Cross Validation Scores: {list_scores[0]}")
    logger.info(f"Search Validation Score: %0.2f" % model.best_score_)
    logger.info(f"Search accuracy on test data: %0.2f (+/- %0.2f)" % (list_scores[1], list_scores[2]))
    logger.info(f"Search test score: %0.2f" % list_scores[3])
    logger.info(f"Search R2 score: %0.2f" % list_scores[4])


if __name__ == '__main__':
    main()

# RANDOMIZEDSEARCHCV OUTPUT

# 2020-12-23 14:02:05,060:MainProcess:root:INFO:Results from Randomized Grid Search (Search):

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

#######################################################################################################################
# s3 Data - Durham, Raleigh, Greensboro, Fayetteville, Charlotte
#######################################################################################################################

# 2020-12-27 13:40:05,885:MainProcess:root:INFO:Results from Randomized Grid Search (RGS):

# 2020-12-27 13:40:05,888:MainProcess:root:INFO:RGS best estimator: Pipeline(steps=[('scaler', RobustScaler()),
#                 ('feature_selection',
#                  RFECV(estimator=<catboost.core.CatBoostRegressor object at 0x0000028F76C2B548>,
#                        step=0.2)),
#                 ('regressor',
#                  <catboost.core.CatBoostRegressor object at 0x0000028F76C2B448>)])

# 2020-12-27 13:40:05,888:MainProcess:root:INFO:RGS Validation Score: 0.8491577972346784

# 2020-12-27 13:40:05,889:MainProcess:root:INFO:RGS Best params:
# {'scaler': RobustScaler(),
# 'regressor__loss_function': 'RMSE',
# 'regressor__learning_rate': 0.05,
# 'regressor__iterations': 1500,
# 'regressor__depth': 8,
# 'regressor': <catboost.core.CatBoostRegressor object at 0x0000028F74F2D448>,
# 'feature_selection': RFECV(estimator=<catboost.core.CatBoostRegressor object at 0x0000028F7674D048>,
#       step=0.2)}

# 2020-12-27 13:40:05,889:MainProcess:root:INFO:RGS Cross Validation Scores:
# [ 0.81946377  0.74768273  0.70388037 -0.10789635  0.62905247]

# 2020-12-27 13:40:05,889:MainProcess:root:INFO:RGS accuracy on all data: 0.56 (+/- 0.68)

# 2020-12-27 13:40:05,889:MainProcess:root:INFO:RGS test score: 0.86

# 2020-12-27 13:40:05,889:MainProcess:root:INFO:RGS R2 score: 0.86
