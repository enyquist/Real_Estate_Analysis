from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import real_estate_analysis.utils.functions as func


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/error_log.log', t_handler_name='../logs/training_log.log')

    ####################################################################################################################
    # Data
    ####################################################################################################################

    bucket = 're-formatted-data'

    df_train = func.fetch_from_s3(bucket=bucket, key='train')
    df_test = func.fetch_from_s3(bucket=bucket, key='test')

    # Split the data
    X_train, y_train = df_train.drop(['list_price'], axis=1).values, df_train['list_price'].values
    X_test, y_test = df_test.drop(['list_price'], axis=1).values, df_test['list_price'].values

    ####################################################################################################################
    # Define Pipeline
    ####################################################################################################################

    # Define Pipeline
    regression_pipe = Pipeline([('regressor', CatBoostRegressor())])

    # Searched with other algorithms listed below, but removed after finding they performed poorly:
    # GaussianProcessRegressor, SVR, ElasticNet, DecisionTreeRegressor, KNeighborsRegressor,
    # RandomForestRegressor, MLPRegressor, LinearRegression

    param_grid = [
        {  # CatBoostRegressor
            'regressor': [CatBoostRegressor()],
            'regressor__depth': [6, 8, 10],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__iterations': [500, 1000, 1500],
            'regressor__loss_function': ['RMSE'],
            'regressor__od_pval': [10e-2],
            'regressor__logging_level': ['Silent']
        },
        {  # XGBoost
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
                                n_jobs=-1)

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
