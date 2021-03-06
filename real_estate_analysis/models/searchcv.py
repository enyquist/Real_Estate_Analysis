from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

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

    df_train = func.fetch_from_s3(bucket=bucket, key='train.tgz')
    df_test = func.fetch_from_s3(bucket=bucket, key='test.tgz')

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

    dict_scores = func.score_my_model(my_model=model, x_test=X_test, y_test=y_test)

    logger.info('Results from Search:')
    logger.info(f'Search Best params: {model.best_params_}')
    logger.info(f"Search Cross Validation Scores: {dict_scores['cross_val_score']}")
    logger.info(f"Search Validation Score: {dict_scores['model_score']:0.2f}")
    logger.info(f"Search accuracy on test data: {dict_scores['mean_cross_val_score']:0.2f}"
                f" (+/- {dict_scores['std_cross_val_score']:0.2f})")
    logger.info(f"Search test score: {dict_scores['model_score']:0.2f}")
    logger.info(f"MSE: {dict_scores['mse']:0.2f}")


if __name__ == '__main__':
    main()

#######################################################################################################################
# s3 Data
#######################################################################################################################

# 2021-02-27 03:42:46,825:MainProcess:root:INFO:Results from Search:

# 2021-02-27 03:42:46,840:MainProcess:root:INFO:Search best estimator: Pipeline(steps=[('regressor',
#                  XGBRegressor(base_score=0.5, booster='gbtree',
#                               colsample_bylevel=1, colsample_bynode=1,
#                               colsample_bytree=1, gamma=0, gpu_id=-1,
#                               importance_type='gain',
#                               interaction_constraints='', learning_rate=0.05,
#                               max_delta_step=0, max_depth=6, min_child_weight=1,
#                               missing=nan, monotone_constraints='()',
#                               n_estimators=3500, n_jobs=32, num_parallel_tree=1,
#                               random_state=0, reg_alpha=0, reg_lambda=1,
#                               scale_pos_weight=1, subsample=1,
#                               tree_method='exact', validate_parameters=1,
#                               verbosity=None))])

# 2021-02-27 03:42:46,841:MainProcess:root:INFO:Search Best params:
# {'regressor': XGBRegressor(base_score=None, booster='gbtree', colsample_bylevel=None,
#              colsample_bynode=None, colsample_bytree=None, gamma=None,
#              gpu_id=None, importance_type='gain', interaction_constraints=None,
#              learning_rate=0.05, max_delta_step=None, max_depth=6,
#              min_child_weight=None, missing=nan, monotone_constraints=None,
#              n_estimators=3500, n_jobs=None, num_parallel_tree=None,
#              random_state=None, reg_alpha=None, reg_lambda=None,
#              scale_pos_weight=None, subsample=None, tree_method=None,
#              validate_parameters=None, verbosity=None),
#              'regressor__booster': 'gbtree',
#              'regressor__learning_rate': 0.05,
#              'regressor__max_depth': 6,
#              'regressor__n_estimators': 3500}

# 2021-02-27 03:42:46,842:MainProcess:root:INFO:Search Cross Validation Scores:
# [0.75089114 0.54762574 0.72706152 0.46637741 0.6725924 ]

# 2021-02-27 03:42:46,842:MainProcess:root:INFO:Search Validation Score: 0.66

# 2021-02-27 03:42:46,842:MainProcess:root:INFO:Search accuracy on test data: 0.63 (+/- 0.22)

# 2021-02-27 03:42:46,842:MainProcess:root:INFO:Search test score: 0.73

# 2021-02-27 03:42:46,842:MainProcess:root:INFO:MSE: 780025278437.83
