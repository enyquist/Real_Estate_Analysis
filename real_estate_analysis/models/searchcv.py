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
            'regressor__depth': [4, 6, 8],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__iterations': [500, 1000, 1500],
            'regressor__loss_function': ['RMSE'],
            'regressor__od_pval': [10e-2],
            'regressor__logging_level': ['Silent'],
            'regressor__l2_leaf_reg': [2, 5, 10],
            'regressor__bagging_temperature': [0.5, 0.75, 1.0],
            'regressor__early_stopping_rounds': [10],
            'regressor__task_type': ['GPU'],
            'regressor__boosting_type': ['Plain']
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
                                n_jobs=1)

    logger.info('Regressor Training Complete')

    ####################################################################################################################
    # Validation
    ####################################################################################################################

    dict_scores = func.score_my_model(my_model=model, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)

    logger.info('Results from Search:')
    logger.info(f'Search Best params: {model.best_params_}')
    logger.info(f"Training Cross Validation Scores: {dict_scores['train_cross_val_score']}")
    logger.info(f"Accuracy on training data: {dict_scores['train_mean_cross_val_score']:0.2f}"
                f" (+/- {dict_scores['train_std_cross_val_score']:0.2f})")
    logger.info(f"Test Cross Validation Scores: {dict_scores['test_cross_val_score']}")
    logger.info(f"Accuracy on test data: {dict_scores['test_mean_cross_val_score']:0.2f}"
                f" (+/- {dict_scores['test_std_cross_val_score']:0.2f})")
    logger.info(f"Test Explained Variance Score: {dict_scores['test_explained_variance_score']:0.2f}")
    logger.info(f"Test Max Error: {dict_scores['test_max_error']:0.2f}")
    logger.info(f"Test Mean Absolute Error: {dict_scores['test_mean_absolute_error']:0.2f}")
    logger.info(f"Test Mean Squared Error: {dict_scores['test_mean_squared_error']:0.2f}")
    logger.info(f"Test Mean Squared Log Error: {dict_scores['test_mean_squared_log_error']:0.2f}")
    logger.info(f"Test Median Absolute Error: {dict_scores['test_median_absolute_error']:0.2f}")
    logger.info(f"Test R2 score: {dict_scores['test_r2']:0.2f}")


if __name__ == '__main__':
    main()

#######################################################################################################################
# s3 Data
#######################################################################################################################

# 2021-03-07 02:08:44,864:MainProcess:root:INFO:Results from Search:

# 2021-03-07 02:08:44,864:MainProcess:root:INFO:Search Best params: {'regressor': <catboost.core.CatBoostRegressor object at 0x00000235C68D9288>, 'regressor__bagging_temperature': 0.75, 'regressor__boosting_type': 'Plain', 'regressor__depth': 8, 'regressor__early_stopping_rounds': 10, 'regressor__iterations': 1500, 'regressor__l2_leaf_reg': 2, 'regressor__learning_rate': 0.05, 'regressor__logging_level': 'Silent', 'regressor__loss_function': 'RMSE', 'regressor__od_pval': 0.1, 'regressor__task_type': 'GPU'}
# 2021-03-07 02:08:44,864:MainProcess:root:INFO:Search Cross Validation Scores:
# [0.48359103 0.45020217 0.49418672 0.47762198 0.37177466]

# 2021-03-07 02:08:44,864:MainProcess:root:INFO:Search Validation Score: 0.32

# 2021-03-07 02:08:44,864:MainProcess:root:INFO:Search accuracy on test data: 0.46 (+/- 0.09)

# 2021-03-07 02:08:44,864:MainProcess:root:INFO:Search test score: 0.32

# 2021-03-07 02:08:44,864:MainProcess:root:INFO:MSE: 1260988713616.69
