import xgboost as xgb
import datetime

import real_estate_analysis.utils.functions as func


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/xgboost_error_log.log',
                                t_handler_name='../logs/xgboost_training_log.log')

    ####################################################################################################################
    # Data
    ####################################################################################################################

    df_train = func.fetch_from_s3(bucket='re-formatted-data', key=f'sold_train.tgz')
    df_test = func.fetch_from_s3(bucket='re-formatted-data', key=f'sold_test.tgz')

    # Split the data
    X_train, y_train = df_train.drop(['price'], axis=1).values, df_train['price'].values
    X_test, y_test = df_test.drop(['price'], axis=1).values, df_test['price'].values

    # Format as DMatrices
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    ####################################################################################################################
    # Bayesian Optimization
    ####################################################################################################################

    dict_params = {
        'max_depth': (3, 10),
        'min_child_weight': (10e-6, 8),
        'eta': (10e-6, 0.2),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1),
        'gamma': (0, 8),
        'lambda_': (0.5, 10),
        'alpha': (5, 10)
    }

    logger.info('Starting Bayesian Optimization')

    optimizer = func.optimize_xgb(dtrain=dtrain, pbounds=dict_params, n_iter=10, init_points=3)

    logger.info('Bayesian Optimization Complete')

    # Extract best params
    best_params = optimizer.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['lambda'] = best_params['lambda_']
    best_params.pop('lambda_')

    # Set up best params for GPU learning
    best_params['objective'] = 'reg:squarederror'
    best_params['eval_metric'] = 'rmse'
    best_params['tree_method'] = 'gpu_hist'
    best_params['max_bin'] = 64
    best_params['predictor'] = 'gpu_predictor'
    best_params['gpu_id'] = 0

    ####################################################################################################################
    # Train Model with Optimized Params
    ####################################################################################################################

    NUM_BOOST_ROUND = 999

    logger.info('Starting Model Training')

    # Train model with those params Model to search for best boosting rounds
    model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtest, 'Test')],
        early_stopping_rounds=10
    )

    best_params['n_estimators'] = model.best_iteration + 1

    optimized_model = xgb.XGBRegressor(**best_params)

    optimized_model.fit(X_train, y_train)

    logger.info('Model Training Complete')

    ####################################################################################################################
    # Validation
    ####################################################################################################################

    dict_scores = func.score_my_model(my_model=optimized_model, x_train=X_train, y_train=y_train,
                                      x_test=X_test, y_test=y_test)

    logger.info('Results from XGBoost Search:')
    logger.info(f'Best params: {best_params}')
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
    logger.info(f"Test Median Absolute Error: {dict_scores['test_median_absolute_error']:0.2f}")
    logger.info(f"Test R2 score: {dict_scores['test_r2']:0.2f}")

    ####################################################################################################################
    # Evaluate and Save
    ####################################################################################################################

    today = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    fname = f'xgboost_{today}.joblib'

    func.validate_model(optimized_model, dict_scores, fname)


if __name__ == '__main__':
    main()
