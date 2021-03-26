import xgboost as xgb
import datetime

import real_estate_analysis.models.functions as func
import real_estate_analysis.models.xgb_model.utils as XGB_utils
import real_estate_analysis.Model.utils as model_utils


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/xgboost_error_log.log',
                                t_handler_name='../logs/xgboost_training_log.log')

    ####################################################################################################################
    # Data
    ####################################################################################################################

    X_train, y_train, X_test, y_test = func.retrieve_and_prepare_data()

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

    optimizer = XGB_utils.optimize_xgb(dtrain=dtrain, pbounds=dict_params, n_iter=10, init_points=3)

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
    func.log_scores(dict_scores)

    ####################################################################################################################
    # Evaluate and Save
    ####################################################################################################################

    today = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    fname = f'xgboost_{today}.joblib'

    model_utils.validate_model(optimized_model, dict_scores, fname)


if __name__ == '__main__':
    main()
