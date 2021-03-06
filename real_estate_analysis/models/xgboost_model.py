import xgboost as xgb
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

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

    bucket = 're-formatted-data'

    df_train = func.fetch_from_s3(bucket=bucket, key='train.tgz')
    df_test = func.fetch_from_s3(bucket=bucket, key='test.tgz')

    # Split the data
    X_train, y_train = df_train.drop(['list_price'], axis=1).values, df_train['list_price'].values
    X_test, y_test = df_test.drop(['list_price'], axis=1).values, df_test['list_price'].values

    # Format as DMatrices
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    ####################################################################################################################
    # Functions
    ####################################################################################################################

    def bo_tune_xgb(max_depth, min_child_weight, eta, subsample, colsample_bytree, gamma, lambda_, alpha):
        """
        Wrapper to apply Bayesian Optimization to tune an xgb model
        :param max_depth: Maximum depth of a tree
        :param min_child_weight: Minimum sum of instance weight (hessian) needed in a child.
        :param eta: Step size shrinkage used in update to prevents overfitting.
        :param subsample: Subsample ratio of the training instances
        :param colsample_bytree: subsample ratio of columns when constructing each tree.
        :param gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree.
        :param lambda_: L2 regularization term on weights.
        :param alpha: L1 regularization term on weights.
        :return:
        """
        params = {
            'max_depth': int(max_depth),
            'min_child_weight': min_child_weight,
            'eta': eta,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'lambda': lambda_,
            'alpha': alpha,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist',
            'max_bin': 64,
            'predictor': 'gpu_predictor',
            'gpu_id': 0
        }

        cv_result = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            seed=42,
            nfold=5,
            metrics=['rmse'],
            early_stopping_rounds=10
        )

        mse = cv_result['test-rmse-mean'].iloc[-1] ** 2

        return -1.0 * mse

    ####################################################################################################################
    # Bayesian Optimization
    ####################################################################################################################

    NUM_BOOST_ROUND = 999

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

    xgb_bo = BayesianOptimization(
        f=bo_tune_xgb,
        pbounds=dict_params,
        random_state=42,
        bounds_transformer=SequentialDomainReductionTransformer()
    )

    logger.info('Starting Bayesian Optimization')

    xgb_bo.maximize(n_iter=45, init_points=15, acq='ei')

    logger.info('Bayesian Optimization Complete')

    # Extract best params
    best_params = xgb_bo.max['params']
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

    dict_scores = func.score_my_model(my_model=optimized_model, x_test=X_test, y_test=y_test)
    r2_training = optimized_model.score(X_train, y_train)

    logger.info('Results from XGBoost Search:')
    logger.info(f'Best params: {best_params}')
    logger.info(f"Cross Validation Scores: {dict_scores['cross_val_score']}")
    logger.info(f"Validation Score: {r2_training:0.2f}")
    logger.info(f"Accuracy on test data: {dict_scores['mean_cross_val_score']:0.2f}"
                f" (+/- {dict_scores['std_cross_val_score']:0.2f})")
    logger.info(f"Test score: {dict_scores['model_score']:0.2f}")
    logger.info(f"MSE: {dict_scores['mse']:0.2f}")


if __name__ == '__main__':
    main()

#######################################################################################################################
# s3 Data
#######################################################################################################################

# Results from XGBoost Search:

# Best params:
# {'alpha': 0.9505062430922577, 'colsample_bytree': 0.6419016010646321, 'eta': 0.03481878178114492,
# 'gamma': 7.328545730764135, 'max_depth': 9, 'min_child_weight': 3.1410157573443875, 'subsample': 0.9309928333029542,
# 'lambda': 1.7017024086650765, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'gpu_hist',
# 'max_bin': 64, 'predictor': 'gpu_predictor', 'gpu_id': 0, 'n_estimators': 566}

# Cross Validation Scores: [0.55291647 0.44046691 0.51831694 0.40844472 0.53878692]

# Validation Score: 0.93

# Accuracy on test data: 0.49 (+/- 0.11)

# Test score: 0.40

# MSE: 1695799635366.35
