from catboost import CatBoostRegressor
from sklearn.model_selection import cross_validate
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

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
    # Functions
    ####################################################################################################################

    def bo_tune_catboost(learning_rate, l2_leaf_reg, bagging_temperature, depth, od_pval):
        """
        Wrapper to apply Bayesian Optimization to tune an xgb model
        :param learning_rate: The maximum number of trees that can be built when solving machine learning problems.
        :param l2_leaf_reg: Coefficient at the L2 regularization term of the cost function.
        :param bagging_temperature: Defines the settings of the Bayesian bootstrap..
        :param depth: Depth of the tree.
        :param od_pval: The threshold for the IncToDec overfitting detector type.
        :return:
        """
        params = {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'iterations': NUM_BOOST_ROUND,
            'learning_rate': learning_rate,
            'random_seed': 42,
            'l2_leaf_reg': l2_leaf_reg,
            'bagging_temperature': bagging_temperature,
            'depth': int(depth),
            'early_stopping_rounds': 10,
            'od_pval': od_pval,
            'task_type': 'GPU'
        }

        catboost = CatBoostRegressor(**params)

        cv_result = cross_validate(estimator=catboost, X=X_train, y=y_train)

        mse = cv_result['test_score'] ** 2

        return -1.0 * mse

    ####################################################################################################################
    # Bayesian Optimization
    ####################################################################################################################

    NUM_BOOST_ROUND = 999

    dict_params = {
        'learning_rate': (1e-6, 0.2),
        'l2_leaf_reg': (2, 10),
        'bagging_temperature': (0.5, 1),
        'depth': (3, 16),
        'od_pval': (10e-10, 10e-2)
    }

    xgb_bo = BayesianOptimization(
        f=bo_tune_catboost,
        pbounds=dict_params,
        random_state=42,
        bounds_transformer=SequentialDomainReductionTransformer()
    )

    logger.info('Starting Bayesian Optimization')

    xgb_bo.maximize(n_iter=45, init_points=15, acq='ei')

    logger.info('Bayesian Optimization Complete')

    # Extract best params
    best_params = xgb_bo.max['params']
    best_params['depth'] = int(best_params['depth'])

    # Set up best params for GPU learning
    best_params['loss_function'] = 'RMSE'
    best_params['eval_metric'] = 'RMSE'
    best_params['iterations'] = NUM_BOOST_ROUND
    best_params['random_seed'] = 42
    best_params['early_stopping_rounds'] = 10
    best_params['task_type'] = 'GPU'

    ####################################################################################################################
    # Training
    ####################################################################################################################

    logger.info('Starting Regressor Training')

    model = CatBoostRegressor(**best_params)

    logger.info('Regressor Training Complete')

    ####################################################################################################################
    # Validation
    ####################################################################################################################

    dict_scores = func.score_my_model(my_model=model, x_test=X_test, y_test=y_test)
    r2_training = model.score(X_train, y_train)

    logger.info('Results from Search:')
    logger.info(f'Search Best params: {best_params}')
    logger.info(f"Search Cross Validation Scores: {dict_scores['cross_val_score']}")
    logger.info(f"Search Validation Score: {r2_training:0.2f}")
    logger.info(f"Search accuracy on test data: {dict_scores['mean_cross_val_score']:0.2f}"
                f" (+/- {dict_scores['std_cross_val_score']:0.2f})")
    logger.info(f"Search test score: {dict_scores['model_score']:0.2f}")
    logger.info(f"MSE: {dict_scores['mse']:0.2f}")


if __name__ == '__main__':
    main()
