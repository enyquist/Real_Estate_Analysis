from catboost import CatBoostRegressor

import real_estate_analysis.models.functions as func
import real_estate_analysis.models.catboost_model.utils as catboost_utils


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/catboost_error_log.log',
                                t_handler_name='../logs/catboost_training_log.log')

    ####################################################################################################################
    # Data
    ####################################################################################################################

    X_train, y_train, X_test, y_test = func.retrieve_and_prepare_data()

    ####################################################################################################################
    # Bayesian Optimization
    ####################################################################################################################

    NUM_BOOST_ROUND = 999

    dict_params = {
        'learning_rate': (1e-6, 0.2),
        'l2_leaf_reg': (2, 10),
        'bagging_temperature': (0.5, 1),
        'depth': (3, 16)
    }

    logger.info('Starting Bayesian Optimization')

    optimizer = catboost_utils.optimize_catboost(x_train=X_train, y_train=y_train, pbounds=dict_params, n_iter=10,
                                                 init_points=3)

    logger.info('Bayesian Optimization Complete')

    # Extract best params
    best_params = optimizer.max['params']
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

    dict_scores = func.score_my_model(my_model=model, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)

    logger.info('Results from Search:')
    logger.info(f'Search Best params: {best_params}')
    func.log_scores(dict_scores)


if __name__ == '__main__':
    main()
