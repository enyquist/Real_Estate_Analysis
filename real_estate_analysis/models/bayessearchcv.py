from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from skopt.space import Real, Categorical, Integer

from real_estate_analysis.utils import functions as func


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
    # Define Pipeline and variables
    ####################################################################################################################

    regression_pipe = Pipeline([('regressor', LinearRegression())])

    skopt_grid = [
        {  # SVR with feature selection
            'regressor': Categorical([SVR()]),
            'regressor__kernel': Categorical(['linear']),
            'regressor__C': Real(low=1e-6, high=1e6)
        },
        {  # ElasticNet
            'regressor': Categorical([ElasticNet()]),
            'regressor__alpha': Real(low=1, high=1e6),
            'regressor__l1_ratio': Real(low=0, high=1)
        },
        {  # SVR
            'regressor': Categorical([SVR()]),
            'regressor__kernel': Categorical(['poly', 'rbf']),
            'regressor__C': Real(low=1e-6, high=1e6)
        },
        {  # DecisionTreeRegressor
            'regressor': Categorical([DecisionTreeRegressor()])
        },
        {  # KNeighborsRegressor
            'regressor': Categorical([KNeighborsRegressor()]),
            'regressor__n_neighbors': Integer(low=5, high=150),
            'regressor__weights': Categorical(['uniform', 'distance'])
        },
        {  # RandomForestRegressor
            'regressor': Categorical([RandomForestRegressor()])
        },
        {  # MLPRegressor
            'regressor': Categorical([MLPRegressor()])
        },
        {  # LinearRegression
            'regressor': Categorical([LinearRegression()])
        }
    ]

    ####################################################################################################################
    # Train Model
    ####################################################################################################################

    logger.info('Starting Regressor Training')

    model = func.train_my_model(my_pipeline=regression_pipe,
                                my_param_grid=skopt_grid,
                                x_train=X_train,
                                y_train=y_train,
                                style='bayes',
                                search=100)

    logger.info('Regressor Training Complete')

    ####################################################################################################################
    # Validation
    ####################################################################################################################

    dict_scores = func.score_my_model(my_model=model, x_test=X_test, y_test=y_test)

    logger.info('Results from Search:')
    logger.info(f'Search best estimator: {model.best_estimator_}')
    logger.info(f'Search Best params: {model.best_params_}')
    logger.info(f"Search Cross Validation Scores: {dict_scores['cross_val_score']}")
    logger.info(f"Search Validation Score: {dict_scores['model_score']:0.2f}")
    logger.info(f"Search accuracy on test data: {dict_scores['mean_cross_val_score']:0.2f}"
                f" (+/- {dict_scores['std_cross_val_score']:0.2f})")
    logger.info(f"Search test score: {dict_scores['model_score']:0.2f}")
    logger.info(f"MSE: {dict_scores['mse']:0.2f}")


if __name__ == '__main__':
    main()
