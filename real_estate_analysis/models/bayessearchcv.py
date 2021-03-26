from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from skopt.callbacks import DeltaXStopper

from real_estate_analysis.models import functions as func


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/error_log.log', t_handler_name='../logs/training_log.log')

    ####################################################################################################################
    # Data
    ####################################################################################################################

    X_train, y_train, X_test, y_test = func.retrieve_and_prepare_data()

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

    cv = BayesSearchCV(estimator=regression_pipe,
                       search_spaces=skopt_grid,
                       n_iter=50,
                       n_jobs=15,
                       cv=5)

    cv.fit(X_train, y_train, callback=DeltaXStopper(0.01))

    logger.info('Regressor Training Complete')

    ####################################################################################################################
    # Validation
    ####################################################################################################################

    dict_scores = func.score_my_model(my_model=cv, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)

    logger.info('Results from Search:')
    logger.info(f'Search best estimator: {cv.best_estimator_}')
    logger.info(f'Search Best params: {cv.best_params_}')
    func.log_scores(dict_scores)


if __name__ == '__main__':
    main()
