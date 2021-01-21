from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from skopt.space import Real, Categorical, Integer

from real_estate_analysis.utils import functions as func


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/error_log.log', t_handler_name='../logs/training_log.log')

    ####################################################################################################################
    # Define Pipeline and variables
    ####################################################################################################################

    bucket = 're-raw-data'

    regression_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('feature_selection', SelectFromModel(CatBoostRegressor())),
        ('regressor', CatBoostRegressor())
    ])

    skopt_grid = [  # todo errors with unhashable types
        {  # CatBoostRegressor
            'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
            'feature_selection': Categorical(['passthrough',
                                              SelectFromModel(CatBoostRegressor())]),
            'regressor': Categorical([CatBoostRegressor()]),
            'regressor__depth': Integer(low=4, high=100),
            'regressor__learning_rate': Real(low=0.01, high=0.1),
            'regressor__iterations': Integer(low=10, high=int(1e6)),
            'regressor__loss_function': Categorical(['RMSE'])
        },
        {  # GaussianProcessRegressor
            'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
            'feature_selection': Categorical(['passthrough']),
            'regressor': Categorical([GaussianProcessRegressor()]),
            'regressor__kernel': Categorical([DotProduct(), WhiteKernel()])
        },
        {  # SVR with feature selection
            'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
            'feature_selection': Categorical(['passthrough',
                                              SelectFromModel(SVR(kernel='linear'))]),
            'regressor': Categorical([SVR()]),
            'regressor__kernel': Categorical(['linear']),
            'regressor__C': Real(low=1e-6, high=1e6)
        },
        {  # ElasticNet
            'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
            'feature_selection': Categorical(['passthrough',
                                              SelectFromModel(ElasticNet())]),
            'regressor': Categorical([ElasticNet()]),
            'regressor__alpha': Real(low=1, high=1e6),
            'regressor__l1_ratio': Real(low=0, high=1)
        },
        {  # SVR
            'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
            'feature_selection': Categorical(['passthrough']),
            'regressor': Categorical([SVR()]),
            'regressor__kernel': Categorical(['poly', 'rbf']),
            'regressor__C': Real(low=1e-6, high=1e6)
        },
        {  # DecisionTreeRegressor
            'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
            'feature_selection': Categorical(['passthrough',
                                              SelectFromModel(DecisionTreeRegressor())]),
            'regressor': Categorical([DecisionTreeRegressor()])
        },
        {  # KNeighborsRegressor
            'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
            'feature_selection': Categorical(['passthrough']),
            'regressor': Categorical([KNeighborsRegressor()]),
            'regressor__n_neighbors': Integer(low=5, high=150),
            'regressor__weights': Categorical(['uniform', 'distance'])
        },
        {  # RandomForestRegressor
            'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
            'feature_selection': Categorical(['passthrough',
                                              SelectFromModel(RandomForestRegressor())]),
            'regressor': Categorical([RandomForestRegressor()])
        },
        {  # MLPRegressor
            'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
            'feature_selection': Categorical(['passthrough']),
            'regressor': Categorical([MLPRegressor()])
        },
        {  # LinearRegression
            'scaler': Categorical([RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()]),
            'feature_selection': Categorical(['passthrough',
                                              SelectFromModel(LinearRegression())]),
            'regressor': Categorical([LinearRegression()])
        }
    ]

    ####################################################################################################################
    # Data
    ####################################################################################################################

    # Retrieve data from s3 and format into dataframe
    df = func.create_df_from_s3(bucket=bucket)

    # Parse into unique DataFrame for each type of real estate
    df_sf = df[df['description.type'] == 'single_family']

    # ID features
    list_features = [
        'description.year_built',
        'description.baths_full',
        'description.baths_3qtr',
        'description.baths_half',
        'description.baths_1qtr',
        'description.lot_sqft',
        'description.sqft',
        'description.garage',
        'description.beds',
        'description.stories',
        'location.address.coordinate.lon',
        'location.address.coordinate.lat',
        'list_price'
    ]

    df_sf_features = df_sf[list_features]

    logger.info('Starting Regressor Training')

    model = func.train_my_model(my_df=df_sf_features,
                                my_pipeline=regression_pipe,
                                my_param_grid=skopt_grid,
                                style='bayes',
                                search=100)

    logger.info('Regressor Training Complete')

    list_scores = func.score_my_model(my_df=df_sf_features, my_model=model)

    logger.info('Results from Search:')
    logger.info(f'Search best estimator: {model.best_estimator_}')
    logger.info(f'Search Best params: {model.best_params_}')
    logger.info(f"Search Cross Validation Scores: {list_scores[0]}")
    logger.info(f"Search Validation Score: %0.2f" % model.best_score_)
    logger.info(f"Search accuracy on test data: %0.2f (+/- %0.2f)" % (list_scores[1], list_scores[2]))
    logger.info(f"Search test score: %0.2f" % list_scores[3])
    logger.info(f"Search R2 score: %0.2f" % list_scores[4])


if __name__ == '__main__':
    main()
