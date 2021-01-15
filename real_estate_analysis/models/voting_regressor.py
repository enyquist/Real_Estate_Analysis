from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.feature_selection import SelectFromModel

import real_estate_analysis.utils.functions as func


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/error_log.log', t_handler_name='../logs/training_log.log')

    ####################################################################################################################
    # Define Pipeline and variables
    ####################################################################################################################

    bucket = 're-raw-data'

    # Read GridSearchCV csv results and create a dictionary with the best models
    dict_best_models = func.create_best_models('../../data/models/output/searchcv.csv')

    # Create VotingRegressors made up of the best models from 2 to N estimators
    list_voting_regressors = func.create_model_combinations(dict_best_models)

    # Define Pipeline - default to catboost because it was the highest performing model
    regression_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(dict_best_models['catboost'])),
        ('regressor', dict_best_models['catboost'])
    ])

    param_grid = [
        {  # VotingRegressor
            'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
            'feature_selection': ['passthrough',
                                  SelectFromModel(dict_best_models['catboost'])],
            'regressor': list_voting_regressors
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

    df_features = df_sf[list_features]

    logger.info('Starting Voting Regressor Training')

    model = func.train_my_model(my_df=df_features, my_pipeline=regression_pipe, my_param_grid=param_grid, style='grid',
                                filename='voting_regressor')

    logger.info('Voting Regressor Training Complete')

    list_scores = func.score_my_model(my_df=df_features, my_model=model)

    logger.info('Results from Voting Regressor Search (VRS):')
    logger.info(f'VRS best estimator: {model.best_estimator_}')
    logger.info(f'VRS Best params: {model.best_params_}')
    logger.info(f"VRS Cross Validation Scores: {list_scores[0]}")
    logger.info(f'VRS Validation Score: %0.2f' % model.best_score_)
    logger.info(f"VRS accuracy on test data: %0.2f (+/- %0.2f)" % (list_scores[1], list_scores[2]))
    logger.info(f"VRS test score: %0.2f" % list_scores[3])
    logger.info(f"VRS R2 score: %0.2f" % list_scores[4])


if __name__ == '__main__':
    main()

#######################################################################################################################
# RDU Data
#######################################################################################################################

# 2020-12-23 14:49:29,555:MainProcess:root:INFO:Results from Voting Regressor Randomized Search (VGS):

# 2020-12-23 14:49:29,579:MainProcess:root:INFO:VGS best estimator: Pipeline(steps=[('scaler', StandardScaler()),
#                 ('feature_selection',
#                  SelectFromModel(estimator=<catboost.core.CatBoostRegressor object at 0x0000021370F43348>)),
#                 ('regressor',
#                  VotingRegressor(estimators=[('catboost',
#                                               <catboost.core.CatBoostRegressor object at 0x0000021370F43088>),
#                                              ('randforest',
#                                               RandomForestRegressor()),
#                                              ('enet',
#                                               ElasticNet(alpha=100.0,
#                                                          l1_ratio=1.0)),
#                                              ('lr', LinearRegression()),
#                                              ('kn',
#                                               KNeighborsRegressor(weights='distance'))]))])

# 2020-12-23 14:49:29,579:MainProcess:root:INFO:VGS Validation Score: 0.76

# 2020-12-23 14:49:29,583:MainProcess:root:INFO:VGS Best params:
# {'scaler': StandardScaler(),
# 'regressor': VotingRegressor(estimators=[('catboost',
#                              <catboost.core.CatBoostRegressor object at 0x0000021370F54F88>),
#                             ('randforest', RandomForestRegressor()),
#                             ('enet', ElasticNet(alpha=100.0, l1_ratio=1.0)),
#                             ('lr', LinearRegression()),
#                             ('kn', KNeighborsRegressor(weights='distance'))]),
# 'feature_selection': SelectFromModel(estimator=<catboost.core.CatBoostRegressor object at 0x0000021370F54508>)}

# 2020-12-23 14:49:29,583:MainProcess:root:INFO:VGS Cross Validation Scores:
# [0.7681488  0.63478995 0.82113333 0.68365414 0.6043568 ]

# 2020-12-23 14:49:29,583:MainProcess:root:INFO:VGS accuracy on all data: 0.70 (+/- 0.16)

# 2020-12-23 14:49:29,583:MainProcess:root:INFO:VGS test score: 0.84

# 2020-12-23 14:49:29,583:MainProcess:root:INFO:VGS R2 score: 0.84

#######################################################################################################################
# Durham Data
#######################################################################################################################

# 2020-12-23 14:53:41,352:MainProcess:root:INFO:Results from Voting Regressor Randomized Search (VGS):

# 2020-12-23 14:53:41,376:MainProcess:root:INFO:VGS best estimator: Pipeline(steps=[('scaler', RobustScaler()),
#                 ('feature_selection',
#                  SelectFromModel(estimator=<catboost.core.CatBoostRegressor object at 0x00000213740966C8>)),
#                 ('regressor',
#                  VotingRegressor(estimators=[('catboost',
#                                               <catboost.core.CatBoostRegressor object at 0x0000021374096288>),
#                                              ('randforest',
#                                               RandomForestRegressor()),
#                                              ('enet',
#                                               ElasticNet(alpha=100.0,
#                                                          l1_ratio=1.0)),
#                                              ('lr', LinearRegression()),
#                                              ('kn',
#                                               KNeighborsRegressor(weights='distance'))]))])

# 2020-12-23 14:53:41,376:MainProcess:root:INFO:VGS Validation Score: 0.64

# 2020-12-23 14:53:41,380:MainProcess:root:INFO:VGS Best params:
# {'scaler': RobustScaler(),
# 'regressor': VotingRegressor(estimators=[('catboost',
#                              <catboost.core.CatBoostRegressor object at 0x0000021371FF5E88>),
#                             ('randforest', RandomForestRegressor()),
#                             ('enet', ElasticNet(alpha=100.0, l1_ratio=1.0)),
#                             ('lr', LinearRegression()),
#                             ('kn', KNeighborsRegressor(weights='distance'))]),
# 'feature_selection': SelectFromModel(estimator=<catboost.core.CatBoostRegressor object at 0x0000021370AEA388>)}

# 2020-12-23 14:53:41,380:MainProcess:root:INFO:VGS Cross Validation Scores:
# [0.66578107 0.83279792 0.69898508 0.63089407 0.71967524]

# 2020-12-23 14:53:41,380:MainProcess:root:INFO:VGS accuracy on all data: 0.71 (+/- 0.14)

# 2020-12-23 14:53:41,380:MainProcess:root:INFO:VGS test score: 0.67

#######################################################################################################################
# s3 Data - Durham, Raleigh, Greensboro, Fayetteville, Charlotte, Maple Grove, Minneapolis, New Hope, Plymouth
#######################################################################################################################

# 2021-01-09 04:16:33,590:MainProcess:root:INFO:Results from Voting Regressor Search (VRS):

# 2021-01-09 04:16:33,599:MainProcess:root:INFO:VRS best estimator: Pipeline(steps=[('scaler', StandardScaler()),
#                 ('feature_selection', 'passthrough'),
#                 ('regressor',
#                  VotingRegressor(estimators=[('catboost',
#                                               <catboost.core.CatBoostRegressor object at 0x00000224769E6908>),
#                                              ('RandomForest',
#                                               RandomForestRegressor())]))])

# 2021-01-09 04:16:33,600:MainProcess:root:INFO:VRS Best params:
# {'feature_selection': 'passthrough',
# 'regressor': VotingRegressor(estimators=[('catboost',
#                              <catboost.core.CatBoostRegressor object at 0x0000022474C43EC8>),
#                             ('RandomForest', RandomForestRegressor())]),
# 'scaler': StandardScaler()}

# 2021-01-09 04:16:33,600:MainProcess:root:INFO:VRS Cross Validation Scores:
# [0.77559715 0.8461415  0.80056253 0.7301138  0.75932304]

# 2021-01-09 04:16:33,600:MainProcess:root:INFO:VRS Validation Score: 0.81

# 2021-01-09 04:16:33,600:MainProcess:root:INFO:VRS accuracy on test data: 0.78 (+/- 0.08)

# 2021-01-09 04:16:33,601:MainProcess:root:INFO:VRS test score: 0.85

# 2021-01-09 04:16:33,601:MainProcess:root:INFO:VRS R2 score: 0.85

#######################################################################################################################
# Durham direct query vs Durham from s3 - confirmed that these objects are identical 12/29/20
#######################################################################################################################

# Durham Direct Query

# Validation Score: 0.66

# VGS accuracy on all data: 0.74 (+/- 0.18)

# VGS test score:  0.58

# VGS R2 score:  0.58

# Durham from s3 - with preprocessing

# VGS Validation Score: 0.67

# VGS accuracy on all data:0.74 (+/- 0.18)

# VGS test score: 0.59

# VGS R2 score: 0.59
