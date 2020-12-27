from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

from utils.functions import train_my_model, score_my_model, create_logger, create_df_from_s3

#######################################################################################################################
# Config Log File
#######################################################################################################################

logger = create_logger(e_handler_name='logs/error_log.log', t_handler_name='logs/training_log.log')

#######################################################################################################################
# Data
#######################################################################################################################

# Retrieve data from s3 and format into dataframe
bucket = 're-raw-data'
prefix = '2020-12-27'

df = create_df_from_s3(bucket=bucket, prefix=prefix)

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

# Force to numeric, as pandas_to_s3 casts everything to strings
df_sf_features = df_sf_features.astype('float32')

best_catboost = CatBoostRegressor(depth=8,
                                  iterations=1500,
                                  learning_rate=0.05,
                                  loss_function='RMSE')

# Define Pipeline
regression_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(best_catboost)),
    ('regressor', CatBoostRegressor())
])

param_grid = [
    {  # VotingRegressor
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough', SelectFromModel(best_catboost)],
        'regressor': [VotingRegressor([
            ('catboost', best_catboost),
            ('randforest', RandomForestRegressor()),
            ('enet', ElasticNet(alpha=100.0, l1_ratio=1.0)),
            ('lr', LinearRegression()),
            ('kn', KNeighborsRegressor(weights='distance', n_neighbors=5))
        ])],
    }
]

logger.info('Starting Voting Regressor Training')

model = train_my_model(my_df=df_sf_features, my_pipeline=regression_pipe, my_param_grid=param_grid)

logger.info('Voting Regressor Training Complete')

list_scores = score_my_model(my_df=df_sf_features, my_model=model)

logger.info('Results from Voting Regressor Randomized Search (VGS):')
logger.info(f'VGS best estimator: {model.best_estimator_}')
logger.info(f'VGS Validation Score: %0.2f' % model.best_score_)
logger.info(f'VGS Best params: {model.best_params_}')
logger.info(f'VGS Cross Validation Scores: {list_scores[0]}')
logger.info(f"VGS accuracy on all data: %0.2f (+/- %0.2f)" % (list_scores[1], list_scores[2]))
logger.info(f"VGS test score: %0.2f" % list_scores[3])
logger.info(f"VGS R2 score: %0.2f" % list_scores[4])

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
# s3 Data - Durham, Raleigh, Greensboro, Fayetteville, Charlotte
#######################################################################################################################

# 2020-12-27 13:59:55,668:MainProcess:root:INFO:Results from Voting Regressor Randomized Search (VGS):

# 2020-12-27 13:59:55,691:MainProcess:root:INFO:VGS best estimator: Pipeline(steps=[('scaler', StandardScaler()),
#                 ('feature_selection', 'passthrough'),
#                 ('regressor',
#                  VotingRegressor(estimators=[('catboost',
#                                               <catboost.core.CatBoostRegressor object at 0x0000020274D79288>),
#                                              ('randforest',
#                                               RandomForestRegressor()),
#                                              ('enet',
#                                               ElasticNet(alpha=100.0,
#                                                          l1_ratio=1.0)),
#                                              ('lr', LinearRegression()),
#                                              ('kn',
#                                               KNeighborsRegressor(weights='distance'))]))])

# 2020-12-27 13:59:55,691:MainProcess:root:INFO:VGS Validation Score: 0.75

# 2020-12-27 13:59:55,695:MainProcess:root:INFO:VGS Best params:
# {'scaler': StandardScaler(),
# 'regressor': VotingRegressor(estimators=[('catboost',
#                              <catboost.core.CatBoostRegressor object at 0x00000202730BD548>),
#                             ('randforest', RandomForestRegressor()),
#                             ('enet', ElasticNet(alpha=100.0, l1_ratio=1.0)),
#                             ('lr', LinearRegression()),
#                             ('kn', KNeighborsRegressor(weights='distance'))]),
#                             'feature_selection': 'passthrough'}

# 2020-12-27 13:59:55,695:MainProcess:root:INFO:VGS Cross Validation Scores:
# [   0.74659476    0.70264914 -182.24063743    0.29708606    0.7109509 ]

# 2020-12-27 13:59:55,695:MainProcess:root:INFO:VGS accuracy on all data: -35.96 (+/- 146.28)

# 2020-12-27 13:59:55,695:MainProcess:root:INFO:VGS test score: 0.75

# 2020-12-27 13:59:55,695:MainProcess:root:INFO:VGS R2 score: 0.75
