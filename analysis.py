import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from catboost import Pool, CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from utils.functions import RealEstateData

#######################################################################################################################
# Placeholder for user input
#######################################################################################################################

str_city = 'Durham'
str_state_code = 'NC'

#######################################################################################################################
# Initialize City Data
#######################################################################################################################

# def main(str_city, str_state_code):
# durham = RealEstateData(str_city, str_state_code)


# if __name__ == '__main__':
df_durham = RealEstateData(str_city, str_state_code).results

#######################################################################################################################
# Data Preparation
#######################################################################################################################

# ID metadata to drop
list_drop_columns = [
    'permalink',
    'photos',
    'community',
    'virtual_tours',
    'matterport',
    'primary_photo.href',
    'source.plan_id',
    'source.agents',
    'source.spec_id',
    'description.sub_type',
    'lead_attributes.show_contact_an_agent',
    'other_listings.rdc',
    'primary_photo',
    'products',
    'community.advertisers',
    'community.description.name',
    'location.address.coordinate',
    'other_listings',
    'location.county'
]

df_durham.drop(list_drop_columns, axis=1, inplace=True)

# Parse into unique DataFrame for each type of real estate
df_sf = df_durham[df_durham['description.type'] == 'single_family']
# df_th = df_durham[df_durham['description.type'] == 'townhomes']
# df_la = df_durham[df_durham['description.type'] == 'land']
# df_mf = df_durham[df_durham['description.type'] == 'multi_family']
# array = ['condos', 'condo_townhome_rowhome_coop', 'condo_townhome']
# df_co = df_durham.loc[df_durham['description.type'].isin(array)]

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
# df_th_features = df_th[list_features]
# df_la_features = df_la[list_features]
# df_mf_features = df_mf[list_features]
# df_co_features = df_co[list_features]

# Clean DataFrame
df_sf_features_clean = df_sf_features.fillna(0, axis=1)

# split into input and output elements
X, y = df_sf_features_clean.drop('list_price', axis=1).values, df_sf_features_clean['list_price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Pipeline
regression_pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('feature_selection', SelectFromModel(LinearSVR())),
    ('regressor', CatBoostRegressor())
])

param_grid = [
    {  # Best Performing after Trials, exploring space further
        'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
        'feature_selection': ['passthrough',
                              SelectFromModel(CatBoostRegressor()),
                              RFECV(CatBoostRegressor(), step=0.2)],
        'regressor': [CatBoostRegressor()],
        'regressor__depth': [4, 6, 8, 10],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__iterations': [500, 1000, 1500],
        'regressor__loss_function': ['RMSE']
    }
    # {  # Removed to save compute
    #     'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
    #     'feature_selection': ['passthrough',
    #                           SelectFromModel(GaussianProcessRegressor()),
    #                           RFECV(GaussianProcessRegressor(), step=0.2)],
    #     'regressor': [GaussianProcessRegressor()],
    #     'regressor__kernel': [DotProduct() + WhiteKernel()]
    # },
    # {  # Removed to save compute
    #     'scaler': [RobustScaler(), StandardScaler(), PowerTransformer()],
    #     'feature_selection': ['passthrough',
    #                           SelectFromModel(DecisionTreeRegressor()),
    #                           RFECV(DecisionTreeRegressor(), step=0.2)],
    #     'regressor': [DecisionTreeRegressor()]
    # },
    # {  # Removed to save compute
    #     'scaler': [RobustScaler(), StandardScaler(), PowerTransformer()],
    #     ''feature_selection': ['passthrough',
    #                            SelectFromModel(RandomForestRegressor()),
    #                            RFECV(RandomForestRegressor(), step=0.2)],
    #     'regressor': [RandomForestRegressor()]
    # },
    # {  # Removed to save compute
    #     'scaler': [RobustScaler(), StandardScaler(), PowerTransformer()],
    #     'feature_selection': ['passthrough',
    #                           SelectFromModel(MLPRegressor()),
    #                           RFECV(MLPRegressor(), step=0.2)],
    #     'regressor': [MLPRegressor()]
    #     # 'regressor__activation': ['identity', 'logistic', 'tanh', 'relu'],
    #     # 'regressor__solver': ['lbfgs', 'sgd', 'adam']
    # },
    # {  # Removed to save compute
    #     'scaler': [RobustScaler(), StandardScaler(), PowerTransformer()],
    #     'feature_selection': ['passthrough',
    #                           SelectFromModel(LinearRegression()),
    #                           RFECV(LinearRegression(), step=0.2)],
    #     'regressor': [LinearRegression()]
    # }
]

grid = GridSearchCV(estimator=regression_pipe,
                    param_grid=param_grid,
                    cv=5,
                    n_jobs=-1)

grid.fit(X_train, y_train)

scores = cross_val_score(grid.best_estimator_, X_test, y_test)

y_pred = grid.predict(X_test)

r2 = r2_score(y_test, y_pred)

print(" Results from Grid Search ")
print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)
print("\n The best score across ALL searched params:\n", grid.best_score_)
print("\n The best parameters across ALL searched params:\n", grid.best_params_)
print("\n Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# The best estimator across ALL searched params:
#  Pipeline(steps=[('scaler', PowerTransformer()),
#                 ('regressor',
#                  <catboost.core.CatBoostRegressor object at 0x0000019BF77B35C8>)])

# The best score across ALL searched params:
#   0.7180125935168109

# The best parameters across ALL searched params:
#  {'regressor': <catboost.core.CatBoostRegressor object at 0x0000019BF865FD08>,
#  'regressor__depth': 4,
#  'regressor__iterations': 1500,
#  'regressor__learning_rate': 0.1,
#  'regressor__loss_function': 'RMSE',
#  'scaler': PowerTransformer()}

# Cross Validation Score:
#  [0.69006381, 0.82823126, 0.74398594, 0.51184881, 0.78272186]

# Accuracy: 0.71 (+/- 0.22)
