import boto3
import numpy as np
import pandas as pd
import logging
from io import StringIO, BytesIO
import gzip
import itertools
import configparser
import joblib
import os
from sklearn import model_selection
from sklearn import ensemble
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from skopt import BayesSearchCV
from skopt.callbacks import DeltaXStopper
from catboost import CatBoostRegressor

from real_estate_analysis.utils.Model import ModelObject

config = configparser.ConfigParser()
config.read('../config.ini')

logger = logging.getLogger(__name__)


def prep_gpu():
    """
    Function to turn on tensorflow and configure gpu
    :return:
    """
    import tensorflow as tf

    # Set GPU
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


def create_logger(e_handler_name, t_handler_name):
    """
    Wrapper to create logger for errors and training records
    :param e_handler_name: filepath to logger as string
    :param t_handler_name: filepath to logger as string
    :return: logger object
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create Handlers
    c_handler = logging.StreamHandler()
    e_handler = logging.FileHandler(filename=e_handler_name)
    t_handler = logging.FileHandler(filename=t_handler_name)

    # Create Log Format(s)
    f_format = logging.Formatter('%(asctime)s:%(processName)s:%(name)s:%(levelname)s:%(message)s')

    # Set handler levels
    c_handler.setLevel(logging.INFO)
    e_handler.setLevel(logging.ERROR)
    t_handler.setLevel(logging.INFO)

    # Set handler formats
    c_handler.setFormatter(f_format)
    e_handler.setFormatter(f_format)
    t_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(e_handler)
    logger.addHandler(t_handler)

    return logger


def prepare_my_data(my_df):
    """

    :param my_df:
    :return:
    """

    folder = 'sold'
    na_subset = ['price', 'address.lon', 'address.lat', 'address.state_code']
    drop_subset = ['price', 'address.state_code']
    price = 'price'
    state = 'address.state_code'
    building_size = 'building_size.size'
    lot_size = 'lot_size.size'
    total_rooms = ['baths_full', 'baths_half', 'beds']

    # Force to numeric, as pandas_to_s3 casts everything to strings, ignore the categorical data
    my_df = my_df.apply(lambda col: pd.to_numeric(col, errors='ignore', downcast='float'))

    # Drop entries that have no price, lat / long, tags (if appropriate to the data set), or associated city
    my_df = my_df.dropna(axis=0, subset=na_subset)

    # Feature Engineering
    my_df['price_per_sqft'] = (my_df[price] / my_df[building_size]).replace([np.nan, np.inf], -1)
    my_df['price_per_lot'] = (my_df[price] / my_df[lot_size]).replace([np.nan, np.inf], -1)
    my_df['total_rooms'] = np.sum(my_df[total_rooms], axis=1)
    my_df['state_encoding'] = pd.Categorical(my_df[state], categories=my_df[state].unique()).codes

    my_df = my_df.reset_index(drop=True)

    # split into features and targets elements, taking logarithm of targets
    X, y = my_df.drop(drop_subset, axis=1), np.log(my_df[[price]])

    # Split data, stratifying by state
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42,
                                                                        stratify=my_df[state])

    # Impute missing values
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    train_imp = imp.fit_transform(X_train)
    test_imp = imp.transform(X_test)

    # Save imputer for later ETL
    with open(f'../../data/models/{folder}/imputer.joblib', 'wb') as file:
        joblib.dump(imp, file, compress='lzma')

    # Outlier Detection and remove outliers
    clf = ensemble.IsolationForest(random_state=42).fit_predict(train_imp)
    mask = np.where(clf == -1)
    train_no_outlier = np.delete(train_imp, mask, axis=0)
    y_train_series = np.delete(y_train.values, mask, axis=0)

    # Scale train and test set
    scaler = StandardScaler()
    train_scale = scaler.fit_transform(train_no_outlier)
    test_scale = scaler.transform(test_imp)

    # Save scaler for later ETL
    with open(f'../../data/models/{folder}/scaler.joblib', 'wb') as file:
        joblib.dump(scaler, file, compress='lzma')

    # Back to DataFrame
    df_train_inter = pd.DataFrame(data=train_scale, columns=X_train.columns)
    df_test_inter = pd.DataFrame(data=test_scale, columns=X_test.columns)
    y_train_inter = pd.DataFrame(data=y_train_series, columns=y_train.columns)

    # Reset indicies
    y_test = y_test.reset_index(drop=True)

    # Concat prices back to DataFrame
    df_train_final = pd.concat([df_train_inter, y_train_inter], axis=1)
    df_test_final = pd.concat([df_test_inter, y_test], axis=1)

    return df_train_final, df_test_final


def train_my_model(my_pipeline, my_param_grid, x_train, y_train, search=50, style='random', filename='searchcv',
                   n_jobs=-1):
    """
    Wrapper for training a regression model to predict housing prices
    :param my_pipeline: Training Pipeline, an instance of sklearn's Pipeline
    :param my_param_grid: Dictionary containing the parameters to train over
    :param x_train: Feature training set
    :param y_train: Target training set
    :param search: Number of iterations to search in RandomizedSearchCV, default 50
    :param style: Search style, default to RandomizedSearchCV
    :param filename: filename to save off files
    :param n_jobs: n_jobs to conduct training
    :return: RandomizedSearchCV object
    """

    if style == 'random':
        # Create RandomizedSearchCV instance
        cv = model_selection.RandomizedSearchCV(estimator=my_pipeline,
                                                param_distributions=my_param_grid,
                                                n_iter=search,
                                                cv=5,
                                                n_jobs=n_jobs,
                                                verbose=3)

        cv.fit(x_train, y_train)

        return cv

    if style == 'grid':
        # Create RandomizedSearchCV instance
        cv = model_selection.GridSearchCV(estimator=my_pipeline,
                                          param_grid=my_param_grid,
                                          cv=5,
                                          n_jobs=n_jobs,
                                          verbose=3)

        cv.fit(x_train, y_train)

        # Save cv_results to inspect for best model params
        df = pd.DataFrame.from_dict(cv.cv_results_)

        try:
            df.to_csv(f'../../data/models/output/{filename}.csv', index=False)
        except FileNotFoundError as e:
            pass

        return cv

    if style == 'bayes':
        cv = BayesSearchCV(estimator=my_pipeline,
                           search_spaces=my_param_grid,
                           n_iter=search,
                           n_jobs=15,
                           cv=5)

        cv.fit(x_train, y_train, callback=DeltaXStopper(0.01))

        return cv

    else:  # todo notify user of error
        return -1


def score_my_model(my_model, x_train, y_train, x_test, y_test):
    """
    Wrapper to score a model
    :param x_train: Features train set
    :param y_train: Target train set
    :param x_test: Features test set
    :param y_test: Target test set
    :param my_model: an instance of an sklearn RandomizedSearchCV model. Must have a .best_estimator_ method implemented
    :return:
    """

    if isinstance(my_model, xgb.XGBRegressor) or isinstance(my_model, CatBoostRegressor):
        test_scores = model_selection.cross_val_score(my_model, x_test, y_test)
        train_scores = model_selection.cross_val_score(my_model, x_train, y_train)
    else:
        test_scores = model_selection.cross_val_score(my_model.best_estimator_, x_test, y_test)
        train_scores = model_selection.cross_val_score(my_model.best_estimator_, x_train, y_train)

    # Score model on Test Data
    y_pred = my_model.predict(x_test)
    test_mse = metrics.mean_squared_error(y_test, y_pred)
    test_mae = metrics.mean_absolute_error(y_test, y_pred)
    test_evs = metrics.explained_variance_score(y_test, y_pred)
    test_mdae = metrics.median_absolute_error(y_test, y_pred)
    test_r2 = metrics.r2_score(y_test, y_pred)
    test_max_error = metrics.max_error(y_test, y_pred)

    dict_scores = {
        'test_cross_val_score': test_scores,
        'test_mean_cross_val_score': test_scores.mean(),
        'test_std_cross_val_score': test_scores.std() * 2,
        'train_cross_val_score': train_scores,
        'train_mean_cross_val_score': train_scores.mean(),
        'train_std_cross_val_score': train_scores.std() * 2,
        'test_r2': test_r2,
        'test_mean_squared_error': test_mse,
        'test_mean_absolute_error': test_mae,
        'test_explained_variance_score': test_evs,
        'test_median_absolute_error': test_mdae,
        'test_max_error': test_max_error,
    }

    return dict_scores


def pandas_to_s3(df, client, bucket, key):
    """
    Wrapper to stream DataFrames to an s3 bucket. Credit to Lukasz uhho
    https://gist.github.com/uhho/a1490ae2abd112b556dcd539750aa151
    :param df: DataFrame
    :param client: s3 client
    :param bucket: s3 bucket
    :param key: Key of object or path to object in non-s3 terms
    :return: API response metadata
    """
    # Write DF to string stream
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Reset stream position
    csv_buffer.seek(0)
    # Create binary stream
    gz_buffer = BytesIO()

    # Compress string stream using gzip
    with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
        gz_file.write(bytes(csv_buffer.getvalue(), 'utf-8'))

    # Write stream to S3
    response = client.put_object(Bucket=bucket, Key=key, Body=gz_buffer.getvalue())

    return response


def s3_to_pandas(client, bucket, key, header=None):
    """
    Wrapper to stream DataFrames from an s3 bucket. Credit to Lukasz uhho
    https://gist.github.com/uhho/a1490ae2abd112b556dcd539750aa151
    :param client: s3 client
    :param bucket: s3 bucket
    :param key: Key of object or path to object in non-s3 terms
    :param header:
    :return:
    """
    # Get key using boto3 client
    obj = client.get_object(Bucket=bucket, Key=key)
    gz = gzip.GzipFile(fileobj=obj['Body'])

    # load stream directly to DF
    return pd.read_csv(gz, header=header, dtype=str)


def s3_to_pandas_with_processing(client, bucket, key, header=None):
    """
    Wrapper to stream DataFrames from an s3 bucket. Credit to Lukasz uhho
    https://gist.github.com/uhho/a1490ae2abd112b556dcd539750aa151
    :param client: s3 client
    :param bucket: s3 bucket
    :param key: Key of object or path to object in non-s3 terms
    :param header:
    :return:
    """
    # Get key using boto3 client
    obj = client.get_object(Bucket=bucket, Key=key)
    gz = gzip.GzipFile(fileobj=obj['Body'])

    # Replace some characters in incomming stream and load it to DF
    lines = "\n".join([line.replace('?', ' ') for line in gz.read().decode('utf-8').split('\n')])
    return pd.read_csv(StringIO(lines), header=header, dtype=str)


def create_df_from_s3(bucket='re-raw-data'):
    """
    Wrapper to collect and append DataFrames from an s3 bucket
    :param bucket: Name of the s3 bucket
    :return: Compiled DataFrame from the s3 bucket
    """
    s3 = boto3.client('s3',
                      region_name='us-east-1',
                      aws_access_key_id=config.get('AWS', 'aws_access_key_id'),
                      aws_secret_access_key=config.get('AWS', 'aws_secret_access_key'))

    # Paginate s3 bucket because objects exceeds 1,000
    paginator = s3.get_paginator('list_objects_v2')

    # Get response from s3 with data from bucket re-raw-data
    pages = paginator.paginate(Bucket=bucket)

    list_data = []
    for page in pages:
        list_contents = page['Contents']
        list_data.extend(list_contents)

    list_formatted_data = []
    for content in list_data:
        key = content['Key']
        data = s3_to_pandas_with_processing(client=s3, bucket=bucket, key=key, header=0)
        list_formatted_data.append(data)

    # Concat data into master Dataframe
    df = pd.concat(list_formatted_data)

    # Drop any duplicates in the dataset
    df = df.drop_duplicates()

    return df


def fetch_from_s3(bucket, key):
    """

    :param bucket:
    :param key:
    :return:
    """
    s3 = boto3.client('s3',
                      region_name='us-east-1',
                      aws_access_key_id=config.get('AWS', 'aws_access_key_id'),
                      aws_secret_access_key=config.get('AWS', 'aws_secret_access_key'))

    data = s3_to_pandas_with_processing(client=s3, bucket=bucket, key=key)
    data.columns = data.iloc[0]
    data = data.drop(0)
    data = data.reset_index(drop=True)
    data = data.apply(lambda col: pd.to_numeric(col, errors='ignore'))

    return data


def create_best_models(results_path):
    """
    Creates the best models from the grid search
    :param results_path: file path to output from grid search
    :return: Dictionary containing best models that meet the threshold requirements
    """

    df = pd.read_csv(results_path)

    # List of model names from searchcv.py, used as dictionary keys
    list_model_names = ['catboost',
                        'RandomForest',
                        'KNeighbors',
                        'DecisionTree',
                        'ElasticNet',
                        'LinearRegression',
                        'SVR',
                        'GaussianProcess',
                        'MLP']

    # Capture models if they meet threshold requirements and store data
    dict_best_models = {}

    for model in list_model_names:
        df_models = df[df['param_regressor'].str.contains(model)]
        df_best_model = df_models.loc[df_models['mean_test_score'].idxmax()]
        if df_best_model['mean_test_score'] > float(config.get('DEFAULT', 'mean_test_score_threshold')) or \
                df_best_model['std_test_score'] < float(config.get('DEFAULT', 'std_test_score_threshold')):
            dict_best_models[model] = df_best_model

    # Dictionary that contains the best params for each model type, used as a Switch-Case statement
    dict_model_format = {
        'catboost': CatBoostRegressor(depth=int(dict_best_models['catboost'].param_regressor__depth),
                                      iterations=int(dict_best_models['catboost'].param_regressor__iterations),
                                      learning_rate=dict_best_models['catboost'].param_regressor__learning_rate,
                                      loss_function='RMSE'),

        # RandomForest is default in searchcv.py
        'RandomForest': model_selection.RandomForestRegressor(),

        'KNeighbors': KNeighborsRegressor(n_neighbors=int(dict_best_models['KNeighbors'].param_regressor__n_neighbors),
                                          weights=dict_best_models['KNeighbors'].param_regressor__weights),

        # DecisionTree is default in searchcv.py
        'DecisionTree': DecisionTreeRegressor(),

        'ElasticNet': linear_model.ElasticNet(alpha=dict_best_models['ElasticNet'].param_regressor__alpha,
                                              l1_ratio=dict_best_models['ElasticNet'].param_regressor__l1_ratio),

        # LinearRegression is default in searchcv.py
        'LinearRegression': linear_model.LinearRegression(),

        'SVR': SVR(kernel=dict_best_models['SVR'].param_regressor__kernel,
                   C=dict_best_models['SVR'].param_regressor__C),

        # GaussianProcess only had one type of kernel in searchcv.py
        'GaussianProcess': GaussianProcessRegressor(kernel=[DotProduct() + WhiteKernel()]),

        # MLP is default in searchcv.py
        'MLP': MLPRegressor()
    }

    # Capture best models for output
    dict_output = {}

    for key in dict_best_models.keys():
        dict_output[key] = dict_model_format[key]

    return dict_output


def create_model_combinations(dict_input):
    """
    Creates combinations length 2 to N of an input dictionary dict_input
    :param dict_input: input dictionary to make combinations from - key is model name, value is model object
    :return: list of tuples length 2 to N
    """

    list_model_combinations = []

    for length in range(2, len(dict_input)):
        list_model_combinations.extend(list(itertools.combinations(dict_input, r=length)))

    # Only maintain combinations with "Catboost" as a model option as it was high performing
    list_model_combinations = [i for i in list_model_combinations if 'catboost' in i]

    list_voting_regressors = []

    for combo in list_model_combinations:
        list_model_format = []
        for model in combo:
            list_model_format.append((f'{model}', dict_input[model]))
        list_voting_regressors.append(model_selection.VotingRegressor(list_model_format))

    return list_voting_regressors


def create_model(input_size=12, hidden_layers=3):
    """
    Simple wrapper to create a deep learning model using the Keras API
    :param input_size: int, the size of the input (number of features)
    :param hidden_layers: Int, how many hidden layers to create in the model
    :return:
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import activations, metrics

    # Create model
    model = Sequential()

    # Input Layer
    model.add(Dense(units=input_size, input_shape=(input_size,),
                    kernel_initializer='normal', activation=activations.relu, name='Input_layer'))

    # Hidden Layers
    for layer in range(1, hidden_layers + 1):
        model.add(Dense(input_size, kernel_initializer='normal', activation=activations.relu, name=f'Hidden_{layer}'))

    model.add(Dropout(0.05, name=f'Dropout_1'))

    # Output Layer
    model.add(Dense(units=1, kernel_initializer='normal', activation='linear', name='Output_layer'))

    optimizer = Adam(lr=10e-6)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[coeff_determination,
                                                                           metrics.MeanSquaredError()])

    return model


def coeff_determination(y_true, y_pred):
    """
    Custom Keras metric, R2 score
    :param y_true: true label
    :param y_pred: predicted label
    :return:
    """
    from keras import backend as K

    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def xgb_cv(max_depth, min_child_weight, eta, subsample, colsample_bytree, gamma, lambda_, alpha, dtrain):
    """
    Function to perform cross validation of an XGBoost model
    :param max_depth: Maximum depth of a tree
    :param min_child_weight: Minimum sum of instance weight (hessian) needed in a child
    :param eta: Step size shrinkage used in update to prevents overfitting
    :param subsample: Subsample ratio of the training instances
    :param colsample_bytree: subsample ratio of columns when constructing each tree
    :param gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree
    :param lambda_: L2 regularization term on weights
    :param alpha: L1 regularization term on weights
    :param dtrain: Training matrix of type xgb.DMatrix
    :return: Mean Squared Error of XGBoost CV
    """
    params = {
        'max_depth': int(max_depth),
        'eta': eta,
        'verbosity': 2,
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'tree_method': 'gpu_hist',
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': alpha,
        'reg_lambda': lambda_,
        'random_state': 42
    }

    cv_result = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=999,
        seed=42,
        nfold=5,
        metrics=['rmse', 'mae'],
        early_stopping_rounds=10
    )

    mse = cv_result['test-rmse-mean'].iloc[-1] ** 2

    return -1 * mse


def optimize_xgb(dtrain, pbounds, n_iter, init_points):
    """
    Performs Bayesian Optimization of an XGBoost Model
    :param dtrain: Training matrix of type xgb.DMatrix
    :param pbounds: Parameter Bounds for Optimization
    :param n_iter: Number of exploitation iterations
    :param init_points: Number of initialization points, to explore the Gaussian Process
    :return: Bayesian Optimized Object
    """

    def xgb_crossval(max_depth, min_child_weight, eta, subsample, colsample_bytree, gamma, lambda_, alpha):
        """
        Wrapper for xgb_cv
        :param max_depth: Maximum depth of a tree
        :param min_child_weight: Minimum sum of instance weight (hessian) needed in a child
        :param eta: Step size shrinkage used in update to prevents overfitting
        :param subsample: Subsample ratio of the training instances
        :param colsample_bytree: subsample ratio of columns when constructing each tree
        :param gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree
        :param lambda_: L2 regularization term on weights
        :param alpha: L1 regularization term on weights
        :return: Mean Squared Error of XGBoost CV
        """
        return xgb_cv(max_depth, min_child_weight, eta, subsample, colsample_bytree, gamma, lambda_, alpha, dtrain)

    optimizer = BayesianOptimization(
        f=xgb_crossval,
        pbounds=pbounds,
        bounds_transformer=SequentialDomainReductionTransformer(),
        random_state=42,
        verbose=2
    )

    optimizer.maximize(n_iter=n_iter, init_points=init_points, acq='ei')

    return optimizer


def catboost_cv(learning_rate, l2_leaf_reg, bagging_temperature, depth, x_train, y_train):
    """
    Function to perform cross validation of an CatBoost model
    :param learning_rate:
    :param l2_leaf_reg:
    :param bagging_temperature:
    :param depth:
    :param x_train:
    :param y_train:
    :return:
    """

    params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'iterations': 999,
        'learning_rate': learning_rate,
        'random_seed': 42,
        'l2_leaf_reg': l2_leaf_reg,
        'bagging_temperature': bagging_temperature,
        'depth': int(depth),
        'early_stopping_rounds': 10,
        'od_pval': 10e-2,
        'task_type': 'GPU',
        'logging_level': 'Silent',
        'boosting_type': 'Plain'
    }

    catboost = CatBoostRegressor(**params)

    cv_result = model_selection.cross_validate(estimator=catboost, X=x_train, y=y_train)

    mse = cv_result['test_score'] ** 2

    return -1.0 * mse


def optimize_catboost(x_train, y_train, pbounds, n_iter, init_points):
    """
    Performs Bayesian Optimization of an CatBoost Model
    :param x_train:
    :param y_train:
    :param pbounds:
    :param n_iter:
    :param init_points:
    :return:
    """

    def catboost_crossval(learning_rate, l2_leaf_reg, bagging_temperature, depth):
        """
        Wrapper for catboost_cv
        :param learning_rate:
        :param l2_leaf_reg:
        :param bagging_temperature:
        :param depth:
        :return:
        """
        return catboost_cv(learning_rate, l2_leaf_reg, bagging_temperature, depth, x_train, y_train)

    optimizer = BayesianOptimization(
        f=catboost_crossval,
        pbounds=pbounds,
        bounds_transformer=SequentialDomainReductionTransformer(),
        random_state=42,
        verbose=2
    )

    optimizer.maximize(n_iter=n_iter, init_points=init_points, acq='ei')

    return optimizer


def validate_model(model, dict_scores, file_name):
    """
    Check if a new model performs better than an existing model. If True, save the model and archive the previous model
    :param model: The model to check
    :param dict_scores: Dictionary of model scores
    :param file_name: Name to save model
    :return:
    """

    dir_, _, file = os.walk('data/models/sold')

    old_path = os.path.join(dir_[0], dir_[1][1], file[2][0]).replace('\\', '/')
    archive_path = os.path.join(dir_[0], dir_[1][0], file[2][0]).replace('\\', '/')
    new_path = os.path.join(dir_[0], dir_[1][1], file_name).replace('\\', '/')

    with open(old_path, 'rb') as f:
        old_model = joblib.load(f)

    old_scores = old_model.get_scores()

    if old_scores['test_mean_squared_error'] < dict_scores['test_mean_squared_error']:
        new_model = ModelObject(model, dict_scores)

        os.rename(old_path, archive_path)

        with open(new_path, 'wb') as f:
            joblib.dump(new_model, f, compress='lmza')
        logger.info('Evaluated Model outperformed previous iteration. Evaluated Model saved.')

    logger.info('Evaluated Model did not outperform previous iteration.')
