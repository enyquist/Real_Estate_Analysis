import numpy as np
import pandas as pd
import logging
from io import StringIO
import configparser
import joblib
import cProfile
import pstats
from sklearn import model_selection
from sklearn import ensemble
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import xgboost as xgb
from catboost import CatBoostRegressor

import real_estate_analysis.AWS.utils as AWS_utils


config = configparser.ConfigParser()
config.read('../config.ini')

logger = logging.getLogger(__name__)


def create_logger(e_handler_name, t_handler_name):
    """
    Wrapper to create logger for errors and training records
    :param e_handler_name: filepath to logger as string
    :param t_handler_name: filepath to logger as string
    :return: logger object
    """

    log = logging.getLogger()
    log.setLevel(logging.INFO)

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
    log.addHandler(c_handler)
    log.addHandler(e_handler)
    log.addHandler(t_handler)

    return log


def prepare_my_data(my_df):
    """
    ETL Pipeline
    :param my_df: Feature dataset
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


def profile(fnc):
    """
    A decorator that uses cProfile to profile a function
    :param fnc:
    :return:
    """
    def inner(*args, **kwargs):

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def log_scores(dict_scores):
    """
    Log model scores
    :param dict_scores: Dictionary of scores
    :return:
    """
    logger.info(f"Training Cross Validation Scores: {dict_scores['train_cross_val_score']}")
    logger.info(f"Accuracy on training data: {dict_scores['train_mean_cross_val_score']:0.2f}"
                f" (+/- {dict_scores['train_std_cross_val_score']:0.2f})")
    logger.info(f"Test Cross Validation Scores: {dict_scores['test_cross_val_score']}")
    logger.info(f"Accuracy on test data: {dict_scores['test_mean_cross_val_score']:0.2f}"
                f" (+/- {dict_scores['test_std_cross_val_score']:0.2f})")
    logger.info(f"Test Explained Variance Score: {dict_scores['test_explained_variance_score']:0.2f}")
    logger.info(f"Test Max Error: {dict_scores['test_max_error']:0.2f}")
    logger.info(f"Test Mean Absolute Error: {dict_scores['test_mean_absolute_error']:0.2f}")
    logger.info(f"Test Mean Squared Error: {dict_scores['test_mean_squared_error']:0.2f}")
    logger.info(f"Test Median Absolute Error: {dict_scores['test_median_absolute_error']:0.2f}")
    logger.info(f"Test R2 score: {dict_scores['test_r2']:0.2f}")


def retrieve_and_prepare_data():
    """
    Collect train and test sets from AWS and split for training and testing
    :return:
    """
    df_train = AWS_utils.fetch_from_s3(bucket='re-formatted-data', key='train.tgz')
    df_test = AWS_utils.fetch_from_s3(bucket='re-formatted-data', key='test.tgz')

    # Split the data
    X_train, y_train = df_train.drop(['list_price'], axis=1).values, df_train['list_price'].values
    X_test, y_test = df_test.drop(['list_price'], axis=1).values, df_test['list_price'].values

    return X_train, y_train, X_test, y_test
