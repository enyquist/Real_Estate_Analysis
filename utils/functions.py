import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import logging
from io import StringIO, BytesIO
import gzip


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
    Wrapper to clean and prepare data for downstream tasks
    :param my_df: DataFrame of features
    :return: Split Dataset into X, y (including outliers), and a train/test split without outliers
    """

    # Imputer Instance
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    # split into input and output elements
    X, y = my_df.drop('list_price', axis=1).values, my_df['list_price'].values

    # Impute data
    X = imp.fit_transform(X)

    # Outlier Detection
    clf = IsolationForest(random_state=42).fit_predict(X)

    # Mask outliers
    mask = np.where(clf == -1)

    # Split data using outlier-free data
    X_train, X_test, y_train, y_test = train_test_split(np.delete(X, mask, axis=0),
                                                        np.delete(y, mask),
                                                        test_size=0.2,
                                                        random_state=42)

    return X, y, X_train, X_test, y_train, y_test


def train_my_model(my_df, my_pipeline, my_param_grid, search=50):
    """
    Wrapper for training a regression model to predict housing prices
    :param my_df: DataFrame of features
    :param my_pipeline: Training Pipeline, an instance of sklearn's Pipeline
    :param my_param_grid: Dictionary containing the parameters to train over
    :param search: Number of iterations to search in RandomizedSearchCV, default 50
    :return: RandomizedSearchCV object
    """

    _, _, X_train, _, y_train, _ = prepare_my_data(my_df)

    # Create RandomizedSearchCV instance
    rscv = RandomizedSearchCV(estimator=my_pipeline,
                              param_distributions=my_param_grid,
                              n_iter=search,
                              cv=5,
                              n_jobs=-1)

    rscv.fit(X_train, y_train)

    return rscv


def score_my_model(my_df, my_model):
    """
    Wrapper to score a model
    :param my_df: DataFrame of features
    :param my_model: an instance of an sklearn RandomizedSearchCV model. Must have a .best_estimator_ method implemented
    :return:
    """

    X, y, _, X_test, _, y_test = prepare_my_data(my_df)

    scores = cross_val_score(my_model.best_estimator_, X, y)

    # Score model
    model_score = my_model.score(X_test, y_test)

    # Predict and Test on test data
    y_pred = my_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    return [scores, scores.mean(), scores.std() * 2, model_score, r2]


def pandas_to_s3(df, client, bucket, key):
    """
    Wrapper to stream DataFrames to an s3 bucket
    :param df: DataFrame
    :param client: S3 client
    :param bucket: S3 bucket
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
