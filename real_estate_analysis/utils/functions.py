import boto3
import numpy as np
import pandas as pd
import logging
from io import StringIO, BytesIO
import gzip
import itertools
import configparser

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import spacy

from skopt import BayesSearchCV
from skopt.callbacks import DeltaXStopper

from catboost import CatBoostRegressor

config = configparser.ConfigParser()
config.read('../config.ini')


def prep_gpu(gb=3):
    """
    Function to turn on tensorflow and configure gpu
    :return:
    """
    import tensorflow as tf

    # Limit GPU usage to 3GB to prevent using all of the available GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_virtual_device_configuration(gpus[0],
    #                                                         [tf.config.experimental.VirtualDeviceConfiguration(
    #                                                             memory_limit=gb*3)])
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


def prepare_my_data(my_df, deep_learning=False):
    """
    Wrapper to clean and prepare data for downstream tasks
    :param my_df: DataFrame of features
    :param deep_learning: Switch if to use deep learning, adds NLP processing to data
    :return: Split Dataset into a train/test split without outliers
    """

    # Imputer Instance
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Force to numeric, as pandas_to_s3 casts everything to strings, ignore the categorical data
    my_df = my_df.apply(lambda col: pd.to_numeric(col, errors='ignore'))

    # Drop entries that have no list_price or lat / long
    my_df = my_df.dropna(axis=0, subset=['list_price',
                                         'location.address.coordinate.lon',
                                         'location.address.coordinate.lat',
                                         'tags'])

    # split into features and targets elements
    X, y = my_df.drop(['list_price', 'tags'], axis=1).values, my_df['list_price'].values

    # Impute features
    X = imp.fit_transform(X)

    scaler = StandardScaler()  # Selected as it was most often the scaler used during GridSearchCV

    X = scaler.fit_transform(X)

    # Outlier Detection
    clf = IsolationForest(random_state=42).fit_predict(X)

    # Mask outliers
    mask = np.where(clf == -1)

    if deep_learning:
        # Load SpaCy model
        nlp = spacy.load('en_core_web_lg', disable=['tagger', 'parser'])

        # Collect SpaCy vectors
        list_vectors = []

        for idx, row in my_df.iterrows():
            row = ' '.join(row.tags.strip("'][' '").split("', '"))
            list_vectors.append(nlp(row).vector)

        vectors = np.stack(list_vectors, axis=0)

        # Concatenate vectors back to X as a part of feature engineering
        X = np.concatenate([X, vectors], axis=1)

    # Split data using outlier-free data
    X_train, X_test, y_train, y_test = train_test_split(np.delete(X, mask, axis=0),
                                                        np.delete(y, mask),
                                                        test_size=0.2,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


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
        cv = RandomizedSearchCV(estimator=my_pipeline,
                                param_distributions=my_param_grid,
                                n_iter=search,
                                cv=5,
                                n_jobs=n_jobs,
                                verbose=1)

        cv.fit(x_train, y_train)

        return cv

    if style == 'grid':
        # Create RandomizedSearchCV instance
        cv = GridSearchCV(estimator=my_pipeline,
                          param_grid=my_param_grid,
                          cv=5,
                          n_jobs=n_jobs,
                          verbose=1)

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
                           n_jobs=n_jobs,
                           cv=5)

        cv.fit(x_train, y_train, callback=DeltaXStopper(0.01))

        return cv

    else:  # todo notify user of error
        return -1


def score_my_model(my_model, x_test, y_test):
    """
    Wrapper to score a model
    :param x_test: Features test set
    :param y_test: Target test set
    :param my_model: an instance of an sklearn RandomizedSearchCV model. Must have a .best_estimator_ method implemented
    :return:
    """

    scores = cross_val_score(my_model.best_estimator_, x_test, y_test)

    # Score model
    model_score = my_model.score(x_test, y_test)

    return [scores, scores.mean(), scores.std() * 2, model_score]


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
                      aws_access_key_id=config['DEFAULT']['aws_access_key_id'],
                      aws_secret_access_key=config['DEFAULT']['aws_secret_access_key'])

    # Get response from s3 with data from bucket re-raw-data
    response = s3.list_objects(Bucket=bucket)

    # Lists
    list_data = []
    list_contents = response['Contents']

    for content in list_contents:
        key = content['Key']
        data = s3_to_pandas_with_processing(client=s3, bucket=bucket, key=key)
        data.columns = data.iloc[0]
        data = data.drop(0)
        data = data.reset_index(drop=True)
        list_data.append(data)

    # Concat data into master Dataframe
    df = pd.concat(list_data)

    # Drop any duplicates in the dataset
    df = df.drop_duplicates()

    return df


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
        if df_best_model['mean_test_score'] > float(config['DEFAULT']['mean_test_score_threshold']) or \
                df_best_model['std_test_score'] < float(config['DEFAULT']['std_test_score_threshold']):
            dict_best_models[model] = df_best_model

    # Dictionary that contains the best params for each model type, used as a Switch-Case statement
    dict_model_format = {
        'catboost': CatBoostRegressor(depth=int(dict_best_models['catboost'].param_regressor__depth),
                                      iterations=int(dict_best_models['catboost'].param_regressor__iterations),
                                      learning_rate=dict_best_models['catboost'].param_regressor__learning_rate,
                                      loss_function='RMSE'),

        # RandomForest is default in searchcv.py
        'RandomForest': RandomForestRegressor(),

        'KNeighbors': KNeighborsRegressor(n_neighbors=int(dict_best_models['KNeighbors'].param_regressor__n_neighbors),
                                          weights=dict_best_models['KNeighbors'].param_regressor__weights),

        # DecisionTree is default in searchcv.py
        'DecisionTree': DecisionTreeRegressor(),

        'ElasticNet': ElasticNet(alpha=dict_best_models['ElasticNet'].param_regressor__alpha,
                                 l1_ratio=dict_best_models['ElasticNet'].param_regressor__l1_ratio),

        # LinearRegression is default in searchcv.py
        'LinearRegression': LinearRegression(),

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
        list_voting_regressors.append(VotingRegressor(list_model_format))

    return list_voting_regressors


def create_model(input_size=12):
    """
    Simple wrapper to create a deep learning model using the Keras API
    :param input_size: int, the size of the input (number of features)
    :return:
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
    from tensorflow.keras.optimizers import Adam
    import tensorflow_addons as tfa

    # Create model
    model = Sequential()

    # Input Layer
    model.add(Dense(units=128, input_shape=(input_size, ), kernel_initializer='normal', activation='relu',
                    name='Input_layer'))

    # Hidden Layers
    model.add(Dense(256, kernel_initializer='normal', activation='relu', name='Hidden_1'))
    model.add(Dropout(0.3, name='Dropout_1'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu', name='Hidden_2'))
    model.add(Dropout(0.3, name='Dropout_2'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu', name='Hidden_3'))
    model.add(Dropout(0.3, name='Dropout_3'))

    # Output Layer
    model.add(Dense(units=1, kernel_initializer='normal', activation='linear', name='Output_layer'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination])

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
