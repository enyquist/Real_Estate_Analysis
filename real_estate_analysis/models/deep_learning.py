import tensorflow.keras.callbacks as callbacks

import real_estate_analysis.utils.functions as func


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/error_log.log', t_handler_name='../logs/training_log.log')

    ####################################################################################################################
    # Data
    ####################################################################################################################

    bucket = 're-raw-data'

    # Retrieve data from s3 and format into dataframe
    df = func.create_df_from_s3(bucket=bucket)

    # Parse DataFrame for single family real estate
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
        'location.address.state_code',
        'tags',
        'list_price'
    ]

    df_sf_features = df_sf[list_features]

    # Prepare and split data
    X_train, X_test, y_train, y_test = func.prepare_my_data(my_df=df_sf_features, deep_learning=True)

    ####################################################################################################################
    # Define Pipeline
    ####################################################################################################################

    func.prep_gpu()

    # keras_model = KerasRegressor(build_fn=func.create_model(), verbose=1)

    model = func.create_model(input_size=X_train.shape[1],
                              hidden_layers=10)

    # # Define Pipeline
    # regression_pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('regressor', keras_model)
    # ])
    #
    # param_grid = {  # Keras Neural Network
    #         'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
    #         'regressor': [keras_model],
    #         'regressor__batch_size': [10, 20, 40, 60, 80, 100],
    #         'regressor__epochs': [10, 50, 100]
    #     }

    ####################################################################################################################
    # Training
    ####################################################################################################################

    logger.info('Starting Regressor Training')

    model.fit(x=X_train,
              y=y_train,
              batch_size=2048,
              epochs=100000,
              verbose=1,
              callbacks=[callbacks.EarlyStopping(patience=200)],
              validation_data=(X_test, y_test))

    # model = func.train_my_model(my_pipeline=regression_pipe,
    #                             my_param_grid=param_grid,
    #                             x_train=X_train,
    #                             y_train=y_train,
    #                             style='grid',
    #                             n_jobs=1)

    logger.info('Regressor Training Complete')

    ####################################################################################################################
    # Validation
    ####################################################################################################################

    # list_scores = func.score_my_model(my_model=model, x_test=X_test, y_test=y_test)
    score = model.evaluate(X_test, y_test, verbose=1)

    logger.info('Results from Deep Learning Model:')
    logger.info(f'Test loss: {score[0]}')
    # Test loss:
    # 2021-02-03 12:08:06,993:MainProcess:root:INFO:Test loss: 39496515584.0
    # 2021-02-04 18:31:22,519:MainProcess:root:INFO:Test loss: 36017999872.0
    logger.info(f'Test Accuracy: {score[1]}')
    # Test Accuracy:
    # 2021-02-03 12:08:06,994:MainProcess:root:INFO:Test Accuracy: 0.5262999534606934
    # 2021-02-04 18:31:22,519:MainProcess:root:INFO:Test Accuracy: 0.6196267008781433

    # logger.info('Results from Deep Learning Search:')
    # logger.info(f'Search best estimator: {model.best_estimator_}')
    # logger.info(f'Search Best params: {model.best_params_}')
    # logger.info(f"Search Cross Validation Scores: {list_scores[0]}")
    # logger.info(f"Search Validation Score: %0.2f" % model.best_score_)
    # logger.info(f"Search accuracy on test data: %0.2f (+/- %0.2f)" % (list_scores[1], list_scores[2]))
    # logger.info(f"Search test score: %0.2f" % list_scores[3])


if __name__ == '__main__':
    main()
