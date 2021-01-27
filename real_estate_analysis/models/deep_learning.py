from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline
import tensorflow.keras.callbacks as callbacks
from sklearn.model_selection import KFold
import matplotlib.pylab as plt

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
        'list_price'
    ]

    df_sf_features = df_sf[list_features]

    # Prepare and split data
    X_train, X_test, y_train, y_test = func.prepare_my_data(my_df=df_sf_features)

    ####################################################################################################################
    # Define Pipeline
    ####################################################################################################################

    # keras_model = KerasRegressor(build_fn=func.create_model(input_dim=X_train.shape[1]),
    #                              epochs=50,
    #                              batch_size=15,
    #                              verbose=1)

    model = func.create_model(input_dim=X_train.shape[1],
                              dense_nparams=X_train.shape[1])

    # # Define Pipeline
    # regression_pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('regressor', keras_model)
    # ])
    #
    # param_grid = [
    #     {  # Keras Neural Network
    #         'scaler': [RobustScaler(), StandardScaler(), PowerTransformer(), QuantileTransformer()],
    #         'regressor': [keras_model],
    #         'regressor__epochs': [10, 100],
    #         'regressor__init': ['uniform', 'zeros', 'normal'],
    #         'regressor__optimizer': ['RMSprop', 'Adam', 'Adamax', 'SGD']
    #     }
    # ]

    ####################################################################################################################
    # Training
    ####################################################################################################################

    logger.info('Starting Regressor Training')

    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=5,
                        epochs=100,
                        verbose=1,
                        # callbacks=[callbacks.EarlyStopping(patience=5)],
                        validation_data=(X_test, y_test))

    logger.info('Regressor Training Complete')

    ####################################################################################################################
    # Inspection
    ####################################################################################################################

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    accuracy = history_dict['coeff_determination']
    val_accuracy = history_dict['val_coeff_determination']

    epochs = range(1, len(loss_values) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(epochs, accuracy, 'bo', label='Training accuracy')
    ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    ax[0].set_title('Training & Validation Accuracy', fontsize=16)
    ax[0].set_xlabel('Epochs', fontsize=16)
    ax[0].set_ylabel('Accuracy', fontsize=16)
    ax[0].legend()

    ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
    ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
    ax[1].set_title('Training & Validation Loss', fontsize=16)
    ax[1].set_xlabel('Epochs', fontsize=16)
    ax[1].set_ylabel('Loss', fontsize=16)
    ax[1].legend()

    ####################################################################################################################
    # Validation
    ####################################################################################################################

    # list_scores = func.score_my_model(my_model=model, x_test=X_test, y_test=y_test)
    score = history.model.evaluate(X_test, y_test, verbose=1)

    logger.info('Results from Deep Learning Model:')
    logger.info(f'Test loss: {score[0]}')  # Test loss:  59806560256.0
    logger.info(f'Test Accuracy: {score[1]}')  # Test Accuracy:  0.3306


if __name__ == '__main__':
    main()
