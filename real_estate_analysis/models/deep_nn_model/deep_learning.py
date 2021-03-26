import tensorflow.keras.callbacks as callbacks

import real_estate_analysis.models.functions as func
import real_estate_analysis.models.deep_nn_model.utils as dnn_utils


def main():
    ####################################################################################################################
    # Config Log File
    ####################################################################################################################

    logger = func.create_logger(e_handler_name='../logs/error_log.log', t_handler_name='../logs/training_log.log')

    ####################################################################################################################
    # Data
    ####################################################################################################################

    X_train, y_train, X_test, y_test = func.retrieve_and_prepare_data()

    ####################################################################################################################
    # Model Creation
    ####################################################################################################################

    dnn_utils.prep_gpu()

    model = dnn_utils.create_model(input_size=X_train.shape[1],
                                   hidden_layers=10)

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

    logger.info('Regressor Training Complete')

    ####################################################################################################################
    # Validation
    ####################################################################################################################

    score = model.evaluate(X_test, y_test, verbose=1)

    logger.info('Results from Deep Learning Model:')
    logger.info(f'Test loss: {score[0]}')
    # Test loss:
    # 2021-02-03 12:08:06,993:MainProcess:root:INFO:Test loss: 39496515584.0
    # 2021-02-04 18:31:22,519:MainProcess:root:INFO:Test loss: 36017999872.0
    # 2021-02-26 11:24:25,958:MainProcess:root:INFO:Test loss: 1496132091904.0
    logger.info(f'Test Accuracy: {score[1]}')
    # Test Accuracy:
    # 2021-02-03 12:08:06,994:MainProcess:root:INFO:Test Accuracy: 0.5262999534606934
    # 2021-02-04 18:31:22,519:MainProcess:root:INFO:Test Accuracy: 0.6196267008781433
    # 2021-02-26 11:24:25,958:MainProcess:root:INFO:Test Accuracy: -0.017010094597935677


if __name__ == '__main__':
    main()
