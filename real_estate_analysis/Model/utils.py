import os
import joblib
import logging

from real_estate_analysis.Model.Model import ModelObject

logger = logging.getLogger(__name__)


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