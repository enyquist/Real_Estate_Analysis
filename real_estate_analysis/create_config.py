import os
import configparser

config = configparser.ConfigParser()

config['DEFAULT'] = {
    'AWS_ACCESS_KEY_ID': os.environ.get('realEstateUserAWS_ID'),
    'AWS_SECRET_ACCESS_KEY': os.environ.get('realEstateUserAWS_Key'),
    'MEAN_TEST_SCORE_THRESHOLD': 0.75,
    'STD_TEST_SCORE_THRESHOLD': 0.05
}

with open('../real_estate_analysis/config.ini', 'w') as configfile:
    config.write(configfile)
