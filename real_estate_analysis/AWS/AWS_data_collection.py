import pandas as pd
import boto3
import datetime
import configparser

from real_estate_analysis.utils.RealEstateData import RealEstateData
from real_estate_analysis.utils import functions as func

config = configparser.ConfigParser()
config.read('../config.ini')

LOG_FILEPATH = '../../data/AWS/city_log.csv'


def main():
    # Generate the boto3 client for interacting with S3
    s3 = boto3.client('s3',
                      region_name='us-east-1',
                      aws_access_key_id=config['DEFAULT']['aws_access_key_id'],
                      aws_secret_access_key=config['DEFAULT']['aws_secret_access_key'])

    # Create AWS Logger
    logger = func.create_logger(e_handler_name='../logs/AWS_e_log.log',
                                t_handler_name='../logs/AWS_log.log')

    # Load CSV
    df_city_log = pd.read_csv(LOG_FILEPATH)

    # Format dates as datetime
    df_city_log['last_modified'] = pd.to_datetime(df_city_log['last_modified'])

    # Capture Time
    today = datetime.datetime.now()

    for idx, row in df_city_log.iterrows():
        if row['last_modified'] < today - datetime.timedelta(365):  # todo set to 30, pending funding

            # Collect Data and log that it was captured
            city = row['city']
            state = row['state']
            df_data = RealEstateData(city, state).results
            str_filename = f'{state}-{city}'

            # Stream to s3
            try:
                response = func.pandas_to_s3(df=df_data, client=s3, bucket='re-raw-data', key=str_filename)

                if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                    logger.info(f'{str_filename} successfully uploaded')
                    df_city_log.loc[idx, 'last_modified'] = today.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    logger.info(f'{str_filename} failed to upload')
            except TypeError as e:
                logger.error(f'{str_filename} resulted in error: {e}')

    # Save df_city_log
    df_city_log.to_csv(LOG_FILEPATH, index=False)

    logger.info(f'Data collection complete')


if __name__ == '__main__':
    main()
