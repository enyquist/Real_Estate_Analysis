import os
import pandas as pd
import boto3
import datetime

from utils.RealEstateData import RealEstateData
from utils.functions import pandas_to_s3, create_logger

AWS_ACCESS_KEY_ID = os.environ.get('realEstateUserAWS_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('realEstateUserAWS_Key')


def main():
    # Generate the boto3 client for interacting with S3
    s3 = boto3.client('s3',
                      region_name='us-east-1',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    # Create AWS Logger
    logger = create_logger(e_handler_name='logs/AWS_e_log.log', t_handler_name='logs/AWS_log.log')

    # Load CSV
    df_city_log = pd.read_csv('AWS/resources/city_log.csv')

    # Format dates as datetime
    df_city_log['last_modified'] = pd.to_datetime(df_city_log['last_modified'])

    # Capture Time
    today = datetime.datetime.now()

    for idx, row in df_city_log.iterrows():
        if row['last_modified'] < today - datetime.timedelta(30):

            # Collect Data and log that it was captured
            city = row['city']
            state = row['state']
            df_data = RealEstateData(city, state).results
            str_filename = f'{today.strftime("%Y-%m-%d")}-{city}-{state}'

            # Stream to s3
            try:
                response = pandas_to_s3(df=df_data, client=s3, bucket='re-raw-data', key=str_filename)

                if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                    logger.info(f'{str_filename} successfully uploaded')
                    df_city_log.loc[idx, 'last_modified'] = today.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    logger.info(f'{str_filename} failed to upload')
            except TypeError as e:
                logger.error(f'{str_filename} resulted in error: {e}')

    # Save df_city_log
    df_city_log.to_csv('AWS/resources/city_log.csv')


if __name__ == '__main__':
    main()
