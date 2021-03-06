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

    API = 'RAPIDAPI_SALE'

    # Generate the boto3 client for interacting with S3
    s3 = boto3.client('s3',
                      region_name='us-east-1',
                      aws_access_key_id=config.get('AWS', 'aws_access_key_id'),
                      aws_secret_access_key=config.get('AWS', 'aws_secret_access_key'))

    # Create AWS Logger
    logger = func.create_logger(e_handler_name='../logs/AWS_e_log.log',
                                t_handler_name='../logs/AWS_log.log')

    # Load CSV
    df_city_log = pd.read_csv(LOG_FILEPATH, encoding='latin-1')

    # Reset Index, in event a previous collection run resulted in deletions
    df_city_log.reset_index(drop=True, inplace=True)

    # Select correct column
    date_column = config.get(API, 'rapidapi_date_modified')

    # Format dates as datetime
    df_city_log[date_column] = pd.to_datetime(df_city_log[date_column])

    # Capture Time
    today = datetime.datetime.now()

    for idx, row in df_city_log.iterrows():
        if row[date_column] < today - datetime.timedelta(365):  # todo set to 30, pending funding

            # Collect Data and log that it was captured
            city = row['city']
            state = row['state']
            df_data = RealEstateData(city=city, state=state, api=API).results
            str_filename = f'{state}-{city}'
            if df_data is not None:

                # Stream to s3
                try:
                    response = func.pandas_to_s3(df=df_data, client=s3, bucket=config.get(API, 'rapidapi_bucket'),
                                                 key=str_filename)

                    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                        logger.info(f'{str_filename} successfully uploaded, collected {API} data')
                        df_city_log.loc[idx, date_column] = today.strftime('%Y-%m-%d %H:%M:%S')

                        # Save df_city_log
                        df_city_log.to_csv(LOG_FILEPATH, index=False)

                    else:
                        logger.info(f'{str_filename} failed to upload')
                except TypeError as e:
                    logger.error(f'{str_filename} resulted in error: {e}')
            else:
                logger.info(f"{str_filename} is not a valid city/state combination in Realtor.com's database")
                df_city_log.drop(idx, inplace=True)
                df_city_log.to_csv(LOG_FILEPATH, index=False)
                logger.info(f"{str_filename} has been deleted from city_log.csv")

    logger.info(f'Data collection complete')


if __name__ == '__main__':
    main()
