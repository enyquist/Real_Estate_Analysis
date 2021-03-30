import pandas as pd
import boto3
import datetime
import configparser

from real_estate_analysis.real_estate_data.RealEstateData import RealEstateData
from real_estate_analysis.models import functions as func
import real_estate_analysis.AWS.utils as AWS_utils

config = configparser.ConfigParser()
config.read('../config.ini')

LOG_FILEPATH = '../../data/AWS/city_log.csv'


def main(api, bool_override=False):
    # Boolean to manage API calls
    BOOL_API_CALLS_REMAINING = True if int(config.get(api, 'rapidapi_api_call_limit')) > 50 else False

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

    # Select correct column for API
    date_column = config.get(api, 'rapidapi_date_modified')

    # Format dates as datetime
    df_city_log[date_column] = pd.to_datetime(df_city_log[date_column])

    # Capture Time
    today = datetime.datetime.now()

    for idx, row in df_city_log.iterrows():  # todo set to 30, pending funding
        if (row[date_column] < today - datetime.timedelta(365)) and (BOOL_API_CALLS_REMAINING ^ bool_override):

            # Collect Data and log that it was captured
            city = row['city']
            state = row['state']
            obj = RealEstateData(city=city, state=state, api=api)
            df_data = obj.get_results()
            str_filename = f'{state}-{city}.tgz'
            if df_data is not None:

                # Stream to s3
                try:
                    response = AWS_utils.pandas_to_s3(df=df_data, client=s3, bucket=config.get(api, 'rapidapi_bucket'),
                                                      key=str_filename)

                    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                        logger.info(f'{str_filename} successfully uploaded, collected {api} data')
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

            if obj.requests_remaining < 50:
                BOOL_API_CALLS_REMAINING = False
                logger.warning(f"{obj.requests_remaining} calls remain for {api}'s API, cancelling data collection")

    logger.info(f'{api} Data collection complete')


if __name__ == '__main__':

    while True:
        value_2 = input('Override API Calls? Enter 1 override API call limits, 0 to limit API calls:')
        try:
            value_2 = int(value_2)
        except ValueError:
            print('Please use numeric digits!')
            continue
        if value_2 not in [0, 1]:
            print('Please enter 1 or 0!')
            continue
        break

    API = 'RAPIDAPI_SOLD'
    BOOL_OVERRIDE = True if value_2 == 1 else False
    print(f'Using {API}, Override API Calls: {BOOL_OVERRIDE}')
    main(API, bool_override=BOOL_OVERRIDE)
