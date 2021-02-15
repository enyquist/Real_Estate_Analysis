import boto3
import configparser

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
    logger = func.create_logger(e_handler_name='../logs/AWS_e_log.log', t_handler_name='../logs/AWS_log.log')

    logger.info('Streaming raw data to local environment')

    # Retrieve data from s3 and format into dataframe
    df = func.create_df_from_s3(bucket='re-raw-data')

    logger.info('Raw data successfully streamed')

    # Filter for Single Family homes
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

    # Create Features DataFrame
    df_sf_features = df_sf[list_features]

    logger.info('Creating train and test sets from raw data')

    df_train, df_test = func.prepare_my_data(df_sf_features)

    logger.info('Train and Test sets created')

    # Stream to s3
    for str_filename, df in zip(['train', 'test'], [df_train, df_test]):
        try:
            response = func.pandas_to_s3(df=df, client=s3, bucket='re-formatted-data', key=str_filename)

            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                logger.info(f'{str_filename} successfully uploaded')

            else:
                logger.info(f'{str_filename} failed to upload')
        except TypeError as e:
            logger.error(f'{str_filename} resulted in error: {e}')

    logger.info(f'Formatted data upload complete')


if __name__ == '__main__':
    main()
