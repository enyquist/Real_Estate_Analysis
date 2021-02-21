import boto3
import configparser

import real_estate_analysis.utils.functions as func

config = configparser.ConfigParser()
config.read('../config.ini')


def main():
    # Generate the boto3 client for interacting with S3
    s3 = boto3.client('s3',
                      region_name='us-east-1',
                      aws_access_key_id=config['AWS']['aws_access_key_id'],
                      aws_secret_access_key=config['AWS']['aws_secret_access_key'])

    # Create AWS Logger
    logger = func.create_logger(e_handler_name='../logs/AWS_e_log.log',
                                t_handler_name='../logs/AWS_log.log')

    list_bucket_names = ['re-raw-data', 're-formatted-data', 're-sold-data']

    # Create buckets
    for bucket in list_bucket_names:
        s3.create_bucket(Bucket=bucket)
        logger.info(f'{bucket} created')


if __name__ == '__main__':
    main()
