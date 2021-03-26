import boto3
import pandas as pd
from io import StringIO, BytesIO
import gzip
import configparser
import logging

config = configparser.ConfigParser()
config.read('../config.ini')

logger = logging.getLogger(__name__)


def pandas_to_s3(df, client, bucket, key):
    """
    Wrapper to stream DataFrames to an s3 bucket. Credit to Lukasz uhho
    https://gist.github.com/uhho/a1490ae2abd112b556dcd539750aa151
    :param df: DataFrame
    :param client: s3 client
    :param bucket: s3 bucket
    :param key: Key of object or path to object in non-s3 terms
    :return: API response metadata
    """
    # Write DF to string stream
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Reset stream position
    csv_buffer.seek(0)
    # Create binary stream
    gz_buffer = BytesIO()

    # Compress string stream using gzip
    with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
        gz_file.write(bytes(csv_buffer.getvalue(), 'utf-8'))

    # Write stream to S3
    response = client.put_object(Bucket=bucket, Key=key, Body=gz_buffer.getvalue())

    return response


def s3_to_pandas(client, bucket, key, header=None):
    """
    Wrapper to stream DataFrames from an s3 bucket. Credit to Lukasz uhho
    https://gist.github.com/uhho/a1490ae2abd112b556dcd539750aa151
    :param client: s3 client
    :param bucket: s3 bucket
    :param key: Key of object or path to object in non-s3 terms
    :param header:
    :return:
    """
    # Get key using boto3 client
    obj = client.get_object(Bucket=bucket, Key=key)
    gz = gzip.GzipFile(fileobj=obj['Body'])

    # load stream directly to DF
    # noinspection PyTypeChecker
    return pd.read_csv(gz, header=header, dtype=str)


def s3_to_pandas_with_processing(client, bucket, key, header=None):
    """
    Wrapper to stream DataFrames from an s3 bucket. Credit to Lukasz uhho
    https://gist.github.com/uhho/a1490ae2abd112b556dcd539750aa151
    :param client: s3 client
    :param bucket: s3 bucket
    :param key: Key of object or path to object in non-s3 terms
    :param header:
    :return:
    """
    # Get key using boto3 client
    obj = client.get_object(Bucket=bucket, Key=key)
    gz = gzip.GzipFile(fileobj=obj['Body'])

    # Replace some characters in incomming stream and load it to DF
    lines = "\n".join([line.replace('?', ' ') for line in gz.read().decode('utf-8').split('\n')])
    return pd.read_csv(StringIO(lines), header=header, dtype=str)


def create_df_from_s3(bucket='re-raw-data'):
    """
    Wrapper to collect and append DataFrames from an s3 bucket
    :param bucket: Name of the s3 bucket
    :return: Compiled DataFrame from the s3 bucket
    """
    s3 = boto3.client('s3',
                      region_name='us-east-1',
                      aws_access_key_id=config.get('AWS', 'aws_access_key_id'),
                      aws_secret_access_key=config.get('AWS', 'aws_secret_access_key'))

    # Paginate s3 bucket because objects exceeds 1,000
    paginator = s3.get_paginator('list_objects_v2')

    # Get response from s3 with data from bucket re-raw-data
    pages = paginator.paginate(Bucket=bucket)

    list_data = []
    for page in pages:
        list_contents = page['Contents']
        list_data.extend(list_contents)

    list_formatted_data = []
    for content in list_data:
        key = content['Key']
        data = s3_to_pandas_with_processing(client=s3, bucket=bucket, key=key, header=0)
        list_formatted_data.append(data)

    # Concat data into master Dataframe
    df = pd.concat(list_formatted_data)

    # Drop any duplicates in the dataset
    df = df.drop_duplicates()

    return df


def fetch_from_s3(bucket, key):
    """

    :param bucket:
    :param key:
    :return:
    """
    s3 = boto3.client('s3',
                      region_name='us-east-1',
                      aws_access_key_id=config.get('AWS', 'aws_access_key_id'),
                      aws_secret_access_key=config.get('AWS', 'aws_secret_access_key'))

    data = s3_to_pandas_with_processing(client=s3, bucket=bucket, key=key)
    data.columns = data.iloc[0]
    data = data.drop(0)
    data = data.reset_index(drop=True)
    data = data.apply(lambda col: pd.to_numeric(col, errors='ignore'))

    return data
