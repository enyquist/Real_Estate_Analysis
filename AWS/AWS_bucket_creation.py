import os
import boto3

AWS_ACCESS_KEY_ID = os.environ.get('realEstateUserAWS_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('realEstateUserAWS_Key')


def main():
    # Generate the boto3 client for interacting with S3
    s3 = boto3.client('s3',
                      region_name='us-east-1',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    list_bucket_names = ['re-raw-data', 're-formatted-data']

    # Create buckets
    for bucket in list_bucket_names:
        s3.create_bucket(Bucket=bucket)


if __name__ == '__main__':
    main()
