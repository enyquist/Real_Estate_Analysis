import os
import boto3

AWS_ACCESS_KEY_ID = os.environ.get('realEstateUserAWS_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('realEstateUserAWS_Key')

# Generate the boto3 client for interacting with S3
s3 = boto3.client('s3',
                  region_name='us-east-1',
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY )
# List the buckets
buckets = s3.list_buckets()

# Print the buckets
print(buckets)

# Create en-real-estate-data bucket. If the bucket exists, this will return the bucket
bucket = s3.create_bucket(Bucket='en-real-estate-data')
