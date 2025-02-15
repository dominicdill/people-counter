import os
import zipfile

import boto3


bucket_name = 'dominic-dill-people-counter'
sns_topic_arn = None


# download data
s3_client = boto3.client('s3')

s3_client.download_file(bucket_name, 'in/datasets.zip', 'datasets.zip')

with zipfile.ZipFile('datasets.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

