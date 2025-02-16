import zipfile
import boto3

'''
This script downloads datasets from S3 bucket and extracts them.
It is intended to be run on an EC2 instance with the appropriate IAM role and permissions.
It expects a file named 'datasets.zip' in the S3 bucket under the 'in' directory.
'''


def download_datasets():
    """
    Download datasets from S3 bucket and extract them.
    """
    # Define the S3 bucket name and SNS topic ARN
    bucket_name = 'dominic-dill-people-counter'
    sns_topic_arn = None


    # download data
    s3_client = boto3.client('s3')

    s3_client.download_file(bucket_name, 'in/datasets.zip', 'datasets.zip')

    with zipfile.ZipFile('datasets.zip', 'r') as zip_ref:
        zip_ref.extractall('./')

if __name__ == "__main__":
    download_datasets()