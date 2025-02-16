from download_s3 import download_datasets
from finetune_model_training import main as finetune_model_training
from upload_s3 import upload_model_results

# This script is used to download datasets from S3, fine-tune a model, and upload the results back to S3.
# It is intended to be run on an EC2 instance with the appropriate IAM role and permissions.

def main():
    # Download the dataset from S3 - must be in the ec2 instance with boto3
    download_datasets()

    # Fine-tune the model
    finetune_model_training()

    # Upload the fine-tuned model to S3
    upload_model_results()

if __name__ == "__main__":
    main()
    