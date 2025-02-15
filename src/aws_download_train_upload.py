import download_s3
import finetune_model_training
import upload_s3


def main():
    # Download the dataset from S3
    download_s3.download_datasets

    # Fine-tune the model
    finetune_model_training.main()

    # Upload the fine-tuned model to S3
    upload_s3.upload_model_results

if __name__ == "__main__":
    main()
    