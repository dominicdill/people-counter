import os
import zipfile

import boto3


def upload_model_results():
    bucket_name = 'dominic-dill-people-counter'
    sns_topic_arn = 'arn:aws:sns:us-east-1:767397842563:dominicdill-people-counter-train-sns'

    s3_client = boto3.client('s3')


    with zipfile.ZipFile('./runs.zip', 'w') as zip:
        for path, directories, files in os.walk('./finetune'):
            for file in files:
                file_name = os.path.join(path, file)
                zip.write(file_name)

    s3_client.upload_file('./runs.zip', bucket_name, 'out/runs.zip')

    # send sns

    sns_client = boto3.client('sns', region_name='us-east-1')

    response = sns_client.publish(
        TargetArn=sns_topic_arn,
        Message="Training completed !"
    )

    # shutdown instance
    os.system('sudo shutdown -h now')

if __name__ == "__main__":
    upload_model_results()