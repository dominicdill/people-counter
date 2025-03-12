import os
import torch
from ultralytics import YOLO
from datetime import datetime


from config.config import finetune_settings

def external_train(timestamp):
    # Load the pre-trained YOLO model
    model = YOLO(finetune_settings.model_path)

    data_yaml = finetune_settings.external_finetune_yaml
    print("Data YAML path:", data_yaml)
    model_name = os.path.splitext(os.path.basename(finetune_settings.model_path))[0]
    
    results = model.train(
        data=data_yaml,
        epochs=finetune_settings.external_finetune_epochs,
        imgsz=finetune_settings.external_finetune_imgsz,
        batch=finetune_settings.external_finetune_batch_size,
        project=finetune_settings.project_name,
        name=f"Stage_1_{model_name}_external_finetune_{timestamp}",
    )
    
    print("Finetuning on external dataset complete!")
    current_run_dir = results.save_dir  # This directory is unique to the current run
    best_model_path = os.path.join(current_run_dir, "weights", "best.pt")
    print("Best model for the current run is saved at:", best_model_path)
    return best_model_path


def webcam_train(model_path, model_name):
    
    model = YOLO(model_path)

    data_yaml = finetune_settings.webcam_finetune_yaml
    print("Data YAML path:", data_yaml)
    print("Model name:", model_name)
    print("Model path:", model_path)
    
    results = model.train(
        data=data_yaml,
        epochs=finetune_settings.webcam_finetune_epochs,
        imgsz=finetune_settings.webcam_finetune_imgsz,
        batch=finetune_settings.webcam_finetune_batch_size,
        project=finetune_settings.project_name,
        name=model_name,
        degrees = 0.0,
        translate = 0.0,
        scale = 0.0,
        fliplr = 0.0,
        mosaic = 0.0,
        erasing = 0.0,
    )
    
    print("Finetuning on webcam dataset complete!")
    current_run_dir = results.save_dir  # This directory is unique to the current run
    best_model_path = os.path.join(current_run_dir, "weights", "best.pt")
    print("Best model for the current run is saved at:", best_model_path)
    return best_model_path

def main():
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    if finetune_settings.external_finetune:
        external_model = external_train(timestamp)
        model_name = f"Stage_2_{os.path.splitext(os.path.basename(finetune_settings.model_path))[0]}_webcam_finetune_{timestamp}"
        webcam_model = webcam_train(external_model, model_name)
    else:
        model_name = f"{os.path.splitext(os.path.basename(finetune_settings.model_path))[0]}_webcam_finetune_{timestamp}"
        webcam_model = webcam_train(finetune_settings.model_path, model_name)
    print("Training complete!")
    print("Best model for the current run is saved at:", webcam_model)

if __name__ == "__main__":
    main()

