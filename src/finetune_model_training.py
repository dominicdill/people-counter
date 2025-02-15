import os
import torch
from ultralytics import YOLO


from config.config import frame_capture_settings

def external_train():
    # Path to your pre-trained model (assumed to be YOLOv8 nano)
    model_path = os.path.join(frame_capture_settings.model_path, frame_capture_settings.model_name)
    
    # Load the pre-trained YOLO model
    model = YOLO(model_path)

    data_yaml = os.path.abspath(os.path.join("datasets", "dataset_external", "data_external.yaml"))
    print("Data YAML path:", data_yaml)
    model_name = frame_capture_settings.model_name.split(".")[0]
    
    # Stage 1: Fine tuning on external dataset
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        project="finetune",
        name=f"{model_name}_external_finetune",
    )
    
    # Optionally, save or print training results
    print("Finetuning on external dataset complete!")
    #print(results)
    current_run_dir = results.save_dir  # This directory is unique to the current run
    best_model_path = os.path.join(current_run_dir, "weights", "best.pt")
    print("Best model for the current run is saved at:", best_model_path)
    return best_model_path

#external_train()

def webcam_train(model_path):
    
    # Load the pre-trained YOLO model
    model = YOLO(model_path)

    data_yaml = os.path.abspath(os.path.join("datasets", "dataset_webcam", "data_webcam.yaml"))
    print("Data YAML path:", data_yaml)
    model_name = frame_capture_settings.model_name.split(".")[0]
    
    # Stage 2: Fine tuning on webcam dataset
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        project="finetune",
        name=f"{model_name}_webcam_finetune"
    )
    
    # Optionally, save or print training results
    print("Finetuning on webcam dataset complete!")
    #print(results)

def main():
    external_model = external_train()
    webcam_model = webcam_train(external_model)
    print("Training complete!")
    print("Best model for the current run is saved at:", webcam_model)

if __name__ == "__main__":
    main()

# def main():
#     # Path to your pre-trained model (assumed to be YOLOv8 nano)
#     model_path = os.path.join(frame_capture_settings.model_path, frame_capture_settings.model_name)
    
#     # Load the pre-trained YOLO model
#     model = YOLO(model_path)

#     data_yaml = "dataset/data.yaml"
    
#     results = model.train(
#         data=data_yaml,          # dataset configuration file
#         epochs=50,               # number of training epochs
#         imgsz=640,               # input image size
#         batch=16,                # batch size (adjust based on your hardware)
#         project="finetune",      # project name for saving results
#         name="yolov8n_person_finetune"  # run name
#     )
    
#     # Optionally, save or print training results
#     print("Training complete!")
#     print(results)

# # if __name__ == "__main__":
# #     main()





