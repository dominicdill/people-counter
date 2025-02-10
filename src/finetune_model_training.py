import os
from ultralytics import YOLO
from config.config import frame_capture_settings

def main():
    # Path to your pre-trained model (assumed to be YOLOv8 nano)
    model_path = os.path.join(frame_capture_settings.model_path, frame_capture_settings.model_name)
    
    # Load the pre-trained YOLO model
    model = YOLO(model_path)

    data_yaml = "dataset/data.yaml"
    
    results = model.train(
        data=data_yaml,          # dataset configuration file
        epochs=50,               # number of training epochs
        imgsz=640,               # input image size
        batch=16,                # batch size (adjust based on your hardware)
        project="finetune",      # project name for saving results
        name="yolov8n_person_finetune"  # run name
    )
    
    # Optionally, save or print training results
    print("Training complete!")
    print(results)

if __name__ == "__main__":
    main()
