import os
import torch
from ultralytics import YOLO
# Set up safe globals for PyTorch 2.6+ (include only if necessary)
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential, ModuleList
from ultralytics.nn.modules.conv import Conv, Concat
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF, DFL
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from ultralytics.nn.modules.head import Detect
torch.serialization.add_safe_globals([
    DetectionModel, Sequential, Conv, Conv2d, BatchNorm2d, SiLU, C2f, ModuleList,
    Bottleneck, SPPF, MaxPool2d, Upsample, Concat, Detect, DFL
])

from config.config import frame_capture_settings

def external_train():
    # Path to your pre-trained model (assumed to be YOLOv8 nano)
    model_path = os.path.join(frame_capture_settings.model_path, frame_capture_settings.model_name)
    
    # Load the pre-trained YOLO model
    model = YOLO(model_path)

    data_yaml = "dataset_external/data_external.yaml"
    data_yaml = os.path.abspath(os.path.join("datasets", "dataset_external", "data_external.yaml"))
    
    # Stage 1: Fine tuning on external dataset
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        project="finetune",
        name="yolov8n_external_finetune",
        pretrained=False
    )
    
    # Optionally, save or print training results
    print("Finetuning on external dataset complete!")
    print(results)

external_train()

# def webcam_train():
#     # Path to your pre-trained model (assumed to be YOLOv8 nano)
#     model_path = os.path.join(frame_capture_settings.model_path, frame_capture_settings.model_name)
    
#     # Load the pre-trained YOLO model
#     model = YOLO(model_path)

#     data_yaml = "dataset_webcam/data_webcam.yaml"
    
#     # Stage 2: Fine tuning on webcam dataset
#     results = model.train(
#         data=data_yaml,
#         epochs=50,
#         imgsz=640,
#         batch=16,
#         project="finetune",
#         name="yolov8n_webcam_finetune"
#     )
    
#     # Optionally, save or print training results
#     print("Finetuning on webcam dataset complete!")
#     print(results)

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





