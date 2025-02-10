import os
import cv2
from datetime import datetime
from ultralytics import YOLO
import torch

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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 model
model = YOLO(os.path.join(frame_capture_settings.model_path, frame_capture_settings.model_name)).to(device)

cap = cv2.VideoCapture(frame_capture_settings.webcam_url)
if not cap.isOpened():
    print("Error: Could not open the video stream.")
    exit()

frame_count = 0

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    # Process only every FRAME_INTERVAL frame
    if frame_count % frame_capture_settings.frame_interval != 1:
        continue

    # Save the raw frame
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    img_filename = os.path.join(frame_capture_settings.img_dir, f"{frame_capture_settings.webcam_name}_{timestamp}.jpg")
    cv2.imwrite(img_filename, frame)
    print(f"Saved image: {img_filename}")

    # Run YOLOv8 inference on the frame
    results = model(frame)
    height, width = frame.shape[:2]

    # Open a .txt file for writing the annotations in YOLO format
    txt_filename = os.path.join(f"{frame_capture_settings.default_label_dir}", f"{frame_capture_settings.webcam_name}_{timestamp}.txt")
    with open(txt_filename, "w") as f:
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()  # each row: [x1, y1, x2, y2, conf, cls]
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    if conf >= frame_capture_settings.conf_threshold:
                        # Convert bounding box to YOLO format (normalized)
                        x_center = ((x1 + x2) / 2.0) / width
                        y_center = ((y1 + y2) / 2.0) / height
                        bbox_width = (x2 - x1) / width
                        bbox_height = (y2 - y1) / height
                        # Write annotation line: class x_center y_center width height
                        f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{int(cls)} {conf:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
    print(f"Saved annotations: {txt_filename}")

    # Optionally, display the frame (with no annotations drawn)
    cv2.imshow("Dataset Collection", frame)


cap.release()
cv2.destroyAllWindows()