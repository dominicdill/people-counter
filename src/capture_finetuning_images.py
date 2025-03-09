import os
import cv2
from datetime import datetime, timedelta
import time
from ultralytics import YOLO
import torch

from config.config import frame_capture_settings

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def main():
    ###########################################################################################
    # Save frames from a webcam stream for collecting finetuning images
    # If a pretrained model is provided that can identify the class listed in the config,
    # it will draw bounding boxes around the detected objects and save the annotations in YOLO format.
    # Otherwise, no inference is performed and only the raw frames are saved along with a blank annotation file
    ###########################################################################################


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(frame_capture_settings.model_path).to(device)
    class_id = get_key_by_value(model.names, frame_capture_settings.target_label) # see if the target label is in the model classes
    if class_id is None: #
        print(f"Target label '{frame_capture_settings.target_label}' not found in model classes. Assisted labeling unavailable.")
        model = None
    else:
        print(f"Target label '{frame_capture_settings.target_label}' found in model classes. Assisted labeling available.")
        print(f"Using device: {device}")


    # load the webcam stream
    cap = cv2.VideoCapture(frame_capture_settings.webcam_url)
    if not cap.isOpened():
        print("Error: Could not open the video stream.")
        exit()

    frame_count = 0

    # This script will run for the number of hours specified in the config
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=frame_capture_settings.duration_hours)
    print("Start time:", start_time)
    print("End time:", end_time)


    while datetime.now() < end_time:
        if cv2.waitKey(1) & 0xFF == ord("q"): # Press 'q' to quit
            break
        try:
            ret, frame = cap.read()
            counter = 0
            while not ret:
                time.sleep(1)
                ret, frame = cap.read()
                counter += 1
                if counter >= 10:
                    raise RuntimeError(f"Failed to capture frame after {counter} retries")
        except Exception as e:
            print(f"Error reading webcam: {e}")
            break

        frame_count += 1
        # Process only every FRAME_INTERVAL frame
        if frame_count % frame_capture_settings.frame_interval != 1:
            continue

        # Save the raw frame
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        img_filename = os.path.join(frame_capture_settings.img_dir, f"{frame_capture_settings.webcam_name}_{timestamp}.jpg")
        cv2.imwrite(img_filename, frame)
        print(f"Saved image: {img_filename}")

        if model:
            results = model(frame, classes=[class_id], conf=frame_capture_settings.conf_threshold)
        else:
            results = []
        height, width = frame.shape[:2]

        # Open a .txt file for writing the annotations in YOLO format
        txt_filename = os.path.join(f"{frame_capture_settings.default_label_dir}", f"{frame_capture_settings.webcam_name}_{timestamp}.txt")
        with open(txt_filename, "w") as f:
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()  # each row: [x1, y1, x2, y2, conf, cls]
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box
                        # Convert bounding box to YOLO format (normalized)
                        x_center = ((x1 + x2) / 2.0) / width
                        y_center = ((y1 + y2) / 2.0) / height
                        bbox_width = (x2 - x1) / width
                        bbox_height = (y2 - y1) / height
                        # Write annotation line: class x_center y_center width height
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
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

if __name__ == "__main__":
    main()