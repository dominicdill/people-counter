import cv2
import os
import time
import imageio
from ultralytics import YOLO
import torch
import numpy as np

def run_yolo_on_frame(frame, model, conf_threshold):
    """
    Runs YOLO inference on the given frame and filters detections based on the confidence threshold.
    Returns a list of filtered results.
    """
    results = model(frame)
    filtered_results = []
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()  # shape: (num_boxes, 6)
            filtered_boxes = [box for box in boxes if box[4] >= conf_threshold]
            if filtered_boxes:
                new_result = result  # shallow copy; modifying boxes directly
                new_result.boxes.data = torch.tensor(filtered_boxes)
                filtered_results.append(new_result)
    return filtered_results

def annotate_frame_multi(frame, models, model_names, colors, conf_threshold):
    """
    For the given frame, runs each model and draws all bounding boxes on one copy of the frame.
    No text is drawn for individual detections.
    Instead, in the top left corner the title for each model is shown (color coded)
    along with the total number of people detected (detections with class 0).
    Returns the annotated frame.
    """
    annotated_frame = frame.copy()
    title_texts = []  # To store title strings for each model
    
    # Process each model's detections
    for model, model_name, color in zip(models, model_names, colors):
        yolo_results = run_yolo_on_frame(frame, model, conf_threshold)
        people_count = 0
        for result in yolo_results:
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    # Count only if detected class is 0 (person)
                    if int(cls) == 0:
                        people_count += 1
        title_texts.append(f"{model_name}: {people_count}")
    
    # Draw the titles in the top left corner, each on a new line
    y0 = 30  # Starting y-coordinate for the first title
    dy = 30  # Vertical spacing between titles
    for i, (text, color) in enumerate(zip(title_texts, colors)):
        cv2.putText(
            annotated_frame,
            text,
            (10, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
    
    return annotated_frame

def save_frame(frame, folder):
    """
    Saves the frame to the specified folder using a timestamp-based filename.
    Returns the filename.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = time.time()
    filename = os.path.join(folder, f"{timestamp:.3f}.jpg")
    cv2.imwrite(filename, frame)
    return filename


def capture_frames(webcam_url, duration, save_folder):
    cap = cv2.VideoCapture(webcam_url)
    if not cap.isOpened():
        raise Exception("Error: Could not open the video stream.")

    start_time = time.time()
    filenames = []
    frame_count = 0

    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame")
            continue

        frame_count += 1
        print(f"Capturing frame {frame_count}")
        filename = save_frame(frame, save_folder)
        filenames.append(filename)
        time.sleep(0.25)  # Wait before capturing the next frame

    cap.release()
    return filenames


def process_saved_frames(filenames, models, model_names, colors, conf_threshold, output_folder):
    """
    Loads each saved frame, runs both YOLO models to annotate the image with their detections,
    and saves the annotated frame in output_folder.
    Returns a list of annotated frame filenames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    annotated_filenames = []
    for idx, fn in enumerate(filenames, start=1):
        print(f"Processing saved frame {idx}")
        frame = cv2.imread(fn)
        annotated_frame = annotate_frame_multi(frame, models, model_names, colors, conf_threshold)
        annotated_fn = save_frame(annotated_frame, output_folder)
        annotated_filenames.append(annotated_fn)
    return annotated_filenames

def create_gif(filenames, gif_name="demo_output.gif", fps=1.0):
    """
    Converts the list of saved frame filenames into a repeating GIF.
    The frames are sorted by timestamp extracted from the filename.
    frame_duration sets the display time for each frame (in seconds).
    """
    sorted_filenames = sorted(filenames, key=lambda x: float(os.path.splitext(os.path.basename(x))[0]))
    images = [imageio.imread(fn) for fn in sorted_filenames]
    imageio.mimsave(gif_name, images, fps=fps, loop=0, optimize=False)
    print(f"GIF saved as {gif_name}")

def main():
    """
    Main function: loads two YOLO models, first captures and saves raw frames,
    then processes each saved frame to annotate detections from both models on a single image.
    Finally, it creates a repeating GIF for review.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the models.
    model1 = YOLO('finetune/yolo11l_webcam_finetune/weights/best.pt').to(device)
    model2 = YOLO('src/models/yolo11l.pt').to(device)
    models = [model1, model2]
    model_names = ["Fine Tuned YOLOv11l", "Default YOLOv11l"]
    
    # Define unique colors for each model: (B, G, R)
    colors = [(0, 255, 0), (0, 0, 255)]  # green for model1, red for model2
    
    # Define parameters.
    webcam_url = "https://streamer4.brownrice.com/camdensnowbowl1/camdensnowbowl1.stream/main_playlist.m3u8"
    raw_frames_folder = "demo/raw_frames"
    annotated_frames_folder = "demo/annotated_frames"
    conf_threshold = 0.42
    capture_duration = 60  # seconds
    
    # First loop: capture and save raw frames.
    raw_filenames = capture_frames(webcam_url, capture_duration, raw_frames_folder)
    
    # Second loop: process each saved frame with both models.
    annotated_filenames = process_saved_frames(raw_filenames, models, model_names, colors, conf_threshold, annotated_frames_folder)
    

    if annotated_filenames:
        create_gif(annotated_filenames, gif_name="demo/demo_output.gif")
    else:
        print("No frames saved for GIF.")

if __name__ == "__main__":
    main()
