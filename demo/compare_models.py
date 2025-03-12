import os
import cv2
import imageio
import argparse
from ultralytics import YOLO

def parse_timestamp(timestamp_str):
    """
    Converts a timestamp string (mm:ss or hh:mm:ss) into seconds.
    """
    parts = timestamp_str.split(':')
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        raise ValueError("Timestamp format should be mm:ss or hh:mm:ss")

class VideoInferenceComparator:
    def __init__(self, video_path, start_time, end_time, model_paths, output_folder, frame_step=2):
        """
        Initializes the comparator with the video, timestamp range, models, and output folder.
        
        Parameters:
            video_path (str): Path to the video file.
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
            model_paths (list of str): List of file paths to the YOLO model weights.
            output_folder (str): Folder where output frames and GIF will be saved.
            frame_step (int): Process every nth frame to save compute (default is 2).
        """
        self.video_path = video_path
        self.start_time = start_time
        self.end_time = end_time
        self.model_paths = model_paths
        self.frame_step = frame_step
        self.output_folder = output_folder
        self.frames_folder = os.path.join(self.output_folder, "frames")
        os.makedirs(self.frames_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Load models
        self.models = []
        for model_path in self.model_paths:
            print(f"Loading model: {model_path}")
            model = YOLO(model_path)
            self.models.append(model)
    
    def run_models_on_frame(self, frame):
        """
        Runs inference on a single frame for each model.
        Returns a list of annotated frames.
        """
        annotated_frames = []
        for idx, model in enumerate(self.models):
            # Run inference (YOLO accepts a frame as a numpy array)
            results = model(frame)
            # Annotate the frame with detections (the result plot is a numpy array)
            annotated = results[0].plot(line_width = 3, font_size = 0.5)
            # Count the number of bounding boxes (detections)
            num_boxes = len(results[0].boxes) if results[0].boxes is not None else 0
            # Create a label that includes the model name and the detection count
            label = f"{os.path.basename(self.model_paths[idx])}: {num_boxes} detections"
            #label = os.path.basename(self.model_paths[idx])
            cv2.putText(annotated, label, (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                        5, (0, 255, 0), 4, cv2.LINE_AA)
            annotated_frames.append(annotated)
        return annotated_frames

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video FPS: {fps}, Total Frames: {total_frames}")

        start_frame = int(self.start_time * fps)
        end_frame = int(self.end_time * fps)
        print(f"Processing frames from {start_frame} to {end_frame}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        frame_counter = 0

        # Process frames and save to disk without storing in memory
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_counter % self.frame_step) == 0:
                annotated_list = self.run_models_on_frame(frame)
                
                # For a grid layout, if needed (for 4 models as an example):
                if len(annotated_list) == 4:
                    row1 = cv2.hconcat([annotated_list[0], annotated_list[1]])
                    row2 = cv2.hconcat([annotated_list[2], annotated_list[3]])
                    combined = cv2.vconcat([row1, row2])
                else:
                    combined = cv2.hconcat(annotated_list)
                
                scale_factor = 0.25  # adjust this value as needed
                new_width = int(combined.shape[1] * scale_factor)
                new_height = int(combined.shape[0] * scale_factor)
                combined = cv2.resize(combined, (new_width, new_height))


                frame_filename = os.path.join(self.frames_folder, f"frame_{current_frame:06d}.jpg")
                cv2.imwrite(frame_filename, combined)
                print(f"Saved frame {current_frame} to {frame_filename}")

            current_frame += 1
            frame_counter += 1

        cap.release()

        # Now read the saved images from disk to create the GIF.
        frame_files = sorted([os.path.join(self.frames_folder, f)
                            for f in os.listdir(self.frames_folder)
                            if f.endswith(".jpg")])
        processed_frames = []
        for file in frame_files:
            # imageio reads the image in RGB format.
            frame = imageio.imread(file)
            processed_frames.append(frame)
        
        # Calculate frame duration for the GIF.
        fps = len(processed_frames) / (self.end_time - self.start_time)
        gif_path = os.path.join(self.output_folder, "comparison.gif")
        imageio.mimsave(gif_path, processed_frames, fps=fps, loop=0)#, optimize=False)
        print(f"GIF saved to {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple YOLO models on a video segment and create a GIF.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--start_time", type=str, required=True,
                        help="Start time (mm:ss or hh:mm:ss) for the segment.")
    parser.add_argument("--end_time", type=str, default=None,
                        help="(Optional) End time (mm:ss or hh:mm:ss) for the segment. Defaults to 10 seconds after start time if not provided.")
    parser.add_argument("--model_paths", type=str, required=True,
                        help="Comma-separated list of file paths for the YOLO models.")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to save output frames and GIF.")
    parser.add_argument("--frame_step", type=int, default=4, help="Process every nth frame (default: 2).")
    args = parser.parse_args()

    start_sec = parse_timestamp(args.start_time)
    if args.end_time is not None:
        end_sec = parse_timestamp(args.end_time)
    else:
        end_sec = start_sec + 10  # Default to 10 seconds after the start time

    models_list = [mp.strip() for mp in args.model_paths.split(",")]

    comparator = VideoInferenceComparator(
        video_path=args.video_path,
        start_time=start_sec,
        end_time=end_sec,
        model_paths=models_list,
        output_folder=args.output_folder,
        frame_step=args.frame_step
    )
    comparator.process_video()

if __name__ == "__main__":
    main()
