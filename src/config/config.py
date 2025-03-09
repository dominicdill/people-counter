from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath


env_path = (Path(__file__).parent / '.env').resolve()  # Using .resolve() for an absolute path


class FrameCaptureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding='utf-8', extra='ignore')
    
    webcam_url: str
    webcam_name: str                    # used for labeling frames used in fine tuning
    model_path: Path                      # path to model used for assisted labeling
    target_label: str                   # assumes only 1 target class
    frame_interval: int                 # process every nth frame
    duration_hours: int                 # duration to capture frames in hours
    conf_threshold: float               # confidence threshold for detections
    img_dir: DirectoryPath              # directory to save captured frames for fine tuning
    default_label_dir: DirectoryPath    # directory to save labels in YOLO format. Will be used for AI assisted labeling. Will move to edited_label_dir after manual editing
    edited_label_dir: DirectoryPath     # directory to save edited labels for fine tuning

class TrainValTestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding='utf-8', extra='ignore')

    train_images_dir: DirectoryPath
    val_images_dir: DirectoryPath
    test_images_dir: DirectoryPath
    train_labels_dir: DirectoryPath
    val_labels_dir: DirectoryPath
    test_labels_dir: DirectoryPath
    sample_ratio_val: float
    sample_ratio_test: float


class FineTuneSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding='utf-8', extra='ignore')

    project_name: str
    model_path: Path                     # path to model to finetune
    external_finetune: bool             # finetune on external dataset
    external_finetune_yaml: Path
    external_finetune_epochs: int
    external_finetune_batch_size: int
    webcam_finetune: bool               # finetune on webcam dataset
    webcam_finetune_yaml: Path
    webcam_finetune_epochs: int
    webcam_finetune_batch_size: int



frame_capture_settings = FrameCaptureSettings()
train_val_test_settings = TrainValTestSettings()
finetune_settings = FineTuneSettings()