from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath


env_path = (Path(__file__).parent / '.env').resolve()  # Using .resolve() for an absolute path


class FrameCaptureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding='utf-8', extra='ignore')
    
    webcam_url: str
    webcam_name: str
    model_path: DirectoryPath
    model_name: str
    target_label_id: int
    frame_interval: int
    conf_threshold: float
    img_dir: DirectoryPath
    default_label_dir: DirectoryPath
    edited_label_dir: DirectoryPath


frame_capture_settings = FrameCaptureSettings()