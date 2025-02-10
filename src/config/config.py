from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath


class FrameCaptureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='src/config/.env', env_file_encoding='utf-8', extra='ignore')
    
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