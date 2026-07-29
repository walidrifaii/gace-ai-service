from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    face_similarity_threshold: float = 0.55
    face_model_name: str = "buffalo_l"
    face_det_size: int = 640
    # -1 = CPU, 0+ = GPU device index. Auto-picks CUDA when face_force_gpu or ctx>=0.
    face_ctx_id: int = -1
    # When true, prefer CUDA if onnxruntime-gpu is installed.
    face_force_gpu: bool = True
    face_api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8001

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
