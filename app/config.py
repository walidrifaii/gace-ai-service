from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    face_similarity_threshold: float = 0.70
    face_model_name: str = "buffalo_l"
    face_det_size: int = 640
    face_api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8001

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
