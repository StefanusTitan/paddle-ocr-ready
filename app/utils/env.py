from pydantic import BaseSettings

class EnvSettings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_TYPE: str
    SECRET_KEY: str 
    ALGORITHM: str
    FIREBASE_STORAGE_BUCKET_URL: str
    
    class Config:
        env_file = '../../.env'
        

env = EnvSettings()