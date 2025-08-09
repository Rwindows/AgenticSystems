from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Config(BaseSettings):
    OPENAI_API_KEY: str    
    QDRANT_URL: str
    OPENAI_EMBEDDING_MODEL: str
    Qdrant_Collections_Name: str
    EMBEDDING_MODEL_PROVIDER: str = "openai"  # Default to openai if not specified
    EMBEDDING_MODEL: str
    GENERATION_MODEL: str
    GENERATION_MODEL_PROVIDER: str
    API_URL: str  = "http://api:8000"
    



    
    model_config = {
        "env_file": ".env",
        "env_prefix": ""
    }

# Create config with environment variables
config = Config()
