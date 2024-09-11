import os
from dotenv import load_dotenv

def load_env_variables():
    """Load environment variables from .env file."""
    load_dotenv()

def get_huggingface_token():
    """Retrieve the Hugging Face token from environment variables."""
    load_dotenv(dotenv_path="../.env")
    return os.getenv('HUGGINGFACE_ACCESS_KEY')
