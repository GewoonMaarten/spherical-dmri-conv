"""
Load variables from a .env file. If the file is not present it will use the 
system environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

dotenv_path = Path(Path(__file__).parent.parent, '.env')
load_dotenv(dotenv_path=dotenv_path)

LOGGING_LEVEL = int(os.getenv('LOGGING_LEVEL', 30))
DATA_PATH = os.getenv('DATA_PATH')

assert DATA_PATH, "No environment variable named DATA_PATH"
