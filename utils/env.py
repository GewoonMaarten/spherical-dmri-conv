"""
Load variables from a .env file. If the file is not present it will use the 
system environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

dotenv_path = Path(Path(__file__).cwd().parent, '.env')
load_dotenv(dotenv_path=dotenv_path)

LOGGING_LEVEL = int(os.getenv('LOGGING_LEVEL'))
MUDI_DATA_PATH = os.getenv('MUDI_DATA_PATH')

assert MUDI_DATA_PATH, "No environment variable named MUDI_DATA_PATH"
