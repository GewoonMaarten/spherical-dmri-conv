"""
Add the parent directory to the system path. This way Python can find our 
packages in the parent directory.
"""

import os
import sys

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
