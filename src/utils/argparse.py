from pathlib import Path
from argparse import ArgumentTypeError


def dir_path(path):
    path = Path(path)
    if path.is_dir():
        return path
    else:
        raise ArgumentTypeError(f"argument value {path} is not a directory")


def file_path(path):
    path = Path(path)
    if path.is_file():
        return path
    else:
        raise ArgumentTypeError(f"argument value {path} is not a file")
