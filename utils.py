import os


def check_file(path: str) -> bool:
    return os.path.exists(path)
