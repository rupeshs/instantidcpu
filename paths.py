import os
from pathlib import Path


def ensure_file(path: str) -> bool:
    return os.path.exists(path)


def join_paths(
    first_path: str,
    second_path: str,
) -> str:
    return os.path.join(first_path, second_path)


def get_file_name(file_path: str) -> str:
    return Path(file_path).stem


def get_app_path() -> str:
    app_dir = os.path.dirname(__file__)
    work_dir = os.path.dirname(app_dir)
    return work_dir
