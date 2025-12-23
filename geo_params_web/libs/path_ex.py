from pathlib import Path

def find_folder_upwards(start_path, folder_name):
    path = Path(start_path).resolve()
    for parent in [path] + list(path.parents):
        candidate = parent / folder_name
        if candidate.is_dir():
            return candidate
    return None
