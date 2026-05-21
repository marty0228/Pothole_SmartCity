import os
import tempfile

def atomic_write(path, data, mode='w', encoding='utf-8'):
    """Safely write `data` to `path` by writing to a temp file and renaming.

    Usage:
        atomic_write('client_3d/result.json', json_string)
    """
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath)
    try:
        with os.fdopen(fd, mode, encoding=encoding) as f:
            f.write(data)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
