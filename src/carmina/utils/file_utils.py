import os


def backup_if_exists(path: str) -> None:
    """
    If *path* already exists, rename it to a numbered backup before the
    caller writes a new file there.

    Backup naming:  ``{stem}_back1{ext}``, ``{stem}_back2{ext}``, …
    The first gap in the sequence is used, so existing backups are never
    overwritten.

    Example::

        backup_if_exists("data/outputs/debug/output_gemini.json")
        # renames the file to "output_gemini_back1.json" (or _back2, etc.)
    """
    if not os.path.exists(path):
        return

    stem, ext = os.path.splitext(path)
    n = 1
    while True:
        candidate = f"{stem}_back{n}{ext}"
        if not os.path.exists(candidate):
            os.rename(path, candidate)
            return
        n += 1
