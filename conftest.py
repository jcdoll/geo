from pathlib import Path
from typing import Any


def pytest_ignore_collect(collection_path: Path, config: Any) -> bool:  # noqa: D401
    """Tell pytest to ignore directories that cannot be stat'ed on Windows.

    On Windows Git checkouts that were created in WSL you often get a `lib64`
    symlink which points to `/usr/lib64`.  The link cannot be resolved from
    Windows, and simply calling `Path.stat()` raises ``OSError(1920)``.

    Pytest calls ``collection_path.is_dir()`` during discovery *before* the
    built-in ``norecursedirs`` filter, so we short-circuit here and tell pytest
    to ignore any path whose ``stat()`` raises *any* ``OSError``.
    """
    try:
        # Trigger Path.stat() via is_dir() – cheapest call that raises if broken
        _ = collection_path.is_dir()
    except OSError:
        return True  # ignore this path entirely
    return False 