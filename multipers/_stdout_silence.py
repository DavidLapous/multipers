from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager


_STDOUT_SILENCE_LOCK = threading.RLock()


@contextmanager
def silence_stdout(enabled: bool = True):
    if not enabled:
        yield
        return
    with _STDOUT_SILENCE_LOCK:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            os.close(devnull)


def call_silencing_stdout(fn, *args, enabled: bool = True, **kwargs):
    with silence_stdout(enabled=enabled):
        return fn(*args, **kwargs)
