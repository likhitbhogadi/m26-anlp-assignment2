"""Mirror everything printed to the terminal (stdout + stderr, incl. tracebacks and
library warnings) into a timestamped log file, so every run leaves a record."""

import os
import shlex
import subprocess
import sys
from datetime import datetime

import yaml


class _Tee:
    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):  # keeps progress bars etc. behaving as on a terminal
        return self.stream.isatty()


def setup_run_log(log_dir: str, script: str, cfg: dict | None = None) -> str:
    """Starts tee-ing stdout/stderr to `<log_dir>/<script>_<timestamp>.log` and writes a
    header with the command, git commit and config. Returns the log path."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{script}_{datetime.now():%Y%m%d_%H%M%S}.log")
    log_file = open(path, "a", buffering=1)  # line-buffered: survives crashes
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        commit += " (uncommitted changes)" if dirty else ""
    except OSError:
        commit = "unknown"
    print(f"# log:     {path}")
    print(f"# started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"# command: {shlex.join(sys.argv)}")
    print(f"# git:     {commit}")
    if cfg is not None:
        print("# config:\n" + "\n".join("#   " + line for line in yaml.safe_dump(cfg, sort_keys=False).splitlines()))
    print()
    return path
