import sys
import os
import time
from datetime import datetime
from graphreg_system.config import LOGS_DIR

class SystemLogger:
    def __init__(self, log_filename="graphreg_execution.log"):
        self.log_path = os.path.join(LOGS_DIR, log_filename)
        
    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def info(self, message):
        msg = f"[{self._timestamp()}] [INFO] {message}"
        print(msg)
        self._write_file(msg)

    def progress(self, current, total, prefix="Progress", suffix="", decimals=1, length=40):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
        filled_length = int(length * current // total)
        bar = '#' * filled_length + '-' * (length - filled_length)
        msg = f"[{self._timestamp()}] [PROGRESS] {prefix} [{bar}] {percent}% {suffix}"
        print(msg)
        self._write_file(msg)

    def success(self, message):
        msg = f"[{self._timestamp()}] [SUCCESS] {message}"
        print(msg)
        self._write_file(msg)

    def warning(self, message):
        msg = f"[{self._timestamp()}] [WARNING] {message}"
        print(msg)
        self._write_file(msg)

    def error(self, message):
        msg = f"[{self._timestamp()}] [ERROR] {message}"
        print(msg)
        self._write_file(msg)

    def _write_file(self, msg):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

logger = SystemLogger()
