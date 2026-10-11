from __future__ import annotations

import json
from pathlib import Path
import threading
import time


class JSONLTraceWriter:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **fields) -> None:
        record = {"event": event, "wall_time": time.time(), **fields}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

