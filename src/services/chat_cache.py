"""Cached chat model loading for VV-GPT3."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, Tuple

from src.chat.chat import ChatBot


class ChatModelCache:
    """Cache ChatBot instances by checkpoint path and mtime.

    Loading large PyTorch checkpoints on every message makes chat feel slow. This
    cache keeps the active model in memory while automatically invalidating when
    the checkpoint file changes.
    """

    def __init__(self, max_models: int = 2):
        self.max_models = max_models
        self._lock = threading.Lock()
        self._cache: Dict[str, Tuple[float, float, ChatBot]] = {}

    def get(self, checkpoint_path: str | Path) -> ChatBot:
        path = Path(checkpoint_path).resolve()
        mtime = path.stat().st_mtime
        key = str(path)
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] == mtime:
                self._cache[key] = (mtime, time.time(), cached[2])
                return cached[2]

            bot = ChatBot(str(path))
            self._cache[key] = (mtime, time.time(), bot)
            self._evict_if_needed()
            return bot

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_models:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"loaded_models": len(self._cache), "max_models": self.max_models}
