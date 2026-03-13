"""Plan cache for compiled LLM plans."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


def cache_key(intent: str, context: Dict[str, Any]) -> str:
    """
    Build a stable cache key from intent and context. Context should be
    JSON-serializable (e.g. source_path, columns, tables, runtime_params).
    """
    payload = {"intent": intent, **context}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class PlanCache(Protocol):
    """Protocol for plan cache backends."""

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return cached plan if present, else None."""
        ...

    def set(self, key: str, plan: Dict[str, Any]) -> None:
        """Store a validated plan under key."""
        ...


class FilePlanCache:
    """
    File-based plan cache. Persists across process restarts so a second
    "python main.py dataframe" can reuse the plan from the first run.
    """

    def __init__(self, cache_dir: str = ".plan_cache"):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def set(self, key: str, plan: Dict[str, Any]) -> None:
        self._path(key).write_text(json.dumps(copy.deepcopy(plan)))


class InMemoryPlanCache:
    """
    In-memory LRU cache for plans. Optional max_size to bound memory; oldest entry is evicted.
    """

    def __init__(self, max_size: Optional[int] = 1024):
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be >= 1 or None")
        self._max_size = max_size
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, plan: Dict[str, Any]) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if self._max_size is not None and len(self._store) >= self._max_size:
                self._store.popitem(last=False)
        self._store[key] = copy.deepcopy(plan)


class PlanCacheGate:
    """
    Wrapper that enables or disables caching at runtime. When enabled=False,
    get() always returns None and set() is a no-op. Set .enabled from the CLI
    or config to turn plan cache on/off without changing decorators.
    """

    def __init__(self, cache: PlanCache, enabled: bool = True):
        self._cache = cache
        self.enabled = enabled

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        return self._cache.get(key)

    def set(self, key: str, plan: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._cache.set(key, plan)


plan_cache = PlanCacheGate(FilePlanCache(cache_dir=".plan_cache"), enabled=True)
