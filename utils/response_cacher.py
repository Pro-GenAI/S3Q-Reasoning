import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Union

# New cache file (single JSON)
CACHE_FILE = "response_cache.json"

# Load cache into memory (initialize if missing)
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            CACHE: Dict[str, str] = json.load(f)
    except (json.JSONDecodeError, OSError):
        CACHE = {}
else:
    CACHE = {}


def get_cache_key(messages: Union[str, List[Dict[str, Any]]], **kwargs: Any) -> str:
    """Generate a unique cache key based on request content."""
    if isinstance(messages, str):
        content = messages
    else:
        content = " ".join([msg.get("content", "") for msg in messages if msg.get("content")])

    cache_data = {
        "content": content,
        "model": kwargs.get("model", ""),
        "temperature": kwargs.get("temperature", 1.0),
        "max_tokens": kwargs.get("max_tokens"),
    }
    cache_str = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_str.encode()).hexdigest()


def get_cached_response(cache_key: str) -> Optional[str]:
    """Retrieve cached response from memory."""
    return CACHE.get(cache_key)


def save_cached_response(cache_key: str, response: str) -> None:
    """Save response to memory + persist to disk."""
    CACHE[cache_key] = response
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(CACHE, f, ensure_ascii=False)
