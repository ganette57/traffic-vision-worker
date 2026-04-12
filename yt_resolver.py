from __future__ import annotations

import logging
from urllib.parse import urlparse

LOGGER = logging.getLogger("traffic_vision.yt_resolver")


def looks_like_youtube_url(value: str) -> bool:
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    return "youtube.com" in host or "youtu.be" in host


def resolve_stream_source(source: str) -> str:
    """
    Stream source resolver hook for camera URLs.
    Keeps behavior safe by returning the original source if no resolver is configured.
    """
    if not source:
        return source
    if looks_like_youtube_url(source):
        LOGGER.info("YouTube source detected; using raw URL as stream source")
    return source
