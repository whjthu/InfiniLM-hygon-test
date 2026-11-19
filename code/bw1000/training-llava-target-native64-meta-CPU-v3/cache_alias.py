"""OneVision compatibility with the exact Hygon Transformers cache provider."""

from contextlib import contextmanager


@contextmanager
def scoped_sliding_cache_alias(cache_utils, *, provider_pin, expected_provider):
    """Expose the upstream alias while the pinned OneVision module is imported."""
    if provider_pin != expected_provider:
        raise ValueError("Unexpected Transformers cache provider bytes")
    attributes = vars(cache_utils)
    if "SlidingWindowCache" in attributes:
        raise ValueError("Transformers already defines SlidingWindowCache")
    static_cache = attributes.get("StaticCache")
    if not isinstance(static_cache, type) or static_cache.__module__ != cache_utils.__name__:
        raise ValueError("StaticCache must come from the pinned provider")

    cache_utils.SlidingWindowCache = static_cache
    try:
        yield static_cache
    finally:
        del cache_utils.SlidingWindowCache
