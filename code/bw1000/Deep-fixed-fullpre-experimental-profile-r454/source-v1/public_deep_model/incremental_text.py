"""Opt-in Native bridge text consumer; no Native imports or second queue.

Use by explicit class composition before constructing a prepared engine:
class StreamingBridge(IncrementalTextMixin, ExistingFamilyBridge): ...
No installed class is patched. Existing providers remain buffered by default.
"""
import asyncio
import math
import queue
import time
from types import SimpleNamespace

_monotonic = time.monotonic


class TextPrefix:
    """Decode complete prefixes without emitting unstable replacement tails.

    Full-prefix decoding preserves tokenizer context/whitespace behavior. A
    decoder that rewrites already-published text fails explicitly; strings
    cannot be retracted from an HTTP stream. Checkpoint-specific decoding and
    parser tests are required before selecting this mixin for a real provider.
    """
    def __init__(self, decode):
        self.decode = decode
        self.published = ""

    def advance(self, token_ids, *, final):
        text = self.decode(token_ids)
        if not isinstance(text, str):
            raise TypeError("Native text decoder must return str")
        if not text.startswith(self.published):
            raise RuntimeError("Decoder rewrote already-published text")
        stable = text if final else text.rstrip("\ufffd")
        if not stable.startswith(self.published):
            raise RuntimeError("Decoder invalidated published Unicode boundary")
        delta = stable[len(self.published):]
        self.published = stable
        return delta


class IncrementalTextMixin:
    async def stream_request(self, req, timeout=100.0, request_timeout=1000.0):
        """Consume original NativeRequest progress/done/error events.

        Only text delivery changes. Native emit/finish/reset, EOS accounting,
        FIFO admission, worker error and cooperative abort ownership do not.
        Coalesced progress may publish multiple tokens in one text delta: its
        wall time must not be expanded into invented per-token client times.
        """
        timeout, request_timeout = float(timeout), float(request_timeout)
        if not all(math.isfinite(v) and v > 0 for v in (timeout, request_timeout)):
            raise ValueError("Finite positive stream/request bounds required")
        started = progress = _monotonic()
        deadline = started + request_timeout
        decode = self.decode_tokens
        if decode is None:
            decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False,
                                                       clean_up_tokenization_spaces=False)
        prefix = TextPrefix(decode)
        with req._lock:
            req.metadata.update(buffered_P0_text=False, incremental_text_candidate_r429=True)
            req.metadata["text_publish_records"] = []

        def timed_out():
            now = _monotonic()
            return now >= deadline or now - progress >= timeout

        def abort_timeout():
            self.add_aborted_req(req, self.reasons.TIMEOUT)
            req.metadata["transport_timeout"] = True
            return SimpleNamespace(token_text="", finished=True, finish_reason=self.reasons.TIMEOUT)

        try:
            while True:
                if timed_out():
                    yield abort_timeout()
                    return
                now = _monotonic()
                wait = max(.001, min(.5, deadline-now, timeout-(now-progress)))
                try:
                    kind, value = await asyncio.to_thread(req._events.get, True, wait)
                except queue.Empty:
                    continue
                # Buffered progress cannot refresh an absolute request deadline.
                if timed_out():
                    yield abort_timeout()
                    return
                if kind == "error":
                    self.add_aborted_req(req, self.reasons.ERROR)
                    raise RuntimeError("Native worker: " + str(value))
                if kind not in ("progress", "done"):
                    raise RuntimeError("Unknown Native worker event")
                progress = _monotonic()
                final = kind == "done"
                with req._lock:
                    ids = list(req.visible_token_ids)
                    finish_reason = req.finish_reason if final else None
                if final and not req.is_finished():
                    raise RuntimeError("Native done event preceded owned finish/reset")
                delta = prefix.advance(ids, final=final)
                # Decoding work must not cause a late normal text publication.
                if timed_out():
                    yield abort_timeout()
                    return
                if delta or final:
                    with req._lock:
                        req.metadata["text_publish_records"].append(dict(
                            consumer_monotonic=_monotonic(),
                            cumulative_visible_token_ids=len(ids),
                            cumulative_text_characters=len(prefix.published), final=final))
                    yield SimpleNamespace(token_text=delta, finished=final,
                                          finish_reason=finish_reason)
                if final:
                    return
        except Exception:
            self.add_aborted_req(req, self.reasons.ERROR)
            raise
        finally:
            if not req.is_finished():
                self.add_aborted_req(req, self.reasons.CANCELED)
            # Do not retire a replacement request if an external integration
            # violated the original single-use request-ID contract.
            with self._lock:
                if self._requests.get(req.request_id) is req:
                    self._requests.pop(req.request_id)
