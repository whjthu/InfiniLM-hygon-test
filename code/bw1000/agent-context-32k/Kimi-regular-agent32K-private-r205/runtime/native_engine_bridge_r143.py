"""Inject a resident Native worker into 7ceb HTTP/SSE with scoped timeout fixes.

No native/runtime imports occur here. The owner thread consumes take(), emits
actual model tokens, resets/drains all owners, and only then finishes a request.
P0 buffers visible token text until completion; no incremental latency claim.
Use the paired server_timeout_overlay_r145.py in a private server overlay.
"""
import asyncio
import queue
import threading
import time
from types import SimpleNamespace

VALIDATION_STREAM_TIMEOUT_S = 3600.0
VALIDATION_REQUEST_TIMEOUT_S = 10800.0


class NativeRequest:
    def __init__(self, request_id, prompt, ids, max_tokens, eos_ids, reasons):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = tuple(ids)
        self.max_new_tokens = max_tokens
        self.generated_token_ids = []
        self.visible_token_ids = []
        self.cancel_event = threading.Event()
        self.finish_reason = None
        self.abort_reason = None
        self.metadata = {}
        self._events = queue.Queue()
        self._lock = threading.Lock()
        self._ended = False
        self._eos_ids = frozenset(eos_ids)
        self._reasons = reasons

    def emit_token(self, token_id, visible=None):
        if type(token_id) is not int or token_id < 0:
            raise ValueError("Native token must be a nonnegative integer")
        with self._lock:
            if self._ended:
                raise RuntimeError("Cannot emit after finish")
            self.generated_token_ids.append(token_id)
            if visible is None:
                visible = token_id not in self._eos_ids
            if visible:
                self.visible_token_ids.append(token_id)
            self.metadata.setdefault("token_monotonic", []).append(time.monotonic())
        self._events.put(("progress", None))

    def finish(self, reason):
        """Call only after the backend's original reset/drain boundary."""
        with self._lock:
            if self._ended:
                return
            self.finish_reason = self.abort_reason or reason
            self._ended = True
        self._events.put(("done", None))

    def fail(self, error):
        with self._lock:
            if self._ended:
                return
            if self.abort_reason not in (self._reasons.TIMEOUT, self._reasons.ERROR):
                self.abort_reason = self._reasons.ERROR
            self.finish_reason = self.abort_reason
            self.metadata["native_error"] = str(error)
            self._ended = True
        self._events.put(("error", str(error)))

    def is_finished(self):
        return self._ended

    def mark_abort(self, reason):
        """Transport intent only; Native reset/drain still owns finish()."""
        with self._lock:
            severe = (self._reasons.TIMEOUT, self._reasons.ERROR)
            if self.abort_reason not in severe:
                if self.abort_reason is None or reason in severe:
                    self.abort_reason = reason
            self.cancel_event.set()

    def get_prompt_length(self):
        return len(self.prompt_token_ids)

    def get_num_generated_tokens(self):
        return len(self.generated_token_ids)

    def get_total_length(self):
        return self.get_prompt_length() + self.get_num_generated_tokens()


class NativeEngineBridge:
    def __init__(self, tokenizer, max_context, finish_reasons, *, encode_chat=None,
                 decode_tokens=None, eos_token_ids=(), default_max_tokens=96,
                 max_pending=8):
        self.tokenizer = tokenizer
        self.max_context = int(max_context)
        self.reasons = finish_reasons
        self.encode_chat = encode_chat
        self.decode_tokens = decode_tokens
        self.eos_ids = tuple(eos_token_ids)
        self.default_max_tokens = int(default_max_tokens)
        self.max_pending = int(max_pending)
        self._pending = queue.Queue()
        self._requests = {}
        self._lock = threading.Lock()
        self._stopped = False
        self._fatal = None

    def start(self):
        if self._stopped:
            raise RuntimeError("Bridge is stopped")

    def stop(self):
        self._stopped = True
        self._pending.put(None)

    def is_healthy(self):
        return not self._stopped and self._fatal is None

    def take(self, timeout=1.0):
        """Owner thread receives NativeRequest, None on stop, or queue.Empty."""
        return self._pending.get(timeout=timeout)

    def add_chat_request(self, *, messages, sampling_params, request_id,
                         request_data=None, add_generation_prompt=True,
                         chat_template_kwargs=None):
        if not self.is_healthy():
            raise RuntimeError("Resident Native worker is unavailable")
        data = request_data or {}
        if data.get("n", 1) != 1:
            raise ValueError("P0 Native bridge supports n=1")
        if float(sampling_params.temperature) != 0.0:
            raise ValueError("This validated Native route supports greedy temperature=0")
        if getattr(sampling_params, "stop", None):
            raise ValueError("Native P0 stop strings are not implemented")
        kwargs = dict(chat_template_kwargs or {})
        if self.encode_chat is not None:
            prompt, ids = self.encode_chat(messages, data,
                                           add_generation_prompt, kwargs)
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt, **kwargs)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        ids = list(ids)
        limit = getattr(sampling_params, "max_tokens", None)
        limit = self.default_max_tokens if limit is None else int(limit)
        if not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("Invalid official tokenizer output")
        if limit < 1 or len(ids) + limit > self.max_context:
            raise ValueError("Complete prompt plus output budget exceeds validated context; no truncation")
        req = NativeRequest(request_id, prompt, ids, limit, self.eos_ids, self.reasons)
        req.metadata.update(buffered_P0_text=True, context_capacity=self.max_context,
                            accepted_monotonic=time.monotonic())
        with self._lock:
            if request_id in self._requests:
                raise ValueError("Duplicate request ID")
            if len(self._requests) >= self.max_pending:
                raise RuntimeError("Native validation queue is full")
            req.metadata["client_session"] = data.get("user") if isinstance(data.get("user"), str) else None
            req.metadata["active_requests_at_accept"] = len(self._requests)
            self._requests[request_id] = req
        self._pending.put(req)
        return req

    def add_aborted_req(self, req, reason):
        req.mark_abort(reason)

    def validate_data(self, data, sampling_params):
        """Pre-SSE input checks without enqueueing or touching Native state."""
        if data.get("n", 1) != 1 or float(sampling_params.temperature) != 0.0:
            raise ValueError("Native P0 supports n=1 and greedy temperature=0")
        if getattr(sampling_params, "stop", None):
            raise ValueError("Native P0 stop strings are not implemented")
        messages = data.get("messages", [])
        kwargs = dict(data.get("chat_template_kwargs") or {})
        generation_prompt = bool(data.get("add_generation_prompt", True))
        if self.encode_chat is not None:
            prompt, ids = self.encode_chat(messages, data, generation_prompt, kwargs)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=generation_prompt, **kwargs)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        limit = getattr(sampling_params, "max_tokens", None)
        limit = self.default_max_tokens if limit is None else int(limit)
        if not ids or limit < 1 or len(ids) + limit > self.max_context:
            raise ValueError("Complete prompt plus output budget exceeds validated context; no truncation")

    async def stream_request(self, req, timeout=100.0, request_timeout=1000.0):
        started = progress = time.monotonic()
        deadline = started + float(request_timeout)
        try:
            while True:
                now = time.monotonic()
                if now >= deadline or now - progress >= timeout:
                    self.add_aborted_req(req, self.reasons.TIMEOUT)
                    req.metadata["transport_timeout"] = True
                    yield SimpleNamespace(token_text="", finished=True,
                                          finish_reason=self.reasons.TIMEOUT)
                    break
                try:
                    wait = max(0.001, min(0.5, deadline - now,
                                         float(timeout) - (now - progress)))
                    kind, value = await asyncio.to_thread(req._events.get, True, wait)
                except queue.Empty:
                    continue
                # A busy progress queue must not extend the absolute deadline.
                now = time.monotonic()
                if now >= deadline or now - progress >= timeout:
                    self.add_aborted_req(req, self.reasons.TIMEOUT)
                    req.metadata["transport_timeout"] = True
                    yield SimpleNamespace(token_text="", finished=True,
                                          finish_reason=self.reasons.TIMEOUT)
                    break
                progress = time.monotonic()
                if kind == "progress":
                    continue
                if kind == "error":
                    self.add_aborted_req(req, self.reasons.ERROR)
                    raise RuntimeError("Native worker: " + value)
                if kind != "done":
                    raise RuntimeError("Unknown Native worker event")
                ids = list(req.visible_token_ids)
                text = (self.decode_tokens(ids) if self.decode_tokens is not None
                        else self.tokenizer.decode(ids, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False))
                yield SimpleNamespace(token_text=text, finished=True,
                                      finish_reason=req.finish_reason)
                break
        finally:
            if not req.is_finished():
                self.add_aborted_req(req, self.reasons.CANCELED)
            with self._lock:
                self._requests.pop(req.request_id, None)


def existing_server_app(engine, model_path, parser, *, host="127.0.0.1", port=43843):
    """Inject the engine into isolated7ceb+timeout routes; keep parser unchanged."""
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    import infinilm.server.inference_server as private_server
    # The private routes pass these module constants explicitly to both streams.
    # Set them before the server instance exists; preserve actual progress and
    # absolute-deadline/abort behavior in NativeEngineBridge.stream_request.
    private_server.DEFAULT_STREAM_TIMEOUT = VALIDATION_STREAM_TIMEOUT_S
    private_server.DEFAULT_REQUEST_TIMEOUT = VALIDATION_REQUEST_TIMEOUT_S
    InferenceServer = private_server.InferenceServer
    engine.validation_timeouts_s = {
        "no_actual_progress": VALIDATION_STREAM_TIMEOUT_S,
        "absolute_request_including_queue": VALIDATION_REQUEST_TIMEOUT_S,
    }
    server = InferenceServer(model_path=model_path, tool_call_parser=parser,
                             temperature=0.0, top_p=1.0, top_k=1,
                             max_tokens=engine.default_max_tokens,
                             max_cache_len=engine.max_context, max_batch_size=1,
                             host=host, port=port)
    server.engine = engine
    original_sampling = server._build_sampling_params

    def checked_sampling(data):
        params = original_sampling(data)
        engine.validate_data(data, params)
        return params

    # Original routes already convert ValueError here into pre-inference HTTP400.
    # No protocol/parser or global server class is modified.
    server._build_sampling_params = checked_sampling

    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(lifespan=lifespan)
    server._register_routes(app)
    return server, app
