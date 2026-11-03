"""Trusted factory for one qualified four-node backend, never a new scheduler."""
from pathlib import Path
import threading
import time


class KimiProvider:
    def __init__(self, config, prepared, owner=None):
        assert config['node'] == 0
        if owner is None:
            from .owner import run_owner
            owner = run_owner
        self.config, self.prepared, self.owner = dict(config), prepared, owner
        self.stop_intent = threading.Event()
        self.ready = threading.Event()
        self.thread = None
        self.failure = None
        self._failure_lock = threading.Lock()
        self.closed = False
        self.factory_used = False

    def __call__(self, **settings):
        if self.factory_used: raise RuntimeError('Provider factory already consumed')
        assert Path(settings['model_path']).resolve() == Path(self.config['model_path']).resolve()
        expected = dict(device='cuda', dtype='bfloat16', tensor_parallel_size=16,
            pipeline_parallel_size=2, pipeline_parallel_stage=0, cache_type='static',
            max_batch_size=1, max_cache_len=32768, max_tokens=4096,
            temperature=0.0, top_p=1.0, top_k=1, enable_graph=False,
            skip_load=False, enable_prefix_caching=False, attn_backend='default',
            use_mla=False, moe_ep_backend='disabled', moe_ep_size=1,
            weight_load_mode='async', kv_transfer_config=None, pre_transpose=False)
        for k, value in expected.items(): assert settings[k] == value, (k, settings[k], value)
        self.factory_used = True
        return self

    def __getattr__(self, name):
        return getattr(self.prepared['engine'], name)

    def _run(self):
        try:
            self.owner(self.config, self.prepared, self.stop_intent, self.ready.set)
        except BaseException as error:
            self._trip(error)
        finally:
            self.closed = True

    def _trip(self, error):
        with self._failure_lock:
            if self.failure is None: self.failure = error
        bridge = self.prepared['engine']
        bridge._fatal = self.failure
        with bridge._lock: pending = list(bridge._requests.values())
        for request in pending:
            if not request.is_finished(): request.fail(self.failure)
        try: self.request_stop()
        except BaseException as cleanup:
            if hasattr(self.failure, 'add_note'): self.failure.add_note('Stop intent: ' + repr(cleanup))

    def start(self):
        if self.thread is not None or self.closed: raise RuntimeError('No provider restart')
        self.prepared['engine'].start()
        self.thread = threading.Thread(target=self._run, name='Kimi-four-node-Native-leader0', daemon=False)
        self.thread.start()
        while not self.ready.wait(.1):
            if self.failure is not None: raise self.failure
            if self.closed: raise RuntimeError('Owner exited before readiness')
            if time.time() >= self.config['gpu_start_epoch'] + self.config['service_seconds']:
                self.stop_intent.set()
                raise TimeoutError('Original service deadline during startup')

    def is_healthy(self):
        return self.ready.is_set() and not self.closed and self.failure is None and self.prepared['engine'].is_healthy()

    def request_stop(self):
        bridge = self.prepared['engine']
        with bridge._admission_lock:
            bridge.stop()
            with bridge._lock: pending = list(bridge._requests.values())
            for request in pending:
                if not request.is_finished(): request.mark_abort(self.prepared['FinishReason'].CANCELED)
            self.stop_intent.set()

    def stop(self):
        if self.thread is None: return
        if threading.current_thread() is self.thread: raise RuntimeError('Native owner cannot join itself/HTTP')
        self.request_stop()
        bridge = self.prepared['engine']
        deadline = min(time.time() + 600, self.config['gpu_start_epoch'] + self.config['Native_seconds'])
        self.thread.join(timeout=max(0, deadline - time.time()))
        if self.thread.is_alive():
            # Native collective cannot be safely killed as a Python thread.
            # Fail-stop this owned leader; peer sockets close and fail cleanup.
            import os
            os._exit(2)
        if self.failure is not None: raise self.failure
        assert self.closed
        # Queued-but-never-active requests finish only after all four leaders'
        # reset/destroy acknowledgement, never on mere cancellation intent.
        with bridge._lock: pending = list(bridge._requests.values())
        for request in pending:
            if not request.is_finished(): request.finish(self.prepared['FinishReason'].CANCELED)
