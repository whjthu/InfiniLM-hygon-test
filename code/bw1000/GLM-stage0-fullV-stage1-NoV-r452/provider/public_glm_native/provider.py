"""Public engine-factory adapter backed by actual owned Native actors."""
import time
from pathlib import Path

from .cache_policy_r452 import global_metadata
from .actors import Actors
from .ready_socket import ReadyReceiver
from .startup_diagnostic import marker, resolve


class GLMProvider:
    def __init__(self, config, prepared):
        self.config, self.prepared = dict(config), prepared
        assert config['stage'] == 1
        self.group = Actors(config, prepared)
        self._factory_used = False
        self._ready = False
        self._stopped = False

    def __call__(self, **settings):
        if self._factory_used: raise RuntimeError('Public provider factory already consumed')
        expected = dict(device='cuda', dtype='bfloat16', tensor_parallel_size=8,
            pipeline_parallel_size=2, pipeline_parallel_stage=1, cache_type='static',
            max_batch_size=1, max_cache_len=32768, max_tokens=4096,
            temperature=0.0, top_p=1.0, top_k=1, enable_graph=False,
            skip_load=False, enable_prefix_caching=False, attn_backend='default',
            use_mla=False, moe_ep_backend='disabled', moe_ep_size=1,
            weight_load_mode='async', kv_transfer_config=None, pre_transpose=False)
        assert Path(settings['model_path']).resolve() == Path(self.config['model_path']).resolve()
        for key, value in expected.items(): assert settings[key] == value, (key, settings[key], value)
        self._factory_used = True
        return self

    def __getattr__(self, name):
        # Original prepared queue/request protocol, no second scheduler/engine.
        return getattr(self.prepared['engine'], name)

    def start(self):
        if self._ready or self._stopped: raise RuntimeError('Provider cannot be restarted')
        self.prepared['engine'].start()
        try:
            # Listener is owned here before Actors.start and any actor load.
            # No object/FD crosses spawn or an SDK/class boundary.
            with ReadyReceiver(self.group.run, cancelled=lambda:
                    self.group.failed.is_set() or self.group.stop_intent.is_set() or
                    (self.group.owner_thread is not None and not self.group.owner_thread.is_alive())) as receiver:
                marker(self.config, 'provider.ready_listener.bound')
                marker(self.config, 'provider.group_start.before')
                self.group.start()  # actual processes + actual same-process owner
                marker(self.config, 'provider.group_start.return')
                marker(self.config, 'provider.verified_ready_socket.before')
                records = receiver.receive()
                marker(self.config, 'provider.verified_ready_socket.return', records=len(records))
            self._before_ready_publication(receiver.stage0_all_ready_return_unix)
            marker(self.config, 'provider.MODEL_READY_write.before')
            self.group.run.write('MODEL_READY.json', dict(Model_READY=True,
                Native_constructed_loaded_by_provider=True, actual_owner_count=len(records),
                TP=8, PP=2, rows=1, capacity=32768, cache_policy=global_metadata(self.config),
                Native_SHA256=self.config['Native_SHA256'], prepared_identity=self.prepared['identity'],
                all16_loaded_postload4GiB_and_service_agreement=True))
            marker(self.config, 'provider.MODEL_READY_write.return')
            resolve(self.config, 'MODEL_READY_published')
            self._ready = True
        except BaseException as error:
            try: resolve(self.config, 'provider.start.error')
            except BaseException as diagnostic_error:
                if hasattr(error, 'add_note'): error.add_note('Diagnostic cancellation: ' + repr(diagnostic_error))
            if self.group._claimed:
                self.group._trip(error)
                try:self.group.stop()
                except BaseException as cleanup:
                    if hasattr(error,'add_note'):error.add_note('Actor startup cleanup: '+repr(cleanup))
            raise

    def _before_ready_publication(self, stage0_all_ready_return_unix):
        """Recheck after socket close, immediately before actual READY write.

        Sender's stage0_all_ready_return_unix is fixed, never reset on receipt;
        its +120s is only an extra guard, not the keeper's first16 timestamp.
        Keeper's original first16ready+120 admission latch remains authoritative.
        Filesystem STOP visibility and cross-process races are not eliminated
        by this instantaneous check; keeper still blocks a late READY/client.
        """
        if self.group.failed.is_set():
            raise self.group.failure or RuntimeError('Actor failed before READY publication')
        if self.group.stop_intent.is_set():raise RuntimeError('Owner stop intent before READY')
        if self.group.owner_thread is None or not self.group.owner_thread.is_alive():
            raise RuntimeError('Emitter exited before READY publication')
        if self.group.run.stop_requested():raise RuntimeError('Run STOP before READY publication')
        now=time.time();epoch=self.config['gpu_start_epoch']
        if stage0_all_ready_return_unix is None or now>=stage0_all_ready_return_unix+120 or now>=epoch+2700:
            raise TimeoutError('Original startup/additional stage0-return120 expired before READY')
        if now+120>=epoch+self.config['service_seconds']:
            raise TimeoutError('Original service margin expired before READY')

    def is_healthy(self):
        return self._ready and not self._stopped and not self.group.failed.is_set() and self.prepared['engine'].is_healthy()

    def stop(self):
        if self._stopped: return
        diagnostic_error = None
        try: resolve(self.config, 'provider.stop')
        except BaseException as error: diagnostic_error = error
        self._stopped = True
        self._ready = False
        try: self.group.stop()  # Native/owned process joins only; never joins HTTP
        except BaseException as error:
            if diagnostic_error is not None and hasattr(error, 'add_note'):
                error.add_note('Diagnostic cancellation: ' + repr(diagnostic_error))
            raise
        if diagnostic_error is not None: raise diagnostic_error
