"""Concrete local actor supervisor, sharing only UID/scalar records by file.

The emitter queue lives only in the public process and its owner thread. No
NativeRequest, processor, tensor, context or lock crosses spawn serialization.
"""
import json
import multiprocessing
import os
import threading
import time

from .checkpoint_loader import bind_library
from .protocol import Run
from .worker import owner_main, process_owner, torch_setup


class Actors:
    def __init__(self, config, prepared=None):
        json.dumps(config, allow_nan=False)  # source config/UID scalar boundary
        self.config = dict(config)
        self.stage = config['stage']
        self.run = Run(config)
        self.prepared = prepared
        assert (self.stage == 1) == (prepared is not None)
        self.processes = []
        self.owner_thread = None
        self.monitor_thread = None
        self.failure = None
        self.failed = threading.Event()
        self.finished = threading.Event()
        self.stop_intent = threading.Event()
        self._started = False
        self._stopping = False
        self._claimed = False
        self._trip_lock = threading.Lock()

    def start(self):
        if self._started: raise RuntimeError('Actor group already submitted; no replay')
        self.run.write('leader-%d.json' % self.stage, dict(stage=self.stage, PID=os.getpid()))
        self._claimed = self._started = True
        if self.stage == 0:
            # Same original parent-side ID generation; no model construction.
            torch_setup(0)
            binding = bind_library(self.config['Native_library'], self.config['Native_SHA256'])
            payload = dict(world=binding.unique(), tp=[binding.unique() for _ in range(2)],
                           control=binding.unique())
            self.run.write('COMMUNICATORS.json', dict(payload=payload))
        self.run.wait('COMMUNICATORS.json')
        self.run.wait('PREPARED.json')
        context = multiprocessing.get_context('spawn')
        for rank in range(8):
            if self.stage == 1 and rank == 0: continue
            proc = context.Process(target=process_owner, args=(rank, self.config),
                                   name='GLM-owner-%d-%d' % (self.stage, rank))
            proc.start()
            self.processes.append(proc)
        if self.stage == 1:
            self.owner_thread = threading.Thread(target=self._emitter,
                name='GLM-Native-emitter-owner', daemon=False)
            self.owner_thread.start()
        self.run.write('actors-%d.json' % self.stage, dict(stage=self.stage,
            owned_process_PIDs=[p.pid for p in self.processes],
            emitter_same_process_thread=self.owner_thread is not None,
            NativeRequest_serialized=False))
        self.monitor_thread = threading.Thread(target=self._monitor,
            name='GLM-owner-supervisor', daemon=False)
        self.monitor_thread.start()

    def _emitter(self):
        try: owner_main(0, self.config, self.prepared, self.stop_intent)
        except BaseException as error: self._trip(error)

    def _trip(self, error):
        if not self._claimed: return  # duplicate intent never poisons an old run
        with self._trip_lock:
            if not self.failed.is_set():
                self.failure = error
                self.failed.set()
                try:self.run.fail(error, stage=self.stage)
                except BaseException as persistence_error:
                    if hasattr(error,'add_note'):error.add_note('Failure-record publication failed: '+repr(persistence_error))
                if self.prepared is not None:
                    bridge = self.prepared['engine']
                    with bridge._lock:
                        bridge._fatal = error
                        pending = list(bridge._requests.values())
                    for request in pending:
                        if not request.is_finished(): request.fail(error)
        # These are the Process objects created by this group, no PID search.
        for process in self.processes:
            if process.is_alive(): process.terminate()

    def _monitor(self):
        try:
            while not self.finished.is_set():
                self.run.raise_if_failed()
                if self.run.remaining() <= 0: raise TimeoutError('Original Native residency expired')
                for process in self.processes:
                    if process.exitcode not in (None, 0):
                        raise RuntimeError('Owned actor failed: %s RC%s' % (process.name, process.exitcode))
                # An actor returning0 without its final record is not accepted.
                for rank in range(8):
                    if self.stage == 1 and rank == 0:
                        exited = self.owner_thread is not None and not self.owner_thread.is_alive()
                    else:
                        process = next(p for p in self.processes if p.name.endswith('-%d' % rank))
                        exited = process.exitcode == 0
                    if exited and not (self.run.path / ('closed-%d-%d.json' % (self.stage, rank))).exists():
                        # NFS publication uses atomic completed files; allow its
                        # visibility bounded by the same original deadline.
                        self.run.wait('closed-%d-%d.json' % (self.stage, rank), minimum=0)
                time.sleep(.2)
        except BaseException as error: self._trip(error)

    def wait_ready(self):
        records = self.run.all_ready()
        if self.failed.is_set(): raise self.failure
        if time.time() + 120 >= self.config['gpu_start_epoch'] + self.config['service_seconds']:
            raise TimeoutError('Insufficient original service window after loading')
        return records

    def wait_peer_node(self):
        assert self.stage == 0
        self.wait_ready()
        while any(p.is_alive() for p in self.processes):
            if self.failed.wait(.2): raise self.failure
        for p in self.processes:
            p.join(timeout=1)
            assert p.exitcode == 0
        for rank in range(8):
            assert self.run.wait('closed-0-%d.json' % rank, minimum=0)['normal']
        self.finished.set()
        self.monitor_thread.join(timeout=2)

    def stop(self):
        if not self._started or self._stopping: return
        if self.owner_thread is threading.current_thread():
            raise RuntimeError('Owner cannot join itself; HTTP owns provider stop')
        self._stopping = True
        if self.prepared is not None:
            bridge = self.prepared['engine']
            # Same-process RLock closes admission atomically with stop intent.
            # No request is finished here: active cancel votes and final STOP
            # drain still own all reset/finish semantics in the actual loop.
            with bridge._admission_lock:
                bridge.stop()
                self.stop_intent.set()
                with bridge._lock: pending = list(bridge._requests.values())
                for request in pending:
                    if not request.is_finished(): request.mark_abort(self.prepared['FinishReason'].CANCELED)
        else:
            self.stop_intent.set()
        try:self.run.stop()
        except BaseException as error:self._trip(error)
        # Original service bound reserves600s for normal cleanup. W4 may still
        # be inside a whole committed prefill group when cancellation arrives;
        # an arbitrary40s join must not kill a healthy group before its vote.
        grace = 20 if self.failed.is_set() else 600
        end = time.monotonic() + min(grace, max(0, self.run.remaining()))
        while time.monotonic() < end:
            if self.failed.is_set(): end = min(end, time.monotonic() + 20)
            alive = any(p.is_alive() for p in self.processes)
            alive = alive or (self.owner_thread is not None and self.owner_thread.is_alive())
            if not alive: break
            time.sleep(.1)
        if self.failed.is_set() or any(p.is_alive() for p in self.processes) or (
                self.owner_thread is not None and self.owner_thread.is_alive()):
            error = self.failure or RuntimeError('Native owner shutdown exceeded original bound')
            self._trip(error)
            for p in self.processes:
                p.join(timeout=2)
                if p.is_alive(): p.kill();p.join(timeout=2)
            if self.owner_thread is not None and self.owner_thread.is_alive():
                # A Native CCL call cannot safely be canceled as a Python
                # thread. Original fail-stop terminates this owned leader.
                os._exit(2)
            self.finished.set()
            if self.monitor_thread: self.monitor_thread.join(timeout=2)
            raise error
        if self.owner_thread is not None: self.owner_thread.join(timeout=0)
        for process in self.processes:
            process.join(timeout=0)
            assert process.exitcode == 0
        # Stop is globally complete only when every actual owner published
        # normal original context destruction and Torch-after-shutdown.
        for stage in range(2):
            for rank in range(8):
                row = self.run.wait('closed-%d-%d.json' % (stage, rank), minimum=0)
                assert row['normal'] and row['post_close_Torch_liveness']
        self.finished.set()
        if self.monitor_thread: self.monitor_thread.join(timeout=2)
