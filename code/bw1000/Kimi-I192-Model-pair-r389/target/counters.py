"""Existing Native HOST branch-selected counters; not GPU hardware events."""
import os
import time

SCHEDULE = [0, 1, 0, 1, 1, 0]
FIXED = {
    'INFINILM_KIMI_MOE_EXECUTION_R4665': '1',
    'INFINILM_KIMI_MOE_WORKSPACE_R4665': '1',
    'INFINILM_KIMI_I192_COMPACT_R8669ZC': '1',
    'INFINILM_KIMI_BATCH_I192_R9125': '1',
    'INFINILM_KIMI_SPEC_MATH_R3676': '0',
    'INFINILM_KIMI_W2_PAIR_R9280': '0',
    'KIMI_MODEL_SPARSE_PROFILE_R135': '0',
    'INFINILM_KIMI_MODEL_EVENTS_R8669ZQ': '0',
}


def subtract(after, before):
    assert len(after) == len(before) == 8
    assert all(len(a) == len(b) == 65 for a,b in zip(after,before))
    delta = [[a-b for a,b in zip(x,y)] for x,y in zip(after,before)]
    if any(v < 0 for row in delta for v in row):
        raise RuntimeError('Native cumulative counter regressed')
    return delta


class CounterNative:
    def __init__(self, native, save):
        self._native, self.save = native, save
        self.records, self.mode, self.ordinal = [], None, None
        self.report = None

    def __getattr__(self, name): return getattr(self._native, name)

    def begin_request(self, ordinal):
        if ordinal not in range(6) or ordinal != (0 if self.ordinal is None else self.ordinal+1):
            raise ValueError('No extra request, skipped arm or replay')
        for key,value in FIXED.items():
            if os.environ.get(key) != value: raise ValueError('Fixed environment changed: '+key)
        # Explicitly absent at process creation, hence the same SDK default branch
        # is inherited by both arms, including its first-use static getenv.
        if os.environ.get('INFINILM_MXFP4_TILE_TP_R2559') is not None:
            raise ValueError('SDK T1 flag must remain absent in both arms')
        self.mode, self.ordinal = SCHEDULE[ordinal], ordinal
        os.environ['INFINILM_KIMI_I192_GROUPED_R8669Y'] = str(self.mode)
        identity = dict(ordinal=ordinal,GROUPED=self.mode,MOE_EXECUTION=1)
        rows = self.control.collect('r389-mode-'+str(ordinal), identity)
        if len(rows)!=4 or any(r!=identity for r in rows):
            raise RuntimeError('Four leader mode agreement required')
        self.records = []
        self.start_hist = self._native.Tstats()

    def _invoke(self, operation, kind, width, past, function, args, kwargs):
        before = self._native.Tstats()
        tick = time.monotonic()
        record = dict(operation=operation,kind=kind,T=width,past=past,
                      node=self.node,PP_stage=self.node//2,TP_node_rank=self.node%2,
                      ordinal=self.ordinal,GROUPED=self.mode,local_monotonic_start=tick,
                      scope='Native HOST success histogram after original owning Runtime drain; no GPU hardware event',
                      fallback_counter_available=False)
        try:
            result = function(*args,**kwargs)
            delta = subtract(self._native.Tstats(),before)
            record.update(completed=True,T_hist_delta_by_local_owner=delta,
                          selected_HOST_branches_by_local_owner=[sum(row) for row in delta],
                          actual_forward_wall_with_counter_sampling_s=time.monotonic()-tick)
            eligible = 8 <= width <= 64
            record.update(source_T_eligible=eligible,
                          source_fallback_expected=(not self.mode or not eligible))
            if not self.mode or not eligible:
                if any(v for row in delta for v in row):
                    raise RuntimeError('OFF or T-ineligible call selected I192')
            else:
                if any(any(v for t,v in enumerate(row) if t!=width) for row in delta):
                    raise RuntimeError('Counter changed outside actual T bucket')
                # Record misses explicitly; the first ON warm must contain genuine
                # T8 selections on every local owner before formal can start.
                record['eligible_owner_misses']=[d for d,row in enumerate(delta) if row[width]==0]
            self.records.append(record)
            return result
        except BaseException as error:
            record.update(error_type=type(error).__name__,error=str(error))
            if not self.records or self.records[-1] is not record:self.records.append(record)
            if self.report is not None:
                self.report['I192_failed_call'] = record
                self.report['I192_partial_request_calls'] = self.records
                self.save(self.report)
            raise

    def call(self,kind,ids,past,tokens,*args,**kwargs):
        return self._invoke('call',kind,sum(map(len,tokens)),list(past),self._native.call,
                            (kind,ids,past,tokens,*args),kwargs)

    def call_append(self,logical,past,tokens,*args,**kwargs):
        return self._invoke('call_append','prefill_append',len(tokens),past,self._native.call_append,
                            (logical,past,tokens,*args),kwargs)

    def replay(self,partial,state):
        # Actual producer signature is [(logical,past,accepted_inputs)], not a
        # second speculation call. Partial replay is necessarily width1..7.
        if len(partial)!=1 or not 1<=partial[0][2]<=7:raise ValueError('Exact C1 partial replay width')
        if state is not True:raise ValueError('Original long publisher uses state-only replay')
        return self._invoke('replay','state_only_replay_no_MoE_forward',0,partial[0][1],self._native.replay,
                            (partial,state),{})

    def end_request(self):
        total = subtract(self._native.Tstats(),self.start_hist)
        hits = [row[8] for row in total]
        summary = dict(GROUPED=self.mode,MOE_EXECUTION=1,node=self.node,
                       T_hist_delta_by_local_owner=total,calls=self.records,
                       T8_HOST_selected_by_local_owner=hits,
                       fallback_counter_available=False,
                       no_fixed_layer_count_assumption=True)
        if self.mode and (not self.records or any(v<=0 for v in hits)):
            if self.report is not None:
                self.report['I192_failed_request']=summary;self.save(self.report)
            raise RuntimeError('ON request lacks real T8 HOST selection on each local owner; stop before later arm')
        return summary
