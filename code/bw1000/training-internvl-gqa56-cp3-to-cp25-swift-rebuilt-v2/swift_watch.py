"""Observe the deployed Swift nested skip callable and its per-rank rebuilt sampler."""
from contextlib import contextmanager
import functools
import inspect
from pathlib import Path
import sys


MIXIN=Path("/swift-runtime/site/swift/trainers/mixin.py")
MIXIN_BYTES=64818
MIXIN_SHA="6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9"
SCOPE_ATTR="_gqa56_cp25_resume_skip_cursor"


def need(ok,code):
    if not ok:
        raise ValueError(code)


def active_skip_callable(hf, plain, owner, path=MIXIN, line=1218):
    active=hf.skip_first_batches
    need(active is not plain and inspect.getsourcefile(active)==str(path)
         and active.__code__.co_firstlineno==line,"EXACT_LIVE_SWIFT_SKIP_SOURCE")
    cells=inspect.getclosurevars(active).nonlocals
    need(set(cells)=={"self","origin_skip_first_batches"} and cells["self"] is owner and
         cells["origin_skip_first_batches"] is plain,"EXACT_SWIFT_SKIP_CLOSURE_OWNER_AND_ORIGIN")
    return active


def sampler_fields(shard,rank,seed):
    fields=dict(rank=rank,world_size=8,tp_size=1,total_samples=8,batch_size=1,base_seed=42,curr_seed=seed,
                shuffle=True,drop_last=False,group_by_length=False)
    need(all(type(getattr(shard,key)) is type(val) and getattr(shard,key)==val for key,val in fields.items()),
         "EXACT_NATIVE_BASE_AND_REBUILT_SAMPLER_FIELDS")


def require_rebuilt(base,result,base_sampler,skip_type,shard_type,rank):
    wrapper=result.batch_sampler
    need(base is not result and type(result) is type(base) and result.dataset is base.dataset
         and result.num_workers==4 and result.in_order is True and len(base)==8 and len(result)==4
         and type(wrapper) is skip_type and wrapper.skip_batches==4,
         "EXACT_NATIVE_REBUILT_FOUR_WORKER_LOADER")
    raw=wrapper.batch_sampler
    need(type(raw) is type(base_sampler) is shard_type and raw is not base_sampler and len(raw)==8
         and base.batch_sampler is base_sampler,"EXACT_FRESH_RAW_SAMPLER_INSIDE_NATIVE_SKIP_WRAPPER")
    sampler_fields(base_sampler,rank,42)
    sampler_fields(raw,rank,42)
    return raw


def wrap_native_context(mixin,pin,path=MIXIN):
    """Arrange finally cleanup inside the original Swift patch's exact lexical scope."""
    original=mixin._patch_skip_first_batches
    pin(path,expected=MIXIN_SHA,extent=MIXIN_BYTES)
    need(inspect.getsourcefile(inspect.unwrap(original))==str(path),
         "EXACT_SWIFT_NATIVE_PATCH_CONTEXT_SOURCE")

    @functools.wraps(original)
    @contextmanager
    def scoped(owner):
        need(mixin._patch_skip_first_batches is scoped and not hasattr(owner,SCOPE_ATTR),
             "ONE_SWIFT_CONTEXT_PROBE_PER_RANK")
        # Only the first real train context consumes the observer. The original method
        # remains authoritative for the context and for any later calls in this process.
        mixin._patch_skip_first_batches=original
        with original(owner):
            setattr(owner,SCOPE_ATTR,None)
            try:
                yield
            finally:
                active_error=sys.exc_info()[0] is not None
                cursor=getattr(owner,SCOPE_ATTR)
                try:
                    if cursor is not None:
                        cursor.detach(strict=not active_error)
                finally:
                    delattr(owner,SCOPE_ATTR)

    mixin._patch_skip_first_batches=scoped
