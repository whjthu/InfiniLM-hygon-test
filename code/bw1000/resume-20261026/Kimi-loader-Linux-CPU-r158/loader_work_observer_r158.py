"""Linux process-wide counters; REVIEW ONLY, not imported by existing loaders."""
import pathlib
import resource
import time


def snapshot():
    tick=time.perf_counter();cpu=time.process_time();usage=resource.getrusage(resource.RUSAGE_SELF)
    io={};unavailable=None
    try:
        for line in pathlib.Path('/proc/self/io').read_text().splitlines():
            key,value=line.split(':',1);io[key]=int(value)
    except (OSError,ValueError) as exc:
        unavailable=type(exc).__name__
    return {'process_cpu_s':cpu,'minor_faults':usage.ru_minflt,'major_faults':usage.ru_majflt,
            'inblock':usage.ru_inblock,'outblock':usage.ru_oublock,'proc_io':io,
            'proc_io_unavailable':unavailable,'observer_snapshot_wall_s':time.perf_counter()-tick}


def delta(before,after):
    keys=('process_cpu_s','minor_faults','major_faults','inblock','outblock')
    out={k:after[k]-before[k] for k in keys}
    assert all(v>=0 for v in out.values())
    out['proc_io_delta']={k:after['proc_io'][k]-v for k,v in before['proc_io'].items()if k in after['proc_io']}
    out['observer_snapshots_wall_s']=before['observer_snapshot_wall_s']+after['observer_snapshot_wall_s']
    out['proc_io_available']=before['proc_io_unavailable'] is None and after['proc_io_unavailable'] is None
    out['scope']='whole node process including all worker/sampler threads; includes observer proc reads; no pure-copy/IO/H2D split; CPU seconds can exceed wall across owners'
    return out
