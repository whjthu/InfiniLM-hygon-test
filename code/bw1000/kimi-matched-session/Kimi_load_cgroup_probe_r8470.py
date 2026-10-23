"""Metadata-only probe to run INSIDE the next actual Model process/container.

No weight data preload, GPU access, cache creation or eviction. Physical freeRAM
is diagnostic only; any unknown cgroup/host demand means NO cache admission.
"""
import pathlib,json,os


def _probe():
    proc=pathlib.Path('/proc/self/cgroup').read_text()
    entries=[v.split(':',2) for v in proc.splitlines()]
    unified=[v[2] for v in entries if v[:2]==['0','']]
    out=dict(pid=os.getpid(),proc_self_cgroup=proc,Slurm_memory_MiB=os.environ.get('SLURM_MEM_PER_NODE'),
             intended_Model_memory_MiB=240000,actual_Model_process_required=True,
             preload_permitted=False,cache_budget_bytes=None,ancestors=[])
    if not unified:
        out['stop_reason']='No resolved cgroupv2 path; legacy/v1 budget unresolved'
        return out
    root=pathlib.Path('/sys/fs/cgroup').resolve()
    current=(root/unified[0].lstrip('/')).resolve()
    assert current.is_relative_to(root)
    while True:
        row=dict(path=str(current))
        for name in ['memory.max','memory.high','memory.current','memory.peak','memory.stat','memory.events','cpuset.cpus.effective','cpu.max']:
            p=current/name
            row[name]=p.read_text().strip() if p.is_file() else 'NOT_AVAILABLE'
        out['ancestors'].append(row)
        if current==root:break
        current=current.parent
    limits=[]
    for row in out['ancestors']:
        if row['memory.max'].isdigit() and row['memory.current'].isdigit():
            limits.append(int(row['memory.max'])-int(row['memory.current']))
    out['minimum_observed_ancestor_headroom_bytes']=min(limits) if limits else None
    out['concurrent_Model_host_peak_and_reserve']='NOT_YET_MEASURED'
    # Cache-directory existence alone is never accepted as residency proof.
    base=pathlib.Path('/dev/shm')/('iloader-r3604-'+str(os.getuid()))
    out['cache_directory_exists']=base.is_dir()
    out['cache_manifest_candidates']=[]
    if base.is_dir():
        for p in list(base.glob('**/*manifest*.json'))[:16]:
            if p.is_file() and p.stat().st_size<=256*1024:
                import hashlib
                raw=p.read_bytes()
                out['cache_manifest_candidates'].append(dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest()))
    out['stop_reason']='Metadata only: actual selected PPkeys/TPslice/usefulbytes, live lease identity, mapping residency and concurrent Model hostpeak/reserve unresolved; no preload. Future slice<=8GiB AND proven ancestor budget only.'
    return out


def probe():
    try:return _probe()
    except Exception as error:
        return dict(pid=os.getpid(),preload_permitted=False,cache_budget_bytes=None,
                    metadata_error=repr(error),stop_reason='Metadata unresolved: no preload or cache admission. Continue necessary Model correctness experiment.')


if __name__=='__main__':print(json.dumps(probe()))
