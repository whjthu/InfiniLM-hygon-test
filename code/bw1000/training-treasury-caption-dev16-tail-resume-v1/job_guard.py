"""Windows native2GiB process/job commit limits and one original epoch watchdog."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import time

LIMIT=2*1024**3
CPU_COUNT=2


class BASIC(ctypes.Structure):
    _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),
        ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),
        ('ActiveProcessLimit',wintypes.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]


class IO(ctypes.Structure):
    _fields_=[(n,ctypes.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
                                           'ReadTransferCount','WriteTransferCount','OtherTransferCount')]


class EXTENDED(ctypes.Structure):
    _fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),
        ('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]


class PROCESS_MEMORY(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
        *[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
        'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]]


def current_affinity():
    if os.name!='nt':raise RuntimeError('Windows actual affinity required')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetCurrentProcess.restype=wintypes.HANDLE
    k.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    process,system=ctypes.c_size_t(),ctypes.c_size_t()
    if not k.GetProcessAffinityMask(k.GetCurrentProcess(),ctypes.byref(process),ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(process_mask=process.value,system_mask=system.value,CPU_count=process.value.bit_count())


def guarded_run(argv,gate,directory,deadline,monotonic_end):
    if os.name!='nt':raise RuntimeError('This prepared local scope requires actual WindowsJobObject; no unbounded fallback')
    k=ctypes.WinDLL('kernel32',use_last_error=True);ps=ctypes.WinDLL('psapi',use_last_error=True)
    k.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];k.CreateJobObjectW.restype=wintypes.HANDLE
    k.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    k.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
    k.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
    k.SetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    k.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    k.TerminateJobObject.argtypes=[wintypes.HANDLE,wintypes.UINT]
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    ps.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    allowed=current_affinity();bits=allowed['process_mask']&allowed['system_mask']
    selected=0
    for _ in range(CPU_COUNT):
        if not bits:raise RuntimeError('Fewer than two allowed logical CPUs; no unbounded fallback')
        bit=bits&-bits;selected|=bit;bits^=bit
    job=k.CreateJobObjectW(None,None)
    if not job:raise ctypes.WinError(ctypes.get_last_error())
    flags=0x10|0x100|0x200|0x2000
    info=EXTENDED();info.BasicLimitInformation.LimitFlags=flags
    info.BasicLimitInformation.Affinity=selected
    info.ProcessMemoryLimit=info.JobMemoryLimit=LIMIT
    child=None;peak_RSS=0;timed_out=False;started=time.monotonic()
    try:
        if not k.SetInformationJobObject(job,9,ctypes.byref(info),ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        observed=EXTENDED()
        if not k.QueryInformationJobObject(job,9,ctypes.byref(observed),ctypes.sizeof(observed),None):
            raise ctypes.WinError(ctypes.get_last_error())
        if ((observed.ProcessMemoryLimit,observed.JobMemoryLimit)!=(LIMIT,LIMIT)
                or observed.BasicLimitInformation.LimitFlags & flags!=flags
                or observed.BasicLimitInformation.Affinity!=selected):
            raise RuntimeError('Job memory/2CPU affinity limit not applied')
        with (directory/'worker.stdout').open('xb') as stdout,(directory/'worker.stderr').open('xb') as stderr:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
            if not k.AssignProcessToJobObject(job,wintypes.HANDLE(int(child._handle))):
                child.terminate();child.wait(timeout=5);raise ctypes.WinError(ctypes.get_last_error())
            if not k.SetProcessAffinityMask(wintypes.HANDLE(int(child._handle)),selected):
                child.terminate();child.wait(timeout=5);raise ctypes.WinError(ctypes.get_last_error())
            actual_mask,system_mask=ctypes.c_size_t(),ctypes.c_size_t()
            if not k.GetProcessAffinityMask(wintypes.HANDLE(int(child._handle)),ctypes.byref(actual_mask),ctypes.byref(system_mask)):
                child.terminate();child.wait(timeout=5);raise ctypes.WinError(ctypes.get_last_error())
            if actual_mask.value!=selected or actual_mask.value.bit_count()!=CPU_COUNT:
                child.terminate();child.wait(timeout=5);raise RuntimeError('Actual worker affinity differs before gate')
            # Child may import stdlib, but cannot fetch/allocate workload before this gate.
            with gate.open('x',encoding='utf8') as stream:json.dump(dict(memory_limit_bytes=LIMIT,deadline=deadline,
                CPU_count=CPU_COUNT,worker_affinity_mask=selected,job_affinity_verified=True),stream)
            while child.poll() is None:
                remaining=min(deadline-time.time(),monotonic_end-time.monotonic())-5
                if remaining<=0:
                    timed_out=True;k.TerminateJobObject(job,124);break
                sample=PROCESS_MEMORY();sample.cb=ctypes.sizeof(sample)
                if ps.GetProcessMemoryInfo(wintypes.HANDLE(int(child._handle)),ctypes.byref(sample),ctypes.sizeof(sample)):
                    peak_RSS=max(peak_RSS,sample.PeakWorkingSetSize)
                time.sleep(min(0.1,remaining))
            child.wait(timeout=5)
        if not k.QueryInformationJobObject(job,9,ctypes.byref(observed),ctypes.sizeof(observed),None):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(returncode=child.returncode,timed_out=timed_out,wall_seconds=time.monotonic()-started,
            WindowsJobObject_limits_verified=True,process_commit_limit_bytes=LIMIT,job_commit_limit_bytes=LIMIT,
            job_CPU_affinity_verified=True,worker_CPU_affinity_verified_before_gate=True,
            worker_CPU_count=CPU_COUNT,worker_affinity_mask=selected,parent_allowed_CPU_count=allowed['CPU_count'],
            actual_peak_process_commit_bytes=observed.PeakProcessMemoryUsed,actual_peak_job_commit_bytes=observed.PeakJobMemoryUsed,
            observed_OS_PeakWorkingSetSize_bytes=peak_RSS,RSS_measurement='OS counter sampled while alive; not cgroup or per-instruction maximum',
            fake_cgroup_claim=False,deadline=deadline)
    finally:
        if child is not None and child.poll() is None:
            k.TerminateJobObject(job,125)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        k.CloseHandle(job)
