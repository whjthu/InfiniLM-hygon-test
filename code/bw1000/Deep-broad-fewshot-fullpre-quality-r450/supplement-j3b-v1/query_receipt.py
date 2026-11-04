"""Persist only the two initial scheduler reads; no retry or GPU operation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(argv,path,identity,timeout=25,*,popen=subprocess.Popen):
    path=Path(path)
    if timeout<=0:raise TimeoutError('Initial metadata deadline already exhausted')
    start=time.monotonic_ns();deadline=start+int(timeout*1e9)
    value=dict(identity=identity,phase='before_Popen',argv=list(argv),parent_PID=os.getpid(),
        started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),started_monotonic_ns=start,
        timeout_s=timeout,deadline_monotonic_ns=deadline,child_PID=None,child_startticks=None,
        returncode=None,stdout=None,stderr=None,timeout=False,retry=False)
    def persist(first=False):
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        if first:
            with path.open('xb')as out:out.write(raw);out.flush();os.fsync(out.fileno())
        else:
            tmp=path.with_name(path.name+'.writing-'+str(os.getpid()))
            with tmp.open('xb')as out:out.write(raw);out.flush();os.fsync(out.fileno())
            tmp.replace(path)
    persist(first=True)
    proc=None
    try:
        remaining=(deadline-time.monotonic_ns())/1e9
        if remaining<=0:raise subprocess.TimeoutExpired(argv,timeout)
        proc=popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        value.update(phase='child_running',child_PID=proc.pid)
        try:
            # Only metadata of the exact child this function just created.
            fields=(Path('/proc')/str(proc.pid)/'stat').read_text().rpartition(') ')[2].split()
            value['child_startticks']=int(fields[19])
        except(OSError,IndexError,ValueError)as error:value['child_startticks_observation_error']=repr(error)
        persist()
        try:stdout,stderr=proc.communicate(timeout=max(.000001,(deadline-time.monotonic_ns())/1e9))
        except subprocess.TimeoutExpired:
            proc.kill();stdout,stderr=proc.communicate(timeout=2)
            value.update(phase='TIMEOUT',timeout=True,returncode=proc.returncode,stdout=stdout,stderr=stderr)
            raise subprocess.TimeoutExpired(argv,timeout,output=stdout,stderr=stderr)
        value.update(phase='returned',returncode=proc.returncode,stdout=stdout,stderr=stderr)
        if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,argv,stdout,stderr)
        return stdout
    except BaseException as error:
        if proc is not None and proc.poll()is None:
            proc.kill()
            try:
                stdout,stderr=proc.communicate(timeout=2)
                value.update(returncode=proc.returncode,stdout=stdout,stderr=stderr)
            except BaseException as cleanup:
                if hasattr(error,'add_note'):error.add_note('Owned metadata child cleanup: '+repr(cleanup))
        value.update(error=repr(error),phase='TIMEOUT'if isinstance(error,subprocess.TimeoutExpired)else'FAILED',timeout=isinstance(error,subprocess.TimeoutExpired))
        raise
    finally:
        primary=sys.exc_info()[1]
        value.update(ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ended_monotonic_ns=time.monotonic_ns(),elapsed_local_monotonic_s=(time.monotonic_ns()-start)/1e9)
        for name in ['stdout','stderr']:
            if value[name]is not None:value[name+'_SHA256']=hashlib.sha256(value[name].encode()).hexdigest()
        try:persist()
        except BaseException as secondary:
            # Retain an existing command error as primary; never invent success.
            if primary is not None and hasattr(primary,'add_note'):primary.add_note('Query receipt persistence: '+repr(secondary))
            else:raise
