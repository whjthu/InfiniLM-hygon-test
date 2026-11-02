"""Actual launcher with recording Popen and temporary receipts; no Slurm/dlopen."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import contextlib,io,json,sys,tempfile
import launch_once as launch
def run():
 cases=[]
 for scenario in ['normal_then_duplicate','existing_watch','existing_intent','GPU_GO_false','bad_CPU_source']:
  with tempfile.TemporaryDirectory()as temporary:
   root=Path(temporary);source=root/'source';source.mkdir();(root/'results').mkdir();(root/'logs').mkdir()
   def write(name,value):
    p=source/name;p.write_text(json.dumps(value));return str(p),launch.sha(p)
   library=source/'not-a-DSO';library.write_bytes(b'CPU recording bytes, never dlopen')
   cp,ch=write('CPU.json',dict(completed=True,passed=True,library=str(library),library_SHA256=launch.sha(library),source_manifest_SHA256='b'*64,CPU_actual_lease_protocol=dict(passed=True),CPU_actual_consumers={'check_interface_CPU.py':dict(passed=True)}))
   _,mh=write('GPU_MANIFEST.json',dict(GPU_GO=scenario!='GPU_GO_false',source_SHA256={},CPU_receipt=cp,CPU_receipt_SHA256=ch,bridge_SHA256=launch.sha(library),CPU_source_manifest_SHA256=('c'*64 if scenario=='bad_CPU_source'else'b'*64)))
   if scenario=='existing_watch':(root/'results'/(launch.LABEL+'-watch.json')).write_bytes(b'')
   if scenario=='existing_intent':(root/'results'/(launch.LABEL+'-intent.json')).write_bytes(b'')
   calls=[]
   def popen(args,**kw):
    assert '--jobid=910407'in args and '--gres=none'in args and '-c1'in args
    assert args[args.index('-t')+1]=='00:15:00'and'880'in args and kw['start_new_session']
    assert str(source/'controller.py')in args and args[-2]=='--manifest-sha'
    assert not any(v.startswith('--gres=dcu')for v in args),'Invoker may submit CPU keeper only'
    calls.append(args);return SimpleNamespace(pid=390780)
   with mock.patch.object(launch,'R',root),mock.patch.object(launch,'S',source),mock.patch.object(launch.subprocess,'Popen',side_effect=popen),mock.patch.object(sys,'argv',['launch_once.py','--publication-commit','a'*40,'--manifest-sha',mh]),contextlib.redirect_stdout(io.StringIO()):
    if scenario=='normal_then_duplicate':
     launch.main()
     try:launch.main()
     except FileExistsError:pass
     else:raise AssertionError('Duplicate intent launched twice')
     assert len(calls)==1
    else:
     try:launch.main()
     except(AssertionError,FileExistsError):pass
     else:raise AssertionError('Invalid/duplicate launch accepted')
     assert not calls
   cases.append(dict(case=scenario,passed=True,recorded_CPU_keeper=len(calls)))
 return dict(passed=True,cases=cases,actual_launcher=True,Slurm=False,GPU=False,Native_dlopen=False)
if __name__=='__main__':print(json.dumps(run()))
