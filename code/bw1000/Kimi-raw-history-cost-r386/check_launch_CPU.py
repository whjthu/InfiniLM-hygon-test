"""Actual launch_once exclusiveintent/keeper argv with recording subprocess only."""
import tempfile,json,sys,io,contextlib
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
import launch_once as l

def main():
 rows=[]
 for case in ['once_then_duplicate','existing_watch','existing_intent']:
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);source=root/'source';source.mkdir();(root/'results').mkdir();(root/'logs').mkdir()
   cp=source/'receipt.json';cp.write_text(json.dumps(dict(completed=True,passed=True,GPU=0,Native_dlopen=False,source_manifest_SHA256='c'*64,actual_CPU='recordingCPU')))
   driver=source/'GPU_driver.py';driver.write_text('# recording bytes')
   m=dict(GPU_GO=True,files_SHA256={'GPU_driver.py':l.sha(driver)},dependencies_SHA256={},CPU_receipt=str(cp),CPU_receipt_SHA256=l.sha(cp),CPU_source_manifest_SHA256='c'*64,actual_CPU='recordingCPU')
   mf=source/'GPU_MANIFEST.json';mf.write_text(json.dumps(m));h=l.sha(mf)
   if case=='existing_watch':(root/'results'/(l.LABEL+'-watch.json')).touch()
   if case=='existing_intent':(root/'results'/(l.LABEL+'-intent.json')).touch()
   calls=[]
   def popen(a,**kw):
    assert '--jobid=910407'in a and '--gres=none'in a and '645'in a
    assert a[a.index('-t')+1]=='00:11:00'and kw['start_new_session']
    calls.append(a);return SimpleNamespace(pid=369)
   with mock.patch.object(l,'R',root),mock.patch.object(l,'S',source),mock.patch.object(l.subprocess,'Popen',side_effect=popen),mock.patch.object(sys,'argv',['launch','--publication-commit','a'*40,'--manifest-sha',h]),contextlib.redirect_stdout(io.StringIO()):
    if case=='once_then_duplicate':l.main()
    try:l.main()
    except (AssertionError,FileExistsError):pass
    else:raise AssertionError('Duplicate or existingmarker was not blocked')
   assert len(calls)==int(case=='once_then_duplicate')
   rows.append(dict(case=case,passed=True,Popen_calls=len(calls)))
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Slurm=False,Native_dlopen=False,cases=rows)))

if __name__=='__main__':main()
