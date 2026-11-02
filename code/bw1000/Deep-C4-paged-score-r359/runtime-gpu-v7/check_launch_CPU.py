"""Actual launch_once with temporary receipts and recording Popen; no Slurm/GPU."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import contextlib,hashlib,io,json,sys,tempfile
import launch_once as launch


def run():
    cases=[]
    for scenario in ['normal_then_duplicate','existing_watch','existing_intent']:
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);source=root/'source';source.mkdir();(root/'results').mkdir();(root/'logs').mkdir()
            def write(name,obj):
                path=source/name;path.write_text(json.dumps(obj));return str(path),launch.sha(path)
            library=source/'recording-not-a-DSO';library.write_bytes(b'CPU boundary, never dlopen')
            driver=source/'GPU_driver.py';driver.write_text('# recording source')
            cp,ch=write('recording-link.json',dict(completed=True,library=str(library),library_SHA256=launch.sha(library)))
            qp,qh=write('recording-consumer.json',dict(completed=True,actual_CPU='910407.725',GPU_initialized=False,Native_dlopen=False,
                driver_SHA256=launch.sha(driver),checks=list(range(6))))
            mp,mh=write('GPU_MANIFEST.json',dict(GPU_GO=True,source_SHA256={'GPU_driver.py':launch.sha(driver)},CPU_receipt=cp,
                CPU_receipt_SHA256=ch,CPU_consumer_receipt=qp,CPU_consumer_receipt_SHA256=qh,DSO_SHA256=launch.sha(library)))
            if scenario=='existing_watch':(root/'results'/(launch.LABEL+'-watch.json')).write_bytes(b'')
            if scenario=='existing_intent':(root/'results'/(launch.LABEL+'-intent.json')).write_bytes(b'')
            calls=[]
            def popen(args,**kw):
                assert '--jobid=910407'in args and '--gres=none'in args and args[args.index('-t')+1]=='00:15:00'
                assert '880'in args and kw['start_new_session']is True and args[-2]=='--manifest-sha'
                calls.append(args);return SimpleNamespace(pid=359725)
            with mock.patch.object(launch,'R',root),mock.patch.object(launch,'S',source),mock.patch.object(launch.subprocess,'Popen',side_effect=popen),mock.patch.object(sys,'argv',['launch_once.py','--publication-commit','a'*40,'--manifest-sha',mh]),contextlib.redirect_stdout(io.StringIO()):
                if scenario=='normal_then_duplicate':
                    launch.main()
                    try:launch.main()
                    except FileExistsError:pass
                    else:raise AssertionError('Duplicate intent launched twice')
                    assert len(calls)==1
                else:
                    try:launch.main()
                    except (AssertionError,FileExistsError):pass
                    else:raise AssertionError('Existing marker did not block duplicate')
                    assert not calls
            cases.append(dict(name=scenario,passed=True,recorded_CPU_keeper_Popen=len(calls)))
    return dict(completed=True,passed=True,GPU=0,Slurm=False,Native_dlopen=False,cases=cases,
        scope='Actual359launch_once CPUkeeper argv/exclusiveintent/unknownresponse suppression; receipt/library/providers are recording bytes')


if __name__=='__main__':print(json.dumps(run()))
