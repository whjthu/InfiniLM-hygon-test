"""Only changed STOP -> actual public watcher handoff; no actor/Native retest."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from public_deep_native.cli import watch_public_shutdown
from public_deep_native.protocol import Run


def run():
    cases=[]
    with tempfile.TemporaryDirectory() as tmp:
        config=dict(runtime_directory=tmp,run='valid-stop',actual_step='CPU.recording',
            source_manifest_SHA256='CPU',provider_config_SHA256='CPU-config',
            gpu_start_epoch=time.time(),Native_seconds=5100,service_seconds=4500)
        for foreign in [False,True]:
            config['run']='foreign-stop'if foreign else'valid-stop'
            actual=Run(config);events=[]
            group=SimpleNamespace(run=actual,failed=threading.Event())
            def trip(error):
                events.append(error);group.failed.set();actual.fail(error,stage=1)
            group._trip=trip
            if foreign:
                other=Run(dict(config,provider_config_SHA256='other-config'));other.stop()
            else:actual.stop()
            http=SimpleNamespace(should_exit=False)
            watch_public_shutdown(SimpleNamespace(group=group),http,threading.Event(),time.time()+100)
            assert http.should_exit
            if foreign:
                assert len(events)==1 and group.failed.is_set()
                assert actual.read('FAILED.json')['error']==repr(events[0])
            else:
                assert not events and not group.failed.is_set()
            cases.append(dict(case='foreign_STOP_trip_and_public_exit'if foreign else'valid_STOP_public_exit_without_trip',passed=True))
    return dict(passed=True,cases=cases,changed_consumer='actual watch_public_shutdown + actual Run records',
        recording_leaves=['HTTP should_exit object','Native group _trip operation'],
        Native_dlopen=False,GPU=False,Model_READY=False,jobs=0,old_suites_repeated=False)


if __name__=='__main__':
    q=run();(Path(__file__).parent/'stop-CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
