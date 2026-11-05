"""Only new concrete launcher/typed entry scope; no old CPU/Native suites."""
from pathlib import Path
import ast,contextlib,io,json,subprocess,sys,tempfile
import controller as c
import launch_once as launch
def recorded_main(m,q):
    """Actual main/parser/namespace/intent/files; only remote leaves recorded."""
    saved={name:getattr(c,name)for name in ['R','verify','membership','rows','host']};saved_host=launch.host_whole_cli;saved_popen=launch.subprocess.Popen
    try:
        with tempfile.TemporaryDirectory()as tmp:
            c.R=Path(tmp);(c.R/'results').mkdir();(c.R/'logs').mkdir()
            c.verify=lambda digest:m;c.membership=lambda:dict(recording=True);c.rows=lambda:[];c.host=lambda p:c.R/'new-runtime'
            def host(digest,commit):
                for suffix in ['stdout','stderr']:(c.R/'logs'/(c.LABEL+'-'+c.RUN+'-HOST.'+suffix)).write_bytes(b'')
                return dict(passed=True,recording=True)
            launch.host_whole_cli=host;calls=[]
            class Process:
                pid=0x12345678
                def __init__(self,argv,**kwargs):calls.append(argv)
            launch.subprocess.Popen=Process;output=io.StringIO()
            with contextlib.redirect_stdout(output):launch.main(['--manifest-sha','0'*64,'--publication-commit','1'*40])
            out=json.loads(output.getvalue());assert out['keeper_submitted']and out['intent_created']and len(calls)==1
            intent=json.loads((c.R/'results'/(c.LABEL+'-'+c.RUN+'-intent.json')).read_bytes());assert intent['planned_questions']==256 and intent['HTTP']==0
            return dict(passed=True,actual_main=True,actual_temp_namespace_and_intent=True,recorded_remote_metadata_membership_Popen=True,Slurm=0,GPU=0)
    finally:
        for name,value in saved.items():setattr(c,name,value)
        launch.host_whole_cli=saved_host;launch.subprocess.Popen=saved_popen
def main():
    d=Path(__file__).resolve().parent;c.D=d
    m=json.loads((d/'SOURCE_MANIFEST.json').read_bytes());q=c.scope_contract(m)
    gpu=c.gpu_command();keeper=launch.keeper_command('0'*64,'1'*40)
    assert gpu[gpu.index('-w')+1]==','.join(q['nodes'])and gpu[gpu.index('-t')+1]=='08:10:00'
    assert keeper[keeper.index('-w')+1]=='a14r2n20'and keeper[keeper.index('-t')+1]=='09:00:00'
    assert(q['service_seconds'],q['Native_seconds'],q['GPU_seconds'],q['keeper_seconds'])==(28200,28800,29400,32400)
    for name in ['node.py','launch_once.py','controller.py','post_once.py']:
        ast.parse((d/name).read_bytes())
        p=subprocess.run([sys.executable,'-B',str(d/name),'--help'],capture_output=True,text=True,timeout=10,check=True);assert '--help'in p.stdout
    s=(d/'node.py').read_text();assert s.index('if args.metadata_only:')<s.index('import torch')
    assert s.index('load_parameters(lib,ctx')<s.index("result['Model_READY']=True")<s.index('model_driver.generate(lib,ctx')
    s=(d/'launch_once.py').read_text();assert s.index('proof=host_whole_cli(')<s.index('c.publish(intent,')
    assert "limit=started+90"in s and 'started+80'in s and 'signal.SIGKILL'in s and "group_gone=not live()"in s
    return dict(passed=True,GPU=0,Slurm=0,Native_dlopen=False,scope='local actual CLI parsers/config/argv/main with filesystem and recorded remote leaves; not actual HOST/fresh/device qualification',GPU_argv=gpu,keeper_argv=keeper,planned_questions=256,whole_main=recorded_main(m,q))
if __name__=='__main__':print(json.dumps(main(),indent=2))
