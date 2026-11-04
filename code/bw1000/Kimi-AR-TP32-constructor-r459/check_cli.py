"""Only new local argv/claim and parser glue; not a HOST/device qualification."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
import controller as c
import launch_once as launch

def main():
    d=Path(__file__).resolve().parent
    c.D=d
    m=json.loads((d/'SOURCE_MANIFEST.json').read_bytes())
    q=c.scope_contract(m)
    gpu=c.gpu_command();keeper=launch.keeper_command('0'*64,'1'*40)
    assert gpu[gpu.index('-w')+1]==','.join(q['nodes'])
    assert keeper[keeper.index('-w')+1]=='a14r2n20' and keeper[keeper.index('-t')+1]=='00:15:00'
    assert keeper[keeper.index('--manifest-sha')+1]=='0'*64
    assert keeper[keeper.index('--publication-commit')+1]=='1'*40
    for name in ['launch_once.py','controller.py','node.py']:
        ast.parse((d/name).read_text())
        z=subprocess.run([sys.executable,'-B',str(d/name),'--help'],capture_output=True,text=True,check=True)
        assert '--help'in z.stdout
    # Exact source-order assertions for the new device/metadata boundary.
    node=(d/'node.py').read_text()
    assert node.index('if a.metadata_all_recording_hosts:')<node.index('import torch')
    assert node.index('if a.metadata_only:')<node.index('import torch')
    launch_source=(d/'launch_once.py').read_text()
    assert launch_source.index('if a.preflight_only:')<launch_source.index('c.publish(intent')<launch_source.index('subprocess.Popen')
    assert 'primary = error'in(d/'constructor.py').read_text()
    print(json.dumps(dict(passed=True,scope='local new argv/parser/source ordering only',
        GPU=0,Native_dlopen=False,Slurm=0,actual_HOST_whole_cli='REQUIRED_AFTER_PUBLICATION',
        GPU_argv=gpu,keeper_argv=keeper,
        source_SHA256={n:hashlib.sha256((d/n).read_bytes()).hexdigest()for n in ['node.py','launch_once.py','controller.py']}),indent=2))
if __name__=='__main__':main()
