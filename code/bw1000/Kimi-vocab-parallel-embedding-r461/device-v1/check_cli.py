"""Only new concrete argv/claim/parser/source-order glue. No Slurm/SDK."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
import controller as c
import launch_once as launch
import embedding
def main():
 d=Path(__file__).resolve().parent;c.D=d
 m=json.loads((d/'SOURCE_MANIFEST.json').read_bytes());q=c.scope_contract(m)
 gpu=c.gpu_command();keeper=launch.keeper_command('0'*64,'1'*40)
 assert gpu[gpu.index('-w')+1]==','.join(q['nodes'])and gpu[gpu.index('-t')+1]=='00:10:00'
 assert keeper[keeper.index('-w')+1]=='a14r2n20'and keeper[keeper.index('-t')+1]=='00:11:00'
 assert q['Native_seconds']==480 and q['floor_bytes']==4294967296 and q['context']==32768
 assert q['lookup_widths']==[len(v)for v in embedding.queries()]==[1,1,64,512]
 assert q['master_address']==q['node_IPs'][q['head']]
 for name in ['node.py','launch_once.py','controller.py','post_once.py']:
  ast.parse((d/name).read_bytes());z=subprocess.run([sys.executable,'-B',str(d/name),'--help'],capture_output=True,text=True,check=True);assert '--help'in z.stdout
 source=(d/'node.py').read_text();assert source.index('if args.metadata_all_recording_hosts:')<source.index('import torch')
 assert source.index('if args.metadata_only:')<source.index('import torch')
 launch_source=(d/'launch_once.py').read_text();assert launch_source.index('proof=c.whole_cli()')<launch_source.index('c.publish(intent')<launch_source.index('subprocess.Popen')
 assert 'primary = error'in(d/'constructor.py').read_text()
 assert 'HOST_TIMEOUT_STOP'in(d/'controller.py').read_text()
 assert 'GLOBAL_MODEL_MODE_R2354'in(d/'inside.sh').read_text()
 return dict(passed=True,scope='local concrete CLI/resource/source glue only; actual HOST whole CLI and fresh four-node qualification required after publication',Native_dlopen=False,GPU=0,Slurm=0,
  GPU_argv=gpu,keeper_argv=keeper,query_ID_SHA256=[embedding.ID_SHA(v)for v in embedding.queries()])
if __name__=='__main__':print(json.dumps(main(),indent=2))
