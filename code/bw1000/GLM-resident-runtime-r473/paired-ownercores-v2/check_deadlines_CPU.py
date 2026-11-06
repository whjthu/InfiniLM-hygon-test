"""Pure CPU cross-file frozen deadline contract, no process or SDK imports."""
from pathlib import Path
import ast,json,re

def verify(root):
 root=Path(root);claim=json.loads((root/'GPU_RESOURCE_CLAIM.json').read_bytes());seconds=claim['seconds'];keeper=claim['keeper']['seconds'];tree=ast.parse((root/'shared.py').read_text());constants={}
 for n in ast.walk(tree):
  if isinstance(n,ast.Assign)and isinstance(n.value,ast.Constant):
   for t in n.targets:
    if isinstance(t,ast.Name):constants[t.id]=n.value.value
 assert constants['GPU_SECONDS']==seconds['GPU'] and constants['WORK_SECONDS']==seconds['Native'] and constants['KEEPER_SECONDS']==keeper
 kill,hard=map(int,re.search(r'--kill-after=(\d+)s (\d+)s',(root/'keeper.sh').read_text()).groups())
 assert seconds['Native']<seconds['inside_timeout']<seconds['GPU']<keeper
 assert hard>seconds['GPU']+40 and hard+kill<=keeper,'keeper shell must outlive GPU+post and fit originalkeeper claim'
 assert int((root/'inside.sh').read_text().split('GLM470_EPOCH',1)[1].split('+',1)[1].split('-',1)[0])==seconds['inside_timeout']
 hms=lambda n:'%02d:%02d:%02d'%(n//3600,n//60%60,n%60)
 constants_in=lambda file:{n.value for n in ast.walk(ast.parse((root/file).read_text()))if isinstance(n,ast.Constant)and isinstance(n.value,str)}
 assert hms(seconds['GPU'])in constants_in('controller.py')and hms(keeper)in constants_in('launch.py')
 # Worker phase/exit checks use the same originalepoch Native andinside caps.
 ctl=(root/'control.py').read_text();assert str(seconds['Native'])in ctl and str(seconds['inside_timeout'])in ctl
 return dict(GPU_seconds=seconds['GPU'],Native_seconds=seconds['Native'],inside_seconds=seconds['inside_timeout'],keeper_seconds=keeper,keeper_shell_hard_seconds=hard,keeper_kill_grace_seconds=kill,passed=True)
if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent)))
