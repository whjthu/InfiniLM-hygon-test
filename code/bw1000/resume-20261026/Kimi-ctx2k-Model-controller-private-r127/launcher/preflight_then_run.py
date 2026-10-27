import pathlib,json,os,sys,ast,hashlib
import common
D=common.SRC;r=common.ROOT;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();pin=json.loads((D/'model-source-pins.json').read_bytes());mp=r/'incoming/Kimi-ctx2k-Model-r127/runtime/model-manifest-r9183.json';assert sha(mp)==pin['manifest_SHA256']
for path,h in pin['files'].items():assert sha(path.replace('/refactor',str(r),1))==h,path
for p in D.glob('*.py'):assert b'\r\n'not in p.read_bytes();ast.parse(p.read_bytes())
e=json.loads((D/'actual-native-CPU-evidence-r127.json').read_bytes());assert e['actual_terminal'][2:4]==['COMPLETED','0:0']and e['receipt']['native_SHA256']==pin['native_DSO_SHA256']
go=json.loads((D/'GPU-GO-after-source-publication.json').read_bytes());assert go['authorized_by']=='parent-root'and go['manifest_SHA256']==pin['manifest_SHA256']and len(go['Git_publication_SHA'])==40 and all(c in '0123456789abcdef'for c in go['Git_publication_SHA'])
os.execv(sys.executable,[sys.executable,str(D/'model_keeper.py')])
