import pathlib,json,os,sys,ast,hashlib
import common
D=common.SRC;r=common.ROOT;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
pin=json.loads((D/'model-source-pins.json').read_bytes());mp=r/'incoming/Kimi-cap8-micro1-Model-r9293/runtime/model-manifest-r9183.json';assert sha(mp)==pin['manifest_SHA256']
for path,h in pin['files'].items():assert sha(path.replace('/refactor',str(r),1))==h,path
for p in D.glob('*.py'):assert b'\r\n'not in p.read_bytes();ast.parse(p.read_bytes())
closure=json.loads((D/'CPU-closure-summary-r9293.json').read_bytes());assert closure['passed']and closure['actual_terminal'][2:4]==['COMPLETED','0:0']and closure['manifest_SHA256']==pin['manifest_SHA256']
go=json.loads((D/'GPU-GO-after-source-publication.json').read_bytes());assert go['authorized_by']=='parent-root'and go['manifest_SHA256']==pin['manifest_SHA256']and len(go['Git_publication_SHA'])==40
os.execv(sys.executable,[sys.executable,str(D/'model_keeper.py')])
