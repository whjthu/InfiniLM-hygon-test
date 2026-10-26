import pathlib,json,os,sys,ast,hashlib
import common
D=common.SRC;r=common.ROOT;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
pin=json.loads((D/'model-source-pins.json').read_bytes());mp=r/'incoming/Kimi-W2-pair-Model-r9282/runtime/model-manifest-r9183.json'
assert sha(mp)==pin['manifest_SHA256']
for path,h in pin['files'].items():assert sha(path.replace('/refactor',str(r),1))==h,path
for p in D.glob('*.py'):assert b'\r\n'not in p.read_bytes();ast.parse(p.read_text())
closure=json.loads((r/'results/Kimi-W2pair-CPU-r9282.json').read_bytes());assert closure['passed']and closure['all96_input_IDs_valid']and closure['all_screen8_fit']and not closure['GPU']
go=json.loads((D/'GPU-GO-after-source-publication.json').read_bytes())
assert go['authorized_by']=='parent-root'and go['manifest_SHA256']==pin['manifest_SHA256']and len(go['Git_publication_SHA'])==40
closure4=json.loads((D/'CPU-closure-summary-r9282.json').read_bytes());assert closure4['passed']and closure4['actual_step']=='910379.468'and closure4['manifest_SHA256']==pin['manifest_SHA256']
os.execv(sys.executable,[sys.executable,str(D/'model_keeper.py')])
