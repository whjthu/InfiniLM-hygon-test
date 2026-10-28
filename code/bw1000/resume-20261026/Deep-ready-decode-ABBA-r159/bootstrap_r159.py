"""Use exact production ac937 successful loader/forward/finally; change matrix only."""
import pathlib,json,hashlib,ast,os,sys
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');BASE=R/'incoming/Deep-C4-independent64-r9278/modern_Deep_independent64_r9278.py';BASE_SHA='d206637dff3e79d17756962974ed6935e85299d137f602b988d8688d29d999d4';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def transform(raw):
 tree=ast.parse(raw);before=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='forward');old=ast.dump(before,include_attributes=False);imports=0;paths=0
 for n in ast.walk(tree):
  if isinstance(n,ast.ImportFrom)and n.module=='Deep_independent64_r9278':n.module='Deep_ready_decode_r159';imports+=1
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='path'for t in n.targets):
   n.value=ast.parse("r/'results'/('Deep-ready-decode-ABBA-r159-'+tag+'.json')",mode='eval').body;paths+=1
 assert imports==1 and paths==1
 after=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='forward');assert old==ast.dump(after,include_attributes=False);ast.fix_missing_locations(tree);compile(tree,str(BASE),'exec');return tree
def main():
 g=json.loads((D/'root-GO-r159.json').read_bytes());manifest=D/'manifest-r159.json';assert g['GPU_authorized']is True and g['source_published']is True and g['source_manifest_SHA256']==sha(manifest)
 m=json.loads(manifest.read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 cp=R/'candidates/Deep-ready-decode-ABBA-r159/CPU-static.json';assert sha(cp)==g['CPU_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['Torch_imported'];assert sha(BASE)==BASE_SHA
 sys.path.insert(0,str(D));ns={'__name__':'__main__','__file__':str(BASE)};exec(compile(transform(BASE.read_bytes()),str(BASE),'exec'),ns);assert ns['report']['engine_worker_shutdown_completed']and ns['report']['completed'];print('DEEP_READY_DECODE_R159_NORMAL',flush=True)
if __name__=='__main__':main()
