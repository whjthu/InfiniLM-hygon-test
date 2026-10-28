"""Only actual original GPU runner layer iterator is changed; no guard changes."""
from pathlib import Path
import ast,hashlib,json,sys
D=Path('/refactor/incoming/Kimi-W13-wavepartial-component-r155');source=D/'run_GPU_r155.py';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(D/'source-manifest-r155.json')=='a4e967fe29460fb7066d220634f13e1def0590d8d4930feb2c18d070879d6183'
manifest=json.loads((D/'source-manifest-r155.json').read_bytes())
for name,h in manifest['files_SHA256'].items():assert sha(D/name)==h
assert sha(source)=='98ff2fc1e9d4b9e7831da8a6c7cb19fe1f5a7e986473e977e60c5cc0dd2ab029'
tree=ast.parse(source.read_bytes(),filename=str(source));before=ast.dump(tree,include_attributes=False)
hits=[n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='layer'and isinstance(n.iter,ast.List)and [v.value for v in n.iter.elts if isinstance(v,ast.Constant)]==[1,47]and len(n.iter.elts)==2]
assert len(hits)==1,'exactly one original [1,47] layer loop required'
old=hits[0].iter;hits[0].iter=ast.copy_location(ast.List(elts=[ast.Constant(value=47)],ctx=ast.Load()),old);ast.fix_missing_locations(tree)
assert ast.dump(tree,include_attributes=False)!=before
# No source file is written and original runtime/manifest/build hashes stay intact.
sys.path.insert(0,str(D));scope={'__name__':'__main__','__file__':str(source)}
exec(compile(tree,str(source),'exec'),scope)
