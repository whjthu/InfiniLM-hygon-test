"""Read only named source/receipt metadata. No source content/snippets, no GPU/Native."""
import pathlib,json,hashlib,os
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();pins=json.loads((D/'source-pins-r177b.json').read_bytes());rows=[]
for q in pins:
 if q['remote'].startswith('/'):
  p=pathlib.Path(q['remote']);b=p.read_bytes();rows.append({'component':q['component'],'path':str(p),'bytes':len(b),'raw_SHA256':hashlib.sha256(b).hexdigest(),'LF_SHA256':hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest(),'matches_local_raw':sha(p)==q['raw_SHA256'],'matches_local_LF':hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()==q['LF_SHA256']})
cp=R/'candidates/glm-transient-layer-r5681/CPU-build.json';proof=json.loads(cp.read_bytes());compiled=[]
for command in proof.get('commands',[]):
 if '-c'in command and '-o'in command:
  source=pathlib.Path(command[command.index('-c')+1]);obj=pathlib.Path(command[command.index('-o')+1])
  if 'indexer_selection'in source.name:
   b=source.read_bytes();compiled.append({'source':str(source),'bytes':len(b),'source_raw_SHA256':hashlib.sha256(b).hexdigest(),'source_LF_SHA256':hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest(),'object':str(obj),'object_SHA256':sha(obj)})
print(json.dumps({'CPU_only':True,'GPU':False,'source_text_exported':False,'source_snippets_exported':False,'named_sources':rows,'actual5681_receipt_SHA256':sha(cp),'selection_compilation_metadata':compiled}))
