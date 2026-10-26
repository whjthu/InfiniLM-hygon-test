import pathlib,json,hashlib,subprocess,re,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');p=r/'candidates/deep-packed-varlen-r9101b/libdeep_packed_varlen_r9101b.so';assert hashlib.sha256(p.read_bytes()).hexdigest()=='edf1e808450519cccfdf0f9d7d4a97ac0ef15901cc4ff3261bca9bf20bd4872a';nm=subprocess.check_output(['nm','-D','--defined-only',str(p)],text=True,timeout=30);symbols=[s.split()[-1]for s in nm.splitlines()if s.split()and not s.split()[-1].startswith('_Z')and re.search(r'profile|window|event|arm|read|attn|attention|pipeline',s.split()[-1],re.I)];rec=json.loads((r/'results/deep-packed-varlen-CPU-r9101b.json').read_bytes());q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU':False,'CPU_source_only':True,'library_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'C_exports':symbols,'receipt_keys':list(rec),'attention_sources':[]}
objects=rec.get('all_linked_object_SHA256',rec.get('all_object_SHA256',{}));q['object_count']=len(objects);paths=[]
for obj in objects:
 z=pathlib.Path(str(obj).replace('/refactor/',str(r)+'/'));dp=z.with_suffix('.d')
 if dp.exists():
  text=dp.read_text(errors='replace').replace('\\\n',' ')
  for word in text.split():
   if word.endswith(('.cpp','.cc','.hpp','.h'))and re.search('attention|reference_attn|latent_prefill',word,re.I):paths.append(word)
for command in rec.get('commands',[]):
 for word in command:
  if isinstance(word,str)and word.endswith(('.cpp','.cc'))and re.search('attention|reference_attn',word,re.I):paths.append(word)
for word in sorted(set(paths)):
 z=pathlib.Path(word.replace('/refactor/',str(r)+'/'))
 if z.exists()and z.is_file():q['attention_sources'].append({'path':word,'SHA256':hashlib.sha256(z.read_bytes()).hexdigest(),'bytes':z.stat().st_size,'text':z.read_text(errors='replace')if z.stat().st_size<=120000 else None})
print(json.dumps(q))
