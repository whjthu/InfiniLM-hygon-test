"""Observe the actual failed metadata predicate without changing its assertions."""
from pathlib import Path
import ast
D=Path(__file__).resolve().parent
s=(D/'read_original_terminal.py').read_text(encoding='utf8').replace('root-reader','root-reader-v3')
needle="model=Path(out['model_path']);s=model.stat();current=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,mode=s.st_mode);assert current==out['after_stat'] and not model.is_symlink() and sorted(p.name for p in model.parent.iterdir())==['model.pt']"
replacement="""model=Path(out['model_path']);s=model.stat();current=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,mode=s.st_mode)
print(json.dumps(dict(stage='POST_FILE_CONTEXT',current_host=socket.gethostname(),hash_owner_host=out['owner']['host'],recorded_stat=out['after_stat'],current_stat=current,differences={k:[out['after_stat'][k],x] for k,x in current.items() if out['after_stat'][k]!=x},symlink=model.is_symlink(),directory_names=sorted(p.name for p in model.parent.iterdir()))),flush=True)
assert current==out['after_stat'] and not model.is_symlink() and sorted(p.name for p in model.parent.iterdir())==['model.pt']"""
assert needle in s;s=s.replace(needle,replacement)
needle="if r.returncode==0:out['value']=json.loads(r.stdout)"
replacement="""contexts=[]
for line in r.stdout.splitlines():
 try:doc=json.loads(line)
 except ValueError:continue
 if doc.get('stage')=='POST_FILE_CONTEXT':contexts.append(doc)
 elif r.returncode==0 and doc.get('passed') and 'original_outcome' in doc:out['value']=doc
out['post_file_contexts']=contexts"""
assert needle in s;s=s.replace(needle,replacement)
s=s.replace("v=out.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,independent_terminal_reader_passed=v.get('passed'),original_hash_owner_identity_present=v.get('original_hash_owner_identity_present'),independent_full_target_read_verified=v.get('independent_full_target_read_verified'),weight_file_bytes_read_by_reader=0)))","v=out.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),post_file_contexts=contexts,weight_file_bytes_read_by_reader=0)))")
ast.parse(s);assert not (D/'read_original_terminal_v3.py').exists();(D/'read_original_terminal_v3.py').write_bytes(s.encode())
