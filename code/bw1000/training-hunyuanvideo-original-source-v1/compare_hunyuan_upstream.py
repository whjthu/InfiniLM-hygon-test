from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).resolve().parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
record=G/'results/bw1000/2026-11-13/training-hunyuanvideo-sft-A100-native-package-file-identity.json'
source=json.loads(record.read_bytes())['package']
git=['git','--git-dir',str(B/'hunyuan-upstream-60783e7.git')]
rev=source['locked_revision']
assert subprocess.check_output(git+['rev-parse','FETCH_HEAD'],text=True).strip()==rev
tree=subprocess.check_output(git+['ls-tree','-r','-z',rev,'--','hyvideo','requirements.txt','setup.py','setup.cfg','pyproject.toml'])
entries={}
for line in tree.split(b'\0'):
    if not line:continue
    info,name=line.decode().split('\t',1)
    mode,kind,oid=info.split()
    assert kind=='blob' and mode in ('100644','100755')
    raw=subprocess.check_output(git+['cat-file','blob',oid])
    entries[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),mode=mode)
rows={}
for name in sorted(set(entries)|set(source['files'])):
    upstream=entries.get(name)
    deployed=source['files'].get(name)
    if upstream is None:kind='deployment_only'
    elif deployed is None:kind='upstream_only'
    elif upstream['SHA256']==deployed['SHA256'] and upstream['bytes']==deployed['bytes']:kind='bytes_match'
    else:kind='bytes_differ'
    rows[name]=dict(classification=kind,upstream=upstream,A100=deployed)
counts={kind:sum(row['classification']==kind for row in rows.values()) for kind in ('bytes_match','bytes_differ','deployment_only','upstream_only')}
result=dict(upstream_commit=rev,upstream_tree=subprocess.check_output(git+['rev-parse',rev+'^{tree}'],text=True).strip(),
    package_record_SHA256=hashlib.sha256(record.read_bytes()).hexdigest(),counts=counts,files=rows)
(B/'hunyuan-upstream-comparison-v1.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(counts=counts,nonmatching=[n for n,v in rows.items() if v['classification']!='bytes_match'])))
