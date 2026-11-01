from pathlib import Path
import ast,hashlib,json,textwrap
D=Path(__file__).resolve().parent;P=D.parent/'GLM-W4-cross-workload-r349/CPU_whole_glue.py';R=D/'CPU-repair-r2';R.mkdir(exist_ok=False)
source=P.read_text();start=source.index(' def guard(name):');end=source.index(' torch.cuda.is_available=',start)
guard=textwrap.indent(textwrap.dedent(source[start:end]),'    ')
old="    def forbidden(*a,**k):raise RuntimeError('GPU initialization forbidden')\n    for n in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,n,forbidden)\n"
cpu=(D/'CPU_whole.py').read_text();assert old in cpu
cpu=cpu.replace(old,guard).replace("HERE=pathlib.Path(__file__).resolve().parent;D=HERE/'owner'","HERE=pathlib.Path('/refactor/incoming/GLM-two-cohort-Model-r355');D=HERE/'owner'")
ast.parse(cpu);(R/'CPU_whole.py').write_text(cpu,encoding='utf8',newline='\n')
inside=(D/'CPU_inside.sh').read_text().replace('GLM-two-cohort-Model-r355/CPU_whole.py','GLM-two-cohort-Model-r355/CPU-repair-r2/CPU_whole.py')
(R/'CPU_inside.sh').write_text(inside,encoding='utf8',newline='\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=dict(GPU=0,repair='Only CPU forbidden-hook identity: exact successful349 per-name closure block; no GPU owner/controller changes',parent_CPU_source_SHA256=sha(P),owner_manifest_SHA256=sha(D/'owner/SOURCE_MANIFEST.json'),controller_manifest_SHA256=sha(D/'controller/SOURCE_MANIFEST.json'),files_SHA256={p.name:sha(p)for p in R.iterdir()if p.is_file()})
(R/'SOURCE_MANIFEST.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps(q))
