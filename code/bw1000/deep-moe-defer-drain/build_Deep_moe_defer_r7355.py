"""Private one-TU incremental closure only; CPU step, no GPU DSO dlopen."""
import pathlib,json,hashlib,shlex,subprocess,time
r=pathlib.Path('/refactor');b=r/'candidates/deep-moe-defer-r7355';parent=r/'candidates/deep-tailSUM-r7212'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
receipt=r/'results/deep-tailSUM-r7212-build.json';old=json.loads(receipt.read_text())
assert sha(old['libraries']['full']['path'])==old['libraries']['full']['SHA256']=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'
unit=next(u for u in old['compiled_units']if u['source'].endswith('/deepseek_v4_moe_prefill_scatter_r194.cpp'))
assert sha(unit['source'])==unit['source_SHA256']=='70dc21ddf87b861120b00522f47c682e7ae3ee6ea42ee0b2593b4ad50b842672'
assert sha(unit['object'])==unit['object_SHA256']=='c888bde05aaa89e311dd71f1cf11869e2b4a2caa8c3dfde174394407f661622f'
spec=json.loads((r/'incoming/Deep-moe-defer-drain-source-r7353.json').read_text())
assert not b.exists();(b/'src').mkdir(parents=True)
private={}
for n in ['deep_moe_defer_drain_r7353.cpp','deep_moe_defer_drain_r7353.hpp']:
 p=r/'incoming'/n;assert sha(p)==spec['files_SHA256'][n];dst=b/'src'/n;dst.write_bytes(p.read_bytes());private[str(dst)]=sha(dst)
argv=old['link']['full']['argv'];oi=argv.index('-o')
flags=['g++','-I'+str(b/'src'),'-I'+str(parent/'incoming'),'-I/opt/dtk/cuda/cuda/include']+[x for i,x in enumerate(argv)if i not in[0,oi,oi+1]and not x.endswith('.o')and x!='-shared'and not x.startswith(('-l','-L','-Wl,'))]
assert any('/opt/dtk/cuda/cuda/include' in x for x in flags),'closed DTK include must remain present'
src=b/'src/deep_moe_defer_drain_r7353.cpp';obj=b/'moe.o';md=b/'moe.d';fm=b/'moe.M';parse=lambda p:list(dict.fromkeys(shlex.split(p.read_text().replace('\\\n',' '))[1:]))
started=time.monotonic();cmd=flags+['-M','-MF',str(fm),'-MT',str(obj),str(src)];p=subprocess.run(cmd,capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr
deps=parse(fm);before={n:sha(n)for n in deps}
for n,h in before.items():
 if n in private:assert h==private[n]
 else:assert n in unit['deps'] and h==unit['deps'][n],('unexpected_dependency',n,h)
for suffix in ['deep_layout_core_r7032.hpp','deep_collective_layout_r7040.hpp','deep_reduce_wide_r3443.hpp','deepseek_routed_mlp_batch_r20.hpp','runtime/runtime.hpp','execution/submit.hpp']:
 matches=[n for n in deps if n.endswith(suffix)];assert len(matches)==1,(suffix,matches)
 if not suffix.startswith(('runtime/','execution/')):assert matches[0].startswith(str(parent/'incoming')+'/'),matches[0]
command=flags+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)];p=subprocess.run(command,capture_output=True,text=True,timeout=180);(b/'compile.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
assert set(parse(md))==set(deps)
for n,h in before.items():assert sha(n)==h
libraries={};links={}
for which in ['partial','full']:
 a=old['link'][which]['argv'];assert a.count(unit['object'])==1
 for n,h in old['link'][which]['objects'].items():assert sha(n)==h
 lib=b/f'libdeep_moe_defer_{which}_r7355.so';link=[str(obj)if n==unit['object']else n for n in a];link[link.index('-o')+1]=str(lib)
 # Preserve full parent closure and require all dynamic references resolve.
 link.insert(1,'-Wl,-z,defs');p=subprocess.run(link,capture_output=True,text=True,timeout=90);(b/f'link-{which}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
 nm=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True);base_nm=subprocess.check_output(['nm','-D','--defined-only',old['libraries'][which]['path']],text=True)
 names=lambda s:{v.split()[-1]for v in s.splitlines()if v.split()};assert names(base_nm)<=names(nm)
 assert 'deep_moe_defer_calls_r7353'in names(nm)
 dynamic=subprocess.check_output(['readelf','-d',str(lib)],text=True);assert 'SYMBOLIC'in dynamic
 libraries[which]={'path':str(lib),'SHA256':sha(lib)};links[which]={'argv':link,'objects':{n:sha(n)for n in link if n.endswith('.o')},'zdefs_passed':True,'added_exports':sorted(names(nm)-names(base_nm)),'dynamic':dynamic}
for n,v in old['actual_DTK_bindings'].items():assert sha(n)==v['SHA256']
out={'completed':True,'producer':__import__('os').environ['SLURM_JOB_ID']+'.'+__import__('os').environ['SLURM_STEP_ID'],'scope':'CPU1TUprivate incremental compile+2links; no dlopen/GPU/weights','parent_receipt_SHA256':sha(receipt),'parent_libraries':old['libraries'],'source_manifest_SHA256':sha(r/'incoming/Deep-moe-defer-drain-source-r7353.json'),'compiled_unit':{'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'command':command,'deps':before,'freshM_equals_MD':True},'libraries':libraries,'link':links,'actual_DTK_bindings':old['actual_DTK_bindings'],'CPU_seconds':time.monotonic()-started}
p=b/'CPU-build.json';p.write_text(json.dumps(out,indent=2));print(json.dumps({'completed':True,'libraries':libraries,'receipt':str(p),'receipt_SHA256':sha(p)}))
