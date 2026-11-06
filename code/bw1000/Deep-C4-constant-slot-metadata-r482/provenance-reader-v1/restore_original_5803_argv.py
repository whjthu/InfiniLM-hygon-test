"""Restore5803 earliest original TU recipe for20 reused5920 objects, no build.
Read existing private metadata only; not a fresh50 preprocessing/ABI closure.
"""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,shlex
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(local):
 L=Path(local);wrapper=L/'GLM472-watch-and-Deep5803-original-read-20261106-0111.json';v=json.loads(json.loads(wrapper.read_bytes())['out'])['Deep5803']
 a=json.loads((L/'Deep482-original35-source-object-provenance-associations-v1.json').read_bytes())['objects'];cpu=json.loads((L/'Deep377-CPU-v1-result-local.json').read_bytes())['receipt']['all50_objects_SHA256']
 assert H(L/'build_deep_csa_r5803.py')==v['producer_script']['SHA256']=='ee2df69be7ce56104bc9ba1d0c00e887c6f69e30bebd67a5d9e859e8b01ccd13'
 assert v['compile_SHA256']==v['report']['sources']['compile.sh']=='cdd42ada8457321dd12692221fe2077f288a8e8381ca540b9d71592973bcee8f' and v['report']['completed'] and v['report']['returncode']==0
 env={'BASE':'/refactor/candidates/deep-csa-prefix-r5803','COMMON':'/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM','LM':'/refactor/candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM','SDK':'/refactor/install/hygon-deep-5f91ad7-r3672','HELPER':'/refactor/candidates/models-5f91ad7-r3675/Deep/helpers','ABI':'1'}
 argv=shlex.split(v['compile_text'].replace('\\\n',' '))
 for i,x in enumerate(argv):
  for k,y in env.items():x=x.replace('${'+k+'}',y).replace('$'+k,y)
  assert '$'not in x;argv[i]=x
 sources=[x for x in argv if x.endswith('.cpp')];oi=argv.index('-o');flags=[x for i,x in enumerate(argv)if i not in[oi,oi+1]and not x.endswith(('.cpp','.o'))and x!='-shared'];rows=[]
 for obj,x in a.items():
  if x.get('reused')and x.get('predecessor','').startswith(env['BASE']+'/unit-'):
   old=x['predecessor'];i=int(old.rsplit('unit-',1)[1].split('.')[0]);src=sources[i];rel=src.removeprefix(env['BASE']+'/');oh=v['report']['sources'][old.removeprefix(env['BASE']+'/')];assert oh==cpu[obj]
   private=src.startswith(env['BASE']+'/')
   if private:assert v['report']['sources'][rel]==x['source_SHA256']
   else:assert src==x['source'],'External source origin must remain literal, not rewritten5920private path'
   rows.append(dict(object=obj,original_object=old,original_object_SHA256=oh,original_source=src,source_SHA256_from5920original=x['source_SHA256'],private_source5803_manifest_match=private,original_recipe_argv_derived=flags+['-c',src,'-o',old],quote_origin=str(PurePosixPath(src).parent),original_argv_recorded_as_stream=False))
 assert len(rows)==20 and sum(x['private_source5803_manifest_match']for x in rows)==5
 return dict(schema='Deep482-5803-predecessor-recipe-v1',scope='Read original5803 compile.sh/report and SHA-identical producer source; restore original quoted source directories for20 copiedobjects. No compiler/preprocessor/SDK/Slurm/GPU.',original_evidence=dict(path=str(wrapper),SHA256=H(wrapper)),original5803_report={k:x for k,x in v['report'].items()if k!='sources'},producer_script=v['producer_script'],compile_script_SHA256=v['compile_SHA256'],ABI='1 from qualified parent producer environment; no independently preserved5803 ambientenv/processargv snapshot',restored_recipe_objects=20,private_source_manifest_SHA_matches=5,external_sources_with5920_original_SHA=15,objects_match_original_CPU377=20,rows=rows,all50_fresh_preprocess_closed=False,CPU_GO=False,GPU_GO=False,next='Use original5803 argv/source directories for reused20, not rewritten5920recipe. Assemble all50 exactsource/compiler/env/quoteorigin/dependencyclaim, then one bounded fullfresh-M reverseclasssize closure before anybuild. MissingMD never meansclean; keep allaffectedTUs. No original50inventory/producer reread.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('local');p.add_argument('output');a=p.parse_args();o=Path(a.output);assert not o.exists();r=read(a.local);r['reader_SHA256']=H(__file__);o.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ['restored_recipe_objects','private_source_manifest_SHA_matches','external_sources_with5920_original_SHA','objects_match_original_CPU377','all50_fresh_preprocess_closed','CPU_GO','GPU_GO']}))
