from pathlib import Path
import hashlib,importlib.util,json
D=Path(__file__).resolve().parent;P=D.parent
def check():
 meta=json.loads((D/'PARENT.json').read_bytes());h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert h(P/'CPU_SOURCE_MANIFEST.json')==meta['production_manifest_SHA256']
 for n,v in json.loads((P/'CPU_SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert h(P/n)==v,n
 old=(P/'native/pair_probe.cpp').read_text();new=(D/'native/pair_probe.cpp').read_text()
 assert old.count('struct Context{')==1 and 'bool vocab_embedding_r461='not in old
 assert new==old.replace('struct Context{','struct Context{bool vocab_embedding_r461=false;',1)
 assert new.count('struct Context{bool vocab_embedding_r461=false;')==1
 assert h(D/'native/pair_probe.cpp')==meta['new_pair_SHA256']
 spec=importlib.util.spec_from_file_location('r461_original_source_contract',P/'check_source.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result=module.check();result.pop('private_Context_bool_only',None)
 return dict(passed=True,parent_unchanged_body_contract=result,actual_private_Context_flag_declared=True,
    sole_native_delta='bool vocab_embedding_r461=false inserted in anonymous pair TU Context',
    old_shared_class_headers_unchanged=True,new_DSO_required=True,old_DSO_qualification_inherited=False)
