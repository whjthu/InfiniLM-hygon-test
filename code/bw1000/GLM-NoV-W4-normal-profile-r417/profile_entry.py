"""Private explicit service entry, preserving the qualified single-context loader."""
from pathlib import Path
import argparse,ast,hashlib,os,sys
D=Path(__file__).resolve().parent
P=D.parent/'GLM-NoV-16card-W4-pair-r414/owner'
sys.path.insert(0,str(P));sys.path.insert(0,str(D))
from service_policy import configure,verify_sources
def loader_tree(parent):
    raw=(Path(parent)/'model_entry_r290.py').read_bytes();tree=ast.parse(raw)
    replaced=0;benchmark=0
    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom)and n.module=='owner_r290':
            assert len(n.names)==1 and n.names[0].name=='run'and n.names[0].asname=='_preloaded_probe_r3091'
            n.module='service_owner';replaced+=1
        if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='benchmark'for t in n.targets):
            assert isinstance(n.value,ast.Dict)and [k.value for k in n.value.keys]==['generation_config','prefill_pair_r240']
            n.value.keys=n.value.keys[:1];n.value.values=n.value.values[:1];benchmark+=1
    assert(replaced,benchmark)==(1,1)
    ast.fix_missing_locations(tree)
    return tree,dict(parent_model_entry_SHA256=hashlib.sha256(raw).hexdigest(),owner_import_replacements=replaced,fixture_payload_removed=benchmark,loader_create_load_finish_destroy_changed=False)
def run_entry():
    p=argparse.ArgumentParser();p.add_argument('--profile',required=True,choices=['glm-nov16-w2','glm-nov16-w4']);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    m=verify_sources(D,P);configure(a.profile,os.environ)
    # Preserve parent source identity for its actual loader checks. The new
    # service policy/source identity is separately bound in ready/owner reports.
    from service_policy import PARENT_OWNER_SHA256
    if os.environ.get('GLM_SOURCE_MANIFEST_SHA',PARENT_OWNER_SHA256)!=PARENT_OWNER_SHA256:raise ValueError('Conflicting loader identity')
    os.environ['GLM_SOURCE_MANIFEST_SHA']=PARENT_OWNER_SHA256
    os.environ['GLM_R417_SOURCE_SHA256']=hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()
    tree,proof=loader_tree(P)
    if a.preflight_only:
        import json
        print(json.dumps(dict(preflight_passed=True,profile=a.profile,source_SHA256=os.environ['GLM_R417_SOURCE_SHA256'],proof=proof,GPU=0,Native_dlopen=False,Model=0)));return
    # Running in this module preserves torch multiprocessing's __mp_main__
    # re-import path. No global module monkeypatch or second create is used.
    exec(compile(tree,str(P/'model_entry_r290.py'),'exec'),globals())
if __name__ in ('__main__','__mp_main__'):run_entry()
