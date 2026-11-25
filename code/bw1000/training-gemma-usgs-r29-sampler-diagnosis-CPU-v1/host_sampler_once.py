"""Run the R29 diagnostic with the existing authenticated zero-GRES HOST envelope."""
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from types import FunctionType

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
LABEL=HERE.name
CARRIER=REPO/"code/bw1000/training-gemma-usgs-r28-sampler-native-CPU-v1/host_sampler_once.py"
CARRIER_SHA="74774933dd02537b26ff146ca3cae9e60d9b60d4e7a23e70628a8eb62ed225b1"
FILES=("probe.py","probe_r29.py","MODEL_PROCESSOR_FILES.json","SOURCE_FILES.json","prepare.py")
CLAIM=REPO/"coordination/tasks"/f"{LABEL}.json"


def replace(text,old,new):
    if text.count(old)!=1:
        raise ValueError(f"Inherited HOST anchor drifted: {old[:55]}")
    return text.replace(old,new,1)


def remote_template(text):
    text=replace(text,"{'probe.py','SOURCE_FILES.json','command.py'}",repr(tuple(FILES)).replace('(', '{', 1).replace(')', '}', 1))
    original="manifest['files']=={'probe.py': {'bytes': len(sources['probe.py']), 'SHA256': digest(sources['probe.py'])}}"
    update="manifest['files']=={n:{'bytes':len(sources[n]),'SHA256':digest(sources[n])} for n in ('probe.py','probe_r29.py','MODEL_PROCESSOR_FILES.json')}"
    text=replace(text,original,update)
    text=replace(text,"sources['command.py']","sources['prepare.py']")
    text=replace(text,"query(*argv['accounting_argv'](parent)[3:])","query('sacct','-nP','-j',parent,'--format',argv['ACCOUNTING'])")
    text=replace(text,"label=='training-gemma-usgs-r28-sampler-native-CPU-v1'",f"label=={LABEL!r}")
    text=replace(text,"    (run/'out').mkdir()","    (run/'out').mkdir()\n    (run/'scratch').mkdir()")
    text=replace(text,"for name in ('probe.py','SOURCE_FILES.json'):","for name in ('probe.py','probe_r29.py','MODEL_PROCESSOR_FILES.json','SOURCE_FILES.json'):")
    text=replace(text,"argv['runner'](parent,node,scope,now,CLAIM['sources']['SOURCE_FILES.json'])",
        "argv['argv'](parent,node,scope,now,CLAIM['sources']['SOURCE_FILES.json'],CLAIM['allow_component_fallback'])")
    text=replace(text,"len(binds)==11","len(binds)==13")
    start=text.index("def project(result):")
    end=text.index("\ndef main():",start)
    text=text[:start]+"""def project(result):
    whitelist=('route','current_phase','phases','passed','step','actual_swift_physical_rows','actual_gpu',
               'actual_optimizer_updates','actual_training_ranks','training_ready','production_adoption',
               'diagnostic','original_sft_binding_diagnosis','scratch','input_environment')
    return {key:result[key] for key in whitelist if key in result}

"""+text[end:]
    old=text.index("        result={'state':'ORIGINAL_ACCOUNTING_NOT_VISIBLE'")
    end=text.index("        return result\n",old)
    text=text[:old]+"""        result={'state':'ORIGINAL_ACCOUNTING_NOT_VISIBLE','accounting':accounting,'replay_allowed':False}
        read=host['hash_read']
        native=None
        if os.path.lexists(run/'out/RESULT.json'):
            native_raw=read(run/'out/RESULT.json',cap=65536)
            native=json.loads(native_raw)
            need(native['request']==scope and native['node']==node,"SAME_NATIVE_REQUEST_IDENTITY")
            result.update(state='ORIGINAL_DIAGNOSTIC_RECORDED',worker=project(native),worker_SHA256=digest(native_raw))
        if accounting:
            row,=accounting
            if row[2].split()[0] in host['TERMINAL']:
                initial=json.loads(read(run/'HOST_INTENT.json'))
                need(initial['scope_request']==scope and initial['claim_SHA256']==CLAIM_SHA,"SAME_ORIGINAL_HOST_INTENT")
                result['independent']=argv['independent'](row,native,parent,node,scope,initial['epoch'],CLAIM['sources']['SOURCE_FILES.json'])
                result['state']='INDEPENDENT_R29_TERMINAL_SCOPE_OBSERVED'
"""+text[end:]
    compile(text,"<R29 adapted HOST template>","exec")
    return text


def main():
    if sha256(CARRIER.read_bytes()).hexdigest()!=CARRIER_SHA:
        raise ValueError("Published R28 HOST carrier changed")
    original=runpy.run_path(str(CARRIER))
    source=remote_template(original["HOST"])
    env=dict(original,__file__=str(Path(__file__).resolve()),ROOT=REPO, PACKAGE=HERE,LABEL=LABEL,CLAIM=CLAIM,BUNDLE=FILES,HOST=source)
    render=original["render"]
    env["render"]=FunctionType(render.__code__,env)
    if len(sys.argv)==2 and sys.argv[1]=="check":
        claim=json.loads(CLAIM.read_bytes())
        original["pinned"](Path(__file__),claim["host_sampler_SHA256"])
        for name in FILES:
            original["pinned"](HERE/name,claim["sources"][name])
        prep=runpy.run_path(str(HERE/"prepare.py"))
        source_bytes=prep["source_files"]()
        assert sha256(source_bytes["SOURCE_FILES.json"]).hexdigest()==claim["sources"]["SOURCE_FILES.json"]
        args=prep["argv"](claim["parent"],claim["node"],claim["request"],1795590000,
                         claim["sources"]["SOURCE_FILES.json"],claim["allow_component_fallback"])
        assert "--gres=none" in args and "--overlap" not in args and args.count("--bind")==13
        compile(env["render"]("launch","2"*32,"d"*40,claim,"a"*64,original["R27_SHA"]),"<candidate source>","exec")
        print(json.dumps(dict(passed=True,CPU=4,GPU=0,mounts=13,scratch_exclusive=True,command_rendered=True)))
        return
    FunctionType(original["main"].__code__,env)()


if __name__=="__main__":
    main()
