"""Run the real answerless input contract and mock the scheduler and native result edges."""
import ast
import json
import os
from pathlib import Path
import tempfile

from contract import ARMS, CAP, CP_ADAPTER, CP_FILES, CP_INVENTORY_SHA, CP_RELOAD_SHA, HEX, PROMPT_SHA, RELOAD_ROW, SLOTS, actual_reload, archive, encode, generation, inputs, seal
from host import HOME_SOURCE, PROVIDER_SOURCE, build_command, parent_clear

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REQUEST = "2cf35d112417459c97b8fd5296f4fdfa"
BASE = REPO/"local-only/minicpm-off-independent-s3-cohort-20261127"


def reject(action, code):
    try:
        action()
    except ValueError as error:
        assert str(error) == code
    else:
        raise AssertionError(code)


def query_fixture(*argv, occupied=None):
    if argv[:3] == ("scontrol","show","job"):
        return [f"JobId=999 JobState=RUNNING UserId=test({os.getuid()}) NodeList=f1r1n2"]
    if argv[:3] == ("scontrol","show","hostnames"):
        return [argv[3]]
    if argv[:3] == ("scontrol","show","step"):
        return ["StepId=999.12 NodeList=f1r1n2"] if occupied=="control" else []
    if argv[:2] == ("squeue","--steps"):
        return ["999.12|f1r1n2"] if occupied=="queue" else []
    return ["999|RUNNING|f1r1n2"]


def main():
    # Scope the real read to the answerless prompt file and referenced JPEG names.
    prompts = inputs(BASE, strict_tree=False)
    assert len(prompts)==14 and len({r["image"] for r in prompts})==11
    reject(lambda: inputs(BASE, strict_tree=True), "ANSWERLESS_EXACT_DIRECTORY")
    original = (REPO/"results/bw1000/2026-11-27/specialized-minicpm-OFF-4484-two-native-AdAMW-updates-and-4486-CP2.json").read_bytes()
    pins = archive(original)
    assert pins==CP_ADAPTER and HEX.fullmatch(CP_INVENTORY_SHA) and HEX.fullmatch(CP_RELOAD_SHA)
    assert all(HEX.fullmatch(p["SHA256"]) for p in CP_FILES.values())
    local_reload = REPO/"local-only/minicpm-two-cp2-reload-9ce5f64ab2c248689c239c9b6253b32b.json"
    envelope = json.loads(local_reload.read_bytes())["response"]
    recorded = json.dumps(envelope, sort_keys=True).encode()
    from contract import sha
    assert sha(recorded) == CP_RELOAD_SHA
    assert sha((REPO/"code/bw1000/training-domain-pilot-v1/internvl_gqa_miopen_home.py").read_bytes()) == HOME_SOURCE
    assert sha((REPO/"code/bw1000/training-minicpm-usgs-native88-CPU-v1/providers.py").read_bytes()) == PROVIDER_SOURCE
    assert actual_reload(recorded, [RELOAD_ROW])["adapter"]==CP_ADAPTER
    reject(lambda: actual_reload(encode(dict(envelope, result=dict(envelope["result"], actual_GPU=1))),
                                         [RELOAD_ROW]),"ACTUAL_INDEPENDENT_CP2_ZERO_GRES_RELOAD")
    assert generation([1,2,3,0], [3], 0)==("eos",3) and generation([2]*CAP,[3],0)==("cap",CAP)
    reject(lambda: generation([1,2],[3],0), "NATIVE_GENERATION_INCOMPLETE")
    reject(lambda: generation([1,3,2],[3],0), "POST_EOS_NONPAD")
    parent_clear("999","f1r1n2",query_fixture)
    reject(lambda: parent_clear("999","f1r1n2",lambda *a: query_fixture(*a,occupied="control")),"NUMBERED_CONTROLLER_STEP")
    reject(lambda: parent_clear("999","f1r1n2",lambda *a: query_fixture(*a,occupied="queue")),"NUMBERED_QUEUE_STEP")
    args = build_command(Path("/target"),Path("/work"),REQUEST,"a"*64,"b"*64,"999","f1r1n2")
    assert args.count("--gres=dcu:1")==1 and "--mem=64000M" in args and "--no-mount" in args and "home,cwd" in args
    assert sum(x.endswith(":rw") for x in args)==2 and sum(x.endswith(":ro") for x in args)==11
    assert "gold" not in " ".join(args) and "prompts.answerless.jsonl" not in " ".join(args)
    witnesses = {arm: {slot:{"SHA256":"c"*64,"tokens":128} for slot in SLOTS} for arm in ARMS}
    rows = {arm:{slot:dict(slot=slot,answer="placeholder",finish_reason="eos") for slot in SLOTS} for arm in ARMS}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        result = seal(folder,REQUEST,rows,witnesses,None,{"actual_live_adapter":CP_ADAPTER,"model_loads":1,"actual_native_calls":28})
        assert result["passed"] and result["planned"]==28 and result["counts"]=={"base":{"eos":14},"adapter":{"eos":14}}
        assert all(set(json.loads(x))=={"slot","answer","finish_reason"} for x in (folder/"base.jsonl").read_text().splitlines())
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        rows["adapter"] = {SLOTS[0]:dict(slot=SLOTS[0],answer=None,finish_reason="error")}
        result=seal(folder,REQUEST,rows,witnesses,{"phase":"fixture"},{})
        assert not result["passed"] and result["counts"]["adapter"]=={"error":1,"missing":13}
        assert len(json.loads((folder/"ledger.private.json").read_bytes()))==28
    for p in HERE.glob("*.py"):
        ast.parse(p.read_bytes(),filename=p.name,feature_version=(3,10))
    print(json.dumps(dict(state="PASS_LOCAL_INPUTS_AND_BOUNDARIES",question_rows=14,JPEGs=11,planned=28,
                         prompt_SHA256=PROMPT_SHA,actual_model_calls=0,actual_GPU=0,gold_read=False),sort_keys=True))


if __name__=="__main__":
    if os.name=="nt":
        os.getuid=lambda:1000
    main()
