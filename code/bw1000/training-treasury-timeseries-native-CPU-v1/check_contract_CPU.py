"""Pure Python failure boundaries; never import Torch/Swift or load blind labels."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from check_native import bind, tokens_contract, batch_contract


def check(root):
    opened = []
    original = Path.read_bytes
    def audited_read(path):
        if path.name == "labels.jsonl" and path.parent.name in ("dev", "blind"):
            raise AssertionError("held-out gold opened")
        opened.append(path.relative_to(root).as_posix())
        return original(path)
    with patch.object(Path, "read_bytes", audited_read):
        _, manifest, rows, media = bind(root)
    assert len(rows)==3 and all(len(v)==144 for v in rows.values()) and len(media)==48
    assert set(opened) == {"manifest.json", "train/labels.jsonl", *[s+"/inputs.jsonl" for s in rows], *media}
    good = dict(input_ids=[4,99,5,6,7], labels=[-100,-100,-100,6,7])
    assert tokens_contract(good,99,True,[4,99,5]) == [5,1,2]
    assert tokens_contract(dict(input_ids=[4,99,5]),99,False) == [3,1,0]
    failures=[]
    def reject(name, fn):
        try:
            fn()
        except (ValueError, KeyError):
            failures.append(name)
        else:
            raise AssertionError("accepted "+name)
    reject("over4096",lambda:tokens_contract(dict(input_ids=[99]*4097),99,False))
    reject("empty",lambda:tokens_contract(dict(input_ids=[]),99,False))
    reject("noimage",lambda:tokens_contract(dict(input_ids=[1,2]),99,False))
    reject("boolean_token",lambda:tokens_contract(dict(input_ids=[99,True]),99,False))
    reject("dev_supervision",lambda:tokens_contract(good,99,False))
    reject("blind_empty_label_array",lambda:tokens_contract(dict(input_ids=[99],labels=[]),99,False))
    reject("prefix_changed",lambda:tokens_contract(good,99,True,[4,99,8]))
    reject("no_answer",lambda:tokens_contract(dict(input_ids=[4,99,5],labels=[-100]*3),99,True,[4,99,5]))
    for name,index,value in [("user_supervision",0,4),("image_supervision",1,99),
                             ("wrong_target",3,8),("boolean_label",3,True)]:
        bad=copy.deepcopy(good);bad["labels"][index]=value
        reject(name,lambda bad=bad:tokens_contract(bad,99,True,[4,99,5]))
    reject("short_labels",lambda:tokens_contract(dict(input_ids=[4,99,5,6],labels=[-100]*3),99,True,[4,99,5]))
    expected=[[-100,-100,6],[-100,7]]
    batch_contract([[-100,-100,6],[-100,7,-100]],[[1,1,1],[1,1,0]],expected)
    reject("padding_supervised",lambda:batch_contract([[-100,-100,6],[-100,7,0]],[[1,1,1],[1,1,0]],expected))
    reject("left_padding",lambda:batch_contract([[-100,-100,6],[-100,-100,7]],[[1,1,1],[0,1,1]],expected))
    reject("collator_target_changed",lambda:batch_contract([[-100,-100,6],[-100,8,-100]],[[1,1,1],[1,1,0]],expected))
    return dict(state="LOCAL_STDLIB_CONTRACT_PASS",actual_native_processor_calls=0,actual_GPU_cards=0,
        actual_updates=0,manifest_SHA256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
        input_rows={s:len(v) for s,v in rows.items()},media_SHA_verified=len(media),
        opened_file_count=len(opened),heldout_label_files_opened=False,
        label_byte_SHA_verified={"train":True,"dev":False,"blind":False},
        valid_stub_cases=3,rejected_cases=failures,rejected_case_count=len(failures),
        target_native_qualification=False,semantic_readability="UNKNOWN")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=check(args.root.resolve(strict=True))
    with args.output.open("x",encoding="utf8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result["state"],rejected_cases=result["rejected_case_count"],
                         native_processor_calls=0,heldout_label_files_opened=False)))
