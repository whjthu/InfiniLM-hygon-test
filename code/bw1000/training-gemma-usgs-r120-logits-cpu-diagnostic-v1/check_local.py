"""Use the three archived provider ASTs with synthetic tensors and test label alignment."""

from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import alignment as a
import probe


def check():
    raw=(HERE.parents[2]/"local-only/gemma-r120-frozen-swift-three-callables-ast-only-v1.private.json").read_bytes()
    assert sha256(raw).hexdigest()==probe.AST_EVIDENCE
    observed=json.loads(raw)
    non_parallel=True
    def pad(arr,padding,value):
        dims=((0,0),padding) if arr.ndim==2 else padding
        return np.pad(arr,dims,constant_values=value)
    globals_={"inspect":inspect,"transformers_5":True,"logger":SimpleNamespace(info_once=lambda _:None),
              "nn":SimpleNamespace(functional=SimpleNamespace(pad=pad)),"is_mp":lambda: not non_parallel}
    methods=observed["trainers/mixin.py"]["functions"] | observed["trainers/seq2seq_trainer.py"]["functions"]
    text="class Parent:\n    def _prepare_inputs(self, inputs): return inputs\nclass Child(Parent):\n"
    for m in methods.values():
        text+="".join("    "+line+"\n" for line in m["ast_source"].splitlines())
    exec(compile(text,"<pinned real provider AST with synthetic tensors>","exec"),globals_)
    trainer=globals_["Child"]()
    fwd=lambda logits_to_keep:None
    trainer.model=SimpleNamespace(forward=fwd,model_meta=SimpleNamespace(is_multimodal=True),
                                  model_info=SimpleNamespace(is_moe_model=False))
    trainer.template=SimpleNamespace(sequence_parallel_size=1,get_base_model=lambda x:x)
    trainer.compute_loss_func=None
    full=[-100,-100,12,-100,12,-100,15]
    for option in probe.MODES:
        trainer.args=SimpleNamespace(tuner_backend="peft",use_logits_to_keep=option)
        batch=trainer._prepare_inputs({"labels":np.array([full],dtype=np.int64)})
        observed=batch.get("logits_to_keep",0)
        positions=a.positions(observed.tolist() if hasattr(observed,"tolist") else observed,len(full))
        labels=batch["labels"][0].tolist()
        mode="no_compression" if option is False else "native_mask"
        assert a.alignment(full,labels,positions,mode=mode)["alignment_passed"]
        if mode=="native_mask":
            assert (positions,labels)==([1,3,5,6],[-100,12,12,15])
            assert a.shifted_one_is_rejected(full,labels,positions,mode=mode)
        assert trainer.args.use_logits_to_keep is (option is not False)
    def reject(fn,code):
        try: fn()
        except ValueError as error: assert str(error)==code
        else: raise AssertionError("expected rejection")
    reject(lambda:a.alignment(full,[-100,12,12,15],[2,4,5,6],mode="native_mask"),"SUPERVISED_CAUSAL_POSITION_LOSS")
    reject(lambda:a.alignment(full,[-100,12,99,15],[1,3,5,6],mode="native_mask"),"PROJECTED_CAUSAL_LABEL_IDENTITY")
    reject(lambda:a.positions([1,1,6],len(full)),"DUPLICATE_OR_UNORDERED_NATIVE_INDEX")
    reject(lambda:probe.source_bind("0"*64),"ORIGINAL_4177_EFFECTIVE_MANIFEST_PIN_ONLY")
    print("PASS original Swift callable AST, full/compact label reference, repeated labels and in-range off-by-one")


if __name__=="__main__":
    check()
