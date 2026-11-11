import json
from pathlib import Path
import torch
from hygon_trellis_compare_first_step import FIELDS, compare, verify_receipt

root=Path("/reference")
reference=torch.load(root/"reference.pt",map_location="cpu",weights_only=True)
policy=json.loads((root/"policy.json").read_bytes())
actual={name:reference[name] for name in FIELDS}
passed=compare(torch,actual,reference,policy)
assert passed["passed"] and len(passed["checks"])==12
rejected=[]
def numeric(label,field,value,check):
    result=compare(torch,{**actual,field:value},reference,policy)
    assert not result["passed"] and not result["checks"][check]
    rejected.append(label)
def malformed(label,field,value,reason):
    try:
        compare(torch,{**actual,field:value},reference,policy)
    except ValueError as error:
        assert reason in str(error)
        rejected.append(label)
    else:
        raise AssertionError(label)
numeric("wrong_loss","loss",actual["loss"]*1.02,"loss")
numeric("zero_update","final_adapter",actual["initial_adapter"],"nonzero_gradient_and_update")
numeric("wrong_gradient_direction","gradients_before_clip",{n:-x for n,x in actual["gradients_before_clip"].items()},"gradient_direction")
numeric("wrong_adamw_moment","final_optimizer",{**actual["final_optimizer"],"state":{
    i:{**values,"exp_avg":values["exp_avg"]*1.10} for i,values in actual["final_optimizer"]["state"].items()}},
    "optimizer_exp_avg")
malformed("wrong_sparse_coordinate_order","prediction_coords",actual["prediction_coords"].flip(0),"prediction_coords")
malformed("nonfinite_output","prediction",actual["prediction"]*float("nan"),"nonfinite")
malformed("changed_input_target","inputs",{**actual["inputs"],"target":actual["inputs"]["target"]+1},"inputs/target")
try:
    verify_receipt({},"a"*64,"b"*64,{})
except ValueError as error:
    assert "frozen binding" in str(error)
    rejected.append("missing_gpu_receipt")
else:
    raise AssertionError("missing receipt accepted")
print(json.dumps({"scope":"CPU comparator qualification against original A100 bytes","checks":passed["checks"],"negative_cases":rejected}))
