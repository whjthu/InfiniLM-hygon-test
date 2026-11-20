"""Small new SE(3)/schema boundaries; stdlib only, no dataset or model access."""
import json
from bidirectional import inverse_pose, reverse_pair, expand


def main():
    identity=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]]
    pose=[[0.,-1.,0.,.3],[1.,0.,0.,.2],[0.,0.,1.,.1]]
    raw=dict(images=["rgb0","rgb1"],depths=["d0","d1"],intrinsics=["k0","k1"],
             point_masks=["m0","m1"],ids=[10,14],seq_name="synthetic-only",extrinsics=[identity,pose])
    reversed_pair=reverse_pair(raw)
    assert reversed_pair["ids"]==[14,10] and reversed_pair["images"]==["rgb1","rgb0"]
    assert reversed_pair["extrinsics"]==[identity,[[0.,1.,0.,-.2],[-1.,0.,0.,.3],[0.,0.,1.,-.1]]]
    assert inverse_pose(inverse_pose(pose))==pose
    assert raw["extrinsics"][1] is pose and raw["images"]==["rgb0","rgb1"]
    views,evidence=expand([raw,dict(raw,seq_name="second-synthetic")],[{"passed":True}]*2)
    assert len(views)==4 and [x["direction"] for x in evidence]==["forward","forward","reverse","reverse"]
    cases=[dict(raw,ids=[14,10]),dict(raw,extrinsics=[pose,pose]),
           dict(raw,extrinsics=[identity,[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,-1.,0.]]]),
           dict(raw,extrinsics=[identity,[[1.,0.,0.,float("nan")],[0.,1.,0.,0.],[0.,0.,1.,0.]]]),
           dict(raw,images=["one"])]
    rejected=0
    for bad in cases:
        try:
            reverse_pair(bad)
        except ValueError:
            rejected+=1
    assert rejected==len(cases)
    return dict(passed=True,valid_checks=5,rejected=5,planned_rejections=5,
                actual_data_read=False,actual_Torch_import=False,actual_GPU=0,actual_optimizer_updates=0)


if __name__=="__main__":
    print(json.dumps(main(),sort_keys=True))
