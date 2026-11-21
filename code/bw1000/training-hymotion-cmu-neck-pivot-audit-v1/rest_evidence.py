"""Static, frozen-rig pivot spans. No second candidate or model evaluation."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from audit_pair import fixed, G, L, OUT


def main():
    h=json.loads(fixed(L/'training-hymotion-cmu-asf-fk-candidate-v1-original/hierarchy-and-conventions-private.json',14452,
        'e7374b7a18d16255506febbc6575272beca20803c8abfb8b00f256fd71032aa0'))
    official=L/'hymotion-wooden-official-lfs-4e426f5-original'
    rest=np.frombuffer(fixed(official/'j_template.bin',624,'4f488cc9d0b650a816f5a8eda49eda7bc796f490e9130a6e0dec5be137d7b929'),'<f4').reshape(52,3).astype(float)
    parents=np.frombuffer(fixed(official/'kintree.bin',208,'98a20fa3b53193790b63d9ac3a9c917f2f70fbe6e053dca495c75317ff4b756a'),'<i4')
    mean=np.load(BytesIO(fixed(official/'Mean.npy',1736,'a4bbe5f68881fe793d509f262ad3dc698a91f79158a9bde9d6f24d46ef9652bd')),allow_pickle=False)
    std=np.load(BytesIO(fixed(official/'Std.npy',1736,'6cc86d1ca4b47fbd0e84b677c4397e51b6615abcb34e38fdd9daabbe6a5685cc')),allow_pickle=False)
    bones={b['name'][0]:b for b in h['bones']}
    # This is the already frozen paired-leg scale from bind_rest, not a new fit.
    scale=0.9132775972368371
    def span(name):
        b=bones[name]
        return float(np.linalg.norm(np.asarray(b['direction'],float)*float(b['length'][0])*h['physical_raw_unit_to_meters'])*scale)
    incoming=float(np.linalg.norm(rest[12]-rest[parents[12]]));outgoing=float(np.linalg.norm(rest[15]-rest[12]))
    r={'schema':'FROZEN_CMU_WOODEN_STATIC_NECK_PIVOT_SPANS_V1','additional_candidates':0,
        'Wood_Neck_parent_index':int(parents[12]),'Wood_Head_parent_index':int(parents[15]),
        'current_CMU_Spine3_to_Neck_anchor_span_rig_units':span('lowerneck'),
        'Wood_Spine3_to_Neck_rest_span_rig_units':incoming,
        'current_CMU_Neck_to_Head_anchor_span_rig_units':span('upperneck'),
        'Wood_Neck_to_Head_rest_span_rig_units':outgoing,
        'incoming_source_to_target_length_ratio':span('lowerneck')/incoming,
        'outgoing_source_to_target_length_ratio':span('upperneck')/outgoing,
        'frozen_paired_leg_length_scale':scale,'channel78_Mean_R11':float(mean[78]),'channel78_Std':float(std[78]),
        'interpretation':'Incoming pivot span mismatch is real. This does not establish anatomical re-anchoring or permit changing the frozen positions. No second correction tried.',
        'training_or_GPU':0,'READY_added':0}
    with (OUT/'pivot-rest-evidence.json').open('x',encoding='utf8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(r,sort_keys=True))


if __name__=='__main__':main()
