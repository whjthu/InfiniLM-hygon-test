"""Optional CPU-only check of the real installed stock sampler classes and 41 input identities."""

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path

from evidence import ROWS, WORLD, dataset_rows, need, pin, read, schedule_position


def check(dataset, inventory):
    import torch
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler, RandomSampler

    ids=dataset_rows(dataset,inventory)
    sources={cls.__module__+"."+cls.__name__:sha256(read(inspect.getfile(cls),1024**2)).hexdigest()
             for cls in (BatchSamplerShard,BatchSampler,RandomSampler)}
    per_epoch=[]
    for epoch in range(2):
        epoch_slots={}
        for rank in range(WORLD):
            generator=torch.Generator(device="cpu").manual_seed(42+epoch)
            sampler=RandomSampler(ids,replacement=False,generator=generator)
            shard=BatchSamplerShard(BatchSampler(sampler,batch_size=1,drop_last=False),
                                    num_processes=WORLD,process_index=rank,split_batches=False,even_batches=True)
            for offset,batch in enumerate(shard):
                position=schedule_position(rank,epoch*len(shard)+offset)
                need(len(batch)==1 and position["epoch"]==epoch,"EXACT_NATIVE_CLASS_FIXTURE_BATCH")
                epoch_slots[position["global_slot_in_epoch"]]=ids[batch[0]]
        sequence=[epoch_slots[i] for i in range(len(epoch_slots))]
        need(len(sequence)==48 and len(set(sequence[:ROWS]))==ROWS and sequence[ROWS:]==sequence[:7],
             "EXACT_INSTALLED_STOCK_PAD_AND_EPOCH_BOUNDARY")
        per_epoch.append({"epoch":epoch,"actual_CPU_sampler_slots":48,"unique_original":41,"padding":7,
                          "sequence_identity_SHA256":pin(sequence)})
    return {"state":"INSTALLED_CPU_SAMPLER_CLASSES_ON_41_ORIGINAL_ROW_IDENTITIES_CHECKED",
            "qualified_native_Trainer_or_GPU_updates":0,"new_GPU":0,"provider_source_SHA256":sources,"epochs":per_epoch}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--inventory",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(check(args.dataset,args.inventory),sort_keys=True))
