"""Local mapping and packet fixtures. Target Swift and Arrow are absent on Windows."""
import json
from unittest.mock import patch
from copy import deepcopy
import multiprocessing as mp
import os
from types import SimpleNamespace
import unittest

import lineage as q
import native_route
import prepare


class Column:
    def __init__(self,values):
        self.values=values

    def __getitem__(self,i):
        return SimpleNamespace(as_py=lambda:self.values[i])

    def to_pylist(self):
        return self.values


class Schema:
    def __init__(self,types,metadata=None):
        self.types=types
        self.metadata=metadata

    def equals(self,other,check_metadata=False):
        return self.types==other.types and (not check_metadata or self.metadata==other.metadata)

    def field(self,key):
        return SimpleNamespace(type=self.types[key])


class Table:
    def __init__(self,cols,*,schema=None):
        self.cols=cols
        self.column_names=list(cols)
        self.num_rows=len(next(iter(cols.values())))
        self.schema=schema or Schema({name:"string" if name=="dataset" else "fixture" for name in cols})

    def column(self,name):
        return Column(self.cols[name])

    def equals(self,other,check_metadata=False):
        return self.cols==other.cols and self.schema.equals(other.schema,check_metadata=check_metadata)


class Dataset:
    def __init__(self,table,order=None):
        self._data=SimpleNamespace(table=table)
        self._indices=SimpleNamespace(table=Table(dict(indices=order))) if order is not None else None

    def __len__(self):
        return self._data.table.num_rows

    def select_columns(self,columns):
        assert columns==["messages"]
        return self

    def with_format(self,fmt):
        assert fmt is None
        return self

    def __getitem__(self,i):
        if self._indices:
            i=self._indices.table.cols["indices"][i]
        raw=self._data.table.cols["messages"][i]
        return dict(messages=q.shape_messages(raw))


ROUTE="/frozen/original/proposed-exposures64.jsonl"


def metadata_process(expected,barrier,records):
    barrier.wait(timeout=10)
    process=mp.current_process()
    rank=int(process.name.rsplit("-",1)[-1])
    fixture_dist=SimpleNamespace(is_initialized=lambda:True,get_world_size=lambda:8,get_rank=lambda:rank)
    def absent_hf_state(_):
        raise AttributeError("distributed_state")
    parsed=type("Primer",(),{"process_index":property(absent_hf_state)})()
    parsed.dataloader_drop_last=False
    parsed.train_dataloader_shuffle=True
    parsed.data_seed=42
    parsed.deepspeed=None
    parsed.group_by_length=False
    args=native_route.MetadataTraversalArguments(parsed,fixture_dist,expected)
    records.put((os.getpid(),args.process_index,args.dataloader_num_workers,args.data_seed,args.train_dataloader_shuffle,
                 args.dataloader_drop_last,args.group_by_length,args.deepspeed))


class Fixture(unittest.TestCase):
    def fixture(self):
        source=list(range(1,57))+list(range(1,9))
        rows=[dict(messages=[dict(role="user",content=f"synthetic-{index}"),
                             dict(role="assistant",content="synthetic")],
                   images=[f"/train/{index}.jpg"]) for index in source]
        lines=[q.encode(row).rstrip(b"\n") for row in rows]
        proof={index:q.digest(raw) for index,raw in zip(source,lines,strict=True)}
        slots=[dict(source_ordinal=index,row_SHA256=proof[index],repeat=physical>=56)
               for physical,index in enumerate(source)]
        expected=q.freeze_physical(b"\n".join(lines),slots,proof,set(proof))
        table=Table(dict(messages=[[json.dumps(item) for item in row["messages"]] for row in rows],
                         images=[[dict(bytes=None,path=row["images"][0])] for row in rows],
                         dataset=[ROUTE]*64))
        order=list(range(63,-1,-1))
        return expected,table,order

    def test_repeat_values_are_distinct_occurrences_from_actual_fetch(self):
        expected,table,order=self.fixture()
        shuffle,messages,values=q.arrow_projection(Dataset(table),Dataset(table,order),Dataset,expected,ROUTE)
        dataset=q.ObservedOccurrences(messages,table,shuffle)
        rank_indices=[list(range(rank,64,8)) for rank in range(8)]
        reports=[q.verify_rank(rank,[q.collate_observed([dataset[index]]) for index in indexes],
                               rank_indices,shuffle,expected,values) for rank,indexes in enumerate(rank_indices)]
        result=q.combine(reports,set(range(1,57)))
        self.assertEqual((result['physical_occurrences'],result['unique_source_rows'],
                          result['repeat_occurrences'],result['internal_source_rows']),(64,56,8,0))
        self.assertEqual(values[0],values[56])
        self.assertNotEqual(dataset[63]['physical'],dataset[7]['physical'])
        self.assertEqual(expected[0][1]['source_ordinal'],expected[56][1]['source_ordinal'])
        wrong=[dataset[index] for index in rank_indices[0]]
        wrong[0],wrong[1]=wrong[1],wrong[0]
        with self.assertRaisesRegex(ValueError,"RANK_OCCURRENCE_DRIFT"):
            q.verify_rank(0,wrong,rank_indices,shuffle,expected,values)

    def test_exact_physical_mapping_rejects_duplicate_non_null_and_changed_message(self):
        expected,table,order=self.fixture()
        with self.assertRaisesRegex(ValueError,"64_PHYSICAL_OCCURRENCES"):
            q.arrow_projection(Dataset(table),Dataset(table,[*order[:-1],order[0]]),Dataset,expected,ROUTE)
        table.cols["images"][0][0]["bytes"]=b"body"
        with self.assertRaisesRegex(ValueError,"NULL_MEDIA_BYTES"):
            q.arrow_projection(Dataset(table),Dataset(table,order),Dataset,expected,ROUTE)
        table.cols["images"][0][0]["bytes"]=None
        table.cols["messages"][0][1]='{"role":"assistant","content":"x","content":"y"}'
        with self.assertRaisesRegex(ValueError,"DUPLICATE_JSON_KEY"):
            q.arrow_projection(Dataset(table),Dataset(table,order),Dataset,expected,ROUTE)

    def test_split_guard_routes_schema_metadata_and_physical_values_separately(self):
        expected,table,order=self.fixture()
        def reject(changed,code):
            with self.assertRaisesRegex(ValueError,code):
                q.arrow_projection(Dataset(table),Dataset(changed,order),Dataset,expected,ROUTE)
        changed=deepcopy(table)
        changed.column_names.remove("dataset")
        reject(changed,"BOTH_ARROW_PHYSICAL_COLUMNS")
        changed=deepcopy(table)
        changed.schema.types["messages"]="wrong"
        reject(changed,"SAME_ARROW_PHYSICAL_SCHEMA")
        changed=deepcopy(table)
        changed.schema.metadata={"lineage":"changed"}
        reject(changed,"SAME_ARROW_SCHEMA_METADATA")
        changed=deepcopy(table)
        changed.cols["dataset"][-1]=ROUTE.rsplit("/",1)[1]
        reject(changed,"ROUTE_BOUND_EXPOSURES_PATH")
        changed.cols["dataset"][-1]=None
        reject(changed,"ROUTE_TYPED_CELL")
        changed=deepcopy(table)
        changed.cols["messages"][-1][0]='{"role":"user","content":"changed"}'
        reject(changed,"SAME_ARROW_PHYSICAL_CELL_VALUES")

    def test_frozen_packet_uses_original_production_cli(self):
        files=prepare.packet()
        self.assertEqual(prepare.packet(),files)
        cmd=json.loads(files["original-launch.json"])["command"]
        args=native_route.rewrite_args(cmd,"/data/train","/model-metadata","/unused-output")
        expected={"--dataset":"/data/train","--dataset_shuffle":"true","--dataloader_num_workers":"4",
                  "--data_seed":"42","--seed":"42","--max_steps":"25","--split_dataset_ratio":"0",
                  "--torch_dtype":"bfloat16","--use_cpu":"true"}
        self.assertTrue(all(args[args.index(key)+1]==value for key,value in expected.items()))
        self.assertNotIn("nvidia_baseline",args)
        self.assertEqual(args.count("--use_cpu"),1)
        with self.assertRaisesRegex(ValueError,"INHERITED_CPU_MODE_OVERRIDE"):
            native_route.rewrite_args(cmd+["--use_cpu=false"],"/data/train","/model-metadata","/unused-output")

    def test_set_epoch_requires_a_real_seed_transition_without_changing_base(self):
        class Loader:
            def __init__(self):
                self.batch_sampler=SimpleNamespace(base_seed=42,curr_seed=42)
                self.calls=[]

            def set_epoch(self,epoch):
                self.calls.append(epoch)
                self.batch_sampler.curr_seed=self.batch_sampler.base_seed+epoch

        loader=Loader()
        for epoch in range(1,13):
            self.assertEqual(native_route.apply_epoch(loader,epoch),
                             dict(data_seed=42,epoch_index=epoch,sampler_seed=42+epoch))
        self.assertEqual(loader.calls,list(range(1,13)))
        with self.assertRaisesRegex(ValueError,"ORIGINAL_DATA_SEED"):
            native_route.apply_epoch(loader,13)
        no_transition=Loader()
        no_transition.set_epoch=lambda index:None
        with self.assertRaisesRegex(ValueError,"SET_EPOCH_AND_SEED"):
            native_route.apply_epoch(no_transition)

    def test_plan_is_exactly_704_and_rejects_epoch1_or_late_seed_drift(self):
        import continuation as c
        order=list(range(64))
        expected={i:(None,dict(physical=i,source_ordinal=i%56+1,row_SHA256=f"{i%56:064x}"))
                  for i in range(64)}
        reports=[]
        for rank in range(8):
            epochs=[]
            for epoch in range(1,13):
                rows=[]
                for offset in range(8):
                    logical=(rank*8+offset+epoch-1)%64
                    rows.append(dict(**expected[logical][1],rank=rank,logical=logical,
                                     update=offset//4+1,accumulation=offset%4+1))
                epochs.append(dict(epoch_index=epoch,data_seed=42,sampler_seed=42+epoch,rows=rows))
            reports.append(dict(rank=rank,order=order,physical_values_verified=64,epochs=epochs))
        epoch1=[row["epochs"][0]["rows"] for row in reports]
        epoch1_ids=[[row["logical"] for row in rank] for rank in epoch1]
        raw=c.encode(epoch1)
        with patch.object(c,"ARROW_ORDER",c.digest(c.encode(order))):
            plan, epochs=c.assemble(reports,expected,raw,epoch1_ids)
            self.assertEqual(len(epochs),12)
            self.assertEqual([row["global_step"] for row in plan["updates"]],list(range(4,26)))
            self.assertTrue(all(len(row["slots"])==32 for row in plan["updates"]))
            self.assertEqual({cell["rank_offset0"] for cell in plan["updates"][0]["slots"]},{4,5,6,7})
            self.assertEqual({cell["rank_offset0"] for cell in plan["updates"][-1]["slots"]},{0,1,2,3})
            self.assertEqual({cell["epoch_index"] for cell in plan["updates"][-1]["slots"]},{12})
            with self.assertRaisesRegex(ValueError,"BYTE_EXACT"):
                c.assemble(reports,expected,raw+b" ",epoch1_ids)
            bad=deepcopy(reports)
            bad[7]["epochs"][-1]["sampler_seed"]=53
            with self.assertRaisesRegex(ValueError,"INDEX_AND_SEED"):
                c.assemble(bad,expected,raw,epoch1_ids)

    def test_nested_training_args_only_accept_cpu_true_and_bf16_true(self):
        cmd=json.loads(prepare.packet()["original-launch.json"])["command"]
        class Pipeline:
            cpu=True
            bf16=True

            def _parse_args(self,argv):
                self.asserted=(argv[argv.index("--use_cpu")+1],argv[argv.index("--torch_dtype")+1])
                assert self.asserted==("true","bfloat16")
                return SimpleNamespace(training_args=SimpleNamespace(use_cpu=self.cpu,bf16=self.bf16),
                    dataset=["/data/train"],val_dataset=None,split_dataset_ratio=0,dataset_shuffle=True,
                    train_dataloader_shuffle=True,seed=42,data_seed=42,per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,max_steps=25,dataloader_drop_last=False,packing=False,
                    padding_free=False,group_by_length=False,dataloader_num_workers=4,deepspeed=None)

        def read(holder):
            return (holder.args.dataset_shuffle,None)

        def build():
            return native_route.original_loader(cmd,"/data/train","/model-metadata","/unused-output",Pipeline,read)

        parsed,load=build()
        self.assertTrue(parsed.training_args.use_cpu and parsed.training_args.bf16)
        self.assertFalse(load(False))
        self.assertTrue(load(True))
        Pipeline.cpu=False
        with self.assertRaisesRegex(ValueError,"PARSED_DIAGNOSTIC_CPU_MODE"):
            build()
        Pipeline.cpu,Pipeline.bf16=True,False
        with self.assertRaisesRegex(ValueError,"PARSED_ORIGINAL_BF16"):
            build()

    def test_eight_real_child_processes_project_dist_rank_without_hf_private_state(self):
        context=mp.get_context("spawn")
        barrier=context.Barrier(8)
        results=context.Queue()
        children=[context.Process(name=f"metadata-fixture-{rank}",target=metadata_process,args=(rank,barrier,results))
                  for rank in range(8)]
        try:
            for child in children:
                child.start()
            observed=[results.get(timeout=10) for _ in children]
            for child in children:
                child.join(2)
            self.assertEqual([child.exitcode for child in children],[0]*8)
            self.assertEqual(len({row[0] for row in observed}),8)
            self.assertEqual(sorted(tuple(row[1:]) for row in observed),
                             [(rank,0,42,True,False,False,None) for rank in range(8)])
        finally:
            for child in children:
                if child.pid is not None and child.is_alive():
                    child.terminate()
                    child.join(1)
            results.close()

    def test_uninitialized_world_or_wrong_rank_is_rejected_at_access(self):
        parsed=SimpleNamespace(dataloader_drop_last=False,train_dataloader_shuffle=True,data_seed=42,
                               deepspeed=None,group_by_length=False)
        dist=SimpleNamespace(is_initialized=lambda:False,get_world_size=lambda:8,get_rank=lambda:3)
        traversal=native_route.MetadataTraversalArguments(parsed,dist,2)
        with self.assertRaisesRegex(ValueError,"EIGHT_RANK_CPU_PROCESS_GROUP"):
            _=traversal.process_index
        dist.is_initialized=lambda:True
        with self.assertRaisesRegex(ValueError,"METADATA_LOADER_LIVE_RANK"):
            _=traversal.process_index


if __name__=="__main__":
    unittest.main()
