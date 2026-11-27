"""Run the rendered InternVL callback gates with bounded local stand-ins."""
import ast
import importlib.util
import math
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import unittest

import render
import slot_guard as guard


def method(tree, owner, name, globals_dict):
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==owner)
    node=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"rendered-callback.py","exec"),globals_dict)
    return globals_dict[name]


class NeverIterate:
    def __iter__(self):
        raise AssertionError("diagnostic changed native sampling")


class BatchSamplerFixture(NeverIterate):
    sampler = NeverIterate()


class LoaderFixture(NeverIterate):
    batch_sampler = BatchSamplerFixture()


class GuardTests(unittest.TestCase):
    def setUp(self):
        original=[f'row-{i}'.encode() for i in range(1,65)]
        sequence=list(range(1,57))+list(range(1,9))
        roster={(i,):dict(ordinal=i,row_SHA256=guard.digest(line)) for i,line in enumerate(original,1)}
        slots=[dict(update=flat//32+1,rank=flat%8,accumulation=flat%32//8+1,source_ordinal=i,
                    row_SHA256=roster[(i,)]["row_SHA256"],repeat=flat>=56) for flat,i in enumerate(sequence)]
        self.raw=lambda rows:b"\n".join(rows)+b"\n"
        self.original,self.roster,self.sequence,self.slots=original,roster,sequence,slots
        self.exposures=[original[index-1] for index in sequence]
        self.subset,self.by_rank=self.bind()
        self.files=render.render()

    def bind(self):
        return guard.bind(self.raw(self.original),self.raw(self.original[:56]),self.raw(self.exposures),
                          self.slots,self.roster)

    def test_exact_plan_and_no_internal_or_wrong_repeats(self):
        self.assertEqual(len(self.subset),56)
        self.assertEqual(sum(map(len,self.by_rank)),64)
        self.exposures[-1]=self.original[62]
        with self.assertRaisesRegex(ValueError,"TRAIN_ONLY"):
            self.bind()
        self.exposures[-1]=self.original[7]
        self.slots[-1]["rank"]=0
        with self.assertRaisesRegex(ValueError,"IMMUTABLE_SOURCE_SLOT"):
            self.bind()

    def test_rendered_pre_optimizer_callback_accepts_exact_sequence_only(self):
        callback=ast.parse(self.files["callback.py"])
        globals_dict=dict(slot_guard=guard,math=math,need=guard.need)
        check=method(callback,"InternVLFirstTwo","on_pre_optimizer_step",globals_dict)
        fired=[]
        cap=SimpleNamespace(updates=0,inputs=4,outputs=4,pending_source=None,guarded=lambda f:f,
                            consensus_check=lambda f:f(),before_optimizer=lambda:fired.append(True))
        trainer=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr":0.0}]))
        owner=SimpleNamespace(capture=cap,trainer=trainer)
        control=object()
        self.assertIs(check(owner,{},SimpleNamespace(global_step=0),control),control)
        cap.inputs=3
        with self.assertRaisesRegex(ValueError,"PRE_ADAMW_SLOT"):
            check(owner,{},SimpleNamespace(global_step=0),control)
        cap.inputs=cap.outputs=8
        cap.updates=1
        with self.assertRaisesRegex(ValueError,"ZERO_THEN_POSITIVE"):
            check(owner,{},SimpleNamespace(global_step=1),control)
        trainer.optimizer.param_groups[0]["lr"]=1e-4
        self.assertIs(check(owner,{},SimpleNamespace(global_step=1),control),control)
        self.assertEqual(fired,[True,True])

    def test_rendered_forward_callback_accepts_actual_rank_slot(self):
        callback=ast.parse(self.files["callback.py"])
        globals_dict=dict(slot_guard=guard,need=guard.need,REQUEST=render.REQUEST,
                          CAPTURE_NAMES={"forward-01-input.pt"},fingerprint=lambda x:{"finite":True,"SHA256":"0"*64},
                          after_sample=lambda kwargs,pending,fp:kwargs["source"],
                          write_once=lambda path,doc:path.write_bytes(b"{}") and "0"*64)
        save=method(callback,"MetadataCapture","save",globals_dict)
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as root:
            cap=SimpleNamespace(inputs=0,rank=0,domain_slots=self.by_rank,files=[],pending_source={},
                                root=Path(root),check_deadline=lambda:None,consensus_check=lambda f:f(),
                                progress=lambda:None)
            bad=dict(micro_ordinal=1,global_step=0,args=[],kwargs={"source":self.by_rank[1][0]})
            with self.assertRaisesRegex(ValueError,"NATIVE_RANK_ACCUMULATION"):
                save(cap,"forward-01-input.pt",bad)
            self.assertEqual(cap.files,[])
            bad["kwargs"]["source"]=self.by_rank[0][0]
            bad["global_step"]=1
            with self.assertRaisesRegex(ValueError,"ACCUMULATION_FORWARD"):
                save(cap,"forward-01-input.pt",bad)
            bad["global_step"]=0
            save(cap,"forward-01-input.pt",bad)
            self.assertEqual(len(cap.files),1)

    def test_real_recipe_builder_keeps_training_options_and_omits_4372_finalizer(self):
        builder_path=render.REPO/"code/bw1000/training-gemma-native-inputs-v1/framework/tools/hygon_vlm_recipe.py"
        spec=importlib.util.spec_from_file_location("actual_framework_builder",builder_path)
        builder=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        scope={"__file__":str(render.HERE/"scope.py")}
        exec(compile(self.files["scope.py"],"rendered-scope.py","exec"),scope)
        for key in ("FRAMEWORK","DATASET","ASSETS","MODEL"):
            scope[key]=PurePosixPath(scope[key])
        command=scope["launch_recipe"](builder.build_launch,self.files["original-launch.json"],
            package=PurePosixPath("/stage/src"),run=PurePosixPath("/stage/run"))["command"]
        self.assertNotIn("nvidia_baseline",command)
        self.assertEqual(command[command.index("--callbacks")+1],"internvl_gqa56_first2_engineering")
        self.assertEqual(command[command.index("--max_steps")+1],"25")
        self.assertEqual(command[command.index("--warmup_ratio")+1],"0.03")
        with self.assertRaisesRegex(SystemExit,"DISABLED_REVIEW_SOURCE"):
            exec(compile(self.files["control_once.py"],"control_once.py","exec"),{})

    def test_sampler_observation_does_not_iterate_and_checkpoint_callback_is_exact(self):
        metadata=guard.sampler_identity(LoaderFixture())
        self.assertEqual([item["link"] for item in metadata["classes"]],["dataloader","batch_sampler","sampler"])
        self.assertFalse(metadata["sampler_iterated"] or metadata["order_verified"])
        with self.assertRaisesRegex(ValueError,"CHAIN_UNAVAILABLE"):
            guard.sampler_identity(None)
        callback=ast.parse(self.files["callback.py"])
        on_save=method(callback,"InternVLFirstTwo","on_save",{"need":guard.need})
        obj=SimpleNamespace(saved_cp2=False)
        control=object()
        with self.assertRaisesRegex(ValueError,"NATIVE_SAVE_CALLBACK"):
            on_save(obj,{},SimpleNamespace(global_step=1),control)
        self.assertIs(on_save(obj,{},SimpleNamespace(global_step=2),control),control)
        with self.assertRaisesRegex(ValueError,"NATIVE_SAVE_CALLBACK"):
            on_save(obj,{},SimpleNamespace(global_step=2),control)


if __name__=="__main__":
    unittest.main()
