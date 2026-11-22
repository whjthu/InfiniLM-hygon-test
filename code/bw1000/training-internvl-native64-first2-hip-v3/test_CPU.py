"""Validate T03 staging and the real callback's stop behavior without Torch or Slurm."""
import ast
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path, PurePosixPath
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import native
import node
import scope


def import_file(name,path):
    spec=spec_from_file_location(name,path)
    module=module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Package(unittest.TestCase):
    def test_actual_original_recipe_keeps_25_step_scheduler_and_full_inputs(self):
        base=scope.HERE.parent/"training-qwen-A100-first-updates-v1/deltas/tools/hygon_vlm_recipe.py"
        builder=import_file("common_vlm_recipe_test",base).build_launch
        raw=(scope.HERE/"original-launch.json").read_bytes()
        with patch.multiple(scope, FRAMEWORK=PurePosixPath("/framework"),DATASET=PurePosixPath("/assets/coco64.jsonl"),
                            ASSETS=PurePosixPath("/assets")):
            launch=scope.launch_recipe(builder,raw,package=PurePosixPath("/source"),run=PurePosixPath("/run"))
        argv=launch["command"]
        self.assertEqual(argv[argv.index("--max_steps")+1],"25")
        self.assertEqual(argv[argv.index("--model")+1],"/internvl-model")
        self.assertIn("internvl_first2_engineering",argv)
        self.assertIn("nvidia_baseline",argv)
        self.assertEqual(launch["environment_overrides"]["NPROC_PER_NODE"],"8")
        with self.assertRaisesRegex(ValueError,"Exact original"):
            scope.launch_recipe(builder,raw+b"\n")

    def test_pinned_internvl_target_selector_with_original_failure_order(self):
        framework=scope.HERE.parent/"training-qwen-target-unified-framework-a68952c-v1/deltas"
        with patch.object(sys,"path",[str(framework),*sys.path]):
            from vlm.runtime.native_recipe_contract import target_signature

        frozen=json.loads((scope.HERE/"NATIVE_BINDING.json").read_bytes())
        observed=r"^(model\.language_model(?=\.).*\.(q_proj|up_proj|down_proj|gate_proj|v_proj|o_proj|k_proj))$"
        actual=dict(trainable_manifest=frozen["trainable_manifest"],target_modules=observed,
                    actual_trainable_tensors=504,actual_trainable_parameters=43646976)
        self.assertNotEqual(observed,frozen["target_modules"])
        self.assertEqual(scope.verify_shape(frozen,actual,target_signature),frozen["trainable_manifest"])
        for wrong in (observed.replace("q_proj","q_proj|q_proj"), observed.replace("q_proj","o_proj"),
                      observed.replace("language_model","visual"), observed.removesuffix("$")):
            with self.subTest(selector=wrong),self.assertRaisesRegex(ValueError,"target selector differs"):
                scope.verify_shape(frozen,dict(actual,target_modules=wrong),target_signature)
        incorrect=[dict(frozen["trainable_manifest"][0],dtype="other"),*frozen["trainable_manifest"][1:]]
        with self.assertRaisesRegex(ValueError,"PEFT structure differs"):
            scope.verify_shape(frozen,dict(actual,trainable_manifest=incorrect),target_signature)

    def test_target_runtime_release_and_reference_release_are_distinct(self):
        versions={"torch":"2.10.0+das.opt1.dtk2604.2606161421.g91a580","transformers":"5.5.0",
                  "ms-swift":"4.4.2","peft":"0.19.1","accelerate":"1.14.0","tokenizers":"0.22.2"}
        native.verify_runtime_versions(versions,"2.10.0","6.3.26113","5.5.0")
        with self.assertRaisesRegex(ValueError,"target Python runtime differs"):
            native.verify_runtime_versions(dict(versions,transformers="5.12.1"),"2.10.0","6.3.26113","5.12.1")
        with self.assertRaisesRegex(ValueError,"target Python runtime differs"):
            native.verify_runtime_versions(versions,"2.10.0","6.3.26113","5.12.1")

    def test_original_eight_rank_fingerprint_order_and_failure_denominator(self):
        rows=[dict(rank=r,micro_ordinal=i,global_step=(i-1)//4,input_fingerprint_SHA256=f"{r*8+i:064x}")
              for r in range(8) for i in range(1,9)]
        reference=dict(schema="native_first_two_inputs_v1",case="T03.sft",rows=rows,
                       fingerprint_source_SHA256=scope.FINGERPRINT_SHA,
                       reference_request_id="031e04f24efa429595364e6f0da1e5b6",
                       world_size=8,accumulation=4,projection=["args","kwargs"])
        result=scope.input_roster(reference)
        self.assertEqual(len(result),8)
        self.assertEqual(result["7"][-1],f"{64:064x}")
        rows[-1]["micro_ordinal"]=7
        with self.assertRaisesRegex(ValueError,"reordered"):
            scope.input_roster(reference)
        scope.finish_step(2,updates=2,inputs=8,outputs=8,clips=2,names=sorted(scope.CAPTURE_NAMES))
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            scope.finish_step(2,updates=1,inputs=8,outputs=8,clips=2,names=sorted(scope.CAPTURE_NAMES))

    def test_callback_stops_exactly_after_second_real_optimizer_update(self):
        def make_module(name,**values):
            module=ModuleType(name)
            module.__dict__.update(values)
            return module
        noop=lambda *args,**kwargs: None
        class TrainerCallback:
            def __init__(self,*args):
                pass
        env={"torch":make_module("torch"),
             "swift.callbacks":make_module("swift.callbacks",TrainerCallback=TrainerCallback,callbacks_map={}),
             "vlm.plugins.native_first_updates":make_module("vlm.plugins.native_first_updates",rng=noop)}
        definitions={
            "checkpointing_lifecycle":dict(validate_checkpointing_lifecycle=noop),
            "first_update_audit":dict(fingerprint=noop),
            "first_update_capture":dict(FirstUpdateCapture=object),
            "initial_trainables":dict(named_parameter_sha256=noop,restore_initial_trainables=noop),
            "native_caller_budget":dict(validate_capture_conditions=noop),
            "native_recipe_contract":dict(validate_native_recipe=noop),
            "provider_sources":dict(model_provider_sources=noop),
        }
        for suffix,values in definitions.items():
            name="vlm.runtime."+suffix
            env[name]=make_module(name,**values)
        with patch.dict(sys.modules,env):
            callback=import_file("t03_callback_fixture",scope.HERE/"callback.py")
            hook=callback.InternVLFirstTwo(None,None)
            order=[]
            hook.capture=SimpleNamespace(step_end=lambda:order.append("capture_handled"),guarded=lambda fn:fn)
            control=SimpleNamespace(should_training_stop=False)
            hook.on_step_end(None,SimpleNamespace(global_step=1),control)
            self.assertFalse(control.should_training_stop)
            hook.on_step_end(None,SimpleNamespace(global_step=2),control)
            self.assertTrue(control.should_training_stop)
            self.assertEqual(order,["capture_handled","capture_handled"])
            with self.assertRaisesRegex(ValueError,"Unexpected third"):
                hook.on_step_begin(None,SimpleNamespace(global_step=2),control)

    def test_compute_proof_exact_step_and_gpu_container_mount_contract(self):
        outcome=dict(state="VERIFIED_ALL17_READONLY_ON_T03_COMPUTE_NODE",step_client_rc=0,
                     accounting=["1087656.1004","old","COMPLETED","0:0","2",scope.NODE,"cpu=1,mem=2G,node=1"],
                     compute_proof_SHA256=node.PROOFS["computed-proof.json"])
        proof=dict(step="1087656.1004",node=scope.NODE,read_only_members=17,localized_stat_views=20,
                   original_inodes_unchanged=True,localized_plan_SHA256=node.PROOFS["localized-plan.json"])
        node.validate_compute_receipts(outcome,proof)
        with self.assertRaisesRegex(ValueError,"readonly"):
            node.validate_compute_receipts(outcome,dict(proof,localized_stat_views=19))
        argv=node.command({"binds":["/test:/internvl-model:ro"]},100,"a"*64)
        self.assertIn("--rocm",argv)
        self.assertIn("/test:/internvl-model:ro",argv)
        self.assertEqual(argv[-2:],["100","a"*64])
        for name in scope.HERE.glob("*.py"):
            ast.parse(name.read_bytes(),feature_version=(3,10))


if __name__=="__main__":
    unittest.main()
