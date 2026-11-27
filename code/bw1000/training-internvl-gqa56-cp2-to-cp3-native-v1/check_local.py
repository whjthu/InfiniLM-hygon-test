import ast
import json
import unittest
from types import SimpleNamespace

import cp3_contract as cp
import render


class TestCP3(unittest.TestCase):
    def test_production_packet_and_only_one_update(self):
        files = render.render()
        self.assertEqual(files, render.render())
        manifest = json.loads(files["SOURCE_FILES.json"])
        self.assertEqual(set(manifest["files"]), set(files)-{"SOURCE_FILES.json"})
        for name, pin in manifest["files"].items():
            self.assertEqual((len(files[name]), cp.digest(files[name])), (pin["bytes"], pin["SHA256"]))
        claim = json.loads(files["CLAIM.json"])
        self.assertEqual((claim["planned_updates"], claim["planned_input_slots"]), (1, 32))
        self.assertEqual(claim["placement"]["cpus"],64)
        self.assertFalse(claim["GPU_GO"])
        for name in ("node.py","native.py","scope.py","control_once.py","callback.py","slot_guard.py"):
            ast.parse(files[name].decode())
        self.assertIn(b"{ORIGINAL_CP2}:{CHECKPOINT}:ro", files["node.py"])
        self.assertIn(b"cp3_contract.resume_argv", files["native.py"])
        self.assertIn(repr(cp.ARROW_ROOT.as_posix()).encode(), files["scope.py"])
        self.assertIn(b'RUN/"train"/"checkpoint-3"', files["native.py"])
        self.assertIn(b'required.add("args.json")', files["slot_guard.py"])
        self.assertIn(b'control.should_training_stop = True', files["callback.py"])
        self.assertIn(b"Current exact holder no longer matches authorized terminal", files["control_once.py"])

    def test_actual_pinned_4726_receipt_and_32_order_rejection(self):
        evidence = (render.ARCHIVES/"specialized-internvl-4726-epoch1-native-order-cp3-32-slots.json").read_bytes()
        self.assertEqual(cp.digest(evidence), cp.EPOCH_ARCHIVE)
        response = json.loads(evidence)["original_evidence"][0]["response"]
        result = dict(response["output/result.json"]["fields"],request=response["request"])
        matrix = [[dict(physical=n) for n in range(rank*8,(rank+1)*8)] for rank in range(8)]
        first = dict(epoch_index=1,next_global_step=3,
                     ranks=[dict(rank=rank,rows=row[:4]) for rank,row in enumerate(matrix)])
        cp.validate_arrow(result,matrix,first)
        with self.assertRaisesRegex(ValueError,"FIRST4"):
            cp.validate_arrow(result,matrix,dict(first, ranks=first["ranks"][::-1]))
        with self.assertRaisesRegex(ValueError,"4726"):
            cp.validate_arrow(dict(result, expected_sampler_seed=42),matrix,first)

    def test_mapped_same_shape_parameters_require_bijection(self):
        params = [object() for _ in range(504)]
        original = {"state":{key:{} for key in range(504)},
                    "param_groups":[{"lr":cp.LR,"params":list(range(504))}]}
        named = [(f"p{i}",p) for i,p in enumerate(params)]
        native = SimpleNamespace(param_groups=[{"lr":cp.LR,"params":params[::-1]}],
                                 state={param:{} for param in params})
        self.assertEqual(cp.optimizer_mapping(native,named,original)[0],(0,"p503",params[-1]))
        native.param_groups[0]["params"][-1] = params[-1]
        with self.assertRaisesRegex(ValueError,"BIJECTION"):
            cp.optimizer_mapping(native,named,original)

    def test_resume_command_retains_nominal_recipe_and_only_cp2(self):
        command=["swift", "sft", "--callbacks", "internvl_gqa56_cp3_observed_engineering",
                 "--max_steps","25","--dataloader_num_workers","4"]
        actual=cp.resume_argv(command)
        self.assertEqual(actual[:len(command)],command)
        self.assertEqual(actual[len(command):],["--resume_from_checkpoint","/gqa-recovery-cp2",
                         "--resume_only_model","false","--ignore_data_skip","false"])
        with self.assertRaisesRegex(ValueError,"FOUR_WORKER"):
            cp.resume_argv(command[:-1]+["0"])


if __name__ == "__main__":
    unittest.main()
