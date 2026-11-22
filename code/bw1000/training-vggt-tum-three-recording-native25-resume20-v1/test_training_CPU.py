"""Pure-CPU boundaries for the exact new VGGT GPU package."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import compare_resume
import launch
import preflight_gate
import read_terminal


def precision_receipt(arm):
    first=compare_resume.ARMS[arm]
    scheduler=dict(type="constant_learning_rate_in_optimizer_param_groups",mutable_scheduler=False,
                   learning_rate=2e-5)
    checkpoints={}
    for number in (10,20,25):
        if number>first:
            checkpoints[str(number)]=dict(
                global_step=number,next_global_step=number+1,next_batch_cursor=number%9,
                file_SHA256="f"*64, model_SHA256="1"*64, optimizer_SHA256="2"*64,placements_SHA256="3"*64,
                rng_SHA256="4"*64,full_boundary_SHA256="5"*64)
    return dict(arm=arm,executed_optimizer_steps=25-first,bundle_SHA256=preflight_gate.BUNDLE_SHA,
                backend_SHA256="a"*64,checkpoint_identity={"native":"identity"},epoch=100.0,deadline=160.0,
                input_order_SHA256=preflight_gate.ORDER_SHA,view_order_SHA256=preflight_gate.VIEW_SHA,
                process=dict(boot_id="fixture",pid=first+1,startticks=30),
                steps=[dict(global_optimizer_step=step,batch_cursor=(step-1)%9,loss=0.2,
                            grad_norm=0.01,camera_head_grad_norm_before_clip=0.01)
                       for step in range(first+1,26)],
                checkpoints=checkpoints,scheduler=scheduler,gradient_accumulation=1,
                resume_source_SHA256="f"*64 if first else None,final_state_SHA256="0"*64)


class TrainingContractTest(unittest.TestCase):
    def test_same_checkpoint20_and_all_five_updated_steps(self):
        receipts={name:precision_receipt(name) for name in compare_resume.ARMS}
        result=compare_resume.compare(receipts)
        self.assertTrue(result["comparison_passed"])
        self.assertEqual(result["actual_update_invocations"],30)
        self.assertEqual([x["global_step"] for x in result["arms"]["resume20"]["steps"]],list(range(21,26)))
        self.assertEqual([x["batch_cursor"] for x in receipts["resume20"]["steps"]],list(range(2,7)))
        self.assertFalse(compare_resume.compare(dict(receipts,resume20=dict(
            receipts["resume20"],final_state_SHA256="1"*64)))["comparison_passed"])

    def test_resume_numerics_and_wrong_cohort_fail_closed(self):
        baseline={name:precision_receipt(name) for name in compare_resume.ARMS}
        for key in ("loss","grad_norm","camera_head_grad_norm_before_clip"):
            wrong=copy.deepcopy(baseline)
            wrong["resume20"]["steps"][0][key]+=0.01
            self.assertFalse(compare_resume.compare(wrong)["comparison_passed"])
        for key,value in (("resume_source_SHA256","0"*64),("view_order_SHA256","0"*64)):
            wrong=copy.deepcopy(baseline)
            wrong["resume20"][key]=value
            with self.assertRaises(ValueError):
                compare_resume.compare(wrong)
        wrong=copy.deepcopy(baseline)
        wrong["resume20"]["steps"][0]["batch_cursor"]=0
        with self.assertRaisesRegex(ValueError,"nine-pair"):
            compare_resume.compare(wrong)

    def test_frozen_public_predecessor_and_bound_controller_compile(self):
        claim,_,body,program=launch.frozen(launch.REPO,"a"*40)
        compile(program,"controller-wire-fixture","exec")
        self.assertLess(len(program),131072)
        self.assertEqual(len(body),64)
        self.assertEqual((claim["planned_updates"],claim["batch_count"]),(30,9))
        self.assertFalse(claim["training_READY"])
        remote=claim["sources"]["actual-922-preflight.json"]
        self.assertEqual(remote["sha"],preflight_gate.CPU_PUBLIC_SHA)
        result=json.loads((launch.REPO/remote["path"]).read_bytes())
        self.assertEqual([row[0] for row in result["actual_accounting"]],
                         ["1103606.921","1103606.922","1103606.923"])
        self.assertEqual(result["native"]["view_order_SHA256"],preflight_gate.VIEW_SHA)
        self.assertEqual(result["persistent_remote_result_SHA256"],preflight_gate.CPU_RESULT_SHA)

    def test_independent_reader_checks_terminal_before_accepting_both_live_receipts(self):
        accounting={phase:[f"1103606.{100+i}",read_terminal.LABEL+"-"+phase,"COMPLETED","0:0",
                           "8",read_terminal.NODE,
                           "cpu=4,mem=64G,gres/dcu=1,node=1" if phase=="gpu" else "cpu=2,mem=2G,node=1"]
                    for i,phase in enumerate(read_terminal.PHASES)}
        pci=[f"0000:0{i}:00.0" for i in range(8)]
        snapshot={"nodes":{read_terminal.NODE:dict(PCIs=pci,parent=read_terminal.PARENT,active_steps=[],
            KFD_clean=True,PID_clean=True,device_probe_observed=True,device_probe_step="1103606.103")}}
        with TemporaryDirectory() as temp:
            root=Path(temp).resolve()/read_terminal.LABEL
            root.mkdir()
            def write(name, value):
                path=root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                raw=json.dumps(value).encode()
                path.write_bytes(raw)
                return preflight_gate.digest(raw)
            write("intent.json",dict(request=read_terminal.REQUEST,parent=read_terminal.PARENT,node=read_terminal.NODE))
            write("preflight-binding.json",dict(preflight_request=preflight_gate.REQUEST,
                preflight_result_SHA256=preflight_gate.CPU_RESULT_SHA,preflight_install_SHA256=preflight_gate.INSTALL_SHA,
                view_order_SHA256=preflight_gate.VIEW_SHA))
            for condition in ("before","after"):
                write(condition+".json",dict(pci=pci,boot_id="boot",kfd_holder_count=0,
                                             full_fd_inventory_performed=True))
            receipts={arm:precision_receipt(arm) for arm in compare_resume.ARMS}
            hashed={arm:dict(SHA256=write("arms/"+arm+"/receipt.json",value),value=value)
                    for arm,value in receipts.items()}
            raw_sha=write("native-result.json",dict(receipts=hashed,comparison=compare_resume.compare(receipts)))
            write("outcome.json",dict(request=read_terminal.REQUEST,passed=True,result_SHA256=raw_sha,
                                       stages={phase:{"row":row} for phase,row in accounting.items()}))
            self.assertTrue(read_terminal.disposition(root,accounting,snapshot)["qualified"])
            snapshot["nodes"][read_terminal.NODE]["PID_clean"]=False
            with self.assertRaisesRegex(ValueError,"independently show device clear"):
                read_terminal.disposition(root,accounting,snapshot)


if __name__=="__main__":
    unittest.main()
