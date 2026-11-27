"""Extend the proven GQA first-two capture through the original 25-step recipe."""
import argparse
import json
from pathlib import Path

import render_internvl_gqa64_first2 as first


REQUEST = "e23c26259c4344bc837a693ae9c2f11a"
LABEL = "training-internvl-gqa64-continuous25-hip-v1"
SOURCE_MANIFEST = "49153458e62d6ff8ee3a14ac5e8fda248ff6557d570a360f0dea6a6abf2ac1fc"
WHOLE, NATIVE, SLURM = 3600, 3180, 3300
ARTIFACT_READER = first.ROSTER.with_name("internvl_gqa25_artifacts.py")


def edit(files, name, substitutions):
    text = files[name].decode()
    for old, new in substitutions:
        text = first.replace(text, old, new)
    files[name] = text.encode()


def adapt_callback(files):
    a = files["callback.py"].decode()
    a = first.replace(a, "        self.capture=None",
                      "        self.capture=None\n        self.optimizer_steps=0\n"
                      "        self.positive_lr_steps=0\n        self.checkpoints=[]\n        self.current_positive_lr=False")
    # Retain the original rank and recipe admission, first-two hooks and broadcast.
    start = a.index("    def on_step_begin(")
    end = a.index("\n\nneed(CALLBACK", start)
    files["callback.py"] = (a[:start] + '''    def on_step_begin(self,args,state,control,**kwargs):
        self.capture.check_deadline()
        need(0 <= state.global_step < 25 and state.global_step == self.optimizer_steps,
             "Unexpected optimizer step sequence")
        return control

    def on_pre_optimizer_step(self,args,state,control,**kwargs):
        self.capture.check_deadline()
        rates = [float(group["lr"]) for group in self.trainer.optimizer.param_groups]
        need(rates and all(math.isfinite(lr) for lr in rates), "Invalid native learning rate")
        if state.global_step == 0:
            need(all(lr == 0 for lr in rates), "Original warmup step differs")
        else:
            need(all(lr > 0 for lr in rates), "Original positive learning-rate schedule differs")
        self.current_positive_lr = state.global_step > 0
        if state.global_step < 2:
            self.capture.guarded(self.capture.before_optimizer)()
        return control

    def on_optimizer_step(self,args,state,control,**kwargs):
        self.capture.check_deadline()
        need(state.global_step == self.optimizer_steps and
             self.trainer.accelerator.optimizer_step_was_skipped is False, "Native optimizer step skipped or duplicated")
        self.optimizer_steps += 1
        self.positive_lr_steps += int(self.current_positive_lr)
        if state.global_step < 2:
            self.capture.guarded(self.capture.after_optimizer)()
        return control

    def on_step_end(self,args,state,control,**kwargs):
        self.capture.check_deadline()
        need(state.global_step == self.optimizer_steps, "Native optimizer/global step mismatch")
        if state.global_step == 2:
            self.capture.guarded(self.capture.step_end)()
            self.capture.guarded(self.capture.train_end)()
        return control

    def on_save(self,args,state,control,**kwargs):
        self.capture.check_deadline()
        need(state.global_step in (10,20,25) and state.global_step not in self.checkpoints,
             "Unexpected native checkpoint")
        self.checkpoints.append(state.global_step)
        return control

    def on_train_end(self,args,state,control,**kwargs):
        self.capture.check_deadline()
        need(state.global_step == self.optimizer_steps == 25 and self.positive_lr_steps == 24
             and self.checkpoints == [10,20,25] and self.capture.detached and not self.capture.failed,
             "Incomplete 25-step native training")
        model = kwargs.get("model", self.trainer.model)
        named = sorted((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
        need(len(named) == 504, "Final trainable parameter scope differs")
        final = named_parameter_sha256(named)
        need(final != NAMED_SHA, "No trainable parameter update")
        self.capture.write_json("run25.json", dict(rank=self.capture.rank, request=REQUEST, global_step=25,
             optimizer_callbacks=self.optimizer_steps, positive_LR_callbacks=self.positive_lr_steps,
             checkpoint_steps=self.checkpoints, initial_parameters=NAMED_SHA, final_parameters=final))
        return control
''' + a[end:]).encode()


def adapt_node(files):
    edit(files, "node.py", [
        ("import sys", "import sys\nimport tempfile"),
        ("def command(plan,epoch,plan_sha):", "def command(plan,epoch,plan_sha,scratch):"),
        ('f"{RUN / \'tmp\'}:/tmp:rw"', 'f"{scratch}:/tmp:rw"'),
        ('for name in ("home","cache","tmp","capture"):', 'for name in ("home","cache","capture"):'),
        ("native_schedule_steps=25,planned_updates=2,planned_micro_forwards_by_rank=8",
         "native_schedule_steps=25,planned_updates=25,planned_first_two_micro_forwards_by_rank=8"),
        ('''        with (RUN/"container.log").open("xb") as out:
            result=subprocess.run(command(plan,args.epoch,plan_sha),stdin=subprocess.DEVNULL,stdout=out,
                                  stderr=subprocess.STDOUT,timeout=seconds_left(args.epoch,SLURM)-20)
        rc=result.returncode''',
         '''        fs=subprocess.check_output(
            ["findmnt","--noheadings","-o","FSTYPE","--target","/tmp"],text=True,timeout=5).strip()
        need(fs=="ext4","Qualified node-local temporary filesystem differs")
        disk=os.statvfs("/tmp")
        need(disk.f_bavail*disk.f_frsize>=2**31,"Node-local temporary filesystem is full")
        with tempfile.TemporaryDirectory(prefix="internvl-gqa25-",dir="/tmp") as temp:
            st=Path(temp).stat()
            write_once(RUN/"local-scratch.json",dict(request=REQUEST,device=st.st_dev,inode=st.st_ino,fs=fs))
            with (RUN/"container.log").open("xb") as out:
                result=subprocess.run(command(plan,args.epoch,plan_sha,temp),stdin=subprocess.DEVNULL,stdout=out,
                                      stderr=subprocess.STDOUT,timeout=seconds_left(args.epoch,SLURM)-20)
            rc=result.returncode
        need(not Path(temp).exists(),"Owned node-local temporary filesystem contents remain")
        write_once(RUN/"local-scratch-clean.json",dict(request=REQUEST,removed=True))'''),
    ])


def adapt_native(files):
    edit(files, "native.py", [
        ("import owned_processes", "import owned_processes\nfrom internvl_gqa25_artifacts import gather"),
        ('scratch = {**private_tmp(RUN / "tmp"), **private_unix_socket()}',
         '''scratch = json.loads(pin(RUN/"local-scratch.json"))
    tmp=Path("/tmp").stat()
    need(scratch["request"]==REQUEST and scratch["fs"]=="ext4"
         and (scratch["device"],scratch["inode"])==(tmp.st_dev,tmp.st_ino),"Node-local scratch bind differs")
    scratch.update(private_unix_socket())'''),
        ('HF_HOME=str(RUN/"cache"),T03_FIRST2_EPOCH=str(epoch)',
         'HF_HOME="/tmp/hf",HF_DATASETS_CACHE="/tmp/datasets",T03_FIRST2_EPOCH=str(epoch)'),
        ('planned_update_slots=2,original_schedule_steps=25,', 'planned_update_slots=25,original_schedule_steps=25,'),
        ('planned_input_slots=64,binding_SHA256=', 'planned_input_slots=800,first_two_witness_slots=64,binding_SHA256='),
        ('    report=dict(request=REQUEST,actual_step=intent["step"],native_returncode=',
         '''    continuous=None
    if complete:
        try:
            continuous=gather(RUN,REQUEST)
        except (OSError,ValueError,KeyError,TypeError) as failure:
            error=type(failure).__name__+": "+str(failure)
    complete=complete and continuous is not None
    report=dict(request=REQUEST,actual_step=intent["step"],continuous25=continuous,
                passed=complete,native_returncode='''),
    ])


def render():
    files = first.render()
    if first.digest(files.pop("SOURCE_FILES.json")) != SOURCE_MANIFEST:
        raise ValueError("Published first-two sources differ")
    for name in ("scope.py", "control_once.py"):
        edit(files, name, [(first.REQUEST, REQUEST), (first.LABEL, LABEL)])
    edit(files, "scope.py", [("WHOLE, NATIVE, SLURM = 1800, 1440, 1500",
                             f"WHOLE, NATIVE, SLURM = {WHOLE}, {NATIVE}, {SLURM}")])
    edit(files, "control_once.py", [("WHOLE, GPU_LIMIT = 1800, 1500", f"WHOLE, GPU_LIMIT = {WHOLE}, {SLURM}")])
    adapt_callback(files)
    adapt_node(files)
    adapt_native(files)
    claim = json.loads(files["CLAIM.json"])
    claim.update(request=REQUEST, label=LABEL, schema="INTERNVL_GQA64_CONTINUOUS25_HIP_V1",
                 planned_updates=25, planned_nonzero_updates=24, planned_microbatches=800,
                 first_two_capture_rows=64, checkpoint_steps=[10,20,25],
                 native_seconds=NATIVE, slurm_seconds=SLURM, whole_seconds=WHOLE,
                 predecessor_clean_step="1103606.4374", node_local_scratch_proof="1103606.4380",
                 replaceable_standby_step="1103606.4382")
    files["CLAIM.json"] = first.encode(claim)
    files[ARTIFACT_READER.name] = ARTIFACT_READER.read_bytes()
    files["SOURCE_FILES.json"] = first.encode(
        {"request": REQUEST, "files": {name: {"bytes": len(data), "SHA256": first.digest(data)}
                                     for name, data in files.items()}})
    for name, data in files.items():
        if name.endswith(".py"):
            compile(data,name,"exec")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = render()
    if args.output is not None:
        args.output.mkdir()
        for name,data in files.items():
            with (args.output/name).open("xb") as fd:
                fd.write(data)
    print(json.dumps({"request":REQUEST,"manifest_SHA256":first.digest(files["SOURCE_FILES.json"]),
                      "files":len(files),"remote_calls":0}))


if __name__=="__main__":
    main()
