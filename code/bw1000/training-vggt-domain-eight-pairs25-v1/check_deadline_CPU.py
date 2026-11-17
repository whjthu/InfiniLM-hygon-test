"""Exercise the actual controller step argv, client timeout and accounting deadline without Slurm."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

HERE=Path(__file__).resolve().parent


def scenario(phase, now, deadline, *, accounting_delay=0, return_delay=0, reject=False):
    spec=importlib.util.spec_from_file_location("deadline_controller_fixture",HERE/"run_tum_continuous_once.py")
    controller=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    clock=types.SimpleNamespace(now=float(now))
    controller.time=types.SimpleNamespace(time=lambda:clock.now,monotonic=lambda:100000+clock.now)
    events=[]
    reads=0
    def output(args, *,timeout,text):
        nonlocal reads
        events.append(dict(kind="accounting",timeout=timeout,wall=clock.now))
        assert 0<timeout<=10 and clock.now+timeout<=deadline
        reads+=1
        if reads==1:
            clock.now+=accounting_delay
            return ""
        return f"1087656.999|{controller.LABEL}-{phase}|COMPLETED|0:0|1|f11r4n19|"+("gres/dcu=1" if phase=="gpu" else "cpu=2")+"\n"
    def run(args, *,stdout,stderr,timeout):
        events.append(dict(kind="srun",args=args,timeout=timeout,wall=clock.now))
        clock.now+=timeout+return_delay
        return types.SimpleNamespace(returncode=0)
    controller.subprocess=types.SimpleNamespace(check_output=output,run=run,STDOUT=subprocess.STDOUT,
                                              TimeoutExpired=subprocess.TimeoutExpired)
    def wait(read, until, *,max_observations):
        events.append(dict(kind="wait_terminal",monotonic_deadline=until,wall=clock.now))
        row=read() if 100000+clock.now<until else None
        return dict(terminal_visible=bool(row),row=row,observations=[row] if row else [])
    fake=types.ModuleType("slurm_accounting")
    fake.wait_terminal=wait
    sys.modules["slurm_accounting"]=fake
    with tempfile.TemporaryDirectory() as folder:
        try:
            result=controller.step(Path(folder),phase,["fixture-no-command"],deadline)
        except ValueError:
            assert reject
            assert not any(item["kind"]=="srun" for item in events)
            return dict(phase=phase,state="NO_NEW_STEP",initial_remaining=deadline-now,events=events)
        assert not reject
    launched=next(item for item in events if item["kind"]=="srun")
    command=launched["args"]
    requested=command[command.index("-t")+1]
    hours,minutes,seconds=map(int,requested.split(":"))
    slurm_seconds=hours*3600+minutes*60+seconds
    assert seconds==0 and slurm_seconds>=60 and launched["timeout"]==slurm_seconds+5
    assert launched["timeout"]<=1440 if phase=="gpu" else launched["timeout"]<=65
    reserve=90 if phase!="post" else 0
    assert launched["wall"]+launched["timeout"]+22+reserve+2<=deadline
    waited=next(item for item in events if item["kind"]=="wait_terminal")
    assert waited["monotonic_deadline"]<=100000+deadline-reserve-2
    for read in events[2:]:
        if read["kind"]=="accounting":
            assert read["wall"]+read["timeout"]<=result["terminal_deadline"]
    return dict(phase=phase,state="BOUNDED_STEP",initial_remaining=deadline-now,slurm_time=requested,
                client_timeout=launched["timeout"],terminal_wait=max(0,waited["monotonic_deadline"]-100000-waited["wall"]),
                reserves=reserve,events=events)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rows=[
        scenario("gpu",100,1900),
        scenario("gpu",1000,1200),
        scenario("pre",1000,1200),
        scenario("post",1110,1200),
        scenario("post",1110,1200,return_delay=15),
        scenario("gpu",1100,1200,reject=True),
        scenario("post",1201,1200,reject=True),
        scenario("gpu",1000,1188,accounting_delay=10,reject=True),
    ]
    result=dict(state="ACTUAL_STEP_DEADLINE_CPU_PASS",scenarios=rows,actual_Slurm_or_DCU=0,
                fixed_epoch_preserved=True,caps_increased=False,geometry_retested=False)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(state=result["state"],scenarios=len(rows),no_new_step=sum(r["state"]=="NO_NEW_STEP" for r in rows),
                          early_GPU_slurm=rows[0]["slurm_time"],late_GPU_slurm=rows[1]["slurm_time"],
                          delayed_post_terminal_wait=rows[4]["terminal_wait"],actual_Slurm_or_DCU=0)))


if __name__=="__main__":
    main()
