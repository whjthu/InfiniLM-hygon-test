"""Capture the real staged owner with no external programs or cluster state."""

from datetime import datetime,timezone
from io import BytesIO
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import owner as entry
from build_packet import build
from contract import (ASSETS_SHA, CACHE_PREFIX_NAME, FRAMEWORK_CACHE_ALLOWLIST, PREFIX, WHOLE, cache_prefix,
    digest, encode, framework_inventory, original_evidence, placement, save, small, stream_check)
from device import hip_gate, observe, trainer_argv
from fixture_1251 import staged_owner_consumes
from install import install
from owner import Owner, config, drive, reconcile


ROOT=Path(__file__).resolve().parents[3]
LABEL_NODE="f00r0n00"
EPOCH=1795586400
PARENT="1194900"


def reject(call, reason):
    try:
        call()
    except (OSError,ValueError) as error:
        assert isinstance(error,FileExistsError) or str(error)==reason,(str(error),reason)
        return reason
    raise AssertionError(reason)


def iso(epoch):
    return datetime.fromtimestamp(epoch,timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def intent_for(request, source_sha):
    return dict(request=request,parent=PARENT,node=LABEL_NODE,publication="a"*40,port=29968,
                DCU=8,CPU=64,memory_MiB=224000,whole_s=WHOLE,GPU_GO=True,exclusive_node=True,
                replay_allowed=False,source_manifest_SHA256=source_sha,label=PREFIX+request,epoch=EPOCH+3,
                deadline=EPOCH+3+WHOLE)


def authority_for(intent):
    return {**{key:intent[key] for key in ("request","parent","node","publication")},
            "coordinator_evidence_SHA256":"c"*64,"exclusive_no_other_dispatch":True,
            "legacy_bridge_state":"VERIFIED_TERMINAL_AND_RELEASED","ordinary_reservation_reconciled":True,
            "observed_epoch":EPOCH,"expires_epoch":EPOCH+WHOLE+60}


class Capture:
    def __init__(self,state):
        self.state=state
        self.rows={}
        self.sends=[]
        self.cards=[f"0000:{i:02x}:00.0" for i in range(8)]
        self.nodes=[LABEL_NODE]+[f"f00r1n{i:02d}" for i in range(39)]

    def query(self,*args):
        if args[0]=="sacct":
            return ["|".join(row) for row in self.rows.values()]
        if args[:3]==("scontrol","show","job"):
            return [f"JobId={PARENT} JobState=RUNNING NumNodes=40 NodeList=whole UserId=fixture(7) "
                    f"AllocTRES=gres/dcu=320 EndTime={iso(EPOCH+2000)}"]
        if args[:3]==("scontrol","show","node"):
            return [f"NodeName={LABEL_NODE} State=ALLOCATED CPUTot=64 RealMemory=260000"]
        if args[:3]==("scontrol","show","hostnames"):
            return self.nodes if args[-1]=="whole" else [args[-1]]
        if args[:4]==("scontrol","-d","show","job"):
            return ["Nodes=whole CPU_IDs=0-63 Mem=224000M"]
        if args[:3]==("scontrol","show","step"):
            return [f"StepId={PARENT}.25 NodeList={self.nodes[-1]} State=RUNNING"]
        if args[:3]==("squeue","--steps","-h"):
            return [f"{PARENT}.25|{self.nodes[-1]}"]
        if args[:3]==("squeue","-h","-j"):
            return [f"{PARENT}|RUNNING|whole"]
        raise AssertionError(args)

    def launch(self,argv,run,phase,budget):
        assert (run/(phase+"-intent.json")).exists()
        assert argv[0]=="srun" and "--jobid="+PARENT in argv and "--nodelist="+LABEL_NODE in argv
        self.sends.append((phase,argv))
        if phase=="gpu":
            assert "--gres=dcu:8" in argv and "--mem=224000M" in argv and "--cpus-per-task=64" in argv
            assert "--rocm" in argv and "source /opt/dtk/env.sh" in argv[-1] and "device.py" in argv[-1]
            assert "PYTHONPYCACHEPREFIX=" in argv[-1] and CACHE_PREFIX_NAME in argv[-1]
            assert all("dev" not in mount for mount in argv if mount.endswith(":ro"))
            if self.state=="UNKNOWN":
                return 255
        else:
            assert "--gres=none" in argv and "device.py" in " ".join(argv) and "--rocm" not in argv
            assert "-X" in argv and any(option.startswith("pycache_prefix=") and CACHE_PREFIX_NAME in option
                                       for option in argv)
        sequence=("pre","gpu","post").index(phase)
        timestamp=EPOCH+5+sequence*5
        target=f"{PARENT}.{880+sequence}"
        state,rc=("FAILED","1:0") if phase=="gpu" and self.state=="FAIL" else ("COMPLETED","0:0")
        self.rows[phase]=[target,run.name+"-"+phase,state,rc,"1",LABEL_NODE,
            "cpu=64,gres/dcu=8,mem=224000M,node=1" if phase=="gpu" else "cpu=2,mem=4G,node=1",
            iso(timestamp),iso(timestamp+1)]
        if phase!="gpu":
            snap=lambda leased:dict(step=target,node=LABEL_NODE,pci=self.cards,boot_id="oneboot",
                monotonic_ns=timestamp*1000000000,full_fd_inventory_performed=True,kfd_holder_count=0)
            output=observe(run,dict(port=29968,request=run.name.removeprefix(PREFIX)),phase,snapshot=snap,
                           check_port=lambda _:None,processes=lambda _:[])
            (run/(phase+".log")).write_bytes(encode(output))
        return int(rc!="0:0")


def check():
    raw,sha=build(ROOT)
    denials=[]
    captures={}
    with TemporaryDirectory(dir=ROOT/"local-only",prefix="qwen-eight-fixture-") as tmp:
        base=Path(tmp).resolve()
        assert raw==build(ROOT)[0]
        denials.append(reject(lambda:install(base,"a"*32,raw,"0"*64),"EXACT_SOURCE_PACKET_MANIFEST"))
        absolute=BytesIO()
        with ZipFile(BytesIO(raw)) as original,ZipFile(absolute,"w") as changed:
            for name in original.namelist():
                changed.writestr("/owner.py" if name=="owner.py" else name,original.read(name))
        denials.append(reject(lambda:install(base,"a"*32,absolute.getvalue(),sha),"EXACT_SOURCE_PACKET_FILE_SET"))
        denials.append(reject(lambda:install(base,"19a0e8677692481ead4c75f76a0361b9",raw,sha),"EXACT_SOURCE_INSTALL_REQUEST"))
        install(base,"f"*32,raw,sha)
        wire=staged_owner_consumes(base/(PREFIX+"f"*32),sha,ROOT,base)
        framework=base/"framework"
        framework.mkdir()
        required=framework/"pinned.py"
        required.write_bytes(b"original")
        pinned={"pinned.py":{}}
        assert framework_inventory(framework,pinned)==[]
        paths=[]
        for name in sorted(FRAMEWORK_CACHE_ALLOWLIST):
            current=framework/name
            current.parent.mkdir(parents=True,exist_ok=True)
            current.write_bytes(b"cache ignored by runtime")
            paths.append(current)
        assert framework_inventory(framework,pinned)==sorted(FRAMEWORK_CACHE_ALLOWLIST)
        added=framework/"foreign.py"
        added.write_bytes(b"unexpected")
        denials.append(reject(lambda:framework_inventory(framework,pinned),"EXACT_220_FRAMEWORK_393_WITH_ONLY_OBSERVED_CACHES"))
        added.unlink()
        required.unlink()
        denials.append(reject(lambda:framework_inventory(framework,pinned),"EXACT_220_FRAMEWORK_393_WITH_ONLY_OBSERVED_CACHES"))
        required.write_bytes(b"original")
        paths[-1].unlink()
        denials.append(reject(lambda:framework_inventory(framework,pinned),"EXACT_220_FRAMEWORK_393_WITH_ONLY_OBSERVED_CACHES"))
        cache_root=base/(PREFIX+"f"*32)
        cache=cache_prefix(cache_root)
        module=base/"cache_probe.py"
        module.write_bytes(b'VALUE="cached"\n')
        original_stat=module.stat()
        py_compile.compile(str(module),doraise=True)
        module.write_bytes(b'VALUE="source"\n')
        os.utime(module,ns=(original_stat.st_atime_ns,original_stat.st_mtime_ns))
        probe=[ "-c","import sys; sys.path.insert(0,sys.argv[1]); import cache_probe; print(cache_probe.VALUE)",str(base)]
        assert subprocess.check_output([sys.executable,"-I","-B",*probe]).strip()==b"cached"
        assert subprocess.check_output([sys.executable,"-I","-B","-X","pycache_prefix="+str(cache),*probe]).strip()==b"source"
        assert not cache.exists()
        cache.mkdir()
        denials.append(reject(lambda:cache_prefix(cache_root),"EXACT_UNUSED_PRIVATE_PYCACHE_PREFIX"))
        for request,case in (("a"*32,"PASS"),("b"*32,"FAIL"),("c"*32,"UNKNOWN")):
            result=install(base,request,raw,sha)
            run=base/(PREFIX+request)
            denials.append(reject(lambda:install(base,request,raw,sha),"DUPLICATE_INSTALL"))
            assert len(config(run,sha))==result["source_files"]==20
            assert original_evidence(run/"src")["accelerate.data_loader.BatchSamplerShard"].startswith("e08c5")
            assert "hipGetDeviceCount" in hip_gate(run).__code__.co_names
            assert digest(small(run/"src/assets220.json"))==ASSETS_SHA
            intent=intent_for(request,sha)
            argv=trainer_argv(run,intent,"f"*64)
            assert Path(argv[0]).as_posix().endswith("src/trainer/native_entry.py") and argv[argv.index("--maximum")+1]=="1320"
            authority=authority_for(intent)
            placement(intent,authority,EPOCH+3)
            denials.append(reject(lambda:placement(intent,dict(authority,legacy_bridge_state="UNKNOWN"),EPOCH+3),
                                  "FRESH_AUTHORITY_AND_RESOLVED_LEGACY_BRIDGE_REQUIRED"))
            save(run/"intent.json",intent)
            fixture=Capture(case)
            owner=Owner(run,intent,query_scheduler=fixture.query,send=fixture.launch,
                        clock=lambda:EPOCH+3,mono=lambda:10.)
            wait=owner.accounting["wait_terminal"]
            owner.accounting["wait_terminal"]=lambda read,deadline,**kw:wait(
                read,deadline,interval=1,max_observations=1,monotonic=lambda:10.)
            end=drive(owner,authority,uid=7)
            phases=[phase for phase,_ in fixture.sends]
            assert phases==(["pre","gpu"] if case=="UNKNOWN" else ["pre","gpu","post"])
            assert end["release"]==(case!="UNKNOWN") and end["planned_optimizer_updates"]==2
            assert end["state"]==("GPU_TERMINAL_UNKNOWN" if case=="UNKNOWN" else "GPU_TERMINAL_AND_INDEPENDENT_CLEAN_POST")
            assert reconcile(run,intent,fixture.query)==end
            denials.append(reject(lambda:owner.phase("gpu"),"EXACT_PHASE_ALREADY_SEEN_NO_REPLAY"))
            assert [phase for phase,_ in fixture.sends].count("gpu")==1
            if case!="UNKNOWN":
                after=run/"after.json"
                known=json.loads(after.read_bytes())
                for change in ({"boot_id":"secondhost"},{"monotonic_ns":1}):
                    after.write_bytes(encode(dict(known,**change)))
                    denials.append(reject(lambda:reconcile(run,intent,fixture.query),
                                          "EXACT_NUMERIC_GPU_THEN_SAME_BOOT_CLEAN_POST"))
                after.write_bytes(encode(dict(known,kfd_holder_count=1)))
                denials.append(reject(lambda:reconcile(run,intent,fixture.query),"EXACT_PHYSICAL_PRE_OR_POST"))
                after.write_bytes(encode(known))
            captures[case]={"phase_names":phases,"GPU_offer_count":phases.count("gpu"),"independent_release":end["release"]}
        asset=base/"asset-pin"
        asset.write_bytes(b"payload")
        assert stream_check(asset,digest(b"payload"),7)[2]==7
        denials.append(reject(lambda:stream_check(asset,"a"*64,7),"EXACT_ASSET_BYTES"))
        detached={}
        for request,fails in (("d"*32,False),("e"*32,True)):
            install(base,request,raw,sha)
            run=base/(PREFIX+request)
            intent=intent_for(request,sha)
            proof=authority_for(intent)
            called=[]
            def launch(argv,**options):
                assert argv[:4]==["/usr/bin/timeout","--kill-after=10s","1710s","/usr/local/python3.12/bin/python3"]
                assert "-X" in argv and "pycache_prefix="+str(run/CACHE_PREFIX_NAME) in argv
                assert "start" in argv and "--from-keeper" in argv and options["start_new_session"]
                assert (run/"keeper-intent.json").is_file()
                assert Path(options["stdout"].name).name=="keeper.stdout" and Path(options["stderr"].name).name=="keeper.stderr"
                called.append(1)
                if fails:
                    raise OSError("fixture wait/transport")
                return SimpleNamespace(pid=71)
            args=(run,base/"scope","a"*64,base/"authority","b"*64,base/"proof","c"*64,sha)
            with patch.object(entry,"prepared",return_value=(intent,proof)):
                kwargs=dict(clock=lambda:EPOCH+3,launch=launch,read_scheduler=lambda *a:[],
                            identity=lambda pid:dict(PID=pid,PGID=pid,startticks=5,hostname="fixture",boot_SHA256="d"*64))
                result=entry.detach(*args,**kwargs)
                denials.append(reject(lambda:entry.detach(*args,**kwargs),"DETACHED_OWNER_ALREADY_OWNS_REQUEST"))
            expected="DETACHED_OWNER_START_UNKNOWN" if fails else "EXACT_DETACHED_OWNER_STARTED"
            assert result["state"]==expected and len(called)==1 and result["deadline"]==EPOCH+3+1720
            state=entry.status(run,sha,lambda *a:[])
            assert state["state"]=="DETACHED_OWNER_PRESTART_OR_TRANSPORT_UNKNOWN" and not state["release"]
            assert not (run/"gpu-intent.json").exists()
            detached[expected]=1
    return {"state":"EXACT_STAGED_OWNER_AND_RUNTIME_ARGV_FIXTURES_CHECKED","source_manifest_SHA256":sha,
            "source_packet_SHA256":digest(raw),"source_packet_bytes":len(raw),"source_members":20,
            "captures":captures,"detached_keeper_fixture":detached,"rejections":denials,"frozen_checker_wire":wire,
            "exact_framework_cache_files_quarantined":len(FRAMEWORK_CACHE_ALLOWLIST),
            "actual_cpython_prefix_ignored_existing_timestamp_hit_and_did_not_write":True,
            "actual_scheduler_or_target_calls":0,"Git":0,"GPU":0}


if __name__=="__main__":
    print(json.dumps(check(),sort_keys=True))
