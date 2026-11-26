"""Exercise the actual source renderer and carrier preparation with synthetic live evidence."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import admission as a
import prepare as p


def check():
    stable,schema=p.stable_members()
    assert len(stable)==32
    static=p.source_manifest(stable,schema)
    base_pin=a.pin(static)
    official=p.read(p.HISTORY,a.HISTORY,8192)
    history=json.loads(official)
    boot="12345678-0000-0000-0000-000000000001"
    pci=[f"0000:0{i}:00.0" for i in range(8)]
    now=a.stamp("2026-11-27T00:00:10")
    at=now-1
    release=dict(request_id=a.OLD_SCOPE, node=a.NODE, parent=a.PARENT, independent=True,released=True,
                 same_boot_and_pci=True,open_kfd_holders=0,scope_owned_processes_remaining=0,
                 holder_accounting=history["holder_accounting"],post_accounting=history["post_accounting"],
                 UTC_accounting=dict(TZ="UTC",rows=[history["holder_accounting"],history["post_accounting"]]),
                 original_release_receipt_SHA256="1"*64,physical_projection_SHA256="2"*64)
    raw=a.encode(release)
    context=dict(schema="GEMMA_R120_NATURAL_RELEASE_AND_FRESH_N12_V1",scope=a.SCOPE,node=a.NODE,
        terminal=dict(step=a.HOLDER,accounting=release["holder_accounting"],worker_terminal_verified=True,
                      termination_kind="NATURAL",handoff_signal_delivered=False,worker_receipt_SHA256=a.WORKER_RECEIPT),
        post=dict(step=a.POST,receipt_SHA256="1"*64,independently_verified=True,kfd_holders=0,
                  owned_processes_remaining=0,boot_id=boot,pci=pci),
        fresh=dict(step="1103606.4191",observed_epoch=at,boot_id=boot,pci=list(pci),node=a.NODE,
                   all_relevant_fd_inventory_complete=True,kfd_holders=0,owned_processes_remaining=0,
                   current_workload_conflict_excluded=True,controller_and_queue_both_clear=True,receipt_SHA256="3"*64,
                   accounting_TZ="UTC",accounting=["1103606.4191","fixture","COMPLETED","0:0","1",a.NODE,
                      "cpu=1,mem=256M,node=1","2026-11-27T00:00:08","2026-11-27T00:00:09"],
                   ports={k:dict(port=a.PORT,clear=True) for k in ("tcp","tcp6","udp","udp6")}))
    fresh=a.encode(context)
    frozen=json.loads(stable["CLAIM.json"])
    rev="4"*40
    approval=dict(request_id=a.SCOPE,dispatch_request_id=a.DISPATCH,node=a.NODE,parent=a.PARENT,master_port=a.PORT,
         issued_epoch=at,authorize_until=at+65,valid_until=now+3000,publication=rev,stable_source_manifest=base_pin,
         original_release_SHA256=sha256(raw).hexdigest(),natural_release_gate_SHA256=sha256(fresh).hexdigest(),
         old_holder_step=a.HOLDER,latest_holder_accounting=release["holder_accounting"],
         approved_eight_card_single_attempt=True,ordinary_exclusion=True,coordinator_identity_verified=True,
         manual_adoption_authorized=True,prior_native_accounting_verified=True,
         release_requires_terminal_independent_post=True,latest_explicit_holder_verified_terminal=True,
         cpu_3481_independent_SHA256=frozen["exact_cpu_3481_receipt_SHA256"],
         sampler_source_manifest_SHA256=frozen["exact_swift_source_manifest_SHA256"],
         prior_diagnostic_terminal_SHA256=frozen["prior_diagnostic_terminal_SHA256"],
         zero_gres_source_SHA256=frozen["cleanup_CPU_source_SHA256"],zero_gres_request="5"*32,
         cpu_original_independent_terminal_SHA256="74bac702f6e94de63d3db93ad8744fffdb45b797881510792587ef76e415de02")
    raw_approval=a.encode(approval)
    def rejects(fn,expected):
        try: fn()
        except ValueError as e: assert str(e)==expected, e
        else: raise AssertionError("bad branch accepted")
    with TemporaryDirectory() as temp:
        root=Path(temp)
        outcome=p.prepare(root/"once",rev,base_pin["SHA256"],raw_approval,raw,fresh,now)
        assert outcome["wire_base64_bytes"]<=560000
        assert outcome["stable_source_manifest"]==base_pin
        assert len([n for n, b in stable.items() if b!=(root/"once"/"effective-source"/n).read_bytes()])==1
        source=(root/"once"/"HOST_PROGRAM.py").read_text()
        assert source.count("spawn.mkdir(mode=0o700)") == source.count("subprocess.Popen(") == 1
        assert source.index("assert rebuilt == (auth,proof)") < source.index("spawn.mkdir(mode=0o700)")
        assert '"-I"' not in source or "start_new_session=True" in source
        try: p.prepare(root/"once",rev,base_pin["SHA256"],raw_approval,raw,fresh,now)
        except FileExistsError: pass
        else: raise AssertionError("duplicate capsule reservation")
        rejects(lambda:p.prepare(root/"stale",rev,base_pin["SHA256"],raw_approval,raw,fresh,now+65),
                "CURRENT_SAME_BOOT_8PCI_FD_PORT_AND_BOTH_VIEWS_REQUIRED")
        assert not (root/"stale").exists()
        rejects(lambda:p.prepare(root/"wrong-manifest",rev,"6"*64,raw_approval,raw,fresh,now),
                "EXACT_PUBLISHED_STABLE_SOURCE_MANIFEST")
        for group,field,val,error in (
           ("terminal","termination_kind","HANDOFF_INTENT_ONLY","ORIGINAL_TERMINAL_OR_POST_PROJECTION_CHANGED"),
           ("fresh","pci",pci[:-1],"CURRENT_SAME_BOOT_8PCI_FD_PORT_AND_BOTH_VIEWS_REQUIRED"),
           ("fresh","controller_and_queue_both_clear",False,"CURRENT_SAME_BOOT_8PCI_FD_PORT_AND_BOTH_VIEWS_REQUIRED"),
           ("fresh","accounting_TZ","Asia/Shanghai","FRESH_ACTUAL_NUMBERED_CPU_ACCOUNTING_REQUIRED")):
            value=deepcopy(context)
            value[group][field]=val
            rejects(lambda:a.authorize(official,raw,a.encode(value),raw_approval,rev,base_pin,now),error)
    print(json.dumps(dict(validation="PASS",static_members=32,stable_manifest=base_pin,
                          actual_test_wire_bytes=outcome["wire_base64_bytes"],target_calls=0)))


if __name__=="__main__":
    check()
