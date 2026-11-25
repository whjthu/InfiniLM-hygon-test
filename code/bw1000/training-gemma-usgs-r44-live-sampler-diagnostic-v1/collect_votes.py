"""Read the sealed production sampler reports without consuming inputs."""

from hashlib import sha256
from pathlib import Path
import re

from contract import INSIDE, REQUEST, ROOT, stable_json


DIAG="live-sampler-diagnostic.json"
WAIT="live-sampler-rendezvous.json"
SCHEMA="GEMMA_R43_LIVE_TRAIN_BEGIN_SAMPLER_PROJECTION_V1"
WAIT_SCHEMA="GEMMA_R43_BOUNDED_LOCAL_RANK_RECEIPT_RENDEZVOUS_V1"
FORBIDDEN=("initialization.json","target-fresh-initial.pt","target-fresh-rng.pt","result.json","native-sampler.json")


def collect(root,step,epoch,boot):
    train=root/"continuous/train"
    witness=train/"continuous-witness"
    try:
        _,cpin=stable_json(root/"continuous/witness-config.json",cap=65536)
    except (OSError,ValueError,TypeError) as error:
        return dict(request_id=REQUEST,step=step,expected_ranks=8,ranks={},missing=list(range(8)),invalid=[],
                    matching_full_rendezvous={},forbidden_output=[],all_eight_clean=False,
                    config_error=type(error).__name__)
    source_payload=f'{cpin["SHA256"]}:{step}:{float(epoch)!r}:{float(epoch+INSIDE)!r}'.encode()
    expected=sha256(source_payload+b":"+boot.encode("ascii")+b"\n").hexdigest()
    reports={}
    waiters={}
    actor_by_rank={}
    invalid=[]
    violations=[]
    for rank in range(8):
        directory=witness/f"rank-{rank:02d}"
        for name in FORBIDDEN:
            if (directory/name).exists() or (directory/name).is_symlink():
                violations.append((rank,name))
        path=directory/DIAG
        try:
            if path.resolve(strict=True)!=path:
                raise ValueError("R44_DIAGNOSTIC_SYMLINK")
            doc,pin=stable_json(path,cap=32768)
            actual=doc["rank_activation"]
            bits=(type(actual["PID"]) is int and actual["PID"]>0 and
                  type(actual["startticks"]) is int and actual["startticks"]>0)
            first_failed=doc["first_failed_check"]
            valid=(doc["schema"]==SCHEMA and type(doc["rank"]) is int and doc["rank"]==rank
                   and actual["step"]==step and actual["activation_SHA256"]==expected and bits
                   and doc["actual_production_callback"] is doc["diagnostic_only_stops_before_optimizer"] is True
                   and type(doc["optimizer_updates"]) is int and doc["optimizer_updates"]==0
                   and doc["live_loader_iterated"] is doc["live_sampler_iterated"] is False
                   and (first_failed is None or type(first_failed) is str
                        and re.fullmatch(r"[a-z][a-z0-9_]{0,99}",first_failed)))
            if not valid:
                raise ValueError("R44_DIAGNOSTIC_IDENTITY")
            actor_by_rank[rank]=(actual["PID"],actual["startticks"])
            reports[str(rank)]={"receipt":pin,"RNG_unchanged":doc["RNG_unchanged"],
                                "first_failed_check":first_failed}
        except FileNotFoundError:
            continue
        except (OSError,KeyError,TypeError,ValueError):
            invalid.append(rank)
            continue
        try:
            path=directory/WAIT
            if path.resolve(strict=True)!=path:
                raise ValueError("R44_WAIT_SYMLINK")
            joined,witness_pin=stable_json(path,cap=32768)
            if (joined["schema"]==WAIT_SCHEMA and joined["rank"]==rank and joined["step"]==step
                    and joined["activation_SHA256"]==expected and joined["expected_ranks"]==8
                    and joined["stop_after_receipt"] is True and joined["optimizer_updates"]==0):
                waiters[str(rank)]=(joined,witness_pin)
        except (OSError,KeyError,TypeError,ValueError):
            pass
    actors=[pid for pid,_ in actor_by_rank.values()]
    duplicates={k for k,(pid,_) in actor_by_rank.items() if actors.count(pid)>1}
    for rank in duplicates:
        reports.pop(str(rank),None)
    invalid=sorted(set(invalid)|duplicates)
    full_commits={}
    for rank,(doc,witness_pin) in waiters.items():
        if (not doc["missing"] and not doc["invalid"] and len(reports)==8 and doc["present"]==
                {expected_rank:value["receipt"]["SHA256"] for expected_rank,value in reports.items()}):
            full_commits[rank]=witness_pin
    if train.exists():
        for path in train.iterdir():
            if path.name.startswith("checkpoint-") or "final-adapter" in path.name:
                violations.append(("train",path.name if re.fullmatch(r"checkpoint-[0-9]{1,4}|r39-final-adapter",path.name) else "other_adapter"))
    return dict(request_id=REQUEST,step=step,activation_SHA256=expected,expected_ranks=8,ranks=reports,
                missing=[k for k in range(8) if str(k) not in reports and k not in invalid],invalid=invalid,
                matching_full_rendezvous=full_commits,forbidden_output=violations,
                all_eight_clean=bool(len(reports)==8 and full_commits and not violations
                                     and all(doc["RNG_unchanged"] is True for doc in reports.values())))
