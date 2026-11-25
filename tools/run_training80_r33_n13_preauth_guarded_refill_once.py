"""Admit one new f12r3n13 standby after a proven pre-authentication no-send."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from types import FunctionType


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local-only"
NODE, PARENT, PREVIOUS = "f12r3n13", "1194900", "1194900.1294"
STEM = "training80-r33-n13-preauth-guarded-bf16-v1"
WRAPPER = ROOT / "tools/run_training80_r33_n13_preauth_guarded_refill_once.py"
CLAIM = ROOT / "coordination/tasks" / (STEM + ".json")
BASE = ROOT / "tools/run_training80_r25_last40_ordinary_guarded_refill_once.py"
BASE_SHA = "575ecd2d9d08f300e287a77c0b25fdeb3a6a48f2ed2135273dbe390bdd16c88e"
OLD_SCOPE = "training80-r25-last40-ordinary-guarded-bf16-v1"
OLD_REQUEST = "9f745ad245535e3a810386ac2e9f8feb"
REQUEST = "91d97b76cb94b1b77dfceda5a85e070b"
OLD_REPLY = LOCAL / f"{OLD_SCOPE}-{NODE}-{OLD_REQUEST}-reply.json"
OLD_REPLY_SHA = "fb5bd9ce14b6f5c5b84c700bfa1507199ad598a57fc8442918b6041c585559f9"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def pin(path, expected):
    raw = path.read_bytes()
    require(path.resolve(strict=True) == path and sha256(raw).hexdigest() == expected,
            "exact previously reviewed source or original no-send receipt changed")
    return raw


def prove_no_send(receipt, row):
    transport = receipt["transport"]
    require((row["parent"], row["previous"], row["new_request"]) == (PARENT, PREVIOUS, OLD_REQUEST)
            and receipt["node"] == NODE and receipt["request"] == OLD_REQUEST
            and receipt["GPU_requested"] == 8 and receipt["replay_allowed"] is False,
            "original target or ordinary predecessor differs")
    require(transport["remote_outcome"] == "PAYLOAD_NOT_SENT" and transport["response"] is None
            and all(transport[key] is False for key in ("authority_verified", "payload_send_attempted",
                                                        "payload_flushed", "remote_received")),
            "original delivery was attempted or remains unknown")


def expected_claim():
    return dict(scope=STEM, state="ORIGINAL_PREAUTH_NO_SEND_FRESH_DEVICE_CHECK_REQUIRED",
                launcher=WRAPPER.relative_to(ROOT).as_posix(), node=NODE, parent=PARENT, predecessor=PREVIOUS,
                old_request=OLD_REQUEST, request=REQUEST, old_local_reply_SHA256=OLD_REPLY_SHA,
                source_dependency_SHA256=BASE_SHA, GPU_per_target=8, Slurm_cap_seconds=11100,
                max_invocations=1, ledger_mutations=0, replay_allowed=False)


def bound_api():
    pin(BASE, BASE_SHA)
    receipt = json.loads(pin(OLD_REPLY, OLD_REPLY_SHA))
    original = runpy.run_path(str(BASE))["bound_api"]()
    frozen = original["load"]()
    row = frozen[-2][NODE]
    prove_no_send(receipt, row)
    rows = {NODE: dict(row, new_request=REQUEST)}
    env = dict(original, __file__=str(WRAPPER), STEM=STEM, WRAPPER=WRAPPER, CLAIM=CLAIM, SPEC=rows)
    for name, function in original.items():
        if isinstance(function, FunctionType) and function.__globals__ is original:
            env[name] = FunctionType(function.__code__, env, name, function.__defaults__, function.__closure__)
            env[name].__kwdefaults__ = function.__kwdefaults__
    inherited_render = env["render"]

    def load(draft=False):
        raw = original["pin"](CLAIM)
        require(json.loads(raw) == expected_claim(), "single published n13 scope differs")
        return (*frozen[:-2], rows, sha256(raw).hexdigest())

    def render(context, row, account, terminal_pin, revision):
        result = inherited_render(context, row, account, terminal_pin, revision)
        launch = result["launch"]
        old_audit = f"{OLD_SCOPE}-{NODE}-{OLD_REQUEST}"
        old_worker = f"training80-dcu-standby-{OLD_REQUEST}"
        old_cpu_label = f"training80-r25-fresh-{OLD_REQUEST[:12]}"
        old_gpu_label = f"training80-dcu-standby-{OLD_REQUEST[:12]}-{NODE}"
        anchor = '            check_previous(account(PREVIOUS))'
        absent = (f"\n            require(not os.path.lexists(RESULTS / {old_audit!r})"
                  f" and not os.path.lexists(RESULTS / {old_worker!r}), 'pre-auth original remote namespace exists')")
        launch = original["unique"](launch, anchor, anchor + absent)
        launch = original["unique"](launch, '([old_label], [gpu_label])',
                                    f"([old_label], [gpu_label], [{old_cpu_label!r}], [{old_gpu_label!r}])")
        require(launch.index('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)') < launch.index(absent)
                < launch.index('outcome["zero_GRES_call_attempted"] = True')
                < launch.index('require(observation.returncode == 0 and not observation.stderr and 0 < len(observation.stdout) <= 8192,'),
                "original lock and fresh strict zero-GRES check are missing")
        compile(launch, "<n13 guarded standby>", "exec")
        result["launch"] = launch
        return result

    def publication(context, revision):
        require(Path(__file__).resolve() == WRAPPER, "exact installed n13 wrapper required")
        original["publication"](context, revision)
        _, publisher = context[3][-2]["modules"]()
        for path in (WRAPPER, CLAIM, BASE):
            ref = revision + ":" + path.relative_to(ROOT).as_posix()
            require(publisher["git"]("cat-file", "blob", ref) == original["pin"](path),
                    "published n13 successor differs")

    env.update(load=load, render=render, publication=publication)
    return env


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-claim"]:
        bound_api()
        with CLAIM.open("x", encoding="utf8", newline="\n") as stream:
            stream.write(json.dumps(expected_claim(), indent=2, sort_keys=True) + "\n")
    else:
        bound_api()["main"]()
