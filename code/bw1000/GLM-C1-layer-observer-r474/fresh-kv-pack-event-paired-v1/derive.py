"""Assemble the new event scope with the qualified GLM numeric launch boundary."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AREA = Path("code/bw1000/GLM-C1-layer-observer-r474")
OLD = AREA / "fresh-kv-pack-device-numeric-v2"
NEW = HERE.relative_to(ROOT)
NUMERIC = Path("results/bw1000/2026-11-22/GLM474-fresh-kv-pack-v2-original941-device-numerics-terminal.json")
NUMERIC_SHA = "240dcb8986383f3d786264a96828f008062f977ecc72feba1343a0508c726628"
OLD_LABEL = "GLM474-fresh-kv-pack-numeric-GPU-v2"
OLD_REQUEST = "93148263aba346dda17b534d419d81c83"
LABEL = "GLM474-fresh-kv-pack-event-paired-GPU-v1"
REQUEST = "7c6ec3205bdf41578b0a6d1698787f13"
PINS = {
    "probe.py": "a9af94ebc3c016cd3174adbe9e48485258968ea5d021ac9dfaaa7c1f62d39818",
    "host_once.py": "a0fd57dd8a9e7847cb29d1c68581cc7a60cf085f80a02db42d697752a0cf2053",
    "device_inside.sh": "bbb84d0e3b1dcffdfa4873130f2466ed65dabb6d12ff5eeaf6d35ed1868e9a10",
    "CLAIM.json": "c6e91a971e21c76ea8a212eea16c609d7bc360d1d9311bee3f64a4bae753f2be",
}


def digest(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def check(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("frozen source differs: " + str(path))
    return raw


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("frozen source context differs: " + old)
    return text.replace(old, new, 1)


def scope(text):
    return once(once(text, OLD_LABEL, LABEL), OLD_REQUEST, REQUEST)


def cut(text, begin, end, replacement):
    opening = text.index(begin)
    finish = text.index(end, opening)
    return once(text, text[opening:finish], replacement)


def probe(text, plan, timing_sha):
    text = scope(text)
    text = once(text, "ARCHIVE_SHA = ", f'NUMERIC = Path("{NUMERIC.as_posix()}")\nNUMERIC_SHA = "{NUMERIC_SHA}"\n'
                f'TIMING = Path("{(NEW / "timing.py").as_posix()}")\nTIMING_SHA = "{timing_sha}"\nARCHIVE_SHA = ')
    text = once(text, "SEEDS = (17, 83)\n", "")
    text = once(text, 'SIZES = (2048, 2048, 3584, 128)\nBAD_FIELDS = ("key", "value", "projected", "rotary", "stream")\n', "")
    text = once(text, "from contextlib import ExitStack\n", "")
    text = once(text, "import struct\n", "")
    text = cut(text, "def fixture(seed):", "def main():", f"PLANNED = {plan!r}\n\n\n")
    text = cut(text, "        def progress(", "        boot_id =", "")
    text = once(text, '            plan = json.loads(read_pin(SOURCE / PLAN, PLAN_SHA, 1_439_584))',
                '''            numeric = json.loads(read_pin(SOURCE / NUMERIC, NUMERIC_SHA, 16_384))
            need(numeric["state"] == "GPU_NUMERICAL_PASS_POST_CLEAN"
                 and numeric["accounting"]["gpu"][:4] == ["1103606.941", "GLM474-fresh-kv-pack-numeric-GPU-v2", "COMPLETED", "0:0"]
                 and numeric["independent_post_accounting"][0] == "1103606.943"
                 and numeric["passed_valid"] == 4 and numeric["passed_argument_rejections"] == 36
                 and numeric["provider_paths_and_SHA256_checked"] == 71, "qualified original device precedent differs")
            timing = load_helper(TIMING, TIMING_SHA, "fresh_kv_timing")
            need(timing.PLAN == PLANNED, "fixed timing denominator differs")
            plan = json.loads(read_pin(SOURCE / PLAN, PLAN_SHA, 1_439_584))''')
    return cut(text, "            with ExitStack() as cleanup:", "            remaining(scope)\n            write",
               "            during = timing.run(hip, guarded, exports, lambda: remaining(scope), write)\n")


def host(text, output, result_sha, timing_sha):
    text = once(scope(text), 'DEVICE = AREA / "fresh-kv-pack-device-numeric-v2"', f'DEVICE = Path("{NEW.as_posix()}")')
    text = once(text, 'RESULT_READER = Path("code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-pack-device-host-result-v2/inspect_result.py")',
                'RESULT_READER = DEVICE / "result.py"\nTIMING = DEVICE / "timing.py"')
    text = once(text, 'RESULT_READER: "266449c1cabe928a33d4d6e2f46421e0c7e9969a49fd3efe33a3ad4dde2ced5c",',
                f'RESULT_READER: "{result_sha}",\n    TIMING: "{timing_sha}",\n    Path("{NUMERIC.as_posix()}"): "{NUMERIC_SHA}",')
    for name in ("probe.py", "device_inside.sh", "CLAIM.json"):
        text = once(text, PINS[name], digest(output[name]))
    text = cut(text, "def inspect_probe(", "def launch(", "")
    text = once(text, "        host = SimpleNamespace(inspect_probe=inspect_probe, bounded=bounded, own_identity=own_identity, need=need)\n"
                '        observed = reader.inspect_probe(run / "probe.jsonl", gpu["row"], intent_sha, before, probe, host)',
                '''        timing = load(TIMING, pin(checkout, TIMING, PINS[TIMING]))
        host = SimpleNamespace(bounded=bounded, own_identity=own_identity, need=need)
        observed = reader.inspect_probe(run / "probe.jsonl", gpu["row"], intent_sha, before, probe, timing, host)''')
    text = once(text, "numerical_pass=False, Model_GO=False", "numerical_pass=False, component_performance_measured=False, Model_GO=False")
    return once(text, '        if result["numerical_pass"]:\n            result["state"] = "GPU_NUMERICAL_PASS_POST_CLEAN"',
                '''        if result["numerical_pass"]:
            result["component_performance_measured"] = True
            result["state"] = "GPU_COMPONENT_PAIRED_POST_CLEAN"''')


def render():
    if not re.fullmatch(r"[0-9a-f]{32}", REQUEST):
        raise ValueError("original event-pair request must be 32 lowercase hex digits")
    previous = {name: check(ROOT / OLD / name, pin) for name, pin in PINS.items()}
    check(ROOT / NUMERIC, NUMERIC_SHA)
    timing = runpy.run_path(str(HERE / "timing.py"))
    timing_sha = digest((HERE / "timing.py").read_bytes())
    result_sha = digest((HERE / "result.py").read_bytes())
    output = {
        "probe.py": probe(previous["probe.py"].decode(), timing["PLAN"], timing_sha).encode(),
        "device_inside.sh": once(previous["device_inside.sh"].decode(), OLD.name, NEW.name).encode(),
    }
    claim = json.loads(previous["CLAIM.json"])
    claim.pop("predecessor")
    claim.pop("provider_path_repair")
    claim.update(label=LABEL, request=REQUEST, operator_GO=True, local_only_preparation=True)
    claim["qualified_device_precedent"] = dict(path=NUMERIC.as_posix(), SHA256=NUMERIC_SHA, GPU="1103606.941",
                                              independent_post="1103606.943", valid=4, rejection=36)
    claim["fixture"] = dict(kind="controlled finite BF16 packing input", tokens=[1, 512], heads=4,
                            T1_role="native TP16 decode layout", T512_role="prefill-sized kernel component proxy",
                            projected=["T",4,448], rotary=["T",1,64], K_V_each=["T",4,256],
                            strides=dict(projected=[1792,448,1], rotary=[64,64,1], K_V_each=[1024,256,1]),
                            warm_per_arm=timing["WARM"], timed_per_tokens=timing["TIMED"], planned=timing["PLAN"],
                            paired_memory="same total capacity and exact equal reported free bytes before and after matched arms",
                            copies_measured="no input upload, canary, CPU reference or D2H in event interval")
    claim["new_files_SHA256"] = {name: digest(raw) for name, raw in output.items()}
    claim["new_files_SHA256"].update({"timing.py":timing_sha, "result.py": result_sha})
    output["CLAIM.json"] = encode(claim)
    output["host_once.py"] = host(previous["host_once.py"].decode(), output, result_sha, timing_sha).encode()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-generated", action="store_true")
    args = parser.parse_args()
    for name, raw in sorted(render().items()):
        path = HERE / name
        if args.write_generated:
            with path.open("xb") as output:
                output.write(raw)
        elif path.read_bytes() != raw:
            raise ValueError("generated source differs: " + name)
        print(digest(raw), name)


if __name__ == "__main__":
    main()
