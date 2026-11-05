"""Derive a private C4 host-split entry from the original r467 and r449 sources."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

PINS = {
    "entry.py": "326cd974e9422cf0ed4b23cf7fee99848534e24f79b488d670ac975f2dfcb138",
    "profile_control.py": "123f27a47b8fb36721edb40fa9321591692bb78554eb94e85ff4736c93a1249d",
    "precision.py": "7443cb788ed2bf7241d0fc64042a1b5def2d692ef66fad8d90d2cca46dd96772",
    "existing_profile_consumer.py": "d5093cd4b4cccb52f6986a9259593dd0d2b51030017bf65116d75c99ae1da1ff",
    "forward_policy0.py": "4f3f6a5a9cf876091e1cab01a36c150dc8f5d48397ef9f3c0418b9bf62bdba2f",
    "native_context.py": "2ba9412b5676c4b2074b4a60f2041db1023a1890c6fcd16dea659650e3d12c4d",
    "config.template.json": "56afc424ef9115329f2b7f0b0c22a7f66dcce32b465ffbdab71469ab11185736",
}
LOCAL = ("candidate_binding.py", "split_capture.py", "consume.py")


def update(text, replacements):
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError(f"Missing or ambiguous source anchor: {before}")
        text = text.replace(before, after, 1)
    return text


def derive(r467, r449, local):
    files = {}
    for name, pin in PINS.items():
        path = (r449 / "public_deep_model" if name == "native_context.py" else r467) / name
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != pin:
            raise ValueError(f"Original source differs: {path}")
        files[name] = raw.decode()
    for name in LOCAL:
        files[name] = (local / name).read_bytes().decode()
    files["entry.py"] = update(files["entry.py"], (
        ("NATIVE='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'",
         "from candidate_binding import NATIVE, LIBRARY, receipt_from_worker, bind_library"),
        ("c['Native_SHA256']==NATIVE and", "c['Native_SHA256']==NATIVE and c['Native_library']==LIBRARY.as_posix() and"),
        ("c['samples']==[8,87]", "c['samples']==[11,12]"),
        (" from public_deep_model.native_context import NativeContext", " from native_context import NativeContext"),
        ("_,receipt,_=worker.verify_native_chain();assert receipt['library_SHA256']==NATIVE",
         "receipt=receipt_from_worker(worker);assert receipt['library_SHA256']==NATIVE"),
        ("binding=worker.bind_library(", "binding=bind_library("),
        ("'deep_attention_inner_read_r9193']};save()", "'deep_attention_inner_read_r9193','deep_c4_host_split_read_v1']};save()"),
    ))
    files["precision.py"] = update(files["precision.py"], ((
        "NATIVE_SHA='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'",
        "from candidate_binding import NATIVE as NATIVE_SHA"),))
    files["native_context.py"] = update(files["native_context.py"], (
        ("from .checkpoint_loader import load_checkpoint,NATIVE_SHA",
         "from public_deep_model.checkpoint_loader import load_checkpoint\nfrom candidate_binding import NATIVE as NATIVE_SHA"),
        ("from .common_mapping_audit", "from public_deep_model.common_mapping_audit"),
        ("from .model_affinity_check_r2628", "from public_deep_model.model_affinity_check_r2628"),
        ("from .forward import make_forward", "from public_deep_model.forward import make_forward"),
    ))
    files["profile_control.py"] = update(files["profile_control.py"], (
        ("MODES=(False,False,False,True,False)",
         "from consume import bind as bind_split\nfrom split_capture import consume_window\nMODES=(False,False,False,True,False)"),
        ("SAMPLES=(8,87)", "SAMPLES=(11,12)"),
        (";bind_inner(lib)\n", ";bind_inner(lib);self.read_split=bind_split(lib)\n"),
        ("   except BaseException as error:\n    self.poisoned=True;",
         "    q['C4_host_split']=consume_window(self.read_split,q,self.output/f'request{request_index}-decode{decode_index}-c4-host.json')\n"
         "   except BaseException as error:\n    self.poisoned=True;"),
    ))
    c = json.loads(files["config.template.json"])
    c.update(Native_library="/refactor/candidates/Deep-C4-current-host-split-CPU-v1/libDeep_c4_host_split_v1.so",
             Native_SHA256="1b729a2d29d39f084addc4978fbe3db7bd4b685360c384f7dfe2ae8908fc9e7f",
             samples=[11, 12], output_root="/refactor/results/Deep-C4-current-host-split/current-v1",
             run="c4-current-host-split-a")
    assert not c["execute_authorized"] and c["actual_step"] is None and c["gpu_start_epoch"] is None
    files["config.template.json"] = json.dumps(c, indent=2) + "\n"
    for name, raw in files.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r467", "r449", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    files = derive(args.r467, args.r449, Path(__file__).resolve().parent)
    args.output.mkdir()
    pins = {}
    for name, raw in files.items():
        data = raw.encode()
        with (args.output / name).open("xb") as stream:
            stream.write(data)
        pins[name] = sha256(data).hexdigest()
    with (args.output / "SOURCE_MANIFEST.json").open("x", encoding="utf8") as stream:
        json.dump(dict(files_SHA256=pins, CPU_GO=False, GPU_GO=False, Model_GO=False), stream, indent=2)
    print(json.dumps(pins))


if __name__ == "__main__":
    main()
