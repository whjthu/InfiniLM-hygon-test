"""Bind the actual .1071 DSO, original complete input and checkpoint loader."""

import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace


MODEL = Path("/old/models/ZhipuAI-GLM-5.3")
CONFIG_SHA = "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69"
INDEX_SHA = "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf"
INDEX_KEYS, SHARDS, LOAD_KEYS, VOCAB = 118629, 141, 117060, 154880
PLAN = Path("/refactor/results/GLM-r474-C1-observer-screen32-v1-private-plan.json")
PLAN_SHA = "b9fd5bf7c18931854fc00bcf73ea9d82d8f9db5048a2750c4088f6f5a852c7d3"
INPUT_SHA = "81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e"
PROMPT_SHA = "b00487acfaeae81d1942d7df97678db30359d169290af61aad5d2198f69f6ac8"
CHECKPOINT_LOADER_SHA = "89fff1fb549b411f41c55c43570937765ff22ccf327c31c4023106f5af196ae7"
NATIVE = Path("/refactor/candidates/GLM474-fresh-kv-prefill512-Native-CPU-v2/libGLM_fresh_kv_pack.so")
NATIVE_SHA = "7f1e2c1846db146d00aa067eeaa3b2289fee5118fae733cf244c2a1008ac1dde"
NATIVE_SIZE = 1970800
RECIPE = NATIVE.with_name("BUILD_RECIPE.json")
RECIPE_SHA = "93c4f33891ad0938144d66f98fec458135ece7233e9e62b6b2e4fd28602b448d"
RECIPE_SIZE = 147409
NATIVE_CPU_ROW = ["1103606.1071","GLM474-fresh-kv-prefill512-Native-CPU-v2","COMPLETED","0:0","46","f12r2n13","cpu=4,mem=16G,node=1"]
NATIVE_CPU_EVIDENCE_SHA = "f900ed78353182039c737adf4739359ba8f4915bf51472af50df9f7131b5de1e"
TYPED_SHA = "7b7002acdb96b172acc5dfb4d1bfec33f3c8910deefdb7b5aae40511c6f09768"
ENTRY = "modern_glm_global_fresh_kv_prefill512_r474"
EXPECTED = {"id_size","id","create","error","keys","load_batch","finish","forward","control","reset","geometry","destroy",
            "decode_forward","prefill512_forward"}
OTHER_ARMS = {"libGLM_fresh_kv_pack.so","libGLM_C1_layer_observer_r474.so",
              "libGLM_depth_observer_B.so","libGLM_history_value_B.so"}


def need(ok, label):
    if not ok:
        raise ValueError(label)


def read(path, digest, bound, size=None):
    path = Path(path)
    need(path.resolve(strict=True) == path and path.is_file(), "unexpected file path: "+path.name)
    with path.open("rb") as stream:
        raw = stream.read(bound+1)
    need(0 < len(raw) <= bound and (size is None or len(raw) == size)
         and sha256(raw).hexdigest() == digest, "actual evidence SHA/size differs: "+path.name)
    return raw


def canonical_hash(value):
    return sha256(json.dumps(value,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def module(path, digest, limit):
    raw = read(path, digest, limit)
    loaded = ModuleType(Path(path).stem + "_verified")
    loaded.__file__ = str(path)
    exec(compile(raw,str(path),"exec"), loaded.__dict__)
    return loaded


def load_checkpoint_function(path):
    need(__debug__, "the original full checkpoint loader requires Python assertions")
    return module(path,CHECKPOINT_LOADER_SHA,8192).load_checkpoint


def model_configuration(model):
    cfg = json.loads(read(model/"config.json",CONFIG_SHA,16384))
    index = json.loads(read(model/"model.safetensors.index.json",INDEX_SHA,32*1024**2))
    mapping = index["weight_map"]
    need(len(mapping) == INDEX_KEYS and len(set(mapping.values())) == SHARDS
         and cfg.get("num_hidden_layers") == 78 and cfg.get("vocab_size") == VOCAB,
         "original GLM-5.3 checkpoint inventory or full model shape differs")
    cfg.update(glm_selected_value_packed_layer0=False,glm_global_tp16_r470=True,glm_global_tp16_Model_r470=True,
               glm_vocab_parallel_embedding_r471=True,glm_c1_layer_observer_r474=False)
    return cfg


def original_prompt(public, plan):
    need(public["model_path"] == MODEL.as_posix() and public["model_config_SHA256"] == CONFIG_SHA
         and public["model_index_SHA256"] == INDEX_SHA and public["model_index_keys"] == INDEX_KEYS
         and public["model_checkpoint_files"] == SHARDS and public["private_plan"]["SHA256"] == PLAN_SHA
         and public["private_plan"]["bytes"] == 817435, "original workload source binding differs")
    need(plan["planned"] == 32 and [row["ordinal"] for row in plan["requests"]] == public["selected_ordinals"],
         "original workload selection differs")
    plan_row = plan["requests"][0]
    public_row = public["private_plan"]["requests"][0]
    keys = ("ordinal","case_id","suite","prompt_tokens","prompt_IDs_SHA256","original_private_SHA256","body_SHA256",
            "max_new_tokens","prefill_window")
    need(all(plan_row[k] == public_row[k] for k in keys) and plan_row["ordinal"] == 1
         and plan_row["prompt_tokens"] == 1312 and plan_row["prefill_window"] == 4096
         and plan_row["max_new_tokens"] == 4096 and plan_row["body"]["temperature"] == 0.0
         and plan_row["prompt_IDs_SHA256"] == PROMPT_SHA, "fixed ordinal1 original prompt metadata differs")
    ids = plan_row["prompt_token_ids"]
    need(len(ids) == 1312 and all(type(value) is int and 0 <= value < VOCAB for value in ids)
         and canonical_hash(ids) == PROMPT_SHA, "exact original full prompt IDs differ")
    return tuple(ids)


def load_input(input_binding):
    public = json.loads(read(input_binding,INPUT_SHA,32768))
    plan = json.loads(read(PLAN,PLAN_SHA,817435,817435))
    return original_prompt(public,plan)


def verify_cpu(raw):
    report = json.loads(raw)
    need(report["state"] == "ACTUAL_T512_V2_NATIVE_BUILD_VERIFIED" and report["original_row"] == NATIVE_CPU_ROW
         and report["native"] == {"SHA256":NATIVE_SHA,"bytes":NATIVE_SIZE} and report["providers"] == 71
         and report["strong_exports"] == 285 and report["GPU_steps"] == 0
         and report["actual"] == dict(GPU_compile=0,HOST_compile=3,fresh_M=3,link_objects=33,links=1,reused_objects=29)
         and set(report["verified_dependencies"]) == {"u00","u15","u17"}
         and all(data["paths_identical"] is True and data["before"] == data["actual"]
                 for data in report["verified_dependencies"].values()), "original .1071 HOST evidence differs")
    return report


class DlInfo(ctypes.Structure):
    _fields_ = [("path",ctypes.c_char_p),("base",ctypes.c_void_p),("name",ctypes.c_char_p),("address",ctypes.c_void_p)]


def verify_providers(native, bound):
    need(set(bound) == EXPECTED and bound["prefill512_forward"].__name__ == ENTRY, "actual prefill and baseline CABI set differs")
    dladdr = ctypes.CDLL("libdl.so.2").dladdr
    dladdr.argtypes,dladdr.restype = [ctypes.c_void_p,ctypes.POINTER(DlInfo)],ctypes.c_int
    for symbol,function in bound.items():
        record = DlInfo()
        address = ctypes.cast(function,ctypes.c_void_p)
        need(dladdr(address,ctypes.byref(record)) and record.path
             and Path(os.fsdecode(record.path)).resolve(strict=True) == native,
             "active C ABI provider differs: "+symbol)


def load_native(cpu_evidence, typed_path):
    report = verify_cpu(read(cpu_evidence,NATIVE_CPU_EVIDENCE_SHA,4096,2950))
    need(read(NATIVE,NATIVE_SHA,NATIVE_SIZE,NATIVE_SIZE).startswith(b"\x7fELF"), "expected actual .1071 ELF")
    for row in Path("/proc/self/maps").read_text().splitlines():
        columns = row.split(maxsplit=5)
        if len(columns) == 6:
            need(Path(columns[-1].removesuffix(" (deleted)")).name not in OTHER_ARMS,
                 "a different or prior GLM Native arm is already mapped")
    typed = module(typed_path,TYPED_SHA,32768)
    lib = ctypes.CDLL(str(NATIVE),mode=os.RTLD_NOW|os.RTLD_LOCAL)
    table = typed.attach(lib)
    verify_providers(NATIVE,table)
    read(NATIVE,NATIVE_SHA,NATIVE_SIZE,NATIVE_SIZE)
    return SimpleNamespace(library=lib,binding=typed,api=table,SHA256=NATIVE_SHA,receipt=report,
                           verified_cabi_count=len(table))


def unique_ids(native):
    """Generate the two actual communicator IDs only in the coordinator's global rank0 owner."""
    size = int(native.api["id_size"]())
    need(0 < size <= 4096, "actual CABI communicator ID width differs")
    world, tp = (ctypes.create_string_buffer(size) for _ in range(2))
    need(native.api["id"](world, size) == native.api["id"](tp, size) == 0 and world.raw != tp.raw,
         "actual CABI did not produce two distinct IDs")
    return world.raw, tp.raw


def verify_node_provider_bytes(observed_paths, deadline):
    """Called once per admitted node after the existing bounded ldd -r check."""
    recipe = json.loads(read(RECIPE, RECIPE_SHA, RECIPE_SIZE, RECIPE_SIZE))
    expected = recipe["expected_providers"]
    need(recipe["linked_library"] == str(NATIVE) and len(expected) == 71
         and set(observed_paths) == set(expected), "actual model-container provider path set differs")
    for name, pinned_sha in sorted(expected.items()):
        digest = sha256()
        deadline("verify provider " + Path(name).name)
        with Path(name).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                deadline("hash provider " + Path(name).name)
        need(digest.hexdigest() == pinned_sha, "actual model-container provider digest differs: " + Path(name).name)
    return {"recipe_SHA256": RECIPE_SHA, "confirmed_providers": len(expected)}
