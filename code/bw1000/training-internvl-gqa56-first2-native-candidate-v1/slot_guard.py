"""Check the frozen 64 slots at the InternVL native input and optimizer boundaries."""
from collections import Counter
from hashlib import sha256
import inspect
import json
import os
import stat


SPLIT_SHA = "28c6a29c501bbb0ef066014aadfc15eaa43fc5f31952d9960917208ca6c669d1"
FILES = {
    "proposed-exposures64.jsonl": (19599,"28643ee29fc0a2d59ccf26883a66bd870dbb89be1f7f710c23dc9aca0727d174"),
    "proposed-slots.json": (9838,"9e6695e5023d713373c9671d287892108077aa6453308127ba6f5d686a719570"),
    "train56.jsonl": (17149,"ffc8b779298e21d7dd697961cf775f72e3a0089a9a3ae42b254d04965a11caad"),
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def bind(original64, train56, exposures64, slots, roster):
    """Compare physical JSONL bytes, source ordinals and the original native roster."""
    original, train, exposures = (x.splitlines() for x in (original64,train56,exposures64))
    need((len(original),len(train),len(exposures),len(slots),len(roster))==(64,56,64,64,64),
         "EXACT_SPLIT_SOURCE_DENOMINATORS")
    original_rows = {row["ordinal"]:row["row_SHA256"] for row in roster.values()}
    need(original_rows=={i:digest(line) for i,line in enumerate(original,1)} and len(set(original))==64,
         "EXACT_4370_SOURCE_ROSTER")
    need(len(set(train))==56 and set(train) < set(original) and set(exposures)==set(train),
         "EXACT_TRAIN_ONLY_EXPOSURE_MEMBERSHIP")
    by_rank = [[] for _ in range(8)]
    for index,(line,slot) in enumerate(zip(exposures,slots,strict=True)):
        rank = index%8
        ordinal = slot["source_ordinal"]
        need(type(ordinal) is int and 1<=ordinal<=64 and line==original[ordinal-1] and
             slot==dict(update=index//32+1,rank=rank,accumulation=index%32//8+1,
                        source_ordinal=ordinal,row_SHA256=digest(line),repeat=index>=56),
             "EXACT_IMMUTABLE_SOURCE_SLOT")
        by_rank[rank].append(dict(ordinal=ordinal,row_SHA256=slot["row_SHA256"]))
    need(exposures[:56]==train and len(set(exposures[:32]))==len(set(exposures[32:]))==32 and
         set(exposures[56:])<=set(exposures[:32]), "EXACT_56_PLUS_8_EXPOSURE_WINDOWS")
    expected_counts = Counter(digest(line) for line in exposures)
    subset = {key:value for key,value in roster.items() if value["row_SHA256"] in expected_counts}
    need(len(subset)==56,"EXACT_NATIVE_TRAIN56_SUBROSTER")
    return subset, by_rank


def load(root, original, roster, pin):
    summary = json.loads(pin(root/"SPLIT.json", expected=SPLIT_SHA, cap=8192))
    need(summary["target_original_binding_verified"] is True and summary["optimizer_updates"]==0 and
         summary["actual_training_sampler_verified"] is False and summary["train_rows"]==56 and
         summary["internal_rows"]==8,"EXACT_TARGET_SPLIT_AUTHORITY")
    payload = {}
    for name,(size,sha) in FILES.items():
        need(summary["files"][name]==dict(bytes=size,SHA256=sha),"EXACT_TARGET_SPLIT_MEMBER")
        payload[name] = pin(root/name, expected=sha, extent=size, cap=32768)
    return bind(original,payload["train56.jsonl"],payload["proposed-exposures64.jsonl"],
                json.loads(payload["proposed-slots.json"]),roster)


def exact_source(by_rank, rank, offset, observation):
    need(0<=rank<8 and 0<=offset<8,"EXACT_NATIVE_FORWARD_SLOT_RANGE")
    expected = by_rank[rank][offset]
    need(isinstance(observation,dict) and all(observation.get(key)==value for key,value in expected.items()),
         "NATIVE_RANK_ACCUMULATION_SOURCE_MISMATCH")
    return expected


def optimizer_boundary(global_step, updates, input_count, output_count, pending, rates):
    need(global_step==updates and global_step in (0,1) and
         input_count==output_count==4*(updates+1) and pending is None,"EXACT_PRE_ADAMW_SLOT_DENOMINATOR")
    need(rates and all(rate==0 if global_step==0 else rate>0 for rate in rates),
         "ORIGINAL_LR_ZERO_THEN_POSITIVE")


def exact_terminal(by_rank, rank, observed):
    need(observed==by_rank[rank],"EXACT_TERMINAL_ORDERED_RANK_SLOTS")


def sampler_identity(loader):
    """Observe wrapper identity and source only. Do not traverse the input iterator."""
    kinds = []
    seen = set()
    for link in ("dataloader","batch_sampler","sampler"):
        need(loader is not None and id(loader) not in seen,"ACTUAL_NATIVE_DATALOADER_CHAIN_UNAVAILABLE")
        seen.add(id(loader))
        cls = type(loader)
        lines, first = inspect.getsourcelines(cls)
        raw = "".join(lines).encode()
        need(len(raw)<=256*1024,"BOUNDED_NATIVE_SAMPLER_SOURCE")
        kinds.append(dict(link=link,module=cls.__module__,name=cls.__qualname__,line=first,SHA256=digest(raw)))
        loader=getattr(loader, "batch_sampler" if link=="dataloader" else "sampler", None)
    return dict(classes=kinds,sampler_iterated=False,order_verified=False)


def seal_checkpoint(root, check_deadline=lambda: None):
    """Fsync and digest native checkpoint files after the Swift child has returned."""
    need(root.resolve(strict=True)==root and stat.S_ISDIR(root.lstat().st_mode),"EXACT_NATIVE_CP2_PATH")
    required = {"adapter_config.json","adapter_model.safetensors","optimizer.pt","scheduler.pt","trainer_state.json",
                *(f"rng_state_{rank}.pth" for rank in range(8))}
    optional = {"training_args.bin","additional_config.json","README.md"}
    observed = {path.name for path in root.iterdir()}
    need(required <= observed <= (required | optional),"EXACT_NATIVE_CP2_MEMBERS")
    members = {}
    for name in sorted(observed):
        path = root/name
        path_stat=path.lstat()
        need(stat.S_ISREG(path_stat.st_mode),"EXACT_NATIVE_CP2_REGULAR_MEMBER")
        fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_BINARY",0))
        with os.fdopen(fd,"rb") as stream:
            before=os.fstat(stream.fileno())
            need(stat.S_ISREG(path_stat.st_mode) and os.path.samestat(before,path_stat)
                 and 0<before.st_size<=2*1024**3,"EXACT_NATIVE_CP2_REGULAR_MEMBER")
            content=sha256()
            while chunk:=stream.read(8*1024*1024):
                check_deadline()
                content.update(chunk)
            os.fsync(stream.fileno())
            after=os.fstat(stream.fileno())
            need(os.path.samestat(before,path.lstat()) and
                 (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                 (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),"CP2_MEMBER_CHANGED")
            members[name]=dict(bytes=before.st_size,SHA256=content.hexdigest())
    for name in ("trainer_state.json","adapter_config.json"):
        with (root/name).open("rb") as stream:
            raw=stream.read(65537)
        need(len(raw)<=65536 and digest(raw)==members[name]["SHA256"],"EXACT_CP2_JSON_MEMBER")
        state=json.loads(raw)
        if name=="trainer_state.json":
            need((state["global_step"],state["max_steps"])==(2,25),"EXACT_CP2_STEP_AND_SCHEDULE")
        else:
            need((state["r"],state["lora_alpha"],state["lora_dropout"])==(16,32,0.05),"EXACT_CP2_ADAPTER_RECIPE")
    need(observed=={path.name for path in root.iterdir()},"EXACT_CP2_FINAL_MEMBER_SET")
    for folder in (root,root.parent):
        fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return dict(step=2,native_schedule_steps=25,files=members,full_reload_verified=False)
