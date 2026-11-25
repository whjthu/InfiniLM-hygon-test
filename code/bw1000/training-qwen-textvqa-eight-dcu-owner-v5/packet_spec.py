"""The exact public source-only packet membership, without data or model bytes."""

HERE="code/bw1000/training-qwen-textvqa-eight-dcu-owner-v5/"
TRAIN="code/bw1000/training-qwen-textvqa-first-two-updates-v2/"
FIRST="code/bw1000/training-qwen-native25-hip-v3/"
SECOND="code/bw1000/training-qwen-native-vproj-layout-native-first2-v1/"
MEMBERS={
    **{name:HERE+name for name in ("owner.py","contract.py","device.py","packet_spec.py","install.py")},
    **{"trainer/"+name:TRAIN+name for name in ("evidence.py","prepare.py")},
    "trainer/street_updates.py":HERE+"street_updates_py310.py",
    "trainer/native_entry.py":HERE+"native_entry_py310.py",
    "snapshot1231.py":SECOND+"snapshot.py",
    "scope1231.py":SECOND+"scope.py",
    "accounting1231.py":SECOND+"slurm_accounting.py",
    "native220.py":FIRST+"native.py",
    "owned220.py":FIRST+"owned_processes.py",
    "assets220.json":FIRST+"ASSETS.json",
    "vision220.json":FIRST+"VISION_FILES.json",
    "native1251.json":"results/bw1000/2026-11-25/specialized01-textvqa-qwen-native-1251-41-training-inputs-pass.json",
    "sampler1252.json":"results/bw1000/2026-11-25/specialized01-qwen-street-1252-stock-sampler-two-native-CPU-epochs.json",
    "terminal220.json":"results/bw1000/2026-11-14/training-qwen-1087656-220-native25-independent-terminal-and-checkpoint-inventory.json",
    "terminal1231.json":"results/bw1000/2026-11-22/training-qwen-native-vproj-upstream-layout-original1231-terminal.json",
}
