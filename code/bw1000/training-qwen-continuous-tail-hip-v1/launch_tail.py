"""Enable the continuous Qwen witness on the validated native launch."""
from pathlib import PurePosixPath

KEY = "qwen_native_continuous_tail"


def add_witness(launch, framework):
    argv = launch["command"]
    if argv.count("--callbacks") != 1 or argv.count("--external_plugins") != 1:
        raise ValueError("expected one native callback group and one plugin group")
    first = argv.index("--callbacks") + 1
    if argv[first:first + 3] != ["native_initial_trainables", "native_first_updates", "nvidia_baseline"]:
        raise ValueError("unexpected native callback order")
    plugin = str(PurePosixPath(framework) / "vlm/plugins/qwen_continuous_tail.py")
    if KEY in argv or plugin in argv:
        raise ValueError("continuous witness already configured")
    last_plugin = argv.index("--external_plugins") + 1
    while last_plugin < len(argv) and not argv[last_plugin].startswith("--"):
        last_plugin += 1
    argv.insert(first + 2, KEY)
    if last_plugin > first + 2:
        last_plugin += 1
    argv.insert(last_plugin, plugin)
    launch["required_assets"]["plugins"].append(plugin)
    launch["diagnostic_observer"]["observers"].append(dict(plugin=plugin, callback=KEY))
    return launch
