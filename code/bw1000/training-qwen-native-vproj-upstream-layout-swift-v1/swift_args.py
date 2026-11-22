"""Add just this observer to existing Swift list-style option groups."""

from pathlib import PurePosixPath


def add_observer(command, plugin):
    result = list(command)
    path = PurePosixPath(plugin)
    if not path.is_absolute() or '..' in path.parts or path.name != 'swift_plugin.py':
        raise ValueError('exact deployed Swift observer path required')
    for flag, value in (('--callbacks', 'qwen_native_vproj_upstream_layout'), ('--external_plugins', str(path))):
        if result.count(flag) != 1 or value in result:
            raise ValueError('ambiguous existing Swift observer option group')
        first = result.index(flag)+1
        last = next((i for i in range(first, len(result)) if result[i].startswith('--')), len(result))
        if first == last:
            raise ValueError('empty existing Swift observer option group')
        result.insert(last, value)
    return result
