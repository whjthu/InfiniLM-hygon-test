"""Append one opt-in external observer to the original Swift argument groups."""
def add_candidate(command, plugin):
    result = list(command)
    for flag, value in (('--callbacks', 'qwen_math_sdpa_recovery'), ('--external_plugins', str(plugin))):
        if result.count(flag) != 1 or value in result:
            raise ValueError('Ambiguous original native Swift argument group')
        index = result.index(flag) + 1
        end = index
        while end < len(result) and not result[end].startswith('--'):
            end += 1
        if end == index:
            raise ValueError('Expected an existing native Swift argument group')
        result.insert(end, value)
    return result
