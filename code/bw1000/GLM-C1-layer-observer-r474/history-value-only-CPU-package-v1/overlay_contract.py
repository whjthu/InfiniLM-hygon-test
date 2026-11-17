"""Pure planning/validation helpers. No compiler, SSH, staging or library loader."""
import posixpath

def owned_path(original, mirror):
    if not original.startswith('/refactor/') or not mirror.startswith('/refactor/results/'):
        raise ValueError('EXACT_PROJECT_ROOT_REQUIRED')
    normalized=posixpath.normpath(original)
    if not normalized.startswith('/refactor/'):
        raise ValueError('PROJECT_PATH_ESCAPE')
    return mirror.rstrip('/')+'/'+normalized[len('/refactor/'):]

def remap_prefix(prefix, mirror):
    """Preserve actual original options; only remap explicit project include paths."""
    out=[];separate=False
    for arg in prefix:
        if separate:
            out.append(owned_path(arg,mirror) if arg.startswith('/refactor/') else arg)
            separate=False
        elif arg in ('-I','-iquote','-isystem'):
            out.append(arg);separate=True
        elif arg.startswith('-I/refactor/'):
            out.append('-I'+owned_path(arg[2:],mirror))
        elif '/refactor/' in arg:
            raise ValueError('UNCLASSIFIED_PROJECT_PATH_OPTION')
        else:out.append(arg)
    if separate:raise ValueError('MISSING_INCLUDE_ARGUMENT')
    return out

def verify_fresh_M(paths, expected_pins, actual_pins, mirror, prepare_aliases, inode_by_path, new_header_sha):
    """Caller must supply actual stable hashes/inodes, not manifest assertions."""
    if set(paths)!=set(expected_pins) or actual_pins!=expected_pins:
        raise ValueError('EXACT_FRESH_M_PINS_REQUIRED')
    if any(p.startswith('/refactor/') and not p.startswith(mirror.rstrip('/')+'/') for p in paths):
        raise ValueError('QUOTE_INCLUDE_ESCAPED_OWNED_MIRROR')
    headers=[p for p in paths if p.endswith('/glm_moe_dsa_attention_prepare.hpp')]
    allowed={owned_path(p,mirror) for p in prepare_aliases}
    if any(p not in allowed or actual_pins[p]!=new_header_sha for p in headers):
        raise ValueError('UNREGISTERED_OR_OLD_PREPARE_HEADER')
    if headers and len({tuple(inode_by_path[p]) for p in headers})!=1:
        raise ValueError('PREPARE_ALIASES_NOT_ONE_OWNED_INODE')
    return True
