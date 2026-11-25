"""Place this worker's native writable state inside the one private work bind."""

from contextlib import ExitStack
import os
from pathlib import Path
import stat


DIRECTORIES = (
    'private-state', 'private-state/home', 'private-state/home/.config',
    'private-state/home/.config/miopen', 'private-state/home/.config/miopen/user-db',
    'private-state/home/.config/miopen/miopen-lockfiles', 'private-state/home/.cache',
    'private-state/home/.cache/miopen', 'private-state/tmp',
)
HOME = '/resume10-work/private-state/home'
TMP = '/resume10-work/private-state/tmp'
ENV = {
    'HOME': HOME, 'TMPDIR': TMP, 'TEMP': TMP, 'TMP': TMP,
    'MIOPEN_USER_DB_PATH': HOME + '/.config/miopen/user-db',
    'MIOPEN_CUSTOM_CACHE_DIR': HOME + '/.cache/miopen',
}
UNSET = (
    'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_RUNTIME_DIR',
    'MIOPEN_SYSTEM_DB_PATH', 'MIOPEN_LOCKFILE_DIR', 'MIOPEN_LOCKFILE_PATH',
)


def shell_environment():
    return ('unset ' + ' '.join(UNSET) + '; export '
            + ' '.join(f'{key}={value}' for key, value in ENV.items()) + '; ')


def check_environment(env):
    if (any(env.get(key) != value for key, value in ENV.items())
            or any(key in env for key in UNSET)):
        raise ValueError('EXACT_EFFECTIVE_PRIVATE_HOME_DB_CACHE_AND_TMP_ENV')


def directory_identities(work, expected=None):
    work = Path(work)
    if work.resolve(strict=True) != work:
        raise ValueError('EXACT_CANONICAL_OWNED_WORK')
    names = ('', *DIRECTORIES)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    identities = {}
    with ExitStack() as stack:
        fds = {}
        for name in names:
            parent, _, leaf = name.rpartition('/')
            fd = os.open(leaf, flags, dir_fd=fds[parent]) if name else os.open(work, flags)
            stack.callback(os.close, fd)
            fds[name] = fd
            actual = os.fstat(fd)
            if (actual.st_uid != os.getuid() or stat.S_IMODE(actual.st_mode) != 0o700
                    or not stat.S_ISDIR(actual.st_mode) or os.fstatvfs(fd).f_flag & os.ST_RDONLY):
                raise ValueError('EXACT_OWNED_RW_0700_NATIVE_STATE_DIRS')
            if name:
                allowed = {part[len(name)+1:] for part in DIRECTORIES
                           if part.startswith(name + '/') and '/' not in part[len(name)+1:]}
                if set(os.listdir(fd)) != allowed:
                    raise ValueError('EXACT_NEW_EMPTY_NATIVE_STATE_LAYOUT')
            identities[name or '.'] = [actual.st_dev, actual.st_ino]
    if expected is not None and identities != expected:
        raise ValueError('EXACT_HOST_AND_CONTAINER_NATIVE_STATE_FD_IDENTITIES')
    return identities


def create(work):
    for name in DIRECTORIES:
        (work / name).mkdir(mode=0o700)
    return directory_identities(work)
