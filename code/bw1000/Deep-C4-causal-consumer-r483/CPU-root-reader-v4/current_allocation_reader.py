"""Bind the independent CPU artifact reader to the published I03 allocation."""

import hashlib


BASE_SHA256 = "cfe3ed31e64de4b4d70903e5d6d9689d2783c303366e5b243700433bd1986212"


def bind(source):
    if hashlib.sha256(source).hexdigest() != BASE_SHA256:
        raise ValueError("historical CPU reader differs")
    reader = source.decode("utf-8")
    replacements = (
        ("730af6c4130270dd7e65122e3c6e5e41fd79741f", "9746283226ac044b588ecadbe46e238a269055df"),
        ("cf54f255ed4c88a23a5b7402e18092e06394eb648666ed0cd004b4bd70e3ea64",
         "f22534d4509ed82921348b5452ab065046f825ad4017e455b277d3e75c33f499"),
        ("0700705e6958fef6cad66cc29c2b0fd38c9f0c1ec6910d7b8224a369c4f8b0db",
         "65443c3d81415156d92caf9f9a0952296acc2313e6e8763de7df0c5aabb9efa6"),
        ('"State", "ExitCode", "ElapsedRaw")', '"State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")'),
        (r'910407\.[1-9][0-9]*', r'1103606\.[1-9][0-9]*'),
        ('or launch["PGID"] != pid or not isinstance(ticks, str)\n'
         '            or not re.fullmatch(r"[1-9][0-9]*", ticks)):',
         'or launch["PGID"] != pid or type(ticks) is not int or ticks <= 0):'),
        ('"srun", "--jobid=910407", "--overlap", "--exact", "--immediate=20",',
         '"srun", "--jobid=1103606", "--exact", "--immediate=5",'),
        ('"-w", "a14r2n20",', '"-w", "f13r2n01",'),
        ('handle, label, state, exit_code, elapsed = rows[0]',
         'handle, label, state, exit_code, elapsed, node, allocated = rows[0]\n'
         '    tres = set(allocated.split(","))\n'
         '    if node != "f13r2n01" or tres not in ({"cpu=4", "mem=8G", "node=1"},\n'
         '                                            {"cpu=4", "mem=8192M", "node=1"}):\n'
         '        raise ValueError("CPU-v2 allocated resources differ")'),
        ('original_epoch=epoch, original_deadline=end,\n'
         '                can_inspect_successful_build=',
         'original_epoch=epoch, original_deadline=end, node=node, allocated_TRES=allocated,\n'
         '                can_inspect_successful_build='),
    )
    for old, current in replacements:
        if reader.count(old) != 1:
            raise ValueError("historical reader substitution anchor differs")
        reader = reader.replace(old, current, 1)
    return reader.encode("utf-8")
