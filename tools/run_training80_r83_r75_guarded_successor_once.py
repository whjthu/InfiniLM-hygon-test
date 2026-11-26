"""Renew only independently closed R75 workers through the published ordinary guard."""

import inspect
from hashlib import sha256
from pathlib import Path
import runpy
import textwrap


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = Path(__file__).resolve()
STEM = 'training80-r83-r75-0907-guarded-bf16-v1'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
R75 = ('tools/run_training80_r75_r57_remaining_twelve_guarded_successor_once.py',
       '6f2e8db0e1e90cce833536d6a1d4ea9aa12c3e9a6635897d3e308d47825737ca')
R75_CLAIM = ('coordination/tasks/training80-r75-r57-remaining-twelve-guarded-bf16-v1.json',
             '8cd300b08be32e1535750464144bb4268500858a223cd842d9f5799547db365a')
READER = ('local-only/read_training80_r83_r75_twelve_0907_exact_originals_once.py',
          'c10eedcee0ee1a24f07fb0394f771f416ba08cc2d96baf8589d6c49e4994b279')
ORIGINALS = {
    (1,('f13r4n07','f13r1n04','f13r3n14')): (
        'local-only/training80-r83-r75-twelve-0907-exact-originals-1103606-5ee19b62738948d1b5f640d39548b0d9.json',
        '9df57b2ddaf7e020589d22676eb8ffcaeb6f58844312302ae6da2b236151855e'),
    (3,('f12r3n13','f12r1n16','f12r3n08','f13r3n10')): (
        'local-only/training80-r83-r75-twelve-0907-exact-originals-1194900-776dd29b9316451cb6ec597cc4daa009.json',
        '90a7f0dac6c7917d0445efe0204bf22ae5a6f33d834242e73a80f65b54978968'),
    (4,('f12r3n12',)): (
        'local-only/training80-r83-r75-twelve-0907-exact-originals-1194900-560cf19baddd4103b4532a42f09d9839.json',
        'f0b4bb395edc5a1a4f02570780815778ecaa327652fd5f53e577250e41e356e9'),
}
ORDER = ('f13r4n07','f12r3n13','f12r1n16','f12r3n08','f13r1n04','f13r3n10','f13r3n14','f12r3n12')


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def pin(spec):
    path = ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True) == path, 'exact original path differs')
    raw = path.read_bytes()
    require(0 < len(raw) <= 262144 and sha256(raw).hexdigest() == spec[1], 'exact original bytes differ')
    return raw


def entry():
    require(ORIGINALS and ORDER, 'awaiting exact qualified original receipts, no scope')
    pin(R75)
    pin(R75_CLAIM)
    reader = runpy.run_path(str(ROOT/READER[0])) if pin(READER) else None
    parent = runpy.run_path(str(ROOT/R75[0]))
    namespace = dict(parent, STEM=STEM, WRAPPER=WRAPPER, CLAIM=CLAIM, READER=READER,
                     ORIGINALS=ORIGINALS, ORDER=ORDER, FENCES=reader['FENCES'],
                     R75=R75,R75_CLAIM=R75_CLAIM,R83_NODES=set(reader['CPU_REPLY']),__file__=__file__)
    source = textwrap.dedent(inspect.getsource(parent['entry']))
    start = source.index('    blocked = ')
    end = source.index('    r46=prior', start)
    source = source[:start]+(
        "    require(set(ORDER)<=R83_NODES and len(ORDER)==len(set(ORDER)),\n"
        "            'new exact R75 subset differs')\n"
    )+source[end:]
    changes = (
        ("R46=r46,__file__=__file__)", "R46=r46,R75=R75,R75_CLAIM=R75_CLAIM,__file__=__file__)"),
        ("    _, _, originals57 = module(R57)['entry']()[0]()",
         "    _, _, originals57 = module(R75)['entry']()[0]()"),
        ("    read_core,read_base,_,groups,_ = reader['load']()",
         "    read_core,read_base,groups,_,_ = reader['load']()"),
        ("        expected = [by_node[n] for n in names]",
         "        expected = [by_node[n] for n in names]\\n"
         "        require(len(names)==len(set(names)), 'repeated original read node')"),
        ("bf0a12ce-79e0-4702-9407-1bea55206a55","6e94424f-8d1a-4632-9e59-d32283d24b0c"),
        ('"==7,\'fixed seven exclusive new requests differ\'","==12,\'fixed exclusive new requests differ\'"',
         '"==7,\'fixed seven exclusive new requests differ\'","==len(ORDER),\'fixed exclusive new requests differ\'"'),
        ('(PREVIOUS,R47,R54,R51,R72,R74,R56,R57,R58,R59,R64,R65,R46)',
         '(PREVIOUS,R47,R54,R51,R72,R74,R56,R57,R58,R59,R64,R65,R46,R75,R75_CLAIM)'),
        ('for spec in (R47,R54,R51,R72,R74,R56,R57,R58,R59,R64,R65,r46):',
         'for spec in (R47,R54,R51,R72,R74,R56,R57,R58,R59,R64,R65,r46,R75,R75_CLAIM):'),
        ("    exec(compile(load,", """    load=adapt(load,(
        ("        for old,proof in zip(expected,observed['targets'],strict=True):",
         "        for old,proof in zip(expected,observed['targets'],strict=True):\\n"
         "            if old['node'] not in ORDER:\\n"
         "                require(proof['qualified_for_separate_fresh_physical'] is False, 'qualified original omitted')\\n"
         "                continue\\n"
         "            require(old['node'] not in rows, 'original candidate supplied twice')"),
        ("and files['submission']==old['submit_SHA256'] and len(set(proof['pci']))==8 and proof['pci']==old['pci']",
         "and files['cgroup']==old['cgroup_SHA256'] and len(set(proof['pci']))==8 and proof['pci']==old['pci']"),
        ("and proof['worker_pid']==old['previous_worker_pid']>0 and proof['worker_start_epoch']==start",
         "and proof['worker_pid']>0 and proof['worker_start_epoch']==start"),
    ))
    exec(compile(load,"""),
        ("for (_,names),s in ORIGINALS.items() for n in names}",
         "for (_,names),s in ORIGINALS.items() for n in names if n in ORDER}"),
    )
    for old, new in changes:
        require(source.count(old) == 1, 'audited R75 guard integration boundary differs')
        source = source.replace(old, new, 1)
    exec(compile(source, '<bounded closed R75 successor>', 'exec'), namespace)
    return namespace['entry']()


if __name__ == '__main__':
    _, _, main = entry()
    main()
