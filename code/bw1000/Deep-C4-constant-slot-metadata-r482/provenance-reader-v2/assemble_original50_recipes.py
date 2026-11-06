"""Restore the fifty CPU377 TU recipes from bounded private original metadata.
No compiler, SDK, Slurm, network, or model calls. Paths in the output retain
original quote-include origins. A derived producer recipe is explicitly distinct
from a saved process argv stream. This does not establish fresh dependencies.
"""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, re, shlex


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assemble(root, repository):
    evidence = {}

    def read(name, out=False):
        p = root / name
        evidence[name] = dict(bytes=p.stat().st_size, SHA256=sha(p))
        q = json.loads(p.read_bytes())
        if out:
            q = q['out']
            if isinstance(q, str):
                q = json.loads(q)
        return q

    w = read('Deep482-original50-recipe-work-v1.json')
    # The intermediate preserves the already reviewed 5803/15-original-record
    # closures; pin its bytes and all subsequent original inputs separately.
    records = w['records']
    original_sources = w['source_records_without_argv']
    producers = read('Deep482-original-parent-producer-commands-first.json', True)
    helpers = read('Deep482-original-helper-stage-and-producer-source-pins.json', True)
    originals = read('Deep482-original15-producer-metadata-sanitized.json')
    recent = read('GLM473-original-watch-and-Deep-two-recipes-20261106-0228.json', True)
    parent_path = repository / 'code/bw1000/Deep-C4-constant-slot-metadata-r482/source-v1/PARENT_PIN.json'
    parent = json.loads(parent_path.read_bytes())
    evidence[str(parent_path.relative_to(repository)).replace('\\', '/')] = dict(SHA256=sha(parent_path), bytes=parent_path.stat().st_size)
    objects = parent['all50_objects_SHA256']
    assert len(objects) == 50
    cpu = read('Deep377-CPU-v1-result-local.json')['receipt']
    assert cpu['all50_objects_SHA256'] == objects
    meta = {v['path']: v['metadata'] for v in producers['files']}
    base = '/refactor/candidates/deep-c4-projection-model-r5920'
    common = '/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM'
    lm = '/refactor/candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM'
    helper = '/refactor/candidates/models-5f91ad7-r3675/Deep/helpers'
    sdk = '/refactor/install/hygon-deep-5f91ad7-r3672'
    script = next(v['metadata']['original_source'] for v in producers['files'] if v['path'].endswith('deep-c4-projection-model-r5920/compile.sh'))
    env = dict(BASE=base, COMMON=common, LM=lm, SDK=sdk, HELPER=helper, ABI='1')
    argv = shlex.split(script.replace('\\\n', ' '))
    for i, value in enumerate(argv):
        for key, replacement in env.items():
            value = value.replace('${'+key+'}', replacement).replace('$'+key, replacement)
        argv[i] = value
    assert not any('$' in value for value in argv)
    oi = argv.index('-o')
    flags5920 = [v for i, v in enumerate(argv) if i not in (oi, oi+1) and not v.endswith(('.cpp', '.o')) and v != '-shared']
    flags7169 = ['g++', '-I/refactor/candidates/deep-combined-r7169/incoming'] + [v for i, v in enumerate(argv) if i not in (0, oi, oi+1) and not v.endswith(('.cpp', '.o')) and v != '-shared']
    derived = 'derived from SHA-matched original producer recipe; not saved ambient environment or process argv stream'

    def check_script(name, expected):
        p = root / name
        assert sha(p) == expected, name
        evidence[name] = dict(bytes=p.stat().st_size, SHA256=sha(p))
        return p.read_text(encoding='utf-8-sig')

    for name in ['build_deep_c4_model_r5920.py', 'build_Deep_combined_r7169.py', 'build_tip_models_r3675_inside.sh']:
        pin = next(v['SHA256'] for v in helpers['scripts'] if v['path'].endswith('/'+name))
        check_script(name, pin)
    live = {v['path']: v for v in recent['metadata']}
    for name in ['build_Deep_quant1_r8881.py', 'build_Deep_latent_prefill_r8936.py']:
        check_script(name, live['incoming/'+name]['SHA256'])

    def add(obj, src, source_hash, args, origin, scope=derived):
        assert obj in objects and obj not in records
        assert re.fullmatch('[0-9a-f]{64}', source_hash), (obj, source_hash)
        assert args[args.index('-c')+1] == src
        assert args[args.index('-o')+1] == obj
        records[obj] = dict(source=src, source_SHA256=source_hash,
                            object_SHA256=objects[obj], compile_argv=args,
                            origin=origin, argv_scope=scope)

    for obj, row in original_sources.items():
        src = row['source']
        if '/deep-c4-projection-model-r5920/' in obj or '/deep-combined-r7169/' in obj:
            flags = flags5920 if '/deep-c4-projection-model-r5920/' in obj else flags7169
            md = obj.replace('/objects/', '/deps/').removesuffix('.o')+'.d'
            add(obj, src, row['source_SHA256'], flags+['-MD','-MF',md,'-c',src,'-o',obj], row['origin']+'; original compile.sh + same-SHA producer')
        else:
            assert obj.endswith('/deep_quant1_gather_r8881.cu.o')
            args = ['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-c',src,'-o',obj]
            add(obj, src, row['source_SHA256'], args, row['origin']+'; original build_Deep_quant1_r8881.py')

    for obj, row in w['commands_without_source'].items():
        args = row['compile_argv'];src=args[args.index('-c')+1]
        if obj.endswith('/HC-unit3.o'):
            hashes = cpu['source_SHA256']
        else:
            hashes = meta[str(PurePosixPath(obj).parent/'CPU-build.json')]['source_SHA256']
        add(obj, src, hashes[PurePosixPath(src).name], args, row.get('origin','original CPU producer commands'), 'saved original producer command; ambient process environment not recorded')

    paths = {
        'routed': 'deepseek_routed_i8_kernels_r20.cu',
        'sinkhorn': 'deepseek_mhc_sinkhorn_r89.cu',
        'gather': 'deep_packed_gather_r374.cu',
        'fused': 'deep_latent_fused_r681.cu',
        'post': 'deepseek_mhc_post_register_r168.cu',
        'scatter': 'deepseek_prefill_scatter_r194.cu',
        'aux_glm_routed': 'glm_routed_fp8_kernels_r19.cu',
    }
    gpu = ['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-I'+lm+'/csrc','-I'+helper+'/src','-I/refactor/candidates/rebased-5f91ad7-r3668/Ops/Deep/InfiniOps/src','-I'+sdk+'/include','-I/refactor/build/ops-deep-5f91ad7-r3672/src','-I/refactor/candidates/rebased-5f91ad7-r3668/Ops/Deep/InfiniOps/generated']
    source_pins = {v['path']: v['SHA256'] for v in helpers['helpers']}
    for name, filename in paths.items():
        src=helper+'/src/'+filename;obj=helper+'/'+name+'.o'
        flags=[] if name in ['routed','aux_glm_routed'] else ['--fmad=false']
        add(obj,src,source_pins[src],gpu+flags+['-c',src,'-o',obj], 'original build_tip_models_r3675_inside.sh; model Deep ABI1; stage source/object SHA association')

    latent='/refactor/candidates/deep-latent-prefill-r8936'
    obj=latent+'/deep_latent_prefill_tiles_r8933.cpp.o';src=latent+'/src/deep_latent_prefill_tiles_r8933.cpp'
    source_hash=live['candidates/deep-latent-prefill-r8936/src/deep_latent_prefill_tiles_r8933.cpp']['SHA256']
    assert live['incoming/Deep-latent-prefill-source-r8936.json']['manifest']['source_SHA256'][PurePosixPath(src).name] == source_hash
    owner = next(v['data'] for v in originals if v['path'].endswith('/deep-owner-device-CPU-r8918.json'))
    prefix=owner['new_compiled_units'][0]['compile_argv'];prefix=prefix[:prefix.index('-MD')];prefix.insert(1,'-I'+latent+'/src')
    add(obj,src,source_hash,prefix+['-MD','-MF',latent+'/actual.d','-c',src,'-o',obj], 'original SHA-matched build_Deep_latent_prefill_r8936.py + original8918 prefix + pinned original input manifest; missing final CPU report')
    # r269 stores only flags under compile_argv. Its pinned producer appends
    # the MD/source/output suffix; do not mislabel the prefix as a full argv.
    worker_obj='/refactor/candidates/latest-family-native-r269/Deep/Native-worker.o'
    worker=records[worker_obj]
    family=next(v['data'] for v in originals if v['path'].endswith('/latest-family-native-r269/CPU-build.json'))
    family_dir=repository/'code/bw1000/public-native-bridge/latest-family-native-r269'
    family_manifest=family_dir/'SOURCE_MANIFEST.json'
    assert sha(family_manifest)==family['source_manifest_SHA256']
    fm=json.loads(family_manifest.read_bytes())
    assert sha(family_dir/'CPU_build_r269.py')==fm['files_SHA256']['CPU_build_r269.py']
    evidence[str(family_manifest.relative_to(repository)).replace('\\','/')]=dict(SHA256=sha(family_manifest),bytes=family_manifest.stat().st_size)
    assert '-c' not in worker['compile_argv']
    worker['compile_argv'] += ['-MD','-MF',str(PurePosixPath(worker_obj).with_suffix('.d')),'-c',worker['source'],'-o',worker_obj]
    worker['argv_scope']=derived+'; r269 saved prefix plus manifest-pinned compile_worker suffix'
    assert set(records) == set(objects), sorted(set(objects)-set(records))
    for obj, row in records.items():
        assert row['object_SHA256'] == objects[obj]
        assert re.fullmatch('[0-9a-f]{64}',row['source_SHA256'])
        args=row['compile_argv'];assert args[args.index('-c')+1] == row['source']
        row['linked_object']=obj
        row['original_compile_output']=args[args.index('-o')+1]
        row['quote_include_origin']=str(PurePosixPath(row['source']).parent)
        row['preprocess_prefix']=args[:args.index('-MD')] if '-MD' in args else args[:args.index('-c')]
        assert '-c' not in row['preprocess_prefix'] and '-o' not in row['preprocess_prefix']
    deps=w['dependencies']
    return dict(schema=1,scope='original50 recipe restoration only; no fresh preprocessing/compiler invocation/ABI closure or numeric qualification',
                CPU_GO=False,GPU_GO=False,Model_GO=False,
                original_CPU377_receipt_SHA256=parent['CPU377_original_receipt_SHA256'], original_parent_actual_CPU=cpu['actual_step'],
                evidence=evidence,objects=records,historical_dependency_SHA256_by_linked_object=deps,
                count=50,compiler_families=sorted(set(v['compile_argv'][0] for v in records.values())),
                original_sources_with_historical_dependency_hashes=len(deps),
                boundaries=['Twenty reused5920 objects keep original5803 source and quote origins.',
                            'Historical source/header hashes do not establish current full dependency bytes.',
                            'All fifty fresh-M reverse dependencies and changed class-layout TU closure remain required before a new Native build.',
                            'A missing original MD is not evidence of no dependency.',
                            'Derived commands preserve known original recipe flags; original ambient environment was not recorded.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private-root',type=Path,required=True);p.add_argument('--repository',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    q=assemble(a.private_root,a.repository)
    a.output.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(count=q['count'],historical_dependency_units=q['original_sources_with_historical_dependency_hashes'],bytes=a.output.stat().st_size,SHA256=sha(a.output),Native_build_authorized=False)))
